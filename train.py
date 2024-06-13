#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from torch import nn
from copy import deepcopy
from typing import NamedTuple
from random import randint
import numpy as np
from utils.loss_utils import l1_loss, ssim, ssim_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel  
from utils.general_utils import safe_state, decode_op
import uuid
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.semantic_utils import trainId2color
from torch.nn import CrossEntropyLoss
from utils.general_utils import seedEverything
from PIL import Image
import json
import torchvision

from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

# seedEverything()

results = {'train': {}, 'test': {}}

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# metrics
m_psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
m_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
m_lpips = LearnedPerceptualImagePatchSimilarity().to('cuda')

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, 
             semantic, optical, unicycle, uc_fit_iter, uc_joint_opt, adapt, affine, data_type, ignore_dynamic):
    
    if semantic:
        semantic_ce = CrossEntropyLoss()

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, feat_mutable=True, affine=affine)
    scene = Scene(dataset, gaussians, shuffle=False, unicycle=unicycle, uc_fit_iter=uc_fit_iter, data_type=data_type, ignore_dynamic=ignore_dynamic)

    os.makedirs(os.path.join(scene.model_path, "unicycle"), exist_ok=True)
    
    scene.gaussians.training_setup(opt)
    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
        dynamic_gaussian.training_setup(opt)
    
    unicycles = scene.unicycles

    if checkpoint:
        # TODO lack dynamic
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    os.makedirs(os.path.join(scene.model_path, "save_train"), exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        scene.gaussians.update_learning_rate(iteration)
        for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
            dynamic_gaussian.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                dynamic_gaussian.oneupSHdegree()


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # cam_index = randint(0, len(viewpoint_stack)//N_cam - 1) * N_cam - 1
        cam_index = randint(0, len(viewpoint_stack)-1)
        if cam_index - 2 >= 0:
            prev_viewpoint_cam = viewpoint_stack[cam_index-2]
        else:
            prev_viewpoint_cam = None
        viewpoint_cam = viewpoint_stack[cam_index]
        # viewpoint_cam = viewpoint_stack.pop(cam_index)
        # Render
        render_pkg = render(viewpoint_cam, prev_viewpoint_cam, gaussians, scene.dynamic_gaussians, unicycles, pipe, background, optical)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if iteration % 500 == 0:
            torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_train", f"{iteration}_{viewpoint_cam.image_name}.png"))

        # Loss
        loss = 0
        gt_image = viewpoint_cam.original_image.cuda()
        mask = None
        if ignore_dynamic:
            mask = viewpoint_cam.mask
        # image, gt_image = image[..., 50:-50], gt_image[..., 50:-50]

        if semantic:
            semantic_map = render_pkg["feats"]
            semantic_gt = viewpoint_cam.semantic2d.to(semantic_map.device)
            if mask is not None:
                semantic_map[~mask] = semantic_gt[~mask]
            semantic_loss = semantic_ce(semantic_map.permute(1,2,0).view(-1, 20), semantic_gt.view(-1)) * 0.01
            loss += semantic_loss

        optical_loss = torch.tensor(0., device='cuda')
        if optical and viewpoint_cam.optical_gt is not None:
            opticalflow = render_pkg["opticalflow"]
            opticalflow = opticalflow.permute(1,2,0)[..., :2]
            gt_optical = viewpoint_cam.optical_gt.cuda()
            # print(torch.max(opticalflow), torch.max(gt_optical))
            if mask is not None:
                opticalflow[~mask] = gt_optical[~mask]
            optical_loss += torch.abs(opticalflow - gt_optical).mean() * 0.005
            loss += optical_loss

        # if mask is not None:
        #     image[:, ~mask] = gt_image[:, ~mask]
        Ll1 = l1_loss(image, gt_image, mask=mask)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss(image, gt_image, mask=mask)
        loss += rgb_loss
        # if iteration % 500 == 60:
        #     torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_train", f"{iteration}_{viewpoint_cam.image_name}.png"))

        reg_loss = 0
        if uc_joint_opt and (len(unicycles) > 0) and (iteration > 500):
            for track_id, unicycle_pkg in unicycles.items():
                model = unicycle_pkg['model']
                reg_loss += 0.1 * model.reg_loss() + 0.02 * model.pos_loss()
            reg_loss = reg_loss / len(unicycles)
            loss += reg_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"RGB": f"{rgb_loss:.{4}f}"}
                tb_writer.add_scalar('train_loss_patches/rgb_loss', rgb_loss.item(), iteration)
                if semantic:
                    postfix["Semantic"] = f"{semantic_loss:.{4}f}"
                    tb_writer.add_scalar('train_loss_patches/semantic_loss', semantic_loss.item(), iteration)
                if optical:
                    postfix["Optical"] = f"{optical_loss:.{4}f}"
                    tb_writer.add_scalar('train_loss_patches/optical_loss', optical_loss.item(), iteration)
                if reg_loss != 0:
                    postfix["UniReg"] = f"{reg_loss:.{4}f}"
                    tb_writer.add_scalar('train_loss_patches/unireg_loss', reg_loss.item(), iteration)
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [scene.dynamic_gaussians, unicycles, pipe, background, optical])
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                w2c = torch.linalg.inv(viewpoint_cam.c2w.float())
                if adapt:
                    points = gaussians._xyz
                    cam_points = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
                    custom_lr = torch.clip(cam_points[:, 2], min=0.1*scene.cameras_extent, max=10*scene.cameras_extent) / scene.cameras_extent
                    gaussians.optimizer.step(custom_lr=custom_lr, name=['xyz'])
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    if adapt:
                        # points = dynamic_gaussian._xyz
                        # cam_points = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
                        # depth = torch.clip(cam_points[:, 2], min=0, max=100) / 3 + 1
                        custom_lr = torch.ones_like(dynamic_gaussian._xyz[:, 0]) * 3.0
                        dynamic_gaussian.optimizer.step(custom_lr=custom_lr, name=['xyz', 'scaling'])
                    else:
                        dynamic_gaussian.optimizer.step()
                    dynamic_gaussian.optimizer.zero_grad(set_to_none = True)

                if uc_joint_opt and iteration > 1000:
                    for track_id, unicycle_pkg in unicycles.items():
                        unicycle_optimizer = unicycle_pkg['optimizer']
                        unicycle_optimizer.step()
                        unicycle_optimizer.zero_grad(set_to_none = True)
                        if iteration % 1000 == 0:
                            for g in unicycle_optimizer.param_groups:
                                g['lr'] /= 2

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + '/ckpts', exist_ok=True)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpts/chkpnt" + str(iteration) + ".pth")

                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    torch.save((dynamic_gaussian.capture(), iteration), scene.model_path + f"/ckpts/dynamic_{iid}_chkpnt{iteration}.pth")
                for track_id, unicycle_pkg in unicycles.items():
                    model = unicycle_pkg['model']
                    torch.save(model.capture(), scene.model_path + f"/ckpts/unicycle_{track_id}_chkpnt{iteration}.pth")
                    model.visualize(os.path.join(scene.model_path, "unicycle", f"{track_id}_{iteration}.png"),
                                    gt_centers=unicycle_pkg['input_centers'])
                        
                # # save semantic pcd
                # print("Saving Semantic")
                # render_set_v2(scene.model_path, None, iteration, None, gaussians, None, None)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                current_index = gaussians.get_xyz.shape[0]
                gaussians.max_radii2D[visibility_filter[:current_index]] = torch.max(gaussians.max_radii2D[visibility_filter[:current_index]], radii[:current_index][visibility_filter[:current_index]])
                gaussians.add_densification_stats_grad(viewspace_point_tensor.grad[:current_index], visibility_filter[:current_index])
                last_index = current_index

                for iid in viewpoint_cam.dynamics.keys():
                    dynamic_gaussian = scene.dynamic_gaussians[iid]
                    current_index = last_index + dynamic_gaussian.get_xyz.shape[0]
                    visible_mask = visibility_filter[last_index:current_index]
                    dynamic_gaussian.max_radii2D[visible_mask] = torch.max(dynamic_gaussian.max_radii2D[visible_mask], radii[last_index:current_index][visible_mask])
                    dynamic_gaussian.add_densification_stats_grad(viewspace_point_tensor.grad[last_index:current_index], visible_mask)
                    last_index = current_index

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                        dynamic_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                        dynamic_gaussian.reset_opacity()

        # if iteration > 15200:
        #     print('Done\n')
        #     exit(0)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    os.makedirs(os.path.join(scene.model_path, "save_test"), exist_ok=True)
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0
                # origin_psnr = 0
                psnr_test = 0
                ssim_test = 0
                lpips_test = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    if idx == 0:
                        prev_viewpoint = None
                    else:
                        prev_viewpoint = config['cameras'][idx-2]
                    image = torch.clamp(renderFunc(viewpoint, prev_viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean()
                    image = image[None, ...]
                    gt_image = gt_image[None, ...]
                    
                    # origin_psnr += psnr(image, gt_image).mean()
                    psnr_test += m_psnr(image, gt_image)
                    ssim_test += m_ssim(image, gt_image)
                    lpips_test += m_lpips(image, gt_image)

                    if config['name'] == 'test':
                        torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_test", f"{viewpoint.image_name}.png"))
                
                # origin_psnr /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {format(l1_test, '.4f')} "
                      f"PSNR {format(psnr_test, '.4f')} SSIM {format(ssim_test, '.4f')} Lpips {format(lpips_test, '.4f')}")
                
                results[config['name']][iteration] = {
                    'psnr': psnr_test.item(),
                    'ssim': ssim_test.item(),
                    'lpips': lpips_test.item(),
                    'l1': l1_test.item()
                }

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    with open(os.path.join(args.model_path, 'results.json'), 'w') as wf:
        json.dump(results, wf, indent=4)

if __name__ == "__main__":
    # Set up command line argument parser
    ckpts = [3000, 9000, 15000, 25000, 30000]
    # ckpts = [200, 500, 1000, 1500, 2000]
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=ckpts)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=ckpts)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=ckpts)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--semantic", action='store_true', default=False)
    parser.add_argument("--unicycle", action="store_true", default=False)
    parser.add_argument("--uc_fit_iter", type=int, default=0)
    parser.add_argument("--uc_joint_opt", action="store_true", default=False)
    parser.add_argument("--optical", action="store_true", default=False)
    parser.add_argument("--adapt", action="store_true", default=False)
    parser.add_argument("--affine", action="store_true", default=False)
    parser.add_argument("--data_type", type=str, default="kitti360")
    parser.add_argument("--ignore_dynamic", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             args.semantic, args.optical, args.unicycle, args.uc_fit_iter, args.uc_joint_opt, args.adapt, args.affine, args.data_type, args.ignore_dynamic)

    # All done
    print("\nTraining complete.")
