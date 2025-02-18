# train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import lpips
import wandb
import math
from dataclasses import dataclass, field
from tqdm import tqdm
from random import randint
from PIL import Image
import numpy as np
import os
from typing import List, Dict
import imageio
import time
import random
import cv2

from scene import Scene, GaussianModel
from gaussian_renderer import render
from torchvision import transforms
from torch.utils.data import DataLoader

from encoders.explicit_encoder import ExplicitEncoder
from models.gnode import GraphNeuralODE
from double_pendulum import DoublePendulum2DPolarDynamics, generate_initial_conditions_polar_2d, polar_to_cartesian_2d

from dataset import PreRenderedGTDataset, ControlPointDataset

from torchdiffeq import odeint

from torch.amp import autocast, GradScaler


debug_render_dir = "training_renders"
if not os.path.exists(debug_render_dir):
    os.makedirs(debug_render_dir)



@dataclass
class OptimizationConfig:
    iterations: int = 20_000          # number of iterations to train the static gaussian model
    epochs: int = 30_000             # dynamic gaussian model training epochs

    total_timesteps: int = 100
    initial_timesteps: int = 2
    framerate: int = 25

    n_objects: int = 3

    ode_iterations: int = 50000
    position_lr_init: float = 0.00016
    position_lr_final: float = 1.6e-06
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    semantic_feature_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 1000
    densify_grad_threshold: float = 0.0002
    load_existing_gaussians: bool = True

@dataclass
class ModelConfig:
    sh_degree: int = 3
    dataset_name: str = "25_cams_1k_res"
    data_path: str = f"data/{dataset_name}"
    source_path: str = f"data/{dataset_name}"
    output_path: str = f"output/{dataset_name}"
    foundation_model: str = ""
    checkpoint_path: str = f"output/{dataset_name}"
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = False
    data_device: str = "cuda"
    eval: bool = False
    speedup: bool = False
    render_items: List[str] = field(default_factory=lambda: ["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"])
    manual_gaussians_bool: bool = False

@dataclass
class PipelineConfig:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = True

@dataclass
class ExperimentConfig:
    device: torch.device = torch.device("cuda")
    learning_rate: float = 1e-3
    num_epochs: int = 300000
    viz_iter: int = 1000
    eval_iter: int = 10
    train_duration: float = 1.0
    test_duration: float = 1.0
    eval_duration: float = 1.0
    train_samples_per_second: int = 10
    test_samples_per_second: int = 10
    eval_samples_per_second: int = 10
    num_time_samples_train: int = 10
    num_time_samples_test: int = 10
    num_samples_eval: int = 10
    num_initial_train_time_samples: int = 5
    train_samples_step: int = 1
    loss_threshold: float = 1e-5
    num_train_data_sequences: int = 2
    num_test_data_sequences: int = 1
    use_all_segments: bool = False
    stride: int = 10
    dynamics_type: str = "double_pendulum_cartesian_rigid"
    data_device: torch.device = torch.device("cuda")
    data_path: str = ModelConfig.data_path
    gt_mode: str = "python"  # or "blender"
    
    batch_size: int = 1
    num_sequences: int = 1    # number of sequences to simulate
    photometric_loss_length: int = 1
    num_train_cams: int = 3
    wandb_name: str = "1_seq_13_dynfeats_1_condfeats_test_iter_10"
    test_length: int = 100
    test_iter: int = 10

@dataclass
class Config:
    experiment: ExperimentConfig = ExperimentConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()



# def select_random_camera(scene):
#     cam_stack = scene.getTrainCameraObjects()
#     viewpoint_cam = cam_stack[randint(0, len(cam_stack) - 1)]
#     return viewpoint_cam


def simulate_dynamic_controlpoints_double_pendulum(dynamics_model, dt, total_timesteps, L1, L2, device, num_sequences=1, seed=0):
    """
    Simulate the double pendulum dynamics and output the control points state over time.
    
    The control points state tensor has shape [B, T, num_controlpoints, 7],
    where each control point is represented as [x, y, z, qw, qx, qy, qz].
    
    Args:
        dynamics_model (nn.Module): The double pendulum dynamics model.
        dt (float): Time step.
        total_timesteps (int): Number of timesteps.
        L1 (float): Length parameter 1.
        L2 (float): Length parameter 2.
        device (torch.device): The device to run on.
        num_sequences (int): Number of sequences (batches) to simulate.
        
    Returns:
        cp_state (torch.Tensor): Control points state with shape [B, T, num_controlpoints, 7].
    """
    # Generate initial conditions for num_sequences sequences
    # x0: shape [B, 4]
    x0 = generate_initial_conditions_polar_2d(num_sequences=num_sequences, device=device, seed=seed)
    
    # Create time span (shape [T])
    t_span = torch.linspace(0, dt * (total_timesteps - 1), total_timesteps, device=device, dtype=torch.float32)
    
    # Simulate the trajectories using odeint.
    # trajectory: shape [T, B, 4]
    trajectory = odeint(dynamics_model, x0, t_span)
    # Permute to shape [B, T, 4]
    trajectory = trajectory.permute(1, 0, 2)
    
    # Convert the polar state to 3D positions for the control points.
    # For example, polar_to_cartesian_2d returns positions for the control points.
    # Assume it returns shape: [B, T, num_controlpoints, 3]
    ctrl_positions = polar_to_cartesian_2d(trajectory, L1=L1, L2=L2)  # [B, T, num_controlpoints, 3]
    
    # Compute control rotations from the trajectory.
    # For the double pendulum, assume we have three control points corresponding to:
    # the anchor (base) and two masses.
    # Extract the relevant angles for each batch at each timestep.
    # theta1: [B, T] from trajectory component 0, and theta2: [B, T] from component 1.
    theta1 = trajectory[..., 0]  # [B, T]
    theta2 = trajectory[..., 1]  # [B, T]
    
    # Compute quaternions for each control point.
    # The first control point (anchor) uses an identity quaternion.
    B, T, _ , _ = ctrl_positions.shape
    q_anchor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, 1, 4)  # [B, T, 3, 4]
    
    def angle_to_quaternion(angle):
        # Returns [cos(angle/2), sin(angle/2), 0, 0]
        return torch.stack([torch.cos(angle/2), torch.sin(angle/2), torch.zeros_like(angle), torch.zeros_like(angle)], dim=-1)


    # Compute quaternion for mass1 using theta1.
    # Assume angle_to_quaternion accepts input of shape [B, T, 1] and returns [B, T, 4].
    q_mass1 = angle_to_quaternion(theta1.unsqueeze(-1))  # [B, T, 4]
    
    # Compute quaternion for mass2 using theta1 + theta2.
    q_mass2 = angle_to_quaternion((theta1 + theta2).unsqueeze(-1))  # [B, T, 4]
    
    # Stack the control rotations into a tensor of shape [B, T, num_controlpoints, 4]
    ctrl_rotations = torch.cat([q_anchor, q_mass1, q_mass2], dim=2)  # [B, T, 3, 4]
    
    # Concatenate positions and rotations to get the control points state.
    # Each control point: [x, y, z, qw, qx, qy, qz]
    cp_state = torch.cat([ctrl_positions, ctrl_rotations], dim=-1)  # [B, T, 3, 7]
    
    return cp_state


def visualize_gaussians(scene, gaussians, config, background, cam_stack, iteration=None):
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        gt_image = camera.original_image.cpu().numpy()
        wandb.log({f"render_{cam_idx}": wandb.Image(rendered_image)})
        wandb.log({f"gt_{cam_idx}": wandb.Image(gt_image)})
        print(f"Logged render and gt for camera {cam_idx}")


def train_static_gaussian_model(scene, config, iterations = 30000):

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene.initialize_gaussians_from_scene_info(scene.gaussians, config.model)

    scene.gaussians.training_setup_0(config.optimization)

    progress_bar = tqdm(range(iterations), desc=f"Training")
    for iteration in range(iterations):
        
        viewpoint_cam = select_random_camera(scene)

        gt_image = viewpoint_cam.original_image.permute(2,0,1)

        render_pkg = render(viewpoint_cam, scene.gaussians, config.pipeline, background)

        rendered_image = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor = render_pkg["viewspace_points"]

        loss = F.mse_loss(rendered_image, gt_image)


        loss.backward()

        scene.gaussians.update_learning_rate(iteration)
        scene.gaussians.optimizer.step()
        scene.gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 500 == 0:
            densification_step(
                iteration,
                config.optimization,
                scene.gaussians,
                render_pkg,
                visibility_filter,
                radii,
                viewspace_point_tensor,
                scene,
                config.model
                )


        # if iteration % 100 == 0:
        #     visualize_gaussians(scene, scene.gaussians, config, background, cam_stack=scene.getTrainCameraObjects()[0:5], iteration=iteration)

        progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
        progress_bar.update(1)
    
    return scene


def densification_step(iteration, opt, gaussians, render_pkg, visibility_filter, radii, viewspace_point_tensor, scene, dataset):
    """Perform densification and pruning steps during training."""
    if iteration < opt.densify_until_iter:
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(
                opt.densify_grad_threshold,
                0.005,
                scene.cameras_extent,
                size_threshold
            )
        if iteration % opt.opacity_reset_interval == 0 or (
            dataset.white_background and iteration == opt.densify_from_iter
        ):
            gaussians.reset_opacity()



def update_gaussians_from_batch(batch, gaussians_template, device):
    """
    For each sequence in the batch, update the gaussians for t0 and t1.
    
    Returns:
        gaussians_t0_list: List of gaussians updated at time 0.
        gaussians_t1_list: List of gaussians updated at time 1.
        batch_gt_xyz_cp: Ground-truth positions (B, T, N, 3).
        batch_gt_rot_cp: Ground-truth rotations (B, T, N, 4).
        batch_pseudo_gt: Pseudo GT (B, T, N, 7).
    """
    batch_gt_xyz_cp = batch["gt_xyz_cp"].to(device)
    batch_gt_rot_cp = batch["gt_rot_cp"].to(device)
    batch_pseudo_gt = batch["pseudo_gt"].to(device)
    num_batch_seqs = batch_gt_xyz_cp.shape[0]
    
    gaussians_t0_list = []
    gaussians_t1_list = []
    for b in range(num_batch_seqs):
        # Create and update t0
        g0 = gaussians_template.clone()
        g0.update_gaussians(batch_gt_xyz_cp[b, 0, :, :], batch_gt_rot_cp[b, 0, :, :])
        gaussians_t0_list.append(g0)
        
        # Clone g0 and update for t1
        g1 = g0.clone()
        g1.update_gaussians(batch_gt_xyz_cp[b, 1, :, :], batch_gt_rot_cp[b, 1, :, :])
        gaussians_t1_list.append(g1)
    
    return gaussians_t0_list, gaussians_t1_list, batch_gt_xyz_cp, batch_gt_rot_cp, batch_pseudo_gt



def log_wandb_video(epoch, config, model, z0_objects, t_span, current_segment_length,
                    gaussians_template, gt_xyz_cp, gt_rot_cp, cam_stack, background, log, split="train"):
    wandb_cam_stack = cam_stack[0:3]
    with torch.no_grad():
        tmp_gaussians_pred = gaussians_template.clone()
        tmp_gaussians_gt = gaussians_template.clone()
        z_traj = model(z0_objects, t_span)

        for b in range(min(4, gt_xyz_cp.shape[0])):
            combined_video_frames_dict = {cam_idx: [] for cam_idx in range(len(wandb_cam_stack))}

            for t in range(current_segment_length):
                # Update predicted gaussians from network output.
                tmp_gaussians_pred.xyz_cp = z_traj[b, t, :, :3]
                tmp_gaussians_pred.rot_cp = z_traj[b, t, :, 3:7]
                tmp_gaussians_pred.update_gaussians_from_controlpoints()

                # Update GT gaussians.
                tmp_gaussians_gt.xyz_cp = gt_xyz_cp[b, t, :, :]
                tmp_gaussians_gt.rot_cp = gt_rot_cp[b, t, :, :]
                tmp_gaussians_gt.update_gaussians_from_controlpoints()
                
                for cam_idx, cam in enumerate(wandb_cam_stack):
                    # Render predicted image.
                    render_pkg_pred = render(cam, tmp_gaussians_pred, config.pipeline, background)
                    pred_image = (render_pkg_pred["render"] * 255).clamp(0, 255).to(torch.uint8)
                    
                    # Render ground-truth image.
                    render_pkg_gt = render(cam, tmp_gaussians_gt, config.pipeline, background)
                    gt_image = (render_pkg_gt["render"] * 255).clamp(0, 255).to(torch.uint8)
                    
                    # Convert images to numpy arrays.
                    pred_np = pred_image.detach().cpu().numpy()
                    gt_np = gt_image.detach().cpu().numpy()
                    
                    # Insert a one-pixel white line between the predicted and GT images.
                    white_line = np.ones((pred_np.shape[0], 1, pred_np.shape[2]), dtype=np.uint8) * 255
                    combined_frame = np.concatenate([pred_np, white_line, gt_np], axis=1)
                    
                    # Overlay the frame number in the top-right corner.
                    frame_text = f"Frame: {t}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7  # smaller font scale
                    thickness = 2
                    text_size, _ = cv2.getTextSize(frame_text, font, font_scale, thickness)
                    text_w, text_h = text_size
                    pos = (combined_frame.shape[1] - text_w - 10, text_h + 10)
                    # First, draw a black outline for better contrast.
                    cv2.putText(combined_frame, frame_text, pos, font, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
                    # Then, draw the white text.
                    cv2.putText(combined_frame, frame_text, pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    
                    combined_video_frames_dict[cam_idx].append(combined_frame)
            # For each camera, stack frames and log the video to wandb.
            for cam_idx in range(len(wandb_cam_stack)):
                combined_video = np.stack(combined_video_frames_dict[cam_idx]).astype(np.uint8)
                log_name = f"{split}_video_sequence_{b}_cam_{cam_idx}"
                log[log_name] = wandb.Video(combined_video, fps=5)
    print(f"[Epoch {epoch}] wandb {split} video logged.")







def update_pseudo_gt(cp_dataset, config, device, gaussians_template, encoder, model, current_segment_length):
    """
    Update the pseudo–3D ground truth for all sequences in the dataset.
    This function loops over the entire cp_dataset (which may have many sequences)
    using a DataLoader and computes model predictions, which are then used to
    update the pseudo ground truth stored in cp_dataset.
    """
    print("[update_pseudo_gt] Updating pseudo ground truth for all sequences...")
    pseudo_gt_xyz_list = []
    pseudo_gt_rot_list = []
    # Use a DataLoader to iterate over the dataset
    pseudo_loader = DataLoader(cp_dataset, batch_size=config.experiment.batch_size, shuffle=False)
    segment_duration = current_segment_length / config.optimization.framerate
    t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
    for batch in pseudo_loader:
        batch_gt_xyz_cp = batch["gt_xyz_cp"].to(device)
        batch_gt_rot_cp = batch["gt_rot_cp"].to(device)
        num_batch_seqs = batch_gt_xyz_cp.shape[0]
        gaussians_t0_list = []
        gaussians_t1_list = []
        for b in range(num_batch_seqs):
            g0 = gaussians_template.clone()
            g0.update_gaussians(batch_gt_xyz_cp[b, 0, :, :], batch_gt_rot_cp[b, 0, :, :])
            gaussians_t0_list.append(g0)
            g1 = g0.clone()
            g1.update_gaussians(batch_gt_xyz_cp[b, 1, :, :], batch_gt_rot_cp[b, 1, :, :])
            gaussians_t1_list.append(g1)
        z0_objects = encoder(gaussians_t0_list, gaussians_t1_list)
        z_traj = model(z0_objects, t_span).detach()  # [B, T, N, D]
        pseudo_gt_xyz = z_traj[:, :, :, :3]
        pseudo_gt_rot = z_traj[:, :, :, 3:7]
        pseudo_gt_xyz_list.append(pseudo_gt_xyz.cpu())
        pseudo_gt_rot_list.append(pseudo_gt_rot.cpu())
    new_pseudo_gt_xyz = torch.cat(pseudo_gt_xyz_list, dim=0)
    new_pseudo_gt_rot = torch.cat(pseudo_gt_rot_list, dim=0)
    cp_dataset.set_pseudo_gt(new_pseudo_gt_xyz, new_pseudo_gt_rot)
    print("[update_pseudo_gt] Pseudo ground truth updated.")

def process_batch(batch, config, device, gaussians_template, encoder, model, t_span, cam_stack, background, current_segment_length):
    """
    Process a single batch: update gaussians, encode, predict, and compute losses.
    
    Returns:
        batch_loss: The computed loss for the batch.
        timing_info: A dictionary with timing info (optional).
    """
    timing_info = {}
    start = time.time()
    (gaussians_t0_list, gaussians_t1_list, batch_gt_xyz_cp, 
     batch_gt_rot_cp, batch_pseudo_gt) = update_gaussians_from_batch(batch, gaussians_template, device)
    timing_info['gaussian_update_time'] = time.time() - start


    # Encode latent state z0
    start = time.time()
    z0_objects = encoder(gaussians_t0_list, gaussians_t1_list) # [B, T, N, D], D = 13 dynamic [xyz, rot, vel, omega] + 1 static [object_id]
    timing_info['encoder_time'] = time.time() - start


    # Predict the trajectory z_traj using GNODE
    start = time.time()
    model.func.nfe = 0
    z_traj = model(z0_objects, t_span)
    timing_info['model_prediction_time'] = time.time() - start


    # Compute losses
    photometric_loss_length = config.experiment.photometric_loss_length
    pseudo_loss_length = max(0, current_segment_length - photometric_loss_length)

    # Pseudo-3D loss
    # start = time.time()
    loss_pseudo3d = F.mse_loss(
        z_traj[:, :pseudo_loss_length, :, :7],
        batch_pseudo_gt[:, :pseudo_loss_length, :, :7]
    )
    # timing_info['loss_pseudo3d_time'] = time.time() - start


    # Photometric loss
    start = time.time()
    loss_photo = 0.0
    photo_count = 0
    num_batch_seqs = batch_gt_xyz_cp.shape[0]
    
    tmp_gaussians_pred = gaussians_template.clone()
    tmp_gaussians_gt = gaussians_template.clone()
    for t in range(pseudo_loss_length, current_segment_length):
        for seq in range(num_batch_seqs):
            # Build predicted gaussians from network output
            
            tmp_gaussians_pred.update_gaussians(z_traj[seq, t, :, :3],
                                                z_traj[seq, t, :, 3:7])
            # Build GT gaussians from batch data
            
            tmp_gaussians_gt.update_gaussians(batch_gt_xyz_cp[seq, t, :, :],
                                              batch_gt_rot_cp[seq, t, :, :])
            # Choose a random camera for rendering (or loop over cam_stack if desired)
            for cam_idx in range(1, 4):
                # viewpoint_cam = random.choice(cam_stack)
                viewpoint_cam = cam_stack[cam_idx]
                # Render predicted image
                render_pkg_pred = render(viewpoint_cam, tmp_gaussians_pred, config.pipeline, background)
                pred_rendered_image = render_pkg_pred["render"]
                
                # Render GT image (with no gradient)
                with torch.no_grad():
                    render_pkg_gt = render(viewpoint_cam, tmp_gaussians_gt, config.pipeline, background)
                    gt_rendered_image = render_pkg_gt["render"]
                
                loss_i = F.mse_loss(pred_rendered_image, gt_rendered_image)
                loss_photo += loss_i
                photo_count += 1

    loss_photo = loss_photo / photo_count
    batch_loss = (loss_pseudo3d * pseudo_loss_length + loss_photo * photometric_loss_length) / current_segment_length
    timing_info['render_time'] = time.time() - start
    # if timing_info['render_time'] > 0.1:
    #     print(f"Render time high: {timing_info['render_time']}")

    return batch_loss, timing_info




def prerender_ground_truth_from_controlpoints(
    gt_xyz_cp: torch.Tensor,       # shape: [num_sequences, T, num_controlpoints, 3]
    gt_rot_cp: torch.Tensor,       # shape: [num_sequences, T, num_controlpoints, 4]
    viewpoint_stack: List,         # list of camera objects; each should have e.g. a camera_id attribute (or you can use its index)
    config,                        # the config object (for pipeline and other settings)
    background: torch.Tensor,
    gaussians_template,            # your template Gaussian model (to be cloned and updated)
    save_dir: str = "prerendered_gt",
    image_format: str = "png"
):
    # Ensure the base directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    num_sequences, total_timesteps, _, _ = gt_xyz_cp.shape
    train_count = int(0.9 * num_sequences)
    # A transform to convert a tensor (assumed in [0,1]) to a PIL image
    transform_to_uint8 = transforms.Compose([
        transforms.Lambda(lambda x: (x.clamp(0, 1) * 255).to(torch.uint8)),
        transforms.ToPILImage()
    ])
    
    # Loop over each sequence
    for seq in tqdm(range(num_sequences), desc="Pre-rendering GT images"):
        # Decide the split based on sequence index
        split = "train" if seq < train_count else "test"
        # Create folder for this scene/sequence
        scene_folder = os.path.join(save_dir, split, f"scene_{seq:03d}")
        os.makedirs(scene_folder, exist_ok=True)
        
        # For each camera, create a subfolder
        for cam_idx, cam in enumerate(viewpoint_stack):
            cam_folder = os.path.join(scene_folder, f"cam_{cam_idx:03d}")
            os.makedirs(cam_folder, exist_ok=True)
        
        # Loop over timesteps
        for t in range(total_timesteps):
            # For each sequence, clone your template and update with GT control points
            temp_gaussians = gaussians_template.clone()
            # Here we assume update_gaussians takes tensors of shape [num_controlpoints, 3] and [num_controlpoints, 4]
            temp_gaussians.update_gaussians(gt_xyz_cp[seq, t], gt_rot_cp[seq, t])
            
            # Render from each camera in the viewpoint stack
            for cam_idx, cam in enumerate(viewpoint_stack):
                with torch.no_grad():
                    render_pkg = render(cam, temp_gaussians, config.pipeline, background)
                    rendered_image = render_pkg["render"].detach().cpu()
                # Convert the image tensor to PIL
                pil_img = transform_to_uint8(rendered_image)
                # Construct filename; e.g. render_{cam_idx}_timestep_{t:05d}.jpg
                filename = f"render_{cam_idx:03d}_timestep_{t:05d}.{image_format}"
                save_path = os.path.join(save_dir, split, f"scene_{seq:03d}", f"cam_{cam_idx:03d}", filename)
                pil_img.save(save_path)


def simulate_cp_set(config, device, num_sequences: int, seed_offset: int = 0):
    """
    Simulate control point data (for either train or test) using the double pendulum dynamics.

    Args:
        config: configuration object.
        device: torch.device to run on.
        num_sequences (int): number of sequences to simulate.
        seed_offset (int): an offset added to a base seed to vary the simulation.

    Returns:
        gt_xyz_cp: Tensor of shape [num_sequences, T, num_controlpoints, 3]
        gt_rot_cp: Tensor of shape [num_sequences, T, num_controlpoints, 4]
    """
    dynamics_model = DoublePendulum2DPolarDynamics(L1=2.0, L2=1.0)
    # Adjust seed by adding the seed_offset.
    seed = 42 + seed_offset
    print(f"Simulating dynamic object nodes for {'TEST' if seed_offset else 'TRAIN'} set with seed {seed}...")
    gt_cp_state = simulate_dynamic_controlpoints_double_pendulum(
        dynamics_model=dynamics_model,
        dt=0.05,
        total_timesteps=config.optimization.total_timesteps,
        L1=2.0,
        L2=2.0,  # or use L2=1.0 if that's what you intend
        device=device,
        num_sequences=num_sequences,
        seed=seed
    )
    gt_xyz_cp = gt_cp_state[..., :3]
    gt_rot_cp = gt_cp_state[..., 3:]
    return gt_xyz_cp, gt_rot_cp


def test_validation(model, encoder, test_cp_dataset, device, current_segment_length, config, gaussians_template):
    """
    Evaluate the model on a test dataset of control points using the ground truth.
    """
    model.eval()
    t_span = torch.linspace(
        0,
        current_segment_length / config.optimization.framerate,
        current_segment_length,
        device=device,
        dtype=torch.float32
    )
    test_loader = DataLoader(test_cp_dataset, batch_size=config.experiment.batch_size, shuffle=False)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Update gaussians from batch and retrieve ground-truth control points.
            (gaussians_t0_list, gaussians_t1_list, batch_gt_xyz_cp,
             batch_gt_rot_cp, _) = update_gaussians_from_batch(batch, gaussians_template, device)
            # Encode latent state z0.
            z0_objects = encoder(gaussians_t0_list, gaussians_t1_list)
            t_test_span = torch.linspace(0, config.experiment.test_length / config.optimization.framerate, config.experiment.test_length, device=device, dtype=torch.float32)
            # Get predicted trajectory from model.
            z_traj = model(z0_objects, t_test_span)
            # Build the ground-truth tensor by concatenating positions and rotations.
            ground_truth = torch.cat([batch_gt_xyz_cp, batch_gt_rot_cp], dim=-1)  # shape: [B, T, N, 7]
            
            loss_gt = F.mse_loss(
                z_traj[:, :config.experiment.test_length, :, :7],
                ground_truth[:, :config.experiment.test_length, :, :]
            )
            total_loss += loss_gt.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    # print(f"Test set validation loss: {avg_loss:.6f}")
    return avg_loss




def select_random_camera(cam_stack):
    cam_idx = torch.randint(0, len(cam_stack), (1,)).item()
    return cam_stack[cam_idx]


def train():
    """
    Main training function.
    """

    config = Config()
    wandb.init(project="debugging", name=config.experiment.wandb_name)
    device = config.experiment.data_device
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")


    ##########################################
    # 0. Set up scene and empty gaussians
    ##########################################

    scene = Scene(config=config, dataset=None)
    scene.gaussians = GaussianModel(sh_degree=3)



    ##########################################
    # 1. Load or train static gaussian model 
    ##########################################

    if config.optimization.load_existing_gaussians:
        gaussian_checkpoint = os.path.join(config.model.checkpoint_path, f"chkpnt{config.optimization.iterations}.pth")
        scene.load_gaussians_from_checkpoint(gaussian_checkpoint, scene.gaussians, config.optimization)

    else:
        scene = train_static_gaussian_model(scene, config, iterations=config.optimization.iterations)
        os.makedirs(config.model.checkpoint_path, exist_ok=True)
        torch.save(scene.gaussians.capture(), os.path.join(config.model.checkpoint_path, f"chkpnt{config.optimization.iterations}.pth"))


    gaussians_t0 = scene.gaussians.clone()
    gaussians_t0.training_setup_t(config.optimization)


    # visualize_gaussians(scene, gaussians_t0, config, background, cam_stack=scene.getTrainCameraObjects()[0:3])



    ##########################################
    # 2. Cluster and initialize control points
    ##########################################

    gaussians_t0.initialize_controlpoints(config.optimization.n_objects)
    gaussians_t0.update_gaussians_from_controlpoints()

    gaussians_t1 = gaussians_t0.clone()
    gaussians_t1.training_setup_t(config.optimization)

    gaussians_template = gaussians_t0.clone()  

    ##########################################
    # 3. Simulate points
    ##########################################

    if config.experiment.gt_mode == "python":
        
        # control point training set:
        gt_xyz_cp, gt_rot_cp = simulate_cp_set(config, device, config.experiment.num_sequences, seed_offset=0)

        # control point test set:
        gt_xyz_cp_test, gt_rot_cp_test = simulate_cp_set(config, device, config.experiment.num_test_data_sequences, seed_offset=1000)

        train_cp_dataset = ControlPointDataset(gt_xyz_cp, gt_rot_cp)
        train_dataloader = DataLoader(train_cp_dataset, batch_size=config.experiment.batch_size, shuffle=False)

        test_cp_dataset = ControlPointDataset(gt_xyz_cp_test, gt_rot_cp_test)
        test_dataloader = DataLoader(test_cp_dataset, batch_size=config.experiment.batch_size, shuffle=False)

    # Assume you have a list of cameras (e.g., obtained from your scene)
    viewpoint_stack = scene.getTrainCameraObjects()  # or another function that returns the desired cameras
    viewpoint_stack = viewpoint_stack[0:5]
    # Call the prerendering function:
    # prerender_ground_truth_from_controlpoints(
    #     gt_xyz_cp=gt_xyz_cp,
    #     gt_rot_cp=gt_rot_cp,
    #     viewpoint_stack=viewpoint_stack,
    #     config=config,
    #     background=background,
    #     gaussians_template=gaussians_template,  # or whichever template you use
    #     save_dir=f"data/{config.model.dataset_name}/python_gt_{config.experiment.num_sequences}",
    #     image_format="png"
    # )

        
    # Create a dataset & dataloader from the simulated GT control point data.
    cp_dataset = ControlPointDataset(gt_xyz_cp, gt_rot_cp)
    # prerendered_dataset = PrerenderedDataset(config, scene, background, gaussians_template, viewpoint_stack)
    dataloader = DataLoader(cp_dataset, batch_size=config.experiment.batch_size, shuffle=False)
    num_batches = len(dataloader)

    ##########################################
    # 4. Setup Dynamic Model    
    ##########################################

    encoder = ExplicitEncoder()

    
    model = GraphNeuralODE(
        node_feature_dim=13,
        node_conditioning_dim=1,
        hidden_dim=256,
        n_hidden_layers=4,
        # solver="rk4",
        # options={"step_size": 0.04},
        solver="dopri5",
        rtol=1e-2,
        atol=1e-4,
        options={"max_num_steps": 200},
        device=device
    )

    optimizer = optim.Adam(
        list(model.parameters()),
        lr=config.experiment.learning_rate
    )

    loss_fn = nn.MSELoss(reduction="mean")

    scaler = GradScaler()


    cam_stack = scene.getTrainCameraObjects() 
    num_cams = len(cam_stack)

    current_segment_length = config.optimization.initial_timesteps


    # Initialize Pseudo-GT

    pseudo_gt_xyz = gt_xyz_cp[:, :1]  # [B, T=1, N, 3]
    pseudo_gt_rot = gt_rot_cp[:, :1]  # [B, T=1, N, 4]

    cp_dataset.set_pseudo_gt(pseudo_gt_xyz, pseudo_gt_rot)

    test_cp_dataset.set_pseudo_gt(pseudo_gt_xyz, pseudo_gt_rot)

    ##########################################
    # 7. Train Dynamic Model (GNODE)
    ##########################################

    # Train-Loop
    epoch_bar = tqdm(range(config.optimization.epochs), desc=f"Training")
    for epoch in epoch_bar:
        
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        log = {}

        segment_duration = current_segment_length / config.optimization.framerate
        t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
        

        for batch_idx, batch in enumerate(dataloader):

            # with autocast(device_type="cuda"):
            batch_loss, timing_info = process_batch(batch, 
                                                    config, 
                                                    device, 
                                                    gaussians_template, 
                                                    encoder, 
                                                    model, 
                                                    t_span, 
                                                    cam_stack, 
                                                    background, 
                                                    current_segment_length)

            # optimizer.zero_grad()
            # scaler.scale(batch_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            start = time.time()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            timing_info['backward_time'] = time.time() - start

            
            epoch_loss += batch_loss

        epoch_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        epochs_per_sec = 1 / epoch_time
        log.update({
            "loss": epoch_loss.item(),
            "epoch_loss": epoch_loss.item(),
            "nfe": model.func.nfe,
            "segment_length": current_segment_length,
            "epoch": epoch,
            "epochs_per_sec": epochs_per_sec,
            "total_time": epoch_time,
        })
        log.update(timing_info)


        # Test Evaluation
        if epoch % config.experiment.test_iter == 0:
            test_loss = test_validation(model, encoder, test_cp_dataset, device, current_segment_length, config, gaussians_template)
            log.update({"test_loss": test_loss})
            
        # Test Video
        if epoch % 500 == 0:
            first_batch = next(iter(test_dataloader))
            (gaussians_t0_list, gaussians_t1_list, batch_gt_xyz_cp, batch_gt_rot_cp, _) = update_gaussians_from_batch(first_batch, gaussians_template, device)
            z0_objects = encoder(gaussians_t0_list, gaussians_t1_list)

            t_test_span = torch.linspace(0, config.experiment.test_length / config.optimization.framerate, config.experiment.test_length, device=device, dtype=torch.float32)
            log_wandb_video(epoch, config, model, z0_objects, t_test_span, len(t_test_span),
                            gaussians_template, batch_gt_xyz_cp, batch_gt_rot_cp, cam_stack, background, log, split="test")


            

        # Train Video
        if epoch % 100 == 0:
            first_batch = next(iter(dataloader))
            (gaussians_t0_list, gaussians_t1_list, batch_gt_xyz_cp, batch_gt_rot_cp, _) = update_gaussians_from_batch(first_batch, gaussians_template, device)
            z0_objects = encoder(gaussians_t0_list, gaussians_t1_list)

            log_wandb_video(epoch, config, model, z0_objects, t_span, current_segment_length,
                            gaussians_template, batch_gt_xyz_cp, batch_gt_rot_cp, cam_stack, background, log, split="train")



        # Increment segment length and update Pseudo-GT
        if epoch > 0 and epoch % 200 == 0:
            update_pseudo_gt(cp_dataset, config, device, gaussians_template, encoder, model, current_segment_length)
            current_segment_length += 1
            segment_duration = current_segment_length / config.optimization.framerate
            t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
        


        wandb.log(log, step=epoch)


        epoch_bar.set_postfix({
            'Loss': f'{batch_loss.item():.7f}',
            'nfe': model.func.nfe,
            'seg_len': current_segment_length,
            'it/s': epochs_per_sec,
            #'test_loss': log['test_loss']
        })





if __name__ == "__main__":
    train()