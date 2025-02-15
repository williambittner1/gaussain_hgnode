# train_hgnode.py
# Changed from train_ode_18.py
# Gaussians now follow the double pendulum motion for all three nodes
# and each node is represented by 30 gaussians (total 90 per snapshot).

import lpips
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.transforms.functional import resize

import wandb
import os
import time
import torch
import glob
import tyro
import numpy as np
import cv2
import math
import copy
from random import randint, sample
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our scene and gaussian model classes.
from scene import Scene, GaussianModel
# Import our training configuration arguments.
from arguments import TrainingConfig
# Import our NeuralODE (assumed to be implemented in NeuralODE_11.py)
from models.hgnode import GraphNeuralODEHierarchical
# Import our gaussian renderer (the original version)
from gaussian_renderer import render

# --- Import double pendulum functions ---
from double_pendulum import (
    DoublePendulum2DPolarDynamics,
    generate_initial_conditions_polar_2d,
    generate_trajectory_2d,
    polar_to_cartesian_2d,
    generate_time_spans
)

# -----------------------------------------------------
# Helper functions for checkpointing and background initialization.
# -----------------------------------------------------
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(f"The directory {checkpoint_dir} does not exist.")
        return None, None
    gaussians_checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "chkpnt*.pth"))
    latest_gaussian_checkpoint = max(gaussians_checkpoint_files, key=os.path.getctime) if gaussians_checkpoint_files else None
    decoder_checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "decoder_chkpnt*.pth"))
    latest_decoder_checkpoint = max(decoder_checkpoint_files, key=os.path.getctime) if decoder_checkpoint_files else None
    return latest_gaussian_checkpoint, latest_decoder_checkpoint

def initialize_background(dataset):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return background

# -----------------------------------------------------
# Main Training Function
# -----------------------------------------------------
def train(dataset, opt, pipe):
    # Initialize the Scene without Gaussians.
    scene = Scene(dataset)
    background = initialize_background(dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Create or load the GaussianModel ---
    # IMPORTANT: We want 90 gaussians per scene snapshot (30 per node × 3 nodes).
    if dataset.manual_gaussians_bool:
        num_gaussians = 90  # 3 nodes × 30 gaussians each
        # For manual creation we will simply create dummy parameters.
        # (In practice you might want to create the 90 gaussians from a pointcloud.)
        positions = [torch.tensor([float(i), 0.0, 0.0], device=device) for i in range(num_gaussians)]
        scales = [torch.tensor([0.1, 0.3, 0.2], device=device) for _ in range(num_gaussians)]
        rotations = [torch.tensor([1.0, 0.2, 0.4, 0.0], device=device) for _ in range(num_gaussians)]
        opacities = [0.9 for _ in range(num_gaussians)]
        colors = [torch.tensor([1.0, 0.0, 0.0], device=device) for _ in range(num_gaussians)]  # Red Gaussians

        gaussians = GaussianModel(dataset.sh_degree, )
        for pos, scl, rot, op, col in zip(positions, scales, rotations, opacities, colors):
            gaussians.add_gaussian(pos, scl, rot, op, col)
        print(f"Number of Gaussians (manual): {gaussians._xyz.shape[0]}")
        scene.create_gaussians_manually(gaussians)
    else:
        gaussians = GaussianModel(dataset.sh_degree)
        checkpoint_dir = dataset.checkpoint_path
        os.makedirs(checkpoint_dir, exist_ok=True)
        gaussian_checkpoint, decoder_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if gaussian_checkpoint:
            scene.load_gaussians_from_checkpoint(gaussian_checkpoint, gaussians, opt)
        else:
            scene.initialize_gaussians_from_scene_info(gaussians, dataset)
    
    # For our purposes, we want to override the ground-truth motion.
    # Save the initial positions.
    gaussians.initial_xyz = gaussians.get_xyz.clone()
    initial_gaussians = gaussians.clone()
    
    # -----------------------------------------------------
    # Ground-Truth Generation using Double Pendulum Simulation
    # -----------------------------------------------------
    total_timesteps = 500
    K = 3
    framerate = 25
    timesteps_tensor = torch.arange(0, total_timesteps, device='cuda').float() / framerate  # [total_timesteps]

    # Double pendulum parameters
    L1 = 1.0
    L2 = 0.5
    g = 9.81
    dp_dynamics = DoublePendulum2DPolarDynamics(L1=L1, L2=L2, g=g).to(device)
    # Generate initial conditions for the double pendulum (for B=1)
    x0_dp = generate_initial_conditions_polar_2d(1, device=device, seed=2)  # shape: [1,4]
    # Generate time spans (using total_timesteps)
    t_span_dp, _ = generate_time_spans(total_timesteps / framerate, total_timesteps / framerate, total_timesteps, total_timesteps, device=device)
    dp_traj = generate_trajectory_2d(dp_dynamics, x0_dp, t_span_dp)  # shape: [1, total_timesteps, 4]
    # Convert 2D polar state to 3D Cartesian positions.
    # The function polar_to_cartesian_2d returns shape: [B, total_timesteps, 3, 3],
    # where the 3 in the third dimension corresponds to: anchor (index 0), mass1 (index 1), mass2 (index 2).
    dp_positions = polar_to_cartesian_2d(dp_traj, L1=L1, L2=L2)  # shape: [1, total_timesteps, 3, 3]
    
    # Now, for each timestep, we want to populate each node (anchor, mass1, mass2)
    # with 30 gaussians. That is, for each t, we form a new GaussianModel with 90 gaussians.
    # For each node i (0,1,2), take its base position from dp_positions[0, t, i, :],
    # add a small random offset to generate 30 positions.
    num_gaussians_per_node = 30
    total_gaussians = 3 * num_gaussians_per_node  # 90
    
    ground_truth_gaussians = []
    with torch.no_grad():
        for t in range(total_timesteps):
            new_positions_list = []
            for node in range(3):
                base_pos = dp_positions[0, t, node, :]  # [3]
                # Generate 30 random offsets (e.g., Gaussian noise with standard deviation 0.01)
                offsets = 0.01 * torch.randn(num_gaussians_per_node, 3, device=device)
                # Compute positions for this node:
                node_positions = base_pos.unsqueeze(0) + offsets  # [30,3]
                new_positions_list.append(node_positions)
            # Concatenate positions from all nodes: shape [90, 3]
            new_positions = torch.cat(new_positions_list, dim=0)
            # Create a clone of the original Gaussians and override the positions.
            gaussians_t = gaussians.clone()
            # Make sure that new_positions has the same number of gaussians.
            # (If gaussians originally had a different number, you might need to reinitialize.)
            if new_positions.shape[0] != gaussians_t._xyz.shape[0]:
                # Reinitialize the gaussian model with the correct number of gaussians.
                gaussians_t = GaussianModel(new_positions.shape[0], sh_degree=gaussians_t.sh_degree)
            gaussians_t._xyz.data = new_positions
            # Optionally, you can update other parameters (scaling, rotation, opacity) with random values.
            gaussians_t._scaling.data = torch.randn(new_positions.shape[0], 3, device=device)
            rand_quat = torch.randn(new_positions.shape[0], 4, device=device)
            gaussians_t._rotation.data = rand_quat / (rand_quat.norm(dim=1, keepdim=True) + 1e-8)
            gaussians_t._opacity.data = torch.randn(new_positions.shape[0], 1, device=device)
            gaussians_t.normalize_rotation()
            ground_truth_gaussians.append(gaussians_t)
    
    # Pre-render ground-truth images
    num_cameras_to_use = 3
    viewpoint_stack = scene.getTrainCameraObjects().copy()
    train_cameras = [viewpoint_stack[i] for i in [3, 16, 30]]
    train_camera_indices = [0, 1, 2]

    gt_images = {}
    print("Pre-rendering ground-truth images...")
    for cam_idx, viewpoint_cam in enumerate(train_cameras):
        gt_images[cam_idx] = []
        for t in tqdm(range(total_timesteps), desc=f"Rendering GT images for camera {cam_idx+1}/{num_cameras_to_use}"):
            gt_gaussians = ground_truth_gaussians[t]
            gt_render_pkg = render(viewpoint_cam, gt_gaussians, pipe, background)
            gt_image = gt_render_pkg['render']
            gt_image = gt_image.detach()
            gt_image.requires_grad = False
            gt_images[cam_idx].append(gt_image)
    print("Finished pre-rendering ground-truth images.")
    
    # Initialize Neural ODE and optimizer
    model = GraphNeuralODEHierarchical(particle_dim=10, 
                                       object_dim=10, 
                                       num_particles=total_gaussians,
                                       hidden_dim=128, 
                                       n_hidden_layers=4, 
                                       solver="dopri5", 
                                       rtol=1e-3, 
                                       atol=1e-5, 
                                       options={"max_num_steps": 200}, 
                                       max_objects=3, 
                                       device=device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    node_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    
    # Prepare initial state h0 (we use current positions and initial positions).
    # For simplicity, we concatenate initial_positions and initial_positions from initial_gaussians.
    initial_rotations = initial_gaussians.get_rotation.detach()
    initial_positions = initial_gaussians.get_xyz.detach()
    h0 = torch.cat([initial_positions, initial_positions], dim=1).to('cuda')  # [n_gaussians, 10]
    
    pred_gaussians = initial_gaussians.clone()

    # Initialize LPIPS model
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    N_sampled_timesteps = 45

    ode_progress_bar = tqdm(range(opt.ode_iterations), desc="Neural ODE Optimization")
    for iteration in range(opt.ode_iterations):
        model.train()
        noise_scale = 1e-2
        h0_perturbed = h0 + torch.randn_like(h0) * noise_scale
        
        num_timesteps_to_sample = min(K, N_sampled_timesteps)
        sampled_timesteps = torch.randperm(K, device='cuda')[:num_timesteps_to_sample]
        t_values = timesteps_tensor[sampled_timesteps]
        sorted_t_values, sorted_indices = torch.sort(t_values)
        sorted_sampled_timesteps = sampled_timesteps[sorted_indices]

        ode_predictions = model.get_ode_predictions(h0_perturbed, sorted_t_values)
        optimizer.zero_grad()
        total_loss = 0.0

        for idx in range(len(sorted_t_values)):
            pred_state = ode_predictions[:, idx, :]
            current_position = pred_state[:, :3]
            pred_gaussians.update_gaussians_from_predictions(torch.cat([current_position, initial_rotations], dim=1))
            t_idx = sorted_sampled_timesteps[idx].item()
            timestep_losses = []
            for cam_idx in train_camera_indices:
                viewpoint_cam = train_cameras[cam_idx]
                pred_render_pkg = render(viewpoint_cam, pred_gaussians, pipe, background)
                pred_image = pred_render_pkg['render']
                gt_image = gt_images[cam_idx][t_idx]
                pred_image_clamped = torch.clamp(pred_image, 0, 1)
                gt_image_clamped = torch.clamp(gt_image, 0, 1)
                mse_loss = F.mse_loss(pred_image_clamped, gt_image_clamped)
                pred_image_normalized = (pred_image_clamped * 2) - 1
                gt_image_normalized = (gt_image_clamped * 2) - 1
                ssim_loss = 1 - structural_similarity_index_measure(pred_image_clamped.unsqueeze(0), gt_image_clamped.unsqueeze(0))
                lpips_loss = loss_fn_lpips(pred_image_normalized.unsqueeze(0), gt_image_normalized.unsqueeze(0)).mean()
                total_timestep_loss = mse_loss * 100.0 + ssim_loss * 1.0 + lpips_loss * 1.0
                timestep_losses.append(total_timestep_loss)
            avg_timestep_loss = torch.stack(timestep_losses).mean()
            total_loss += avg_timestep_loss

        total_loss.backward(retain_graph=False)
        optimizer.step()
        node_scheduler.step()
        wandb.log({
            "MSE Loss": mse_loss.item(),
            "SSIM Loss": ssim_loss.item(),
            "LPIPS Loss": lpips_loss.item(),
            "Total Loss": total_loss.item(),
            "Iteration": iteration,
            "Current K": K
        }, step=iteration)
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({f"Gradient/{name}": param.grad.norm().item()}, step=iteration)
                wandb.log({f"Gradient Histogram/{name}": wandb.Histogram(param.grad.cpu().numpy())}, step=iteration)
        if iteration == 0: 
            loss_threshold = total_loss * 0.2
            print(f"\nloss_threshold = {loss_threshold}\n")
        with torch.no_grad():
            if total_loss.item() <= loss_threshold and K < total_timesteps - 1:
                K += 1
                loss_threshold *= K / (K - 1)
                print(f"\nAdded timestep {K} to contributing loss-timesteps.\n")
        wandb.log({"Neural ODE Loss": total_loss.item(), "Iteration": iteration, "Current K": K}, step=iteration)
        ode_evaluations = model.diffeq_solver.func.num_evaluations
        ode_progress_bar.set_postfix({"Loss": f"{total_loss.item():.7f}", "ODE Func Evaluations": ode_evaluations})
        model.diffeq_solver.func.num_evaluations = 0
        ode_progress_bar.update(1)
        if (iteration > 0 and iteration % 200 == 0) or iteration == 15:
            with torch.no_grad():
                camera_indices = [0, 1]
                render_and_log_to_wandb(
                    model, h0, timesteps_tensor, camera_indices, scene,
                    initial_gaussians, ground_truth_gaussians, pipe, background, iteration, gt_images, train_cameras, initial_rotations, K
                )
    ode_progress_bar.close()
    print("Training complete.")

def select_random_camera(scene):
    viewpoint_stack = scene.getTrainCameraObjects().copy()
    viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]
    return viewpoint_cam

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(f"The directory {checkpoint_dir} does not exist.")
        return None, None
    gaussians_checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "chkpnt*.pth"))
    latest_gaussian_checkpoint = max(gaussians_checkpoint_files, key=os.path.getctime) if gaussians_checkpoint_files else None
    decoder_checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "decoder_chkpnt*.pth"))
    latest_decoder_checkpoint = max(decoder_checkpoint_files, key=os.path.getctime) if decoder_checkpoint_files else None
    return latest_gaussian_checkpoint, latest_decoder_checkpoint


def render_and_log_to_wandb(
    model, h0, timesteps_tensor, camera_indices, scene,
    initial_gaussians, ground_truth_gaussians, pipe, background, iteration, gt_images, train_cameras, initial_rotations, current_K
):
    """
    Renders videos from specified camera views over all timesteps and logs them to WandB.
    Also computes per-pixel loss heatmaps based on the ground-truth image differences and logs them as videos.
    
    Args:
        current_K (int): The current value of K from the training loop.
    """
    total_timesteps = len(timesteps_tensor)
    sorted_t_values, _ = torch.sort(timesteps_tensor)

    # Initialize predicted Gaussians
    pred_gaussians = initial_gaussians.clone()

    # Get Neural ODE predictions for timesteps
    ode_predictions = model.get_ode_predictions(h0, sorted_t_values)

    # Initialize lists for WandB logging
    rendered_videos = []
    loss_heatmap_videos = []
    gt_videos = []

    # Loop over the specified camera views
    for cam_idx in camera_indices:
        viewpoint_cam = train_cameras[cam_idx]
        rendered_images, loss_heatmaps, gt_images_list = [], [], []

        # Loop over each timestep
        for idx, t in enumerate(sorted_t_values):
            # Predicted Gaussians at time t
            pred_state = ode_predictions[:, idx, :]  # Shape: (n_gaussians, 6)

            # Extract current position and initial position
            current_position = pred_state[:, :3]
            initial_position = pred_state[:, 3:6]  # This remains constant

            # Update pred_gaussians with the new position
            pred_gaussians.update_gaussians_from_predictions(torch.cat([current_position, initial_rotations], dim=1))

            # Render predicted image
            pred_render_pkg = render(viewpoint_cam, pred_gaussians, pipe, background)
            pred_image = pred_render_pkg["render"]
            pred_image_np = (pred_image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

            # Retrieve pre-rendered ground-truth image
            gt_image = gt_images[cam_idx][idx]
            gt_image_np = (gt_image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

            # Ensure both pred_image and gt_image are not empty before proceeding
            if pred_image_np.size == 0 or gt_image_np.size == 0:
                continue

            # Compute per-pixel loss before adding text
            per_pixel_loss = np.abs(pred_image_np.astype(np.float32) - gt_image_np.astype(np.float32))

            # For visualization, compute mean across color channels
            per_pixel_loss_np = per_pixel_loss.mean(axis=2)  # Shape: (H, W)

            # Normalize the loss to the range [0, 1]
            normalized_loss = per_pixel_loss_np / max(per_pixel_loss_np.max(), 1e-8)

            # Scale the normalized loss to [0, 255] for visualization in grayscale
            loss_heatmap_uint8 = (normalized_loss * 255).astype(np.uint8)

            # Convert the grayscale image to a 3-channel image
            loss_heatmap_rgb = cv2.cvtColor(loss_heatmap_uint8, cv2.COLOR_GRAY2RGB)

            # Overlay the timestep text on the predicted image
            timestep_text = f"Timestep: {idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size, _ = cv2.getTextSize(timestep_text, font, font_scale, thickness)
            text_x = pred_image_np.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
            text_y = text_size[1] + 10  # 10 pixels from the top edge

            # Determine text color based on K
            if idx <= current_K:
                color = (0, 255, 0)  # Green in BGR
            else:
                color = (255, 0, 0)  # Blue in BGR

            # Convert RGB to BGR for OpenCV
            pred_image_bgr = cv2.cvtColor(pred_image_np, cv2.COLOR_RGB2BGR)
            loss_heatmap_bgr = cv2.cvtColor(loss_heatmap_rgb, cv2.COLOR_RGB2BGR)
            gt_image_bgr = cv2.cvtColor(gt_image_np, cv2.COLOR_RGB2BGR)

            # Put the text on the images
            cv2.putText(pred_image_bgr, timestep_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(loss_heatmap_bgr, timestep_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(gt_image_bgr, timestep_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

            # Convert back to RGB
            pred_image_np = cv2.cvtColor(pred_image_bgr, cv2.COLOR_BGR2RGB)
            loss_heatmap_rgb = cv2.cvtColor(loss_heatmap_bgr, cv2.COLOR_BGR2RGB)
            gt_image_np = cv2.cvtColor(gt_image_bgr, cv2.COLOR_BGR2RGB)

            # Append the modified images
            rendered_images.append(pred_image_np)
            gt_images_list.append(gt_image_np)
            loss_heatmaps.append(loss_heatmap_rgb)

        # Convert lists to numpy arrays and transpose for WandB if non-empty
        if rendered_images:
            images_np = np.stack(rendered_images, axis=0).transpose(0, 3, 1, 2)  # (T, C, H, W)
            rendered_video = wandb.Video(images_np, fps=25, format="mp4")
            rendered_videos.append(rendered_video)

        if loss_heatmaps:
            loss_heatmaps_np = np.stack(loss_heatmaps, axis=0).transpose(0, 3, 1, 2)  # (T, C, H, W)
            loss_heatmap_video = wandb.Video(loss_heatmaps_np, fps=25, format="mp4")
            loss_heatmap_videos.append(loss_heatmap_video)

        if gt_images_list:
            gt_images_np = np.stack(gt_images_list, axis=0).transpose(0, 3, 1, 2)  # (T, C, H, W)
            gt_video = wandb.Video(gt_images_np, fps=25, format="mp4")
            gt_videos.append(gt_video)

    # Log all videos if they exist and contain data
    log_data = {}
    for i, rendered_video in enumerate(rendered_videos):
        log_data[f"Rendered_Video_Cam_{i}"] = rendered_video

    for i, loss_heatmap_video in enumerate(loss_heatmap_videos):
        log_data[f"Loss_Heatmap_Cam_{i}"] = loss_heatmap_video

    for i, gt_video in enumerate(gt_videos):
        log_data[f"GT_Video_Cam_{i}"] = gt_video

    # Log all collected data
    wandb.log(log_data, step=iteration)



if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    print("Starting optimization")
    wandb.init(project="5-Layer MLP",
               notes="Modified: Each double pendulum node now has 30 Gaussians (total 90 per snapshot)",
               force=True)
    train(dataset=config.model, opt=config.optimization, pipe=config.pipeline)
    print("Training complete.")
