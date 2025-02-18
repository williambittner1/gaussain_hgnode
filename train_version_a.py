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

from scene import Scene, GaussianModel
from gaussian_renderer import render
from torchvision import transforms
from torch.utils.data import DataLoader

from encoders.explicit_encoder import ExplicitEncoder
from models.gnode import GraphNeuralODE
from double_pendulum import DoublePendulum2DPolarDynamics, generate_initial_conditions_polar_2d, polar_to_cartesian_2d

from dataset import PreRenderedGTDataset, ControlPointDataset

from torchdiffeq import odeint


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
    batch_size: int = 1
    use_all_segments: bool = False
    stride: int = 10
    dynamics_type: str = "double_pendulum_cartesian_rigid"
    data_device: torch.device = torch.device("cuda")
    data_path: str = ModelConfig.data_path
    gt_mode: str = "python"  # or "blender"
    num_sequences: int = 1    # number of sequences to simulate
    photometric_loss_length: int = 1
    num_train_cams: int = 3


@dataclass
class Config:
    experiment: ExperimentConfig = ExperimentConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()



def select_random_camera(scene):
    cam_stack = scene.getTrainCameraObjects()
    viewpoint_cam = cam_stack[randint(0, len(cam_stack) - 1)]
    return viewpoint_cam


def simulate_dynamic_controlpoints_double_pendulum(dynamics_model, dt, total_timesteps, L1, L2, device, num_sequences=1):
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
    x0 = generate_initial_conditions_polar_2d(num_sequences=num_sequences, device=device, seed=42)
    
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



def train():
    """
    Main training function.
    """

    config = Config()
    wandb.init(project="debugging")
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



    ##########################################
    # 3. Simulate points
    ##########################################

    if config.experiment.gt_mode == "python":
        
        dynamics_model = DoublePendulum2DPolarDynamics(L1=2.0, L2=1.0)

        print("Simulating dynamic object nodes...")

        gt_cp_state = simulate_dynamic_controlpoints_double_pendulum(dynamics_model=dynamics_model,
                                                                     dt=0.05, 
                                                                     total_timesteps=config.optimization.total_timesteps, 
                                                                     L1=2.0, 
                                                                     L2=1.0, 
                                                                     device=device,
                                                                     num_sequences=config.experiment.num_sequences) # torch.Size([B, T, N_o, 7])
        
        gt_xyz_cp, gt_rot_cp = gt_cp_state[..., :3], gt_cp_state[..., 3:] # shape: [B, T, N_o, 3], [B, T, N_o, 4]

        """
        Update protocol to update gaussians_t0 from gt control points :
            gaussians_t0.xyz_cp = gt_xyz_cp[b, t]  # shape: [N, 3]
            gaussians_t0.rot_cp = gt_rot_cp[b, t]  # shape: [N, 4]
            gaussians_t0.update_gaussians_from_controlpoints()
        """


        
    # Create a dataset & dataloader from the simulated GT control point data.
    cp_dataset = ControlPointDataset(gt_xyz_cp, gt_rot_cp)
    dataloader = DataLoader(cp_dataset, batch_size=config.experiment.batch_size, shuffle=False)

    ##########################################
    # 4. Setup Dynamic Model    
    ##########################################

    encoder = ExplicitEncoder()

    
    model = GraphNeuralODE(
        node_dim=13,
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


    cam_stack = scene.getTrainCameraObjects() 
    num_cams = len(cam_stack)

    current_segment_length = config.optimization.initial_timesteps


    # Initialize Pseudo-GT
  
    pseudo_gt_xyz_list = []
    pseudo_gt_rot_list = []

    for i in range(config.experiment.num_sequences):
                
        # Get positions and rotations for timestep t0
        pseudo_gt_xyz_t0 = gt_xyz_cp[i, 0, :, :].squeeze(0)  # [N, 3]
        pseudo_gt_rot_t0 = gt_rot_cp[i, 0, :, :].squeeze(0)   # [N, 4]
        
        # Get positions and rotations for timestep t1
        pseudo_gt_xyz_t1 = gt_xyz_cp[i, 1, :, :].squeeze(0)  # [N, 3]
        pseudo_gt_rot_t1 = gt_rot_cp[i, 1, :, :].squeeze(0)   # [N, 4]
  
        # Stack timesteps for this batch
        batch_xyz = torch.stack([pseudo_gt_xyz_t0, pseudo_gt_xyz_t1], dim=0)  # [T=2, N, 3]
        batch_rot = torch.stack([pseudo_gt_rot_t0, pseudo_gt_rot_t1], dim=0)  # [T=2, N, 4]
        
        pseudo_gt_xyz_list.append(batch_xyz)
        pseudo_gt_rot_list.append(batch_rot)
    
    # Stack all batches
    pseudo_gt_xyz = torch.stack(pseudo_gt_xyz_list, dim=0)  # [B, T, N, 3]
    pseudo_gt_rot = torch.stack(pseudo_gt_rot_list, dim=0)  # [B, T, N, 4]
    
    # Concatenate xyz and rotation along last dimension
    pseudo_gt = torch.cat([pseudo_gt_xyz, pseudo_gt_rot], dim=-1)  # [B, T, N, 7]



    ##########################################
    # 7. Train Dynamic Model (GNODE)
    ##########################################

    # Train-Loop
    epoch_bar = tqdm(range(config.optimization.epochs), desc=f"Training")
    for epoch in epoch_bar:
        
        torch.cuda.empty_cache()

        epoch_start_time = time.time()

        epoch_loss = 0.0
        loss_list = []
        log = {}


        segment_duration = current_segment_length / config.optimization.framerate
        t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
        

        num_batches = len(dataloader)
        epoch_loss = 0.0

        # Loop over the dataloader batches (each batch is a full sequence)
        for batch_idx, batch in enumerate(dataloader):


            # Each batch is a dictionary with keys "gt_xyz_cp" and "gt_rot_cp"
            # Their shapes: [B, T, num_controlpoints, 3] and [B, T, num_controlpoints, 4]
            batch_gt_xyz_cp = batch["gt_xyz_cp"].to(device)  # [B, T, N, 3]
            batch_gt_rot_cp = batch["gt_rot_cp"].to(device)  # [B, T, N, 4]

            num_batch_seqs = batch_gt_xyz_cp.shape[0]
            batch_loss = 0.0

            if epoch <= 300:
                gaussians_t1_list = []
                gaussians_t0_list = []
                
                # Retrieve gaussians_t0 and gaussians_t1 from batch
                


                gaussian_retrieval_start_time = time.time()
                # retrieve gaussians_t0 and gaussians_t1 for each sequence in batch
                for b in range(num_batch_seqs):

                    
                    # set gaussians_t0 from simulation (or data)
                    gaussians_t0 = gaussians_t0.clone()
                    gaussians_t0.xyz_cp = batch_gt_xyz_cp[b, 0, :, :].squeeze(0)  # [N, 3]
                    gaussians_t0.rot_cp = batch_gt_rot_cp[b, 0, :, :].squeeze(0)   # [N, 4]
                    gaussians_t0.update_gaussians_from_controlpoints()
                    gaussians_t0_list.append(gaussians_t0)

                    
                    # set gaussians_t1 from simulation (or data)
                    gaussians_t1 = gaussians_t0.clone()
                    gaussians_t1.xyz_cp = batch_gt_xyz_cp[b, 1, :, :].squeeze(0)  # [N, 3]
                    gaussians_t1.rot_cp = batch_gt_rot_cp[b, 1, :, :].squeeze(0)   # [N, 4]
                    gaussians_t1.update_gaussians_from_controlpoints()
                    gaussians_t1_list.append(gaussians_t1)

                gaussian_retrieval_end_time = time.time()
                gaussian_retrieval_time = gaussian_retrieval_end_time - gaussian_retrieval_start_time
                # print(f"Gaussian retrieval time: {gaussian_retrieval_time:.5f} seconds")

                
                encoder_start_time = time.time()
                z0_objects = encoder(gaussians_t0_list, gaussians_t1_list) # shape: [B, N, 13]
                encoder_end_time = time.time()
                encoder_time = encoder_end_time - encoder_start_time
                # print(f"Encoder time: {encoder_time:.5f} seconds")
            
            model_prediction_start_time = time.time()
            model.func.nfe = 0
            z_traj = model(z0_objects, t_span) # shape: [B, T, N, D] with D = 13 [pos, quat, vel, omega]

            model_prediction_end_time = time.time()
            model_prediction_time = model_prediction_end_time - model_prediction_start_time
            # print(f"Model prediction time: {model_prediction_time:.5f} seconds")


            render_start_time = time.time()

            for seq in range(num_batch_seqs):

                num_timesteps = 0
                sequence_loss = 0.0


                for t in range(current_segment_length):

                    # Use pseudo 3d ground-truth for all timesteps except the last photometric loss length timesteps
                    if t < current_segment_length - config.experiment.photometric_loss_length:
                        pseudo_3d_loss = F.mse_loss(z_traj[seq, t, :, :3], pseudo_gt[seq, t, :, :3])
                        
                        sequence_loss += pseudo_3d_loss
                        num_timesteps += 1

                    elif t >= current_segment_length - config.experiment.photometric_loss_length:
                        # Update temporary prediction gaussians from predicted control points
                        tmp_pred_xyz_cp = z_traj[seq, t, :, :3]
                        tmp_pred_rot_cp = z_traj[seq, t, :, 3:7]
                        tmp_gaussians_pred = gaussians_t0.clone()
                        tmp_gaussians_pred.xyz_cp = tmp_pred_xyz_cp
                        tmp_gaussians_pred.rot_cp = tmp_pred_rot_cp
                        tmp_gaussians_pred.update_gaussians_from_controlpoints()

                        # Update temporary gt gaussians from gt control points
                        tmp_gt_xyz_cp = batch_gt_xyz_cp[seq, t, :, :]
                        tmp_gt_rot_cp = batch_gt_rot_cp[seq, t, :, :]
                        tmp_gaussians_gt = gaussians_t0.clone()
                        tmp_gaussians_gt.xyz_cp = tmp_gt_xyz_cp
                        tmp_gaussians_gt.rot_cp = tmp_gt_rot_cp
                        tmp_gaussians_gt.update_gaussians_from_controlpoints()
                
                        timestep_loss = 0.0

                        for cam in range(config.experiment.num_train_cams):
                            viewpoint_cam = cam_stack[cam]

                            # Render prediction
                            render_pkg = render(viewpoint_cam, tmp_gaussians_pred, config.pipeline, background) # ~6-13ms
                            pred_rendered_image = render_pkg["render"]

                            # Render gt
                            with torch.no_grad():
                                render_pkg_gt = render(viewpoint_cam, tmp_gaussians_gt, config.pipeline, background)
                                gt_rendered_image = render_pkg_gt["render"]

                            loss_i = F.mse_loss(pred_rendered_image, gt_rendered_image)
                            timestep_loss += loss_i / config.experiment.num_train_cams


                        sequence_loss += timestep_loss
                        num_timesteps += 1
                
                    
                batch_loss += sequence_loss / num_timesteps
            
            batch_loss = batch_loss / num_batch_seqs
            
            epoch_loss += batch_loss

            render_end_time = time.time()
            render_time = render_end_time - render_start_time
            # print(f"Render time: {render_time:.5f} seconds")

        

            optimizer.zero_grad()
            
            backward_start_time = time.time()
            batch_loss.backward(retain_graph=False)
            backward_end_time = time.time()
            backward_time = backward_end_time - backward_start_time
            # print(f"Backward time: {backward_time:.5f} seconds")

            optimizer.step()

        epoch_loss = epoch_loss / num_batches

        epoch_time = time.time() - epoch_start_time
        epochs_per_sec = 1 / epoch_time

        log.update({"loss": epoch_loss.item(),
                    "epoch_loss": epoch_loss.item(),
                    "batch_loss": batch_loss.item(),
                    "nfe": model.func.nfe,
                    "segment_length": current_segment_length,
                    "epoch": epoch,
                    "epochs_per_sec": epochs_per_sec,
                    "encoder_time": encoder_time,
                    "model_prediction_time": model_prediction_time,
                    "render_time": render_time,
                    "backward_time": backward_time,
                    "total_time": epoch_time
                    })
        


        # Log video
        """
        if epoch % 100 == 0:
            print(f"logging debug video at epoch {epoch}...")
            wandb_cam_stack = cam_stack[1:2]
            with torch.no_grad():   
                for b in range(config.experiment.num_sequences):
                    # Create a dictionary (or list) to collect video frames for each camera.
                    pred_video_frames_dict = {cam_idx: [] for cam_idx in range(len(wandb_cam_stack))}
                    gt_video_frames_dict = {cam_idx: [] for cam_idx in range(len(wandb_cam_stack))}
                    
                    for t in range(current_segment_length):
                        # Update temporary prediction gaussians from predicted control points
                        tmp_gaussians_pred = gaussians_t0.clone()
                        tmp_gaussians_pred.xyz_cp = z_traj[b, t, :, :3]
                        tmp_gaussians_pred.rot_cp = z_traj[b, t, :, 3:7]
                        tmp_gaussians_pred.update_gaussians_from_controlpoints()

                        # Update temporary GT gaussians from GT control points
                        tmp_gaussians_gt = gaussians_t0.clone()
                        tmp_gaussians_gt.xyz_cp = gt_xyz_cp[b, t, :, :]
                        tmp_gaussians_gt.rot_cp = gt_rot_cp[b, t, :, :]
                        tmp_gaussians_gt.update_gaussians_from_controlpoints()
                        
                        for cam_idx, cam in enumerate(wandb_cam_stack):  
                            # Render predicted image
                            render_pkg = render(cam, tmp_gaussians_pred, config.pipeline, background)
                            debug_image = (render_pkg["render"] * 255).clamp(0, 255).to(torch.uint8)
                            pred_video_frames_dict[cam_idx].append(debug_image.detach().cpu().numpy())
                            
                            # Render ground-truth image
                            render_pkg_gt = render(cam, tmp_gaussians_gt, config.pipeline, background)
                            debug_image_gt = (render_pkg_gt["render"] * 255).clamp(0, 255).to(torch.uint8)
                            gt_video_frames_dict[cam_idx].append(debug_image_gt.detach().cpu().numpy())
                    
                    # After looping over time steps, stack frames per camera.
                    for cam_idx in range(len(wandb_cam_stack)):
                        pred_video_frames = np.stack(pred_video_frames_dict[cam_idx]).astype(np.uint8)
                        gt_video_frames = np.stack(gt_video_frames_dict[cam_idx]).astype(np.uint8)
                        log.update({
                            f"pred_video_sequence_{b}_cam_{cam_idx}": wandb.Video(pred_video_frames, fps=5),
                            f"gt_video_sequence_{b}_cam_{cam_idx}": wandb.Video(gt_video_frames, fps=5)
                        })
            print(f"logged debug video at epoch {epoch}")
        """

                # Log video
        if epoch % 100 == 0:
            print(f"logging debug video at epoch {epoch}...")
            wandb_cam_stack = cam_stack[1:2]
            with torch.no_grad():   
                for b in range(config.experiment.batch_size):
                    # Create a dictionary to collect combined video frames for each camera
                    combined_video_frames_dict = {cam_idx: [] for cam_idx in range(len(wandb_cam_stack))}
                    
                    z_traj = model(z0_objects, t_span)

                    for t in range(current_segment_length):
                        # Update temporary prediction gaussians from predicted control points
                        tmp_gaussians_pred = gaussians_t0.clone()
                        tmp_gaussians_pred.xyz_cp = z_traj[b, t, :, :3]
                        tmp_gaussians_pred.rot_cp = z_traj[b, t, :, 3:7]
                        tmp_gaussians_pred.update_gaussians_from_controlpoints()

                        # Update temporary GT gaussians from GT control points
                        tmp_gaussians_gt = gaussians_t0.clone()
                        tmp_gaussians_gt.xyz_cp = gt_xyz_cp[b, t, :, :]
                        tmp_gaussians_gt.rot_cp = gt_rot_cp[b, t, :, :]
                        tmp_gaussians_gt.update_gaussians_from_controlpoints()
                        
                        for cam_idx, cam in enumerate(wandb_cam_stack):  
                            # Render predicted image
                            render_pkg = render(cam, tmp_gaussians_pred, config.pipeline, background)
                            pred_image = (render_pkg["render"] * 255).clamp(0, 255).to(torch.uint8)
                            
                            # Render ground-truth image
                            render_pkg_gt = render(cam, tmp_gaussians_gt, config.pipeline, background)
                            gt_image = (render_pkg_gt["render"] * 255).clamp(0, 255).to(torch.uint8)
                            
                            # Convert to numpy and concatenate horizontally
                            pred_np = pred_image.detach().cpu().numpy()
                            gt_np = gt_image.detach().cpu().numpy()
                            combined_frame = np.concatenate([pred_np, gt_np], axis=2)  # Concatenate along width
                            
                            combined_video_frames_dict[cam_idx].append(combined_frame)
                    
                    # After looping over time steps, stack frames per camera
                    for cam_idx in range(len(wandb_cam_stack)):
                        combined_video = np.stack(combined_video_frames_dict[cam_idx]).astype(np.uint8)
                        log.update({
                            f"pred_gt_video_sequence_{b}_cam_{cam_idx}": wandb.Video(combined_video, fps=5)
                        })
            print(f"logged debug video at epoch {epoch}")


        # if epoch_loss < config.experiment.loss_threshold:
        if epoch % 100 == 0:
            
            added_timestep = current_segment_length - config.experiment.photometric_loss_length
            pseudo_gt_xyz = z_traj[:, added_timestep:added_timestep+1, :, :3]
            pseudo_gt_rot = z_traj[:, added_timestep:added_timestep+1, :, 3:7]
            new_pseudo_gt = torch.cat([pseudo_gt_xyz, pseudo_gt_rot], dim=-1).detach()
            
            # Pseudo-GT is fixed once it is created
            # if pseudo_gt is None:
            #     pseudo_gt = new_pseudo_gt  # First timestep
            # else:
            #     pseudo_gt = torch.cat([pseudo_gt, new_pseudo_gt], dim=1)  # shape: [B, T+=1, N, 7]

            # Pseudo-GT is always the trajectory from the last model prediction
            pseudo_gt = z_traj.detach()


            current_segment_length += 1
            print(f"\nupdated current_segment_length to: {current_segment_length}")
        
        # if epoch % 100 == 0:
            # render video with model prediction gaussians and gt gaussians


        wandb.log(log, step=epoch)


        epoch_bar.set_postfix({
            'Loss': f'{batch_loss.item():.7f}',
            'nfe': model.func.nfe,
            'seg_len': current_segment_length,
            'it/s': epochs_per_sec,

        })





if __name__ == "__main__":
    train()