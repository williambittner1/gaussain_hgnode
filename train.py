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

#from models.hgnode import GraphNeuralODEHierarchical
from models.gnode_hierarchical import GraphNeuralODEHierarchical
from double_pendulum import DoublePendulum2DPolarDynamics, generate_initial_conditions_polar_2d, polar_to_cartesian_2d

from dataset import PreRenderedGTDataset

from torchdiffeq import odeint


debug_render_dir = "training_renders"
if not os.path.exists(debug_render_dir):
    os.makedirs(debug_render_dir)



@dataclass
class OptimizationConfig:
    iterations: int = 20_000          # number of iterations to train the static gaussian model
    epochs: int = 10_000              # dynamic gaussian model training epochs

    total_timesteps: int = 100
    initial_timesteps: int = 3
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
    learning_rate: float = 1e-4
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
    loss_threshold: float = 5e-5
    num_train_data_sequences: int = 2
    num_test_data_sequences: int = 1
    batch_size: int = 2
    use_all_segments: bool = False
    stride: int = 10
    dynamics_type: str = "double_pendulum_cartesian_rigid"
    data_device: torch.device = torch.device("cuda")
    data_path: str = ModelConfig.data_path
    gt_mode: str = "python"  # or "blender"
    num_sequences: int = 4    # number of sequences to simulate



@dataclass
class Config:
    experiment: ExperimentConfig = ExperimentConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()


def select_random_camera(scene):
    viewpoint_stack = scene.getTrainCameraObjects()
    viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]
    return viewpoint_cam


def angle_to_quaternion(angle):
    # Returns [cos(angle/2), sin(angle/2), 0, 0]
    return torch.stack([torch.cos(angle/2), torch.sin(angle/2), torch.zeros_like(angle), torch.zeros_like(angle)], dim=-1)


def simulate_dynamic_gaussians_double_pendulum_1seq(gaussians_t0, dynamics_model, dt, total_timesteps, L1, L2, device):
    """
    Simulate the dynamics of the Gaussian points using a double pendulum dynamics model.
    
    Args:
        gaussians_t0 (GaussianModel): A static GaussianModel that has been initialized and clustered.
        dynamics_model (nn.Module): A dynamics model (e.g. DoublePendulum2DPolarDynamics) that 
            simulates the system dynamics in polar coordinates.
        dt (float): Time step size.
        total_timesteps (int): Total number of timesteps for simulation.
        L1 (float): Length parameter 1 (for converting polar to Cartesian).
        L2 (float): Length parameter 2.
        device (torch.device): The device to run the simulation on.
    
    Returns:
        gt_xyz (torch.Tensor): Ground-truth positions over time with shape [T, N, 3],
                               where T is the number of timesteps and N is the number of Gaussians.
        gt_rot (torch.Tensor): Ground-truth rotations over time with shape [T, N, 4].
    """
    # Generate initial condition x0 (system state: [theta1, theta2, dtheta1, dtheta2])
    x0 = generate_initial_conditions_polar_2d(num_sequences=1, device=device, seed=42)[0]  
    # Define time span
    t_span = torch.linspace(0, dt * (total_timesteps - 1), total_timesteps, device=device, dtype=torch.float32)
    
    # Simulate the trajectory in polar state: shape [total_timesteps, 4]
    trajectory = odeint(dynamics_model, x0.unsqueeze(0), t_span).squeeze(1)
    
    # Convert the 4d polar state to 3D positions for 3 masses: shape [total_timesteps, 3, 3]
    positions_3d = polar_to_cartesian_2d(trajectory.unsqueeze(0), L1=L1, L2=L2).squeeze(0)
    
    # Compute dynamic control parameters for each timestep:
    dynamic_control_positions = []  # List of [3, 3] tensors (one per timestep)
    dynamic_control_rotations = []  # List of [3, 4] tensors (one per timestep)
    for t_idx in range(total_timesteps):
        # The control positions for each cluster (assumed to be 3 clusters) are the 3 rows of positions_3d at time t_idx.
        ctrl_pos = positions_3d[t_idx]  # shape: [3, 3]
        dynamic_control_positions.append(ctrl_pos)
        
        # Compute control rotations:
        # The anchor (first mass) remains fixed (identity quaternion)
        q_anchor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        # For the first movable mass, use theta1
        q_mass1 = angle_to_quaternion(trajectory[t_idx, 0].unsqueeze(0))[0]
        # For the second movable mass, use (theta1 + theta2)
        q_mass2 = angle_to_quaternion((trajectory[t_idx, 0] + trajectory[t_idx, 1]).unsqueeze(0))[0]
        # Stack into a [3, 4] tensor
        ctrl_rot = torch.stack([q_anchor, q_mass1, q_mass2], dim=0)
        dynamic_control_rotations.append(ctrl_rot)
    
    # Prepare lists to collect the ground-truth positions and rotations for all timesteps.
    gt_xyz_list = []
    gt_rot_list = []
    
    # Get the static cluster assignments and relative offsets from gaussians_t0:
    static_labels = gaussians_t0.cluster_label.to(device)    # shape: [N]
    static_relatives = gaussians_t0.relative_positions.to(device)  # shape: [N, 3]
    
    # For each timestep, compute the updated positions and rotations using the control parameters:
    for t_idx in range(total_timesteps):
        # Retrieve control parameters for timestep t_idx:
        ctrl_pos = dynamic_control_positions[t_idx].to(device)   # [3, 3]
        ctrl_rot = dynamic_control_rotations[t_idx].to(device)     # [3, 4]
        
        # "Look-up operation":Use static_labels to broadcast control parameters to all Gaussians:
        cp_dyn = ctrl_pos[static_labels]    # shape: [N, 3] = [3, 3][N]
        q_dyn = ctrl_rot[static_labels]     # shape: [N, 4] = [3, 4][N]
        
        # Rotate each Gaussian's stored relative offset by its control rotation:
        new_offset = rotate_vectors(q_dyn, static_relatives)  # shape: [N, 3]
        new_positions = cp_dyn + new_offset  # New absolute positions for each Gaussian
        
        # Update rotations: compose control rotation with the static (initial) rotation.
        old_rotations = gaussians_t0._rotation  # shape: [N, 4]
        new_rotations = quaternion_multiply(q_dyn, old_rotations)  # shape: [N, 4]
        
        gt_xyz_list.append(new_positions)
        gt_rot_list.append(new_rotations)
    
    # Stack the lists into tensors of shape [T, N, ...]
    gt_xyz = torch.stack(gt_xyz_list, dim=0)  # [total_timesteps, N, 3]
    gt_rot = torch.stack(gt_rot_list, dim=0)  # [total_timesteps, N, 4]
    
    return gt_xyz, gt_rot


def simulate_dynamic_gaussians_double_pendulum_old(gaussians_t0, dynamics_model, dt, total_timesteps, L1, L2, device, num_sequences=1):
    """
    Simulate the dynamics for num_batches sequences. Each sequence has its own 
    initial condition and is simulated for total_timesteps.
    
    Args:
        gaussians_t0 (GaussianModel): Static Gaussian model (whose cluster labels 
            and relative offsets will be used for all batches).
        dynamics_model (nn.Module): The double pendulum dynamics model.
        dt (float): Time step.
        total_timesteps (int): Number of timesteps.
        L1 (float): Length parameter 1.
        L2 (float): Length parameter 2.
        device (torch.device): The device to run on.
        num_batches (int): Number of sequences (batches) to simulate.
    
    Returns:
        gt_xyz (torch.Tensor): Ground-truth positions with shape [B, T, N, 3].
        gt_rot (torch.Tensor): Ground-truth rotations with shape [B, T, N, 4].
    """
    # Generate initial conditions for num_batches sequences
    # x0: shape [B, 4]
    x0 = generate_initial_conditions_polar_2d(num_sequences=num_sequences, device=device, seed=42)
    
    # Create time span (shape [T])
    t_span = torch.linspace(0, dt * (total_timesteps - 1), total_timesteps, device=device, dtype=torch.float32)
    
    # Simulate the trajectories using odeint.
    # trajectory: shape [T, B, 4]
    trajectory = odeint(dynamics_model, x0, t_span)
    # Permute to shape [B, T, 4]
    trajectory = trajectory.permute(1, 0, 2)
    
    # Convert the polar state to 3D positions.
    # Assuming polar_to_cartesian_2d accepts input of shape [B, T, 4] and 
    # returns output of shape [B, T, 3, 3] (for 3 objects per sequence)
    positions_3d = polar_to_cartesian_2d(trajectory, L1=L1, L2=L2)  # shape: [B, T, 3, 3]
    
    # Retrieve static information from the gaussians.
    static_labels = gaussians_t0.cluster_label.to(device)         # shape: [N]
    static_relatives = gaussians_t0.relative_positions.to(device)   # shape: [N, 3]
    old_rotations = gaussians_t0._rotation.to(device)               # shape: [N, 4]
    
    # Expand static quantities to work with batches when needed.
    # For example, we will need to expand static_relatives and old_rotations
    # later to shape [B, N, ...].
    
    # Prepare lists to collect the per-timestep ground-truth states.
    gt_xyz_list = []
    gt_rot_list = []
    
    B = num_sequences
    for t in range(total_timesteps):
        # For each timestep t:
        # 1. Get the control positions from positions_3d at time t: shape [B, 3, 3]
        ctrl_pos = positions_3d[:, t, :, :]  # [B, 3, 3]
        
        # 2. Compute control rotations from the trajectory at time t.
        # Extract angles for each batch.
        theta1 = trajectory[:, t, 0]  # shape: [B]
        theta2 = trajectory[:, t, 1]  # shape: [B]
        
        # Compute quaternions in a vectorized manner:
        q_anchor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)  # [B, 4]
        q_mass1 = angle_to_quaternion(theta1.unsqueeze(1)).squeeze(1)  # [B, 4]
        q_mass2 = angle_to_quaternion((theta1 + theta2).unsqueeze(1)).squeeze(1)  # [B, 4]
        # Stack into shape [B, 3, 4] for the three control rotations.
        ctrl_rot = torch.stack([q_anchor, q_mass1, q_mass2], dim=1)  # [B, 3, 4]
        
        # 3. "Lookup" the control parameters for each Gaussian using its cluster label.
        # Using advanced indexing:
        # For positions: shape [B, N, 3]
        cp_dyn = ctrl_pos[:, static_labels, :]  
        # For rotations: shape [B, N, 4]
        q_dyn = ctrl_rot[:, static_labels, :]
        
        # 4. Compute the new offset by rotating the static relative positions.
        # Expand static_relatives to [B, N, 3]
        static_relatives_exp = static_relatives.unsqueeze(0).expand(B, -1, -1)
        # Reshape to merge batch and particle dimensions for vectorized rotation.
        q_dyn_flat = q_dyn.reshape(-1, 4)
        static_relatives_flat = static_relatives_exp.reshape(-1, 3)
        new_offset_flat = rotate_vectors(q_dyn_flat, static_relatives_flat)
        new_offset = new_offset_flat.reshape(B, -1, 3)  # [B, N, 3]
        
        # 5. Compute the new positions.
        new_positions = cp_dyn + new_offset  # [B, N, 3]
        
        # 6. Update rotations by composing the control rotation with the static rotation.
        old_rotations_exp = old_rotations.unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]
        # Reshape for quaternion multiplication.
        q_dyn_flat = q_dyn.reshape(-1, 4)
        old_rotations_flat = old_rotations_exp.reshape(-1, 4)
        new_rotations_flat = quaternion_multiply(q_dyn_flat, old_rotations_flat)
        new_rotations = new_rotations_flat.reshape(B, -1, 4)  # [B, N, 4]
        
        # Append the per-timestep results.
        gt_xyz_list.append(new_positions)
        gt_rot_list.append(new_rotations)
    
    # Stack along a new time dimension.
    # Resulting shapes: [B, T, N, 3] and [B, T, N, 4]
    gt_xyz = torch.stack(gt_xyz_list, dim=1)
    gt_rot = torch.stack(gt_rot_list, dim=1)
    
    return gt_xyz, gt_rot


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
    q_anchor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # [B, T, 4]
    
    # Compute quaternion for mass1 using theta1.
    # Assume angle_to_quaternion accepts input of shape [B, T, 1] and returns [B, T, 4].
    q_mass1 = angle_to_quaternion(theta1.unsqueeze(-1))  # [B, T, 4]
    
    # Compute quaternion for mass2 using theta1 + theta2.
    q_mass2 = angle_to_quaternion((theta1 + theta2).unsqueeze(-1))  # [B, T, 4]
    
    # Stack the control rotations into a tensor of shape [B, T, num_controlpoints, 4]
    ctrl_rotations = torch.stack([q_anchor, q_mass1, q_mass2], dim=2)  # [B, T, 3, 4]
    
    # Concatenate positions and rotations to get the control points state.
    # Each control point: [x, y, z, qw, qx, qy, qz]
    cp_state = torch.cat([ctrl_positions, ctrl_rotations], dim=-1)  # [B, T, 3, 7]
    
    return cp_state




def encode_initial_state(gt_gaussians_t0, gt_gaussians_t1):
    """
    Encode the initial state z₀ from the static Gaussian model (at t=0) and its
    dynamic update at t=1. The state is represented as a tuple (z0_particles, z0_objects),
    where:
      - z0_particles is a tensor with concatenated particle features:
          [current_position, velocity, quaternion, initial_position]
      - z0_objects is a tensor with concatenated object-level features:
          [COM, object_velocity, object_quat, object_omega]

    Args:
        gt_gaussians_t0 (GaussianModel): The static Gaussian model at t=0.
        gt_gaussians_t1 (GaussianModel): The Gaussian model updated to t=1 
                                         (e.g. using gt_xyz and gt_rot).
    
    Returns:
        state0 (tuple): A tuple (z0_particles, z0_objects) suitable for the GraphNeuralODEHierarchical.
    """
    # Particle state:
    # p0: positions at t=0, p1: positions at t=1, v0: estimated velocity (p1 - p0)
    p0 = gt_gaussians_t0.get_xyz           # Shape: [N, 3]
    p1 = gt_gaussians_t1.get_xyz           # Shape: [N, 3]
    v0 = p1 - p0                           # Shape: [N, 3]
    q0 = gt_gaussians_t0._rotation         # Shape: [N, 4]

    # Concatenate particle features. Here we use:
    # [current_position, velocity, quaternion, initial_position]
    z0_particles = torch.cat([p0, v0, q0, p0], dim=1)  # [N, 13]

    # Object state:
    # Compute the cluster-level state (COM, object velocity, etc.) from the particle states.
    cluster_labels = gt_gaussians_t0.cluster_label  # Shape: [N]
    COM, obj_vel, obj_quat, obj_omega = compute_object_state_from_labels(
        cluster_labels, p0, v0, q0
    )
    # Concatenate object features: [COM, object_velocity, object_quat, object_omega]
    z0_objects = torch.cat([
        COM.unsqueeze(0),       # [1, num_clusters, 3]
        obj_vel.unsqueeze(0),   # [1, num_clusters, 3]
        obj_quat.unsqueeze(0),  # [1, num_clusters, 4]
        obj_omega.unsqueeze(0)  # [1, num_clusters, 3]
    ], dim=-1)  # [1, num_clusters, 13]

    # Add a batch dimension (B=1) to the particle state as well.
    z0_particles = z0_particles.unsqueeze(0)  # [1, N, 13]

    return z0_particles, z0_objects


def encode_initial_state_batched(gaussians_t0, gt_xyz, gt_rot):
    """
    Compute the initial state from gaussians_t0 and the batched ground truth
    at t=0 and t=1. Here, gt_xyz and gt_rot have shape [B, T, N, 3] and [B, T, N, 4].
    """
    B = gt_xyz.shape[0]
    # p0 is taken from the static gaussians (assumed same for all batches)
    p0 = gt_xyz[:, 0, :, :]
    # Replicate for each batch
    

    # p1 is taken from the ground truth at t=1 for each batch
    p1 = gt_xyz[:, 1, :, :]  # [B, N, 3]
    v0 = p1 - p0             # Estimated velocity for each batch [B, N, 3]

    # Get quaternion (assumed static at t=0) and replicate across batches
    q0 = gt_rot[:, 0, :, :]  # [B, N, 4]

    # Concatenate features: [current_position, velocity, quaternion, initial_position]
    z0_particles = torch.cat([p0, v0, q0, p0], dim=-1)  # [B, N, 13]

    # For the object state, you need to compute the cluster-level (object) features per batch.
    # For example, you might compute each batch’s COM and velocity using a loop or vectorized operation.
    # Here’s a simple (non-vectorized) approach:
    COM_list, vel_list, quat_list, omega_list = [], [], [], []
    num_clusters = int(gaussians_t0.cluster_label.max().item()) + 1
    for b in range(B):
        cluster_labels = gaussians_t0.cluster_label  # shape: [N]
        COM, obj_vel, obj_quat, obj_omega = compute_object_state_from_labels(
            cluster_labels,
            p0[b],
            v0[b],
            q0[b]
        )
        COM_list.append(COM)
        vel_list.append(obj_vel)
        quat_list.append(obj_quat)
        omega_list.append(obj_omega)
    # Stack along a new batch dimension: each will be [B, num_clusters, ...]
    z0_objects = torch.cat([
        torch.stack(COM_list, dim=0),       # [B, num_clusters, 3]
        torch.stack(vel_list, dim=0),         # [B, num_clusters, 3]
        torch.stack(quat_list, dim=0),        # [B, num_clusters, 4]
        torch.stack(omega_list, dim=0)         # [B, num_clusters, 3]
    ], dim=-1)  # resulting in [B, num_clusters, 13]

    return z0_particles, z0_objects


def save_image(tensor_image, iteration, image_type, debug_render_dir):
    # Permute the tensor to move channels to the last dimension and convert to CPU and NumPy
    image_np = tensor_image.permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)  # Convert to 8-bit for saving
    
    # Convert to PIL Image and save
    pil_image = Image.fromarray(image_np)
    pil_image.save(os.path.join(debug_render_dir, f"{image_type}_iteration_{iteration}.png"))


def visualize_gaussians(scene, gaussians, config, background, viewpoint_stack, iteration=None):
    for cam_idx, camera in enumerate(viewpoint_stack):
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
        #     visualize_gaussians(scene, scene.gaussians, config, background, viewpoint_stack=scene.getTrainCameraObjects()[0:5], iteration=iteration)

        progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
        progress_bar.update(1)
    
    return scene


def render_and_save_gt_images_from_simulation(scene, gaussians, gt_xyz, gt_rot, config, device):
    """
    Pre-render ground-truth images from the simulated dynamic gaussians and save them to disk.
    The images are saved in a folder structure like:
      <output_path>/gt/<cam_id>/render_<cam_id>_timestep_<timestep>.jpg
    """

    # Set up output folder for GT images
    gt_dir = os.path.join(config.model.data_path, "python_gt")
    os.makedirs(gt_dir, exist_ok=True)
    
    # Get the camera objects from the scene
    cams = scene.getTrainCameraObjects()  # assume these are the cameras you want to use
    num_cams = len(cams)
    
    # For each camera, create a subfolder (with a 3-digit id)
    for cam_idx in range(num_cams):
        cam_folder = os.path.join(gt_dir, f"{cam_idx:03d}")
        os.makedirs(cam_folder, exist_ok=True)
    
    # For each timestep, update the dynamic state and render GT for each camera
    for t in range(config.optimization.total_timesteps):
        # Update gaussians for timestep t
        gaussians.update_dynamic_state(t, gt_xyz, gt_rot)
        # Render for each camera
        for cam_idx, cam in enumerate(cams):
            render_pkg = render(cam, gaussians, config.pipeline, torch.tensor([0,0,0], dtype=torch.float32, device=device))
            img = render_pkg["render"]  # tensor image, e.g., [C,H,W] in [0,1]
            # Convert to PIL Image
            img_np = (img.permute(1,2,0).detach().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            # Save image with naming convention: render_<cam>_timestep_<t>.jpg
            filename = os.path.join(gt_dir, f"{cam_idx:03d}", f"render_{cam_idx:03d}_timestep_{t:05d}.jpg")
            pil_img.save(filename)
        print(f"Pre-rendered GT for timestep {t}")








def simulate_object_dynamics(gaussians_t0, dynamics_model, dt, total_timesteps, device, num_sequences=1):
    """
    Simulate the dynamics for the object nodes.
    
    We assume that the initial object state is given by the control points and
    control orientations computed during clustering. For simplicity, we also
    assume zero initial velocities and angular velocities.
    
    The state for each object node is represented as:
      [position (3), velocity (3), quaternion (4), angular_velocity (3)]
    i.e. a 13-dimensional state.
    
    Returns:
        gt_xyz_o: [B, T, M, 3]  -- positions of M object nodes over time.
        gt_rot_o: [B, T, M, 4]  -- orientations (quaternions) of object nodes over time.
    """
    # Retrieve initial object states from the static model.
    # These were computed during clustering.
    # cluster_control_points: [M, 3]
    # cluster_control_orientations: [M, 4]
    M = gaussians_t0.cluster_control_points.shape[0]
    init_pos = gaussians_t0.cluster_control_points  # [M, 3]
    init_quat = gaussians_t0.cluster_control_orientations  # [M, 4]
    
    # For simplicity, assume initial velocity and angular velocity are zeros.
    init_vel = torch.zeros_like(init_pos)  # [M, 3]
    init_ang_vel = torch.zeros_like(init_pos)  # [M, 3]
    
    # Build initial state: shape [M, 13]
    init_state = torch.cat([init_pos, init_vel, init_quat, init_ang_vel], dim=-1)  # [M, 13]
    # Expand to batch dimension: [B, M, 13]
    B = num_sequences
    init_state = init_state.unsqueeze(0).expand(B, -1, -1)
    # Flatten batch and objects for odeint: shape [B*M, 13]
    init_state_flat = init_state.reshape(B * M, 13)
    
    # Create time span
    t_span = torch.linspace(0, dt * (total_timesteps - 1), total_timesteps, device=device, dtype=torch.float32)
    
    # Simulate dynamics using odeint (assumes dynamics_model is defined for 13-dim states)
    traj = odeint(dynamics_model, init_state_flat, t_span)
    # traj: [T, B*M, 13]
    traj = traj.reshape(total_timesteps, B, M, 13).permute(1, 0, 2, 3)  # [B, T, M, 13]
    
    # Extract object positions and quaternions.
    gt_xyz_o = traj[..., :3]    # [B, T, M, 3]
    gt_rot_o = traj[..., 6:10]   # [B, T, M, 4]
    return gt_xyz_o, gt_rot_o

def compute_particle_from_objects(gaussians_t0, gt_xyz_o, gt_rot_o):
    """
    Compute particle trajectories given object (control point) trajectories.
    
    For each particle, use its stored cluster label, its stored relative offset, and
    its stored "static" rotation to rigidly transform the object state.
    
    For a particle i in a given batch and time:
      gt_xyz_p[i] = gt_xyz_o[b, t, cluster_label[i]] + rotate(gt_rot_o[b, t, cluster_label[i]], relative_offset[i])
      gt_rot_p[i] = quaternion_multiply(gt_rot_o[b, t, cluster_label[i]], static_rotation[i])
    
    Returns:
        gt_xyz_p: [B, T, N, 3] particle positions.
        gt_rot_p: [B, T, N, 4] particle orientations.
    """
    # Retrieve stored per-particle info from gaussians_t0.
    # Assume these were computed during clustering.
    static_labels = gaussians_t0.cluster_label  # shape [N]
    static_relatives = gaussians_t0.relative_positions  # shape [N, 3]
    static_rotations = gaussians_t0._rotation  # shape [N, 4]
    
    # For each batch/time, select the corresponding object state for each particle.
    # gt_xyz_o: [B, T, M, 3] and gt_rot_o: [B, T, M, 4]
    # Use advanced indexing to get control parameters per particle:
    # Expand static_labels to use as index: we need them as a LongTensor.
    cp = gt_xyz_o[:, :, static_labels, :]  # [B, T, N, 3]
    q_obj = gt_rot_o[:, :, static_labels, :]  # [B, T, N, 4]
    
    # Rotate the stored relative offsets by the object quaternions.
    # First, expand static_relatives to [1, 1, N, 3]
    static_relatives_exp = static_relatives.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 3]
    # We now use a quaternion rotation function for batched inputs.
    rot_offset = quat_rotate(q_obj.reshape(-1, 4), static_relatives_exp.expand(cp.shape[0], cp.shape[1], -1, -1).reshape(-1, 3))
    rot_offset = rot_offset.reshape(cp.shape)  # [B, T, N, 3]
    
    gt_xyz_p = cp + rot_offset  # [B, T, N, 3]
    
    # For rotations, compose the object quaternion with the particle's static rotation.
    static_rotations_exp = static_rotations.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 4]
    gt_rot_p = quaternion_multiply(q_obj.reshape(-1, 4), static_rotations_exp.expand(q_obj.shape[0], q_obj.shape[1], -1, -1).reshape(-1, 4))
    gt_rot_p = gt_rot_p.reshape(q_obj.shape)  # [B, T, N, 4]
    return gt_xyz_p, gt_rot_p


def quat_conjugate(q):
    """
    Compute the conjugate of quaternion(s) q.
    
    Args:
        q (torch.Tensor): Tensor of shape (..., 4) representing quaternions in [w, x, y, z] format.
    
    Returns:
        torch.Tensor: Conjugate of q, same shape as q.
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate(q, v):
    """
    Rotate vector(s) v by quaternion(s) q.
    
    The rotation is given by:
      v_rot = q * [0, v] * q^{-1}
    
    Args:
        q (torch.Tensor): Tensor of shape (..., 4) representing quaternion(s) in [w, x, y, z] format.
        v (torch.Tensor): Tensor of shape (..., 3) representing vector(s) to be rotated.
    
    Returns:
        torch.Tensor: Rotated vector(s) of shape (..., 3).
    """
    # Create a pure quaternion from v with zero scalar part.
    zero = torch.zeros_like(v[..., :1])
    v_quat = torch.cat([zero, v], dim=-1)  # shape (..., 4)
    
    # Compute the rotated quaternion: q * v_quat * q_conjugate
    q_conj = quat_conjugate(q)
    rotated_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    
    # Return only the vector part.
    return rotated_quat[..., 1:]




def train():
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


    # visualize_gaussians(scene, gaussians_t0, config, background, viewpoint_stack=scene.getTrainCameraObjects()[0:3])


    ##########################################
    # 2. Initialize num_objects control points for each cluster
    ##########################################
    gaussians_t0.initialize_controlpoints(config.optimization.n_objects)
    gaussians_t0.update_gaussians_from_controlpoints()


    ##########################################
    # 3. Initialize control points for t=0
    ##########################################
    
    # Control Points
    control_points = gaussians_t0.cluster_control_points
    control_orientations = gaussians_t0.cluster_control_orientations
    # Gaussians
    relative_positions = gaussians_t0.relative_positions
    relative_rotations = gaussians_t0.relative_rotations
    cluster_labels = gaussians_t0.cluster_label
    xyz = gaussians_t0.get_xyz
    rotation = gaussians_t0.get_rotation





    ##########################################
    # 3. Simulate points
    ##########################################
    if config.experiment.gt_mode == "python":
        print("Simulating dynamic gaussians...")
        dynamics_model = DoublePendulum2DPolarDynamics(L1=2.0, L2=1.0)

        print("Simulating dynamic object nodes...")

        gt_cp_state = simulate_dynamic_controlpoints_double_pendulum(dynamics_model=dynamics_model,
                                                                     dt=0.05, 
                                                                     total_timesteps=config.optimization.total_timesteps, 
                                                                     L1=2.0, 
                                                                     L2=1.0, 
                                                                     device=device,
                                                                     num_sequences=config.experiment.num_sequences)
        
        print("Updating control points...")
        gt_xyz_cp, gt_rot_cp = gt_cp_state[..., :3], gt_cp_state[..., 3:] # shape: [B, T, N, 3], [B, T, N, 4]

        """
        Update protocol to update gaussians_t0 from gt control points :
            gaussians_t0.xyz_cp = gt_xyz_cp[b, t]  # shape: [N, 3]
            gaussians_t0.rot_cp = gt_rot_cp[b, t]  # shape: [N, 4]
            gaussians_t0.update_gaussians_from_controlpoints()
        """

        print("Updating gaussians...")

        # gt_xyz, gt_rot = simulate_dynamic_gaussians_double_pendulum(gaussians_t0, 
        #                                                             dynamics_model, 
        #                                                             dt=0.05, 
        #                                                             total_timesteps=config.optimization.total_timesteps, 
        #                                                             L1=2.0, 
        #                                                             L2=1.0, 
        #                                                             device=device,
        #                                                             num_sequences=config.experiment.num_sequences)
        # 3. Simulate points: First simulate object node dynamics.

        # gt_xyz_o, gt_rot_o = simulate_object_dynamics(gaussians_t0, dynamics_model, dt=0.05, 
        #                                             total_timesteps=config.optimization.total_timesteps, 
        #                                             device=device,
        #                                             num_sequences=config.experiment.num_sequences)
        
        # Then compute particle dynamics by rigidly attaching particles to the object nodes.
        # gt_xyz_p, gt_rot_p = compute_particle_from_objects(gaussians_t0, gt_xyz_o, gt_rot_o)
        print("Simulated dynamic particles.")


        gt_xyz = gt_xyz_p
        gt_rot = gt_rot_p

        cams = scene.getTrainCameraObjects()  # assume these are the cameras you want to use
        num_cams = len(cams)
        # check if gt_images folder exists
        # if not os.path.exists(os.path.join(config.model.data_path, "python_gt")):
        #     print("Rendering and saving GT images from simulation...")
        #     render_and_save_gt_images_from_simulation(scene, gaussians_t0, gt_xyz, gt_rot, config, device)   
        #     print("Python GT images saved")
        # else:
        #     print("Python GT images already exist")
        # gt_base_dir = os.path.join(config.model.data_path, "python_gt")
      
        # gt_dataset = PreRenderedGTDataset(gt_base_dir, config.optimization.total_timesteps, num_cams)
        # # Optionally, create a DataLoader (if you want to sample images in mini-batches)
        # gt_loader = DataLoader(gt_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        # print("Loading GT images into dictionary...")
        # gt_images = {}
        # for img, cam, t in gt_loader:
        #     # Remove batch dimension (assumed batch_size=1)
        #     gt_images[(int(cam.item()), int(t.item()))] = img.squeeze(0)  # tensor shape: [C, H, W]

        # print("GT images loaded into dictionary")

        # gaussians_t0.update_dynamic_state(0, gt_xyz, gt_rot)


    ##########################################
    # 4. Setup Dynamic Model    
    ##########################################

    model = GraphNeuralODEHierarchical(
        particle_dim=13,
        object_dim=13,
        hidden_dim=256,
        n_hidden_layers=4,
        solver="rk4",
        rtol=1e-4,
        atol=1e-5,
        options={"step_size": 0.04,
                 # "max_num_steps": 200
                 },
        max_objects=3,
        device=device
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters()),
        lr=1e-3,#config.learning_rate
    )
    loss_fn = nn.MSELoss(reduction="mean")


    current_segment_length = config.optimization

  
    ##########################################
    # 5. Initial State Encoder z0 (explicit)
    ##########################################

    # Extract z0_particles and z0_objects from gt_gaussians
    # gt_gaussians_t0 = gaussians_t0.clone()
    # gt_gaussians_t1 = gaussians_t0.clone()
    # gt_gaussians_t1.update_dynamic_state(1, gt_xyz, gt_rot)
    
    # Encode the initial state for the dynamic ODE model.
    # z0_particles, z0_objects = encode_initial_state(gt_gaussians_t0, gt_gaussians_t1)
    z0_particles, z0_objects = encode_initial_state_batched(gaussians_t0, gt_xyz, gt_rot)
    state0 = (z0_particles, z0_objects)



    ##########################################
    # 6. Setup Assignment Matrix S
    ##########################################
    # num_objects = number of clusters
    # Convert hard cluster labels to one-hot representation.
    num_objects = config.optimization.n_objects
    S = F.one_hot(cluster_labels, num_classes=num_objects).float().unsqueeze(0)  # shape: [B, N, 3] 
    model.func.S = S
    current_segment_length = config.optimization.initial_timesteps


    ##########################################
    # 7. Train Dynamic Model (HGNODE)
    ##########################################
    
    # Train-Loop Setup
    viewpoint_stack = scene.getTrainCameraObjects()

    # Train-Loop
    epoch_bar = tqdm(range(config.optimization.epochs), desc=f"Training")
    for epoch in epoch_bar:
        
        torch.cuda.empty_cache()

        epoch_start_time = time.time()

        epoch_loss = 0.0
        loss_list = []
        log = {}

        viewpoint_cam = select_random_camera(scene)

        segment_duration = current_segment_length / config.optimization.framerate

        t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
        
        model.func.nfe = 0

        state0_detached = (state0[0].detach(), state0[1].detach())
        pred_particles, pred_objects = model(state0_detached, t_span)
        pred_xyz = pred_particles[:, :, :, :3]# .clone()
        pred_rot = pred_particles[:, :, :, 6:10]# .clone()

        # tmp_gaussians = gaussians_t0.clone()
        # gt_gaussians = gaussians_t0.clone()

        for b in range(pred_xyz.shape[0]):
            for t in range(current_segment_length):
                # Clone the predicted state *once per (b,t)*
                pred_xyz_bt = pred_xyz[b, t].clone()
                pred_rot_bt = pred_rot[b, t].clone()

                gt_xyz_bt = gt_xyz[b, t]
                gt_rot_bt = gt_rot[b, t]

                for cam in range(2):
                    viewpoint_cam = viewpoint_stack[cam]
                    tmp_gaussians_b = gaussians_t0.clone()
                    # Pass the cloned tensors (wrapped with unsqueeze to match expected dimensions)
                    tmp_gaussians_b.update_dynamic_state(pred_xyz_bt, pred_rot_bt)
                    render_pkg = render(viewpoint_cam, tmp_gaussians_b, config.pipeline, background)
                    pred_rendered_image = render_pkg["render"]

                    with torch.no_grad():
                        gt_gaussians_b = gaussians_t0.clone()
                        gt_gaussians_b.update_dynamic_state(gt_xyz_bt, gt_rot_bt)
                        render_pkg_gt = render(viewpoint_cam, gt_gaussians_b, config.pipeline, background)
                        gt_rendered_image = render_pkg_gt["render"]

                    loss_i = F.mse_loss(pred_rendered_image, gt_rendered_image)
                    loss_list.append(loss_i)
                    
        epoch_loss = torch.stack(loss_list).mean()

        optimizer.zero_grad()
        epoch_loss.backward(retain_graph=False)
        optimizer.step()


        epoch_time = time.time() - epoch_start_time
        epochs_per_sec = 1 / epoch_time

        log.update({"loss": epoch_loss.item(),
                    "nfe": model.func.nfe,
                    "segment_length": current_segment_length,
                    "epoch": epoch,
                    "epochs_per_sec": epochs_per_sec
                    })
        

        if epoch % 100 == 0:
            print(f"logging debug video at epoch {epoch}...")
            with torch.no_grad():   
                for b in range(config.experiment.num_sequences):
                    for cam_idx, cam in enumerate(viewpoint_stack[1:2]):  
                        pred_video_frames = []
                        gt_video_frames = []

                        # render prediction
                        for t in range(current_segment_length):
                            
                            pred_xyz_bt = pred_xyz[b, t]
                            pred_rot_bt = pred_rot[b, t]

                            tmp_gaussians_b.update_dynamic_state(pred_xyz_bt, pred_rot_bt)
                            render_pkg = render(cam, tmp_gaussians_b, config.pipeline, background)
                            debug_image = (render_pkg["render"] * 255).clamp(0, 255).to(torch.uint8)
                            pred_video_frames.append(debug_image.detach().cpu().numpy())
                        pred_video_frames = np.stack(pred_video_frames).astype(np.uint8)

                        # render gt
                        for t in range(current_segment_length):
                            gt_xyz_bt = gt_xyz[b, t]
                            gt_rot_bt = gt_rot[b, t]
                            
                            gt_gaussians_b.update_dynamic_state(gt_xyz_bt, gt_rot_bt)
                            render_pkg_gt = render(cam, gt_gaussians_b, config.pipeline, background)
                            debug_image_gt = (render_pkg_gt["render"] * 255).clamp(0, 255).to(torch.uint8)
                            # debug_image_gt = (gt_images[(i, t)] * 255).clamp(0, 255).to(torch.uint8)

                            gt_video_frames.append(debug_image_gt.detach().cpu().numpy())
                        gt_video_frames = np.stack(gt_video_frames).astype(np.uint8)

                        log.update({
                            f"pred_video_sequence_{b}_cam_{cam_idx}": wandb.Video(pred_video_frames, fps=5),
                            f"gt_video_sequence_{b}_cam_{cam_idx}": wandb.Video(gt_video_frames, fps=5)})                
                        
            print(f"logged debug video at epoch {epoch}")


        if epoch_loss < 1e-5:
            current_segment_length += 1
            print(f"updated current_segment_length to: {current_segment_length}")

        # if epoch % 100 == 0:
            # render video with model prediction gaussians and gt gaussians


        wandb.log(log, step=epoch)


        epoch_bar.set_postfix({
            'Epoch': epoch,
            'Loss': f'{epoch_loss.item():.7f}',
            'nfe': model.func.nfe,
            'segment_length': current_segment_length,
            'iter_per_sec': epochs_per_sec
        })



def compute_object_state_from_labels(cluster_labels, particle_positions, particle_velocities, particle_quats, num_clusters=None, eps=1e-8):
    """
    Compute the object state from particle features using hard cluster labels.
    
    Instead of a soft assignment S, we use a hard assignment vector, `cluster_labels` (of shape [N])
    where each entry is an integer in {0, 1, ..., num_clusters-1}.
    
    For each cluster, the object state is defined as:
      - COM (position): the average of the particle positions in that cluster.
      - object_velocity: the average of the particle velocities in that cluster.
      - object_quat: set to the identity quaternion [1, 0, 0, 0] (could later be updated).
      - object_omega: computed as the average over particles in the cluster of
            cross(r, (v - object_velocity)) / (||r||^2 + eps)
        where r = (particle_position - COM).
    
    Args:
      cluster_labels: LongTensor of shape [N] with cluster assignments (0-indexed).
      particle_positions: Tensor of shape [N, 3].
      particle_velocities: Tensor of shape [N, 3].
      particle_quats: Tensor of shape [N, 4] (not used here; we assume identity for t=0).
      num_clusters: (optional) number of clusters. If None, it is inferred from cluster_labels.
      eps: Small constant to avoid division by zero.
      
    Returns:
      COM: Tensor of shape [num_clusters, 3]
      object_velocity: Tensor of shape [num_clusters, 3]
      object_quat: Tensor of shape [num_clusters, 4] (all identity quaternions)
      object_omega: Tensor of shape [num_clusters, 3]
    """
    if num_clusters is None:
        num_clusters = int(cluster_labels.max().item()) + 1

    COM_list = []
    velocity_list = []
    omega_list = []
    identity_quats = []
    # Loop over each cluster label
    for c in range(num_clusters):
        # Get indices of particles in cluster c
        indices = (cluster_labels == c).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            # If no particles in this cluster, skip it.
            continue
        pos_c = particle_positions[indices]  # [n_c, 3]
        vel_c = particle_velocities[indices]   # [n_c, 3]
        # Compute COM and average velocity
        COM_c = pos_c.mean(dim=0)
        vel_avg = vel_c.mean(dim=0)
        # For each particle in the cluster, compute the relative position and relative velocity.
        r = pos_c - COM_c  # [n_c, 3]
        v_rel = vel_c - vel_avg  # [n_c, 3]
        # Compute per-particle omega estimates: cross(r, v_rel) / (||r||^2 + eps)
        r_norm_sq = (r ** 2).sum(dim=-1, keepdim=True)  # [n_c, 1]
        omega_est = torch.cross(r, v_rel, dim=-1) / (r_norm_sq + eps)  # [n_c, 3]
        omega_c = omega_est.mean(dim=0)  # average over cluster
        # Append results
        COM_list.append(COM_c)
        velocity_list.append(vel_avg)
        omega_list.append(omega_c)
        identity_quats.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=particle_positions.device, dtype=particle_positions.dtype))
    
    COM = torch.stack(COM_list, dim=0)            # [num_clusters, 3]
    object_velocity = torch.stack(velocity_list, dim=0)  # [num_clusters, 3]
    object_omega = torch.stack(omega_list, dim=0)        # [num_clusters, 3]
    object_quat = torch.stack(identity_quats, dim=0)     # [num_clusters, 4]
    
    return COM, object_velocity, object_quat, object_omega


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


def rotate_vectors(q, v):
    """
    Rotate a batch of vectors v by a batch of quaternions q.
    q: Tensor of shape (N, 4) with each row [w, x, y, z].
    v: Tensor of shape (N, 3).
    Returns: Rotated vectors of shape (N, 3).
    """
    # Using the formula: v_rot = v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)
    q_w = q[:, :1]    # shape (N, 1)
    q_vec = q[:, 1:]  # shape (N, 3)
    uv = torch.cross(q_vec, v, dim=1)
    uuv = torch.cross(q_vec, uv, dim=1)
    return v + 2 * (q_w * uv + uuv)


def quaternion_multiply(q1, q2):
    """
    Multiply two batches of quaternions.

    Args:
        q1 (torch.Tensor): Tensor of shape (N, 4) representing quaternions in [w, x, y, z] format.
        q2 (torch.Tensor): Tensor of shape (N, 4) representing quaternions in [w, x, y, z] format.

    Returns:
        torch.Tensor: Tensor of shape (N, 4) representing the elementwise quaternion products.
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


if __name__ == "__main__":
    train()