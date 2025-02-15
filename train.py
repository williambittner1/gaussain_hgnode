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

from scene import Scene, GaussianModel
from gaussian_renderer import render

#from models.hgnode import GraphNeuralODEHierarchical
from models.gnode_hierarchical import GraphNeuralODEHierarchical
from double_pendulum import DoublePendulum2DPolarDynamics, generate_initial_conditions_polar_2d, polar_to_cartesian_2d



from torchdiffeq import odeint


debug_render_dir = "training_renders"
if not os.path.exists(debug_render_dir):
    os.makedirs(debug_render_dir)



@dataclass
class OptimizationConfig:
    iterations: int = 20_000          # number of iterations to train the static gaussian model
    epochs: int = 10_000              # dynamic gaussian model training epochs

    total_timesteps: int = 50
    initial_timesteps: int = 30
    framerate: int = 25

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
    learning_rate: float = 5e-3
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

def save_initial_renders_and_gt(scene, config, debug_render_dir, background):
    """
    Save initial renders and ground truth images for all cameras.
    
    Args:
        scene: Scene object containing cameras and gaussians
        config: Configuration object with pipeline settings
        debug_render_dir: Directory path to save renders
        background: Background color for rendering
    """
    print("Saving initial renders and ground truth for all cameras...")
    for cam_idx, camera in enumerate(scene.getTrainCameraObjects()):
        gt_image = camera.original_image.permute(2,0,1)
        
        # Render initial state
        render_pkg = render(camera, scene.gaussians, config.pipeline, background)
        rendered_image = render_pkg["render"]
        
        # Save both images
        save_image(gt_image, cam_idx, 'gt_initial', debug_render_dir)
        save_image(rendered_image, cam_idx, 'render_initial', debug_render_dir)


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

    # visualize_gaussians(scene, gaussians_t0, config, background, viewpoint_stack=scene.getTrainCameraObjects()[0:3])

    ##########################################
    # 2. Cluster gaussians into n_clusters clusters (K-means)
    ##########################################
    # for each cluster:
    #   - self.cluster_control_points holds the mean positions (shape: [3,3])
    #   - self.cluster_control_orientations holds the orientations (all identity initially)
    #   - self.relative_positions holds each gaussian’s offset from its cluster’s control point.
    gaussians_t0.cluster_gaussians(n_clusters=3)






    ##########################################
    # 3. Simulate points
    ##########################################

    # Define dynamics model
    dynamics_model = DoublePendulum2DPolarDynamics(L1=2.0, L2=1.0, g=9.81).to(device)

    # Generate initial condition x0: torch.Tensor (system_state=4={theta1, theta2, dtheta1, dtheta2})
    x0 = generate_initial_conditions_polar_2d(num_sequences=1, device=device, seed=42)[0]  

    # Define time_span: shape=(total_timesteps,)
    dt = 0.05  # time step (in seconds)
    total_timesteps = config.optimization.total_timesteps
    t_span = torch.linspace(0, dt * (total_timesteps - 1), total_timesteps, device=device, dtype=torch.float32)

    # Simulate trajectory: 
    trajectory = odeint(dynamics_model, x0.unsqueeze(0), t_span).squeeze(1)  # torch.Tensor (total_timesteps, system_state=4={theta1, theta2, dtheta1, dtheta2})

    # Convert 4d system's polar state to three masses' 3D positions
    positions_3d = polar_to_cartesian_2d(trajectory.unsqueeze(0), L1=2.0, L2=1.0).squeeze(0)  # torch.Tensor (total_timesteps, num_masses=3, pos=3)

    # Compute control-point positions and quaternions for each time step
    dynamic_control_positions = []      # List of total_timesteps x tensors [3, 3]
    dynamic_control_rotations = []   # List of total_timesteps x tensors [3, 4]
    for t_idx in range(total_timesteps):
        dynamic_control_positions.append(positions_3d[t_idx])
        q_anchor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        q_mass1 = angle_to_quaternion(trajectory[t_idx, 0].unsqueeze(0))[0]
        q_mass2 = angle_to_quaternion((trajectory[t_idx, 0] + trajectory[t_idx, 1]).unsqueeze(0))[0]
        dynamic_control_rotations.append(torch.stack([q_anchor, q_mass1, q_mass2], dim=0))

    # Compute particle-point positions and quaternions for each time step
    N = gaussians_t0.get_xyz.shape[0]
    particle_labels = gaussians_t0.cluster_label.to(device)         # torch.Tensor (N,)
    particle_offsets = gaussians_t0.relative_positions.to(device)   # torch.Tensor (N, 3)

    gt_xyz = []
    gt_rot = []
    
    for t_idx in range(total_timesteps):
        ctrl_pos = dynamic_control_positions[t_idx].to(device)       # shape: [3, 3]
        ctrl_rot = dynamic_control_rotations[t_idx].to(device)   # shape: [3, 4]
        
        # Broadcast control point positions and rotations to each particle-point:
        q_dyn = ctrl_rot[particle_labels]           # shape: (N, 4)
        cp_dyn = ctrl_pos[particle_labels]          # shape: (N, 3)

        # Compute new particle-point positions:
        new_offset = rotate_vectors(q_dyn, particle_offsets)  # shape: (N, 3)
        new_positions = cp_dyn + new_offset
        
        # Compute new particle-point rotations:
        old_rotations = gaussians_t0._rotation  # (N,4)
        new_rotations = quaternion_multiply(q_dyn, old_rotations)
        
        gt_xyz.append(new_positions)
        gt_rot.append(new_rotations)

    gt_xyz = torch.stack(gt_xyz, dim=0) # torch.Tensor (T_dynamic, N, 3)
    gt_rot = torch.stack(gt_rot, dim=0) # torch.Tensor (T_dynamic, N, 4)
    


    
    
    

    model = GraphNeuralODEHierarchical(
        particle_dim=13,
        object_dim=13,
        hidden_dim=256,
        n_hidden_layers=4,
        solver="rk4",
        rtol=1e-4,
        atol=1e-5,
        options={"step_size": 0.01,
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

  

    # Extract z0_particles and z0_objects from gt_gaussians
    gt_gaussians_t0 = gaussians_t0.clone()
    gt_gaussians_t1 = gaussians_t0.clone().update_dynamic_state(1, gt_xyz, gt_rot)

    p0 = gt_gaussians_t0.get_xyz                   # Shape: [N, 3]
    p1 = gt_gaussians_t1.get_xyz                   # Shape: [N, 3]

    v0 = p1-p0

    q0 = gt_gaussians_t0._rotation                   # Shape: [N, 4]

    z0_particles = torch.cat([p0, v0, q0, p0], dim=1)

    cluster_labels = gt_gaussians_t0.cluster_label  # shape [N]

    COM, obj_vel, obj_quat, obj_omega = compute_object_state_from_labels(cluster_labels, p0, v0, q0)

    z0_objects = torch.cat([
        COM.unsqueeze(0),        # [1, num_clusters, 3]
        obj_vel.unsqueeze(0),     # [1, num_clusters, 3]
        obj_quat.unsqueeze(0),    # [1, num_clusters, 4]
        obj_omega.unsqueeze(0)    # [1, num_clusters, 3]
    ], dim=-1)  # Final shape: [1, num_clusters, 13]

    # Now your initial state is:
    state0 = (z0_particles.unsqueeze(0), z0_objects)  # [1, N, 13] and [1, num_clusters, 13]


    # Convert hard cluster labels to one-hot representation.
    # Here, num_objects should equal the number of clusters (3).
    num_objects = 3
    S = F.one_hot(cluster_labels, num_classes=num_objects).float()  # shape: [N, 3]

    # Add a batch dimension so that S has shape [B, N, 3] with B=1.
    S = S.unsqueeze(0)

    # Set the assignment in the GraphNeuralODE function.
    model.func.S = S

    viewpoint_stack = scene.getTrainCameraObjects()

    current_segment_length = config.optimization.initial_timesteps

    epoch_bar = tqdm(range(config.optimization.epochs), desc=f"Training")
    for epoch in epoch_bar:
        
        torch.cuda.empty_cache()

        epoch_loss = 0.0
        loss_list = []

        viewpoint_cam = select_random_camera(scene)

        segment_duration = current_segment_length / config.optimization.framerate

        t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
        
        model.func.nfe = 0

        pred_particles, pred_objects = model(state0, t_span)


        for t in range(current_segment_length):
            for cam in range(2):
                viewpoint_cam = viewpoint_stack[cam]
                tmp_gaussians = gt_gaussians[0].clone()
                tmp_gaussians._xyz = pred_particles[0, t, :, :3].clone()
                tmp_gaussians._rotation = pred_particles[0, t, :, 6:10].clone()

                render_pkg = render(viewpoint_cam, tmp_gaussians, config.pipeline, background)
                pred_rendered_image = render_pkg["render"]
                
                with torch.no_grad():
                    gt_gaussians = gaussians_t0.update_dynamic_state(t, gt_xyz, gt_rot)
                    render_pkg_gt = render(viewpoint_cam, gt_gaussians, config.pipeline, background)
                    gt_rendered_image = render_pkg_gt["render"]
                
                loss_i = F.mse_loss(pred_rendered_image, gt_rendered_image)
                loss_list.append(loss_i)
                
        epoch_loss = torch.stack(loss_list).sum()

        optimizer.zero_grad()
        epoch_loss.backward(retain_graph=True)
        optimizer.step()

        # log to wandb
        wandb.log({"loss": epoch_loss.item()})

        if epoch % 100 == 0:
            with torch.no_grad():   
                for i, cam in enumerate(viewpoint_stack[0:5]):  
                    pred_video_frames = []
                    gt_video_frames = []
                    for t in range(current_segment_length):
                        tmp_gaussians = gt_gaussians[0].clone()
                        tmp_gaussians._xyz = pred_particles[0, t, :, :3]
                        tmp_gaussians._rotation = pred_particles[0, t, :, 6:10]
                        render_pkg = render(cam, tmp_gaussians, config.pipeline, background)
                        debug_image = (render_pkg["render"] * 255).clamp(0, 255).to(torch.uint8)
                        pred_video_frames.append(debug_image.detach().cpu().numpy())
                    pred_video_frames = np.stack(pred_video_frames).astype(np.uint8)

                    for t in range(current_segment_length):
                        gt_gaussians = gaussians_t0.update_dynamic_state(t, gt_xyz, gt_rot)
                        render_pkg_gt = render(cam, gt_gaussians, config.pipeline, background)
                        debug_image_gt = (render_pkg_gt["render"] * 255).clamp(0, 255).to(torch.uint8)
                        gt_video_frames.append(debug_image_gt.detach().cpu().numpy())
                    gt_video_frames = np.stack(gt_video_frames).astype(np.uint8)

                    wandb.log({
                        f"pred_video_cam_{i}": wandb.Video(pred_video_frames, fps=5),
                        f"gt_video_cam_{i}": wandb.Video(gt_video_frames, fps=5)})                
                    
                print(f"logged debug video at epoch {epoch}")

        # if epoch % 100 == 0:
            # render video with model prediction gaussians and gt gaussians




        epoch_bar.set_postfix({"Loss": f"{epoch_loss.item():.7f}"})
        epoch_bar.update(1)



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