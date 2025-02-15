# train.py
import torch
import torch.nn.functional as F
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

from scene import Scene, GaussianModel
from gaussian_renderer import render

from models.hgnode import GraphNeuralODEHierarchical
from double_pendulum import (DoublePendulum2DPolarDynamics, 
                             generate_initial_conditions_polar_2d, 
                             generate_trajectory_2d, 
                             polar_to_10d, 
                             dense_ground_truth_from_sparse, 
                             generate_points_data)


training_renders_dir = "training_renders"
if not os.path.exists(training_renders_dir):
    os.makedirs(training_renders_dir)



@dataclass
class OptimizationConfig:
    iterations: int = 30000
    total_timesteps: int = 40
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
    exp_name: str = "Monkey"
    data_path: str = "data/Monkey"
    source_path: str = "data/Monkey"
    output_path: str = "output/Monkey"
    foundation_model: str = ""
    checkpoint_path: str = "output/Monkey"
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
    data_path: str = "data/DoublePendulumSpheres_5cams"


@dataclass
class Config:
    experiment: ExperimentConfig = ExperimentConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()

def init_config():
    config = Config()
    return config


def select_random_camera(scene):
    viewpoint_stack = scene.getTrainCameraObjects()
    viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]
    return viewpoint_cam

def compute_loss(render_pkg, viewpoint_cam):
    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image.permute(2,0,1)
    loss = F.l1_loss(image, gt_image)
    return loss

def save_image(tensor_image, iteration, image_type, training_renders_dir):
    # Permute the tensor to move channels to the last dimension and convert to CPU and NumPy
    image_np = tensor_image.permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)  # Convert to 8-bit for saving
    
    # Convert to PIL Image and save
    pil_image = Image.fromarray(image_np)
    pil_image.save(os.path.join(training_renders_dir, f"{image_type}_iteration_{iteration}.png"))


def train():
    config = init_config()

    # Initialize the Scene without Gaussians
    scene = Scene(config=config, dataset=None)
    gaussians = GaussianModel(sh_degree=3)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    device = config.experiment.data_device
    # loss_fn_lpips = lpips.LPIPS(net='alex').to(device)


    scene.initialize_gaussians_from_scene_info(gaussians, config.model)

    timestep = 0


    # Setup training in the Gaussian model.
    gaussians.training_setup_0(config.optimization)
    progress_bar = tqdm(range(config.optimization.iterations), desc=f"Training timestep {timestep}")
    for iteration in range(config.optimization.iterations):
        # Select a random camera from the scene.
        viewpoint_cam = select_random_camera(scene)

        # Render using your renderer.
        render_pkg = render(viewpoint_cam, gaussians, config.pipeline, background)
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor = render_pkg["viewspace_points"]

        # Save rendered and GT images every 1000 iterations.
        if iteration == 0:
            save_image(render_pkg["render"], iteration, 'render', training_renders_dir)
            save_image(viewpoint_cam.original_image.permute(2, 0, 1), iteration, 'gt', training_renders_dir)


        # Compute loss between render and ground truth.
        loss = compute_loss(render_pkg, viewpoint_cam)

        loss.backward()
        gaussians.update_learning_rate(iteration)
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Save rendered and GT images every 1000 iterations.
        if iteration % 1000 == 0:
            save_image(render_pkg["render"], iteration, 'render', training_renders_dir)
            save_image(viewpoint_cam.original_image.permute(2, 0, 1), iteration, 'gt', training_renders_dir)

        # Perform densification/pruning if needed.
        densification_step(
            iteration,
            config.optimization,
            gaussians,
            render_pkg,
            visibility_filter,
            radii,
            viewspace_point_tensor,
            scene,
            config.model
        )

        # Update progress bar.
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
            progress_bar.update(10)

        # Save checkpoint every 5000 iterations.
        if iteration % 5000 == 0:
            checkpoint_dir = config.model.checkpoint_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save((gaussians.capture(), iteration), os.path.join(checkpoint_dir, f"chkpnt{iteration}.pth"))

    progress_bar.close()
    print("\nTraining complete.")

    return











    L1 = 1.0
    L2 = 0.5
    g = 9.81

    # Create dynamics instance for the 2D double pendulum.
    dynamics = DoublePendulum2DPolarDynamics(L1=L1, L2=L2, g=g).to(device)
    

    # Generate full time spans
    t_train_span = torch.linspace(0., config.train_duration, config.num_time_samples_train).to(device)
    t_test_span = torch.linspace(0., config.test_duration, config.num_time_samples_test).to(device)
    

    # Generate initial conditions for training and testing
    x0_train = generate_initial_conditions_polar_2d(config.num_train_data_sequences, device=device, seed=2)
    x0_test  = generate_initial_conditions_polar_2d(config.num_test_data_sequences, device=device, seed=3)


    # Generate training data
    points_train = generate_points_data(
        dynamics=dynamics,
        num_sequences=config.num_train_data_sequences,
        t_span=t_train_span,
        num_scattered_points=30,
        radius=0.1,
        L1=L1,
        L2=L2,
        device=device,
        x0=x0_train,
        seed=2
    )
    B, T, N_train, features = points_train.shape    # features = [p, v, q, semantic_feature] = [3+3+4+1] = [11]


    # Generate test data
    points_test = generate_points_data(
        dynamics=dynamics,
        num_sequences=config.num_test_data_sequences,
        t_span=t_test_span,
        num_scattered_points=10,
        radius=0.1,
        L1=L1,
        L2=L2,
        device=device,
        x0=x0_test,
        seed=3
    )
    
    num_points = 5
    T = 1

    def sample_points_on_sphere(num_points, radius, device):
        # Uniformly sample points on a sphere of given radius.
        theta = torch.rand(num_points, device=device) * 2 * math.pi         # Uniform in [0, 2pi)
        u = torch.rand(num_points, device=device) * 2 - 1                     # Uniform in [-1, 1]
        # Compute spherical coordinates
        phi = torch.acos(u)
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * u
        points = torch.stack([x, y, z], dim=1)  # shape (num_points, 3)
        return points

    sphere_radius = 15.0
    sphere_points = sample_points_on_sphere(num_points, sphere_radius, device)  # shape (num_points, 3)
    semantic_features_dummy = torch.ones(num_points, 1, device=device)  # shape (num_points, 1)

    distance = 30.0
    for t in range(T):
        # positions = torch.tensor([[ 0.0, 0.0, distance]], dtype=torch.float32, device=device)
        # colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        # scales = torch.ones((1, 3), device=device)
        # rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).expand(1, 4)
        # opacities = torch.full((1, 1), 0.9, device=device)
        # semantic_features_dummy = torch.ones(1, 1, device=device)
        
        positions = [torch.tensor([i, 0.0, 10.0], device=device) for i in range(num_points)]
        scales = [torch.tensor([0.001, 0.001, 0.001], device=device) for _ in range(num_points)]
        rotations = [torch.tensor([1.0, 0.2, 0.4, 0.0], device=device) for _ in range(num_points)]
        opacities = [0.9 for _ in range(num_points)]
        colors = [torch.tensor([1.0, 0.0, 0.0], device=device) for _ in range(num_points)]  # Red Gaussians

        
        gaussians = GaussianModel(sh_degree=1)
        gaussians.optimizer = torch.optim.Adam([
            {'params': [], 'name': 'xyz'},
            {'params': [], 'name': 'f_dc'},
            {'params': [], 'name': 'f_rest'},
            {'params': [], 'name': 'opacity'},
            {'params': [], 'name': 'scaling'},
            {'params': [], 'name': 'rotation'},
            {'params': [], 'name': 'semantic_feature'}
        ], lr=0.001)
        
        # Loop over the six points and add each as a gaussian.
        for pos, scl, rot, op, col, sem in zip(positions, scales, rotations, opacities, colors, semantic_features_dummy):
            gaussians.add_gaussian(pos, scl, rot, op, col, sem)
        
        # Append the model (which now contains 6 gaussians) to the scene.
        scene.gaussians.append(gaussians)

    print(f"Number of Gaussians: {len(scene.gaussians)}")

    viewpoint_stack = scene.getTrainCameraObjects().copy()
    train_cameras = [viewpoint_stack[i] for i in range(len(viewpoint_stack))]

    
    viewpoint_cam = train_cameras[0]
    
    pipe = PipelineParams()
    pipe.debug = False
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    
    gaussians = scene.gaussians[0]
    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    rendered_image = render_pkg['render']
    
    gt_image_wandb = rendered_image.detach().cpu().permute(1, 2, 0).numpy()
    wandb.log({"rendered_image": wandb.Image(gt_image_wandb)})



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



if __name__ == "__main__":
    train()