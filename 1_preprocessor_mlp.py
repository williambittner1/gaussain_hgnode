# 1_static_preprocessing.py
# This script is used to preprocess the dataset and train the static gaussian models for each sequence in the dataset.
# For the GM1, we use a deformation MLP to deform the Gaussians to fit the scene at t=1

# global imports
import os
import torch
import wandb
from dataclasses import dataclass, field, fields
from tqdm import tqdm
import torch.nn.functional as F
from random import randint
from torch.utils.data import Dataset, DataLoader
import json
from typing import List
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn



# local imports
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer_gsplat import render, render_batch
# from gaussian_renderer_inria import render



@dataclass
class ExperimentConfig:
    # data_path: str = f"/work/williamb/datasets"    # "/work/williamb/datasets"  or "/users/williamb/dev/gaussain_hgnode/data"
    data_path: str = f"/users/williamb/dev/gaussain_hgnode/data"
    # dataset_name: str = "pendulum_100seq_250ts"
    dataset_name: str = "pendulum_5seq_250ts"
    dataset_path: str = f"{data_path}/{dataset_name}"
    gm_output_path: str = f"{dataset_path}/"
    data_device: str = "cuda"


@dataclass
class OptimizationConfig:
    semantic_feature_lr: float = 0.001
    iterations = 12_000
    arap_iterations = 1_000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.025
    scaling_lr = 0.005
    rotation_lr = 0.001
    exposure_lr_init = 0.01
    exposure_lr_final = 0.001
    exposure_lr_delay_steps = 0
    exposure_lr_delay_mult = 0.0
    percent_dense = 0.01
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    depth_l1_weight_init = 1.0
    depth_l1_weight_final = 0.01
    random_background = False
    optimizer_type = "default"
    arap_weight: float = 0.1  # weight of ARAP regularization


@dataclass
class ModelConfig:
    sh_degree: int = 3
    #foundation_model: str = ""
    model_path: str = ""
    # images: str = "images"
    resolution: int = -1
    white_background: bool = False
    eval: bool = False
    speedup: bool = False
    render_items: List[str] = field(default_factory=lambda: ["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"])
    manual_gaussians_bool: bool = False

@dataclass
class PipelineConfig:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = True

def flatten_dataclass(cls):
    """Decorator to flatten nested dataclass attributes while preserving original structure."""
    original_init = cls.__init__
    
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Flatten all attributes from nested dataclasses
        for field in fields(cls):
            if hasattr(field.type, '__dataclass_fields__'):  # Check if field is a dataclass
                nested_obj = getattr(self, field.name)
                for nested_field in fields(nested_obj):
                    # Don't override existing attributes
                    if not hasattr(cls, nested_field.name):
                        # Create a closure to capture the current values
                        def make_property(field_name, nested_field_name):
                            return property(
                                lambda self: getattr(getattr(self, field_name), nested_field_name)
                            )
                        # Set the property
                        setattr(cls, nested_field.name, make_property(field.name, nested_field.name))
    
    cls.__init__ = __init__
    return cls

@flatten_dataclass
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

def visualize_gaussians(scene, gaussians, config, background, cam_stack, name=None):
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        gt_image = camera.original_image.cpu().numpy()
        
        # Create white separator line
        H, W, C = rendered_image.shape
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)  # 2-pixel wide white line
        
        # Concatenate horizontally: [rendered | white_line | gt]
        combined_image = np.concatenate([rendered_image, white_line, gt_image], axis=1)

        # Convert to uint8 [0-255] range
        combined_image = (combined_image * 255).astype(np.uint8)

        # wandb.log({f"render_{cam_idx}": wandb.Image(rendered_image)})
        # wandb.log({f"gt_{cam_idx}": wandb.Image(gt_image)})
        wandb.log({f"{name}_cam{cam_idx}": wandb.Image(combined_image)})
    #print(f"Logged render and gt")




def quaternion_inverse(q):
    """
    Returns the inverse of a quaternion q (w, x, y, z).
    q_inv = conjugate(q) / ||q||^2.
    """
    # q shape (..., 4)
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True).clamp_min(1e-10)
    return conj / norm_sq

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1, q2 (w, x, y, z).
    Returns q1 * q2 of shape (..., 4).
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def train_static_gaussian_model_batchrendering(scene, config, iterations = 30000):

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene.initialize_gaussians_from_scene_info(scene.gaussians, config.model)

    scene.gaussians.training_setup_0(config.optimization)

    viewpoint_stack = scene.getTrainCameraObjects()

    progress_bar = tqdm(range(iterations), desc=f"Training")
    for iteration in range(iterations):
        
        # viewpoint_cam = select_random_camera(scene)

        # gt_image = viewpoint_cam.original_image.permute(2,0,1)

        # # render_pkg = render(viewpoint_cam, scene.gaussians, config.pipeline, background)
        
        
        # rendered_image = render_pkg["render"]
        # visibility_filter = render_pkg["visibility_filter"]
        # radii = render_pkg["radii"]
        # viewspace_point_tensor = render_pkg["viewspace_points"]

        # loss = F.mse_loss(rendered_image, gt_image)


        width = viewpoint_stack[0].image_width
        height = viewpoint_stack[0].image_height    
        gt_batch = torch.zeros((len(viewpoint_stack), 3, height, width), device="cuda")
        for cam_idx, camera in enumerate(viewpoint_stack):
            gt_image = camera.original_image.permute(2,0,1)
            gt_batch[cam_idx] = gt_image
        gt_batch = gt_batch.permute(0,2,3,1)

        render_pkg = render_batch(viewpoint_stack, scene.gaussians, config.pipeline, background)
        rendered_batch = render_pkg["render"].permute(1,0,2,3)
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor = render_pkg["viewspace_points"]

        loss = F.mse_loss(rendered_batch, gt_batch)

        loss.backward()

        scene.gaussians.update_learning_rate(iteration)
        scene.gaussians.optimizer.step()
        scene.gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 500 == 0:
            viewpoint_cam = select_random_camera(scene)
            gt_image = viewpoint_cam.original_image.permute(2,0,1)

            viewpoint_cam = select_random_camera(scene)
            render_pkg = render(viewpoint_cam, scene.gaussians, config.pipeline, background)
        
            rendered_image = render_pkg["render"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor = render_pkg["viewspace_points"]


            loss = F.mse_loss(rendered_image, gt_image)

            loss.backward()


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
        
            print("Number of Gaussians: ", len(scene.gaussians._xyz))

        if iteration % 5000 == 0:
            visualize_gaussians(scene, scene.gaussians, config, background, cam_stack=scene.getTrainCameraObjects()[:10], iteration=iteration)

        progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
        progress_bar.update(1)
    




def train_GM0(scene, config, iterations = 30000):

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene.initialize_gaussians_from_scene_info(scene.gaussians, config.model)

    scene.gaussians.training_setup_0(config.optimization)

    viewpoint_stack = scene.getTrainCameraObjects()

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

        progress_bar.set_postfix({"Loss": f"{loss.item():.7f}", "Number of Gaussians": f"{len(scene.gaussians._xyz)}"})
        progress_bar.update(1)
    
    visualize_gaussians(scene, scene.gaussians, config, background, cam_stack=scene.getTrainCameraObjects()[:1], name=f"final_{scene.dataset_path.split('/')[-1]}_t0")
    
        

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




class DeformationMLP(nn.Module):
    """
    Example MLP that takes reference positions (x0) and quaternions (q0)
    and predicts new positions (x1) and quaternions (q1).
    
    For brevity, we do not add sophisticated encodings or skip connections.
    You can adapt as needed.
    """
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()

        # We will treat (x0, q0) \in R^7 as input
        # and output (dx, dq) \in R^7. 
        # Then final x1 = x0 + dx,   q1 = (q0 + dq) normalized.
        in_dim = 7
        out_dim = 7

        layers = []
        dim = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            dim = hidden_dim
        # final layer
        layers.append(nn.Linear(dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x0, q0):
        """
        x0: (N,3)
        q0: (N,4)  (assumed normalized)
        return: (N, 3) new positions, (N,4) new quaternions (normalized).
        """
        # Flatten input
        inp = torch.cat([x0, q0], dim=-1)  # shape: (N,7)
        out = self.mlp(inp)               # shape: (N,7)

        dx  = out[..., :3]
        dq  = out[..., 3:]  # We'll add this to q0, then normalize

        x1 = x0 + dx
        q1_unnorm = q0 + dq
        # Normalize to ensure valid quaternion
        q1 = q1_unnorm / torch.norm(q1_unnorm, dim=-1, keepdim=True).clamp(min=1e-10)
        return x1, q1


def train_GM1_deformation_MLP(scene, config, iterations=3000):
    """
    Instead of directly optimizing gaussians._xyz, we learn a function (MLP) that deforms
    GM0’s (x0, q0) to new (x1, q1). We do a photometric loss on the rendered images at t=1.
    """
    device = scene.gaussians._xyz.device
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Freeze GM0 reference:
    x0_ref = scene.gaussians._xyz.detach().clone()       # shape (N,3)
    q0_ref = scene.gaussians.get_rotation.detach().clone()  # shape (N,4)

    # Create an MLP
    mlp = DeformationMLP(hidden_dim=128, num_layers=4).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    # We do not require grad on the original GM0 parameters:
    scene.gaussians._xyz.requires_grad_(False)
    scene.gaussians._rotation.requires_grad_(False)

    viewpoint_stack = scene.getTrainCameraObjects()
    progress_bar = tqdm(range(iterations), desc="Training GM1 (MLP Deformation)")

    for iteration in progress_bar:
        viewpoint_cam = select_random_camera(scene)
        gt_image = viewpoint_cam.original_image.permute(2, 0, 1)

        # Forward pass: Deform all reference gaussians
        x1, q1 = mlp(x0_ref, q0_ref)

        # Temporarily assign these deformed parameters to the scene
        scene.gaussians._xyz = x1
        scene.gaussians._rotation = q1

        # Render
        render_pkg = render(viewpoint_cam, scene.gaussians, config.pipeline, bg_color)
        rendered_image = render_pkg["render"]

        # Photometric loss w.r.t. ground-truth at t=1
        photo_loss = F.mse_loss(rendered_image, gt_image)

        # Optionally, you can add a "regularization" term if you like:
        # reg_loss = F.mse_loss(x1, x0_ref) * 0.01  # example: keep them close
        # total_loss = photo_loss + reg_loss
        total_loss = photo_loss  # pure photometric

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        progress_bar.set_postfix({
            "photo": f"{photo_loss.item():.5f}",
            "iter": iteration
        })

    # Final pass: store the final deformed positions in the scene, or keep the MLP
    with torch.no_grad():
        x1, q1 = mlp(x0_ref, q0_ref)
        scene.gaussians._xyz = x1
        scene.gaussians._rotation = q1

    # Visualize final GM1 after MLP-based deformation
    visualize_gaussians(
        scene,
        scene.gaussians,
        config,
        bg_color,
        cam_stack=scene.getTrainCameraObjects()[:1],
        name=f"final_{scene.dataset_path.split('/')[-1]}_MLP_t1"
    )

    # Return the MLP or store it
    return mlp



class PreprocessingDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.experiment.dataset_path
        self.samples = []
        self.to_tensor = T.ToTensor()

        self.scene_dirs = sorted([
            os.path.join(self.dataset_path, d)
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d)) and d.startswith("sequence")
        ])

        if len(self.scene_dirs) == 0:
            raise ValueError(f"No scene directories found in {self.dataset_path}. "
                             "Make sure your data folder has subdirectories named like 'sequence1', 'sequence2', etc.")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx_tuple):
        """Get data for a specific sequence and timestep.
        
        Args:
            sequence_idx (int): Index of the sequence directory
            time_index (int): Time index to retrieve (0 for static image, >0 for video frame)
            
        Returns:
            dict: Contains scene_dir and gt_images tensor
        """
        if isinstance(idx_tuple, tuple):
            sequence_idx, time_index = idx_tuple
        # Get the scene directory path
        scene_dir = self.scene_dirs[sequence_idx]

        dynamic_dir = os.path.join(scene_dir, "dynamic") 
        video_files = sorted([f for f in os.listdir(dynamic_dir) if f.endswith(".mp4")])

        frames = []
        for video_file in video_files:
            video_path = os.path.join(dynamic_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            # Set frame position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, time_index)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError(f"Could not read frame {time_index} from {video_path}")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame_tensor = self.to_tensor(frame)
            frames.append(frame_tensor)
            
        # Stack all frames into a single tensor
        frames_tensor = torch.stack(frames)

        sample = {
            "scene_dir": scene_dir,
            "gt_images": frames_tensor
        }

        return sample




def convert_cameras_gt_to_train_meta(config, scene_path, cameras_gt_json):
    """
    Convert cameras_gt.json to train_meta.json
    """
    data = {}
    # Use the resolution from the first entry (assumes all cameras share the same size)
    data['width'] = cameras_gt_json[0]['width']
    data['height'] = cameras_gt_json[0]['height']
    
    # Initialize lists to be stored in the meta file.
    # (w2c, c2w, k, cam_id lists will correspond only to dynamic frames.)
    w2c, c2w, k, cam_id, img_path, vid_path = [], [], [], [], [], []

    for entry in cameras_gt_json:
        cam_idx = entry['camera_id']
        static_img_path = os.path.join(scene_path, "static", entry['img_fn'])
        img_path.append(static_img_path)
        vid_path.append(os.path.join(scene_path, "dynamic", entry['vid_fn']))
        w2c.append(entry['w2c'])
        c2w.append(entry['c2w'])
        k.append(get_intrinsics(entry))
        cam_id.append(str(cam_idx))

    data['w2c'] = w2c
    data['c2w'] = c2w
    data['k'] = k
    data['cam_id'] = cam_id
    data['img_path'] = img_path
    data['vid_path'] = vid_path

    return data


def get_intrinsics(camera_info):
    """Get intrisics matrix from camera info"""
    fx = camera_info['fx']
    fy = camera_info['fy']
    cx = camera_info['width'] / 2
    cy = camera_info['height'] / 2
    k = [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]
    
    return k      

def train(config):
    """
    Main training function.
    """

    
    device = config.experiment.data_device
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")


    ##########################################
    # 0. Set up scene and empty gaussians
    ##########################################

    
    dataset = PreprocessingDataset(config)


    print(f"Found {len(dataset)} scenes for training.")
    
    for seq_idx in range(len(dataset)):
        time_index = 0
        sample = dataset[(seq_idx, time_index)]
        print()
        print(f"\nProcessing scene: {os.path.basename(sample['scene_dir'])}")

    
        # Convert cameras_gt.json to train_meta.json if needed.
        cameras_gt_path = os.path.join(sample['scene_dir'], 'cameras_gt.json')
        train_meta_path = os.path.join(sample['scene_dir'], "train_meta.json")
        with open(cameras_gt_path, 'r') as f:
            cameras_gt_json = json.load(f)
        train_meta_json = convert_cameras_gt_to_train_meta(config, sample['scene_dir'], cameras_gt_json)
        with open(train_meta_path, 'w') as f:
            json.dump(train_meta_json, f)
        print(f"Saved train_meta.json to {train_meta_path}")


        scene_dir = sample['scene_dir']
        gm0_checkpoint_path = os.path.join(scene_dir, "gm_checkpoints", "GM0.pth")
        os.makedirs(os.path.dirname(gm0_checkpoint_path), exist_ok=True)


        # GM0
        scene = Scene(config=config, scene_path=scene_dir)
        scene.gaussians = GaussianModel(sh_degree=config.model.sh_degree)
        
        train_GM0(scene, config, iterations=config.optimization.iterations)
        
        torch.save(scene.gaussians.capture(), gm0_checkpoint_path)
        print(f"Saved GM0 with {len(scene.gaussians._xyz)} gaussians to {gm0_checkpoint_path}")






        # --- Prepare Camera GT Images for GM1 training ---
        time_index = 1
        gt_images = dataset[seq_idx, time_index]['gt_images']
        cam_stack = scene.getTrainCameraObjects()
        for i, cam in enumerate(cam_stack):
            cam.original_image = gt_images[i].permute(1,2,0).to(device)

        # --- Train GM1 using MLP deformation ---
        print("Starting GM1 refinement with MLP deformation")

        mlp = train_GM1_deformation_MLP(scene, config, iterations=config.optimization.arap_iterations)

        gm1_checkpoint_path = os.path.join(scene_dir, "gm_checkpoints", "GM1.pth")
        torch.save(scene.gaussians.capture(), gm1_checkpoint_path)
        print(f"Saved GM1 with MLP deformation to {gm1_checkpoint_path}")






if __name__ == "__main__":
    config = Config()
    wandb.init(project="blender_static_preprocessing_debug")
    train(config)
    wandb.finish()
