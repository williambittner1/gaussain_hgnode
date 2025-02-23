# 1_static_preprocessing.py
# This script is used to preprocess the dataset and train the static gaussian models for each sequence in the dataset
# For the GM1, we use the ARAP regularization to preserve the pairwise distances between the Gaussians.

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



# local imports
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer_gsplat import render, render_batch
# from gaussian_renderer_inria import render



@dataclass
class ExperimentConfig:
    data_path: str = f"/users/williamb/dev/gaussain_hgnode/data"    # "/work/williamb/datasets"  or "/users/williamb/dev/gaussain_hgnode/data"
    dataset_name: str = "pendulum_5seq_250ts"
    dataset_path: str = f"{data_path}/{dataset_name}"
    gm_output_path: str = f"{dataset_path}/"
    data_device: str = "cuda"


@dataclass
class OptimizationConfig:
    semantic_feature_lr: float = 0.001
    iterations = 30_000
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
    print(f"Logged render and gt")




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




def precompute_arap_data(positions_ref: torch.Tensor, arap_radius: float):
    """
    Precompute neighbor lists and reference edge vectors for ARAP.

    Args:
        positions_ref : (N,3) reference positions of the gaussians (torch float tensor).
        arap_radius   : Only neighbors with distance < arap_radius are considered.

    Returns:
        A dict with:
          - 'neighbors': list of length N; each entry is a 1D LongTensor of neighbor indices
          - 'ref_edges': list of length N; each entry shape (k, 3)
          - 'weights':   list of length N; each entry shape (k,)
    """
    device = positions_ref.device
    N = positions_ref.size(0)

    # Compute all pairwise distances (N,N)
    # For large N, consider a more efficient neighbor search.
    diff_ref = positions_ref.unsqueeze(1) - positions_ref.unsqueeze(0)  # (N,N,3)
    dist_ref = diff_ref.norm(dim=-1)  # (N,N)

    # Build output containers
    neighbors_list = []
    ref_edges_list = []
    weights_list   = []

    for i in range(N):
        # find neighbors within radius (excluding i itself)
        mask = (dist_ref[i] < arap_radius) & (dist_ref[i] > 1e-12)
        idx_neighbors = mask.nonzero(as_tuple=True)[0]  # 1D LongTensor

        neighbors_list.append(idx_neighbors)

        # reference edges = p_j^0 - p_i^0
        r_edges = positions_ref[idx_neighbors] - positions_ref[i]  # shape (k,3)
        ref_edges_list.append(r_edges)

        # weights = 1 / dist_ref(i,j)
        d_ij = dist_ref[i, idx_neighbors]       # shape (k,)
        w_ij = 1.0 / d_ij
        weights_list.append(w_ij)

    return {
        "neighbors": neighbors_list,
        "ref_edges": ref_edges_list,
        "weights":   weights_list
    }


def compute_full_arap_loss_svd(current_positions: torch.Tensor,
                               arap_data: dict) -> torch.Tensor:
    """
    'Full ARAP' loss with per-point SVD for best-fit rotation.

    Args:
      current_positions : (N,3) current positions of the gaussians.
      arap_data: dict from precompute_arap_data(...).
        - neighbors[i] : (k_i,) neighbor indices
        - ref_edges[i] : (k_i,3)
        - weights[i]   : (k_i,)

    Returns:
      (scalar) The ARAP loss, typically to be multiplied by a weight and
      added to the photometric loss.
    """
    device = current_positions.device
    N = current_positions.size(0)

    total_loss = 0.0
    # We'll accumulate in float, to keep some numeric stability if needed
    for i in range(N):
        idx_neighbors = arap_data["neighbors"][i]
        if idx_neighbors.numel() == 0:
            continue

        # 1) X_j = p_j - p_i
        p_i = current_positions[i]                 # (3,)
        p_j = current_positions[idx_neighbors]     # (k,3)
        X = p_j - p_i                              # (k,3)

        # 2) Y_j = ref_edges[i], shape (k,3), already stored
        Y = arap_data["ref_edges"][i].to(device)   # (k,3)
        w = arap_data["weights"][i].to(device)     # (k,)

        # 3) Build A_i = sum_j w_ij * X_j (Y_j)^T
        #    We can do a quick outer product in a loop or vectorized approach:
        #    A_i is (3,3)
        A_i = X.new_zeros((3, 3))
        for n in range(len(idx_neighbors)):
            xj = X[n].unsqueeze(1)     # (3,1)
            yj = Y[n].unsqueeze(0)     # (1,3)
            A_i += w[n] * (xj @ yj)    # outer product

        # 4) SVD, then fix sign if det < 0
        U, S, Vt = torch.linalg.svd(A_i, full_matrices=False)
        det = torch.det(U @ Vt)
        if det < 0:
            # flip last row of Vt
            Vt[-1, :] *= -1.0
        R_i = U @ Vt  # (3,3)

        # 5) local error = sum_j w_ij * || X_j - R_i Y_j ||^2
        Y_rot = (R_i @ Y.T).T  # (k,3)
        diff_ij = X - Y_rot    # (k,3)
        dist_sq = diff_ij.pow(2).sum(dim=-1)  # shape (k,)
        local_loss = (w * dist_sq).sum()

        total_loss += local_loss

    # average by N for consistent scale
    arap_loss = total_loss / N
    return arap_loss

def compute_simple_arap_loss(gaussians, ref_positions, neighbor_radius=0.1):
    """
    A simpler ARAP-like loss that just preserves pairwise distances 
    for neighbors (i.e. no local rotation via SVD).

    For each pair of points (i,j) within neighbor_radius in the reference,
    we penalize [||p_i - p_j|| - ||p^0_i - p^0_j||]^2.

    Args:
        gaussians: GaussianModel containing current positions (gaussians._xyz).
        ref_positions: (N,3) reference positions (usually the original GM0 state).
        neighbor_radius: Only consider neighbors whose ref distance < neighbor_radius.

    Returns:
        Scalar arap_loss (Tensor).
    """
    current_positions = gaussians._xyz  # (N,3)
    diff_ref = ref_positions.unsqueeze(1) - ref_positions.unsqueeze(0)  # (N,N,3)
    dist_ref = diff_ref.norm(dim=-1)                                    # (N,N)

    diff_cur = current_positions.unsqueeze(1) - current_positions.unsqueeze(0) 
    dist_cur = diff_cur.norm(dim=-1)

    # neighbor_mask: only pairs with 0 < dist_ref < neighbor_radius
    neighbor_mask = (dist_ref < neighbor_radius) & (dist_ref > 1e-6)

    # distance difference
    dist_diff = dist_cur - dist_ref
    loss_map = dist_diff**2

    # sum over neighbor pairs
    pairwise_loss = loss_map[neighbor_mask].mean() if neighbor_mask.any() else 0.0
    return pairwise_loss

def train_GM1_full_arap(scene, config, iterations=3000, arap_radius=0.1):
    """
    Example refinement from GM0 to GM1 with full ARAP.
    We do NOT do densification or pruning here to preserve 1:1 correspondences.
    """
    device = scene.gaussians._xyz.device
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Save the reference positions for all gaussians at t=0
    ref_positions = scene.gaussians._xyz.clone().detach()
    # Precompute neighbor data
    arap_data = precompute_arap_data(ref_positions, arap_radius=arap_radius)

    # (Optionally adjust your optimizer so that you only train pos+quat, etc.)
    scene.gaussians.training_setup_arap(config.optimization)

    from tqdm import tqdm
    progress_bar = tqdm(range(iterations), desc="Refining GM1 w/ Full ARAP")

    for iteration in progress_bar:
        # 1) Pick random viewpoint & photometric loss
        viewpoint_cam = select_random_camera(scene)
        gt_image = viewpoint_cam.original_image.permute(2, 0, 1)
        render_pkg = render(viewpoint_cam, scene.gaussians, config.pipeline, bg_color)
        rendered_image = render_pkg["render"]
        photo_loss = F.mse_loss(rendered_image, gt_image)

        # 2) ARAP loss
        current_positions = scene.gaussians._xyz
        arap_loss = compute_full_arap_loss_svd(current_positions, arap_data)

        # 3) Weighted sum
        total_loss = photo_loss + config.optimization.arap_weight * arap_loss

        # 4) Backprop & step
        total_loss.backward()
        scene.gaussians.update_learning_rate(iteration)
        scene.gaussians.optimizer.step()
        scene.gaussians.optimizer.zero_grad(set_to_none=True)

        progress_bar.set_postfix({
            "Photo": f"{photo_loss.item():.3e}",
            "ARAP": f"{arap_loss.item():.3e}",
            "Total": f"{total_loss.item():.3e}",
            "NumG": len(scene.gaussians._xyz)
        })

    progress_bar.close()
    print("Done refining GM1 with full ARAP!")


def train_GM1_simple_arap(scene, config, iterations=3000, neighbor_radius=0.1):
    """
    Refines GM0 to GM1 with a simpler ARAP that merely preserves pairwise 
    distances, rather than doing a local SVD for each point.
    """
    device = scene.gaussians._xyz.device
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Save the reference positions for all gaussians at t=0
    ref_positions = scene.gaussians._xyz.clone().detach()

    # Possibly adjust your optimizer so that only pos & rot are trained:
    scene.gaussians.training_setup_arap(config.optimization)

    from tqdm import tqdm
    progress_bar = tqdm(range(iterations), desc="Refining GM1 w/ Simple ARAP")

    for iteration in progress_bar:
        viewpoint_cam = select_random_camera(scene)
        gt_image = viewpoint_cam.original_image.permute(2, 0, 1)
        render_pkg = render(viewpoint_cam, scene.gaussians, config.pipeline, background)
        rendered_image = render_pkg["render"]
        photo_loss = F.mse_loss(rendered_image, gt_image)

        # simpler ARAP distance-preserving loss
        arap_loss = compute_simple_arap_loss(scene.gaussians, ref_positions, neighbor_radius)

        total_loss = photo_loss + config.optimization.arap_weight * arap_loss
        total_loss.backward()

        scene.gaussians.update_learning_rate(iteration)
        scene.gaussians.optimizer.step()
        scene.gaussians.optimizer.zero_grad(set_to_none=True)

        progress_bar.set_postfix({
            "Photo": f"{photo_loss.item():.3e}",
            "ARAP": f"{arap_loss.item():.3e}",
            "Total": f"{total_loss.item():.3e}",
            "NumG": len(scene.gaussians._xyz)
        })

    progress_bar.close()
    visualize_gaussians(scene, scene.gaussians, config, background, cam_stack=scene.getTrainCameraObjects()[:1], name=f"final_{scene.dataset_path.split('/')[-1]}_arap")
    print("Done refining GM1 with simple pairwise-distance ARAP!")


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



    


class GM_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.experiment.dataset_path

        self.scene_dirs = sorted([
            os.path.join(self.dataset_path, d)
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d)) and d.startswith("sequence")
        ])

        if len(self.scene_dirs) == 0:
            raise ValueError(f"No scene directories found in {self.dataset_path}. "
                             "Make sure your data folder has subdirectories named like 'scene1', 'scene2', etc.")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        # Return the path to the scene directory. 
        # This can be used by the Scene class to load scene-specific files (e.g., cameras_gt.json, images, etc.)
        scene_path = self.scene_dirs[idx]
        return scene_path

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

def convert_cameras_gt_to_static_train_meta(config, scene_path, cameras_gt_json):
    
    data = dict()
    
    cameras_info = cameras_gt_json

    data = dict()
    data['w'] = cameras_info[0]['width']
    data['h'] = cameras_info[0]['height']
    
    w2c, c2w, k, cam_id, fn = [], [], [], [], []    
    
    total_timesteps = 0 #cameras_info[0]['total_timesteps']
    
    total_cameras = len(os.listdir(os.path.join(scene_path, 'static')))

    # For each timestep, loop over cameras.

    k_inner = []
    w2c_inner = []
    c2w_inner = []
    cam_id_inner = []
    fn_inner = []
    
    # Choose the number of cameras to process at this timestep:

    num_cams = total_cameras

    for c in range(num_cams):
        # Find the camera info for camera c at timestep t.
        curr_camera_info = [ci for ci in cameras_info if ci['camera_id'] == c and ci['frame'] == 0][0]
        k_inner.append(get_intrinsics(curr_camera_info))
        w2c_inner.append(curr_camera_info['w2c'])
        c2w_inner.append(curr_camera_info['c2w'])
        cam_id_inner.append(str(curr_camera_info['camera_id']))
        fn_inner.append(f"{scene_path}/static/{curr_camera_info['img_name']}.png")
        
    w2c.append(w2c_inner)
    k.append(k_inner)
    cam_id.append(cam_id_inner)
    fn.append(fn_inner)
    c2w.append(c2w_inner)
    
    data['w2c'] = w2c_inner
    data['k'] = k_inner
    data['cam_id'] = cam_id_inner
    data['static_fn'] = fn_inner
    data['c2w'] = c2w_inner

    return data

def convert_cameras_gt_to_dynamic_train_meta(config, scene_path, cameras_gt_json):
    
    data = dict()
    
    cameras_info = cameras_gt_json

    data = dict()
    data['w'] = cameras_info[0]['width']
    data['h'] = cameras_info[0]['height']
    
    w2c, c2w, k, cam_id, fn = [], [], [], [], []    
    
    total_timesteps = 0 #cameras_info[0]['total_timesteps']
    
    total_cameras = len(os.listdir(os.path.join(scene_path, 'dynamic')))

    # For each timestep, loop over cameras.

    k_inner = []
    w2c_inner = []
    c2w_inner = []
    cam_id_inner = []
    fn_inner = []
    
    # Choose the number of cameras to process at this timestep:

    num_cams = total_cameras

    for c in range(num_cams):
        # Find the camera info for camera c at timestep t.
        curr_camera_info = [ci for ci in cameras_info if ci['camera_id'] == c and ci['frame'] == 0][0]
        k_inner.append(get_intrinsics(curr_camera_info))
        w2c_inner.append(curr_camera_info['w2c'])
        c2w_inner.append(curr_camera_info['c2w'])
        cam_id_inner.append(str(curr_camera_info['camera_id']))
        fn_inner.append(f"{scene_path}/static/{curr_camera_info['img_name']}.png")
        
    w2c.append(w2c_inner)
    k.append(k_inner)
    cam_id.append(cam_id_inner)
    fn.append(fn_inner)
    c2w.append(c2w_inner)
    
    data['w2c'] = w2c_inner
    data['k'] = k_inner
    data['cam_id'] = cam_id_inner
    data['static_fn'] = fn_inner
    data['c2w'] = c2w_inner

    return data

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
        print(f"Processing scene: {os.path.basename(sample['scene_dir'])}")

    
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






        # --- Prepare for GM1 training ---
        time_index = 1
        gt_images = dataset[seq_idx, time_index]['gt_images']
        cam_stack = scene.getTrainCameraObjects()
        for i, cam in enumerate(cam_stack):
            cam.original_image = gt_images[i].permute(1,2,0).to(device)


        # --- Train GM1 using ARAP regularization ---
        print("Starting GM1 refinement with ARAP-Regularization")

        train_GM1_simple_arap(scene, config, iterations=config.optimization.arap_iterations, neighbor_radius=0.2)

        gm1_checkpoint_path = os.path.join(scene_dir, "gm_checkpoints", "GM1.pth")
        torch.save(scene.gaussians.capture(), gm1_checkpoint_path)
        print(f"Saved GM1 with ARAP regularization to {gm1_checkpoint_path}")






if __name__ == "__main__":
    config = Config()
    wandb.init(project="blender_static_preprocessing_debug")
    train(config)
    wandb.finish()
