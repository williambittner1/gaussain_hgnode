# global imports
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, fields
import os
import cv2
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors


# local imports
from scene import Scene
from scene.gaussian_model import GaussianModel
from pointnet.semantic_segmentation.pointnet2_sem_seg import get_model
from gaussian_renderer_gsplat import render
from models.segmenter import PointNetSegmenter
from models.msgnode import MSGNODEProcessor
from gaussian_renderer_gsplat import render_batch, render

from dataset import GM_Dataset

@dataclass
class ExperimentConfig:
    project_dir: str = "/users/williamb/dev/gaussain_hgnode"
    data_path: str = f"{project_dir}/data"     # "/work/williamb/datasets" 
    dataset_name: str = "pendulum_5seq_250ts"
    dataset_path: str = f"{data_path}/{dataset_name}"
    gm_output_path: str = f"{dataset_path}/"
    data_device: str = "cuda"
    
    framerate: int = 25
    dataset_initial_length: int = 10
    loss_threshold: float = 1e-3

# Static gaussian model optimization config
@dataclass
class OptimizationConfig:
    semantic_feature_lr: float = 0.001
    iterations = 30_000
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

    ms_gnode_epochs = 1000
    batch_size = 1
    initial_segment_length = 4
    photometric_loss_length = 3
    learning_rate = 1e-3

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
    pipeline: PipelineConfig = PipelineConfig()

# Modified visualization functions to handle list-based data
def visualize_gaussians(scene, gaussians, config, background, cam_stack, gt_frame, iteration=None):
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        
        gt_image = gt_frame[cam_idx].permute(1, 2, 0).cpu().numpy()
        H, W, C = rendered_image.shape
        
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)
        combined_image = np.concatenate([rendered_image, white_line, gt_image], axis=1)
        combined_image = (combined_image * 255).astype(np.uint8)
            
        wandb.log({f"combined_{cam_idx}": wandb.Image(combined_image)}, step=iteration)
    print(f"Logged render and gt")

def visualize_gaussians_semantic_colors(scene, gaussians, config, background, cam_stack, gt_frame, iteration=None):
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background, override_color=gaussians.semantic_class_color)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        gt_image = gt_frame[cam_idx].permute(1, 2, 0).cpu().numpy()
        
        H, W, C = rendered_image.shape
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)
        
        combined_image = np.concatenate([rendered_image, white_line, gt_image], axis=1)
        combined_image = (combined_image * 255).astype(np.uint8)
            
        wandb.log({f"combined_{cam_idx}": wandb.Image(combined_image)}, step=iteration)
    print(f"Logged render and gt")

def visualize_gaussians_cluster_colors(scene, gaussians, config, background, cam_stack, iteration=None):
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background, override_color=gaussians.cluster_color)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        gt_image = camera.original_image.cpu().numpy()
        
        H, W, C = rendered_image.shape
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)
        
        combined_image = np.concatenate([rendered_image, white_line, gt_image], axis=1)
        combined_image = (combined_image * 255).astype(np.uint8)
            
        wandb.log({f"combined_{cam_idx}": wandb.Image(combined_image)}, step=iteration)
    print(f"Logged render and gt")

def quaternion_inverse(q):
    """
    Compute the inverse of a quaternion [w,x,y,z].
    Args:
        q: Tensor of shape [..., 4]
    Returns:
        q_inv: Tensor of shape [..., 4]
    """
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions [w,x,y,z].
    Args:
        q1, q2: Tensors of shape [..., 4]
    Returns:
        q_prod: Tensor of shape [..., 4]
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)

def get_nearest_neighbor_features(points, features, labels, query_points, k=1):
    """
    Find nearest neighbors and get their features.
    Args:
        points: (N, 3) points that have features
        features: (N, F) features for each point
        query_points: (M, 3) points to assign features to
        k: number of nearest neighbors (default=1 for closest point assignment)
    Returns:
        nn_features: (M, F) features assigned from nearest neighbors
    """
    # Build KNN index
    points_np = points.detach().cpu().numpy()
    query_np = query_points.detach().cpu().numpy()
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points_np)
    distances, indices = nbrs.kneighbors(query_np)
    
    # Get features of nearest neighbors (just take the first neighbor if k>1)
    nn_features = features[torch.from_numpy(indices).to(features.device)[:, 0]]
    nn_labels = labels[torch.from_numpy(indices).to(labels.device)[:, 0]]

    return nn_features, nn_labels


def extract_features_batch(GM0s, GM1s, segmenter=None, include_semantic=True, include_augm=True):
    """
    Extract features from batches of GM0 and GM1 for ODE integration.
    Args:
        GM0s: List of B GaussianModel objects at t=0
        GM1s: List of B GaussianModel objects at t=1
        segmenter: Optional segmenter object for semantic feature extraction
        include_semantic: Whether to include semantic features
        include_augm: Whether to include augmentation features
    Returns:
        z_h_batch: (B, N, feature_dim) - Batched concatenated features
    """
    # First collect all raw features from Gaussian models
    pos0_list = []
    pos1_list = []
    q0_list = []
    q1_list = []
    color_list = []
    
    for gm0, gm1 in zip(GM0s, GM1s):
        pos0_list.append(gm0.get_xyz)
        pos1_list.append(gm1.get_xyz)
        q0_list.append(gm0.get_rotation)
        q1_list.append(gm1.get_rotation)
        if include_semantic:
            color_list.append(gm0.colors.float() / 255.0)
    
    # Stack to create batched tensors
    pos0_batch = torch.stack(pos0_list)      # [B, N, 3]
    pos1_batch = torch.stack(pos1_list)      # [B, N, 3]
    q0_batch = torch.stack(q0_list)          # [B, N, 4]
    q1_batch = torch.stack(q1_list)          # [B, N, 4]
    
    # Compute velocities
    dt_pos_batch = pos1_batch - pos0_batch    # [B, N, 3]
    
    # Compute angular velocities
    q0_inv_batch = quaternion_inverse(q0_batch)  # [B, N, 4]
    q_diff_batch = quaternion_multiply(q1_batch, q0_inv_batch)  # [B, N, 4]
    omega_batch = 2 * q_diff_batch[..., 1:] / (q_diff_batch[..., 0:1] + 1e-6)  # [B, N, 3]

    # Base features list
    feature_list = [
        pos0_batch,             # [B, N, 3]
        dt_pos_batch,           # [B, N, 3]
        q0_batch,               # [B, N, 4]
        omega_batch,            # [B, N, 3]
    ]

    if include_semantic:
        if segmenter is None:
            raise ValueError("Segmenter must be provided when include_semantic=True")
            
        color_batch = torch.stack(color_list)     # [B, N, 3]
        semantic_features_batch = []
        semantic_labels_batch = []
        B = pos0_batch.shape[0]
        num_sample_points = 1024

        for i in range(B):
            pos = pos0_batch[i]
            colors = color_batch[i]
            N = pos.shape[0]
            
            # Sample points for segmentation
            if N > num_sample_points:
                idx = torch.randperm(N)[:num_sample_points]
                sample_pos = pos[idx]
                sample_colors = colors[idx]
            else:
                idx = torch.randint(N, (num_sample_points,))
                sample_pos = pos[idx]
                sample_colors = colors[idx]
            
            # Get semantic features
            sample_semantic_features, sample_semantic_labels = segmenter(sample_pos, sample_colors)
            
            # Assign features to all points
            full_semantic_features, full_semantic_labels = get_nearest_neighbor_features(
                sample_pos, sample_semantic_features, sample_semantic_labels, pos
            )
            
            semantic_features_batch.append(full_semantic_features)
            semantic_labels_batch.append(full_semantic_labels)

        semantic_features_batch = torch.stack(semantic_features_batch)
        semantic_labels_batch = torch.stack(semantic_labels_batch).unsqueeze(-1)

        # Store labels for visualization
        for i, gm0 in enumerate(GM0s):
            gm0.semantic_labels = semantic_labels_batch[i]

        feature_list.extend([
            color_batch,
            semantic_features_batch,
            semantic_labels_batch,
        ])

    if include_augm:
        B, N, _ = pos0_batch.shape
        augm_batch = torch.zeros(B, N, 49, device=pos0_batch.device)
        feature_list.append(augm_batch)

    # Concatenate all features
    z_h_batch = torch.cat(feature_list, dim=-1)
    
    return z_h_batch

def extract_features(GM0, GM1, segmenter, include_semantic=False, include_augm=False, augm_dim=9):
    """Extract features from a single pair of Gaussian models"""
    # Process single sequence
    xyz0 = GM0.get_xyz  # [N, 3]
    rot0 = GM0.get_rotation  # [N, 4]
    xyz1 = GM1.get_xyz  # [N, 3]
    rot1 = GM1.get_rotation  # [N, 4]
    
    # Compute velocities
    vel = xyz1 - xyz0  # [N, 3]

    # Compute angular velocities
    q0_inv = quaternion_inverse(rot0)  # [N, 4]
    q_diff = quaternion_multiply(rot1, q0_inv)  # [N, 4]
    omega = 2 * q_diff[..., 1:] / (q_diff[..., 0:1] + 1e-6)  # [N, 3]

    feature_list = [xyz0, rot0, vel, omega]
        
    if include_semantic:
        if segmenter is None:
            raise ValueError("Segmenter must be provided when include_semantic=True")
            
        color = GM0.colors.float() / 255.0     # [N, 3]
        pos = xyz0
        N = pos.shape[0]
        num_sample_points = 1024
        
        # Sample points for segmentation
        if N > num_sample_points:
            sparse_idx = torch.randperm(N)[:num_sample_points]
            sparse_pos = pos[sparse_idx]
            sparse_colors = color[sparse_idx]
        else:
            sparse_idx = torch.randint(N, (num_sample_points,))
            sparse_pos = pos[sparse_idx]
            sparse_colors = color[sparse_idx]
        
        # Get semantic features for sampled points
        sparse_semantic_features, sparse_semantic_labels = segmenter(sparse_pos, sparse_colors)
        
        # Assign features to all points
        semantic_features, semantic_labels = get_nearest_neighbor_features(
                                                            sparse_pos, sparse_semantic_features, sparse_semantic_labels, pos
                                                        )

        # Store labels for visualization
        GM0.semantic_labels = semantic_labels

        feature_list.extend([
            color,
            semantic_features,
            semantic_labels,
        ])
        
    if include_augm:
        B, N, _ = xyz0.shape
        augm = torch.zeros(B, N, augm_dim, device=xyz0.device)
        feature_list.append(augm)

    features = torch.cat(feature_list, dim=-1)
        
    return features  # [N, F]

def custom_collate(batch):
    """
    Custom collate function that handles batched Scene objects with batch_size = B
    
    Returns:
        dict: Contains scene, GM0, GM1, gt_frame, and pseudo_3d_gt
    """
    return {
        "seq_idx": [item["seq_idx"] for item in batch],  # List of sequence indices
        "scene": [item["scene"] for item in batch],  # List of B Scene objects
        "GM0": [item["GM0"] for item in batch],      # List of B GM0 models
        "GM1": [item["GM1"] for item in batch],      # List of B GM1 models
        # "gt_image": torch.stack([item["gt_image"] for item in batch]),  # [B, ...]
        # "gt_video": torch.stack([item["gt_video"] for item in batch]),  # [B, ...]
        # "pseudo_3d_gt": torch.stack([item["pseudo_3d_gt"] for item in batch]),    # [B, ...]
        "gt_image": [item["gt_image"] for item in batch],  # List of [1, num_cams, H, W, 3]
        "gt_video": [item["gt_video"] for item in batch],  # List of [T, num_cams, H, W, 3]        
        "pseudo_3d_gt": [item["pseudo_3d_gt"] for item in batch],  # List of [T, N, 7]
        "z_h_0": [item["z_h_0"] for item in batch] if "z_h_0" in batch[0] else None,
        "z_l_0": [item["z_l_0"] for item in batch] if "z_l_0" in batch[0] else None
    }



# -------------------------------------------------------------------------------------------------
# Main Training Function
# -------------------------------------------------------------------------------------------------

def train():
    """
    Main training function.
    """

    config = Config()
    device = config.experiment.data_device
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    batch_size = config.optimization.batch_size
    current_segment_length = config.optimization.initial_segment_length
    photometric_loss_length = config.optimization.photometric_loss_length

    ##########################################  
    # 0. Load Dataset (GM0, GM1, gt_images, gt_videos)
    ##########################################

    train_path = os.path.join(config.experiment.dataset_path, "train")
    dataset = GM_Dataset(config, train_path)

    # test_path = os.path.join(config.experiment.dataset_path, "test")
    # test_dataset = GM_Dataset(config, test_path)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


    ##########################################  
    # 1. Model Setup
    ##########################################
    pointnet_ssg_model = get_model(num_classes=13)
    checkpoint = torch.load(os.path.join(config.experiment.project_dir, "pointnet", "semantic_segmentation", "best_model.pth"), 
                          map_location=torch.device('cuda'))
    pointnet_ssg_model.load_state_dict(checkpoint['model_state_dict'])
    pointnet_ssg_model.eval()

    segmenter = PointNetSegmenter(num_classes=2, num_sample_points=1024, knn_k=8, device="cuda")
    loss_fn = nn.MSELoss(reduction="mean")

    processor = MSGNODEProcessor(feature_dim=13, message_dim=64, hidden_dim=256, device=device)

    optimizer = optim.Adam(
        list(processor.parameters()), #+ list(pointnet_ssg_model.parameters()),
        lr=config.optimization.learning_rate
    )

    ##########################################
    # 2. Preprocess Dataset
    ##########################################

    for seq_idx in range(len(dataset)):
        scene = dataset.scenes[seq_idx]
        GM0 = dataset.GM0[seq_idx]
        GM1 = dataset.GM1[seq_idx]
        points = extract_features(GM0, GM1, segmenter, include_semantic=False, include_augm=False, augm_dim=9)
        N = points.shape[0]
        fine_indices = torch.randperm(N, device=points.device)
        coarse_indices = torch.randperm(N, device=points.device)[:int(N * 0.01)]
        dataset.fine_indices[seq_idx] = fine_indices
        dataset.coarse_indices[seq_idx] = coarse_indices
        dataset.set_z_h_0_z_l_0(
            seq_idx,
            points[fine_indices],
            points[coarse_indices]
        )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    ##########################################
    # 3. Train Loop
    ##########################################

    # Training loop
    epoch_bar = tqdm(range(config.optimization.ms_gnode_epochs), desc="Training")
    for epoch in epoch_bar:
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        epoch_loss = 0.0

        segment_duration = current_segment_length / config.experiment.framerate
        t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)

        for batch_idx, batch in enumerate(train_loader):
            scenes = batch['scene']
            GM0s = batch['GM0']
            GM1s = batch['GM1']
            gt_images = batch['gt_image']  # List of [T, num_cams, H, W, 3]
            gt_videos = batch['gt_video']  # List of [T, num_cams, H, W, 3]
            pseudo_3d_gts = batch['pseudo_3d_gt']  # List of [T, N, 7]
            z_h_0_list = batch['z_h_0']  # List of [N, F]
            z_l_0_list = batch['z_l_0']  # List of [N_coarse, F]
            seq_indices = batch['seq_idx']
            batch_size = len(scenes)

            loss_pseudo3d = 0.0
            loss_photo = 0.0
            photo_count = 0

            for b in range(batch_size):
                # Sequence-specific data
                z_h_0 = z_h_0_list[b].unsqueeze(0).to(device).detach()  # [1, N, F]
                z_l_0 = z_l_0_list[b].unsqueeze(0).to(device).detach()  # [1, N_coarse, F]
                pseudo_3d_gt = pseudo_3d_gts[b].to(device)  # [T, N, 7]
                gt_image = gt_images[b].to(device)  # [T, num_cams, H, W, 3]
                scene = scenes[b]
                GM0 = GM0s[b]
                GM1 = GM1s[b]

                # Get fine indices for this sequence
                seq_idx = seq_indices[b]  # Correct sequence index
                fine_indices = dataset.fine_indices[seq_idx].to(device)

                # Processor forward pass
                processor.ode_func.reset_edges()
                processor.ode_func.nfe = 0
                z_h_traj, z_l_traj = processor(z_h_0, z_l_0, t_span)  # [B=1, T, N, F]

                # Compute pseudo 3D loss
                pseudo_loss_length = max(0, current_segment_length - photometric_loss_length)
                seq_loss_pseudo3d = F.mse_loss(
                    z_h_traj[0, :pseudo_loss_length, :, :7],
                    pseudo_3d_gt[:pseudo_loss_length, fine_indices, :7]
                )
                loss_pseudo3d += seq_loss_pseudo3d

                # Compute photometric loss
                tmp_gaussians_pred = GM0.clone()
                num_train_cams = 5
                for t in range(pseudo_loss_length, current_segment_length):
                    tmp_gaussians_pred.update_gaussians(
                        z_h_traj[0, t, :, :3],
                        z_h_traj[0, t, :, 3:7]
                    )

                    cam_stack = scene.getTrainCameraObjects()
                    viewpoint_cams = cam_stack[:num_train_cams]
                    render_pkg_pred = render_batch(viewpoint_cams, tmp_gaussians_pred, config.pipeline, background)
                    pred_rendered_image = render_pkg_pred["render"].permute(1, 0, 2, 3)  # [C, H, W, 3]
                    gt_image_t = gt_image[:num_train_cams].permute(0, 2, 3, 1)  # [C, H, W, 3]
                    loss_i = F.mse_loss(pred_rendered_image, gt_image_t)
                    loss_photo += loss_i
                    photo_count += 1

            # Average losses over batch
            loss_pseudo3d /= batch_size
            loss_photo /= photo_count if photo_count > 0 else 1
            batch_loss = (
                loss_pseudo3d * pseudo_loss_length +
                loss_photo * photometric_loss_length
            ) / current_segment_length
            # batch_loss = loss_pseudo3d

            optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        epoch_loss /= len(train_loader)


        if epoch % 10 == 0:
            visualize_gaussians(scenes[0], GM0s[0], config, background, scenes[0].getTrainCameraObjects()[:1], gt_images[0], iteration=epoch)

        epoch_bar.set_postfix({
            'Loss': f'{batch_loss.item():.7f}',
            # 'nfe': model.func.nfe,
            # 'seg_len': current_segment_length,
            # 'it/s': epochs_per_sec,
            #'test_loss': log['test_loss']
        })

    
        # Increment segment length and update Pseudo-GT
        if epoch > 0 and loss_photo < config.experiment.loss_threshold:
            current_segment_length += 1
            

            for seq in range(len(dataset)):
                z_h_traj, z_l_traj = processor(dataset[seq]["z_h_0"], dataset[seq]["z_l_0"], t_span)
                last_xyz = z_h_traj[:, -1, :, :3].detach()  # Detach here
                last_quat = z_h_traj[:, -1, :, 3:7].detach()  # Detach here
                dataset[seq].update_pseudo_3d_gt(seq, last_xyz.to(device), last_quat.to(device))
                dataset[seq].update_image(seq, current_segment_length)
                dataset[seq].update_video(seq, current_segment_length)

            


if __name__ == "__main__":
    wandb.init(project="2_dynamic_training_debugging")
    train()
    wandb.finish()




