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

from dataset import GM_Dataset

@dataclass
class ExperimentConfig:
    project_dir: str = "/users/williamb/dev/gaussain_hgnode"
    data_path: str = f"{project_dir}/data"     # "/work/williamb/datasets" 
    dataset_name: str = "pendulum_5seq_250ts"
    dataset_path: str = f"{data_path}/{dataset_name}"
    gm_output_path: str = f"{dataset_path}/"
    data_device: str = "cuda"


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



class GM_Dataset1(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.experiment.dataset_path
        self.scenes = []
        self.gaussian_models_t0 = []  # GM0
        self.gaussian_models_t1 = []  # GM1
        self.gt_images = {}  # Dictionary to store gt images by scene_idx
        self.gt_videos = {}  # Dictionary to store gt videos by scene_idx
        self.to_tensor = T.ToTensor()

        # List subdirectories that start with "sequence"
        self.scene_dirs = sorted([
            os.path.join(self.dataset_path, d)
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d)) and d.startswith("sequence")
        ])

        if len(self.scene_dirs) == 0:
            raise ValueError(f"No scene directories found in {self.dataset_path}.")
        
        # Load scenes, gaussians, and ground truth data during initialization
        self.load_all_data()

    def load_all_data(self):
        """Load all scenes, gaussian models, and ground truth data."""
        for idx, sequence in enumerate(self.scene_dirs):
            # Load scene
            scene = Scene(config=self.config, scene_path=sequence)
            cam_stack = scene.getTrainCameraObjects()
            
            # Load GM0 (t=0) and GM1 (t=1)
            gm0_checkpoint = os.path.join(sequence, "gm_checkpoints", "GM0.pth")
            gm1_checkpoint = os.path.join(sequence, "gm_checkpoints", "GM1.pth")
            
            if not (os.path.exists(gm0_checkpoint) and os.path.exists(gm1_checkpoint)):
                raise FileNotFoundError(f"Missing Gaussian model checkpoints in {sequence}/gm_checkpoints/")
            
            # Initialize and load GM0
            gm0 = GaussianModel(sh_degree=3)
            scene.load_gaussians_from_checkpoint(gm0_checkpoint, gm0, self.config.optimization)
            
            # Initialize and load GM1
            gm1 = GaussianModel(sh_degree=3)
            scene.load_gaussians_from_checkpoint(gm1_checkpoint, gm1, self.config.optimization)
            
            self.scenes.append(scene)
            self.gaussian_models_t0.append(gm0)
            self.gaussian_models_t1.append(gm1)








            # # Load ground truth images and videos
            # self.gt_images[idx] = {}
            # self.gt_videos[idx] = {}
            
            # # Load static images (t=0)
            # static_dir = os.path.join(sequence, "static")
            # if os.path.exists(static_dir):
            #     for img_file in sorted(os.listdir(static_dir)):
            #         if img_file.endswith('.png'):
            #             cam_id = os.path.splitext(img_file)[0]
            #             img_path = os.path.join(static_dir, img_file)
            #             img = Image.open(img_path).convert("RGB")
            #             self.gt_images[idx][cam_id] = self.to_tensor(img)

            # # Load dynamic videos (t>0)
            # dynamic_dir = os.path.join(sequence, "dynamic")
            # if os.path.exists(dynamic_dir):
            #     for video_file in sorted(os.listdir(dynamic_dir)):
            #         if video_file.endswith('.mp4'):
            #             cam_id = os.path.splitext(video_file)[0]
            #             video_path = os.path.join(dynamic_dir, video_file)
            #             cap = cv2.VideoCapture(video_path)
            #             frames = []
            #             while cap.isOpened():
            #                 ret, frame = cap.read()
            #                 if not ret:
            #                     break
            #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #                 frame = self.to_tensor(frame)
            #                 frames.append(frame)
            #             cap.release()
            #             self.gt_videos[idx][cam_id] = torch.stack(frames)

        print(f"Loaded {len(self.scenes)} scenes with their gaussian models (GM0 and GM1) and ground truth data.")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__1(self, idx):
        """Get a scene, its cameras, gaussian models, and ground truth data."""
        return {
            'scene': self.scenes[idx],
            'gaussians0': self.gaussian_models_t0[idx],  # GM0 (t=0)
            'gaussians1': self.gaussian_models_t1[idx],  # GM1 (t=1)
            'scene_path': self.scene_dirs[idx],
            'gt_images': self.gt_images[idx],
            'gt_videos': self.gt_videos[idx]
        }

    def __getitem__2(self, idx_tuple):
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
            'scene': self.scenes[sequence_idx],
            'GM0': self.gaussian_models_t0[sequence_idx],  # GM0 (t=0)
            'GM1': self.gaussian_models_t1[sequence_idx],  # GM1 (t=1)
            'scene_path': self.scene_dirs[sequence_idx],
            "gt_images": frames_tensor
        }

        return sample

    def __getitem__3(self, idx):
        # If idx is a tuple, extract both the sequence index and the specific time index.
        # Otherwise, treat idx as the sequence index and load all frames.
        if isinstance(idx, tuple):
            sequence_idx, time_index = idx
        else:
            sequence_idx = idx
            time_index = None

        # Get the scene directory path.
        scene_dir = self.scene_dirs[sequence_idx]
        dynamic_dir = os.path.join(scene_dir, "dynamic")
        video_files = sorted([f for f in os.listdir(dynamic_dir) if f.endswith(".mp4")])

        frames = []
        for video_file in video_files:
            video_path = os.path.join(dynamic_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            if time_index is None:
                # Load all frames for this camera.
                video_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame_tensor = self.to_tensor(frame)
                    video_frames.append(frame_tensor)
                cap.release()
                # Stack frames: shape (num_frames, C, H, W)
                video_tensor = torch.stack(video_frames)
            else:
                # Load only the specific frame.
                cap.set(cv2.CAP_PROP_POS_FRAMES, time_index)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError(f"Could not read frame {time_index} from {video_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                video_tensor = self.to_tensor(frame)
            frames.append(video_tensor)

        # If all frames were loaded, the result will be a list where each element is a tensor 
        # of shape (num_frames, C, H, W). Otherwise, each element is (C, H, W).
        if time_index is None:
            # Stack along a new dimension for cameras: shape (num_cameras, num_frames, C, H, W)
            frames_tensor = torch.stack(frames)
        else:
            # Stack to shape (num_cameras, C, H, W)
            frames_tensor = torch.stack(frames)

        sample = {
            'scene': self.scenes[sequence_idx],
            'GM0': self.gaussian_models_t0[sequence_idx],
            'GM1': self.gaussian_models_t1[sequence_idx],
            'scene_path': scene_dir,
            'gt_images': frames_tensor
        }
        return sample


    def __getitem__(self, idx_input):
        """
        If a tuple (sequence_idx, time_index) is provided, load the specific frame from each camera.
        Otherwise, return metadata (e.g. dynamic video file paths) so that frames can be loaded later.
        """
        # Allow idx_input to be an int or tuple
        if isinstance(idx_input, tuple):
            sequence_idx, time_index = idx_input
        else:
            sequence_idx = idx_input
            time_index = None

        scene_dir = self.scene_dirs[sequence_idx]
        dynamic_dir = os.path.join(scene_dir, "dynamic")
        # Get full video file paths for each camera
        video_paths = sorted([
            os.path.join(dynamic_dir, f)
            for f in os.listdir(dynamic_dir) if f.endswith(".mp4")
        ])

        if time_index is not None:
            # Load only the specified frame from each video
            frames = []
            for video_path in video_paths:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, time_index)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError(f"Could not read frame {time_index} from {video_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_tensor = self.to_tensor(frame)
                frames.append(frame_tensor)
            # Stack into tensor of shape (num_cameras, C, H, W)
            gt_images = torch.stack(frames)
        else:
            # Return the video file paths instead of loading all frames.
            # Later, during loss computation, you can load the needed frame using the path.
            gt_images = video_paths

        sample = {
            "scene_path": scene_dir,
            "video_paths": video_paths,
            "GM0": self.gaussian_models_t0[sequence_idx],
            "GM1": self.gaussian_models_t1[sequence_idx],
        }
        return sample




    def get_all_scenes(self):
        """Return all scenes."""
        return self.scenes

    def get_all_gaussians(self):
        """Return all gaussian model pairs (GM0, GM1)."""
        return list(zip(self.gaussian_models_t0, self.gaussian_models_t1))

    def get_scene_and_gaussians(self, idx):
        """Get a specific scene and its gaussian models."""
        return self.scenes[idx], self.gaussian_models_t0[idx], self.gaussian_models_t1[idx]

    def get_gt_data(self, scene_idx, cam_id=None, timestep=None):
        """
        Get ground truth data for a specific scene, camera, and timestep.
        
        Args:
            scene_idx: Index of the scene
            cam_id: Camera ID (e.g., "cam001"). If None, returns all cameras.
            timestep: Timestep to retrieve. If None, returns all timesteps.
                     If 0, returns static image. If >0, returns video frame.
        """
        if timestep == 0:
            data = self.gt_images[scene_idx]
            return data if cam_id is None else data.get(f"{cam_id}_img000")
        else:
            videos = self.gt_videos[scene_idx]
            if cam_id is None:
                return videos
            video = videos.get(f"{cam_id}_vid")
            return video if timestep is None else video[timestep-1] if video is not None else None


def visualize_gaussians(scene, gaussians, config, background, cam_stack, iteration=None):
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        gt_image = camera.original_image.cpu().numpy()
        
        H, W, C = rendered_image.shape
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)  # 2-pixel wide white line
        
        combined_image = np.concatenate([rendered_image, white_line, gt_image], axis=1)

        combined_image = (combined_image * 255).astype(np.uint8)
            
        wandb.log({f"combined_{cam_idx}": wandb.Image(combined_image)}, step=iteration)
    print(f"Logged render and gt")


def visualize_gaussians_semantic_colors(scene, gaussians, config, background, cam_stack, iteration=None):
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background, override_color=gaussians.semantic_class_color)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        gt_image = camera.original_image.cpu().numpy()
        
        H, W, C = rendered_image.shape
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)  # 2-pixel wide white line
        
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
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)  # 2-pixel wide white line
        
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


def extract_features(GM0s, GM1s, segmenter):
    """
    Extract features from batches of GM0 and GM1 for ODE integration.
    Args:
        GM0s: List of B GaussianModel objects at t=0
        GM1s: List of B GaussianModel objects at t=1
        segmenter: Segmenter object for semantic feature extraction
    Returns:
        z_h_batch: (B, N, feature_dim) - Batched concatenated features
    """
    # First collect all raw features from Gaussian models
    pos0_list = []
    pos1_list = []
    q0_list = []
    q1_list = []
    color_list = []
    semantic_list = []
    
    for gm0, gm1 in zip(GM0s, GM1s):
        pos0_list.append(gm0.get_xyz)
        pos1_list.append(gm1.get_xyz)
        q0_list.append(gm0.get_rotation)
        q1_list.append(gm1.get_rotation)
        color_list.append(gm0.colors.float() / 255.0)
        
    
    # Stack to create batched tensors
    pos0_batch = torch.stack(pos0_list)      # [B, N, 3]
    pos1_batch = torch.stack(pos1_list)      # [B, N, 3]
    q0_batch = torch.stack(q0_list)          # [B, N, 4]
    q1_batch = torch.stack(q1_list)          # [B, N, 4]
    color_batch = torch.stack(color_list)     # [B, N, 3]
    
    
    # Compute velocities in parallel
    dt_pos_batch = pos1_batch - pos0_batch    # [B, N, 3]
    
    # Compute angular velocities in parallel
    q0_inv_batch = quaternion_inverse(q0_batch)  # [B, N, 4]
    q_diff_batch = quaternion_multiply(q1_batch, q0_inv_batch)  # [B, N, 4]
    
    # Convert to angular velocity (omega)
    omega_batch = 2 * q_diff_batch[..., 1:] / (q_diff_batch[..., 0:1] + 1e-6)  # [B, N, 3]
    

    # Semantic Segmentation (PointNet++)
    semantic_features_batch = []
    semantic_labels_batch = []

    B = pos0_batch.shape[0]
    num_sample_points = 1024

    for i in range(B):
        pos = pos0_batch[i]  # [N, 3]
        colors = color_batch[i]  # [N, 3]
        N = pos.shape[0]
        # Randomly sample 1024 points for segmentation
        if N > num_sample_points:
            idx = torch.randperm(N)[:num_sample_points]
            sample_pos = pos[idx]
            sample_colors = colors[idx]
        else:
            # If we have fewer points, repeat some points
            idx = torch.randint(N, (num_sample_points,))
            sample_pos = pos[idx]
            sample_colors = colors[idx]
            
        # Get semantic features for sampled points
        sample_semantic_features, sample_semantic_labels = segmenter(sample_pos, sample_colors)  # [1024, feature_dim]
        
        # Assign features to all points based on nearest sampled point
        full_semantic_features, full_semantic_labels = get_nearest_neighbor_features(
            sample_pos, sample_semantic_features, sample_semantic_labels, pos
        )  # [N, feature_dim], [N]
       
        semantic_features_batch.append(full_semantic_features)
        semantic_labels_batch.append(full_semantic_labels)

    semantic_features_batch = torch.stack(semantic_features_batch)  # [B, N, feature_dim]
    semantic_labels_batch = torch.stack(semantic_labels_batch).unsqueeze(-1)  # [B, N, 1]

    # Store labels in gaussian models for visualization
    for i, gm0 in enumerate(GM0s):
        gm0.semantic_labels = semantic_labels_batch[i]


    # Create augmented working space
    B, N, _ = pos0_batch.shape
    augm_batch = torch.zeros(B, N, 49, device=pos0_batch.device)
    
    # Concatenate all features
    z_h_batch = torch.cat([
        pos0_batch,             # [B, N, 3]
        dt_pos_batch,           # [B, N, 3]
        q0_batch,               # [B, N, 4]
        omega_batch,            # [B, N, 3]
        color_batch,            # [B, N, 3]
        semantic_features_batch,# [B, N, num_classes]
        semantic_labels_batch,   # [B, N, 1]
        augm_batch              # [B, N, 49]
    ], dim=-1)                  # [B, N, 66]
    
    
    return z_h_batch

def custom_collate(batch):
    """
    Custom collate function that handles batched Scene objects with batch_size = B
    
    Returns:
        dict: Contains scene, GM0, GM1, frames_tensor, and pseudo_3d_gt
    """
    return {
        "scene": [item["scene"] for item in batch],  # List of B Scene objects
        "GM0": [item["GM0"] for item in batch],      # List of B GM0 models
        "GM1": [item["GM1"] for item in batch],      # List of B GM1 models
        "frames_tensor": torch.stack([item["frames_tensor"] for item in batch]),  # [B, ...]
        "pseudo_3d_gt": torch.stack([item["pseudo_3d_gt"] for item in batch])    # [B, ...]
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

    ##########################################  
    # 0. Load Dataset (GM0, GM1, gt_images, gt_videos)
    ##########################################

    train_path = os.path.join(config.experiment.dataset_path, "train")
    dataset = GM_Dataset(config, train_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

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

    segmenter = PointNetSegmenter(num_classes=2, num_sample_points=1024, knn_k=10, device="cuda")
    loss_fn = nn.MSELoss(reduction="mean")

    processor = MSGNODEProcessor(feature_dim=67, message_dim=64, hidden_dim=256)

    optimizer = optim.Adam(
        list(processor.parameters()), #+ list(pointnet_ssg_model.parameters()),
        lr=config.experiment.learning_rate
    )

    ##########################################
    # 3. Train Loop
    ##########################################

    epoch_bar = tqdm(range(config.optimization.ms_gnode_epochs), desc=f"Training")
    for epoch in epoch_bar:
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        epoch_loss = 0.0
        log = {}

        segment_duration = current_segment_length / config.optimization.framerate
        t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
        


        for batch in train_loader:
           
            scenes          = batch['scene']
            GM0s            = batch['GM0']
            GM1s            = batch['GM1']
            frames_tensors  = batch['frames_tensor']
            pseudo_3d_gts   = batch['pseudo_3d_gt']
            batch_size      = len(scenes)


            points = extract_features(GM0s, GM1s, segmenter)  # [B, N, feature_dim]
           

            ##########################################
            # Subsample fine and coarse nodes from gaussians 
            ##########################################
            # - Subsample Sparse Gaussians from all/dense Gaussians (e.g. 30% subsampling factor w.r.t. all Gaussians)
            # - Subsample Coarse Gaussians from Sparse Gaussians (e.g. 10% subsampling factor w.r.t. Sparse Gaussians)
            # - Dense Gaussians are rigidly/softly connected to the Sparse Gaussians (either by hard-/soft-assignment)
            # - fine nodes V_h in the Fine Graph G_h are created from Sparse Gaussians (e.g. 30% subsampling factor)
            # - coarse nodes V_l in the Coarse Graph G_l are created from Coarse Gaussians (e.g. 10% subsampling factor)
            
            
            # Downsample z_h to get z_l by randomly selecting a subset of nodes
            B, N, F = points.shape 
            num_fine_nodes = int(N * 0.7)
            num_coarse_nodes = int(N * 0.01)  # 1% of fine nodes become coarse nodes

            # Generate random indices for this batch
            # Same indices will be used for all items in batch, but different across batches
            fine_indices = torch.randperm(N, device=points.device)[:num_fine_nodes]     # [num_fine_nodes]
            coarse_indices = torch.randperm(N, device=points.device)[:num_coarse_nodes] # [num_coarse_nodes]
            
            # Select fine and coarse nodes using broadcasting
            z_h_0 = points[:, fine_indices]     # [B, num_fine_nodes, F]
            z_l_0 = points[:, coarse_indices] # [B, num_coarse_nodes, F]


            visualize_gaussians_semantic_colors(scenes[0], GM0s[0], config, background, scenes[0].getTrainCameraObjects()[:1], iteration=epoch)

            
            z_h_traj, z_l_traj = processor(z_h_0, z_l_0, t_span)




            # Compute losses
            photometric_loss_length = config.optimization.photometric_loss_length
            pseudo_loss_length = max(0, current_segment_length - photometric_loss_length)




            loss_pseudo3d = F.mse_loss(
                z_traj[:, :pseudo_loss_length, :, :7],
                batch_pseudo_gt[:, :pseudo_loss_length, :, :7]
            )





    

if __name__ == "__main__":
    # wandb.init(project="2_clustering_and_dynamic_keypoint_debugging")
    train()
    # wandb.finish()