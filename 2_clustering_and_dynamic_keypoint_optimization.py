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

# local imports
from scene import Scene
from scene.gaussian_model import GaussianModel
from pointnet.semantic_segmentation.pointnet2_sem_seg import get_model
from gaussian_renderer_gsplat import render
from models.pre_models import PointNetSegmenter, Clusterer
from models.gnode import GraphNeuralODE
from encoders.explicit_encoder import ExplicitEncoder

@dataclass
class ExperimentConfig:
    project_dir: str = "/users/williamb/dev/gaussain_hgnode"
    data_path: str = f"{project_dir}/data"     # "/work/williamb/datasets" 
    dataset_name: str = "pendulum_3seq_25ts"
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



class GaussianModelDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.experiment.dataset_path
        self.samples = []
        self.to_tensor = T.ToTensor()

        # List all scene directories inside the dataset path
        scene_dirs = sorted([
            os.path.join(self.dataset_path, d)
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ])

        for scene_dir in scene_dirs:
            static_dir = os.path.join(scene_dir, "static")
            dynamic_dir = os.path.join(scene_dir, "dynamic")
            gm_checkpoint_dir = os.path.join(scene_dir, "gm_checkpoints")
            
            # --- Process static images (t = 0) ---
            if os.path.isdir(static_dir):
                static_files = sorted([
                    f for f in os.listdir(static_dir)
                    if f.endswith(".png")
                ])
                for fname in static_files:
                    camera_id = os.path.splitext(fname)[0]  # e.g., "cam001"
                    gt_image_path = os.path.join(static_dir, fname)
                    gm_checkpoint_path = os.path.join(gm_checkpoint_dir, "gm_ckpnt0.pth")
                    
                    sample = {
                        "scene_dir": scene_dir,
                        "camera_id": camera_id,
                        "timestep": 0,
                        "gt_image_path": gt_image_path,
                        "gm_checkpoint_path": gm_checkpoint_path
                    }
                    self.samples.append(sample)

            # --- Process dynamic videos (t > 0) ---
            if os.path.isdir(dynamic_dir):
                video_files = sorted([
                    f for f in os.listdir(dynamic_dir)
                    if f.endswith(".mp4")
                ])
                for fname in video_files:
                    camera_id = os.path.splitext(fname)[0]  # e.g., "cam001"
                    video_path = os.path.join(dynamic_dir, fname)
                    
                    # Open video to determine frame count.
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Warning: Could not open video {video_path}")
                        continue
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # Assume frame 0 is reserved for t=0 (static) and add samples for t >= 1.
                    for t in range(1, frame_count):
                        sample = {
                            "scene_dir": scene_dir,
                            "camera_id": camera_id,
                            "timestep": t,
                            "video_path": video_path,
                            "frame_index": t
                        }
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        timestep = sample["timestep"]

        if timestep == 0:
            # Static sample: load the ground truth image from file.
            gt_image = Image.open(sample["gt_image_path"]).convert("RGB")
            gt_image_tensor = self.to_tensor(gt_image)
            # Load the pretrained Gaussian model checkpoint.
            gm_checkpoint = torch.load(
                sample["gm_checkpoint_path"],
                map_location=self.config.experiment.data_device
            )
            return {
                "scene_dir": sample["scene_dir"],
                "camera_id": sample["camera_id"],
                "timestep": timestep,
                "gt_image": gt_image_tensor,
                "gm_checkpoint": gm_checkpoint
            }
        else:
            # Dynamic sample: extract the frame from the video.
            video_path = sample["video_path"]
            frame_index = sample["frame_index"]
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gt_image = Image.fromarray(frame)
            gt_image_tensor = self.to_tensor(gt_image)
            return {
                "scene_dir": sample["scene_dir"],
                "camera_id": sample["camera_id"],
                "timestep": timestep,
                "gt_image": gt_image_tensor
            }

class GM_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.experiment.dataset_path
        self.scenes = []
        self.gaussian_models = []
        self.gt_images = {}  # Dictionary to store gt images by scene_idx
        self.gt_videos = {}  # Dictionary to store gt videos by scene_idx
        self.to_tensor = T.ToTensor()

        # List subdirectories that start with "scene"
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
            # Load scene and gaussians
            scene = Scene(config=self.config, scene_path=sequence)
            scene.gaussians = GaussianModel(sh_degree=3)    
            cam_stack = scene.getTrainCameraObjects()
            
            gaussian_checkpoint = os.path.join(
                sequence, 
                "gm_checkpoints", 
                f"chkpnt30k_{sequence.split('/')[-1]}.pth"
            )
            scene.load_gaussians_from_checkpoint(gaussian_checkpoint, scene.gaussians, self.config.optimization)
            
            self.scenes.append(scene)
            self.gaussian_models.append(scene.gaussians)

            # Load ground truth images and videos
            self.gt_images[idx] = {}
            self.gt_videos[idx] = {}
            
            # Load static images (t=0)
            static_dir = os.path.join(sequence, "static")
            if os.path.exists(static_dir):
                for img_file in sorted(os.listdir(static_dir)):
                    if img_file.endswith('.png'):
                        cam_id = os.path.splitext(img_file)[0]  # e.g., "cam001"
                        img_path = os.path.join(static_dir, img_file)
                        img = Image.open(img_path).convert("RGB")
                        self.gt_images[idx][cam_id] = self.to_tensor(img)

            # Load dynamic videos (t>0)
            dynamic_dir = os.path.join(sequence, "dynamic")
            if os.path.exists(dynamic_dir):
                for video_file in sorted(os.listdir(dynamic_dir)):
                    if video_file.endswith('.mp4'):
                        cam_id = os.path.splitext(video_file)[0]
                        video_path = os.path.join(dynamic_dir, video_file)
                        cap = cv2.VideoCapture(video_path)
                        frames = []
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = self.to_tensor(frame)
                            frames.append(frame)
                        cap.release()
                        self.gt_videos[idx][cam_id] = torch.stack(frames)

        print(f"Loaded {len(self.scenes)} scenes with their gaussian models and ground truth data.")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        """Get a scene, its cameras, gaussian model, and ground truth data."""
        return {
            'scene': self.scenes[idx],
            'gaussians': self.gaussian_models[idx],
            'scene_path': self.scene_dirs[idx],
            'gt_images': self.gt_images[idx],
            'gt_videos': self.gt_videos[idx]
        }

    def get_all_scenes(self):
        """Return all scenes."""
        return self.scenes

    def get_all_gaussians(self):
        """Return all gaussian models."""
        return self.gaussian_models

    def get_scene_and_gaussian(self, idx):
        """Get a specific scene and its gaussian model."""
        return self.scenes[idx], self.gaussian_models[idx]

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



def train():
    """
    Main training function.
    """

    config = Config()
    device = config.experiment.data_device
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")



    ##########################################  
    # 0. Load Dataset
    ##########################################
    dataset = GM_Dataset(config)
    
    # Get all data for a scene
    data = dataset[0]
    scene = data['scene']
    gaussians = data['gaussians']
    gt_images = data['gt_images']  # Dict of static images by camera ID
    gt_videos = data['gt_videos']  # Dict of videos by camera ID

    # Get specific ground truth data
    static_image = dataset.get_gt_data(scene_idx=0, cam_id="cam001", timestep=0)
    video_frame = dataset.get_gt_data(scene_idx=0, cam_id="cam001", timestep=5)
    all_frames = dataset.get_gt_data(scene_idx=0, cam_id="cam001")  # All frames for one camera



    ##########################################  
    # 1. Model Setup
    ##########################################
    pointnet_ssg_model = get_model(num_classes=13)
    checkpoint = torch.load(os.path.join(config.experiment.project_dir, "pointnet", "semantic_segmentation", "best_model.pth"), 
                          map_location=torch.device('cuda'))
    pointnet_ssg_model.load_state_dict(checkpoint['model_state_dict'])
    pointnet_ssg_model.eval()

    segmenter = PointNetSegmenter()

    clusterer = Clusterer()

    encoder = ExplicitEncoder()
    encoder = Encoder()

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


    num_params = count_trainable_parameters(model)
    print(f"Total trainable parameters: {num_params}")
    wandb.log({"total_trainable_params": num_params})

    optimizer = optim.Adam(
        list(model.parameters()),
        lr=config.experiment.learning_rate
    )

    loss_fn = nn.MSELoss(reduction="mean")


    current_segment_length = config.optimization.initial_timesteps

    num_clusters = 4  

    ##########################################
    # 3. Train Loop
    ##########################################
    epoch_bar = tqdm(range(config.optimization.epochs), desc=f"Training")
    for epoch in epoch_bar:
        torch.cuda.empty_cache()
        epoch_loss = 0.0

        # For simplicity, loop over scenes one by one (you can add batching later)
        for scene_idx in range(len(dataset)):
            data = dataset[scene_idx]
            scene = data['scene']
            gaussians = data['gaussians']

            # ---- Semantic Segmentation ----
            # Get gaussian positions (N,3) and colors (N,3), normalized to [0,1].
            gaussian_positions = gaussians.get_xyz  # (N, 3)
            gaussian_colors = gaussians.colors.float() / 255.0  # (N, 3)
            # Run the segmentation module (PointNetSegmenter).
            semantic_labels = segmenter(gaussian_positions, gaussian_colors)
            gaussians.semantic_labels = semantic_labels  # store labels for later use or visualization
            
            # ---- Clustering ----
            cluster_labels, cluster_colors = clusterer.cluster(gaussian_positions, semantic_labels, num_clusters)
            gaussians.semantic_cluster_labels = cluster_labels
            gaussians.cluster_color = cluster_colors
            # (Optional) Visualize cluster colors for debugging.
            if epoch == 0:
                visualize_gaussians_cluster_colors(scene, gaussians, config, background, scene.getTrainCameraObjects()[:1], iteration=epoch)
            

            # TODO: Encode the initial graph including nodes and edges. 
            # In the beginning I have a dense set of gaussian nodes: Gaussian nodes.
            # Sparsify this set of nodes to create a set of keypoint nodes: Keypoint nodes.
            # Cluster/Assign the dense set of gaussian nodes to the keypoint nodes.
            # Create a set of edges between the gaussian nodes and their corresponding keypoint nodes.
            # Create a set of edges between the keypoint nodes themselves.
            # I also want dynamic edges constructed between gaussian nodes from different keypoints every time they come close to each other.
            # The edges should be weighted by the distance between the nodes.
            # Edges that are too far away should be removed.

            # 0. Cluster/sparsify gaussian nodes:
            #   - clustering of the gaussians through subsampling in Hyperspace (Position, Velocity and Semantic Features)

            # 1. Initialize gaussian nodes: 
            #   - Encode gaussian node level Latent Semantic Features (PointNet++ Semantic Segmentation)
            #   - Set Relative Position (3D)

            # 2. Initialize keypoint nodes:
            #   - clustering of the gaussians through subsampling in Hyperspace (Position, Velocity and Semantic Features)
            #   - Encoding keypoint node level Latent Semantic Features through PointNet++ Encoder Classification)
            # Then the edges are initialized


            ##########################################
            # Train initial Gaussian Model for t=0 (GM0)
            ##########################################


            ##########################################
            # Optimize GM1 with ARAP-Regularization
            ##########################################
            # -> velocity and angular velocity via finite difference between GM0 and GM1



            ##########################################
            # Subsample fine and coarse nodes from gaussians 
            ##########################################
            # - Dense Gaussians are rigidly/softly connected to the sparse gaussians (either by hard-/soft-assignment)
            # - fine nodes V_h in the Fine Graph G_h Sparse Gaussians (e.g. 30% subsampling factor)
            # - coarse nodes V_l in the Coarse Graph G_l Sparse Gaussians (e.g. 5% subsampling factor)
            

            ##########################################
            # Train - Loop
            ##########################################


                ##########################################
                # Semantic Encoder
                ##########################################
                # Encode gaussian node level Latent Semantic Features (PointNet++ Semantic Segmentation)
                # (PointNet++ Semantic Segmentation on the gaussians)


                ##########################################
                # Multi-Scale Message Passing with Graph Neural ODE 
                ##########################################

                # returns: z_traj (trajectory of the sparse gaussians, fine nodes and coarse nodes)
                
                # 4 directed Graphs:
                # - G_h(V_h, E_h_h, E_h_w)
                #       Fine (high-res) Graph (Sparse Gaussian Nodes)
                # - G_l(V_l, E_l_l, E_l_w)
                #       Coarse (low-res) Graph (Abstract Nodes)
                # - G_h_l(V_h, V_l, E_h_l)
                #       Fine -> Coarse Graph    
                # - G_l_h(V_l, V_h, E_l_h)
                #       Coarse -> Fine Graph   
                

                #################
                # Fine Graph G_h (V_h, E_h_h, E_h_w):
                #################

                # v_h = [pos, dt_pos, quat, dt_quat, neighbor_vectors, wedge, dt_wedge, color, semantic_features]
                # e_ij = [diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij, dt_wedge_ij]
                # e_h_w = [distance, vel_crossprod_distance, abs_vel_crossprod_distance, dt_wedge_distance]




                # Nodes V_h (Gaussian Nodes):
                #########################

                # v_h = [pos, dt_pos, quat, dt_quat, abc, wedge, dt_wedge, rgb, semantic]

                # - pos (3D) (dynamic)  
                # - dt_pos (3D) (dynamic)
                # - quat (4D) (dynamic)
                # - dt_quat (3D) (dynamic)
                # - abc (9D) (neighbor vectors) (either constant neighbors or dynamically updated neighbors via k-NN)
                # - wedge (1D) (dynamic) (computed from the 3 nearest neighbors)
                # - dt_wedge (1D) (dynamic) (dt_wedge = wedge(dt_a, b, c)+wedge(a, dt_b, c)+wedge(a, b, dt_c))
                # - color (3D) (constant, potentially dynamic in the future)
                # - semantic_features (e.g. 32D) (constant)  


                # Edges E_h_h, E_h_w:  
                ###################

                # Intra-Graph Edges:
                # e_ij = [diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij, dt_wedge_ij]

                # World Edges:
                # e_h_w = [diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij, dt_wedge_ij]

                # - diff_ij (3D)                (relative position between node i and node j)
                # - dist_ij (1D)                (squared distance (x_i-x_j)^2)
                # - vel_crossprod_ij (3D)       (cross product of vel_i and vel_j) (Do both nodes move in the same direction?)
                # - abs_vel_crossprod_ij (1D)   (absolute value of vel_crossprod_ij) (Do both nodes move in the same direction?)
                # - dt_wedge_ij (1D)            (relative squared difference of dt_wedge_i - dt_wedge_j)




                # Message Passing Update m_h_h:
                ###############################################

                # message
                # m_h_h = MLP_Edge_h_h(v_h_1, v_h_2, e_h_h)

                # World Message:
                # m_h_w = MLP_Edge_h_w(v_h_1, v_h_2, e_h_w)

                # message weight
                # w_h_h =   MLP_Weight_h_h(v_h_1, v_h_2, e_h_h)  (learned weight)
                #         1/distance(v_h_1, v_h_2)  (inverse distance)
                #         1  (constant weight)

                # fine message aggregation
                # m_agg_h = sum(weight_h_h * m_h_h)

                # world message aggregation
                # m_agg_w = sum(weight_h_w * m_h_w)
                
                # fine node update
                # v_h_new = MLP_Node_h(v_h, m_agg_h, m_agg_w)

                # edge connectivity
                # radius-based connectivity (fine node to k nearest fine nodes)



                #################
                # Coarse Graph G_l(V_l, E_l_l, E_l_w)
                #################
                
                # Same as Fine Graph but on set of coarse nodes.

                # - v_l = [pos, dt_pos, quat, dt_quat, wedge, dt_wedge, rgb, semantic]
                # - e_ij = [diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij, dt_wedge_ij]
                # - e_l_w = [diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij, dt_wedge_ij]

                # message
                # m_l_l = MLP_Edge_l_l(v_l_1, v_l_2, e_l_l)

                # message world:
                # m_l_w = MLP_Edge_l_w(v_l_1, v_l_2, e_l_w)

                # message weight
                # w_l_l =   MLP_Weight_l_l(v_l_1, v_l_2, e_l_l)  (learned weight)
                #         1/distance(v_l_1, v_l_2)  (inverse distance)
                #         1  (constant weight)

                # coarse message aggregation
                # m_agg_l = sum(weight_l_l * m_l_l)

                # world message aggregation
                # m_agg_w = sum(weight_l_w * m_l_w)

                # coarse node update
                # v_l_new = MLP_Node_l(v_l, m_agg_l, m_agg_w)

                # edge connectivity
                # radius-based connectivity (coarse node to k nearest coarse nodes)


                ##########################################
                # Downsample Graph G_h_l(V_h, V_l, E_h_l)
                ##########################################

                # - v_h = [pos, dt_pos, quat, dt_quat, abc, wedge, dt_wedge, rgb, semantic]
                # - v_l = [pos, dt_pos, quat, dt_quat, wedge, dt_wedge, rgb, semantic]
                # - e_h_l = [diff_hl, dist_hl, vel_crossprod_hl, abs_vel_crossprod_hl, dt_wedge_hl]

                # message
                # m_h_l = MLP_Edge_h_l(v_h, v_l, e_h_l)

                # message aggregation
                # m_agg_l = sum(weight_h_l * m_h_l)

                # coarse node update
                # v_l_new = MLP_Node_l(v_l, m_agg_l)

                # edge connectivity
                # radius-based connectivity (fine node to k nearest coarse nodes)



                ##########################################
                # Upsample Graph G_l_h(V_l, V_h, E_l_h)
                ##########################################

                # - v_h = [pos, dt_pos, quat, dt_quat, abc, wedge, dt_wedge, rgb, semantic]
                # - v_l = [pos, dt_pos, quat, dt_quat, wedge, dt_wedge, rgb, semantic]
                # - e_l_h = [diff_lh, dist_lh, vel_crossprod_lh, abs_vel_crossprod_lh, dt_wedge_lh]

                # message
                # m_l_h = MLP_Edge_l_h(v_l_1, v_l_2, e_l_h)

                # message aggregation
                # m_agg_h = sum(weight_l_h * m_l_h)

                # fine node update
                # v_h_new = MLP_Node_h(v_h, m_agg_h)

                # edge connectivity
                # radius-based connectivity (coarse node to k nearest fine nodes)



                # Edge-Connectivity:
                # - Fine Edges:
                #   - radius-based connectivity (intra-graph)
                # - Coarse Edges:
                #   - radius-based connectivity (intra-graph)
                # - Fine -> Coarse Edges:
                #   - k-NN connectivity (inter-graph)
                # - Coarse -> Fine Edges:
                #   - k-NN connectivity (inter-graph)
                # - World Edges:
                #   - radius-based connectivity (exclude edges to fine/coarse nodes, that are close in Hyperspace or are already connected by a fine/coarse edge)








            # Fine Nodes V_h (Gaussian Nodes):

            # - Relative Position (3D)                      
            #   (constant)  (initialized by gaussian positions - keypoint positions)
            # - Position (3D)                               
            #   (dynamic)   (initialized with keypoint position + relative position) 
            #   (update: by keypoint position + rotated relative position, 
            #   where rotation is defined by the keypoint quaternion (q_t - q_0))
            # - Color (3D)                                  
            #   (constant)  
            #   (initialized with gaussian colors)
            # - Latent Semantic Features (e.g. 32D)         (constant)  (initialized with gaussian node level Latent Semantic Features through PointNet++ Semantic Segmentation on the gaussians)


            # CoarseKeypoint Nodes:
            # - Position (3D)                               (dynamic)   (initialized with gaussian positions)             (updated by velocity)
            # - Velocity (3D)                               (dynamic)   (initialized with ?)                              (updated by node velocity update net)
            # - Quaternion (4D)                             (dynamic)   (initialized with gaussian quaternions)           (updated by angular velocity)
            # - Angular Velocity (3D)                       (dynamic)   (initialized with ?)                              (updated by node angular velocity update net)
            # - Wedge Product of 3 closest keypoints (1D)   (dynamic)   (initialized with 3 closest neighbor)             (updated by 3 closest neighbors)
            #   -> Defines the spanned volume of the 4 keypoint nodes.
            # - Gradient of the Wedge Product (1D)          (dynamic)   (initialized with ?)                              (updated through finite difference of current and previous Wedge Product)
            #   -> Defines the change the spanned volume over time. Tells whether node is in rigid or soft region.
            # - Latent Semantic Features (e.g. 32D)         (constant)  (initialized via classifying the local segmented gaussians with PointNet++ Classifier all gaussians of the keypoint node)            
            #   -> latent shape/semantic feature space. 
            # - Color (3D)                                  (constant)  (initialized with gaussian colors)
            # - Augmented Working Space (49D)               (constant)  (initialized by copying and concatenating the initial node state or simply setting zeros)


            # The keypoint-gaussian edges have the following features:
            # - Weight (1D)

            # The keypoint-keypoint edges have the following features:
            # - Weight (1D)




            z0 = encoder(gaussians)

            # Optionally, incorporate conditioning from the explicit encoder.
            # For example, encoder(scene) might produce a tensor of shape (1, 1) per scene.
            conditioning = encoder(scene)  # Adjust according to your encoder implementation
            # In this design, we assume the conditioning is already included in the node state as the last 13 channels.
            # Otherwise, you may need to concatenate or add it in an appropriate manner.

            # ---- Run Graph Neural ODE ----
            traj = model(node_state, t_span)









    # Train-Loop
    for epoch in epoch_bar:

        epoch_start_time = time.time()
        epoch_loss = 0.0
        log = {}

        segment_duration = current_segment_length / config.optimization.framerate
        t_span = torch.linspace(0, segment_duration, current_segment_length, device=device, dtype=torch.float32)
        

        for scene in scenes:

            # Retrieve scene from dataset
            scene_data = gm_dataset[scene_idx]





            # 1. PointNet Semantic Segmentation
            # visualize_gaussians(scene, scene.gaussians, config, background, scene.getTrainCameraObjects()[:1], iteration=0)
            # scene.gaussians.run_pointnet_semantic_part_segmentation(pointnet_ssg_model)
            # visualize_gaussians_semantic_colors(scene, scene.gaussians, config, background, scene.getTrainCameraObjects()[:1], iteration=1)
            gaussian_positions = scene.gaussians.get_xyz  # (N,3)
            gaussian_colors = scene.gaussians.colors.float() / 255.0  # Normalize to [0,1]
            semantic_labels = segmenter(gaussian_positions, gaussian_colors)
            scene.gaussians.semantic_labels = semantic_labels


            # 2. Clustering
            # scene.gaussians.cluster_gaussians_with_semantic_features(num_clusters=13)
            # visualize_gaussians_cluster_colors(scene, scene.gaussians, config, background, scene.getTrainCameraObjects()[:1], iteration=2)
            # Run clustering based on positions and semantic labels.
            cluster_labels, cluster_colors = clusterer.cluster(gaussian_positions, semantic_labels, num_clusters=13)
            scene.gaussians.semantic_cluster_labels = cluster_labels
            scene.gaussians.cluster_color = cluster_colors



    print("Done.")
    


if __name__ == "__main__":
    wandb.init(project="2_clustering_and_dynamic_keypoint_debugging")
    train()
    wandb.finish()