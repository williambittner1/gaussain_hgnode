# 1_static_preprocessing.py
# This script is used to preprocess the dataset and train the static gaussian models for each sequence in the dataset

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


# local imports
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer_gsplat import render, render_batch
# from gaussian_renderer_inria import render



@dataclass
class ExperimentConfig:
    data_path: str = f"/work/williamb/datasets"    # "/work/williamb/datasets"  or "/users/williamb/dev/gaussain_hgnode/data"
    dataset_name: str = "pendulum_100seq_250ts"
    dataset_path: str = f"{data_path}/{dataset_name}"
    gm_output_path: str = f"{dataset_path}/"
    data_device: str = "cuda"
# Static gaussian model optimization config
@dataclass
class OptimizationConfig:
    # iterations: int = 30_000          # number of iterations to train the static gaussian model
    # position_lr_init: float = 0.00016
    # position_lr_final: float = 1.6e-06
    # position_lr_delay_mult: float = 0.01
    # position_lr_max_steps: int = 30000
    # feature_lr: float = 0.0025
    # opacity_lr: float = 0.05
    # scaling_lr: float = 0.005
    # rotation_lr: float = 0.001
    semantic_feature_lr: float = 0.001
    # percent_dense: float = 0.01
    # lambda_dssim: float = 0.2
    # densification_interval: int = 100
    # opacity_reset_interval: int = 3000
    # densify_from_iter: int = 500
    # densify_until_iter: int = 1000
    # densify_grad_threshold: float = 0.0002
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
        wandb.log({f"{name}_{cam_idx}": wandb.Image(combined_image)})
    print(f"Logged render and gt")


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
    


def train_static_gaussian_model(scene, config, iterations = 30000):

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
    
    visualize_gaussians(scene, scene.gaussians, config, background, cam_stack=scene.getTrainCameraObjects()[:1], name=f"final_{scene.dataset_path.split('/')[-1]}")
    
        
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


class GM_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.experiment.dataset_path

        # List subdirectories that start with "scene"
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




def convert_cameras_gt_to_dynamic_train_meta(config, scene_path, cameras_gt_json):
    """
    Create a train_meta dictionary that includes:
      - dynamic_fn: a list of dictionaries (one per dynamic frame) with keys:
            "video_path": path to the video file,
            "timestep": the frame index in that video.
      - static_fn: a list of file paths to the static images for each camera (timestep 0).
      
    The cameras_gt_json is expected to be a list of dictionaries with keys such as:
      "id", "t", "w2c", "c2w", "img_name", "width", "height", "fx", "fy", etc.
      
    Note: The 5 dynamic cameras are assumed to be the same as the first 5 of the 25 static cameras.
    """

    data = {}
    # Use the resolution from the first entry (assumes all cameras share the same size)
    data['width'] = cameras_gt_json[0]['width']
    data['height'] = cameras_gt_json[0]['height']
    
    # Initialize lists to be stored in the meta file.
    # (w2c, c2w, k, cam_id lists will correspond only to dynamic frames.)
    w2c, c2w, k, cam_id, dynamic_fn, static_fn = [], [], [], [], [], []

    def get_intrinsics(camera_info):
        fx = camera_info['fx']
        fy = camera_info['fy']
        cx = camera_info['width'] / 2
        cy = camera_info['height'] / 2
        return [[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]]
    
    # --- Build static_fn entries ---
    # For each camera (all 25), look for the entry with t == 0 and build the static image path.
    for entry in cameras_gt_json:
        if entry['frame'] == 0:
            cam_idx = entry['camera_id']
            # Construct the expected static image path.
            # (Adjust the folder structure if needed.)
            static_img_path = os.path.join(scene_path, "static", f"Cam_{cam_idx:03d}", f"{entry['img_name']}.png")
            static_fn.append(static_img_path)
    
    # --- Build dynamic_fn entries for dynamic cameras (first 5) ---
    for cam_index in range(5):
        # Build the dynamic video filename for this camera.
        video_filename = f"Cam_{cam_index:03d}_0000-0200.mp4"
        video_path = os.path.join(scene_path, "dynamic", video_filename)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Expected video for camera {cam_index} not found: {video_path}")
        
        # Open the video to determine total timesteps.
        cap = cv2.VideoCapture(video_path)
        total_timesteps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # For each timestep in this video, add a dynamic entry.
        for t in range(total_timesteps):
            # Find the corresponding GT entry for this camera and timestep.
            # (Assumes the GT JSON contains an entry for each frame for each camera.)
            matches = [ci for ci in cameras_gt_json if ci['camera_id'] == cam_index and ci['frame'] == t]
            if not matches:
                # If no matching entry exists, skip this timestep.
                continue
            curr_camera_info = matches[0]
            
            # Save intrinsics and extrinsics for this dynamic frame.
            k.append(get_intrinsics(curr_camera_info))
            w2c.append(curr_camera_info['w2c'])
            c2w.append(curr_camera_info['c2w'])
            cam_id.append(str(cam_index))
            
            # Add the dynamic_fn entry with video path and current timestep.
            dynamic_fn.append({
                "video_path": video_path,
                "timestep": t
            })
    
    # Pack everything into the data dictionary.
    data['w2c'] = w2c
    data['c2w'] = c2w
    data['k'] = k
    data['cam_id'] = cam_id
    data['dynamic_fn'] = dynamic_fn
    data['static_fn'] = static_fn

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

    # wandb.init(project="blender_static_preprocessing")
    device = config.experiment.data_device
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")


    ##########################################
    # 0. Set up scene and empty gaussians
    ##########################################
    
    # Create dataset and dataloader
    dataset = GM_Dataset(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    
    ##########################################
    # 1. Train and save static gaussian models 
    ##########################################

    for i in range(len(dataset)):

        scene_path = dataset[i]

        # convert cameras_gt.json to train_meta.json
        # print(f"Converting cameras_gt.json to train_static_meta.json for {scene_path}")
        cameras_gt_path = os.path.join(scene_path, 'cameras_gt.json')
        # train_static_meta_path = os.path.join(scene_path, 'train_static_meta.json')

        # if not os.path.exists(train_static_meta_path):
        #     with open(cameras_gt_path, 'r') as f:
        #         cameras_gt_json = json.load(f)
        #     train_static_meta_json = convert_cameras_gt_to_static_train_meta(config, scene_path, cameras_gt_json)
        #     with open(train_static_meta_path, 'w') as f:
        #         json.dump(train_static_meta_json, f)
        #     print(f"Saved static-video-based train_meta.json to {train_static_meta_path}")
        # else:
        #     print(f"train_meta.json already exists for {scene_path}")  

        # if not os.path.exists(os.path.join(scene_path, "train_dynamic_meta.json")):
        #     print(f"Converting cameras_gt.json to train_dynamic_meta.json for {scene_path}")
        #     cameras_gt_path = os.path.join(scene_path, 'cameras_gt.json')
        #     train_dynamic_meta_path = os.path.join(scene_path, 'train_dynamic_meta.json')
        #     with open(cameras_gt_path, 'r') as f:
        #         cameras_gt_json = json.load(f)
        #     train_dynamic_meta_json = convert_cameras_gt_to_dynamic_train_meta(config, scene_path, cameras_gt_json)
        #     with open(train_dynamic_meta_path, 'w') as f:
        #         json.dump(train_dynamic_meta_json, f)
        #     print(f"Saved dynamic-video-based train_meta.json to {train_dynamic_meta_path}")

        # if not os.path.exists(os.path.join(scene_path, "train_meta.json")):
        print(f"Converting cameras_gt.json to train_meta.json for {scene_path.split('/')[-1]}")
        train_meta_path = os.path.join(scene_path, "train_meta.json")
        with open(cameras_gt_path, 'r') as f:
            cameras_gt_json = json.load(f)
        train_meta_json = convert_cameras_gt_to_train_meta(config, scene_path, cameras_gt_json)
        with open(train_meta_path, 'w') as f:
            json.dump(train_meta_json, f)
        print(f"Saved train_meta.json to {train_meta_path}")

        scene = Scene(config=config, scene_path=scene_path)
        scene.gaussians = GaussianModel(sh_degree=3)

        scene = train_static_gaussian_model(scene, config, iterations=config.optimization.iterations)

        gaussian_checkpoint_path = os.path.join(scene_path, "gm_checkpoints", f"chkpnt{dataset.scene_dirs[i].split('/')[-1]}.pth")
        os.makedirs(os.path.dirname(gaussian_checkpoint_path), exist_ok=True)
        torch.save(scene.gaussians.capture(), gaussian_checkpoint_path)
        print(f"Saved static gaussian model with {len(scene.gaussians._xyz)} gaussians to {gaussian_checkpoint_path}")






if __name__ == "__main__":
    config = Config()
    wandb.init(project="blender_static_preprocessing_debug")
    train(config)
    wandb.finish()