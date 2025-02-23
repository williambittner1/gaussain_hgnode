# Standard imports
import os
import json
from PIL import Image
import torch
import cv2
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List

# Local imports
from scene import Scene
from scene.gaussian_model import GaussianModel

class PreRenderedGTDataset(torch.utils.data.Dataset):
    def __init__(self, gt_base_dir, split: str, num_timesteps: int, cam_ids: List[int], transform=None):
        """
        Args:
            gt_base_dir (str): Base directory where images were saved.
            split (str): 'train' or 'test'
            num_timesteps (int): Number of timesteps per sequence.
            cam_ids (List[int]): List of camera IDs.
            transform (callable, optional): Transform to apply to the image.
        """
        self.samples = []
        self.transform = transform if transform is not None else transforms.ToTensor()
        # Assume each scene is stored in a subfolder under gt_base_dir/split/
        split_dir = os.path.join(gt_base_dir, split)
        scene_dirs = sorted(os.listdir(split_dir))
        for scene in scene_dirs:
            scene_path = os.path.join(split_dir, scene)
            for cam_id in cam_ids:
                cam_folder = os.path.join(scene_path, f"{cam_id:03d}")
                if not os.path.exists(cam_folder):
                    continue
                for t in range(num_timesteps):
                    filename = f"render_{cam_id:03d}_timestep_{t:05d}.jpg"
                    filepath = os.path.join(cam_folder, filename)
                    if os.path.exists(filepath):
                        self.samples.append({
                            "scene": scene,
                            "cam_id": cam_id,
                            "timestep": t,
                            "filepath": filepath
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample["cam_id"], sample["timestep"]


class ControlPointDataset(Dataset):
    """
    Dataset for ground-truth control point trajectories.
    
    Expects:
      - gt_xyz_cp: Tensor of shape [B, T, num_controlpoints, 3]
      - gt_rot_cp: Tensor of shape [B, T, num_controlpoints, 4]
      
    Each item is a dictionary with keys:
      "gt_xyz_cp": [T, num_controlpoints, 3]
      "gt_rot_cp": [T, num_controlpoints, 4]
    """
    def __init__(self, gt_xyz_cp, gt_rot_cp):
        self.gt_xyz_cp = gt_xyz_cp
        self.gt_rot_cp = gt_rot_cp
        self.pseudo_gt_xyz_cp = None
        self.pseudo_gt_rot_cp = None
        self.pseudo_gt = None

    def set_pseudo_gt(self, pseudo_gt_xyz_cp, pseudo_gt_rot_cp):
        self.pseudo_gt_xyz_cp = pseudo_gt_xyz_cp # [B, T, N, 3]
        self.pseudo_gt_rot_cp = pseudo_gt_rot_cp # [B, T, N, 4]
        self.pseudo_gt = torch.cat([self.pseudo_gt_xyz_cp, self.pseudo_gt_rot_cp], dim=-1) # [B, T, N, 7]
        
    def __len__(self):
        return self.gt_xyz_cp.shape[0]

    def __getitem__(self, idx):
        return {
            "gt_xyz_cp": self.gt_xyz_cp[idx],  # [T, num_controlpoints, 3]
            "gt_rot_cp": self.gt_rot_cp[idx],     # [T, num_controlpoints, 4]
            "pseudo_gt_xyz_cp": self.pseudo_gt_xyz_cp[idx] if self.pseudo_gt_xyz_cp is not None else None,  # [T, num_controlpoints, 3]
            "pseudo_gt_rot_cp": self.pseudo_gt_rot_cp[idx] if self.pseudo_gt_rot_cp is not None else None,     # [T, num_controlpoints, 4]
            "pseudo_gt": self.pseudo_gt[idx] if self.pseudo_gt is not None else None     # [T, num_controlpoints, 7]    
        }

class GM_Dataset(Dataset):
    """
    Dataset for GM0, GM1, gt_videos and scene_path
    """

    def __init__(self, config, dataset_path):
        self.config = config
        self.to_tensor = T.ToTensor()
        self.dataset_path = dataset_path
        self.scene_dirs = sorted([
            os.path.join(self.dataset_path, d)
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d)) and d.startswith("sequence")
        ])
        if len(self.scene_dirs) == 0:
            raise ValueError(f"No scene directories found in {self.dataset_path}.")
        
        self.scenes = []
        self.GM0 = []  
        self.GM1 = []  
        self.frames_tensor = []
        self.pseudo_3d_gt = []

        self.initialize_data()


    def initialize_data(self):
        # Load scenes (including camera objects) and gaussians during initialization
        
        for seq_idx, sequence in enumerate(self.scene_dirs):
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
            print(f"Loaded GM0 with {gm0._xyz.shape[0]} points")
            
            # Initialize GM1 and update its parameters from checkpoint
            gm1 = GaussianModel(sh_degree=3)
            checkpoint = torch.load(gm1_checkpoint, map_location='cuda')
            gm1.update_from_checkpoint(checkpoint)
            print(f"Loaded GM1 with {gm1._xyz.shape[0]} points")
            

            """
            In case we want to use separate scene objects for each Gaussian model

            # Create separate scene objects for each Gaussian model
            scene0 = Scene(config=self.config, scene_path=sequence)
            scene1 = Scene(config=self.config, scene_path=sequence)
            
            # Initialize and load GM0
            gm0 = GaussianModel(sh_degree=3)
            scene0.load_gaussians_from_checkpoint(gm0_checkpoint, gm0, self.config.optimization)
            print(f"Loaded GM0 with {gm0._xyz.shape[0]} points")
            
            # Initialize and load GM1
            gm1 = GaussianModel(sh_degree=3)
            scene1.load_gaussians_from_checkpoint(gm1_checkpoint, gm1, self.config.optimization)
            print(f"Loaded GM1 with {gm1._xyz.shape[0]} points")
            """



            self.scenes.append(scene)
            self.GM0.append(gm0)
            self.GM1.append(gm1)

            # Initialize frames tensor with the first frame of each video
            frames_tensor = self.load_gt_frames(seq_idx, 0)
            self.frames_tensor.append(frames_tensor)

            # Initialize pseudo_3d_gt with the 3d positions and quaternions of the the GM0 and GM1
            pseudo_3d_gt = self.load_pseudo_3d_gt(seq_idx)
            self.pseudo_3d_gt.append(pseudo_3d_gt)


    def __getitem__(self, idx):
        item = {
            "scene": self.scenes[idx],
            "GM0": self.GM0[idx],
            "GM1": self.GM1[idx],
            "frames_tensor": self.frames_tensor[idx],
            "pseudo_3d_gt": self.pseudo_3d_gt[idx]
        }
        return item
    
    def __len__(self):
        return len(self.scenes)
    

    def load_pseudo_3d_gt(self, idx):
        """
        Initialize pseudo_3d_gt with the 3d positions and quaternions of the GM0 and GM1
        Returns:
            pseudo_3d_gt: Tensor of shape [N, 7] where:
                - [:, 0:3] are the 3D positions
                - [:, 3:7] are the quaternions (w,x,y,z)
        """
        # Get positions and rotations from GM0
        xyz0 = self.GM0[idx]._xyz.detach()  # [N, 3]
        rot0 = self.GM0[idx]._rotation.detach()  # [N, 4] quaternion

        # Get positions and rotations from GM1
        xyz1 = self.GM1[idx]._xyz.detach()  # [N, 3]
        rot1 = self.GM1[idx]._rotation.detach()  # [N, 4] quaternion

        # Concatenate positions and rotations for each timestep
        gt_t0 = torch.cat([xyz0, rot0], dim=-1)  # [N, 7]
        gt_t1 = torch.cat([xyz1, rot1], dim=-1)  # [N, 7]

        # Stack both timesteps
        pseudo_3d_gt = torch.stack([gt_t0, gt_t1], dim=0)  # [2, N, 7]

        return pseudo_3d_gt


    def load_gt_frames(self, seq_idx, timestep):
        """Load ground truth frames for a particular timestep from videos."""
        frames = []
        dynamic_dir = os.path.join(self.scene_dirs[seq_idx], "dynamic")
        if os.path.exists(dynamic_dir):
            for video_file in sorted(os.listdir(dynamic_dir)):
                if video_file.endswith('.mp4'):
                    cam_id = os.path.splitext(video_file)[0]
                    video_path = os.path.join(dynamic_dir, video_file)
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, timestep)
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError(f"Could not read frame {timestep} from {video_path}")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self.to_tensor(frame)
                    frames.append(frame)
                    cap.release()
        return torch.stack(frames)


    def extend_pseudo_gt(self, idx):
        """
        Extend pseudo_3d_gt with the latest 3d positions and quaternions of the scene
        """
        # TODO: Implement this
        pass

    def update_frames_tensor(self, idx, timestep):
        """
        Update frames_tensor with the respective timestep frame of each video
        """
        # TODO: Implement this
        pass















