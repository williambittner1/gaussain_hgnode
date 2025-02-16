import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PreRenderedGTDataset(Dataset):
    def __init__(self, gt_base_dir, total_timesteps, num_cams, transform=None):
        """
        Args:
            gt_base_dir (str): Path to the directory where GT images are saved.
              Expected structure: 
                gt_base_dir/<cam_id>/render_<cam_id>_timestep_<t>.jpg
            total_timesteps (int): Total number of timesteps.
            num_cams (int): Total number of cameras.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.gt_base_dir = gt_base_dir
        self.total_timesteps = total_timesteps
        self.num_cams = num_cams
        self.transform = transform if transform is not None else transforms.ToTensor()
        # Build a list of (cam_id, timestep, filepath)
        self.samples = []
        for cam in range(num_cams):
            cam_folder = os.path.join(gt_base_dir, f"{cam:03d}")
            for t in range(total_timesteps):
                filename = f"render_{cam:03d}_timestep_{t:05d}.jpg"
                filepath = os.path.join(cam_folder, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"File not found: {filepath}")
                self.samples.append((cam, t, filepath))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cam, t, filepath = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        img = self.transform(img)  # tensor in [0,1]
        return img, cam, t




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

    def __len__(self):
        return self.gt_xyz_cp.shape[0]

    def __getitem__(self, idx):
        return {
            "gt_xyz_cp": self.gt_xyz_cp[idx],  # [T, num_controlpoints, 3]
            "gt_rot_cp": self.gt_rot_cp[idx]     # [T, num_controlpoints, 4]
        }