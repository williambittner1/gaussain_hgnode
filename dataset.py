import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List

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
