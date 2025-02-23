# global imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# local imports
from pointnet.semantic_segmentation.pointnet2_utils import farthest_point_sample
from pointnet.semantic_segmentation.pointnet2_sem_seg import get_model


def get_cluster_colors(num_clusters):
    """
    Returns an array of distinct colors based on the number of clusters.
    The colors are chosen from the 'tab20' colormap and scaled to RGB values in [0, 255].
    """
    cmap = plt.get_cmap('tab20', num_clusters)  # Use a colormap with 20 distinct colors
    colors = [cmap(i)[:3] for i in range(num_clusters)]  # Extract RGB values
    colors = np.array(colors) * 255  # Scale to RGB [0, 255]
    return colors.astype(np.uint8)

def compute_normals(points_np, k=30):
    """
    Compute normals for a point cloud (numpy array of shape (N,3)) using PCA on local neighborhoods.
    Returns an array of shape (N,3) with normals.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points_np)
    distances, indices = nbrs.kneighbors(points_np)
    normals = []
    for i in range(points_np.shape[0]):
        # Exclude the point itself (first neighbor)
        neighbors = points_np[indices[i][1:]]
        cov = np.cov(neighbors - points_np[i], rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # eigenvector corresponding to the smallest eigenvalue
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        normals.append(normal)
    return np.array(normals)

class PointNetSegmenter(nn.Module):
    """
    A separate module for semantic segmentation using PointNet++.
    Given a set of gaussian points (with colors), it subsamples a fixed number of points,
    constructs a 9-dimensional feature (xyz, rgb, normals), and runs segmentation.
    """
    def __init__(self, num_classes=13, num_sample_points=1024, knn_k=10, device="cuda"):
        super(PointNetSegmenter, self).__init__()
        self.num_classes = num_classes
        self.num_sample_points = num_sample_points
        self.knn_k = knn_k
        self.device = device
        # Load the segmentation model (pretrained weights should be loaded externally or here)
        self.segmentation_model = get_model(num_classes=num_classes).to(device)
    
    def forward(self, gaussian_positions, gaussian_colors):
        """
        Args:
            gaussian_positions: Tensor of shape (N, 3) containing gaussian xyz coordinates.
            gaussian_colors: Tensor of shape (N, 3) containing rgb colors in [0,1].
        Returns:
            predicted_labels_full: Tensor of shape (N,) with semantic labels for each gaussian.
        """
        N_total = gaussian_positions.shape[0]
        if N_total < self.num_sample_points:
            raise ValueError("Not enough gaussians to sample from.")
        # Subsample using farthest point sampling.
        # Input for farthest_point_sample must be (B, N, 3); here B=1.
        gaussian_positions_unsq = gaussian_positions.unsqueeze(0)  # (1, N, 3)
        sample_indices = farthest_point_sample(gaussian_positions_unsq, self.num_sample_points).squeeze(0)  # (num_sample_points,)
        downsampled_xyz = gaussian_positions[sample_indices]  # (num_sample_points, 3)
        downsampled_rgb = gaussian_colors[sample_indices]       # (num_sample_points, 3)
        
        # Compute normals for the downsampled points.
        downsampled_xyz_np = downsampled_xyz.cpu().detach().numpy()
        normals_np = compute_normals(downsampled_xyz_np, k=self.knn_k)
        downsampled_normals = torch.tensor(normals_np, device=self.device, dtype=downsampled_xyz.dtype)
        
        # Concatenate features: xyz, rgb, normals → (num_sample_points, 9)
        point_features = torch.cat([downsampled_xyz, downsampled_rgb, downsampled_normals], dim=1)
        point_features = point_features.transpose(0, 1).unsqueeze(0)  # (1, 9, num_sample_points)
            
        # Run segmentation
        pred, _ = self.segmentation_model(point_features)  # (1, num_sample_points, num_classes)
        pred = pred.squeeze(0)  # (num_sample_points, num_classes)
        
        # Apply softmax to get probabilities
        semantic_features_down = torch.softmax(pred, dim=1)  # (num_sample_points, num_classes)
        semantic_labels_down = torch.argmax(pred, dim=1)     # (num_sample_points,)

        
        # For each full gaussian, assign features from nearest downsampled point
        dists = torch.cdist(gaussian_positions, downsampled_xyz)  # (N_total, num_sample_points)
        nearest_idx = torch.argmin(dists, dim=1)  # (N_total,)
        semantic_features = semantic_features_down[nearest_idx]  # (N_total, num_classes)
        semantic_labels = semantic_labels_down[nearest_idx]      # (N_total,)

        
        return semantic_features, semantic_labels