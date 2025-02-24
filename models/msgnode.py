import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch_geometric.nn import knn, radius
from torch_geometric.nn import radius_graph

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions using the Hamilton product.
    Args:
        q1: (B, N, 4) - First quaternion
        q2: (B, N, 4) - Second quaternion
    Returns:
        q_out: (B, N, 4) - Product quaternion
    """
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-8)
    # Extract components while preserving batch dimensions
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Compute Hamilton product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

def rotate_vector(v, q, inverse=False):
    """
    Rotate vector v by quaternion q.
    Args:
        v: (N, 3) - Vectors to rotate
        q: (N, 4) - Quaternions (normalized)
        inverse: bool - If True, apply inverse rotation (local to global)
    Returns:
        v_rot: (N, 3) - Rotated vectors
    """
    q = q / torch.norm(q, dim=-1, keepdim=True)
    if inverse:
        q = torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
    v_quat = torch.cat([torch.zeros_like(v[:, :1]), v], dim=-1)
    q_inv = torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
    v_rot_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_inv)
    return v_rot_quat[:, 1:]

# Feature Layout

feature_layout = {
    'pos': slice(0, 3),    # [B, N, 3]
    'vel': slice(3, 6),    # [B, N, 3]
    'quat': slice(6, 10),  # [B, N, 4]
    'omega': slice(10, 13) # [B, N, 3]
}
feature_dim = 13  # Total dimension: 3 + 3 + 4 + 3

# feature_layout = {
#     'pos':              slice(0, 3),        # 3D position
#     'vel':              slice(3, 6),       # Position velocity
#     'quat':             slice(6, 10),     # Quaternion
#     'omega':            slice(10, 13),   # Quaternion velocity
#     'rgb':              slice(13, 16),     # RGB colors
#     'semantic_feat':    slice(16, 29),  # Semantic features (13D for num_classes=13)
#     'semantic_label':   slice(29, 30), # Semantic label (1D)
#     'augm':             slice(30, 79),     # Placeholder augmentation (49D)
# }


"""
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
"""


# Utility Functions

def get_feature(z, key):
    """Extract feature from z using feature_layout."""
    return z[:, :, feature_layout[key]]

def set_feature(z, key, value):
    """Set feature in z using feature_layout."""
    z[:, :, feature_layout[key]] = value
    return z

# MLP Definition

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.out_features = out_dim
    
    def forward(self, x):
        return self.layers(x)
    

# Edge Constructors


def build_knn_edges1(pos_source, pos_target, k=3):
    """
    Build k-NN edges between source and target points.
    Args:
        pos_source: (batch_size, num_source, 3) source point positions
        pos_target: (batch_size, num_target, 3) target point positions
        k: Number of nearest neighbors
    Returns:
        edge_index: (2, num_edges) edge indices
    """
    # Handle batched input
    if pos_source.dim() == 3:
        batch_size, num_source, dim = pos_source.shape
        _, num_target, _ = pos_target.shape
        k = min(k, num_target)
        
        pos_source_flat = pos_source.reshape(-1, dim)
        pos_target_flat = pos_target.reshape(-1, dim)
        
        batch_source = torch.arange(batch_size, device=pos_source.device).repeat_interleave(num_source).long()
        batch_target = torch.arange(batch_size, device=pos_target.device).repeat_interleave(num_target).long()
        

        edge_index = knn(
            x=pos_source_flat,  # Target points (pos_l)
            y=pos_target_flat,  # Source points (pos_h)
            k=k,
            batch_x=batch_source,
            batch_y=batch_target
        )
 
        assert edge_index[0].max() < batch_size * num_target, "Target indices out of bounds"
        assert edge_index[1].max() < batch_size * num_source, "Source indices out of bounds"
        
        return edge_index
    else:
        k = min(k, pos_target.shape[0])
        return knn(pos_target, pos_source, k)

def build_hyper_radius_edges1(pos, vel, radius):
    """
    Build edges between points within a radius in hyperspace (pos + vel).
    Args:
        pos: (B, N, 3) node positions
        vel: (B, N, 3) node velocities
        radius: Float radius value in hyperspace
    Returns:
        edge_index: (2, E) edge indices
    """
    batch_size, num_points, _ = pos.shape
    hyperspace = torch.cat([pos, vel], dim=-1)  # [B, N, 6]
    hyperspace_flat = hyperspace.reshape(-1, 6)  # [B * N, 6]
    batch = torch.arange(batch_size, device=pos.device).repeat_interleave(num_points).long()
    
    edge_index = radius_graph(hyperspace_flat, r=radius, batch=batch)
    return edge_index

def build_world_edges1(pos, edge_index_normal, radius_pos):
    """
    Build world edges for surface interactions in normal space (position only), excluding normal edges.
    Args:
        pos: [B, N, 3] - Node positions
        edge_index_normal: [2, E_normal] - Existing normal edges
        radius_pos: float - Radius in position space
    Returns:
        edge_index_world: [2, E_world] - World edges
    """
    batch_size, num_points, _ = pos.shape
    pos_flat = pos.reshape(-1, 3)  # [B * N, 3]
    batch = torch.arange(batch_size, device=pos.device).repeat_interleave(num_points).long()

    # Compute radius-based graph in position space
    edge_index_pos = radius_graph(pos_flat, r=radius_pos, batch=batch)

    # Exclude existing normal edges
    normal_set = set(zip(edge_index_normal[0].tolist(), edge_index_normal[1].tolist()))
    edge_index_world = [e for e in zip(edge_index_pos[0].tolist(), edge_index_pos[1].tolist()) 
                        if e not in normal_set]
    edge_index_world = torch.tensor(edge_index_world, dtype=torch.long, device=pos.device).t()

    return edge_index_world

def build_hyper_radius_edges(pos, vel, radius):
    """
    Build edges between points within a radius in hyperspace (pos + vel).
    Args:
        pos: (B, N, 3) node positions
        vel: (B, N, 3) node velocities
        radius: Float radius value in hyperspace
    Returns:
        edge_indices: List of [2, E_b] tensors, one for each batch
    """
    batch_size, num_points, _ = pos.shape
    edge_indices = []
    for b in range(batch_size):
        hyperspace_b = torch.cat([pos[b], vel[b]], dim=-1)  # [N, 6]
        edge_index_b = radius_graph(hyperspace_b, r=radius)  # [2, E_b]
        edge_indices.append(edge_index_b.to(pos.device))
    return edge_indices

def build_knn_edges(pos_source, pos_target, k=3):
    """
    Build k-NN edges between source and target points.
    Args:
        pos_source: (B, N_source, 3) source point positions
        pos_target: (B, N_target, 3) target point positions
        k: Number of nearest neighbors
    Returns:
        edge_indices: List of [2, E_b] tensors, one for each batch
    """
    batch_size, num_source, _ = pos_source.shape
    _, num_target, _ = pos_target.shape
    edge_indices = []
    for b in range(batch_size):
        pos_source_b = pos_source[b]  # [N_source, 3]
        pos_target_b = pos_target[b]  # [N_target, 3]
        k_b = min(k, num_target)  # Ensure k doesn’t exceed available targets
        edge_index_b = knn(pos_target_b, pos_source_b, k_b)  # [2, N_source * k_b]
        edge_indices.append(edge_index_b.to(pos_source.device))
    return edge_indices

def build_world_edges(pos, edge_index_normal, radius_pos):
    """
    Build world edges for surface interactions in position space, excluding normal edges.
    Args:
        pos: [B, N, 3] - Node positions
        edge_index_normal: List of [2, E_normal_b] - Existing normal edges per batch
        radius_pos: float - Radius in position space
    Returns:
        edge_indices_world: List of [2, E_world_b] - World edges per batch
    """
    batch_size, num_points, _ = pos.shape
    edge_indices_world = []
    for b in range(batch_size):
        pos_b = pos[b]  # [N, 3]
        edge_index_normal_b = edge_index_normal[b]  # [2, E_normal_b]
        edge_index_pos_b = radius_graph(pos_b, r=radius_pos)  # [2, E_pos_b]
        normal_set_b = set(zip(edge_index_normal_b[0].tolist(), edge_index_normal_b[1].tolist()))
        edge_list = [e for e in zip(edge_index_pos_b[0].tolist(), edge_index_pos_b[1].tolist()) 
                     if e not in normal_set_b]
        edge_index_world_b = torch.tensor(edge_list, dtype=torch.long, device=pos.device).t()
        edge_indices_world.append(edge_index_world_b)
    return edge_indices_world


# GNN Layers

class GNN_h_noworldedges(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8  # diff_ij(3) + dist_ij(1) + vel_crossprod_ij(3) + abs_vel_crossprod_ij(1)
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, feature_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim * 2, hidden_dim, feature_dim)
        self.mlp_world = MLP(feature_dim + 3, hidden_dim, feature_dim)

    def compute_edge_features(self, z, edge_index):
        """
        Compute edge features from node features.
        Args:
            z: (B, N, feature_dim) node features
            edge_index: (2, E) edge indices
        Returns:
            edge_features: (E, edge_feature_dim)
        """
        B, N, _ = z.shape
        source, target = edge_index
        
        # Convert global indices to batch and local indices
        batch_idx = source // N  # Determine which batch each edge belongs to
        local_source = source % N
        local_target = target % N
        
        # Index into the correct batch and nodes
        batch_idx_expanded = batch_idx.unsqueeze(-1)  # [E, 1]
        
        # Get features for source and target nodes from the correct batch
        pos_source = z[batch_idx, local_source, feature_layout['pos']]  # [E, 3]
        pos_target = z[batch_idx, local_target, feature_layout['pos']]  # [E, 3]
        vel_source = z[batch_idx, local_source, feature_layout['vel']]  # [E, 3]
        vel_target = z[batch_idx, local_target, feature_layout['vel']]  # [E, 3]
        
        # Compute edge features
        diff_ij = pos_source - pos_target  # [E, 3]
        dist_ij = (diff_ij ** 2).sum(-1, keepdim=True)  # [E, 1]
        vel_crossprod_ij = torch.cross(vel_source, vel_target, dim=-1)  # [E, 3]
        abs_vel_crossprod_ij = torch.norm(vel_crossprod_ij, dim=-1, keepdim=True)  # [E, 1]
        
        edge_features = torch.cat([
            diff_ij,                # [E, 3]
            dist_ij,               # [E, 1]
            vel_crossprod_ij,      # [E, 3]
            abs_vel_crossprod_ij   # [E, 1]
        ], dim=-1)                 # [E, 8]
        
        return edge_features

    def forward(self, z_h, edge_index_h_h, pos_world=None):
        """GNN forward pass with proper batch handling"""
        B, N, F = z_h.shape
        source, target = edge_index_h_h
        
        # Convert global indices to batch and local indices
        batch_idx = source // N
        local_source = source % N
        local_target = target % N
        
        # Get features for source and target nodes from correct batches
        z_source = z_h[batch_idx, local_source]  # [E, F]
        z_target = z_h[batch_idx, local_target]  # [E, F]
        
        # Compute edge features
        e_h_h = self.compute_edge_features(z_h, edge_index_h_h)  # [E, edge_feat_dim]
        
        # Message passing
        input_edge = torch.cat([z_source, z_target, e_h_h], dim=-1)  # [E, 2*F + edge_feat_dim]
        m_h_h = self.mlp_edge(input_edge)  # [E, F]
        w_h_h = torch.sigmoid(self.mlp_weight(input_edge))  # [E, 1]
        
        # Message aggregation (maintaining batch dimension)
        m_agg_h = torch.zeros_like(z_h)  # [B, N, F]
        for b in range(B):
            batch_mask = batch_idx == b
            batch_target = local_target[batch_mask]
            batch_messages = w_h_h[batch_mask] * m_h_h[batch_mask]
            m_agg_h[b].index_add_(0, batch_target, batch_messages)
        
        # Node update
        input_node = torch.cat([z_h, m_agg_h], dim=-1)  # [B, N, 2*F]
        delta_z_h = self.mlp_node(input_node)  # [B, N, F]
        
        return delta_z_h
    

class GNN_h1(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        # Edge feature dimension: diff_ij(3) + dist_ij(1) + vel_crossprod_ij(3) + abs_vel_crossprod_ij(1)
        edge_dim = 8

        # MLPs for normal edges
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)

        # MLPs for world edges
        self.mlp_edge_world = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight_world = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)

        # Node update MLP: takes [z_h, m_agg_h, m_agg_w]
        self.mlp_node = MLP(feature_dim + message_dim * 2, hidden_dim, feature_dim)

    def compute_edge_features(self, z, edge_index):
        """
        Compute edge features from node features.
        Args:
            z: (B, N, feature_dim) node features
            edge_index: (2, E) edge indices
        Returns:
            edge_features: (E, edge_feature_dim)
        """
        B, N, _ = z.shape
        source, target = edge_index

        # Convert global indices to batch and local indices
        batch_idx = source // N  # Determine which batch each edge belongs to
        local_source = source % N
        local_target = target % N

        # Index into the correct batch and nodes
        pos_source = z[batch_idx, local_source, feature_layout['pos']]  # [E, 3]
        pos_target = z[batch_idx, local_target, feature_layout['pos']]  # [E, 3]
        vel_source = z[batch_idx, local_source, feature_layout['vel']]  # [E, 3]
        vel_target = z[batch_idx, local_target, feature_layout['vel']]  # [E, 3]

        # Compute edge features
        diff_ij = pos_source - pos_target  # [E, 3]
        dist_ij = (diff_ij ** 2).sum(-1, keepdim=True)  # [E, 1]
        vel_crossprod_ij = torch.cross(vel_source, vel_target, dim=-1)  # [E, 3]
        abs_vel_crossprod_ij = torch.norm(vel_crossprod_ij, dim=-1, keepdim=True)  # [E, 1]

        edge_features = torch.cat([
            diff_ij,               # [E, 3]
            dist_ij,              # [E, 1]
            vel_crossprod_ij,     # [E, 3]
            abs_vel_crossprod_ij  # [E, 1]
        ], dim=-1)                # [E, 8]

        return edge_features

    def forward(self, z_h, edge_index_h_h, edge_index_world):
        """
        GNN forward pass with normal and world edges.
        Args:
            z_h: (B, N, feature_dim) node features
            edge_index_h_h: (2, E_h) normal edge indices
            edge_index_world: (2, E_w) world edge indices
        Returns:
            delta_z_h: (B, N, feature_dim) node feature updates
        """
        B, N, F = z_h.shape

        # --- Normal Edges ---
        source, target = edge_index_h_h
        batch_idx = source // N
        local_source = source % N
        local_target = target % N

        z_source = z_h[batch_idx, local_source]  # [E_h, F]
        z_target = z_h[batch_idx, local_target]  # [E_h, F]

        e_h_h = self.compute_edge_features(z_h, edge_index_h_h)  # [E_h, edge_dim]

        # Compute messages and weights for normal edges
        input_edge = torch.cat([z_source, z_target, e_h_h], dim=-1)  # [E_h, 2*F + edge_dim]
        m_h_h = self.mlp_edge(input_edge)  # [E_h, message_dim]
        w_h_h = torch.sigmoid(self.mlp_weight(input_edge))  # [E_h, 1]

        # Aggregate normal messages
        m_agg_h = torch.zeros(B, N, m_h_h.shape[-1], device=z_h.device)  # [B, N, message_dim]
        for b in range(B):
            batch_mask = batch_idx == b
            batch_target = local_target[batch_mask]
            batch_messages = w_h_h[batch_mask] * m_h_h[batch_mask]
            m_agg_h[b].index_add_(0, batch_target, batch_messages)

        # --- World Edges ---
        source_w, target_w = edge_index_world
        batch_idx_w = source_w // N
        local_source_w = source_w % N
        local_target_w = target_w % N

        z_source_w = z_h[batch_idx_w, local_source_w]  # [E_w, F]
        z_target_w = z_h[batch_idx_w, local_target_w]  # [E_w, F]

        e_h_w = self.compute_edge_features(z_h, edge_index_world)  # [E_w, edge_dim]

        # Compute messages and weights for world edges
        input_edge_w = torch.cat([z_source_w, z_target_w, e_h_w], dim=-1)  # [E_w, 2*F + edge_dim]
        m_h_w = self.mlp_edge_world(input_edge_w)  # [E_w, message_dim]
        w_h_w = torch.sigmoid(self.mlp_weight_world(input_edge_w))  # [E_w, 1]

        # Aggregate world messages
        m_agg_w = torch.zeros(B, N, m_h_w.shape[-1], device=z_h.device)  # [B, N, message_dim]
        for b in range(B):
            batch_mask_w = batch_idx_w == b
            batch_target_w = local_target_w[batch_mask_w]
            batch_messages_w = w_h_w[batch_mask_w] * m_h_w[batch_mask_w]
            m_agg_w[b].index_add_(0, batch_target_w, batch_messages_w)

        # --- Node Update ---
        input_node = torch.cat([z_h, m_agg_h, m_agg_w], dim=-1)  # [B, N, F + message_dim + message_dim]
        delta_z_h = self.mlp_node(input_node)  # [B, N, F]

        return delta_z_h
    
class GNN_h(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        # Edge feature dimension: diff_ij(3) + dist_ij(1) + vel_crossprod_ij(3) + abs_vel_crossprod_ij(1)
        edge_dim = 8

        # MLPs for normal edges
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)

        # MLPs for world edges
        self.mlp_edge_world = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight_world = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)

        # Node update MLP: takes [z_h, m_agg_h, m_agg_w]
        self.mlp_node = MLP(feature_dim + message_dim * 2, hidden_dim, feature_dim)

    def compute_edge_features(self, z_b, edge_index_b):
        """
        Compute edge features from node features for a single batch.
        Args:
            z_b: (N, feature_dim) node features for batch b
            edge_index_b: (2, E_b) edge indices for batch b, local to the batch
        Returns:
            edge_features: (E_b, edge_feature_dim)
        """
        source, target = edge_index_b  # [E_b], [E_b]
        pos_source = z_b[source, feature_layout['pos']]  # [E_b, 3]
        pos_target = z_b[target, feature_layout['pos']]  # [E_b, 3]
        vel_source = z_b[source, feature_layout['vel']]  # [E_b, 3]
        vel_target = z_b[target, feature_layout['vel']]  # [E_b, 3]

        # Compute edge features
        diff_ij = pos_source - pos_target  # [E_b, 3]
        dist_ij = (diff_ij ** 2).sum(-1, keepdim=True)  # [E_b, 1]
        vel_crossprod_ij = torch.cross(vel_source, vel_target, dim=-1)  # [E_b, 3]
        abs_vel_crossprod_ij = torch.norm(vel_crossprod_ij, dim=-1, keepdim=True)  # [E_b, 1]

        return torch.cat([diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij], dim=-1)  # [E_b, 8]

    def forward(self, z_h, edge_index_h_h, edge_index_world):
        """
        GNN forward pass with normal and world edges.
        Args:
            z_h: (B, N, feature_dim) node features
            edge_index_h_h: list of (2, E_b) normal edge indices, one per batch
            edge_index_world: list of (2, E_b) world edge indices, one per batch
        Returns:
            delta_z_h: (B, N, feature_dim) node feature updates
        """
        B, N, F = z_h.shape
        assert len(edge_index_h_h) == B and len(edge_index_world) == B, "Edge index lists must match batch size"

        message_dim = self.mlp_edge.out_features
        m_agg_h = torch.zeros(B, N, message_dim, device=z_h.device)
        m_agg_w = torch.zeros(B, N, message_dim, device=z_h.device)

        for b in range(B):
            z_b = z_h[b]  # [N, F]

            # --- Normal Edges ---
            source, target = edge_index_h_h[b]  # [E_b], [E_b], local indices
            e_h_h = self.compute_edge_features(z_b, edge_index_h_h[b])  # [E_b, edge_dim]
            z_source = z_b[source]  # [E_b, F]
            z_target = z_b[target]  # [E_b, F]
            input_edge = torch.cat([z_source, z_target, e_h_h], dim=-1)  # [E_b, 2*F + edge_dim]
            m_h_h = self.mlp_edge(input_edge)  # [E_b, message_dim]
            w_h_h = torch.sigmoid(self.mlp_weight(input_edge))  # [E_b, 1]
            m_agg_h[b].index_add_(0, target, w_h_h * m_h_h)  # Aggregate to targets in batch b

            # --- World Edges ---
            if len(edge_index_world[b]) > 0:  # Check if there are world edges for this batch
                source_w, target_w = edge_index_world[b]  # [E_b], [E_b], local indices
                e_h_w = self.compute_edge_features(z_b, edge_index_world[b])  # [E_b, edge_dim]
                z_source_w = z_b[source_w]  # [E_b, F]
                z_target_w = z_b[target_w]  # [E_b, F]
                input_edge_w = torch.cat([z_source_w, z_target_w, e_h_w], dim=-1)  # [E_b, 2*F + edge_dim]
                m_h_w = self.mlp_edge_world(input_edge_w)  # [E_b, message_dim]
                w_h_w = torch.sigmoid(self.mlp_weight_world(input_edge_w))  # [E_b, 1]
                m_agg_w[b].index_add_(0, target_w, w_h_w * m_h_w)  # Aggregate to targets in batch b
            else:
                # If no world edges, m_agg_w[b] remains zero, meaning no update from world edges
                pass

        # --- Node Update ---
        input_node = torch.cat([z_h, m_agg_h, m_agg_w], dim=-1)  # [B, N, F + message_dim * 2]
        delta_z_h = self.mlp_node(input_node)  # [B, N, F]
        return delta_z_h
    
class GNN_l(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8  # diff_ij(3) + dist_ij(1) + vel_crossprod_ij(3) + abs_vel_crossprod_ij(1)
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_edge_world = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight_world = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim + message_dim * 2, hidden_dim, feature_dim)

    def compute_edge_features(self, z_b, edge_index_b):
        """
        Compute edge features from node features for a single batch.
        Args:
            z_b: (N, feature_dim) node features for batch b
            edge_index_b: (2, E_b) edge indices for batch b, local to the batch
        Returns:
            edge_features: (E_b, edge_feature_dim)
        """
        source, target = edge_index_b  # [E_b], [E_b]
        pos_source = z_b[source, feature_layout['pos']]  # [E_b, 3]
        pos_target = z_b[target, feature_layout['pos']]  # [E_b, 3]
        vel_source = z_b[source, feature_layout['vel']]  # [E_b, 3]
        vel_target = z_b[target, feature_layout['vel']]  # [E_b, 3]

        diff_ij = pos_source - pos_target  # [E_b, 3]
        dist_ij = (diff_ij ** 2).sum(-1, keepdim=True)  # [E_b, 1]
        vel_crossprod_ij = torch.cross(vel_source, vel_target, dim=-1)  # [E_b, 3]
        abs_vel_crossprod_ij = torch.norm(vel_crossprod_ij, dim=-1, keepdim=True)  # [E_b, 1]

        return torch.cat([diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij], dim=-1)  # [E_b, 8]

    def forward(self, z_l, edge_index_l_l, edge_index_world):
        B, N, F = z_l.shape
        assert len(edge_index_l_l) == B and len(edge_index_world) == B, "Edge index lists must match batch size"

        message_dim = self.mlp_edge.out_features
        m_agg_l = torch.zeros(B, N, message_dim, device=z_l.device)
        m_agg_w = torch.zeros(B, N, message_dim, device=z_l.device)

        for b in range(B):
            z_b = z_l[b]  # [N, F]

            # --- Normal Edges ---
            source, target = edge_index_l_l[b]  # [E_b], [E_b], local indices
            e_l_l = self.compute_edge_features(z_b, edge_index_l_l[b])  # [E_b, edge_dim]
            z_source = z_b[source]  # [E_b, F]
            z_target = z_b[target]  # [E_b, F]
            input_edge = torch.cat([z_source, z_target, e_l_l], dim=-1)  # [E_b, 2*F + edge_dim]
            m_l_l = self.mlp_edge(input_edge)  # [E_b, message_dim]
            w_l_l = torch.sigmoid(self.mlp_weight(input_edge))  # [E_b, 1]
            m_agg_l[b].index_add_(0, target, w_l_l * m_l_l)  # Aggregate to targets in batch b

            # --- World Edges ---
            if len(edge_index_world[b]) > 0:  # Check if there are world edges for this batch
                source_w, target_w = edge_index_world[b]  # [E_b], [E_b], local indices
                e_l_w = self.compute_edge_features(z_b, edge_index_world[b])  # [E_b, edge_dim]
                z_source_w = z_b[source_w]  # [E_b, F]
                z_target_w = z_b[target_w]  # [E_b, F]
                input_edge_w = torch.cat([z_source_w, z_target_w, e_l_w], dim=-1)  # [E_b, 2*F + edge_dim]
                m_l_w = self.mlp_edge_world(input_edge_w)  # [E_b, message_dim]
                w_l_w = torch.sigmoid(self.mlp_weight_world(input_edge_w))  # [E_b, 1]
                m_agg_w[b].index_add_(0, target_w, w_l_w * m_l_w)  # Aggregate to targets in batch b
            else:
                # If no world edges, m_agg_w[b] remains zero
                pass

        # --- Node Update ---
        input_node = torch.cat([z_l, m_agg_l, m_agg_w], dim=-1)  # [B, N, F + message_dim * 2]
        delta_z_l = self.mlp_node(input_node)  # [B, N, F]
        return delta_z_l
    

class GNN_h_l(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8  # diff_hl(3) + dist_hl(1) + vel_crossprod_hl(3) + abs_vel_crossprod_hl(1)
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim + message_dim, hidden_dim, feature_dim)

    def compute_edge_features(self, z_h_b, z_l_b, edge_index_b):
        source, target = edge_index_b  # [E_b], [E_b]
        pos_source = z_h_b[source, feature_layout['pos']]  # [E_b, 3]
        pos_target = z_l_b[target, feature_layout['pos']]  # [E_b, 3]
        vel_source = z_h_b[source, feature_layout['vel']]  # [E_b, 3]
        vel_target = z_l_b[target, feature_layout['vel']]  # [E_b, 3]
        diff_hl = pos_source - pos_target
        dist_hl = (diff_hl ** 2).sum(-1, keepdim=True)
        vel_crossprod_hl = torch.cross(vel_source, vel_target, dim=-1)
        abs_vel_crossprod_hl = torch.norm(vel_crossprod_hl, dim=-1, keepdim=True)
        return torch.cat([diff_hl, dist_hl, vel_crossprod_hl, abs_vel_crossprod_hl], dim=-1)

    def forward(self, z_h, z_l, edge_index_h_l):
        B, N_h, F = z_h.shape
        _, N_l, _ = z_l.shape
        assert len(edge_index_h_l) == B, "Edge index list must match batch size"

        m_agg_l = torch.zeros(B, N_l, self.mlp_edge.out_features, device=z_h.device)
        for b in range(B):
            edge_index_b = edge_index_h_l[b]
            source, target = edge_index_b
            z_h_b = z_h[b]  # [N_h, F]
            z_l_b = z_l[b]  # [N_l, F]
            e_h_l = self.compute_edge_features(z_h_b, z_l_b, edge_index_b)  # [E_b, edge_dim]
            z_source = z_h_b[source]  # [E_b, F]
            z_target = z_l_b[target]  # [E_b, F]
            input_edge = torch.cat([z_source, z_target, e_h_l], dim=-1)
            m_h_l = self.mlp_edge(input_edge)  # [E_b, message_dim]
            w_h_l = torch.sigmoid(self.mlp_weight(input_edge))  # [E_b, 1]
            m_agg_l[b].index_add_(0, target, w_h_l * m_h_l)

        input_node = torch.cat([z_l, m_agg_l], dim=-1)  # [B, N_l, F + message_dim]
        delta_z_l = self.mlp_node(input_node)  # [B, N_l, F]
        return delta_z_l
    

class GNN_l_h(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8  # diff_lh(3) + dist_lh(1) + vel_crossprod_lh(3) + abs_vel_crossprod_lh(1)
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim + message_dim, hidden_dim, feature_dim)

    def compute_edge_features(self, z_l_b, z_h_b, edge_index_b):
        """
        Compute edge features from low to high resolution for a single batch.
        Args:
            z_l_b: (N_l, feature_dim) low-res node features for batch b
            z_h_b: (N_h, feature_dim) high-res node features for batch b
            edge_index_b: (2, E_b) edge indices, source in z_l_b, target in z_h_b
        Returns:
            edge_features: (E_b, edge_feature_dim)
        """
        source, target = edge_index_b  # [E_b], [E_b], source in [0, N_l), target in [0, N_h)
        pos_source = z_l_b[source, feature_layout['pos']]  # [E_b, 3]
        pos_target = z_h_b[target, feature_layout['pos']]  # [E_b, 3]
        vel_source = z_l_b[source, feature_layout['vel']]  # [E_b, 3]
        vel_target = z_h_b[target, feature_layout['vel']]  # [E_b, 3]

        diff_lh = pos_source - pos_target  # [E_b, 3]
        dist_lh = (diff_lh ** 2).sum(-1, keepdim=True)  # [E_b, 1]
        vel_crossprod_lh = torch.cross(vel_source, vel_target, dim=-1)  # [E_b, 3]
        abs_vel_crossprod_lh = torch.norm(vel_crossprod_lh, dim=-1, keepdim=True)  # [E_b, 1]

        return torch.cat([diff_lh, dist_lh, vel_crossprod_lh, abs_vel_crossprod_lh], dim=-1)  # [E_b, 8]

    def forward(self, z_l, z_h, edge_index_l_h):
        """
        GNN forward pass for upsampling from low to high resolution.
        Args:
            z_l: (B, N_l, feature_dim) low-res node features
            z_h: (B, N_h, feature_dim) high-res node features
            edge_index_l_h: list of (2, E_b) edge indices, source in z_l, target in z_h
        Returns:
            delta_z_h: (B, N_h, feature_dim) high-res node feature updates
        """
        B, N_l, F = z_l.shape
        _, N_h, _ = z_h.shape
        assert len(edge_index_l_h) == B, "Edge index list must match batch size"

        message_dim = self.mlp_edge.out_features
        m_agg_h = torch.zeros(B, N_h, message_dim, device=z_h.device)

        for b in range(B):
            z_l_b = z_l[b]  # [N_l, F]
            z_h_b = z_h[b]  # [N_h, F]
            source, target = edge_index_l_h[b]  # [E_b], [E_b], source in [0, N_l), target in [0, N_h)
            e_l_h = self.compute_edge_features(z_l_b, z_h_b, edge_index_l_h[b])  # [E_b, edge_dim]
            z_source = z_l_b[source]  # [E_b, F]
            z_target = z_h_b[target]  # [E_b, F]
            input_edge = torch.cat([z_source, z_target, e_l_h], dim=-1)  # [E_b, 2*F + edge_dim]
            m_l_h = self.mlp_edge(input_edge)  # [E_b, message_dim]
            w_l_h = torch.sigmoid(self.mlp_weight(input_edge))  # [E_b, 1]
            m_agg_h[b].index_add_(0, target, w_l_h * m_l_h)  # Aggregate to targets in z_h_b

        # --- Node Update ---
        input_node = torch.cat([z_h, m_agg_h], dim=-1)  # [B, N_h, F + message_dim]
        delta_z_h = self.mlp_node(input_node)  # [B, N_h, F]
        return delta_z_h
    

class ODEFunc(nn.Module):
    def __init__(self, feature_dim=13, message_dim=64, hidden_dim=256, device='cuda'):
        super().__init__()
        self.gnn_h = GNN_h(feature_dim, message_dim, hidden_dim).to(device)
        self.gnn_l = GNN_l(feature_dim, message_dim, hidden_dim).to(device)
        self.gnn_h_l = GNN_h_l(feature_dim, message_dim, hidden_dim).to(device)
        self.gnn_l_h = GNN_l_h(feature_dim, message_dim, hidden_dim).to(device)
        self.mlp_vel = MLP(feature_dim, hidden_dim, 3).to(device)
        self.mlp_omega = MLP(feature_dim, hidden_dim, 3).to(device)
        self.device = device
        self.nfe = 0

        # Flags and storage for precomputed edges and radii
        self.edges_computed = False
        self.edge_index_h_h = None
        self.edge_index_l_l = None
        self.edge_index_h_l = None
        self.edge_index_l_h = None
        self.radius_h = None
        self.radius_l = None
        self.radius_hyper = None
        self.radius_pos = None

    def reset_edges(self):
        """Reset edge computation state for a new ODE integration."""
        self.edges_computed = False
        self.edge_index_h_h = None
        self.edge_index_l_l = None
        self.edge_index_h_l = None
        self.edge_index_l_h = None
        self.radius_h = None
        self.radius_l = None
        self.radius_hyper = None
        self.radius_pos = None

    def compute_radius(self, pos, k=10):
        """
        Compute an adaptive radius based on the average distance of k nearest neighbors.
        """
        if pos.dim() == 3:
            batch_size, num_points, dim = pos.shape
            pos_flat = pos.reshape(-1, dim)
            batch = torch.arange(batch_size, device=self.device).repeat_interleave(num_points).long()
            edge_index = knn(pos_flat, pos_flat, k + 1, batch, batch)
        else:
            pos_flat = pos
            edge_index = knn(pos, pos, k + 1)
        
        source, target = edge_index
        distances = torch.norm(pos_flat[source] - pos_flat[target], dim=1)
        total_points = pos_flat.shape[0]
        distances = distances.view(total_points, k + 1)
        distances_no_self = distances[:, 1:]
        return distances_no_self.mean().item()

    def forward(self, t, z):
        z_h, z_l = z
        pos_h = get_feature(z_h, 'pos')
        vel_h = get_feature(z_h, 'vel')
        pos_l = get_feature(z_l, 'pos')
        vel_l = get_feature(z_l, 'vel')

        # Lazy initialization of normal edges and radii on the first call
        if not self.edges_computed:
            # Compute radii based on initial positions
            self.radius_h = self.compute_radius(pos_h, k=2)
            self.radius_l = self.compute_radius(pos_l, k=3)
            self.radius_hyper = max(self.radius_h, self.radius_l) * 0.1
            self.radius_pos = min(self.radius_h, self.radius_l) * 0.5
            
            # Compute normal edges in hyperspace
            self.edge_index_h_h = build_hyper_radius_edges(pos_h, vel_h, self.radius_hyper)
            self.edge_index_l_l = build_hyper_radius_edges(pos_l, vel_l, self.radius_hyper)
            
            # Compute inter-level edges (k-NN, position-based)
            self.edge_index_h_l = build_knn_edges(pos_h, pos_l, k=3)
            self.edge_index_l_h = build_knn_edges(pos_l, pos_h, k=3)
            
            self.edges_computed = True

        # Compute world edges dynamically in position space
        edge_index_h_w = build_world_edges(pos_h, self.edge_index_h_h, self.radius_pos)
        edge_index_l_w = build_world_edges(pos_l, self.edge_index_l_l, self.radius_pos)

        # Compute GNN updates with 1H1D5L1U1H procedure
        # 1H: First high-resolution update
        delta_z_h_1 = self.gnn_h(z_h, self.edge_index_h_h, edge_index_h_w)

        # 1D: Downsample from high to low
        delta_z_l_from_h = self.gnn_h_l(z_h, z_l, self.edge_index_h_l)
        z_l_temp = z_l + delta_z_l_from_h

        # 5L: Five low-resolution updates
        for _ in range(5):
            delta_z_l = self.gnn_l(z_l_temp, self.edge_index_l_l, edge_index_l_w)
            z_l_temp = z_l_temp + delta_z_l

        # 1U: Upsample from low to high
        delta_z_h_from_l = self.gnn_l_h(z_l_temp, z_h, self.edge_index_l_h)

        # 1H: Final high-resolution update
        z_h_temp = z_h + delta_z_h_1 + delta_z_h_from_l
        delta_z_h_2 = self.gnn_h(z_h_temp, self.edge_index_h_h, edge_index_h_w)

        # Combine updates for dz_h_dt
        dz_h_dt = delta_z_h_1 + delta_z_h_2

        # Compute physical derivatives for high-level nodes
        dz_h_dt_full = torch.zeros_like(z_h)
        set_feature(dz_h_dt_full, 'pos', get_feature(z_h, 'vel'))
        set_feature(dz_h_dt_full, 'vel', self.mlp_vel(z_h))
        omega = get_feature(z_h, 'omega')
        quat = get_feature(z_h, 'quat')
        # Create zero tensor with correct batch and node dimensions
        zeros = torch.zeros_like(omega[:, :, :1])  # [B, N, 1]
        omega_quat = torch.cat([zeros, omega], dim=-1)  # [B, N, 4]
        
        quat_deriv = 0.5 * quaternion_multiply(omega_quat, quat)
        
        set_feature(dz_h_dt_full, 'quat', quat_deriv)
        set_feature(dz_h_dt_full, 'omega', self.mlp_omega(z_h))

        # Placeholder for low-level nodes
        dz_l_dt = z_l_temp - z_l

        return [dz_h_dt_full + dz_h_dt, dz_l_dt]
    

class MSGNODEProcessor(nn.Module):
    def __init__(self, feature_dim=13, message_dim=64, hidden_dim=256, device='cuda'):
        super().__init__()
        self.ode_func = ODEFunc(feature_dim, message_dim, hidden_dim, device)
        
    
    def forward(self, z0_h, z0_l, t):
        z0 = (z0_h, z0_l)
        z_traj = odeint(self.ode_func, z0, t, method='rk4')
        z_h_traj, z_l_traj = z_traj
        return z_h_traj.permute(1,0,2,3), z_l_traj.permute(1,0,2,3)




"""

MSGNODE Model Description

##########################################
# Semantic Encoder
##########################################
# Encode gaussian node level Latent Semantic Features (PointNet++ Semantic Segmentation)
# (PointNet++ Semantic Segmentation on the gaussians)


##########################################
# Multi-Scale Message Passing with Graph Neural ODE 
##########################################

# returns: z_traj (trajectory of the fine nodes and coarse nodes)

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

# v_h = [pos, dt_pos, quat, dt_quat, abc, wedge, dt_wedge, rgb, semantic, augm]

# - pos (3D) (dynamic)  
# - dt_pos (3D) (dynamic)
# - quat (4D) (dynamic)
# - dt_quat (3D) (dynamic)
# - abc (9D) (neighbor vectors) (either constant neighbors or dynamically updated neighbors via k-NN)
# - wedge (1D) (dynamic) (computed from the 3 nearest neighbors)
# - dt_wedge (1D) (dynamic) (dt_wedge = wedge(dt_a, b, c)+wedge(a, dt_b, c)+wedge(a, b, dt_c))
# - color (3D) (constant, potentially dynamic in the future)
# - semantic_features (e.g. 32D) (constant)  
# - augmented_working_space (49D) (constant) (initialized by copying and concatenating the initial node state or simply setting zeros)


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

# - v_l = [pos, dt_pos, quat, dt_quat, wedge, dt_wedge, rgb, semantic, augm]
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

# - v_h = [pos, dt_pos, quat, dt_quat, abc, wedge, dt_wedge, rgb, semantic, augm]
# - v_l = [pos, dt_pos, quat, dt_quat, wedge, dt_wedge, rgb, semantic, augm]
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

# - v_h = [pos, dt_pos, quat, dt_quat, abc, wedge, dt_wedge, rgb, semantic, augm]
# - v_l = [pos, dt_pos, quat, dt_quat, wedge, dt_wedge, rgb, semantic, augm]
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


"""
