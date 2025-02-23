import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch_geometric.nn import knn, radius
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions using the Hamilton product.
    Args:
        q1: (N, 4) - First quaternion
        q2: (N, 4) - Second quaternion
    Returns:
        q_out: (N, 4) - Product quaternion
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
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
    'pos':              slice(0, 3),        # 3D position
    'vel':              slice(3, 6),       # Position velocity
    'quat':             slice(6, 10),     # Quaternion
    'omega':            slice(10, 13),   # Quaternion velocity
    'rgb':              slice(13, 16),     # RGB colors
    'semantic_feat':    slice(16, 29),  # Semantic features (13D for num_classes=13)
    'semantic_label':   slice(29, 30), # Semantic label (1D)
    'augm':             slice(30, 79),     # Placeholder augmentation (49D)
}


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
    return z[:, feature_layout[key]]

def set_feature(z, key, value):
    """Set feature in z using feature_layout."""
    z[:, feature_layout[key]] = value
    return z

# MLP Definition

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    

# Edge Constructors

def build_knn_edges(pos_source, pos_target, k=3):
    """Build edges using k-nearest neighbors."""
    return knn(pos_source, pos_target, k)

def build_radius_edges(pos_source, pos_target, r):
    """Build edges within a specified radius."""
    return radius(pos_source, pos_target, r)


# GNN Layers

class GNN_h(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8  # diff_ij(3) + dist_ij(1) + vel_crossprod_ij(3) + abs_vel_crossprod_ij(1)
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim + message_dim, hidden_dim, feature_dim)
        self.mlp_world = MLP(feature_dim + 3, hidden_dim, message_dim)  # World edge MLP

    def compute_edge_features(self, z_h, edge_index):
        source, target = edge_index
        diff_ij = z_h[source, 0:3] - z_h[target, 0:3]
        dist_ij = (diff_ij ** 2).sum(-1, keepdim=True)
        vel_crossprod_ij = torch.cross(z_h[source, 3:6], z_h[target, 3:6], dim=-1)
        abs_vel_crossprod_ij = torch.norm(vel_crossprod_ij, dim=-1, keepdim=True)
        return torch.cat([diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij], dim=-1)

    def forward(self, z_h, edge_index_h_h, pos_world):
        # Local message passing (h-h edges)
        e_h_h = self.compute_edge_features(z_h, edge_index_h_h)
        source, target = edge_index_h_h
        input_edge = torch.cat([z_h[source], z_h[target], e_h_h], dim=-1)
        m_h_h = self.mlp_edge(input_edge)
        w_h_h = torch.sigmoid(self.mlp_weight(input_edge))
        m_agg_h = torch.zeros_like(z_h).scatter_add_(0, target.unsqueeze(-1).expand_as(m_h_h), w_h_h * m_h_h)

        # World message (h-world edges)
        diff_world = z_h[:, 0:3] - pos_world
        input_world = torch.cat([z_h, diff_world], dim=-1)
        m_h_w = self.mlp_world(input_world)
        m_agg_w = m_h_w.sum(dim=0, keepdim=True).expand_as(z_h)

        # Combine messages
        m_agg_total = m_agg_h + m_agg_w
        input_node = torch.cat([z_h, m_agg_total], dim=-1)
        delta_z_h = self.mlp_node(input_node)
        return delta_z_h
    

    
class GNN_l(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim + message_dim, hidden_dim, feature_dim)
        self.mlp_world = MLP(feature_dim + 3, hidden_dim, message_dim)

    def compute_edge_features(self, z_l, edge_index):
        source, target = edge_index
        diff_ij = z_l[source, 0:3] - z_l[target, 0:3]
        dist_ij = (diff_ij ** 2).sum(-1, keepdim=True)
        vel_crossprod_ij = torch.cross(z_l[source, 3:6], z_l[target, 3:6], dim=-1)
        abs_vel_crossprod_ij = torch.norm(vel_crossprod_ij, dim=-1, keepdim=True)
        return torch.cat([diff_ij, dist_ij, vel_crossprod_ij, abs_vel_crossprod_ij], dim=-1)

    def forward(self, z_l, edge_index_l_l, pos_world):
        # Local message passing (l-l edges)
        e_l_l = self.compute_edge_features(z_l, edge_index_l_l)
        source, target = edge_index_l_l
        input_edge = torch.cat([z_l[source], z_l[target], e_l_l], dim=-1)
        m_l_l = self.mlp_edge(input_edge)
        w_l_l = torch.sigmoid(self.mlp_weight(input_edge))
        m_agg_l = torch.zeros_like(z_l).scatter_add_(0, target.unsqueeze(-1).expand_as(m_l_l), w_l_l * m_l_l)

        # World message (l-world edges)
        diff_world = z_l[:, 0:3] - pos_world
        input_world = torch.cat([z_l, diff_world], dim=-1)
        m_l_w = self.mlp_world(input_world)
        m_agg_w = m_l_w.sum(dim=0, keepdim=True).expand_as(z_l)

        # Combine messages
        m_agg_total = m_agg_l + m_agg_w
        input_node = torch.cat([z_l, m_agg_total], dim=-1)
        delta_z_l = self.mlp_node(input_node)
        return delta_z_l
    

class GNN_h_l(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim + message_dim, hidden_dim, feature_dim)

    def compute_edge_features(self, z_h, z_l, edge_index):
        source, target = edge_index
        diff_hl = z_h[source, 0:3] - z_l[target, 0:3]
        dist_hl = (diff_hl ** 2).sum(-1, keepdim=True)
        vel_crossprod_hl = torch.cross(z_h[source, 3:6], z_l[target, 3:6], dim=-1)
        abs_vel_crossprod_hl = torch.norm(vel_crossprod_hl, dim=-1, keepdim=True)
        return torch.cat([diff_hl, dist_hl, vel_crossprod_hl, abs_vel_crossprod_hl], dim=-1)

    def forward(self, z_h, z_l, edge_index_h_l):
        e_h_l = self.compute_edge_features(z_h, z_l, edge_index_h_l)
        source, target = edge_index_h_l
        input_edge = torch.cat([z_h[source], z_l[target], e_h_l], dim=-1)
        m_h_l = self.mlp_edge(input_edge)
        w_h_l = torch.sigmoid(self.mlp_weight(input_edge))
        m_agg_l = torch.zeros_like(z_l).scatter_add_(0, target.unsqueeze(-1).expand_as(m_h_l), w_h_l * m_h_l)
        input_node = torch.cat([z_l, m_agg_l], dim=-1)
        delta_z_l = self.mlp_node(input_node)
        return delta_z_l
    
class GNN_l_h(nn.Module):
    def __init__(self, feature_dim, message_dim, hidden_dim):
        super().__init__()
        edge_dim = 8
        self.mlp_edge = MLP(feature_dim * 2 + edge_dim, hidden_dim, message_dim)
        self.mlp_weight = MLP(feature_dim * 2 + edge_dim, hidden_dim, 1)
        self.mlp_node = MLP(feature_dim + message_dim, hidden_dim, feature_dim)

    def compute_edge_features(self, z_l, z_h, edge_index):
        source, target = edge_index
        diff_lh = z_l[source, 0:3] - z_h[target, 0:3]
        dist_lh = (diff_lh ** 2).sum(-1, keepdim=True)
        vel_crossprod_lh = torch.cross(z_l[source, 3:6], z_h[target, 3:6], dim=-1)
        abs_vel_crossprod_lh = torch.norm(vel_crossprod_lh, dim=-1, keepdim=True)
        return torch.cat([diff_lh, dist_lh, vel_crossprod_lh, abs_vel_crossprod_lh], dim=-1)

    def forward(self, z_l, z_h, edge_index_l_h):
        e_l_h = self.compute_edge_features(z_l, z_h, edge_index_l_h)
        source, target = edge_index_l_h
        input_edge = torch.cat([z_l[source], z_h[target], e_l_h], dim=-1)
        m_l_h = self.mlp_edge(input_edge)
        w_l_h = torch.sigmoid(self.mlp_weight(input_edge))
        m_agg_h = torch.zeros_like(z_h).scatter_add_(0, target.unsqueeze(-1).expand_as(m_l_h), w_l_h * m_l_h)
        input_node = torch.cat([z_h, m_agg_h], dim=-1)
        delta_z_h = self.mlp_node(input_node)
        return delta_z_h
    
class ODEFunc(nn.Module):
    def __init__(self, feature_dim=67, message_dim=64, hidden_dim=256):
        super().__init__()
        self.gnn_h = GNN_h(feature_dim, message_dim, hidden_dim)
        self.gnn_l = GNN_l(feature_dim, message_dim, hidden_dim)
        self.gnn_h_l = GNN_h_l(feature_dim, message_dim, hidden_dim)
        self.gnn_l_h = GNN_l_h(feature_dim, message_dim, hidden_dim)
        self.mlp_vel = MLP(feature_dim, hidden_dim, 3)    # For dt_pos
        self.mlp_omega = MLP(feature_dim, hidden_dim, 3)  # For dt_quat
        self.radius_h = None
        self.radius_l = None
        self.pos_world = torch.zeros(3, device='cuda')

    def compute_radius(self, pos, k=10):
        edge_index = knn(pos, pos, k + 1)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remove self-loops
        diff = pos[edge_index[0]] - pos[edge_index[1]]
        distances = torch.norm(diff, dim=-1)
        mean_distances = torch.zeros(pos.size(0), device=pos.device)
        mean_distances.scatter_add_(0, edge_index[1], distances)
        counts = torch.ones(pos.size(0), device=pos.device) * k
        return (mean_distances / counts).mean().item()

    def forward(self, t, z):
        z_h, z_l = z
        pos_h = get_feature(z_h, 'pos')
        pos_l = get_feature(z_l, 'pos')

        # Compute radii if not already set
        if self.radius_h is None:
            self.radius_h = self.compute_radius(pos_h, k=10)
        if self.radius_l is None:
            self.radius_l = self.compute_radius(pos_l, k=10)

        # Build edges
        edge_index_h_h = build_radius_edges(pos_h, pos_h, self.radius_h)
        edge_index_l_l = build_radius_edges(pos_l, pos_l, self.radius_l)
        edge_index_h_l = build_knn_edges(pos_h, pos_l, k=3)
        edge_index_l_h = build_knn_edges(pos_l, pos_h, k=3)

        # Compute GNN updates
        delta_z_h = self.gnn_h(z_h, edge_index_h_h, self.pos_world)
        delta_z_l = self.gnn_l(z_l, edge_index_l_l, self.pos_world)

        # Compute derivatives for high-level nodes
        dz_h_dt = torch.zeros_like(z_h)
        set_feature(dz_h_dt, 'pos', get_feature(z_h, 'dt_pos'))
        set_feature(dz_h_dt, 'dt_pos', self.mlp_vel(z_h))
        omega = get_feature(z_h, 'dt_quat')[:, :3]
        quat = get_feature(z_h, 'quat')
        quat_deriv = 0.5 * quaternion_multiply(torch.cat([omega, torch.zeros_like(omega[:, :1])], dim=-1), quat)
        set_feature(dz_h_dt, 'quat', quat_deriv)
        set_feature(dz_h_dt, 'dt_quat', torch.cat([self.mlp_omega(z_h), torch.zeros_like(z_h[:, :1])], dim=-1))
        set_feature(dz_h_dt, 'rgb', torch.zeros_like(get_feature(z_h, 'rgb')))
        set_feature(dz_h_dt, 'semantic', torch.zeros_like(get_feature(z_h, 'semantic')))
        set_feature(dz_h_dt, 'augm', torch.zeros_like(get_feature(z_h, 'augm')))

        # Placeholder for low-level nodes (simplified here; extend as needed)
        dz_l_dt = dz_h_dt.clone()

        return [dz_h_dt + delta_z_h, dz_l_dt + delta_z_l]


class MSGNODEProcessor(nn.Module):
    def __init__(self, feature_dim=67, message_dim=64, hidden_dim=256):
        super().__init__()
        self.ode_func = ODEFunc(feature_dim, message_dim, hidden_dim)
    
    def forward(self, z0_h, z0_l, t):
        from torchdiffeq import odeint
        z0 = [z0_h, z0_l]
        z_traj = odeint(self.ode_func, z0, t, method='rk4')
        z_h_traj, z_l_traj = z_traj
        return z_h_traj, z_l_traj




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
