import torch
import torch.nn as nn
from torchdiffeq import odeint

# Helper functions (unchanged)
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

def quaternion_derivative(q, omega):
    zeros = torch.zeros_like(omega[..., :1])
    omega_quat = torch.cat([zeros, omega], dim=-1)
    return 0.5 * quaternion_multiply(q, omega_quat)

# =============================================
# Augmented Graph Neural ODE Function with Augmented Edges
# =============================================
class AugmentedGraphNeuralODEFunc(nn.Module):
    def __init__(self, original_dim, hidden_dim, n_hidden_layers):
        """
        ODE function for augmented node dynamics.
        
        The input state has shape [B, N, 2*original_dim]:
          - The first original_dim entries are dynamic.
          - The second original_dim entries are a static copy of z0.
          
        For message passing, we now augment the edge features:
          - Dynamic edge features computed from the full augmented node state.
          - Static edge features computed from the static copy.
        These are concatenated and fed into the edge network.
        
        We assume that in the dynamic (evolving) part the first 3 dimensions correspond to position.
        """
        super(AugmentedGraphNeuralODEFunc, self).__init__()
        self.original_dim = original_dim
        self.augmented_dim = 2 * original_dim
        
        # Compute dynamic edge features:
        #   From full augmented state: dimension = 2 * augmented_dim + 4.
        dyn_edge_dim = 2 * original_dim + 4  # = 4*original_dim + 4
        # Compute static edge features:
        #   From static node state (of dimension original_dim): dimension = 2 * original_dim + 4.
        stat_edge_dim = 4
        # Combined edge feature dimension:
        self.edge_input_dim = dyn_edge_dim + stat_edge_dim  # = (2*original_dim+4) + (2*original_dim+4) = 4*original_dim + 8
        
        # Let the edge network output a message of dimension original_dim.
        self.edge_net = self._build_mlp(self.edge_input_dim, hidden_dim, original_dim, n_hidden_layers)
        
        # Update network: takes as input the concatenation of the evolving part and the aggregated message.
        # Input dimension = 2 * original_dim.
        self.update_net = self._build_mlp(2 * original_dim, hidden_dim, original_dim, n_hidden_layers)
        self.nfe = 0

    def _build_mlp(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, t, z_aug):
        """
        z_aug: [B, N, 2*original_dim]
        Returns dz/dt with shape [B, N, 2*original_dim],
        where the derivative of the second half (the static copy) is zero.
        """
        B, N, _ = z_aug.shape
        # Split augmented state into dynamic (evolving) and static parts.
        evolving = z_aug[..., :self.original_dim]   # [B, N, original_dim]
        static = z_aug[..., self.original_dim:]       # [B, N, original_dim]
        
        # -------------------------------
        # Dynamic edge features (from full augmented state)
        # -------------------------------
        # For relative positions, use the evolving part (assume first 3 dims are position).
        pos_dyn = evolving[..., :3]  # [B, N, 3]
        diff_dyn = pos_dyn.unsqueeze(2) - pos_dyn.unsqueeze(1)  # [B, N, N, 3]
        dist_dyn = torch.norm(diff_dyn, dim=-1, keepdim=True)     # [B, N, N, 1]
        # Expand the full augmented state for each edge.
        x_i_dyn = evolving.unsqueeze(2).expand(B, N, N, self.original_dim)
        x_j_dyn = evolving.unsqueeze(1).expand(B, N, N, self.original_dim)
        edge_features_dyn = torch.cat([x_i_dyn, x_j_dyn, diff_dyn, dist_dyn], dim=-1)  # shape: [B, N, N, 2*augmented_dim + 4]
        
        # -------------------------------
        # Static edge features (from static part)
        # -------------------------------
        # Use static node state (which is constant) to compute relative position.
        pos_stat = static[..., :3]  # [B, N, 3]
        diff_stat = pos_stat.unsqueeze(2) - pos_stat.unsqueeze(1)  # [B, N, N, 3]
        dist_stat = torch.norm(diff_stat, dim=-1, keepdim=True)      # [B, N, N, 1]
        # Expand static node state for each edge.
        x_i_stat = static.unsqueeze(2).expand(B, N, N, self.original_dim)
        x_j_stat = static.unsqueeze(1).expand(B, N, N, self.original_dim)
        # edge_features_stat = torch.cat([x_i_stat, x_j_stat, diff_stat, dist_stat], dim=-1)  # shape: [B, N, N, 2*original_dim+4]
        edge_features_stat = torch.cat([diff_stat, dist_stat], dim=-1)  # shape: [B, N, N, 4]
        # -------------------------------
        # Combine dynamic and static edge features.
        # -------------------------------
        edge_features = torch.cat([edge_features_dyn, edge_features_stat], dim=-1)  # shape: [B, N, N, 2*original_dim+8]
        
        # Compute edge messages.
        edge_messages = self.edge_net(edge_features)  # [B, N, N, original_dim]
        agg_message = edge_messages.sum(dim=2)         # [B, N, original_dim]
        
        # Update input for the dynamic part: concatenate evolving part with aggregated message.
        update_input = torch.cat([evolving, agg_message], dim=-1)  # [B, N, 2*original_dim]
        d_evolving = self.update_net(update_input)  # [B, N, original_dim]
        
        # The static part remains unchanged.
        d_static = torch.zeros(B, N, self.original_dim, device=z_aug.device, dtype=z_aug.dtype)
        dz_aug = torch.cat([d_evolving, d_static], dim=-1)  # [B, N, 2*original_dim]
        self.nfe += 1
        return dz_aug

# =============================================
# Augmented GraphNeuralODE Module with Augmented Edge Features
# =============================================
class AugmentedGraphNeuralODE(nn.Module):
    def __init__(self, original_dim, hidden_dim=256, n_hidden_layers=4,
                 solver="rk4", rtol=1e-4, atol=1e-5, options={"step_size": 0.04}, device="cpu"):
        """
        Graph Neural ODE with augmentation.
        
        Given an explicit initial state z₀ of dimension original_dim for each node, we form an augmented state:
            z₀_aug = [z₀, z₀]  (dimension 2*original_dim)
        The dynamic part (first half) is evolved; the second half is kept constant.
        Additionally, edge features are augmented with static information computed from z₀.
        """
        super(AugmentedGraphNeuralODE, self).__init__()
        self.original_dim = original_dim
        self.augmented_dim = 2 * original_dim
        self.func = AugmentedGraphNeuralODEFunc(original_dim, hidden_dim, n_hidden_layers).to(device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = options
        self.device = device
        self.to(device)
        
    def forward(self, z0_explicit, t_span):
        """
        z0_explicit: [B, N, original_dim] explicit initial state.
        Returns:
            explicit_traj: [B, T, N, original_dim] trajectory for the evolving part.
            (The static part remains equal to the initial state.)
        """
        # Augment z₀ by concatenating it with itself.
        z0_aug = torch.cat([z0_explicit, z0_explicit], dim=-1)  # [B, N, 2*original_dim]
        z0_aug = z0_aug.to(self.device)
        t_span = t_span.to(self.device)
        traj_aug = odeint(self.func, z0_aug, t_span, method=self.solver,
                            rtol=self.rtol, atol=self.atol, options=self.options)
        # traj_aug shape: [T, B, N, 2*original_dim] --> permute to [B, T, N, 2*original_dim]
        traj_aug = traj_aug.permute(1, 0, 2, 3)
        # Extract the evolving part (first half) as our explicit trajectory.
        explicit_traj = traj_aug[..., :self.original_dim]  # [B, T, N, original_dim]
        return explicit_traj
