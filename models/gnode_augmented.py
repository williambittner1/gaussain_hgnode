# =============================================
# Augmented GraphNeuralODE Module
# the z-feature is augmented with concatenation of initial state for each node
# =============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

# Example helper functions (quaternion multiply/derivative remain unchanged if needed)
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
# Augmented Graph Neural ODE Function
# =============================================
class AugmentedGraphNeuralODEFunc(nn.Module):
    def __init__(self, original_dim, hidden_dim, n_hidden_layers):
        """
        ODE function for augmented node dynamics.
        
        The input state has shape [B, N, 2*original_dim]:
          - The first original_dim entries are dynamic.
          - The second original_dim entries are a static copy of z0.
          
        For message passing, we use the full augmented state. However, only the evolving
        (first half) is updated; the static part has zero derivative.
        
        We assume that in the evolving part the first 3 dimensions correspond to position.
        """
        super(AugmentedGraphNeuralODEFunc, self).__init__()
        self.original_dim = original_dim
        self.augmented_dim = 2 * original_dim
        # For message passing, use the full augmented state.
        # Each edge gets: state_i | state_j | relative position (from evolving part) | distance.
        # Input dimension = 2 * augmented_dim + 4.
        edge_input_dim = 2 * self.augmented_dim + 4 + 4 # 2 * augmented node dim + 4 (current relative position and distance) + 4 (initial relative position and distance)
        # Let the edge network output a message of dimension original_dim.
        self.edge_net = self._build_mlp(edge_input_dim, hidden_dim, original_dim, n_hidden_layers)
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
        Returns dz/dt with shape [B, N, 2*original_dim], where the derivative of the second half is zero.
        """
        B, N, _ = z_aug.shape
        # Split into dynamic and static parts.
        evolving = z_aug[..., :self.original_dim]   # [B, N, original_dim]
        static = z_aug[..., self.original_dim:]       # [B, N, original_dim] (should remain unchanged)
        
        # For message passing, use the full augmented state.
        # Compute relative position from the evolving part (assume first 3 dims are position).
        pos = evolving[..., :3]  # [B, N, 3]
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]


        pos_static = static[..., :3]
        diff_static = pos_static.unsqueeze(2) - pos_static.unsqueeze(1)  # [B, N, N, 3]
        dist_static = torch.norm(diff_static, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        # Expand full augmented state for each edge.
        x_i = z_aug.unsqueeze(2).expand(B, N, N, self.augmented_dim)
        x_j = z_aug.unsqueeze(1).expand(B, N, N, self.augmented_dim)
        edge_features = torch.cat([x_i, x_j, diff, dist, diff_static, dist_static], dim=-1)  # shape: [B, N, N, 2*augmented_dim + 8]
        
        # Compute edge messages.
        edge_messages = self.edge_net(edge_features)  # [B, N, N, original_dim]
        agg_message = edge_messages.sum(dim=2)         # [B, N, original_dim]
        
        # Update input: concatenate evolving part with aggregated message.
        update_input = torch.cat([evolving, agg_message], dim=-1)  # [B, N, 2*original_dim]
        d_evolving = self.update_net(update_input)  # [B, N, original_dim]
        
        # Static part remains constant.
        d_static = torch.zeros(B, N, self.original_dim, device=z_aug.device, dtype=z_aug.dtype)
        dz_aug = torch.cat([d_evolving, d_static], dim=-1)  # [B, N, 2*original_dim]
        self.nfe += 1
        return dz_aug

# =============================================
# Augmented GraphNeuralODE Module
# =============================================
class AugmentedGraphNeuralODE(nn.Module):
    def __init__(self, original_dim, hidden_dim=256, n_hidden_layers=4,
                 solver="rk4", rtol=1e-4, atol=1e-5, options={"step_size": 0.04}, device="cpu"):
        """
        Graph Neural ODE with augmentation.
        
        Given an initial explicit state z0 of dimension original_dim, we form an augmented state:
            z0_aug = [z0, z0]  (dimension 2*original_dim)
        The dynamic part (first half) is evolved; the second half is kept constant.
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
        self.z0_offset = nn.Parameter(torch.zeros(1, 1, self.augmented_dim, device=device))
    def forward(self, z0_explicit, t_span):
        """
        z0_explicit: [B, N, original_dim] explicit initial state.
        Returns:
            explicit_traj: [B, T, N, original_dim] trajectory for the evolving part.
            (The static part remains equal to the initial state.)
        """
        # Augment z0 by concatenating it with itself.
        z0_aug = torch.cat([z0_explicit, z0_explicit], dim=-1)  # [B, N, 2*original_dim]
        z0_aug = z0_aug.to(self.device)
        t_span = t_span.to(self.device)
        traj_aug = odeint(self.func, z0_aug, t_span, method=self.solver,
                            rtol=self.rtol, atol=self.atol, options=self.options)
        # traj_aug shape: [T, B, N, 2*original_dim]. Permute to [B, T, N, 2*original_dim]
        traj_aug = traj_aug.permute(1, 0, 2, 3)
        # Extract the evolving part (first half) as our explicit trajectory.
        explicit_traj = traj_aug[..., :self.original_dim]  # [B, T, N, original_dim]
        return explicit_traj
