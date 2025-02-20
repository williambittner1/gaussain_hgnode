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
# Augmented Graph Neural ODE Function with 2 Message Passing Steps
# =============================================
class AugmentedGraphNeuralODEFunc(nn.Module):
    def __init__(self, original_dim, hidden_dim, n_hidden_layers):
        """
        ODE function for augmented node dynamics.
        
        Input state has shape [B, N, 2*original_dim]:
          - First original_dim entries: dynamic (evolving)
          - Second original_dim entries: static (a copy of z₀ that remains unchanged)
          
        In addition, edge features are built from both dynamic and static information.
        This version performs two sequential message passing update steps.
        """
        super(AugmentedGraphNeuralODEFunc, self).__init__()
        self.original_dim = original_dim
        self.augmented_dim = 2 * original_dim
        
        # Define the edge network.
        # For dynamic edge features, we use the evolving part.
        # Here we compute dynamic features from evolving nodes (of dim original_dim)
        # plus we add a small static edge feature from the static part (here we simply use [diff, dist] of size 4).
        # So the combined edge input dimension is: (2*original_dim + 4) + 4 = 2*original_dim + 8.
        self.edge_input_dim = 2 * original_dim + 8
        self.edge_net = self._build_mlp(self.edge_input_dim, hidden_dim, original_dim, n_hidden_layers)
        
        # Instead of one update network, we now define two sequential update networks.
        # Each takes as input the concatenation of the current evolving state and the aggregated edge message.
        update_input_dim = 2 * original_dim
        self.update_net1 = self._build_mlp(update_input_dim, hidden_dim, original_dim, n_hidden_layers)
        self.update_net2 = self._build_mlp(update_input_dim, hidden_dim, original_dim, n_hidden_layers)
        
        self.nfe = 0

    def _build_mlp(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def compute_agg_message(self, z_aug):
        """
        Computes aggregated edge messages using augmented node state z_aug.
        
        z_aug: [B, N, 2*original_dim]
        Returns:
            agg_message: [B, N, original_dim]
        """
        B, N, _ = z_aug.shape
        # Split augmented state.
        evolving = z_aug[..., :self.original_dim]   # dynamic part
        static   = z_aug[..., self.original_dim:]       # static part
        
        # Dynamic edge features: use evolving part.
        pos_dyn = evolving[..., :3]  # assume first 3 dims are position
        diff_dyn = pos_dyn.unsqueeze(2) - pos_dyn.unsqueeze(1)  # [B, N, N, 3]
        dist_dyn = torch.norm(diff_dyn, dim=-1, keepdim=True)     # [B, N, N, 1]
        # For dynamic features, use the evolving node features.
        x_i_dyn = evolving.unsqueeze(2).expand(B, N, N, self.original_dim)
        x_j_dyn = evolving.unsqueeze(1).expand(B, N, N, self.original_dim)
        edge_features_dyn = torch.cat([x_i_dyn, x_j_dyn, diff_dyn, dist_dyn], dim=-1)  # shape: [B, N, N, 2*original_dim+4]
        
        # Static edge features: computed from static part.
        pos_stat = static[..., :3]
        diff_stat = pos_stat.unsqueeze(2) - pos_stat.unsqueeze(1)  # [B, N, N, 3]
        dist_stat = torch.norm(diff_stat, dim=-1, keepdim=True)      # [B, N, N, 1]
        # Here, for static edge features, we simply use diff and distance.
        edge_features_stat = torch.cat([diff_stat, dist_stat], dim=-1)  # shape: [B, N, N, 4]
        
        # Combine the two: total dimension = (2*original_dim+4) + 4 = 2*original_dim+8.
        edge_features = torch.cat([edge_features_dyn, edge_features_stat], dim=-1)
        
        # Pass through edge network.
        edge_messages = self.edge_net(edge_features)  # [B, N, N, original_dim]
        agg_message = edge_messages.sum(dim=2)         # [B, N, original_dim]
        return agg_message

    def forward(self, t, z_aug):
        """
        z_aug: [B, N, 2*original_dim]
        Returns dz/dt with shape [B, N, 2*original_dim], where the derivative for the static part is zero.
        """
        B, N, _ = z_aug.shape
        # Split augmented state.
        evolving = z_aug[..., :self.original_dim]
        static = z_aug[..., self.original_dim:]
        
        # --- First message passing update ---
        agg_message1 = self.compute_agg_message(z_aug)  # [B, N, original_dim]
        update_input1 = torch.cat([evolving, agg_message1], dim=-1)  # [B, N, 2*original_dim]
        d_evolving1 = self.update_net1(update_input1)  # [B, N, original_dim]
        # Compute intermediate evolving state.
        evolving_intermediate = evolving + d_evolving1  # (this is an intermediate value)
        
        # --- Second message passing update ---
        # Rebuild augmented state with updated evolving part.
        z_aug_updated = torch.cat([evolving_intermediate, static], dim=-1)
        agg_message2 = self.compute_agg_message(z_aug_updated)
        update_input2 = torch.cat([evolving_intermediate, agg_message2], dim=-1)
        d_evolving2 = self.update_net2(update_input2)  # [B, N, original_dim]
        
        # Define overall derivative for evolving part as d_evolving2.
        d_evolving = d_evolving2
        d_static = torch.zeros_like(static)
        dz_aug = torch.cat([d_evolving, d_static], dim=-1)
        self.nfe += 1
        return dz_aug

# =============================================
# Augmented GraphNeuralODE Module with 2 Message Passing Updates
# =============================================
class AugmentedGraphNeuralODE(nn.Module):
    def __init__(self, original_dim, hidden_dim=256, n_hidden_layers=4,
                 solver="rk4", rtol=1e-4, atol=1e-5, options={"step_size": 0.04}, device="cpu"):
        """
        Graph Neural ODE with augmentation and two message passing updates per integration step.
        
        Given an explicit initial state z₀ (shape [B, N, original_dim]), we form an augmented state:
            z₀_aug = [z₀, z₀]  (shape [B, N, 2*original_dim])
        The dynamic part (first half) is evolved; the static part remains constant.
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
        """
        # Form augmented initial state.
        z0_aug = torch.cat([z0_explicit, z0_explicit], dim=-1)  # [B, N, 2*original_dim]
        z0_aug = z0_aug.to(self.device)
        t_span = t_span.to(self.device)
        traj_aug = odeint(self.func, z0_aug, t_span, method=self.solver,
                            rtol=self.rtol, atol=self.atol, options=self.options)
        # traj_aug shape: [T, B, N, 2*original_dim] -> permute to [B, T, N, 2*original_dim]
        traj_aug = traj_aug.permute(1, 0, 2, 3)
        # Extract evolving part.
        explicit_traj = traj_aug[..., :self.original_dim]
        return explicit_traj
