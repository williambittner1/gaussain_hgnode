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
# and Split Update Networks for dt_vel and dt_omega
# =============================================
class AugmentedGraphNeuralODEFunc(nn.Module):
    def __init__(self, original_dim, hidden_dim, n_hidden_layers, trans_dim, rot_dim):
        """
        ODE function for augmented node dynamics.
        
        Input state z has shape [B, N, 2*original_dim], where original_dim = 14.
        The first 14 entries (dynamic part) are arranged as:
            pos (3), vel (3), quat (4), omega (3), object_id (1)
        The second 14 entries form a static copy of z₀.
        
        We update only the physical dynamic parts:
          - Only vel (indices 3:6) and omega (indices 10:13) are updated.
          - dt_pos is set to the current vel.
          - dt_quat is computed from quaternion_derivative(quat, omega).
          - dt_object_id is zero.
        
        Two sequential message passing updates are performed. In each update, the input for
        the update networks is formed by concatenating [vel, omega, aggregated_message].
        trans_net outputs a delta for vel (3 dims) and rot_net outputs a delta for omega (3 dims).
        """
        super(AugmentedGraphNeuralODEFunc, self).__init__()
        assert original_dim == 14, "original_dim must be 14 (for [pos, vel, quat, omega, object_id])."
        # For updating, only vel and omega (each 3 dims) are updated.
        assert trans_dim == 3 and rot_dim == 3, "trans_dim and rot_dim must both be 3."
        
        self.original_dim = original_dim  # 14
        self.augmented_dim = 2 * original_dim  # 28
        
        # Build edge features.
        # Dynamic edge features: computed from the evolving part (first 14 dims).
        # We use: [x_i, x_j, diff_dyn, dist_dyn]
        #   x_i and x_j: each of dimension original_dim (14),
        #   diff_dyn (from pos): 3, dist_dyn: 1.
        # So dynamic part has size: 14+14+3+1 = 32.
        # Static edge features: from the static part.
        # We use: [diff_stat, dist_stat] (3+1 = 4).
        # Combined edge input dimension = 32 + 4 = 36.
        self.edge_input_dim = 36
        self.edge_net = self._build_mlp(self.edge_input_dim, hidden_dim, original_dim, n_hidden_layers)
        
        # Update input: we update only vel (3) and omega (3). We form update input by concatenating:
        # current vel (3) + current omega (3) + aggregated message (dimension = original_dim, i.e. 14).
        # So update input dimension = 3+3+14 = 20.
        update_input_dim = 20
        self.trans_net1 = self._build_mlp(update_input_dim, hidden_dim, trans_dim, n_hidden_layers)
        self.rot_net1   = self._build_mlp(update_input_dim, hidden_dim, rot_dim, n_hidden_layers)
        self.trans_net2 = self._build_mlp(update_input_dim, hidden_dim, trans_dim, n_hidden_layers)
        self.rot_net2   = self._build_mlp(update_input_dim, hidden_dim, rot_dim, n_hidden_layers)
        
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
        z_aug: [B, N, 2*original_dim] with original_dim = 14.
        Returns:
            agg_message: [B, N, original_dim] (i.e. 14)
        """
        B, N, _ = z_aug.shape
        evolving = z_aug[..., :self.original_dim]   # dynamic part (14)
        static   = z_aug[..., self.original_dim:]      # static part (14)
        
        # Dynamic edge features: from evolving part.
        pos_dyn = evolving[..., :3]  # [B, N, 3]
        diff_dyn = pos_dyn.unsqueeze(2) - pos_dyn.unsqueeze(1)  # [B, N, N, 3]
        dist_dyn = torch.norm(diff_dyn, dim=-1, keepdim=True)     # [B, N, N, 1]
        x_i_dyn = evolving.unsqueeze(2).expand(B, N, N, self.original_dim)  # [B, N, N, 14]
        x_j_dyn = evolving.unsqueeze(1).expand(B, N, N, self.original_dim)  # [B, N, N, 14]
        edge_features_dyn = torch.cat([x_i_dyn, x_j_dyn, diff_dyn, dist_dyn], dim=-1)  # [B, N, N, 14+14+3+1=32]
        
        # Static edge features: from static part.
        pos_stat = static[..., :3]  # [B, N, 3]
        diff_stat = pos_stat.unsqueeze(2) - pos_stat.unsqueeze(1)  # [B, N, N, 3]
        dist_stat = torch.norm(diff_stat, dim=-1, keepdim=True)      # [B, N, N, 1]
        edge_features_stat = torch.cat([diff_stat, dist_stat], dim=-1)  # [B, N, N, 4]
        
        # Combined edge features: [B, N, N, 32+4=36]
        edge_features = torch.cat([edge_features_dyn, edge_features_stat], dim=-1)
        edge_messages = self.edge_net(edge_features)  # [B, N, N, original_dim] i.e. [B, N, N, 14]
        agg_message = edge_messages.sum(dim=2)         # [B, N, 14]
        return agg_message

    def forward(self, t, z_aug):
        """
        z_aug: [B, N, 2*original_dim] with original_dim = 14.
        Returns dz/dt with shape [B, N, 2*original_dim],
        where the derivative for the static part is zero.
        """
        B, N, _ = z_aug.shape
        # Split augmented state.
        evolving = z_aug[..., :self.original_dim]  # dynamic part: [B, N, 14]
        static = z_aug[..., self.original_dim:]      # static part: [B, N, 14]
        
        # --- Extract explicit components from evolving ---
        pos    = evolving[..., :3]        # [B, N, 3]
        vel    = evolving[..., 3:6]       # [B, N, 3]
        quat   = evolving[..., 6:10]      # [B, N, 4]
        omega  = evolving[..., 10:13]     # [B, N, 3]
        obj_id = evolving[..., 13:]       # [B, N, 1] (should remain constant)
        
        # --- First message passing update ---
        agg_message1 = self.compute_agg_message(z_aug)  # [B, N, 14]
        # Form update input: concatenate current vel, omega, and aggregated message.
        # Shape: [B, N, 3 + 3 + 14] = [B, N, 20]
        update_input1 = torch.cat([vel, omega, agg_message1], dim=-1)
        delta_vel1 = self.trans_net1(update_input1)  # [B, N, 3]
        delta_omega1 = self.rot_net1(update_input1)    # [B, N, 3]
        # Compute intermediate updated velocity and omega.
        vel_int = vel + delta_vel1
        omega_int = omega + delta_omega1
        # Construct intermediate evolving state: keep pos, quat, obj_id unchanged.
        evolving_int = torch.cat([pos, vel_int, quat, omega_int, obj_id], dim=-1)  # [B, N, 14]
        # Rebuild updated augmented state.
        z_aug_updated = torch.cat([evolving_int, static], dim=-1)  # [B, N, 28]
        
        # --- Second message passing update ---
        agg_message2 = self.compute_agg_message(z_aug_updated)  # [B, N, 14]
        update_input2 = torch.cat([vel_int, omega_int, agg_message2], dim=-1)  # [B, N, 20]
        delta_vel2 = self.trans_net2(update_input2)  # [B, N, 3]
        delta_omega2 = self.rot_net2(update_input2)    # [B, N, 3]
        
        # Combine updates.
        total_delta_vel = delta_vel1 + delta_vel2  # [B, N, 3]
        total_delta_omega = delta_omega1 + delta_omega2  # [B, N, 3]
        
        # Now define the derivative for the dynamic (evolving) part.
        dt_pos = vel                         # derivative of pos is current vel
        dt_vel = total_delta_vel             # learned update for velocity
        dt_quat = quaternion_derivative(quat, omega)  # computed from physical relation
        dt_omega = total_delta_omega         # learned update for angular velocity
        dt_obj_id = torch.zeros(B, N, 1, device=z_aug.device, dtype=z_aug.dtype)  # object_id remains constant
        
        d_dynamic = torch.cat([dt_pos, dt_vel, dt_quat, dt_omega, dt_obj_id], dim=-1)  # [B, N, 14]
        d_static = torch.zeros(B, N, self.original_dim, device=z_aug.device, dtype=z_aug.dtype)  # static part derivative: [B, N, 14]
        dz_aug = torch.cat([d_dynamic, d_static], dim=-1)  # [B, N, 28]
        self.nfe += 1
        return dz_aug

# =============================================
# Augmented GraphNeuralODE Module with 2 Message Passing Updates
# and Split Update Networks (Trans and Rot) for dt_vel and dt_omega
# =============================================
class AugmentedGraphNeuralODE(nn.Module):
    def __init__(self, original_dim, hidden_dim=256, n_hidden_layers=4,
                 trans_dim=None, rot_dim=None, solver="rk4", rtol=1e-4, atol=1e-5,
                 options={"step_size": 0.04}, device="cpu"):
        """
        Graph Neural ODE with augmentation and two message passing updates per integration step,
        with split update networks for translation and rotation updates.
        
        Given an explicit initial state z₀ of dimension original_dim (here 14 for [pos, vel, quat, omega, object_id])
        for each node, we form an augmented state:
            z₀_aug = [z₀, z₀]  (shape [B, N, 28])
        The dynamic part (first 14) is evolved; the static part remains constant.
        
        We assume the dynamic state is structured as:
            pos (3), vel (3), quat (4), omega (3), object_id (1).
        We update only vel and omega (total 6 dimensions). Thus, set:
            trans_dim = 3, rot_dim = 3.
        """
        super(AugmentedGraphNeuralODE, self).__init__()
        self.original_dim = original_dim  # should be 14
        if trans_dim is None or rot_dim is None:
            trans_dim = 3
            rot_dim = 3
        else:
            assert trans_dim == 3 and rot_dim == 3, "For a 14-dim state, the update part (vel and omega) is 6-dim (3 each)."
        self.trans_dim = trans_dim
        self.rot_dim = rot_dim
        self.augmented_dim = 2 * original_dim  # 28
        self.func = AugmentedGraphNeuralODEFunc(original_dim, hidden_dim, n_hidden_layers, trans_dim, rot_dim).to(device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = options
        self.device = device
        self.to(device)
        
    def forward(self, z0_explicit, t_span):
        """
        z0_explicit: [B, N, original_dim] explicit initial state (14-dim).
        Returns:
            explicit_traj: [B, T, N, original_dim] trajectory for the evolving (dynamic) part.
        """
        # Form augmented initial state.
        z0_aug = torch.cat([z0_explicit, z0_explicit], dim=-1)  # [B, N, 28]
        z0_aug = z0_aug.to(self.device)
        t_span = t_span.to(self.device)
        traj_aug = odeint(self.func, z0_aug, t_span, method=self.solver,
                            rtol=self.rtol, atol=self.atol, options=self.options)
        # traj_aug shape: [T, B, N, 28] -> permute to [B, T, N, 28]
        traj_aug = traj_aug.permute(1, 0, 2, 3)
        # Extract evolving part.
        explicit_traj = traj_aug[..., :self.original_dim]  # [B, T, N, 14]
        return explicit_traj
