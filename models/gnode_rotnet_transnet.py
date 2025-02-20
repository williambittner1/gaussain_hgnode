import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

# Helper: quaternion multiplication.
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    q1, q2: tensors of shape (..., 4) in (w, x, y, z) format.
    Returns: quaternion product of shape (..., 4).
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

# Helper: quaternion derivative from angular velocity.
def quaternion_derivative(q, omega):
    """
    Compute the quaternion derivative.
    q: tensor of shape (..., 4) representing quaternions (w, x, y, z).
    omega: tensor of shape (..., 3) representing angular velocity.
    Returns:
        dq/dt = 0.5 * q ⊗ [0, omega]
    """
    zeros = torch.zeros_like(omega[..., :1])
    omega_quat = torch.cat([zeros, omega], dim=-1)  # make it (w, x, y, z) with w=0
    return 0.5 * quaternion_multiply(q, omega_quat)


class GraphNeuralODEFunc(nn.Module):
    def __init__(self, node_feature_dim, node_conditioning_dim, hidden_dim, n_hidden_layers):
        """
        ODE function for node dynamics using graph message passing.
        
        The full node state is assumed to have:
          - Evolving features of dimension `node_feature_dim`
          - Conditioning features of dimension `node_conditioning_dim`
        The conditioning features remain constant during integration.
        
        For the evolving state, we assume it is partitioned as:
            pos (3), vel (3), quat (4), omega (3)
        so that node_feature_dim is 13.
        
        Two separate update networks are used:
            - transnet: computes dt_vel (3 dims)
            - rotnet: computes dt_omega (3 dims)
        dt_pos is simply the current velocity, and dt_quat is given by the quaternion derivative formula.
        
        Args:
            node_feature_dim (int): Dimension of the evolving state (e.g., 13).
            node_conditioning_dim (int): Dimension of the constant conditioning features.
            hidden_dim (int): Hidden layer size for the MLPs.
            n_hidden_layers (int): Number of hidden layers in each MLP.
        """
        super(GraphNeuralODEFunc, self).__init__()
        self.node_feature_dim = node_feature_dim   # expect 13
        self.node_conditioning_dim = node_conditioning_dim
        self.total_dim = node_feature_dim + node_conditioning_dim

        # For message passing, we use the full node state (evolving + conditioning).
        # For each edge between nodes i and j, we concatenate:
        #    state_i (total_dim) | state_j (total_dim) | relative position (3) | distance (1)
        # Total input dim = 2 * total_dim + 4.
        edge_input_dim = 2 * self.total_dim + 4
        # We let the edge network output a message; here we choose an output dimension of node_feature_dim
        # (you may choose a different message dimension if desired).
        self.edge_net = self._build_mlp(edge_input_dim, hidden_dim, node_feature_dim, n_hidden_layers)
        
        # Instead of one update network, we use two:
        # For transnet (velocity update), we feed in the current velocity (3 dims) and the aggregated message.
        # For rotnet (angular velocity update), we feed in the current omega (3 dims) and the aggregated message.
        # Here we assume the edge network produces an aggregated message of dimension node_feature_dim.
        # We set the input dimension for both nets as: node_feature_dim + 3.
        update_input_dim = node_feature_dim + 3
        self.transnet = self._build_mlp(update_input_dim, hidden_dim, 3, n_hidden_layers)
        self.rotnet = self._build_mlp(update_input_dim, hidden_dim, 3, n_hidden_layers)
        
        self.nfe = 0

    def _build_mlp(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, t, z_objects):
        B, N, D = z_objects.shape  # D = total_dim
        # Split the evolving part from the full state (for computing relative distances, etc.)
        evolving = z_objects[..., :self.node_feature_dim]   # [B, N, node_feature_dim]
        
        # Partition the evolving state into its components.
        pos   = evolving[..., :3]        # [B, N, 3]
        vel   = evolving[..., 3:6]       # [B, N, 3]
        quat  = evolving[..., 6:10]      # [B, N, 4]
        omega = evolving[..., 10:13]     # [B, N, 3]
        
        # Normalize quaternion
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        
        # Compute known derivatives.
        dt_pos = vel
        dt_quat = quaternion_derivative(quat, omega)
        
        # Compute pairwise differences using position.
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        # For message passing, use the full node state.
        x_i = z_objects.unsqueeze(2).expand(B, N, N, D)
        x_j = z_objects.unsqueeze(1).expand(B, N, N, D)
        edge_features = torch.cat([x_i, x_j, diff, dist], dim=-1)  # [B, N, N, 2*total_dim + 4]
        edge_messages = self.edge_net(edge_features)  # [B, N, N, node_feature_dim]
        agg_message = edge_messages.sum(dim=2)         # [B, N, node_feature_dim]
        
        # Concatenate the entire node state (both evolving and conditioning) with agg_message.
        full_input = torch.cat([z_objects, agg_message], dim=-1)  # [B, N, total_dim + node_feature_dim]
        
        # Use full_input for both networks.
        d_vel = self.transnet(full_input)  # [B, N, 3]
        d_omega = self.rotnet(full_input)  # [B, N, 3]
        
        # Reconstruct the derivative for the evolving state.
        d_evolving = torch.cat([dt_pos, d_vel, dt_quat, d_omega], dim=-1)
        
        # Conditioning remains constant.
        d_conditioning = torch.zeros(B, N, self.node_conditioning_dim, device=z_objects.device, dtype=z_objects.dtype)
        
        dz = torch.cat([d_evolving, d_conditioning], dim=-1)
        self.nfe += 1
        return dz



class GraphNeuralODE(nn.Module):
    def __init__(self, node_feature_dim, node_conditioning_dim, hidden_dim=256, n_hidden_layers=4,
                 solver="rk4", rtol=1e-4, atol=1e-5, options={"step_size": 0.04}, device="cpu"):
        """
        Graph Neural ODE that operates on object nodes.
        
        Args:
            node_feature_dim (int): Dimension of the evolving state (e.g., 13).
            node_conditioning_dim (int): Dimension of the conditioning features.
            hidden_dim (int): Hidden layer size.
            n_hidden_layers (int): Number of hidden layers in the ODE function.
            solver (str): ODE solver to use.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.
            options (dict): Additional solver options.
            device (str): Device to run the model on.
        """
        super(GraphNeuralODE, self).__init__()
        total_dim = node_feature_dim + node_conditioning_dim
        self.func = GraphNeuralODEFunc(node_feature_dim, node_conditioning_dim, hidden_dim, n_hidden_layers).to(device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = options
        self.device = device
        self.to(device)

    def forward(self, z0_object, t_span):
        """
        Simulate the object dynamics.
        
        Args:
            z0_object: Tensor of shape [B, N, D], the initial node states.
            t_span: 1D tensor of timesteps at which to evaluate the solution.
            
        Returns:
            traj_objects: Tensor of shape [B, T, N, D] representing the object state trajectory.
        """
        z0_object = z0_object.to(self.device)
        t_span = t_span.to(self.device)
        traj = odeint(self.func, z0_object, t_span, method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        traj_objects = traj.permute(1, 0, 2, 3)  # from [T, B, N, D] to [B, T, N, D]
        return traj_objects

