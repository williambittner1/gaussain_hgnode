# ==============================
# Graph Latent Neural ODE
# Seems to perform much worse than gnode.
# ==============================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

# ==============================
# Helper Functions
# ==============================
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

# ==============================
# graph_Encoder and graph_Decoder Modules
# ==============================
class Graph_Encoder(nn.Module):
    def __init__(self, explicit_dim, latent_dim, hidden_dim, n_hidden_layers):
        """
        Maps the explicit state (e.g. [pos, vel, quat, omega, conditioning]) to a latent state.
        """
        super(Graph_Encoder, self).__init__()
        layers = []
        layers.append(nn.Linear(explicit_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: [B, N, explicit_dim]
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        latent = self.net(x)
        return latent.view(B, N, -1)

class Graph_Decoder(nn.Module):
    def __init__(self, latent_dim, explicit_dim, hidden_dim, n_hidden_layers):
        """
        Maps the latent state back to the explicit state.
        """
        super(Graph_Decoder, self).__init__()
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, explicit_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        # z shape: [B, N, latent_dim] or [B*T, N, latent_dim]
        B, N, _ = z.shape
        z = z.view(B * N, -1)
        explicit = self.net(z)
        return explicit.view(B, N, -1)

# ==============================
# Latent ODE Function (Process)
# ==============================
class GraphNeuralLatentODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_hidden_layers):
        """
        ODE function that operates on the latent state.
        We assume the latent state preserves spatial information in its first 3 dimensions.
        """
        super(GraphNeuralLatentODEFunc, self).__init__()
        self.latent_dim = latent_dim
        # For message passing, assume the first 3 dims of the latent state represent position.
        # For each edge, input dimension = 2*latent_dim + 4 (using relative differences computed from first 3 dims)
        edge_input_dim = 2 * latent_dim + 4
        self.edge_net = self._build_mlp(edge_input_dim, hidden_dim, latent_dim, n_hidden_layers)
        # Update net: input dimension = 2 * latent_dim.
        self.update_net = self._build_mlp(2 * latent_dim, hidden_dim, latent_dim, n_hidden_layers)
        self.nfe = 0

    def _build_mlp(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, t, z_latent):
        """
        Compute dz/dt in the latent space.
        z_latent: Tensor of shape [B, N, latent_dim].
        We use the first 3 dims of z_latent as the "position" for message passing.
        """
        B, N, D = z_latent.shape  # D = latent_dim
        pos = z_latent[..., :3]   # assume first 3 dims are spatial coordinates
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        # Expand latent state for each edge.
        x_i = z_latent.unsqueeze(2).expand(B, N, N, D)
        x_j = z_latent.unsqueeze(1).expand(B, N, N, D)
        edge_features = torch.cat([x_i, x_j, diff, dist], dim=-1)  # [B, N, N, 2*latent_dim+4]
        edge_messages = self.edge_net(edge_features)  # [B, N, N, latent_dim]
        agg_message = edge_messages.sum(dim=2)         # [B, N, latent_dim]
        update_input = torch.cat([z_latent, agg_message], dim=-1)  # [B, N, 2*latent_dim]
        dz_latent = self.update_net(update_input)  # [B, N, latent_dim]
        self.nfe += 1
        return dz_latent

# ==============================
# GraphNeuralODE with Encode-Process-Decode
# ==============================
class GraphLatentNeuralODE(nn.Module):
    def __init__(self, explicit_dim, latent_dim, hidden_dim=256, n_hidden_layers=4,
                 solver="rk4", rtol=1e-4, atol=1e-5, options={"step_size": 0.04}, device="cpu"):
        """
        Graph Neural ODE in an encode-process-decode framework.
        
        Args:
            explicit_dim (int): Dimension of the explicit state (e.g., node_feature_dim + node_conditioning_dim).
            latent_dim (int): Dimension of the latent space.
            hidden_dim (int): Hidden layer size.
            n_hidden_layers (int): Number of hidden layers in graph_encoder, graph_decoder, and latent ODE function.
            solver, rtol, atol, options: ODE solver parameters.
            device (str): Device.
        """
        super(GraphLatentNeuralODE, self).__init__()
        self.explicit_dim = explicit_dim
        self.latent_dim = latent_dim
        self.graph_encoder = Graph_Encoder(explicit_dim, latent_dim, hidden_dim, n_hidden_layers)
        self.graph_decoder = Graph_Decoder(latent_dim, explicit_dim, hidden_dim, n_hidden_layers)
        self.func = GraphNeuralLatentODEFunc(latent_dim, hidden_dim, n_hidden_layers).to(device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = options
        self.device = device
        self.to(device)
        
    def forward(self, z0_explicit, t_span):
        """
        Simulate dynamics in latent space and decode to explicit state.
        
        Args:
            z0_explicit: Tensor of shape [B, N, explicit_dim] (initial explicit state).
            t_span: 1D tensor of timesteps.
            
        Returns:
            explicit_traj: Tensor of shape [B, T, N, explicit_dim] (explicit state trajectory).
        """
        # Encode explicit state into latent space.
        latent0 = self.graph_encoder(z0_explicit)  # [B, N, latent_dim]
        latent0 = latent0.to(self.device)
        t_span = t_span.to(self.device)
        
        # Integrate the latent ODE.
        latent_traj = odeint(self.func, latent0, t_span,
                             method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        # latent_traj shape: [T, B, N, latent_dim] --> permute to [B, T, N, latent_dim]
        latent_traj = latent_traj.permute(1, 0, 2, 3)
        
        # Decode the latent trajectory at each timestep.
        B, T, N, _ = latent_traj.shape
        latent_traj_flat = latent_traj.reshape(B * T, N, self.latent_dim)
        explicit_traj_flat = self.graph_decoder(latent_traj_flat)  # [B*T, N, explicit_dim]
        explicit_traj = explicit_traj_flat.view(B, T, N, self.explicit_dim)
        return explicit_traj
