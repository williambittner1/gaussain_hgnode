import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

import torch.nn.functional as F
from torchdiffeq import odeint

class GraphNeuralODEFunc_no_conditioning(nn.Module):
    def __init__(self, object_dim, hidden_dim, n_hidden_layers):
        """
        ODE function that models object (node) dynamics via graph message passing.
        
        Args:
            object_dim (int): Dimension of the object state (e.g., 13 for [pos, vel, quat, omega]).
            hidden_dim (int): Hidden layer dimension for the MLPs.
            n_hidden_layers (int): Number of hidden layers in each MLP.
        """
        super(GraphNeuralODEFunc, self).__init__()
        self.object_dim = object_dim
        
        # Edge network: takes as input the concatenation of:
        #  - state of object i: [object_dim] - pos + rot + vel + omega (+ latent shape feature conditioning)
        #  - state of object j: [object_dim] - pos + rot + vel + omega (+ latent shape feature conditioning)
        #  - relative position: [3]
        #  - distance: [1]
        # Total input dimension: 2*object_dim + 3 + 1.
        edge_input_dim = 2 * object_dim + 4
        self.edge_net = self._build_mlp(edge_input_dim, hidden_dim, object_dim, n_hidden_layers)
        
        # Update network: takes as input the concatenation of:
        #  - current object state: [object_dim]
        #  - aggregated message: [object_dim]
        # Total input dimension: 2*object_dim.
        update_input_dim = 2 * object_dim
        self.update_net = self._build_mlp(update_input_dim, hidden_dim, object_dim, n_hidden_layers)
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
        """
        Compute the time derivative dx/dt for the object nodes.
        
        Args:
            t: current time (not used explicitly here, but can be concatenated if desired).
            z_objects: Tensor of shape [B, N, object_dim], where B is the batch size,
                      N is the number of object nodes.
                      
        Returns:
            dx: Tensor of shape [B, N, object_dim] representing the time derivative.
        """
        B, N, D = z_objects.shape
        # Extract positions from each object state.
        # Assume the first 3 dimensions correspond to the COM (position).
        pos = z_objects[..., :3]  # [B, N, 3]
        
        # Compute pairwise relative positions.
        # For each pair (i, j), compute diff = pos_i - pos_j.
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        # Compute the Euclidean distance for each pair.
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        # Expand the object state for each edge.
        x_i = z_objects.unsqueeze(2).expand(B, N, N, D)  # state of node i for each edge [B, N, N, D]
        x_j = z_objects.unsqueeze(1).expand(B, N, N, D)  # state of node j for each edge [B, N, N, D]
        
        # Concatenate edge features: [x_i, x_j, diff, dist]
        edge_features = torch.cat([x_i, x_j, diff, dist], dim=-1)  # [B, N, N, 2*D+3+1]
        
        # Pass edge features through the edge network.
        edge_messages = self.edge_net(edge_features)  # [B, N, N, object_dim]
        
        # Aggregate messages for each node by summing over j.
        agg_message = edge_messages.sum(dim=2)  # [B, N, object_dim]
        
        # Concatenate the aggregated message with the current object state.
        update_input = torch.cat([z_objects, agg_message], dim=-1)  # [B, N, 2*object_dim]
        
        # Compute the state derivative using the update network.
        dx = self.update_net(update_input)  # [B, N, object_dim]
        
        self.nfe += 1
        return dx
    

class GraphNeuralODEFunc(nn.Module):
    def __init__(self, node_feature_dim, node_conditioning_dim, hidden_dim, n_hidden_layers):
        """
        ODE function for object dynamics using graph message passing.
        
        The full node state is assumed to have:
          - Evolving features of dimension `node_feature_dim`
          - Conditioning features of dimension `node_conditioning_dim`
        The conditioning features remain constant during integration.
        
        Args:
            node_feature_dim (int): Dimension of the state features that evolve.
            node_conditioning_dim (int): Dimension of the conditioning features (constant).
            hidden_dim (int): Hidden layer size for the MLPs.
            n_hidden_layers (int): Number of hidden layers in each MLP.
        """
        super(GraphNeuralODEFunc, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_conditioning_dim = node_conditioning_dim
        self.total_dim = node_feature_dim + node_conditioning_dim

        # When computing messages, use the entire node state (both evolving and conditioning).
        # For each edge between nodes i and j, we concatenate:
        #    state_i (total_dim) | state_j (total_dim) | relative position (3) | distance (1)
        # Total input dim = 2 * total_dim + 4.
        edge_input_dim = 2 * self.total_dim + 4
        # We let the edge network output a message of size equal to node_feature_dim.
        self.edge_net = self._build_mlp(edge_input_dim, hidden_dim, node_feature_dim, n_hidden_layers)
        
        # The update network takes as input the concatenation of:
        #    current evolving state (node_feature_dim) and aggregated message (node_feature_dim).
        # Total input dim = 2 * node_feature_dim.
        self.update_net = self._build_mlp(2 * node_feature_dim, hidden_dim, node_feature_dim, n_hidden_layers)
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
        """
        Compute the time derivative dz/dt for each node.
        
        Args:
            t: current time (not explicitly used, but available if needed).
            z_objects: Tensor of shape [B, N, total_dim] where total_dim = node_feature_dim + node_conditioning_dim.
                      The first node_feature_dim entries evolve over time; the last node_conditioning_dim entries
                      are constant conditioning signals.
                      
        Returns:
            dz: Tensor of shape [B, N, total_dim] containing the time derivatives.
                The derivative for the conditioning part is zero.
        """
        B, N, D = z_objects.shape  # D should equal total_dim
        # Split the node state into the evolving features and conditioning features.
        evolving_state = z_objects[..., :self.node_feature_dim]    # [B, N, node_feature_dim]
        conditioning = z_objects[..., self.node_feature_dim:]        # [B, N, node_conditioning_dim]
        
        # For spatial message passing, assume the first 3 dimensions of the evolving state are positions.
        pos = evolving_state[..., :3]  # [B, N, 3]
        
        # Compute pairwise relative positions and distances.
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        # For message passing, use the full state (concatenate evolving state and conditioning).
        # Expand the full state for each edge.
        x_i = z_objects.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, total_dim]
        x_j = z_objects.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, total_dim]
        
        # Concatenate edge features: [state_i, state_j, relative position, distance]
        edge_features = torch.cat([x_i, x_j, diff, dist], dim=-1)  # [B, N, N, 2*total_dim + 4]
        
        # Compute messages for each edge.
        edge_messages = self.edge_net(edge_features)  # [B, N, N, node_feature_dim]
        
        # Aggregate messages for each node (sum over the second node index).
        agg_message = edge_messages.sum(dim=2)  # [B, N, node_feature_dim]
        
        # Concatenate the evolving state and the aggregated message.
        update_input = torch.cat([evolving_state, agg_message], dim=-1)  # [B, N, 2*node_feature_dim]
        
        # Compute derivative for the evolving part.
        d_evolving = self.update_net(update_input)  # [B, N, node_feature_dim]
        
        # The conditioning part should remain constant.
        d_conditioning = torch.zeros(B, N, self.node_conditioning_dim, device=z_objects.device, dtype=z_objects.dtype)
        
        # Concatenate to form the full derivative.
        dz = torch.cat([d_evolving, d_conditioning], dim=-1)  # [B, N, total_dim]
        
        self.nfe += 1
        return dz
    


class GraphNeuralODE(nn.Module):
    def __init__(self, node_feature_dim, node_conditioning_dim, hidden_dim=256, n_hidden_layers=4,
                 solver="rk4", rtol=1e-4, atol=1e-5, options={"step_size": 0.04}, device="cpu"):
        """
        Graph Neural ODE that only operates on object nodes.
        
        Args:
            node_dim (int): Dimension of the node features (e.g., 13).
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
       
        # self.func = GraphNeuralODEFunc(node_dim, hidden_dim, n_hidden_layers).to(device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = options
        self.device = device
        # Move the entire model to device
        self.to(device)

    def forward(self, z0_object, t_span):
        """
        Simulate the object dynamics.
        
        Args:
            z0_object: Tensor of shape [B, N, D], the initial node states.
            t_span: 1D tensor of timesteps at which to evaluate the solution.
            
        Returns:
            traj_objects: Tensor of shape [B, T, N, D] representing the object state trajectory.
            D: Dimension of the object state (e.g., 13 for [pos, quat, vel, omega]).
        """
        # Ensure inputs are on the correct device
        z0_object = z0_object.to(self.device)
        t_span = t_span.to(self.device)

        traj = odeint(self.func, z0_object, t_span, method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        
        traj_objects = traj.permute(1, 0, 2, 3) # [T, B, N, D] -> [B, T, N, D] 
        
        return traj_objects