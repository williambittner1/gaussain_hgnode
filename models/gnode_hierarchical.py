# gnode_hierarchical.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

def compute_adjacency_batch(x, sigma=0.1):
    """
    Compute a batch of adjacency matrices from node features.
    Assumes the first 3 features are the positions.
    """
    pos = x[..., :3]  # [B, N, 3]
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
    dist_sq = (diff ** 2).sum(dim=-1)            # [B, N, N]
    A = torch.exp(-dist_sq / (sigma ** 2))         # [B, N, N]
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0)
    A = A * (1 - I) + I  # set diagonal to 1
    A = A / A.sum(dim=-1, keepdim=True)
    return A

class GraphNeuralODEFuncHierarchical(nn.Module):
    def __init__(self, particle_dim, object_dim, hidden_dim, n_hidden_layers, max_objects=10):
        """
        Hierarchical Graph Neural ODE function.
        - particle_dim: dimension of particle node features.
        - object_dim: dimension of object node features.
        - max_objects: maximum number of object nodes.
        """
        super(GraphNeuralODEFuncHierarchical, self).__init__()
        self.nfe = 0
        self.max_objects = max_objects
        self.particle_dim = particle_dim
        self.object_dim = object_dim

        # Assignment network: takes [particle_dim + 1] and outputs logits over max_objects.
        self.assign_net = nn.Sequential(
            nn.Linear(3 + 1, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_hidden_layers - 2),
            nn.Linear(hidden_dim, max_objects)
        )

        # Particle-to-object message network (for aggregation)
        self.object_msg_net = nn.Sequential(
            nn.Linear(particle_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim)
        )

        # Object update network (using aggregated particle messages and object–object interactions)
        self.object_update_net = nn.Sequential(
            nn.Linear(object_dim + object_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim)
        )

        # For dx_o_vel: correction for the object's linear velocity.
        self.f_vel = nn.Sequential(
            nn.Linear(object_dim + object_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )
        # For dx_o_omega: correction for the object's angular velocity.
        self.f_omega = nn.Sequential(
            nn.Linear(object_dim + object_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )



        # Object-to-object message network: used to compute messages among object nodes.
        self.edge_net = nn.Sequential(
            nn.Linear(2*object_dim + 3 + 1 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, object_dim)
        )

        # For transforming the object message to the particle space.
        self.obj_to_particle = nn.Linear(object_dim, particle_dim)


    def forward(self, t, state):
        """
        Forward pass for the ODE function.
        state: a tuple (x_p, x_o)
          - x_p: [B, N_p, particle_dim] with features [pos, vel, quat, pos_t0]
          - x_o: [B, max_objects, object_dim] with features [COM, vel, quat, omega]
        Updates:
          - Object state via object-to-object messaging.
          - Particle state by rigidly transforming each particle's pos_t0.
        """
        self.nfe += 1
        x_p, x_o = state
        B, N_p, D_p = x_p.shape
        B, N_o, D_o = x_o.shape
        
        
        t_exp_obj = t.view(1, 1, 1).expand(B, self.max_objects, self.max_objects, 1)

        # --- Compute soft assignments from particles to objects ---
        # t_exp = t.view(1, 1, 1).expand(B, N_p, 1)
        # assign_input = torch.cat([x_p, t_exp], dim=-1)  # [B, N_p, particle_dim+1]
        # logits = self.assign_net(assign_input)            # [B, N_p, max_objects]
        # self.S = F.softmax(logits, dim=-1)                     # soft assignment: [B, N_p, max_objects]


        # --- Object-to-object message passing ---
        A_obj = compute_adjacency_batch(x_o, sigma=0.3)  # [B, max_objects, max_objects]
        # A_obj = torch.ones(B, self.max_objects, self.max_objects, device=x_o.device)
        pos = x_o[..., :3]

        diff = pos.unsqueeze(1) - pos.unsqueeze(2)  # [B, N, N, 3]

        distance = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]

        h_i = x_o.unsqueeze(2).expand(B, N_o, N_o, x_o.size(-1))  # [B, N, N, object_dim]
        h_j = x_o.unsqueeze(1).expand(B, N_o, N_o, x_o.size(-1))
        

        edge_input = torch.cat([h_i, h_j, diff, distance, t_exp_obj], dim=-1)
        edge_message = self.edge_net(edge_input)  # [B, N, N, F+1]

        # Weight the edge messages by the adjacency matrix.
        A_exp = A_obj.unsqueeze(-1)  # [B, N, N, 1]
        weighted_edge_message = A_exp * edge_message  # [B, N, N, F+1]

        # Aggregate messages over the neighbors: sum over j.
        aggregated_message = weighted_edge_message.sum(dim=2)  # [B, N, F+1]

        m_obj = aggregated_message



        # --- Compute separate object state derivatives ---
        # Decompose current object state:
        COM_obj = x_o[:, :, :3]    # [B, max_objects, 3]
        vel_obj = x_o[:, :, 3:6]     # [B, max_objects, 3]
        quat_obj = x_o[:, :, 6:10]   # [B, max_objects, 4]
        omega_obj = x_o[:, :, 10:13] # [B, max_objects, 3]

        # dx_o_com: derivative of COM is simply the object's velocity.
        dx_o_com = vel_obj  # [B, max_objects, 3]

        t0 = t.view(1, 1, 1).expand(B, self.max_objects, 1)
        # dx_o_vel: use a small network to compute correction from m_obj.
        dx_o_vel = self.f_vel(torch.cat([x_o, m_obj, t0], dim=-1))  # 13 + 13 + 1 -> 13

        # dx_o_quat: use standard rigid-body kinematics for quaternions.
        # Represent omega as a pure quaternion: [0, omega].
        omega_quat = torch.cat([torch.zeros_like(omega_obj[..., :1]), omega_obj], dim=-1)  # [B, max_objects, 4]
        dx_o_quat = 0.5 * quat_mul(omega_quat, quat_obj)  # [B, max_objects, 4]

        # dx_o_omega: use a small network to compute correction from m_obj.
        dx_o_omega = self.f_omega(torch.cat([x_o, m_obj, t0], dim=-1))  # [B, max_objects, 3]

        # Concatenate to form dx_o:
        dx_o = torch.cat([dx_o_com, dx_o_vel, dx_o_quat, dx_o_omega], dim=-1)  # [B, max_objects, 13]
        
        # --- Object-to-particle message passing (Rigid Transformation) ---
        if self.S is None:
            raise ValueError("External soft assignment S must be provided (set self.S) before calling forward.")
        # S is [B, N_p, max_objects].



        # Parse object state:
        COM_obj = x_o[:, :, :3]      # [B, max_objects, 3]
        vel_obj = x_o[:, :, 3:6]       # [B, max_objects, 3]
        quat_obj = x_o[:, :, 6:10]     # [B, max_objects, 4]
        omega_obj = x_o[:, :, 10:13]   # [B, max_objects, 3]
        
        # For particles, use the stored pos_t0 (last 3 dims of x_p).
        pos_t0 = x_p[..., 10:13]  # [B, N_p, 3]
        
        # Compute local coordinates relative to each object's COM.
        local = pos_t0.unsqueeze(2) - COM_obj.unsqueeze(1)  # [B, N_p, max_objects, 3]
        
        # Rotate the local coordinates using the object quaternions.
        # We need to apply quat_rotate in a batched manner.
        B, N_p, M, _ = local.shape
        quat_obj_exp = quat_obj.unsqueeze(1).expand(B, N_p, M, 4)  # [B, N_p, max_objects, 4]
        local_flat = local.reshape(B * N_p * M, 3)
        quat_flat = quat_obj_exp.reshape(B * N_p * M, 4)
        rot_local_flat = quat_rotate(quat_flat, local_flat)  # [B*N_p*M, 3]
        rot_local = rot_local_flat.reshape(B, N_p, M, 3)  # [B, N_p, max_objects, 3]
        
        # Predicted particle position from each object: COM + rotated local.
        pred_pos = COM_obj.unsqueeze(1) + rot_local  # [B, N_p, max_objects, 3]
        
        # Predicted particle velocity: object velocity + (omega cross rotated local).
        omega_exp = omega_obj.unsqueeze(1).expand(B, N_p, M, 3)  # [B, N_p, max_objects, 3]
        rigid_vel_correction = torch.cross(omega_exp, rot_local, dim=-1)  # [B, N_p, max_objects, 3]
        pred_vel = vel_obj.unsqueeze(1) + rigid_vel_correction  # [B, N_p, max_objects, 3]
        
        # Combine contributions from all objects using soft assignment S.
        S_expanded = self.S.unsqueeze(-1)  # [B, N_p, max_objects, 1]
        final_pos = (S_expanded * pred_pos).sum(dim=2)  # [B, N_p, 3]
        final_vel = (S_expanded * pred_vel).sum(dim=2)  # [B, N_p, 3]
        



        # --- Particle State Update ---
        # Update particle position and velocity based on the rigid transformation.
        current_pos = x_p[..., :3]
        current_vel = x_p[..., 3:6]
        dpos = final_vel  # Let particle's derivative of position be the new velocity.
        dvel = final_vel - current_vel  # A simple correction toward the rigid prediction.
        
        # For the remaining features (particle quat and pos_t0), assume zero derivatives.
        dquat = torch.zeros_like(x_p[..., 6:10])
        dpos_t0 = torch.zeros_like(x_p[..., 10:13])
        dx_p = torch.cat([dpos, dvel, dquat, dpos_t0], dim=-1)  # [B, N_p, particle_dim]
        
        return (dx_p, dx_o)

class GraphNeuralODEHierarchical(nn.Module):
    def __init__(self, particle_dim, object_dim, hidden_dim=128, n_hidden_layers=4,
                 solver="dopri5", rtol=1e-3, atol=1e-5, options={"max_num_steps": 200}, max_objects=10, device="cpu"):
        """
        Hierarchical Graph Neural ODE that operates on a tuple of states (particles and objects).
        """
        super(GraphNeuralODEHierarchical, self).__init__()
        self.func = GraphNeuralODEFuncHierarchical(particle_dim, object_dim, hidden_dim, n_hidden_layers, max_objects=max_objects)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.options = options
        self.max_objects = max_objects

        # self.assigner = Assigner(particle_dim=particle_dim, hidden_dim=256, max_objects=3).to(device)
        self.S = None
        self.device = device

    def forward(self, z0, t_span):
        # z0 is a tuple: (x_p, x_o)
        B, N_p, D_p = z0[0].shape   
        B, N_o, D_o = z0[1].shape

        t0 = torch.zeros(B, N_p, 1, device=self.device)

        z0_particles = z0[0]
        z0_objects = z0[1]
    
        

        # odeint supports nested structures.
        traj = odeint(self.func, z0, t_span, method=self.solver, rtol=self.rtol, atol=self.atol, options=self.options)
        # traj is a tuple: (traj_particles, traj_objects) with shapes [T, B, ...]
        traj_particles, traj_objects = traj
        traj_particles = traj_particles.permute(1, 0, 2, 3)  # [B, T, N_p, particle_dim]
        traj_objects = traj_objects.permute(1, 0, 2, 3)        # [B, T, max_objects, object_dim]
        return traj_particles, traj_objects

    def rollout(self, z0, t_span):
        return self.forward(z0, t_span)


class Assigner(nn.Module):
    def __init__(self, particle_dim, hidden_dim, max_objects):
        super(Assigner, self).__init__()
        self.max_objects = max_objects
        self.net = nn.Sequential(
            nn.Linear(particle_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_objects)
        )
    def forward(self, particle_features, t):
        # particle_features: [B, N_particles, particle_dim]
        # t: [B, N_particles, 1]
        x = torch.cat([particle_features, t], dim=-1)
        logits = self.net(x)
        S = F.softmax(logits, dim=-1)
        return S
    


def quat_mul(q, r):
    """
    Multiply two quaternions.
    q, r: tensors of shape (..., 4) in the format [w, x, y, z].
    Returns a tensor of shape (..., 4).
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    q: tensor of shape (..., 4) in the format [w, x, y, z].
    Returns a tensor of shape (..., 4).
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_rotate(q, v):
    """
    Rotate a vector v by a quaternion q.
    q: tensor of shape (..., 4) in the format [w, x, y, z].
    v: tensor of shape (..., 3)
    Returns the rotated vector of shape (..., 3).
    """
    zero = torch.zeros_like(v[..., :1])
    v_quat = torch.cat([zero, v], dim=-1)  # Represent v as a pure quaternion [0, v].
    q_conj = quat_conjugate(q)
    rotated = quat_mul(quat_mul(q, v_quat), q_conj)
    return rotated[..., 1:]  # Return only the vector part.