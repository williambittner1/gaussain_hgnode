import torch

import torch.nn as nn

def quaternion_inverse(q):
    """
    Compute the inverse of a quaternion.
    q: Tensor of shape [M, 4] in [w, x, y, z] format.
    Returns a tensor of the same shape.
    """
    # For a unit quaternion, the inverse is just the conjugate.
    w = q[:, :1]
    xyz = q[:, 1:]
    return torch.cat([w, -xyz], dim=-1)

def quaternion_multiply(q1, q2):
    """
    Multiply two batches of quaternions.
    q1, q2: Tensors of shape [M, 4] in [w, x, y, z] format.
    Returns a tensor of shape [M, 4].
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_delta_to_angular_velocity(delta_q, dt=1.0, eps=1e-8):
    """
    Convert a delta quaternion (representing a rotation) to an angular velocity.
    delta_q: Tensor of shape [M, 4] (assumed normalized) in [w, x, y, z] format.
    dt: time step (assumed 1.0 if not provided).
    Returns: Tensor of shape [M, 3] representing angular velocity.
    """
    # Clamp w for numerical stability.
    w = torch.clamp(delta_q[:, 0], -1+eps, 1-eps)
    # The rotation angle (in radians) is 2 * acos(w)
    angle = 2 * torch.acos(w)
    # Compute sin(theta/2) safely.
    sin_half_angle = torch.sqrt(torch.clamp(1 - w**2, min=eps))
    # The rotation axis is the normalized vector part.
    axis = delta_q[:, 1:] / sin_half_angle.unsqueeze(-1)
    # Angular velocity: (angle / dt) * axis.
    omega = (angle / dt).unsqueeze(-1) * axis
    return omega

# Single version
class ExplicitEncoder_Single(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gaussians_t0, gaussians_t1):
        # TODO: make this work for batch size > 1
        # Extract control point positions and rotations at t=0 and t=1.
        p0 = gaussians_t0.xyz_cp  # [num_objects, 3]
        q0 = gaussians_t0.rot_cp  # [num_objects, 4]
        p1 = gaussians_t1.xyz_cp  # [num_objects, 3]
        q1 = gaussians_t1.rot_cp  # [num_objects, 4]
        
        # Compute linear velocity as difference (assuming dt = 1)
        v = p1 - p0  # [num_objects, 3]
        
        # Compute the delta quaternion: delta_q = rot1 * inverse(q0)
        inv_q0 = quaternion_inverse(q0)  # [num_objects, 4]
        delta_q = quaternion_multiply(q1, inv_q0)  # [num_objects, 4]
        
        # Convert the delta quaternion into an angular velocity (assuming dt = 1)
        omega = quat_delta_to_angular_velocity(delta_q, dt=1.0)  # [num_objects, 3]
        
        # For timestep 0, we set velocity and angular velocity to zeros.
        z0_objects = torch.cat([
            p0,          # position: [num_objects, 3]
            q0,          # rotation: [num_objects, 4]
            v,    # velocity: [num_objects, 3]
            omega # angular velocity: [num_objects, 3]
        ], dim=-1)  # [num_objects, 13]
        
        return z0_objects # [B, T, N, D], where D: (xyz, rot, vel, omega) 
    
# Batch version
class ExplicitEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gaussians_t0, gaussians_t1):
        # Extract control point positions and rotations at t=0 and t=1.
        batch_states = []
        # Assume the two lists have the same length (batch size B)
        for gm0, gm1 in zip(gaussians_t0, gaussians_t1):
            # Each gm0 and gm1 represent one scene (with N control points)
            p0 = gm0.xyz_cp  # [N, 3]
            q0 = gm0.rot_cp  # [N, 4]
            p1 = gm1.xyz_cp  # [N, 3]
            q1 = gm1.rot_cp  # [N, 4]
            
            # Compute linear velocity (assuming dt = 1)
            v = p1 - p0  # [N, 3]
            
            # Compute delta quaternion: delta_q = q1 * inverse(q0)
            inv_q0 = quaternion_inverse(q0)  # [N, 4]
            delta_q = quaternion_multiply(q1, inv_q0)  # [N, 4]
            
            # Convert delta quaternion to angular velocity (assuming dt = 1)
            omega = quat_delta_to_angular_velocity(delta_q, dt=1.0)  # [N, 3]
            
            # Concatenate to form state: [position, quaternion, velocity, angular velocity] => [N, 13]
            state = torch.cat([p0, q0, v, omega], dim=-1)  # [N, 13]


            # Add 1 static conditioning feature (object id) per control point.
            N = state.shape[0]
            object_ids = torch.arange(N, device=state.device, dtype=state.dtype).unsqueeze(-1)  # [N, 1]
            state = torch.cat([state, object_ids], dim=-1)  # [N, 14]
            



            batch_states.append(state)
        
        # Stack the states into a batch: [B, N, 13]
        batch_states = torch.stack(batch_states, dim=0)
        # Add a singleton time dimension (T=1): [B, 1, N, 13]
        z0_objects = batch_states#.unsqueeze(1)
        return z0_objects