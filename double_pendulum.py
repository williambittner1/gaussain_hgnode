# double_pendulum.py
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import math
##########################################
# 1. Full Dynamics Model for a 2D Double Pendulum in Polar Coordinates
##########################################

class DoublePendulum2DPolarDynamics(nn.Module):
    """
    Dynamics for a planar (2D) double pendulum in polar coordinates.
    
    State: 
      x = [theta1, theta2, dot_theta1, dot_theta2] ∈ ℝ⁴,
    where theta1 and theta2 are the angles (from the vertical) of the first and second pendulum, respectively.
    
    The equations (for m1 = m2 = 1) are:
    
      ddtheta1 = [ -g*(2)*sin(theta1) - g*sin(theta1-2*theta2) - 2*sin(theta1-theta2) *
                    (dot_theta2^2 * L2 + dot_theta1^2 * L1*cos(theta1-theta2) )
                  ] / [ L1*(2 - cos(2*theta1-2*theta2)) ]
    
      ddtheta2 = [ 2*sin(theta1-theta2)*( dot_theta1^2 * L1 + g*cos(theta1) + dot_theta2^2 * L2*cos(theta1-theta2) )
                  ] / [ L2*(2 - cos(2*theta1-2*theta2)) ]
    
    (This is one common formulation.)
    """
    def __init__(self, L1=1.0, L2=1.0, g=9.81):
        super(DoublePendulum2DPolarDynamics, self).__init__()
        self.L1 = L1
        self.L2 = L2
        self.g = g

    def forward(self, t, x):
        # x: [batch, 4] where x = [theta1, theta2, dot_theta1, dot_theta2]
        theta1 = x[:, 0:1]
        theta2 = x[:, 1:2]
        dtheta1 = x[:, 2:3]
        dtheta2 = x[:, 3:4]
        
        # For brevity, denote delta = theta1 - theta2.
        delta = theta1 - theta2
        
        # Denominator common term:
        D = 2 - torch.cos(2*theta1 - 2*theta2) + 1e-6  # add epsilon to avoid division by zero
        
        # Equation for theta1:
        ddtheta1 = (
            - self.g * (2 * torch.sin(theta1) + torch.sin(theta1 - 2 * theta2))
            - 2 * torch.sin(delta) * (dtheta2**2 * self.L2 + dtheta1**2 * self.L1 * torch.cos(delta))
        ) / (self.L1 * D)
        
        # Equation for theta2:
        ddtheta2 = (
            2 * torch.sin(delta) * (dtheta1**2 * self.L1 + self.g * torch.cos(theta1) + dtheta2**2 * self.L2 * torch.cos(delta))
        ) / (self.L2 * D)
        
        # Concatenate derivatives to form the state derivative.
        dxdt = torch.cat([dtheta1, dtheta2, ddtheta1, ddtheta2], dim=1)
        return dxdt

##########################################
# 2. Initial Conditions and Time Spans (for 2D Model)
##########################################

def generate_initial_conditions_polar_2d(num_sequences, device="cpu", seed=1):
    """
    Generate initial conditions in polar coordinates for a planar double pendulum.
    
    State: [theta1, theta2, dot_theta1, dot_theta2] ∈ ℝ⁴.
    Here we use fixed nominal angles and zero initial angular velocities,
    with a small additive noise.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    theta1 = np.pi/4
    theta2 = np.pi/4
    base = torch.tensor([theta1, theta2, 0.0, 0.0],
                        device=device, dtype=torch.float32).unsqueeze(0)
    noise = np.pi * torch.randn(num_sequences, 4, device=device, dtype=torch.float32)
    x0 = base.repeat(num_sequences, 1) + noise
    return x0

def generate_time_spans(train_duration, test_duration, num_train_samples, num_test_samples, device="cpu"):
    t_train = torch.linspace(0., train_duration, num_train_samples, device=device, dtype=torch.float32)
    t_test  = torch.linspace(0., test_duration, num_test_samples, device=device, dtype=torch.float32)
    return t_train, t_test

##########################################
# 3. Conversion from 2D Polar State to 3D Cartesian Positions
##########################################

def polar_to_cartesian_2d(x, L1=1.0, L2=1.0):
    """
    Convert a 2D polar state trajectory to 3D Cartesian positions.
    
    Input:
      x: [batch, time_steps, 4] where:
           x[..., 0] = theta1, x[..., 1] = theta2.
           (Velocities are ignored.)
           
    We assume the motion is planar (in the vertical plane) with a fixed azimuth φ = π/2.
    Then:
      p1 = [0, L1*sin(theta1), -L1*cos(theta1)]
      p2 = p1 + [0, L2*sin(theta2), -L2*cos(theta2)]
      
    An anchor point is fixed at [0, 0, 0].
    
    Returns:
      positions: [batch, time_steps, 3, 3], where the three points are:
                 anchor, mass1 (p1), and mass2 (p2).
    """
    batch, time_steps, _ = x.shape
    theta1 = x[..., 0]
    theta2 = x[..., 1]
    
    # For φ = π/2: cos(π/2)=0 and sin(π/2)=1.
    # Thus, p1 = [0, L1*sin(theta1), -L1*cos(theta1)].
    p1_x = torch.zeros_like(theta1)
    p1_y = L1 * torch.sin(theta1)
    p1_z = - L1 * torch.cos(theta1)
    p1 = torch.stack([p1_x, p1_y, p1_z], dim=-1)  # [batch, time_steps, 3]
    
    # p2 = p1 + [0, L2*sin(theta2), -L2*cos(theta2)]
    dp2_x = torch.zeros_like(theta2)
    dp2_y = L2 * torch.sin(theta2)
    dp2_z = - L2 * torch.cos(theta2)
    dp2 = torch.stack([dp2_x, dp2_y, dp2_z], dim=-1)
    
    p2 = p1 + dp2
    
    # Create an anchor point at [0, 0, 0] for each batch and time step.
    anchor = torch.zeros(batch, time_steps, 1, 3, device=x.device, dtype=x.dtype)
    p1 = p1.unsqueeze(2)  # [batch, time_steps, 1, 3]
    p2 = p2.unsqueeze(2)
    
    positions = torch.cat([anchor, p1, p2], dim=2)  # [batch, time_steps, 3, 3]
    return positions


def polar_to_cartesian_2d_extended(x, L1=1.0, L2=1.0):
    """
    Convert a 2D polar state trajectory to 3D Cartesian positions and velocities.
    
    Input:
      x: [batch, time_steps, 4] where:
           x[..., 0] = theta1, x[..., 1] = theta2,
           x[..., 2] = dot_theta1, x[..., 3] = dot_theta2.
           (Velocities in the polar state are the angular velocities.)
           
    We assume the motion is planar (vertical plane) with a fixed azimuth φ = π/2.
    Then for mass 1:
         p1 = [0, L1*sin(theta1), -L1*cos(theta1)]
         v1 = [0, L1*cos(theta1)*dot_theta1, L1*sin(theta1)*dot_theta1]
    For mass 2:
         p2 = p1 + [0, L2*sin(theta2), -L2*cos(theta2)]
         v2 = v1 + [0, L2*cos(theta2)*dot_theta2, L2*sin(theta2)*dot_theta2]
         
    The anchor is fixed at [0, 0, 0] with zero velocity.
    
    Returns:
      positions: Tensor of shape [batch, time_steps, num_points, features],
                 where num_points = 3 (anchor, mass1, mass2) and features = 6
                 (first 3 are position, last 3 are velocity).
    """
    batch, T, _ = x.shape
    theta1 = x[..., 0]
    theta2 = x[..., 1]
    dtheta1 = x[..., 2]
    dtheta2 = x[..., 3]
    
    # For fixed azimuth φ = π/2:
    # Mass 1:
    p1_x = torch.zeros_like(theta1)
    p1_y = L1 * torch.sin(theta1)
    p1_z = -L1 * torch.cos(theta1)
    p1 = torch.stack([p1_x, p1_y, p1_z], dim=-1)  # [batch, T, 3]
    
    # Derivative of p1 with respect to time:
    v1_x = torch.zeros_like(theta1)
    v1_y = L1 * torch.cos(theta1) * dtheta1
    v1_z = L1 * torch.sin(theta1) * dtheta1
    v1 = torch.stack([v1_x, v1_y, v1_z], dim=-1)  # [batch, T, 3]
    
    # Mass 2:
    dp2_x = torch.zeros_like(theta2)
    dp2_y = L2 * torch.sin(theta2)
    dp2_z = -L2 * torch.cos(theta2)
    dp2 = torch.stack([dp2_x, dp2_y, dp2_z], dim=-1)  # [batch, T, 3]
    
    p2 = p1 + dp2  # [batch, T, 3]
    
    # Relative velocity for mass 2:
    dv2_x = torch.zeros_like(theta2)
    dv2_y = L2 * torch.cos(theta2) * dtheta2
    dv2_z = L2 * torch.sin(theta2) * dtheta2
    dv2 = torch.stack([dv2_x, dv2_y, dv2_z], dim=-1)  # [batch, T, 3]
    
    v2 = v1 + dv2  # [batch, T, 3]
    
    # Anchor: fixed at [0,0,0] with zero velocity.
    anchor_pos = torch.zeros(batch, T, 3, device=x.device, dtype=x.dtype)
    anchor_vel = torch.zeros(batch, T, 3, device=x.device, dtype=x.dtype)
    anchor = torch.cat([anchor_pos, anchor_vel], dim=-1)  # [batch, T, 6]
    
    # For mass 1 and mass 2, concatenate position and velocity.
    mass1 = torch.cat([p1, v1], dim=-1)  # [batch, T, 6]
    mass2 = torch.cat([p2, v2], dim=-1)  # [batch, T, 6]
    
    # Stack the three points along a new dimension.
    positions = torch.stack([anchor, mass1, mass2], dim=2)  # [batch, T, 3, 6]
    return positions

def polar_to_13d(x, L1=1.0, L2=1.0):
    """
    Convert a 2D polar state trajectory to 3D Cartesian positions, velocities, quaternions,
    and angular velocities.
    
    Input:
      x: [batch, time_steps, 4] where:
           x[..., 0] = theta1, x[..., 1] = theta2,
           x[..., 2] = dot_theta1, x[..., 3] = dot_theta2.
           
    We assume the motion is planar (vertical plane) with a fixed azimuth φ = π/2.
    Then:
      For mass 1:
         p1 = [0, L1*sin(theta1), -L1*cos(theta1)]
         v1 = [0, L1*cos(theta1)*dot_theta1, L1*sin(theta1)*dot_theta1]
         q1 = [cos(theta1/2), sin(theta1/2), 0, 0]   (rotation about x-axis)
         ω₁ = [dot_theta1, 0, 0]
         
      For mass 2:
         p2 = p1 + [0, L2*sin(theta2), -L2*cos(theta2)]
         v2 = v1 + [0, L2*cos(theta2)*dot_theta2, L2*sin(theta2)*dot_theta2]
         q2 = [cos(theta2/2), sin(theta2/2), 0, 0]
         ω₂ = [dot_theta2, 0, 0]
         
      For the anchor:
         p_anchor = [0, 0, 0]
         v_anchor = [0, 0, 0]
         q_anchor = [1, 0, 0, 0]
         ω_anchor = [0, 0, 0]
    
    Returns:
      out: Tensor of shape [batch, time_steps, 3, 13],
           where 13 = 3 (position) + 3 (velocity) + 4 (quaternion) + 3 (angular velocity).
    """
    batch, T, _ = x.shape
    theta1 = x[..., 0]
    theta2 = x[..., 1]
    dtheta1 = x[..., 2]
    dtheta2 = x[..., 3]
    
    # For fixed azimuth φ = π/2:
    # Mass 1:
    p1_x = torch.zeros_like(theta1)
    p1_y = L1 * torch.sin(theta1)
    p1_z = -L1 * torch.cos(theta1)
    p1 = torch.stack([p1_x, p1_y, p1_z], dim=-1)  # [batch, T, 3]
    
    # Derivative of p1:
    v1_x = torch.zeros_like(theta1)
    v1_y = L1 * torch.cos(theta1) * dtheta1
    v1_z = L1 * torch.sin(theta1) * dtheta1
    v1 = torch.stack([v1_x, v1_y, v1_z], dim=-1)  # [batch, T, 3]
    
    # Quaternion for mass 1: rotation about x by theta1.
    q1_w = torch.cos(theta1 / 2)
    q1_x = torch.sin(theta1 / 2)
    q1_y = torch.zeros_like(theta1)
    q1_z = torch.zeros_like(theta1)
    q1 = torch.stack([q1_w, q1_x, q1_y, q1_z], dim=-1)  # [batch, T, 4]
    
    # Angular velocity for mass 1:
    omega1 = torch.stack([dtheta1, torch.zeros_like(dtheta1), torch.zeros_like(dtheta1)], dim=-1)  # [batch, T, 3]
    
    # Mass 2:
    dp2_x = torch.zeros_like(theta2)
    dp2_y = L2 * torch.sin(theta2)
    dp2_z = -L2 * torch.cos(theta2)
    dp2 = torch.stack([dp2_x, dp2_y, dp2_z], dim=-1)  # [batch, T, 3]
    
    p2 = p1 + dp2  # [batch, T, 3]
    
    # Derivative for mass 2:
    dv2_x = torch.zeros_like(theta2)
    dv2_y = L2 * torch.cos(theta2) * dtheta2
    dv2_z = L2 * torch.sin(theta2) * dtheta2
    dv2 = torch.stack([dv2_x, dv2_y, dv2_z], dim=-1)  # [batch, T, 3]
    
    v2 = v1 + dv2  # [batch, T, 3]
    
    # Quaternion for mass 2: rotation about x by theta2.
    q2_w = torch.cos(theta2 / 2)
    q2_x = torch.sin(theta2 / 2)
    q2_y = torch.zeros_like(theta2)
    q2_z = torch.zeros_like(theta2)
    q2 = torch.stack([q2_w, q2_x, q2_y, q2_z], dim=-1)  # [batch, T, 4]
    
    # Angular velocity for mass 2:
    omega2 = torch.stack([dtheta2, torch.zeros_like(dtheta2), torch.zeros_like(dtheta2)], dim=-1)  # [batch, T, 3]
    
    # Anchor: fixed at [0,0,0] with zero velocity, identity quaternion, and zero angular velocity.
    anchor_pos = torch.zeros(batch, T, 3, device=x.device, dtype=x.dtype)
    anchor_vel = torch.zeros(batch, T, 3, device=x.device, dtype=x.dtype)
    anchor_q = torch.zeros(batch, T, 4, device=x.device, dtype=x.dtype)
    anchor_q[:, :, 0] = 1.0  # identity quaternion [1, 0, 0, 0]
    anchor_omega = torch.zeros(batch, T, 3, device=x.device, dtype=x.dtype)
    
    # Concatenate for each:
    mass1 = torch.cat([p1, v1, q1, omega1], dim=-1)  # [batch, T, 13]
    mass2 = torch.cat([p2, v2, q2, omega2], dim=-1)  # [batch, T, 13]
    anchor = torch.cat([anchor_pos, anchor_vel, anchor_q, anchor_omega], dim=-1)  # [batch, T, 13]
    
    # Stack them along a new dimension for the three points (anchor, mass1, mass2)
    out = torch.stack([anchor, mass1, mass2], dim=2)  # [batch, T, 3, 13]
    return out



def polar_to_10d(x, L1=1.0, L2=1.0):
    """
    Convert a 2D polar state trajectory to 3D Cartesian positions, velocities, and quaternions.
    
    Input:
      x: [batch, time_steps, 4] where:
           x[..., 0] = theta1, x[..., 1] = theta2,
           x[..., 2] = dot_theta1, x[..., 3] = dot_theta2.
           
    We assume the motion is planar (vertical plane) with a fixed azimuth φ = π/2.
    Then:
      p1 = [0, L1*sin(theta1), -L1*cos(theta1)]
      v1 = [0, L1*cos(theta1)*dot_theta1, L1*sin(theta1)*dot_theta1]
      
      p2 = p1 + [0, L2*sin(theta2), -L2*cos(theta2)]
      v2 = v1 + [0, L2*cos(theta2)*dot_theta2, L2*sin(theta2)*dot_theta2]
      
    We set the quaternion for a rotation about the x-axis by angle theta as:
      q = [cos(theta/2), sin(theta/2), 0, 0].
    For the anchor we use the identity quaternion [1,0,0,0].
    
    Returns:
      out: Tensor of shape [batch, time_steps, num_points, 10],
           with num_points = 3 (anchor, mass1, mass2) and features = 10:
           [position (3), velocity (3), quaternion (4)].
    """
    batch, T, _ = x.shape
    theta1 = x[..., 0]
    theta2 = x[..., 1]
    dtheta1 = x[..., 2]
    dtheta2 = x[..., 3]
    
    # For fixed azimuth φ = π/2:
    # Mass 1:
    p1_x = torch.zeros_like(theta1)
    p1_y = L1 * torch.sin(theta1)
    p1_z = - L1 * torch.cos(theta1)
    p1 = torch.stack([p1_x, p1_y, p1_z], dim=-1)  # [batch, T, 3]
    
    # Derivative of p1:
    v1_x = torch.zeros_like(theta1)
    v1_y = L1 * torch.cos(theta1) * dtheta1
    v1_z = L1 * torch.sin(theta1) * dtheta1
    v1 = torch.stack([v1_x, v1_y, v1_z], dim=-1)  # [batch, T, 3]
    
    # Quaternion for mass 1: rotation about x by theta1.
    q1_w = torch.cos(theta1 / 2)
    q1_x = torch.sin(theta1 / 2)
    q1_y = torch.zeros_like(theta1)
    q1_z = torch.zeros_like(theta1)
    q1 = torch.stack([q1_w, q1_x, q1_y, q1_z], dim=-1)  # [batch, T, 4]
    
    # Mass 2:
    dp2_x = torch.zeros_like(theta2)
    dp2_y = L2 * torch.sin(theta2)
    dp2_z = - L2 * torch.cos(theta2)
    dp2 = torch.stack([dp2_x, dp2_y, dp2_z], dim=-1)  # [batch, T, 3]
    
    p2 = p1 + dp2  # [batch, T, 3]
    
    # Derivative for mass 2:
    dv2_x = torch.zeros_like(theta2)
    dv2_y = L2 * torch.cos(theta2) * dtheta2
    dv2_z = L2 * torch.sin(theta2) * dtheta2
    dv2 = torch.stack([dv2_x, dv2_y, dv2_z], dim=-1)  # [batch, T, 3]
    
    v2 = v1 + dv2  # [batch, T, 3]
    
    # Quaternion for mass 2: rotation about x by theta2.
    q2_w = torch.cos(theta2 / 2)
    q2_x = torch.sin(theta2 / 2)
    q2_y = torch.zeros_like(theta2)
    q2_z = torch.zeros_like(theta2)
    q2 = torch.stack([q2_w, q2_x, q2_y, q2_z], dim=-1)  # [batch, T, 4]
    
    # Anchor: fixed at [0, 0, 0] with zero velocity and identity quaternion.
    anchor_pos = torch.zeros(batch, T, 3, device=x.device, dtype=x.dtype)
    anchor_vel = torch.zeros(batch, T, 3, device=x.device, dtype=x.dtype)
    anchor_q = torch.zeros(batch, T, 4, device=x.device, dtype=x.dtype)
    anchor_q[:, :, 0] = 1.0  # identity quaternion [1, 0, 0, 0]
    anchor = torch.cat([anchor_pos, anchor_vel, anchor_q], dim=-1)  # [batch, T, 10]
    
    mass1 = torch.cat([p1, v1, q1], dim=-1)  # [batch, T, 10]
    mass2 = torch.cat([p2, v2, q2], dim=-1)  # [batch, T, 10]
    
    # Stack them: output shape: [batch, T, 3, 10]
    out = torch.stack([anchor, mass1, mass2], dim=2)
    return out

##########################################
# 4. Trajectory Generation for 2D Model
##########################################

def generate_trajectory_2d(dynamics, x0, t_span):
    """
    Simulate the 2D double pendulum dynamics using odeint.
    
    Input:
      dynamics: an instance of DoublePendulum2DPolarDynamics,
      x0: initial state [batch, 4],
      t_span: time vector [num_timesteps]
    
    Returns:
      trajectory: raw state trajectory with shape [batch, num_timesteps, 4].
    """
    with torch.no_grad():
        raw = odeint(dynamics, x0, t_span, method="dopri5", rtol=1e-6, atol=1e-8)
        raw = raw.permute(1, 0, 2)
    return raw





##########################################
# 4.5 dense from sparse
##########################################



def quaternion_rotate_vector(q, v):
    """
    Rotate vector v by quaternion q.
    
    Args:
        q: shape [..., 4], each is [w, x, y, z] (assumed normalized).
        v: shape [..., 3].
    
    Returns:
        rotated v of shape [..., 3].
    """
    # Represent v as pure quaternion [0, v].
    zero = torch.zeros_like(v[..., :1])
    v_quat = torch.cat([zero, v], dim=-1)
    
    def quat_mul(a, b):
        # Multiply quaternions a and b, each shape [..., 4].
        w1, x1, y1, z1 = a.unbind(dim=-1)
        w2, x2, y2, z2 = b.unbind(dim=-1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=-1)
    
    q_conj = q.clone()
    q_conj[..., 1:] = -q[..., 1:]  # conjugate
    
    tmp = quat_mul(q, v_quat)
    rotated = quat_mul(tmp, q_conj)
    return rotated[..., 1:]  # drop the w component




def fibonacci_sphere_offsets(num_points, radius, device):
    """
    Generate `num_points` offsets on a sphere using the Fibonacci sphere sampling method.
    
    Args:
        num_points (int): Number of points to sample.
        radius (float): Sphere radius (scales the offsets).
        device: Torch device to create the tensor on.
        
    Returns:
        offsets: Tensor of shape [num_points, 3] with the sampled 3D offsets.
    """
    # Create indices from 0.5 to num_points - 0.5 to avoid clustering at the poles.
    indices = torch.arange(num_points, dtype=torch.float32, device=device) + 0.5
    
    # Golden angle in radians.
    golden_angle = math.pi * (3. - math.sqrt(5.))
    
    # Compute the y coordinate (from 1 to -1).
    y = 1 - 2 * indices / num_points  # y goes from nearly 1 to nearly -1.
    
    # Radius at each y (the projection on the XZ plane).
    r = torch.sqrt(1 - y ** 2)
    
    # Compute the azimuthal angle.
    theta = golden_angle * indices
    
    # Compute the x and z coordinates.
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    
    # Stack and scale the points.
    offsets = torch.stack([x, y, z], dim=-1) * radius  # shape: [num_points, 3]
    return offsets


def create_initial_dense_offsets(sparse_traj, num_scattered, radius):
    """
    For each node (in the sparse trajectory), create a set of local offsets using the
    Fibonacci sphere sampling method.
    
    Args:
        sparse_traj: Tensor of shape [B, T, N, 10] (the 3 points each w/ position, velocity, and quaternion).
        num_scattered (int): How many scattered points per node.
        radius (float): The radius used for scattered points.
        
    Returns:
        local_offsets: Tensor of shape [B, N, num_scattered, 3] containing the local offsets.
    """
    B, T, N, feat = sparse_traj.shape
    device = sparse_traj.device

    # Use Fibonacci sphere sampling to get canonical offsets.
    canonical_offsets = fibonacci_sphere_offsets(num_scattered, radius, device)  # [num_scattered, 3]
    
    # Expand the offsets for each batch and node.
    canonical_offsets = canonical_offsets.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, num_scattered, 3]
    canonical_offsets = canonical_offsets.expand(B, N, num_scattered, 3).clone()

    # Get the node quaternions at time t=0.
    q0 = sparse_traj[:, 0, :, 6:10]  # shape: [B, N, 4]
    q0_expanded = q0.unsqueeze(2).expand(B, N, num_scattered, 4)
    
    # Rotate the canonical offsets by the inverse of the node's initial quaternion.
    # (Assuming the quaternions are normalized, their inverse is the conjugate.)
    q0_conj = q0_expanded.clone()
    q0_conj[..., 1:] = -q0_conj[..., 1:]
    
    # Use your existing quaternion_rotate_vector function.
    offset_local = quaternion_rotate_vector(q0_conj, canonical_offsets)
    return offset_local  # shape: [B, N, num_scattered, 3]




def apply_scattered_offsets13d(sparse_traj, offset_local):
    """
    Given a sparse trajectory [B, T, N, 13], and local offsets [B, N, num_scattered, 3] that define 
    how each scattered point is positioned in local coordinates,
    produce a new dense trajectory of shape [B, T, N, num_scattered, 13].
    
    For each time step t, each node i:
      p_dense(t) = p(t) + R(q(t)) * offset_local
      v_dense(t) = v(t)  (copied)
      q_dense(t) = q(t)  (copied)
      ω_dense(t) = ω(t)  (copied)
    
    Args:
        sparse_traj: Tensor of shape [B, T, N, 13], where each node has:
                     position (3), velocity (3), quaternion (4), and angular velocity (3).
        offset_local: Tensor of shape [B, N, num_scattered, 3].
    
    Returns:
        dense_traj: Tensor of shape [B, T, N, num_scattered, 13].
    """
    B, T, N, feat = sparse_traj.shape
    device = sparse_traj.device
    num_scattered = offset_local.shape[2]
    
    # Extract the features from the sparse trajectory.
    p = sparse_traj[..., :3]       # position: [B, T, N, 3]
    v = sparse_traj[..., 3:6]      # velocity: [B, T, N, 3]
    q = sparse_traj[..., 6:10]     # quaternion: [B, T, N, 4]
    omega = sparse_traj[..., 10:13]  # angular velocity: [B, T, N, 3]
    
    # Expand offset_local to the time dimension: [B, T, N, num_scattered, 3]
    offset_local = offset_local.unsqueeze(1).expand(B, T, N, num_scattered, 3)
    
    # Expand q so it can be used to rotate the offsets.
    q_expanded = q.unsqueeze(3).expand(B, T, N, num_scattered, 4)
    
    # Rotate the canonical offsets by the node's current quaternion.
    offset_rotated = quaternion_rotate_vector(q_expanded, offset_local)  # [B, T, N, num_scattered, 3]
    
    # Translate the rotated offsets by the node's position.
    p_expanded = p.unsqueeze(3)  # [B, T, N, 1, 3]
    p_dense = p_expanded + offset_rotated  # [B, T, N, num_scattered, 3]
    
    # Expand the remaining features so they can be concatenated.
    v_expanded = v.unsqueeze(3).expand(B, T, N, num_scattered, 3)
    q_expanded2 = q.unsqueeze(3).expand(B, T, N, num_scattered, 4)
    omega_expanded = omega.unsqueeze(3).expand(B, T, N, num_scattered, 3)
    
    # Concatenate all features along the last dimension.
    dense_features = torch.cat([p_dense, v_expanded, q_expanded2, omega_expanded], dim=-1)  # shape: [B, T, N, num_scattered, 13]
    return dense_features


def apply_scattered_offsets10d(sparse_traj, offset_local):
    """
    Given a sparse trajectory [B,T,N,10], and local offsets [B,N,num_scattered,3] that define 
    how each scattered point is positioned in local coords,
    produce a new dense trajectory of shape [B,T,N,num_scattered,10].
    
    For each time step t, each node i:
      p_dense(t) = p(t) + R(q(t)) * offset_local
      v_dense(t) = v(t)  (copied)
      q_dense(t) = q(t)  (copied)
    
    Args:
        sparse_traj: [B, T, N, 10], each node has p(3), v(3), q(4).
        offset_local: [B, N, num_scattered, 3]
    
    Returns:
        dense_traj: [B, T, N, num_scattered, 10]
    """
    B, T, N, feat = sparse_traj.shape
    device = sparse_traj.device
    num_scattered = offset_local.shape[2]
    
    # We'll do: p_dense(t) = p(t) + R(q(t))* offset_local
    # We'll replicate velocity & quaternion from the node.
    
    p = sparse_traj[..., :3]    # [B,T,N,3]
    v = sparse_traj[..., 3:6]   # [B,T,N,3]
    q = sparse_traj[..., 6:10]  # [B,T,N,4]
    
    # We'll expand offset_local for the T dimension:
    # shape => [B, T, N, num_scattered, 3]
    offset_local = offset_local.unsqueeze(1).expand(B, T, N, num_scattered, 3)
    
    # Expand q to match offsets: [B,T,N,1,4] => [B,T,N,num_scattered,4].
    q_expanded = q.unsqueeze(3).expand(B, T, N, num_scattered, 4)
    
    # Rotate offset_local by q(t)
    offset_rotated = quaternion_rotate_vector(q_expanded, offset_local)  # => [B,T,N,num_scattered,3]
    
    # Now p(t) has shape [B,T,N,3], so expand p(t) -> [B,T,N,1,3]
    p_expanded = p.unsqueeze(3)
    p_dense = p_expanded + offset_rotated  # => [B,T,N,num_scattered,3]
    
    # The velocity & quaternion for the dense points is the same as the node’s:
    v_expanded = v.unsqueeze(3).expand(B, T, N, num_scattered, 3)
    q_expanded2 = q.unsqueeze(3).expand(B, T, N, num_scattered, 4)
    
    # Combine:
    dense_features = torch.cat([p_dense, v_expanded, q_expanded2], dim=-1)  # => [B,T,N,num_scattered,10]
    return dense_features


def dense_ground_truth_from_sparse(sparse_traj, num_scattered_points=10, radius=0.1):
    """
    Full pipeline:
      1) Create local offsets at t=0 for each node (sparse).
      2) At each timestep, apply the node’s position & quaternion to produce the scattered dense points.
    
    Returns:
      dense_traj: shape [B,T,N,num_scattered_points,10]
    """
    # 1) build offset_local from t=0
    offset_local = create_initial_dense_offsets(sparse_traj, num_scattered_points, radius)
    # 2) apply the offset for every time step
    dense_traj = apply_scattered_offsets10d(sparse_traj, offset_local)
    return dense_traj



def generate_points_data_old(dynamics, num_sequences, t_span, num_scattered_points=20, radius=0.1, L1=1.0, L2=1.0, device='cuda', x0=None, seed=1):
    """
    Generate both sparse and dense point trajectories for the double pendulum.

    Args:
        dynamics: the dynamics model.
        num_sequences: Number of trajectory sequences to generate.
        t_span: Time span tensor for the trajectories.
        num_scattered_points: Number of scattered points around each pendulum point.
        radius: Radius for scattered points.
        L1, L2: Pendulum lengths.
        device: Device to place tensors on.
        x0 (optional): Initial conditions; if not provided, new ones are generated.

    Returns:
        points: Combined sparse and dense trajectories [B, T, N_total, features].
    """
    # If no initial conditions are provided, generate them.
    if x0 is None:
        x0 = generate_initial_conditions_polar_2d(num_sequences, device=device, seed=seed)
    # Step A: generate the sparse system trajectories
    traj_system = generate_trajectory_2d(dynamics, x0, t_span)
    sparse_points = polar_to_10d(traj_system, L1=L1, L2=L2)  # [B, T, N, features]

    # Step B: generate the dense ground truth
    dense_points = dense_ground_truth_from_sparse(sparse_points, 
                                                  num_scattered_points=num_scattered_points, 
                                                  radius=radius)
    # Flatten the scattered features
    B, T, N, num_scattered, features = dense_points.shape
    dense_points = dense_points.reshape(B, T, N*num_scattered, features)

    # Step C: concatenate the sparse and dense points
    # points = torch.cat([sparse_points, dense_points], dim=-2)
    # points = sparse_points.to(device)
    points = dense_points
    return points

def generate_points_data(dynamics, num_sequences, t_span, num_scattered_points=20, radius=0.1, L1=1.0, L2=1.0, device='cuda', x0=None, seed=1):
    """
    Generate both sparse and dense point trajectories for the double pendulum.
    The returned points tensor has shape (B, T, N_particles, features) where:
        features = [3d pos, 3d vel, 4d quat, 1d object id].
    
    For the 3 spheres (anchor, mass1, mass2) the object ids are 0, 1, and 2, respectively.
    
    Args:
        dynamics: the dynamics model.
        num_sequences: Number of trajectory sequences to generate.
        t_span: Time span tensor for the trajectories.
        num_scattered_points: Number of scattered points around each pendulum point.
        radius: Radius for scattered points.
        L1, L2: Pendulum lengths.
        device: Device to place tensors on.
        x0 (optional): Initial conditions; if not provided, new ones are generated.
        seed: Random seed.
    
    Returns:
        points: Tensor of shape (B, T, N_particles, 11), where 11 = 3 (position) + 3 (velocity) +
                4 (quaternion) + 1 (object id).
    """
    # If no initial conditions are provided, generate them.
    if x0 is None:
        x0 = generate_initial_conditions_polar_2d(num_sequences, device=device, seed=seed)
    # Step A: generate the sparse system trajectories
    traj_system = generate_trajectory_2d(dynamics, x0, t_span)
    # sparse_points has shape [B, T, 3, 10] (the 3 points for the pendulum: anchor, mass1, mass2)
    sparse_points = polar_to_10d(traj_system, L1=L1, L2=L2)
    
    # Step B: generate the dense ground truth from the sparse trajectory.
    # dense_points has shape [B, T, 3, num_scattered_points, 10]
    dense_points = dense_ground_truth_from_sparse(sparse_points, 
                                                  num_scattered_points=num_scattered_points, 
                                                  radius=radius)
    
    # Append the object id for each node.
    # The original sparse trajectory has 3 nodes (axis-2), with object ids 0, 1, and 2.
    B, T, N, num_scattered, feat = dense_points.shape  # feat is currently 10
    # Create a tensor of object ids for the 3 nodes:
    # Shape: [1, 1, N, 1, 1] with values [0, 1, 2] for the three nodes.
    object_ids = torch.arange(N, device=dense_points.device, dtype=dense_points.dtype).view(1, 1, N, 1, 1)
    # Expand to [B, T, N, num_scattered, 1]
    object_ids = object_ids.expand(B, T, N, num_scattered, 1)
    
    # Concatenate the object id as an extra feature channel.
    # Now each point will have 10+1 = 11 features.
    dense_points = torch.cat([dense_points, object_ids], dim=-1)
    
    # Flatten the nodes and scattered dimensions:
    # The result will have shape [B, T, N * num_scattered, 11]
    dense_points = dense_points.reshape(B, T, N * num_scattered, feat + 1)
    
    points = dense_points
    return points
##########################################
# 5. Main Routine (Example Usage)
##########################################

if __name__ == "__main__":
    device = torch.device("cpu")
    
    # Simulation parameters.
    num_sequences = 16
    train_duration = 10.0
    test_duration = 10.0
    num_train_samples = 1000
    num_test_samples = 1000
    L1 = 1.0
    L2 = 1.0
    g = 9.81

    # Create dynamics instance for the planar double pendulum.
    dynamics_2d = DoublePendulum2DPolarDynamics(L1=L1, L2=L2, g=g).to(device)
    
    # Generate initial conditions (state dimension = 4).
    x0_2d = generate_initial_conditions_polar_2d(num_sequences, device=device)
    
    # Generate time spans.
    t_train, t_test = generate_time_spans(train_duration, test_duration, num_train_samples, num_test_samples, device=device)
    
    # Simulate trajectories (raw state shape: [batch, num_train_samples, 4]).
    traj_train_2d = generate_trajectory_2d(dynamics_2d, x0_2d, t_train)
    traj_test_2d = generate_trajectory_2d(dynamics_2d, x0_2d, t_test)
    
    # Convert 2D polar trajectories to 3D Cartesian positions.
    positions_train_2d = polar_to_cartesian_2d(traj_train_2d, L1=L1, L2=L2)
    positions_test_2d = polar_to_cartesian_2d(traj_test_2d, L1=L1, L2=L2)
    
    print("Training positions shape (2D model):", positions_train_2d.shape)  # Expected: [batch, num_train_samples, 3, 3]
    print("Testing positions shape (2D model):", positions_test_2d.shape)
    
    # (Optional: Save or visualize these trajectories.)

