import tyro
from typing import List
from dataclasses import dataclass, field

@dataclass
class ModelParams:
    sh_degree: int = 3
    exp_name: str = "Monkey"
    data_path: str = f"data/{exp_name}"
    source_path: str = f"data/{exp_name}"
    output_path: str = f"output/{exp_name}"
    foundation_model: str = ""
    checkpoint_path: str = f"output/{exp_name}"
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = False
    data_device: str = "cuda"
    eval: bool = False
    speedup: bool = False
    render_items: List[str] = field(default_factory=lambda: ["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"])
    manual_gaussians_bool: bool = False

@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = True

@dataclass
class OptimizationParams:
    iterations: int = 30000
    total_timesteps: int = 40
    framerate: int = 25
    ode_iterations: int = 50000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    semantic_feature_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 1000
    densify_grad_threshold: float = 0.0002
    load_existing_gaussians: bool = True

@dataclass
class TrainingConfig:
    model: ModelParams = ModelParams()
    pipeline: PipelineParams = PipelineParams()
    optimization: OptimizationParams = OptimizationParams()

@tyro.conf.configure
class ConfigWrapper:
    config: TrainingConfig = TrainingConfig()