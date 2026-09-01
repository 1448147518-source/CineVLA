try:
    import tyro
except ImportError:
    tyro = None
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class Options:
    # ── Trajectory ──
    pose_length: int = 30
    pose_dim: int = 7
    dense_frames: int = 120

    # ── Perception ──
    perception_dim: int = 512
    image_size: int = 224
    num_frames: int = 8

    # ── Planner ──
    planner_hidden_dim: int = 256
    planner_num_layers: int = 6
    planner_num_heads: int = 4
    planner_text_ca_layers: int = 3

    # ── Refiner ──
    refiner_hidden_dim: int = 256
    refiner_num_layers: int = 4
    refiner_num_heads: int = 4
    refiner_lookahead: int = 5
    refiner_flow_steps: int = 4
    refiner_correction_min_scale: float = 0.25

    # ── Training ──
    batch_size: int = 4
    lr: float = 1e-4
    planner_pretrain_epochs: int = 30
    refiner_pretrain_epochs: int = 20
    joint_epochs: int = 50
    grad_accum: int = 4
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    mixed_precision: str = 'bf16'
    seed: int = 42
    num_workers: int = 2

    # ── Paths ──
    workspace: str = 'workspace'
    exp_name: str = 'default'
    resume: Optional[str] = None
    data_path: str = 'DataDoP/train'
    test_size: int = 16
    image_path: Optional[str] = None
    text: Optional[str] = None

    # ── Inference ──
    closed_loop_steps: int = 30
    cfg_scale: float = 2.0
    discrepancy_threshold: float = 0.01

    # ── Visualization ──
    vis_latent: bool = False
    vis_latent_every: int = 100

    # ── Loss weights ──
    lambda_rot: float = 1.0
    lambda_trans: float = 0.5
    lambda_rel: float = 0.05
    lambda_smooth: float = 0.1
    rel_window_size: int = 5
    lambda_rot_smooth: float = 0.5
    lambda_rel_t: float = 0.1

    lambda_pose_delta: float = 1.0
    lambda_z_pred: float = 0.1
    lambda_flow: float = 1.0
    lambda_rel_ref: float = 0.02
    lambda_smooth_ref: float = 0.05

    pose_start_len: int = 10
    curriculum_ramp_epochs: Optional[int] = None


config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['default'] = 'default CineVLA settings'
config_defaults['default'] = Options()

AllConfigs = (tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
              if tyro is not None else Options)
