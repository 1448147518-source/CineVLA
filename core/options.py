import tyro
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class Options:
    # ── Trajectory ──
    pose_length: int = 30              # trajectory frames
    pose_dim: int = 7                  # quat(4) + trans(3)
    dense_frames: int = 120            # SLERP interpolation output

    # ── Perception ──
    perception_dim: int = 512          # environment latent dimension
    image_size: int = 224              # perception encoder input size
    num_frames: int = 8               # input frame sequence length

    # ── Planner ──
    planner_hidden_dim: int = 256
    planner_num_layers: int = 6
    planner_num_heads: int = 4
    planner_text_ca_layers: int = 3    # text cross-attention in last N layers

    # ── Refiner ──
    refiner_hidden_dim: int = 256
    refiner_num_layers: int = 4
    refiner_num_heads: int = 4
    refiner_lookahead: int = 5         # refine next K frames per step

    # ── Music ──
    music_dim: int = 128
    music_seq_len: int = 30
    music_ca_layers: int = 2            # top N planner layers with music cross-attn

    # ── Training ──
    batch_size: int = 4
    lr: float = 1e-4
    num_epochs: int = 100
    planner_pretrain_epochs: int = 30
    refiner_pretrain_epochs: int = 20
    joint_epochs: int = 50
    grad_accum: int = 4
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    mixed_precision: str = 'bf16'
    seed: int = 42
    freeze_encoders: bool = True

    # ── Paths ──
    workspace: str = 'workspace'
    exp_name: str = 'default'
    resume: Optional[str] = None
    data_path: str = 'DataDoP/train'
    image_path: Optional[str] = None
    text: Optional[str] = None
    music_path: Optional[str] = None

    # ── Inference (closed-loop + CFG) ──
    closed_loop_steps: int = 30        # max closed-loop steps
    refinement_interval: int = 1       # refine every N steps
    cfg_scale: float = 2.0            # classifier-free guidance scale for text

    # ── Visualization ──
    vis_latent: bool = False           # enable latent state PCA visualization
    vis_latent_every: int = 100        # log latent state every N training steps


config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['default'] = 'default CineVLA v3 settings'
config_defaults['default'] = Options()

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
