"""
CineVLA evaluation suite — benchmark metrics from the CLaTr framework.

Provides:
  - TrajectoryEncoder : lightweight encoder that maps [N, 7] trajectories → feature vectors
  - eval_benchmark    : batch evaluation script producing FCD / PRDC / CLaTr Score
"""

from .trajectory_encoder import TrajectoryEncoder
from .eval_benchmark import run_benchmark
