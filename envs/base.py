"""Interfaces for action-conditioned camera environments.

The policy is deliberately restricted to observations returned by this API.
This makes causal rollout evaluation possible with either an offline replay
source or a future renderer / physical camera adapter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch


@dataclass
class CameraObservation:
    """One RGB observation emitted after a camera action."""

    rgb: torch.Tensor                     # [3, H, W], float in [0, 1]
    step: int
    info: Dict[str, Any] = field(default_factory=dict)


class CameraEnvironment(ABC):
    """Minimal environment contract for receding-horizon camera control."""

    @abstractmethod
    def reset(self) -> CameraObservation:
        """Reset an episode and return the only initially observable frame."""

    @abstractmethod
    def step(self, camera_pose: torch.Tensor) -> Tuple[CameraObservation, bool]:
        """Execute a [7] camera pose and return (next_observation, terminated)."""
