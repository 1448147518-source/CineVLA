"""Adapter for action-conditioned renderers and physical camera bridges.

The adapter has no knowledge of Blender, Unity, Gaussian splatting, or a
hardware camera.  Integrations only need to supply ``render_fn(pose)`` which
returns the RGB image actually observed after executing that pose.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

from envs.base import CameraEnvironment, CameraObservation

Frame = Union[torch.Tensor, np.ndarray]
RenderFn = Callable[[torch.Tensor], Frame]


class RendererCameraEnv(CameraEnvironment):
    """Action-conditioned environment backed by a synchronous renderer callback."""

    def __init__(self, render_fn: RenderFn, initial_pose: torch.Tensor,
                 image_size: int, max_steps: int = 30):
        if initial_pose.shape != (7,):
            raise ValueError('initial_pose must have shape [7]')
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.render_fn = render_fn
        self.initial_pose = initial_pose.detach().cpu().float().clone()
        self.image_size = image_size
        self.max_steps = max_steps
        self._step = 0
        self._terminated = False

    def _to_rgb_tensor(self, frame: Frame) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame.copy())
        if not isinstance(frame, torch.Tensor):
            raise TypeError('render_fn must return a numpy array or torch tensor')
        if frame.ndim != 3:
            raise ValueError('rendered frame must have three dimensions')
        if frame.shape[0] == 3:
            rgb = frame
        elif frame.shape[-1] == 3:
            rgb = frame.permute(2, 0, 1)
        else:
            raise ValueError('rendered frame must be RGB with 3 channels')
        rgb = rgb.detach().cpu().float()
        if rgb.max() > 1:
            rgb = rgb / 255.0
        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = torch.nn.functional.interpolate(
                rgb.unsqueeze(0), size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
        return rgb.clamp(0, 1)

    def _render(self, pose: torch.Tensor) -> torch.Tensor:
        return self._to_rgb_tensor(self.render_fn(pose.detach().cpu().float()))

    def reset(self) -> CameraObservation:
        self._step = 0
        self._terminated = False
        return CameraObservation(
            rgb=self._render(self.initial_pose), step=0,
            info={'environment': 'renderer', 'executed_pose': self.initial_pose.clone()},
        )

    def step(self, camera_pose: torch.Tensor) -> Tuple[CameraObservation, bool]:
        if self._terminated:
            raise RuntimeError('episode is already terminated; call reset()')
        if camera_pose.shape != (7,):
            raise ValueError(f'camera_pose must have shape [7], got {tuple(camera_pose.shape)}')
        self._step += 1
        pose = camera_pose.detach().cpu().float()
        self._terminated = self._step >= self.max_steps
        return CameraObservation(
            rgb=self._render(pose), step=self._step,
            info={'environment': 'renderer', 'executed_pose': pose.clone()},
        ), self._terminated
