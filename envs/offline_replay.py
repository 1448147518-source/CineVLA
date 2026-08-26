"""Offline replay environment for deterministic closed-loop integration tests.

This environment exposes observations one frame at a time and never reveals a
future frame to the policy.  It does *not* render from the submitted action:
the supplied pose is logged against an optional reference trajectory.  It is a
debugging and reproducibility baseline, not evidence of action-conditioned
closed-loop performance; use a renderer-backed environment for that claim.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from envs.base import CameraEnvironment, CameraObservation


class OfflineReplayEnv(CameraEnvironment):
    def __init__(self, frames: torch.Tensor,
                 reference_poses: Optional[torch.Tensor] = None):
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError('frames must have shape [T, 3, H, W]')
        if frames.shape[0] < 2:
            raise ValueError('offline replay requires at least two frames')
        if reference_poses is not None and reference_poses.shape[0] != frames.shape[0]:
            raise ValueError('reference_poses must align one-to-one with frames')
        self.frames = frames
        self.reference_poses = reference_poses
        self._index = 0

    def __len__(self) -> int:
        return self.frames.shape[0]

    @staticmethod
    def _resize(image: torch.Tensor, image_size: int) -> torch.Tensor:
        h, w = image.shape[-2:]
        if h > image_size:
            image = image[:, (h - image_size) // 2:(h - image_size) // 2 + image_size, :]
        if w > image_size:
            image = image[:, :, (w - image_size) // 2:(w - image_size) // 2 + image_size]
        if image.shape[-2:] != (image_size, image_size):
            padded = torch.zeros(3, image_size, image_size, dtype=image.dtype)
            hc, wc = image.shape[-2:]
            padded[:, (image_size - hc) // 2:(image_size - hc) // 2 + hc,
                   (image_size - wc) // 2:(image_size - wc) // 2 + wc] = image
            image = padded
        return image

    @classmethod
    def from_path(cls, path: str, image_size: int,
                  max_frames: Optional[int] = None) -> 'OfflineReplayEnv':
        """Create an episode from a frame directory or video without exposing it yet."""
        source = Path(path)
        if source.is_dir():
            files: Sequence[Path] = sorted(
                list(source.glob('*.png')) + list(source.glob('*.jpg'))
            )
            if len(files) < 2:
                raise RuntimeError(f'Frame directory {source} needs at least two PNG/JPG frames')
            if max_frames is not None and len(files) > max_frames:
                indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
                files = [files[i] for i in indices]
            raw_frames = []
            for frame_path in files:
                image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'Could not read frame: {frame_path}')
                raw_frames.append(image)
        elif source.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
            cap = cv2.VideoCapture(str(source))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2:
                cap.release()
                raise RuntimeError(f'Video {source} needs at least two frames')
            count = min(total, max_frames) if max_frames is not None else total
            indices = np.linspace(0, total - 1, count, dtype=int)
            raw_frames = []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, image = cap.read()
                if not ok:
                    cap.release()
                    raise RuntimeError(f'Could not read frame {index} from {source}')
                raw_frames.append(image)
            cap.release()
        else:
            raise RuntimeError('Provide a frame directory or video file (.mp4/.avi/.mov/.mkv)')

        tensors = []
        for image in raw_frames:
            rgb = torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
            tensors.append(cls._resize(rgb, image_size))
        return cls(torch.stack(tensors))

    def reset(self) -> CameraObservation:
        self._index = 0
        return CameraObservation(self.frames[0], step=0, info={'environment': 'offline_replay'})

    def step(self, camera_pose: torch.Tensor) -> Tuple[CameraObservation, bool]:
        if camera_pose.shape != (7,):
            raise ValueError(f'camera_pose must have shape [7], got {tuple(camera_pose.shape)}')
        if self._index >= len(self) - 1:
            raise RuntimeError('episode is already terminated; call reset()')
        self._index += 1
        info = {'environment': 'offline_replay'}
        if self.reference_poses is not None:
            reference = self.reference_poses[self._index]
            info['reference_pose'] = reference
            info['action_l1_to_reference'] = (camera_pose.detach().cpu() - reference.cpu()).abs().mean().item()
        terminated = self._index == len(self) - 1
        return CameraObservation(self.frames[self._index], step=self._index, info=info), terminated
