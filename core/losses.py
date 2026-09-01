"""
CineVLA v4 Loss Functions — geometrically grounded pose supervision.

Design principles (informed by π³, LingBot-Map, VGGT, Kendall & Cipolla):
  1. Decouple rotation (SO(3) geodesic) from translation (L1)
  2. Enforce pairwise relative-pose consistency in a causal sliding window
  3. Regularise second-order trajectory smoothness (acceleration penalty)
  4. Progressive trajectory-length curriculum for planner pretraining
  5. Train Refiner residuals with conditional Flow Matching velocity supervision
"""

from typing import Optional

import torch
import torch.nn.functional as F

from core.quaternion import quat_conjugate, quat_multiply


def geodesic_rotation_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
    dot = dot.clamp(0.0, 1.0)
    return 2 * torch.acos(dot).mean()


def l1_translation_loss(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    return t_pred.sub(t_gt).abs().mean()


def relative_pose_loss(
    poses_pred: torch.Tensor,
    poses_gt: torch.Tensor,
    window_size: int = 5,
    lambda_trans: float = 0.1,
) -> torch.Tensor:
    B, N, _ = poses_pred.shape
    W = min(window_size, N - 1)
    if W < 1:
        return torch.tensor(0.0, device=poses_pred.device)

    q_pred = F.normalize(poses_pred[..., :4], dim=-1, eps=1e-8)
    q_gt = F.normalize(poses_gt[..., :4], dim=-1, eps=1e-8)
    t_pred = poses_pred[..., 4:7]
    t_gt = poses_gt[..., 4:7]

    total_rot, total_trans, count = 0.0, 0.0, 0
    for i in range(1, N):
        for j in range(max(0, i - W), i):
            q_rel_pred = quat_multiply(quat_conjugate(q_pred[:, j]), q_pred[:, i])
            q_rel_gt = quat_multiply(quat_conjugate(q_gt[:, j]), q_gt[:, i])
            dot = torch.abs(torch.sum(q_rel_pred * q_rel_gt, dim=-1)).clamp(0.0, 1.0)
            total_rot += 2 * torch.acos(dot).sum()

            t_rel_pred = t_pred[:, i] - t_pred[:, j]
            t_rel_gt = t_gt[:, i] - t_gt[:, j]
            total_trans += t_rel_pred.sub(t_rel_gt).abs().sum()
            count += B

    if count == 0:
        return torch.tensor(0.0, device=poses_pred.device)
    return (total_rot + lambda_trans * total_trans) / count


def trajectory_smoothness_loss(
    poses: torch.Tensor,
    lambda_rot_smooth: float = 0.5,
) -> torch.Tensor:
    B, N = poses.shape[:2]
    if N < 3:
        return torch.tensor(0.0, device=poses.device)

    trans = poses[..., 4:7]
    acc_trans = (trans[:, 2:] - 2 * trans[:, 1:-1] + trans[:, :-2]).abs().mean()

    quat = F.normalize(poses[..., :4], dim=-1, eps=1e-8)
    dot_omega = torch.abs(torch.sum(quat[:, 1:] * quat[:, :-1], dim=-1)).clamp(0.0, 1.0)
    omega = 2 * torch.acos(dot_omega)
    acc_rot = (omega[:, 1:] - omega[:, :-1]).abs().mean()
    return acc_trans + lambda_rot_smooth * acc_rot


def planner_loss(
    poses_pred: torch.Tensor,
    poses_gt: torch.Tensor,
    effective_N: Optional[int] = None,
    lambda_rot: float = 1.0,
    lambda_trans: float = 0.5,
    lambda_rel: float = 0.05,
    lambda_smooth: float = 0.1,
    window_size: int = 5,
    lambda_rot_smooth: float = 0.5,
    lambda_rel_t: float = 0.1,
):
    if effective_N is not None and effective_N < poses_pred.shape[1]:
        poses_pred = poses_pred[:, :effective_N, :]
        poses_gt = poses_gt[:, :effective_N, :]

    q_pred = F.normalize(poses_pred[..., :4], dim=-1, eps=1e-8)
    q_gt = F.normalize(poses_gt[..., :4], dim=-1, eps=1e-8)
    t_pred = poses_pred[..., 4:7]
    t_gt = poses_gt[..., 4:7]

    L_rot = geodesic_rotation_loss(q_pred, q_gt)
    L_trans = l1_translation_loss(t_pred, t_gt)
    L_rel = relative_pose_loss(poses_pred, poses_gt, window_size, lambda_rel_t)
    L_smooth = trajectory_smoothness_loss(poses_pred, lambda_rot_smooth)

    total = (lambda_rot * L_rot + lambda_trans * L_trans
             + lambda_rel * L_rel + lambda_smooth * L_smooth)
    with torch.no_grad():
        components = {
            'L_rot': L_rot.detach(),
            'L_trans': L_trans.detach(),
            'L_rel': L_rel.detach(),
            'L_smooth': L_smooth.detach(),
        }
    return total, components


def refiner_loss(
    refined: torch.Tensor,
    target_poses: torch.Tensor,
    z_next: torch.Tensor,
    z_next_target: torch.Tensor,
    flow_velocity: Optional[torch.Tensor] = None,
    flow_target: Optional[torch.Tensor] = None,
    lambda_pose: float = 1.0,
    lambda_z: float = 0.1,
    lambda_flow: float = 1.0,
    lambda_rel: float = 0.02,
    lambda_smooth: float = 0.05,
    window_size: int = 5,
    lambda_rot_smooth: float = 0.5,
    lambda_rel_t: float = 0.1,
):
    """Composite Refiner objective.

    Flow Matching is the primary residual-generation objective:
        x_t = (1-t) x_0 + t x_1,
        v*  = x_1 - x_0,
        L_flow = ||v_theta(x_t,t,c) - v*||^2.

    Geometric losses remain as auxiliary supervision on the one-step estimate
    so the generated residual respects camera pose geometry and smoothness.
    """
    device = refined.device

    L_pose_rot = geodesic_rotation_loss(
        F.normalize(refined[..., :4], dim=-1, eps=1e-8),
        F.normalize(target_poses[..., :4], dim=-1, eps=1e-8),
    )
    L_pose_trans = l1_translation_loss(refined[..., 4:7], target_poses[..., 4:7])
    L_pose_delta = L_pose_rot + L_pose_trans

    L_z_pred = F.mse_loss(z_next, z_next_target)

    if flow_velocity is not None and flow_target is not None:
        L_flow = F.mse_loss(flow_velocity, flow_target)
    else:
        L_flow = torch.tensor(0.0, device=device)

    K = refined.shape[1]
    if K >= 2:
        L_rel_chunk = relative_pose_loss(
            refined, target_poses,
            window_size=min(window_size, K - 1),
            lambda_trans=lambda_rel_t,
        )
        L_smooth_chunk = trajectory_smoothness_loss(refined, lambda_rot_smooth)
    else:
        L_rel_chunk = torch.tensor(0.0, device=device)
        L_smooth_chunk = torch.tensor(0.0, device=device)

    total = (
        lambda_pose * L_pose_delta
        + lambda_z * L_z_pred
        + lambda_flow * L_flow
        + lambda_rel * L_rel_chunk
        + lambda_smooth * L_smooth_chunk
    )

    with torch.no_grad():
        components = {
            'loss_pose': L_pose_delta.detach(),
            'loss_pose_rot': L_pose_rot.detach(),
            'loss_pose_trans': L_pose_trans.detach(),
            'loss_z': L_z_pred.detach(),
            'loss_flow': L_flow.detach(),
            'loss_rel': L_rel_chunk.detach(),
            'loss_smooth': L_smooth_chunk.detach(),
        }
    return total, components


def get_effective_pose_length(
    epoch: int,
    total_epochs: int,
    start_len: int = 10,
    full_len: int = 30,
    ramp_epochs: Optional[int] = None,
) -> int:
    if ramp_epochs is None:
        ramp_epochs = max(1, int(total_epochs * 0.67))
    r = min(1.0, epoch / max(1, ramp_epochs))
    return int(start_len + (full_len - start_len) * r)
