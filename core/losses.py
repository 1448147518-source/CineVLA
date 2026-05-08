"""
CineVLA v4 Loss Functions — geometrically grounded pose supervision.

Design principles (informed by π³, LingBot-Map, VGGT, Kendall & Cipolla):
  1. Decouple rotation (SO(3) geodesic) from translation (L1)
  2. Enforce pairwise relative-pose consistency in a causal sliding window
  3. Regularise second-order trajectory smoothness (acceleration penalty)
  4. Progressive trajectory-length curriculum for planner pretraining

References:
  π³:    arXiv:2507.13347  — pairwise relative pose + geodesic + Huber
  GCT:   arXiv:2604.14141  — c2w abs-pose + causal relative + progressive training
  VGGT:  arXiv:2503.11651  — explicit camera parameter supervision
  Kendall & Cipolla, CVPR 2017 — learned homoscedastic uncertainty weighting
"""

from typing import Optional

import torch
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# Primitive loss functions
# ═════════════════════════════════════════════════════════════════════════════

def geodesic_rotation_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic (angular) distance on SO(3) via unit quaternions.

        d_rot(q̂, q*) = 2 * arccos(|⟨q̂, q*⟩|)

    The absolute value handles the quaternion double-cover (q ≡ -q).
    Clamp protects against numerical drift outside [-1, 1].

    Args:
        q_pred: [..., 4]  predicted unit quaternion (w,x,y,z)
        q_gt:   [..., 4]  ground-truth unit quaternion (w,x,y,z)

    Returns:
        scalar tensor — mean geodesic distance in radians [0, π]
    """
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
    dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return 2 * torch.acos(dot).mean()


def l1_translation_loss(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """
    L1 translation error (element-wise mean over all dimensions).

        L_trans = mean(|t̂ - t*|)

    Args:
        t_pred: [..., 3]  predicted translation (scene-normalised)
        t_gt:   [..., 3]  ground-truth translation

    Returns:
        scalar tensor
    """
    return t_pred.sub(t_gt).abs().mean()


# ═════════════════════════════════════════════════════════════════════════════
# Quaternion algebra helpers
# ═════════════════════════════════════════════════════════════════════════════

def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions).  q: [..., 4] (w,x,y,z)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ∘ q2.  Both [..., 4] (w,x,y,z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
# Relative-pose consistency loss
# ═════════════════════════════════════════════════════════════════════════════

def relative_pose_loss(
    poses_pred: torch.Tensor,
    poses_gt:   torch.Tensor,
    window_size: int = 5,
    lambda_trans: float = 0.1,
) -> torch.Tensor:
    """
    Causal sliding-window pairwise relative-pose consistency.

    For every pair (i, j) with  1 ≤ i < N,  max(0, i-W) ≤ j < i :

        q_rel_pred = conj(q̂_j) ∘ q̂_i      q_rel_gt  = conj(q*_j) ∘ q*_i
        t_rel_pred = t̂_i − t̂_j             t_rel_gt  = t*_i − t*_j

        L_rel = avg over pairs [ d_rot(q_rel_pred, q_rel_gt)
                                + λ_t · L1(t_rel_pred, t_rel_gt) ]

    Translations are compared in the global (first-frame-normalised) frame.
    The window is causal (only observed frames), preventing look-ahead.

    Args:
        poses_pred: [B, N, 7]  predicted poses  (quat wxyz + trans xyz)
        poses_gt:   [B, N, 7]  ground-truth poses
        window_size: max frame distance for pair construction
        lambda_trans: weight of translation term inside the relative loss

    Returns:
        scalar tensor
    """
    B, N, _ = poses_pred.shape
    W = min(window_size, N - 1)
    if W < 1:
        return torch.tensor(0.0, device=poses_pred.device)

    q_pred = F.normalize(poses_pred[..., :4], dim=-1, eps=1e-8)
    q_gt   = F.normalize(poses_gt[..., :4],   dim=-1, eps=1e-8)
    t_pred = poses_pred[..., 4:7]
    t_gt   = poses_gt[..., 4:7]

    total_rot, total_trans, count = 0.0, 0.0, 0

    for i in range(1, N):
        for j in range(max(0, i - W), i):
            # Relative quaternion: q_{j→i} = conj(q_j) ∘ q_i
            q_rel_pred = _quat_multiply(_quat_conjugate(q_pred[:, j]), q_pred[:, i])
            q_rel_gt   = _quat_multiply(_quat_conjugate(q_gt[:, j]),   q_gt[:, i])

            dot = torch.abs(torch.sum(q_rel_pred * q_rel_gt, dim=-1))
            dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            total_rot += 2 * torch.acos(dot).sum()

            # Relative translation (global-frame difference)
            t_rel_pred = t_pred[:, i] - t_pred[:, j]
            t_rel_gt   = t_gt[:, i]   - t_gt[:, j]
            total_trans += t_rel_pred.sub(t_rel_gt).abs().sum()

            count += B

    if count == 0:
        return torch.tensor(0.0, device=poses_pred.device)

    return (total_rot + lambda_trans * total_trans) / count


# ═════════════════════════════════════════════════════════════════════════════
# Trajectory smoothness loss (second-order)
# ═════════════════════════════════════════════════════════════════════════════

def trajectory_smoothness_loss(
    poses: torch.Tensor,
    lambda_rot_smooth: float = 0.5,
) -> torch.Tensor:
    """
    Second-order smoothness — penalise acceleration (change in velocity).

    Translation:
        a_i = t_i − 2·t_{i−1} + t_{i−2}      (central 2nd-order finite diff)
        L_acc_trans = mean(|a_i|)

    Rotation:
        ω_i = d_rot(q_i, q_{i−1})             (angular speed, scalar rad/frame)
        L_acc_rot   = mean(|ω_i − ω_{i−1}|)   (angular acceleration)

    L_smooth = L_acc_trans + λ_rot_smooth · L_acc_rot

    L1 (not L2) is used so that occasional legitimate turns are not
    over-penalised — the penalty grows linearly, not quadratically.

    Args:
        poses: [B, N, 7]  trajectory (quat wxyz + trans xyz)

    Returns:
        scalar tensor
    """
    B, N = poses.shape[:2]
    if N < 3:
        return torch.tensor(0.0, device=poses.device)

    trans = poses[..., 4:7]

    # Translation acceleration (scaled to scene-normalised units)
    acc_trans = (trans[:, 2:] - 2 * trans[:, 1:-1] + trans[:, :-2]).abs().mean()

    # Rotation angular-acceleration
    quat = F.normalize(poses[..., :4], dim=-1, eps=1e-8)
    dot_omega = torch.abs(torch.sum(quat[:, 1:] * quat[:, :-1], dim=-1))
    dot_omega = dot_omega.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    omega = 2 * torch.acos(dot_omega)                          # [B, N-1]
    acc_rot = (omega[:, 1:] - omega[:, :-1]).abs().mean()

    return acc_trans + lambda_rot_smooth * acc_rot


# ═════════════════════════════════════════════════════════════════════════════
# Composite loss functions (used by training loop)
# ═════════════════════════════════════════════════════════════════════════════

def planner_loss(
    poses_pred: torch.Tensor,
    poses_gt:   torch.Tensor,
    effective_N: Optional[int] = None,
    lambda_rot:   float = 1.0,
    lambda_trans: float = 0.5,
    lambda_rel:   float = 0.05,
    lambda_smooth: float = 0.1,
    window_size:  int   = 5,
    lambda_rot_smooth: float = 0.5,
    lambda_rel_t: float = 0.1,
):
    """
    Planner composite loss.

        L_planner = λ_rot·L_rot + λ_trans·L_trans + λ_rel·L_rel + λ_smooth·L_smooth

    Args:
        poses_pred:  [B, N, 7]  predicted trajectory
        poses_gt:    [B, N, 7]  ground-truth trajectory
        effective_N: if set, only supervise the first effective_N frames
                     (used for progressive length curriculum)
        lambda_* :   loss component weights

    Returns:
        total_loss:  scalar tensor (for backward)
        components:  dict of detached scalars (for logging)
    """
    if effective_N is not None and effective_N < poses_pred.shape[1]:
        poses_pred = poses_pred[:, :effective_N, :]
        poses_gt   = poses_gt[:, :effective_N, :]

    q_pred = F.normalize(poses_pred[..., :4], dim=-1, eps=1e-8)
    q_gt   = F.normalize(poses_gt[..., :4],   dim=-1, eps=1e-8)
    t_pred = poses_pred[..., 4:7]
    t_gt   = poses_gt[..., 4:7]

    L_rot    = geodesic_rotation_loss(q_pred, q_gt)
    L_trans  = l1_translation_loss(t_pred, t_gt)
    L_rel    = relative_pose_loss(poses_pred, poses_gt, window_size, lambda_rel_t)
    L_smooth = trajectory_smoothness_loss(poses_pred, lambda_rot_smooth)

    total = lambda_rot * L_rot + lambda_trans * L_trans \
            + lambda_rel * L_rel + lambda_smooth * L_smooth

    with torch.no_grad():
        components = {
            'L_rot':    L_rot.detach(),
            'L_trans':  L_trans.detach(),
            'L_rel':    L_rel.detach(),
            'L_smooth': L_smooth.detach(),
        }

    return total, components


def refiner_loss(
    refined:       torch.Tensor,      # [B, K, 7]
    planned:       torch.Tensor,      # [B, K, 7]  (GT remaining trajectory)
    z_next:        torch.Tensor,      # [B, dim]
    z_real:        torch.Tensor,      # [B, dim]
    lambda_pose:   float = 1.0,
    lambda_z:      float = 0.1,
    lambda_rel:    float = 0.02,
    lambda_smooth: float = 0.05,
    window_size:   int   = 5,
    lambda_rot_smooth: float = 0.5,
    lambda_rel_t:  float = 0.1,
):
    """
    Refiner composite loss.

        L_refiner = λ_pose·L1(delta) + λ_z·MSE(z_next, z_real)
                    + λ_rel·L_rel(refined, planned) + λ_smooth·L_smooth(refined)

    L1 on the delta (|refined − planned|) encourages conservative corrections.
    L_rel and L_smooth on the refined chunk ensure the refiner does not break
    local geometric consistency when it does correct.

    Args:
        refined:  [B, K, 7]  refined trajectory chunk
        planned:  [B, K, 7]  GT trajectory chunk (input to refiner)
        z_next:   [B, dim]   predicted next-frame latent feature
        z_real:   [B, dim]   actual frame latent feature (from perception)
        lambda_*: loss component weights

    Returns:
        total_loss:  scalar tensor
        components:  dict of detached scalars
    """
    device = refined.device

    # Pose delta regularisation — L1 (more robust than old MSE)
    L_pose_delta = (refined - planned).abs().mean()

    # Feature prediction (unchanged from original)
    L_z_pred = F.mse_loss(z_next, z_real)

    # Geometric constraints on the refined chunk
    K = refined.shape[1]
    if K >= 2:
        L_rel_chunk = relative_pose_loss(
            refined, planned,
            window_size=min(window_size, K - 1),
            lambda_trans=lambda_rel_t,
        )
        L_smooth_chunk = trajectory_smoothness_loss(refined, lambda_rot_smooth)
    else:
        L_rel_chunk = torch.tensor(0.0, device=device)
        L_smooth_chunk = torch.tensor(0.0, device=device)

    total = (lambda_pose * L_pose_delta + lambda_z * L_z_pred
             + lambda_rel * L_rel_chunk + lambda_smooth * L_smooth_chunk)

    with torch.no_grad():
        components = {
            'loss_pose':   L_pose_delta.detach(),
            'loss_z':      L_z_pred.detach(),
            'loss_rel':    L_rel_chunk.detach(),
            'loss_smooth': L_smooth_chunk.detach(),
        }

    return total, components


# ═════════════════════════════════════════════════════════════════════════════
# Progressive trajectory-length curriculum
# ═════════════════════════════════════════════════════════════════════════════

def get_effective_pose_length(
    epoch: int,
    total_epochs: int,
    start_len: int = 10,
    full_len: int = 30,
    ramp_epochs: Optional[int] = None,
) -> int:
    """
    Linearly increase effective trajectory length during planner pretraining.

        N_eff(epoch) = start_len + ⌊(full_len - start_len) · min(1, epoch / ramp_epochs)⌋

    This lets the planner first learn short-range motion patterns,
    then gradually handle longer-range trajectory structure — analogous
    to LingBot-Map's 24→320 progressive view training.

    Args:
        epoch:        current epoch (0-indexed)
        total_epochs: total epochs for this stage
        start_len:    initial trajectory length (default 10)
        full_len:     final trajectory length (default 30 = pose_length)
        ramp_epochs:  number of epochs over which to ramp up
                      (default: 2/3 of total_epochs)

    Returns:
        effective_N (int)
    """
    if ramp_epochs is None:
        ramp_epochs = max(1, int(total_epochs * 0.67))
    r = min(1.0, epoch / max(1, ramp_epochs))
    return int(start_len + (full_len - start_len) * r)
