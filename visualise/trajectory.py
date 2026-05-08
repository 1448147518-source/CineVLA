"""
Camera trajectory visualization — 3D path with camera pose coordinate frames.

Saves to ./results/ (relative to project root).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Quaternion utilities ──

def _q2r(q):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix (numpy version for matplotlib)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


# ── Plotting helpers ──

def _camera_frame(ax, pos, R, scale, alpha, label=None):
    """Draw camera coordinate frame: X=red, Y=green, Z=blue."""
    for axis_idx, color in enumerate(['#e74c3c', '#2ecc71', '#3498db']):
        end = pos + R[:, axis_idx] * scale
        ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                color=color, linewidth=1.2, alpha=alpha)
    if label:
        ax.text(pos[0], pos[1], pos[2], f'  {label}', fontsize=7, alpha=alpha)


def _set_axes_equal(ax):
    """Force 3D axes to equal scale so camera frames aren't distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = (limits[:, 1] - limits[:, 0]).max() / 2.0
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


# ── Main API ──

def plot_trajectory(
    trajectory,                  # [N, 7]  (quat + trans) — final refined trajectory
    dense=None,                  # [M, 3, 4] — SLERP-interpolated dense path
    steps=None,                  # list[dict] — per-step poses + errors from closed loop
    save_dir='results',
    title='CineVLA Camera Trajectory',
    frame_stride=None,           # show a camera frame every N poses (auto if None)
):
    """
    3D camera trajectory plot.

    Parameters
    ----------
    trajectory : np.ndarray  shape [N, 7]
        Final camera trajectory (quaternion wxyz + translation xyz).
    dense : np.ndarray or None  shape [M, 3, 4]
        Dense SLERP-interpolated trajectory for a smoother path line.
    steps : list[dict] or None
        Per-step dicts with keys 'step', 'pose', 'error' from the closed loop.
    save_dir : str
        Output directory (created if missing).
    title : str
        Figure title.
    frame_stride : int or None
        Draw a camera coordinate frame every N poses. Auto-computed if None.
    """
    os.makedirs(save_dir, exist_ok=True)

    trajectory = np.asarray(trajectory)
    N = trajectory.shape[0]
    positions = trajectory[:, 4:7]

    # ── 3D trajectory plot ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Path line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color='#2c3e50', linewidth=1.8, alpha=0.75, label='Camera path')

    # Dense SLERP path (thin, translucent)
    if dense is not None:
        dense = np.asarray(dense)
        dpos = dense[:, :3, 3]
        ax.plot(dpos[:, 0], dpos[:, 1], dpos[:, 2],
                color='#95a5a6', linewidth=0.5, alpha=0.35,
                linestyle='--', label='Dense (SLERP)')

    # Camera frames along the path
    if frame_stride is None:
        frame_stride = max(1, N // 8)
    max_dim = max(positions.ptp(axis=0)) + 1e-6
    frame_scale = max_dim * 0.04

    for i in range(0, N, frame_stride):
        q = trajectory[i, :4]
        pos = trajectory[i, 4:7]
        R = _q2r(q)
        alpha = 0.35 + 0.60 * (i / max(1, N - 1))
        _camera_frame(ax, pos, R, frame_scale, alpha, label=str(i))

    # Start / end markers
    ax.scatter(*positions[0],  c='#27ae60', s=120, marker='o',
               edgecolors='white', linewidth=1.5, zorder=10, label='Start')
    ax.scatter(*positions[-1], c='#e74c3c', s=120, marker='s',
               edgecolors='white', linewidth=1.5, zorder=10, label='End')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    _set_axes_equal(ax)

    fig.tight_layout()
    traj_path = os.path.join(save_dir, 'trajectory.png')
    fig.savefig(traj_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[visualise] Trajectory saved → {traj_path}")

    # ── Refinement error over time ──
    if steps and len(steps) > 0:
        _plot_refinement_error(steps, save_dir)


def _plot_refinement_error(steps, save_dir):
    """Line plot of per-step perception error during closed-loop refinement."""
    step_ix = np.array([s['step'] for s in steps])
    errors = np.array([s['error'] for s in steps])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(step_ix, errors, color='#e74c3c', marker='o', markersize=5,
            linewidth=1.5, alpha=0.85)
    ax.axhline(y=0.01, color='#7f8c8d', linestyle='--', linewidth=1,
               alpha=0.6, label='Refinement threshold (0.01)')
    ax.fill_between(step_ix, 0, errors, alpha=0.12, color='#e74c3c')
    ax.set_xlabel('Closed-loop step')
    ax.set_ylabel('Perception MSE')
    ax.set_title('Refinement error over closed-loop steps')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    err_path = os.path.join(save_dir, 'refinement_error.png')
    fig.savefig(err_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[visualise] Error plot saved → {err_path}")
