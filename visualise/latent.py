"""
Latent environment state visualization — PCA projection of ẑ_pred vs z_real.

Works in both training and inference phases.  Uses numpy SVD-based PCA so
no extra dependencies are required beyond numpy + matplotlib.

Saves to ./pred_latent/ (relative to project root).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


# ── NumPy-only PCA (avoids sklearn dependency) ──

class _SimplePCA:
    """PCA via SVD.  Fit on a data matrix, then transform."""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.n_components].copy()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xc = X - self.mean_
        return Xc @ self.components_.T


# ── Latent logger ──

class LatentLogger:
    """
    Collects (z_real, z_pred) pairs during training or inference,
    then generates PCA-based visualizations.

    Usage
    -----
    >>> logger = LatentLogger(save_dir='pred_latent')
    >>> logger.log(z_real, z_pred, step=0, phase='train')
    >>> logger.log(z_real, z_pred, step=1, phase='infer')
    >>> logger.finalize()               # produce PNG summaries
    >>> logger.reset()                  # start fresh for next epoch / run
    """

    def __init__(self, save_dir='pred_latent', n_pca_components=2,
                 max_log=2000):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.n_pca_components = n_pca_components
        self.max_log = max_log          # cap stored entries to avoid OOM

        self.pca = _SimplePCA(n_components=n_pca_components)

        # buffers — store as numpy arrays
        self.z_real_buf: list = []
        self.z_pred_buf: list = []
        self.errors: list = []
        self.steps: list = []
        self.phases: list = []          # 'train' | 'infer'

    # ── Logging ──

    def log(self, z_real, z_pred, step, phase='train'):
        """
        Record one (z_real, z_pred) pair.

        Parameters
        ----------
        z_real : torch.Tensor or np.ndarray  shape [D] or [B, D]
        z_pred : torch.Tensor or np.ndarray  shape [D] or [B, D]
        step : int       Global step index.
        phase : str      'train' or 'infer'.
        """
        # Convert
        zr = _to_numpy1d(z_real)
        zp = _to_numpy1d(z_pred)
        err = float(np.mean((zr - zp) ** 2))

        self.z_real_buf.append(zr)
        self.z_pred_buf.append(zp)
        self.errors.append(err)
        self.steps.append(step)
        self.phases.append(phase)

        # Keep buffer bounded
        if len(self.z_real_buf) > self.max_log:
            # drop oldest half
            keep = self.max_log // 2
            self.z_real_buf = self.z_real_buf[-keep:]
            self.z_pred_buf = self.z_pred_buf[-keep:]
            self.errors = self.errors[-keep:]
            self.steps = self.steps[-keep:]
            self.phases = self.phases[-keep:]

    # ── Visualization ──

    def finalize(self, max_pca_points=1000):
        """
        Generate two summary plots:
          1. latent_space.png   — PCA-2D scatter of z_real / z_pred
          2. latent_error.png   — MSE over time
        """
        n = len(self.z_real_buf)
        if n < 3:
            print(f"[visualise] LatentLogger: only {n} entries — skipping plots")
            return

        all_z = np.stack(self.z_real_buf + self.z_pred_buf)

        # Fit PCA (subsample if huge)
        if len(all_z) > max_pca_points:
            idx = np.linspace(0, len(all_z) - 1, max_pca_points, dtype=int)
            self.pca.fit(all_z[idx])
        else:
            self.pca.fit(all_z)

        zr_2d = self.pca.transform(np.stack(self.z_real_buf))
        zp_2d = self.pca.transform(np.stack(self.z_pred_buf))

        errors = np.array(self.errors)
        steps  = np.array(self.steps)
        phases = np.array(self.phases)

        train_mask = phases == 'train'
        infer_mask = phases == 'infer'

        # ── Figure 1: PCA latent space ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        for ax, mask, name, cmap_real, cmap_pred in [
            (axes[0], train_mask, 'Training', 'Blues', 'Reds'),
            (axes[1], infer_mask, 'Inference', 'Blues', 'Reds'),
        ]:
            self._plot_latent_2d(
                ax, zr_2d[mask], zp_2d[mask],
                steps[mask], errors[mask],
                name, cmap_real, cmap_pred,
            )

        fig.tight_layout()
        p1 = os.path.join(self.save_dir, 'latent_space.png')
        fig.savefig(p1, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[visualise] Latent space → {p1}")

        # ── Figure 2: error over time ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        for ax, mask, name in [
            (axes[0], train_mask, 'Training'),
            (axes[1], infer_mask, 'Inference'),
        ]:
            self._plot_error(ax, steps[mask], errors[mask], name)

        fig.tight_layout()
        p2 = os.path.join(self.save_dir, 'latent_error.png')
        fig.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[visualise] Latent error → {p2}")

    # ── Internal plotting ──

    @staticmethod
    def _plot_latent_2d(ax, real, pred, steps, errors, title,
                        cmap_real, cmap_pred):
        if len(real) == 0:
            ax.text(0.5, 0.5, f'(no {title.lower()} data)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='gray')
            ax.set_title(title, fontsize=12)
            return

        # Normalize step → [0, 1] for colormap
        norm_steps = steps.astype(np.float32)
        if norm_steps.max() > norm_steps.min():
            norm_steps = (norm_steps - norm_steps.min()) / (norm_steps.max() - norm_steps.min())
        else:
            norm_steps[:] = 0.5

        sc_r = ax.scatter(real[:, 0], real[:, 1], c=norm_steps,
                          cmap=cmap_real, s=25, alpha=0.7, marker='o',
                          edgecolors='none', label='z_real (actual)')
        sc_p = ax.scatter(pred[:, 0], pred[:, 1], c=norm_steps,
                          cmap=cmap_pred, s=25, alpha=0.7, marker='x',
                          linewidths=0.6, label='z_pred (predicted)')

        # Thin connector lines (max 80 pairs to avoid clutter)
        n_connect = min(len(real), 80)
        for i in range(n_connect):
            ax.plot([real[i, 0], pred[i, 0]], [real[i, 1], pred[i, 1]],
                    color='#bdc3c7', linewidth=0.3, alpha=0.45)

        ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
        ax.set_title(f'{title} — Latent States ({len(real)} steps)', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

    @staticmethod
    def _plot_error(ax, steps, errors, title):
        if len(steps) == 0:
            ax.text(0.5, 0.5, f'(no {title.lower()} data)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='gray')
            ax.set_title(title, fontsize=12)
            return

        ax.plot(steps, errors, color='#2980b9', linewidth=1.0, alpha=0.8)
        ax.scatter(steps, errors, c='#2980b9', s=8, alpha=0.4)
        ax.set_xlabel('Step'); ax.set_ylabel('MSE (z_real vs z_pred)')
        ax.set_title(f'{title} — Latent Prediction Error', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25)

    def reset(self):
        """Clear buffers — call between epochs or runs."""
        self.z_real_buf.clear()
        self.z_pred_buf.clear()
        self.errors.clear()
        self.steps.clear()
        self.phases.clear()


# ── Helpers ──

def _to_numpy1d(x):
    """Convert tensor/array to 1-D numpy float32.

    Batched tensors [B, D] are mean-pooled over B to preserve
    a representative feature vector rather than silently discarding
    all but the first sample.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > 1:
        x = x.mean(axis=0)
    return x
