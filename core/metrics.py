"""
Standalone benchmark metrics — adapted from the CLaTr evaluation framework.

No lightning / torchmetrics / hydra dependency.  Uses numpy + scipy only.

Metrics:
  - FCD  (Frechet CLaTr Distance) — multivariate Gaussian distance
  - PRDC (Precision / Recall / Density / Coverage) — manifold metrics
  - CLaTr Score — trajectory-text cosine alignment
"""

import numpy as np
import torch
from scipy import linalg


# ── Helper: multivariate Gaussian stats ──

def _gaussian_stats(features: np.ndarray):
    """Return (mu, sigma) of a feature matrix [N, D]."""
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ── FCD ──

def compute_fcd(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    """
    Frechet CLaTr Distance between two multivariate Gaussians.

    d^2 = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 * sqrt(S1 @ S2))

    Parameters
    ----------
    feats_real : np.ndarray  shape [N, D]
    feats_fake : np.ndarray  shape [M, D]

    Returns
    -------
    float
    """
    mu1, sigma1 = _gaussian_stats(feats_real)
    mu2, sigma2 = _gaussian_stats(feats_fake)

    diff = mu1 - mu2
    a = np.dot(diff, diff)

    cov_prod = sigma1 @ sigma2
    # sqrtm returns (result, error_estimate); take real part
    sqrt_cov = linalg.sqrtm(cov_prod)
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    b = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_cov)
    return float(max(a + b, 0.0))


# ── PRDC ──

def compute_prdc(
    feats_real: np.ndarray,
    feats_fake: np.ndarray,
    nearest_k: int = 3,
) -> dict:
    """
    Precision, Recall, Density, Coverage between two manifolds.

    Uses Euclidean distance in feature space.

    Parameters
    ----------
    feats_real : np.ndarray  shape [N, D]
    feats_fake : np.ndarray  shape [M, D]
    nearest_k : int

    Returns
    -------
    dict with keys: precision, recall, density, coverage
    """
    # Pairwise distances
    dist_rr = _pairwise_euclidean(feats_real, feats_real)   # [N, N]
    dist_ff = _pairwise_euclidean(feats_fake, feats_fake)   # [M, M]
    dist_rf = _pairwise_euclidean(feats_real, feats_fake)   # [N, M]

    # k-th nearest neighbour radius for each real / fake point
    nn_r = _kth_values(dist_rr, k=nearest_k + 1)  # +1 because self is always closest
    nn_f = _kth_values(dist_ff, k=nearest_k + 1)

    # Precision: fraction of fake points within some real point's k-NN radius
    precision = float((dist_rf < nn_r[:, np.newaxis]).any(axis=0).mean())

    # Recall: fraction of real points within some fake point's k-NN radius
    recall = float((dist_rf < nn_f[np.newaxis, :]).any(axis=1).mean())

    # Density: avg number of real-sphere neighbours per fake point / k
    density = float(
        (dist_rf < nn_r[:, np.newaxis]).sum(axis=0).mean() / nearest_k
    )

    # Coverage: fraction of real points that have at least one fake neighbour
    coverage = float((dist_rf.min(axis=1) < nn_r).mean())

    return {
        'precision': precision,
        'recall': recall,
        'density': density,
        'coverage': coverage,
    }


def _pairwise_euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between two matrices. [N, D] vs [M, D] → [N, M]."""
    xx = (x ** 2).sum(axis=1, keepdims=True)       # [N, 1]
    yy = (y ** 2).sum(axis=1, keepdims=True).T     # [1, M]
    dist = xx + yy - 2.0 * (x @ y.T)
    return np.sqrt(np.maximum(dist, 0.0))


def _kth_values(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return k-th smallest value along last axis for each row."""
    return np.partition(dist_matrix, k - 1, axis=-1)[:, k - 1]


# ── CLaTr Score ──

def compute_clatr_score(
    traj_feats: np.ndarray,
    text_feats: np.ndarray,
) -> float:
    """
    Mean cosine similarity × 100 between trajectory and text features.

    Parameters
    ----------
    traj_feats : np.ndarray  shape [N, D]
    text_feats : np.ndarray  shape [N, D]

    Returns
    -------
    float  (clamped to ≥ 0)
    """
    # L2 normalize
    t = traj_feats / (np.linalg.norm(traj_feats, axis=-1, keepdims=True) + 1e-8)
    txt = text_feats / (np.linalg.norm(text_feats, axis=-1, keepdims=True) + 1e-8)
    score = 100.0 * (t * txt).sum(axis=-1).mean()
    return float(max(score, 0.0))


# ── Batch utilities ──

class MetricsAccumulator:
    """Accumulate features across a dataset, then compute all metrics."""

    def __init__(self, prdc_k: int = 3):
        self.prdc_k = prdc_k
        self.traj_gen: list = []      # generated trajectory features
        self.traj_ref: list = []      # reference (GT) trajectory features
        self.text_feats: list = []    # text features (one per sample)

    def add(self, traj_feat_gen, traj_feat_ref, text_feat):
        self.traj_gen.append(_to_numpy(traj_feat_gen))
        self.traj_ref.append(_to_numpy(traj_feat_ref))
        self.text_feats.append(_to_numpy(text_feat))

    def compute(self):
        gen = np.stack(self.traj_gen)
        ref = np.stack(self.traj_ref)
        txt = np.stack(self.text_feats)

        fcd = compute_fcd(ref, gen)
        prdc = compute_prdc(ref, gen, self.prdc_k)
        clatr = compute_clatr_score(gen, txt)

        return {
            'clatr/fcd': fcd,
            'clatr/precision': prdc['precision'],
            'clatr/recall': prdc['recall'],
            'clatr/density': prdc['density'],
            'clatr/coverage': prdc['coverage'],
            'clatr/clatr_score': clatr,
            'num_samples': len(gen),
        }


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)
