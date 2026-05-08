"""
Standalone benchmark metrics — adapted from the CLaTr evaluation framework.

No lightning / torchmetrics / hydra dependency.  Uses numpy + scipy only.

Metrics:
  - FCD  (Frechet CLaTr Distance) — multivariate Gaussian distance
  - PRDC (Precision / Recall / Density / Coverage) — manifold metrics
  - CLaTr Score — trajectory-text cosine alignment
  - Caption metrics — motion-pattern segmentation precision / recall / F1
"""

import numpy as np
import torch
from scipy import linalg
from scipy.stats import mode


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

        # Caption metrics — collect full trajectories in [N, 3, 4] format
        self.traj_gen_34: list = []
        self.traj_ref_34: list = []

    def add(self, traj_feat_gen, traj_feat_ref, text_feat,
            traj_34_gen=None, traj_34_ref=None):
        self.traj_gen.append(_to_numpy(traj_feat_gen))
        self.traj_ref.append(_to_numpy(traj_feat_ref))
        self.text_feats.append(_to_numpy(text_feat))
        if traj_34_gen is not None:
            self.traj_gen_34.append(_to_numpy_34(traj_34_gen))
        if traj_34_ref is not None:
            self.traj_ref_34.append(_to_numpy_34(traj_34_ref))

    def compute(self):
        gen = np.stack(self.traj_gen)
        ref = np.stack(self.traj_ref)
        txt = np.stack(self.text_feats)

        fcd = compute_fcd(ref, gen)
        prdc = compute_prdc(ref, gen, self.prdc_k)
        clatr = compute_clatr_score(gen, txt)

        results = {
            'clatr/fcd': fcd,
            'clatr/precision': prdc['precision'],
            'clatr/recall': prdc['recall'],
            'clatr/density': prdc['density'],
            'clatr/coverage': prdc['coverage'],
            'clatr/clatr_score': clatr,
            'num_samples': len(gen),
        }

        # Caption metrics (motion-pattern segmentation)
        if self.traj_gen_34 and self.traj_ref_34:
            cap = compute_caption_metrics(self.traj_gen_34, self.traj_ref_34)
            results['captions/precision'] = cap['precision']
            results['captions/recall'] = cap['recall']
            results['captions/fscore'] = cap['fscore']

        return results


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _to_numpy_34(x):
    """Convert trajectory to numpy [N, 3, 4]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# ── Caption metrics: trajectory motion-pattern segmentation ──

# Translation direction → class index (27 classes)
_TRANS_PATTERNS = {
    ( 0,  0,  0): 0,   ( 1,  0,  0): 1,   (-1,  0,  0): 2,
    ( 0,  1,  0): 3,   ( 0, -1,  0): 6,   ( 0,  0,  1): 9,
    ( 0,  0, -1): 18,
    # two-axis
    ( 1,  1,  0): 4,   ( 1, -1,  0): 5,   ( 1,  0,  1): 10,
    ( 1,  0, -1): 11,  (-1,  1,  0): 7,   (-1, -1,  0): 8,
    (-1,  0,  1): 19,  (-1,  0, -1): 20,  ( 0,  1,  1): 12,
    ( 0,  1, -1): 15,  ( 0, -1,  1): 21,  ( 0, -1, -1): 24,
    # three-axis
    ( 1,  1,  1): 13,  ( 1,  1, -1): 14,  ( 1, -1,  1): 16,
    ( 1, -1, -1): 17,  (-1,  1,  1): 22,  (-1,  1, -1): 23,
    (-1, -1,  1): 25,  (-1, -1, -1): 26,
}

# Angular → class index (7 classes): 0=static, 1=pitch up, 2=pitch down,
#   3=yaw left, 4=yaw right, 5=roll left, 6=roll right
# (we map roll → index 5/6, yaw → 3/4, pitch → 1/2)


def _c2w_to_w2c(c2w):
    """Invert a batch of 3x4 or 4x4 c2w matrices to w2c."""
    R = c2w[..., :3, :3]                 # [N, 3, 3]
    t = c2w[..., :3, 3]                  # [N, 3]
    R_inv = R.transpose(0, 2, 1)         # R^T
    t_inv = -np.einsum('nij,nj->ni', R_inv, t)
    w2c = np.zeros((c2w.shape[0], 3, 4), dtype=c2w.dtype)
    w2c[:, :3, :3] = R_inv
    w2c[:, :3, 3] = t_inv
    return w2c


def _segment_trajectory(poses_34, fps=30, static_thr=0.02, ang_thr=0.005):
    """
    Segment a [N, 3, 4] trajectory into motion-pattern labels.

    Returns array of int labels (one per frame pair), where
    label = trans_class * 7 + ang_class.
    """
    N = poses_34.shape[0]
    if N < 2:
        return np.array([0], dtype=int)

    w2c = _c2w_to_w2c(poses_34)          # [N, 3, 4]

    # Frame-to-frame relative transforms (w2c_i @ c2w_{i+1})
    velocities = []
    for i in range(N - 1):
        # v = w2c[i] @ [c2w[i+1]; 0 0 0 1]  (4x4 multiply, keep upper 3x4)
        w2c_i = np.eye(4); w2c_i[:3, :] = w2c[i]
        c2w_i1 = np.eye(4); c2w_i1[:3, :] = poses_34[i + 1]
        v = w2c_i @ c2w_i1
        velocities.append(v[:3])
    velocities = np.array(velocities)     # [N-1, 3, 4]

    t_vel = velocities[:, :3, 3] * fps   # translation velocity
    R_vel = velocities[:, :3, :3]         # rotation matrices

    # Translation classification
    t_signs = np.zeros((N - 1, 3), dtype=int)
    t_signs[np.abs(t_vel[:, 0]) > static_thr, 0] = np.sign(t_vel[np.abs(t_vel[:, 0]) > static_thr, 0])
    t_signs[np.abs(t_vel[:, 1]) > static_thr, 1] = np.sign(t_vel[np.abs(t_vel[:, 1]) > static_thr, 1])
    t_signs[np.abs(t_vel[:, 2]) > static_thr, 2] = np.sign(t_vel[np.abs(t_vel[:, 2]) > static_thr, 2])

    trans_classes = np.array([
        _TRANS_PATTERNS.get(tuple(s), 0) for s in t_signs
    ], dtype=int)

    # Angular classification (euler angles from rotation matrix)
    ang_classes = np.zeros(N - 1, dtype=int)
    for i in range(N - 1):
        R = R_vel[i]
        # Extract Euler angles (approximate small rotations)
        rx = np.arctan2(R[2, 1], R[2, 2])   # pitch
        ry = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))  # yaw
        rz = np.arctan2(R[1, 0], R[0, 0])   # roll
        angles = np.array([rx, ry, rz])

        if np.abs(angles).max() < ang_thr:
            ang_classes[i] = 0          # static
        else:
            dominant = np.argmax(np.abs(angles))
            sign = 1 if angles[dominant] > 0 else -1
            if dominant == 0:           # pitch (X)
                ang_classes[i] = 1 if sign > 0 else 2
            elif dominant == 1:         # yaw (Y)
                ang_classes[i] = 3 if sign > 0 else 4
            else:                       # roll (Z)
                ang_classes[i] = 5 if sign > 0 else 6

    # Combine: trans_class * 7 + ang_class
    combined = trans_classes * 7 + ang_classes
    return combined


def _smooth_labels(labels, window=15):
    """Majority-vote smoothing."""
    if len(labels) < window:
        return labels.copy()
    smoothed = labels.copy()
    half = window // 2
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        smoothed[i] = mode(labels[lo:hi], keepdims=False).mode
    return smoothed


def compute_caption_metrics(
    trajs_gen: list,
    trajs_ref: list,
    fps=30,
    static_thr=0.02,
    ang_thr=0.005,
    smooth_window=15,
) -> dict:
    """
    Compute caption metrics (precision / recall / F1) by comparing
    motion-pattern segmentations between generated and reference trajectories.

    Parameters
    ----------
    trajs_gen : list[np.ndarray]  each [N_i, 3, 4] generated trajectories
    trajs_ref : list[np.ndarray]  each [N_i, 3, 4] reference trajectories
    fps : frame rate for velocity scaling
    static_thr : translation velocity magnitude threshold for "moving"
    ang_thr : angular velocity threshold for "rotating"

    Returns
    -------
    dict with precision, recall, fscore
    """
    all_pred, all_target = [], []

    for traj_g, traj_r in zip(trajs_gen, trajs_ref):
        # Match lengths
        L = min(traj_g.shape[0], traj_r.shape[0])
        if L < 2:
            continue

        seg_g = _segment_trajectory(traj_g[:L], fps, static_thr, ang_thr)
        seg_r = _segment_trajectory(traj_r[:L], fps, static_thr, ang_thr)

        seg_g = _smooth_labels(seg_g, smooth_window)
        seg_r = _smooth_labels(seg_r, smooth_window)

        all_pred.append(seg_g)
        all_target.append(seg_r)

    if not all_pred:
        return {'precision': 0.0, 'recall': 0.0, 'fscore': 0.0}

    pred_flat = np.concatenate(all_pred)
    target_flat = np.concatenate(all_target)

    # Per-class weighted metrics
    classes = np.unique(np.concatenate([pred_flat, target_flat]))
    p_sum, r_sum, f_sum, w_sum = 0.0, 0.0, 0.0, 0.0

    for c in classes:
        tp = ((pred_flat == c) & (target_flat == c)).sum()
        fp = ((pred_flat == c) & (target_flat != c)).sum()
        fn = ((target_flat == c) & (pred_flat != c)).sum()
        support = (target_flat == c).sum()

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        p_sum += prec * support
        r_sum += rec * support
        f_sum += f1 * support
        w_sum += support

    if w_sum == 0:
        return {'precision': 0.0, 'recall': 0.0, 'fscore': 0.0}

    return {
        'precision': float(p_sum / w_sum),
        'recall': float(r_sum / w_sum),
        'fscore': float(f_sum / w_sum),
    }
