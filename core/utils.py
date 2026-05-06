"""Camera math: quaternion ↔ matrix, SLERP, and token conversion."""

import math
import torch
import numpy as np


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k),  two_s * (i * j - k * r),  two_s * (i * k + j * r),
        two_s * (i * j + k * r),      1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r),      two_s * (j * k + i * r),      1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(M):
    prefix = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)
    tr = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]
    Qs = []
    for i in range(Ms.shape[0]):
        m, t = Ms[i], tr[i]
        if t > 0:
            r = torch.sqrt(t) / 2.0
            x = (m[2, 1] - m[1, 2]) / (4 * r)
            y = (m[0, 2] - m[2, 0]) / (4 * r)
            z = (m[1, 0] - m[0, 1]) / (4 * r)
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            r = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            r = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            r = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
        Qs += [torch.stack([r, x, y, z], dim=-1)]
    return torch.stack(Qs, dim=0).reshape(*prefix, 4)


def quaternion_slerp(q0, q1, fraction, shortestpath=True):
    d = (q0 * q1).sum(-1)
    if shortestpath:
        d[d < 0] = -d[d < 0]
        q1[d < 0] = -q1[d < 0]
    d = d.clamp(0, 1.0)
    angle = torch.acos(d)
    isin = 1.0 / (torch.sin(angle) + 1e-10)
    q = q0 * torch.sin((1.0 - fraction) * angle) * isin + q1 * torch.sin(fraction * angle) * isin
    q[angle < 1e-5] = q0[angle < 1e-5]
    return q


def sample_from_two_pose(pose_a, pose_b, fraction):
    quat_a = matrix_to_quaternion(pose_a[..., :3, :3])
    quat_b = matrix_to_quaternion(pose_b[..., :3, :3])
    dot = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    quat_b = torch.where(dot < 0, -quat_b, quat_b)
    q = quaternion_slerp(quat_a, quat_b, fraction)
    R = quaternion_to_matrix(q)
    T = (1 - fraction) * pose_a[..., :3, 3] + fraction * pose_b[..., :3, 3]
    new = pose_a.clone()
    new[..., :3, :3] = R
    new[..., :3, 3] = T
    return new


def slerp_trajectory(poses_34, num_dense):
    """Interpolate sparse [N, 3, 4] trajectory to [num_dense, 3, 4]."""
    N = poses_34.shape[0]
    dense = []
    for i in range(num_dense):
        frac = i / (num_dense - 1) * (N - 1) if num_dense > 1 else 0
        idx = int(frac)
        w = frac - idx
        if idx >= N - 1:
            dense.append(poses_34[N - 1])
        else:
            dense.append(sample_from_two_pose(poses_34[idx], poses_34[idx + 1], w))
    return torch.stack(dense)


def init_logger(filename):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger
