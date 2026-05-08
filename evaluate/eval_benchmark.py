"""
CineVLA benchmark evaluation — FCD, PRDC, CLaTr Score, caption metrics.

Usage:
  python -m evaluate.eval_benchmark --resume ckpt.safetensors --data_path DataDoP/train

Reuses CineVLAInference for trajectory generation — no duplicated inference logic.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import tyro
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.options import Options
from core.dataset import CineVLADataset
from core.metrics import MetricsAccumulator
from core.utils import quaternion_to_matrix
from evaluate.trajectory_encoder import TrajectoryEncoder
from eval import CineVLAInference


@dataclass
class BenchmarkOptions(Options):
    output_csv: str = 'metrics/benchmark.csv'
    prdc_k: int = 3
    device: str = 'cuda'


def trajectory_7d_to_34(poses_7d):
    N = poses_7d.shape[0]
    R = quaternion_to_matrix(poses_7d[:, :4])
    T = poses_7d[:, 4:7].unsqueeze(-1)
    return torch.cat([R, T], dim=-1)


@torch.no_grad()
def run_benchmark(opt: BenchmarkOptions):
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f"[benchmark] device = {device}")

    # ── CineVLA inference engine ──
    engine = CineVLAInference(opt)
    engine.device = device
    engine.perception = engine.perception.to(device)
    engine.planner = engine.planner.to(device)
    engine.refiner = engine.refiner.to(device)

    # ── Trajectory encoder ──
    traj_encoder = TrajectoryEncoder(
        pose_dim=opt.pose_dim, hidden_dim=256, output_dim=768,
    ).eval().to(device)

    enc_path = opt.resume.replace('.safetensors', '_traj_enc.safetensors') if opt.resume else None
    if enc_path and os.path.exists(enc_path):
        from safetensors.torch import load_file
        traj_encoder.load_state_dict(load_file(enc_path), strict=False)
        print(f"[benchmark] loaded trajectory encoder: {enc_path}")

    # ── Load test dataset ──
    ds = CineVLADataset(
        opt.data_path, split_txt='DataDoP/train_valid.txt',
        pose_length=opt.pose_length, test=True,
        test_size=max(opt.test_size, 16), num_frames=opt.num_frames,
    )
    print(f"[benchmark] test samples: {len(ds)}")

    # ── Metrics accumulator ──
    metrics = MetricsAccumulator(prdc_k=opt.prdc_k)

    for idx in range(len(ds)):
        sample = ds[idx]
        gt_poses = sample['poses'].to(device)
        text = sample['text'] or 'camera movement'

        # Use CineVLAInference.infer() — no file I/O, no visualization
        # The image_path must point to the _frames/ directory or _video.mp4
        base = sample['path']
        frames_dir = base + '_frames'
        video_path = base + '_video.mp4'

        if os.path.exists(video_path):
            image_path = video_path
        elif os.path.isdir(frames_dir):
            image_path = frames_dir
        else:
            print(f"[benchmark] skipping {base}: no frames/video")
            continue

        result = engine.infer(image_path, text)

        trajectory = result['trajectory']
        text_feats = result['text_feats']

        gen_feat = traj_encoder(trajectory.unsqueeze(0)).squeeze(0)
        ref_feat = traj_encoder(gt_poses.unsqueeze(0)).squeeze(0)
        txt_feat = text_feats.mean(dim=1).squeeze(0)

        traj_34 = trajectory_7d_to_34(trajectory)
        gt_34 = trajectory_7d_to_34(gt_poses)

        metrics.add(gen_feat, ref_feat, txt_feat,
                    traj_34_gen=traj_34, traj_34_ref=gt_34)

        if (idx + 1) % 10 == 0:
            print(f"[benchmark] {idx + 1}/{len(ds)} samples processed")

    # ── Compute & save ──
    results = metrics.compute()
    os.makedirs(os.path.dirname(opt.output_csv) or '.', exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(opt.output_csv, index=False)
    print(f"[benchmark] metrics saved to {opt.output_csv}")
    print(df.to_string(index=False))

    return results


def main():
    opt = tyro.cli(BenchmarkOptions)
    run_benchmark(opt)


if __name__ == '__main__':
    main()
