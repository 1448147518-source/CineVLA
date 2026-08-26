"""
CineVLA benchmark evaluation — computes FCD, PRDC, CLaTr Score on a test split.

Usage:
  python -m evaluate.eval_benchmark --resume ckpt.safetensors --data_path DataDoP/train

Flow:
  1. Load CineVLA model from checkpoint
  2. Load test split via CineVLADataset(test=True)
  3. For each sample: generate trajectory → encode to feature
  4. For each sample: encode GT trajectory → reference feature
  5. For each sample: encode text → text feature
  6. Aggregate across all samples → compute metrics → save CSV
"""

import os, sys, json
import numpy as np
import torch
import tyro
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quaternion import quat_to_rotmat_batched
from core.options import Options
from core.dataset import CineVLADataset, collate_fn
from core.perception import VideoPerceptionEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.utils import repeat_perception
from core.metrics import MetricsAccumulator, compute_pose_metrics
from evaluate.trajectory_encoder import TrajectoryEncoder


@dataclass
class BenchmarkOptions(Options):
    """Extended options for benchmark evaluation."""
    output_csv: str = 'metrics/benchmark.csv'
    prdc_k: int = 3
    device: str = 'cuda'
    trajectory_encoder: Optional[str] = None  # pretrained trajectory-text encoder; optional


def _poses_7d_to_34(poses_7d):
    """Convert [N, 7] (quat+trans) → [N, 3, 4] pose matrices."""
    R = quat_to_rotmat_batched(poses_7d[:, :4])
    T = poses_7d[:, 4:7].unsqueeze(-1)
    return torch.cat([R, T], dim=-1)


@torch.no_grad()
def run_benchmark(opt: BenchmarkOptions):
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f"[benchmark] device = {device}")

    # ── Load model ──
    perception = VideoPerceptionEncoder(opt.perception_dim, opt.image_size)
    planner = Planner(
        pose_dim=opt.pose_dim, pose_length=opt.pose_length,
        perception_dim=opt.perception_dim,
        hidden_dim=opt.planner_hidden_dim,
        num_layers=opt.planner_num_layers,
        num_heads=opt.planner_num_heads,
    )
    refiner = Refiner(
        pose_dim=opt.pose_dim, perception_dim=opt.perception_dim,
        hidden_dim=opt.refiner_hidden_dim,
        num_layers=opt.refiner_num_layers,
        num_heads=opt.refiner_num_heads,
    )

    if opt.resume:
        from safetensors.torch import load_file
        ckpt = load_file(opt.resume) if opt.resume.endswith('.safetensors') \
            else torch.load(opt.resume, map_location='cpu')
        for name, model in [('perception', perception), ('planner', planner),
                             ('refiner', refiner)]:
            sub = {k.replace(f'{name}.', ''): v for k, v in ckpt.items()
                   if k.startswith(f'{name}.')}
            model.load_state_dict(sub, strict=False)
        print(f"[benchmark] loaded checkpoint: {opt.resume}")

    perception = perception.eval().to(device)
    planner = planner.eval().to(device)
    refiner = refiner.eval().to(device)

    # Distribution/text metrics are valid only with an explicitly supplied,
    # pretrained encoder.  A random trajectory encoder would produce numbers
    # but not an evaluable scientific claim.
    traj_encoder = None
    if opt.trajectory_encoder:
        traj_encoder = TrajectoryEncoder(
            pose_dim=opt.pose_dim, hidden_dim=256, output_dim=768,
        ).eval().to(device)
        from safetensors.torch import load_file
        traj_encoder.load_state_dict(load_file(opt.trajectory_encoder), strict=True)
        print(f"[benchmark] loaded trajectory encoder: {opt.trajectory_encoder}")
    else:
        print('[benchmark] CLaTr/FCD/PRDC disabled: no pretrained trajectory encoder supplied')

    # ── Load test dataset ──
    ds = CineVLADataset(
        opt.data_path,
        split_txt='DataDoP/train_valid.txt',
        pose_length=opt.pose_length,
        test=True,
        test_size=opt.test_size,
        num_frames=opt.num_frames,
    )
    print(f"[benchmark] test samples: {len(ds)}")

    # ── Metrics accumulator ──
    metrics = MetricsAccumulator(prdc_k=opt.prdc_k)
    pose_totals = {}

    for idx in range(len(ds)):
        sample = ds[idx]
        frames = sample['frames'].unsqueeze(0).to(device)          # [1, T, 3, H, W]
        gt_poses = sample['poses'].to(device)                       # [N, 7]
        text = sample['text']

        if not text:
            text = 'camera movement'

        # ── Generate trajectory ──
        perc = perception(frames)
        out = planner.forward(repeat_perception(perc, repeats=2), [text, ''])
        cond_plan = out['poses'][0]
        uncond_plan = out['poses'][1]
        cfg_scale = getattr(opt, 'cfg_scale', 2.0)
        plan = uncond_plan + cfg_scale * (cond_plan - uncond_plan)

        # Closed-loop refinement (simplified — single pass for speed)
        trajectory = plan.clone()
        z_pred = perc['features_0']
        text_feats = planner.encode_text([text])
        for t in range(min(opt.closed_loop_steps - 1, plan.shape[0] - 1)):
            fi = min(int(t / opt.closed_loop_steps * frames.shape[1]), frames.shape[1] - 1)
            z_real = perc['features'][:, fi, :].squeeze(0)
            error = torch.nn.functional.mse_loss(z_real, z_pred).item()
            if error > 0.01:
                remaining = trajectory[t + 1:]
                if remaining.shape[0] > 0:
                    refined, z_pred_next = refiner.refine(
                        z_real, z_pred, remaining, text_feats.squeeze(0))
                    trajectory[t + 1:] = refined
                    z_pred = z_pred_next.unsqueeze(0)

        # ── Extract features ──
        direct = compute_pose_metrics(trajectory, gt_poses)
        for key, value in direct.items():
            pose_totals[key] = pose_totals.get(key, 0.0) + value

        # Convert to [N, 3, 4] for caption metrics
        traj_34 = _poses_7d_to_34(trajectory)
        gt_34 = _poses_7d_to_34(gt_poses)

        if traj_encoder is not None:
            gen_feat = traj_encoder(trajectory.unsqueeze(0)).squeeze(0)
            ref_feat = traj_encoder(gt_poses.unsqueeze(0)).squeeze(0)
            txt_feat = text_feats.mean(dim=1).squeeze(0)
            metrics.add(gen_feat, ref_feat, txt_feat,
                        traj_34_gen=traj_34, traj_34_ref=gt_34)

        if (idx + 1) % 10 == 0:
            print(f"[benchmark] {idx + 1}/{len(ds)} samples processed")

    # ── Compute & save ──
    results = {key: value / len(ds) for key, value in pose_totals.items()}
    results['num_samples'] = len(ds)
    if traj_encoder is not None:
        results.update(metrics.compute())
    os.makedirs(os.path.dirname(opt.output_csv) or '.', exist_ok=True)

    import pandas as pd
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
