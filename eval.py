"""
CineVLA v3.1 Inference — RGB frame sequence → camera trajectory.

Supported inputs:
  - Frame directory (_frames/): loads multiple PNG/JPG frames
  - MP4 video: extracts evenly-spaced frames automatically
  - Single images: NOT supported — use a _frames/ directory or video file

Usage:
  python eval.py default --image_path "frames/" --text "..." --resume "ckpt.safetensors"
  python eval.py default --image_path "video.mp4" --text "..." --resume "ckpt.safetensors"
"""

import os, json, glob
import cv2, numpy as np
import torch, tyro
import torch.nn.functional as F

from core.options import AllConfigs
from core.perception import VideoPerceptionEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.utils import slerp_trajectory, quat_to_rot
from visualise.trajectory import plot_trajectory
from visualise.latent import LatentLogger


class CineVLAInference:
    def __init__(self, opt):
        self.opt = opt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.perception = VideoPerceptionEncoder(opt.perception_dim, opt.image_size,
                                                 freeze_backbone=opt.freeze_encoders)
        self.planner = Planner(pose_dim=opt.pose_dim, pose_length=opt.pose_length,
                               perception_dim=opt.perception_dim,
                               hidden_dim=opt.planner_hidden_dim,
                               num_layers=opt.planner_num_layers,
                               num_heads=opt.planner_num_heads,
                               music_ca_layers=opt.music_ca_layers,
                               music_dim=opt.music_dim,
                               freeze_text_encoder=opt.freeze_encoders)
        self.refiner = Refiner(pose_dim=opt.pose_dim, perception_dim=opt.perception_dim,
                               hidden_dim=opt.refiner_hidden_dim,
                               num_layers=opt.refiner_num_layers,
                               num_heads=opt.refiner_num_heads)

        if opt.resume:
            from safetensors.torch import load_file
            ckpt = load_file(opt.resume) if opt.resume.endswith('.safetensors') \
                else torch.load(opt.resume, map_location='cpu')
            for name, model in [('perception', self.perception), ('planner', self.planner),
                                 ('refiner', self.refiner)]:
                sub = {k.replace(f'{name}.', ''): v for k, v in ckpt.items() if k.startswith(f'{name}.')}
                model.load_state_dict(sub, strict=False)
            print(f"[INFO] Loaded checkpoint from {opt.resume}")

        self.perception = self.perception.eval().to(device)
        self.planner = self.planner.eval().to(device)
        self.refiner = self.refiner.eval().to(device)
        self.device = device

    def _load_frame_sequence(self, path, num_frames=8):
        """Load T frames: _frames/ directory or video file.  No single-image fallback.

        Priority:
          1. Directory of PNG/JPG files — real multi-view frames
          2. MP4/AVI/MOV/MKV — auto-extract evenly-spaced frames
          3. Anything else → RuntimeError (single images not supported)
        """
        path = str(path)
        Ht = Wt = 224

        def _load_one(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = img[..., [2, 1, 0]]
            t = torch.from_numpy(img).permute(2, 0, 1).float()
            h, w = t.shape[1], t.shape[2]
            if h > Ht: t = t[:, (h - Ht) // 2:(h - Ht) // 2 + Ht, :]
            if w > Wt: t = t[:, :, (w - Wt) // 2:(w - Wt) // 2 + Wt]
            return t

        # ── Directory of frames ──
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.png')) +
                           glob.glob(os.path.join(path, '*.jpg')))[:num_frames]
            if not files:
                raise RuntimeError(f"Frame directory is empty: {path}")
            if len(files) < num_frames:
                raise RuntimeError(
                    f"Frame directory {path}: {len(files)} images found, "
                    f"need >= {num_frames}"
                )
            frames = torch.stack([_load_one(f) for f in files])
            return frames.unsqueeze(0).to(self.device)

        # ── Video file ──
        if path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < num_frames:
                cap.release()
                raise RuntimeError(
                    f"Video {path}: {total} frames total, need >= {num_frames}"
                )
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break
                frame = frame.astype(np.float32) / 255.
                frame = frame[..., [2, 1, 0]]
                t = torch.from_numpy(frame).permute(2, 0, 1).float()
                h, w = t.shape[1], t.shape[2]
                if h > Ht: t = t[:, (h - Ht) // 2:(h - Ht) // 2 + Ht, :]
                if w > Wt: t = t[:, :, (w - Wt) // 2:(w - Wt) // 2 + Wt]
                frames.append(t)
            cap.release()
            if len(frames) < num_frames:
                raise RuntimeError(
                    f"Video {path}: could only read {len(frames)}/{num_frames} frames"
                )
            return torch.stack(frames).unsqueeze(0).to(self.device)

        # ── Unsupported ──
        raise RuntimeError(
            f"Unsupported input: {path}\n"
            f"Provide a _frames/ directory or a video file (.mp4/.avi/.mov/.mkv). "
            f"Single-image input is not supported."
        )

    @torch.no_grad()
    def infer(self, image_path, text, music_path=None):
        """Core inference: returns trajectory + perception + text features.

        No file I/O or visualization.  Benchmarks call this directly.
        """
        # Phase 1: Plan with CFG
        frames = self._load_frame_sequence(image_path, num_frames=self.opt.num_frames)
        perc = self.perception(frames)
        text_feats = self.planner.encode_text([text])

        cond_plan = self.planner.plan(perc, [text], music_path=music_path)
        uncond_plan = self.planner.plan(perc, [''], music_path=music_path)
        cfg_scale = getattr(self.opt, 'cfg_scale', 2.0)
        plan = uncond_plan + cfg_scale * (cond_plan - uncond_plan)

        # Phase 2: Closed-loop refinement
        trajectory = plan.clone()
        z_pred = perc['features_0'].squeeze(0)
        N = min(self.opt.closed_loop_steps, plan.shape[0])
        steps = []

        for t in range(N - 1):
            T = frames.shape[1]
            fi = min(int(t / N * T), T - 1)
            z_real = perc['features'][:, fi, :].squeeze(0)

            error = F.mse_loss(z_real, z_pred).item()
            if error > 0.01 and t < N - 1:
                remaining = trajectory[t + 1:]
                if remaining.shape[0] > 0:
                    refined, z_pred_next = self.refiner.refine(
                        z_real, z_pred, remaining, text_feats.squeeze(0))
                    trajectory[t + 1:] = refined
                    z_pred = z_pred_next

            steps.append({'step': t, 'pose': trajectory[t].tolist(), 'error': error})

        return {
            'trajectory': trajectory, 'steps': steps, 'perc': perc,
            'text_feats': text_feats, 'cfg_scale': cfg_scale,
        }

    @torch.no_grad()
    def run(self, image_path, text, music_path=None, output_dir='outputs'):
        os.makedirs(output_dir, exist_ok=True)

        result = self.infer(image_path, text, music_path=music_path)
        trajectory = result['trajectory']
        perc = result['perc']
        text_feats = result['text_feats']
        steps = result['steps']

        print(f"[Planner] {trajectory.shape[0]} frames (CFG={result['cfg_scale']})")

        # ── Latent visualization (optional) ──
        latent_logger = None
        if getattr(self.opt, 'vis_latent', False):
            latent_logger = LatentLogger(save_dir=os.path.join(output_dir, 'pred_latent'))
            N = min(self.opt.closed_loop_steps, trajectory.shape[0])
            collected = set()
            for t in range(N - 1):
                fi = min(int(t / N * self.opt.num_frames), self.opt.num_frames - 1)
                z_real = perc['features'][0, fi, :]
                if fi not in collected:
                    latent_logger.log(z_real, perc['features_0'][0], step=t, phase='infer')
                    collected.add(fi)

        # ── Output ──
        poses_34 = torch.zeros(trajectory.shape[0], 3, 4)
        for i, p in enumerate(trajectory):
            R = quat_to_rot(p[:4])
            poses_34[i, :, :3] = R
            poses_34[i, :, 3] = p[4:7]
        dense = slerp_trajectory(poses_34, self.opt.dense_frames)

        np.save(os.path.join(output_dir, 'trajectory.npy'), trajectory.cpu().numpy())
        np.save(os.path.join(output_dir, 'trajectory_dense.npy'), dense.cpu().numpy())
        json.dump(steps, open(os.path.join(output_dir, 'steps.json'), 'w'), indent=2)
        print(f"[Done] → {output_dir}/")

        # Expose intermediate features for benchmark reuse
        self._last_perc = perc
        self._last_text_feats = text_feats

        # ── Trajectory visualization (always) ──
        plot_trajectory(
            trajectory.cpu().numpy(),
            dense=dense.cpu().numpy(),
            steps=steps,
            save_dir=os.path.join(output_dir, 'results'),
            title=f'CineVLA — {text[:60]}'
        )

        # ── Latent visualization finalize ──
        if latent_logger is not None:
            latent_logger.finalize()
            print(f"[visualise] Latent plots → {output_dir}/pred_latent/")

        return {'trajectory': trajectory, 'dense': dense}


def main():
    opt = tyro.cli(AllConfigs)
    engine = CineVLAInference(opt)
    engine.run(opt.image_path or 'input.jpg', opt.text or '',
               music_path=opt.music_path,
               output_dir=os.path.join(opt.workspace, opt.exp_name or 'output'))


if __name__ == '__main__':
    main()
