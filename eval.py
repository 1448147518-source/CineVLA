"""
CineVLA v3.1 Inference — RGB frame sequence → camera trajectory.

Supports:
  - Single image: creates synthetic frame sequence via augmentations
  - Frame directory: loads multiple PNG frames
  - MP4 video: extracts frames automatically

Usage:
  python eval.py default --image_path "scene.jpg" --text "..." --resume "ckpt.safetensors"
  python eval.py default --image_path "frames/" --text "..." --resume "ckpt.safetensors"
  python eval.py default --image_path "video.mp4" --text "..." --resume "ckpt.safetensors"
"""

import os, json, time, glob, random
import cv2, numpy as np
import torch, tyro
import torch.nn.functional as F

from core.options import AllConfigs
from core.perception import VideoPerceptionEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.utils import slerp_trajectory
from visualise.trajectory import plot_trajectory
from visualise.latent import LatentLogger


class CineVLAInference:
    def __init__(self, opt):
        self.opt = opt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.perception = VideoPerceptionEncoder(opt.perception_dim, opt.image_size)
        self.planner = Planner(pose_dim=opt.pose_dim, pose_length=opt.pose_length,
                               perception_dim=opt.perception_dim,
                               hidden_dim=opt.planner_hidden_dim,
                               num_layers=opt.planner_num_layers,
                               num_heads=opt.planner_num_heads,
                               music_ca_layers=opt.music_ca_layers,
                               music_dim=opt.music_dim)
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
        """Load T frames from image, directory, or video."""
        path = str(path)
        Ht = Wt = 224

        def _load_one(p):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
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
            if files:
                frames = torch.stack([_load_one(f) for f in files])
                if len(frames) < num_frames:
                    pad = frames[-1:].repeat(num_frames - len(frames), 1, 1, 1)
                    frames = torch.cat([frames, pad])
                return frames.unsqueeze(0).to(self.device)

        # ── Video file ──
        if path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
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
                if len(frames) >= 2:
                    while len(frames) < num_frames: frames.append(frames[-1])
                    return torch.stack(frames[:num_frames]).unsqueeze(0).to(self.device)

        # ── Single image: create synthetic sequence ──
        img = _load_one(path)
        frames = [img]
        for _ in range(num_frames - 1):
            aug = img.clone()
            s = random.uniform(0.85, 0.98)
            h, w = int(224 * s), int(224 * s)
            y, x = random.randint(0, 224 - h), random.randint(0, 224 - w)
            patch = aug[:, y:y + h, x:x + w]
            aug = F.interpolate(patch.unsqueeze(0), (224, 224), mode='bilinear',
                                align_corners=False).squeeze(0)
            aug = torch.clamp(aug * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05), 0, 1)
            frames.append(aug)
        return torch.stack(frames).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def run(self, image_path, text, music_path=None, output_dir='outputs'):
        os.makedirs(output_dir, exist_ok=True)

        # ── Phase 1: Plan with CFG ──
        frames = self._load_frame_sequence(image_path)
        perc = self.perception(frames)
        text_feats = self.planner.encode_text([text])

        # Conditional trajectory (with text guidance)
        cond_plan = self.planner.plan(perc, [text], music_path=music_path)
        # Unconditional trajectory (empty text)
        uncond_plan = self.planner.plan(perc, [''], music_path=music_path)
        # CFG extrapolation
        cfg_scale = getattr(self.opt, 'cfg_scale', 2.0)
        plan = uncond_plan + cfg_scale * (cond_plan - uncond_plan)
        print(f"[Planner] {plan.shape[0]} frames initial trajectory (CFG={cfg_scale})")

        # ── Phase 2: Closed-loop refinement ──
        trajectory = plan.clone()
        z_pred = perc['features_0']  # initial prediction
        N = min(self.opt.closed_loop_steps, plan.shape[0])
        steps = []

        latent_logger = None
        if getattr(self.opt, 'vis_latent', False):
            latent_logger = LatentLogger(save_dir='pred_latent')

        for t in range(N - 1):
            T = frames.shape[1]
            fi = min(int(t / N * T), T - 1)
            z_real = perc['features'][:, fi, :].squeeze(0)  # current frame feature

            # Log latent state if visualization enabled
            if latent_logger is not None:
                latent_logger.log(z_real, z_pred, step=t, phase='infer')

            error = F.mse_loss(z_real, z_pred).item()
            if error > 0.01 and t < N - 1:
                remaining = trajectory[t + 1:]
                if remaining.shape[0] > 0:
                    refined, z_pred_next = self.refiner.refine(
                        z_real, z_pred, remaining, text_feats.squeeze(0))
                    trajectory[t + 1:] = refined
                    z_pred = z_pred_next.unsqueeze(0)
                    print(f"  Step {t}: refined {remaining.shape[0]} frames, err={error:.4f}")

            steps.append({'step': t, 'pose': trajectory[t].tolist(), 'error': error})

        # ── Output ──
        poses_34 = torch.zeros(trajectory.shape[0], 3, 4)
        for i, p in enumerate(trajectory):
            R = _q2r(p[:4])
            poses_34[i, :, :3] = R
            poses_34[i, :, 3] = p[4:7]
        dense = slerp_trajectory(poses_34, self.opt.dense_frames)

        np.save(os.path.join(output_dir, 'trajectory.npy'), trajectory.cpu().numpy())
        np.save(os.path.join(output_dir, 'trajectory_dense.npy'), dense.cpu().numpy())
        json.dump(steps, open(os.path.join(output_dir, 'steps.json'), 'w'), indent=2)
        print(f"[Done] → {output_dir}/")

        # ── Trajectory visualization (always) ──
        plot_trajectory(
            trajectory.cpu().numpy(),
            dense=dense.cpu().numpy(),
            steps=steps,
            save_dir='results',
            title=f'CineVLA — {text[:60]}'
        )

        # ── Latent visualization finalize ──
        if latent_logger is not None:
            latent_logger.finalize()
            print(f"[visualise] Latent state plots saved to pred_latent/")

        return {'trajectory': trajectory, 'dense': dense}


def _q2r(q):
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def main():
    opt = tyro.cli(AllConfigs)
    engine = CineVLAInference(opt)
    engine.run(opt.image_path or 'input.jpg', opt.text or '',
               music_path=opt.music_path,
               output_dir=os.path.join(opt.workspace, opt.exp_name or 'output'))


if __name__ == '__main__':
    main()
