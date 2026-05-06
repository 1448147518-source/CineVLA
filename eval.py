"""
CineVLA v3 Closed-Loop Inference

Pipeline:
  1. Planner: image_0 + text → initial trajectory [p_1 ... p_N]
  2. Loop for t = 0, 1, 2, ... N:
     a. Capture image_t at current pose p_t
     b. Perception → z_t (real environment latent)
     c. Compare z_t vs predicted ẑ_t → error
     d. Refiner → refined remaining trajectory + predicted ẑ_{t+1}
     e. Move to refined p_{t+1}

Usage:
  python eval.py default --image_path "scene.jpg" --text "..." --resume "ckpt.safetensors"
"""

import os, json, time
import cv2, numpy as np
import torch, tyro
import torch.nn.functional as F

from core.options import AllConfigs
from core.perception import PerceptionEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.music_encoder import MusicEncoder
from core.utils import slerp_trajectory


class CineVLAInference:
    def __init__(self, opt):
        self.opt = opt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.perception = PerceptionEncoder(opt.perception_dim, opt.image_size)
        self.planner = Planner(
            pose_dim=opt.pose_dim, pose_length=opt.pose_length,
            perception_dim=opt.perception_dim,
            hidden_dim=opt.planner_hidden_dim,
            num_layers=opt.planner_num_layers,
            num_heads=opt.planner_num_heads,
        )
        self.refiner = Refiner(
            pose_dim=opt.pose_dim,
            perception_dim=opt.perception_dim,
            hidden_dim=opt.refiner_hidden_dim,
            num_layers=opt.refiner_num_layers,
            num_heads=opt.refiner_num_heads,
        )
        self.music = MusicEncoder(dim=opt.music_dim, seq_len=opt.music_seq_len)

        if opt.resume:
            from safetensors.torch import load_file
            ckpt = load_file(opt.resume) if opt.resume.endswith('.safetensors') \
                else torch.load(opt.resume, map_location='cpu')
            self.perception.load_state_dict({k.replace('perception.', ''): v for k, v in ckpt.items() if k.startswith('perception.')}, strict=False)
            self.planner.load_state_dict({k.replace('planner.', ''): v for k, v in ckpt.items() if k.startswith('planner.')}, strict=False)
            self.refiner.load_state_dict({k.replace('refiner.', ''): v for k, v in ckpt.items() if k.startswith('refiner.')}, strict=False)

        self.perception = self.perception.eval().to(device)
        self.planner = self.planner.eval().to(device)
        self.refiner = self.refiner.eval().to(device)
        self.device = device

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img = img[..., [2, 1, 0]]
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        h, w = t.shape[1], t.shape[2]
        if h > 224: t = t[:, (h - 224) // 2: (h - 224) // 2 + 224, :]
        if w > 224: t = t[:, :, (w - 224) // 2: (w - 224) // 2 + 224]
        return t.unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

    @torch.no_grad()
    def run(self, image_path: str, text: str, music_path: str = None,
            output_dir: str = 'outputs') -> dict:
        """
        Run full closed-loop inference.

        In real deployment, images at steps t>0 would come from the camera.
        Here we simulate by starting from the initial frame only.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── Phase 1: Initial Planning ──
        img_0 = self._load_image(image_path)
        z_0 = self.perception(img_0)  # [1, perception_dim]
        music_feats = self.music(music_path, self.device) if music_path else None
        plan = self.planner.plan(z_0, [text], music_feats)  # [N, 7]

        text_feats = self.planner.encode_text([text])

        print(f"[Planner] Initial trajectory: {plan.shape[0]} frames")

        # ── Phase 2: Closed-Loop Refinement ──
        trajectory = plan.clone()
        z_pred = z_0  # initial prediction
        N = min(self.opt.closed_loop_steps, plan.shape[0])
        step_results = []

        for t in range(N - 1):
            # In real deployment: capture image_t from camera at pose trajectory[t]
            # Here: use initial frame as proxy (simulated closed-loop)
            z_real = self.perception(img_0)  # would be: perception(camera_capture())

            # Check if refinement needed
            error = F.mse_loss(z_real, z_pred).item()
            if error > 0.01 and t < N - 1:
                remaining = trajectory[t + 1:]
                # Ensure at least 1 frame remaining
                if remaining.shape[0] > 0:
                    refined_remaining, z_pred_next = self.refiner.refine(
                        z_real.squeeze(0), z_pred.squeeze(0),
                        remaining, text_feats.squeeze(0),
                    )
                        # Replace remaining trajectory with refined version
                    trajectory[t + 1:] = refined_remaining
                    z_pred = z_pred_next.unsqueeze(0)
                    print(f"  Step {t}: refined {remaining.shape[0]} frames, error={error:.4f}")

            step_results.append({
                'step': t,
                'pose': trajectory[t].cpu().tolist(),
                'perception_error': error,
            })

        # ── Output ──
        # SLERP to dense trajectory
        poses_34 = torch.zeros(trajectory.shape[0], 3, 4)
        for i, p in enumerate(trajectory):
            R = quaternion_to_matrix_pt(p[:4])
            poses_34[i, :, :3] = R
            poses_34[i, :, 3] = p[4:7]
        dense = slerp_trajectory(poses_34, self.opt.dense_frames)

        np.save(os.path.join(output_dir, 'trajectory.npy'), trajectory.cpu().numpy())
        np.save(os.path.join(output_dir, 'trajectory_dense.npy'), dense.cpu().numpy())

        with open(os.path.join(output_dir, 'steps.json'), 'w') as f:
            json.dump(step_results, f, indent=2)

        print(f"[Done] Trajectory saved to {output_dir}/")
        return {'trajectory': trajectory, 'dense': dense, 'steps': step_results}


def quaternion_to_matrix_pt(q):
    """Single quaternion → 3x3 rotation matrix."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)]),
        torch.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)]),
        torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]),
    ], dim=-2)


def main():
    opt = tyro.cli(AllConfigs)
    engine = CineVLAInference(opt)
    engine.run(
        image_path=opt.image_path or 'assets/scene.jpg',
        text=opt.text or 'Camera moves forward smoothly.',
        music_path=opt.music_path,
        output_dir=os.path.join(opt.workspace, opt.exp_name or 'output'),
    )


if __name__ == '__main__':
    main()
