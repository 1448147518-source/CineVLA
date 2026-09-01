"""Closed-loop inference for CineVLA.

Input frames are exposed only through a ``CameraEnvironment``. The default
``OfflineReplayEnv`` is useful for deterministic integration tests; it does
not render observations from commanded poses and must not be used to claim
action-conditioned closed-loop performance.
"""

import json
import os

import numpy as np
import torch
import tyro

from core.options import AllConfigs
from core.perception import VideoPerceptionEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.quaternion import quat_to_rotmat, slerp_trajectory
from core.utils import repeat_perception
from envs.base import CameraEnvironment
from envs.offline_replay import OfflineReplayEnv
from visualise.latent import LatentLogger
from visualise.trajectory import plot_trajectory


class CineVLAInference:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.perception = VideoPerceptionEncoder(opt.perception_dim, opt.image_size)
        self.planner = Planner(
            pose_dim=opt.pose_dim, pose_length=opt.pose_length,
            perception_dim=opt.perception_dim, hidden_dim=opt.planner_hidden_dim,
            num_layers=opt.planner_num_layers, num_heads=opt.planner_num_heads,
        )
        self.refiner = Refiner(
            pose_dim=opt.pose_dim, perception_dim=opt.perception_dim,
            hidden_dim=opt.refiner_hidden_dim, num_layers=opt.refiner_num_layers,
            num_heads=opt.refiner_num_heads,
            flow_steps=opt.refiner_flow_steps,
            correction_min_scale=opt.refiner_correction_min_scale,
        )

        if opt.resume:
            from safetensors.torch import load_file
            checkpoint = load_file(opt.resume) if opt.resume.endswith('.safetensors') \
                else torch.load(opt.resume, map_location='cpu')
            for name, model in [('perception', self.perception), ('planner', self.planner),
                                ('refiner', self.refiner)]:
                submodule = {key.replace(f'{name}.', ''): value for key, value in checkpoint.items()
                             if key.startswith(f'{name}.')}
                model.load_state_dict(submodule, strict=False)
            print(f'[INFO] Loaded checkpoint from {opt.resume}')

        self.perception = self.perception.eval().to(self.device)
        self.planner = self.planner.eval().to(self.device)
        self.refiner = self.refiner.eval().to(self.device)

    def _perceive_history(self, history):
        frames = torch.stack(history[-self.opt.num_frames:]).unsqueeze(0).to(self.device)
        return self.perception(frames)

    def _plan_with_cfg(self, perception, text):
        cfg_perception = repeat_perception(perception, repeats=2)
        outputs = self.planner(cfg_perception, [text, ''])['poses']
        conditional, unconditional = outputs[0], outputs[1]
        return unconditional + self.opt.cfg_scale * (conditional - unconditional)

    @torch.no_grad()
    def run_environment(self, environment: CameraEnvironment, text, output_dir='outputs'):
        """Run Predict -> Observe -> Compare -> Correct closed-loop inference."""
        os.makedirs(output_dir, exist_ok=True)
        observation = environment.reset()
        history = [observation.rgb]

        perception = self._perceive_history(history)
        current_z = perception['global'].squeeze(0)
        text_features = self.planner.encode_text([text])
        trajectory = self._plan_with_cfg(perception, text)

        episode_length = len(environment) if hasattr(environment, '__len__') else None
        episode_text = f'; episode has {episode_length} observations' if episode_length else ''
        print(f'[Planner] {trajectory.shape[0]} poses{episode_text}')

        logger = LatentLogger(save_dir='pred_latent') if self.opt.vis_latent else None
        steps = []
        max_actions = min(self.opt.closed_loop_steps - 1, trajectory.shape[0] - 1)
        if episode_length is not None:
            max_actions = min(max_actions, episode_length - 1)

        for action_index in range(1, max_actions + 1):
            action = trajectory[action_index].clone()

            # Predict what the next observation should look like before executing.
            z_predicted = self.refiner.predict_next_latent(
                current_z, action.unsqueeze(0)
            )

            observation, terminated = environment.step(action.cpu())
            history.append(observation.rgb)
            perception = self._perceive_history(history)
            z_real = perception['global'].squeeze(0)

            if logger is not None:
                logger.log(z_real, z_predicted, step=observation.step, phase='infer')

            # Cheap scalar gate; the actual correction condition is the learned
            # discrepancy representation inside Refiner.
            error = torch.nn.functional.mse_loss(z_real, z_predicted).item()
            refined = False
            remaining = trajectory[action_index + 1:]
            if error > self.opt.discrepancy_threshold and len(remaining) > 0:
                corrected, _ = self.refiner.refine(
                    z_real, z_predicted, remaining, text_features.squeeze(0)
                )
                trajectory[action_index + 1:] = corrected
                refined = True

            steps.append({
                'step': observation.step,
                'pose': action.tolist(),
                'error': error,
                'refined': refined,
                'environment': observation.info.get('environment'),
            })
            if refined:
                print(f'  Step {observation.step}: refined {len(remaining)} future poses, err={error:.4f}')

            current_z = z_real
            if terminated:
                break

        poses_34 = torch.zeros(trajectory.shape[0], 3, 4)
        for index, pose in enumerate(trajectory):
            poses_34[index, :, :3] = quat_to_rotmat(pose[:4])
            poses_34[index, :, 3] = pose[4:7]
        dense = slerp_trajectory(poses_34, self.opt.dense_frames)

        np.save(os.path.join(output_dir, 'trajectory.npy'), trajectory.cpu().numpy())
        np.save(os.path.join(output_dir, 'trajectory_dense.npy'), dense.cpu().numpy())
        with open(os.path.join(output_dir, 'steps.json'), 'w') as handle:
            json.dump(steps, handle, indent=2)
        print(f'[Done] → {output_dir}/')

        plot_trajectory(trajectory.cpu().numpy(), dense=dense.cpu().numpy(), steps=steps,
                        save_dir='results', title=f'CineVLA — {text[:60]}')
        if logger is not None:
            logger.finalize()
            print('[visualise] Latent state plots saved to pred_latent/')
        return {'trajectory': trajectory, 'dense': dense, 'steps': steps}

    def run(self, image_path, text, output_dir='outputs'):
        environment = OfflineReplayEnv.from_path(
            image_path, image_size=self.opt.image_size,
            max_frames=self.opt.closed_loop_steps,
        )
        return self.run_environment(environment, text, output_dir)


def main():
    opt = tyro.cli(AllConfigs)
    if not opt.image_path:
        raise RuntimeError('Provide --image_path to a frame directory or video file.')
    CineVLAInference(opt).run(
        opt.image_path, opt.text or '',
        output_dir=os.path.join(opt.workspace, opt.exp_name or 'output'),
    )


if __name__ == '__main__':
    main()
