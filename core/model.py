"""CineVLA top-level model — CLIP Planner + VAE-state Flow Refiner."""

import random
import torch
import torch.nn as nn

from core.options import Options
from core.perception import VideoPerceptionEncoder
from core.vae_perception import RefinerVAEEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.losses import planner_loss, refiner_loss


class CineVLA(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        # Planner visual pathway: semantic/contextual CLIP features.
        self.perception = VideoPerceptionEncoder(opt.perception_dim, opt.image_size)
        self.planner = Planner(
            pose_dim=opt.pose_dim, pose_length=opt.pose_length,
            perception_dim=opt.perception_dim,
            hidden_dim=opt.planner_hidden_dim,
            num_layers=opt.planner_num_layers,
            num_heads=opt.planner_num_heads,
        )

        # Refiner visual pathway: reconstructive spatial VAE latent.
        self.refiner_vae = RefinerVAEEncoder(
            model_id=opt.refiner_vae_model,
            image_size=opt.refiner_vae_image_size,
            freeze=opt.freeze_refiner_vae,
        )
        self.refiner = Refiner(
            pose_dim=opt.pose_dim,
            latent_channels=self.refiner_vae.latent_channels,
            hidden_dim=opt.refiner_hidden_dim,
            num_layers=opt.refiner_num_layers,
            num_heads=opt.refiner_num_heads,
            flow_steps=opt.refiner_flow_steps,
            correction_min_scale=opt.refiner_correction_min_scale,
        )

    def forward_planner(self, batch):
        frames, texts, gt_poses = batch['frames'], batch['text'], batch['poses']
        if self.training and random.random() < 0.1:
            texts = [''] * len(texts)

        perc = self.perception(frames)
        out = self.planner(perc, texts)
        eff_N = getattr(self, 'effective_pose_length', None)
        loss, comps = planner_loss(
            out['poses'], gt_poses, effective_N=eff_N,
            lambda_rot=self.opt.lambda_rot,
            lambda_trans=self.opt.lambda_trans,
            lambda_rel=self.opt.lambda_rel,
            lambda_smooth=self.opt.lambda_smooth,
            window_size=self.opt.rel_window_size,
            lambda_rot_smooth=self.opt.lambda_rot_smooth,
            lambda_rel_t=self.opt.lambda_rel_t,
        )
        return {'loss': loss, 'pred_poses': out['poses'], **comps}

    def forward_refiner(self, batch):
        frames = batch['frames']
        gt_poses = batch['poses']
        texts = batch['text']
        N = self.opt.pose_length

        # Planner still consumes CLIP-based perception and supplies cached text semantics.
        perc = self.perception(frames)
        text_feats, text_padding_mask = self.planner.encode_text(texts, return_padding_mask=True)
        plan_out = self.planner(perc, texts)
        planned_trajectory = plan_out['poses']
        if not self.training or getattr(self, '_refiner_only', False):
            planned_trajectory = planned_trajectory.detach()

        if self.training:
            t = torch.randint(0, N - 1, (1,)).item()
            max_k = min(self.opt.refiner_lookahead, N - 1 - t)
            K = torch.randint(1, max_k + 1, (1,)).item()
        else:
            t = max(0, N // 2 - 1)
            K = min(self.opt.refiner_lookahead, N - 1 - t)

        T = frames.shape[1]
        frame_idx = min(round(t * (T - 1) / max(N - 1, 1)), T - 1)
        next_frame_idx = min(frame_idx + 1, T - 1)

        # VAE visual states are independent from Planner CLIP features.
        z_current = self.refiner_vae(frames[:, frame_idx])
        z_next_target = self.refiner_vae(frames[:, next_frame_idx]).detach()

        planned = planned_trajectory[:, t + 1:t + 1 + K, :]
        target = gt_poses[:, t + 1:t + 1 + K, :]

        # Predict next VAE state from the current state and the next planned motion,
        # then compare that prediction with the actual next observation state.
        z_pred = self.refiner.predict_next_latent(z_current, planned[:, :1])
        z_real = z_next_target

        raw = self.refiner(
            z_real, z_pred, planned, text_feats, text_padding_mask,
            target_poses=target,
        )

        # World-model supervision is aligned to the same t -> t+1 transition.
        loss, comps = refiner_loss(
            raw['refined'], target,
            z_pred, z_next_target,
            flow_velocity=raw['flow_velocity'],
            flow_target=raw['flow_target'],
            lambda_pose=self.opt.lambda_pose_delta,
            lambda_z=self.opt.lambda_z_pred,
            lambda_flow=self.opt.lambda_flow,
            lambda_rel=self.opt.lambda_rel_ref,
            lambda_smooth=self.opt.lambda_smooth_ref,
            window_size=self.opt.rel_window_size,
            lambda_rot_smooth=self.opt.lambda_rot_smooth,
            lambda_rel_t=self.opt.lambda_rel_t,
        )
        return {
            'loss': loss,
            'z_real': z_real,
            'z_pred': z_pred,
            'discrepancy': raw['discrepancy'],
            **comps,
        }

    def forward(self, batch, stage='joint'):
        if stage == 'planner':
            return self.forward_planner(batch)
        if stage == 'refiner':
            return self.forward_refiner(batch)
        p = self.forward_planner(batch)
        r = self.forward_refiner(batch)
        result = {
            'loss': p['loss'] + r['loss'],
            'loss_p': p['loss'], 'loss_r': r['loss'],
            'z_real': r.get('z_real'), 'z_pred': r.get('z_pred'),
        }
        for k, v in p.items():
            if k.startswith('L_'):
                result[f'p_{k}'] = v
        for k, v in r.items():
            if k.startswith('loss_'):
                result[f'r_{k}'] = v
        return result
