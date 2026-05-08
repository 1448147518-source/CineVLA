"""CineVLA top-level model — composes Perception, Planner, and Refiner."""

import random
import torch
import torch.nn as nn

from core.options import Options
from core.perception import VideoPerceptionEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.losses import planner_loss, refiner_loss


class CineVLA(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.perception = VideoPerceptionEncoder(opt.perception_dim, opt.image_size)
        self.planner = Planner(
            pose_dim=opt.pose_dim, pose_length=opt.pose_length,
            perception_dim=opt.perception_dim,
            hidden_dim=opt.planner_hidden_dim,
            num_layers=opt.planner_num_layers,
            num_heads=opt.planner_num_heads,
            music_ca_layers=opt.music_ca_layers,
            music_dim=opt.music_dim,
        )
        self.refiner = Refiner(
            pose_dim=opt.pose_dim, perception_dim=opt.perception_dim,
            hidden_dim=opt.refiner_hidden_dim,
            num_layers=opt.refiner_num_layers,
            num_heads=opt.refiner_num_heads,
        )

    def forward_planner(self, batch):
        frames, texts, gt_poses = batch['frames'], batch['text'], batch['poses']
        music = batch.get('music_path')

        # CFG: 10% text dropout during training
        if self.training and random.random() < 0.1:
            texts = [''] * len(texts)

        perc = self.perception(frames)
        out = self.planner(perc, texts, music_path=music if isinstance(music, str) else None)

        # v4: decoupled geometric loss
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

        perc = self.perception(frames)
        text_feats = self.planner.encode_text(texts)

        # Variable chunk-size: sample K in [1, refiner_lookahead]
        t = torch.randint(1, N - 1, (1,)).item()
        max_k = min(self.opt.refiner_lookahead, N - 1 - t)
        K = torch.randint(1, max_k + 1, (1,)).item()
        T = frames.shape[1]
        frame_idx = min(int(t / N * T), T - 1)

        z_real = perc['features'][:, frame_idx, :]
        z_pred = perc['global']
        remaining = gt_poses[:, t:t + K, :]
        raw = self.refiner(z_real, z_pred, remaining, text_feats)

        # v4: geometric refiner loss
        loss, comps = refiner_loss(
            raw['refined'], remaining, raw['z_next_pred'], z_real,
            lambda_pose=self.opt.lambda_pose_delta,
            lambda_z=self.opt.lambda_z_pred,
            lambda_rel=self.opt.lambda_rel_ref,
            lambda_smooth=self.opt.lambda_smooth_ref,
            window_size=self.opt.rel_window_size,
            lambda_rot_smooth=self.opt.lambda_rot_smooth,
            lambda_rel_t=self.opt.lambda_rel_t,
        )
        return {'loss': loss, 'z_real': z_real, 'z_pred': z_pred, **comps}

    def forward(self, batch, stage='joint'):
        if stage == 'planner':
            return self.forward_planner(batch)
        if stage == 'refiner':
            return self.forward_refiner(batch)
        p = self.forward_planner(batch)
        r = self.forward_refiner(batch)
        result = {'loss': p['loss'] + r['loss'],
                  'loss_p': p['loss'], 'loss_r': r['loss'],
                  'z_real': r.get('z_real'), 'z_pred': r.get('z_pred')}
        for k, v in p.items():
            if k.startswith('L_'):
                result[f'p_{k}'] = v
        for k, v in r.items():
            if k.startswith('loss_'):
                result[f'r_{k}'] = v
        return result
