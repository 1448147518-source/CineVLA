"""
CineVLA v3 Training — 3-stage: Planner pretrain → Refiner pretrain → Joint

Usage:
  accelerate launch --config_file acc_configs/gpu1.yaml train.py default \
      --workspace workspace --exp_name run1
"""

import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from safetensors.torch import load_file, save_file

from core.options import AllConfigs, Options
from core.perception import PerceptionEncoder, warp_viewpoint
from core.planner import Planner
from core.refiner import Refiner
from core.music_encoder import MusicEncoder
from core.dataset import CineVLADataset, collate_fn
from core.utils import init_logger


class CineVLA(nn.Module):
    """Full CineVLA v3 model: Perception + Planner + Refiner."""

    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.perception = PerceptionEncoder(opt.perception_dim, opt.image_size)
        self.planner = Planner(
            pose_dim=opt.pose_dim, pose_length=opt.pose_length,
            perception_dim=opt.perception_dim,
            hidden_dim=opt.planner_hidden_dim,
            num_layers=opt.planner_num_layers,
            num_heads=opt.planner_num_heads,
            text_ca_layers=opt.planner_text_ca_layers,
        )
        self.refiner = Refiner(
            pose_dim=opt.pose_dim,
            perception_dim=opt.perception_dim,
            hidden_dim=opt.refiner_hidden_dim,
            num_layers=opt.refiner_num_layers,
            num_heads=opt.refiner_num_heads,
        )
        self.music = MusicEncoder(dim=opt.music_dim, seq_len=opt.music_seq_len)

    def forward_planner(self, batch):
        """Stage 1: train Planner."""
        images = batch['rgb']
        texts = batch['text']
        gt_poses = batch['poses']
        z_0 = self.perception(images)
        out = self.planner(z_0, texts)
        loss = F.mse_loss(out['poses'], gt_poses)
        return {'loss': loss, 'pred_poses': out['poses']}

    def forward_refiner(self, batch):
        """Stage 2: train Refiner with synthetic intermediate views."""
        B = len(batch['text'])
        device = batch['rgb'].device
        perception = self.perception
        planner = self.planner
        refiner = self.refiner

        images_0 = batch['rgb']
        depths_0 = batch['depth']
        gt_poses = batch['poses']
        c2ws = batch['c2ws']
        intr = batch['intrinsics']
        texts = batch['text']

        N = self.opt.pose_length

        # Generate initial plan
        z_0 = perception(images_0)
        plan = self.planner(z_0, texts)['poses']  # [B, N, 7]
        text_feats = self.planner.encode_text(texts)

        # Pick random step t for refinement training
        t = torch.randint(1, N - 1, (1,)).item()
        remaining_gt = gt_poses[:, t:, :]

        # Warp source image to viewpoint at pose_t for synthetic "real" perception
        z_real_list = []
        for b in range(B):
            T = c2ws[b, t]  # [4, 4]
            warped = warp_viewpoint(images_0[b], depths_0[b].squeeze(0),
                                    c2ws[b, 0], T, intr[b], 224)
            z_real_list.append(perception(warped.unsqueeze(0)))
        z_real = torch.cat(z_real_list, dim=0)  # [B, perception_dim]

        # Predicted perception from planner's trajectory
        z_pred = perception(images_0)  # crude: use initial perception as predicted

        # Refine
        refiner_input = gt_poses[:, t:, :]  # use GT as "planned" for training
        out = refiner(z_real, z_pred, refiner_input, text_feats)
        return out

    def forward(self, batch, stage='joint'):
        if stage == 'planner':
            return self.forward_planner(batch)
        elif stage == 'refiner':
            return self.forward_refiner(batch)
        else:
            p = self.forward_planner(batch)
            r = self.forward_refiner(batch)
            return {'loss': p['loss'] + r['loss'], 'loss_p': p['loss'], 'loss_r': r['loss']}


def main():
    opt = tyro.cli(AllConfigs)
    accelerator = Accelerator(mixed_precision=opt.mixed_precision,
                              gradient_accumulation_steps=opt.grad_accum)
    os.makedirs(os.path.join(opt.workspace, opt.exp_name), exist_ok=True)
    logger = init_logger(os.path.join(opt.workspace, opt.exp_name, 'log.txt'))

    model = CineVLA(opt)
    if opt.resume:
        ckpt = load_file(opt.resume) if opt.resume.endswith('.safetensors') \
            else torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        logger.info("Resumed checkpoint")

    ds = CineVLADataset(opt.data_path, split_txt='DataDoP/train_valid.txt',
                         pose_length=opt.pose_length, test_size=opt.test_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=opt.batch_size, shuffle=True,
                                     num_workers=2, pin_memory=True, collate_fn=collate_fn)
    optm = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01)
    model, optm, dl = accelerator.prepare(model, optm, dl)

    # Stage 1: Planner pretrain
    for ep in range(opt.planner_pretrain_epochs):
        model.train()
        tl = sum(model(batch, stage='planner')['loss'].item() for batch in dl) / len(dl)
        if accelerator.is_main_process:
            logger.info(f"[Planner] ep{ep} loss={tl:.4f}")

    # Stage 2: Refiner pretrain
    for ep in range(opt.refiner_pretrain_epochs):
        model.train()
        tl = sum(model(batch, stage='refiner')['loss'].item() for batch in dl) / len(dl)
        if accelerator.is_main_process:
            logger.info(f"[Refiner] ep{ep} loss={tl:.4f}")

    # Stage 3: Joint
    for ep in range(opt.joint_epochs):
        model.train()
        for batch in dl:
            optm.zero_grad()
            out = model(batch)
            accelerator.backward(out['loss'])
            optm.step()
        if accelerator.is_main_process and ep % 5 == 0:
            logger.info(f"[Joint] ep{ep} loss={out['loss'].item():.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_file(model.state_dict(), os.path.join(opt.workspace, opt.exp_name, 'model.safetensors'))
        logger.info("Saved.")


if __name__ == '__main__':
    main()
