"""
CineVLA v3 Training — 3-stage: Planner → Refiner → Joint
Usage:
  accelerate launch train.py default --workspace workspace --exp_name run1
"""

import os, math
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
from core.provider import CineVLADataset, collate_fn
from core.utils import init_logger


class CineVLA(nn.Module):
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
            pose_dim=opt.pose_dim, perception_dim=opt.perception_dim,
            hidden_dim=opt.refiner_hidden_dim,
            num_layers=opt.refiner_num_layers,
            num_heads=opt.refiner_num_heads,
        )
        self.music = MusicEncoder(dim=opt.music_dim, seq_len=opt.music_seq_len)

    def forward_planner(self, batch):
        images, texts, gt_poses = batch['rgb'], batch['text'], batch['poses']
        z_0 = self.perception(images)
        out = self.planner(z_0, texts)
        loss = F.mse_loss(out['poses'], gt_poses)
        return {'loss': loss, 'pred_poses': out['poses']}

    def forward_refiner(self, batch):
        B = len(batch['text'])
        device = batch['rgb'].device
        images_0, depths_0 = batch['rgb'], batch['depth']
        gt_poses, c2ws = batch['poses'], batch['c2ws']
        intr, texts = batch['intrinsics'], batch['text']
        N = self.opt.pose_length

        z_0 = self.perception(images_0)
        text_feats = self.planner.encode_text(texts)

        t = torch.randint(1, N - 1, (1,)).item()

        z_real_list = []
        for b in range(B):
            warped = warp_viewpoint(
                images_0[b], depths_0[b].squeeze(0),
                c2ws[b, 0], c2ws[b, t], intr[b], 224)
            z_real_list.append(self.perception(warped.unsqueeze(0)))
        z_real = torch.cat(z_real_list, dim=0)

        z_pred = self.perception(images_0)
        refiner_input = gt_poses[:, t:, :]
        out = self.refiner(z_real, z_pred, refiner_input, text_feats)
        return out

    def forward(self, batch, stage='joint'):
        if stage == 'planner':
            return self.forward_planner(batch)
        if stage == 'refiner':
            return self.forward_refiner(batch)
        p = self.forward_planner(batch)
        r = self.forward_refiner(batch)
        return {'loss': p['loss'] + r['loss'], 'loss_p': p['loss'], 'loss_r': r['loss']}


def main():
    opt = tyro.cli(AllConfigs)
    acc = Accelerator(mixed_precision=opt.mixed_precision,
                       gradient_accumulation_steps=opt.grad_accum)
    os.makedirs(os.path.join(opt.workspace, opt.exp_name), exist_ok=True)
    logger = init_logger(os.path.join(opt.workspace, opt.exp_name, 'log.txt'))

    # ── Wandb init ──
    wandb = None
    if acc.is_main_process:
        try:
            import wandb
            wandb.init(project='cinevla-v3', name=opt.exp_name, config=vars(opt),
                       dir=os.path.join(opt.workspace, opt.exp_name), resume='allow')
            logger.info("wandb initialized")
        except ImportError:
            logger.info("wandb not installed, skipping")

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
    total = (opt.planner_pretrain_epochs + opt.refiner_pretrain_epochs + opt.joint_epochs) * len(dl)

    def lr_lambda(s):
        p = s / max(1, total)
        if p < opt.warmup_ratio: return p / opt.warmup_ratio
        p = (p - opt.warmup_ratio) / (1 - opt.warmup_ratio)
        return 0.5 * (1 + math.cos(math.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(optm, lr_lambda)
    model, optm, dl, sch = acc.prepare(model, optm, dl, sch)

    global_step = 0
    best_loss = float('inf')

    def run_stage(stage_name, epochs, loss_keys=('loss',)):
        nonlocal global_step, best_loss
        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            sub_losses = {k: 0.0 for k in loss_keys}
            seen = 0
            for batch in dl:
                with acc.accumulate(model):
                    optm.zero_grad()
                    out = model(batch, stage=stage_name)
                    loss = out['loss']
                    acc.backward(loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    optm.step()
                    sch.step()

                total_loss += loss.detach()
                for k in loss_keys:
                    if k in out:
                        sub_losses[k] += out[k].detach()
                seen += 1

                if acc.is_main_process and global_step % 20 == 0:
                    mem = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 1)
                    lr = sch.get_last_lr()[0]
                    logger.info(f"[{stage_name}] ep{ep} step{global_step} loss={loss.item():.4f} lr={lr:.2e}")
                    if wandb:
                        wandb.log({'train/loss': loss.item(), 'train/lr': lr,
                                   'train/epoch': ep, 'train/step': global_step,
                                   'train/stage': stage_name})

                global_step += 1

            avg = total_loss.item() / seen
            if acc.is_main_process:
                log_msg = f"[{stage_name}] ep{ep} avg_loss={avg:.4f}"
                for k in loss_keys:
                    log_msg += f" {k}={sub_losses[k].item()/seen:.4f}"
                logger.info(log_msg)
                if wandb:
                    d = {f'train/avg_loss': avg, f'train/stage_epoch': ep}
                    for k in loss_keys:
                        d[f'train/{k}'] = sub_losses[k].item() / seen
                    wandb.log(d)

                if avg < best_loss:
                    best_loss = avg
                    acc.wait_for_everyone()
                    save_file(model.state_dict(),
                              os.path.join(opt.workspace, opt.exp_name, 'best.safetensors'))

    # ── Stage 1: Planner ──
    run_stage('planner', opt.planner_pretrain_epochs)

    # ── Stage 2: Refiner ──
    run_stage('refiner', opt.refiner_pretrain_epochs, loss_keys=('loss', 'loss_pose', 'loss_z'))

    # ── Stage 3: Joint ──
    run_stage('joint', opt.joint_epochs, loss_keys=('loss', 'loss_p', 'loss_r'))

    acc.wait_for_everyone()
    if acc.is_main_process:
        save_file(model.state_dict(),
                  os.path.join(opt.workspace, opt.exp_name, 'model.safetensors'))
        logger.info("Saved.")
        if wandb:
            wandb.finish()


if __name__ == '__main__':
    main()
