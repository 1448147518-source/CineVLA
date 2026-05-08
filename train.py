"""
CineVLA v3.1 Training — RGB-only, frame sequences, no depth.

Usage:
  python train.py default --workspace workspace --exp_name run1
"""

import os, math, sys, contextlib, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from safetensors.torch import load_file, save_file

from core.options import AllConfigs, Options
from core.perception import VideoPerceptionEncoder
from core.planner import Planner
from core.refiner import Refiner
from core.dataset import CineVLADataset, collate_fn
from core.utils import init_logger
from visualise.latent import LatentLogger


class _SimpleAccelerator:
    def __init__(self, grad_accum=1):
        self.gradient_accumulation_steps = grad_accum
        self._step = 0
        self._sync_gradients = False

    @property
    def is_main_process(self): return True
    def wait_for_everyone(self): pass
    def prepare(self, *args): return args
    def backward(self, loss):
        loss.backward()
        self._step += 1
        self._sync_gradients = (self._step % self.gradient_accumulation_steps == 0)

    @property
    def sync_gradients(self):
        return self._sync_gradients

    def clip_grad_norm_(self, p, m): torch.nn.utils.clip_grad_norm_(p, m)
    @contextlib.contextmanager
    def accumulate(self, model): yield

def _get_accelerator(opt):
    try:
        from accelerate import Accelerator
        return Accelerator(mixed_precision=opt.mixed_precision,
                           gradient_accumulation_steps=opt.grad_accum), True
    except Exception:
        print("[WARN] using single GPU/CPU mode")
        return _SimpleAccelerator(opt.grad_accum), False


class CineVLA(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.perception = VideoPerceptionEncoder(opt.perception_dim, opt.image_size,
                                                 freeze_backbone=opt.freeze_encoders)
        self.planner = Planner(
            pose_dim=opt.pose_dim, pose_length=opt.pose_length,
            perception_dim=opt.perception_dim,
            hidden_dim=opt.planner_hidden_dim,
            num_layers=opt.planner_num_layers,
            num_heads=opt.planner_num_heads,
            music_ca_layers=opt.music_ca_layers,
            music_dim=opt.music_dim,
            freeze_text_encoder=opt.freeze_encoders,
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

        # CFG: 10% per-sample text dropout during training
        if self.training:
            texts = ['' if random.random() < 0.1 else t for t in texts]

        perc = self.perception(frames)
        out = self.planner(perc, texts, music_path=music if isinstance(music, str) else None)
        loss = F.mse_loss(out['poses'], gt_poses)
        return {'loss': loss, 'pred_poses': out['poses']}

    def forward_refiner(self, batch):
        B = len(batch['text'])
        frames = batch['frames']
        gt_poses = batch['poses']
        texts = batch['text']
        music = batch.get('music_path')
        N = self.opt.pose_length
        T = frames.shape[1]

        perc = self.perception(frames)
        text_feats = self.planner.encode_text(texts)

        # Variable chunk-size: sample K ∈ [1, refiner_lookahead]
        t = torch.randint(1, N - 1, (1,)).item()
        max_k = min(self.opt.refiner_lookahead, N - 1 - t)
        K = torch.randint(1, max_k + 1, (1,)).item()
        frame_idx = min(int(t / N * T), T - 1)

        # Planner output serves as the noisy input the Refiner must correct
        pred_poses = self.planner(perc, texts, music_path=music if isinstance(music, str) else None)['poses']

        z_real = perc['features'][:, frame_idx, :]
        # z_pred: feature from an earlier frame, simulating step-by-step prior knowledge
        prev_idx = max(0, frame_idx - torch.randint(1, min(4, frame_idx + 1), (1,)).item())
        z_pred = perc['features'][:, prev_idx, :]

        # Refiner input = planner output (noisy), target = ground truth
        remaining = pred_poses[:, t:t + K, :].detach()
        gt_slice = gt_poses[:, t:t + K, :]

        # z_next target = feature of the NEXT frame (true future prediction)
        next_idx = min(frame_idx + 1, T - 1)
        z_next_target = perc['features'][:, next_idx, :]

        out = self.refiner(z_real, z_pred, remaining, text_feats,
                           gt_poses=gt_slice, z_next_target=z_next_target)
        out['z_real'] = z_real  # for latent visualization
        out['z_pred'] = z_pred
        return out

    def forward(self, batch, stage='joint'):
        if stage == 'planner': return self.forward_planner(batch)
        if stage == 'refiner': return self.forward_refiner(batch)
        p = self.forward_planner(batch)
        r = self.forward_refiner(batch)
        return {'loss': p['loss'] + r['loss'], 'loss_p': p['loss'], 'loss_r': r['loss'],
                'z_real': r.get('z_real'), 'z_pred': r.get('z_pred')}


def main():
    opt = tyro.cli(AllConfigs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(opt.workspace, opt.exp_name), exist_ok=True)
    logger = init_logger(os.path.join(opt.workspace, opt.exp_name, 'log.txt'))

    acc, has_acc = _get_accelerator(opt)

    wandb = None
    try:
        import wandb
        wandb.init(project='cinevla-v3', name=opt.exp_name, config=vars(opt),
                   dir=os.path.join(opt.workspace, opt.exp_name), resume='allow')
        logger.info("wandb initialized")
    except ImportError: pass

    model = CineVLA(opt).to(device)
    if opt.resume:
        ckpt = load_file(opt.resume) if opt.resume.endswith('.safetensors') \
            else torch.load(opt.resume, map_location=device)
        model.load_state_dict(ckpt, strict=False)

    ds = CineVLADataset(opt.data_path, split_txt='DataDoP/train_valid.txt',
                         pose_length=opt.pose_length, test_size=opt.test_size,
                         num_frames=opt.num_frames)
    dl = torch.utils.data.DataLoader(ds, batch_size=opt.batch_size, shuffle=True,
                                     num_workers=0 if device.type == 'cpu' else 2,
                                     pin_memory=(device.type == 'cuda'),
                                     collate_fn=collate_fn)

    optm = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01)
    total = (opt.planner_pretrain_epochs + opt.refiner_pretrain_epochs + opt.joint_epochs) * len(dl)

    def lr_lambda(s):
        p = s / max(1, total)
        if p < opt.warmup_ratio: return p / opt.warmup_ratio
        p = (p - opt.warmup_ratio) / (1 - opt.warmup_ratio)
        return 0.5 * (1 + math.cos(math.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(optm, lr_lambda)
    if has_acc: model, optm, dl, sch = acc.prepare(model, optm, dl, sch)

    gs, best = 0, float('inf')

    # ── Latent state visualization (optional) ──
    latent_logger = None
    if opt.vis_latent:
        latent_logger = LatentLogger(
            save_dir=os.path.join(opt.workspace, opt.exp_name, 'pred_latent'))
        logger.info("[visualise] Latent state logging enabled")

    def run_stage(name, epochs, keys=('loss',)):
        nonlocal gs, best
        for ep in range(epochs):
            model.train()
            tl, sl, seen = 0.0, {k: 0.0 for k in keys}, 0
            for batch in dl:
                with acc.accumulate(model):
                    optm.zero_grad()
                    out = model(batch, stage=name)
                    loss = out['loss']
                    acc.backward(loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    optm.step(); sch.step()

                lv = loss.detach().item() if has_acc else loss.item()
                tl += lv
                for k in keys:
                    if k in out:
                        sl[k] += (out[k].detach().item() if has_acc else out[k].item())
                seen += 1

                # Latent state logging (for refiner / joint stages)
                if latent_logger is not None and 'z_real' in out:
                    if gs % opt.vis_latent_every == 0:
                        latent_logger.log(out['z_real'], out['z_pred'],
                                         step=gs, phase='train')

                if gs % 20 == 0:
                    lr = sch.get_last_lr()[0]
                    logger.info(f"[{name}] ep{ep} step{gs} loss={lv:.4f} lr={lr:.2e}")
                    if wandb:
                        wandb.log({'train/loss': lv, 'train/lr': lr,
                                   'train/epoch': ep, 'train/stage': name, 'train/step': gs})
                gs += 1

            avg = tl / seen
            logger.info(f"[{name}] ep{ep} avg={avg:.4f} " +
                        ' '.join(f'{k}={sl[k]/seen:.4f}' for k in keys))
            if wandb:
                wandb.log({'train/avg_loss': avg, 'train/stage_epoch': ep,
                          **{f'train/{k}': sl[k]/seen for k in keys}})
            if avg < best:
                best = avg; acc.wait_for_everyone()
                save_file(model.state_dict(), os.path.join(opt.workspace, opt.exp_name, 'best.safetensors'))

    run_stage('planner', opt.planner_pretrain_epochs)
    run_stage('refiner', opt.refiner_pretrain_epochs, keys=('loss', 'loss_pose', 'loss_z'))
    run_stage('joint', opt.joint_epochs, keys=('loss', 'loss_p', 'loss_r'))

    acc.wait_for_everyone()
    save_file(model.state_dict(), os.path.join(opt.workspace, opt.exp_name, 'model.safetensors'))
    logger.info("Saved.")

    if latent_logger is not None:
        latent_logger.finalize()
        logger.info("[visualise] Latent state summaries saved")

    if wandb: wandb.finish()


if __name__ == '__main__':
    main()
