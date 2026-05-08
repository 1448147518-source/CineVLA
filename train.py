"""
CineVLA v3.1 Training — RGB-only, frame sequences, no depth.

Usage:
  python train.py default --workspace workspace --exp_name run1
"""

import os, math, contextlib, random
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
from core.losses import planner_loss, refiner_loss, get_effective_pose_length
from visualise.latent import LatentLogger


class _SimpleAccelerator:
    def __init__(self, grad_accum=1):
        self.gradient_accumulation_steps = grad_accum
        self._step = 0
        self._should_step = True
    @property
    def is_main_process(self): return True
    def wait_for_everyone(self): pass
    def prepare(self, *args): return args
    def backward(self, loss):
        (loss / self.gradient_accumulation_steps).backward()
    @property
    def sync_gradients(self):
        self._step += 1
        self._should_step = self._step % self.gradient_accumulation_steps == 0
        return self._should_step
    def clip_grad_norm_(self, p, m):
        if self._should_step:
            torch.nn.utils.clip_grad_norm_(p, m)
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

        # Variable chunk-size: sample K ∈ [1, refiner_lookahead]
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
        if stage == 'planner': return self.forward_planner(batch)
        if stage == 'refiner': return self.forward_refiner(batch)
        p = self.forward_planner(batch)
        r = self.forward_refiner(batch)
        result = {'loss': p['loss'] + r['loss'],
                  'loss_p': p['loss'], 'loss_r': r['loss'],
                  'z_real': r.get('z_real'), 'z_pred': r.get('z_pred')}
        # Merge component losses with prefix for logging clarity
        for k, v in p.items():
            if k.startswith('L_'):
                result[f'p_{k}'] = v
        for k, v in r.items():
            if k.startswith('loss_'):
                result[f'r_{k}'] = v
        return result


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
        latent_logger = LatentLogger(save_dir='pred_latent')
        logger.info("[visualise] Latent state logging enabled → pred_latent/")

    def run_stage(name, epochs, keys=('loss',)):
        nonlocal gs, best
        for ep in range(epochs):
            # Progressive trajectory-length curriculum (planner stage only)
            if name == 'planner':
                model.effective_pose_length = get_effective_pose_length(
                    ep, epochs,
                    start_len=opt.pose_start_len,
                    full_len=opt.pose_length,
                    ramp_epochs=opt.curriculum_ramp_epochs,
                )
            model.train()
            tl, sl, seen = 0.0, {k: 0.0 for k in keys}, 0
            for batch in dl:
                with acc.accumulate(model):
                    out = model(batch, stage=name)
                    loss = out['loss']
                    acc.backward(loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(model.parameters(), opt.grad_clip)
                        optm.step(); sch.step()
                        optm.zero_grad()

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
                        log_dict = {'train/loss': lv, 'train/lr': lr,
                                    'train/epoch': ep, 'train/stage': name, 'train/step': gs}
                        for k in keys:
                            if k in out:
                                log_dict[f'train/{k}'] = (out[k].detach().item() if has_acc else out[k].item())
                        wandb.log(log_dict)
                gs += 1

            avg = tl / seen
            msg = f"[{name}] ep{ep} avg={avg:.4f} " + ' '.join(f'{k}={sl[k]/seen:.4f}' for k in keys)
            if name == 'planner':
                msg += f' N_eff={model.effective_pose_length}'
            logger.info(msg)
            if wandb:
                log_dict = {'train/avg_loss': avg, 'train/stage_epoch': ep,
                          **{f'train/{k}': sl[k]/seen for k in keys}}
                if name == 'planner':
                    log_dict['train/effective_pose_length'] = model.effective_pose_length
                wandb.log(log_dict)
            if avg < best:
                best = avg; acc.wait_for_everyone()
                save_file(model.state_dict(), os.path.join(opt.workspace, opt.exp_name, 'best.safetensors'))

    run_stage('planner', opt.planner_pretrain_epochs,
              keys=('loss', 'L_rot', 'L_trans', 'L_rel', 'L_smooth'))
    run_stage('refiner', opt.refiner_pretrain_epochs,
              keys=('loss', 'loss_pose', 'loss_z', 'loss_rel', 'loss_smooth'))
    run_stage('joint', opt.joint_epochs,
              keys=('loss', 'loss_p', 'loss_r'))

    acc.wait_for_everyone()
    save_file(model.state_dict(), os.path.join(opt.workspace, opt.exp_name, 'model.safetensors'))
    logger.info("Saved.")

    if latent_logger is not None:
        latent_logger.finalize()
        logger.info("[visualise] Latent state summaries saved to pred_latent/")

    if wandb: wandb.finish()


if __name__ == '__main__':
    main()
