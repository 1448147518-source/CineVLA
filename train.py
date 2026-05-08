"""
CineVLA v4 Training — RGB-only, frame sequences, no depth.

Usage:
  python train.py default --workspace workspace --exp_name run1
"""

import os, math
import torch
import tyro
from safetensors.torch import load_file, save_file

from core.options import AllConfigs
from core.model import CineVLA
from core.dataset import CineVLADataset, collate_fn
from core.utils import init_logger
from core.accelerator import get_accelerator
from core.losses import get_effective_pose_length
from visualise.latent import LatentLogger


def main():
    opt = tyro.cli(AllConfigs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(opt.workspace, opt.exp_name), exist_ok=True)
    logger = init_logger(os.path.join(opt.workspace, opt.exp_name, 'log.txt'))

    acc, has_acc = get_accelerator(opt)

    wandb = None
    try:
        import wandb
        wandb.init(project='cinevla-v3', name=opt.exp_name, config=vars(opt),
                   dir=os.path.join(opt.workspace, opt.exp_name), resume='allow')
        logger.info("wandb initialized")
    except ImportError:
        pass

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
        if p < opt.warmup_ratio:
            return p / opt.warmup_ratio
        p = (p - opt.warmup_ratio) / (1 - opt.warmup_ratio)
        return 0.5 * (1 + math.cos(math.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(optm, lr_lambda)
    if has_acc:
        model, optm, dl, sch = acc.prepare(model, optm, dl, sch)

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
                        optm.step()
                        sch.step()
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
                best = avg
                acc.wait_for_everyone()
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

    if wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
