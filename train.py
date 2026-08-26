"""
CineVLA v4 Training — RGB-only, frame sequences, no depth.

Usage:
  python train.py default --workspace workspace --exp_name run1
"""

import json
import os, math
import torch
import tyro
from safetensors.torch import load_file, save_file

from core.options import AllConfigs
from core.model import CineVLA
from core.dataset import CineVLADataset, collate_fn
from core.utils import init_logger, set_seed
from core.accelerator import get_accelerator
from core.losses import get_effective_pose_length
from visualise.latent import LatentLogger


def move_to_device(value, device):
    """Recursively move a collated batch without modifying its text fields."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def main():
    opt = tyro.cli(AllConfigs)
    set_seed(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = os.path.join(opt.workspace, opt.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = init_logger(os.path.join(run_dir, 'log.txt'))

    acc, has_acc = get_accelerator(opt)
    if acc.is_main_process:
        with open(os.path.join(run_dir, 'config.json'), 'w') as handle:
            json.dump(vars(opt), handle, indent=2, sort_keys=True)

    wandb = None
    try:
        import wandb
        wandb.init(project='cinevla-v3', name=opt.exp_name, config=vars(opt),
                   dir=run_dir, resume='allow')
        logger.info("wandb initialized")
    except ImportError:
        pass

    model = CineVLA(opt).to(device)
    resume_state = {}
    if opt.resume:
        ckpt = load_file(opt.resume) if opt.resume.endswith('.safetensors') \
            else torch.load(opt.resume, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        state_path = opt.resume.rsplit('.', 1)[0] + '.training.pt'
        if os.path.exists(state_path):
            resume_state = torch.load(state_path, map_location='cpu')
            logger.info(f"restoring optimizer/scheduler state from {state_path}")

    train_ds = CineVLADataset(opt.data_path, split_txt='DataDoP/train_valid.txt',
                              pose_length=opt.pose_length, test_size=opt.test_size,
                              num_frames=opt.num_frames, split_seed=opt.seed)
    valid_ds = CineVLADataset(opt.data_path, split_txt='DataDoP/train_valid.txt',
                              pose_length=opt.pose_length, test=True, test_size=opt.test_size,
                              num_frames=opt.num_frames, split_seed=opt.seed)
    workers = 0 if device.type == 'cpu' else opt.num_workers
    dl = torch.utils.data.DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                                     num_workers=workers,
                                     pin_memory=(device.type == 'cuda'),
                                     collate_fn=collate_fn)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=opt.batch_size, shuffle=False,
                                           num_workers=workers,
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
    if resume_state:
        optm.load_state_dict(resume_state['optimizer'])
        sch.load_state_dict(resume_state['scheduler'])
        if 'torch_rng_state' in resume_state:
            torch.set_rng_state(resume_state['torch_rng_state'])
        if torch.cuda.is_available() and resume_state.get('cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(resume_state['cuda_rng_state_all'])
    if has_acc:
        model, optm, dl, valid_dl, sch = acc.prepare(model, optm, dl, valid_dl, sch)

    gs = int(resume_state.get('global_step', 0))

    # ── Latent state visualization (optional) ──
    latent_logger = None
    if opt.vis_latent:
        latent_logger = LatentLogger(save_dir='pred_latent')
        logger.info("[visualise] Latent state logging enabled → pred_latent/")

    def save_model(filename):
        if acc.is_main_process:
            save_file(acc.unwrap_model(model).state_dict(), os.path.join(run_dir, filename))

    def save_training_state(filename, stage, epoch):
        if acc.is_main_process:
            state_path = os.path.join(run_dir, filename.rsplit('.', 1)[0] + '.training.pt')
            torch.save({
                'optimizer': optm.state_dict(), 'scheduler': sch.state_dict(),
                'global_step': gs, 'stage': stage, 'epoch': epoch,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, state_path)

    @torch.no_grad()
    def validate(stage):
        model.eval()
        model._refiner_only = (stage == 'refiner')
        previous_length = getattr(model, 'effective_pose_length', None)
        if stage == 'planner':
            model.effective_pose_length = opt.pose_length
        total_loss, totals, seen = 0.0, {}, 0
        for batch in valid_dl:
            if not has_acc:
                batch = move_to_device(batch, device)
            out = model(batch, stage=stage)
            total_loss += out['loss'].detach().item()
            for key, value in out.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + value.detach().item()
            seen += 1
        if stage == 'planner':
            model.effective_pose_length = previous_length
        return total_loss / max(seen, 1), {key: value / max(seen, 1) for key, value in totals.items()}

    def run_stage(name, epochs, keys=('loss',)):
        nonlocal gs
        stage_best = float('inf')
        start_ep = 0
        if resume_state.get('stage') == name:
            start_ep = int(resume_state.get('epoch', -1)) + 1
        elif resume_state.get('stage') in ('refiner', 'joint') and name == 'planner':
            return
        elif resume_state.get('stage') == 'joint' and name == 'refiner':
            return
        for ep in range(start_ep, epochs):
            # Progressive trajectory-length curriculum (planner stage only)
            if name == 'planner':
                model.effective_pose_length = get_effective_pose_length(
                    ep, epochs,
                    start_len=opt.pose_start_len,
                    full_len=opt.pose_length,
                    ramp_epochs=opt.curriculum_ramp_epochs,
                )
            model.train()
            # See CineVLA.forward_refiner: detach planner outputs while the
            # refiner is pre-trained, but allow end-to-end gradients in joint.
            model._refiner_only = (name == 'refiner')
            tl, sl, seen = 0.0, {k: 0.0 for k in keys}, 0
            for batch in dl:
                if not has_acc:
                    batch = move_to_device(batch, device)
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

            # The lightweight fallback has no dataloader-end hook like
            # Accelerate.  Flush a final partial accumulation window instead
            # of silently discarding its gradients.
            if not has_acc and acc._step % opt.grad_accum:
                acc._should_step = True
                acc.clip_grad_norm_(model.parameters(), opt.grad_clip)
                optm.step()
                sch.step()
                optm.zero_grad()

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
            val_loss, val_components = validate(name)
            logger.info(f"[{name}] ep{ep} val={val_loss:.4f}")
            if wandb:
                wandb.log({'val/loss': val_loss, 'val/stage': name, 'val/epoch': ep,
                           **{f'val/{key}': value for key, value in val_components.items()}})
            if val_loss < stage_best:
                stage_best = val_loss
                acc.wait_for_everyone()
                save_model(f'best_{name}.safetensors')
                if name == 'joint':
                    save_model('best.safetensors')
            acc.wait_for_everyone()
            save_model('last.safetensors')
            save_training_state('last.safetensors', name, ep)

    run_stage('planner', opt.planner_pretrain_epochs,
              keys=('loss', 'L_rot', 'L_trans', 'L_rel', 'L_smooth'))
    run_stage('refiner', opt.refiner_pretrain_epochs,
              keys=('loss', 'loss_pose', 'loss_z', 'loss_rel', 'loss_smooth'))
    run_stage('joint', opt.joint_epochs,
              keys=('loss', 'loss_p', 'loss_r'))

    acc.wait_for_everyone()
    save_model('model.safetensors')
    logger.info("Saved.")

    if latent_logger is not None:
        latent_logger.finalize()
        logger.info("[visualise] Latent state summaries saved to pred_latent/")

    if wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
