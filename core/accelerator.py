"""Gradient-accumulation wrapper — mirrors the HuggingFace Accelerate API."""

import contextlib
import torch


class _SimpleAccelerator:
    def __init__(self, grad_accum=1):
        self.gradient_accumulation_steps = grad_accum
        self._step = 0
        self._should_step = True

    @property
    def is_main_process(self):
        return True

    def wait_for_everyone(self):
        pass

    def prepare(self, *args):
        return args

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
    def accumulate(self, model):
        yield


def get_accelerator(opt):
    try:
        from accelerate import Accelerator
        return Accelerator(mixed_precision=opt.mixed_precision,
                           gradient_accumulation_steps=opt.grad_accum), True
    except Exception:
        print("[WARN] using single GPU/CPU mode")
        return _SimpleAccelerator(opt.grad_accum), False
