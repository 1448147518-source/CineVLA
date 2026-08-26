"""Small shared utilities."""

import random

import numpy as np
import torch


def repeat_perception(perception: dict, repeats: int) -> dict:
    """Repeat a perception batch for batched classifier-free guidance.

    Text CFG needs conditional and unconditional samples to have the same
    visual context.  Repeating every tensor here avoids the silent batch-size
    mismatch that otherwise occurs inside cross-attention.
    """
    if repeats < 1:
        raise ValueError(f'repeats must be positive, got {repeats}')
    return {
        name: value.repeat_interleave(repeats, dim=0)
        if isinstance(value, torch.Tensor) else value
        for name, value in perception.items()
    }


def set_seed(seed: int):
    """Seed all local RNGs used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(filename):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger
