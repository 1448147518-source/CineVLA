"""Frozen VAE encoder used only by the Refiner visual-state pathway."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinerVAEEncoder(nn.Module):
    """Encode RGB observations into spatial latent states.

    Planner perception remains CLIP-based.  This module is intentionally
    separate so the Refiner can reason over a reconstructive latent space
    without changing the Planner interface.
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse",
                 image_size: int = 224, freeze: bool = True):
        super().__init__()
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.image_size = image_size
        self.freeze = freeze
        if freeze:
            self.vae.eval()
            for parameter in self.vae.parameters():
                parameter.requires_grad = False

        self.latent_channels = int(getattr(self.vae.config, "latent_channels", 4))
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vae.eval()
        return self

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images, (self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        # Dataset frames are [0, 1]; AutoencoderKL expects [-1, 1].
        return images.mul(2.0).sub(1.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images [B,3,H,W] -> deterministic spatial latent [B,C,h,w]."""
        images = self._prepare(images)
        if self.freeze:
            with torch.no_grad():
                posterior = self.vae.encode(images).latent_dist
                latent = posterior.mode()
        else:
            posterior = self.vae.encode(images).latent_dist
            latent = posterior.mode()
        return latent * self.scaling_factor

    @torch.no_grad()
    def encode_frame(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.forward(image)[0]
