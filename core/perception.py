"""
Video Perception Encoder — extracts 3D-aware features from RGB frame sequences.

Architecture:
  1. Per-frame: frozen CLIP ViT-B/32 extracts 512-dim features
  2. Temporal: lightweight transformer aggregates across frames
  3. Output: per-frame latents [T, dim] with cross-frame 3D understanding

No depth maps required. The model learns multi-view geometry implicitly
from RGB frame sequences, similar to monocular 3D reconstruction methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return an additive mask that prevents a timestep from reading the future."""
    return torch.triu(
        torch.full((length, length), float('-inf'), device=device), diagonal=1
    )


class VideoPerceptionEncoder(nn.Module):
    """
    Encode an RGB frame sequence into per-frame environment latents.

    Input:  [B, T, 3, H, W] frame sequence (T >= 1)
    Output: [B, T, dim] per-frame latents with temporal 3D context
    """

    def __init__(self, dim: int = 512, image_size: int = 224, freeze_backbone: bool = True,
                 temporal_layers: int = 3, temporal_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.image_size = image_size

        # ── Frame-level encoder: CLIP ViT-B/32 ──
        import clip
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_model = self.clip_model.visual
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # ── Per-frame projection ──
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, dim), nn.GELU(), nn.Linear(dim, dim),
        )

        # ── Temporal aggregator ──
        self.temporal_pos = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=temporal_heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.temporal_norm = nn.LayerNorm(dim)

    # CLIP ViT-B/32 normalization constants
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def _clip_features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B*T, 3, H, W] → [B*T, 512]"""
        if frames.shape[-1] != 224:
            frames = F.interpolate(frames, (224, 224), mode='bilinear', align_corners=False)
        frames = (frames - self.CLIP_MEAN.to(frames.device)) / self.CLIP_STD.to(frames.device)
        if self.freeze_backbone:
            with torch.no_grad():
                return self.clip_model(frames)
        return self.clip_model(frames)

    def train(self, mode=True):
        """Do not let a frozen CLIP backbone acquire train-time stochasticity."""
        super().train(mode)
        if self.freeze_backbone:
            self.clip_model.eval()
        return self

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames: [B, T, 3, H, W] RGB frame sequence

        Returns:
            dict with:
              'features':   [B, T, dim] per-frame latents (temporal-contextualized)
              'global':     [B, dim]    pooled global scene representation
              'features_0': [B, dim]    first-frame feature (before temporal)
        """
        B, T = frames.shape[:2]
        device = frames.device
        if T > self.temporal_pos.shape[1]:
            raise ValueError(f"Received {T} frames, but temporal encoder supports at most "
                             f"{self.temporal_pos.shape[1]}.")

        # Per-frame CLIP
        flat = frames.reshape(B * T, *frames.shape[2:])
        clip_feats = self._clip_features(flat)  # [B*T, 512]
        feats = self.frame_proj(clip_feats)      # [B*T, dim]
        feats = feats.reshape(B, T, self.dim)    # [B, T, dim]

        # Save first-frame raw feature (used by Planner as initial perception)
        features_0 = feats[:, 0, :]

        # Add temporal positional encoding
        feats = feats + self.temporal_pos[:, :T, :]

        # Temporal aggregation.  This mask is essential: the latent at t is
        # used by the online controller and must not encode frames t+1...T.
        feats = self.temporal(feats, mask=causal_attention_mask(T, device))
        feats = self.temporal_norm(feats)  # [B, T, dim]

        # The final causal state summarizes the available observation history.
        # Mean pooling would be unsuitable as an online state if callers append
        # future observations later.
        global_feat = feats[:, -1, :]  # [B, dim]

        return {
            'features': feats,
            'global': global_feat,
            'features_0': features_0,
        }

    @torch.no_grad()
    def encode_frame(self, image: torch.Tensor) -> torch.Tensor:
        """Single-frame inference: image [3, H, W] → latent [dim]."""
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        return self.forward(image)['global'][0]
