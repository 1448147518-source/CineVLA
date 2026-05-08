"""
Video Perception Encoder — extracts causal 3D-aware features from RGB frame sequences.

Causal paradigm (closed-loop):
  frame_0 → Planner initial trajectory
  frame_t → Refiner gets z_real_t (only sees frames ≤ t) → corrects remaining path

Architecture:
  1. Per-frame: frozen CLIP ViT-B/32 extracts 512-dim features
  2. Temporal: lightweight transformer with CAUSAL mask — frame_i only
     attends to frames [0..i], preventing future information leakage
  3. Output: per-frame latents [T, dim] where each frame's feature is
     causally scoped to its own timestamp and earlier

No depth maps required. The model learns multi-view geometry implicitly
from RGB frame sequences, similar to monocular 3D reconstruction methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if freeze_backbone:
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # ── Per-frame projection ──
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, dim), nn.GELU(), nn.Linear(dim, dim),
        )

        # ── Temporal aggregator ──
        self.max_frames = 16               # generous upper bound, not 64
        self.temporal_pos = nn.Parameter(torch.randn(1, self.max_frames, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=temporal_heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.temporal_norm = nn.LayerNorm(dim)

    @torch.no_grad()
    def _clip_features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B*T, 3, H, W] → [B*T, 512]"""
        size = self.image_size
        if frames.shape[-1] != size:
            frames = F.interpolate(frames, (size, size),
                                   mode='bilinear', align_corners=False)
        return self.clip_model(frames)

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames: [B, T, 3, H, W] RGB frame sequence

        Returns:
            dict with:
              'features':   [B, T, dim] per-frame latents, causally scoped
                            (frame_i only attends to frames [0..i])
              'global':     [B, dim]    pooled global scene representation
              'features_0': [B, dim]    first-frame causally-encoded feature
                            (only sees frame_0 itself, used for Planner init)
        """
        B, T = frames.shape[:2]
        device = frames.device

        # Per-frame CLIP
        flat = frames.reshape(B * T, *frames.shape[2:])
        clip_feats = self._clip_features(flat)  # [B*T, 512]
        feats = self.frame_proj(clip_feats)      # [B*T, dim]
        feats = feats.reshape(B, T, self.dim)    # [B, T, dim]

        # Add temporal positional encoding
        feats = feats + self.temporal_pos[:, :T, :]

        # ── Causal temporal aggregation ──
        # frame_i can only attend to frames [0..i]; future frames are masked.
        # Boolean mask: True = keep, False = mask.  tril gives lower triangle.
        causal_mask = torch.tril(
            torch.ones(T, T, device=device, dtype=torch.bool),
        )
        feats = self.temporal(feats, mask=causal_mask)
        feats = self.temporal_norm(feats)  # [B, T, dim]

        # First-frame causally-encoded feature (only sees frame_0)
        features_0 = feats[:, 0, :]

        # Global scene embedding (mean pool over time)
        global_feat = feats.mean(dim=1)  # [B, dim]

        return {
            'features': feats,
            'global': global_feat,
            'features_0': features_0,
        }
