"""
Planner — generates the initial trajectory from first-frame perception + text [+ music].

Architecture: causal Transformer decoder that takes:
  - z_0 (initial environment perception)
  - text features (SD 2.1 text encoder)
  - music rhythm features (optional)
and outputs the planned trajectory [pose_length × pose_dim].

This runs ONCE at the start, before the closed-loop begins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from contextlib import nullcontext
from typing import Optional


class Planner(nn.Module):
    """
    Causal Transformer that generates 30-frame trajectory from initial conditions.
    """

    def __init__(self,
                 pose_dim: int = 7,
                 pose_length: int = 30,
                 perception_dim: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 text_ca_layers: int = 3,
                 music_dim: int = 128,
                 freeze_encoders: bool = True):
        super().__init__()
        self.pose_dim = pose_dim
        self.pose_length = pose_length
        self.hidden_dim = hidden_dim

        # ── Condition encoders ──
        self._init_text_encoder(freeze_encoders)
        self._init_music_encoder(music_dim)

        # ── Input projections ──
        self.z_proj = nn.Linear(perception_dim, hidden_dim)
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.text_proj = nn.Linear(1024, hidden_dim)

        # ── Learnable query tokens for trajectory ──
        self.traj_queries = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)
        self.z_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # ── Causal Transformer decoder ──
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.text_ca_layers = text_ca_layers

        # ── Output heads ──
        self.norm = nn.LayerNorm(hidden_dim)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, pose_dim),
        )

        # ── Positional encoding ──
        self.pos_embed = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)

    def _init_text_encoder(self, freeze: bool):
        from transformers import CLIPTextModel, CLIPTokenizer
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if freeze:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def _init_music_encoder(self, music_dim: int):
        self.music_proj = nn.Sequential(
            nn.Linear(256, music_dim), nn.SiLU(),  # takes from MusicEncoder output
        )

    def encode_text(self, texts):
        grad_ctx = torch.no_grad if all(not p.requires_grad for p in self.text_encoder.parameters()) else nullcontext
        with grad_ctx():
            tokens = self.tokenizer(texts, padding=True, truncation=True,
                                    return_tensors="pt", max_length=77)
            tokens = {k: v.to(next(self.text_encoder.parameters()).device) for k, v in tokens.items()}
            return self.text_encoder(**tokens).last_hidden_state  # [B, L, 768]

    def forward(self, z_0: torch.Tensor, texts, music_feats: Optional[torch.Tensor] = None) -> dict:
        """
        Generate planned trajectory.

        Args:
            z_0: [B, perception_dim] initial perception
            texts: list[str] text instructions
            music_feats: [B, music_seq, 256] optional rhythm features

        Returns:
            dict with 'poses' [B, pose_length, pose_dim], 'loss'
        """
        B = z_0.shape[0]
        device = z_0.device

        # ── Encode conditions ──
        z_tok = self.z_proj(z_0).unsqueeze(1)  # [B, 1, hidden]
        text_feats = self.encode_text(texts)     # [B, L_t, 768]
        text_mem = self.text_proj(text_feats)    # [B, L_t, hidden]

        # ── Prepare queries ──
        queries = self.traj_queries.expand(B, -1, -1) + self.pos_embed
        # prepend z_0 query
        z_q = self.z_query.expand(B, -1, -1)
        queries = torch.cat([z_q, queries], dim=1)  # [B, 1+N, hidden]

        # ── Self-attention with causal mask ──
        tgt_mask = torch.triu(
            torch.ones(1 + self.pose_length, 1 + self.pose_length, device=device) * float('-inf'),
            diagonal=1,
        )

        # ── Transformer ──
        x = self.transformer(tgt=queries, memory=text_mem, tgt_mask=tgt_mask)

        # ── Extract pose predictions ──
        pose_features = x[:, 1:, :]  # skip z_0 position
        poses = self.pose_head(self.norm(pose_features))  # [B, N, 7]

        return {'poses': poses, 'features': pose_features}

    @torch.no_grad()
    def plan(self, z_0: torch.Tensor, texts, music_feats=None) -> torch.Tensor:
        """Inference: return planned trajectory [N, 7]."""
        return self.forward(z_0, texts, music_feats)['poses'][0]
