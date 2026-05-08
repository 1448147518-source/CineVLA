"""
Planner — generates initial trajectory from frame sequence features + text [+ music].

Input:  perception dict + text captions + optional music features
Output: planned trajectory [B, pose_length, 7]

Music influences the trajectory rhythm via cross-attention in upper layers,
enabling beat-synchronized camera movement for dance/concert filming.
"""

import math
import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import Optional


class Planner(nn.Module):
    def __init__(self, pose_dim=7, pose_length=30, perception_dim=512,
                 hidden_dim=256, num_layers=6, num_heads=4,
                 music_dim=128, music_ca_layers=2,
                 freeze_text_encoder=True):
        super().__init__()
        self.pose_dim = pose_dim
        self.pose_length = pose_length
        self.hidden_dim = hidden_dim
        self.music_ca_layers = music_ca_layers  # top N layers

        # ── Text encoder ──
        from transformers import CLIPTextModel, CLIPTokenizer
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if freeze_text_encoder:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # ── Music encoder (rhythm features) ──
        from core.music_encoder import MusicEncoder
        self.music_encoder = MusicEncoder(dim=music_dim, seq_len=pose_length)

        # ── Projections ──
        self.z_proj = nn.Linear(perception_dim, hidden_dim)
        self.text_proj = nn.Linear(768, hidden_dim)
        self.music_proj = nn.Linear(music_dim, hidden_dim)

        # ── Trajectory queries ──
        self.traj_queries = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)

        # ── Causal Transformer decoder layers ──
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_music_ca = (i >= num_layers - music_ca_layers)
            self.layers.append(
                _PlannerLayer(hidden_dim, num_heads, use_music_ca)
            )

        # ── Output ──
        self.norm = nn.LayerNorm(hidden_dim)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, pose_dim),
        )

        # ── Action stream weight initialization (LingBot-VA) ──
        # Initialize pose projection weights from perception projection,
        # scaling by sqrt(d_v / d_a) to preserve activation variance.
        with torch.no_grad():
            d_v = perception_dim
            d_a = hidden_dim
            scale = math.sqrt(d_v / d_a)

            # pose_head[0]: Linear(hidden_dim, hidden_dim)
            w0 = self.z_proj.weight.data[:, :hidden_dim] * scale
            self.pose_head[0].weight.data.copy_(w0)

            # pose_head[2]: Linear(hidden_dim, pose_dim)
            w2 = self.z_proj.weight.data[:pose_dim, :hidden_dim] * scale
            self.pose_head[2].weight.data.copy_(w2)

    def encode_text(self, texts):
        device = next(self.text_encoder.parameters()).device
        grad_ctx = torch.no_grad if all(not p.requires_grad for p in self.text_encoder.parameters()) else nullcontext
        with grad_ctx():
            tokens = self.tokenizer(texts, padding=True, truncation=True,
                                    return_tensors="pt", max_length=77)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            return self.text_encoder(**tokens).last_hidden_state  # [B, L, 768]

    def encode_music(self, music_path: Optional[str], device=None) -> Optional[torch.Tensor]:
        """Encode music file → [B, pose_length, music_dim], or None if no music."""
        if music_path is None:
            return None
        feats = self.music_encoder(music_path, device=device)  # [1, L, music_dim]
        return feats  # [1, L, music_dim]

    def forward(self, perception: dict, texts,
                music_path: Optional[str] = None,
                music_feats: Optional[torch.Tensor] = None):
        """
        Args:
            perception: from VideoPerceptionEncoder
            texts: list[str]
            music_path: optional path to MP3/WAV
            music_feats: optional pre-encoded music features [B, L, music_dim]
        """
        z_0_raw = perception['features_0']                      # [B, dim] causally-scoped frame_0
        B = z_0_raw.shape[0]
        device = z_0_raw.device
        z_0 = self.z_proj(z_0_raw).unsqueeze(1)                  # [B, 1, hidden]

        # Text
        text_feats = self.encode_text(texts)
        text_mem = self.text_proj(text_feats)  # [B, L_t, hidden]

        # Music
        if music_feats is None and music_path is not None:
            music_feats = self.encode_music(music_path, device)
        if music_feats is not None:
            if music_feats.shape[0] == 1 and B > 1:
                music_feats = music_feats.expand(B, -1, -1)
            music_mem = self.music_proj(music_feats)  # [B, L_m, hidden]
        else:
            music_mem = None

        # Trajectory queries — prepend causally-scoped frame_0 feature as context token
        queries = self.traj_queries.expand(B, -1, -1) + self.pos_embed
        queries = torch.cat([z_0, queries], dim=1)  # [B, 1+N, hidden]

        # Causal mask
        total_len = 1 + self.pose_length
        tgt_mask = torch.triu(
            torch.ones(total_len, total_len, device=device) * float('-inf'), diagonal=1)

        # Layer stack
        for layer in self.layers:
            queries = layer(queries, tgt_mask, text_mem, music_mem)

        poses = self.pose_head(self.norm(queries[:, 1:, :]))  # [B, N, 7]
        return {'poses': poses, 'features': queries[:, 1:, :]}

    @torch.no_grad()
    def plan(self, perception: dict, texts, music_path=None, music_feats=None) -> torch.Tensor:
        return self.forward(perception, texts, music_path, music_feats)['poses'][0]


class _PlannerLayer(nn.Module):
    """One decoder layer with optional music cross-attention."""

    def __init__(self, dim, num_heads, use_music_ca=False):
        super().__init__()
        self.use_music_ca = use_music_ca

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

        self.norm_text = nn.LayerNorm(dim)
        self.text_cross = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

        if use_music_ca:
            self.norm_music = nn.LayerNorm(dim)
            self.music_cross = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim * 4, dim), nn.Dropout(0.1),
        )

    def forward(self, x, tgt_mask, text_mem, music_mem=None):
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                               attn_mask=tgt_mask)[0]
        # Text cross-attention
        x = x + self.text_cross(self.norm_text(x), text_mem, text_mem)[0]
        # Music cross-attention (only in top layers)
        if self.use_music_ca and music_mem is not None:
            x = x + self.music_cross(self.norm_music(x), music_mem, music_mem)[0]
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x
