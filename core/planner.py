"""
Planner — generates initial trajectory from frame sequence features + text.

Input:  perception dict {'features': [B,T,dim], 'global': [B,dim], 'features_0': [B,dim]}
        + text captions
Output: planned trajectory [B, pose_length, 7]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from contextlib import nullcontext
from typing import Optional


class Planner(nn.Module):
    def __init__(self, pose_dim=7, pose_length=30, perception_dim=512,
                 hidden_dim=256, num_layers=6, num_heads=4,
                 freeze_text_encoder=True):
        super().__init__()
        self.pose_dim = pose_dim
        self.pose_length = pose_length
        self.hidden_dim = hidden_dim

        # ── Text encoder ──
        from transformers import CLIPTextModel, CLIPTokenizer
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if freeze_text_encoder:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # ── Projections ──
        self.z_proj = nn.Linear(perception_dim, hidden_dim)
        self.text_proj = nn.Linear(768, hidden_dim)  # CLIP text dim

        # ── Learnable trajectory queries ──
        self.traj_queries = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)
        self.z_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)

        # ── Causal Transformer decoder ──
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ── Output ──
        self.norm = nn.LayerNorm(hidden_dim)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def encode_text(self, texts):
        device = next(self.text_encoder.parameters()).device
        grad_ctx = torch.no_grad if all(not p.requires_grad for p in self.text_encoder.parameters()) else nullcontext
        with grad_ctx():
            tokens = self.tokenizer(texts, padding=True, truncation=True,
                                    return_tensors="pt", max_length=77)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            return self.text_encoder(**tokens).last_hidden_state  # [B, L, 768]

    def forward(self, perception: dict, texts, music_feats=None):
        """
        Args:
            perception: from VideoPerceptionEncoder
                {'features': [B,T,dim], 'global': [B,dim], 'features_0': [B,dim]}
            texts: list[str]

        Returns:
            dict with 'poses' [B, N, 7], 'loss'
        """
        B = perception['global'].shape[0]
        device = perception['global'].device

        # Use global perception + first-frame feature as context
        z_global = self.z_proj(perception['global']).unsqueeze(1)  # [B, 1, hidden]
        z_0 = self.z_proj(perception['features_0']).unsqueeze(1)    # [B, 1, hidden]

        # Text conditioning
        text_feats = self.encode_text(texts)          # [B, L, 768]
        text_mem = self.text_proj(text_feats)          # [B, L, hidden]

        # Trajectory queries
        queries = self.traj_queries.expand(B, -1, -1) + self.pos_embed
        queries = torch.cat([self.z_query.expand(B, -1, -1), queries], dim=1)  # [B, 1+N, hidden]

        # Causal mask
        tgt_mask = torch.triu(
            torch.ones(1 + self.pose_length, 1 + self.pose_length, device=device) * float('-inf'),
            diagonal=1)

        x = self.transformer(tgt=queries, memory=text_mem, tgt_mask=tgt_mask)
        poses = self.pose_head(self.norm(x[:, 1:, :]))  # [B, N, 7]

        return {'poses': poses, 'features': x[:, 1:, :]}

    @torch.no_grad()
    def plan(self, perception: dict, texts) -> torch.Tensor:
        return self.forward(perception, texts)['poses'][0]
