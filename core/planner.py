"""Planner — generates an initial camera trajectory from visual context + text."""

import math
import torch
import torch.nn as nn
from contextlib import nullcontext


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
        self.freeze_text_encoder = freeze_text_encoder

        # ── Projections ──
        self.z_proj = nn.Linear(perception_dim, hidden_dim)
        self.text_proj = nn.Linear(768, hidden_dim)

        # ── Trajectory queries ──
        self.traj_queries = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, pose_length, hidden_dim) * 0.02)

        # ── Causal Transformer decoder layers ──
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_PlannerLayer(hidden_dim, num_heads))

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

    def train(self, mode=True):
        """Keep a frozen language backbone deterministic during optimisation."""
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def encode_text(self, texts, return_padding_mask=False):
        device = next(self.text_encoder.parameters()).device
        grad_ctx = torch.no_grad if all(not p.requires_grad for p in self.text_encoder.parameters()) else nullcontext
        with grad_ctx():
            tokens = self.tokenizer(texts, padding=True, truncation=True,
                                    return_tensors="pt", max_length=77)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            features = self.text_encoder(**tokens).last_hidden_state  # [B, L, 768]
        if return_padding_mask:
            # MultiheadAttention expects True for tokens that must be ignored.
            return features, ~tokens['attention_mask'].bool()
        return features

    def forward(self, perception: dict, texts):
        """
        Args:
            perception: from VideoPerceptionEncoder
            texts: list[str]
        """
        z_0_raw = perception['features_0']                      # [B, dim] causally-scoped frame_0
        B = z_0_raw.shape[0]
        device = z_0_raw.device
        z_0 = self.z_proj(z_0_raw).unsqueeze(1)                  # [B, 1, hidden]

        # Text
        text_feats, text_padding_mask = self.encode_text(texts, return_padding_mask=True)
        text_mem = self.text_proj(text_feats)  # [B, L_t, hidden]

        # Trajectory queries — prepend causally-scoped frame_0 feature as context token
        queries = self.traj_queries.expand(B, -1, -1) + self.pos_embed
        queries = torch.cat([z_0, queries], dim=1)  # [B, 1+N, hidden]

        # Causal mask
        total_len = 1 + self.pose_length
        tgt_mask = torch.triu(
            torch.ones(total_len, total_len, device=device) * float('-inf'), diagonal=1)

        # Layer stack
        for layer in self.layers:
            queries = layer(queries, tgt_mask, text_mem, text_padding_mask)

        poses = self.pose_head(self.norm(queries[:, 1:, :]))  # [B, N, 7]
        # A camera rotation is a point on S^3, not an unconstrained 4-vector.
        poses = torch.cat([torch.nn.functional.normalize(poses[..., :4], dim=-1, eps=1e-8),
                           poses[..., 4:]], dim=-1)
        return {'poses': poses, 'features': queries[:, 1:, :]}

    @torch.no_grad()
    def plan(self, perception: dict, texts) -> torch.Tensor:
        return self.forward(perception, texts)['poses'][0]


class _PlannerLayer(nn.Module):
    """One causal decoder layer conditioned on the text sequence."""

    def __init__(self, dim, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

        self.norm_text = nn.LayerNorm(dim)
        self.text_cross = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim * 4, dim), nn.Dropout(0.1),
        )

    def forward(self, x, tgt_mask, text_mem, text_padding_mask=None):
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                               attn_mask=tgt_mask)[0]
        # Text cross-attention
        x = x + self.text_cross(self.norm_text(x), text_mem, text_mem,
                                key_padding_mask=text_padding_mask)[0]
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x
