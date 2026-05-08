"""
Refiner — closed-loop trajectory correction driven by real camera frames.

At each step t during inference:
  1. Camera captures image_t → VideoPerceptionEncoder.encode_frame() → z_t
  2. Compare z_t vs predicted ẑ_t → perception error
  3. Refiner: error + remaining trajectory → refined poses + ẑ_{t+1}

Training: the Refiner learns to predict corrections from noisy-to-GT trajectory,
with frame features providing environment context. No depth needed.
"""

import torch
import torch.nn as nn


class Refiner(nn.Module):
    def __init__(self, pose_dim=7, perception_dim=512, hidden_dim=256,
                 num_layers=4, num_heads=4, text_dim=768):
        super().__init__()
        self.pose_dim = pose_dim
        self.perception_dim = perception_dim

        # ── Input projections ──
        self.z_proj = nn.Linear(perception_dim * 2, hidden_dim)  # z_t + ẑ_t
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # ── Error encoder ──
        self.error_encoder = nn.Sequential(
            nn.Linear(perception_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Text cross-attention ──
        self.text_cross = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.text_norm = nn.LayerNorm(hidden_dim)

        # ── Outputs ──
        self.norm = nn.LayerNorm(hidden_dim)
        self.pose_delta_head = nn.Linear(hidden_dim, pose_dim)
        self.z_pred_head = nn.Linear(hidden_dim, perception_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

    def forward(self,
                z_real: torch.Tensor,          # [B, dim] real frame feature at step t
                z_predicted: torch.Tensor,      # [B, dim] predicted feature for step t
                planned_poses: torch.Tensor,    # [B, remaining, 7]
                text_features: torch.Tensor,    # [B, L_t, text_dim]
                ) -> dict:
        B, N = planned_poses.shape[:2]
        device = planned_poses.device

        # Perception comparison + error
        z_cat = torch.cat([z_real, z_predicted], dim=-1)       # [B, 2*dim]
        z_token = self.z_proj(z_cat).unsqueeze(1)              # [B, 1, hidden]
        error = self.error_encoder(z_real - z_predicted).unsqueeze(1)  # [B, 1, hidden]

        # Encode remaining trajectory
        pose_tokens = self.pose_proj(planned_poses)
        pose_tokens = pose_tokens + self.pos_embed[:, :N, :]

        # Sequence: [z_token, error_token, pose_1, ..., pose_N]
        seq = torch.cat([z_token, error_token, pose_tokens], dim=1)  # [B, 2+N, hidden]
        seq = self.transformer(seq)

        # Text conditioning
        text_cond = self.text_proj(text_features)
        seq_norm = self.text_norm(seq)
        seq_cross, _ = self.text_cross(seq_norm, text_cond, text_cond)
        seq = seq + seq_cross

        seq = self.norm(seq)
        pose_feats = seq[:, 2:, :]  # skip z + error tokens

        pose_delta = self.pose_delta_head(pose_feats)

        # Predict next frame feature from context
        z_ctx = seq[:, 0, :] + seq[:, 2:, :].mean(dim=1)
        z_next = self.z_pred_head(z_ctx)

        refined = planned_poses + pose_delta

        return {
            'pose_delta': pose_delta,
            'refined': refined,
            'z_next_pred': z_next,
        }

    @torch.no_grad()
    def refine(self, z_real, z_predicted, planned_poses, text_features):
        out = self.forward(z_real.unsqueeze(0), z_predicted.unsqueeze(0),
                           planned_poses.unsqueeze(0), text_features.unsqueeze(0))
        return out['refined'][0], out['z_next_pred'][0]
