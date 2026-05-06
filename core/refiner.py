"""
Refiner — closed-loop trajectory correction driven by real visual feedback.

At each step t during inference:
  1. Camera captures image_t at current pose p_t
  2. Perception Encoder → z_t (real environment perception)
  3. Refiner compares z_t against predicted ẑ_t → perception error
  4. Uses error + remaining trajectory + context → refined future poses

Training: learns to correct "noisy" trajectories back to ground truth,
with perception features as guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Refiner(nn.Module):
    """
    Lightweight causal Transformer that refines the remaining trajectory
    at each closed-loop step based on real visual feedback.

    Input per step:
      - z_t: real perception from this step's camera image [perception_dim]
      - pred_z_t: what the model predicted z_t would be [perception_dim]
      - current_pose_t: [pose_dim]
      - remaining_plan: [remaining, pose_dim] from Planner
      - text_context: persistent text features

    Output:
      - refined_remaining: [remaining, pose_dim] adjusted trajectory
      - pred_z_next: [perception_dim] predicted next-step environment
      - confidence: [remaining] per-pose confidence
    """

    def __init__(self,
                 pose_dim: int = 7,
                 perception_dim: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 text_dim: int = 768):
        super().__init__()
        self.pose_dim = pose_dim
        self.perception_dim = perception_dim

        # ── Input projections ──
        self.z_proj = nn.Linear(perception_dim * 2, hidden_dim)  # z_t + pred_z_t concat
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # ── Error encoding ──
        self.error_encoder = nn.Sequential(
            nn.Linear(perception_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Causal Transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Cross-attention to text ──
        self.text_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)

        # ── Output heads ──
        self.norm = nn.LayerNorm(hidden_dim)
        self.pose_delta_head = nn.Linear(hidden_dim, pose_dim)     # pose correction delta
        self.z_pred_head = nn.Linear(hidden_dim, perception_dim)   # next perception prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid()                  # per-pose confidence
        )

        # ── Positional encoding ──
        self.pos_embed = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

    def forward(self,
                z_real: torch.Tensor,
                z_predicted: torch.Tensor,
                planned_poses: torch.Tensor,
                text_features: torch.Tensor,
                current_pose: Optional[torch.Tensor] = None,
                step_idx: int = 0,
                ) -> dict:
        """
        Training forward pass.

        Args:
            z_real:        [B, perception_dim] — real perception at current step
            z_predicted:   [B, perception_dim] — what model predicted for this step
            planned_poses: [B, remaining, pose_dim] — planner output from step onwards
            text_features: [B, L_t, text_dim] — persistent text context
            current_pose:  [B, pose_dim] — optional, actual pose at this step

        Returns:
            dict with 'pose_delta', 'z_next_pred', 'confidence', 'loss'
        """
        B, N, _ = planned_poses.shape
        device = planned_poses.device

        # ── Perception error ──
        z_concat = torch.cat([z_real, z_predicted], dim=-1)  # [B, 2×perception_dim]
        z_token = self.z_proj(z_concat).unsqueeze(1)  # [B, 1, hidden]

        error = z_real - z_predicted  # [B, perception_dim]
        error_token = self.error_encoder(error).unsqueeze(1)  # [B, 1, hidden]

        # ── Encode remaining trajectory ──
        pose_tokens = self.pose_proj(planned_poses)  # [B, N, hidden]
        pose_tokens = pose_tokens + self.pos_embed[:, :N, :]

        # ── Build sequence: [z_token, error_token, pose_1, pose_2, ...] ──
        seq = torch.cat([z_token, error_token, pose_tokens], dim=1)  # [B, 2+N, hidden]

        # ── Transformer ──
        seq = self.transformer(seq)

        # ── Cross-attention to text ──
        text_cond = self.text_proj(text_features)
        seq_norm = self.text_norm(seq)
        seq_cross, _ = self.text_cross_attn(seq_norm, text_cond, text_cond)
        seq = seq + seq_cross

        # ── Extract predictions ──
        seq = self.norm(seq)
        pose_features = seq[:, 2:, :]  # skip z and error tokens

        pose_delta = self.pose_delta_head(pose_features)  # [B, N, pose_dim]
        confidence = self.confidence_head(pose_features).squeeze(-1)  # [B, N]

        # Predict next-step perception (from last pose feature + z context)
        z_context = seq[:, 0, :]
        z_next_pred = self.z_pred_head(z_context + seq[:, 2:, :].mean(dim=1))

        # ── Loss (when ground truth available) ──
        # In training, planned_poses are the ground-truth remaining trajectory
        refined = planned_poses + pose_delta
        loss_pose = F.mse_loss(refined, planned_poses)  # should be close to GT
        loss_z = F.mse_loss(z_next_pred, z_real)  # predict next perception

        loss = loss_pose + 0.1 * loss_z

        return {
            'pose_delta': pose_delta,
            'refined': refined,
            'z_next_pred': z_next_pred,
            'confidence': confidence,
            'loss': loss,
            'loss_pose': loss_pose,
            'loss_z': loss_z,
        }

    @torch.no_grad()
    def refine(self,
               z_real: torch.Tensor,
               z_predicted: torch.Tensor,
               planned_poses: torch.Tensor,
               text_features: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference: refine remaining trajectory based on real perception.

        Returns:
            refined_poses: [remaining, pose_dim]
            z_next_pred: [perception_dim] predicted next environment
        """
        out = self.forward(z_real.unsqueeze(0), z_predicted.unsqueeze(0),
                           planned_poses.unsqueeze(0), text_features.unsqueeze(0))
        return out['refined'][0], out['z_next_pred'][0]
