"""
Lightweight trajectory encoder — maps [N, 7] pose sequences to fixed-dim features.

Used as the feature extractor for FCD / PRDC / CLaTr Score computations.
Analogue of CLaTr's trajectory encoder; no external model dependency.
"""

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """
    Encodes a variable-length trajectory [N, 7] (quat wxyz + trans xyz)
    into a fixed-length feature vector [D].

    Architecture: Linear projection → small Transformer encoder → global pool → MLP head.
    """

    def __init__(self, pose_dim=7, hidden_dim=256, num_layers=2, num_heads=4,
                 output_dim=768, max_len=512):
        super().__init__()

        self.input_proj = nn.Linear(pose_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, trajectory):
        """
        Parameters
        ----------
        trajectory : torch.Tensor  shape [B, N, 7]  (quat wxyz + trans xyz)

        Returns
        -------
        torch.Tensor  shape [B, output_dim]
        """
        B, N, _ = trajectory.shape

        x = self.input_proj(trajectory)                     # [B, N, hidden]
        x = x + self.pos_embed[:, :N, :]                    # add position encoding
        x = self.transformer(x)                             # [B, N, hidden]
        x = self.norm(x).mean(dim=1)                        # global average pool
        x = self.output_proj(x)                             # [B, output_dim]
        return x
