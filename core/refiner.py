"""
Refiner — VAE-latent world-model-guided closed-loop trajectory correction.

Planner perception stays CLIP-based.  The Refiner visual state is a spatial
VAE latent [B, C, H, W], allowing a lightweight convolutional discrepancy
encoder and a spatial latent-dynamics world model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        ) if half > 0 else t.new_zeros((0,))
        angles = t * frequencies[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class LearnableDiscrepancyEncoder(nn.Module):
    """Linear-cost comparator over spatial VAE latents.

    Inputs are [z_real, z_pred, delta, |delta|, cosine_map].  A pointwise
    bottleneck, depthwise 3x3 convolution and pointwise projection learn local
    mismatch patterns without cross-attention.
    """

    def __init__(self, latent_channels: int, hidden_dim: int):
        super().__init__()
        in_channels = latent_channels * 4 + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_real: torch.Tensor, z_pred: torch.Tensor):
        delta = z_real - z_pred
        cosine = F.cosine_similarity(z_real, z_pred, dim=1, eps=1e-6).unsqueeze(1)
        features = torch.cat([z_real, z_pred, delta, delta.abs(), cosine], dim=1)
        feature_map = self.net(features)
        token = self.norm(feature_map.mean(dim=(-1, -2)))
        return feature_map, token


class LatentDynamicsModel(nn.Module):
    """One-step spatial world model in frozen VAE latent space."""

    def __init__(self, latent_channels: int, pose_dim: int, hidden_dim: int):
        super().__init__()
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Conv2d(latent_channels, hidden_dim, 1)
        self.block = nn.Sequential(
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.out_proj = nn.Conv2d(hidden_dim, latent_channels, 1)

    def forward(self, z: torch.Tensor, planned_poses: torch.Tensor) -> torch.Tensor:
        if planned_poses.dim() == 2:
            planned_poses = planned_poses.unsqueeze(0)
        if planned_poses.shape[1] == 0:
            pose = torch.zeros(z.shape[0], self.pose_mlp[0].in_features,
                               device=z.device, dtype=z.dtype)
        else:
            pose = planned_poses.mean(dim=1)
        pose_cond = self.pose_mlp(pose).unsqueeze(-1).unsqueeze(-1)
        hidden = self.in_proj(z) + pose_cond
        return z + self.out_proj(self.block(hidden))


class FlowVelocityField(nn.Module):
    def __init__(self, pose_dim: int, hidden_dim: int, time_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.x_proj = nn.Linear(pose_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None, None]
        elif t.dim() == 2:
            t = t[:, :, None]
        t_flat = t[:, 0, 0]
        t_token = self.time_proj(self.time_embed(t_flat)).unsqueeze(1)
        t_token = t_token.expand(-1, x_t.shape[1], -1)
        x_token = self.x_proj(x_t)
        return self.net(torch.cat([x_token, context, t_token], dim=-1))


class Refiner(nn.Module):
    def __init__(self, pose_dim=7, latent_channels=4, hidden_dim=256,
                 num_layers=4, num_heads=4, text_dim=768,
                 flow_steps=4, correction_min_scale=0.25):
        super().__init__()
        self.pose_dim = pose_dim
        self.latent_channels = latent_channels
        self.flow_steps = max(1, int(flow_steps))
        self.correction_min_scale = float(correction_min_scale)

        self.discrepancy_encoder = LearnableDiscrepancyEncoder(latent_channels, hidden_dim)
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.text_cross = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.world_model = LatentDynamicsModel(latent_channels, pose_dim, hidden_dim)
        self.velocity_field = FlowVelocityField(pose_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

    @staticmethod
    def _align_target_quaternion_sign(target: torch.Tensor,
                                      planned: torch.Tensor) -> torch.Tensor:
        target_q = target[..., :4]
        planned_q = planned[..., :4]
        sign = torch.where(
            (target_q * planned_q).sum(dim=-1, keepdim=True) < 0,
            -torch.ones_like(target_q[..., :1]),
            torch.ones_like(target_q[..., :1]),
        )
        return torch.cat([target_q * sign, target[..., 4:]], dim=-1)

    def predict_next_latent(self, z_real: torch.Tensor,
                            planned_poses: torch.Tensor) -> torch.Tensor:
        single = z_real.dim() == 3
        if single:
            z_real = z_real.unsqueeze(0)
        if planned_poses.dim() == 2:
            planned_poses = planned_poses.unsqueeze(0)
        pred = self.world_model(z_real, planned_poses)
        return pred[0] if single else pred

    def _build_context(self, z_real: torch.Tensor, z_predicted: torch.Tensor,
                       planned_poses: torch.Tensor, text_features: torch.Tensor,
                       text_padding_mask: torch.Tensor = None):
        B, N = planned_poses.shape[:2]
        if N > self.pos_embed.shape[1]:
            raise ValueError(f"Refiner received {N} poses, max is {self.pos_embed.shape[1]}.")

        _, discrepancy_token = self.discrepancy_encoder(z_real, z_predicted)
        discrepancy = discrepancy_token.unsqueeze(1)
        pose_tokens = self.pose_proj(planned_poses) + self.pos_embed[:, :N, :]
        seq = torch.cat([discrepancy, pose_tokens], dim=1)
        seq = self.transformer(seq)

        text_cond = self.text_proj(text_features)
        seq_norm = self.text_norm(seq)
        seq_cross, _ = self.text_cross(
            seq_norm, text_cond, text_cond, key_padding_mask=text_padding_mask
        )
        seq = self.norm(seq + seq_cross)
        return seq[:, 1:, :], discrepancy_token

    def _sample_flow(self, context: torch.Tensor, dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
        B, N = context.shape[:2]
        x = torch.zeros(B, N, self.pose_dim, device=device, dtype=dtype)
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t = torch.full((B,), step * dt, device=device, dtype=dtype)
            x = x + dt * self.velocity_field(x, t, context)
        if N > 0:
            weights = torch.linspace(
                self.correction_min_scale, 1.0, N, device=device, dtype=dtype
            ).view(1, N, 1)
            x = x * weights
        return x

    def forward(self, z_real: torch.Tensor, z_predicted: torch.Tensor,
                planned_poses: torch.Tensor, text_features: torch.Tensor,
                text_padding_mask: torch.Tensor = None,
                target_poses: torch.Tensor = None) -> dict:
        B = planned_poses.shape[0]
        pose_context, discrepancy_token = self._build_context(
            z_real, z_predicted, planned_poses, text_features, text_padding_mask
        )

        flow_velocity = flow_target = flow_t = None
        if target_poses is not None:
            aligned_target = self._align_target_quaternion_sign(target_poses, planned_poses)
            target_delta = aligned_target - planned_poses
            x0 = torch.randn_like(target_delta)
            flow_t = torch.rand(B, device=planned_poses.device, dtype=planned_poses.dtype)
            t_view = flow_t.view(B, 1, 1)
            x_t = (1.0 - t_view) * x0 + t_view * target_delta
            flow_target = target_delta - x0
            flow_velocity = self.velocity_field(x_t, flow_t, pose_context)
            pose_delta = x_t + (1.0 - t_view) * flow_velocity
        else:
            pose_delta = self._sample_flow(pose_context, planned_poses.dtype, planned_poses.device)

        refined_raw = planned_poses + pose_delta
        refined = torch.cat([
            F.normalize(refined_raw[..., :4], dim=-1, eps=1e-8),
            refined_raw[..., 4:],
        ], dim=-1)
        z_next = self.world_model(z_real, planned_poses[:, :1])

        return {
            'pose_delta': pose_delta,
            'refined': refined,
            'z_next_pred': z_next,
            'discrepancy': discrepancy_token,
            'flow_velocity': flow_velocity,
            'flow_target': flow_target,
            'flow_t': flow_t,
        }

    @torch.no_grad()
    def refine(self, z_real, z_predicted, planned_poses, text_features,
               text_padding_mask=None):
        out = self.forward(
            z_real.unsqueeze(0), z_predicted.unsqueeze(0),
            planned_poses.unsqueeze(0), text_features.unsqueeze(0),
            text_padding_mask.unsqueeze(0) if text_padding_mask is not None else None,
        )
        return out['refined'][0], out['z_next_pred'][0]
