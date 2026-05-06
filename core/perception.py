"""
Perception Encoder — runs at every closed-loop step.
Encodes a real RGB image into an environment latent z_t [perception_dim].

Uses a frozen CLIP ViT-B/32 backbone for efficiency, with a learnable
projection head that adapts the features for trajectory refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PerceptionEncoder(nn.Module):
    """
    Encode RGB image → environment latent z_t.

    Backbone: frozen CLIP ViT-B/32 (lightweight, 512-dim output)
    Projection: 2-layer MLP with LayerNorm
    """

    def __init__(self, dim: int = 512, image_size: int = 224, freeze_backbone: bool = True):
        super().__init__()
        self.dim = dim
        self.image_size = image_size

        # ── Backbone: CLIP ViT-B/32 ──
        import clip
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
        self.clip_model = self.clip_model.visual  # only vision part

        if freeze_backbone:
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # ── Projection head ──
        self.proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    @torch.no_grad()
    def _extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features from batch of images [B, 3, H, W]."""
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, (224, 224), mode='bilinear', align_corners=False)
        return self.clip_model(images)  # [B, 512]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode real camera image into environment latent.

        Args:
            images: [B, 3, H, W] RGB in [0, 1]

        Returns:
            z: [B, dim] environment perception latent
        """
        features = self._extract_clip_features(images)
        z = self.proj(features)
        return z

    @torch.no_grad()
    def perceive(self, image: torch.Tensor) -> torch.Tensor:
        """Inference wrapper: single image → latent."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.forward(image)


# ─────────────────────────────────────────────────────────────
#  Viewpoint warper — for training data synthesis
# ─────────────────────────────────────────────────────────────


def warp_viewpoint(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    source_pose: torch.Tensor,    # [4, 4] c2w matrix
    target_pose: torch.Tensor,    # [4, 4] c2w matrix
    intrinsics: torch.Tensor,     # [fx, fy, cx, cy]
    image_size: int = 224,
) -> torch.Tensor:
    """
    Synthesize the view at target_pose from a single source image + depth.

    Uses simple depth-based 3D warping (no learned inpainting).
    This produces approximate intermediate views for training the Refiner.

    Returns:
        warped_rgb: [3, image_size, image_size]
    """
    H, W = source_rgb.shape[-2], source_rgb.shape[-1]
    device = source_rgb.device

    # Camera intrinsics
    fx, fy, cx, cy = intrinsics.unbind(-1)

    # Pixel coordinate grid
    u = torch.arange(W, device=device).float()
    v = torch.arange(H, device=device).float()
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    uu, vv = uu.flatten(), vv.flatten()

    # Unproject to 3D world coordinates
    depth = source_depth.flatten()
    X = (uu - cx) * depth / fx
    Y = (vv - cy) * depth / fy
    Z = depth
    points_3d_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=0)  # [4, H*W]

    # Transform from source camera to world, then to target camera
    T_world_from_src = source_pose  # c2w
    T_cam_from_world = torch.inverse(target_pose)  # w2c
    points_3d_target = T_cam_from_world @ T_world_from_src @ points_3d_cam
    points_3d_target = points_3d_target[:3]  # [3, H*W]

    # Project to target image plane
    Xt, Yt, Zt = points_3d_target[0], points_3d_target[1], points_3d_target[2]
    valid = Zt > 1e-6
    ut = (Xt / Zt.clamp(min=1e-6)) * fx + cx
    vt = (Yt / Zt.clamp(min=1e-6)) * fy + cy
    ut = ut.clamp(0, W - 1).long()
    vt = vt.clamp(0, H - 1).long()

    # Scatter warp
    warped = torch.zeros(3, H, W, device=device)
    colors = source_rgb.reshape(3, -1)
    valid_mask = valid & (ut >= 0) & (ut < W) & (vt >= 0) & (vt < H)
    idx = vt[valid_mask] * W + ut[valid_mask]
    warped_flat = warped.reshape(3, -1)
    warped_flat[:, idx] = colors[:, valid_mask]

    if image_size != H:
        warped = F.interpolate(warped.unsqueeze(0), (image_size, image_size),
                               mode='bilinear', align_corners=False).squeeze(0)

    return warped
