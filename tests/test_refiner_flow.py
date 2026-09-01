import torch
import torch.nn.functional as F

from core.refiner import LearnableDiscrepancyEncoder, Refiner


def _poses(batch, length):
    q = F.normalize(torch.randn(batch, length, 4), dim=-1)
    return torch.cat([q, torch.randn(batch, length, 3)], dim=-1)


def test_discrepancy_encoder_is_learnable_and_shape_stable():
    encoder = LearnableDiscrepancyEncoder(latent_channels=4, hidden_dim=16)
    z_real = torch.randn(3, 4, 8, 8, requires_grad=True)
    z_pred = torch.randn(3, 4, 8, 8)
    fmap, token = encoder(z_real, z_pred)
    assert fmap.shape == (3, 16, 8, 8)
    assert token.shape == (3, 16)
    token.square().mean().backward()
    assert z_real.grad is not None
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_refiner_flow_matching_training_contract():
    model = Refiner(
        pose_dim=7,
        latent_channels=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        text_dim=12,
        flow_steps=2,
    )
    planned = _poses(2, 3)
    target = _poses(2, 3)
    out = model(
        z_real=torch.randn(2, 4, 8, 8),
        z_predicted=torch.randn(2, 4, 8, 8),
        planned_poses=planned,
        text_features=torch.randn(2, 4, 12),
        target_poses=target,
    )
    assert out['refined'].shape == planned.shape
    assert out['flow_velocity'].shape == planned.shape
    assert out['flow_target'].shape == planned.shape
    assert torch.isfinite(out['flow_velocity']).all()
    assert torch.isfinite(out['discrepancy']).all()


def test_refiner_inference_integrates_flow_and_normalizes_rotation():
    model = Refiner(
        pose_dim=7,
        latent_channels=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        text_dim=12,
        flow_steps=2,
    ).eval()
    planned = _poses(1, 4)
    with torch.no_grad():
        out = model(
            z_real=torch.randn(1, 4, 8, 8),
            z_predicted=torch.randn(1, 4, 8, 8),
            planned_poses=planned,
            text_features=torch.randn(1, 5, 12),
        )
    assert out['refined'].shape == planned.shape
    norms = out['refined'][..., :4].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_latent_world_model_predicts_expected_spatial_shape():
    model = Refiner(
        pose_dim=7,
        latent_channels=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        text_dim=12,
    )
    z = torch.randn(2, 4, 8, 8)
    pose = _poses(2, 1)
    pred = model.predict_next_latent(z, pose)
    assert pred.shape == z.shape
    assert torch.isfinite(pred).all()
