"""Fast contract tests for the research-baseline invariants."""

import torch

from core.losses import refiner_loss
from core.perception import causal_attention_mask
from core.utils import repeat_perception
from core.dataset import split_items
from envs.offline_replay import OfflineReplayEnv
from envs.renderer import RendererCameraEnv


def test_causal_attention_mask_blocks_only_future_tokens():
    mask = causal_attention_mask(4, torch.device('cpu'))
    assert torch.isneginf(mask[0, 1])
    assert torch.isneginf(mask[1, 3])
    assert mask[3, 0] == 0
    assert mask.diag().eq(0).all()


def test_repeat_perception_matches_cfg_batch_size():
    perception = {
        'features': torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        'global': torch.ones(1, 3),
    }
    repeated = repeat_perception(perception, repeats=2)
    assert repeated['features'].shape == (2, 2, 3)
    assert torch.equal(repeated['features'][0], repeated['features'][1])


def test_refiner_is_supervised_by_future_target_not_input_plan():
    target = torch.tensor([[[1., 0., 0., 0., 0., 0., 0.]]])
    predicted_next = torch.zeros(1, 3)
    next_target = torch.zeros(1, 3)
    loss_at_target, _ = refiner_loss(target, target, predicted_next, next_target)
    wrong_refined = torch.ones(1, 1, 7)
    loss_away_from_target, _ = refiner_loss(
        wrong_refined, target, predicted_next, next_target
    )
    assert loss_at_target == 0
    assert loss_away_from_target > 0


def test_offline_replay_hides_future_frames_until_step():
    frames = torch.stack([torch.full((3, 2, 2), float(step)) for step in range(3)])
    environment = OfflineReplayEnv(frames)
    initial = environment.reset()
    assert initial.step == 0
    assert initial.rgb.eq(0).all()

    next_observation, terminated = environment.step(torch.zeros(7))
    assert next_observation.step == 1
    assert next_observation.rgb.eq(1).all()
    assert not terminated

    final_observation, terminated = environment.step(torch.zeros(7))
    assert final_observation.step == 2
    assert final_observation.rgb.eq(2).all()
    assert terminated


def test_renderer_environment_observes_the_submitted_action():
    rendered_poses = []

    def render(pose):
        rendered_poses.append(pose.clone())
        return torch.full((2, 2, 3), pose[-1].item())

    environment = RendererCameraEnv(
        render, initial_pose=torch.zeros(7), image_size=4, max_steps=1
    )
    initial = environment.reset()
    action = torch.tensor([0., 0., 0., 1., 0., 0., 0.8])
    observation, terminated = environment.step(action)

    assert initial.rgb.shape == (3, 4, 4)
    assert torch.equal(rendered_poses[-1], action)
    assert abs(observation.rgb.mean().item() - 0.8) < 1e-6
    assert terminated


def test_split_items_is_deterministic_and_disjoint():
    items = [f'sample-{index}' for index in range(10)]
    train_a, validation_a = split_items(items, validation_size=3, seed=7)
    train_b, validation_b = split_items(list(reversed(items)), validation_size=3, seed=7)
    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)
    assert len(train_a) == 7
    assert len(validation_a) == 3
