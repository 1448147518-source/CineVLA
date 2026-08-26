"""Model-level contracts with tiny local stand-ins for external encoders."""

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 512)

    def forward(self, images):
        return self.proj(images.mean(dim=(-1, -2)))


class _FakeTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 768)

    @classmethod
    def from_pretrained(cls, _name):
        return cls()

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _name):
        return cls()

    def __call__(self, texts, **_kwargs):
        width = max(2, max(len(text.split()) for text in texts) + 1)
        ids = torch.ones(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            length = max(1, len(text.split()) + 1)
            ids[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return {'input_ids': ids, 'attention_mask': mask}


def _install_encoder_stubs(monkeypatch):
    clip_stub = types.ModuleType('clip')
    clip_stub.load = lambda *_args, **_kwargs: (types.SimpleNamespace(visual=_FakeVisual()), None)
    transformers_stub = types.ModuleType('transformers')
    transformers_stub.CLIPTextModel = _FakeTextModel
    transformers_stub.CLIPTokenizer = _FakeTokenizer
    monkeypatch.setitem(sys.modules, 'clip', clip_stub)
    monkeypatch.setitem(sys.modules, 'transformers', transformers_stub)


def _unit_quaternions(batch, length):
    q = F.normalize(torch.randn(batch, length, 4), dim=-1)
    return torch.cat([q, torch.randn(batch, length, 3)], dim=-1)


def test_all_training_stages_have_finite_loss_and_normalised_rotations(monkeypatch):
    _install_encoder_stubs(monkeypatch)
    from core.model import CineVLA
    from core.options import Options

    opt = Options(
        pose_length=4, num_frames=3, image_size=32, perception_dim=16,
        planner_hidden_dim=16, planner_num_layers=1, planner_num_heads=4,
        refiner_hidden_dim=16, refiner_num_layers=1, refiner_num_heads=4,
        refiner_lookahead=2,
    )
    model = CineVLA(opt).train()
    batch = {
        'frames': torch.rand(2, 3, 3, 32, 32),
        'poses': _unit_quaternions(2, 4),
        'text': ['slow dolly forward', 'pan left'],
    }

    for stage in ('planner', 'refiner', 'joint'):
        model.zero_grad(set_to_none=True)
        out = model(batch, stage=stage)
        assert torch.isfinite(out['loss'])
        out['loss'].backward()

    with torch.no_grad():
        planned = model.planner(model.perception(batch['frames']), batch['text'])['poses']
        assert torch.allclose(planned[..., :4].norm(dim=-1), torch.ones_like(planned[..., 0]), atol=1e-5)
