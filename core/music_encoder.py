"""Music rhythm encoder — extracts BPM + beat + onset features from MP3/WAV."""

import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicEncoder(nn.Module):
    def __init__(self, dim: int = 256, seq_len: int = 30):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.input_proj = nn.Linear(3, dim)  # BPM + onset + beat → dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2,
            dropout=0.1, batch_first=True)
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

    @torch.no_grad()
    def extract(self, path: str) -> torch.Tensor:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=22050, duration=30.0)
        if y is None or len(y) == 0 or np.max(np.abs(y)) < 1e-6:
            return torch.zeros(self.seq_len, 3)
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        onset_norm = onset / (onset.max() + 1e-8)
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        phases = np.zeros_like(onset_norm)
        for bt in beats:
            idx = min(int(bt * sr / 512), len(phases) - 1)
            phases[idx] = 1.0
        feats = np.stack([np.full_like(onset_norm, min(bpm / 180, 1.0)), onset_norm, phases], axis=-1)
        return torch.from_numpy(feats).float()

    def forward(self, path: str, device=None) -> torch.Tensor:
        if path is None or not os.path.exists(path):
            d = device or next(self.parameters()).device
            return torch.zeros(1, self.seq_len, self.dim, device=d)
        feats = self.extract(path)
        T = feats.shape[0]
        feats = feats.unsqueeze(0).permute(0, 2, 1)
        feats = F.interpolate(feats, self.seq_len, mode='linear', align_corners=False)
        feats = feats.permute(0, 2, 1)
        d = device or next(self.parameters()).device
        feats = feats.to(d)
        x = self.input_proj(feats) + self.pos[:, :self.seq_len, :]
        return self.temporal(x)
