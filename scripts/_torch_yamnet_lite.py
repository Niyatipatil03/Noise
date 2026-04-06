"""
PyTorch YAMNet-lite BSR definition — mirrors src/models/yamnet_bsr.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class YAMNetLite_BSR(nn.Module):
    """Waveform-input model with log-mel front-end + 4-block CNN."""

    def __init__(self, input_length=48000, num_noise_types=8,
                 embedding_dim=256, dropout_rate=0.4):
        super().__init__()
        self.input_length = input_length

        # Mel front-end parameters (match config.yaml)
        self.sample_rate  = 16000
        self.n_fft        = 512
        self.hop_length   = 160
        self.n_mels       = 64
        self.fmin         = 125.0
        self.fmax         = 7500.0

        # Register mel filterbank as buffer (non-trainable)
        fb = self._mel_filterbank()
        self.register_buffer("mel_fb", fb)   # (n_mels, n_fft//2+1)

        # CNN backbone
        self.cnn = nn.Sequential(
            self._conv_block(1, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
        )

        self.fc1    = nn.Linear(512 * 2, embedding_dim, bias=False)
        self.bn1    = nn.BatchNorm1d(embedding_dim)
        self.drop1  = nn.Dropout(dropout_rate)
        self.fc2    = nn.Linear(embedding_dim, embedding_dim // 2, bias=False)
        self.bn2    = nn.BatchNorm1d(embedding_dim // 2)
        self.drop2  = nn.Dropout(dropout_rate * 0.6)

        self.binary_head = nn.Linear(embedding_dim // 2, 1)
        self.type_head   = nn.Linear(embedding_dim // 2, num_noise_types)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )

    def _mel_filterbank(self) -> torch.Tensor:
        def hz2mel(h): return 2595.0 * math.log10(1.0 + h / 700.0)
        def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        n_bins = self.n_fft // 2 + 1
        mel_lo = hz2mel(self.fmin)
        mel_hi = hz2mel(self.fmax)
        mel_pts = [mel2hz(mel_lo + i * (mel_hi - mel_lo) / (self.n_mels + 1))
                   for i in range(self.n_mels + 2)]
        bin_pts = [int(math.floor(f / self.sample_rate * 2 * (n_bins - 1)))
                   for f in mel_pts]

        fb = torch.zeros(self.n_mels, n_bins)
        for m in range(self.n_mels):
            for b in range(n_bins):
                lo, mid, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
                if lo <= b <= mid and mid > lo:
                    fb[m, b] = (b - lo) / (mid - lo)
                elif mid < b <= hi and hi > mid:
                    fb[m, b] = (hi - b) / (hi - mid)
        return fb

    # ── forward ──────────────────────────────────────────────────────────────

    def _log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, N) → log-mel: (B, 1, n_mels, T)"""
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            pad_mode="reflect",
        )                               # (B, n_fft//2+1, T)
        power = stft.abs() ** 2        # (B, F, T)
        mel   = torch.matmul(self.mel_fb, power)   # (B, n_mels, T)
        log_m = torch.log(mel + 1e-10)
        return log_m.unsqueeze(1)      # (B, 1, n_mels, T)

    def forward(self, x: torch.Tensor):
        # x: (B, N) raw waveform
        spec = self._log_mel(x)        # (B, 1, n_mels, T)
        feat = self.cnn(spec)

        avg = feat.mean(dim=[2, 3])
        mx  = feat.flatten(2).max(dim=2).values
        h   = torch.cat([avg, mx], dim=1)

        h = self.drop1(F.relu(self.bn1(self.fc1(h))))
        h = self.drop2(F.relu(self.bn2(self.fc2(h))))

        binary = torch.sigmoid(self.binary_head(h))
        ntype  = torch.softmax(self.type_head(h), dim=-1)
        return binary, ntype
