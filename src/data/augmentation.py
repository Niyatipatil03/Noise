"""
Audio augmentation for small BSR datasets.
All transforms operate on raw waveforms (np.float32).
"""

import numpy as np
import librosa
from typing import List
import random


def time_shift(waveform: np.ndarray, sample_rate: int, max_fraction: float = 0.2) -> np.ndarray:
    """Randomly circular-shift the waveform in time."""
    shift = int(random.uniform(-max_fraction, max_fraction) * len(waveform))
    return np.roll(waveform, shift)


def pitch_shift(waveform: np.ndarray, sample_rate: int, n_steps: float = 2.0) -> np.ndarray:
    """Randomly shift pitch by ±n_steps semitones."""
    shift = random.uniform(-n_steps, n_steps)
    return librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=shift)


def time_stretch(waveform: np.ndarray, sample_rate: int, rates: List[float] = None) -> np.ndarray:
    """Randomly stretch time by one of the given rates, then resize back."""
    if rates is None:
        rates = [0.9, 1.0, 1.1]
    rate = random.choice(rates)
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    # Restore original length
    if len(stretched) > len(waveform):
        return stretched[:len(waveform)]
    pad = len(waveform) - len(stretched)
    return np.pad(stretched, (0, pad), mode="constant")


def add_gaussian_noise(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at the given SNR (dB)."""
    signal_rms = np.sqrt(np.mean(waveform ** 2) + 1e-9)
    noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise = np.random.normal(0.0, noise_rms, size=len(waveform))
    return (waveform + noise).astype(np.float32)


def random_gain(waveform: np.ndarray, gain_db_range: float = 6.0) -> np.ndarray:
    """Apply random gain in ±gain_db_range dB."""
    gain_db = random.uniform(-gain_db_range, gain_db_range)
    gain_lin = 10 ** (gain_db / 20.0)
    return (waveform * gain_lin).astype(np.float32)


def spec_augment(
    spec: np.ndarray,
    time_mask_param: int = 30,
    freq_mask_param: int = 20,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
) -> np.ndarray:
    """SpecAugment: random time and frequency masking on a spectrogram.

    Parameters
    ----------
    spec : np.ndarray  shape (n_mels, T) or (n_mels, T, 1)
    """
    squeeze = spec.ndim == 3
    if squeeze:
        spec = spec[..., 0]
    spec = spec.copy()
    n_mels, T = spec.shape

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, T))
        t0 = random.randint(0, T - t)
        spec[:, t0:t0 + t] = spec.mean()

    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, n_mels))
        f0 = random.randint(0, n_mels - f)
        spec[f0:f0 + f, :] = spec.mean()

    if squeeze:
        spec = spec[..., np.newaxis]
    return spec


def mixup_batch(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_type: np.ndarray,
    alpha: float = 0.4,
):
    """Mixup augmentation over a batch.

    Returns mixed X, y_binary, y_type (soft labels).
    """
    lam = np.random.beta(alpha, alpha, size=(len(X), 1, 1, 1)).astype(np.float32)
    idx = np.random.permutation(len(X))
    X_mix = lam * X + (1 - lam) * X[idx]
    lam2d = lam.squeeze()
    y_binary_mix = lam2d * y_binary + (1 - lam2d) * y_binary[idx]
    y_type_mix = lam2d[:, np.newaxis] * y_type + (1 - lam2d[:, np.newaxis]) * y_type[idx]
    return X_mix, y_binary_mix, y_type_mix


class WaveformAugmenter:
    """Pipeline that applies random augmentation to raw waveforms."""

    def __init__(self, cfg: dict, sample_rate: int = 16000):
        self.cfg = cfg
        self.sr = sample_rate

    def augment(self, waveform: np.ndarray) -> np.ndarray:
        aug = cfg = self.cfg

        # Time shift
        if random.random() < 0.5:
            waveform = time_shift(waveform, self.sr, cfg.get("time_shift_max", 0.2))

        # Pitch shift
        if random.random() < 0.4:
            waveform = pitch_shift(waveform, self.sr, cfg.get("pitch_shift_steps", 2))

        # Time stretch
        if random.random() < 0.4:
            waveform = time_stretch(waveform, self.sr, cfg.get("time_stretch_rates", [0.9, 1.1]))

        # Gaussian noise
        if random.random() < 0.5:
            snr = random.uniform(*cfg.get("gaussian_noise_snr_db", [20, 40]))
            waveform = add_gaussian_noise(waveform, snr)

        # Random gain
        if random.random() < 0.5:
            waveform = random_gain(waveform, 4.0)

        return waveform.astype(np.float32)

    def generate_augmented(self, waveform: np.ndarray, n: int) -> List[np.ndarray]:
        """Return n augmented copies."""
        return [self.augment(waveform.copy()) for _ in range(n)]
