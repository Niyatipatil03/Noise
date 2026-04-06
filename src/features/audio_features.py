"""
Feature extraction: log-mel spectrograms, MFCC, delta features.
All functions operate on raw waveforms (numpy float32 arrays).
"""

import numpy as np
import librosa


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """Compute log-mel spectrogram.

    Returns
    -------
    S : np.ndarray  shape (n_mels, T) — float32
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    return log_mel.astype(np.float32)


def normalise_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Normalise spectrogram to zero mean, unit variance per clip."""
    mean = spec.mean()
    std = spec.std() + 1e-9
    return ((spec - mean) / std).astype(np.float32)


def mfcc_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute MFCC + delta + delta-delta features.

    Returns
    -------
    features : np.ndarray  shape (3*n_mfcc, T)
    """
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sample_rate,
        n_mfcc=n_mfcc, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length,
    )
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta1, delta2], axis=0).astype(np.float32)


def extract_spectrogram_for_model(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    top_db: float = 80.0,
    target_frames: int = 299,
) -> np.ndarray:
    """Return log-mel spectrogram shaped for CNN input: (n_mels, T, 1).

    Pads or truncates T dimension to target_frames.
    """
    spec = log_mel_spectrogram(
        waveform, sample_rate=sample_rate,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, fmin=fmin, fmax=fmax, top_db=top_db,
    )
    spec = normalise_spectrogram(spec)

    # Ensure fixed time dimension
    if spec.shape[1] < target_frames:
        pad_width = target_frames - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :target_frames]

    return spec[..., np.newaxis]  # (n_mels, T, 1)


def extract_yamnet_input(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Prepare waveform for YAMNet (expects 16 kHz mono float32 in [-1,1])."""
    # YAMNet is happy with raw waveform; just ensure correct dtype
    return waveform.astype(np.float32)
