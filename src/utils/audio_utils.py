"""
Audio utilities: loading, resampling, normalisation, and segmentation.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple


def load_audio(
    path: str,
    sample_rate: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Returns
    -------
    waveform : np.ndarray  shape (N,)
    sample_rate : int
    """
    waveform, sr = librosa.load(str(path), sr=sample_rate, mono=mono, duration=duration)
    return waveform.astype(np.float32), sr


def pad_or_truncate(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Pad (repeat) or truncate waveform to exactly target_length samples."""
    length = len(waveform)
    if length == target_length:
        return waveform
    if length > target_length:
        return waveform[:target_length]
    # Pad by repeating
    repeats = int(np.ceil(target_length / length))
    return np.tile(waveform, repeats)[:target_length]


def normalise_waveform(waveform: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Peak-normalise audio to [-1, 1]."""
    peak = np.abs(waveform).max()
    return waveform / (peak + eps)


def segment_audio(
    waveform: np.ndarray,
    sample_rate: int,
    clip_duration: float,
    hop_duration: float,
    min_duration: float = 0.5,
) -> list:
    """Split a long waveform into overlapping clips.

    Returns list of (start_sample, clip_waveform) tuples.
    """
    clip_len = int(clip_duration * sample_rate)
    hop_len = int(hop_duration * sample_rate)
    min_len = int(min_duration * sample_rate)
    segments = []
    start = 0
    while start + min_len <= len(waveform):
        end = start + clip_len
        clip = waveform[start:end]
        clip = pad_or_truncate(clip, clip_len)
        segments.append((start, clip))
        start += hop_len
    return segments


def rms_db(waveform: np.ndarray, eps: float = 1e-9) -> float:
    """Return RMS level in dBFS."""
    rms = np.sqrt(np.mean(waveform ** 2) + eps)
    return 20.0 * np.log10(rms)


def supported_audio_extensions() -> Tuple[str, ...]:
    return (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")


def list_audio_files(directory: str) -> list:
    """Recursively list all audio files in directory."""
    exts = supported_audio_extensions()
    return sorted([
        str(p) for p in Path(directory).rglob("*")
        if p.suffix.lower() in exts
    ])
