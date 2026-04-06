"""
BSR Dataset Loader.

Expected directory layout:
  dataset/
    OK/            ← audio files (any supported format)
    NOT_OK/
      IP_NOISE/
      SUNROOF_NOISE/
      REAR_NOISE/
      STEERING_NOISE/
      DOOR_NOISE/
      GLASS_NOISE/

OR a flat NOT_OK/ folder with all files when noise_type is unknown
(files will get label UNKNOWN_NOISE).
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

from src.utils.audio_utils import load_audio, pad_or_truncate, normalise_waveform, list_audio_files
from src.features.audio_features import extract_spectrogram_for_model
from src.data.augmentation import WaveformAugmenter, spec_augment


# ─── Label maps ───────────────────────────────────────────────────────────────

BINARY_LABELS = {"OK": 0, "NOT_OK": 1}

NOISE_TYPE_LABELS = {
    "OK":             0,   # no noise
    "IP_NOISE":       1,
    "SUNROOF_NOISE":  2,
    "REAR_NOISE":     3,
    "STEERING_NOISE": 4,
    "DOOR_NOISE":     5,
    "GLASS_NOISE":    6,
    "UNKNOWN_NOISE":  7,
}

NUM_NOISE_TYPES = len(NOISE_TYPE_LABELS)  # 8


def save_labels(path: str) -> None:
    labels = {
        "binary": BINARY_LABELS,
        "noise_type": NOISE_TYPE_LABELS,
    }
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


# ─── File discovery ───────────────────────────────────────────────────────────

def discover_files(root: str, cfg: dict) -> pd.DataFrame:
    """Crawl dataset root and return a DataFrame with columns:
    [file_path, binary_label, noise_type_label, noise_type_name]
    """
    records = []
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")

    ok_dir = Path(root) / cfg["ok_dir"]
    for p in ok_dir.rglob("*"):
        if p.suffix.lower() in exts:
            records.append({
                "file_path": str(p),
                "binary_label": BINARY_LABELS["OK"],
                "noise_type_label": NOISE_TYPE_LABELS["OK"],
                "noise_type_name": "OK",
            })

    notok_dir = Path(root) / cfg["notok_dir"]
    noise_types = cfg.get("noise_types", [])
    for nt in noise_types:
        sub = notok_dir / nt
        if sub.exists():
            for p in sub.rglob("*"):
                if p.suffix.lower() in exts:
                    records.append({
                        "file_path": str(p),
                        "binary_label": BINARY_LABELS["NOT_OK"],
                        "noise_type_label": NOISE_TYPE_LABELS.get(nt, NOISE_TYPE_LABELS["UNKNOWN_NOISE"]),
                        "noise_type_name": nt,
                    })

    # Fallback: flat NOT_OK directory (no subdirectories)
    flat_notok = [
        p for p in notok_dir.glob("*")
        if p.suffix.lower() in exts
    ]
    for p in flat_notok:
        records.append({
            "file_path": str(p),
            "binary_label": BINARY_LABELS["NOT_OK"],
            "noise_type_label": NOISE_TYPE_LABELS["UNKNOWN_NOISE"],
            "noise_type_name": "UNKNOWN_NOISE",
        })

    # Always create DataFrame with fixed columns even if empty
    if len(records) == 0:
        df = pd.DataFrame(columns=["file_path", "binary_label", "noise_type_label", "noise_type_name"])
    else:
        df = pd.DataFrame(records).drop_duplicates("file_path").reset_index(drop=True)

    ok_count  = int((df["binary_label"] == 0).sum()) if len(df) > 0 else 0
    nok_count = int((df["binary_label"] == 1).sum()) if len(df) > 0 else 0

    print(f"[Dataset] Found {len(df)} files: {ok_count} OK, {nok_count} NOT_OK")

    if len(df) == 0:
        abs_ok  = Path(root) / cfg["ok_dir"]
        abs_nok = Path(root) / cfg["notok_dir"]
        print("\n" + "="*60)
        print("  ERROR: No audio files found!")
        print("="*60)
        print(f"\n  Place your audio files in these folders:")
        print(f"  OK files   → {abs_ok.resolve()}")
        print(f"  NOT OK     → {abs_nok.resolve()}")
        print(f"\n  Supported formats: .wav  .mp3  .flac  .ogg  .m4a")
        print("="*60 + "\n")

    return df


def train_val_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split on binary label."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df["binary_label"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# ─── Sample loading ───────────────────────────────────────────────────────────

class BSRSample:
    """Loads and preprocesses a single audio sample."""

    def __init__(self, cfg: dict):
        self.sr = cfg["audio"]["sample_rate"]
        self.clip_len = int(cfg["audio"]["sample_rate"] * cfg["audio"]["clip_duration"])
        self.n_mels = cfg["audio"]["n_mels"]
        self.n_fft = cfg["audio"]["n_fft"]
        self.hop_length = cfg["audio"]["hop_length"]
        self.win_length = cfg["audio"]["win_length"]
        self.fmin = cfg["audio"]["fmin"]
        self.fmax = cfg["audio"]["fmax"]
        self.top_db = cfg["audio"]["top_db"]
        # Compute expected time frames: ceil(clip_len / hop_length)
        self.target_frames = int(np.ceil(self.clip_len / self.hop_length))

    def load(self, path: str) -> np.ndarray:
        waveform, _ = load_audio(path, sample_rate=self.sr)
        waveform = pad_or_truncate(waveform, self.clip_len)
        waveform = normalise_waveform(waveform)
        return waveform

    def to_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        return extract_spectrogram_for_model(
            waveform,
            sample_rate=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
            top_db=self.top_db,
            target_frames=self.target_frames,
        )


# ─── TF Dataset builder ───────────────────────────────────────────────────────

class BSRDataset:
    """Builds augmented tf.data.Dataset pipelines for training and validation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_proc = BSRSample(cfg)
        self.augmenter = WaveformAugmenter(
            cfg["augmentation"], sample_rate=cfg["audio"]["sample_rate"]
        ) if cfg["augmentation"]["enabled"] else None
        self.n_aug = cfg["augmentation"].get("num_augmented_per_sample", 5)
        self.num_noise_types = NUM_NOISE_TYPES

    # ── raw data building ─────────────────────────────────────────────────────

    def _build_arrays(
        self,
        df: pd.DataFrame,
        augment: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load all files → spectrograms.  Returns X, y_binary, y_type."""
        specs, y_bin, y_type = [], [], []

        for _, row in df.iterrows():
            try:
                wav = self.sample_proc.load(row["file_path"])
            except Exception as e:
                print(f"  [WARN] Skipping {row['file_path']}: {e}")
                continue

            wavs = [wav]
            if augment and self.augmenter:
                wavs += self.augmenter.generate_augmented(wav, self.n_aug)

            for w in wavs:
                spec = self.sample_proc.to_spectrogram(w)
                specs.append(spec)
                y_bin.append(row["binary_label"])
                y_type.append(row["noise_type_label"])

        X = np.array(specs, dtype=np.float32)
        yb = np.array(y_bin, dtype=np.float32)
        yt = np.array(y_type, dtype=np.int32)
        print(f"  Built array: X={X.shape}, OK={int((yb==0).sum())}, NOT_OK={int((yb==1).sum())}")
        return X, yb, yt

    # ── public interface ──────────────────────────────────────────────────────

    def build(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        batch_size: int = 16,
        use_spec_augment: bool = True,
    ):
        """Return (train_ds, val_ds, class_weights)."""

        print("[Dataset] Loading training set …")
        X_tr, yb_tr, yt_tr = self._build_arrays(train_df, augment=True)
        print("[Dataset] Loading validation set …")
        X_val, yb_val, yt_val = self._build_arrays(val_df, augment=False)

        # One-hot encode noise types
        yt_tr_oh  = tf.keras.utils.to_categorical(yt_tr,  self.num_noise_types)
        yt_val_oh = tf.keras.utils.to_categorical(yt_val, self.num_noise_types)

        # Class weights for binary head
        n_ok = int((yb_tr == 0).sum())
        n_nok = int((yb_tr == 1).sum())
        total = n_ok + n_nok
        class_weights = {0: total / (2.0 * n_ok), 1: total / (2.0 * n_nok)}
        print(f"  Class weights: OK={class_weights[0]:.2f}, NOT_OK={class_weights[1]:.2f}")

        # Build tf.data pipelines
        def make_ds(X, yb, yt_oh, shuffle: bool, apply_spec_aug: bool) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices((X, {"binary_out": yb, "type_out": yt_oh}))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
            if apply_spec_aug:
                time_p  = self.cfg["augmentation"]["spec_time_mask_param"]
                freq_p  = self.cfg["augmentation"]["spec_freq_mask_param"]

                def apply_sa(spec, labels):
                    spec_aug = tf.numpy_function(
                        lambda s: spec_augment(s, time_p, freq_p).astype(np.float32),
                        [spec], tf.float32,
                    )
                    spec_aug.set_shape(spec.shape)
                    return spec_aug, labels

                ds = ds.map(apply_sa, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = make_ds(X_tr, yb_tr, yt_tr_oh, shuffle=True, apply_spec_aug=use_spec_augment)
        val_ds   = make_ds(X_val, yb_val, yt_val_oh, shuffle=False, apply_spec_aug=False)

        return train_ds, val_ds, class_weights

    def get_input_shape(self) -> Tuple[int, int, int]:
        """Return spectrogram shape (n_mels, T, 1)."""
        t = self.sample_proc.target_frames
        return (self.sample_proc.n_mels, t, 1)
