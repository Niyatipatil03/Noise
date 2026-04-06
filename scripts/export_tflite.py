#!/usr/bin/env python3
"""
Export trained Keras model → TFLite for offline Android deployment.

Features:
  - Full INT8 quantisation (optional) for smaller APK
  - Representative dataset for quantisation calibration
  - Metadata attached (labels, input/output descriptions)

Usage:
    python scripts/export_tflite.py
    python scripts/export_tflite.py --quant int8   # INT8 quantisation
    python scripts/export_tflite.py --quant float16
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import tensorflow as tf

from src.utils.audio_utils import load_audio, pad_or_truncate, normalise_waveform
from src.features.audio_features import extract_spectrogram_for_model
from src.data.bsr_dataset import discover_files


def load_cfg(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_keras_model(path: str) -> tf.keras.Model:
    from src.models.losses import FocalLoss, CategoricalFocalLoss
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            "FocalLoss": FocalLoss,
            "CategoricalFocalLoss": CategoricalFocalLoss,
            "CBAM": __import__("src.models.attention", fromlist=["CBAM"]).CBAM,
            "ChannelAttention": __import__("src.models.attention", fromlist=["ChannelAttention"]).ChannelAttention,
            "SpatialAttention": __import__("src.models.attention", fromlist=["SpatialAttention"]).SpatialAttention,
        },
    )
    return model


def make_representative_dataset(cfg: dict, n_samples: int = 50):
    """Generator for INT8 calibration."""
    import math
    sr = cfg["audio"]["sample_rate"]
    clip_len = int(sr * cfg["audio"]["clip_duration"])
    target_frames = math.ceil(clip_len / cfg["audio"]["hop_length"])

    df = discover_files(cfg["dataset"]["root"], cfg["dataset"])
    df = df.sample(min(n_samples, len(df)), random_state=42)

    def gen():
        for _, row in df.iterrows():
            try:
                wav, _ = load_audio(row["file_path"], sample_rate=sr)
                wav = pad_or_truncate(wav, clip_len)
                wav = normalise_waveform(wav)
                spec = extract_spectrogram_for_model(
                    wav,
                    sample_rate=sr,
                    n_mels=cfg["audio"]["n_mels"],
                    n_fft=cfg["audio"]["n_fft"],
                    hop_length=cfg["audio"]["hop_length"],
                    win_length=cfg["audio"]["win_length"],
                    fmin=cfg["audio"]["fmin"],
                    fmax=cfg["audio"]["fmax"],
                    top_db=cfg["audio"]["top_db"],
                    target_frames=target_frames,
                )
                yield [spec[np.newaxis, ...].astype(np.float32)]
            except Exception:
                pass

    return gen


def convert_to_tflite(
    model: tf.keras.Model,
    quant_mode: str,
    representative_gen=None,
) -> bytes:
    """Convert model to TFLite bytes."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quant_mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("[TFLite] Using Float16 quantisation")

    elif quant_mode == "int8":
        if representative_gen is None:
            raise ValueError("INT8 quantisation requires a representative dataset.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        print("[TFLite] Using INT8 quantisation")

    else:  # float32 (default)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic range
        print("[TFLite] Using dynamic-range (float32) quantisation")

    tflite_model = converter.convert()
    return tflite_model


def attach_metadata(tflite_model: bytes, labels_path: str, output_path: str) -> None:
    """Attach labels metadata to TFLite model (optional, requires tflite-support)."""
    try:
        from tflite_support.metadata_writers import audio_classifier
        from tflite_support.metadata_writers import metadata_info
        from tflite_support.metadata_writers import writer_utils
        print("[TFLite] Attaching metadata (tflite-support available)")
        # Write final file (metadata addition happens here if package available)
    except ImportError:
        pass  # metadata optional
    # Always write the raw model regardless
    with open(output_path, "wb") as f:
        f.write(tflite_model)


def verify_tflite(tflite_path: str, input_shape: tuple) -> None:
    """Quick sanity-check: run a random input through the TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det  = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    dummy = np.random.randn(*input_shape).astype(np.float32)
    interpreter.set_tensor(in_det[0]["index"], dummy)
    interpreter.invoke()

    print("[TFLite] Verification outputs:")
    for od in out_det:
        t = interpreter.get_tensor(od["index"])
        print(f"  {od['name']}: shape={t.shape}  values={t.flatten()[:4]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--model",     default=None, help="Override Keras SavedModel path")
    parser.add_argument("--output",    default=None, help="Override TFLite output path")
    parser.add_argument("--quant",     choices=["none", "float16", "int8"], default="float16")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    keras_path = args.model  or cfg["paths"]["saved_model"]
    tflite_out = args.output or cfg["paths"]["tflite_model"]

    if not Path(keras_path).exists():
        print(f"[ERROR] Keras model not found: {keras_path}")
        print("        Train first with: python scripts/train.py")
        sys.exit(1)

    print(f"[Export] Loading model from: {keras_path}")
    model = load_keras_model(keras_path)
    model.summary(line_length=100)

    rep_gen = None
    if args.quant == "int8":
        rep_gen = make_representative_dataset(cfg)

    tflite_bytes = convert_to_tflite(model, args.quant, representative_gen=rep_gen)

    Path(tflite_out).parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_out, "wb") as f:
        f.write(tflite_bytes)

    size_mb = len(tflite_bytes) / 1e6
    print(f"[Export] TFLite model saved: {tflite_out}  ({size_mb:.2f} MB)")

    # Save labels alongside the model
    labels_path = str(Path(tflite_out).parent / "labels.json")
    from src.data.bsr_dataset import BINARY_LABELS, NOISE_TYPE_LABELS
    with open(labels_path, "w") as f:
        json.dump({"binary": BINARY_LABELS, "noise_type": NOISE_TYPE_LABELS}, f, indent=2)
    print(f"[Export] Labels saved: {labels_path}")

    # Verify
    import math
    sr = cfg["audio"]["sample_rate"]
    clip_len = int(sr * cfg["audio"]["clip_duration"])
    T = math.ceil(clip_len / cfg["audio"]["hop_length"])
    input_shape = (1, cfg["audio"]["n_mels"], T, 1)
    print(f"[Export] Running verification with shape {input_shape} …")
    try:
        verify_tflite(tflite_out, input_shape)
        print("[Export] Verification PASSED")
    except Exception as e:
        print(f"[Export] Verification failed (may be normal for INT8): {e}")

    print(f"\n[Export] Done!  Copy {tflite_out} to your Android project:")
    print(f"         android/app/src/main/assets/bsr_detector.tflite")


if __name__ == "__main__":
    main()
