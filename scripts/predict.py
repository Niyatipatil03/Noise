#!/usr/bin/env python3
"""
Run inference on a single audio file using the saved Keras model.

Usage:
    python scripts/predict.py path/to/audio.wav
    python scripts/predict.py path/to/audio.wav --model models/bsr_detector_saved
    python scripts/predict.py path/to/audio.wav --tflite models/bsr_detector.tflite
"""

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


NOISE_TYPE_NAMES = [
    "OK",
    "IP_NOISE",
    "SUNROOF_NOISE",
    "REAR_NOISE",
    "STEERING_NOISE",
    "DOOR_NOISE",
    "GLASS_NOISE",
    "UNKNOWN_NOISE",
]


def load_cfg(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess_audio(path: str, cfg: dict) -> np.ndarray:
    """Load and convert audio to model input."""
    sr = cfg["audio"]["sample_rate"]
    clip_len = int(sr * cfg["audio"]["clip_duration"])

    waveform, _ = load_audio(path, sample_rate=sr)
    waveform = pad_or_truncate(waveform, clip_len)
    waveform = normalise_waveform(waveform)

    import math
    target_frames = math.ceil(clip_len / cfg["audio"]["hop_length"])

    spec = extract_spectrogram_for_model(
        waveform,
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
    return spec[np.newaxis, ...]  # (1, n_mels, T, 1)


def predict_keras(model_path: str, spec: np.ndarray):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "FocalLoss": __import__("src.models.losses", fromlist=["FocalLoss"]).FocalLoss,
            "CategoricalFocalLoss": __import__("src.models.losses", fromlist=["CategoricalFocalLoss"]).CategoricalFocalLoss,
        },
    )
    binary_pred, type_pred = model.predict(spec, verbose=0)
    return float(binary_pred[0, 0]), type_pred[0]


def predict_tflite(tflite_path: str, spec: np.ndarray):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    interpreter.set_tensor(in_details[0]["index"], spec.astype(np.float32))
    interpreter.invoke()

    # Find binary and type outputs
    binary_pred = None
    type_pred   = None
    for od in out_details:
        t = interpreter.get_tensor(od["index"])
        if t.shape[-1] == 1:
            binary_pred = float(t.flat[0])
        else:
            type_pred = t[0]
    return binary_pred, type_pred


def print_result(binary_score: float, type_scores: np.ndarray, threshold: float = 0.5):
    is_notok = binary_score > threshold
    status   = "NOT OK ⚠" if is_notok else "OK ✓"

    print("\n" + "=" * 50)
    print(f"  BSR Detection Result")
    print("=" * 50)
    print(f"  Status        : {status}")
    print(f"  Confidence    : {binary_score*100:.1f}% NOT_OK  /  {(1-binary_score)*100:.1f}% OK")

    if is_notok and type_scores is not None:
        top_idx = int(np.argmax(type_scores))
        top_name = NOISE_TYPE_NAMES[top_idx] if top_idx < len(NOISE_TYPE_NAMES) else f"TYPE_{top_idx}"
        print(f"\n  Noise Type    : {top_name}")
        print(f"  Type Confidence: {type_scores[top_idx]*100:.1f}%")
        print("\n  All noise-type probabilities:")
        for i, (name, score) in enumerate(zip(NOISE_TYPE_NAMES, type_scores)):
            bar = "█" * int(score * 20)
            print(f"    {name:<18} {bar:<20} {score*100:5.1f}%")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--model",   default=None, help="Keras SavedModel path")
    parser.add_argument("--tflite",  default=None, help="TFLite model path")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # Resolve model path
    tflite_path = args.tflite or cfg["paths"]["tflite_model"]
    keras_path  = args.model  or cfg["paths"]["saved_model"]

    print(f"[Predict] Processing: {args.audio_file}")
    spec = preprocess_audio(args.audio_file, cfg)

    if args.tflite or Path(tflite_path).exists():
        print(f"[Predict] Using TFLite model: {tflite_path}")
        binary_score, type_scores = predict_tflite(tflite_path, spec)
    elif Path(keras_path).exists():
        print(f"[Predict] Using Keras model: {keras_path}")
        binary_score, type_scores = predict_keras(keras_path, spec)
    else:
        print("[ERROR] No model found. Train first with: python scripts/train.py")
        sys.exit(1)

    print_result(binary_score, type_scores, threshold=args.threshold)


if __name__ == "__main__":
    main()
