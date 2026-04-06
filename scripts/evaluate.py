#!/usr/bin/env python3
"""
Detailed evaluation script — prints metrics, plots confusion matrices, ROC curve.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --tflite models/bsr_detector.tflite
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

import tensorflow as tf

from src.data.bsr_dataset import (
    BSRDataset, discover_files, train_val_split, NOISE_TYPE_LABELS, BINARY_LABELS
)
from src.utils.visualization import plot_confusion_matrix, print_classification_report


def load_cfg(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def gather_predictions(model_or_interp, val_ds, use_tflite: bool = False):
    y_bin_true, y_bin_score = [], []
    y_type_true, y_type_pred = [], []

    if use_tflite:
        interp = model_or_interp
        in_det  = interp.get_input_details()
        out_det = interp.get_output_details()

    for X_batch, labels in val_ds:
        X_np = X_batch.numpy()
        bin_true  = labels["binary_out"].numpy().flatten()
        type_true = np.argmax(labels["type_out"].numpy(), axis=1)

        if use_tflite:
            b_scores, t_scores = [], []
            for i in range(len(X_np)):
                x = X_np[i:i+1]
                interp.set_tensor(in_det[0]["index"], x)
                interp.invoke()
                for od in out_det:
                    t = interp.get_tensor(od["index"])
                    if t.shape[-1] == 1:
                        b_scores.append(float(t.flat[0]))
                    else:
                        t_scores.append(t[0])
            bin_score  = np.array(b_scores)
            type_score = np.array(t_scores)
        else:
            bin_score, type_score = model_or_interp.predict(X_np, verbose=0)
            bin_score = bin_score.flatten()

        y_bin_true.extend(bin_true)
        y_bin_score.extend(bin_score)
        y_type_true.extend(type_true)
        y_type_pred.extend(np.argmax(type_score, axis=1))

    return (
        np.array(y_bin_true, dtype=int),
        np.array(y_bin_score, dtype=float),
        np.array(y_type_true, dtype=int),
        np.array(y_type_pred, dtype=int),
    )


def plot_roc(y_true, y_score, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("BSR Detector — ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] ROC AUC = {roc_auc:.4f}  → saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--model",   default=None)
    parser.add_argument("--tflite",  default=None)
    parser.add_argument("--outdir",  default="outputs/eval")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Dataset
    df = discover_files(cfg["dataset"]["root"], cfg["dataset"])
    _, val_df = train_val_split(df, cfg["dataset"]["test_size"], cfg["dataset"]["random_seed"])
    dataset = BSRDataset(cfg)
    _, val_ds, _ = dataset.build(val_df, val_df, batch_size=16)

    # Load model
    use_tflite = False
    tflite_path = args.tflite or cfg["paths"]["tflite_model"]
    keras_path  = args.model  or cfg["paths"]["saved_model"]

    if args.tflite or Path(tflite_path).exists():
        print(f"[Eval] Using TFLite: {tflite_path}")
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        mdl = interp
        use_tflite = True
    else:
        from src.models.losses import FocalLoss, CategoricalFocalLoss
        mdl = tf.keras.models.load_model(
            keras_path,
            custom_objects={"FocalLoss": FocalLoss, "CategoricalFocalLoss": CategoricalFocalLoss},
        )
        print(f"[Eval] Using Keras model: {keras_path}")

    y_bin_true, y_bin_score, y_type_true, y_type_pred = gather_predictions(mdl, val_ds, use_tflite)
    y_bin_pred = (y_bin_score > 0.5).astype(int)

    # Reports
    print_classification_report(y_bin_true, y_bin_pred, ["OK", "NOT_OK"])
    plot_confusion_matrix(y_bin_true, y_bin_pred, ["OK", "NOT_OK"],
                          save_path=f"{args.outdir}/cm_binary.png")
    plot_roc(y_bin_true, y_bin_score, save_path=f"{args.outdir}/roc.png")

    inv = {v: k for k, v in NOISE_TYPE_LABELS.items()}
    type_names = [inv[i] for i in range(len(NOISE_TYPE_LABELS))]
    print_classification_report(y_type_true, y_type_pred, type_names)
    plot_confusion_matrix(y_type_true, y_type_pred, type_names,
                          title="Noise-Type Confusion Matrix",
                          save_path=f"{args.outdir}/cm_type.png")


if __name__ == "__main__":
    main()
