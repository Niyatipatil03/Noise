#!/usr/bin/env python3
"""
Main training script for BSR Noise Detector.

Usage:
    python scripts/train.py                          # use config.yaml defaults
    python scripts/train.py --backbone cnn14         # CNN14 (PANN-style)
    python scripts/train.py --backbone yamnet_lite   # YAMNet-like end-to-end
    python scripts/train.py --folds 5                # k-fold cross-validation
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from src.data.bsr_dataset import (
    BSRDataset, discover_files, train_val_split, save_labels, NUM_NOISE_TYPES
)
from src.models.cnn14_bsr   import build_cnn14_bsr_from_config
from src.models.yamnet_bsr  import build_yamnet_lite_bsr
from src.training.trainer   import compile_model, train, save_training_summary
from src.utils.visualization import plot_training_history, plot_confusion_matrix, print_classification_report


def parse_args():
    p = argparse.ArgumentParser(description="Train BSR Noise Detector")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--backbone", choices=["cnn14", "yamnet_lite"], default=None,
                   help="Override backbone model from config")
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--batch",    type=int, default=None)
    p.add_argument("--folds",    type=int, default=1,
                   help="k-fold cross-validation (1 = single split)")
    p.add_argument("--gpu",      default="0", help="CUDA_VISIBLE_DEVICES")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, backbone: str, input_shape, num_noise_types: int):
    """Instantiate the requested backbone."""
    if backbone == "yamnet_lite":
        sr = cfg["audio"]["sample_rate"]
        dur = cfg["audio"]["clip_duration"]
        return build_yamnet_lite_bsr(
            input_length=int(sr * dur),
            num_noise_types=num_noise_types,
            embedding_dim=cfg["model"]["embedding_dim"],
            dropout_rate=cfg["model"]["dropout_rate"],
            l2_reg=cfg["model"]["l2_reg"],
        )
    else:
        return build_cnn14_bsr_from_config(cfg)


def run_single(cfg, args, df, run_dir):
    """Single train/val split training."""
    train_df, val_df = train_val_split(
        df,
        test_size=cfg["dataset"]["test_size"],
        seed=cfg["dataset"]["random_seed"],
    )

    dataset = BSRDataset(cfg)
    input_shape = dataset.get_input_shape()
    backbone = args.backbone or cfg["model"]["backbone"]

    print(f"\n[Model] Backbone: {backbone}  |  Input shape: {input_shape}")

    train_ds, val_ds, class_weights = dataset.build(
        train_df, val_df, batch_size=cfg["training"]["batch_size"]
    )

    model = build_model(cfg, backbone, input_shape, NUM_NOISE_TYPES)
    model.summary(line_length=100)
    compile_model(model, cfg)

    history = train(model, train_ds, val_ds, cfg, run_dir, class_weights)
    save_training_summary(history, run_dir)

    # Plots
    plot_training_history(history, save_path=os.path.join(run_dir, "training_curves.png"))

    # Validation predictions
    evaluate_and_report(model, val_ds, run_dir)

    # Save final model
    saved_path = cfg["paths"]["saved_model"]
    model.save(saved_path)
    print(f"\n[Model] Saved to {saved_path}")

    return model


def evaluate_and_report(model, val_ds, run_dir):
    """Run predictions on val_ds and print classification report."""
    from src.data.bsr_dataset import BINARY_LABELS, NOISE_TYPE_LABELS

    y_bin_true, y_bin_pred = [], []
    y_type_true, y_type_pred = [], []

    for X_batch, labels in val_ds:
        bin_pred, type_pred = model.predict(X_batch, verbose=0)
        bin_true  = labels["binary_out"].numpy()
        type_true = np.argmax(labels["type_out"].numpy(), axis=1)

        y_bin_true.extend(bin_true.flatten())
        y_bin_pred.extend((bin_pred.flatten() > 0.5).astype(int))
        y_type_true.extend(type_true)
        y_type_pred.extend(np.argmax(type_pred, axis=1))

    # Binary report
    print_classification_report(
        np.array(y_bin_true, dtype=int),
        np.array(y_bin_pred, dtype=int),
        class_names=["OK", "NOT_OK"],
    )
    plot_confusion_matrix(
        np.array(y_bin_true, dtype=int),
        np.array(y_bin_pred, dtype=int),
        class_names=["OK", "NOT_OK"],
        title="Binary Confusion Matrix",
        save_path=os.path.join(run_dir, "cm_binary.png"),
    )

    # Noise-type report (on NOT_OK samples only)
    inv_type = {v: k for k, v in NOISE_TYPE_LABELS.items()}
    type_names = [inv_type[i] for i in range(len(NOISE_TYPE_LABELS))]
    print_classification_report(
        np.array(y_type_true, dtype=int),
        np.array(y_type_pred, dtype=int),
        class_names=type_names,
    )
    plot_confusion_matrix(
        np.array(y_type_true, dtype=int),
        np.array(y_type_pred, dtype=int),
        class_names=type_names,
        title="Noise-Type Confusion Matrix",
        save_path=os.path.join(run_dir, "cm_noise_type.png"),
    )


def run_kfold(cfg, args, df, run_dir):
    """K-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold

    k = args.folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg["dataset"]["random_seed"])
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df["binary_label"]), 1):
        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        Path(fold_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{k}")
        print(f"{'='*60}")

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        dataset = BSRDataset(cfg)
        input_shape = dataset.get_input_shape()
        backbone = args.backbone or cfg["model"]["backbone"]

        train_ds, val_ds, class_weights = dataset.build(
            train_df, val_df, batch_size=cfg["training"]["batch_size"]
        )

        model = build_model(cfg, backbone, input_shape, NUM_NOISE_TYPES)
        compile_model(model, cfg)

        history = train(model, train_ds, val_ds, cfg, fold_dir, class_weights)
        save_training_summary(history, fold_dir)

        best_auc = max(history.history.get("val_binary_out_auc", [0]))
        fold_scores.append(best_auc)
        print(f"[Fold {fold}] Best val AUC: {best_auc:.4f}")

    print(f"\n[K-Fold] AUC per fold: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"[K-Fold] Mean AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    with open(os.path.join(run_dir, "kfold_results.json"), "w") as f:
        json.dump({"fold_aucs": fold_scores, "mean": float(np.mean(fold_scores)),
                   "std": float(np.std(fold_scores))}, f, indent=2)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # GPU memory growth
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch:
        cfg["training"]["batch_size"] = args.batch

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["paths"]["output_dir"], f"run_{timestamp}")
    Path(cfg["paths"]["model_dir"]).mkdir(exist_ok=True)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # Labels file
    save_labels(cfg["paths"]["labels_file"])
    print(f"[Labels] Saved to {cfg['paths']['labels_file']}")

    # Discover dataset
    df = discover_files(cfg["dataset"]["root"], cfg["dataset"])
    if len(df) == 0:
        dataset_path = Path(cfg["dataset"]["root"]).resolve()
        print(f"\n  Copy your audio files to:")
        print(f"  {dataset_path / 'OK'}         ← all OK audio files")
        print(f"  {dataset_path / 'NOT_OK'}     ← all NOT OK audio files")
        sys.exit(1)

    # Train
    if args.folds > 1:
        run_kfold(cfg, args, df, run_dir)
    else:
        model = run_single(cfg, args, df, run_dir)


if __name__ == "__main__":
    main()
