"""
Training logic: compile, fit, callbacks, k-fold cross-validation.
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, Tuple

from src.models.losses import FocalLoss, CategoricalFocalLoss


def compile_model(model: tf.keras.Model, cfg: dict) -> None:
    """Compile model with focal losses, metrics, and Adam optimizer."""
    lr = cfg["training"]["learning_rate"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss={
            "binary_out": FocalLoss(alpha=0.25, gamma=2.0),
            "type_out":   CategoricalFocalLoss(gamma=2.0),
        },
        loss_weights={
            "binary_out": 1.0,
            "type_out":   0.5,
        },
        metrics={
            "binary_out": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
            "type_out": [
                tf.keras.metrics.CategoricalAccuracy(name="cat_acc"),
            ],
        },
    )


def build_callbacks(cfg: dict, run_dir: str) -> list:
    """Return training callbacks list."""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, "best_model.h5"),
            monitor="val_binary_out_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_binary_out_auc",
            mode="max",
            factor=cfg["training"]["lr_factor"],
            patience=cfg["training"]["lr_patience"],
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_out_auc",
            mode="max",
            patience=cfg["training"]["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(run_dir, "training_log.csv"),
            append=True,
        ),
    ]

    # TensorBoard is optional — skip if not installed
    try:
        import tensorboard  # noqa: F401
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(run_dir, "tb_logs"),
                histogram_freq=0,
            )
        )
    except ImportError:
        print("[Callbacks] TensorBoard not installed — skipping (training will still work)")

    return callbacks


def _make_sample_weight_ds(
    ds: tf.data.Dataset,
    class_weights: Dict,
) -> tf.data.Dataset:
    """Convert class_weights dict → per-sample weight dataset.

    Keras supports class_weight only for single-output models.
    For multi-output models we attach sample weights to the dataset instead.
    """
    def attach_weight(X, labels):
        binary_label = labels["binary_out"]          # float32 scalar per sample
        # weight = class_weights[0] if label==0 else class_weights[1]
        w0 = tf.constant(class_weights[0], dtype=tf.float32)
        w1 = tf.constant(class_weights[1], dtype=tf.float32)
        sample_w = tf.where(binary_label < 0.5, w0, w1)
        return X, labels, sample_w

    return ds.map(attach_weight, num_parallel_calls=tf.data.AUTOTUNE)


def train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: dict,
    run_dir: str,
    class_weights: Optional[Dict] = None,
) -> tf.keras.callbacks.History:
    """Full training run."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    callbacks = build_callbacks(cfg, run_dir)

    # class_weight is NOT supported for multi-output models in Keras.
    # Use per-sample weights attached to the dataset instead.
    if class_weights and cfg["training"].get("class_weight_auto", True):
        train_ds = _make_sample_weight_ds(train_ds, class_weights)
        print(f"[Train] Sample weights applied: "
              f"OK={class_weights[0]:.2f}, NOT_OK={class_weights[1]:.2f}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_training_summary(history: tf.keras.callbacks.History, run_dir: str) -> None:
    """Save final metrics to JSON."""
    hist = history.history
    summary = {
        "best_val_auc":     float(max(hist.get("val_binary_out_auc", [0]))),
        "best_val_acc":     float(max(hist.get("val_binary_out_acc", [0]))),
        "best_val_recall":  float(max(hist.get("val_binary_out_recall", [0]))),
        "best_val_prec":    float(max(hist.get("val_binary_out_precision", [0]))),
        "total_epochs":     len(hist.get("loss", [])),
    }
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Training Summary]")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
