"""
Visualisation utilities: training curves, confusion matrices, spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(history, save_path: str = None):
    """Plot loss and key metrics over epochs."""
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("BSR Noise Detector — Training History", fontsize=14)

    pairs = [
        ("loss", "val_loss", "Total Loss"),
        ("binary_out_acc", "val_binary_out_acc", "Binary Accuracy"),
        ("binary_out_auc", "val_binary_out_auc", "Binary AUC"),
        ("binary_out_precision", "val_binary_out_precision", "Precision"),
        ("binary_out_recall", "val_binary_out_recall", "Recall"),
        ("type_out_cat_acc", "val_type_out_cat_acc", "Noise-Type Accuracy"),
    ]

    for ax, (tr_key, val_key, title) in zip(axes.flat, pairs):
        if tr_key in h:
            ax.plot(epochs, h[tr_key], "b-", label="train", linewidth=1.5)
        if val_key in h:
            ax.plot(epochs, h[val_key], "r--", label="val", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved training history → {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: str = None,
):
    """Plot normalised confusion matrix — only shows classes present in data."""
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    filtered_names = [
        class_names[i] for i in present_labels
        if i < len(class_names)
    ]

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(6, len(filtered_names)), max(5, len(filtered_names) - 1)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=filtered_names, yticklabels=filtered_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved confusion matrix → {save_path}")
    plt.close()
    return cm


def plot_spectrogram(
    spec: np.ndarray,
    title: str = "Log-Mel Spectrogram",
    save_path: str = None,
):
    """Visualise a single log-mel spectrogram."""
    if spec.ndim == 3:
        spec = spec[..., 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    ax.set_title(title)
    plt.colorbar(img, ax=ax, label="dB")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
):
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)

    # Only include names for classes actually present in the data
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    filtered_names = [
        class_names[i] for i in present_labels
        if i < len(class_names)
    ]

    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=filtered_names,
        zero_division=0,
    ))
