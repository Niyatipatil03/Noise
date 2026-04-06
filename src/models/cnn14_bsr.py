"""
CNN14-BSR: PANN-style CNN14 backbone adapted for vehicle BSR noise detection.

Architecture:
  Log-Mel Spectrogram (n_mels × T × 1)
  → 6 ConvBlocks (doubling channels: 64→128→256→512→1024→2048)
     each block: Conv2D → BN → ReLU → CBAM → AvgPool
  → Global Average Pooling
  → Shared Dense(512) + BN + Dropout
  → Head-1: Dense(1, sigmoid)           binary  OK/NOT_OK
  → Head-2: Dense(num_noise_types, softmax)  noise-type classification

Reference: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks
           for Audio Pattern Recognition", TASLP 2020.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

from src.models.attention import CBAM


def _conv_block(
    x,
    filters: int,
    use_attention: bool = True,
    l2: float = 1e-4,
    name: str = "block",
):
    """Two 3×3 convolutions + BN + ReLU + optional CBAM + AvgPool."""
    for i in range(2):
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(l2),
            name=f"{name}_conv{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i+1}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i+1}")(x)

    if use_attention:
        x = CBAM(filters, name=f"{name}_cbam")(x)

    x = layers.AveragePooling2D(pool_size=(2, 2), name=f"{name}_pool")(x)
    return x


def build_cnn14_bsr(
    input_shape: tuple,          # (n_mels, T, 1)
    num_noise_types: int = 8,
    embedding_dim: int = 512,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    use_attention: bool = True,
    name: str = "CNN14_BSR",
) -> Model:
    """Build and return the CNN14-BSR Keras model."""

    inp = layers.Input(shape=input_shape, name="spectrogram_input")

    # Initial batch normalisation on input spectrogram
    x = layers.BatchNormalization(name="input_bn")(inp)

    # ── 6 convolutional blocks ────────────────────────────────────────────────
    x = _conv_block(x, 64,   use_attention, l2_reg, name="block1")
    x = _conv_block(x, 128,  use_attention, l2_reg, name="block2")
    x = _conv_block(x, 256,  use_attention, l2_reg, name="block3")
    x = _conv_block(x, 512,  use_attention, l2_reg, name="block4")
    x = _conv_block(x, 1024, use_attention, l2_reg, name="block5")

    # Final conv without pool (feature map may be too small after 5 pools)
    for i in range(2):
        x = layers.Conv2D(
            2048, 3, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"block6_conv{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"block6_bn{i+1}")(x)
        x = layers.Activation("relu", name=f"block6_relu{i+1}")(x)

    if use_attention:
        x = CBAM(2048, name="block6_cbam")(x)

    # ── Global aggregation ────────────────────────────────────────────────────
    avg = layers.GlobalAveragePooling2D(name="gap")(x)
    mx  = layers.GlobalMaxPooling2D(name="gmp")(x)
    x   = layers.Add(name="gap_gmp_add")([avg, mx])

    # ── Shared embedding ──────────────────────────────────────────────────────
    x = layers.Dense(
        embedding_dim, use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="embedding_dense",
    )(x)
    x = layers.BatchNormalization(name="embedding_bn")(x)
    x = layers.Activation("relu", name="embedding_relu")(x)
    x = layers.Dropout(dropout_rate, name="embedding_dropout")(x)

    x2 = layers.Dense(
        embedding_dim // 2, use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="embedding2_dense",
    )(x)
    x2 = layers.BatchNormalization(name="embedding2_bn")(x2)
    x2 = layers.Activation("relu", name="embedding2_relu")(x2)
    x2 = layers.Dropout(dropout_rate * 0.75, name="embedding2_dropout")(x2)

    # ── Output heads ─────────────────────────────────────────────────────────
    binary_out = layers.Dense(1, activation="sigmoid", name="binary_out")(x2)
    type_out   = layers.Dense(num_noise_types, activation="softmax", name="type_out")(x2)

    model = Model(inputs=inp, outputs=[binary_out, type_out], name=name)
    return model


def build_cnn14_bsr_from_config(cfg: dict) -> Model:
    """Convenience wrapper that reads from config.yaml dict."""
    audio_cfg  = cfg["audio"]
    model_cfg  = cfg["model"]
    dataset_cfg = cfg["dataset"]

    import math
    # compute T dimension
    clip_samples = int(audio_cfg["sample_rate"] * audio_cfg["clip_duration"])
    T = math.ceil(clip_samples / audio_cfg["hop_length"])

    input_shape = (audio_cfg["n_mels"], T, 1)
    num_noise_types = len(dataset_cfg["noise_types"]) + 2  # +OK +UNKNOWN

    return build_cnn14_bsr(
        input_shape=input_shape,
        num_noise_types=num_noise_types,
        embedding_dim=model_cfg["embedding_dim"],
        dropout_rate=model_cfg["dropout_rate"],
        l2_reg=model_cfg["l2_reg"],
        use_attention=model_cfg.get("use_attention", True),
    )
