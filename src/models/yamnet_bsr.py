"""
YAMNet-BSR: Transfer learning using Google's YAMNet as a frozen feature extractor.

The YAMNet model (from TF-Hub) produces 1024-dimensional embeddings for each
0.96 s audio frame.  We aggregate those embeddings over time and add a
custom classification head for BSR noise detection.

Usage during training (needs TF-Hub):
    model = build_yamnet_bsr(yamnet_url, num_noise_types)

Usage for TFLite export:
    Call export_combined_tflite() which creates a single SavedModel
    that runs YAMNet preprocessing + custom head end-to-end.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


# ─── Custom YAMNet-compatible preprocessing (for offline TFLite) ─────────────

YAMNET_SAMPLE_RATE  = 16000
YAMNET_WIN_LEN      = 0.025   # 25 ms
YAMNET_HOP_LEN      = 0.010   # 10 ms
YAMNET_N_MELS       = 64
YAMNET_FMIN         = 125.0
YAMNET_FMAX         = 7500.0
YAMNET_N_FFT        = 512
YAMNET_EMBEDDING_DIM = 1024


def yamnet_log_mel(waveform: tf.Tensor) -> tf.Tensor:
    """Replicate YAMNet's log-mel spectrogram preprocessing.

    Parameters
    ----------
    waveform : tf.Tensor  shape (N,)  float32, 16 kHz
    Returns  : tf.Tensor  shape (frames, 64)
    """
    sr = YAMNET_SAMPLE_RATE
    win_samples = int(YAMNET_WIN_LEN * sr)   # 400
    hop_samples = int(YAMNET_HOP_LEN * sr)   # 160

    # STFT → power spectrum
    stft = tf.signal.stft(
        waveform,
        frame_length=win_samples,
        frame_step=hop_samples,
        fft_length=YAMNET_N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    magnitude = tf.abs(stft)
    power     = magnitude ** 2

    # Mel filterbank
    num_spec_bins = YAMNET_N_FFT // 2 + 1
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=YAMNET_N_MELS,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=sr,
        lower_edge_hertz=YAMNET_FMIN,
        upper_edge_hertz=YAMNET_FMAX,
    )
    mel = tf.tensordot(power, mel_matrix, 1)
    mel.set_shape([None, YAMNET_N_MELS])

    log_mel = tf.math.log(mel + 1e-6)
    return log_mel   # (frames, 64)


# ─── Lightweight CNN head on top of YAMNet embeddings ────────────────────────

def build_yamnet_bsr(
    num_noise_types: int = 8,
    embedding_dim: int = 256,
    dropout_rate: float = 0.5,
    l2_reg: float = 1e-4,
    yamnet_url: str = "https://tfhub.dev/google/yamnet/1",
    freeze_yamnet: bool = True,
) -> Model:
    """Build YAMNet + custom BSR head.

    The model expects a raw waveform input of shape (N,) at 16 kHz.
    YAMNet processes it frame-by-frame; we aggregate embeddings over time.
    """
    try:
        import tensorflow_hub as hub
    except ImportError:
        raise ImportError("tensorflow-hub is required for the YAMNet model.  "
                          "Install with: pip install tensorflow-hub")

    # Load YAMNet from TF-Hub
    yamnet_layer = hub.KerasLayer(yamnet_url, trainable=not freeze_yamnet, name="yamnet")

    # Input: raw waveform
    waveform_input = layers.Input(shape=(None,), dtype=tf.float32, name="waveform_input")

    # YAMNet returns (scores, embeddings, log_mel_spectrogram)
    _, embeddings, _ = yamnet_layer(waveform_input)
    # embeddings: (num_frames, 1024)

    # Aggregate over time
    avg_emb = layers.Lambda(lambda e: tf.reduce_mean(e, axis=0, keepdims=True),
                             name="avg_pool_time")(embeddings)
    max_emb = layers.Lambda(lambda e: tf.reduce_max(e, axis=0, keepdims=True),
                             name="max_pool_time")(embeddings)
    x = layers.Concatenate(axis=-1, name="emb_concat")([avg_emb, max_emb])
    # x: (1, 2048)
    x = layers.Flatten(name="flatten")(x)

    # Shared head
    x = layers.Dense(
        embedding_dim, use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc1",
    )(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout_rate, name="fc1_drop")(x)

    x = layers.Dense(
        embedding_dim // 2, use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc2",
    )(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation("relu", name="fc2_relu")(x)
    x = layers.Dropout(dropout_rate * 0.6, name="fc2_drop")(x)

    binary_out = layers.Dense(1, activation="sigmoid", name="binary_out")(x)
    type_out   = layers.Dense(num_noise_types, activation="softmax", name="type_out")(x)

    model = Model(inputs=waveform_input, outputs=[binary_out, type_out], name="YAMNet_BSR")
    return model


# ─── Self-contained TFLite-ready model (no TF-Hub dependency) ────────────────

def build_yamnet_lite_bsr(
    input_length: int = 48000,     # 3 s × 16 kHz
    num_noise_types: int = 8,
    embedding_dim: int = 256,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> Model:
    """YAMNet-like model that embeds its own log-mel front-end.

    Fully self-contained — no TF-Hub required at inference time.
    This is what gets exported to TFLite for the Android APK.

    Input  : waveform  shape (input_length,)  float32  16 kHz
    Outputs: binary_out (1,), type_out (num_noise_types,)
    """

    waveform_input = layers.Input(shape=(input_length,), dtype=tf.float32, name="waveform_input")

    # ── Log-Mel front-end (runs on-device) ────────────────────────────────────
    x = layers.Lambda(
        yamnet_log_mel, name="log_mel_frontend"
    )(waveform_input)
    # x: (frames, 64)  →  reshape to image (frames, 64, 1)
    x = layers.Lambda(lambda t: tf.expand_dims(t, -1), name="expand_dims")(x)
    # x: (frames, 64, 1)

    # ── CNN backbone ─────────────────────────────────────────────────────────
    filters = [64, 128, 256, 512]
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name=f"cnn_conv{i+1}a")(x)
        x = layers.BatchNormalization(name=f"cnn_bn{i+1}a")(x)
        x = layers.Activation("relu", name=f"cnn_relu{i+1}a")(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name=f"cnn_conv{i+1}b")(x)
        x = layers.BatchNormalization(name=f"cnn_bn{i+1}b")(x)
        x = layers.Activation("relu", name=f"cnn_relu{i+1}b")(x)
        x = layers.AveragePooling2D((2, 2), name=f"cnn_pool{i+1}")(x)

    avg = layers.GlobalAveragePooling2D(name="gap")(x)
    mx  = layers.GlobalMaxPooling2D(name="gmp")(x)
    x   = layers.Add(name="agg")([avg, mx])

    # ── Shared dense ─────────────────────────────────────────────────────────
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_regularizer=regularizers.l2(l2_reg), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout_rate, name="fc1_drop")(x)

    x = layers.Dense(embedding_dim // 2, use_bias=False,
                     kernel_regularizer=regularizers.l2(l2_reg), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation("relu", name="fc2_relu")(x)
    x = layers.Dropout(dropout_rate * 0.6, name="fc2_drop")(x)

    # ── Output heads ─────────────────────────────────────────────────────────
    binary_out = layers.Dense(1, activation="sigmoid", name="binary_out")(x)
    type_out   = layers.Dense(num_noise_types, activation="softmax", name="type_out")(x)

    model = Model(inputs=waveform_input, outputs=[binary_out, type_out],
                  name="YAMNetLite_BSR")
    return model
