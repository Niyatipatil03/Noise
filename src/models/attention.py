"""
Attention mechanisms: Channel Attention, Spatial Attention, CBAM.
Used inside the CNN14 backbone to focus on BSR noise patterns.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        r = max(1, channels // reduction)
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.gmp = layers.GlobalMaxPooling2D(keepdims=True)
        self.fc1 = layers.Dense(r, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(channels, use_bias=False)
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x, training=None):
        avg = self.fc2(self.fc1(self.gap(x)))
        mx  = self.fc2(self.fc1(self.gmp(x)))
        scale = self.sigmoid(avg + mx)
        return x * scale


class SpatialAttention(layers.Layer):
    """Spatial attention using avg- and max-pool along the channel axis."""

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            1, kernel_size, padding="same", activation="sigmoid", use_bias=False
        )

    def call(self, x, training=None):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat   = tf.concat([avg_pool, max_pool], axis=-1)
        scale    = self.conv(concat)
        return x * scale


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (Woo et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def call(self, x, training=None):
        x = self.ca(x, training=training)
        x = self.sa(x, training=training)
        return x
