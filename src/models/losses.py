"""
Custom losses for multi-task BSR classifier.

- Focal Loss  : handles hard-to-detect low-intensity BSR sounds
- Multi-task Loss : weighted sum of binary + type losses
"""

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """Binary Focal Loss (Lin et al., 2017).

    Focuses training on hard, misclassified examples — ideal for subtle BSR noises.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce   = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        pt   = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * ce
        return tf.reduce_mean(focal)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma})
        return cfg


class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical Focal Loss for multi-class noise-type head."""

    def __init__(self, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce  = -y_true * tf.math.log(y_pred)
        pt  = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal = (1 - pt) ** self.gamma * ce
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma})
        return cfg
