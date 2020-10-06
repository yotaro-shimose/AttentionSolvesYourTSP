import tensorflow as tf


class ResidualBatchNorm(tf.keras.models.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.batch_normalization = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x1 = tf.add(self.layer(x), x)
        return tf.add(self.batch_normalization(x1), x1)


class ResidualLayerNorm(tf.keras.models.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.layer_normalization = tf.keras.layers.LayerNormalization()

    def call(self, x):
        residual = x
        x = self.layer_normalization(x)
        return tf.add(self.layer(x), residual)
