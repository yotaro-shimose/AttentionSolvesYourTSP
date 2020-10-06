import tensorflow as tf


class ResidualBatchNorm(tf.keras.models.Model):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.batch_normalization = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x1 = tf.add(self.layer(x), x)
        return tf.add(self.batch_normalization(x1), x1)
