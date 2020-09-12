import tensorflow as tf
import numpy as np


class MaskedAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_key, weight_balancer):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.weight_balancer = weight_balancer

    def build(self, input_shape):
        self.wq = self.add_weight(
            name="wq",
            shape=(input_shape[1][-1], self.d_key),
            initializer=tf.random_uniform_initializer(
                -np.sqrt(6 / (input_shape[1][-1] +
                              self.d_key)) * self.weight_balancer,
                np.sqrt(6 / (input_shape[1][-1] +
                             self.d_key)) * self.weight_balancer
            ),
            trainable=True
        )
        self.wk = self.add_weight(
            name="wk",
            shape=(input_shape[0][-1], self.d_key),
            initializer=tf.random_uniform_initializer(
                -np.sqrt(6/(input_shape[0][-1] + self.d_key)
                         ) * self.weight_balancer,
                np.sqrt(6/(input_shape[0][-1] + self.d_key)
                        ) * self.weight_balancer
            ),
            trainable=True
        )
        self.wv = self.add_weight(
            name="wv",
            shape=(input_shape[0][-1], self.d_model),
            initializer=tf.random_uniform_initializer(
                -np.sqrt(6/(input_shape[0][-1] +
                            self.d_model)) * self.weight_balancer,
                np.sqrt(6/(input_shape[0][-1]+self.d_model)
                        ) * self.weight_balancer
            ),
            trainable=True
        )

    def masked_attention(self, Q, K, V, mask):
        divide_const = tf.sqrt(tf.cast(tf.constant(K.shape[-1]), tf.float32))
        QK = tf.matmul(Q, K, transpose_b=True)
        shape = tf.shape(QK)
        QK = tf.reshape(QK, (shape[1], shape[0], shape[-1]))

        masked_QK = tf.reshape(tf.where(mask, tf.float32.min, QK), shape)

        return tf.matmul(tf.nn.softmax(tf.divide(masked_QK, divide_const)), V)

    def call(self, inputs, training=None):
        return self.masked_attention(tf.matmul(inputs[1], self.wq), tf.matmul(
            inputs[0], self.wk), tf.matmul(inputs[0], self.wv), inputs[2])
