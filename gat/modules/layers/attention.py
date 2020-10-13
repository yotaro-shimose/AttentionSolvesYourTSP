import tensorflow as tf
import numpy as np


def attention(Q, K, V):
    divide_const = tf.sqrt(tf.cast(tf.constant(K.shape[-1]), tf.float32))
    return tf.matmul(tf.nn.softmax(tf.divide(tf.matmul(Q, K, transpose_b=True), divide_const)), V)


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_key, weight_balancer=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.weight_balancer = weight_balancer

    def build(self, input_shape):
        self.wq = self.add_weight(
            name="wq",
            shape=(input_shape[-1], self.d_key),
            initializer=tf.random_uniform_initializer(-np.sqrt(
                6/(input_shape[-1] + self.d_key)) * self.weight_balancer,
                np.sqrt(6/(input_shape[-1] + self.d_key)) * self.weight_balancer),
            trainable=True
        )
        self.wk = self.add_weight(
            name="wq",
            shape=(input_shape[-1], self.d_key),
            initializer=tf.random_uniform_initializer(-np.sqrt(
                6/(input_shape[-1] + self.d_key)) * self.weight_balancer,
                np.sqrt(6/(input_shape[-1] + self.d_key)) * self.weight_balancer),
            trainable=True
        )
        self.wv = self.add_weight(
            name="wv",
            shape=(input_shape[-1], self.d_model),
            initializer=tf.random_uniform_initializer(-np.sqrt(
                6/(input_shape[-1] + self.d_model)) * self.weight_balancer,
                np.sqrt(6/(input_shape[-1] + self.d_model)) * self.weight_balancer),
            trainable=True
        )

    def call(self, x):
        return attention(tf.matmul(x, self.wq), tf.matmul(
            x, self.wk), tf.matmul(x, self.wv))
