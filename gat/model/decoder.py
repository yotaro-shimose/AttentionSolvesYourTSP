import tensorflow as tf

from gat.attention.multi_head_masked_attention import MultiHeadMaskedAttention
from gat.model.preprocessor import Preprocessor
from gat.model.residual import ResidualLayerNorm
import numpy as np


class Decoder(tf.keras.models.Model):
    def __init__(self, d_model, d_key, n_heads, weight_balancer=0.01):
        super().__init__()
        if (d_model % n_heads) != 0:
            raise ValueError('割り切れる数字を入れてね！！')
        self.d_model = d_model
        self.d_key = d_key
        self.attention = MultiHeadMaskedAttention(
            d_model, d_key, n_heads, weight_balancer)
        self.preprocesser = Preprocessor(d_model, d_key, n_heads)
        self.weight_balancer = weight_balancer

        self.residual_layer_norm = ResidualLayerNorm(
            tf.keras.layers.Dense(d_model))

    def build(self, input_shape):

        initializer = tf.random_uniform_initializer(
            -np.sqrt(6/(self.d_model + self.d_key)) * self.weight_balancer,
            np.sqrt(6/(self.d_model + self.d_key)) * self.weight_balancer
        )

        self.wq = self.add_weight(name="wq", shape=(self.d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

        self.wk = self.add_weight(name="wk", shape=(self.d_model, self.d_key),
                                  initializer=initializer,
                                  trainable=True)

    def calc_Q(self, Q, K):
        # calculate Q values based on queries and keys.
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = tf.squeeze(QK, axis=1)
        return QK

    @tf.function
    def call(self, inputs):
        '''
        inputs ===[H (BATCH_SIZE, n_nodes, d_model), trajectory(BATCH_SIZE, n_nodes)]
        outputs === (BATCH_SIZE, n_nodes)
        '''
        inputs = self.preprocesser(inputs)
        output = self.attention(inputs)

        return self.calc_Q(tf.matmul(output, self.wq), tf.matmul(
            inputs[0], self.wk))
