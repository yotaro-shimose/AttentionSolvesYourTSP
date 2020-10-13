import tensorflow as tf
from gat.modules.functions import masked_softmax
from gat.modules.layers.multi_head_masked_attention import MultiHeadMaskedAttention
from gat.modules.models.preprocessor import Preprocessor
from gat.modules.models.residual import ResidualLayerNorm
import numpy as np


class QDecoder(tf.keras.models.Model):
    def __init__(self, d_model, d_key, n_heads, weight_balancer=0.01):
        super().__init__()
        if (d_model % n_heads) != 0:
            raise ValueError('d_model must be multiple of n_heads!')
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


class PolicyDecoder(tf.keras.models.Model):
    def __init__(self, d_model, d_key, n_heads, th_range, weight_balancer=0.01):
        super().__init__()
        if (d_model % n_heads) != 0:
            raise ValueError('d_model must be multiple of n_heads!')
        self.d_model = d_model
        self.d_key = d_key
        self.attention = MultiHeadMaskedAttention(
            d_model, d_key, n_heads, weight_balancer)
        self.th_range = th_range
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

    def calc_policy(self, Q, K, V, mask):
        divide_const = tf.sqrt(tf.constant(K.shape[-1], dtype=tf.float32))
        QK = tf.matmul(Q, K, transpose_b=True) / divide_const
        # mask is tensor of shape (batch_size, n_nodes) by default.
        # but it must be tensor of shape (batch_size, 1, n_nodes).
        batch_size = Q.shape[0]
        n_nodes = V.shape[1]
        n_query = Q.shape[1]  # always one in ordinary use
        mask = tf.reshape(mask, (batch_size, n_query, n_nodes))
        policy = masked_softmax(
            self.th_range * tf.keras.activations.tanh(QK), mask)
        # now policy is tensor of shape(batch_size, 1, n_nodes) which must be turned into tensor of
        # shape(batch_size, n_nodes)
        return tf.reshape(policy, (batch_size, n_nodes))

    @tf.function
    def call(self, inputs):
        '''
        inputs ===[H (BATCH_SIZE, n_nodes, d_model), trajectory(BATCH_SIZE, n_nodes)]
        outputs === (BATCH_SIZE, n_nodes)
        '''
        inputs = self.preprocesser(inputs)
        output = self.attention(inputs)
        return self.calc_policy(tf.matmul(output, self.wq), tf.matmul(
            inputs[0], self.wk), inputs[2])
