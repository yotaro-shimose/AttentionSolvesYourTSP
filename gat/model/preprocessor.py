import tensorflow as tf


class Preprocessor(tf.keras.models.Model):

    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.n_heads = n_heads

    def build(self, input_shape):

        initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.0001, seed=None)

        self.v_init = self.add_weight(name="v_init", shape=(1, self.d_model),
                                      initializer=initializer,
                                      trainable=True)

        self.v_last = self.add_weight(name="v_last", shape=(1, self.d_model),
                                      initializer=initializer,
                                      trainable=True)

    def get_first(self, H, trajectory, v_first):
        """compute vector agent visited at the beginning of the episode.

        Args:
            H : output of encoder
            trajectory : trajectory with shape (batch_size, node_size)
            v_last : learnable vector to replace h_0 when no node have been visited.

        Returns:
            h_0: embedding vector representing node where agent visited at the beginning of the episode.
        """

        H_v = tf.concat([H, tf.broadcast_to(v_first, tf.concat(
            [tf.shape(H)[0:1], v_first.get_shape()], axis=0))], axis=1)
        f_index = trajectory[:, 0]
        f_index = tf.map_fn(lambda y: tf.cond(
            y == -1, lambda: H_v.shape[1] - 1, lambda: y), f_index)
        indice = tf.stack([tf.range(tf.size(f_index)), f_index], axis=1)
        return tf.gather_nd(H_v, indice)

    def get_last(self, H, trajectory, v_last):
        """compute vector agent visited at t-1

        Args:
            H : output of encoder
            trajectory : trajectory with shape (batch_size, node_size)
            v_last : learnable vector to replace h_(t-1) when no node have been visited.

        Returns:
            h_(t-1): embedding vector representing node where agent visited at t-1
        """
        H_v = tf.concat([H, tf.broadcast_to(v_last, tf.concat(
            [tf.shape(H)[0:1], v_last.get_shape()], axis=0))], axis=1)
        l_index = self.get_last_index(trajectory)
        indice = tf.stack([tf.range(tf.size(l_index)), l_index], axis=1)
        return tf.gather_nd(H_v, indice)

    def get_last_index(self, x):
        def _last(x):
            return tf.scan(lambda a, y: tf.cond(y == -1, lambda: a, lambda: y), x,
                           initializer=tf.constant(-1))[-1]
        value = tf.map_fn(_last, x)
        return tf.map_fn(lambda y: tf.cond(y == -1, lambda: x.shape[1], lambda: y), value)

    # compute mask for trajectory with shape(batch_size, node_size)
    def create_mask(self, trajectory):
        def _create_mask(trajectory):
            tf_range = tf.range(tf.size(trajectory))
            return tf.map_fn(lambda x: tf.size(tf.where(trajectory == x))
                             != 0, tf_range, fn_output_signature=tf.bool)
        return tf.map_fn(_create_mask, trajectory, fn_output_signature=tf.bool)

    def call(self, inputs, training=None):
        '''
        inputs === [H, trajectory]
        outputs === [H, h_c, mask]
        h_c === [h_g, h_0, h_t]
        '''
        # parse inputs.
        H = inputs[0]
        trajectory = inputs[1]

        # calculate h_c
        h_g = tf.reduce_mean(H, axis=1)
        first = self.get_first(H, trajectory, self.v_init)
        last = self.get_last(H, trajectory, self.v_last)
        h_c = tf.expand_dims(tf.concat([h_g, first, last], axis=1), axis=1)

        # calculate mask
        mask = self.create_mask(trajectory)

        # calculate probability over non-visited nodes
        return [H, h_c, mask]
