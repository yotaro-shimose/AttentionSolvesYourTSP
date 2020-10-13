import tensorflow as tf


def masked_cross_entropy_from_Q(Q, Q_target, mask):
    p_target = masked_softmax(Q_target, mask)
    p = masked_softmax(Q, mask)
    return -tf.math.reduce_mean(tf.keras.layers.dot([p_target, masked_log(p, mask)], axes=1))


def masked_softmax(tensor, mask):
    tensor = tensor
    exps = tf.math.exp(tensor) * (1 - tf.cast(mask, tf.float32))
    softmax = exps / tf.math.reduce_sum(exps, -1, keepdims=True)
    return softmax


def masked_log(tensor, mask):
    float_mask = tf.cast(mask, tf.float32)
    log = tf.math.log((1 - float_mask) * tensor + float_mask * 1)
    return log


# compute mask for trajectory with shape(batch_size, node_size)
def create_mask(trajectory):
    def _create_mask(trajectory):
        tf_range = tf.range(tf.size(trajectory))
        return tf.map_fn(lambda x: tf.size(tf.where(trajectory == x))
                         != 0, tf_range, fn_output_signature=tf.bool)
    return tf.map_fn(_create_mask, trajectory, fn_output_signature=tf.bool)


def masked_argmax(tensor, mask):
    min = tf.math.reduce_min(tensor)
    return tf.argmax(tf.where(mask, min, tensor), axis=1, output_type=tf.int32)
