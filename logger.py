import tensorflow as tf


class Logger:

    def __init__(self, logdir):
        self.writer = tf.summary.create_file_writer(logdir=logdir)

    def log(self, metrics, step):
        with self.writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step)
