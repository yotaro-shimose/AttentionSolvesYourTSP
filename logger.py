import tensorflow as tf
import os


class Logger:

    def __init__(self, logdir):
        self.pid = os.getpid()
        self.writer = tf.summary.create_file_writer(logdir=logdir)

    def log(self, metrics, step):
        with self.writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(str(self.pid) + key, value, step)
