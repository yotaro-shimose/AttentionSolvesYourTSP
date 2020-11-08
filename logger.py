import tensorflow as tf
import os


class TFLogger:

    def __init__(self, logdir):
        self.writer = tf.summary.create_file_writer(logdir=logdir)

    def log(self, metrics, step):
        with self.writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step)
            self.writer.flush()


class ConsoleLogger:

    def __init__(self, logdir):
        self.pid = os.getpid()

    def log(self, metrics, step):
        for key, value in metrics.items():
            print(f"{key} : {value}")
