import gpublock
import tensorflow as tf
from gat.reinforce.reinforce import Reinforce
import datetime
import pathlib


class ConsoleLogger:
    def log(self, metrics, step):
        print(f"Step: {step},   ", end="")
        for key, value in metrics.items():
            print(f"{key}: {value}   ", end="")
        print()


class TFLogger:

    def __init__(self, logdir):
        self.writer = tf.summary.create_file_writer(logdir=logdir)

    def log(self, metrics, step):
        with self.writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step)


if __name__ == '__main__':
    # Debugger V2
    # tf.debugging.experimental.enable_dump_debug_info(
    #     dump_root="./logs/",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1)
    date = datetime.datetime.today().strftime("%Y%m%d%H%M%S/")
    path = str(pathlib.Path("./logs/") / date)
    reinforce = Reinforce(logger=TFLogger(path))
    reinforce.start()
