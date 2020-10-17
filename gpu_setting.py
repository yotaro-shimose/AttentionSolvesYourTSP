import tensorflow as tf
import os
# debug
os.environ["CUDA_VISIBLE_DEVICES"] = ""

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
