if __name__ == '__main__':
    import os
    # works much faster on cpu probably because of the overhead copying values on gpu or tf.map_fn.
    # hopefully it is due to tf.map_fn. Trying to replace create_mask function not to use tf.map_fn.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gat.save.default_parameters import default_parameters
    from gat.save.save import SAVE
    tuning_parameters = {}
    parameters = default_parameters
    parameters.update(tuning_parameters)
    launcher = SAVE(**parameters)
    launcher.start()
