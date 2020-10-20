if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gat.save.default_parameters import default_parameters
    from gat.save.save import SAVE
    tuning_parameters = {}
    parameters = default_parameters
    parameters.update(tuning_parameters)
    launcher = SAVE(**parameters)
    launcher.start()
