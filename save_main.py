import gpublock  # currently using gpu makes calculation slower
from gat.save.default_parameters import default_parameters
from gat.save.save import SAVE

if __name__ == '__main__':

    tuning_parameters = {}
    parameters = default_parameters
    parameters.update(tuning_parameters)
    launcher = SAVE(**parameters)
    launcher.start()
