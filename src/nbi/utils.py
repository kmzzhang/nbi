import numpy as np
from tqdm import tqdm_notebook as tqdm


def simulator_wrapper(simulator):
    """
        thetas, paths are packaged into args because this function will be called using multiprocessing

        thetas: shape = (num_simulations, dim_parameters)
        paths: shape = (num_simulations,)
    """

    def _sim2file(args):
        thetas, paths = args[0], args[1]
        for i, params in tqdm(enumerate(thetas)):
            simulation = simulator(params)
            np.save(paths[i], simulation)

    return _sim2file
