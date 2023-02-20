import numpy as np
from tqdm import tqdm


def parallel_simulate(args):
    """
    thetas, paths, simulator are packaged into args because this function will be called using multiprocessing

    :param args: thetas, paths, simulator
        thetas: shape = (num_simulations, dim_parameters)
        paths: shape = (num_simulations,)
        simulator: callable.
    :return: mask of good simulations
    """

    thetas, paths, simulator = args
    use_tqdm = "/0.npy" in paths[0]
    mask = list()
    if use_tqdm:
        print("Generating simulations")
    for i, params in tqdm(enumerate(thetas), disable=not use_tqdm):
        simulation = simulator(params)
        np.save(paths[i], simulation)
        mask.append(not np.isnan(simulation).any() and not np.isinf(simulation).any())

    return mask
