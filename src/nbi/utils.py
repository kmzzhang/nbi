from functools import partial

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
    mask = []
    if use_tqdm:
        print("Generating simulations")
    for i, params in tqdm(enumerate(thetas), disable=not use_tqdm):
        simulation = simulator(params)
        np.save(paths[i], simulation)
        mask.append(not np.isnan(simulation).any() and not np.isinf(simulation).any())

    return mask


def add_noise(x_err, x, y=None):
    """
    x: light curve of shape (length,)
    y: parameter of shape (dim,)
    """
    rand = np.random.normal(0, 1, size=x.shape[0])
    x_noise = x + rand * x_err
    return x_noise, y


def iid_gaussian(x_err):
    return partial(add_noise, x_err)


def log_like(x_err, x, x_path, y):
    # x is observed data, x_path is path to saved model prediction
    model = np.load(x_path)
    chi2 = (((x - model) / x_err) ** 2).sum()
    return -chi2 / 2


def log_like_iidg(x_err):
    return partial(log_like, x_err)
