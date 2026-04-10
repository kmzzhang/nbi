from functools import partial

import numpy as np
from tqdm import tqdm
import torch


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
    # TODO: generalize to multi-channel / multi-modal x (e.g., list of arrays)
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

def collate_list_channels(batch):
    """
    Custom collate function that supports both single-tensor and multi-channel inputs.

    For multi-channel data, batch is a list of tuples: [(x_list, y), ...]
    where x_list is a list/tuple of tensors, one per channel: [x0, x1, ..., xK].

    For single-tensor data, batch is a list of tuples: [(x, y), ...]
    where x is a tensor or ndarray.

    Returns
    -------
    x_batched : list of tensors (multi-channel) or single tensor (single-channel)
        Each channel stacked to [B, ...], or a single stacked tensor [B, ...].
    y_batched : tensor [B, D]
    """
    xs, ys = zip(*batch)

    # Convert y to tensors
    y_out = torch.stack([torch.as_tensor(y) for y in ys], dim=0)

    # Multi-channel: x is a list/tuple of tensors
    if isinstance(xs[0], (list, tuple)):
        k = len(xs[0])
        x_out = [torch.stack([torch.as_tensor(x_i[j]) for x_i in xs], dim=0) for j in range(k)]
    else:
        # Single tensor/ndarray
        x_out = torch.stack([torch.as_tensor(x) for x in xs], dim=0)

    return x_out, y_out


def masked_mean_std(xs, mask_dim, masked_dims, reduce_axes=(0, 2), eps=1e-8):
    """
    Compute masked mean/std for selected channels.

    Parameters
    ----------
    xs : np.ndarray, shape [N, C, L]
    mask_dim : int
        Channel index containing the mask (same length L).
    masked_dims : list[int]
        Channels to compute masked stats for.
    reduce_axes : tuple[int]
        Axes to reduce over when computing stats.
        - (0,2): reduce over batch and length -> outputs [1, C, 1]
        - (0,):  reduce over batch only        -> outputs [1, C, L]
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    mu, sd : np.ndarray
        Shapes follow keepdims=True reduction:
        - reduce_axes=(0,2) => [1, C, 1]
        - reduce_axes=(0,)  => [1, C, L]
    """
    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(f"xs must be [N,C,L], got shape {xs.shape}")

    N, C, L = xs.shape
    reduce_axes = tuple(reduce_axes)

    # Determine expected output shape with keepdims=True
    out_shape = [N, C, L]
    for ax in reduce_axes:
        out_shape[ax] = 1
    out_shape = tuple(out_shape)

    mu = np.zeros(out_shape, dtype=np.float32)
    sd = np.ones(out_shape, dtype=np.float32)

    # mask: [N, 1, L] broadcastable across channels
    mask = xs[:, mask_dim:mask_dim + 1, :]  # [N,1,L]
    mask = np.where(np.isfinite(mask), mask, 0.0)
    mask = (mask > 0.5).astype(np.float32)

    for d in masked_dims:
        v = xs[:, d:d + 1, :]  # [N,1,L]
        finite = np.isfinite(v).astype(np.float32)

        w = mask * finite  # [N,1,L], 0/1 weights
        wsum = w.sum(axis=reduce_axes, keepdims=True)  # out_shape but with channel=1

        # Safe denom
        denom = np.maximum(wsum, 0.0)

        # Masked mean
        v_filled = np.where(np.isfinite(v), v, 0.0)
        m = (v_filled * w).sum(axis=reduce_axes, keepdims=True) / np.maximum(denom, eps)

        # Masked variance
        var = ((v_filled - m) ** 2 * w).sum(axis=reduce_axes, keepdims=True) / np.maximum(denom, eps)
        s = np.sqrt(var)

        # Where denom==0, fall back to finite-only unmasked stats over the same reduce_axes
        # This fallback is done elementwise in the reduced shape.
        if np.any(denom <= 0):
            w2 = finite
            w2sum = w2.sum(axis=reduce_axes, keepdims=True)
            m2 = (v_filled * w2).sum(axis=reduce_axes, keepdims=True) / np.maximum(w2sum, eps)
            var2 = ((v_filled - m2) ** 2 * w2).sum(axis=reduce_axes, keepdims=True) / np.maximum(w2sum, eps)
            s2 = np.sqrt(var2)

            use_fallback = (denom <= 0)
            m = np.where(use_fallback, m2, m)
            s = np.where(use_fallback, s2, s)

        # Final cleanup
        m = np.where(np.isfinite(m), m, 0.0)
        s = np.where(np.isfinite(s) & (s > 0), s, 1.0)

        # write into channel d (broadcast along reduced axes)
        mu[:, d:d + 1, :] = m.astype(np.float32)
        sd[:, d:d + 1, :] = s.astype(np.float32)

    return mu, sd