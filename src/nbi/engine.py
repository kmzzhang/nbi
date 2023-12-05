# rull: noqa: E402 F401
import copy
import os

# this needs to go in before importing torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import corner
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import torch
from multiprocess import Pool
from torch import optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
)

# this seems to be required for some environments
from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn

dataloader.multiprocessing = mp

from .data import BaseContainer
from .model import DataParallelFlow, get_featurizer, get_flow
from .utils import iid_gaussian, log_like_iidg, parallel_simulate


class NBI:
    """Neural Bayesian Inference Engine.

    NBI is an open-source software introduced to support both amortized and sequential Neural Posterior Estimation
    (NPE) methods, particularly tailored for astronomical inference problems, such as those involving
    light curves and spectra.

    The design of NBI addresses critical issues in the adaptation of NPE methods in astronomy. It provides built-in
    "featurizer" networks with demonstrated efficacy on sequential data, removing the need for custom featurizer
    networks by users. It also employs a modified algorithm, SNPE-IS, which enables asymptotically exact inference
    by using the surrogate posterior under NPE as a proposal distribution for importance sampling.


    Parameters
    ----------
    flow : dict or nn.Module
        Dictionary containing hyperparameters for the Masked Autoregressive Flow.
        If dictionary, the keys include:

        - 'n_dims', dimension of model parameter space.
        - 'num_blocks', number of Masked Autoregressive Flow (MAF) blocks.
        - 'flow_hidden', hidden dimension for each MAF block
        - 'perm_seed', random seed for dimension permutation of each MAF block
        - 'n_mog', number of mixture of Gaussians as the base density. Recommended > 1 for multi-modal
          and/or non-Gaussian posterior distributions. Rule of thumb: twice the maximum number of posterior modes.

        Only 'n_dims' required. See class attribute default_flow_config for default values.

        If nn.Module, it's a custom normalizing flow.

    featurizer : dict or nn.Module
        Dictionary of hyperparameters for the pre-built neural network for dimensionality reduction
        or a custom PyTorch network module.

        If dictionary, the keys include:
            - 'type': Name of pre-built network architecture. Default: 'resnet-gru'.
            - 'norm': Either 'weight_norm' or 'batch_norm'.
            - 'dim_in': Number of input data channels.
            - 'dim_out': Dimension of output feature vector.
            - 'dim_conv_max': Maximum hidden dimensions of ResNet layers.
            - 'depth': ResNet network depth.
            - 'n_rnn': Number of GRU layers.

        If nn.Module, it's a custom featurizer network that maps input sequence of shape [Batch, Channel, Length] to
        output feature vector of shape [Batch, Dimension].

    simulator : function, optional
        Simulator function to generate data. Requires input of model parameters and returns simulated data.

    priors : list of scipy.stats objects, optional
        Prior distribution for Bayesian inference. If not provided, uniform prior is assumed.

    device : {'cpu', 'cuda', 'mps'} str, optional
        Device for neural network, default is 'cpu'.

    path : str, optional
        Path to save training set and model checkpoints.

    n_jobs : int, optional
        Number of parallel jobs for computation.

    labels : list of str, optional
        Names of parameters for inference. Must be in the same order as priors.

    tqdm_notebook : bool, optional
        If True, uses notebook version of tqdm for progress bars.

    network_reinit : bool, optional
        If True, re-initializes the network weights every round. Default is False, which often yield better results
        than True.

    scale_reinit : bool, optional
        If True, re-initializes data pre-processing scales every round. Default is True.

    """

    corner_kwargs = {
        "quantiles": [0.16, 0.5, 0.84],
        "show_titles": True,
        "title_kwargs": {"fontsize": 16},
        "fill_contours": True,
        "levels": 1.0 - np.exp(-0.5 * np.arange(0.5, 2.6, 0.5) ** 2),
    }

    default_flow_config = {
        "flow_hidden": 64,
        "num_blocks": 5,
        "perm_seed": 3,
        "n_mog": 1,
    }

    def __init__(
        self,
        flow,
        featurizer,
        simulator=None,
        priors=None,
        device="cpu",
        path="test",
        n_jobs=1,
        labels=None,
        tqdm_notebook=False,
        network_reinit=False,
        scale_reinit=True,
    ):
        self.device = device
        self.init_env()
        flow_config_all = copy.copy(self.default_flow_config)
        flow_config_all.update(flow)
        self.corner_kwargs.update({"labels": labels})
        self.network_reinit = network_reinit
        self.scale_reinit = scale_reinit

        # if featurizer is not user provided pytorch module
        # generate featurizer network based on user specified type and hyperparameters
        if type(featurizer) == dict:
            featurizer = get_featurizer(featurizer.pop("type"), featurizer)
            flow_config_all["num_cond_inputs"] = featurizer.num_outputs

        self.network = get_flow(featurizer, **flow_config_all)
        self.network = DataParallelFlow(self.network).to(
            self.device, dtype=torch.float32
        )

        self.epoch = 0
        self.prev_clip = 1e8
        self.tloss = []
        self.vloss = []

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.norm = []

        self.prior = priors
        self.param_names = labels
        self.simulator = simulator
        self.directory = path
        self.n_jobs = n_jobs
        self.x_obs = None
        self.y_true = None

        self.x = None
        self.y = None

        self.process = None
        self.like = None

        self.round = 0
        self.early_stop_count = 0
        self.x_all = []
        self.y_all = []
        self.weights = []
        self.neff = []
        self.state_dict_0 = self.get_state_dict()

        self.prev_state = []
        self.prev_x_mean = []
        self.prev_x_std = []
        self.prev_y_mean = []
        self.prev_y_std = []

        try:
            os.mkdir(self.directory)
        except:
            pass

        if tqdm_notebook:
            self.tqdm = tqdmn
        else:
            self.tqdm = tqdm

    def fit(
        self,
        x=None,
        y=None,
        noise=None,
        log_like=None,
        n_epochs=10,
        n_rounds=1,
        n_sims=-1,
        x_obs=None,
        y_true=None,
        n_reuse=0,
        batch_size=64,
        project="test",
        wandb_enabled=False,
        neff_stop=-1,
        early_stop_train=False,
        early_stop_patience=-1,
        f_val=0.1,
        lr=0.001,
        min_lr=None,
        decay_type="SGDR",
        plot=True,
        f_accept_min=-1,
        workers=8,
    ):
        """
        Fit the Neural Bayesian Inference Engine.

        Trains the network based on provided data and parameters.

        Parameters
        ----------
        x : ndarray of paths to individual simulations, optional
            First round training simulations. Only required when simulation and prior not specified during engine
            initialization.

        y : ndarray of shape (N, D) where D is parameter space dimension, optional
            First round training parameters. Only required when simulation and prior not specified during engine
            initialization.

        noise : ndarray or function, optional
            Measurement error and/or data augmentation during training. Array of Gaussian errorbars for fixed
             iid Gaussian noise. For ANPE, provide a function that takes in noiseless data and parameters (x, y) and
             outputs the noisified data and parameters (x', y'), which is the last pre-processing step before feeding
             into the neural network.

        log_like : function, optional
            Log-likelihood function that takes in (x, x_path, y) and returns the log likelihood.
            Required for importance sampling (SNPE) but not required when noise is iid Gaussian and specified
            as an errorbar array.

        n_epochs : int, optional
            Number of training epochs.

        n_rounds : int, optional
            Number of training rounds.

        n_sims : int, optional
            Number of simulations.

        x_obs : ndarray, optional
            Observed data.

        y_true : ndarray, optional
            True target values.

        n_reuse : int, optional
            Number of previous round training data to be reused for the current round.

        batch_size : int, optional
            Batch size for training and validation.

        project : str, optional
            Name of the project for logging.

        wandb_enabled : bool, optional
            If True, enables wandb logging.

        neff_stop : int, optional
            Early stopping criteria based on Effective Sample Size (ESS). Terminate inference when ESS exceeds this
            value.

        early_stop_train : bool, optional
            If True, terminates inference when the surrogate posterior (as measured by the ESS) does not improve for the current round.

        early_stop_patience : int, optional
            Number of epochs without improvement to trigger early stopping.

        f_val : float, optional
            Fraction of data to use for validation. Default: 0.1

        lr : float, optional
            Learning rate. Default: 0.001

        min_lr : float, optional
            Minimum learning rate for learning rate decay. Automatically calculated when not specified.

        decay_type : {'SGDR'} str, optional
            Type of learning rate decay. Default is Cosine annealing decay ("SGDR").

        plot : bool, optional
            If True, plots results after training.

        f_accept_min : float, optional
            Minimum round sampling efficiency (defined as the ratio from the effective sample size to the total sample
            size) to terminate inference early.

        workers : int, optional
            Number of workers for data loading.

        Returns
        -------

        """
        assert n_sims > 0 or y is not None

        if type(noise) == np.ndarray:
            # for i.i.d. gaussian noise
            self.process = iid_gaussian(noise)
            self.like = log_like_iidg(noise)
        else:
            # for custom noise
            self.process = noise
            self.like = log_like

        self.n_epochs = n_epochs
        self.x_obs = x_obs
        self.y_true = y_true

        self.x = x
        self.y = y

        self._init_wandb(project, wandb_enabled)

        if min_lr is None:
            min_lr = min(lr, lr / (n_sims / batch_size * n_epochs) * 10)
            print("Auto learning rate to min_lr =", min_lr)

        # for restarting training
        if len(self.x_all) == self.round:
            # this is not a restart because
            # data for this round has not been generated
            self.prepare_data(x_obs, n_sims)

        for i in range(n_rounds):
            print(
                f"\n---------------------- Round: {self.round} ----------------------"
            )

            self._init_train(lr)
            self._init_scheduler(min_lr, decay_type=decay_type)
            x_round, y_round = self.get_round_data(n_reuse)
            data_container = BaseContainer(
                x_round, y_round, f_test=0, f_val=f_val, process=self.process
            )
            self._init_loader(data_container, batch_size, workers=workers)

            for epoch in range(n_epochs):
                self.epoch = epoch
                self._train_step()
                self._step_scheduler()
                self._validate_step()
                if self.wandb:
                    wandb.log(
                        {
                            "Train Loss": self.training_losses[-1],
                            "Val Loss": self.validation_losses[-1],
                        }
                    )

                self.save_state_dict()

                if self.stop_training(early_stop_patience):
                    print(
                        "early stopping, loading state dict from epoch",
                        self.epoch - early_stop_patience - 2,
                    )
                    self.load_state_dict(self.epoch - early_stop_patience - 2)
                    break

            self.save_current_state()
            self.round += 1

            if n_rounds == 1:
                return

            self.prepare_data(x_obs, n_sims)

            if self.round > 0 and plot:
                self.weighted_corner(x_obs, y_true)

            # early stopping
            if np.sum(self.neff) > neff_stop > 0:
                print("Success: Exceed specified stopping sample size!")
                self._corner_all()
                return

            f_accept_round = self.neff[-1] / n_sims
            if self.neff[-1] / n_sims > f_accept_min > 0:
                print(f"Success: Sampling efficiency is {f_accept_round:.1f}!")
                self._corner_all()
                return

            if early_stop_train and self.round > 1:
                if 1 < self.neff[-1] < self.neff[-2]:
                    print(
                        "Early stop: Surrogate posterior did not improve for this round"
                    )
                    self.load_prev_state(self.round - 2)
                    return

        if x_obs is not None:
            self._corner_all()

    def save_current_state(self):
        """
        Saves the network state from current round.

        Returns
        -------

        """
        prev_state = self.get_state_dict()
        self.prev_state.append(prev_state)
        self.prev_x_mean.append(self.x_mean)
        self.prev_x_std.append(self.x_std)
        self.prev_y_mean.append(self.y_mean)
        self.prev_y_std.append(self.y_std)

    def load_prev_state(self, round):
        """
        Load state of the engine from a previous round.

        Parameters
        ----------
        round : int
            Round number to load state from.

        Returns
        -------

        """
        print("Loaded state from round ", round)
        self.get_network().load_state_dict(self.prev_state[round])
        self.x_mean = self.prev_x_mean[round]
        self.x_std = self.prev_x_std[round]
        self.y_mean = self.prev_y_mean[round]
        self.y_std = self.prev_y_std[round]

    def _corner_all(self):
        """
        SNPE: Corner plot for the reweighted posterior from all rounds.

        Returns
        -------

        """
        print("reweighted posterior from all rounds")
        all_thetas, all_weights = self.result()
        self.corner(self.x_obs, all_thetas, y_true=self.y_true, weights=all_weights)

    def prepare_data(self, x_obs, n_sims):
        """
        Generate training data for the current round.

        Parameters
        ----------
        x_obs : ndarray
            Observed data for producing simulations.
        n_sims : int
            Number of simulations.

        Returns
        -------

        """

        ys = self._draw_params(x_obs, n_sims)

        np.save(os.path.join(self.directory, str(self.round)) + "_y_all.npy", ys)

        x_path, good = self.simulate(ys)
        np.save(os.path.join(self.directory, str(self.round)) + "_x.npy", x_path[good])
        np.save(os.path.join(self.directory, str(self.round)) + "_y.npy", ys[good])

        self.x_all.append(np.array(x_path)[good])
        self.y_all.append(np.array(ys)[good])

        weights = self.importance_reweight(x_obs, self.x_all[-1], self.y_all[-1])
        self.weights.append(weights)
        np.save(os.path.join(self.directory, str(self.round)) + "_w.npy", weights)

        if self.like is not None and x_obs is not None:
            neff = 1 / (weights**2).sum() - 1
            self.neff.append(neff)
            print(
                "Effective sample size for current/all rounds",
                f"{neff:.1f}/{np.sum(self.neff):.1f}",
            )

    def weighted_corner(self, x_obs, y_true):
        """
        SNPE: Reweighted corner plot for the current round.

        Parameters
        ----------
        x_obs : ndarray
            Observed data.
        y_true : ndarray
            True parameters, if known.

        Returns
        -------

        """
        try:
            self.corner(x_obs, self.y_all[-1], y_true=y_true, weights=self.weights[-1])

        except:
            print("corner plot failed")

    def stop_training(self, patience=1):
        """
        Early stopping criteria based on validation loss.

        Parameters
        ----------
        patience : int
            Number of epochs without improvement to trigger early stopping.

        Returns
        -------
        bool
            True if early stopping criteria is met.

        """
        if self.epoch < patience + 2 or patience == -1:
            return False

        prev_losses = np.array(self.vloss[-1 * patience - 1 :])
        base_loss = self.vloss[-1 * patience - 2]
        return (prev_losses > base_loss).all()

    def result(self):
        """
        SNPE: Returns the reweighted posterior from all rounds.

        Returns
        -------
            all_thetas : ndarray
                Parameter values from all rounds.
            all_weights: ndarray
                Importance weights from all rounds.
        """
        all_weights = np.concatenate(
            [self.weights[i] * self.neff[i] for i in range(self.round + 1)]
        )
        all_weights /= all_weights.sum()
        all_thetas = np.concatenate(self.y_all)

        return all_thetas, all_weights

    def get_round_data(self, n_reuse):
        """
        Returns training data for the current round.

        Parameters
        ----------
        n_reuse : int
            Number of previous round training data to be reused for the current round.

        Returns
        -------
        x_round : ndarray
            Training data for the current round.
        y_round : ndarray
            Training parameters for the current round.

        """
        if n_reuse == -1:
            return np.concatenate(self.x_all), np.concatenate(self.y_all)
        else:
            x_round = self.x_all[max(0, self.round - n_reuse) : self.round + 1]
            x_round = np.concatenate(x_round)

            y_round = self.y_all[max(0, self.round - n_reuse) : self.round + 1]
            y_round = np.concatenate(y_round)

            return x_round, y_round

    def importance_reweight(self, x_obs, x, y):
        """
        SNPE: Calculate importance reweights for the current round.

        Parameters
        ----------
        x_obs : ndarray
            Observed data.
        x : ndarray
            Simulated data.
        y : ndarray
            Simulated parameters.

        Returns
        -------
        weights : ndarray
            Importance weights.

        """
        if self.like is None or x_obs is None:
            return None

        loglike = self.log_like(x_obs, x, y)
        logprior = self.log_prior(y)
        logproposal = self.log_prob(x_obs, y)

        log_weights = loglike + logprior - logproposal

        bad = np.isnan(log_weights) + np.isinf(log_weights)
        log_weights -= log_weights[~bad].max()

        weights = np.exp(log_weights)
        weights[bad] = 0
        weights /= weights.sum()

        return weights

    def importance_reweight_like_only(self, x_obs, x, y):
        """
        SNPE: Calculate importance reweights for the current round, using only the likelihood.

        Parameters
        ----------
        x_obs : ndarray
            Observed data.
        x : ndarray
            Simulated data.
        y : ndarray
            Simulated parameters.

        Returns
        -------
        weights : ndarray

        """
        if self.like is None or x_obs is None:
            return None

        log_weights = self.log_like(x_obs, x, y)
        bad = np.isnan(log_weights) + np.isinf(log_weights)
        log_weights -= log_weights[~bad].max()

        weights = np.exp(log_weights)
        weights[bad] = 0
        weights /= weights.sum()

        return weights

    def init_env(self):
        """
        Initialize environment for training.

        Returns
        -------

        """
        torch.manual_seed(0)
        np.random.seed(0)
        if self.device == "mps":
            try:
                torch.mps.manual_seed(0)
            except:
                print(
                    "MPS not supported by current PyTorch installation. Reverting to CPU"
                )
                self.device = "cpu"
        elif "cuda" in self.device:
            if not torch.cuda.is_available():
                print(
                    "CUDA not supported by current PyTorch installation. Reverting to CPU"
                )
            else:
                torch.cuda.manual_seed(0)

    def get_network(self):
        """
        Returns the network module without DataParallel wrapper, if any.

        Returns
        -------
        nn.Module
            Network module without the DataParallel wrapper.

        """
        if type(self.network) == DataParallelFlow:
            return self.network.module
        else:
            return self.network

    def get_state_dict(self):
        """
        Returns the current network state dictionary.

        Returns
        -------

        """
        return copy.deepcopy(self.get_network().state_dict())

    def load_state_dict(self, epoch):
        """
        Loads the network state dictionary from a previous epoch.

        Parameters
        ----------
        epoch : int
            Epoch number to load state from.

        Returns
        -------

        """
        path_round = os.path.join(self.directory, str(self.round))
        path = os.path.join(path_round, str(epoch) + ".pth")
        self.get_network().load_state_dict(torch.load(path, map_location=self.device))

    def set_params(self, network, x_scale, y_scale):
        """
        Load engine parameters from disk, including network weights and data pre-processing scales.

        Parameters
        ----------
        network : str
            Path of network state dict.
        x_scale : str or ndarray
            Path of x-scale or x-scale array.
        y_scale : str or ndarray
            Path of y-scale or y-scale array.

        Returns
        -------

        """
        if type(x_scale) == str:
            x_scale = np.load(x_scale)
        if type(y_scale) == str:
            y_scale = np.load(y_scale)

        self.x_mean = x_scale[0]
        self.x_std = x_scale[1]
        self.y_mean = y_scale[0]
        self.y_std = y_scale[1]
        self.get_network().load_state_dict(
            torch.load(network, map_location=self.device)
        )

    def save_state_dict(self):
        """
        Saves the network weights and pre-processing scales to disk

        Returns
        -------

        """
        path_round = os.path.join(self.directory, str(self.round))
        path_network = os.path.join(path_round, str(self.epoch) + ".pth")
        torch.save(self.get_state_dict(), path_network)

        path_xscales = os.path.join(path_round, "x_scales.npy")
        path_yscales = os.path.join(path_round, "y_scales.npy")
        np.save(path_xscales, np.array([self.x_mean, self.x_std]))
        np.save(path_yscales, np.array([self.y_mean, self.y_std]))

    def scale_y(self, y, back=False):
        """
        Scale parameters to zero mean and unit variance, and vice versa

        Parameters
        ----------
        y : ndarray
            Parameters to be scaled.
        back : bool, optional
            If True, scales parameters back to original values.

        Returns
        -------

        """
        if back:
            return y * self.y_std + self.y_mean
        else:
            if len(y.shape) != 2:
                y = np.expand_dims(y, axis=list(range(2 - len(y.shape))))
            return (y - self.y_mean) / self.y_std

    def scale_x(self, x, back=False):
        """
        Scale data to zero mean and unit variance, and vice versa.

        Parameters
        ----------
        x : ndarray
            Data to be scaled.
        back : bool, optional
            If True, scales data back to original values.

        Returns
        -------

        """
        if back:
            return x * self.x_std + self.x_mean
        else:
            # shape needs to be (N, D, L) for ResNet-GRU
            # todo: make more generic
            if len(x.shape) != 3:
                x = np.expand_dims(x, axis=list(range(3 - len(x.shape))))
            return (x - self.x_mean) / self.x_std

    def predict(
        self,
        x,
        x_err=None,
        y_true=None,
        log_like=None,
        n_samples=1000,
        neff_min=0,
        n_max=-1,
        corner=False,
        corner_reweight=False,
    ):
        """
        Generates the posterior distribution of parameters given input data.

        Parameters
        ----------
        x : ndarray
            Input data for inference
        x_err : ndarray, optional
            Measurement error for input data. Required for importance sampling. If not specified, use log_like instead.
        y_true : ndarray, optional
            True parameters, if known.
        log_like : function, optional
            Log-likelihood function that takes in (x, x_path, y) and returns the log likelihood. Required for importance
            sampling when x_err not specified.
        n_samples : int, optional
            Number of posterior samples to generate.
        neff_min : int, optional
            Minimum effective sample size required. If neff_min > n_samples, additional simulations will be generated
            until an ESS of neff_min is reached. Default: 0
        n_max : int, optional
            Maximum number of simulations to generate to achieve neff_min.
        corner : bool, optional
            If True, generates a corner plot of the posterior before reweighting.
        corner_reweight : bool, optional
            If True, generates a corner plot of the posterior after reweighting.

        Returns
        -------
        ys : ndarray
            Posterior samples.
        weights : ndarray
            Importance weights.

        """
        self.like = log_like_iidg(x_err) if type(x_err) == np.ndarray else log_like

        if self.round == 0:
            self.round = 1
        ys = self._draw_params(x, n_samples)

        if corner:
            print("surrogate posterior")
            self.corner(x, ys, y_true=y_true)

        if x_err is None and log_like is None:
            return ys

        x_path, good = self.simulate(ys)
        x_path = x_path[good]
        ys = ys[good]
        weights = self.importance_reweight(x, x_path, ys)

        neff = 1 / (weights**2).sum() - 1

        f_accept = neff / n_samples
        print(f"Effective Sample Size = {neff:.1f}")
        print(f"Sampling efficiency = {f_accept * 100:.1f}%")

        if n_max > n_samples and neff > neff_min:
            n_required = int(n_samples * (1 / f_accept - 1))
            print("Requires N =", n_required, "more simulations to reach n_samples")
            n_required = min(n_required, n_max - n_samples)

            ys_extra = self._draw_params(x, n_required)
            x_path, good = self.simulate(ys_extra)
            x_path = x_path[good]
            ys_extra = ys_extra[good]
            weights_extra = self.importance_reweight(x, x_path, ys_extra)

            neff_extra = 1 / (weights_extra**2).sum() - 1
            print("Total effective sample size N =", "%.1f" % (neff + neff_extra))

            ys = np.concatenate([ys, ys_extra])
            weights = np.concatenate([weights, weights_extra])

        if corner_reweight:
            self.corner(x, ys, y_true=y_true, weights=weights)

        return ys, weights

    def sample(self, x, y=None, n=5000, corner=False):
        """
        Generates samples from the surrogate posterior.

        Parameters
        ----------
        x : ndarray
            Input data for inference
        y : ndarray, optional
            True parameters (for corner plot), if known.
        n : int, optional
            Number of samples to generate.
        corner : bool, optional
            If True, generates a corner plot of the surrogate posterior samples.

        Returns
        -------
        samples : ndarray
            Samples from the surrogate posterior.

        """
        self.network.eval()
        x = self.scale_x(x)
        x = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            # GPU memory control (make larger?)
            if n > 20000:
                s = []
                for i in range(n // 100000 + 1):
                    s.append(self.get_network()(x, n=n, sample=True).cpu().numpy())
                s = np.concatenate(s)[:n]
            else:
                s = self.get_network()(x, n=n, sample=True).cpu().numpy()
        samples = self.scale_y(s, back=True)[0]
        if corner:
            self.corner(x, samples, y_true=y)
        return samples

    def simulate(self, thetas):
        """
        Generates simulations for provided parameters, which are saved to disk. An array containing paths to the
         simulations is returned.

        Parameters
        ----------
        thetas : ndarray
            Parameters to generate simulations for.

        Returns
        -------
        x_path : ndarray
            Paths to generated simulations.
        """
        path_round = os.path.join(self.directory, str(self.round))
        try:
            os.mkdir(path_round)
        except:
            pass

        if self.x is not None and self.round == 0:
            print("Use precomputed simulations for round ", self.round)
            masks = np.array([True] * len(self.x))
            return self.x, masks
        else:
            n = len(thetas)
            paths = np.array(
                [os.path.join(path_round, str(i) + ".npy") for i in range(n)]
            )
            per_job = n // self.n_jobs
            njobs = np.zeros(self.n_jobs) + per_job
            njobs[np.arange(n % self.n_jobs)] += 1
            njobs = np.array(
                [njobs[:i].sum() for i in range(self.n_jobs + 1)], dtype=int
            )

            jobs = [
                [
                    thetas[njobs[i] : njobs[i + 1]],
                    paths[njobs[i] : njobs[i + 1]],
                    self.simulator,
                ]
                for i in range(self.n_jobs)
            ]

            with Pool(self.n_jobs) as p:
                masks = p.map(parallel_simulate, jobs)
            masks = np.concatenate(masks)
            return paths, masks

    def _train_step(self):
        """
        Single training step.

        Returns
        -------

        """
        np.random.seed(self.epoch)
        self.network.train()
        train_loss = []
        with self.tqdm(total=len(self.train_loader.dataset)) as pbar:
            for batch_idx, data in enumerate(self.train_loader):
                if len(data) == 2:
                    x, y = data
                    aux = None
                else:
                    x, y, aux = data
                    aux = aux.to(self.device, dtype=torch.float32)
                x = self.scale_x(x).to(self.device, dtype=torch.float32)
                y = self.scale_y(y).to(self.device, dtype=torch.float32)
                self.optimizer.zero_grad()
                loss = self.network(x, y, aux=aux)
                loss = loss.mean()
                train_loss.append(loss.item())
                loss.backward()
                if self.clip > 0:
                    self.norm.append(
                        torch.nn.utils.clip_grad_norm_(
                            self.network.parameters(), self.prev_clip
                        ).cpu()
                    )
                self.optimizer.step()

                pbar.update(x.shape[0])
                pbar.set_description(
                    "Epoch {:d}: Train, Loglike in nats: {:.6f}".format(
                        self.epoch, -np.mean(train_loss)
                    )
                )

        if self.clip > 0:
            self.prev_clip = np.percentile(np.array(self.norm), self.clip)
        train_loss = np.array(train_loss).mean()
        self.tloss.append(train_loss)

    def _validate_step(self):
        """
        Single validation step.

        Returns
        -------

        """
        np.random.seed(0)
        self.network.eval()
        val_loss = []
        with self.tqdm(total=len(self.valid_loader.dataset)) as pbar:
            objs = 0
            for batch_idx, data in enumerate(self.valid_loader):
                x, y = data
                x = self.scale_x(x).to(self.device, dtype=torch.float32)
                y = self.scale_y(y).to(self.device, dtype=torch.float32)
                objs += x.shape[0]
                self.optimizer.zero_grad()
                with torch.no_grad():
                    loss = self.network(x, y).mean()
                    val_loss.append(loss.detach().cpu().numpy())
                pbar.update(x.shape[0])
                pbar.set_description(
                    f"- Val, Loglike in nats: {-np.sum(val_loss) / (batch_idx + 1):.6f}"
                )

        val_loss = np.median(val_loss)
        pbar.set_description(f"- Val, Loglike in nats: {-val_loss:.6f}")
        self.vloss.append(val_loss)

    def _init_wandb(self, project, enable):
        """
        Initialize weights & biases logging.

        Parameters
        ----------
        project : str
            Project name.
        enable : bool
            If True, enable weights & biases logging.

        Returns
        -------

        """
        self.wandb = enable
        if enable:
            try:
                import wandb
            except ModuleNotFoundError:
                print("weights & biases not installed")
                self.wandb = False
                return
            wandb.init(project=project, config=self.args, name=self.name)
            wandb.watch(self.network)

    def _init_train(self, lr, clip=85):
        """
        Initialize training environment.

        Parameters
        ----------
        lr : float
            Learning rate.
        clip : [0, 100] float, optional
            Gradient clipping percentile for each epoch. If 0, no clipping is performed.

        Returns
        -------

        """
        self.clip = clip
        if self.network_reinit:
            self.get_network().load_state_dict(self.state_dict_0)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        torch.manual_seed(0)
        self.training_losses = []
        self.validation_losses = []

    def _init_scheduler(
        self, min_lr, decay_type="SGDR", patience=5, decay_threshold=0.01
    ):
        """
        Initialize learning rate scheduler.

        Parameters
        ----------
        min_lr : float
            Minimum learning rate.
        decay_type : str, optional
            Learning rate decay type. Options: "SGDR", "plateau", or a comma-separated list/string of epochs to decay at.
        patience : int, optional
            plateau: Number of epochs without improvement to trigger learning rate decay.
        decay_threshold : float, optional
            plateau: Threshold for measuring the new optimum, to only focus on significant changes.

        Returns
        -------

        """
        self.decay_type = decay_type
        if decay_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                factor=0.1,
                patience=patience,
                threshold_mode="abs",
                cooldown=0,
                verbose=True,
                threshold=decay_threshold,
                min_lr=1e-6,
            )

        elif "SGDR" in decay_type:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.n_epochs, T_mult=1, eta_min=min_lr
            )
        else:
            self.scheduler = MultiStepLR(
                self.optimizer, np.array(decay_type.split(","), dtype=int), gamma=0.1
            )

    def _step_scheduler(self):
        """
        Step learning rate scheduler.

        Returns
        -------

        """
        if self.decay_type == "plateau":
            self.scheduler.step(self.training_losses[-1])
        else:
            self.scheduler.step()

    def _init_loader(self, data_container, batch_size, workers=4):
        """
        Initialize data loader.

        Parameters
        ----------
        data_container : DataContainer
            Data container object.
        batch_size : int
            Batch size.
        workers : int, optional
            Number of workers for data loader.

        Returns
        -------

        """
        train_container, val_container, test_container = data_container.get_splits()

        kwargs = {
            "num_workers": workers,
            "pin_memory": False,
            "drop_last": True,
            "persistent_workers": True,
        }

        self.train_loader = DataLoader(
            train_container, batch_size=batch_size, shuffle=True, **kwargs
        )
        self.valid_loader = DataLoader(val_container, batch_size=batch_size, **kwargs)

        # if self.network_reinit or self.round == 0:
        if self.scale_reinit or self.round == 0:
            self._init_scales()

    def _draw_params(self, x, n):
        """
        Draw parameters from prior (ANPE or SNPE first round) or surrogate posterior.

        Parameters
        ----------
        x : ndarray
            Input data for inference.
        n : int
            Number of parameters to draw.

        Returns
        -------

        """
        # first round: precomputed data or draw from prior
        if self.round == 0:
            if self.y is not None:
                return self.y
            else:
                params = []
                for prior in self.prior:
                    params.append(prior.rvs(n))
                params = np.array(params).T
                return params
        # 2+ round: sample from surrogate posterior
        else:
            params = self.sample(x, n=n)
            logprior = self.log_prior(params)
            if np.isinf(logprior).any():
                print("Samples outside prior N =", np.isinf(logprior).sum())
                params = params[~np.isinf(logprior)]
            return params

    def _init_scales(self):
        """
            Calculate data pre-processing scales from the current round training data.

        Returns
        -------

        """
        x_list = []
        y_list = []
        n = 0
        for batch_idx, data in enumerate(self.train_loader):
            if len(data) == 2:
                x, y = data
                aux = None
            else:
                x, y, aux = data
            x_list.append(x.cpu().numpy())
            y_list.append(y.cpu().numpy())
            n += x_list[-1].shape[0]
            if n > 5000:
                break
        x_list = np.concatenate(x_list, axis=0)
        y_list = np.concatenate(y_list, axis=0)
        self.x_mean = x_list.mean(0).mean(-1, keepdims=True)
        self.x_std = x_list.mean(0).std(-1, keepdims=True)
        self.y_mean = y_list.mean(0, keepdims=True)
        self.y_std = y_list.std(0, keepdims=True)

    def log_prior(self, y):
        """
        Calculate log prior probability.

        Parameters
        ----------
        y : ndarray
            Parameters to calculate prior for.

        Returns
        -------
        log_prob : ndarray
            Log prior probability.

        """
        if self.prior is None:
            return np.zeros(len(y))
        else:
            log_prob = np.zeros(len(y))
            for i, prior in enumerate(self.prior):
                log_prob += prior.logpdf(y[:, i])
        return log_prob

    def log_like(self, x_obs, x, y):
        """
        Calculate log likelihood.

        Parameters
        ----------
        x_obs : ndarray
            Observed data.
        x : ndarray
            Simulated data.
        y : ndarray
            Simulated parameters.

        Returns
        -------
        log_prob : ndarray
            Log likelihood.

        """
        values = []
        for i in range(len(x)):
            values.append(self.like(x_obs, x[i], y[i]))
        return np.array(values)

    def log_prob(self, x, y):
        """
        Calculate log probability under surrogate posterior.

        Parameters
        ----------
        x : ndarray
            Observations.
        y : ndarray
            Pparameters.

        Returns
        -------
        log_prob : ndarray
            Log probability under surrogate posterior

        """
        if self.round == 0:
            return self.log_prior(y)

        x = self.scale_x(x)
        y = self.scale_y(y)

        x = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        y = torch.from_numpy(y).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            # it appears that DataParallal doesn't work properly here
            log_prob = self.network.module(x, y).cpu().numpy()[:, 0] * -1
        return log_prob

    def corner(
        self,
        x,
        y=None,
        weights=None,
        color="k",
        y_true=None,
        plot_datapoints=True,
        plot_density=False,
        range_=None,
        truth_color="r",
        n=5000,
    ):
        """
        Wrapper function to make corner plot.

        Parameters
        ----------
        x : ndarray
            Input data for inference.
        y : ndarray, optional
            Parameters to plot. If None, parameters are drawn from the surrogate posterior.
        weights : ndarray, optional
            Importance weights for reweighting.
        color : str, optional
            Color of the corner plot
        y_true : ndarray, optional
            True parameters for crosshairs
        plot_datapoints : bool, optional
            If True, plot data points as scatter.
        plot_density : bool, optional
            If True, plot 2D densities.
        `range_` : list, optional
            Percentile of data to plot in corner plot.
        truth_color : str, optional
            Color of crosshairs.
        n : int, optional
            Number of samples to draw from the surrogate posterior, if y is not specified.

        Returns
        -------

        """
        if y is None:
            y = self.sample(x, n=n)
        corner.corner(
            y,
            truths=y_true,
            color=color,
            plot_datapoints=plot_datapoints,
            range=range_,
            plot_density=plot_density,
            truth_color=truth_color,
            weights=weights,
            **self.corner_kwargs,
        )
        plt.show()
