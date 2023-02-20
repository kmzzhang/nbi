import os
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import corner
import numpy as np
import matplotlib.pyplot as plt
import wandb
from multiprocess import Pool
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn

import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    MultiStepLR,
    CosineAnnealingWarmRestarts,
)

from .model import get_flow, DataParallelFlow
from .data import BaseContainer
from .utils import parallel_simulate

corner_kwargs = {
    "quantiles": [0.16, 0.5, 0.84],
    "show_titles": True,
    "title_kwargs": {"fontsize": 16},
    "fill_contours": True,
    "levels": 1.0 - np.exp(-0.5 * np.arange(0.5, 2.6, 0.5) ** 2),
}

default_flow_config = {
    "flow_hidden": 256,
    "num_cond_inputs": 256,
    "num_blocks": 20,
    "perm_seed": 3,
    "n_mog": 8,
}


class NBI:
    """Neural bayesian inference engine for astronomical data"""

    def __init__(
        self,
        featurizer,
        dim_param,
        physics=None,
        prior_sampler=None,
        log_prior=None,
        log_like=None,
        instrumental=None,
        flow_config={},
        idx_gpu=0,
        parallel=False,
        directory="",
        n_jobs=1,
        n_jobs_loader=0,
        modify_scales=None,
        labels=None,
        tqdm_notebook=False,
    ):
        """

        Parameters
        ----------
        featurizer (nn.Module): pytorch network which maps input sequence of shape [Batch, Channel, Length] to
                output feature vector of shape [Batch, Dimension]. See NBI.get_featurizer() for pre-defined ones.
        dim_param (int): number of inferred parameters
        physics (callable): a function which takes (thetas, files)
        prior_sampler
        log_prior
        log_like
        instrumental
        flow_config
        idx_gpu
        parallel
        directory
        n_jobs
        modify_scales
        labels
        """

        self.init_env(idx_gpu)
        self.ndim = dim_param
        config = copy.copy(default_flow_config)
        config.update(flow_config)
        corner_kwargs.update({"labels": labels})

        self.network = get_flow(featurizer, dim_param, **config).type(self.dtype)
        if parallel:
            self.network = DataParallelFlow(self.network)

        self.epoch = 0
        self.prev_clip = 50000
        self.tloss = list()
        self.vloss = list()

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.norm = list()

        self.modify_scales = modify_scales

        self.draw_prior = prior_sampler
        self.prior = log_prior
        self.like = log_like
        self.simulator = physics
        self.process = instrumental
        self.directory = directory
        self.n_jobs = n_jobs
        self.n_jobs_loader = n_jobs_loader

        self.round = 0
        self.early_stop_count = 0
        self.x_all = list()
        self.y_all = list()
        self.weights = list()
        self.neff = list()
        self.state_dict_0 = self.get_state_dict()

        self.prev_state = None
        self.prev_x_mean = None
        self.prev_x_std = None
        self.prev_y_mean = None
        self.prev_y_std = None

        try:
            os.mkdir(self.directory)
        except:
            pass

        if tqdm_notebook:
            self.tqdm = tqdmn
        else:
            self.tqdm = tqdm

    def train(self, *args, **kwargs):
        # deprecated
        return self.run(*args, **kwargs)

    def run(
        self,
        obs,
        n_per_round,
        n_rounds=1,
        n_epochs=100,
        n_reuse=0,
        y_true=None,
        train_batch=512,
        val_batch=512,
        project="test",
        wandb_enabled=False,
        neff_stop=-1,
        early_stop_train=True,
        early_stop_patience=-1,
        f_val=0.1,
        lr=0.001,
        min_lr=None,
        x_file=None,
        y_file=None,
        decay_type="SGDR",
        debug=False,
    ):
        self.n_epochs = n_epochs
        self.x_file = x_file
        self.y_file = y_file

        self._init_wandb(project, wandb_enabled)

        if min_lr is None:
            min_lr = lr * 0.001

        """
         restart training:
          - len(self.x_all) == self.round + 1
            - data already generated
          - len(self.x_all) == self.round
            - data not available
        """
        if len(self.x_all) == self.round:
            self.prepare_data(obs, n_per_round, y_true=y_true)

        for i in range(n_rounds):
            print(
                "\n---------------------- Round: {} ----------------------".format(
                    self.round
                )
            )

            self._init_train(lr)
            self._init_scheduler(min_lr, decay_type=decay_type)

            x_round, y_round = self.get_round_data(n_reuse)
            data_container = BaseContainer(
                x_round, y_round, f_test=0, f_val=f_val, process=self.process
            )
            self._init_loader(data_container, train_batch, val_batch)

            for epoch in range(n_epochs):
                self.epoch = epoch
                print("\nEpoch: {}".format(epoch))
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
                if debug:
                    ys = self.sample(obs, n_per_round)
                    self.corner(obs, ys, y_true=y_true)

            self.save_current_state()

            if obs is None:
                self.round += 1
                return

            self.round += 1
            self.prepare_data(obs, n_per_round, y_true)

            if np.sum(self.neff) > neff_stop > 0:
                print("early stopping")
                self.corner_all(obs, y_true)
                return

            if early_stop_train and self.round > 1 and neff_stop > 0:
                if self.neff[-1] < self.neff[-2]:
                    self.load_prev_state()
                    n_required = neff_stop - np.sum(self.neff)
                    f_accept = self.neff[-2] / n_per_round
                    if f_accept < 0.005:
                        print("failed: acceptance rate < 0.5%")
                        return
                    n_required /= f_accept
                    n_required = int(n_required)
                    print("stop training")
                    print("importance sampling N =", n_required)

                    self.round += 1
                    self.prepare_data(obs, n_required, y_true)
                    self.corner_all(obs, y_true)
                    return

        if obs is not None:
            self.corner_all(obs, y_true)

    def save_current_state(self):
        self.prev_state = self.get_state_dict()
        self.prev_x_mean = self.x_mean
        self.prev_x_std = self.x_std
        self.prev_y_mean = self.y_mean
        self.prev_y_std = self.y_std

    def load_prev_state(self):
        self.get_network().load_state_dict(self.prev_state)
        self.x_mean = self.prev_x_mean
        self.x_std = self.prev_x_std
        self.y_mean = self.prev_y_mean
        self.y_std = self.prev_y_std

    def corner_all(self, obs, y_true):
        print("reweighted posterior from all rounds")
        all_thetas, all_weights = self.result()
        self.corner(obs, all_thetas, y_true=y_true, weights=all_weights)

    def prepare_data(self, obs, n_per_round, y_true=None):
        ys = self._draw_params(obs, n_per_round)
        np.save(os.path.join(self.directory, str(self.round)) + "_y_all.npy", ys)

        if self.round > 0:
            print("surrogate posterior")
            self.corner(obs, ys, y_true=y_true)

        x_path, good = self.simulate(ys)
        np.save(os.path.join(self.directory, str(self.round)) + "_x.npy", x_path[good])
        np.save(os.path.join(self.directory, str(self.round)) + "_y.npy", ys[good])

        self.add_round_data(x_path, ys, good)

        weights = self.importance_reweight(obs, self.x_all[-1], self.y_all[-1])
        self.weights.append(weights)
        np.save(os.path.join(self.directory, str(self.round)) + "_w.npy", weights)

        if self.round > 0:
            self.weighted_corner(obs, y_true)

        if self.like is not None and obs is not None:
            neff = 1 / (weights**2).sum() - 1
            self.neff.append(neff)
            print("Effective sample size for this round", "%.1f" % neff)
            print("Effective sample size for all rounds: ", "%.1f" % np.sum(self.neff))

    def weighted_corner(self, obs, y_true):
        try:
            print("reweighted posterior from current round")
            self.corner(obs, self.y_all[-1], y_true=y_true, weights=self.weights[-1])

        except:
            print("corner plot failed")

    def stop_training(self, patience=1):
        if self.epoch < patience + 2 or patience == -1:
            return False

        prev_losses = np.array(self.vloss[-1 * patience - 1 :])
        base_loss = self.vloss[-1 * patience - 2]
        return (prev_losses > base_loss).all()

    def add_round_data(self, x, y, good):
        self.x_all.append(np.array(x)[good])
        self.y_all.append(np.array(y)[good])
        if len(x) != good.sum():
            print("Number of simulations with nan/inf:", len(x) - good.sum())

    def result(self):
        all_weights = np.concatenate(
            [self.weights[i] * self.neff[i] for i in range(self.round + 1)]
        )
        all_weights /= all_weights.sum()
        all_thetas = np.concatenate(self.y_all)

        return all_thetas, all_weights

    def get_round_data(self, n_reuse):
        if n_reuse == -1:
            return np.concatenate(self.x_all), np.concatenate(self.y_all)
        else:
            x_round = self.x_all[max(0, self.round - n_reuse) : self.round + 1]
            x_round = np.concatenate(x_round)

            y_round = self.y_all[max(0, self.round - n_reuse) : self.round + 1]
            y_round = np.concatenate(y_round)

            return x_round, y_round

    def importance_reweight(self, obs, x, y):
        if self.like is None or obs is None:
            return None

        loglike = self.log_like(obs, x, y)
        logprior = self.log_prior(y)
        logproposal = self.log_prob(obs, y)

        log_weights = loglike + logprior - logproposal
        bad = np.isnan(log_weights) + np.isinf(log_weights)
        log_weights -= log_weights[~bad].max()

        weights = np.exp(log_weights)
        weights[bad] = 0
        weights /= weights.sum()

        return weights

    def init_env(self, idx_gpu):
        torch.manual_seed(0)
        np.random.seed(0)
        self.dtype = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            self.map_location = f"cuda:{idx_gpu}"
            torch.cuda.set_device(idx_gpu)
        else:
            self.map_location = "cpu"

    def get_network(self):
        if type(self.network) == DataParallelFlow:
            return self.network.module
        else:
            return self.network

    def get_state_dict(self):
        return self.get_network().state_dict()

    def load_state_dict(self, epoch):
        path_round = os.path.join(self.directory, str(self.round))
        path = os.path.join(path_round, str(epoch) + ".pth")
        self.get_network().load_state_dict(
            torch.load(path, map_location=self.map_location)
        )

    def load_checkpoint(self, network, x_scale, y_scale):
        if type(x_scale) == str:
            x_scale = np.load(x_scale)
        if type(y_scale) == str:
            y_scale = np.load(y_scale)

        self.x_mean = x_scale[0]
        self.x_std = x_scale[1]
        self.y_mean = y_scale[0]
        self.y_std = y_scale[1]
        self.get_network().load_state_dict(
            torch.load(network, map_location=self.map_location)
        )

    def save_state_dict(self):
        path_round = os.path.join(self.directory, str(self.round))
        path_network = os.path.join(path_round, str(self.epoch) + ".pth")
        torch.save(self.get_state_dict(), path_network)

        path_xscales = os.path.join(path_round, "x_scales.npy")
        path_yscales = os.path.join(path_round, "y_scales.npy")
        np.save(path_xscales, np.array([self.x_mean, self.x_std]))
        np.save(path_yscales, np.array([self.y_mean, self.y_std]))

    def scale_y(self, y, back=False):
        if back:
            return y * self.y_std + self.y_mean
        else:
            if len(y.shape) != 2:
                y = np.expand_dims(y, axis=list(range(2 - len(y.shape))))
            return (y - self.y_mean) / self.y_std

    def scale_x(self, x, back=False):
        if back:
            return x * self.x_std + self.x_mean
        else:
            # shape needs to be (N, D, L) for ResNet-GRU
            # todo: make more generic
            if len(x.shape) != 3:
                x = np.expand_dims(x, axis=list(range(3 - len(x.shape))))
            return (x - self.x_mean) / self.x_std

    def infer(
        self, obs, neff_target, y_true=None, corner_before=False, corner_after=False
    ):
        if self.round == 0:
            self.round = 1
        ys = self._draw_params(obs, neff_target)

        if corner_before:
            print("surrogate posterior")
            self.corner(obs, ys, y_true=y_true)

        x_path, good = self.simulate(ys)
        x_path = x_path[good]
        ys = ys[good]
        weights = self.importance_reweight(obs, x_path, ys)

        neff = 1 / (weights**2).sum() - 1
        # print('Initial effective sample size N =', '%.1f' % neff)

        f_accept = neff / neff_target
        if f_accept < 0.005:
            print("failed: sampling efficiency < 0.5%")
            return ys, weights, neff
        print("Sampling efficiency = %.1f" % f_accept)

        n_required = int(neff_target * (1 / f_accept - 1))
        print("Requires N =", n_required, "more simulations")

        ys_extra = self._draw_params(obs, n_required)
        x_path, good = self.simulate(ys_extra)
        x_path = x_path[good]
        ys_extra = ys_extra[good]
        weights_extra = self.importance_reweight(obs, x_path, ys_extra)

        neff_extra = 1 / (weights_extra**2).sum() - 1
        print("Total effective sample size N =", "%.1f" % (neff + neff_extra))

        ys = np.concatenate([ys, ys_extra])
        weights = np.concatenate([weights, weights_extra])

        if corner_after:
            self.corner(obs, ys, y_true=y_true, weights=weights)

        return ys, weights

    def sample(self, x, y=None, n=5000, corner=False):
        x = self.scale_x(x)
        x = torch.from_numpy(x).type(self.dtype)
        with torch.no_grad():
            # GPU memory control (make larger?)
            if n > 20000:
                s = list()
                for i in range(n // 20000 + 1):
                    s.append(self.get_network()(x, n=n, sample=True).cpu().numpy())
                s = np.concatenate(s)[:n]
            else:
                s = self.get_network()(x, n=n, sample=True).cpu().numpy()
        samples = self.scale_y(s, back=True)[0]
        if corner:
            self.corner(x, samples, y_true=y)
        return samples

    def simulate(self, thetas):
        path_round = os.path.join(self.directory, str(self.round))
        try:
            os.mkdir(path_round)
        except:
            pass

        if self.x_file is not None and self.round == 0:
            paths = np.load(self.x_file)
            print("Use precomputed simulations for round ", self.round)
            masks = np.array([True] * len(paths))
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
        np.random.seed(self.epoch)
        self.network.train()
        train_loss = list()
        pbar = self.tqdm(total=len(self.train_loader.dataset))
        for batch_idx, data in enumerate(self.train_loader):
            if len(data) == 2:
                x, y = data
                aux = None
            else:
                x, y, aux = data
                aux = aux.type(self.dtype)
            x = self.scale_x(x).type(self.dtype)
            y = self.scale_y(y).type(self.dtype)
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
                "Train, Log likelihood in nats: {:.6f}".format(-np.mean(train_loss))
            )

        if self.clip > 0:
            self.prev_clip = np.percentile(np.array(self.norm), self.clip)
        pbar.close()
        train_loss = np.array(train_loss).mean()
        self.tloss.append(train_loss)

    def _validate_step(self):
        np.random.seed(0)
        self.network.eval()
        val_loss = list()
        pbar = self.tqdm(total=len(self.valid_loader.dataset))
        pbar.set_description("Eval")
        objs = 0
        for batch_idx, data in enumerate(self.valid_loader):
            x, y = data
            x = self.scale_x(x).type(self.dtype)
            y = self.scale_y(y).type(self.dtype)
            objs += x.shape[0]
            self.optimizer.zero_grad()
            with torch.no_grad():
                loss = self.network(x, y).mean()
                val_loss.append(loss.detach().cpu().numpy())
            pbar.update(x.shape[0])
            pbar.set_description(
                "Val, Log likelihood in nats: {:.6f}".format(
                    -np.sum(val_loss) / (batch_idx + 1)
                )
            )

        # pbar.close()
        val_loss = np.array(val_loss)
        val_loss = val_loss[val_loss < np.percentile(val_loss, 90)].mean()
        pbar.set_description("Val, Log likelihood in nats: {:.6f}".format(-val_loss))
        self.vloss.append(val_loss)

    def _init_wandb(self, project, enable=True):
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
        self.clip = clip
        self.get_network().load_state_dict(self.state_dict_0)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.prev_clip = 1e8
        torch.manual_seed(0)
        self.training_losses = list()
        self.validation_losses = list()

    def _init_scheduler(
        self, min_lr, decay_type="SGDR", patience=5, decay_threshold=0.01
    ):
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
        if self.decay_type == "plateau":
            self.scheduler.step(self.training_losses[-1])
        else:
            self.scheduler.step()

    def _init_loader(self, data_container, train_batch, val_batch):
        train_container, val_container, test_container = data_container.get_splits()

        kwargs = {
            "num_workers": self.n_jobs_loader,
            "pin_memory": False,
            "drop_last": True,
        }

        self.train_loader = DataLoader(
            train_container, batch_size=train_batch, shuffle=True, **kwargs
        )
        self.valid_loader = DataLoader(val_container, batch_size=val_batch, **kwargs)

        self._init_scales()

    def _draw_params(self, x, n):
        # first round: precomputed data or draw from prior
        if self.round == 0:
            if self.y_file is not None:
                return np.load(self.y_file)
            else:
                return self.draw_prior(n)
        # 2+ round: sample from surrogate posterior
        else:
            params = self.sample(x, n=n)
            logprior = self.log_prior(params)
            if np.isinf(logprior).any():
                print("Samples outside prior N =", np.isinf(logprior).sum())
                params = params[~np.isinf(logprior)]
            return params

    def _init_scales(self):
        x_list = list()
        y_list = list()
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
        values = list()
        for i in range(len(y)):
            values.append(self.prior(y[i]))
        return np.array(values)

    def log_like(self, obs, x, y):
        values = list()
        for i in range(len(x)):
            values.append(self.like(obs, x[i], y[i]))
        return np.array(values)

    def log_prob(self, x, y):
        if self.round == 0:
            return self.log_prior(y)

        x = self.scale_x(x)
        y = self.scale_y(y)

        x = torch.from_numpy(x).type(self.dtype)
        y = torch.from_numpy(y).type(self.dtype)
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
            **corner_kwargs,
        )
        plt.show()
