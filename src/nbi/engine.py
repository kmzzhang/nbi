import os
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import corner
import numpy as np
import matplotlib.pyplot as plt
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

from .model import get_flow, DataParallelFlow, get_featurizer
from .data import BaseContainer
from .utils import parallel_simulate, iid_gaussian, log_like_iidg

corner_kwargs = {
    "quantiles": [0.16, 0.5, 0.84],
    "show_titles": True,
    "title_kwargs": {"fontsize": 16},
    "fill_contours": True,
    "levels": 1.0 - np.exp(-0.5 * np.arange(0.5, 2.6, 0.5) ** 2),
}

default_flow_config = {
    "flow_hidden": 64,
    "num_cond_inputs": 64,
    "num_blocks": 5,
    "perm_seed": 3,
    "n_mog": 1,
}

class NBI:
    """Neural bayesian inference engine for astronomical data"""

    def __init__(
        self,
        flow,
        featurizer,
        simulator=None,
        priors=None,
        idx_gpu=0,
        path="test",
        n_jobs=1,
        labels=None,
        tqdm_notebook=False,
        network_reinit=False,
        scale_reinit=True
    ):
        """

        Parameters
        ----------
        featurizer (nn.Module): pytorch network which maps input sequence of shape [Batch, Channel, Length] to
                output feature vector of shape [Batch, Dimension]. See NBI.get_featurizer() for pre-defined ones.
        dim_param (int): number of inferred parameters
        simulator (callable): a function which takes (thetas, files)
        prior_sampler
        log_prior
        log_like
        noise
        flow_config
        idx_gpu
        parallel
        directory
        n_jobs
        labels
        """

        # if labels is None and prior is None:
        #     raise AssertionError('Parameter name must be provided via labels when prior not supplied!')

        self.init_env(idx_gpu)
        flow_config_all = copy.copy(default_flow_config)
        flow_config_all.update(flow)
        corner_kwargs.update({"labels": labels})
        self.network_reinit = network_reinit
        self.scale_reinit = scale_reinit

        # if featurizer is not user provided pytorch module
        # generate featurizer network based on user specified type and hyperparameters
        if type(featurizer) == dict:
            featurizer["dim_out"] = flow_config_all["num_cond_inputs"]
            featurizer = get_featurizer(featurizer.pop("type"), featurizer)

        self.network = get_flow(featurizer, **flow_config_all).type(
            self.dtype
        )
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


        # self.draw_prior = prior_sampler
        self.prior = priors
        self.param_names = labels
        self.simulator = simulator
        self.directory = path
        self.n_jobs = n_jobs
        self.obs = None
        self.y_true = None

        self.x = None
        self.y = None

        self.process = None
        self.like = None

        self.round = 0
        self.early_stop_count = 0
        self.x_all = list()
        self.y_all = list()
        self.weights = list()
        self.neff = list()
        self.state_dict_0 = self.get_state_dict()

        self.prev_state = list()
        self.prev_x_mean = list()
        self.prev_x_std = list()
        self.prev_y_mean = list()
        self.prev_y_std = list()

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
        workers=4
    ):
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
        self.obs = x_obs
        self.y_true = y_true
        
        self.x = x
        self.y = y


        self._init_wandb(project, wandb_enabled)

        if min_lr is None:
            min_lr = lr / n_epochs

        # for restarting training
        if len(self.x_all) == self.round:
            # this is not a restart because
            # data for this round has not been generated
            self.prepare_data(x_obs, n_sims)

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


            self.prepare_data(x_obs, n_sims)

            if self.round > 0 and plot:
                self.weighted_corner(x_obs, y_true)

            # early stopping
            if np.sum(self.neff) > neff_stop > 0:
                print("Success: Exceed specified stopping sample size!")
                self.corner_all(x_obs, y_true)
                return

            f_accept_round = self.neff[-1] / n_sims
            if self.neff[-1] / n_sims > f_accept_min > 0:
                print("Success: Sampling efficiency is {:.1f}!".format(f_accept_round))
                self.corner_all(x_obs, y_true)
                return

            if early_stop_train and self.round > 1:
                if 1 < self.neff[-1] < self.neff[-2]:
                    print(
                        "Early stop: Surrogate posterior did not improve for this round"
                    )
                    self.load_prev_state(self.round - 2)
                    return

        if x_obs is not None:
            self.corner_all(x_obs, y_true)

    def save_current_state(self):
        prev_state = self.get_state_dict()
        self.prev_state.append(prev_state)
        self.prev_x_mean.append(self.x_mean)
        self.prev_x_std.append(self.x_std)
        self.prev_y_mean.append(self.y_mean)
        self.prev_y_std.append(self.y_std)

    def load_prev_state(self, round):
        print("Loaded state from round ", round)
        self.get_network().load_state_dict(self.prev_state[round])
        self.x_mean = self.prev_x_mean[round]
        self.x_std = self.prev_x_std[round]
        self.y_mean = self.prev_y_mean[round]
        self.y_std = self.prev_y_std[round]

    def corner_all(self, x_obs, y_true):
        print("reweighted posterior from all rounds")
        all_thetas, all_weights = self.result()
        self.corner(x_obs, all_thetas, y_true=y_true, weights=all_weights)

    def prepare_data(self, x_obs, n_sims):
        ys = self._draw_params(x_obs, n_sims)

        np.save(os.path.join(self.directory, str(self.round)) + "_y_all.npy", ys)

        x_path, good = self.simulate(ys)
        np.save(os.path.join(self.directory, str(self.round)) + "_x.npy", x_path[good])
        np.save(os.path.join(self.directory, str(self.round)) + "_y.npy", ys[good])

        self.x_all.append(np.array(x_path)[good])
        self.y_all.append(np.array(ys)[good])

        weights = self.importance_reweight(
            x_obs,
            self.x_all[-1],
            self.y_all[-1]
        )
        self.weights.append(weights)
        np.save(os.path.join(self.directory, str(self.round)) + "_w.npy", weights)

        if self.like is not None and x_obs is not None:
            neff = 1 / (weights**2).sum() - 1
            self.neff.append(neff)
            print(
                "Effective sample size for current/all rounds",
                "%.1f/%.1f" % (neff, np.sum(self.neff)),
            )
            # print("Effective sample size for all rounds: ", "%.1f" % np.sum(self.neff))

    def weighted_corner(self, x_obs, y_true):
        try:
            # print("reweighted posterior from current round")
            self.corner(x_obs, self.y_all[-1], y_true=y_true, weights=self.weights[-1])

        except:
            print("corner plot failed")

    def stop_training(self, patience=1):
        if self.epoch < patience + 2 or patience == -1:
            return False

        prev_losses = np.array(self.vloss[-1 * patience - 1 :])
        base_loss = self.vloss[-1 * patience - 2]
        return (prev_losses > base_loss).all()

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

    def importance_reweight(self, x_obs, x, y):
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
        if self.like is None or x_obs is None:
            return None

        log_weights = self.log_like(x_obs, x, y)
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
        return copy.deepcopy(self.get_network().state_dict())

    def load_state_dict(self, epoch):
        path_round = os.path.join(self.directory, str(self.round))
        path = os.path.join(path_round, str(epoch) + ".pth")
        self.get_network().load_state_dict(
            torch.load(path, map_location=self.map_location)
        )

    def set_params(self, network, x_scale, y_scale):
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

    def predict(
        self,
        x,
        x_err=None,
        y_true=None,
        log_like=None,
        n_samples=1000,
        n_max=-1,
        neff_min=1,
        corner=False,
        corner_reweight=False
    ):

        self.like = (
            log_like_iidg(x_err) if type(x_err) == np.ndarray else log_like
        )

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
        print("Effective Sample Size = {0:.1f}".format(neff))
        print("Sampling efficiency = {0:.1f}%".format(f_accept * 100))

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
        x = self.scale_x(x)
        x = torch.from_numpy(x).type(self.dtype)
        with torch.no_grad():
            # GPU memory control (make larger?)
            if n > 20000:
                s = list()
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
                "Epoch {:d}: Train, Loglike in nats: {:.6f}".format(
                    self.epoch, -np.mean(train_loss)
                )
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
                "- Val, Loglike in nats: {:.6f}".format(
                    -np.sum(val_loss) / (batch_idx + 1)
                )
            )

        # pbar.close()
        val_loss = np.array(val_loss)
        val_loss = val_loss[val_loss < np.percentile(val_loss, 90)].mean()
        pbar.set_description("- Val, Loglike in nats: {:.6f}".format(-val_loss))
        self.vloss.append(val_loss)

    def _init_wandb(self, project, enable):
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
        if self.network_reinit:
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

    def _init_loader(self, data_container, batch_size, workers=4):
        train_container, val_container, test_container = data_container.get_splits()

        kwargs = {
            "num_workers": workers,
            "pin_memory": False,
            "drop_last": True,
        }

        self.train_loader = DataLoader(
            train_container, batch_size=batch_size, shuffle=True, **kwargs
        )
        self.valid_loader = DataLoader(val_container, batch_size=batch_size, **kwargs)

        # if self.network_reinit or self.round == 0:
        if self.scale_reinit or self.round == 0:
            self._init_scales()

    def _draw_params(self, x, n):
        # first round: precomputed data or draw from prior
        if self.round == 0:
            if self.y is not None:
                return self.y
            else:
                params = list()
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
        if self.prior is None:
            return np.zeros(len(y))
        else:
            log_prob = np.zeros(len(y))
            for i, prior in enumerate(self.prior):
                log_prob += prior.logpdf(y[:, i])
        return log_prob

    def log_like(self, x_obs, x, y):
        values = list()
        for i in range(len(x)):
            values.append(self.like(x_obs, x[i], y[i]))
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
