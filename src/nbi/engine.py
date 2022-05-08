from .model import get_flow, DataParallelFlow
from .data import BaseContainer

import os
import corner
import copy
import numpy as np
import matplotlib.pyplot as plt
import wandb
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm

import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingWarmRestarts

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

corner_kwargs = {'quantiles': [0.16, 0.5, 0.84],
                 'show_titles': True,
                 'title_kwargs': {"fontsize": 16},
                 'fill_contours': True,
                 'levels': 1.0 - np.exp(-0.5 * np.arange(0.5, 2.6, 0.5) ** 2)}

default_flow_config = {
    'flow_hidden': 256,
    'num_cond_inputs': 256,
    'num_blocks': 20,
    'perm_seed': 3,
    'n_mog': 8
}


class NBI:
    """ Neural bayesian inference engine for astronomical data
    """

    def __init__(
            self,
            featurizer,
            dim_param,
            physics=None,
            prior=None,
            log_prior=None,
            log_like=None,
            instrumental=None,
            flow_config={},
            idx_gpu=0,
            parallel=False,
            directory='',
            n_jobs=1,
            modify_scales=None,
            labels=None
    ):
        """

        Parameters
        ----------
        featurizer (nn.Module): pytorch network which maps input sequence of shape [Batch, Channel, Length] to
                output feature vector of shape [Batch, Dimension]. See NBI.get_featurizer() for pre-defined ones.
        dim_param (int): number of inferred parameters
        physics (callable): a function which takes (thetas, files)
        prior
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
        corner_kwargs.update({'labels':labels})

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

        self.prior = prior
        self.log_prior = log_prior
        self.log_like = log_like
        self.simulator = physics
        self.process = instrumental
        self.directory = directory
        self.n_jobs = n_jobs

        self.round = 0
        self.x_all = list()
        self.y_all = list()
        self.weights = list()
        self.neff = list()

        try:
            os.mkdir(self.directory)
        except:
            pass

    def train(self, *args, **kwargs):
        # deprecated
        return self.run(*args, **kwargs)

    def run(
            self,
            obs,
            n_rounds,
            n_per_round,
            n_epochs,
            n_reuse=0,
            y_true=None,
            train_batch=512,
            val_batch=512,
            project='test',
            wandb_enabled=False,
            neff_stop=-1,
            early_stop_train=True,
            f_val=0,
            lr=0.001,
            min_lr=None,
            x_file=None,
            y_file=None,
            decay_type='SGDR'
    ):

        self.n_epochs = n_epochs
        self.x_file = x_file
        self.y_file = y_file
        self._init_train(lr)
        self._init_wandb(project, wandb_enabled)

        if min_lr is None:
            min_lr = lr * 0.001

        ys = self.sample(obs, n_per_round)
        x_path = self.simulate(ys)
        weights = self.importance_reweight(obs, ys, x_path)
        np.save(os.path.join(self.directory, '0_x.npy'), x_path)
        np.save(os.path.join(self.directory, '0_y.npy'), ys)
        self.x_all.append(x_path)
        self.y_all.append(ys)
        self.weights.append(weights)

        for i in range(n_rounds):
            self._init_scheduler(min_lr, decay_type=decay_type)
            print('\nRound: {}'.format(i))

            x_round, y_round = self.get_round_data(n_reuse)
            data_container = BaseContainer(x_round, y_round, f_test=0, f_val=f_val, process=self.process)
            self._init_loader(data_container, train_batch, val_batch)

            for epoch in range(n_epochs):
                print('\nEpoch: {}'.format(epoch))
                self._train_step()
                self._step_scheduler()
                # self._validate_step()
                if self.wandb:
                    wandb.log({"Train Loss": self.training_losses[-1], "Val Loss": self.validation_losses[-1]})
                self.epoch = epoch

            self.round += 1
            ys = self.sample(obs, n_per_round)
            x_path = self.simulate(ys)
            weights = self.importance_reweight(obs, ys, x_path)
            np.save(os.path.join(self.directory, str(self.round + 1)) + '_x.npy', x_path)
            np.save(os.path.join(self.directory, str(self.round + 1)) + '_y.npy', ys)
            self.x_all.append(x_path)
            self.y_all.append(ys)
            self.weights.append(weights)

            if self.log_like is not None:
                neff = 1 / (weights ** 2).sum()
                self.neff.append(neff)
                print('Effective sample size for round ' + str(self.round) + ': ', '%.1f' % neff)
                print('Effective sample size for all rounds: ', '%.1f' % np.sum(self.neff))

            print('posterior from round ' + str(self.round))
            self.corner(obs, ys, truth=y_true, weights=weights)

            print('posterior from all rounds')
            all_thetas, all_weights = self.result()
            self.corner(obs, all_thetas, truth=y_true, weights=all_weights)

            if y_true is not None:
                loglike = self.log_prob(obs, y_true)
                if self.wandb:
                    wandb.log({"loglike": loglike})
                print(loglike)
            self.epoch = 0

            if np.sum(self.neff) > neff_stop and neff_stop > 0:
                print('early stopping')
                return
            elif early_stop_train and self.neff[-1] * 1.1 < self.neff[-2]:
                n_required = neff_stop - np.sum(self.neff)
                n_required /= self.neff[-1] / n_per_round
                n_required = int(n_required)
                print('stop training')
                print('importance sampling N =', n_required)
                ys, weights = self.sample(obs, n_required)
                neff = 1 / (weights ** 2).sum()

                self.round += 1
                x_path = self.simulate(ys)
                np.save(os.path.join(self.directory, str(self.round + 1)) + '_x.npy', x_path)
                np.save(os.path.join(self.directory, str(self.round + 1)) + '_y.npy', ys)
                self.x_all.append(x_path)
                self.y_all.append(ys)
                self.neff.append(neff)
                self.weights.append(weights)
                print('Effective sample size is %.1f' % neff)
                return

    def result(self):
        all_weights = np.concatenate(np.array(self.weights) * np.array(self.neff)[:, None])
        all_weights /= all_weights.sum()
        all_thetas = np.concatenate(self.y_all)

        return all_thetas, all_weights

    def get_round_data(self, n_reuse):
        if n_reuse == -1:
            return np.concatenate(self.x_all), np.concatenate(self.y_all)
        else:
            x_round = self.x_all[max(0, self.round - n_reuse): self.round + 1]
            x_round = np.concatenate(x_round)

            y_round = self.y_all[max(0, self.round - n_reuse): self.round + 1]
            y_round = np.concatenate(y_round)

            return x_round, y_round

    def sample(self, x, n):
        thetas = self._draw_params(x, n)
        return thetas

    def importance_reweight(self, x, y, x_path):
        if self.log_like is None:
            return None
        try:
            loglike = self.log_like(x, y, x_path)
        except:
            loglike = self.log_like(x, y)

        logprior = self.log_prior(y)
        logproposal = self.log_prob(x, y)

        log_weights = loglike + logprior - logproposal
        log_weights -= log_weights[~np.isnan(weights)].sum()
        log_weights[np.isnan(weights)] = 0
        weights = np.exp(log_weights)

        return weights

    def init_env(self, idx_gpu):
        torch.manual_seed(0)
        np.random.seed(0)
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            self.map_location = f'cuda:{idx_gpu}'
            torch.cuda.set_device(idx_gpu)
        else:
            self.map_location = 'cpu'

    def get_network(self):
        if type(self.network) == DataParallelFlow:
            return self.network.module
        else:
            return self.network

    def get_state_dict(self):
        return self.get_network().state_dict()

    def load_state_dict(self, file, x_scale, y_scale):
        self.x_mean = x_scale[:, 0]
        self.x_std = x_scale[:, 1]
        self.y_mean = y_scale[:, 0]
        self.y_std = y_scale[:, 1]
        self.get_network().load_state_dict(torch.load(file, map_location=self.map_location))

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
            if len(x.shape) != 3:
                x = np.expand_dims(x, axis=list(range(3 - len(x.shape))))
            return (x - self.x_mean) / self.x_std

    def infer(self, x, n=5000):
        x = self.scale_x(x)
        x = torch.from_numpy(x).type(self.dtype)
        with torch.no_grad():
            s = self.get_network()(x, n=n, sample=True).cpu().numpy()
        return self.scale_y(s, back=True)[0]

    def simulate(self, thetas):
        if self.x_file is not None:
            paths = np.load(self.x_file)
        else:
            path_round = os.path.join(self.directory, str(self.round))
            try:
                os.mkdir(path_round)
            except:
                pass
            n = len(thetas)
            paths = np.array([os.path.join(path_round, str(i)+'.npy') for i in range(n)])
            per_job = n // self.n_jobs
            jobs = [[thetas[i * per_job: (i + 1) * per_job], paths[i * per_job: (i + 1) * per_job]] for i in range(self.n_jobs)]
            with Pool(self.n_jobs) as p:
                p.map(self.simulator, jobs)
        return paths

    def _train_step(self):
        np.random.seed(self.epoch)
        self.network.train()
        train_loss = list()
        pbar = tqdm(total=len(self.train_loader.dataset))
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
                self.norm.append(torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.prev_clip).cpu())
            self.optimizer.step()

            pbar.update(x.shape[0])
            pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
                -np.mean(train_loss)))

        if self.clip > 0:
            self.prev_clip = np.percentile(np.array(self.norm), self.clip)
        pbar.close()
        train_loss = np.array(train_loss).mean()
        self.tloss.append(train_loss)

    def _validate_step(self):
        np.random.seed(0)
        self.network.eval()
        val_loss = list()
        pbar = tqdm(total=len(self.valid_loader.dataset))
        pbar.set_description('Eval')
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
            pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
                -np.sum(val_loss) / (batch_idx + 1)))

        # pbar.close()
        val_loss = np.array(val_loss)
        val_loss = val_loss.mean()
        self.vloss.append(val_loss)

    def _init_wandb(self, project, enable=True):
        self.wandb = enable
        if enable:
            wandb.init(project=project, config=self.args, name=self.name)
            wandb.watch(self.network)

    def _init_train(self, lr, clip=85):

        self.epoch = 0
        self.clip = clip
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.best_validation_loss = float('inf')
        self.best_validation_epoch = 0

        self.prev_clip = 1e8
        torch.manual_seed(0)
        self.training_losses = list()
        self.validation_losses = list()

    def _init_scheduler(self, min_lr, decay_type='SGDR', patience=5, decay_threshold=0.01):
        self.decay_type = decay_type
        if decay_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=patience,
                                               threshold_mode='abs', cooldown=0, verbose=True,
                                               threshold=decay_threshold, min_lr=1e-6)

        elif 'SGDR' in decay_type:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.n_epochs,
                T_mult=1,
                eta_min=min_lr
            )
        else:
            self.scheduler = MultiStepLR(self.optimizer, np.array(decay_type.split(','), dtype=int), gamma=0.1)

    def _step_scheduler(self):
        if self.decay_type == 'plateau':
            self.scheduler.step(self.training_losses[-1])
        else:
            self.scheduler.step()

    def _init_loader(self, data_container, train_batch, val_batch):
        train_container, \
        val_container, \
        test_container \
            = data_container.get_splits()

        kwargs = {'num_workers': 8 if torch.cuda.is_available() else 1, 'pin_memory': False, 'drop_last': True}

        self.train_loader = DataLoader(train_container, batch_size=train_batch, shuffle=True, **kwargs)
        self.valid_loader = DataLoader(val_container, batch_size=val_batch, **kwargs)

        if self.round == 0:
            self._init_scales()

    def _draw_params(self, x, n):
        if self.y_file is not None:
            return np.load(self.y_file)
        elif self.round == 0:
            return self.prior(n)
        else:
            return self.infer(x, n)

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

    def log_prob(self, x, y):
        if self.round == 0:
            return self.log_prior(y)
        x = self.scale_x(x)
        y = self.scale_y(y)
        x = torch.from_numpy(x).type(self.dtype)
        y = torch.from_numpy(y).type(self.dtype)
        with torch.no_grad():
            log_prob = self.network(x, y).cpu().numpy()[:,0] * -1
        return log_prob

    def corner(
            self,
            x,
            y=None,
            weights=None,
            color='k',
            truth=None,
            plot_datapoints=True,
            plot_density=False,
            range_=0.95,
            truth_color='r'
    ):

        range_ = [range_] * self.ndim
        if y is None:
            y = self.infer(x)
        corner.corner(y,
                       truths=truth,
                       color=color,
                       plot_datapoints=plot_datapoints,
                       # range=range_,
                       plot_density=plot_density,
                       truth_color=truth_color,
                       weights=weights,
                       **corner_kwargs)
        plt.show()
