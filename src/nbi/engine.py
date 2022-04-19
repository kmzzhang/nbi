import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import corner
import copy
import numpy as np
import matplotlib.pyplot as plt
import wandb
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm

from .model import get_flow, DataParallelFlow
from .data import BaseContainer

import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingWarmRestarts


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
    def __init__(self, featurizer, dim_param, flow_config={}, idx_gpu=0, parallel=False,
                 simulator=None, directory='', prior=None, n_jobs=1, process=None, modify_scales=None, labels=None):
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
        self.simulator = simulator
        self.process = process
        self.directory = directory
        self.n_jobs=n_jobs

        try:
            os.mkdir(self.directory)
        except:
            pass

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
            return (y - self.y_mean) / self.y_std

    def scale_x(self, x, back=False):
        if back:
            return x * self.x_std + self.x_mean
        else:
            return (x - self.x_mean) / self.x_std

    def infer(self, x, n=5000):
        x = self.scale_x(x)
        x = torch.from_numpy(x).type(self.dtype)
        with torch.no_grad():
            s = self.get_network()(x, n=n, sample=True).cpu().numpy()
        return self.scale_y(s, back=True)[0]

    def simulate(self, thetas, x_files=None):
        if x_files is not None:
            paths = np.load(x_files)
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

    def _init_scheduler(self, lr, decay_type='SGDR', patience=5, decay_threshold=0.01):
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
                eta_min=1e-6
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

    def _draw_params(self, x, n, y_file=None):
        if y_file is not None:
            return np.load(y_file)
        elif self.round == 0:
            return self.prior(n)
        else:
            if len(x.shape) == 1:
                return self.infer(x[None, None, :], n)
            else:
                return self.infer(x[None, :], n)
    """
    def apt_loss(self, x, y):
       
        Code taken from sbi: https://github.com/mackelab/sbi
        Return log probability of the proposal posterior for atomic proposals.
        We have two main options when evaluating the proposal posterior.
            (1) Generate atoms from the proposal prior.
            (2) Generate atoms from a more targeted distribution, such as the most
                recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly-initialized neural density
        estimator.
        Args:
            theta: Batch of parameters θ.  (N, D)
            x: Batch of data.              (N, 1, 7200)
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
        Returns:
            Log-probability of the proposal posterior.
        

        batch_size = theta.shape[0]

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self.network(repeated_x, atomic_theta) * -1
        # _assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)
        # print(nde.transform_pm(atomic_theta))
        # print(log_prob_posterior)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = ln_prior(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        # _assert_all_finite(log_prob_prior, "prior eval")

        # Compute unnormalized proposal posterior.
        # print(log_prob_posterior, torch.from_numpy(log_prob_prior))
        unnormalized_log_prob = log_prob_posterior - torch.from_numpy(log_prob_prior).type(dfloat)
        # unnormalized_log_prob = log_prob_posterior
        # print('log_prob: [true theta, contrasting atoms ...]')
        # print(log_prob_posterior.mean())

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
            unnormalized_log_prob, dim=-1
        )
        # print(log_prob_proposal_posterior[0])
        # _assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        # XXX This evaluates the posterior on _all_ prior samples
        # if _use_combined_loss:
        #     log_prob_posterior_non_atomic = _posterior.net.log_prob(theta, x)
        #     masks = masks.reshape(-1)
        #     log_prob_proposal_posterior = (
        #             masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
        #     )

        return log_prob_proposal_posterior * -1, unnormalized_log_prob[:, 0]
    """

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

    def train(self, x, n_rounds, n_per_round, n_epochs, y=None, train_batch=512, val_batch=512,
              project='test', wandb_enabled=False, f_val=0.2, lr=0.001, x_file=None, y_file=None, decay_type='SGDR'):

        self.n_epochs = n_epochs
        self._init_train(lr)
        self._init_wandb(project, wandb_enabled)
        self.x_paths = list()
        self.ys = None

        for round in range(n_rounds):
            self._init_scheduler(lr, decay_type=decay_type)
            self.round = round
            print('\nRound: {}'.format(round))
            thetas = self._draw_params(x, n_per_round, y_file if round == 0 else None)
            x_path = self.simulate(thetas, x_file if round == 0 else None)
            np.save(os.path.join(self.directory, str(round)) + '_x.npy', x_path)
            np.save(os.path.join(self.directory, str(round)) + '_y.npy', thetas)
            self.x_paths.extend(x_path)
            # if self.ys is None:
            #     self.ys = thetas
            # else:
            #     self.ys = np.concatenate([self.ys, thetas], axis=0)
            data_container = BaseContainer(x_path, thetas, f_test=0, f_val=f_val, process=self.process)
            self._init_loader(data_container, train_batch, val_batch)
            if round == 0:
                self._init_scales()
            for epoch in range(n_epochs):
                print('\nEpoch: {}'.format(epoch))
                self._train_step()
                self._step_scheduler()
                # self._validate_step()
                if self.wandb:
                    wandb.log({"Train Loss": self.training_losses[-1], "Val Loss": self.validation_losses[-1]})
                self.epoch = epoch
            self.epoch=0
            self.corner(x, n=100000, y=y)
            loglike = self.log_prob(x, y)
            if self.wandb:
                wandb.log({"loglike": loglike})
            print(loglike)

    def log_prob(self, x, y):
        if len(x.shape) == 1:
            x = x[None, None, :]
        else:
            x = x[None, :]
        x = self.scale_x(x)
        y = self.scale_y(y[None, :])
        x = torch.from_numpy(x).type(self.dtype)
        y = torch.from_numpy(y).type(self.dtype)
        with torch.no_grad():
            log_prob = self.network(x, y).cpu().numpy()[0] * -1
        return log_prob

    def corner(self, x, n, color='k', y=None, plot_datapoints=False, plot_density=False, range_=0.95, truth_color='r', seed=0):
        range_ = [range_] * self.ndim
        if len(x.shape) == 1:
            x = x[None, None, :]
        else:
            x = x[None, :]
        sample = self.infer(x)
        figure = corner.corner(sample,
                               truths=y,
                               color=color,
                               plot_datapoints=plot_datapoints,
                               range=range_,
                               plot_density=plot_density,
                               truth_color=truth_color,
                               **corner_kwargs)
        plt.show()
