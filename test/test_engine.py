import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import nbi
import numpy as np
import pytest
import torch
from scipy.stats import uniform
from torch import nn
from torch.utils.data import Dataset
from nbi.empirical_prior import EmpiricalPrior

# Common setup variables
t = np.linspace(0, 1, 50)


def sine(param):
    phi0, A, omega = param
    return np.sin(omega * t + phi0) * A


# Define prior
prior = {
    "phi0": uniform(loc=0, scale=np.pi * 2),
    "A": uniform(loc=1, scale=4),
    "omega": uniform(loc=2 * np.pi, scale=10 * np.pi),
}
labels = list(prior.keys())
priors = [prior[k] for k in labels]

# Global y_true and x_obs setup
np.random.seed(0)
y_true = np.array([var.rvs(1)[0] for var in priors])
x_err = 1
x_obs = sine(y_true) + np.random.normal(size=50) * x_err


def fit_and_predict(engine):
    engine.fit(
        x_obs=x_obs,
        y_true=y_true,
        n_sims=320,
        n_rounds=3,
        n_epochs=100,
        batch_size=32,
        lr=0.001,
        min_lr=0.001,
        early_stop_train=True,
        early_stop_patience=1,
        noise=np.array([1] * 50),
        workers=10,
        plot=False,
    )

    y, w = engine.predict(
        x_obs,
        x_err=np.array([0.2] * 50),
        y_true=y_true,
        n_samples=1000,
        neff_min=100,
        f_accept_min=0.1,
        seed=0,
    )

    best_params = engine.best_params
    engine = nbi.NBI(
        state_dict=best_params,
        simulator=sine,
        priors=priors,
        labels=labels,
        path="test",
        device="cpu",
        n_jobs=4,
    )
    y1, w1 = engine.predict(
        x_obs,
        x_err=np.array([0.2] * 50),
        y_true=y_true,
        n_samples=1000,
        neff_min=100,
        f_accept_min=0.1,
        seed=0,
    )

    assert np.allclose(y, y1)


def fit_and_predict_anpe(engine):
    engine.fit(
        x_obs=x_obs,
        n_sims=320,
        n_rounds=1,
        n_epochs=1,
        batch_size=32,
        lr=0.001,
        min_lr=0.001,
        early_stop_train=True,
        early_stop_patience=1,
        workers=10,
        plot=False,
    )

    y = engine.predict(x_obs, n_samples=1000, seed=0)
    assert len(y) == 1000


def test_default_featurizer():
    flow = {
        "n_dims": 3,
        "flow_hidden": 32,
        "num_blocks": 4,
    }

    featurizer = {
        "type": "resnet-gru",
        "norm": "weight_norm",
        "dim_in": 1,
        "dim_out": 32,
        "dim_conv_max": 256,
        "depth": 3,
    }

    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        simulator=sine,
        priors=priors,
        labels=labels,
        path="test",
        device="cpu",
        n_jobs=4,
    )

    fit_and_predict(engine)

    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        simulator=sine,
        priors=priors,
        labels=labels,
        path="test",
        device="cpu",
        n_jobs=4,
    )
    fit_and_predict_anpe(engine)


def test_custom_featurizer():
    flow = {"n_dims": 3, "flow_hidden": 32, "num_blocks": 4, "num_cond_inputs": 64}

    featurizer = nn.Sequential(
        nn.Linear(50, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )

    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        simulator=sine,
        priors=priors,
        labels=labels,
        path="test",
        device="cpu",
        n_jobs=4,
    )

    fit_and_predict(engine)
    fit_and_predict_anpe(engine)


class MultiChannelSineDataset(Dataset):
    """Synthetic dataset returning two channels per sample: (sine, cosine)."""

    def __init__(self, n_samples=500, seq_len=50, seed=0):
        rng = np.random.default_rng(seed)
        self.t = np.linspace(0, 1, seq_len).astype(np.float32)
        self.params = np.column_stack([
            rng.uniform(0, 2 * np.pi, n_samples),   # phi0
            rng.uniform(1, 5, n_samples),            # A
            rng.uniform(2 * np.pi, 12 * np.pi, n_samples),  # omega
        ]).astype(np.float32)

        # Pre-compute both channels
        self.ch0 = []  # sine channel
        self.ch1 = []  # cosine channel
        for phi0, A, omega in self.params:
            self.ch0.append(A * np.sin(omega * self.t + phi0))
            self.ch1.append(A * np.cos(omega * self.t + phi0))

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        # Each channel: [1, seq_len]  (C=1 per channel)
        x0 = torch.tensor(self.ch0[idx][None, :], dtype=torch.float32)
        x1 = torch.tensor(self.ch1[idx][None, :], dtype=torch.float32)
        y = torch.tensor(self.params[idx], dtype=torch.float32)
        return [x0, x1], y


def test_multi_modal():
    """Test multi-channel input with per-channel featurizers and DatasetContainer."""
    dim_out = 32
    flow = {
        "n_dims": 3,
        "flow_hidden": 32,
        "num_blocks": 4,
        "num_cond_inputs": dim_out * 2,
    }

    featurizer_ch0 = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(50, 64),
        nn.ReLU(),
        nn.Linear(64, dim_out),
    )
    featurizer_ch0.num_outputs = dim_out

    featurizer_ch1 = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(50, 64),
        nn.ReLU(),
        nn.Linear(64, dim_out),
    )
    featurizer_ch1.num_outputs = dim_out

    dataset = MultiChannelSineDataset(n_samples=500, seq_len=50)

    engine = nbi.NBI(
        flow=flow,
        featurizer=[featurizer_ch0, featurizer_ch1],
        labels=labels,
        path="test_multimodal",
        device="cpu",
    )

    engine.fit(
        x=dataset,
        n_sims=-1,
        n_rounds=1,
        n_epochs=2,
        batch_size=32,
        lr=0.001,
        min_lr=0.001,
        workers=0,
        plot=False,
    )

    # Build a single observation from both channels
    y_test = np.array([1.0, 2.0, 6.0], dtype=np.float32)
    x0_obs = (2.0 * np.sin(6.0 * t + 1.0)).astype(np.float32)[None, :]  # [1, L]
    x1_obs = (2.0 * np.cos(6.0 * t + 1.0)).astype(np.float32)[None, :]
    x_obs_multi = [x0_obs, x1_obs]

    samples = engine.predict(x_obs_multi, n_samples=100, seed=0)
    assert samples.shape == (100, 3)

    # Test save/load round-trip
    best_params = engine.get_params()
    engine2 = nbi.NBI(
        state_dict=best_params,
        featurizer=[featurizer_ch0, featurizer_ch1],
        labels=labels,
        path="test_multimodal2",
        device="cpu",
    )
    samples2 = engine2.predict(x_obs_multi, n_samples=100, seed=0)
    assert np.allclose(samples, samples2)


def test_importance_reweight_returns_zeros_when_all_log_weights_invalid():
    flow = {
        "n_dims": 3,
        "flow_hidden": 16,
        "num_blocks": 2,
    }

    featurizer = {
        "type": "resnet-gru",
        "norm": "weight_norm",
        "dim_in": 1,
        "dim_out": 16,
        "dim_conv_max": 32,
        "depth": 2,
    }

    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        simulator=sine,
        priors=priors,
        labels=labels,
        path="test_invalid_weights",
        device="cpu",
        n_jobs=1,
    )

    engine.like = True  # must be non-None to enter the reweight logic
    engine.log_like = lambda x_obs, x, y: np.full(len(y), np.nan)
    engine.log_prob = lambda x_obs, y: np.zeros(len(y))
    y = np.zeros((5, 3), dtype=np.float32)

    weights = engine.importance_reweight(np.zeros(50), np.zeros((5, 50)), y)
    assert weights is not None
    assert weights.shape == (5,)
    assert np.all(weights == 0)


def test_empirical_prior_logpdf_with_list_priors():
    """EmpiricalPrior.logpdf must handle a list of per-parameter scipy distributions."""
    lookup = np.column_stack([
        uniform(loc=0, scale=2 * np.pi).rvs(100),
        uniform(loc=1, scale=4).rvs(100),
        uniform(loc=2 * np.pi, scale=10 * np.pi).rvs(100),
    ])
    ep = EmpiricalPrior(lookup, priors=priors)

    # Samples inside the prior support should give finite log-probabilities
    params_in = ep.rvs(10)
    lp = ep.logpdf(params_in)
    assert lp.shape == (10,)
    assert np.all(np.isfinite(lp))

    # A point outside every prior's support should give -inf
    params_out = np.array([[-999, -999, -999]])
    lp_out = ep.logpdf(params_out)
    assert np.all(np.isinf(lp_out))


def test_empirical_prior_logpdf_raises_without_priors():
    """EmpiricalPrior without priors must raise NotImplementedError on logpdf."""
    lookup = np.random.randn(50, 3)
    ep = EmpiricalPrior(lookup)
    with pytest.raises(NotImplementedError, match="logpdf not implemented"):
        ep.logpdf(np.random.randn(5, 3))


def test_snpe_with_empirical_prior():
    """SNPE (n_rounds > 1) must work when the prior is an EmpiricalPrior with list priors."""
    # Build lookup table by sampling from the analytic priors
    np.random.seed(42)
    lookup = np.column_stack([p.rvs(500) for p in priors])
    emp_prior = EmpiricalPrior(lookup, priors=priors, random_state=42)

    flow = {
        "n_dims": 3,
        "flow_hidden": 32,
        "num_blocks": 4,
    }

    featurizer = {
        "type": "resnet-gru",
        "norm": "weight_norm",
        "dim_in": 1,
        "dim_out": 32,
        "dim_conv_max": 256,
        "depth": 3,
    }

    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        simulator=sine,
        priors=emp_prior,
        labels=labels,
        path="test_empirical_snpe",
        device="cpu",
        n_jobs=4,
    )

    # Run 2 rounds of SNPE — this exercises log_prior via importance_reweight
    engine.fit(
        x_obs=x_obs,
        y_true=y_true,
        n_sims=320,
        n_rounds=2,
        n_epochs=100,
        batch_size=32,
        lr=0.001,
        min_lr=0.001,
        early_stop_train=True,
        early_stop_patience=1,
        noise=np.array([1] * 50),
        workers=10,
        plot=False,
    )

    y, w = engine.predict(
        x_obs,
        x_err=np.array([0.2] * 50),
        y_true=y_true,
        n_samples=1000,
        neff_min=100,
        f_accept_min=0.1,
        seed=0,
    )

    assert y.shape[1] == 3
    assert w is not None


def test_anpe_with_empirical_prior():
    """ANPE (n_rounds=1) must work with EmpiricalPrior with list priors."""
    np.random.seed(42)
    lookup = np.column_stack([p.rvs(500) for p in priors])
    # priors needed because predict() calls _draw_params() → log_prior()
    emp_prior = EmpiricalPrior(lookup, priors=priors, random_state=42)

    flow = {
        "n_dims": 3,
        "flow_hidden": 32,
        "num_blocks": 4,
    }

    featurizer = {
        "type": "resnet-gru",
        "norm": "weight_norm",
        "dim_in": 1,
        "dim_out": 32,
        "dim_conv_max": 256,
        "depth": 3,
    }

    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        simulator=sine,
        priors=emp_prior,
        labels=labels,
        path="test_empirical_anpe",
        device="cpu",
        n_jobs=4,
    )

    engine.fit(
        x_obs=x_obs,
        n_sims=320,
        n_rounds=1,
        n_epochs=1,
        batch_size=32,
        lr=0.001,
        min_lr=0.001,
        workers=10,
        plot=False,
    )

    y = engine.predict(x_obs, n_samples=1000, seed=0)
    assert len(y) == 1000


if __name__ == "__main__":
    pytest.main()
