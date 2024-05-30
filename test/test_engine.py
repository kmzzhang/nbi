import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import nbi
import numpy as np
import pytest
from scipy.stats import uniform
from torch import nn

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
        x_obs, x_err=np.array([0.2] * 50), y_true=y_true, n_samples=1000, seed=0
    )

    best_params = engine.best_params
    engine = nbi.NBI(
        state_dict=best_params,
        simulator=sine,
        priors=priors,
        labels=labels,
        path="test",
        device="cpu",
        n_jobs=10,
    )
    y1, w1 = engine.predict(
        x_obs, x_err=np.array([0.2] * 50), y_true=y_true, n_samples=1000, seed=0
    )

    assert np.allclose(y, y1)


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
        n_jobs=10,
    )

    fit_and_predict(engine)


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
        n_jobs=10,
    )

    fit_and_predict(engine)


if __name__ == "__main__":
    pytest.main()
