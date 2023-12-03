import nbi
import numpy as np
from scipy.stats import uniform


def test():
    t = np.linspace(0, 1, 50)

    def sine(param):
        phi0, A, omega = param
        return np.sin(omega * t + phi0) * A

    # nbi requires prior to be defined with scipy functions
    # alternatively, you may supply pre-generated parameters with numpy arrays

    prior = {
        "phi0": uniform(loc=0, scale=np.pi * 2),
        "A": uniform(loc=1, scale=4),
        "omega": uniform(loc=2 * np.pi, scale=10 * np.pi),
    }
    labels = list(prior.keys())
    priors = [prior[k] for k in labels]

    np.random.seed(0)

    # draw random parameter from prior
    y_true = [var.rvs(1)[0] for var in priors]

    # add fixed gaussian noise of 1
    x_err = 1
    x_obs = sine(y_true) + np.random.normal(size=50) * x_err

    # hyperparameters for the normalizing flow
    flow = {
        "n_dims": 3,  # dimension of parameter space
        "flow_hidden": 32,
        "num_blocks": 4,
    }

    # the NBI package provides the "ResNet-GRU" network as the default
    # featurizer network for sequential data
    featurizer = {
        "type": "resnet-gru",
        "norm": "weight_norm",
        "dim_in": 1,
        "dim_out": 32,
        "dim_conv_max": 256,
        "depth": 3,
    }

    # initialize NBI engine
    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        simulator=sine,
        priors=priors,
        labels=labels,
        device="cpu",
        n_jobs=10,
    )

    engine.fit(
        x_obs=x_obs,
        y_true=y_true,
        n_sims=1280,
        n_rounds=1,
        n_epochs=1,
        batch_size=64,
        lr=0.001,
        min_lr=0.001,
        early_stop_train=True,  # If sampling efficiency is reduced, stop and revert to previous round
        early_stop_patience=10,  # Within a round, wait this many epochs before early stopping
        noise=np.array([1] * 50),  # homogeneous noise; used for importance sampling
        workers=4,
    )

    y_pred, weights = engine.predict(
        x_obs,
        x_err=np.array([0.2] * 50),
        y_true=y_true,
        n_samples=10000,
        corner_reweight=True,
    )


if __name__ == "__main__":
    test()
