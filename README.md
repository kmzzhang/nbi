[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![PyPI](https://img.shields.io/pypi/v/nbi)
[![Tests](https://github.com/kmzzhang/nbi/actions/workflows/tests.yml/badge.svg)](https://github.com/kmzzhang/nbi/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/nbi/badge/?version=latest)](https://nbi.readthedocs.io/en/latest/?badge=latest)

## nbi: neural bayesian inference

[Documentation](https://nbi.readthedocs.io/en/latest/)

Do you have challenging inference problems that are difficult to solve with standard optimization and/or MCMC methods?
Are you looking to fit the same forward model to thousands or millions of observed targets?
`nbi` may be your solution.

`nbi` is an engine for Neural Posterior Estimation (NPE) focused on out-of-the-box functionality for astronomical data,
particularly light curves and spectra.
`nbi` provides effective embedding/featurizer networks for spectra and light-curve data, along
with importance-sampling integration that enables asymptotically exact inference so that the inference results are
interpretable and trustworthy.

## Installation

You may either install `nbi` with `pip install nbi` or directly from source. As `nbi` is currently under active development,
installing from source may be preferable at this stage.

```bash
git clone https://github.com/kmzzhang/nbi.git
cd nbi
pip install .
```

If you are using Mac ARM **CPU** (i.e. M1/M2/M3), you might want to install PyTorch from source and disable NNPACK, which is known to
reduce performance (see [issue](https://github.com/pytorch/pytorch/issues/107534)). Note that currently the `MPS`
Also support for `weight_norm` on Mac M1-M3 GPUs is recently
[implemented](https://github.com/pytorch/pytorch/pull/108008) but has not been included in a stable
release yet. Installing the nightly version from source also enables `weight_norm` for
the MPS device.

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
USE_NNPACK=0 python setup.py install
```

## Quick Start

The `examples/` directory contains complete examples that demonstrates the functionality of `nbi`. A bare-bone
example below illustrates the basic API, which follows the scikit-learn style. The default featurizer network for
sequential data is `resnet-gru`, which is a hybrid CNN-RNN architecture.

Here are a rule of thumb for resnet-gru hyperparameters:

- dim_in: this is your number of input data channels
- depth: number of ResNet blocks. Start near log2(L)-5, where L is length of your sequential data.
- max_hidden: Maximum hidden dimensions for ResNet. Hidden dimensions double (from hidden_conv=32 by default) every
  depth. At least a few times D^2, where D is the dimension of the physical parameter space.

```python
import nbi

# hyperparameters
featurizer = {
    "type": "resnet-gru",
    "dim_in": 1,
    "max_hidden": 64
}

flow = {
    "n_dims": 1,        # parameter space dimension
    "flow_hidden": 32,  # generally no larger than max_hidden
    "num_blocks": 10    # depends on complexity of posterior shape
}

engine = nbi.NBI(
    flow,
    featurizer,
    simulator,
    noise,
    priors,
    device='cpu'        # 'cuda', 'cuda:0', 'mps' for M1/M2 Mac GPU
)
engine.fit(
    n_sims=1000,
    n_rounds=1,
    n_epochs=100
)
y_pred, weights = engine.predict(x_obs, x_err, n_samples=2000)
```

## References

nbi: the Astronomer's Package for Neural Posterior Estimation
([Zhang et al. 2023](https://ml4astro.github.io/icml2023/assets/71.pdf)).
Accepted to the "Machine Learning for Astrophysics" workshop at the 2023
International Conference for Machine Learning (ICML). Will be posted to arXiv soon.

Masked Autoregressive Flow for Density Estimation (Papamakarios et al. 2017)\
https://arxiv.org/abs/1705.07057

Featurizers: ResNet (He et al. 2015; https://arxiv.org/abs/1512.03385), Gated Recurrent Units
(GRU; Cho et al. 2014; https://arxiv.org/abs/1406.1078),
ResNet-GRU (Zhang et al. 2021; https://iopscience.iop.org/article/10.3847/1538-3881/abf42e)

## Acknowledgments

The `nbi` package is expanded from code originally written for _''Real-time Likelihood-free Inference of Roman Binary Microlensing Events
with Amortized Neural Posterior Estimation'''_ ([Zhang et al. 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abf42e)).
The Masked Autoregressive Flow in this package is partly adapted from the implementation in
https://github.com/kamenbliznashki/normalizing_flows.
Work on this project was supported by the [National Science Foundation award #2206744](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2206744&HistoricalAwards=false) ("CDS&E: Accelerating Astrophysical Insight at Scale with Likelihood-Free Inference").

<center><img src="https://new.nsf.gov/themes/custom/nsf_theme/components/images/logo/logo-desktop.svg" width="30%"></center>
