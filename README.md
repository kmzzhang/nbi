[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![PyPI](https://img.shields.io/pypi/v/nbi)

## nbi: neural bayesian inference

Do you have challanging inference problems that are difficult to solve with standard MCMC or Nested Sampling methods?
Are you looking to fit the same forward model to thousands or millions of observed datasets?
`nbi` may be your solution. 

`nbi` is an engine for Neural Posterior Estimation (NPE) focused on out-of-the-box application to common astronomical data,
such as light curves and spectra.
Compared to related packages, `nbi` requires minimal customization and can easily substitute for
It also implements a custom NPE algorithm that integrates importance sampling, which allows for
efficient, asymptotically exact results.

## Installation

To install this package, we recommend that you create a dedicated `conda` environment with Python 3.7 or higher

```bash
conda create -n nbi python=3.10 && conda activate nbi
```

Then `pip` install this package

```bash
pip install nbi
```

## Quick Start

The `examples/` directory contains complete examples that demonstrates the functionality of `nbi`. A bare-bone
example below illustrates the basic API, which follows the scikit-learn style:

```python
import nbi

# specify hyperparameters
flow = {
    "n_dims": 1,
    "flow_hidden": 32,
    "num_blocks": 4
}
featurizer = {
    "type": "resnet-gru",
    "dim_in": 1,
    "depth": 3
}
engine = nbi.NBI(
    flow,
    featurizer,
    simulator,
    noise,
    priors
)
engine.fit(
    n_sims=1000,
    n_rounds=1,
    n_epochs=100
)
y_pred, weights = engine.predict(x_obs, x_err, n_samples=2000)
```

## References

nbi: the Astronomer's Package for Neural Posterior Estimation (Zhang et al. 2023)
 - Accepted to the "Machine Learning for Astrophysics" workshop at the 
International Conference for Machine Learning (ICML). Link to the paper will be updated here.

Masked Autoregressive Flow for Density Estimation (Papamakarios et al. 2017)\
https://arxiv.org/abs/1705.07057

Featurizers: ResNet (He et al. 2015; https://arxiv.org/abs/1512.03385), Gated Recurrent Units
(GRU; Cho et al. 2014; https://arxiv.org/abs/1406.1078), 
ResNet-GRU (Zhang et al. 2021; https://iopscience.iop.org/article/10.3847/1538-3881/abf42e)



## Acknowledgments
The `nbi` package is expanded from code originally written for *''Real-time Likelihood-free Inference of Roman Binary Microlensing Events
with Amortized Neural Posterior Estimation'''* ([Zhang et al. 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abf42e)).
The Masked Autoregressive Flow in this package is partly adapted from the implementation in
https://github.com/kamenbliznashki/normalizing_flows.
Work on this project was supported by the [National Science Foundation award #2206744](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2206744&HistoricalAwards=false) ("CDS&E: Accelerating Astrophysical Insight at Scale with Likelihood-Free Inference").

<center><img src="https://www.nsf.gov/policies/images/NSF_Official_logo.svg" width="10%"></center>
