nbi - Neural Bayesian Inference
====


Installation
======

To install this package, we recommend that you create a dedicated `conda` environment, using the latest `pytorch` package:

```bash
conda create -n nbi python=3.10
conda activate nbi
conda install pytorch torchvision torchaudio -c pytorch
```

Then `pip` install this package from pypi

```bash
pip install nbi
```

or directly from GitHub:

```bash
git clone https://github.com/kmzzhang/nbi
cd nbi
pip install -e .
```

The `-e` option for pip is the development mode, where changes will immediately apply.

Usage
======

See the `examples/` directory for Jupyter notebooks showing the basic functionality of the package.

Contributing
=====

nbi is released under the BSD license. We encourage you to modify it, reuse it, and contribute changes back for the benefit of others. We follow standard open source development practices: changes are submitted as pull requests and, once they pass the test suite, reviewed by the team before inclusion. 

To contribute back via pull requests, please first fork this report and make sure that you add the original repo as the `upstream` remote:

```bash
git remote add upstream https://github.com/kmzzhang/nbi.git
```
When you are ready to submit a PR push to your fork:

```bash
git push -u origin <your_local_branch_with_changes_name>
```

Then create a PR back to `upstream`:

```
https://github.com/<your_username>/nbi/pull/new/<your_local_branch_with_changes_name>
```

Acknowledgments
=====
Work on this project was supported by the [National Science Foundation award #2206744](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2206744&HistoricalAwards=false) ("CDS&E: Accelerating Astrophysical Insight at Scale with Likelihood-Free Inference").

<center><img src="https://www.nsf.gov/policies/images/NSF_Official_logo.svg" width="20%"></center>