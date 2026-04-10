import numpy as np

class EmpiricalPrior:
    def __init__(self, lookup_table, priors=None, random_state=None):
        self.lookup_table = np.asarray(lookup_table)
        self.n_samples = self.lookup_table.shape[0]
        self.priors = priors
        self.rng = np.random.default_rng(random_state)

    def rvs(self, size=1, replace=True):
        if not replace and size > self.n_samples:
            raise ValueError(
                f"Requested {size} samples without replacement, but "
                f"lookup_table only has {self.n_samples} entries."
            )
        idx = self.rng.choice(self.n_samples, size=size, replace=replace)
        return self.lookup_table[idx]

    def logpdf(self, params):
        if self.priors is None:
            raise NotImplementedError(
                "logpdf not implemented: provide a `priors` argument (a list of "
                "scipy-style distributions or a single object with a .logpdf method) "
                "to enable log-probability evaluation, which is required for SNPE."
            )
        params = np.asarray(params)
        if isinstance(self.priors, list):
            log_prob = np.zeros(len(params))
            for i, prior in enumerate(self.priors):
                log_prob += prior.logpdf(params[:, i])
            return log_prob
        return self.priors.logpdf(params)
