"""
Multi-fidelity fusion (co-kriging), the recursive Le Gratiet form of Kennedy & O'Hagan:

    f_high(x) = rho * f_low(x) + delta(x)

A cheap source (simulation) supplies the SHAPE of the response surface everywhere; a
scarce, trusted source (experiment) supplies the correction. This is the technique that
makes a useful model out of a few hundred literature rows, because those rows only have
to learn the offset - not the whole physics.

Fitted in two stages rather than jointly. The joint likelihood is badly conditioned when
the high-fidelity set is tiny, which is exactly the regime this exists for; the recursive
form is stable there and its rho stays interpretable (rho ~ 1 means the cheap source has
the right shape and only needs shifting; rho far from 1 means it is systematically
mis-scaled).

Today "low" is a coarse mesh and "high" is a fine one, which lets the machinery be
validated against known ground truth. Swap in curated literature rows as `high` and the
same code is the real thing.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from . import config as C


def _make_gp(seed=C.SEED, length_scale=1.0, noise=1e-3):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise, (1e-8, 1e1)))
    # 10 restarts, not 3. The marginal likelihood here has a degenerate optimum in which
    # the WhiteKernel absorbs everything and the GP returns a near-constant mean; with 3
    # restarts it was found on roughly half of the fits, so a model given MORE expensive
    # data appeared to get worse (R2 0.77 at 16 points, 0.26 at 32 - an artefact of the
    # optimiser, not of the data). 25 restarts reproduces 10 exactly, so this is settled
    # rather than merely raised.
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=10, random_state=seed)


class MultiFidelityGP:
    """
    Co-kriging surrogate. `fit` takes both tiers; `predict` returns mean and (optionally)
    standard deviation propagated through both stages.
    """

    def __init__(self, seed=C.SEED):
        self.seed = seed
        self.rho_ = 1.0

    def fit(self, X_low, y_low, X_high, y_high):
        X_low = np.asarray(X_low, float)
        X_high = np.asarray(X_high, float)
        y_low = np.asarray(y_low, float)
        y_high = np.asarray(y_high, float)

        self.scaler_ = StandardScaler().fit(np.vstack([X_low, X_high]))
        self.gp_low_ = _make_gp(self.seed).fit(self.scaler_.transform(X_low), y_low)

        mu_low_at_high = self.gp_low_.predict(self.scaler_.transform(X_high))

        # rho by least squares through the origin. An intercept is deliberately omitted:
        # a constant offset is exactly what delta(x) is there to absorb, and fitting it
        # twice makes the two stages fight.
        denom = float(mu_low_at_high @ mu_low_at_high)
        self.rho_ = float(mu_low_at_high @ y_high / denom) if denom > 1e-12 else 1.0

        resid = y_high - self.rho_ * mu_low_at_high
        self.gp_delta_ = _make_gp(self.seed).fit(self.scaler_.transform(X_high), resid)
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_.transform(np.asarray(X, float))
        if return_std:
            mu_l, sd_l = self.gp_low_.predict(Xs, return_std=True)
            mu_d, sd_d = self.gp_delta_.predict(Xs, return_std=True)
            var = (self.rho_ ** 2) * sd_l ** 2 + sd_d ** 2
            return self.rho_ * mu_l + mu_d, np.sqrt(var)
        return self.rho_ * self.gp_low_.predict(Xs) + self.gp_delta_.predict(Xs)


class SingleFidelityGP:
    """Plain GP, used as the two controls the fused model has to beat."""

    def __init__(self, seed=C.SEED):
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.scaler_ = StandardScaler().fit(X)
        self.gp_ = _make_gp(self.seed).fit(self.scaler_.transform(X), np.asarray(y, float))
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_.transform(np.asarray(X, float))
        return self.gp_.predict(Xs, return_std=return_std)


METRIC_KEYS = ("r2", "rmse", "mae", "spearman", "r2_linear", "median_fold_error")


def fidelity_experiment(X_low, y_low, X_high, y_high, X_test, y_test,
                        n_high_grid=(4, 8, 16, 24), n_repeats=5, seed=C.SEED):
    """
    How many expensive points do you actually need?

    Sweeps the high-fidelity budget and, at each budget, compares three models on the
    same held-out high-fidelity test set:

      low_only   - trust the cheap source directly (no correction at all)
      high_only  - throw the cheap source away and fit the few expensive points
      fused      - co-kriging

    This is the plot that justifies the curation effort: it converts "collect more data"
    into "collect N rows and here is the accuracy you get". Reported as median fold-error
    as well as R^2, because a factor is what a designer can act on.

    Two things make that claim survive contact with a reader:

    NESTED subsets. Each repeat draws ONE permutation and gives budget n its first n
    points, so a larger budget always contains the smaller one. Drawing each budget
    independently makes the curve a record of which points each draw happened to get -
    16 trusted points can then score far below 8, which says nothing about budget.

    REPEATS. One draw of 4 points from a 36-point pool is a coin toss, so every budget is
    averaged over `n_repeats` permutations and the spread is returned as `r2_std`. A
    curation programme is going to be costed off this plot; the honest version shows how
    much of it is still luck.
    """
    from .validation import metrics
    X_high, y_high = np.asarray(X_high, float), np.asarray(y_high, float)
    budgets = [n for n in n_high_grid if 4 <= n <= len(y_high)]

    # The cheap source ignores the high-fidelity budget entirely: fitted once, drawn as a
    # flat reference line. Refitting it per budget produced identical numbers.
    m_low = metrics(y_test, SingleFidelityGP(seed).fit(X_low, y_low).predict(X_test))

    draws = {}
    for rep in range(n_repeats):
        order = np.random.default_rng(seed + rep).permutation(len(y_high))
        for n_high in budgets:
            Xh, yh = X_high[order[:n_high]], y_high[order[:n_high]]

            high_only = SingleFidelityGP(seed).fit(Xh, yh)
            draws.setdefault(("high_only", n_high), []).append(
                metrics(y_test, high_only.predict(X_test)))

            fused = MultiFidelityGP(seed).fit(X_low, y_low, Xh, yh)
            rec = metrics(y_test, fused.predict(X_test))
            rec["rho"] = fused.rho_
            draws.setdefault(("fused", n_high), []).append(rec)

    rows = []
    for n_high in budgets:
        rows.append(dict(model="low_only", n_high=n_high, n_repeats=n_repeats,
                         r2_std=0.0, **{k: m_low[k] for k in METRIC_KEYS}))
        for name in ("high_only", "fused"):
            ds = draws[(name, n_high)]
            rec = dict(model=name, n_high=n_high, n_repeats=len(ds),
                       r2_std=float(np.std([d["r2"] for d in ds])),
                       **{k: float(np.mean([d[k] for d in ds])) for k in METRIC_KEYS})
            if name == "fused":
                rec["rho"] = float(np.mean([d["rho"] for d in ds]))
            rows.append(rec)
    return pd.DataFrame(rows)
