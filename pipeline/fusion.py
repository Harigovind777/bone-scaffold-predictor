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


# --------------------------------------------------------------------------
# How tightly rho is held to 1, and why it is not a tuning knob.
#
# rho rescales the CHEAP source to the trusted one. Here both tiers are the same
# physical quantity - relative modulus - computed on a coarse and a fine mesh, so a
# refinement can shift the answer but cannot rescale what it means: rho is 1 up to
# discretisation error. The convergence table measures that error directly (median
# 5.98% at grid 20 against grid 44), and rho multiplies log E, which spans roughly
# -6 to -1. A multiplicative slip of delta therefore moves a prediction by about
# delta * |log E| ~ 3*delta, so matching a 6% mesh error needs delta ~ 0.02.
#
# Hence N(1, 0.02^2) - read off the mesh study, not fitted to the fusion result. It
# is what stops least squares through the origin from inventing a rho of 1.13 from
# four points, which is what made the fused model worse than ignoring the expensive
# data entirely.
# --------------------------------------------------------------------------
RHO_PRIOR_MEAN = 1.0
RHO_PRIOR_SD = 0.02

# How much of each fitted correction to believe - 0 keeps the physics prior (rho = 1, no
# discrepancy), 1 is textbook co-kriging. Same coarse grid, same one-standard-error rule
# and same reasoning as `models.SHRINKAGE_GRID`: the question is "all of it, none of it,
# or somewhere between", and a finer grid only gives the rule more noise to chew on.
TRUST_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def _make_gp(seed=C.SEED, length_scale=1.0, noise=1e-3, normalize_y=True,
             amplitude_bounds=(1e-3, 1e3)):
    kernel = (ConstantKernel(1.0, amplitude_bounds)
              * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise, (1e-8, 1e1)))
    # 10 restarts, not 3. The marginal likelihood here has a degenerate optimum in which
    # the WhiteKernel absorbs everything and the GP returns a near-constant mean; with 3
    # restarts it was found on roughly half of the fits, so a model given MORE expensive
    # data appeared to get worse (R2 0.77 at 16 points, 0.26 at 32 - an artefact of the
    # optimiser, not of the data). 25 restarts reproduces 10 exactly, so this is settled
    # rather than merely raised.
    return GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y,
                                    n_restarts_optimizer=10, random_state=seed)


class MultiFidelityGP:
    """
    Co-kriging surrogate. `fit` takes both tiers; `predict` returns mean and (optionally)
    standard deviation propagated through both stages.
    """

    def __init__(self, seed=C.SEED, rho_prior_sd=RHO_PRIOR_SD, adaptive_trust=True):
        self.seed = seed
        self.rho_prior_sd = rho_prior_sd
        self.adaptive_trust = adaptive_trust
        self.rho_ = RHO_PRIOR_MEAN
        self.rho_trust_ = 1.0
        self.delta_trust_ = 1.0

    def _fit_rho(self, mu_low_at_high, y_high):
        noise_var = max(float(np.var(y_high - mu_low_at_high)), 1e-9)
        prior_prec = 1.0 / (self.rho_prior_sd ** 2)
        precision = float(mu_low_at_high @ mu_low_at_high) / noise_var + prior_prec
        return float((float(mu_low_at_high @ y_high) / noise_var
                      + prior_prec * RHO_PRIOR_MEAN) / precision)

    def _fit_delta(self, Xs_high, resid):
        return _make_gp(self.seed, normalize_y=False,
                        amplitude_bounds=(1e-6, 1.0)).fit(Xs_high, resid)

    def _estimate_trust(self, Xs_high, mu_low_at_high, y_high):
        """
        How much of (rho - 1) and of delta survives leave-one-out on the trusted points.

        WHY THIS EXISTS. Giving rho a prior stopped it exploding and bounding delta's
        amplitude stopped that exploding; neither made either correction EARN its place.
        Measured against the cheap source alone on held-out fine-mesh cases, the
        discrepancy term cost -0.104 R2 at four trusted points and was still negative at
        sixty-four, and rho drifting to 1.018 cost another -0.009. A coarse solve already
        scoring 0.818 leaves a correction very little to learn from a handful of points
        and plenty of room to invent.

        So both are multiplied by a factor MEASURED rather than chosen, on the same LOO
        folds and by the same one-standard-error rule the rest of the project uses: leave
        out each trusted point in turn, refit, and take the least total trust whose error
        is within one standard error of the best. When a correction transfers its factor
        goes to 1 and this is textbook co-kriging; when it does not, the factor goes to 0
        and the model falls back on the cheap source rather than below it.

        Predictions are linear in both factors, so one refit per fold prices the whole
        7 x 7 grid. `gp_low_` is deliberately NOT refitted inside the loop: it is a
        function of the cheap tier alone and has never seen a trusted point.

        MEASURED: both factors come back 0 at every budget from 4 to 64 trusted points -
        the model reporting, correctly, that on this mesh pair the expensive tier has
        nothing to add that the coarse one did not already have.
        """
        n = len(y_high)
        if n < 4:
            return 0.0, 0.0                 # no evidence, so no trust
        per_fold = {}
        for i in range(n):
            tr = np.ones(n, bool)
            tr[i] = False
            rho_i = self._fit_rho(mu_low_at_high[tr], y_high[tr])
            gp_i = self._fit_delta(Xs_high[tr], y_high[tr] - rho_i * mu_low_at_high[tr])
            mu_i = float(mu_low_at_high[i])
            d_i = float(gp_i.predict(Xs_high[[i]])[0])
            for t in TRUST_GRID:
                base = (1.0 + t * (rho_i - RHO_PRIOR_MEAN)) * mu_i
                for u in TRUST_GRID:
                    per_fold.setdefault((t, u), []).append(
                        float((y_high[i] - (base + u * d_i)) ** 2))
        mean = {k: float(np.mean(v)) for k, v in per_fold.items()}
        best = min(mean, key=mean.get)
        se = float(np.std(per_fold[best], ddof=1)) / np.sqrt(n)
        within = [k for k in mean if mean[k] <= mean[best] + se]
        # Least total trust among the candidates the evidence cannot separate: when the
        # data do not argue, the answer that wins is the one that changes nothing.
        return min(within, key=lambda k: (k[0] + k[1], k))

    def fit(self, X_low, y_low, X_high, y_high):
        X_low = np.asarray(X_low, float)
        X_high = np.asarray(X_high, float)
        y_low = np.asarray(y_low, float)
        y_high = np.asarray(y_high, float)

        self.scaler_ = StandardScaler().fit(np.vstack([X_low, X_high]))
        self.gp_low_ = _make_gp(self.seed).fit(self.scaler_.transform(X_low), y_low)

        mu_low_at_high = self.gp_low_.predict(self.scaler_.transform(X_high))

        # rho as a POSTERIOR under N(RHO_PRIOR_MEAN, rho_prior_sd^2), not least squares
        # through the origin. An intercept is still deliberately omitted - a constant
        # offset is what delta(x) exists to absorb, and fitting it twice makes the two
        # stages fight - but the unregularised slope was the thing that broke this stage.
        #
        # With four high-fidelity points, least squares through the origin fitted rho to
        # whatever those four happened to say; a 3% slip on a target of magnitude ~3
        # is a 0.1 shift on EVERY prediction, which is why the fused model scored 0.27
        # where simply believing the cheap source scored 0.82. Shrinking rho to the value
        # physics already implies costs nothing when the data agree and saves the model
        # when there are too few points to argue.
        self.rho_ = self._fit_rho(mu_low_at_high, y_high)

        # The discrepancy reverts to ZERO away from its data, not to the mean of a
        # handful of residuals. normalize_y=True centred delta on that mean and added it
        # back everywhere, so four noisy residuals became a constant bias applied to the
        # whole design space. delta is a CORRECTION: with no evidence it must vanish and
        # leave rho * f_low standing, which is the graceful-degradation property the
        # whole argument for fusion rests on. The amplitude ceiling is the same statement
        # in the kernel - a correction cannot be larger than the signal it corrects.
        Xs_high = self.scaler_.transform(X_high)
        self.gp_delta_ = self._fit_delta(Xs_high, y_high - self.rho_ * mu_low_at_high)
        if self.adaptive_trust:
            self.rho_trust_, self.delta_trust_ = self._estimate_trust(
                Xs_high, mu_low_at_high, y_high)
            # rho is reported as the value actually used, so `rho_` stays the number a
            # reader can check against the physics rather than an intermediate.
            self.rho_ = RHO_PRIOR_MEAN + self.rho_trust_ * (self.rho_ - RHO_PRIOR_MEAN)
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_.transform(np.asarray(X, float))
        if return_std:
            mu_l, sd_l = self.gp_low_.predict(Xs, return_std=True)
            mu_d, sd_d = self.gp_delta_.predict(Xs, return_std=True)
            # The trust factor scales delta's uncertainty with delta itself: a correction
            # the data said to ignore must not go on widening the interval.
            var = (self.rho_ ** 2) * sd_l ** 2 + (self.delta_trust_ * sd_d) ** 2
            return self.rho_ * mu_l + self.delta_trust_ * mu_d, np.sqrt(var)
        return (self.rho_ * self.gp_low_.predict(Xs)
                + self.delta_trust_ * self.gp_delta_.predict(Xs))


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
            rec["delta_trust"] = fused.delta_trust_
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
                rec["delta_trust"] = float(np.mean([d["delta_trust"] for d in ds]))
            rows.append(rec)
    return pd.DataFrame(rows)
