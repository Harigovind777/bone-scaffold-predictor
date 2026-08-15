"""
The model ladder, from textbook physics to the physics-informed hybrid.

Every model here predicts a target in the space config.TARGETS declares (log, for
modulus). They share the sklearn fit/predict interface so validation.cross_validate can
swap them without special-casing.

The ladder exists to answer one question honestly: how much does machine learning
actually add over the closed-form theory a supervisor could write on a whiteboard?

  1. GibsonAshbyGlobal        one (n, C) for everything            - the textbook answer
  2. GibsonAshbyPerArchitecture  one (n, C) per architecture        - the best physics can do
  3. PlainML                  gradient boosting, no physics         - the usual ML paper
  4. PhysicsInformedResidual  GA prior + learned residual           - the proposal

Model 2 is the interesting control. Under leave-one-architecture-out it CANNOT see the
held-out architecture's parameters and must fall back to the global fit, which is
precisely why a per-architecture lookup table is not a design tool - and why the
residual model, which reads the geometry rather than a topology label, is.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from . import config as C

RHO_COL = "log_relative_density"


def make_regressor(kind="hgb", seed=C.SEED):
    """Small-data-friendly defaults; these sweeps are hundreds of rows, not millions."""
    if kind == "rf":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                     max_features=0.6, n_jobs=-1, random_state=seed)
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06,
                                         max_iter=400, min_samples_leaf=5,
                                         l2_regularization=1.0, random_state=seed)


# --------------------------------------------------------------------------
# Physics baselines
# --------------------------------------------------------------------------

class GibsonAshbyGlobal:
    """
    log E_rel = log C + n * log(rho_rel), one global (n, C).

    Fitted, not assumed: the textbook values (n=2, C=1) are a prior, and letting the
    data set them is the fairest possible version of the baseline. Beating an ASSUMED
    n=2 would be a rigged comparison.
    """

    def __init__(self):
        self.n_ = 2.0
        self.logC_ = 0.0

    def fit(self, X, y):
        lr = np.asarray(X[RHO_COL], float)
        y = np.asarray(y, float)
        ok = np.isfinite(lr) & np.isfinite(y)
        if ok.sum() >= 3:
            self.n_, self.logC_ = np.polyfit(lr[ok], y[ok], 1)
        return self

    def predict(self, X):
        return self.logC_ + self.n_ * np.asarray(X[RHO_COL], float)


class GibsonAshbyPerArchitecture:
    """
    One (n, C) per architecture, falling back to the global fit for unseen ones.

    `groups` must be supplied at construction for both fit and predict, since the
    architecture label is metadata rather than a model feature.
    """

    def __init__(self, groups):
        self.groups = pd.Series(groups).reset_index(drop=True)
        self.fits_ = {}
        self.global_ = GibsonAshbyGlobal()

    def fit(self, X, y):
        idx = X.index if hasattr(X, "index") else np.arange(len(y))
        g = self.groups.iloc[idx].values if len(self.groups) > len(idx) else self.groups.values
        lr, y = np.asarray(X[RHO_COL], float), np.asarray(y, float)
        self.global_.fit(X, y)
        for label in pd.unique(g):
            m = (g == label) & np.isfinite(lr) & np.isfinite(y)
            if m.sum() >= 4:
                n, logC = np.polyfit(lr[m], y[m], 1)
                self.fits_[label] = (n, logC)
        return self

    def predict(self, X):
        idx = X.index if hasattr(X, "index") else np.arange(len(X))
        g = self.groups.iloc[idx].values if len(self.groups) > len(idx) else self.groups.values
        lr = np.asarray(X[RHO_COL], float)
        out = self.global_.predict(X)
        for label, (n, logC) in self.fits_.items():
            m = g == label
            if m.any():
                out[m] = logC + n * lr[m]
        return out


# --------------------------------------------------------------------------
# Machine learning
# --------------------------------------------------------------------------

class PlainML:
    """Gradient boosting straight onto the target. No physics. The control."""

    def __init__(self, kind="hgb", seed=C.SEED):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        self.model_ = make_regressor(self.kind, self.seed).fit(X, np.asarray(y, float))
        self.columns_ = list(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.columns_] if hasattr(X, "columns") else X)


class PhysicsInformedResidual:
    """
    Gibson-Ashby as a prior mean; ML learns only what the power law gets wrong.

        log E_pred = [logC + n*log(rho)]  +  f(geometry)
                      \______ physics ______/   \___ learned ___/

    Why this should beat fitting the raw target: the density scaling - which dominates
    the variance and is genuinely universal - is handled by a two-parameter physical law
    that extrapolates sensibly. The learner spends its limited capacity on the
    architecture-dependent departure, which is the part it can actually learn from a few
    hundred rows.

    Both stages are fitted inside each CV fold, so the prior never sees held-out rows.

    MEASURED BEHAVIOUR, leave-one-architecture-out on the simulated tier (747 rows):
    this model scores R2 = 0.55 against 0.73 for the bare Gibson-Ashby prior it is built
    on. It does NOT degrade gracefully. The residual it learns is architecture-specific,
    so on a topology held out entirely the learned correction is not merely uninformative
    but actively wrong, and it is added to a prior that was already right. Reporting that
    is the point of the ladder. Two honest readings, to be settled with Tier 3 data:
    simulated E_rel really is a clean power law in relative density, leaving no residual
    structure that transfers between topologies; or the residual stage needs to shrink
    toward zero when the held-out geometry falls outside the training manifold.
    """

    def __init__(self, kind="hgb", seed=C.SEED):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.prior_ = GibsonAshbyGlobal().fit(X, y)
        resid = y - self.prior_.predict(X)
        self.model_ = make_regressor(self.kind, self.seed).fit(X, resid)
        self.columns_ = list(X.columns)
        return self

    def predict(self, X):
        Xc = X[self.columns_] if hasattr(X, "columns") else X
        return self.prior_.predict(X) + self.model_.predict(Xc)

    def predict_std(self, X):
        """
        Spread across trees as a cheap epistemic-uncertainty estimate.

        Only meaningful for the forest variant; boosting's stages are additive rather
        than independent, so their spread is not an uncertainty. Returns NaN there
        instead of a number that would be quietly wrong.
        """
        Xc = X[self.columns_] if hasattr(X, "columns") else X
        if not isinstance(self.model_, RandomForestRegressor):
            return np.full(len(Xc), np.nan)
        preds = np.stack([t.predict(np.asarray(Xc, float)) for t in self.model_.estimators_])
        return preds.std(axis=0)


def model_zoo(groups=None, kind="hgb"):
    """The ladder as {name: factory}. Factories, so each CV fold gets a fresh model."""
    zoo = {
        "gibson_ashby_global": GibsonAshbyGlobal,
        "plain_ml": lambda: PlainML(kind),
        "physics_informed": lambda: PhysicsInformedResidual(kind),
    }
    if groups is not None:
        zoo["gibson_ashby_per_arch"] = lambda: GibsonAshbyPerArchitecture(groups)
    return zoo


def permutation_importance(model, X, y, n_repeats=5, seed=C.SEED):
    """Degradation in R^2 when each column is shuffled. Model-agnostic, so it compares
    across the whole ladder."""
    from .validation import metrics
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    base = metrics(y, model.predict(X))["r2"]
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - metrics(y, model.predict(Xp))["r2"])
        rows.append(dict(feature=col, importance=float(np.mean(drops)),
                         std=float(np.std(drops))))
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
