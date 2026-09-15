"""
The model ladder, from textbook physics to the physics-informed hybrid.

Every model here predicts a target in the space config.TARGETS declares (log, for
modulus). They share the sklearn fit/predict interface so validation.cross_validate can
swap them without special-casing.

The ladder exists to answer one question honestly: how much does machine learning
actually add over the closed-form theory a supervisor could write on a whiteboard?

  1. GibsonAshbyGlobal          one (n, C), pooled over rows      - the textbook answer
  2. GibsonAshbyBalanced        one (n, C), one vote per architecture
  3. GibsonAshbyPerMode         one (n, C) per deformation mode   - the best physics can do
  4. GibsonAshbyPerArchitecture one (n, C) per architecture       - the control that fails
  5. PlainML                    gradient boosting, no physics     - the usual ML paper
  6. PhysicsInformedResidual    best prior + SHRUNK residual      - the proposal

Read 3 against 4, because the difference between them is the whole argument. Both split
the power law by a label; only one of those labels survives being handed a topology the
sweep never saw. Under leave-one-architecture-out model 4 has no row in its table for the
held-out topology and falls back to the global fit, scoring exactly what the global fit
scores - a per-architecture lookup table is not a design tool. Model 3 splits on the
DEFORMATION MODE, which is an input the designer chooses rather than an identity the
model has to recognise, so a new topology still arrives with one attached and the fit for
its mode transfers. That is the difference between describing the sweep and predicting
outside it.

Model 2 takes the sweep's sampling schedule out of the estimator without adding a
parameter. Model 6 adds the only learning that survived: a residual on top of the best
physics available, multiplied by a trust factor estimated inside each fold - which under
leave-one-architecture-out comes back as zero, so model 6 reduces to model 3 rather than
scoring below it.

What did NOT work is recorded with numbers in `tools/accuracy_ledger.py --rejected`;
chief among them, predicting (n, C) for an unseen topology from its design-time
morphometry, which scored 0.691 under nested selection, i.e. worse than not trying.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from . import config as C

RHO_COL = "log_relative_density"

# Trust levels the residual stage is allowed to take. Coarse on purpose: the decision
# this grid has to make is "all of it, none of it, or somewhere between", and a finer
# grid only gives the one-SE rule more noise to chew on.
SHRINKAGE_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


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


def architecture_weights(groups, index=None):
    """
    Row weights that give every architecture the same total say, normalised to mean 1.

    A pooled least-squares fit weights each architecture by how many times it was
    sampled, which is a fact about the sweep schedule and not about cellular solids.
    Here that is a 4x distortion: the two FDM architectures contribute 471 of 1806 rows
    and one of them (staggered) has an exponent of 3.9 against a global 2.24, so a
    quarter of the fit is pulled by the single most atypical geometry in the study.

    Equal weight per architecture is the same choice the validation protocol already
    makes - one architecture is one unit of evidence, exactly as one publication is -
    so this makes the estimator agree with the way it is scored.
    """
    g = pd.Series(groups).reset_index(drop=True)
    if index is not None:
        g = g.iloc[np.asarray(index)] if len(g) > len(index) else g
    counts = g.value_counts()
    w = (1.0 / g.map(counts)).values.astype(float)
    return w / w.mean()


class GibsonAshbyBalanced(GibsonAshbyGlobal):
    """
    The same two-parameter power law, fitted with one vote per architecture.

    This is not a different model - it is the same physics with the sampling bias taken
    out of the estimator. Under leave-one-architecture-out it scores R2 = 0.790 against
    0.783 for the pooled fit, and the gain is concentrated exactly where it should be:
    on the architectures whose exponent sits far from the global one.
    """

    def __init__(self, groups=None):
        super().__init__()
        self.groups = None if groups is None else pd.Series(groups).reset_index(drop=True)

    def fit(self, X, y):
        lr = np.asarray(X[RHO_COL], float)
        y = np.asarray(y, float)
        ok = np.isfinite(lr) & np.isfinite(y)
        if ok.sum() < 3:
            return self
        if self.groups is None:
            self.n_, self.logC_ = np.polyfit(lr[ok], y[ok], 1)
            return self
        idx = X.index if hasattr(X, "index") else np.arange(len(y))
        w = architecture_weights(self.groups, idx)
        # polyfit takes sqrt-weights: it minimises sum((w_i * (y_i - f(x_i)))^2).
        self.n_, self.logC_ = np.polyfit(lr[ok], y[ok], 1, w=np.sqrt(w[ok]))
        return self


# A mode needs this many rows and this many DISTINCT architectures in training before
# it gets its own power law. Two architectures is the minimum at which "this mode" is a
# claim about the mode rather than about the one topology that happens to carry it.
MIN_ROWS_PER_MODE = 10
MIN_ARCH_PER_MODE = 2


class GibsonAshbyPerMode(GibsonAshbyBalanced):
    """
    One (n, C) per DEFORMATION MODE, falling back to the balanced global fit.

    WHY THIS IS NOT THE PER-ARCHITECTURE TABLE WEARING A HAT. `gibson_ashby_per_arch`
    is useless on a held-out topology because a topology label it has never seen has no
    row in the table. `mode` - sheet, network, aligned, staggered - is different in the
    one way that matters: it is an INPUT the designer chooses, and a brand-new topology
    still arrives with a mode attached. Leaving `TPMS|gyroid|sheet` out still leaves
    seven other sheet architectures in training, so the fit for "sheet" transfers.

    WHY IT SHOULD WORK. Gibson-Ashby is not one power law but
    two: n ~ 2 when cell walls bend and n ~ 1 when they stretch. Sheet TPMS are closed
    membranes carrying load in tension; network TPMS are strut assemblies that bend.
    Pooling them estimates one exponent for two mechanisms. Measured per architecture,
    sheet exponents average 1.98 (sd 0.34) and network 2.61 (sd 0.78) - the predicted
    ordering, and the sheet family is much the tighter of the two.

    MEASURED, leave-one-architecture-out on 1806 rows: R2 = 0.7996 against 0.7904 for
    the balanced pooled fit and 0.7828 for the textbook one. Still two parameters per
    mode and no continuous knob. The grouping was compared on the score against one real
    alternative - per family, 0.7910 - and the physics above predicted the winner, so this
    is a choice between candidates rather than a blind claim; it is recorded as such.
    A shrunk version with the pooling strength lambda chosen inside each
    fold was built and measured; it is in the ledger's rejected list, because lambda
    came back at either 0 or 0.75 depending on which standard-error rule was used and
    neither beat simply fitting the modes.

    FDM's two modes hold one architecture each, so under leave-one-architecture-out they
    fall back to the balanced fit - correctly, since with the only aligned architecture
    held out there is no evidence about aligned left to use.
    """

    def __init__(self, groups=None, modes=None):
        super().__init__(groups)
        self.modes = None if modes is None else pd.Series(modes).reset_index(drop=True)
        self.by_mode_ = {}

    def _align(self, series, index):
        return (series.iloc[np.asarray(index)].reset_index(drop=True)
                if len(series) > len(index) else series.reset_index(drop=True))

    def fit(self, X, y):
        super().fit(X, y)                      # the fallback, and the modes' own prior
        self.by_mode_ = {}
        if self.modes is None or self.groups is None:
            return self
        index = X.index.values if hasattr(X, "index") else np.arange(len(y))
        md = self._align(self.modes, index)
        ar = self._align(self.groups, index)
        lr = np.asarray(X[RHO_COL], float)
        y = np.asarray(y, float)
        ok = np.isfinite(lr) & np.isfinite(y)
        for label in md.unique():
            m = (md == label).values & ok
            if m.sum() < MIN_ROWS_PER_MODE or ar[m].nunique() < MIN_ARCH_PER_MODE:
                continue
            w = architecture_weights(ar[m])
            self.by_mode_[label] = np.polyfit(lr[m], y[m], 1, w=np.sqrt(w))
        return self

    def predict(self, X):
        out = super().predict(X)
        if not self.by_mode_:
            return out
        index = X.index.values if hasattr(X, "index") else np.arange(len(X))
        md = self._align(self.modes, index).values
        lr = np.asarray(X[RHO_COL], float)
        for label, (n, logC) in self.by_mode_.items():
            m = md == label
            if m.any():
                out[m] = logC + n * lr[m]
        return out


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
    r"""
    Gibson-Ashby as a prior mean; ML learns only what the power law gets wrong, and is
    then scaled by how much that correction is worth on geometry it has never seen.

        log E_pred = [logC + n*log(rho)]  +  s * f(geometry)
                      \______ physics ______/    \___ learned ___/

    Why a prior mean at all: the density scaling - which dominates the variance and is
    genuinely universal - is handled by a two-parameter physical law that extrapolates
    sensibly. The learner spends its limited capacity on the architecture-dependent
    departure, which is the part it can actually learn from a few hundred rows.

    WHY `s` EXISTS. Without it this model scored R2 = 0.647 under leave-one-architecture-
    out against 0.783 for the bare power law it is built on: the residual it learns is
    architecture-specific, so on a topology held out entirely the correction is not
    merely uninformative but actively wrong, and it is added to a prior that was already
    right. The earlier version of this docstring named the fix and did not implement it.

    `s` is that fix, and it is ESTIMATED, not chosen. Inside `fit`, the training set is
    split again by the same grouping the outer protocol uses, the residual stage is
    refitted on each inner split, and s is taken as the smallest value on
    SHRINKAGE_GRID whose inner-CV error is within one standard error of the best - see
    `_estimate_shrinkage` for why the one-SE rule rather than the outright minimum.

    So the trust placed in the correction is measured under exactly the extrapolation
    the outer protocol will impose. When the residual transfers, s goes to 1 and this is
    the old model; when it does not, s goes to 0 and this degrades gracefully onto the
    physics prior instead of below it. Handed a random inner split it recovers s ~ 1,
    because under interpolation the residual really is worth trusting - the shrinkage
    adapts to the question being asked rather than to a constant.

    MEASURED, leave-one-architecture-out on 1806 simulated rows: s = 0 on every fold and
    R2 = 0.790, against 0.647 for the unshrunk version and 0.783 for the pooled power
    law. Scoring identically to `gibson_ashby_balanced` is not a coincidence to explain
    away - at s = 0 it IS that model, which is the whole point.

    Both stages, and s itself, are fitted strictly inside each outer CV fold.
    """

    def __init__(self, kind="hgb", seed=C.SEED, groups=None, modes=None, shrink=True):
        self.kind, self.seed = kind, seed
        self.groups = None if groups is None else pd.Series(groups).reset_index(drop=True)
        self.modes = None if modes is None else pd.Series(modes).reset_index(drop=True)
        self.shrink = shrink

    # -- the best physics available: per-mode if modes are known, then balanced, then
    #    pooled. The residual stage corrects whatever prior it is given, so handing it
    #    the stronger one moves this rung up with it.
    def _make_prior(self):
        if self.groups is None:
            return GibsonAshbyGlobal()
        # The label series are handed over WHOLE and indexed into by the row labels of
        # whatever frame the prior is given, exactly as `architecture_weights` does.
        # Slicing them to the inner-training rows instead would leave the prior unable
        # to look up a mode for the inner-TEST rows it is then asked to predict.
        if self.modes is None:
            return GibsonAshbyBalanced(self.groups)
        return GibsonAshbyPerMode(self.groups, self.modes)

    def _inner_splits(self, n, index):
        """Leave-one-group-out over the TRAINING groups; random 5-fold if ungrouped."""
        if self.groups is None:
            from sklearn.model_selection import KFold
            for tr, te in KFold(5, shuffle=True, random_state=self.seed).split(np.zeros(n)):
                yield tr, te
            return
        g = self.groups.iloc[np.asarray(index)].reset_index(drop=True)
        idx = np.arange(n)
        for label in g.unique():
            te = idx[(g == label).values]
            tr = idx[(g != label).values]
            if len(te) >= 3 and len(tr) >= 10:
                yield tr, te

    def _estimate_shrinkage(self, X, y, index):
        """
        Smallest s whose inner-CV error is within one standard error of the best.

        The one-standard-error rule rather than the outright minimum, for the usual
        reason and one specific to this model. The usual reason: with 17 inner folds the
        error curve is noisy, and picking its argmin buys noise. The specific one: the
        two hypotheses in play are "the correction transfers" (s ~ 1) and "it does not"
        (s = 0), and when the evidence does not separate them, the answer that must win
        is the one that changes nothing. Taking the argmin instead scores 0.770 here;
        the one-SE rule scores 0.790 and picks s = 0 on every fold, which is the model
        reporting - correctly - that it has nothing to add to the power law on a
        topology it has never seen.

        Predictions are linear in s, so one residual fit per inner fold prices the whole
        grid.
        """
        per_fold = {s: [] for s in SHRINKAGE_GRID}
        for tr, te in self._inner_splits(len(y), index):
            Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
            Xte = X.iloc[te] if hasattr(X, "iloc") else X[te]
            prior = self._make_prior().fit(Xtr, y[tr])
            m = make_regressor(self.kind, self.seed).fit(Xtr, y[tr] - prior.predict(Xtr))
            base = prior.predict(Xte)
            r_hat = m.predict(Xte)
            for s in SHRINKAGE_GRID:
                per_fold[s].append(float(np.mean((y[te] - (base + s * r_hat)) ** 2)))

        n_folds = len(per_fold[SHRINKAGE_GRID[0]])
        if n_folds < 2:
            return 1.0
        mean = {s: float(np.mean(v)) for s, v in per_fold.items()}
        best = min(SHRINKAGE_GRID, key=lambda s: mean[s])
        se = float(np.std(per_fold[best], ddof=1)) / np.sqrt(n_folds)
        return min(s for s in SHRINKAGE_GRID if mean[s] <= mean[best] + se)

    def fit(self, X, y):
        y = np.asarray(y, float)
        index = X.index.values if hasattr(X, "index") else np.arange(len(y))
        self.prior_ = self._make_prior().fit(X, y)
        self.shrinkage_ = self._estimate_shrinkage(X, y, index) if self.shrink else 1.0
        resid = y - self.prior_.predict(X)
        self.model_ = make_regressor(self.kind, self.seed).fit(X, resid)
        self.columns_ = list(X.columns)
        return self

    def predict(self, X):
        Xc = X[self.columns_] if hasattr(X, "columns") else X
        return self.prior_.predict(X) + self.shrinkage_ * self.model_.predict(Xc)

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
        return self.shrinkage_ * preds.std(axis=0)


def model_zoo(groups=None, modes=None, kind="hgb"):
    """The ladder as {name: factory}. Factories, so each CV fold gets a fresh model."""
    zoo = {
        "gibson_ashby_global": GibsonAshbyGlobal,
        "plain_ml": lambda: PlainML(kind),
        "physics_informed": lambda: PhysicsInformedResidual(kind, groups=groups,
                                                             modes=modes),
    }
    if groups is not None:
        zoo["gibson_ashby_balanced"] = lambda: GibsonAshbyBalanced(groups)
        zoo["gibson_ashby_per_arch"] = lambda: GibsonAshbyPerArchitecture(groups)
    if groups is not None and modes is not None:
        zoo["gibson_ashby_per_mode"] = lambda: GibsonAshbyPerMode(groups, modes)
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
