"""
Tier 1: predicting printability from an ink's composition and its printing parameters.

This is the only stage that runs on real publications rather than on simulation, so it
is where the leakage claim gets tested against the thing it is a claim about. It is also
where the project's headline accuracy number was being measured with the wrong ruler.

TWO THINGS WERE WRONG.

1. Printability is ORDINAL - 0, 1, 2, 3 - and was being predicted with a multiclass
   classifier that treats those four levels as four unrelated names. Calling a 3 a 0 was
   scored exactly as badly as calling it a 2, so the model had no reason to prefer being
   close, and its ranking was nearly worthless (Spearman 0.28) even while its accuracy
   looked respectable.

2. Accuracy is a degenerate metric on this target. 55.5% of rows are level 3, so a model
   that answers "3" to everything scores 0.555 and the honest grouped number to beat was
   only +0.029 above that. A model can buy accuracy simply by collapsing toward the
   majority level, which is precisely what the classifier was doing, and no amount of
   accuracy chasing distinguishes that from having learned something.

THE FIX, and it is one change rather than two: keep the classifier - the levels really
are discrete and it models them well - but read out its PROBABILITY-WEIGHTED EXPECTED
LEVEL, sum(k * p_k), instead of its argmax. That single number is ordinal by
construction, so a model that is unsure between 2 and 3 says 2.4 instead of gambling.
Scored against grouped-by-DOI cross-validation it is strictly better on every measure at
once, accuracy included:

    metric                        classifier argmax   expected level
    accuracy                            0.587             0.595
    quadratic-weighted kappa            0.144             0.379
    Spearman rho                        0.278             0.543
    mean absolute error (levels)        0.645             0.608

Quadratic-weighted kappa is the metric reported alongside accuracy from here on, because
it is the one that cannot be won by answering "3" to everything: guessing the majority
level scores exactly 0.

The features are enriched at the same time. Raw %w/v columns describe an ink by how much
of each of fifty possible ingredients it contains, which makes two inks of identical
chemistry at different dilutions look unrelated. Composition is therefore also expressed
as FRACTIONS of total solids, plus total solids itself, the component count, and two
extrusion ratios (pressure and speed per unit nozzle diameter) that carry the shear the
ink actually sees.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, KFold
from scipy.stats import spearmanr

from . import config as C

TARGET = "Printability"
# Every target column, not just the one being predicted. "Scaffold Quality (P*C)" is
# Printability x Cell Response, so leaving either of those in hands the model the answer.
LEAKY = ["Printability", "Cell Response", "Scaffold Quality (P*C)"]


def build_features(m, enrich=True):
    """Numeric design matrix. `enrich=False` reproduces the original feature block."""
    f = m.select_dtypes(include=[np.number]).drop(columns=LEAKY, errors="ignore")
    f = f.loc[:, f.notna().mean() > 0.5]
    if not enrich:
        return f.fillna(f.median())

    comp = [c for c in f.columns if "%w/v" in c or "%wt" in c]
    total = f[comp].fillna(0).sum(axis=1)
    X = pd.DataFrame(index=f.index)

    # Composition as fractions of total solids: the same chemistry at two dilutions
    # should look like the same ink, which in raw %w/v it does not.
    denom = total.replace(0, np.nan)
    for c in comp:
        X["frac_" + c] = (f[c].fillna(0) / denom).fillna(0.0)
    X["total_solids"] = total
    X["log_total_solids"] = np.log1p(total)
    X["n_components"] = (f[comp].fillna(0) > 0).sum(axis=1)
    X["max_fraction"] = X[[c for c in X.columns if c.startswith("frac_")]].max(axis=1)

    # What the ink actually experiences on the way out: shear scales with pressure and
    # with speed, both per unit nozzle diameter.
    for a, b, name in (("Extrusion_Pressure (kPa)", "Nozzle_Diameter_(µm)", "pressure_per_dia"),
                       ("Nozzle_Movement_Speed_(mm/s)", "Nozzle_Diameter_(µm)", "speed_per_dia")):
        if a in f.columns and b in f.columns:
            X[name] = f[a] / f[b].replace(0, np.nan)

    for c in [c for c in f.columns if c not in comp]:
        X[c] = f[c]
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median()).fillna(0.0)


def expected_level(X_train, y_train, X_test, seed=C.SEED):
    """
    sum(k * p_k) from a random forest classifier - a continuous, ordinal readout.

    The argmax of the same probabilities is what the previous version reported, and it
    throws away exactly the information that makes the answer ordinal: a row the forest
    splits 0.5/0.5 between levels 2 and 3 is a 2.5, not a coin flip between two labels
    that the metric will treat as equally distant from every other label.
    """
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                 random_state=seed).fit(X_train, y_train)
    return clf.predict_proba(X_test) @ clf.classes_.astype(float)


def argmax_level(X_train, y_train, X_test, seed=C.SEED):
    """The original readout, kept so the improvement is measured rather than asserted."""
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                  random_state=seed).fit(X_train, y_train).predict(X_test)


def _score(y, scores, levels):
    pred = np.clip(np.round(scores), levels.min(), levels.max()).astype(int)
    return dict(
        accuracy=float((pred == y).mean()),
        qwk=float(cohen_kappa_score(y, pred, weights="quadratic")),
        mae_levels=float(np.abs(scores - y).mean()),
        spearman=float(spearmanr(y, scores).statistic),
    )


def evaluate(X, y, groups, readout=expected_level, grouped=True, n_splits=5,
             n_repeats=4, seed=C.SEED):
    """
    Out-of-fold scores under repeated shuffled grouped (or random) k-fold.

    sklearn's GroupKFold is deterministic on group order, so a single run reports one
    arbitrary partition of the 88 DOIs. Repeating over shuffled group orders and
    returning the spread is what makes a 0.008 accuracy difference readable as signal or
    as noise.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    levels = np.unique(y)
    uniq = pd.unique(pd.Series(groups))
    runs = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        order = {g: i for i, g in enumerate(rng.permutation(uniq))}
        shuffled = np.array([order[g] for g in groups])
        oof = np.full(len(y), np.nan)
        if grouped:
            splits = GroupKFold(n_splits).split(X, y, groups=shuffled)
        else:
            splits = KFold(n_splits, shuffle=True, random_state=seed + rep).split(X)
        for train, test in splits:
            oof[test] = readout(X[train], y[train], X[test], seed)
        runs.append(_score(y, oof, levels))
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out["accuracy_std"] = float(np.std([r["accuracy"] for r in runs]))
    out["n_repeats"] = n_repeats
    return out


def majority_baseline(y):
    """What answering the commonest level to everything scores. qwk is 0 by definition."""
    y = np.asarray(y, int)
    mode = pd.Series(y).value_counts().idxmax()
    return dict(accuracy=float((y == mode).mean()), qwk=0.0,
                mae_levels=float(np.abs(y - mode).mean()), spearman=0.0)
