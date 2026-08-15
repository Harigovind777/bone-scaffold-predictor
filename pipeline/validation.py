"""
Splitters and metrics. This module is where the project's central methodological claim
lives, so it is deliberately blunt about it.

The claim: on small, publication-derived scaffold datasets, a random train/test split
does not measure generalisation. Rows from one paper share a material batch, a printer
and an operator, so random splitting puts near-duplicates on both sides. The honest
question is not "can it predict a held-out row" but "can it predict a paper, or an
architecture, it has never seen" - which is the only question a design tool must answer.

Hence two grouped protocols, and `leakage_gap` to quantify what random splitting buys
you in false confidence.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold

from . import config as C


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metrics(y_true, y_pred, log_space=True):
    """
    Accuracy in both spaces.

    A model fitted on log(E) is scored in log space by default, because that is the
    loss it minimised. The linear-space R^2 is reported alongside since it is what a
    reader assumes "R^2 = 0.9" means, and on a decade-spanning target the two can
    disagree sharply.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 3:
        return dict(n=int(len(y_true)), r2=np.nan, rmse=np.nan, mae=np.nan,
                    spearman=np.nan, r2_linear=np.nan)

    resid = y_true - y_pred
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    out = dict(
        n=int(len(y_true)),
        r2=float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
        rmse=float(np.sqrt((resid ** 2).mean())),
        mae=float(np.abs(resid).mean()),
        spearman=float(spearmanr(y_true, y_pred).statistic),
    )
    if log_space:
        a, b = np.exp(y_true), np.exp(y_pred)
        ss = ((a - a.mean()) ** 2).sum()
        out["r2_linear"] = float(1 - ((a - b) ** 2).sum() / ss) if ss > 0 else np.nan
        # Median fold-error is the number a designer actually feels: "predictions are
        # typically within a factor of X".
        out["median_fold_error"] = float(np.median(np.maximum(a / b, b / a)))
    else:
        out["r2_linear"] = out["r2"]
    return out


# --------------------------------------------------------------------------
# Splitters
# --------------------------------------------------------------------------

def leave_one_group_out(groups):
    """Yield (train_idx, test_idx, held_out_label) holding out one whole group at a time."""
    groups = pd.Series(groups).reset_index(drop=True)
    idx = np.arange(len(groups))
    for g in groups.unique():
        test = idx[(groups == g).values]
        train = idx[(groups != g).values]
        if len(test) >= 3 and len(train) >= 10:
            yield train, test, str(g)


def grouped_kfold(groups, n_splits=5, seed=C.SEED):
    """GroupKFold, but shuffled - sklearn's is deterministic on group order."""
    groups = pd.Series(groups).reset_index(drop=True)
    uniq = groups.unique()
    rng = np.random.default_rng(seed)
    perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
    shuffled = groups.map(perm).values
    n_splits = min(n_splits, len(uniq))
    for train, test in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)),
                                                          groups=shuffled):
        yield train, test, None


def random_kfold(n, n_splits=5, seed=C.SEED):
    for train, test in KFold(n_splits=n_splits, shuffle=True,
                             random_state=seed).split(np.zeros(n)):
        yield train, test, None


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def cross_validate(make_model, X, y, splitter, log_space=True, return_preds=False):
    """
    Out-of-fold evaluation. `make_model` is a zero-arg factory so every fold gets a
    fresh, unfitted estimator - reusing one instance silently leaks fitted state.
    """
    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
    y = np.asarray(y, float)
    oof = np.full(len(y), np.nan)
    per_group = []

    for train, test, label in splitter:
        model = make_model()
        Xtr = X.iloc[train] if hasattr(X, "iloc") else X[train]
        Xte = X.iloc[test] if hasattr(X, "iloc") else X[test]
        model.fit(Xtr, y[train])
        pred = np.asarray(model.predict(Xte), float)
        oof[test] = pred
        if label is not None:
            per_group.append(dict(held_out=label, **metrics(y[test], pred, log_space)))

    res = metrics(y[~np.isnan(oof)], oof[~np.isnan(oof)], log_space)
    res["coverage"] = float(np.mean(~np.isnan(oof)))
    if per_group:
        res["per_group"] = sorted(per_group, key=lambda d: (d["r2"] is np.nan, d["r2"]))
    if return_preds:
        res["oof"] = oof
    return res


def leakage_gap(make_model, X, y, groups, n_splits=5, log_space=True):
    """
    The headline diagnostic: score the same model under a random split and a
    group-aware split. The difference is the optimism a random split manufactures.
    """
    rand = cross_validate(make_model, X, y, random_kfold(len(y), n_splits), log_space)
    grp = cross_validate(make_model, X, y, grouped_kfold(groups, n_splits), log_space)
    return dict(
        random=rand, grouped=grp,
        r2_gap=float(rand["r2"] - grp["r2"]),
        n_groups=int(pd.Series(groups).nunique()),
    )
