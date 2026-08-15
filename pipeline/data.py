"""
Loaders and the single feature builder every model in the pipeline uses.

One rule enforced here: a feature is either a DESIGN variable (available before any
solve) or a SOLVED one. `build_features(..., allow_solved=False)` is the default because
the surrogate exists to replace the solve; letting a solved column in would make the
reported accuracy unreachable at design time.
"""

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_simulated():
    """Tier 2 + 2b: the unified simulated table from sim/generate_all.py."""
    if not C.SIMULATED.exists():
        raise FileNotFoundError(
            f"{C.SIMULATED} not found - run: .venv/bin/python sim/generate_all.py")
    df = pd.read_csv(C.SIMULATED)
    df["architecture"] = df.family + "|" + df.topology + "|" + df["mode"]

    # A solid phase that does not span the loading direction returns the ersatz-stiffness
    # floor (void_ratio = 1e-6), not a modulus. Those samples are a valid finding - the
    # geometry falls apart - but they are NOT power-law data, and a single one of them
    # will drag a log-log fit by whole units of exponent. Flagged here rather than in the
    # generator so the CSV stays raw measurements and the interpretation stays reviewable.
    df["structurally_dead"] = (df.E_rel_z < C.DEAD_MODULUS).astype(int)
    return df


def load_convergence():
    """Mesh-ladder table; also the low/high-fidelity pairs used by the fusion stage."""
    if not C.CONVERGENCE.exists():
        return None
    df = pd.read_csv(C.CONVERGENCE)
    df["architecture"] = df.family + "|" + df.topology + "|" + df["mode"]
    return df


def load_mlate(bone_only=True):
    """Tier 1: borrowed printability/biology data. Carries a DOI, hence groupable."""
    path = C.MLATE_BONE if bone_only else C.MLATE_FULL
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_tier3():
    """
    Tier 3: the curated literature table. Returns None until curation starts.

    Every stage that can use real measurements checks for this and silently falls back
    to simulation-only, so the pipeline runs end-to-end today and gets better - without
    code changes - the moment rows land.
    """
    if not C.TIER3.exists():
        return None
    df = pd.read_csv(C.TIER3)
    required = {"doi", "time_point_weeks"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{C.TIER3} is missing required column(s): {sorted(missing)}")
    return df


def tier3_status():
    """Human-readable one-liner on curation progress, for the report."""
    df = load_tier3()
    if df is None:
        schema_n = len(pd.read_csv(C.CURATION_SCHEMA)) if C.CURATION_SCHEMA.exists() else 0
        return dict(present=False, rows=0, dois=0,
                    message=f"not started - schema ready ({schema_n} columns), "
                            f"target 300-600 rows from 80-150 papers")
    col = "compressive_modulus"
    with_mod = int(df[col].notna().sum()) if col in df.columns else 0
    traj = 0
    if "modulus_retention_pct" in df.columns:
        traj = int(df.modulus_retention_pct.notna().sum())
    return dict(present=True, rows=len(df), dois=int(df.doi.nunique()),
                with_modulus=with_mod, with_retention=traj,
                message=f"{len(df)} rows / {df.doi.nunique()} DOIs, "
                        f"{with_mod} with a compressive modulus, "
                        f"{traj} with a modulus-retention trajectory point")


# --------------------------------------------------------------------------
# Feature building
# --------------------------------------------------------------------------

def build_features(df, allow_solved=False, fit_columns=None):
    """
    Design matrix from the unified simulated schema.

    Categoricals are one-hot encoded; NaN numerics (e.g. `strut_d` on a TPMS row, which
    has no struts) are filled with a sentinel and paired with an explicit `<col>_isna`
    flag, so "this family has no such parameter" is information the tree can use rather
    than a value it has to guess around.

    `fit_columns` reindexes to a previously built column set - required when predicting
    on a design catalogue that happens not to contain every topology seen in training.
    """
    cols = list(C.DESIGN_NUMERIC)
    if allow_solved:
        cols += [c for c in C.SOLVED_ONLY if c in df.columns]

    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c == "log_relative_density":
            v = np.log(df["relative_density"].clip(lower=1e-6))
        elif c in df.columns:
            v = df[c]
        else:
            continue
        v = pd.to_numeric(v, errors="coerce")
        X[c + "_isna"] = v.isna().astype(int)
        X[c] = v.fillna(-1.0)

    for c in C.DESIGN_CATEGORICAL:
        if c in df.columns:
            X = X.join(pd.get_dummies(df[c].astype(str), prefix=c, dtype=float))

    if fit_columns is None:
        # Constant `_isna` flags carry no signal and make the feature-importance table
        # noisy. Only those: a real feature that happens to be constant is dropped
        # harmlessly at fit time, but dropping it while reindexing to a fit column set
        # would refill it with 0.0 - so a catalogue of one family would silently predict
        # as `family_TPMS = 0`, i.e. not TPMS. Prune flags, never features.
        flags = [c for c in X.columns if c.endswith("_isna")]
        dead = [c for c in flags if X[c].nunique(dropna=False) <= 1]
        X = X.drop(columns=dead)
    else:
        X = X.reindex(columns=fit_columns, fill_value=0.0)
    return X


def get_target(df, name):
    """Target vector in the space the model should be fitted in (see config.TARGETS)."""
    y = pd.to_numeric(df[name], errors="coerce")
    if C.TARGETS.get(name) == "log":
        y = np.log(y.clip(lower=1e-9))
    return y


def invert_target(y, name):
    return np.exp(y) if C.TARGETS.get(name) == "log" else y


def clean_for_target(df, name):
    """Drop rows a given target cannot be fitted on, and say how many went."""
    ok = pd.to_numeric(df[name], errors="coerce").notna()
    if C.TARGETS.get(name) == "log":
        ok &= pd.to_numeric(df[name], errors="coerce") > 1e-9
    if "cg_converged" in df.columns:
        ok &= df.cg_converged == 1
    if "structurally_dead" in df.columns:
        ok &= df.structurally_dead == 0
    return df[ok].reset_index(drop=True), int((~ok).sum())
