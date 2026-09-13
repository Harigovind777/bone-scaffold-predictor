"""
The accuracy ledger: every headline model, re-measured against a frozen baseline.

    .venv/bin/python tools/accuracy_ledger.py            # the ledger
    .venv/bin/python tools/accuracy_ledger.py --rejected # what was tried and dropped
    .venv/bin/python tools/accuracy_ledger.py --freeze    # adopt current as the baseline

WHY THIS EXISTS. "The model got better" is not a claim anyone can check, and on this
project it was three claims in a trench coat: three models, scored on three different
targets under three different protocols, one of them with a broken ruler. Improving them
needs a fixed reference point, so `results/accuracy_baseline.json` holds the numbers as
they stood before any of this and every run prints the delta against it.

THE SEVEN RULES THIS ENFORCES, in the order they mattered.

1  FREEZE A BASELINE FIRST. Numbers move for boring reasons - a seed, a library version,
   a row that stopped converging. Without a frozen file you cannot tell your improvement
   from the weather.

2  FIND THE CEILING BEFORE CHASING THE GAP. `--ceiling` fits each held-out architecture's
   own power law, which no honest model can beat, and the same with only its exponent
   revealed. Those came out at R2 = 0.953 and 0.848 against a baseline of 0.783. That
   said the whole remaining prize in the exponent was 0.065, which is what stopped a
   week going into hierarchical models to chase it.

3  NEST EVERY CHOICE MADE ON THE SCORE. A k-nearest-architecture borrow of (n, C) scored
   0.818 with k picked by looking at the answer, and 0.691 with k picked inside each fold.
   The 0.035 "gain" was the tuning. Anything selected against the reported metric has to
   be selected again inside the fold, and usually stops existing when it is.

4  FIX THE ESTIMATOR BEFORE ADDING A MODEL. The pooled Gibson-Ashby fit weighted each
   architecture by how many times the sweep happened to sample it. One vote per
   architecture instead - no new parameters, no new model - was worth more than every
   learned correction tried afterwards.

5  CHECK THE RULER. Printability accuracy was being reported against a 0.555 majority
   baseline on an ordinal target, so a model could buy accuracy by collapsing onto the
   commonest level, and the previous one had. A metric a constant answer nearly wins is
   not measuring the model.

6  DEGRADE GRACEFULLY OR NOT AT ALL. A wrapper that scores below the thing it wraps is a
   bug, not a trade-off. Fusion scored 0.269 where believing the cheap source alone
   scored 0.818; the physics-informed model scored 0.647 where its own prior scored
   0.783. Both now hold their own floor by construction, because the amount of each
   learned correction to believe is MEASURED on held-out folds rather than assumed - and
   on this data both measurements come back at zero, which is the models saying they
   have nothing to add and saying it in the only way that is checkable.

7  SPLIT ON SOMETHING THE DESIGNER CAN CHOOSE. The per-architecture power law is useless
   on an unseen topology because the topology label is an IDENTITY - a new design has no
   row in the table. The deformation mode is an INPUT, so a new topology still arrives
   with one, and the same estimator grouped by mode instead of by architecture went from
   contributing nothing to +0.0092. Before reaching for a bigger model, check whether the
   grouping variable you already have is one that transfers.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pipeline import config as C, data as D, fusion, models as M  # noqa: E402
from pipeline import printability as P, validation as V           # noqa: E402

BASELINE = C.RESULTS / "accuracy_baseline.json"

# Variants that were built, measured and dropped. Recorded so the next person does not
# spend the same week finding out, and so a claim of "you should have tried X" has a
# number attached to it rather than an opinion.
REJECTED = [
    ("ladder", "hierarchical GA: ridge from design-time morphometry -> (n, C)",
     0.7160, "17 architecture-level points, 8 descriptors; best descriptor correlates "
             "0.61 with n and that is not enough"),
    ("ladder", "hierarchical GA: kernel pooling, h and lambda nested per fold",
     0.6910, "the honest version of the kNN borrow below; nesting removed the gain "
             "entirely"),
    ("ladder", "kNN over architectures, k=8 chosen on the score",
     0.8180, "REPORTED HIGH AND FALSE - the same estimator scores 0.691 once k is "
             "chosen inside the fold. Rule 3."),
    ("ladder", "quadratic in log rho", 0.7754, "curvature is confounded with the "
                                               "between-architecture spread"),
    ("ladder", "cubic in log rho", 0.7829, "as above, plus variance"),
    ("ladder", "percolation form C*(rho - rho_c)^n", 0.7826,
     "rho_c fits to its lower bound; no architecture-transferable percolation threshold"),
    ("ladder", "two-term C1*rho^n + C2*rho (bending + stretching)", 0.7904,
     "not rejected on score - C2 fits to zero, so it IS the adopted single-term fit "
     "with two extra parameters"),
    ("ladder", "Huber loss on the power law", 0.7770,
     "the tails it downweights are real geometries, not outliers"),
    ("ladder", "dimensionless-only feature block for the residual", 0.7615,
     "scale-free features transfer no better than the existing design block"),
    ("ladder", "shrinkage chosen by inner-CV argmin instead of one-SE", 0.7743,
     "argmin buys noise; one-SE correctly returns s=0"),
    ("ladder", "per-mode power law SHRUNK toward the pooled fit, lambda nested per fold",
     0.7904, "the honest test of rule 3 on the per-mode split, and it came back "
             "ambiguous: the one-SE rule picks lambda=0 (0.7904) and the argmin picks "
             "0.75 (0.8000) on the same inner folds. A knob whose answer depends on "
             "which standard-error convention you hold is not evidence, so the adopted "
             "model has no knob - it fits the modes outright (0.7996), a grouping "
             "chosen over per-family by the score and predicted by the physics."),
    ("ladder", "per-FAMILY power law (FDM vs TPMS) instead of per-mode", 0.7910,
     "family is also design-time knowledge, but it splits the data where the physics "
     "does not: sheet and network TPMS share a family and differ by 0.6 in exponent"),
    ("fusion", "discrepancy trust chosen by LOO argmin instead of one-SE", 0.7208,
     "at 4 trusted points argmin believes 35% of a correction fitted to 3 of them; "
     "one-SE returns 0 and holds the floor"),
    ("fusion", "stacking rho*f_low with a high-fidelity GP, weight nested by LOO",
     0.8221, "that score is at 4 points only, where one of five draws took a 10% "
             "weight; at every other budget the weight is 0 and it is the adopted model "
             "plus a second GP per LOO fold. +0.003 on one budget does not buy the "
             "machinery"),
    ("printability", "cut points fitted to maximise QWK inside each fold", 0.512,
     "ACCURACY shown. QWK rises to 0.477 but accuracy falls 0.08 - the cut points buy "
     "ordinal agreement by moving mass off the majority level. A real trade-off rather "
     "than an improvement, so the naive 0.5/1.5/2.5 cuts stay"),
    ("printability", "ExtraTrees in place of the random forest", 0.572,
     "ACCURACY shown; QWK 0.341, worse than the forest on both"),
]


def _sim_setup():
    sim = D.load_simulated()
    df, dropped = D.clean_for_target(sim, "E_rel_z")
    X = D.build_features(df, allow_solved=False)
    y = D.get_target(df, "E_rel_z")
    return sim, df, X, y, df.architecture, dropped


def measure_ladder():
    """Leave-one-architecture-out R2 for every rung. The headline modulus number."""
    _, df, X, y, groups, _ = _sim_setup()
    out = {}
    for name, factory in M.model_zoo(groups=groups, modes=df["mode"]).items():
        r = V.cross_validate(factory, X, y, V.leave_one_group_out(groups))
        out[name] = dict(r2=r["r2"], rmse=r["rmse"], spearman=r["spearman"],
                         median_fold_error=r["median_fold_error"], n=r["n"])
    return out


def measure_ceiling():
    """
    Bounds no honest model can pass, so the size of the remaining prize is known.

    `oracle_per_arch` fits the held-out architecture's own two parameters - it is what a
    perfect predictor of (n, C) would score. `oracle_exponent` reveals only the exponent
    and takes the level from the training fit, which is the half of the gap that a
    descriptor-based model would have had to earn.
    """
    _, df, _, y, groups, _ = _sim_setup()
    y = np.asarray(y, float)
    lr = np.log(df.relative_density.values)
    g = groups.reset_index(drop=True)

    def run(fn):
        oof = np.full(len(y), np.nan)
        for a in g.unique():
            te = (g == a).values
            tr = ~te
            if te.sum() < 3 or tr.sum() < 10:
                continue
            oof[te] = fn(tr, te)
        ok = ~np.isnan(oof)
        return V.metrics(y[ok], oof[ok])["r2"]

    def per_arch(tr, te):
        n, c = np.polyfit(lr[te], y[te], 1)
        return c + n * lr[te]

    def exponent_only(tr, te):
        piv = lr[tr].mean()
        ng, cg = np.polyfit(lr[tr], y[tr], 1)
        n_t, _ = np.polyfit(lr[te], y[te], 1)
        return (cg + ng * piv) + n_t * (lr[te] - piv)

    return dict(oracle_per_arch=run(per_arch), oracle_exponent=run(exponent_only))


def measure_printability():
    """Grouped-by-DOI scores for the old readout and the new one, on the same folds."""
    m = D.load_mlate(bone_only=False)
    if m is None or P.TARGET not in m.columns:
        return None
    m = m.dropna(subset=[P.TARGET, "DOI"]).copy()
    y = m[P.TARGET].astype(int).values
    g = m["DOI"].astype(str).values
    return dict(
        majority_baseline=P.majority_baseline(y),
        argmax_readout=P.evaluate(P.build_features(m, enrich=False), y, g,
                                  readout=P.argmax_level),
        expected_readout=P.evaluate(P.build_features(m, enrich=True), y, g,
                                    readout=P.expected_level),
    )


def measure_fusion():
    """Fused R2 across the high-fidelity budget, plus the cheap-source floor it must hold."""
    conv = D.load_convergence()
    if conv is None:
        return None
    hi = conv.grid.max()
    conv = conv[conv.E_rel_z > C.DEAD_MODULUS]
    high = conv[conv.grid == hi].set_index("case")
    families = set(high.family)
    lo = None
    for grid in sorted(conv.grid.unique()):
        if grid >= hi:
            break
        paired = conv[(conv.grid == grid) & conv.case.isin(high.index)]
        if set(paired.family) == families and len(paired) >= 12:
            lo = grid
            break
    if lo is None:
        return None
    low = conv[conv.grid == lo].set_index("case")
    cases = sorted(set(low.index) & set(high.index))
    perm = np.random.default_rng(C.SEED).permutation(cases)
    n_test = max(6, len(cases) // 3)
    test, pool = list(perm[:n_test]), list(perm[n_test:])

    def feats(d):
        cols = ["relative_density", "pore_size_um", "strut_thickness_um",
                "specific_surface_per_mm"]
        X = d[cols].astype(float).copy()
        X["is_fdm"] = (d.family == "FDM").astype(float)
        return X.values

    mf = fusion.fidelity_experiment(
        feats(low.loc[cases]), np.log(low.loc[cases].E_rel_z.clip(lower=1e-9)),
        feats(high.loc[pool]), np.log(high.loc[pool].E_rel_z.clip(lower=1e-9)),
        feats(high.loc[test]), np.log(high.loc[test].E_rel_z.clip(lower=1e-9)),
        n_high_grid=(4, 8, 12, 16, 24, 32, 48, 64))
    piv = mf.pivot_table(index="n_high", columns="model", values="r2")
    fused_rows = mf[mf.model == "fused"]
    return dict(low_only=float(piv["low_only"].iloc[0]),
                fused={int(k): float(v) for k, v in piv["fused"].items()},
                high_only={int(k): float(v) for k, v in piv["high_only"].items()},
                rho_mean=float(fused_rows.rho.mean()),
                delta_trust_mean=float(fused_rows.delta_trust.mean()))


def collect():
    t0 = time.time()
    print("measuring the model ladder (leave-one-architecture-out) ...", flush=True)
    ladder = measure_ladder()
    print("measuring the oracle ceiling ...", flush=True)
    ceiling = measure_ceiling()
    print("measuring printability (grouped by DOI) ...", flush=True)
    printab = measure_printability()
    print("measuring multi-fidelity fusion ...", flush=True)
    fuse = measure_fusion()
    return dict(ladder=ladder, ceiling=ceiling, printability=printab, fusion=fuse,
                seconds=round(time.time() - t0, 1))


def _delta(now, was, digits=4, higher_is_better=True):
    if now is None or not np.isfinite(now):
        return "—", ""
    if was is None or not np.isfinite(was):
        # A rung the frozen file has never seen. Saying so beats a blank column that
        # reads like a formatting bug.
        return f"{now:+.{digits}f}", "  (not in baseline)"
    d = now - was
    if abs(d) < 10 ** (-digits):
        return f"{now:+.{digits}f}", "  ="
    good = (d > 0) == higher_is_better
    return f"{now:+.{digits}f}", f"  {d:+.{digits}f} {'better' if good else 'WORSE'}"


def report(now, base):
    W = 78
    print("\n" + "=" * W + f"\n{'ACCURACY LEDGER':^{W}}\n" + "=" * W)
    if base is None:
        print("\nNo frozen baseline. Run with --freeze to record the current numbers.")

    b = (base or {}).get("ladder", {})
    print("\n1  MODULUS, leave-one-architecture-out (R2 on log E/Es)")
    print("   every model predicts a topology absent from training\n")
    rows = sorted(now["ladder"].items(), key=lambda kv: -kv[1]["r2"])
    print(f"   {'rung':26s} {'R2':>9s} {'vs baseline':>22s} {'typ. error':>11s}")
    for name, r in rows:
        val, note = _delta(r["r2"], (b.get(name) or {}).get("r2"))
        print(f"   {name:26s} {val:>9s} {note:>22s}   x{r['median_fold_error']:.3f}")
    if now.get("ceiling"):
        c = now["ceiling"]
        best = rows[0][1]["r2"]
        print(f"\n   ceiling: R2 = {c['oracle_per_arch']:.4f} with the held-out "
              f"architecture's own (n, C),")
        print(f"            R2 = {c['oracle_exponent']:.4f} with only its exponent "
              f"revealed.")
        print(f"   best honest model is {best:.4f}, so {c['oracle_exponent'] - best:+.4f} "
              f"of the gap is reachable")
        print(f"   by predicting the exponent and "
              f"{c['oracle_per_arch'] - c['oracle_exponent']:+.4f} more needs the level "
              f"too.")

    if now.get("printability"):
        p, pb = now["printability"], (base or {}).get("printability") or {}
        print("\n2  PRINTABILITY, grouped by DOI (ordinal 0-3, 88 publications)")
        print("   accuracy is shown but QWK is the number to read: answering the")
        print("   majority level to everything scores 0.000 on it and "
              f"{p['majority_baseline']['accuracy']:.3f} on accuracy\n")
        print(f"   {'readout':26s} {'acc':>8s} {'QWK':>8s} {'MAE':>8s} {'rho':>8s}")
        print(f"   {'majority level':26s} "
              f"{p['majority_baseline']['accuracy']:8.3f} {0.0:8.3f} "
              f"{p['majority_baseline']['mae_levels']:8.3f} {0.0:8.3f}")
        for key, label in (("argmax_readout", "classifier argmax"),
                           ("expected_readout", "expected level (now)")):
            r = p[key]
            print(f"   {label:26s} {r['accuracy']:8.3f} {r['qwk']:8.3f} "
                  f"{r['mae_levels']:8.3f} {r['spearman']:8.3f}")
        if pb.get("expected_readout"):
            for k, hib in (("accuracy", True), ("qwk", True), ("mae_levels", False),
                           ("spearman", True)):
                _, note = _delta(p["expected_readout"][k],
                                 pb["expected_readout"][k], 3, hib)
                print(f"   {k:>26s} {note}")

    if now.get("fusion"):
        f, fb = now["fusion"], (base or {}).get("fusion") or {}
        print("\n3  MULTI-FIDELITY FUSION (R2 on held-out fine-mesh cases)")
        print(f"   the cheap source alone scores {f['low_only']:.4f} - the floor fusion "
              f"must never fall below\n")
        print(f"   {'trusted pts':>12s} {'fused':>9s} {'vs baseline':>22s} "
              f"{'high_only':>10s}")
        for n_high in sorted(f["fused"], key=int):
            was = (fb.get("fused") or {}).get(str(n_high), (fb.get("fused") or {}).get(n_high))
            val, note = _delta(f["fused"][n_high], was)
            print(f"   {n_high:>12d} {val:>9s} {note:>22s} "
                  f"{f['high_only'][n_high]:>10.4f}")
        print(f"\n   mean rho = {f['rho_mean']:.4f}  (physics says 1; the prior is "
              f"N(1, {fusion.RHO_PRIOR_SD}^2))")
        if f.get("delta_trust_mean") is not None:
            print(f"   mean discrepancy trust = {f['delta_trust_mean']:.4f}  "
                  f"(0 = the correction earned nothing on held-out trusted points)")
    print("\n" + "=" * W)


def report_rejected():
    print("\nTRIED, MEASURED, DROPPED. Recorded so the same week is not spent twice, and"
          "\nso 'you should have tried X' comes with a number attached rather than an"
          "\nopinion. Each score is in its own stage's metric against its own adopted"
          "\nmodel:\n"
          "\n  ladder        leave-one-architecture-out R2   textbook 0.7828, adopted 0.7996"
          "\n  fusion        R2 on held-out fine-mesh cases  cheap source 0.8184, adopted 0.8182+"
          "\n  printability  grouped-by-DOI accuracy         majority 0.555, adopted 0.593\n")
    for stage in ("ladder", "fusion", "printability"):
        rows = [r for r in REJECTED if r[0] == stage]
        if not rows:
            continue
        print(f"  --- {stage} " + "-" * (66 - len(stage)))
        for _, what, score, why in rows:
            print(f"  {score:.4f}  {what}")
            print(f"          {why}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true",
                    help="write the current numbers as the new frozen baseline")
    ap.add_argument("--rejected", action="store_true",
                    help="print the variants that were measured and dropped, and why")
    ap.add_argument("--json", action="store_true", help="emit the ledger as JSON")
    args = ap.parse_args()

    if args.rejected:
        report_rejected()
        return 0

    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    now = collect()

    if args.json:
        print(C.dump_json(dict(current=now, baseline=base), indent=2))
    else:
        report(now, base)

    if args.freeze:
        C.RESULTS.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(C.dump_json(now, indent=2))
        print(f"\nfroze {BASELINE.relative_to(C.ROOT)} as the new baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
