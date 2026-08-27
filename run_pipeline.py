"""
End-to-end pipeline. One command, every stage, all artefacts.

    .venv/bin/python run_pipeline.py

Stages
  1  dataset audit          what exists, what converged, what is still missing
  2  model ladder           physics vs ML vs physics-informed, leave-one-architecture-out
  3  leakage audit          what a random split would have told you instead
  4  mesh convergence       how much of the answer is the mesh
  5  multi-fidelity fusion  how many expensive points are actually needed
  6  inverse design         the constrained search, per anatomical site

Everything lands in results/: metrics.json, figures/, tables/, and README.md.
Stages whose inputs are missing report themselves as skipped rather than failing, so
the pipeline runs today on simulation alone and strengthens as curated data arrives.
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from pipeline import config as C
from pipeline import data as D
from pipeline import fusion, inverse, models, report, validation as V

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

TARGET = "E_rel_z"
RESULTS = {}
FIGURES = []


def section(n, title):
    print(f"\n{'='*74}\n[{n}] {title}\n{'='*74}", flush=True)


# --------------------------------------------------------------------------

def stage_audit():
    section(1, "Dataset audit")
    sim = D.load_simulated()
    by_fam = sim.groupby("family").agg(
        rows=("sample_id", "size"),
        architectures=("architecture", "nunique"),
        P_min=("porosity", "min"), P_max=("porosity", "max"),
        converged=("cg_converged", "sum"), open_pored=("pore_connected", "sum"),
    ).reset_index()
    print(by_fam.to_string(index=False))

    closed = int((sim.pore_connected == 0).sum())
    print(f"\n{closed} of {len(sim)} geometries have CLOSED porosity - useless for bone "
          f"ingrowth whatever their modulus, and a hard constraint downstream.")

    t3 = D.tier3_status()
    print(f"Tier 3 (curated literature): {t3['message']}")

    RESULTS["dataset"] = dict(
        n_simulated=len(sim), by_family=by_fam.to_dict("records"),
        n_closed_porosity=closed, tier3=t3,
        n_architectures=int(sim.architecture.nunique()),
    )
    return sim


def stage_model_ladder(sim):
    section(2, "Model ladder - leave-one-architecture-out")
    df, dropped = D.clean_for_target(sim, TARGET)
    print(f"{len(df)} usable rows ({dropped} dropped: non-converged or non-positive)")

    X = D.build_features(df, allow_solved=False)
    y = D.get_target(df, TARGET)
    groups = df.architecture
    print(f"{X.shape[1]} design features · {groups.nunique()} architectures held out in turn")

    rows = []
    for name, factory in models.model_zoo(groups=groups).items():
        res = V.cross_validate(factory, X, y, V.leave_one_group_out(groups))
        rows.append(dict(model=name, r2=res["r2"], rmse=res["rmse"],
                         spearman=res["spearman"], r2_linear=res["r2_linear"],
                         median_fold_error=res.get("median_fold_error"),
                         n=res["n"]))
        print(f"  {name:26s} R2={res['r2']:+.3f}  rho={res['spearman']:.3f}  "
              f"typical error x{res.get('median_fold_error', float('nan')):.2f}")

    ladder = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    best = ladder.iloc[0]
    ga = ladder[ladder.model == "gibson_ashby_global"].iloc[0]
    print(f"\nBest: {best.model} (R2={best.r2:.3f}) vs textbook Gibson-Ashby "
          f"(R2={ga.r2:.3f}) - a gain of {best.r2 - ga.r2:+.3f}")

    # Which design variables the winner leans on - the interpretability the viva needs.
    winner = models.PhysicsInformedResidual(kind="rf").fit(X, y)
    imp = models.permutation_importance(winner, X, y).head(10)
    print("\nTop design variables (permutation importance on the residual model):")
    print(imp.to_string(index=False))

    RESULTS["model_ladder"] = ladder.to_dict("records")
    RESULTS["feature_importance"] = imp.to_dict("records")
    return ladder, X, y, groups, df


def stage_leakage(sim, X, y, groups):
    section(3, "Leakage audit - what a random split would have claimed")

    # (a) simulated data, grouped by architecture
    gap_sim = V.leakage_gap(lambda: models.PhysicsInformedResidual(), X, y, groups)
    print(f"Simulated modulus  random R2={gap_sim['random']['r2']:.3f}  "
          f"grouped R2={gap_sim['grouped']['r2']:.3f}  "
          f"inflation={gap_sim['r2_gap']:+.3f}")

    # (b) MLATE printability, grouped by DOI - the real-publication version of the claim
    panel = None
    mlate = D.load_mlate(bone_only=False)
    if mlate is not None and "Printability" in mlate.columns:
        m = mlate.dropna(subset=["Printability", "DOI"]).copy()
        # Every target column has to go, not just the one being predicted. "Scaffold
        # Quality (P*C)" is Printability x Cell Response, so leaving either of those in
        # hands the model the answer and inflates BOTH split protocols - which would
        # have hidden the very leakage this stage exists to measure.
        leaky = ["Printability", "Cell Response", "Scaffold Quality (P*C)"]
        feats = m.select_dtypes(include=[np.number]).drop(columns=leaky, errors="ignore")
        feats = feats.loc[:, feats.notna().mean() > 0.5].fillna(feats.median())
        yy = m["Printability"].astype(int)
        gg = m["DOI"].astype(str)

        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     n_jobs=-1, random_state=C.SEED)
        acc_rand = cross_val_score(clf, feats, yy,
                                   cv=KFold(5, shuffle=True, random_state=C.SEED)).mean()
        acc_grp = cross_val_score(clf, feats, yy, groups=gg,
                                  cv=GroupKFold(5)).mean()
        baseline = yy.value_counts(normalize=True).max()
        print(f"MLATE printability random acc={acc_rand:.3f}  grouped acc={acc_grp:.3f}  "
              f"majority baseline={baseline:.3f}  inflation={acc_rand-acc_grp:+.3f}")
        verdict = ("the honest score does NOT beat guessing the majority class"
                   if acc_grp <= baseline else
                   f"the honest score beats the baseline by {acc_grp-baseline:+.3f}")
        print(f"  -> {verdict}")

        panel = dict(dataset="MLATE printability", metric="accuracy", group_by="DOI",
                     random=float(acc_rand), grouped=float(acc_grp),
                     baseline=float(baseline), n_groups=int(gg.nunique()),
                     verdict=verdict)

    RESULTS["leakage"] = dict(
        simulated=dict(random_r2=gap_sim["random"]["r2"], grouped_r2=gap_sim["grouped"]["r2"],
                       inflation=gap_sim["r2_gap"], n_groups=gap_sim["n_groups"]),
        mlate=panel)
    return panel


def _gp_features(df):
    """Compact numeric vector for the GP. Kernels degrade fast above a handful of dims."""
    cols = ["relative_density", "pore_size_um", "strut_thickness_um",
            "specific_surface_per_mm"]
    X = df[cols].astype(float).copy()
    X["is_fdm"] = (df.family == "FDM").astype(float)
    return X.values


def stage_fusion():
    section(4, "Mesh convergence and multi-fidelity fusion")
    conv = D.load_convergence()
    if conv is None:
        print("SKIPPED - data/mesh_convergence.csv not found "
              "(run: .venv/bin/python sim/convergence.py)")
        RESULTS["convergence"] = dict(skipped=True)
        RESULTS["fusion"] = dict(skipped=True)
        return None, None

    hi_grid = conv.grid.max()
    finest = conv.sort_values("grid").groupby("case").last()
    d = conv.join(finest[["E_rel_z"]].rename(columns={"E_rel_z": "E_ref"}), on="case")
    # A case whose solid phase does not span the specimen at the finest grid has a
    # reference modulus of exactly zero, and a percentage deviation from zero is
    # undefined - not merely large. Excluded and counted rather than left to poison the
    # summary with inf.
    n_cases = d.case.nunique()
    d = d[d.E_ref > C.DEAD_MODULUS]
    n_excluded = n_cases - d.case.nunique()
    d["pct_err"] = 100 * (d.E_rel_z - d.E_ref).abs() / d.E_ref
    summ = d.groupby("grid")["pct_err"].agg(["median", "max"]).round(2)
    print(f"Deviation from the grid-{hi_grid} answer "
          f"({n_cases - n_excluded} cases; {n_excluded} excluded as non-spanning):")
    print(summ.to_string())
    RESULTS["convergence"] = dict(
        skipped=False, finest_grid=int(hi_grid), n_cases=int(n_cases - n_excluded),
        n_excluded_nonspanning=int(n_excluded),
        by_grid={int(g): dict(median_pct=float(r["median"]), max_pct=float(r["max"]))
                 for g, r in summ.iterrows()})

    # ---- paired low/high fidelity ----
    # Same exclusion as the convergence table, and for a sharper reason here. The
    # compression solve returns E_rel = 0 for a solid phase that spans nothing, and the
    # GP is fitted on log E: clipped at 1e-9 that is -20.7 against a normal range of
    # -6 to -1, so one non-spanning row dominates the kernel and drags the whole fusion
    # experiment negative. A geometry that carries no load has no modulus to fuse.
    #
    # The filter is per ROW, not per case. Filtering only on the finest grid leaves a
    # case that spans at 44 but not at 20 in the cheap tier with E = 0, which is exactly
    # the row that does the damage - it sits in y_low, the source the fused model is
    # supposed to trust. Pairing then intersects what survives at each grid.
    conv = conv[conv.E_rel_z > C.DEAD_MODULUS]
    high = conv[conv.grid == hi_grid].set_index("case")
    hi_families = set(high.family)

    # The cheap tier has to span the same families as the expensive one, so the low grid
    # is chosen rather than assumed to be the coarsest. A filament thinner than ~3 voxels
    # is not meshed at all, so grid 16 holds no FDM case whatsoever; pairing against it
    # would drop a whole family from the fusion experiment while still reporting a single
    # global R2 - the fused model would be credited for geometries it never saw cheaply.
    lo_grid = None
    for g in sorted(conv.grid.unique()):
        if g >= hi_grid:
            break
        paired = conv[(conv.grid == g) & conv.case.isin(high.index)]
        if set(paired.family) == hi_families and len(paired) >= 12:
            lo_grid = g
            break
    if lo_grid is None:
        print("SKIPPED fusion - no coarse grid covers every family present at the finest")
        RESULTS["fusion"] = dict(skipped=True, reason="no family-complete coarse grid")
        return conv, None

    low = conv[conv.grid == lo_grid].set_index("case")
    cases = sorted(set(low.index) & set(high.index))
    if len(cases) < 12:
        print(f"SKIPPED fusion - only {len(cases)} cases resolved at both grids")
        RESULTS["fusion"] = dict(skipped=True, reason=f"{len(cases)} paired cases")
        return conv, None

    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(cases)
    n_test = max(6, len(cases) // 3)
    test_cases, pool_cases = list(perm[:n_test]), list(perm[n_test:])

    X_low = _gp_features(low.loc[cases])
    y_low = np.log(low.loc[cases].E_rel_z.clip(lower=1e-9))
    X_high = _gp_features(high.loc[pool_cases])
    y_high = np.log(high.loc[pool_cases].E_rel_z.clip(lower=1e-9))
    X_test = _gp_features(high.loc[test_cases])
    y_test = np.log(high.loc[test_cases].E_rel_z.clip(lower=1e-9))

    print(f"\nfusion: low=grid {lo_grid} ({len(cases)} cases) · "
          f"high=grid {hi_grid} ({len(pool_cases)} available, {len(test_cases)} held out)")
    # Budgets are filtered against the pool rather than fixed, so widening the
    # convergence sweep to every topology extends the curve instead of truncating it -
    # and the top of the curve is where "the cheap source stops helping" becomes visible.
    budgets = [b for b in (4, 8, 12, 16, 24, 32, 48, 64) if b <= len(pool_cases)]
    mf = fusion.fidelity_experiment(X_low, y_low, X_high, y_high, X_test, y_test,
                                    n_high_grid=budgets)
    piv = mf.pivot_table(index="n_high", columns="model", values="r2").round(3)
    print(piv.to_string())

    best_fused = mf[mf.model == "fused"].sort_values("r2").iloc[-1]
    print(f"\nbest fused R2={best_fused.r2:.3f} at {int(best_fused.n_high)} high-fidelity "
          f"points (rho={best_fused.rho:.3f})")
    RESULTS["fusion"] = dict(skipped=False, low_grid=int(lo_grid), high_grid=int(hi_grid),
                             n_paired_cases=len(cases), table=mf.round(4).to_dict("records"))
    return conv, mf


def stage_inverse(sim):
    section(5, "Inverse design - constrained search")
    cat = inverse.build_catalogue(sim)
    kept, why = inverse.geometric_filter(cat)
    print(f"catalogue: {len(cat)} simulated geometries in {cat.config.nunique()} configs")
    for reason, n in why.items():
        print(f"  rejected {n:4d}  {reason}")
    print(f"  -> {len(kept)} printable, ingrowth-capable geometries")

    if kept.empty:
        print("\nNo geometry survives the printability/ingrowth constraints.")
        RESULTS["inverse"] = dict(feasible=0, rejections=why)
        return None, None

    out = {}
    best_overall = None
    for site in ("trabecular_low", "trabecular_mid", "trabecular_high"):
        res = inverse.screen(kept, site=site)
        print(f"\n{site}: {res['n_feasible']}/{res['n_evaluated']} candidates meet "
              f"every clinical constraint")
        if res["n_feasible"]:
            top = res["ranked"].head(3)
            for _, r in top.iterrows():
                print(f"   cost={r.load_transfer_cost:.4f}  {r.config:28s} "
                      # 4dp, not 2: E_rel falls steeply near the top of a config's
                      # porosity range, so a rounded figure pasted into tools/predict.py
                      # describes a measurably different scaffold.
                      f"P={r.porosity:.4f} pore={r.pore_size_um:.0f}um  "
                      f"{r.polymer}/{r.ceramic} {r.ceramic_wt:.0%}  "
                      f"E0={r.E0_MPa:.0f}MPa sigma0={r.sigma0_MPa:.1f}MPa")
            cand = res["ranked"].iloc[0]
            if best_overall is None or site == "trabecular_mid":
                best_overall = cand
            out[site] = res["ranked"].head(10).to_dict("records")
        else:
            print("   none feasible - nearest misses:")
            for _, r in res["near_miss"].head(3).iterrows():
                failed = [k for k in ("meets_stiffness", "meets_strength", "pH_safe")
                          if not r[k]]
                print(f"   cost={r.load_transfer_cost:.4f}  {r.config:28s} "
                      f"{r.polymer}/{r.ceramic} - fails {', '.join(failed)}")
            out[site] = []

        C.RESULTS.mkdir(parents=True, exist_ok=True)
        res["ranked"].to_csv(C.RESULTS / f"design_{site}.csv", index=False)

    RESULTS["inverse"] = dict(catalogue=len(cat), printable=len(kept), rejections=why,
                              by_site={k: v[:5] for k, v in out.items()})
    return best_overall, kept


def stage_figures(sim, ladder, leak_panel, conv, mf, best_cand):
    section(6, "Figures and tables")
    FIGURES.append(report.fig_gibson_ashby(sim))
    FIGURES.append(report.fig_model_ladder(ladder, leak_panel))
    FIGURES.append(report.fig_anisotropy(sim))
    if conv is not None:
        FIGURES.append(report.fig_convergence(conv))
    if mf is not None:
        FIGURES.append(report.fig_multifidelity(mf))
    if best_cand is not None:
        FIGURES.append(report.fig_design(inverse.trajectory_for(best_cand), best_cand))
    for p in FIGURES:
        print(f"  wrote {p.relative_to(C.ROOT)}")


def write_summary():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "metrics.json").write_text(json.dumps(RESULTS, indent=2, default=str))

    ds, lad = RESULTS["dataset"], pd.DataFrame(RESULTS["model_ladder"])
    lad = lad.dropna(subset=["r2"])          # a model that failed to score is not "best"
    best = lad.sort_values("r2").iloc[-1]
    ga = lad[lad.model == "gibson_ashby_global"].iloc[0]
    L = ["# Results", "",
         f"Generated by `run_pipeline.py` from {ds['n_simulated']} simulated geometries "
         f"across {ds['n_architectures']} architectures.", "",
         "## 1. Can the model predict an architecture it has never seen?", "",
         "Leave-one-architecture-out: every model is trained with one whole topology "
         "removed, then asked to predict it. This is the only protocol that reflects how "
         "a design tool is actually used.", "",
         "| model | R² (log E/Es) | Spearman | typical error |", "|---|---|---|---|"]
    for _, r in lad.sort_values("r2", ascending=False).iterrows():
        L.append(f"| {r.model.replace('_', ' ')} | {r.r2:+.3f} | {r.spearman:.3f} | "
                 f"×{r.median_fold_error:.2f} |")
    L += ["", f"Best model **{best.model}** beats the textbook Gibson-Ashby power law by "
              f"**{best.r2 - ga.r2:+.3f} R²**.", "",
          "`gibson_ashby_per_arch` scoring identically to the global fit is not a bug - it "
          "is the point. Held-out architectures have no fitted (n, C) to look up, so a "
          "per-architecture table has nothing to say about a new design.", ""]

    lk = RESULTS.get("leakage", {})
    if lk.get("mlate"):
        m = lk["mlate"]
        L += ["## 2. What a random split would have claimed instead", "",
              f"| protocol | accuracy |", "|---|---|",
              f"| random 5-fold | {m['random']:.3f} |",
              f"| grouped by {m['group_by']} ({m['n_groups']} groups) | {m['grouped']:.3f} |",
              f"| majority-class baseline | {m['baseline']:.3f} |", "",
              f"Random splitting inflates accuracy by **{m['random']-m['grouped']:+.3f}** "
              f"on {m['dataset']}. Rows from one publication share a material batch, a "
              f"printer and an operator, so a random split puts near-duplicates on both "
              f"sides. Verdict: {m['verdict']}.", ""]

    cv = RESULTS.get("convergence", {})
    if not cv.get("skipped"):
        L += ["## 3. How much of the answer is the mesh?", "",
              f"| grid | median deviation from grid-{cv['finest_grid']} | worst |",
              "|---|---|---|"]
        for g, r in sorted(cv["by_grid"].items()):
            L.append(f"| {g}³ | {r['median_pct']:.2f}% | {r['max_pct']:.2f}% |")
        L.append("")

    fu = RESULTS.get("fusion", {})
    if not fu.get("skipped"):
        mf = pd.DataFrame(fu["table"])
        bf = mf[mf.model == "fused"].sort_values("r2").iloc[-1]
        # The interesting comparison is at the SMALLEST budget, not the best row: fusion's
        # whole claim is that it is usable when trusted data is scarce, and by the largest
        # budget every model has converged and there is nothing left to demonstrate.
        n_min = int(mf.n_high.min())
        at_min = mf[mf.n_high == n_min].set_index("model").r2
        L += ["## 4. How many expensive datapoints are actually needed?", "",
              f"Co-kriging fuses the cheap source (grid {fu['low_grid']}) with a handful of "
              f"trusted points (grid {fu['high_grid']}), scored on held-out grid-"
              f"{fu['high_grid']} data and averaged over repeated nested draws.", "",
              f"With only **{n_min}** trusted points the fused model reaches R² = "
              f"**{at_min['fused']:.3f}**, where fitting those same {n_min} points alone "
              f"gives R² = {at_min['high_only']:.3f} — an unusable model. That gap is the "
              f"argument for fusion. Best fused R² = {bf.r2:.3f} at {int(bf.n_high)} points "
              f"(ρ = {bf.rho:.3f}); by then the single-fidelity model has caught up "
              f"({mf[(mf.n_high == bf.n_high) & (mf.model == 'high_only')].r2.iloc[0]:.3f}), "
              "which is itself the answer to 'how many rows is enough'.", "",
              "Swap curated literature rows in as the high-fidelity source and this stage "
              "runs unchanged - that is the argument for the curation effort, quantified.",
              ""]

    inv = RESULTS.get("inverse", {})
    if inv:
        L += ["## 5. Inverse design", "",
              f"Of {inv['catalogue']} simulated geometries, **{inv['printable']}** are "
              f"printable and ingrowth-capable. Rejections:", ""]
        for reason, n in inv["rejections"].items():
            L.append(f"- {n} — {reason}")
        L.append("")
        for site, rows in inv.get("by_site", {}).items():
            if rows:
                r = rows[0]
                L.append(f"**{site}** — best: {r['config']}, P={r['porosity']:.4f}, "
                         f"pore {r['pore_size_um']:.0f} µm, {r['polymer']}/{r['ceramic']} "
                         f"{r['ceramic_wt']:.0%} → E₀ = {r['E0_MPa']:.0f} MPa, "
                         f"cost = {r['load_transfer_cost']:.4f}")
            else:
                L.append(f"**{site}** — no candidate meets every constraint (see "
                         f"`design_{site}.csv` for near misses)")
        L.append("")

    t3 = ds["tier3"]
    L += ["## Status of the three tiers", "",
          f"- **Tier 1 (MLATE, borrowed)** — in use for the leakage audit.",
          f"- **Tier 2/2b (simulated)** — {ds['n_simulated']} rows, "
          f"{ds['n_closed_porosity']} with closed porosity (excluded by constraint).",
          f"- **Tier 3 (curated literature)** — {t3['message']}.", "",
          "The pipeline runs end-to-end on simulation alone today. Tier 3 rows are picked "
          "up automatically from `data/tier3_curated.csv` — no code changes.", "",
          "## Figures", ""]
    L += [f"- `figures/{p.name}`" for p in FIGURES]
    L.append("")

    (C.RESULTS / "README.md").write_text("\n".join(L))
    print(f"\nwrote {(C.RESULTS / 'metrics.json').relative_to(C.ROOT)}")
    print(f"wrote {(C.RESULTS / 'README.md').relative_to(C.ROOT)}")


def main():
    t0 = time.time()
    sim = stage_audit()
    ladder, X, y, groups, df = stage_model_ladder(sim)
    leak_panel = stage_leakage(sim, X, y, groups)
    conv, mf = stage_fusion()
    best_cand, _ = stage_inverse(sim)
    stage_figures(sim, ladder, leak_panel, conv, mf, best_cand)
    write_summary()
    print(f"\nPipeline complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
