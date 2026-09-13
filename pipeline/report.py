"""
Figures and tables.

Static PNGs for a thesis/report, so these deliberately commit to the light surface
rather than shipping a dark variant. Every figure has a companion CSV written beside it
in results/tables/ - that is both the table view the accessibility pass requires and the
thing a reader needs to re-plot a panel their own way.

Palette: the validated reference categorical order. Scatter and small-multiple forms use
at most the first THREE slots, which are the ones that clear the all-pairs CVD and
normal-vision floors; anything needing more identities is faceted instead of coloured.
Slot 3 (aqua) sits below 3:1 on the light surface, so every series is directly labelled -
colour never carries identity alone.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

# ---- validated reference palette, light mode ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STATUS = dict(good="#0ca30c", warning="#fab219", critical="#d03b3b")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,      # gridlines behind the marks, never across them
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def _save(fig, name, table=None):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    if table is not None:
        tdir = C.RESULTS / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(tdir / name.replace(".png", ".csv"), index=False)
    return path


def _title(ax, title, subtitle=None):
    """Title above subtitle above the axes. The pad has to clear the subtitle line,
    otherwise the two render on top of each other."""
    ax.set_title(title, loc="left", color=INK, fontweight="bold",
                 pad=24 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=8.5, va="bottom", ha="left")


# --------------------------------------------------------------------------

def fig_gibson_ashby(sim):
    """
    Small multiples, one architecture per panel, log-log.

    Faceted rather than coloured because there are eighteen architectures and no palette
    keeps even ten identities separable in a scatter. Each panel carries its own fitted
    exponent, which is the number the panel exists to communicate; the shared dashed
    line is the textbook n = 2 so the departure is readable at a glance.
    """
    n_dead = int(sim.get("structurally_dead", pd.Series(0, index=sim.index)).sum())
    sim = sim[(sim.cg_converged == 1) & (sim.get("structurally_dead", 0) == 0)].copy()
    sim["arch"] = np.where(sim.family == "TPMS",
                           sim.topology + " " + sim["mode"],
                           "FDM " + sim["mode"])
    archs = sorted(sim.arch.unique())
    ncol = 5
    nrow = int(np.ceil(len(archs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.45 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    rows = []

    rd = np.linspace(max(sim.relative_density.min(), 0.05), sim.relative_density.max(), 50)
    for ax, arch in zip(axes, archs):
        g = sim[sim.arch == arch]
        color = SERIES[0] if g.family.iloc[0] == "TPMS" else SERIES[1]
        ax.loglog(g.relative_density, g.E_rel_z, "o", ms=4.5, color=color,
                  markeredgecolor=SURFACE, markeredgewidth=0.6, alpha=0.95)
        if len(g) >= 4:
            n, logC = np.polyfit(np.log(g.relative_density), np.log(g.E_rel_z), 1)
            ax.loglog(rd, np.exp(logC) * rd ** n, "-", lw=2, color=color, alpha=0.85)
            ax.loglog(rd, rd ** 2, "--", lw=1.2, color=MUTED, alpha=0.7)
            ax.text(0.04, 0.93, f"n = {n:.2f}", transform=ax.transAxes,
                    fontsize=9, color=INK, fontweight="bold", va="top")
            rows.append(dict(architecture=arch, n=round(n, 3),
                             C=round(float(np.exp(logC)), 3), points=len(g)))
        ax.set_title(arch, loc="left", fontsize=9.5, color=INK)
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(False, which="minor")

    for ax in axes[len(archs):]:
        ax.set_visible(False)
    # Log minor ticks collide at this panel width; a few named majors read better.
    ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 0.8) if
             sim.relative_density.min() * 0.9 <= t <= sim.relative_density.max() * 1.1]
    for ax in axes[:len(archs)]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("relative density")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("$E/E_s$")

    note = "solid line: fitted power law   ·   dashed: textbook n = 2"
    if n_dead:
        note += (f"   ·   {n_dead} non-spanning geometries excluded "
                 f"(solid phase does not percolate)")
    fig.suptitle("Stiffness-density scaling is architecture-specific",
                 x=0.005, y=1.0, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.005, 0.972, note, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save(fig, "fig1_gibson_ashby.png", pd.DataFrame(rows))


def fig_model_ladder(ladder_df, leakage=None):
    """Panel A: the model ladder under honest validation. Panel B: the leakage gap."""
    has_b = leakage is not None
    fig, axes = plt.subplots(1, 2 if has_b else 1,
                             figsize=(11 if has_b else 6.2, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1]} if has_b else None)
    axes = np.atleast_1d(axes)
    ax = axes[0]

    d = ladder_df.sort_values("r2")
    y = np.arange(len(d))
    winner = d.model.iloc[-1]
    # Highlight whichever model actually won. A title asserting a conclusion the numbers
    # do not support is how a figure starts lying, so both are read off the data.
    colors = [SERIES[2] if m == winner else SERIES[0] for m in d.model]
    ax.barh(y, d.r2, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ") for m in d.model], fontsize=9, color=INK)
    for yi, v in zip(y, d.r2):
        ax.text(v + 0.015 if v > 0 else 0.015, yi, f"{v:.3f}", va="center",
                fontsize=9, color=INK, fontweight="bold")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("out-of-fold $R^2$  (log $E/E_s$)")
    ax.set_xlim(min(0, d.r2.min() * 1.15), max(1.0, d.r2.max() * 1.18))
    ax.grid(axis="y", visible=False)

    ga = d[d.model == "gibson_ashby_global"]
    margin = float(d.r2.iloc[-1] - ga.r2.iloc[0]) if len(ga) else np.nan
    if np.isfinite(margin) and margin >= 0.02:
        headline = f"{winner.replace('_', ' ')} beats the textbook power law by {margin:+.3f} R²"
    elif np.isfinite(margin):
        headline = "No model meaningfully beats the textbook power law here"
    else:
        headline = "Model ladder on unseen architectures"
    _title(ax, headline,
           "leave-one-architecture-out · each model must predict a topology absent from training")

    if has_b:
        ax = axes[1]
        labels = ["random\n5-fold", f"grouped\nby {leakage['group_by']}"]
        vals = [leakage["random"], leakage["grouped"]]
        bars = ax.bar(labels, vals, width=0.55,
                      color=[STATUS["critical"], SERIES[0]],
                      edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=10, color=INK, fontweight="bold")
        # The baseline is what makes the honest score interpretable: a grouped score
        # that merely matches "always guess the majority class" has learned nothing.
        if leakage.get("baseline") is not None:
            ax.axhline(leakage["baseline"], ls="--", lw=1.3, color=MUTED)
            ax.text(-0.42, leakage["baseline"] + 0.018, "majority-class baseline",
                    fontsize=8.5, color=MUTED, va="bottom", ha="left")
        ax.text(0.5, max(vals) * 1.14, f"random split inflates by +{vals[0]-vals[1]:.3f}",
                ha="center", va="center", fontsize=9.5, color=INK2, fontweight="bold")
        ax.set_ylabel(leakage["metric"])
        ax.set_ylim(0, max(vals) * 1.26)
        ax.grid(axis="x", visible=False)
        _title(ax, "Random splitting manufactures accuracy",
               f"{leakage['dataset']} · same model & data, {leakage['n_groups']} groups")

    fig.tight_layout()
    return _save(fig, "fig2_model_ladder.png", ladder_df)


def fig_convergence(conv):
    """Deviation from the finest grid. Answers the 'is 20^3 enough?' caveat with a number."""
    finest = conv.sort_values("grid").groupby("case").last()
    d = conv.join(finest[["E_rel_z"]].rename(columns={"E_rel_z": "E_ref"}), on="case")
    # Non-spanning cases have E_ref == 0 exactly, so their relative deviation is undefined.
    # Excluded here as well as in run_pipeline, so the figure and the metric agree.
    n_cases = d.case.nunique()
    d = d[d.E_ref > C.DEAD_MODULUS]
    n_excluded = n_cases - d.case.nunique()
    d["pct_err"] = 100 * (d.E_rel_z - d.E_ref).abs() / d.E_ref
    summ = (d.groupby(["family", "grid"])["pct_err"]
            .agg(median="median", p90=lambda s: s.quantile(0.9)).reset_index())

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for i, (fam, g) in enumerate(summ.groupby("family")):
        g = g.sort_values("grid")
        ax.plot(g.grid, g["median"], "-o", lw=2, ms=7, color=SERIES[i],
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=fam)
        ax.fill_between(g.grid, g["median"], g["p90"], color=SERIES[i], alpha=0.13)
        # No inline series label here, unlike the other figures: every curve converges on
        # the finest grid by construction, so direct labels always land on top of one
        # another. The legend carries the identification instead.

    ax.axhline(5, ls="--", lw=1.2, color=MUTED)
    ax.text(conv.grid.min(), 5.4, "5% tolerance", fontsize=8.5, color=MUTED)
    ax.set_xlabel("voxel grid (per side)")
    ax.set_ylabel("|deviation| from finest grid, %")
    ax.set_xticks(sorted(conv.grid.unique()))
    ax.legend(frameon=False, loc="upper right")
    sub = "line: median across geometries · band: up to the 90th percentile"
    if n_excluded:
        sub += f" · {n_excluded} non-spanning cases excluded"
    _title(ax, "Mesh sensitivity of the apparent modulus", sub)
    fig.tight_layout()
    return _save(fig, "fig3_convergence.png", summ.round(3))


def fig_multifidelity(mf):
    """How much expensive data the fused model actually needs. The curation business case."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    order = ["low_only", "high_only", "fused"]
    label = {"low_only": "cheap source alone", "high_only": "expensive alone",
             "fused": "co-kriging (fused)"}
    for i, m in enumerate(order):
        g = mf[mf.model == m].sort_values("n_high")
        if g.empty:
            continue
        if m == "low_only":
            # The floor is drawn as a dashed line ON TOP. When fusion correctly declines
            # to correct the cheap source the two curves coincide exactly, and a solid
            # floor drawn first disappears under the fused line - hiding the very
            # comparison the figure exists to make.
            ax.plot(g.n_high, g.r2, "--", lw=1.8, color=SERIES[i], zorder=5,
                    label=label[m])
        else:
            ax.plot(g.n_high, g.r2, "-o", lw=2, ms=7, color=SERIES[i], zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label[m])
        # Spread across repeated draws of the high-fidelity subset. Without it the reader
        # cannot tell a real budget effect from the luck of one draw.
        if "r2_std" in g and g.r2_std.notna().any() and g.r2_std.abs().sum() > 0:
            ax.fill_between(g.n_high, g.r2 - g.r2_std, g.r2 + g.r2_std,
                            color=SERIES[i], alpha=0.15, lw=0)
        # Legend only, for the same reason as fig3: the three models converge as the
        # high-fidelity budget grows - which is the result - so right-edge direct labels
        # overprint each other exactly where the curves are most interesting.

    ax.set_xlabel("number of high-fidelity points used")
    ax.set_ylabel("$R^2$ on held-out high-fidelity data")
    ax.set_xticks(sorted(mf.n_high.unique()))
    ax.set_xlim(right=mf.n_high.max() * 1.06)     # was 1.38 to clear the inline labels
    ax.legend(frameon=False, loc="lower right")
    reps = int(mf.n_repeats.max()) if "n_repeats" in mf else 1
    _title(ax, "Fusing a cheap biased source with a few trusted points",
           f"mean of {reps} nested draws · band: ±1 SD · "
           "swap in curated literature rows unchanged")
    fig.tight_layout()
    return _save(fig, "fig4_multifidelity.png", mf.round(4))


def fig_anisotropy(sim):
    """
    Anisotropy in BOTH transported quantities: stiffness on the left, diffusivity on
    the right.

    A 0/90 print is not isotropic in either, and the ratio depends on the lay-down
    offset - a variable most papers neither control nor report. The transport panel only
    became plottable once effective_diffusivity stopped ignoring its own axis argument;
    before that D_eff_x was a copy of D_eff_z and every ratio here was exactly 1.000,
    which would have read as a physical result rather than a bug.
    """
    sim = sim[(sim.anisotropy_z_over_x > 0) & (sim.cg_converged == 1)
              & (sim.get("structurally_dead", 0) == 0)].copy()
    sim["cls"] = np.where(sim.family == "TPMS", "TPMS", "FDM " + sim["mode"])
    classes = [c for c in ["TPMS", "FDM aligned", "FDM staggered"] if c in set(sim.cls)]

    has_transport = ("D_anisotropy_z_over_x" in sim.columns
                     and sim.D_anisotropy_z_over_x.gt(0).any())
    fig, axes = plt.subplots(1, 2 if has_transport else 1,
                             figsize=(11 if has_transport else 6.6, 4.2))
    axes = np.atleast_1d(axes)

    # `direct`: whether the series separate enough to carry right-edge labels. They do
    # in stiffness, where a staggered lay-down runs an order of magnitude off isotropic.
    # They do not in transport - TPMS is exactly 1.0 and the printed lattices stay inside
    # +-25% of it - so that panel is identified by its legend alone, the same call fig3
    # and fig4 make where their curves converge.
    panels = [("anisotropy_z_over_x", axes[0], True,
               "$E_z / E_x$  (build direction / in-plane)",
               "The compression axis changes the answer several-fold",
               "TPMS architectures are cubic and land on 1.0 · printed lattices do not")]
    if has_transport:
        panels.append(("D_anisotropy_z_over_x", axes[1], False,
                       "$D_z / D_x$  (build direction / in-plane)",
                       "So does the transport axis",
                       "aligned pores stack into straight channels; staggering breaks them"))

    for col, ax, direct, ylabel, title, subtitle in panels:
        g_all = sim[sim[col] > 0]
        labels = []
        for i, cls in enumerate(classes):
            g = g_all[g_all.cls == cls]
            if g.empty:
                continue
            ax.scatter(g.porosity, g[col], s=26, color=SERIES[i],
                       edgecolor=SURFACE, linewidth=0.6, alpha=0.9, label=cls)
            gx = g.sort_values("porosity")
            labels.append((float(gx[col].iloc[-1]), float(gx.porosity.iloc[-1]),
                           f"  {cls}", SERIES[i], "bold"))
        ax.axhline(1.0, ls="--", lw=1.2, color=MUTED)
        ax.text(g_all.porosity.min(), 1.04, "isotropic", fontsize=8.5, color=MUTED)
        ax.set_yscale("log")
        ax.set_xlabel("porosity")
        ax.set_ylabel(ylabel)
        ax.set_xlim(right=g_all.porosity.max() * 1.16)
        ax.legend(frameon=False, loc="lower left")
        _title(ax, title, subtitle)
        if direct:
            # After the scale and the limits are final, never before: the labels are
            # placed against the axis geometry.
            _stack_right_labels(ax, labels)

    agg = dict(n="size", median="median", q10=lambda s: s.quantile(.1),
               q90=lambda s: s.quantile(.9))
    tbl = sim.groupby("cls")["anisotropy_z_over_x"].agg(**agg).reset_index()
    tbl.columns = ["cls", "n", "E_median", "E_q10", "E_q90"]
    if has_transport:
        t2 = (sim[sim.D_anisotropy_z_over_x > 0]
              .groupby("cls")["D_anisotropy_z_over_x"].agg(**agg).reset_index())
        t2.columns = ["cls", "D_n", "D_median", "D_q10", "D_q90"]
        tbl = tbl.merge(t2, on="cls", how="left")
    fig.tight_layout()
    return _save(fig, "fig5_anisotropy.png", tbl.round(3))


def _stack_right_labels(ax, entries, min_gap_frac=0.058):
    """
    Place right-edge series labels, nudged apart so converging curves stay readable.

    Direct labelling beats a legend while curves are separated, but here the construct and
    the new bone converge on the native line by construction - that convergence IS the
    result - so three labels arrive at one y and overprint into an unreadable smear.
    Sorting by value and enforcing a minimum gap keeps label order faithful to curve order.

    Positions are converted to axes fraction before the gap is applied, so the same
    minimum spacing means the same thing on a log scale as on a linear one - fig5's
    panels are log-y, where a gap measured in data units is enormous at the top of the
    axis and invisible at the bottom. On a linear axis this is identical to measuring
    the gap as a fraction of the y range, which is what it used to do.
    """
    from matplotlib.transforms import blended_transform_factory
    ax.autoscale_view()          # transData is stale until the limits are resolved; the
                                 # earlier version got this for free from get_ylim()
    to_frac = ax.transAxes.inverted()
    items = sorted(
        ((float(to_frac.transform(ax.transData.transform((0, y)))[1]), x, t, c, w)
         for y, x, t, c, w in entries),
        key=lambda e: e[0])
    for i in range(1, len(items)):
        if items[i][0] - items[i - 1][0] < min_gap_frac:
            items[i] = (items[i - 1][0] + min_gap_frac,) + tuple(items[i][1:])
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for yf, x, text, color, weight in items:
        ax.text(x, yf, text, transform=tr, fontsize=9, color=color,
                fontweight=weight, va="center")


def fig_design(traj, cand):
    """The winning candidate: does construct stiffness track healing bone?"""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax.plot(traj["weeks"], traj["E_scaffold"], lw=2, color=SERIES[0], label="scaffold")
    ax.plot(traj["weeks"], traj["E_bone"], lw=2, color=SERIES[1], label="new bone")
    ax.plot(traj["weeks"], traj["E_total"], lw=2.4, color=SERIES[2], label="construct")
    ax.axhline(traj["E_native"], ls="--", lw=1.3, color=MUTED)
    x_end = traj["weeks"][-1]
    _stack_right_labels(ax, [
        (float(traj["E_native"]), x_end, " native", MUTED, "normal"),
        (float(traj["E_scaffold"][-1]), x_end, "  scaffold", SERIES[0], "bold"),
        (float(traj["E_bone"][-1]), x_end, "  new bone", SERIES[1], "bold"),
        (float(traj["E_total"][-1]), x_end, "  construct", SERIES[2], "bold"),
    ])
    ax.set_xlabel("weeks post-implantation")
    ax.set_ylabel("compressive modulus, MPa")
    ax.set_xlim(right=traj["weeks"][-1] * 1.22)
    _title(ax, "Load transfer over healing",
           f"{cand.polymer} / {cand.ceramic} {cand.ceramic_wt:.0%} · {cand.config}")

    ax2b = ax2
    ax2b.plot(traj["weeks"], 100 * traj["mass"], lw=2, color=SERIES[0])
    ax2b.text(traj["weeks"][-1], 100 * traj["mass"][-1], "  mass remaining",
              fontsize=9, color=SERIES[0], fontweight="bold", va="center")
    ax2b.plot(traj["weeks"], 100 * traj["porosity"], lw=2, color=SERIES[1])
    ax2b.text(traj["weeks"][-1], 100 * traj["porosity"][-1], "  porosity",
              fontsize=9, color=SERIES[1], fontweight="bold", va="center")
    ax2b.set_xlabel("weeks post-implantation")
    ax2b.set_ylabel("%")
    ax2b.set_xlim(right=traj["weeks"][-1] * 1.3)
    ax2b.set_ylim(0, 105)
    _title(ax2b, "Why stiffness falls",
           f"minimum pH {traj['pH'].min():.2f} · degradation opens the structure up")

    fig.tight_layout()
    tbl = pd.DataFrame(dict(weeks=traj["weeks"], E_scaffold=traj["E_scaffold"],
                            E_bone=traj["E_bone"], E_total=traj["E_total"],
                            porosity=traj["porosity"], mass_remaining=traj["mass"],
                            pH=traj["pH"])).round(4)
    return _save(fig, "fig6_design.png", tbl)
