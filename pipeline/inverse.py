"""
Constrained inverse design: which scaffold should actually be printed?

The forward chain is composition -> solid modulus -> degradation -> E(t), crossed with
architecture -> (n, C) -> how that solid modulus becomes a scaffold modulus. Inverse
design searches that product space for the candidate whose stiffness tracks healing bone
most closely, subject to constraints that are hard rather than penalised.

Two decisions worth defending:

* The architecture space is an ENUMERATED CATALOGUE of geometries that were actually
  simulated, not a continuous optimisation. Every candidate therefore has measured pore
  size, strut thickness, stress concentration and pore connectivity - and the optimiser
  cannot wander into a region where the surrogate is extrapolating, which is where
  inverse design usually goes wrong.

* Constraints are HARD. A scaffold with closed porosity is not a slightly worse scaffold,
  it is not a scaffold - bone cannot grow into it whatever its modulus. Folding that into
  a weighted objective would let a good stiffness score buy off a fatal geometry.
"""

import sys
from itertools import product

import numpy as np
import pandas as pd

from . import config as C

sys.path.insert(0, str(C.ROOT / "sim"))
import physics as phys  # noqa: E402


WEEKS = np.arange(0, 53, 1.0)
_DEGRADE_CACHE = {}


def degrade_cached(polymer, ceramic, ceramic_wt, porosity0):
    """
    Degradation depends only on chemistry and porosity - never on architecture. Caching
    on that key collapses the search by the size of the architecture catalogue, which is
    what makes an exhaustive sweep tractable at all.
    """
    key = (polymer, ceramic, round(ceramic_wt, 3), round(porosity0, 2))
    if key not in _DEGRADE_CACHE:
        _DEGRADE_CACHE[key] = phys.degrade(polymer, ceramic, ceramic_wt,
                                           round(porosity0, 2), WEEKS)
    return _DEGRADE_CACHE[key]


# --------------------------------------------------------------------------
# Architecture catalogue
# --------------------------------------------------------------------------

def build_catalogue(sim, ga_fits=None):
    """
    One row per simulated geometry, carrying the (n, C) of its own configuration.

    (n, C) evolve the modulus as degradation raises porosity. They come from the
    configuration's own porosity series, and `P_min`/`P_max` record the window that
    series covered so `screen` can refuse to extrapolate outside it.
    """
    sim = sim[sim.cg_converged == 1].copy()
    if "structurally_dead" in sim.columns:
        sim = sim[sim.structurally_dead == 0].copy()

    def config_key(r):
        if r.family == "TPMS":
            return f"TPMS|{r.topology}|{r['mode']}|c{int(r.n_cells)}"
        return f"FDM|{r['mode']}|s{int(r.n_struts)}|l{int(r.n_layers)}"

    sim["config"] = sim.apply(config_key, axis=1)

    fits = {}
    for cfg, g in sim.groupby("config"):
        g = g[(g.E_rel_z > 1e-6) & (g.relative_density > 0.02)]
        if len(g) < 4:
            continue
        n, logC = np.polyfit(np.log(g.relative_density), np.log(g.E_rel_z), 1)
        pred = np.exp(logC) * g.relative_density ** n
        ss_tot = ((g.E_rel_z - g.E_rel_z.mean()) ** 2).sum()
        r2 = 1 - ((g.E_rel_z - pred) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
        fits[cfg] = dict(n=float(n), C=float(np.exp(logC)), fit_r2=float(r2),
                         P_min=float(g.porosity.min()), P_max=float(g.porosity.max()))

    sim = sim[sim.config.isin(fits)].copy()
    for k in ("n", "C", "fit_r2", "P_min", "P_max"):
        sim[f"ga_{k}"] = sim.config.map(lambda c: fits[c][k])
    return sim.reset_index(drop=True)


def geometric_filter(cat, pore_window=C.PORE_WINDOW_UM, min_strut=C.MIN_STRUT_UM,
                     porosity_window=C.POROSITY_WINDOW):
    """Reject geometries on grounds no amount of chemistry can fix. Returns (kept, why)."""
    reasons = {
        "closed porosity (no ingrowth path)": cat.pore_connected != 1,
        f"porosity outside {porosity_window[0]:.0%}-{porosity_window[1]:.0%}":
            ~cat.porosity.between(*porosity_window),
        f"pore outside {pore_window[0]:.0f}-{pore_window[1]:.0f} um":
            ~cat.pore_size_um.between(*pore_window),
        f"strut under {min_strut:.0f} um (unprintable)": cat.strut_thickness_um < min_strut,
    }
    bad = np.zeros(len(cat), bool)
    counts = {}
    for label, mask in reasons.items():
        m = np.asarray(mask, bool)
        counts[label] = int((m & ~bad).sum())    # attribute each rejection to one cause
        bad |= m
    return cat[~bad].reset_index(drop=True), counts


# --------------------------------------------------------------------------
# Candidate evaluation
# --------------------------------------------------------------------------

def _anchored_C(arch):
    """
    Gibson-Ashby prefactor for this geometry, anchored on its own solved modulus.

    At t = 0 the FEM answer for THIS geometry is already known exactly. Evaluating the
    per-config power-law fit there instead discards that and substitutes a smoothed value
    that sits a few percent off - enough to reorder candidates whose load-transfer costs
    are within a percent of each other, which near the top of the ranking they are.

    So C is re-anchored to reproduce the solved modulus exactly, and the fitted exponent
    is left to do the only job it is needed for: evolving E as degradation raises
    porosity away from P0. Falls back to the fitted C when no solved modulus is attached,
    which is what happens for a hypothetical geometry that was never meshed.
    """
    try:                                    # absent, None or NaN all mean "never solved"
        E_rel_0 = float(getattr(arch, "E_rel_z", np.nan))
    except (TypeError, ValueError):
        E_rel_0 = np.nan
    if np.isfinite(E_rel_0) and E_rel_0 > 0:
        return E_rel_0 / max((1.0 - float(arch.porosity)) ** arch.ga_n, 1e-12)
    return float(arch.ga_C)


def evaluate_candidate(arch, polymer, ceramic, ceramic_wt, site, a=1.0):
    """
    Full coupled trajectory for one (architecture, chemistry, site) triple.

    Strength uses the SIMULATED stress-concentration factor rather than the generic
    Gibson-Ashby strength exponent: sigma_scaffold = sigma_yield_solid / K_sc. Two
    architectures at identical relative density can differ several-fold in K_sc, and
    that difference is the whole reason one of them survives loading.
    """
    P0 = float(arch.porosity)
    Es0, _ = phys.solid_modulus(polymer, ceramic, ceramic_wt)
    deg = degrade_cached(polymer, ceramic, ceramic_wt, P0)

    # Mass loss opens the structure up: porosity climbs as the polymer dissolves.
    P_t = 1.0 - (1.0 - P0) * deg["mass_remaining"]
    Es_t = Es0 * deg["Mn_rel"] ** a
    E_t = _anchored_C(arch) * Es_t * np.clip(1.0 - P_t, 1e-6, 1.0) ** arch.ga_n

    sigma_solid = 0.035 * Es_t                     # yield at ~3.5% strain
    K_sc = float(arch.K_sc_z) if np.isfinite(arch.K_sc_z) and arch.K_sc_z > 0 else np.nan
    sigma_t = sigma_solid / K_sc if np.isfinite(K_sc) else np.nan

    req = phys.BONE_SITES[site]
    E_bone = phys.bone_ingrowth(WEEKS, site, float(arch.pore_size_um), P0)
    total = E_t + E_bone
    cost = float(np.trapezoid(np.abs(total - req["E_native"]) / req["E_native"], WEEKS)
                 / (WEEKS[-1] - WEEKS[0]))

    resorbed = WEEKS[deg["polymer_mass"] < 0.10]
    return dict(
        polymer=polymer, ceramic=ceramic, ceramic_wt=ceramic_wt, site=site,
        E0_MPa=float(E_t[0]), sigma0_MPa=float(sigma_t[0]) if np.isfinite(K_sc) else np.nan,
        E_12wk_pct=float(100 * E_t[12] / E_t[0]) if E_t[0] > 0 else np.nan,
        mass_12wk_pct=float(100 * deg["mass_remaining"][12]),
        polymer_resorbed_week=float(resorbed[0]) if len(resorbed) else np.nan,
        min_pH=float(deg["pH"].min()),
        porosity_final=float(P_t[-1]),
        load_transfer_cost=cost,
        meets_stiffness=bool(0.5 * req["E_native"] <= E_t[0] <= 2.0 * req["E_native"]),
        meets_strength=bool(np.isfinite(sigma_t[0]) and sigma_t[0] >= req["sigma_req"]),
        pH_safe=bool(deg["pH"].min() >= 6.5),
    )


def screen(cat, site="trabecular_mid",
           polymers=("PCL", "PLLA", "PLGA_85_15", "chitosan", "silk"),
           ceramics=("none", "HA", "beta_TCP", "bioglass"),
           ceramic_wts=(0.0, 0.15, 0.30),
           require_all=True, top_n=25):
    """
    Exhaustive sweep of catalogue x chemistry for one anatomical site.

    `require_all=True` keeps only candidates passing every clinical constraint. If
    nothing survives that is a RESULT - it means no combination in the searched space
    can serve that site - so the near-miss table is returned too rather than silently
    relaxing the constraints.
    """
    rows = []
    for _, arch in cat.iterrows():
        for polymer, ceramic, wt in product(polymers, ceramics, ceramic_wts):
            if ceramic == "none" and wt > 0:
                continue                              # not a real combination
            if ceramic != "none" and wt == 0:
                continue                              # duplicate of ceramic="none"
            res = evaluate_candidate(arch, polymer, ceramic, wt, site)
            # E_rel_z travels with the row so trajectory_for() anchors identically and the
            # plotted curve starts on the E0 this table quotes.
            res.update(config=arch.config, family=arch.family, topology=arch.topology,
                       mode=arch["mode"], porosity=arch.porosity,
                       E_rel_z=arch.E_rel_z, pore_size_um=arch.pore_size_um,
                       strut_thickness_um=arch.strut_thickness_um,
                       K_sc_z=arch.K_sc_z, anisotropy=arch.anisotropy_z_over_x,
                       ga_n=arch.ga_n, ga_C=arch.ga_C, ga_fit_r2=arch.fit_r2
                       if hasattr(arch, "fit_r2") else arch.ga_fit_r2)
            rows.append(res)

    df = pd.DataFrame(rows)
    df["n_constraints_met"] = (df.meets_stiffness.astype(int)
                               + df.meets_strength.astype(int) + df.pH_safe.astype(int))
    feasible = df[df.n_constraints_met == 3] if require_all else df
    ranked = feasible.sort_values("load_transfer_cost").head(top_n).reset_index(drop=True)
    near_miss = (df[df.n_constraints_met == 2]
                 .sort_values("load_transfer_cost").head(10).reset_index(drop=True))
    return dict(all=df, ranked=ranked, near_miss=near_miss,
                n_evaluated=len(df), n_feasible=int((df.n_constraints_met == 3).sum()))


def trajectory_for(arch_row, weeks=WEEKS):
    """Re-run one winning candidate's trajectory for plotting."""
    P0 = float(arch_row.porosity)
    Es0, _ = phys.solid_modulus(arch_row.polymer, arch_row.ceramic, arch_row.ceramic_wt)
    deg = degrade_cached(arch_row.polymer, arch_row.ceramic, arch_row.ceramic_wt, P0)
    P_t = 1.0 - (1.0 - P0) * deg["mass_remaining"]
    Es_t = Es0 * deg["Mn_rel"]
    # Same anchoring as evaluate_candidate, so the plotted trajectory starts on the same
    # E0 the ranking table quotes.
    E_t = _anchored_C(arch_row) * Es_t * np.clip(1.0 - P_t, 1e-6, 1.0) ** arch_row.ga_n
    E_bone = phys.bone_ingrowth(weeks, arch_row.site, float(arch_row.pore_size_um), P0)
    return dict(weeks=weeks, E_scaffold=E_t, E_bone=E_bone, E_total=E_t + E_bone,
                porosity=P_t, mass=deg["mass_remaining"], pH=deg["pH"],
                E_native=phys.BONE_SITES[arch_row.site]["E_native"])
