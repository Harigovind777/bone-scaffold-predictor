"""
Prediction interface: one scaffold specification in, its predicted behaviour out.

This is the entry point a researcher actually uses. Everything else in the repo either
generates data or evaluates models; this answers the question the project exists to
answer - "if I print THIS, what do I get?"

    .venv/bin/python tools/predict.py --topology gyroid --mode network --porosity 0.68 \
        --polymer PLGA_85_15 --ceramic beta_TCP --wt 0.30 --site trabecular_mid

Two ways to resolve the geometry, because they trade accuracy against seconds:

  catalogue (default)  Interpolate the architecture's own simulated porosity series.
                       Instant. Valid only inside the porosity window that series
                       covered - outside it you are extrapolating a power law and the
                       tool says so rather than quietly returning a number.

  --solve              Build and mesh THIS geometry and run the voxel FEM now. Seconds,
                       and the only honest option for a geometry the sweep never sampled.

WHAT PREDICTS WHAT - worth being explicit, because it is the defensible part:

The modulus is NOT taken from a learned surrogate. Under leave-one-architecture-out the
surrogate scores R2 = 0.55 against 0.73 for the Gibson-Ashby power law it was built on
(see pipeline/models.py), so using it here would be dressing up a worse predictor as a
better one. What runs instead is the fitted (n, C) for the requested architecture plus
the mechanistic decoder in sim/physics.py, which has zero learned parameters. Time
evolution - hydrolysis, mass loss, the porosity that opens up as polymer leaves, the
stiffness that follows - is all physics.

Anchoring: where a measured E_rel exists at the requested porosity (always under
--solve, and inside the sampled window otherwise), C is re-anchored so the trajectory
starts exactly on the measured value and the fitted exponent governs only how the
modulus MOVES as degradation raises porosity. That is strictly better than evaluating
the global fit at a point where the true answer is already known.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config as C           # noqa: E402
from pipeline import data as D             # noqa: E402
from pipeline import inverse               # noqa: E402

sys.path.insert(0, str(ROOT / "sim"))
import physics as phys                     # noqa: E402

# Interpolated off the architecture's own porosity series. E_rel_z is handled separately
# (it is the anchor); these are the descriptors the constraints and strength model need.
_INTERP_COLS = ("pore_size_um", "strut_thickness_um", "K_sc_z", "anisotropy_z_over_x",
                "specific_surface_per_mm", "D_eff_z")

TPMS_GRID, FDM_GRID = 32, 44               # production grids, from sim/generate_all.py


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------

def config_key(family, topology, mode, n_cells=1, n_struts=3, n_layers=12):
    """The catalogue's identity for an architecture. Must match pipeline/inverse.py."""
    if family == "TPMS":
        return f"TPMS|{topology}|{mode}|c{int(n_cells)}"
    return f"FDM|{mode}|s{int(n_struts)}|l{int(n_layers)}"


def list_configs():
    """Every architecture the catalogue can predict without a fresh solve."""
    cat = inverse.build_catalogue(D.load_simulated())
    rows = (cat.groupby("config")
               .agg(rows=("porosity", "size"), P_min=("porosity", "min"),
                    P_max=("porosity", "max"), n=("ga_n", "first"), C=("ga_C", "first"),
                    fit_r2=("ga_fit_r2", "first"))
               .round(3).sort_index())
    return rows


# --------------------------------------------------------------------------
# Geometry resolution
# --------------------------------------------------------------------------

def _from_catalogue(family, topology, mode, porosity, n_cells, n_struts, n_layers):
    key = config_key(family, topology, mode, n_cells, n_struts, n_layers)
    cat = inverse.build_catalogue(D.load_simulated())
    g = cat[cat.config == key].sort_values("porosity")
    if g.empty:
        raise KeyError(
            f"no simulated data for '{key}'. Run with --solve to mesh it now, or "
            f"--list-configs to see what the sweep covered.")

    P_min, P_max = float(g.porosity.min()), float(g.porosity.max())
    inside = P_min <= porosity <= P_max

    arch = {c: float(np.interp(porosity, g.porosity, g[c])) for c in _INTERP_COLS
            if c in g.columns}
    arch["porosity"] = porosity
    arch["ga_n"] = float(g.ga_n.iloc[0])
    arch["ga_C"] = float(g.ga_C.iloc[0])

    # Nearest sampled geometry decides connectivity: it is a topological yes/no, and
    # interpolating between "open" and "closed" would invent a state neither row is in.
    nearest = g.iloc[(g.porosity - porosity).abs().argmin()]
    arch["pore_connected"] = int(nearest.pore_connected)

    prov = dict(source="catalogue", config=key, n_sampled=int(len(g)),
                P_window=[round(P_min, 3), round(P_max, 3)],
                ga_fit_r2=round(float(g.ga_fit_r2.iloc[0]), 4),
                extrapolating=not inside)
    if inside:
        # Anchor on the interpolated FEM modulus (see module docstring).
        arch["E_rel_measured"] = float(np.interp(porosity, g.porosity, g.E_rel_z))
        prov["modulus_from"] = "interpolated FEM E_rel at this porosity"
    else:
        arch["E_rel_measured"] = None
        prov["modulus_from"] = "Gibson-Ashby fit, EXTRAPOLATED beyond the sampled window"
    return arch, prov


def _fdm_mask_at_porosity(porosity, grid, n_struts, n_layers, stagger, tol=3e-3):
    """
    Bisect filament DIAMETER at fixed in-plane spacing 1/n_struts.

    This is the parameterisation the production sweep uses, and the only one under which
    n_struts keeps the meaning the config key claims for it. tpms_fem.build_fdm_at_porosity
    does the opposite - fixes the diameter and bisects the spacing - so using it here would
    solve some other strut count and then label the answer 's3'.

    Porosity falls as the filament thickens, so the reachable window is bounded at both
    ends: by the thinnest filament that still fuses to the layer below (h = 0.8*d), and by
    the diameter at which neighbouring filaments merge into a solid slab.
    """
    from tpms_fem import fdm_lattice                                # noqa: E402

    spacing = 1.0 / n_struts
    d_thin = (1.0 / n_layers) / 0.8          # thinner and the layers stop bonding
    d_thick = 1.4 / n_struts                 # thicker and the lay-down is a slab
    if d_thin * grid < 3.0:
        raise ValueError(
            f"filament is {d_thin*grid:.1f} voxels across at grid {grid}; under 3 it is "
            f"not resolved. Raise --grid or reduce --layers.")

    def porosity_at(d):
        return 1.0 - float(fdm_lattice(grid, d, spacing, 0.8, stagger).mean())

    P_hi, P_lo = porosity_at(d_thin), porosity_at(d_thick)
    if not (P_lo - tol) <= porosity <= (P_hi + tol):
        raise ValueError(
            f"s{n_struts}/l{n_layers} can only reach porosity {P_lo:.3f}-{P_hi:.3f} "
            f"(you asked for {porosity:.3f}). Filament diameter is bounded below by layer "
            f"fusion and above by filament merging. Change --struts or --layers.")

    lo, hi = d_thin, d_thick
    mask = None
    for _ in range(40):
        d = 0.5 * (lo + hi)
        mask = fdm_lattice(grid, d, spacing, 0.8, stagger)
        P = 1.0 - float(mask.mean())
        if abs(P - porosity) < tol:
            break
        if P > porosity:
            lo = d                            # too porous -> thicken the filament
        else:
            hi = d
    return mask


def _from_solve(family, topology, mode, porosity, n_cells, n_struts, n_layers, grid=None):
    """Mesh this exact geometry and run the full physics stack on it now."""
    sys.path.insert(0, str(ROOT / "sim"))
    from tpms_fem import build_at_porosity                          # noqa: E402
    from generate_all import evaluate, DOMAIN_MM                    # noqa: E402

    if family == "TPMS":
        grid = grid or TPMS_GRID
        mask = build_at_porosity(topology, mode, porosity, n=grid, cells=n_cells)
    else:
        grid = grid or FDM_GRID
        mask = _fdm_mask_at_porosity(porosity, grid, n_struts, n_layers,
                                     stagger=(mode == "staggered"))

    row = evaluate(mask, grid, DOMAIN_MM)
    actual = float(row["porosity"])
    if abs(actual - porosity) > 0.05:
        raise ValueError(
            f"the bisection reached porosity {actual:.3f}, not {porosity:.3f} - this "
            f"geometry cannot express that porosity on a {grid}^3 grid.")
    if not row["cg_converged"]:
        raise RuntimeError("the FEM solve did not converge; the result is not usable.")

    # (n, C) still come from the architecture's fitted series where one exists, because a
    # single solve cannot give a density exponent. Falls back to the textbook n = 2.
    try:
        _, cat_prov = _from_catalogue(family, topology, mode, actual,
                                      n_cells, n_struts, n_layers)
        cat = inverse.build_catalogue(D.load_simulated())
        g = cat[cat.config == cat_prov["config"]]
        ga_n, ga_C, fit_r2 = float(g.ga_n.iloc[0]), float(g.ga_C.iloc[0]), cat_prov["ga_fit_r2"]
        exponent_from = f"fitted for {cat_prov['config']}"
    except KeyError:
        ga_n, ga_C, fit_r2 = 2.0, 1.0, None
        exponent_from = "textbook n = 2 (no fitted series for this architecture)"

    arch = {c: float(row[c]) for c in _INTERP_COLS if c in row}
    arch.update(porosity=actual, ga_n=ga_n, ga_C=ga_C,
                pore_connected=int(row["pore_connected"]),
                E_rel_measured=float(row["E_rel_z"]))
    prov = dict(source=f"voxel FEM, grid {grid}^3", config=config_key(
                    family, topology, mode, n_cells, n_struts, n_layers),
                requested_porosity=porosity, achieved_porosity=round(actual, 4),
                anisotropy_z_over_x=row.get("anisotropy_z_over_x"),
                ga_fit_r2=fit_r2, exponent_from=exponent_from, extrapolating=False,
                modulus_from="this solve's own E_rel")
    return arch, prov


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict(topology="gyroid", mode="network", porosity=0.65, polymer="PLGA_85_15",
            ceramic="none", ceramic_wt=0.0, site="trabecular_mid", family=None,
            n_cells=1, n_struts=3, n_layers=12, solve=False, grid=None,
            include_trajectory=False):
    """
    Predicted properties and a feasibility verdict for one scaffold specification.

    Returns a dict with `geometry`, `mechanics`, `degradation`, `verdict` and
    `provenance`. Nothing is rounded away: `provenance` always states where the modulus
    came from and whether the request sat inside the data.
    """
    family = family or ("FDM" if topology in ("fdm", "fdm_strut") else "TPMS")
    if family == "FDM" and mode not in ("aligned", "staggered"):
        mode = "aligned"
    if polymer not in phys.POLYMERS:
        raise KeyError(f"unknown polymer '{polymer}'. Known: {sorted(phys.POLYMERS)}")
    if ceramic not in phys.CERAMICS:
        raise KeyError(f"unknown ceramic '{ceramic}'. Known: {sorted(phys.CERAMICS)}")
    if site not in phys.BONE_SITES:
        raise KeyError(f"unknown site '{site}'. Known: {sorted(phys.BONE_SITES)}")
    if ceramic == "none" and ceramic_wt > 0:
        raise ValueError("ceramic_wt > 0 with ceramic='none' - pick a ceramic.")
    if not 0.0 <= ceramic_wt < 1.0:
        raise ValueError(
            f"--wt takes a FRACTION in [0, 1), got {ceramic_wt}"
            + (f" - did you mean {ceramic_wt/100:g}?" if ceramic_wt >= 1.0 else ""))
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"--porosity takes a fraction in (0, 1), got {porosity}")

    resolve = _from_solve if solve else _from_catalogue
    kwargs = dict(grid=grid) if solve else {}
    arch, prov = resolve(family, topology, mode, porosity, n_cells,
                         n_struts, n_layers, **kwargs)

    # Anchoring happens in exactly one place - inverse._anchored_C - so this interface and
    # the inverse search cannot drift into two opinions about the same scaffold. Passing
    # E_rel_z is what opts in; without it the fitted C is used, which is the correct
    # behaviour when the porosity was never solved.
    prov["ga_n"] = round(arch["ga_n"], 4)
    prov["ga_C_fitted"] = round(arch["ga_C"], 4)
    row = pd.Series({**arch, "family": family, "topology": topology, "mode": mode,
                     "config": prov["config"], "E_rel_z": arch.get("E_rel_measured")})
    if arch.get("E_rel_measured") is not None:
        prov["ga_C_anchored"] = round(inverse._anchored_C(row), 4)
        prov["E_rel_measured"] = round(arch["E_rel_measured"], 6)
    res = inverse.evaluate_candidate(row, polymer, ceramic, ceramic_wt, site)

    # Same hard geometric constraints the inverse search applies, so a spec accepted here
    # is one that search would also have kept. One row in, one verdict out.
    frame = pd.DataFrame([{k: row.get(k) for k in
                           ("pore_connected", "porosity", "pore_size_um",
                            "strut_thickness_um")}])
    kept, counts = inverse.geometric_filter(frame)
    geometry_ok = len(kept) == 1
    geometry_fail = [why for why, n in counts.items() if n]

    req = phys.BONE_SITES[site]
    checks = {
        "geometry printable and ingrowth-capable": geometry_ok,
        f"stiffness within 0.5-2x native ({req['E_native']:.0f} MPa)": res["meets_stiffness"],
        f"strength >= {req['sigma_req']:.1f} MPa": res["meets_strength"],
        "local pH stays >= 6.5": res["pH_safe"],
    }
    trajectory = None
    if include_trajectory:
        # The week-by-week arrays behind the summary scalars, for plotting. Same call the
        # report figure uses, so a chart drawn from this cannot disagree with fig6.
        traj_row = pd.Series({**row.to_dict(), "polymer": polymer, "ceramic": ceramic,
                              "ceramic_wt": ceramic_wt, "site": site})
        t = inverse.trajectory_for(traj_row)
        trajectory = {k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in t.items()}

    return dict(
        trajectory=trajectory,
        # The integers that define the architecture belong in the specification, not only
        # encoded inside provenance.config. A caller reconstructing this prediction - the
        # web UI explaining a rejection, or a bookmarked URL being re-run - needs the cell
        # or strut count as a number, and parsing it back out of "TPMS|gyroid|network|c2"
        # is a format dependency nothing should take on.
        specification=dict(family=family, topology=topology, mode=mode,
                           porosity=round(arch["porosity"], 4), polymer=polymer,
                           ceramic=ceramic, ceramic_wt=ceramic_wt, site=site,
                           **(dict(n_cells=int(n_cells)) if family == "TPMS"
                              else dict(n_struts=int(n_struts), n_layers=int(n_layers)))),
        geometry={k: (round(v, 2) if isinstance(v, float) else v)
                  for k, v in arch.items()
                  if k not in ("E_rel_measured", "ga_n", "ga_C")},
        mechanics=dict(E0_MPa=res["E0_MPa"], sigma0_MPa=res["sigma0_MPa"],
                       E_12wk_pct_of_E0=res["E_12wk_pct"],
                       native_E_MPa=req["E_native"], required_sigma_MPa=req["sigma_req"]),
        degradation=dict(mass_12wk_pct=res["mass_12wk_pct"],
                         polymer_resorbed_week=res["polymer_resorbed_week"],
                         min_pH=res["min_pH"], porosity_final=res["porosity_final"]),
        verdict=dict(feasible=all(checks.values()), checks=checks,
                     geometry_rejections=geometry_fail,
                     load_transfer_cost=res["load_transfer_cost"]),
        provenance=prov)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _render(p):
    s, g, m, d, v, pr = (p["specification"], p["geometry"], p["mechanics"],
                         p["degradation"], p["verdict"], p["provenance"])
    w = 62
    # The canonical config key, not family+topology: for FDM the latter reads "FDM
    # fdm/aligned" and drops the strut and layer counts that define the geometry. 4 dp on
    # porosity to match what run_pipeline.py prints, so specs copy between them cleanly.
    L = ["", "=" * w,
         f"  {pr['config']}  P={s['porosity']:.4f}  ->  {s['site']}",
         f"  {s['polymer']}"
         + (f" + {s['ceramic']} {s['ceramic_wt']:.0%}" if s["ceramic"] != "none" else ""),
         "=" * w, "", "GEOMETRY",
         f"  pore size            {g.get('pore_size_um', float('nan')):8.0f} um",
         f"  strut thickness      {g.get('strut_thickness_um', float('nan')):8.0f} um",
         f"  stress concentration {g.get('K_sc_z', float('nan')):8.2f}",
         "", "MECHANICS"]
    sig = m["sigma0_MPa"]
    L += [f"  E0                   {m['E0_MPa']:8.1f} MPa   (native {m['native_E_MPa']:.0f})",
          f"  sigma0               {sig:8.1f} MPa   (need {m['required_sigma_MPa']:.1f})"
          if np.isfinite(sig) else "  sigma0                    n/a",
          f"  E at 12 weeks        {m['E_12wk_pct_of_E0']:8.1f} %  of E0",
          "", "DEGRADATION",
          f"  mass at 12 weeks     {d['mass_12wk_pct']:8.1f} %",
          f"  polymer resorbed     {d['polymer_resorbed_week']:8.0f} wk"
          if np.isfinite(d["polymer_resorbed_week"]) else
          "  polymer resorbed         >52 wk",
          f"  minimum local pH     {d['min_pH']:8.2f}",
          f"  porosity at 52 wk    {d['porosity_final']:8.3f}",
          "", "VERDICT"]
    for label, ok in v["checks"].items():
        L.append(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    for why in v["geometry_rejections"]:
        L.append(f"          reason: {why}")
    L += ["", f"  {'FEASIBLE' if v['feasible'] else 'NOT FEASIBLE'}"
              f"   load-transfer cost = {v['load_transfer_cost']:.4f}",
          "", "PROVENANCE",
          f"  modulus from   {pr['modulus_from']}",
          f"  geometry from  {pr['source']}"]
    if pr.get("P_window"):
        L.append(f"  fitted window  P = {pr['P_window'][0]} - {pr['P_window'][1]}"
                 f"  (fit R2 = {pr.get('ga_fit_r2')})")
    if pr.get("exponent_from"):
        L.append(f"  exponent       {pr['exponent_from']}")
    if pr.get("extrapolating"):
        L += ["", "  *** WARNING: the requested porosity is outside the range this",
              "      architecture was ever simulated at. The power law is being",
              "      extrapolated and the result is not backed by a solve. Re-run",
              "      with --solve to mesh this geometry properly. ***"]
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description="Predict the behaviour of one biopolymer bone scaffold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  .venv/bin/python tools/predict.py --topology gyroid "
               "--mode network \\\n      --porosity 0.68 --polymer PLGA_85_15 "
               "--ceramic beta_TCP --wt 0.30")
    ap.add_argument("--topology", default="gyroid",
                    help="any of sim.tpms_fem.TOPOLOGIES (gyroid, diamond, schwarzP, iwp, "
                         "fischerKochS, neovius, lidinoid, splitP), or fdm for a "
                         "printed lattice")
    ap.add_argument("--mode", default="network",
                    help="TPMS: network|sheet · FDM: aligned|staggered")
    ap.add_argument("--porosity", type=float, default=0.65)
    ap.add_argument("--cells", type=int, default=1, help="TPMS unit cells per side")
    ap.add_argument("--struts", type=int, default=3, help="FDM filaments per side")
    ap.add_argument("--layers", type=int, default=12, help="FDM layers")
    ap.add_argument("--polymer", default="PLGA_85_15", choices=sorted(phys.POLYMERS))
    ap.add_argument("--ceramic", default="none", choices=sorted(phys.CERAMICS))
    ap.add_argument("--wt", type=float, default=0.0, help="ceramic weight fraction, 0-1")
    ap.add_argument("--site", default="trabecular_mid", choices=sorted(phys.BONE_SITES))
    ap.add_argument("--solve", action="store_true",
                    help="mesh and solve this geometry now instead of interpolating")
    ap.add_argument("--grid", type=int, default=None, help="voxel grid for --solve")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--list-configs", action="store_true",
                    help="architectures predictable without a solve, and their windows")
    a = ap.parse_args()

    if a.list_configs:
        print(list_configs().to_string())
        return 0
    try:
        p = predict(topology=a.topology, mode=a.mode, porosity=a.porosity,
                    polymer=a.polymer, ceramic=a.ceramic, ceramic_wt=a.wt, site=a.site,
                    n_cells=a.cells, n_struts=a.struts, n_layers=a.layers,
                    solve=a.solve, grid=a.grid)
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(C.dump_json(p, indent=2) if a.json else _render(p))
    return 0 if p["verdict"]["feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
