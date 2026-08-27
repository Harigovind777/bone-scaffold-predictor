"""
Unified simulated-tier generator: TPMS + FDM, one schema, run in parallel.

Supersedes generate_dataset.py and generate_fdm.py, which each wrote their own columns
and could not be concatenated. Everything downstream reads ONE table, so an architecture
descriptor means the same thing whichever family produced it.

What each sample now carries beyond the earlier sweeps:
  * E_rel along TWO axes -> anisotropy ratio. A 0/90 print is not isotropic, and the
    compression axis is exactly the thing papers forget to state.
  * a stress-concentration factor -> a strength proxy from a linear-elastic solve.
  * micro-CT-comparable pore size and strut thickness from a distance transform, rather
    than a nominal pore size back-computed from the unit cell.
  * effective diffusivity along BOTH axes, hence a transport anisotropy ratio. This was
    nominally present before but was not being measured: effective_diffusivity looped over
    the same name as its own `axis` argument, so D_eff_x came back as an exact copy of
    D_eff_z on every row.

Run:  .venv/bin/python sim/generate_all.py [--quick]
"""

import os

# Must precede numpy: each worker gets one core, otherwise BLAS oversubscribes and the
# pool runs slower than serial.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpms_fem import (TOPOLOGIES, build_at_porosity, fdm_lattice,
                      compression_response_axes, effective_diffusivity, specific_surface,
                      largest_connected_fraction, morphometry)

# Physical size of the simulated cube. Every length in the output is scaled by this, so
# it is the one number to change if you want to model a different specimen size.
DOMAIN_MM = 2.0

TPMS_GRID = 32
FDM_GRID = 44          # thin filaments at 16 layers need ~3.4 voxels across; 44 gives that
MODES = ("network", "sheet")
CELLS = (1, 2)

# The sweep was widened once the compression solve got ~4x cheaper (see
# sim/benchmark_solver.py). Grids are unchanged, deliberately - they are what
# sim/convergence.py characterises, and moving them would invalidate that study - so the
# extra budget went into the two axes that actually buy something:
#
#   ARCHITECTURES. Eight topologies rather than four. Leave-one-architecture-out is the
#   headline protocol and every architecture is one fold, so this is the only lever that
#   improves the honest score's precision rather than just its sample size.
#
#   POROSITY DENSITY. 45 levels rather than 30, out to 0.92. A Gibson-Ashby exponent is
#   fitted per architecture from these points alone; the literature cannot resolve it
#   because published scaffolds cluster at two or three porosities per paper, so
#   resolution along this axis is the simulated tier's whole reason to exist.
TPMS_POROSITIES = np.round(np.linspace(0.20, 0.92, 45), 4)

FDM_STRUTS = (2, 3, 4, 5, 6)
FDM_LAYERS = (8, 10, 12, 14, 16)
FDM_STAGGER = (False, True)
FDM_DIAMETER_STEPS = 12
POROSITY_WINDOW = (0.30, 0.92)


# --------------------------------------------------------------------------
# One sample = one geometry evaluated on every property
# --------------------------------------------------------------------------

def evaluate(mask, grid, domain_mm=DOMAIN_MM):
    """Run the full physics stack on a voxel mask. Shared by both families."""
    # One assembly, both loading axes: the global stiffness does not depend on which
    # face is pushed, and rebuilding it per axis was pure repetition.
    r = compression_response_axes(mask, (2, 0))
    rz, rx = r[2], r[0]
    d_z = effective_diffusivity(mask, axis=2)
    d_x = effective_diffusivity(mask, axis=0)
    porosity = 1.0 - float(mask.mean())
    morph = morphometry(mask, domain_mm)

    # Below ~5e-3 the Laplace solve is riding the ersatz conductivity, not a real pore
    # path: the porosity is closed and tortuosity is meaningless rather than merely large.
    connected = d_z > 5e-3

    return dict(
        porosity=round(porosity, 4),
        relative_density=round(1.0 - porosity, 4),
        E_rel_z=round(rz["E_rel"], 6),
        E_rel_x=round(rx["E_rel"], 6),
        anisotropy_z_over_x=round(rz["E_rel"] / rx["E_rel"], 4) if rx["E_rel"] > 1e-9 else np.nan,
        K_sc_z=round(rz["stress_concentration"], 3),
        K_sc_x=round(rx["stress_concentration"], 3),
        D_eff_z=round(float(d_z), 6),
        D_eff_x=round(float(d_x), 6),
        # Transport anisotropy, the counterpart of anisotropy_z_over_x. Until the axis
        # argument of effective_diffusivity was honoured this column was a copy of D_eff_z
        # on all 747 rows, so a 0/90 lay-down - which is plainly not isotropic in
        # transport either - reported a ratio of exactly 1.
        D_anisotropy_z_over_x=round(float(d_z / d_x), 4) if d_x > 1e-9 else np.nan,
        tortuosity_z=round(porosity / d_z, 4) if connected else np.nan,
        specific_surface_per_mm=round(specific_surface(mask) / domain_mm, 4),
        solid_connectivity=round(largest_connected_fraction(mask), 4),
        pore_connected=int(connected),
        cg_converged=int(rz["cg_info"] == 0 and rx["cg_info"] == 0),
        **morph,
    )


def run_tpms(spec):
    topo, mode, cells, target_P, grid = spec
    try:
        mask = build_at_porosity(topo, mode, target_P, n=grid, cells=cells)
        actual_P = 1.0 - float(mask.mean())
        if mask.sum() < 50 or not (0.05 < actual_P < 0.97):
            return None
        # A bisection that lands far from its target means the level set cannot express
        # that porosity on this grid; keeping the row would put a mislabelled design
        # point in the training set.
        if abs(actual_P - target_P) > 0.05:
            return None
        cell_mm = DOMAIN_MM / cells
        return dict(
            family="TPMS", topology=topo, mode=mode, grid=grid,
            n_cells=cells, unit_cell_mm=cell_mm,
            strut_d=np.nan, strut_spacing=np.nan, n_struts=np.nan,
            n_layers=np.nan, stagger=np.nan,
            target_porosity=round(float(target_P), 4),
            **evaluate(mask, grid),
        )
    except Exception as exc:
        print(f"  skip TPMS {topo}/{mode}/{cells}/P={target_P}: {exc}", flush=True)
        return None


def run_fdm(spec):
    n_struts, n_layers, stagger, d, grid = spec
    try:
        spacing = 1.0 / n_struts
        mask = fdm_lattice(grid, d, spacing, layer_ratio=0.8, stagger=stagger)
        P = 1.0 - float(mask.mean())
        if mask.sum() < 50 or not (POROSITY_WINDOW[0] <= P <= POROSITY_WINDOW[1]):
            return None
        return dict(
            family="FDM", topology="fdm_strut",
            mode="staggered" if stagger else "aligned", grid=grid,
            n_cells=np.nan, unit_cell_mm=DOMAIN_MM * spacing,
            strut_d=round(float(d), 4), strut_spacing=round(spacing, 4),
            n_struts=n_struts, n_layers=n_layers, stagger=int(stagger),
            target_porosity=np.nan,
            **evaluate(mask, grid),
        )
    except Exception as exc:
        print(f"  skip FDM s={n_struts} l={n_layers} d={d:.3f}: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------
# Sweep definition
# --------------------------------------------------------------------------

def tpms_specs(grid, porosities):
    return [(t, m, c, float(p), grid)
            for t, m, c, p in itertools.product(TOPOLOGIES, MODES, CELLS, porosities)]


def fdm_specs(grid):
    """
    Forward-sample filament diameter rather than bisecting to a target porosity: strut
    and layer counts are integers, so porosity is quantised and bisection can only reach
    a few discrete values.

    The fusion constraint h = 1/n_layers <= 0.8*d sets the thinnest filament that still
    bonds to the layer below. Go under it and the print becomes a stack of unbonded rods
    with zero through-thickness stiffness - a modelling artefact, not a scaffold.
    """
    specs = []
    for n_struts, n_layers, stag in itertools.product(FDM_STRUTS, FDM_LAYERS, FDM_STAGGER):
        d_min = (1.0 / n_layers) / 0.8
        for d in np.linspace(d_min, 1.7 * d_min, FDM_DIAMETER_STEPS):
            if d >= 1.4 / n_struts:        # struts already merged in-plane: solid slab
                continue
            if d * grid < 3.0:             # filament under 3 voxels across: unresolved
                continue
            specs.append((n_struts, n_layers, stag, float(d), grid))
    return specs


def fit_gibson_ashby(df, group_cols, family_label):
    """
    Fit E_rel = C * rho_rel^n per architecture group, on the z (build-direction) modulus.

    C > 1.2 is flagged, not dropped: it is physically impossible for a cellular solid
    (E cannot exceed Es at full density), so it marks a LOCAL fit that must not be
    extrapolated toward rho_rel = 1. Those rows are still valid inside their sampled
    porosity window, which is where the ML actually uses them.
    """
    out = []
    for keys, g in df.groupby(group_cols, dropna=False):
        g = g[(g.E_rel_z > 1e-5) & (g.relative_density > 0.02)]
        if len(g) < 4:
            continue
        n_exp, lnC = np.polyfit(np.log(g.relative_density), np.log(g.E_rel_z), 1)
        pred = np.exp(lnC) * g.relative_density ** n_exp
        ss_res = ((g.E_rel_z - pred) ** 2).sum()
        ss_tot = ((g.E_rel_z - g.E_rel_z.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        C = float(np.exp(lnC))
        out.append(dict(family=family_label, **rec, pts=len(g),
                        P_min=round(float(g.porosity.min()), 3),
                        P_max=round(float(g.porosity.max()), 3),
                        n=round(float(n_exp), 3), C=round(C, 3),
                        r2=round(float(r2), 4), C_unphysical=int(C > 1.2)))
    return out


def main():
    quick = "--quick" in sys.argv
    workers = max(1, min(8, (os.cpu_count() or 4) - 2))

    porosities = TPMS_POROSITIES[::6] if quick else TPMS_POROSITIES
    t_specs = tpms_specs(20 if quick else TPMS_GRID, porosities)
    f_specs = fdm_specs(24 if quick else FDM_GRID)
    if quick:
        f_specs = f_specs[::6]

    print(f"TPMS: {len(t_specs)} specs | FDM: {len(f_specs)} specs | {workers} workers",
          flush=True)
    t0 = time.time()
    rows = []
    with mp.Pool(workers) as pool:
        for tag, fn, specs in (("TPMS", run_tpms, t_specs), ("FDM", run_fdm, f_specs)):
            done = 0
            for res in pool.imap_unordered(fn, specs, chunksize=1):
                done += 1
                if res is not None:
                    rows.append(res)
                if done % 25 == 0:
                    print(f"  [{tag} {done}/{len(specs)}] kept {len(rows)} "
                          f"| {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    order = ["family", "topology", "mode", "grid", "n_cells", "unit_cell_mm",
             "strut_d", "strut_spacing", "n_struts", "n_layers", "stagger",
             "target_porosity", "porosity", "relative_density",
             "E_rel_z", "E_rel_x", "anisotropy_z_over_x", "K_sc_z", "K_sc_x",
             "D_eff_z", "D_eff_x", "D_anisotropy_z_over_x", "tortuosity_z",
             "specific_surface_per_mm",
             "pore_size_um", "strut_thickness_um", "solid_connectivity",
             "pore_connected", "cg_converged"]
    df = df[order].sort_values(["family", "topology", "mode", "porosity"]).reset_index(drop=True)
    df.insert(0, "sample_id", [f"SIM{i:04d}" for i in range(len(df))])

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/simulated_all.csv", index=False)

    fits = (fit_gibson_ashby(df[df.family == "TPMS"], ["topology", "mode"], "TPMS")
            + fit_gibson_ashby(df[df.family == "FDM"],
                               ["mode", "n_struts", "n_layers"], "FDM"))
    fdf = pd.DataFrame(fits).sort_values(["family", "n"])
    fdf.to_csv("data/gibson_ashby_fits.csv", index=False)

    dt = time.time() - t0
    print(f"\nwrote data/simulated_all.csv  {df.shape}  in {dt:.0f}s")
    print(df.groupby("family").agg(
        rows=("sample_id", "size"),
        P_min=("porosity", "min"), P_max=("porosity", "max"),
        converged=("cg_converged", "sum"), connected=("pore_connected", "sum"),
    ).to_string())
    print(f"\nwrote data/gibson_ashby_fits.csv  {fdf.shape}  "
          f"({int(fdf.C_unphysical.sum())} local-only fits flagged)")


if __name__ == "__main__":
    main()
