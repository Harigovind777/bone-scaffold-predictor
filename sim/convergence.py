"""
Mesh convergence study - and the paired low/high-fidelity table the fusion stage needs.

Two jobs from one sweep:

1. CONVERGENCE. data/README.md lists "20^3 voxel grid: modulus is mesh-sensitive" as an
   open limitation. This quantifies it: the same geometry at 16/20/24/32/44 voxels, with
   the deviation of each from the finest grid. That turns a caveat into a number.

2. MULTI-FIDELITY PAIRS. The framework's headline method is co-kriging - fuse a cheap,
   biased, plentiful source with an expensive, accurate, scarce one. Tier 3 (literature)
   is the eventual "high" source, but it does not exist yet. A coarse grid is cheap and
   biased in exactly the same way, so grid 16 -> grid 44 lets the fusion code be built,
   tested and validated NOW, on data where ground truth is known. When curated rows
   arrive, only the loader changes.

Run:  .venv/bin/python sim/convergence.py
"""

import os

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
from tpms_fem import (build_at_porosity, fdm_lattice, compression_response,
                      effective_diffusivity, specific_surface, morphometry)

GRIDS = (16, 20, 24, 32, 44)
DOMAIN_MM = 2.0

TPMS_CASES = [(t, m, c, p)
              for t, m, c, p in itertools.product(
                  ("gyroid", "diamond", "schwarzP", "iwp"), ("network", "sheet"),
                  (1, 2), (0.40, 0.60, 0.75))]

# Strut/layer combinations drawn from the production sweep's range. Note these do NOT all
# resolve across the whole ladder: the thinnest fusing filament here is 1.4375/n_layers of
# the domain, so nothing survives the 3-voxel floor at grid 16 and only n_layers=8 does at
# grid 20. That is a property of the geometry, not a bug - evaluate_case returns None and
# the FDM curve simply starts at the first grid that resolves it. run_pipeline picks the
# low-fidelity grid accordingly rather than assuming the coarsest one has both families.
FDM_CASES = [(s, l, stag) for s, l, stag in itertools.product(
    (2, 3, 4), (8, 10), (False, True))]


def evaluate_case(spec):
    kind, params, grid = spec
    try:
        if kind == "TPMS":
            topo, mode, cells, P = params
            mask = build_at_porosity(topo, mode, P, n=grid, cells=cells)
            key = f"TPMS|{topo}|{mode}|c{cells}|P{P:.2f}"
            meta = dict(family="TPMS", topology=topo, mode=mode, n_cells=cells,
                        target_porosity=P, n_struts=np.nan, n_layers=np.nan,
                        strut_d=np.nan, stagger=np.nan)
        else:
            n_struts, n_layers, stag = params
            d = (1.0 / n_layers) / 0.8 * 1.15
            if d * grid < 3.0:
                return None
            mask = fdm_lattice(grid, d, 1.0 / n_struts, layer_ratio=0.8, stagger=stag)
            key = f"FDM|s{n_struts}|l{n_layers}|{'stag' if stag else 'align'}"
            meta = dict(family="FDM", topology="fdm_strut",
                        mode="staggered" if stag else "aligned", n_cells=np.nan,
                        target_porosity=np.nan, n_struts=n_struts, n_layers=n_layers,
                        strut_d=round(d, 4), stagger=int(stag))

        P_actual = 1.0 - float(mask.mean())
        if mask.sum() < 30 or not (0.05 < P_actual < 0.97):
            return None
        rz = compression_response(mask, axis=2)
        rx = compression_response(mask, axis=0)
        d_z = effective_diffusivity(mask, axis=2)
        return dict(case=key, grid=grid, porosity=round(P_actual, 4),
                    relative_density=round(1 - P_actual, 4),
                    E_rel_z=round(rz["E_rel"], 6), E_rel_x=round(rx["E_rel"], 6),
                    K_sc_z=round(rz["stress_concentration"], 3),
                    D_eff_z=round(float(d_z), 6),
                    specific_surface_per_mm=round(specific_surface(mask) / DOMAIN_MM, 4),
                    **morphometry(mask, DOMAIN_MM), **meta)
    except Exception as exc:
        print(f"  skip {spec[:2]} grid={spec[2]}: {exc}", flush=True)
        return None


def main():
    specs = ([("TPMS", c, g) for c in TPMS_CASES for g in GRIDS]
             + [("FDM", c, g) for c in FDM_CASES for g in GRIDS])
    workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    print(f"{len(specs)} solves over grids {GRIDS} | {workers} workers", flush=True)

    t0 = time.time()
    rows = []
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(evaluate_case, specs, chunksize=1), 1):
            if res is not None:
                rows.append(res)
            if i % 40 == 0:
                print(f"  [{i}/{len(specs)}] {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["case", "grid"]).reset_index(drop=True)
    df.to_csv("data/mesh_convergence.csv", index=False)
    print(f"\nwrote data/mesh_convergence.csv  {df.shape}  in {time.time()-t0:.0f}s")

    # ---- convergence: deviation of each grid from the finest available for that case ----
    finest = df.sort_values("grid").groupby("case").last()
    dev = df.join(finest[["E_rel_z"]].rename(columns={"E_rel_z": "E_ref"}), on="case")
    dev["pct_err"] = 100 * (dev.E_rel_z - dev.E_ref).abs() / dev.E_ref
    summary = dev.groupby(["family", "grid"])["pct_err"].agg(
        median="median", p90=lambda s: s.quantile(0.90), worst="max").round(2)
    print("\nModulus deviation from the finest grid, % (per family):")
    print(summary.to_string())

    # Pairing needs a case at the finest grid AND at some coarser one - not at EVERY grid,
    # which no FDM case manages (see the note on FDM_CASES).
    at_finest = set(df[df.grid == max(GRIDS)].case)
    per_grid = {int(g): len(at_finest & set(df[df.grid == g].case))
                for g in sorted(df.grid.unique()) if g < max(GRIDS)}
    print(f"\n{len(at_finest)} cases resolved at the finest grid ({max(GRIDS)})")
    print(f"paired against each coarser grid: {per_grid}")


if __name__ == "__main__":
    main()
