"""Sweep TPMS architectures x porosity and write the simulated scaffold dataset."""

import sys, time, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, "sim")
from tpms_fem import (TOPOLOGIES, build_at_porosity, apparent_modulus,
                      effective_diffusivity, specific_surface,
                      largest_connected_fraction)

GRID = 20
POROSITIES = np.round(np.arange(0.30, 0.881, 0.05), 3)
MODES = ("network", "sheet")
CELLS = (1, 2)

# Unit-cell physical size, used to convert dimensionless results into real pore sizes.
CELL_SIZE_MM = {1: 2.0, 2: 1.0}


def run():
    rows, t0 = [], time.time()
    combos = list(itertools.product(TOPOLOGIES, MODES, CELLS, POROSITIES))
    for i, (topo, mode, cells, P) in enumerate(combos, 1):
        try:
            mask = build_at_porosity(topo, mode, P, n=GRID, cells=cells)
            actual_P = 1.0 - float(mask.mean())
            if mask.sum() < 50 or actual_P < 0.05 or actual_P > 0.97:
                continue
            conn = largest_connected_fraction(mask)
            E_rel, info = apparent_modulus(mask)
            D_eff = effective_diffusivity(mask)
            sav = specific_surface(mask)

            cell_mm = CELL_SIZE_MM[cells]
            rows.append(dict(
                topology=topo, mode=mode, n_cells=cells, grid=GRID,
                unit_cell_mm=cell_mm,
                target_porosity=P, porosity=round(actual_P, 4),
                relative_density=round(1 - actual_P, 4),
                E_rel=round(float(E_rel), 6),
                D_eff_rel=round(float(D_eff), 6),
                tortuosity=round(actual_P / D_eff, 4) if D_eff > 1e-9 else np.nan,
                specific_surface_per_mm=round(sav / cell_mm, 4),
                solid_connectivity=round(conn, 4),
                # nominal pore size: cell size scaled by porosity, in microns
                nominal_pore_um=round(cell_mm * 1000 * actual_P / cells, 1),
                cg_converged=int(info == 0),
            ))
            if i % 10 == 0:
                print(f"[{i}/{len(combos)}] {time.time()-t0:.0f}s  last: "
                      f"{topo}/{mode} P={actual_P:.2f} E_rel={E_rel:.4f}", flush=True)
        except Exception as e:                       # keep the sweep alive
            print(f"  skip {topo}/{mode}/{cells}/P={P}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("data/tpms_simulated.csv", index=False)
    print(f"\nwrote data/tpms_simulated.csv  shape={df.shape}  in {time.time()-t0:.0f}s")

    # Fit Gibson-Ashby parameters per architecture - these become ML targets.
    fits = []
    for (topo, mode), g in df.groupby(["topology", "mode"]):
        g = g[(g.E_rel > 1e-5) & (g.relative_density > 0.02)]
        if len(g) < 4:
            continue
        n_exp, lnC = np.polyfit(np.log(g.relative_density), np.log(g.E_rel), 1)
        pred = np.exp(lnC) * g.relative_density ** n_exp
        r2 = 1 - ((g.E_rel - pred) ** 2).sum() / ((g.E_rel - g.E_rel.mean()) ** 2).sum()
        fits.append(dict(topology=topo, mode=mode, n_points=len(g),
                         GA_exponent_n=round(n_exp, 3), GA_prefactor_C=round(float(np.exp(lnC)), 3),
                         fit_r2=round(float(r2), 4)))
    fdf = pd.DataFrame(fits).sort_values("GA_exponent_n")
    fdf.to_csv("data/gibson_ashby_fits.csv", index=False)
    print("\nGibson-Ashby parameters per architecture:")
    print(fdf.to_string(index=False))


if __name__ == "__main__":
    run()
