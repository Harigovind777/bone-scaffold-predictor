"""
FDM strut-lattice sweep.

Forward-samples (struts per layer, layers, filament diameter) rather than bisecting to a
target porosity: the strut/layer counts are integers, so porosity is quantised and
bisection can only reach a couple of discrete values. Sampling diameter continuously
above the layer-fusion limit gives smooth porosity coverage instead.

Fusion constraint: layer height h = 1/n_layers must satisfy h <= 0.8*d, i.e.
d >= 1/(0.8*n_layers). Below that the layers do not bond and the lattice has no
through-thickness stiffness.
"""

import sys, time, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, "sim")
from tpms_fem import (fdm_lattice, apparent_modulus, effective_diffusivity,
                      specific_surface, largest_connected_fraction)

# Bone scaffolds live at 60-85% porosity. Reaching that with fused layers needs WIDE
# strut spacing (few struts) and THIN filaments (many layers) - and a thin filament must
# still span several voxels, so the grid has to be fine enough to resolve it.
# d_min = 1/(0.8*n_layers); at n_layers=12 that is 0.104, i.e. ~3.3 voxels at GRID=32.
GRID = 32
POROSITY_WINDOW = (0.35, 0.90)                # discard geometry outside the useful range


def run():
    rows, t0 = [], time.time()
    combos = list(itertools.product((2, 3), (8, 10, 12), (False, True)))
    for n_struts, n_layers, stag in combos:
        h = 1.0 / n_layers
        d_min = h / 0.8                       # thinnest filament that still fuses layers
        for d in np.linspace(d_min, 1.6 * d_min, 7):
            if d >= 1.4 / n_struts:           # struts fully merged in-plane
                continue
            if d * GRID < 3.0:                # filament thinner than 3 voxels: unresolved
                continue
            mask = fdm_lattice(GRID, d, 1.0 / n_struts, layer_ratio=0.8, stagger=stag)
            P = 1.0 - mask.mean()
            if mask.sum() < 50 or not (POROSITY_WINDOW[0] <= P <= POROSITY_WINDOW[1]):
                continue
            E, info = apparent_modulus(mask)
            D = effective_diffusivity(mask)
            rows.append(dict(
                topology="fdm_strut", mode="staggered" if stag else "aligned",
                n_struts=n_struts, n_layers=n_layers, strut_d=round(float(d), 4),
                porosity=round(P, 4), relative_density=round(1 - P, 4),
                E_rel=round(float(E), 6), D_eff_rel=round(float(D), 6),
                tortuosity=round(P / D, 3) if D > 5e-3 else np.nan,
                specific_surface_per_mm=round(specific_surface(mask), 4),
                solid_connectivity=round(largest_connected_fraction(mask), 4),
                pore_connected=int(D > 5e-3), cg_converged=int(info == 0)))
            print(f"  s={n_struts} l={n_layers} d={d:.3f} P={P:.3f} E={E:.5f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("data/fdm_simulated.csv", index=False)
    print(f"\n{len(df)} rows in {time.time()-t0:.0f}s | "
          f"converged {df.cg_converged.sum()}/{len(df)} | "
          f"dead {(df.E_rel < 1e-5).sum()}")

    print("\nGibson-Ashby fits (fused-layer FDM):")
    for mo, g in df.groupby("mode"):
        g = g[(g.E_rel > 1e-5) & (g.relative_density > 0.02)]
        if len(g) < 4:
            continue
        n, lnC = np.polyfit(np.log(g.relative_density), np.log(g.E_rel), 1)
        pred = np.exp(lnC) * g.relative_density ** n
        r2 = 1 - ((g.E_rel - pred) ** 2).sum() / ((g.E_rel - g.E_rel.mean()) ** 2).sum()
        print(f"  {mo:10s} n={n:.3f} C={np.exp(lnC):.3f} R2={r2:.4f} "
              f"({len(g)} pts, P={g.porosity.min():.2f}-{g.porosity.max():.2f})")


if __name__ == "__main__":
    run()
