# Dataset package

## The headline finding from the search

**There is no public dataset covering mechanical + degradation properties of biopolymer bone
scaffolds.** I checked the obvious candidates — Zenodo, Mendeley Data, figshare, Data in Brief,
NIMS PoLyInfo, the NIH/NIBIB bioink database, and the supplementary material of the recent ML
scaffold papers. What exists is either the wrong target variable (printability, cell response),
the wrong material class (hydrogel bioinks), or single-paper datasets of 20–60 rows.

That is not a dead end — it is the project's opening. **The curated dataset becomes your
contribution.** Structure it as three tiers.

---

## Tier 1 — MLATE (ready today, borrowed)

| | |
|---|---|
| Files | `mlate_full.csv` (1171 × 82), `mlate_bone_subset.csv` (199 × 82, 38 DOIs) |
| Source | Rafieyan et al., *Biofabrication* 16 (2024) 045014 — https://github.com/saeedrafieyan/MLATE |
| Targets | Printability (0–3), Cell Response (1–5), Scaffold Quality |
| Licence | Check the repo before redistributing; cite the paper. |

61 composition columns, 11 process columns, plus engineered features (`total_solids_pct`,
`ceramic_pct`, `ceramic_fraction`, `n_components`, `is_bone_relevant`).

**Use it for:** the printability/biological arm, and to have a real baseline running in week 1.
**Do not use it for:** mechanics or degradation — it contains neither. It is also hydrogel-heavy;
only 199 rows are bone-relevant, and those skew toward soft formulations.

### ⚠️ The leakage result — put this in your thesis

Predicting Printability with a random forest:

| Validation | Accuracy |
|---|---|
| Random 5-fold | **0.851** |
| GroupKFold by DOI | **0.549** |
| Majority-class baseline | 0.555 |

**Random splitting inflates accuracy by 30 points, and the honest score does not beat guessing the
majority class.** Rows from one publication share a material batch, a printer and an operator, so
random splitting puts near-duplicates in train and test. This is very likely happening in published
ML-scaffold papers reporting R² > 0.95 on small datasets.

Demonstrating this on someone else's benchmark, then building a framework that avoids it, is a
strong, defensible thesis contribution — and it costs you one figure.

---

## Tier 2 + 2b — simulated (generated here, unlimited)

| | |
|---|---|
| Files | `simulated_all.csv` (747 rows), `gibson_ashby_fits.csv` (43 fits), `mesh_convergence.csv` (282 rows) |
| Code | `sim/tpms_fem.py`, `sim/generate_all.py`, `sim/convergence.py` |
| Cost | 68 min for 747 samples on 8 workers (`--quick` gives 96 rows in ~30 s) |

Both families live in **one table with one schema** — `sim/generate_all.py` supersedes the earlier
`generate_dataset.py` / `generate_fdm.py`, which wrote separate columns that could not be
concatenated. The legacy `tpms_simulated.csv` and `fdm_simulated.csv` are kept only for provenance;
nothing downstream reads them.

Voxel finite-element pipeline: TPMS level-set → bisect to target porosity → uniaxial compression
FEM (symmetry rollers, prescribed strain) → Laplace solve on the pore phase → interface counting.
Every sample additionally carries a **second loading axis** (hence an anisotropy ratio), a
**stress-concentration factor** as a strength proxy, and µCT-comparable pore size and strut
thickness from a distance transform rather than a nominal unit-cell figure.

**Validation:** `tools/selftest.py` — 16 checks, all passing. Solid block returns E/Es = 1 on every
axis; fully open box transports at D_eff/D_bulk = 1; cubic TPMS is elastically isotropic and a 0/90
print is not. All 747 CG solves converged; 21 geometries have closed porosity and are excluded by
constraint downstream, not silently.

| Family | Rows | Architectures | Porosity range |
|---|---|---|---|
| TPMS | 469 | 8 | 0.164 – 0.915 |
| FDM | 278 | 2 lay-down modes | 0.302 – 0.860 |

### Why this tier matters — it resolves what literature cannot

Fitted Gibson–Ashby parameters, 8 TPMS architectures (median fit R² = 0.993, all ≥ 0.93):

| Topology | Mode | n | C | points | R² |
|---|---|---|---|---|---|
| iwp | network | 1.52 | 0.84 | 58 | 0.996 |
| diamond | sheet | 1.56 | 0.85 | 54 | 0.993 |
| gyroid | sheet | 1.79 | 0.81 | 60 | 0.992 |
| schwarzP | network | 1.98 | 1.13 | 51 | 0.983 |
| schwarzP | sheet | 2.03 | 0.96 | 55 | 0.994 |
| iwp | sheet | 2.08 | 1.26 | 60 | 0.934 |
| gyroid | network | 2.41 | 1.10 | 60 | 0.988 |
| diamond | network | 2.43 | 1.01 | 60 | 0.995 |

The exponent spans **1.52 → 2.43** — the stretch-dominated to bending-dominated range, matching
published values. Published scaffolds cluster at two or three porosities per architecture, so
literature data can never resolve this. Simulation can, densely and for free. One fit (iwp/sheet,
C = 1.26 > 1.2) is flagged `C_unphysical`: E cannot exceed Es at full density, so it is a *local*
fit valid inside its sampled window and must not be extrapolated toward ρ_rel = 1. It is flagged
rather than dropped — 16 of the 43 fits across both families carry that flag.

### Mesh convergence (`mesh_convergence.csv`)

The same geometries re-solved on a 16/20/24/32/44 ladder, which doubles as the low/high-fidelity
pairs for the co-kriging stage. Median deviation from the grid-44 answer: 13.2% at grid 16, 6.3% at
20, 3.7% at 32. Worst-case deviations are far larger (223% at grid 16) — coarse grids do not merely
add noise, they can lose the load path entirely.

Note filaments thinner than ~3 voxels are not meshed at all, so **no FDM case resolves at grid 16**
and only half do at grid 20. `run_pipeline.py` therefore picks the coarsest grid that covers *both*
families as its cheap tier rather than assuming the coarsest one does.

---

## Tier 2b — FDM strut lattices (within `simulated_all.csv`, 278 rows, grid 44³)

This is what the biopolymer bone-scaffold literature actually prints: parallel filaments rotated
90° each layer. Its load path differs fundamentally from a TPMS, so the simulated tier needs it to
be transferable to Tier 3. All 278 solves converged, none degenerate, porosity 0.30–0.86 — the
bone-relevant window.

**Modelling note (a trap worth knowing).** Layer height must be tied to filament diameter
(`h = 0.8·d`), because melt extrusion squashes each filament onto the layer below so layers always
bond. Treating layer count as an independent parameter is wrong: thinning the filament separates
the print into a stack of unbonded rods with *exactly zero* through-thickness stiffness. That
failure looks like a mesh-resolution problem and is not — it persisted unchanged at 20³, 28³ and 36³.

### ⚠️ The finding that reshapes the model design

**Relative density alone does not determine modulus.** At matched relative density:

| Comparison | Modulus spread |
|---|---|
| FDM, within one lay-down mode (varying strut count / layer count) | **1.1–2.5×** |
| FDM, aligned vs staggered lay-down | the dominant effect — n ≈ 1.5–2.0 vs 2.8–4.7 |
| TPMS, across the 8 architectures | **2.1–2.5×** |

So a single Gibson–Ashby (n, C) per architecture class is *insufficient* — which contradicts the
simplest reading of the framework. Two consequences:

1. **Stage 1 must predict n and C from the full architecture descriptor set** — filament diameter,
   strut spacing, layer count, lay-down offset — not from a topology label. Better still, treat
   Gibson–Ashby as a physics-informed *prior* and let the ML learn the geometry-dependent residual,
   rather than forcing the power-law form.
2. **Lay-down offset is a first-order design variable.** Aligned n ≈ 1.5–2.0; staggered n ≈ 2.8–4.7.
   Offsetting alternate layers roughly doubles the exponent. Most scaffold papers neither control
   nor report this.

Per-configuration fits are in `gibson_ashby_fits.csv` (20 rows). Note the `C_unphysical` flag:
8 fits have **C > 1.2**, which is impossible for a real cellular solid (E cannot exceed Es at
ρ_rel = 1). Those are *local* fits valid only across the sampled porosity window — **do not
extrapolate them toward full density.** The worst (staggered 2s/12l, R² = 0.49) shows the power law
barely holding at all for sparse staggered lattices.

### On validating against published values
I attempted to anchor against a PCL/HA paper reporting 46.6 → 71.6 MPa across 40→60% infill
(implying n ≈ 1.06). **That anchor is not usable:** two points cannot constrain an exponent, and
slicer *infill %* is not measured porosity — shells and over-extrusion break the correspondence.
Real validation must come from Tier 3 rows with directly measured porosity. Another reason
curation is the critical path.

**Proof it adds value** (predicting `E_rel`):

| Model | R² |
|---|---|
| Gibson–Ashby single global power law | 0.444 |
| Random forest, leave-one-architecture-out | **0.841** |
| Random forest, random 5-fold | 0.947 (optimistic) |

Learning topology-dependent n and C nearly doubles R² over textbook theory, and it holds up on
*architectures never seen in training*. That is the core thesis of the framework, demonstrated.

### A real finding already in the data
14 of 192 samples have **non-percolating (closed-cell) porosity** — mostly sheet-mode Schwarz-P at
P = 0.31–0.41. Those scaffolds are useless for bone ingrowth regardless of how good their modulus
looks. `pore_connected` flags them, and it should be a hard constraint in the inverse-design stage.

### Limits — state these in your thesis
- 20³ voxel grid: modulus is mesh-sensitive; run a convergence study at 30³/40³ on a few points.
- `E_rel` is *relative* to solid-phase modulus. Multiply by Halpin–Tsai `Es` for real MPa.
- Transport is effective diffusivity from a Laplace solve, **not** Stokes permeability.
- Linear elastic and small-strain — no yield, no buckling, no post-collapse behaviour.
- `nominal_pore_um` scales with the assumed `unit_cell_mm`; it is a design variable, not a result.

---

## Tier 3 — Literature curation (your contribution, the critical path)

Schema: **`curation_schema.csv`** — 56 columns across meta / chemistry / process / architecture /
mechanical / degradation / biological.

Design decisions baked into the schema, each of which prevents a specific failure:

- **`doi`** — the grouping key. Without it you cannot do honest validation. See the leakage result.
- **`time_point_weeks`** — *one row per time point*, not one row per scaffold. This is what makes
  trajectory prediction possible and is the single most important schema decision.
- **`modulus_retention_pct`, `Mw_remaining_pct`, `mass_remaining_pct`** — the trajectory targets.
  Modulus retention is the scarcest and most valuable field in the whole literature.
- **`test_condition`** (dry vs wet PBS) and **`strain_rate`** — a wet-tested polymer scaffold can
  read several-fold softer than the same scaffold dry. Papers that omit this are the main source of
  cross-study scatter. Record it as a feature so the model can account for it.
- **`porosity_method`** — gravimetric and µCT porosity differ by up to 10 points on one sample.
- **`extractor`** + **`data_source`** — re-extract 10% of rows with a second person and report
  inter-rater agreement. Cheap, and a genuine rigour marker few FYPs have.

**Target: 300–600 rows from 80–150 papers.** Split across the team, ~4 papers/person/day is realistic.

Search string to start from:
```
(PCL OR polycaprolactone OR PLA OR PLGA OR chitosan OR gelatin OR "silk fibroin")
AND ("bone scaffold" OR "bone tissue engineering")
AND ("compressive modulus" OR "compressive strength" OR degradation OR "mass loss")
```

Prioritise papers reporting **4/8/12-week degradation series** — they yield several rows each and
are exactly what nobody else is mining. Papers reporting only day-0 modulus are still useful (they
become `time_point_weeks = 0` rows), so nothing is wasted.

**Start in week 3.** The classic failure mode is a beautiful architecture with 60 datapoints.

---

## How the tiers combine

Tier 2 is dense, cheap and physically exact but idealised. Tier 3 is scarce, noisy and real.
Fuse them with a **multi-fidelity Gaussian process** (co-kriging): `f_high(x) = ρ·f_low(x) + δ(x)`.
Simulation supplies the shape of the response surface; the scarce experiments correct the offset.

That is the most defensible way to build a useful model from 400 literature rows, and it is the
technique most likely to distinguish this from a standard regression project.

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scikit-learn scipy matplotlib openpyxl
.venv/bin/python sim/generate_dataset.py   # Tier 2, ~85 s
.venv/bin/python sim/prepare_mlate.py      # Tier 1 (edit SRC to your MLATE checkout)
```

## Sources
- MLATE dataset — https://github.com/saeedrafieyan/MLATE ; paper https://doi.org/10.1088/1758-5090/ad6374
- Bioink database — https://doi.org/10.1088/1758-5090/ac933a ; portal https://cect.umd.edu/database
- PoLyInfo (free registration) — https://polymer.nims.go.jp/
- PolyHIPE scaffold data — https://doi.org/10.1016/j.dib.2015.09.051
- Inverse design of anisotropic bone scaffold — https://doi.org/10.3389/fbioe.2023.1241151
- Gyroid scaffold in-silico mechanics — https://doi.org/10.3390/computation11090181
- PLA/cHAP ML bone implant — https://doi.org/10.3390/biomimetics9100587
