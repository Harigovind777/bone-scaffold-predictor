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
| Files | `simulated_all.csv` (1896 rows), `gibson_ashby_fits.csv` (60 fits), `mesh_convergence.csv` (522 rows) |
| Code | `sim/tpms_fem.py`, `sim/generate_all.py`, `sim/convergence.py` |
| Cost | ~41 min for 1896 samples on 8 workers (`--quick` gives ~280 rows in ~25 s). The previous solver took 68 min for 747 samples; see `sim/benchmark_solver.py`. |

Both families live in **one table with one schema** — `sim/generate_all.py` supersedes the earlier
`generate_dataset.py` / `generate_fdm.py`, which wrote separate columns that could not be
concatenated. The legacy `tpms_simulated.csv` and `fdm_simulated.csv` are kept only for provenance;
nothing downstream reads them.

Voxel finite-element pipeline: TPMS level-set → bisect to target porosity → uniaxial compression
FEM (symmetry rollers, prescribed strain) → Laplace solve on the pore phase → interface counting.
Every sample additionally carries a **second loading axis** (hence an anisotropy ratio), a
**stress-concentration factor** as a strength proxy, and µCT-comparable pore size and strut
thickness from a distance transform rather than a nominal unit-cell figure.

**Validation:** `tools/selftest.py` — 23 checks, all passing. Solid block returns E/Es = 1 on every
axis; fully open box transports at D_eff/D_bulk = 1; cubic TPMS is elastically isotropic and a 0/90
print is not. 1891 of 1896 CG solves converged; the five that did not are all the same
Fischer–Koch S sheet cell at P = 0.88, whose largest connected solid cluster holds 0.2% of its own
material — they carry `cg_converged = 0` and are dropped by `clean_for_target`, not silently
averaged in. 121 geometries have closed porosity and are excluded by constraint downstream.

| Family | Rows | Architectures | Porosity range |
|---|---|---|---|
| TPMS | 1425 | 16 | 0.164 – 0.953 |
| FDM | 471 | 2 lay-down modes | 0.302 – 0.860 |

### Why this tier matters — it resolves what literature cannot

Fitted Gibson–Ashby parameters, 16 TPMS architectures (median fit R² = 0.976):

| Topology | Mode | n | C | points | R² |
|---|---|---|---|---|---|
| iwp | network | 1.52 | 0.84 | 86 | 0.997 |
| neovius | sheet | 1.56 | 0.91 | 90 | 0.995 |
| diamond | sheet | 1.61 | 0.88 | 83 | 0.994 |
| gyroid | sheet | 1.86 | 0.85 | 90 | 0.993 |
| schwarzP | network | 1.99 | 1.14 | 74 | 0.981 |
| schwarzP | sheet | 2.00 | 0.94 | 82 | 0.992 |
| iwp | sheet | 2.02 | 1.20 | 89 | 0.955 |
| splitP | sheet | 2.16 | 1.23 | 90 | 0.931 |
| splitP | network | 2.39 | 1.19 | 82 | 0.972 |
| gyroid | network | 2.41 | 1.11 | 90 | 0.987 |
| fischerKochS | network | 2.46 | 1.14 | 84 | 0.984 |
| diamond | network | 2.63 | 1.17 | 90 | 0.961 |
| lidinoid | sheet | 2.64 | 1.36 | 90 | 0.821 |
| fischerKochS | sheet | 2.94 | 2.10 | 89 | 0.403 |
| neovius | network | 3.56 | 2.12 | 58 | 0.857 |
| lidinoid | network | 3.91 | 1.79 | 73 | 0.793 |

The exponent spans **1.52 → 3.91**. Read the two ends differently. From 1.52 to about 2.5 is the
stretch-dominated to bending-dominated range Gibson–Ashby describes and published scaffolds occupy.
Above roughly 3 it is not: those fits (neovius/network n = 3.56, lidinoid/network n = 3.91) come
with the worst R² in the table (0.40–0.86) and a prefactor well over
1, which means the power law is being stretched to describe an architecture that is *losing its load
path* as porosity rises rather than thinning uniformly. The exponent there is a symptom, not a
material property. Fischer–Koch S sheet is the clearest case: R² = 0.40, i.e. barely a
power law at all.

Published scaffolds cluster at two or three porosities per architecture, so literature data can
never resolve any of this. Simulation can, densely and for free — including the negative result that
some architectures do not obey the law being fitted to them.

`C_unphysical` flags a fitted prefactor above 1.2: E cannot exceed Es at full density, so such a fit
is *local*, valid inside its sampled porosity window and not to be extrapolated toward ρ_rel = 1.
Flagged rather than dropped — 23 of the 60 fits across both families carry it, up from 16 of 43,
because the four new surfaces sit disproportionately at that end.

### Mesh convergence (`mesh_convergence.csv`)

The same geometries re-solved on a 16/20/24/32/44 ladder, which doubles as the low/high-fidelity
pairs for the co-kriging stage. Median deviation from the grid-44 answer: 14.0% at grid 16, 6.0% at
20, 3.5% at 32. Worst-case deviations are far larger (1800% at grid 16) — coarse grids do not merely
add noise, they can lose the load path entirely. Three of the 108 cases are excluded from that
summary: their solid phase does not span the specimen even at grid 44, so the reference modulus is
exactly zero and a relative deviation from it is undefined rather than infinite.

Note filaments thinner than ~3 voxels are not meshed at all, so **no FDM case resolves at grid 16**
and only half do at grid 20. `run_pipeline.py` therefore picks the coarsest grid that covers *both*
families as its cheap tier rather than assuming the coarsest one does.

---

## Tier 2b — FDM strut lattices (within `simulated_all.csv`, 471 rows, grid 44³)

This is what the biopolymer bone-scaffold literature actually prints: parallel filaments rotated
90° each layer. Its load path differs fundamentally from a TPMS, so the simulated tier needs it to
be transferable to Tier 3. All 471 solves converged, porosity 0.30–0.86 — the
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

Per-configuration fits are in `gibson_ashby_fits.csv` (44 FDM rows). Note the `C_unphysical` flag:
17 fits have **C > 1.2**, which is impossible for a real cellular solid (E cannot exceed Es at
ρ_rel = 1). Those are *local* fits valid only across the sampled porosity window — **do not
extrapolate them toward full density.** The worst (staggered 2s/12l, R² = 0.49) shows the power law
barely holding at all for sparse staggered lattices.

### On validating against published values
I attempted to anchor against a PCL/HA paper reporting 46.6 → 71.6 MPa across 40→60% infill
(implying n ≈ 1.06). **That anchor is not usable:** two points cannot constrain an exponent, and
slicer *infill %* is not measured porosity — shells and over-extrusion break the correspondence.
Real validation must come from Tier 3 rows with directly measured porosity. Another reason
curation is the critical path.

**What the model ladder actually reports** (predicting `E_rel_z`, leave-one-architecture-out
over 18 architectures — see `results/metrics.json`):

| Model | R² |
|---|---|
| Gibson–Ashby single global power law | **0.783** |
| Gradient boosting, no physics | 0.654 |
| Gibson–Ashby prior + learned residual | 0.647 |
| Same residual model, random 5-fold | 0.979 (optimistic) |

**The textbook power law wins.** On an architecture the model has never seen, neither learned
variant beats the two-parameter law it is built on. That is a result, not a failure: it says the
simulated tier's `E_rel` is a clean power law in relative density with no residual structure that
transfers between topologies, and it is why `tools/predict.py` uses the fitted (n, C) rather than a
surrogate. Whether a learned residual earns its place is a question for Tier 3 data, where the
scatter is real rather than numerical.

> **Superseded numbers, kept here as a warning.** Earlier revisions of this file reported
> R² = 0.444 for the power law and **0.841** for a random forest under the same protocol, and
> concluded that learning topology-dependent (n, C) "nearly doubles R² over textbook theory".
> Those came from the 192-row, TPMS-only sweep preserved in `tpms_simulated.csv` — the legacy
> file this README itself describes as read by nothing downstream. They disagreed with
> `results/metrics.json` by roughly 0.3 R², in the flattering direction, inside the document that
> teaches the leakage lesson. Regenerate numbers; do not retype them.

### A real finding already in the data
121 of 1896 samples have **non-percolating (closed-cell) porosity**. Those scaffolds are useless for bone ingrowth regardless of how good their modulus
looks. `pore_connected` flags them, and it should be a hard constraint in the inverse-design stage.

### Limits — state these in your thesis
- Mesh sensitivity is now measured rather than warned about: the sweep runs at 32³ (TPMS)
  and 44³ (FDM), and `mesh_convergence.csv` gives the deviation of every coarser grid from
  the finest. The residual caveat is that 44³ is the finest rung, so it anchors the ladder
  and cannot itself be checked against anything.
- `E_rel` is *relative* to solid-phase modulus. Multiply by Halpin–Tsai `Es` for real MPa.
- Transport is effective diffusivity from a Laplace solve, **not** Stokes permeability.
- Linear elastic and small-strain — no yield, no buckling, no post-collapse behaviour.
- `nominal_pore_um` scales with the assumed `unit_cell_mm`; it is a design variable, not a result.
- The compression solve meshes the solid phase only. Void voxels carry no equations, so a
  solid phase that touches none of the loaded faces returns E_rel = 0 rather than the old
  ersatz-stiffness floor. That is the correct reading — a platen cannot compress what it
  does not touch — but it means `structurally_dead` rows are exactly zero, not merely small.

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
