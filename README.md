# Bone scaffolds — property prediction and inverse design

Predict the mechanical and degradation behaviour of biopolymer bone scaffolds from
composition, process and architecture — then run that model backwards to propose a
scaffold worth printing.

The project exists because **there is no public dataset covering mechanical *and*
degradation properties of biopolymer bone scaffolds** (see `data/README.md` for the
search that established this). So the dataset is part of the contribution, and it is
built in three tiers of decreasing volume and increasing trustworthiness.

```
Tier 1  MLATE            borrowed, 1171 rows   printability & biology, no mechanics
Tier 2  simulated        1896 rows here       dense, exact, idealised
Tier 3  curated papers   the critical path     scarce, noisy, real
```

The modelling strategy follows directly from that shape: simulation supplies the *shape*
of the response surface, scarce real data corrects its *offset*, and the two are fused by
co-kriging rather than pooled.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scikit-learn scipy matplotlib openpyxl

.venv/bin/python sim/generate_all.py     # Tier 2+2b, parallel   (~41 min, or --quick for ~25 s)
.venv/bin/python sim/convergence.py      # mesh ladder + fidelity pairs
.venv/bin/python run_pipeline.py         # everything downstream -> results/
```

`run_pipeline.py` is the single entry point for *building* the framework. Stages whose
inputs are missing report themselves as skipped rather than failing, so it runs
end-to-end on whatever exists.

---

## The web interface

```bash
.venv/bin/python webapp/server.py        # then open http://127.0.0.1:8000
```

Three tabs, and it drives the real code — not a mock-up:

- **Predict** — pick an architecture and chemistry, get E₀, strength, the 52-week
  trajectory charted, every clinical constraint as a pass/fail chip, and a provenance
  block saying where the number came from. Tick *solve this exact geometry* to run the
  voxel FEM live (~1 s) on something the sweep never sampled.
- **Results** — the dashboard: dataset size, the model ladder, the leakage inflation, the
  fusion result, and all six figures (click to enlarge).
- **Pipeline** — a **Run pipeline** button that executes `run_pipeline.py` and streams its
  output into the page line by line, then reloads the dashboard when it finishes.

Standard library only — no Flask, no `npm install`, no CDN. A demo's usual failure mode is
a missing dependency five minutes beforehand, so there are none to miss. It binds to
127.0.0.1, serves figures from a whitelist, and starts the pipeline as a fixed command
with no caller-supplied arguments.

Every prediction updates the URL with the full specification, so a result shown in a viva
can be bookmarked, pasted into a report, or reopened later and re-run against whatever the
pipeline says then.

---

## Predicting one scaffold

`tools/predict.py` is the entry point for *using* it — one specification in, its
predicted behaviour and a feasibility verdict out.

```bash
.venv/bin/python tools/predict.py --topology gyroid --mode network --cells 2 \
    --porosity 0.68 --polymer PLGA_85_15 --ceramic beta_TCP --wt 0.30 \
    --site trabecular_mid
```

```
MECHANICS
  E0                      268.0 MPa   (native 300)
  sigma0                    9.6 MPa   (need 6.0)
  E at 12 weeks            40.5 %  of E0
VERDICT
  [PASS] geometry printable and ingrowth-capable   [PASS] stiffness within 0.5-2x native
  [PASS] strength >= 6.0 MPa                       [PASS] local pH stays >= 6.5
  FEASIBLE   load-transfer cost = 0.2239
```

Exit code is 0 when feasible, 1 when not and 2 on a bad request, so it scripts. Add
`--json` for machine-readable output, `--list-configs` for every architecture predictable
without a fresh solve, and `--solve` to mesh and solve a geometry the sweep never sampled
(~1 s, measured end-to-end through the web API). Importable too:
`from tools.predict import predict`.

**What predicts what.** The modulus is *not* taken from a learned surrogate — under
leave-one-architecture-out the surrogate scores R² = 0.65 against 0.78 for the
Gibson–Ashby law it is built on, so using it would dress up a worse predictor as a better
one. What runs is the fitted (n, C) for that architecture plus the zero-learned-parameter
decoder in `sim/physics.py`. Where a measured `E_rel` exists at the requested porosity, C
is re-anchored so the trajectory starts exactly on it and the fitted exponent governs only
how stiffness *moves* as degradation opens the structure up.

Every answer carries a `PROVENANCE` block saying where the modulus came from and whether
the request sat inside the data. Ask for a porosity the architecture was never simulated
at and it says so loudly rather than returning a confident number.

---

## What lives where

| Path | Role |
|---|---|
| `sim/tpms_fem.py` | Voxel FEM core: TPMS and FDM geometry, compression, transport, morphometry |
| `sim/physics.py` | Mechanistic decoder — Halpin–Tsai, hydrolysis kinetics, Gibson–Ashby, bone healing. **Zero learned parameters** |
| `sim/generate_all.py` | Parallel sweep → `data/simulated_all.csv` (one unified schema for both families) |
| `sim/convergence.py` | Mesh ladder → `data/mesh_convergence.csv`, doubling as low/high-fidelity pairs |
| `sim/benchmark_solver.py` | The compression-solver measurement: solid-only vs ersatz-void assembly, against a void→0 reference |
| `pipeline/data.py` | Loaders + the design/solved feature contract |
| `pipeline/validation.py` | Grouped splitters and the leakage diagnostic |
| `pipeline/models.py` | The model ladder, physics → physics-informed |
| `pipeline/fusion.py` | Multi-fidelity co-kriging |
| `pipeline/inverse.py` | Constrained inverse design |
| `pipeline/report.py` | Figures and companion tables |
| `webapp/server.py` | **The web interface** — stdlib HTTP server over the same code paths |
| `webapp/index.html` | Single-page UI: predict, results dashboard, live pipeline runner |
| `tools/predict.py` | **The prediction interface** — one scaffold spec → properties, trajectory, verdict, provenance |
| `tools/selftest.py` | 23 physics and pipeline-contract checks |
| `tools/validate_curation.py` | Gate for Tier 3 curation sheets |
| `results/` | Everything the pipeline produces — `README.md`, `metrics.json`, `figures/`, `tables/` |

---

## Three design decisions that shape everything

**1. Validation is grouped, never random.**
Rows from one publication share a material batch, a printer and an operator; rows from one
architecture share a load path. A random split puts near-duplicates on both sides and
reports a number that cannot be reproduced on a new paper or a new topology. Every headline
score here is leave-one-architecture-out or grouped by DOI, and `run_pipeline.py` reports
what a random split *would* have claimed so the gap is visible rather than assumed.

**2. A feature is either a design variable or a solved one.**
`pipeline/config.py` splits them explicitly. Anything requiring an FEM or Laplace solve is
barred from the model used for inverse design — otherwise "designing" a scaffold requires
first computing the answer the model exists to predict. This is enforced in code, not by
convention.

**3. Constraints are hard, not weighted.**
A scaffold with closed porosity is not a slightly worse scaffold — bone cannot grow into
it at any modulus. Folding that into a weighted objective lets a good stiffness score buy
off a fatal geometry, so connectivity, pore size, strut printability and porosity are
filters applied before scoring.

---

## Adding Tier 3 data

1. Copy `data/tier3_template.csv` (56 columns, from `data/curation_schema.csv`).
2. **One row per time point**, not per scaffold — this is what makes trajectory
   prediction possible and is the single most important schema decision.
3. Always record `doi` (the grouping key), `test_condition` (a wet-tested polymer scaffold
   reads several-fold softer than the same scaffold dry) and `porosity_method`
   (gravimetric and µCT differ by up to 10 points on one sample).
4. Re-extract ~10% of rows with a second person so an inter-rater figure can be reported.
5. Validate before committing:

```bash
.venv/bin/python tools/validate_curation.py data/tier3_curated.csv
```

Save the result as `data/tier3_curated.csv`. The pipeline picks it up automatically — no
code changes.

Prioritise papers reporting **4/8/12-week degradation series**: they yield several rows
each and are exactly what nobody else is mining. Papers with only day-0 modulus still
count as `time_point_weeks = 0` rows, so nothing is wasted.
