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
.venv/bin/pip install -r requirements.txt     # pinned: the versions behind results/

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

**What predicts what.** The modulus is *not* taken from a learned surrogate. Under
leave-one-architecture-out the surrogate used to score R² = 0.65 against 0.78 for the
Gibson–Ashby law it is built on — using it would have dressed up a worse predictor as a
better one. It now scores 0.800, because it learned to switch itself off: the residual
stage is scaled by a factor estimated inside each fold, and on a topology held out
entirely that factor comes back as exactly zero. A model that knows when it has nothing
to add is the useful kind. What runs is still the fitted (n, C) for that architecture
plus the zero-learned-parameter decoder in `sim/physics.py`. Where a measured `E_rel` exists at the requested porosity, C
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
| `pipeline/printability.py` | Tier 1 printability as an **ordinal** target, and the readout that makes it one |
| `tools/predict.py` | **The prediction interface** — one scaffold spec → properties, trajectory, verdict, provenance |
| `tools/accuracy_ledger.py` | **The accuracy framework** — every headline model re-measured against a frozen baseline |
| `ACCURACY.md` | The framework written up: seven rules, every change and why, and the measured graveyard |
| `tools/selftest.py` | 26 physics, pipeline-contract and JSON-output checks |
| `tools/validate_curation.py` | Gate for Tier 3 curation sheets |
| `results/` | Everything the pipeline produces — `README.md`, `metrics.json`, `figures/`, `tables/` |

---

## Four design decisions that shape everything

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

**4. Accuracy is a delta against a frozen number, not an adjective.**
`results/accuracy_baseline.json` holds every headline score as it stood before the last
round of work, and `tools/accuracy_ledger.py` re-measures all of them and prints the
difference. It also carries the ceiling — what a model that knew the held-out
architecture's own power law would score — so the size of the remaining prize is visible
rather than assumed, and a list of the variants that were built, measured and dropped,
so nobody spends that week twice.

---

## Improving accuracy: what worked and what did not

**Published report: https://harigovind777.github.io/bone-scaffold-predictor/** — rebuilt by
`.github/workflows/accuracy.yml` on every push, which reruns the self-tests, the accuracy
ledger and the full pipeline on GitHub's servers and publishes the page from that run's
results. The run log is linked from the page footer.

**The full write-up is [`ACCURACY.md`](ACCURACY.md)** — the seven rules, every change with
its reasoning, and the graveyard of variants that were measured and dropped.

```bash
.venv/bin/python tools/accuracy_ledger.py             # the ledger, with deltas
.venv/bin/python tools/accuracy_ledger.py --rejected  # the graveyard, with numbers
```

| model | metric | was | now |
|---|---|---|---|
| physics-informed residual, leave-one-architecture-out | R² | 0.647 | **0.800** |
| Gibson–Ashby power law, leave-one-architecture-out | R² | 0.783 | **0.800** |
| multi-fidelity fusion, 4 trusted points | R² | 0.269 | **0.819** |
| multi-fidelity fusion, 64 trusted points | R² | 0.768 | **0.821** |
| printability (grouped by DOI) | QWK | 0.144 | **0.375** |
| printability (grouped by DOI) | Spearman | 0.278 | **0.546** |

Five changes, and none of them is a bigger model.

**One vote per architecture.** The pooled power-law fit weighted each architecture by how
often the sweep sampled it — two FDM architectures held 471 of 1806 rows. Equal weight
per architecture is the choice the validation protocol already makes. 0.783 → 0.790.

**Split the power law by deformation mode.** Gibson–Ashby predicts different exponents for
bending- and stretch-dominated structures, and the data agree: sheet TPMS average n = 1.98,
network TPMS 2.61. Unlike the topology label, the mode is an *input* a new design arrives
with, so a per-mode fit transfers to topologies the sweep never saw where a
per-architecture table cannot. 0.790 → 0.800, no parameter tuned on the score.

**The residual learns when to shut up.** The physics-informed rung used to score *below*
its own prior. Its correction is now scaled by a trust factor estimated inside each fold;
on an unseen topology it comes back as 0 and the rung reduces to the per-mode prior
instead of undercutting it. 0.647 → 0.800.

**ρ and δ have to earn their place.** Co-kriging fits `f_high = ρ·f_low + δ`. ρ now has a
physics prior N(1, 0.02²) read off the mesh study, and both ρ − 1 and δ are scaled by trust
factors measured by leave-one-out on the trusted points. Both come back at 0 at every
budget: on this mesh pair the fine solve adds nothing the coarse one lacks, and the model
now says so instead of losing 0.10 R² pretending otherwise. It holds the cheap-source
floor (0.818) at all eight budgets, where it used to fall as low as 0.269.

**Printability is ordinal, and accuracy was the wrong ruler.** 55.5% of rows are level 3,
so answering "3" to everything scores 0.555. Reading the classifier's expected level
Σk·pₖ instead of its argmax improves every metric at once, and quadratic-weighted kappa —
which the majority answer scores 0 on — is reported alongside accuracy.

**What did not work.** Hierarchical (n, C) from morphometry, kNN over architectures once
`k` is nested, curvature and percolation forms, Huber loss, stacking fusion, QWK-optimised
cut points — all measured, all in `ACCURACY.md` with their numbers. The oracle ceiling is
0.953, so the remaining prize is real, but it is not reachable from the descriptors
available here.

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
