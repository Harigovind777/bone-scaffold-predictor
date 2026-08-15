"""
Self-test for the physics core and the pipeline's data contract.

These are the invariants the whole project rests on. Each one is a case where the answer
is known independently of the code, so a regression shows up as a failed assertion rather
than as a plausible-looking number in a results table.

    .venv/bin/python tools/selftest.py

Small grids throughout - this is a correctness check, not a benchmark. Runs in ~30 s.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

import physics as phys                                              # noqa: E402
from tpms_fem import (build_at_porosity, compression_response, effective_diffusivity,
                      fdm_lattice, morphometry, specific_surface)   # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------------------
# FEM core
# --------------------------------------------------------------------------

@check("solid block returns E/Es = 1 on every axis")
def _():
    solid = np.ones((12, 12, 12), dtype=bool)
    for axis in range(3):
        r = compression_response(solid, axis=axis)
        assert abs(r["E_rel"] - 1.0) < 1e-9, f"axis {axis}: E={r['E_rel']}"


@check("solid block has no stress concentration (K_sc = 1)")
def _():
    # A uniform block loads uniformly, so the 95th-percentile von Mises stress equals
    # the macroscopic stress exactly. Any deviation means the stress recovery is wrong.
    r = compression_response(np.ones((12, 12, 12), dtype=bool))
    assert abs(r["stress_concentration"] - 1.0) < 1e-6, r["stress_concentration"]


@check("fully open box transports at D_eff/D_bulk = 1 on every axis")
def _():
    void = np.zeros((12, 12, 12), dtype=bool)
    for axis in range(3):
        d = effective_diffusivity(void, axis=axis)
        assert abs(d - 1.0) < 0.02, f"axis {axis}: D={d}"


@check("cubic TPMS is elastically isotropic")
def _():
    # Gyroid, diamond, schwarzP and IWP all have cubic symmetry, so E_x = E_y = E_z.
    # This is the sharpest available test that the axis generalisation is correct:
    # an indexing error in the boundary conditions breaks it immediately.
    mask = build_at_porosity("gyroid", "network", 0.6, n=16, cells=1)
    E = [compression_response(mask, axis=a)["E_rel"] for a in range(3)]
    spread = (max(E) - min(E)) / np.mean(E)
    assert spread < 1e-6, f"anisotropic cubic cell: {E}"


@check("0/90 printed lattice is NOT isotropic")
def _():
    # The converse check: if this passed as isotropic too, the axis argument would be
    # doing nothing and the previous test would be vacuous.
    mask = fdm_lattice(24, 0.16, 0.5, stagger=False)
    Ez = compression_response(mask, axis=2)["E_rel"]
    Ex = compression_response(mask, axis=0)["E_rel"]
    assert Ez / Ex < 0.9, f"expected through-thickness compliance, got Ez/Ex={Ez/Ex:.3f}"


@check("staggering a lay-down raises its stress concentration")
def _():
    a = fdm_lattice(24, 0.16, 0.5, stagger=False)
    s = fdm_lattice(24, 0.16, 0.5, stagger=True)
    Ka = compression_response(a, axis=2)["stress_concentration"]
    Ks = compression_response(s, axis=2)["stress_concentration"]
    assert Ks > Ka, f"staggered {Ks:.1f} should concentrate more than aligned {Ka:.1f}"


@check("porosity bisection reaches its target")
def _():
    for topo in ("gyroid", "diamond", "schwarzP", "iwp"):
        for target in (0.4, 0.6, 0.8):
            mask = build_at_porosity(topo, "network", target, n=16, cells=1)
            got = 1.0 - mask.mean()
            assert abs(got - target) < 0.05, f"{topo} P={target}: got {got:.3f}"


@check("modulus decreases monotonically with porosity")
def _():
    E = [compression_response(build_at_porosity("gyroid", "network", p, n=16),
                              axis=2)["E_rel"] for p in (0.3, 0.5, 0.7, 0.85)]
    assert all(a > b for a, b in zip(E, E[1:])), E


@check("morphometry recovers known thicknesses")
def _():
    solid = np.ones((16, 16, 16), dtype=bool)
    m = morphometry(solid, domain_mm=2.0)
    assert m["pore_size_um"] == 0.0, m
    assert m["strut_thickness_um"] > 900, m          # inscribed sphere fills the 2 mm cube
    assert specific_surface(solid) == 0.0            # no interface in a solid block


# --------------------------------------------------------------------------
# Mechanistic decoder
# --------------------------------------------------------------------------

@check("Halpin-Tsai reduces to the matrix modulus at zero filler")
def _():
    Es, Vf = phys.solid_modulus("PCL", "none", 0.0)
    assert abs(Es - phys.POLYMERS["PCL"]["E"]) < 1e-9, Es
    assert Vf == 0.0


@check("adding stiff ceramic raises the composite modulus monotonically")
def _():
    E = [phys.solid_modulus("PCL", "HA", w)[0] for w in (0.0, 0.1, 0.2, 0.3)]
    assert all(a < b for a, b in zip(E, E[1:])), E


@check("degradation is monotonic and bounded")
def _():
    d = phys.degrade("PLGA_50_50", "none", 0.0, porosity=0.6)
    assert np.all(np.diff(d["Mn_rel"]) <= 1e-12), "Mn must not increase"
    assert np.all(np.diff(d["mass_remaining"]) <= 1e-9), "mass must not increase"
    assert d["mass_remaining"][0] > 0.999 and d["mass_remaining"][-1] >= 0.0


@check("ceramic buffering slows polyester degradation")
def _():
    # The mechanism behind the well-known experimental result. If this inverts, the
    # buffering term is wired backwards - and every SHAP plot downstream would be wrong.
    plain = phys.degrade("PLGA_50_50", "none", 0.0, porosity=0.6)
    buffered = phys.degrade("PLGA_50_50", "beta_TCP", 0.30, porosity=0.6)
    assert buffered["Mn_rel"][-1] > plain["Mn_rel"][-1], "ceramic should retard hydrolysis"
    assert buffered["pH"].min() > plain["pH"].min(), "ceramic should raise the acid floor"


@check("PCL degrades far slower than PLGA 50:50")
def _():
    pcl = phys.degrade("PCL", "none", 0.0, porosity=0.6)["mass_remaining"][-1]
    plga = phys.degrade("PLGA_50_50", "none", 0.0, porosity=0.6)["mass_remaining"][-1]
    assert pcl > plga, f"PCL {pcl:.3f} should outlast PLGA 50:50 {plga:.3f} at one year"


# --------------------------------------------------------------------------
# Pipeline contracts
# --------------------------------------------------------------------------

@check("design features exclude everything requiring a solve")
def _():
    import pandas as pd
    from pipeline import config as C
    from pipeline import data as D

    row = {c: 1.0 for c in C.DESIGN_NUMERIC + C.SOLVED_ONLY}
    row.update(family="TPMS", topology="gyroid", mode="network",
               relative_density=0.4, porosity=0.6)
    df = pd.DataFrame([row, {**row, "relative_density": 0.5, "topology": "diamond"}])

    X = D.build_features(df, allow_solved=False)
    leaked = [c for c in C.SOLVED_ONLY if c in X.columns]
    assert not leaked, f"solved features leaked into the design matrix: {leaked}"

    X2 = D.build_features(df, allow_solved=True)
    assert any(c in X2.columns for c in C.SOLVED_ONLY), "allow_solved=True had no effect"


@check("grouped validation detects leakage that random splitting hides")
def _():
    import pandas as pd
    from pipeline import models, validation as V

    # Construct data with a per-group offset the model can memorise. A random split sees
    # every group in training and scores well; a grouped split must extrapolate to an
    # unseen offset and cannot. If this gap fails to appear, the splitters are not
    # actually grouping and every headline number in the project is unverified.
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(12), 20)
    offset = rng.normal(0, 3, 12)[groups]
    x = rng.uniform(0.1, 0.9, len(groups))
    y = 2 * np.log(x) + offset + rng.normal(0, 0.01, len(groups))
    X = pd.DataFrame({"log_relative_density": np.log(x), "relative_density": x,
                      "group_hint": offset})

    gap = V.leakage_gap(lambda: models.PlainML(), X, y, groups)
    assert gap["r2_gap"] > 0.1, f"expected inflation under random splitting, got {gap['r2_gap']:.3f}"


# --------------------------------------------------------------------------
# Prediction interface
# --------------------------------------------------------------------------

@check("predict() agrees with the inverse search on the same candidate")
def _():
    # The interface and the pipeline must not be two opinions about one scaffold. Both
    # routes end in inverse.evaluate_candidate; this pins that they are fed the same
    # architecture, so a divergence here means the interface is resolving geometry wrongly
    # rather than that the physics changed.
    from tools.predict import predict
    from pipeline import data as D, inverse

    cat = inverse.build_catalogue(D.load_simulated())
    row = cat[cat.config == "FDM|aligned|s3|l16"].sort_values("porosity").iloc[len(
        cat[cat.config == "FDM|aligned|s3|l16"]) // 2]
    ref = inverse.evaluate_candidate(row, "PLGA_85_15", "beta_TCP", 0.30, "trabecular_mid")

    got = predict(topology="fdm", mode="aligned", n_struts=3, n_layers=16,
                  porosity=float(row.porosity), polymer="PLGA_85_15",
                  ceramic="beta_TCP", ceramic_wt=0.30, site="trabecular_mid")
    assert abs(got["mechanics"]["E0_MPa"] - ref["E0_MPa"]) / ref["E0_MPa"] < 0.02, (
        f"E0 {got['mechanics']['E0_MPa']:.1f} vs inverse search {ref['E0_MPa']:.1f}")
    assert abs(got["verdict"]["load_transfer_cost"] - ref["load_transfer_cost"]) < 5e-3


@check("predict() refuses to extrapolate silently")
def _():
    # The failure mode that matters for a design tool is a confident number for a geometry
    # nobody simulated. Outside the sampled window the answer must be labelled.
    from tools.predict import predict
    inside = predict(topology="gyroid", mode="network", n_cells=2, porosity=0.60)
    assert not inside["provenance"]["extrapolating"]
    assert "interpolated" in inside["provenance"]["modulus_from"]

    outside = predict(topology="gyroid", mode="network", n_cells=2, porosity=0.97)
    assert outside["provenance"]["extrapolating"], "P=0.97 is outside every sampled window"
    assert "EXTRAPOLATED" in outside["provenance"]["modulus_from"]


@check("FDM solve keeps n_struts meaning what the config key says")
def _():
    # tpms_fem.build_fdm_at_porosity bisects SPACING, which silently changes the strut
    # count; predict.py bisects DIAMETER at fixed spacing instead. If that ever regresses,
    # the reachable porosity window stops matching the production sweep's.
    from tools.predict import _fdm_mask_at_porosity
    from pipeline import data as D, inverse

    cat = inverse.build_catalogue(D.load_simulated())
    g = cat[cat.config == "FDM|aligned|s3|l12"]
    target = float(g.porosity.max()) - 0.02

    mask = _fdm_mask_at_porosity(target, grid=32, n_struts=3, n_layers=12, stagger=False)
    got = 1.0 - float(mask.mean())
    assert abs(got - target) < 0.02, f"asked {target:.3f}, got {got:.3f}"

    try:
        _fdm_mask_at_porosity(0.95, grid=32, n_struts=3, n_layers=12, stagger=False)
    except ValueError:
        pass
    else:
        raise AssertionError("porosity 0.95 is unreachable for s3/l12 and must be refused")


def main():
    passed = failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
            if "-v" in sys.argv:
                traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
