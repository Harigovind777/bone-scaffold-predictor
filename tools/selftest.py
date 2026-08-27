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
from tpms_fem import (TOPOLOGIES, build_at_porosity, compression_response,
                      compression_response_axes, effective_diffusivity, fdm_lattice,
                      morphometry, specific_surface)                # noqa: E402

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
    #
    # The bound is 1e-4, not machine zero: K_sc is a derivative of the displacement field
    # and so inherits the CG tolerance roughly one-for-one (4e-6 at the default rtol of
    # 1e-6, 4e-8 at 1e-8), while E_rel - an energy - converges quadratically and is exact
    # to 1e-11 either way. Four orders of margin still catches a wrong B matrix or a wrong
    # von Mises contraction, which are percent-level errors, and 1e-4 is two orders below
    # the third decimal the sweep writes K_sc out to.
    r = compression_response(np.ones((12, 12, 12), dtype=bool))
    assert abs(r["stress_concentration"] - 1.0) < 1e-4, r["stress_concentration"]


@check("fully open box transports at D_eff/D_bulk = 1 on every axis")
def _():
    void = np.zeros((12, 12, 12), dtype=bool)
    for axis in range(3):
        d = effective_diffusivity(void, axis=axis)
        assert abs(d - 1.0) < 0.02, f"axis {axis}: D={d}"


@check("every TPMS topology is elastically isotropic")
def _():
    # All eight level sets are invariant under a cyclic permutation of x, y, z, so
    # E_x = E_y = E_z. This is the sharpest available test that the axis generalisation
    # is correct: an indexing error in the boundary conditions breaks it immediately.
    # Run over the whole tuple rather than one topology, so adding a ninth surface that
    # is NOT cubic is caught here instead of silently entering the dataset as one.
    for topo in TOPOLOGIES:
        mask = build_at_porosity(topo, "network", 0.6, n=16, cells=1)
        E = [compression_response(mask, axis=a)["E_rel"] for a in range(3)]
        if max(E) < 1e-6:
            continue                    # solid phase spans no face here; nothing to load
        spread = (max(E) - min(E)) / np.mean(E)
        assert spread < 1e-5, f"{topo} came out anisotropic: {E}"


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
    for topo in TOPOLOGIES:
        for target in (0.4, 0.6, 0.8):
            mask = build_at_porosity(topo, "network", target, n=16, cells=1)
            got = 1.0 - mask.mean()
            assert abs(got - target) < 0.05, f"{topo} P={target}: got {got:.3f}"


@check("modulus decreases monotonically with porosity")
def _():
    E = [compression_response(build_at_porosity("gyroid", "network", p, n=16),
                              axis=2)["E_rel"] for p in (0.3, 0.5, 0.7, 0.85)]
    assert all(a > b for a, b in zip(E, E[1:])), E


@check("transport anisotropy is measured, not copied from the z axis")
def _():
    # Regression test. effective_diffusivity looped `for axis in range(3)` over the same
    # name as its own argument, so the Dirichlet faces and the flux plane were always
    # taken along z: D_eff_x came back as an exact copy of D_eff_z on every row of the
    # dataset, and a 0/90 lay-down - obviously not isotropic in transport - reported a
    # ratio of 1.000. A slab of parallel z-channels is the unambiguous case: open along
    # z, blocked across it.
    n = 20
    mask = np.ones((n, n, n), dtype=bool)
    mask[4:8, 4:8, :] = False                       # one straight pore along z
    d_z = effective_diffusivity(mask, axis=2)
    d_x = effective_diffusivity(mask, axis=0)
    assert d_z > 0.02, f"z channel should conduct, got {d_z}"
    assert d_x < 1e-3 * d_z, f"x is blocked but conducted {d_x} against {d_z} along z"


@check("solid-only assembly agrees with the ersatz-void solve")
def _():
    # The compression solve stopped meshing void voxels. That is a change of formulation,
    # not just of speed, so the two must be pinned against each other: the ersatz phase
    # only ever contributed at the 1e-6 level, and dropping it moves E_rel by less than
    # the sixth decimal the sweep writes out.
    for mask in (build_at_porosity("gyroid", "network", 0.75, n=20),
                 fdm_lattice(20, 0.16, 0.5, stagger=True)):
        fast = compression_response(mask, axis=2)
        ersatz = compression_response(mask, axis=2, void_ratio=1e-6)
        rel = abs(fast["E_rel"] - ersatz["E_rel"]) / ersatz["E_rel"]
        assert rel < 5e-4, f"E_rel disagrees by {rel:.2%}"
        k = abs(fast["stress_concentration"] - ersatz["stress_concentration"])
        assert k / ersatz["stress_concentration"] < 5e-3, f"K_sc disagrees by {k}"


@check("one assembly for several axes matches solving them separately")
def _():
    mask = fdm_lattice(20, 0.16, 0.5, stagger=False)
    both = compression_response_axes(mask, (2, 0))
    for ax in (2, 0):
        one = compression_response(mask, axis=ax)
        assert both[ax]["E_rel"] == one["E_rel"], f"axis {ax}: {both[ax]} vs {one}"


@check("a solid phase touching no face carries no load")
def _():
    # The ersatz void used to hand a floating structure a small non-zero modulus that came
    # entirely from the fictitious material around it - Neovius/network at P = 0.70 reads
    # 2.4e-6 under the old formulation and exactly 0 under this one. A platen cannot
    # compress something it does not touch, and the pipeline's structurally_dead flag
    # depends on that reading as dead rather than merely soft.
    n = 16
    mask = np.zeros((n, n, n), dtype=bool)
    mask[4:12, 4:12, 4:12] = True                   # a cube floating clear of every face
    r = compression_response(mask, axis=2)
    assert r["E_rel"] == 0.0, f"floating solid returned E_rel={r['E_rel']}"
    assert r["cg_info"] == 0, "should converge trivially, not fall back"


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
