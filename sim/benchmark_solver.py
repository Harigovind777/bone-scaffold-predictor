"""
Compression-solver benchmark: what the solid-only assembly changed, and by how much.

The sweep's cost is almost entirely one conjugate-gradient solve per loading axis - at
grid 44 the assembly is under a second and the CG is eleven - so this is the only place
where making the framework faster is worth any effort at all. This script is the
measurement that settled two decisions, kept runnable so neither has to be taken on
trust.

    .venv/bin/python sim/benchmark_solver.py [--full]

WHAT IS COMPARED

  reference   every voxel meshed, void_ratio 1e-10, rtol 1e-10.
              The void_ratio -> 0 limit, to the precision CG can deliver. This is the
              answer the other rows are scored against; it is not a candidate, it is
              far too slow.
  ersatz      every voxel meshed, void_ratio 1e-6, rtol 1e-8.  The former default.
  solid-only  solid voxels meshed, nothing else, rtol 1e-6.    The current default.

WHAT IT SHOWS

Meshing the void was never neutral. The ersatz phase carries a little of the platen load
and constrains the solid where it borders it, and both effects are largest exactly where
the modulus is smallest - the deviation from the reference runs ~0.002% at P = 0.70 and
~0.04% at P = 0.85, i.e. it grows into the high-porosity range the bone-scaffold
literature actually occupies. Solid-only assembly is not an approximation of that: it IS
the void_ratio -> 0 limit, and it reproduces the reference to seven figures while
dropping two thirds of the degrees of freedom and two thirds of the non-zeros.

The tolerance was then re-set from 1e-8 to 1e-6 against the same reference. E_rel is an
energy and converges quadratically, so it is exact to 1e-11 either way; K_sc is a
derivative of the displacement field and tracks the tolerance one-for-one, reaching 4e-6
at 1e-6 - still two orders below the third decimal the sweep writes it out to.

The second section measures what the sweep actually runs: TWO loading axes per geometry.
Every TPMS level set here is invariant under the x -> y -> z -> x rotation on the voxel
grid, so rotating the first solution lands exactly on the second axis's answer and CG
stops at iteration zero. A 0/90 lay-down is not symmetric that way and pays full price -
which is the control that shows the saving is real rather than an assumption being
smuggled in.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpms_fem import (build_at_porosity, compression_response,
                      compression_response_axes, fdm_lattice)

CASES = (
    ("gyroid network P0.70", lambda: build_at_porosity("gyroid", "network", 0.70, n=44)),
    ("gyroid network P0.85", lambda: build_at_porosity("gyroid", "network", 0.85, n=44)),
    ("iwp sheet P0.60 n32", lambda: build_at_porosity("iwp", "sheet", 0.60, n=32)),
    ("fdm 0/90 s3 l12", lambda: fdm_lattice(44, 1.0 / 12 / 0.8, 1 / 3.0, 0.8, False)),
)

VARIANTS = (
    ("reference  void1e-10 rtol1e-10", dict(void_ratio=1e-10, rtol=1e-10)),
    ("ersatz     void1e-6  rtol1e-8", dict(void_ratio=1e-6, rtol=1e-8)),
    ("solid-only           rtol1e-8", dict(void_ratio=0.0, rtol=1e-8)),
    ("solid-only           rtol1e-6", dict(void_ratio=0.0, rtol=1e-6)),
    ("solid-only           rtol1e-5", dict(void_ratio=0.0, rtol=1e-5)),
)


def main():
    cases = CASES if "--full" in sys.argv else CASES[:1] + CASES[2:]
    print(f"{'variant':32s} {'E_rel':>11s} {'vs ref':>9s} {'K_sc':>9s} {'seconds':>8s} {'speedup':>8s}")
    for name, build in cases:
        mask = build()
        print(f"\n{name}   n={mask.shape[0]}  porosity={1 - mask.mean():.3f}")
        ref = base = None
        for label, kw in VARIANTS:
            t0 = time.time()
            r = compression_response(mask, axis=2, **kw)
            dt = time.time() - t0
            if ref is None:
                ref = r["E_rel"]
            if "ersatz" in label:
                base = dt
            dev = 100 * abs(r["E_rel"] - ref) / ref if ref else 0.0
            speed = f"{base / dt:6.1f}x" if base else "     -"
            print(f"  {label:30s} {r['E_rel']:11.7f} {dev:8.4f}% "
                  f"{r['stress_concentration']:9.4f} {dt:8.2f} {speed:>8s}")

    print(f"\n{'-' * 78}\nTwo axes per geometry, as the sweep runs them\n{'-' * 78}")
    print(f"{'geometry':24s} {'cyclic':>7s} {'separate':>9s} {'shared':>9s} {'speedup':>8s}"
          f"  {'E_z':>10s} {'E_x':>10s}")
    for name, build in cases:
        mask = build()
        t0 = time.time()
        a = compression_response(mask, axis=2)
        b = compression_response(mask, axis=0)
        t_sep = time.time() - t0
        t0 = time.time()
        both = compression_response_axes(mask, (2, 0))
        t_sh = time.time() - t0
        assert abs(both[2]["E_rel"] - a["E_rel"]) < 1e-9 * max(a["E_rel"], 1e-9)
        assert abs(both[0]["E_rel"] - b["E_rel"]) < 1e-6 * max(b["E_rel"], 1e-9)
        cyc = bool((mask.transpose(1, 2, 0) == mask).all())
        print(f"{name:24s} {str(cyc):>7s} {t_sep:8.2f}s {t_sh:8.2f}s {t_sep / t_sh:7.1f}x"
              f"  {both[2]['E_rel']:10.6f} {both[0]['E_rel']:10.6f}")


if __name__ == "__main__":
    main()
