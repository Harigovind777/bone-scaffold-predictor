"""
TPMS scaffold dataset generator: geometry -> effective mechanical & transport properties.

Generates the LOW-FIDELITY, HIGH-VOLUME tier of the scaffold dataset. Each sample:
  1. builds a triply-periodic minimal surface (TPMS) unit-cell lattice on a voxel grid,
  2. bisects the level-set constant to hit a target porosity,
  3. runs a voxel finite-element uniaxial compression to get apparent modulus,
  4. solves a Laplace problem on the pore phase for effective diffusivity / tortuosity,
  5. counts solid-void interfaces for specific surface area.

The point of this tier is to densely learn the Gibson-Ashby topology parameters
(exponent n, prefactor C) which literature data alone can never resolve, because
published scaffolds cluster at a handful of porosities per architecture.

Outputs a tidy CSV consumable by the property-prediction framework.

WHERE THE TIME GOES. Step 3 is the whole cost - at grid 44 the compression solve is
about 250x the Laplace solve and the morphometry put together, and inside it the
conjugate-gradient iteration is ~90% of the wall clock. Two changes follow from that and
are measured in sim/benchmark_solver.py: the stiffness matrix is assembled over the SOLID
PHASE ONLY rather than meshing void voxels at a token modulus, and a geometry loaded
along several axes is assembled once, with the later axes warm-started by rotating the
first solution. Neither is an approximation - the first IS the void_ratio -> 0 limit the
ersatz phase was approximating, and the second is a starting guess CG checks for itself.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator

# --------------------------------------------------------------------------
# TPMS level-set fields
# --------------------------------------------------------------------------

# The four classical scaffold surfaces, then four less-sampled ones added to widen the
# architecture axis. Leave-one-architecture-out is the framework's headline protocol and
# its variance is set by how many architectures there are to hold out - eight topologies
# x two modes gives sixteen groups where four gave eight, which is the cheapest available
# improvement to that estimate. All eight are cubic (their level sets are invariant under
# a cyclic permutation of x, y, z), so E_x = E_z to solver tolerance; tools/selftest.py
# checks this on every one of them, and it is the sharpest test that the loading-axis
# generalisation is right.
TOPOLOGIES = ("gyroid", "diamond", "schwarzP", "iwp",
              "fischerKochS", "neovius", "lidinoid", "splitP")


def tpms_field(kind, n, cells):
    """Evaluate a TPMS level-set on an n^3 voxel grid spanning `cells` unit cells."""
    lin = (np.arange(n) + 0.5) / n * (2 * np.pi * cells)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")

    if kind == "gyroid":
        return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    if kind == "diamond":
        return (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
                + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z))
    if kind == "schwarzP":
        return np.cos(x) + np.cos(y) + np.cos(z)
    if kind == "iwp":
        return (2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z) + np.cos(z) * np.cos(x))
                - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))
    if kind == "fischerKochS":
        return (np.cos(2 * x) * np.sin(y) * np.cos(z)
                + np.cos(2 * y) * np.sin(z) * np.cos(x)
                + np.cos(2 * z) * np.sin(x) * np.cos(y))
    if kind == "neovius":
        return (3 * (np.cos(x) + np.cos(y) + np.cos(z))
                + 4 * np.cos(x) * np.cos(y) * np.cos(z))
    if kind == "lidinoid":
        return (0.5 * (np.sin(2 * x) * np.cos(y) * np.sin(z)
                       + np.sin(2 * y) * np.cos(z) * np.sin(x)
                       + np.sin(2 * z) * np.cos(x) * np.sin(y))
                - 0.5 * (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
                         + np.cos(2 * z) * np.cos(2 * x))
                + 0.15)
    if kind == "splitP":
        return (1.1 * (np.sin(2 * x) * np.sin(z) * np.cos(y)
                       + np.sin(2 * y) * np.sin(x) * np.cos(z)
                       + np.sin(2 * z) * np.sin(y) * np.cos(x))
                - 0.2 * (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
                         + np.cos(2 * z) * np.cos(2 * x))
                - 0.4 * (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))
    raise ValueError(f"unknown topology {kind}")


def solid_mask(field, c, mode):
    """`network` (skeletal) keeps one side of the surface; `sheet` keeps a shell around it."""
    return np.abs(field) <= c if mode == "sheet" else field <= c


def build_at_porosity(kind, mode, target_porosity, n=20, cells=1, tol=2e-3):
    """Bisect the level-set threshold until the voxel model hits the target porosity."""
    field = tpms_field(kind, n, cells)
    lo, hi = float(field.min()), float(field.max())
    if mode == "sheet":
        lo, hi = 0.0, float(np.abs(field).max())

    mask = None
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        mask = solid_mask(field, mid, mode)
        porosity = 1.0 - mask.mean()
        if abs(porosity - target_porosity) < tol:
            break
        # raising the threshold always adds solid, i.e. lowers porosity
        if porosity > target_porosity:
            lo = mid
        else:
            hi = mid
    return mask


# --------------------------------------------------------------------------
# FDM / melt-extrusion strut lattices
#
# This is what the biopolymer bone-scaffold literature actually prints: layers of
# parallel cylindrical struts, rotated 90 deg each layer. Its load path is very
# different from a TPMS, so it lands in a different Gibson-Ashby regime - which is
# exactly why the simulated tier needs it to be transferable to literature data.
# --------------------------------------------------------------------------

def fdm_lattice(n, strut_d, spacing, layer_ratio=0.8, stagger=False):
    """
    0/90 lay-down lattice with FUSED layers.

    Layer height is tied to filament diameter (h = layer_ratio * strut_d, with
    layer_ratio < 1), which is what actually happens in melt extrusion: the filament
    is squashed onto the layer below, so consecutive layers always bond. Treating
    layer count as independent of strut diameter is wrong - thin the struts and the
    print separates into a stack of unbonded rods with no through-thickness stiffness.

    Porosity is controlled by in-plane strut SPACING, not by thinning the filament.
    """
    lin = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    r = strut_d / 2.0
    h = layer_ratio * strut_d

    n_layers = max(1, int(round(1.0 / h)))
    h = 1.0 / n_layers                            # renormalise to fill the cube exactly
    layer = np.clip((z / h).astype(int), 0, n_layers - 1)

    solid = np.zeros_like(x, dtype=bool)
    n_struts = max(1, int(round(1.0 / spacing)))
    s = 1.0 / n_struts
    for k in range(n_layers):
        sel = layer == k
        if not sel.any():
            continue
        t = y if k % 2 == 0 else x                # even layers along x, odd along y
        off = 0.5 * s if (stagger and (k // 2) % 2 == 1) else 0.0
        axes = (np.arange(n_struts) + 0.5) * s + off
        d = np.abs(t[sel][..., None] - axes)
        d = np.minimum(d, 1.0 - d).min(axis=-1)   # periodic wrap
        dz = z[sel] - (k + 0.5) * h
        solid[sel] = np.sqrt(d ** 2 + dz ** 2) <= r
    return solid


def build_fdm_at_porosity(target_porosity, n=20, strut_d=0.25, layer_ratio=0.8,
                          stagger=False, tol=3e-3):
    """Bisect in-plane strut spacing (filament diameter fixed) to hit target porosity."""
    lo, hi = strut_d, 8.0 * strut_d               # spacing below strut_d would overlap
    mask = None
    for _ in range(40):
        s = 0.5 * (lo + hi)
        mask = fdm_lattice(n, strut_d, s, layer_ratio, stagger)
        porosity = 1.0 - mask.mean()
        if abs(porosity - target_porosity) < tol:
            break
        if porosity > target_porosity:
            hi = s                                # tighter spacing -> less porosity
        else:
            lo = s
    return mask


# --------------------------------------------------------------------------
# Trilinear hexahedral element
# --------------------------------------------------------------------------

NODE_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                      dtype=float)  # natural coords of the 8 nodes, matching node_ids()


def elasticity_matrix(E=1.0, nu=0.3):
    """Isotropic 6x6 constitutive matrix, Voigt order [xx, yy, zz, xy, yz, xz]."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.array([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ])


def strain_displacement(dNdx):
    """Assemble the 6x24 B matrix from nodal shape-function gradients."""
    B = np.zeros((6, 24))
    for i in range(8):
        bx, by, bz = dNdx[i]
        B[0, 3 * i + 0] = bx
        B[1, 3 * i + 1] = by
        B[2, 3 * i + 2] = bz
        B[3, 3 * i + 0] = by; B[3, 3 * i + 1] = bx
        B[4, 3 * i + 1] = bz; B[4, 3 * i + 2] = by
        B[5, 3 * i + 0] = bz; B[5, 3 * i + 2] = bx
    return B


def hex8_B_centroid(h=1.0):
    """B matrix evaluated at the element centroid - used for stress recovery."""
    dN = 0.125 * NODE_SIGNS                      # dN/dnatural at (0,0,0)
    return strain_displacement(dN * (2.0 / h))


def hex8_stiffness(E=1.0, nu=0.3, h=1.0):
    """8-node brick stiffness by 2x2x2 Gauss quadrature on a cube of side h."""
    D = elasticity_matrix(E, nu)
    sgn = NODE_SIGNS
    g = 1.0 / np.sqrt(3.0)
    Ke = np.zeros((24, 24))
    detJ = (h / 2.0) ** 3
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                p = np.array([xi, eta, zeta])
                # dN/dnatural
                dN = np.empty((8, 3))
                for i, s in enumerate(sgn):
                    dN[i, 0] = 0.125 * s[0] * (1 + s[1] * p[1]) * (1 + s[2] * p[2])
                    dN[i, 1] = 0.125 * (1 + s[0] * p[0]) * s[1] * (1 + s[2] * p[2])
                    dN[i, 2] = 0.125 * (1 + s[0] * p[0]) * (1 + s[1] * p[1]) * s[2]
                B = strain_displacement(dN * (2.0 / h))  # jacobian is diagonal h/2
                Ke += B.T @ D @ B * detJ
    return Ke


def node_ids(n):
    """Element -> 8 node indices on an (n+1)^3 node grid, matching hex8_stiffness order."""
    nn = n + 1
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    ids = np.empty((i.size, 8), dtype=np.int64)
    col = 0
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                ids[:, col] = (i + di) * nn * nn + (j + dj) * nn + (k + dk)
                col += 1
    return ids


# --------------------------------------------------------------------------
# Uniaxial compression FEM
# --------------------------------------------------------------------------

CG_RTOL = 1e-6
"""
Conjugate-gradient tolerance for the compression solve.

Calibrated, not guessed: against a void_ratio -> 0, rtol = 1e-10 reference solve, 1e-6
reproduces E_rel to all seven printed figures on gyroid (P = 0.70 and 0.85), IWP-sheet
(P = 0.60) and 0/90 FDM (P = 0.72), while cutting CG iterations by 7-30% depending on
geometry. E_rel is an energy and converges quadratically, so it is insensitive to this;
K_sc is a derivative of the displacement field and tracks the tolerance one-for-one,
reaching 4e-6 here - still two orders below the third decimal the sweep writes it out
to. sim/benchmark_solver.py reproduces the comparison.
"""


def _element_dofs(ids):
    """(nel, 8) node ids -> (nel, 24) dof ids, in the node order hex8_stiffness assumes."""
    edof = np.empty((ids.shape[0], 24), dtype=np.int64)
    edof[:, 0::3] = ids * 3
    edof[:, 1::3] = ids * 3 + 1
    edof[:, 2::3] = ids * 3 + 2
    return edof


def assemble_compression(mask, nu=0.3, void_ratio=0.0):
    """
    Global stiffness for a voxel mask, independent of which axis will be loaded.

    Two assembly modes, and the default changed:

    `void_ratio = 0` (default) assembles the SOLID PHASE ONLY. Void voxels are not
    elements at all and their exclusive nodes carry no equations, so the system is the
    scaffold skeleton and nothing else.

    `void_ratio > 0` reproduces the older ersatz-stiffness assembly, in which every
    voxel is an element and void ones are given a token modulus to keep the matrix
    non-singular. That is a numerical regularisation, and it is not free: the ersatz
    phase both carries a little load and constrains the solid, and against a
    void_ratio -> 0 reference it inflates E_rel by 0.003% at P = 0.70 and by 0.04% at
    P = 0.85 - small, but growing an order of magnitude across the porosity range, i.e.
    worst exactly where the modulus is smallest and the scaffold literature is densest.
    It also triples the non-zeros. It is kept only as a fallback for geometries whose
    solid phase floats free (see `compression_response_axes`) and to reproduce earlier
    runs.

    Returns everything the per-axis solve needs, so a mask loaded along several axes is
    assembled once.
    """
    n = mask.shape[0]
    nn = n + 1
    ndof = 3 * nn ** 3
    h = 1.0 / n

    Ke = hex8_stiffness(1.0, nu, h)
    ids_all = node_ids(n)
    flat = mask.ravel()

    if void_ratio > 0:
        ids = ids_all
        dens = np.where(flat, 1.0, void_ratio)
        is_solid = flat
    else:
        keep = np.flatnonzero(flat)
        ids = ids_all[keep]
        dens = np.ones(keep.size)
        is_solid = np.ones(keep.size, dtype=bool)

    edof = _element_dofs(ids)
    if edof.shape[0] == 0:                     # nothing solid: no system to build
        K = sparse.csr_matrix((ndof, ndof))
        active = np.zeros(nn ** 3, dtype=bool)
    else:
        rows = np.repeat(edof, 24, axis=1).ravel()
        cols = np.tile(edof, (1, 24)).ravel()
        vals = (dens[:, None] * Ke.ravel()[None, :]).ravel()
        K = sparse.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
        active = np.zeros(nn ** 3, dtype=bool)
        active[np.unique(ids)] = True

    return dict(K=K, edof=edof, dens=dens, is_solid=is_solid, active=active,
                n=n, nn=nn, ndof=ndof, h=h, nu=nu)


def _cyclic_shift(u, nn):
    """
    Turn a displacement field one step around the x -> y -> z -> x rotation.

    If the geometry is invariant under that rotation, the image of the solution for a
    compression along `axis` IS the solution along (axis + 1) % 3 - not an approximation
    of it. Both the node grid and the three displacement components have to be permuted
    together, which is what the transpose and the component reindex do.
    """
    U = u.reshape(nn, nn, nn, 3)
    return np.transpose(U, (2, 0, 1, 3))[..., [2, 0, 1]].ravel()


def _is_cyclic(mask):
    """True if the voxel geometry is unchanged by the x -> y -> z -> x rotation."""
    return bool((mask.transpose(1, 2, 0) == mask).all())


def _solve_axis(sys_, axis, Es=1.0, rtol=CG_RTOL, x0=None):
    """Uniaxial compression of an assembled system along `axis`. See compression_response."""
    K, n, nn, ndof = sys_["K"], sys_["n"], sys_["nn"], sys_["ndof"]

    grid_idx = np.unravel_index(np.arange(nn ** 3), (nn, nn, nn))
    fixed = np.concatenate([
        3 * np.flatnonzero(grid_idx[d] == 0) + d for d in range(3)  # symmetry rollers
    ])
    top = np.flatnonzero(grid_idx[axis] == n)
    top_dof = 3 * top + axis

    delta = 1e-3                               # prescribed compressive strain of 0.1%
    u = np.zeros(ndof)
    u[top_dof] = -delta

    constrained = np.unique(np.concatenate([fixed, top_dof]))
    # Only nodes carrying an equation are unknowns. Under solid-only assembly that
    # excludes every purely-void node, which is where the speed comes from: at P = 0.70
    # it is a third of the degrees of freedom and a third of the non-zeros.
    free = np.setdiff1d(np.flatnonzero(np.repeat(sys_["active"], 3)), constrained)

    info = 0
    if free.size:
        f = -(K[:, constrained] @ u[constrained])
        Kff = K[free][:, free]
        diag = Kff.diagonal()
        diag[diag == 0] = 1.0
        M = LinearOperator(Kff.shape, matvec=lambda v: v / diag)
        uf, info = cg(Kff, f[free], x0=(None if x0 is None else x0[free]),
                      rtol=rtol, maxiter=8000, M=M)
        if not np.all(np.isfinite(uf)):
            info = -1
        else:
            u[free] = uf

    reaction = float((K[top_dof, :] @ u).sum())
    macro_stress = abs(reaction / 1.0)         # unit cross-sectional area
    strain = delta / 1.0                       # unit height
    E_rel = macro_stress / strain * Es

    # ---- stress recovery at element centroids, for the strength proxy ----
    edof, dens, is_solid = sys_["edof"], sys_["dens"], sys_["is_solid"]
    if edof.shape[0] and is_solid.any() and macro_stress > 0:
        D = elasticity_matrix(1.0, sys_["nu"])
        B = hex8_B_centroid(sys_["h"])
        eps = u[edof] @ B.T                     # (nel, 6)
        sig = (eps @ D.T) * dens[:, None]       # ersatz-scaled, so void carries ~nothing
        vm = np.sqrt(0.5 * ((sig[:, 0] - sig[:, 1]) ** 2 + (sig[:, 1] - sig[:, 2]) ** 2
                            + (sig[:, 2] - sig[:, 0]) ** 2)
                     + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))
        k_sc = float(np.percentile(vm[is_solid], 95) / macro_stress)
    else:
        k_sc = np.nan

    return dict(E_rel=E_rel, cg_info=info, stress_concentration=k_sc, u=u)


def apparent_modulus(mask, Es=1.0, nu=0.3, void_ratio=0.0, axis=2):
    """Backwards-compatible wrapper: returns (E_rel, cg_info) only."""
    r = compression_response(mask, Es, nu, void_ratio, axis)
    return r["E_rel"], r["cg_info"]


def compression_response(mask, Es=1.0, nu=0.3, void_ratio=0.0, axis=2, rtol=CG_RTOL):
    """
    Apparent Young's modulus under uniaxial compression along `axis`, normalised by Es.

    Symmetry rollers on the x=0, y=0 and z=0 faces; prescribed displacement on the
    far face of the loading axis. This mirrors a physical platen compression test, so
    results are directly comparable to published compressive-modulus values (unlike
    periodic homogenisation).

    `axis` is 0/1/2 for x/y/z. Loading along more than one axis is how the orthotropy
    of a 0/90 lay-down is measured: a print is much stiffer across the filaments than
    along them, and that ratio is a design variable the literature rarely reports. Use
    `compression_response_axes` when several axes are wanted - it assembles once.

    Also returns a STRESS CONCENTRATION FACTOR: the 95th-percentile von Mises stress
    in the solid phase divided by the applied macroscopic stress. Strength then follows
    as sigma_scaffold ~ sigma_yield_solid / K_sc, which is a defensible first-order
    strength predictor from a purely linear-elastic solve - no plasticity model needed.
    A high K_sc means the architecture funnels load through a few sharp junctions and
    will fail early even if its modulus looks good.
    """
    return compression_response_axes(mask, (axis,), Es, nu, void_ratio, rtol)[axis]


def compression_response_axes(mask, axes=(2,), Es=1.0, nu=0.3, void_ratio=0.0,
                              rtol=CG_RTOL):
    """
    Compression response along several axes from ONE assembly. Returns {axis: result}.

    A cubic geometry gets its later axes almost for nothing. Every TPMS level set here
    is invariant under the x -> y -> z -> x rotation - on the voxel grid, not merely in
    the continuum - so rotating the solved displacement field lands exactly on the next
    axis's solution and CG stops at iteration ZERO, having checked the residual itself.
    That is the point: it is offered as a starting guess, never asserted, so a geometry
    that is not actually symmetric (a 0/90 lay-down is not) simply costs what it always
    did. The invariance is tested per mask rather than inferred from the topology name,
    which also keeps it honest for a bisection that happened to land off-symmetry.

    Solid-only assembly drops the ersatz void that used to keep the matrix
    non-singular, so a solid phase that floats free of every boundary condition can
    leave the system singular. That is rare - it needs an island touching the loaded
    face and nothing else - but a sweep must not fall over on it, so a non-converged
    solve is retried on the regularised assembly rather than returned.
    """
    sys_ = assemble_compression(mask, nu, void_ratio)
    cyclic = _is_cyclic(mask)
    nn = sys_["nn"]

    out, solved = {}, {}
    for ax in axes:
        x0 = None
        first = next(iter(solved.items()), None)
        if cyclic and first is not None:
            src, x0 = first
            for _ in range((ax - src) % 3):
                x0 = _cyclic_shift(x0, nn)
        out[ax] = _solve_axis(sys_, ax, Es, rtol, x0=x0)
        solved[ax] = out[ax]["u"]

    if void_ratio <= 0 and any(r["cg_info"] != 0 for r in out.values()):
        fallback = assemble_compression(mask, nu, void_ratio=1e-6)
        for ax, r in out.items():
            if r["cg_info"] != 0:
                out[ax] = _solve_axis(fallback, ax, Es, rtol)
                out[ax]["regularised"] = True
    for r in out.values():
        r.pop("u", None)                       # the field itself is not part of the result
    return out


# --------------------------------------------------------------------------
# Transport: effective diffusivity of the pore phase
# --------------------------------------------------------------------------

def effective_diffusivity(mask, axis=2):
    """
    Solve a steady Laplace problem on the VOID phase across the `axis` direction.
    Returns D_eff/D_bulk. Tortuosity follows as porosity / D_eff.
    Disconnected pores are handled by the ersatz conductivity.
    """
    n = mask.shape[0]
    void = ~mask
    sig = np.where(void, 1.0, 1e-6).ravel()
    N = n ** 3
    idx = np.arange(N).reshape(n, n, n)

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    def harm(a, c):
        return 2 * a * c / (a + c)

    # `d` deliberately, not `axis`: this loop sums conductances over all three lattice
    # directions and must not clobber the transport direction the caller asked for.
    # It used to, so D_eff came back along z whatever `axis` said, and the anisotropy
    # of a 0/90 lay-down was silently reported as isotropic.
    for d in range(3):
        a_idx = np.take(idx, np.arange(n - 1), axis=d).ravel()
        b_idx = np.take(idx, np.arange(1, n), axis=d).ravel()
        k = harm(sig[a_idx], sig[b_idx])
        rows.extend([a_idx, b_idx, a_idx, b_idx])
        cols.extend([a_idx, b_idx, b_idx, a_idx])
        vals.extend([k, k, -k, -k])

    # Dirichlet: phi=1 on the inlet layer, phi=0 on the outlet layer
    lo = np.take(idx, 0, axis=axis).ravel()
    hi = np.take(idx, n - 1, axis=axis).ravel()
    big = 1e6
    rows.append(lo); cols.append(lo); vals.append(np.full(lo.size, big))
    rows.append(hi); cols.append(hi); vals.append(np.full(hi.size, big))
    b[lo] = big * 1.0

    A = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N)).tocsr()
    diag = A.diagonal(); diag[diag == 0] = 1.0
    M = LinearOperator(A.shape, matvec=lambda v: v / diag)
    phi, _ = cg(A, b, rtol=1e-8, maxiter=5000, M=M)
    phi = phi.reshape(n, n, n)

    # flux across the mid-plane normal to the transport direction
    mid = n // 2
    sig3 = sig.reshape(n, n, n)
    k = harm(np.take(sig3, mid - 1, axis=axis), np.take(sig3, mid, axis=axis))
    flux = float((k * (np.take(phi, mid - 1, axis=axis) - np.take(phi, mid, axis=axis))).sum())
    # Normalise by the flux an identically sized fully-open channel would carry:
    # n^2 faces of unit conductance, each seeing a potential drop of 1/(n-1).
    return flux * (n - 1) / (n * n) if n > 1 else 0.0


def specific_surface(mask):
    """Solid-void interface area per unit total volume (voxel faces, normalised)."""
    n = mask.shape[0]
    faces = 0
    for axis in range(3):
        a = np.take(mask, np.arange(n - 1), axis=axis)
        b = np.take(mask, np.arange(1, n), axis=axis)
        faces += int((a ^ b).sum())
    return faces * (1.0 / n) ** 2 / 1.0


def morphometry(mask, domain_mm=2.0):
    """
    Pore diameter and strut thickness by maximum-inscribed-sphere, in microns.

    This is the same quantity a micro-CT morphometry package reports, so it is directly
    comparable to the `pore_size_um` / `strut_thickness_um` fields curated from papers -
    unlike a nominal pore size back-computed from unit-cell size, which is a design input
    rather than a measurement. The 95th percentile is used instead of the max because a
    single voxel of noise sets the max.
    """
    from scipy.ndimage import distance_transform_edt
    vox_um = domain_mm * 1000.0 / mask.shape[0]

    def thickness(phase):
        if not phase.any():
            return 0.0
        d = distance_transform_edt(phase)
        return float(2.0 * np.percentile(d[phase], 95) * vox_um)

    return dict(pore_size_um=round(thickness(~mask), 1),
                strut_thickness_um=round(thickness(mask), 1))


def largest_connected_fraction(mask):
    """Fraction of solid voxels in the largest 6-connected cluster (printability proxy)."""
    from scipy.ndimage import label
    lab, nlab = label(mask)
    if nlab == 0:
        return 0.0
    counts = np.bincount(lab.ravel())[1:]
    return float(counts.max() / mask.sum())
