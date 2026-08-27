"""Paths, seeds and the feature contracts every stage agrees on."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SIMULATED = DATA / "simulated_all.csv"
CONVERGENCE = DATA / "mesh_convergence.csv"
GA_FITS = DATA / "gibson_ashby_fits.csv"
MLATE_FULL = DATA / "mlate_full.csv"
MLATE_BONE = DATA / "mlate_bone_subset.csv"
CURATION_SCHEMA = DATA / "curation_schema.csv"
TIER3 = DATA / "tier3_curated.csv"          # created by the curation team; optional

SEED = 20260808

# --------------------------------------------------------------------------
# Feature contracts
#
# DESIGN features are knowable BEFORE any solve - they are geometry you specify or
# measure off the voxel model in milliseconds. Only these may feed a model used for
# inverse design, otherwise the "design" requires the FEM answer it is meant to replace.
#
# SOLVED features come out of a Laplace or FEM solve. They are legitimate for a
# post-hoc analysis model but NOT for the design surrogate, and mixing the two is the
# easiest way to publish an R^2 that cannot be reproduced at design time.
# --------------------------------------------------------------------------

DESIGN_NUMERIC = [
    "relative_density", "log_relative_density",
    "unit_cell_mm", "n_cells", "strut_d", "strut_spacing", "n_struts", "n_layers",
    "stagger", "specific_surface_per_mm", "pore_size_um", "strut_thickness_um",
]
DESIGN_CATEGORICAL = ["family", "topology", "mode"]
SOLVED_ONLY = ["D_eff_z", "D_eff_x", "D_anisotropy_z_over_x", "tortuosity_z",
               "solid_connectivity", "K_sc_z", "K_sc_x", "anisotropy_z_over_x"]

TARGETS = {
    "E_rel_z": "log",       # spans decades -> fit in log space
    "E_rel_x": "log",
    "K_sc_z": "log",
    "D_eff_z": "identity",
}

# Bone-ingrowth and printability windows used as hard constraints in inverse design.
PORE_WINDOW_UM = (200.0, 600.0)      # <100 blocks vascularisation, >600 loses surface
MIN_STRUT_UM = 150.0                 # a 150 um nozzle is about the practical floor

# Porosity is a constraint in its OWN right, not merely a route to pore size. A 20%-porous
# block can have a perfect 400 um pore and still be useless: there is nowhere near enough
# void volume for vascularised bone, and it will not resorb on any clinical timescale. The
# bone-scaffold literature works at 60-85%; 50% is a permissive floor.
POROSITY_WINDOW = (0.50, 0.90)

# Below this the FEM is reporting the ersatz void stiffness rather than a load path:
# the solid phase does not span the specimen. Two orders of magnitude above the
# void_ratio floor of 1e-6, so it separates cleanly without catching real soft samples.
DEAD_MODULUS = 1e-4
