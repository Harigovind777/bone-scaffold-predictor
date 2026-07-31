"""
Gibson-Ashby synthetic data generator for bone scaffold mechanical properties.

Generates physically-plausible (compressive_strength, elastic_modulus) pairs
from porosity/density inputs, using published bulk (fully dense, 0% porosity)
material properties as the reference point for each biopolymer.

This does NOT generate cell_viability, degradation_rate, or bone_regeneration_score
- there's no clean physics equation for those; they must come from literature mining
(see bone_scaffold_dataset_template.csv) or be modeled separately.

Gibson-Ashby open-cell foam relations:
    E_scaffold / E_solid  = C1 * (rho_scaffold / rho_solid) ** n1
    S_scaffold / S_solid  = C2 * (rho_scaffold / rho_solid) ** n2

Typical exponents for open-cell foams: n1 ~ 2 (modulus), n2 ~ 1.5 (strength).
C1, C2 are geometry-dependent constants, ~0.1-1 for real (imperfect) foams;
we default to 1.0 and let noise + porosity range capture real-world scatter.
"""

import csv
import random

random.seed(42)

# Bulk (fully dense / 0% porosity) reference material properties.
# Sources: literature values for PLA, PCL, PGA, PLGA, HA, chitosan, gelatin.
# modulus in MPa, strength in MPa, density in g/cm3.
# For composites (e.g. PLA/HA), a simple rule-of-mixtures blend is used at
# generation time based on the wt% HA specified.
MATERIALS = {
    "PLA":       {"density": 1.24, "modulus": 3500, "strength": 65},
    "PCL":       {"density": 1.145, "modulus": 400,  "strength": 40},
    "PGA":       {"density": 1.53, "modulus": 3500, "strength": 100},
    "PLGA":      {"density": 1.34, "modulus": 2000, "strength": 50},
    "Chitosan":  {"density": 1.40, "modulus": 3000, "strength": 35},
    "Gelatin":   {"density": 1.35, "modulus": 150,   "strength": 10},
    "HA":        {"density": 3.15, "modulus": 90000, "strength": 300},  # ceramic filler
}

PRINTING_METHODS = ["FDM", "SLA", "SLS", "salt leaching", "freeze drying", "electrospinning"]

# Gibson-Ashby exponents (open-cell foam)
N_MODULUS = 2.0
N_STRENGTH = 1.5

def blend_material(base_polymer, ha_wt_pct):
    """Simple rule-of-mixtures blend of a base polymer with HA filler."""
    base = MATERIALS[base_polymer]
    ha = MATERIALS["HA"]
    f = ha_wt_pct / 100.0
    return {
        "density":  base["density"] * (1 - f) + ha["density"] * f,
        "modulus":  base["modulus"] * (1 - f) + ha["modulus"] * f,
        "strength": base["strength"] * (1 - f) + ha["strength"] * f,
    }

def generate_row(base_polymer, ha_wt_pct, porosity_pct, pore_size_um,
                  layer_thickness_um, printing_method, noise=0.15):
    solid = blend_material(base_polymer, ha_wt_pct) if ha_wt_pct > 0 else MATERIALS[base_polymer]

    relative_density = 1 - (porosity_pct / 100.0)
    scaffold_density = solid["density"] * relative_density

    modulus = solid["modulus"] * (relative_density ** N_MODULUS)
    strength = solid["strength"] * (relative_density ** N_STRENGTH)

    # Add realistic experimental scatter (Gibson-Ashby is an idealization)
    modulus *= random.uniform(1 - noise, 1 + noise)
    strength *= random.uniform(1 - noise, 1 + noise)

    composition = base_polymer if ha_wt_pct == 0 else f"{base_polymer}/HA ({ha_wt_pct}wt% HA)"

    return {
        "source_doi": "SYNTHETIC_GIBSON_ASHBY",
        "source_title": "",
        "biopolymer_type": base_polymer if ha_wt_pct == 0 else f"{base_polymer}/HA",
        "material_composition": composition,
        "pore_size_um": pore_size_um,
        "porosity_pct": porosity_pct,
        "printing_method": printing_method,
        "layer_thickness_um": layer_thickness_um,
        "density_g_cm3": round(scaffold_density, 3),
        "compressive_strength_MPa": round(strength, 2),
        "elastic_modulus_MPa": round(modulus, 1),
        "degradation_rate_pct_per_week": "",
        "cell_viability_pct": "",
        "bone_regeneration_score": "",
        "notes": "synthetic - Gibson-Ashby model, not experimental",
    }

def generate_dataset(n_rows=500, out_path="synthetic_scaffold_data.csv"):
    rows = []
    polymers = ["PLA", "PCL", "PGA", "PLGA", "Chitosan"]
    for _ in range(n_rows):
        polymer = random.choice(polymers)
        ha_pct = random.choice([0, 0, 10, 15, 20, 30])  # weight polymer-only cases more
        porosity = random.uniform(30, 90)               # typical scaffold porosity range
        pore_size = random.uniform(100, 600)             # microns, typical range
        layer_thickness = random.uniform(100, 400)       # microns, typical FDM/SLA range
        method = random.choice(PRINTING_METHODS)
        rows.append(generate_row(polymer, ha_pct, porosity, pore_size, layer_thickness, method))

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic rows to {out_path}")

if __name__ == "__main__":
    generate_dataset(n_rows=500, out_path="synthetic_scaffold_data.csv")
