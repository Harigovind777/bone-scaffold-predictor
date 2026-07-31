"""
Full synthetic bone scaffold dataset generator.

Extends the original Gibson-Ashby mechanical model (compressive_strength,
elastic_modulus -- these have real physics backing) with literature-informed
HEURISTIC formulas for the three properties that have no clean physics
equation: degradation_rate_pct_per_week, cell_viability_pct,
bone_regeneration_score.

IMPORTANT / HONESTY NOTE:
These three heuristic targets are NOT experimentally validated data. They
encode qualitative relationships reported in tissue-engineering literature
(e.g. Karageorgiou & Kaplan 2005 on pore size vs. bone ingrowth; HA buffering
of acidic polyester degradation byproducts; optimal porosity ranges for
vascularization) as smooth functions with added noise, so a model can be
demonstrated end-to-end. They should be replaced with real experimental data
as it becomes available. The `data_provenance` column on every row marks
which columns are physics-grounded vs heuristic.

Gibson-Ashby open-cell foam relations (physics-grounded):
    E_scaffold / E_solid = (rho_scaffold / rho_solid) ** n1   (n1 ~ 2.0)
    S_scaffold / S_solid = (rho_scaffold / rho_solid) ** n2   (n2 ~ 1.5)
"""

import csv
import math
import random

random.seed(42)

# Bulk (fully dense) reference material properties.
# modulus/strength in MPa, density in g/cm3.
MATERIALS = {
    "PLA":      {"density": 1.24, "modulus": 3500,  "strength": 65,
                 "degr_base": 1.2, "biocompat": 0.75, "acid_producing": True},
    "PCL":      {"density": 1.145, "modulus": 400,   "strength": 40,
                 "degr_base": 0.25, "biocompat": 0.80, "acid_producing": True},
    "PGA":      {"density": 1.53, "modulus": 3500,  "strength": 100,
                 "degr_base": 7.5, "biocompat": 0.65, "acid_producing": True},
    "PLGA":     {"density": 1.34, "modulus": 2000,  "strength": 50,
                 "degr_base": 3.5, "biocompat": 0.70, "acid_producing": True},
    "Chitosan": {"density": 1.40, "modulus": 3000,  "strength": 35,
                 "degr_base": 2.0, "biocompat": 0.85, "acid_producing": False},
    "Gelatin":  {"density": 1.35, "modulus": 150,   "strength": 10,
                 "degr_base": 6.0, "biocompat": 0.85, "acid_producing": False},
    "HA":       {"density": 3.15, "modulus": 90000, "strength": 300,
                 "degr_base": 0.05, "biocompat": 0.90, "acid_producing": False},
}

PRINTING_METHODS = ["FDM", "SLA", "SLS", "salt leaching", "freeze drying", "electrospinning"]

# Cytotoxicity/thermal-stress modifier per printing method (heuristic, 0 = neutral)
METHOD_VIABILITY_MODIFIER = {
    "FDM": -0.05,
    "SLA": -0.08,
    "SLS": -0.10,
    "salt leaching": -0.02,
    "freeze drying": 0.02,
    "electrospinning": -0.03,
}

N_MODULUS = 2.0
N_STRENGTH = 1.5


def blend_material(base_polymer, ha_wt_pct):
    base = MATERIALS[base_polymer]
    ha = MATERIALS["HA"]
    f = ha_wt_pct / 100.0
    return {
        "density":  base["density"] * (1 - f) + ha["density"] * f,
        "modulus":  base["modulus"] * (1 - f) + ha["modulus"] * f,
        "strength": base["strength"] * (1 - f) + ha["strength"] * f,
    }


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def gaussian_factor(x, center, width):
    """0-1 bell curve, 1.0 at center, decaying with distance."""
    return math.exp(-((x - center) / width) ** 2)


def compute_mechanical(base_polymer, ha_wt_pct, porosity_pct, noise):
    solid = blend_material(base_polymer, ha_wt_pct) if ha_wt_pct > 0 else MATERIALS[base_polymer]
    relative_density = 1 - (porosity_pct / 100.0)
    scaffold_density = solid["density"] * relative_density

    modulus = solid["modulus"] * (relative_density ** N_MODULUS)
    strength = solid["strength"] * (relative_density ** N_STRENGTH)

    modulus *= random.uniform(1 - noise, 1 + noise)
    strength *= random.uniform(1 - noise, 1 + noise)
    return scaffold_density, strength, modulus


def compute_degradation_rate(base_polymer, ha_wt_pct, porosity_pct, noise):
    """Heuristic: base polymer resorption rate, buffered by HA (for acid-
    producing polyesters), accelerated by higher porosity (more hydrolysis
    surface area)."""
    mat = MATERIALS[base_polymer]
    f = ha_wt_pct / 100.0

    ha_buffer = (1 - 0.4 * f) if mat["acid_producing"] else (1 - 0.1 * f)
    porosity_factor = 1 + 0.8 * ((porosity_pct / 100.0) - 0.5)

    rate = mat["degr_base"] * ha_buffer * porosity_factor
    rate *= random.uniform(1 - noise, 1 + noise)
    return clamp(rate, 0.05, 15.0)


def compute_cell_viability(base_polymer, ha_wt_pct, pore_size_um, porosity_pct,
                            printing_method, noise):
    """Heuristic: optimal pore size ~300um (Karageorgiou & Kaplan 2005),
    optimal porosity band for nutrient/oxygen diffusion ~60-85%, base
    material biocompatibility, small HA osteoconductive bonus, printing
    method cytotoxicity/thermal-stress modifier."""
    mat = MATERIALS[base_polymer]
    f = ha_wt_pct / 100.0

    pore_factor = gaussian_factor(pore_size_um, center=300, width=250)
    porosity_factor = gaussian_factor(porosity_pct, center=72, width=35)

    base = 45 + 35 * (0.5 * pore_factor + 0.5 * porosity_factor)
    base *= (0.7 + 0.3 * mat["biocompat"] / 0.9)  # normalize biocompat to ~[0.77,1.0] multiplier
    base += 8 * f  # HA osteoconductive bonus, diminishing already via small coefficient
    base += 100 * METHOD_VIABILITY_MODIFIER.get(printing_method, 0.0)

    base *= random.uniform(1 - noise, 1 + noise)
    return clamp(base, 20.0, 98.0)


def compute_bone_regeneration_score(strength, modulus, pore_size_um, porosity_pct,
                                     degradation_rate, cell_viability, ha_wt_pct, noise):
    """Heuristic composite (0-100): mechanical fit to trabecular bone range,
    pore-size/porosity osteoconductivity, degradation rate matched to new-bone
    ingrowth timeline (~8-16 weeks -> ideal ~1.5-3 %/week), cell viability,
    and HA osteoconductive bonus."""
    f = ha_wt_pct / 100.0

    # Trabecular bone: strength ~2-12 MPa, modulus ~50-500 MPa (log-scale fit)
    strength_factor = gaussian_factor(math.log10(max(strength, 0.01)), center=math.log10(7), width=0.6)
    modulus_factor = gaussian_factor(math.log10(max(modulus, 0.01)), center=math.log10(200), width=0.7)

    pore_factor = gaussian_factor(pore_size_um, center=350, width=220)
    porosity_factor = gaussian_factor(porosity_pct, center=75, width=30)
    degradation_factor = gaussian_factor(degradation_rate, center=2.2, width=2.0)
    viability_factor = cell_viability / 100.0
    ha_bonus = 0.10 * min(f, 0.25) / 0.25  # saturating bonus, HA >25wt% adds little further

    score = 100 * (
        0.20 * strength_factor +
        0.15 * modulus_factor +
        0.15 * pore_factor +
        0.15 * porosity_factor +
        0.15 * degradation_factor +
        0.20 * viability_factor
    )
    score += 100 * ha_bonus * 0.05
    score *= random.uniform(1 - noise, 1 + noise)
    return clamp(score, 0.0, 100.0)


def generate_row(base_polymer, ha_wt_pct, porosity_pct, pore_size_um,
                  layer_thickness_um, printing_method, noise=0.15):
    density, strength, modulus = compute_mechanical(base_polymer, ha_wt_pct, porosity_pct, noise)
    degradation_rate = compute_degradation_rate(base_polymer, ha_wt_pct, porosity_pct, noise)
    cell_viability = compute_cell_viability(base_polymer, ha_wt_pct, pore_size_um,
                                             porosity_pct, printing_method, noise)
    regen_score = compute_bone_regeneration_score(strength, modulus, pore_size_um, porosity_pct,
                                                   degradation_rate, cell_viability, ha_wt_pct, noise)

    composition = base_polymer if ha_wt_pct == 0 else f"{base_polymer}/HA ({ha_wt_pct}wt% HA)"

    return {
        "source": "SYNTHETIC_GIBSON_ASHBY_PLUS_HEURISTIC",
        "biopolymer_type": base_polymer if ha_wt_pct == 0 else f"{base_polymer}/HA",
        "material_composition": composition,
        "ha_wt_pct": ha_wt_pct,
        "pore_size_um": round(pore_size_um, 2),
        "porosity_pct": round(porosity_pct, 2),
        "printing_method": printing_method,
        "layer_thickness_um": round(layer_thickness_um, 2),
        "density_g_cm3": round(density, 3),
        "compressive_strength_MPa": round(strength, 2),
        "elastic_modulus_MPa": round(modulus, 1),
        "degradation_rate_pct_per_week": round(degradation_rate, 3),
        "cell_viability_pct": round(cell_viability, 2),
        "bone_regeneration_score": round(regen_score, 2),
        "data_provenance": (
            "compressive_strength_MPa,elastic_modulus_MPa=physics(Gibson-Ashby); "
            "degradation_rate_pct_per_week,cell_viability_pct,bone_regeneration_score=heuristic_literature_informed"
        ),
    }


def generate_dataset(n_rows=4000, out_path="data/full_scaffold_dataset.csv"):
    rows = []
    polymers = ["PLA", "PCL", "PGA", "PLGA", "Chitosan", "Gelatin"]
    for _ in range(n_rows):
        polymer = random.choice(polymers)
        ha_pct = random.choice([0, 0, 10, 15, 20, 30])
        porosity = random.uniform(30, 90)
        pore_size = random.uniform(80, 700)
        layer_thickness = random.uniform(100, 400)
        method = random.choice(PRINTING_METHODS)
        rows.append(generate_row(polymer, ha_pct, porosity, pore_size, layer_thickness, method))

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    generate_dataset(n_rows=4000, out_path="data/full_scaffold_dataset.csv")
