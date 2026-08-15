"""
Clean the MLATE 3D-(bio)printed scaffold dataset into a modelling-ready table.

Source: Rafieyan et al., Biofabrication 16 (2024) 045014, https://github.com/saeedrafieyan/MLATE
1171 formulations x 88 publications, with ordinal Printability (0-3), Cell Response (1-5)
and Scaffold Quality targets.

This dataset covers the BIOLOGICAL / PRINTABILITY arm of the framework only - it carries
no mechanical or degradation measurements. It is used here to (a) get a validated baseline
running in week 1, and (b) demonstrate publication-grouped cross-validation, since the DOI
column lets us group correlated rows properly.
"""

import numpy as np
import pandas as pd

SRC = ("/private/tmp/claude-501/-Users-harigovind-Desktop-Demo/"
       "bc5e456a-0f3d-40de-a440-48213c3bbfea/scratchpad/MLATE-main/V2/df.xlsx")

# Components that put a formulation in scope for a load-bearing bone scaffold.
BONE_COMPONENTS = [
    "PCL (%w/v)", "PLGA (%w/v)", "hydroxyapatite (%w/v)", "TCP (%w/v)",
    "bioactive glass (%w/v)", "Chitosan (%w/v)", "Collagen (%w/v)",
    "Fibroin/Fibrinogen (%w/v)", "PVA-HA (%w/v)", "graphene oxide (%w/v)",
    "laponite (%w/v)",
]

PROCESS = [
    "Cell_Density_(cells/mL)", "Physical_Crosslinking_Durantion_(s)",
    "Photocrosslinking_Duration_(s)", "Extrusion_Pressure (kPa)",
    "Extrusion_Rate_Lengthwise_(mm/s)", "Extrusion_Rate_Volume-wise_(mL/s)",
    "Nozzle_Movement_Speed_(mm/s)", "Nozzle_Diameter_(µm)",
    "chamber Temperature  (°C)", "Syringe_Temperature_(°C)",
    "Substrate_Temperature_(°C)",
]

TARGETS = ["Printability", "Cell Response", "Scaffold Quality (P*C)"]


def main():
    df = pd.read_excel(SRC)
    comp_cols = [c for c in df.columns
                 if ("%w/v" in c or "%wt" in c or "(mM)" in c or "(U/ml)" in c or "(M)" in c)]

    out = df.copy()
    out[comp_cols] = out[comp_cols].fillna(0.0)

    # Engineered descriptors - these are what actually help the model generalise
    out["total_solids_pct"] = out[[c for c in comp_cols if "%w/v" in c]].sum(axis=1)
    out["n_components"] = (out[comp_cols] > 0).sum(axis=1)
    out["ceramic_pct"] = out[["hydroxyapatite (%w/v)", "TCP (%w/v)",
                              "bioactive glass (%w/v)"]].sum(axis=1)
    out["ceramic_fraction"] = out["ceramic_pct"] / out["total_solids_pct"].replace(0, np.nan)
    out["is_bone_relevant"] = (out[BONE_COMPONENTS] > 0).any(axis=1).astype(int)
    out["has_cells"] = (out["Cell_Density_(cells/mL)"].fillna(0) > 0).astype(int)

    keep = (["Reference", "DOI"] + comp_cols + PROCESS + TARGETS
            + ["total_solids_pct", "n_components", "ceramic_pct",
               "ceramic_fraction", "is_bone_relevant", "has_cells"])
    out = out[keep]

    out.to_csv("data/mlate_full.csv", index=False)
    bone = out[out.is_bone_relevant == 1]
    bone.to_csv("data/mlate_bone_subset.csv", index=False)

    print(f"data/mlate_full.csv         {out.shape}  DOIs={out.DOI.nunique()}")
    print(f"data/mlate_bone_subset.csv  {bone.shape}  DOIs={bone.DOI.nunique()}")
    print("\nTarget balance (full):")
    for t in TARGETS:
        print(f"  {t:26s} {out[t].value_counts().sort_index().to_dict()}")
    print("\nMissingness in process columns (full):")
    miss = out[PROCESS].isna().mean().sort_values(ascending=False)
    print((miss * 100).round(1).to_string())


if __name__ == "__main__":
    main()
