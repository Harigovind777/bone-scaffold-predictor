"""
Validate a curated Tier 3 sheet against data/curation_schema.csv.

Curation is the project's critical path and it is done by several people over weeks, so
errors are systematic rather than random: one person reads "infill %" as porosity, another
records wet moduli without setting test_condition, a third omits the DOI on rows typed
after lunch. Every one of those silently corrupts the model, and none of them look wrong
in a spreadsheet.

Run it after every curation session:

    .venv/bin/python tools/validate_curation.py data/tier3_curated.csv

Exit status is 1 if any ERROR is found, so it can gate a commit. WARNINGs are judgement
calls worth a second look, not blockers.

It also reports INTER-RATER AGREEMENT wherever two extractors independently coded the same
record_id. Re-extracting ~10% of rows and reporting that number is cheap, and it is a
rigour marker very few undergraduate projects have.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "data" / "curation_schema.csv"

# Physically or clinically implausible values. Deliberately WIDE - the aim is to catch
# unit errors and transcription slips, not to police unusual-but-real results.
RANGES = {
    "porosity": (0.0, 1.0),
    "pore_size_mean": (1.0, 5000.0),
    "strut_diameter": (1.0, 5000.0),
    "compressive_modulus": (0.01, 30000.0),
    "compressive_strength": (0.001, 500.0),
    "ceramic_wt_pct": (0.0, 100.0),
    "polymer_1_wt_pct": (0.0, 100.0),
    "time_point_weeks": (0.0, 260.0),
    "mass_remaining_pct": (0.0, 110.0),
    "Mw_remaining_pct": (0.0, 110.0),
    "modulus_retention_pct": (0.0, 200.0),
    "medium_pH": (1.0, 14.0),
    "viability_pct": (0.0, 150.0),
    "interconnectivity": (0.0, 100.0),
}

AGREEMENT_FIELDS = ["porosity", "pore_size_mean", "compressive_modulus",
                    "ceramic_wt_pct", "time_point_weeks"]


def load_schema():
    s = pd.read_csv(SCHEMA)
    enums = {}
    for _, r in s[s.type == "enum"].iterrows():
        note = str(r.notes)
        if "|" in note:
            # Enum members are the pipe-separated head of the note; prose follows a period.
            head = note.split(".")[0]
            enums[r.column] = {v.strip() for v in head.split("|") if v.strip()}
    return s, enums


def validate(path):
    schema, enums = load_schema()
    df = pd.read_csv(path)
    errors, warnings, info = [], [], []

    # ---- structure ----
    expected = list(schema.column)
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        errors.append(f"missing {len(missing)} schema column(s): {missing[:8]}"
                      + (" ..." if len(missing) > 8 else ""))
    if extra:
        warnings.append(f"{len(extra)} column(s) not in schema: {extra[:8]}")

    required = schema[schema.required == "yes"].column
    for col in required:
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                errors.append(f"{col}: {n} row(s) missing a REQUIRED value")

    # ---- keys ----
    if "record_id" in df.columns:
        dup = df.record_id[df.record_id.duplicated(keep=False)]
        # Duplicated record_ids are only legitimate as double-extraction for agreement.
        if len(dup) and "extractor" in df.columns:
            pairs = df[df.record_id.isin(dup)].groupby("record_id").extractor.nunique()
            bad = pairs[pairs < 2].index.tolist()
            if bad:
                errors.append(f"{len(bad)} record_id(s) duplicated by the SAME extractor "
                              f"(not a double-extraction): {bad[:5]}")

    # ---- enums ----
    for col, allowed in enums.items():
        if col not in df.columns:
            continue
        seen = set(df[col].dropna().astype(str).str.strip()) - {""}
        unknown = seen - allowed
        if unknown:
            warnings.append(f"{col}: value(s) outside the schema enum: {sorted(unknown)[:6]}"
                            f"  (allowed: {sorted(allowed)[:6]})")

    # ---- ranges ----
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        bad = v.notna() & ((v < lo) | (v > hi))
        if bad.any():
            errors.append(f"{col}: {int(bad.sum())} value(s) outside [{lo}, {hi}] "
                          f"e.g. {v[bad].head(3).tolist()} - check units")

    # ---- cross-field logic ----
    if {"porosity", "porosity_method"} <= set(df.columns):
        nominal = df.porosity_method.astype(str).str.strip() == "design_nominal"
        if nominal.any():
            warnings.append(f"{int(nominal.sum())} row(s) use design_nominal porosity. "
                            f"Slicer infill is not measured porosity - shells and "
                            f"over-extrusion break the correspondence. Prefer uCT or "
                            f"gravimetric, and never validate a simulation against these.")

    if {"time_point_weeks", "compressive_modulus"} <= set(df.columns):
        t = pd.to_numeric(df.time_point_weeks, errors="coerce")
        m = pd.to_numeric(df.compressive_modulus, errors="coerce")
        n0 = int(((t == 0) & m.isna()).sum())
        if n0:
            warnings.append(f"{n0} as-fabricated row(s) (week 0) have no modulus - these "
                            f"are the anchor for every trajectory in their series")

    if {"doi", "time_point_weeks"} <= set(df.columns):
        series = df.groupby("doi").time_point_weeks.nunique()
        multi = int((series > 1).sum())
        info.append(f"{multi}/{len(series)} DOIs contribute a multi-time-point series "
                    f"(these are the valuable ones - they yield trajectory targets)")

    if "test_condition" in df.columns:
        vc = df.test_condition.value_counts(dropna=False).to_dict()
        info.append(f"test_condition mix: {vc}")
        wet = df.test_condition.astype(str).str.startswith("wet").sum()
        if wet and wet == len(df):
            warnings.append("every row is wet-tested - with no dry rows the model cannot "
                            "learn the wet/dry offset, it can only absorb it as bias")

    # ---- inter-rater agreement ----
    if {"record_id", "extractor"} <= set(df.columns):
        dbl = df[df.duplicated("record_id", keep=False)]
        dbl = dbl.groupby("record_id").filter(lambda g: g.extractor.nunique() >= 2)
        if len(dbl):
            n_rec = dbl.record_id.nunique()
            info.append(f"inter-rater agreement over {n_rec} double-extracted record(s), "
                        f"{100*n_rec/max(df.record_id.nunique(),1):.1f}% of the corpus:")
            for f in AGREEMENT_FIELDS:
                if f not in dbl.columns:
                    continue
                rel = []
                for _, g in dbl.groupby("record_id"):
                    v = pd.to_numeric(g[f], errors="coerce").dropna()
                    if len(v) >= 2 and abs(v.mean()) > 1e-9:
                        rel.append(abs(v.max() - v.min()) / abs(v.mean()))
                if rel:
                    info.append(f"    {f:24s} median disagreement "
                                f"{100*np.median(rel):5.1f}%  (n={len(rel)})")
        else:
            warnings.append("no double-extracted rows found - re-extract ~10% with a "
                            "second person so an agreement figure can be reported")

    return df, errors, warnings, info


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"not found: {path}")
        return 2

    df, errors, warnings, info = validate(path)
    print(f"{path.name}: {len(df)} rows x {len(df.columns)} columns\n")
    for line in info:
        print(f"  INFO   {line}")
    for line in warnings:
        print(f"  WARN   {line}")
    for line in errors:
        print(f"  ERROR  {line}")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
