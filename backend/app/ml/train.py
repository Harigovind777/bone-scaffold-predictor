import math
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import lightgbm as lgb

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Matches the polymers/printing methods actually covered by the project's
# source data (gibson_ashby_generator.py) so predictions stay grounded in a
# real physics model rather than arbitrary made-up categories.
BIOPOLYMERS = ["PLA", "PCL", "PGA", "PLGA", "Chitosan", "Gelatin"]
PRINTING_METHODS = ["FDM", "SLA", "SLS", "salt leaching", "freeze drying", "electrospinning"]

# Bulk (fully dense, 0% porosity) reference properties. density/modulus/
# strength are real literature values for these biopolymers and HA filler.
# degr_months = approximate time for a non-porous bulk sample to fully
# resorb in vivo (literature-typical order of magnitude, not a per-batch
# measurement). biocompat is a qualitative 0-1 baseline biocompatibility
# score; acid_producing marks polyesters whose hydrolysis byproducts are
# acidic (relevant because HA buffers them).
MATERIALS = {
    "PLA":      {"density": 1.24, "modulus": 3500, "strength": 65,
                 "degr_months": 18, "biocompat": 0.75, "acid_producing": True},
    "PCL":      {"density": 1.145, "modulus": 400, "strength": 40,
                 "degr_months": 30, "biocompat": 0.80, "acid_producing": True},
    "PGA":      {"density": 1.53, "modulus": 3500, "strength": 100,
                 "degr_months": 2.5, "biocompat": 0.65, "acid_producing": True},
    "PLGA":     {"density": 1.34, "modulus": 2000, "strength": 50,
                 "degr_months": 6, "biocompat": 0.70, "acid_producing": True},
    "Chitosan": {"density": 1.40, "modulus": 3000, "strength": 35,
                 "degr_months": 8, "biocompat": 0.85, "acid_producing": False},
    "Gelatin":  {"density": 1.35, "modulus": 150, "strength": 10,
                 "degr_months": 3, "biocompat": 0.85, "acid_producing": False},
    "HA":       {"density": 3.15, "modulus": 90000, "strength": 300,
                 "biocompat": 0.90, "acid_producing": False},
}

# Cytotoxicity / thermal-stress modifier per printing method (heuristic, 0=neutral)
METHOD_VIABILITY_MODIFIER = {
    "FDM": -0.05, "SLA": -0.08, "SLS": -0.10,
    "salt leaching": -0.02, "freeze drying": 0.02, "electrospinning": -0.03,
}

N_MODULUS = 2.0   # Gibson-Ashby open-cell foam exponent (modulus)
N_STRENGTH = 1.5  # Gibson-Ashby open-cell foam exponent (strength)


def _blend(polymer, ha_fraction):
    base = MATERIALS[polymer]
    ha = MATERIALS["HA"]
    return {
        "density": base["density"] * (1 - ha_fraction) + ha["density"] * ha_fraction,
        "modulus": base["modulus"] * (1 - ha_fraction) + ha["modulus"] * ha_fraction,
        "strength": base["strength"] * (1 - ha_fraction) + ha["strength"] * ha_fraction,
    }


def _gaussian(x, center, width):
    return math.exp(-((x - center) / width) ** 2)


def generate_synthetic_data(n_samples=5000):
    """
    compressive_strength / elastic_modulus: physics-grounded, from the
    Gibson-Ashby open-cell foam relations calibrated to the bulk properties
    above (same model as ml/generate_dataset.py / gibson_ashby_generator.py).

    degradation_rate (months), cell_viability (%), bone_regeneration_score
    (0-10): NOT experimentally validated. No clean physics equation exists
    for these, so they are literature-informed heuristic formulas (pore-size
    and porosity optima per Karageorgiou & Kaplan 2005; HA buffering of
    acidic polyester degradation byproducts; degradation timeline matched to
    new-bone ingrowth). Treat as illustrative until replaced with real
    experimental data.
    """
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        biopolymer = np.random.choice(BIOPOLYMERS)
        printing = np.random.choice(PRINTING_METHODS)

        pore_size = np.random.uniform(50, 800)
        porosity = np.random.uniform(30, 95)
        layer_thickness = np.random.uniform(50, 400)
        # HA filler fraction (0-1), weighted toward realistic 0-30wt% but
        # covering the full range the input schema allows.
        material_composition = float(np.clip(np.random.beta(1.5, 5), 0, 1))

        mat = MATERIALS[biopolymer]
        relative_density = 1 - (porosity / 100.0)
        solid = _blend(biopolymer, material_composition)

        noise = np.random.uniform(0.85, 1.15)
        density = solid["density"] * relative_density
        compressive_strength = max(0.3, solid["strength"] * (relative_density ** N_STRENGTH) * noise)
        elastic_modulus = max(1.0, solid["modulus"] * (relative_density ** N_MODULUS) * noise)

        # --- degradation_rate: months to substantially fully resorb ---
        ha_buffer = (1 + 0.4 * material_composition) if mat["acid_producing"] else (1 + 0.1 * material_composition)
        porosity_speedup = 1 - 0.3 * ((porosity / 100.0) - 0.5)
        degradation_rate = mat["degr_months"] * ha_buffer * porosity_speedup * np.random.uniform(0.85, 1.15)
        degradation_rate = max(1, min(24, degradation_rate))

        # --- cell_viability: pore-size/porosity optima + biocompatibility ---
        pore_factor = _gaussian(pore_size, center=300, width=250)
        porosity_factor = _gaussian(porosity, center=72, width=35)
        cell_viability = 45 + 35 * (0.5 * pore_factor + 0.5 * porosity_factor)
        cell_viability *= (0.7 + 0.3 * mat["biocompat"] / 0.9)
        cell_viability += 8 * min(material_composition, 0.3) / 0.3
        cell_viability += 100 * METHOD_VIABILITY_MODIFIER.get(printing, 0.0)
        cell_viability *= np.random.uniform(0.9, 1.1)
        cell_viability = max(25, min(98, cell_viability))

        # --- bone_regeneration_score: composite, rescaled to a 0-10 scale ---
        strength_factor = _gaussian(math.log10(max(compressive_strength, 0.01)), math.log10(7), 0.6)
        modulus_factor = _gaussian(math.log10(max(elastic_modulus, 0.01)), math.log10(200), 0.7)
        regen_pore_factor = _gaussian(pore_size, center=350, width=220)
        regen_porosity_factor = _gaussian(porosity, center=75, width=30)
        degradation_factor = _gaussian(24 / max(degradation_rate, 0.5), center=2.2, width=2.0)  # ~%/week equivalent
        viability_factor = cell_viability / 100.0
        ha_bonus = 0.10 * min(material_composition, 0.25) / 0.25

        bone_regeneration = 10 * (
            0.20 * strength_factor + 0.15 * modulus_factor +
            0.15 * regen_pore_factor + 0.15 * regen_porosity_factor +
            0.15 * degradation_factor + 0.20 * viability_factor
        )
        bone_regeneration += 10 * ha_bonus * 0.05
        bone_regeneration *= np.random.uniform(0.9, 1.1)
        bone_regeneration = max(0, min(10, bone_regeneration))

        data.append({
            "biopolymer_type": biopolymer,
            "pore_size": pore_size,
            "porosity": porosity,
            "printing_method": printing,
            "layer_thickness": layer_thickness,
            "material_composition": material_composition,
            "density": density,
            "compressive_strength": compressive_strength,
            "elastic_modulus": elastic_modulus,
            "degradation_rate": degradation_rate,
            "cell_viability": cell_viability,
            "bone_regeneration_score": bone_regeneration,
        })

    return pd.DataFrame(data)


def preprocess_data(df, encoders=None, fit_encoders=True):
    df = df.copy()
    le_biopolymer = LabelEncoder() if fit_encoders else encoders["biopolymer"]
    le_printing = LabelEncoder() if fit_encoders else encoders["printing"]

    if fit_encoders:
        df["biopolymer_type"] = le_biopolymer.fit_transform(df["biopolymer_type"])
        df["printing_method"] = le_printing.fit_transform(df["printing_method"])
    else:
        df["biopolymer_type"] = le_biopolymer.transform(df["biopolymer_type"])
        df["printing_method"] = le_printing.transform(df["printing_method"])

    feature_cols = [
        "biopolymer_type", "pore_size", "porosity", "printing_method",
        "layer_thickness", "material_composition", "density"
    ]
    target_cols = [
        "compressive_strength", "elastic_modulus", "degradation_rate",
        "cell_viability", "bone_regeneration_score"
    ]

    X = df[feature_cols].values
    y = df[target_cols].values

    encoders_out = {"biopolymer": le_biopolymer, "printing": le_printing}
    return X, y, feature_cols, target_cols, encoders_out


def build_and_train_nn(X_train, y_train, X_test, y_test, target_cols):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train_scaled, y_train_scaled)

    return model, X_scaler, y_scaler


def train_and_save_models():
    print("Generating synthetic data...")
    df = generate_synthetic_data(5000)
    print(f"  Generated {len(df)} samples")

    X, y, feature_cols, target_cols, encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}
    scores = {}

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=4,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    models["random_forest"] = rf
    scores["random_forest"] = {"r2": float(rf_r2), "rmse": float(rmse)}
    print(f"  R² = {rf_r2:.4f}, RMSE = {rmse:.4f}")

    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=10, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    models["xgboost"] = xgb_model
    scores["xgboost"] = {"r2": float(xgb_r2), "rmse": float(rmse)}
    print(f"  R² = {xgb_r2:.4f}, RMSE = {rmse:.4f}")

    # LightGBM
    print("Training LightGBM...")
    lgb_model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=200, max_depth=10, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1
        )
    )
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    lgb_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    models["lightgbm"] = lgb_model
    scores["lightgbm"] = {"r2": float(lgb_r2), "rmse": float(rmse)}
    print(f"  R² = {lgb_r2:.4f}, RMSE = {rmse:.4f}")

    # Neural Network
    print("\nTraining Neural Network (this may take a while)...")
    nn_model, nn_X_scaler, nn_y_scaler = build_and_train_nn(
        X_train, y_train, X_test, y_test, target_cols
    )
    y_pred_scaled = nn_model.predict(nn_X_scaler.transform(X_test))
    y_pred = nn_y_scaler.inverse_transform(y_pred_scaled)
    nn_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    models["neural_network"] = {
        "model": nn_model, "X_scaler": nn_X_scaler, "y_scaler": nn_y_scaler
    }
    scores["neural_network"] = {"r2": float(nn_r2), "rmse": float(rmse)}
    print(f"  R² = {nn_r2:.4f}, RMSE = {rmse:.4f}")

    # Save models
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, path)
        print(f"  Saved {name} to {path}")

    # Save encoders
    encoders_path = os.path.join(MODELS_DIR, "encoders.joblib")
    joblib.dump(encoders, encoders_path)
    print(f"  Saved encoders to {encoders_path}")

    # Save metadata
    metadata = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "scores": scores,
    }
    metadata_path = os.path.join(MODELS_DIR, "metadata.joblib")
    joblib.dump(metadata, metadata_path)
    print(f"  Saved metadata to {metadata_path}")

    # Save training data reference
    df.to_csv(os.path.join(MODELS_DIR, "training_data.csv"), index=False)

    print("\n=== Training Summary ===")
    for name, score in scores.items():
        print(f"  {name}: R² = {score['r2']:.4f}, RMSE = {score['rmse']:.4f}")

    return models, scores


if __name__ == "__main__":
    train_and_save_models()
