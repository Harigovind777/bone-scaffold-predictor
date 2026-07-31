"""
Trains and compares RandomForest, XGBoost, LightGBM, and a small Keras neural
network for each of the 5 scaffold property targets, then saves the best
model per target plus a shared preprocessing pipeline.

Usage:
    python train.py
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = "data/full_scaffold_dataset.csv"
MODELS_DIR = "models"

NUMERIC_FEATURES = ["ha_wt_pct", "pore_size_um", "porosity_pct", "layer_thickness_um", "density_g_cm3"]
CATEGORICAL_FEATURES = ["biopolymer_type", "printing_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = [
    "compressive_strength_MPa",
    "elastic_modulus_MPa",
    "degradation_rate_pct_per_week",
    "cell_viability_pct",
    "bone_regeneration_score",
]

PHYSICS_GROUNDED = {"compressive_strength_MPa", "elastic_modulus_MPa"}


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_nn(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def evaluate(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_target(target, X_train, X_test, y_train, y_test, preprocessor):
    Xtr = preprocessor.transform(X_train)
    Xte = preprocessor.transform(X_test)

    candidates = {}

    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    candidates["random_forest"] = rf

    xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    xgb.fit(Xtr, y_train)
    candidates["xgboost"] = xgb

    lgbm = LGBMRegressor(n_estimators=400, max_depth=-1, learning_rate=0.05,
                          num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1)
    lgbm.fit(Xtr, y_train)
    candidates["lightgbm"] = lgbm

    nn = build_nn(Xtr.shape[1])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    nn.fit(Xtr, y_train, validation_split=0.15, epochs=200, batch_size=32,
           callbacks=[early_stop], verbose=0)
    candidates["neural_network"] = nn

    results = {}
    for name, model in candidates.items():
        if name == "neural_network":
            preds = model.predict(Xte, verbose=0).flatten()
        else:
            preds = model.predict(Xte)
        results[name] = evaluate(y_test, preds)

    best_name = max(results, key=lambda n: results[n]["r2"])
    best_model = candidates[best_name]

    return best_name, best_model, results


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))

    summary = {}
    for target in TARGETS:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_name, best_model, results = train_target(target, X_train, X_test, y_train, y_test, preprocessor)

        if best_name == "neural_network":
            best_model.save(os.path.join(MODELS_DIR, f"{target}__neural_network.keras"))
        else:
            joblib.dump(best_model, os.path.join(MODELS_DIR, f"{target}__{best_name}.joblib"))

        summary[target] = {
            "best_model": best_name,
            "physics_grounded": target in PHYSICS_GROUNDED,
            "metrics_by_model": results,
        }

        print(f"\n=== {target} ===")
        for name, m in results.items():
            marker = " <-- selected" if name == best_name else ""
            print(f"  {name:16s} R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}{marker}")

    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_schema.json"), "w") as f:
        json.dump({
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "categorical_options": {
                "biopolymer_type": sorted(df["biopolymer_type"].unique().tolist()),
                "printing_method": sorted(df["printing_method"].unique().tolist()),
            },
            "targets": TARGETS,
            "physics_grounded_targets": sorted(PHYSICS_GROUNDED),
        }, f, indent=2)

    print("\nSaved models, preprocessor, training_summary.json, feature_schema.json to", MODELS_DIR)


if __name__ == "__main__":
    main()
