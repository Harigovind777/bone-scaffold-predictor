import os
import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "saved_models")


class MLService:
    def __init__(self):
        self.models = {}
        self.encoders = None
        self.metadata = None
        self.loaded = False

    def load_models(self):
        if self.loaded:
            return

        if not os.path.exists(MODELS_DIR):
            raise FileNotFoundError(
                f"Models directory not found at {MODELS_DIR}. "
                "Run `python -m app.ml.train` first."
            )

        model_names = ["random_forest", "xgboost", "lightgbm", "neural_network"]
        for name in model_names:
            path = os.path.join(MODELS_DIR, f"{name}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            self.models[name] = joblib.load(path)

        self.encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
        self.metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
        self.loaded = True

    def predict(self, input_data, model_name="random_forest"):
        self.load_models()

        if model_name not in self.models:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {list(self.models.keys())}"
            )

        model = self.models[model_name]

        biopolymer_encoded = self.encoders["biopolymer"].transform(
            [input_data["biopolymer_type"]]
        )[0]
        printing_encoded = self.encoders["printing"].transform(
            [input_data["printing_method"]]
        )[0]

        features = np.array([[
            biopolymer_encoded,
            input_data["pore_size"],
            input_data["porosity"],
            printing_encoded,
            input_data["layer_thickness"],
            input_data["material_composition"],
            input_data["density"],
        ]], dtype=np.float32)

        if model_name == "neural_network":
            features_scaled = model["X_scaler"].transform(features)
            prediction_scaled = model["model"].predict(features_scaled)
            prediction = model["y_scaler"].inverse_transform(prediction_scaled.reshape(1, -1))
        else:
            prediction = model.predict(features)

        prediction = prediction.flatten()

        target_cols = self.metadata["target_cols"]
        result = {
            target_cols[i]: round(float(prediction[i]), 4)
            for i in range(len(target_cols))
        }

        scores = self.metadata["scores"].get(model_name, {})
        result["model_used"] = model_name
        result["model_confidence"] = round(scores.get("r2", 0), 4)

        return result

    def get_available_models(self):
        return list(self.models.keys())

    def get_model_scores(self):
        self.load_models()
        return self.metadata.get("scores", {})


ml_service = MLService()
