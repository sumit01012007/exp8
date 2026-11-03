
from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Iris_Model.pkl"
SCALER_PATH = BASE_DIR / "Iris_Scaler.pkl"

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

SPECIES = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}


def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Model files missing. Run python train.py first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris Flower Prediction API",
        "status": "running",
        "endpoints": ["/health", "/predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Send a valid JSON object."
            }), 400

        if "features" in data:
            values = data["features"]
        else:
            values = [data.get(feature) for feature in FEATURES]

        if not isinstance(values, list) or len(values) != 4:
            return jsonify({
                "error": "Provide exactly four numeric features."
            }), 400

        try:
            values = [float(value) for value in values]
        except (TypeError, ValueError):
            return jsonify({
                "error": "All features must be numeric."
            }), 400

        if not all(pd.notna(value) for value in values):
            return jsonify({
                "error": "Features must be finite numbers."
            }), 400

        if not all(abs(value) < float("inf") for value in values):
            return jsonify({
                "error": "Features must be finite numbers."
            }), 400

        input_data = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        scaled_data = scaler.transform(input_data)

        prediction = int(model.predict(scaled_data)[0])

        return jsonify({
            "prediction": prediction,
            "species": SPECIES[prediction],
            "status": "success"
        })

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

