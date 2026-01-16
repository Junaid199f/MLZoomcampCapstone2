import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_PATH = Path("models/model.pkl")
META_PATH = Path("models/metadata.json")

DEFAULT_FEATURES = [
    "AF3",
    "AF4",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC5",
    "FC6",
    "T7",
    "T8",
    "P7",
    "P8",
    "O1",
    "O2",
]

app = Flask(__name__, template_folder="templates", static_folder="static")


def load_metadata():
    if META_PATH.exists():
        with META_PATH.open() as f:
            return json.load(f)
    return {"features": DEFAULT_FEATURES}


meta = load_metadata()
FEATURES = meta.get("features", DEFAULT_FEATURES)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run train.py first to create it."
    )

model = joblib.load(MODEL_PATH)


def build_dataframe(payload):
    if isinstance(payload, dict) and "features" in payload:
        values = payload["features"]
        if not isinstance(values, list) or len(values) != len(FEATURES):
            raise ValueError("features must be a list with 14 values")
        return pd.DataFrame([values], columns=FEATURES)

    if isinstance(payload, dict):
        missing = [col for col in FEATURES if col not in payload]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        row = {col: payload[col] for col in FEATURES}
        return pd.DataFrame([row])

    raise ValueError("Payload must be a JSON object")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        features = build_dataframe(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    proba = float(model.predict_proba(features)[:, 1][0])
    prediction = int(proba >= 0.5)

    return jsonify({"eye_closed_probability": proba, "prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
