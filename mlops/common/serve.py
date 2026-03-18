"""
Generic Model Serving Endpoint.

Serves any trained model from this repository via a REST API.
Works with all three cloud platforms (SageMaker, Azure ML, Vertex AI)
as they all support custom containers with HTTP endpoints.

SageMaker expects: /ping (health) and /invocations (predict)
Azure ML expects: /score
Vertex AI expects: /predict

This server implements ALL endpoints so one container works everywhere.
"""

import json
import logging
import os
import pickle

import numpy as np
from flask import Flask, Response, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = None
METADATA = None
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")


def load_model():
    """Load model and metadata from disk."""
    global MODEL, METADATA
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    meta_path = os.path.join(MODEL_DIR, "metadata.json")

    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            METADATA = json.load(f)
        logger.info(f"Metadata: {METADATA}")


def predict(input_data):
    """Run prediction on input data."""
    X = np.array(input_data)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Handle sklearn models (have .predict)
    if hasattr(MODEL, "predict"):
        predictions = MODEL.predict(X)
    # Handle numpy/dict-based models (weights stored in dict)
    elif isinstance(MODEL, dict) and "weights" in MODEL:
        weights = MODEL["weights"]
        bias = MODEL.get("bias", 0)
        predictions = X @ weights + bias
    else:
        raise ValueError(f"Unsupported model type: {type(MODEL)}")

    return predictions.tolist()


# --- SageMaker endpoints ---

@app.route("/ping", methods=["GET"])
def ping():
    """SageMaker health check."""
    return Response(response="", status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """SageMaker inference endpoint."""
    data = request.get_json(force=True)
    instances = data.get("instances", data.get("data", data))
    predictions = predict(instances)
    return jsonify({"predictions": predictions})


# --- Azure ML endpoint ---

@app.route("/score", methods=["POST"])
def score():
    """Azure ML inference endpoint."""
    data = request.get_json(force=True)
    instances = data.get("data", data.get("instances", data))
    predictions = predict(instances)
    return jsonify({"result": predictions})


# --- Vertex AI endpoint ---

@app.route("/predict", methods=["POST"])
def vertex_predict():
    """Google Vertex AI inference endpoint."""
    data = request.get_json(force=True)
    instances = data.get("instances", data.get("data", data))
    predictions = predict(instances)
    return jsonify({"predictions": predictions})


# --- Common endpoints ---

@app.route("/health", methods=["GET"])
def health():
    """Generic health check."""
    return jsonify({"status": "healthy", "model_loaded": MODEL is not None, "metadata": METADATA})


@app.route("/metadata", methods=["GET"])
def metadata():
    """Return model metadata."""
    return jsonify(METADATA or {})


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
