"""
Generic Training Wrapper for MLOps Deployment.

This wrapper dynamically imports any algorithm from the repository and
executes its training pipeline, then serializes the model for deployment.

Usage:
    python train_wrapper.py \
        --algorithm-path 01_regression/linear_regression_sklearn.py \
        --output-dir /opt/ml/model \
        --hyperparams '{"learning_rate": 0.01}'

To deploy a DIFFERENT algorithm, just change --algorithm-path:
    --algorithm-path 02_classification/random_forest_sklearn.py
    --algorithm-path 04_time_series/arima_sklearn.py
    --algorithm-path 05_nlp/sentiment_analysis_sklearn.py
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_algorithm_module(algorithm_path: str):
    """Dynamically import an algorithm module from its file path."""
    path = Path(algorithm_path).resolve()
    if not path.exists():
        # Try relative to repo root
        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / algorithm_path
    if not path.exists():
        raise FileNotFoundError(f"Algorithm not found: {algorithm_path}")

    spec = importlib.util.spec_from_file_location("algorithm", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_and_save(algorithm_path: str, output_dir: str, hyperparams: dict = None):
    """Train an algorithm and save the model artifact."""
    logger.info(f"Loading algorithm from: {algorithm_path}")
    module = load_algorithm_module(algorithm_path)

    # All algorithms in this repo have generate_data() and train()
    logger.info("Generating training data...")
    data = module.generate_data()

    # Unpack based on return structure (most return train/val/test splits)
    if len(data) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = data
    elif len(data) == 4:
        X_train, X_test, y_train, y_test = data
        X_val, y_val = X_test, y_test
    else:
        raise ValueError(f"Unexpected data shape from generate_data(): {len(data)} elements")

    # Train with optional hyperparameters
    logger.info(f"Training with hyperparams: {hyperparams or 'defaults'}")
    if hyperparams:
        model = module.train(X_train, y_train, **hyperparams)
    else:
        model = module.train(X_train, y_train)

    # Validate
    if hasattr(module, "validate"):
        logger.info("Running validation...")
        val_metrics = module.validate(model, X_val, y_val)
        logger.info(f"Validation metrics: {val_metrics}")

    # Test
    if hasattr(module, "test"):
        logger.info("Running test evaluation...")
        test_metrics = module.test(model, X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")

    # Save model artifact
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {model_path}")

    # Save metadata
    metadata = {
        "algorithm_path": algorithm_path,
        "algorithm_name": Path(algorithm_path).stem,
        "hyperparams": hyperparams or {},
        "train_samples": len(X_train),
        "features": X_train.shape[1] if hasattr(X_train, "shape") and len(X_train.shape) > 1 else 1,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {meta_path}")

    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Train any ML algorithm for deployment")
    parser.add_argument("--algorithm-path", required=True, help="Path to algorithm script (e.g., 01_regression/linear_regression_sklearn.py)")
    parser.add_argument("--output-dir", default="/opt/ml/model", help="Directory to save model artifacts")
    parser.add_argument("--hyperparams", default="{}", help="JSON string of hyperparameters")
    args = parser.parse_args()

    hyperparams = json.loads(args.hyperparams)
    train_and_save(args.algorithm_path, args.output_dir, hyperparams or None)


if __name__ == "__main__":
    main()
