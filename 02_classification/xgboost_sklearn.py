"""
XGBoost Classifier - Scikit-Learn API Implementation
=====================================================

Theory & Mathematics:
    XGBoost (eXtreme Gradient Boosting) is a scalable, distributed gradient-
    boosted decision tree (GBDT) system. It builds an additive ensemble of
    weak learners (decision trees) where each new tree corrects the errors
    of the previous ensemble.

    Additive Model:
        F(x) = sum_{m=1}^{M} f_m(x)
        where f_m is the m-th tree.

    At each step m, we add a new tree f_m that minimizes:
        Obj = sum_{i=1}^{N} L(y_i, F_{m-1}(x_i) + f_m(x_i)) + Omega(f_m)

    Regularization term:
        Omega(f) = gamma * T + (1/2) * lambda * sum_{j=1}^{T} w_j^2
        where T = number of leaves, w_j = leaf weights.

    Second-order Taylor expansion of the loss:
        Obj ~ sum_{i=1}^{N} [g_i * f_m(x_i) + (1/2) * h_i * f_m(x_i)^2] + Omega(f_m)
        where:
            g_i = dL/dF|_{F=F_{m-1}(x_i)}  (first-order gradient)
            h_i = d^2L/dF^2|_{F=F_{m-1}(x_i)}  (second-order gradient / Hessian)

    Optimal leaf weight for leaf j:
        w_j* = -sum_{i in I_j} g_i / (sum_{i in I_j} h_i + lambda)

    Split gain:
        Gain = (1/2) * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma

    Key innovations in XGBoost:
        - Column subsampling (like Random Forest)
        - Weighted quantile sketch for approximate split finding
        - Sparsity-aware split finding for missing values
        - Cache-aware block structure for out-of-core computing
        - Shrinkage (learning rate) to prevent overfitting

Business Use Cases:
    - Competition-winning approach for tabular data (Kaggle)
    - Credit scoring and insurance risk modeling
    - Real-time bid optimization in programmatic advertising
    - Anomaly detection in cybersecurity
    - Demand forecasting for supply chain

Advantages:
    - State-of-the-art performance on tabular data
    - Built-in regularization (L1/L2 on weights, tree complexity)
    - Handles missing values natively
    - Column/row subsampling for robustness
    - Efficient implementation (parallelized, cache-aware)
    - Early stopping support

Disadvantages:
    - Many hyperparameters to tune
    - Can overfit on noisy data without proper regularization
    - Sequential tree building limits parallelism in boosting
    - Less interpretable than single trees
    - Memory-intensive for very large datasets

Hyperparameters:
    - n_estimators: Number of boosting rounds (trees)
    - max_depth: Maximum depth of each tree
    - learning_rate (eta): Shrinkage factor for each tree
    - subsample: Row subsampling ratio
    - colsample_bytree: Column subsampling ratio per tree
    - reg_alpha: L1 regularization on leaf weights
    - reg_lambda: L2 regularization on leaf weights
    - gamma: Minimum loss reduction for a split
    - min_child_weight: Minimum sum of instance weight in a child
"""

import logging
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> xgb.XGBClassifier:
    """Train an XGBoost classifier using the sklearn API."""
    defaults = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    n_classes = len(np.unique(y_train))
    if n_classes > 2:
        defaults["objective"] = "multi:softprob"
        defaults["num_class"] = n_classes

    defaults.update(hyperparams)
    model = xgb.XGBClassifier(**defaults)
    model.fit(X_train, y_train)
    logger.info("XGBoost trained with %d estimators, max_depth=%d, lr=%.3f",
                defaults["n_estimators"], defaults["max_depth"], defaults["learning_rate"])
    return model


def _evaluate(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    n_classes = len(np.unique(y))
    if n_classes == 2:
        auc = roc_auc_score(y, y_proba[:, 1])
    else:
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: xgb.XGBClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    logger.info("\nTop 10 Feature Importances (gain):")
    for rank, idx in enumerate(indices[:10], 1):
        logger.info("  %d. Feature %d: %.4f", rank, idx, importances[idx])

    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = train(X_train, y_train, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "n_estimators": ray_tune.choice([100, 200, 300, 500]),
        "max_depth": ray_tune.randint(3, 12),
        "learning_rate": ray_tune.loguniform(0.01, 0.3),
        "subsample": ray_tune.uniform(0.5, 1.0),
        "colsample_bytree": ray_tune.uniform(0.5, 1.0),
        "reg_alpha": ray_tune.loguniform(1e-8, 10.0),
        "reg_lambda": ray_tune.loguniform(1e-8, 10.0),
        "gamma": ray_tune.uniform(0.0, 5.0),
        "min_child_weight": ray_tune.randint(1, 10),
    }

    tuner = ray_tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=ray_tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="f1", mode="max")
    logger.info("Ray Tune best config: %s", best.config)
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 70)
    logger.info("XGBoost Classifier - Scikit-Learn API Implementation")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Baseline
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # Optuna
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=30,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    best_model = train(X_train, y_train, **study.best_params)

    # Test
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
