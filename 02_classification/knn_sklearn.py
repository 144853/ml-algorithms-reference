"""
K-Nearest Neighbors Classifier - Scikit-Learn Implementation
==============================================================

Theory & Mathematics:
    K-Nearest Neighbors (KNN) is a non-parametric, instance-based (lazy) learning
    algorithm. It makes no assumptions about the underlying data distribution and
    defers all computation to prediction time.

    Algorithm:
        1. Store all training data (no explicit training phase).
        2. For a new query point x:
            a. Compute the distance from x to every training point.
            b. Select the K closest training points (nearest neighbors).
            c. Return the majority class among the K neighbors (classification).

    Distance Metrics:
        Euclidean Distance (L2 norm):
            d(x, y) = sqrt(sum_{i=1}^{n} (x_i - y_i)^2)
            WHY: The most natural notion of distance in Euclidean space.
            Sensitive to feature scaling (features with larger ranges dominate).

        Manhattan Distance (L1 norm):
            d(x, y) = sum_{i=1}^{n} |x_i - y_i|
            WHY: More robust to outliers than Euclidean. Better for data where
            features are measured on different scales or are sparse.

        Minkowski Distance (Lp norm):
            d(x, y) = (sum_{i=1}^{n} |x_i - y_i|^p)^(1/p)
            Special cases: p=1 (Manhattan), p=2 (Euclidean), p->inf (Chebyshev).

    Voting Schemes:
        Uniform: Each of the K neighbors gets one vote regardless of distance.
            WHY: Simple, works well when neighbors are roughly equidistant.
        Distance-weighted: Each neighbor's vote is weighted by 1/distance.
            WHY: Closer neighbors have more influence, which is more intuitive
            and often produces better boundaries.

    Choosing K:
        - K=1: Most flexible (decision boundary passes through every point),
          but extremely sensitive to noise (any single mislabeled point
          creates a wrong prediction region).
        - Large K: Smoother decision boundary, more robust to noise, but may
          smooth over important local patterns (underfitting).
        - Rule of thumb: K = sqrt(N) where N is number of training samples.
        - K should be odd for binary classification to avoid ties.

    Curse of Dimensionality:
        In high-dimensional spaces, all points become roughly equidistant,
        making KNN ineffective. WHY: The volume of a hypersphere grows
        exponentially with dimension, so neighbors in high-D are not truly "near".
        Mitigation: dimensionality reduction (PCA) or feature selection.

Business Use Cases:
    - Recommendation systems (collaborative filtering is KNN on user/item space)
    - Medical diagnosis with similar patient lookup
    - Anomaly detection (outliers have distant nearest neighbors)
    - Image classification (pixel-space similarity)
    - Missing value imputation (fill with neighbors' values)

Advantages:
    - No training phase (lazy learning): just store the data
    - Non-parametric: makes no assumptions about data distribution
    - Naturally handles multi-class problems
    - Simple to understand and implement
    - Decision boundary can be arbitrarily complex

Disadvantages:
    - Slow prediction: O(n*d) per query (must scan all training points)
    - Memory intensive: must store entire training set
    - Sensitive to feature scaling (must standardize)
    - Curse of dimensionality: degrades in high-D spaces
    - Sensitive to noisy features and irrelevant dimensions
    - No built-in feature importance or model explanation
"""

# --- Standard library imports ---
import logging  # Structured logging
import warnings  # Warning suppression
from typing import Any, Dict, Tuple  # Type annotations

# --- Third-party imports ---
import numpy as np  # Numerical computing
import optuna  # Bayesian hyperparameter optimization

from sklearn.neighbors import KNeighborsClassifier  # KNN implementation
from sklearn.datasets import make_classification  # Synthetic data
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardization

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data, split, and standardize for KNN.

    WHY standardize: KNN relies on distance metrics. Features with larger scales
    dominate the distance computation, making smaller-scale features invisible.
    Standardizing to zero mean and unit variance ensures equal contribution.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Generate synthetic classification dataset.
    X, y = make_classification(
        n_samples=n_samples,            # Total data points
        n_features=n_features,          # Feature dimensionality
        n_informative=n_features // 2,  # Half carry real signal
        n_redundant=n_features // 4,    # Quarter are linear combinations
        n_classes=n_classes,            # Number of classes
        random_state=random_state,      # Seed for reproducibility
    )

    # Split: 60% train, 20% val, 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Standardize features for KNN.
    # WHY fit only on train: Prevents data leakage from val/test into scaling.
    scaler = StandardScaler()  # Create scaler
    X_train = scaler.fit_transform(X_train)  # Fit and transform train
    X_val = scaler.transform(X_val)          # Transform val with train statistics
    X_test = scaler.transform(X_test)        # Transform test with train statistics

    logger.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "euclidean",
    p: int = 2,
) -> KNeighborsClassifier:
    """Train (fit) a KNN classifier on training data.

    WHY "train" for a lazy learner: KNN's fit() simply stores the training data.
    No model parameters are learned. The actual computation happens at predict().

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_neighbors: Number of neighbors to consider (K).
            WHY: K controls the bias-variance tradeoff. K=1 is high variance
            (noisy boundary), large K is high bias (smooth boundary).
        weights: Voting scheme ("uniform" or "distance").
            WHY uniform: Simple majority vote. WHY distance: 1/d weighting
            gives closer neighbors more influence.
        metric: Distance metric ("euclidean", "manhattan", "minkowski").
            WHY: Different metrics suit different data types and distributions.
        p: Power parameter for Minkowski distance (p=1: Manhattan, p=2: Euclidean).

    Returns:
        Fitted KNeighborsClassifier model.
    """
    logger.info(
        f"Training KNN: k={n_neighbors}, weights={weights}, "
        f"metric={metric}, p={p}"
    )

    # Create KNN classifier with specified hyperparameters.
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,  # Number of nearest neighbors
        weights=weights,          # Voting scheme
        metric=metric,            # Distance metric
        p=p,                      # Minkowski p parameter
        n_jobs=-1,                # Use all CPU cores for distance computation
    )

    # Fit: stores training data in the model's internal data structure.
    # WHY: KNN needs access to all training points at prediction time.
    # sklearn uses a KD-tree or Ball tree for efficient neighbor lookup.
    model.fit(X_train, y_train)  # Store training data

    # Compute training accuracy (KNN with K=1 always gets 100% on training data).
    train_preds = model.predict(X_train)  # Predict on training data
    train_acc = accuracy_score(y_train, train_preds)  # Compute accuracy
    logger.info(f"Training accuracy: {train_acc:.4f}")  # Log result

    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: KNeighborsClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate on validation set.

    Args:
        model: Fitted KNN classifier.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_val)  # Hard predictions
    y_proba = model.predict_proba(X_val)  # Probability predictions

    n_classes = len(np.unique(y_val))
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_val, y_proba[:, 1]) if n_classes == 2 else 0.0,
    }

    logger.info("Validation Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test(
    model: KNeighborsClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Final evaluation on test set.

    Args:
        model: Fitted KNN classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    n_classes = len(np.unique(y_test))
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba[:, 1]) if n_classes == 2 else 0.0,
    }

    logger.info("=" * 50)
    logger.info("TEST SET RESULTS (Final Evaluation):")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    logger.info("=" * 50)
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial) -> float:
    """Optuna objective for KNN hyperparameter search.

    Searches over K, weights, and distance metric.

    Args:
        trial: Optuna Trial.

    Returns:
        Validation F1 score.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    n_neighbors = trial.suggest_int("n_neighbors", 1, 50)  # K from 1 to 50
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
    p = trial.suggest_int("p", 1, 5) if metric == "minkowski" else 2

    model = train(X_train, y_train, n_neighbors=n_neighbors,
                  weights=weights, metric=metric, p=p)
    metrics = validate(model, X_val, y_val)

    return metrics["f1"]


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune search space for KNN.

    Returns:
        Dictionary defining the search space.
    """
    search_space = {
        "n_neighbors": {"type": "randint", "lower": 1, "upper": 51},
        "weights": {"type": "choice", "values": ["uniform", "distance"]},
        "metric": {"type": "choice", "values": ["euclidean", "manhattan", "minkowski"]},
        "p": {"type": "randint", "lower": 1, "upper": 6},
    }
    logger.info("Ray Tune search space:")
    for param, config in search_space.items():
        logger.info(f"  {param}: {config}")
    return search_space


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare different KNN configurations.

    Configurations:
    1. K=1, Euclidean: Nearest-neighbor (most flexible, highest variance).
    2. K=5, Euclidean: Small neighborhood (balanced).
    3. K=20, Manhattan, distance-weighted: Larger neighborhood, alternative metric.
    4. K=50, Euclidean, distance-weighted: Very smooth boundary.

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    configs = {
        "k1_euclidean_uniform": {
            "params": {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"},
            "reasoning": (
                "K=1 (nearest neighbor) with Euclidean distance. Most flexible decision "
                "boundary: exactly follows training data. 100% train accuracy guaranteed. "
                "Expected: high variance (overfitting), sensitive to noise. Every single "
                "training point creates its own Voronoi cell in the decision boundary."
            ),
        },
        "k5_euclidean_uniform": {
            "params": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
            "reasoning": (
                "K=5 with Euclidean distance and uniform voting. Standard default choice. "
                "Smooths out noise by requiring 5 nearby points to agree. "
                "Expected: good balance between flexibility and robustness. "
                "Odd K avoids ties in binary classification."
            ),
        },
        "k20_manhattan_distance": {
            "params": {"n_neighbors": 20, "weights": "distance", "metric": "manhattan"},
            "reasoning": (
                "K=20 with Manhattan distance and distance-weighted voting. Large K "
                "creates smooth boundaries. Manhattan distance is more robust to outliers "
                "than Euclidean (doesn't square large differences). Distance weighting "
                "ensures the 20 neighbors are not all equally influential."
            ),
        },
        "k50_euclidean_distance": {
            "params": {"n_neighbors": 50, "weights": "distance", "metric": "euclidean"},
            "reasoning": (
                "K=50 with Euclidean distance and distance weighting. Very smooth boundary "
                "approaching a global average for small datasets. Expected: underfitting "
                "for complex decision boundaries. Distance weighting partially compensates "
                "by down-weighting distant neighbors."
            ),
        },
    }

    results = {}
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {name}")
        logger.info(f"Reasoning: {config['reasoning']}")

        model = train(X_train, y_train, **config["params"])
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY:")
    for name, metrics in results.items():
        logger.info(f"  {name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate KNN on wine quality classification.

    Domain: Wine quality assessment.
    Features: fixed_acidity, residual_sugar, alcohol_content, pH_level
    Target: quality_class (0 = standard, 1 = premium)

    WHY this domain: Wine quality depends on chemical properties that create
    local clusters in feature space, making it ideal for KNN's locality-based
    approach. Similar wines (by chemical composition) tend to have similar quality.
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Wine Quality Classification")
    logger.info("=" * 60)

    np.random.seed(42)
    n_samples = 600

    # Generate wine chemistry features with realistic ranges.
    fixed_acidity = np.random.normal(8.0, 1.5, n_samples).clip(4.0, 14.0)  # g/L tartaric acid
    residual_sugar = np.random.exponential(2.5, n_samples).clip(0.5, 20.0)  # g/L
    alcohol_content = np.random.normal(10.5, 1.2, n_samples).clip(8.0, 15.0)  # % vol
    pH_level = np.random.normal(3.3, 0.15, n_samples).clip(2.7, 4.0)  # pH scale

    X = np.column_stack([fixed_acidity, residual_sugar, alcohol_content, pH_level])
    feature_names = ["fixed_acidity", "residual_sugar", "alcohol_content", "pH_level"]

    # Create quality labels based on chemistry.
    # WHY: Higher alcohol, moderate acidity, and lower sugar -> premium wine.
    quality_score = (
        0.3 * (alcohol_content - 10.0)  # Higher alcohol = better
        - 0.2 * (fixed_acidity - 7.0)   # Moderate acidity preferred
        - 0.1 * (residual_sugar - 2.0)   # Less sugar = drier, premium
        + 0.1 * (3.3 - pH_level) * 10    # Slightly acidic preferred
    )
    probability = 1.0 / (1.0 + np.exp(-quality_score))
    y = (probability + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

    # Split and standardize.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        logger.info(f"  {name}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}")
    logger.info(f"Class dist: standard={np.sum(y_train==0)}, premium={np.sum(y_train==1)}")

    model = train(X_train, y_train, n_neighbors=7, weights="distance")
    validate(model, X_val, y_val)
    test(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete KNN (sklearn) pipeline."""
    logger.info("=" * 60)
    logger.info("K-Nearest Neighbors - Scikit-Learn Implementation")
    logger.info("=" * 60)

    logger.info("\n--- Step 1: Generating Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    logger.info("\n--- Step 2: Training Baseline (K=5) ---")
    baseline = train(X_train, y_train, n_neighbors=5)

    logger.info("\n--- Step 3: Validating Baseline ---")
    validate(baseline, X_val, y_val)

    logger.info("\n--- Step 4: Comparing Parameter Sets ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    logger.info("\n--- Step 5: Optuna HPO ---")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(optuna_objective, n_trials=25)
    logger.info(f"Best F1: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    logger.info("\n--- Step 6: Ray Tune Search Space ---")
    ray_tune_search()

    logger.info("\n--- Step 7: Training Best Model ---")
    best_p = study.best_trial.params
    best_model = train(X_train, y_train, **best_p)

    logger.info("\n--- Step 8: Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    logger.info("\n--- Step 9: Real-World Demo ---")
    real_world_demo()

    logger.info("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    main()
