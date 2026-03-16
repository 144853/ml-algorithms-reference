"""
K-Nearest Neighbors Classifier - NumPy From-Scratch Implementation
===================================================================

Theory & Mathematics:
    This implements KNN from scratch using NumPy, including multiple distance
    metrics and voting schemes.

    KNN Algorithm:
        Training: Store all training data (X_train, y_train). No computation.
        Prediction for query point x:
            1. Compute d(x, x_i) for all training points x_i.
            2. Sort distances and select K smallest.
            3. Aggregate labels of K nearest neighbors (majority vote).

    Distance Metrics Implemented:

        Euclidean (L2):
            d(x, y) = sqrt(sum_i (x_i - y_i)^2)
            Vectorized: d = ||x - y||_2
            WHY: Standard geometric distance. Most common for continuous features.

        Manhattan (L1):
            d(x, y) = sum_i |x_i - y_i|
            WHY: More robust to outliers (no squaring). Better for sparse or
            high-dimensional data. Measures distance along grid lines.

        Minkowski (Lp):
            d(x, y) = (sum_i |x_i - y_i|^p)^(1/p)
            WHY: Generalizes L1 and L2. p parameter lets you tune the distance.
            Higher p emphasizes the largest component difference.

    Voting Schemes:
        Uniform: y_hat = mode(y[nearest_k])
            Each neighbor contributes one equal vote.
        Distance-weighted: y_hat = argmax_c sum_{i in N_k} (1/d_i) * I(y_i == c)
            Neighbors closer to the query get more influence.

    Vectorized Pairwise Distance Computation:
        Instead of looping over pairs, we use the identity:
        ||x - y||^2 = ||x||^2 - 2*x^T*y + ||y||^2

        This allows computing ALL pairwise distances with matrix operations:
        D[i,j] = ||X_test[i]||^2 - 2*X_test[i]^T*X_train[j] + ||X_train[j]||^2

        WHY: Matrix operations are 100-1000x faster than Python loops because
        NumPy delegates to optimized BLAS libraries (written in C/Fortran).

Business Use Cases:
    - Similar customer lookup for recommendation
    - Fraud detection via anomaly scoring
    - Medical diagnosis based on similar patient outcomes
    - Quality control via nearest conforming/nonconforming samples

Advantages:
    - No training time (lazy learning)
    - Non-parametric, no distribution assumptions
    - Handles complex decision boundaries
    - Simple to understand and debug

Disadvantages:
    - O(n*d) prediction per query point
    - Entire training set must be stored in memory
    - Feature scaling is critical
    - Degrades with high dimensionality (curse of dimensionality)
"""

# --- Standard library imports ---
import logging  # Structured logging
import warnings  # Warning suppression
from typing import Any, Dict, Optional, Tuple  # Type annotations

# --- Third-party imports ---
import numpy as np  # Numerical computing - the core of this implementation
import optuna  # Bayesian hyperparameter optimization

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
# KNN Classifier (From Scratch)
# ---------------------------------------------------------------------------

class KNNFromScratch:
    """K-Nearest Neighbors classifier implemented from scratch with NumPy.

    Supports multiple distance metrics and voting schemes with fully
    vectorized distance computation for efficiency.

    WHY from scratch: Understanding vectorized distance computation,
    the effect of K on bias-variance, and how voting schemes affect
    predictions builds deep intuition for instance-based learning.
    """

    def __init__(
        self,
        n_neighbors: int = 5,        # Number of nearest neighbors (K)
        weights: str = "uniform",     # Voting scheme: "uniform" or "distance"
        metric: str = "euclidean",    # Distance metric
        p: int = 2,                   # Minkowski p parameter
    ):
        """Initialize the KNN classifier.

        Args:
            n_neighbors: Number of nearest neighbors to consider.
                WHY: K controls smoothness. K=1 = most complex boundary.
                Large K = smoother, more biased boundary.
            weights: How to weight neighbor votes.
                "uniform" = equal votes. "distance" = 1/d weighting.
            metric: Which distance function to use.
                "euclidean", "manhattan", or "minkowski".
            p: Power for Minkowski distance (only used when metric="minkowski").
        """
        self.n_neighbors = n_neighbors  # Store K value
        self.weights = weights          # Store voting scheme
        self.metric = metric            # Store distance metric name
        self.p = p                      # Store Minkowski p parameter
        self.X_train: Optional[np.ndarray] = None  # Training features (set in fit)
        self.y_train: Optional[np.ndarray] = None  # Training labels (set in fit)
        self.n_classes: int = 0         # Number of unique classes (set in fit)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNFromScratch":
        """Store training data (lazy learning - no actual model is built).

        WHY just store: KNN is a "lazy learner" - it defers all computation
        to prediction time. The training phase is O(1): just save the data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self (for method chaining).
        """
        self.X_train = X.copy()             # Store copy of training features
        self.y_train = y.astype(int).copy()  # Store copy of training labels (as int)
        self.n_classes = len(np.unique(y))   # Count unique classes

        logger.info(  # Log dataset size
            f"KNN fit: stored {X.shape[0]} samples, {X.shape[1]} features, "
            f"{self.n_classes} classes"
        )
        return self  # Return self for method chaining

    def _compute_distances(self, X_query: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between query points and training data.

        Uses fully vectorized computation for efficiency. No Python loops.

        For Euclidean:
            ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
            d = sqrt(||x||^2 + ||y||^2 - 2*x^T*y)

        WHY vectorized: Matrix operations are orders of magnitude faster than
        Python loops. For 1000 queries x 5000 training points x 20 features,
        vectorized takes milliseconds vs seconds for loops.

        Args:
            X_query: Query points of shape (n_queries, n_features).

        Returns:
            Distance matrix of shape (n_queries, n_train).
        """
        if self.metric == "euclidean":
            # Vectorized Euclidean distance using the expansion trick.
            # ||x-y||^2 = ||x||^2 - 2*x^T*y + ||y||^2

            # Compute ||x||^2 for each query point.
            # WHY sum(axis=1): Squaring and summing across features gives squared norm.
            query_sq = np.sum(X_query ** 2, axis=1, keepdims=True)  # Shape: (n_query, 1)

            # Compute ||y||^2 for each training point.
            train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True)  # Shape: (n_train, 1)

            # Compute cross term: -2 * x^T * y via matrix multiplication.
            # X_query @ X_train.T has shape (n_query, n_train).
            cross_term = -2.0 * X_query @ self.X_train.T  # Shape: (n_query, n_train)

            # Combine: ||x-y||^2 = ||x||^2 - 2*x^T*y + ||y||^2.
            # Broadcasting: (n_query, 1) + (n_query, n_train) + (1, n_train).
            squared_dists = query_sq + cross_term + train_sq.T  # Shape: (n_query, n_train)

            # Clip to zero to avoid sqrt of negative numbers due to floating point.
            # WHY clip: Numerical errors can make squared_dists slightly negative.
            squared_dists = np.clip(squared_dists, 0.0, None)  # Ensure non-negative

            # Take square root to get Euclidean distances.
            distances = np.sqrt(squared_dists)  # Shape: (n_query, n_train)

        elif self.metric == "manhattan":
            # Manhattan distance: sum of absolute differences.
            # Computed using broadcasting: expand dims to enable pairwise computation.
            # X_query[:, None, :] has shape (n_query, 1, n_features).
            # X_train[None, :, :] has shape (1, n_train, n_features).
            # Subtraction broadcasts to (n_query, n_train, n_features).
            diff = X_query[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]  # Pairwise diffs
            distances = np.sum(np.abs(diff), axis=2)  # Sum absolute diffs: (n_query, n_train)

        elif self.metric == "minkowski":
            # Minkowski distance: (sum |x_i - y_i|^p)^(1/p).
            # Generalization of L1 (p=1) and L2 (p=2).
            diff = X_query[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]  # Pairwise diffs
            distances = np.sum(np.abs(diff) ** self.p, axis=2) ** (1.0 / self.p)  # Minkowski

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return distances  # Shape: (n_queries, n_train)

    def _get_k_nearest(
        self, distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the K nearest neighbors for each query point.

        Uses np.argpartition for efficient partial sorting (O(n) vs O(n log n)).

        WHY argpartition: We only need the K smallest distances, not a full sort.
        argpartition finds the K-th smallest element and partitions the array
        so elements smaller than K-th are on the left. This is O(n) average.

        Args:
            distances: Pairwise distance matrix of shape (n_queries, n_train).

        Returns:
            Tuple of (neighbor_indices, neighbor_distances), both shape (n_queries, K).
        """
        k = self.n_neighbors  # Number of neighbors to find

        # Use argpartition to find the K smallest distances efficiently.
        # WHY: argpartition is O(n) average case, vs O(n log n) for full sort.
        # The first k elements of the partitioned array are the k smallest (unsorted).
        knn_indices = np.argpartition(distances, k, axis=1)[:, :k]  # (n_queries, K)

        # Gather the actual distances for the K nearest neighbors.
        # WHY: We need distances for distance-weighted voting.
        knn_distances = np.take_along_axis(distances, knn_indices, axis=1)  # (n_queries, K)

        return knn_indices, knn_distances  # Return indices and distances

    def _vote(
        self,
        knn_indices: np.ndarray,  # Shape: (n_queries, K)
        knn_distances: np.ndarray,  # Shape: (n_queries, K)
    ) -> np.ndarray:
        """Aggregate neighbor labels into predictions using the voting scheme.

        Uniform voting: each neighbor contributes 1 vote.
        Distance-weighted: each neighbor contributes 1/distance votes.

        Args:
            knn_indices: Indices of K nearest neighbors per query.
            knn_distances: Distances to K nearest neighbors per query.

        Returns:
            Predicted class labels of shape (n_queries,).
        """
        n_queries = knn_indices.shape[0]  # Number of query points
        predictions = np.zeros(n_queries, dtype=int)  # Initialize predictions

        # Get the labels of the K nearest neighbors.
        knn_labels = self.y_train[knn_indices]  # Shape: (n_queries, K)

        for i in range(n_queries):  # Process each query point
            if self.weights == "uniform":
                # Uniform voting: simple majority vote.
                # WHY bincount: Efficiently counts occurrences of each class label.
                counts = np.bincount(knn_labels[i], minlength=self.n_classes)  # Class counts
                predictions[i] = np.argmax(counts)  # Class with most votes wins

            elif self.weights == "distance":
                # Distance-weighted voting: closer neighbors get more influence.
                # Weight = 1/distance (with epsilon to avoid division by zero).
                # WHY: Intuitively, a neighbor at distance 0.1 should have 10x
                # more influence than one at distance 1.0.
                dists = knn_distances[i]  # Distances for this query
                weights = 1.0 / (dists + 1e-10)  # Inverse distance weights

                # Accumulate weighted votes for each class.
                weighted_counts = np.zeros(self.n_classes)  # Initialize
                for j in range(self.n_neighbors):  # For each neighbor
                    weighted_counts[knn_labels[i, j]] += weights[j]  # Add weighted vote

                predictions[i] = np.argmax(weighted_counts)  # Class with highest weight

        return predictions  # Return predicted labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for all query points.

        Pipeline: compute distances -> find K nearest -> vote.

        Args:
            X: Query features of shape (n_queries, n_features).

        Returns:
            Predicted labels of shape (n_queries,).
        """
        distances = self._compute_distances(X)  # Pairwise distances
        knn_indices, knn_distances = self._get_k_nearest(distances)  # K nearest
        predictions = self._vote(knn_indices, knn_distances)  # Majority vote

        return predictions  # Return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities based on neighbor proportions.

        For uniform: P(class) = count(class in neighbors) / K
        For distance-weighted: P(class) = sum(1/d for class neighbors) / sum(1/d for all)

        Args:
            X: Query features of shape (n_queries, n_features).

        Returns:
            Probability matrix of shape (n_queries, n_classes).
        """
        distances = self._compute_distances(X)  # Pairwise distances
        knn_indices, knn_distances = self._get_k_nearest(distances)  # K nearest
        knn_labels = self.y_train[knn_indices]  # Neighbor labels

        n_queries = X.shape[0]  # Number of queries
        proba = np.zeros((n_queries, self.n_classes))  # Initialize probabilities

        for i in range(n_queries):  # Process each query
            if self.weights == "uniform":
                # Simple proportion of each class among neighbors.
                counts = np.bincount(knn_labels[i], minlength=self.n_classes)
                proba[i] = counts / self.n_neighbors  # Normalize to sum to 1

            elif self.weights == "distance":
                # Distance-weighted proportions.
                dists = knn_distances[i]
                weights = 1.0 / (dists + 1e-10)  # Inverse distance

                for j in range(self.n_neighbors):
                    proba[i, knn_labels[i, j]] += weights[j]

                total_weight = np.sum(proba[i])  # Sum for normalization
                if total_weight > 0:
                    proba[i] /= total_weight  # Normalize to sum to 1

        return proba  # Return probability matrix


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

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_features // 2, n_redundant=n_features // 4,
        n_classes=n_classes, random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Standardize for KNN (critical for distance-based methods).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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
) -> KNNFromScratch:
    """Create and fit a from-scratch KNN classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_neighbors: Number of neighbors (K).
        weights: Voting scheme.
        metric: Distance metric.
        p: Minkowski p parameter.

    Returns:
        Fitted KNNFromScratch model.
    """
    logger.info(f"Training KNN (scratch): k={n_neighbors}, weights={weights}, metric={metric}")

    model = KNNFromScratch(
        n_neighbors=n_neighbors, weights=weights, metric=metric, p=p
    )
    model.fit(X_train, y_train)

    # Training accuracy check.
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    logger.info(f"Training accuracy: {train_acc:.4f}")

    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: KNNFromScratch,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate on validation set.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

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
    model: KNNFromScratch,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Final evaluation on test set.

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

    Returns:
        Validation F1 score.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
    p = trial.suggest_int("p", 1, 5) if metric == "minkowski" else 2

    model = train(X_train, y_train, n_neighbors=n_neighbors,
                  weights=weights, metric=metric, p=p)
    metrics = validate(model, X_val, y_val)

    return metrics["f1"]


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune search space.

    Returns:
        Search space dictionary.
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
    """Compare different KNN configurations with reasoning.

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    configs = {
        "k1_euclidean_uniform": {
            "params": {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"},
            "reasoning": (
                "K=1 nearest neighbor with Euclidean distance. Maximum flexibility: "
                "decision boundary passes through every training point's Voronoi cell. "
                "100% train accuracy guaranteed. High variance, sensitive to noise."
            ),
        },
        "k7_euclidean_distance": {
            "params": {"n_neighbors": 7, "weights": "distance", "metric": "euclidean"},
            "reasoning": (
                "K=7 with distance-weighted Euclidean. Moderate smoothing with closer "
                "neighbors contributing more. Odd K avoids ties in binary classification. "
                "Expected: good generalization for most datasets."
            ),
        },
        "k15_manhattan_uniform": {
            "params": {"n_neighbors": 15, "weights": "uniform", "metric": "manhattan"},
            "reasoning": (
                "K=15 with Manhattan (L1) distance and uniform voting. Larger K smooths "
                "the boundary. Manhattan distance is more robust to outliers and works "
                "better in high dimensions. Expected: smoother but may miss fine patterns."
            ),
        },
        "k5_minkowski_p3_distance": {
            "params": {"n_neighbors": 5, "weights": "distance", "metric": "minkowski", "p": 3},
            "reasoning": (
                "K=5 with Minkowski p=3 distance and distance weighting. p=3 sits between "
                "L2 (Euclidean) and L-inf (Chebyshev), placing more weight on the largest "
                "component difference. Tests whether non-standard p values help."
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
    """Demonstrate from-scratch KNN on wine quality classification.

    Domain: Wine quality assessment.
    Features: fixed_acidity, residual_sugar, alcohol_content, pH_level
    Target: quality_class (0 = standard, 1 = premium)
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Wine Quality Classification (From Scratch)")
    logger.info("=" * 60)

    np.random.seed(42)
    n_samples = 600

    # Generate wine features.
    fixed_acidity = np.random.normal(8.0, 1.5, n_samples).clip(4.0, 14.0)
    residual_sugar = np.random.exponential(2.5, n_samples).clip(0.5, 20.0)
    alcohol_content = np.random.normal(10.5, 1.2, n_samples).clip(8.0, 15.0)
    pH_level = np.random.normal(3.3, 0.15, n_samples).clip(2.7, 4.0)

    X = np.column_stack([fixed_acidity, residual_sugar, alcohol_content, pH_level])
    feature_names = ["fixed_acidity", "residual_sugar", "alcohol_content", "pH_level"]

    # Create quality labels.
    quality_score = (
        0.3 * (alcohol_content - 10.0) - 0.2 * (fixed_acidity - 7.0)
        - 0.1 * (residual_sugar - 2.0) + 0.1 * (3.3 - pH_level) * 10
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

    model = train(X_train, y_train, n_neighbors=7, weights="distance", metric="euclidean")
    validate(model, X_val, y_val)
    test(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete KNN (from scratch) pipeline."""
    logger.info("=" * 60)
    logger.info("K-Nearest Neighbors - NumPy From-Scratch Implementation")
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
    study.optimize(optuna_objective, n_trials=20)
    logger.info(f"Best F1: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    logger.info("\n--- Step 6: Ray Tune Search Space ---")
    ray_tune_search()

    logger.info("\n--- Step 7: Training Best Model ---")
    best_model = train(X_train, y_train, **study.best_trial.params)

    logger.info("\n--- Step 8: Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    logger.info("\n--- Step 9: Real-World Demo ---")
    real_world_demo()

    logger.info("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    main()
