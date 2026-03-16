"""
K-Nearest Neighbors Classifier - PyTorch (GPU-Accelerated) Implementation
===========================================================================

Theory & Mathematics:
    This implements KNN using PyTorch tensors and GPU operations for accelerated
    pairwise distance computation and top-k neighbor selection.

    GPU Advantage for KNN:
        KNN's prediction bottleneck is computing pairwise distances between query
        points and ALL training points. This is a massively parallel operation:
        - N_query * N_train independent distance computations
        - Each distance involves a dot product over D features

        GPUs excel at exactly this pattern: thousands of simple, independent
        arithmetic operations executed in parallel by CUDA cores.

    Vectorized Pairwise Distance (PyTorch):
        For Euclidean distance, we use the expansion:
            ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y

        In PyTorch:
            x_sq = torch.sum(X_query ** 2, dim=1, keepdim=True)  # (n_query, 1)
            y_sq = torch.sum(X_train ** 2, dim=1, keepdim=True)  # (n_train, 1)
            cross = torch.mm(X_query, X_train.T)                  # (n_query, n_train)
            dist = torch.sqrt(x_sq + y_sq.T - 2 * cross)          # (n_query, n_train)

    Top-K Selection (torch.topk):
        Instead of full sorting (O(n log n)), torch.topk uses a partial sort
        (O(n + k log k)) to find the K smallest/largest elements. On GPU, this
        is implemented with parallel reduction, making it extremely fast.

        WHY torch.topk: Full sorting is wasteful when we only need K elements.
        topk is both asymptotically faster and GPU-optimized.

    Key Operations:
        - torch.cdist: Computes batched pairwise distances (L1, L2, Lp)
        - torch.topk: Efficient K-smallest element selection
        - torch.scatter_add_: Aggregates votes by class (for voting)

Business Use Cases:
    - Large-scale similarity search (millions of training points)
    - Real-time recommendation systems with GPU inference
    - Image retrieval (nearest neighbors in embedding space)
    - Approximate nearest neighbor with GPU acceleration

Advantages:
    - GPU acceleration: 10-100x speedup for large datasets
    - torch.cdist: optimized pairwise distance for multiple metrics
    - torch.topk: efficient partial sort for neighbor selection
    - Seamless integration with PyTorch deep learning pipelines
    - Can use learned embeddings as features (from neural networks)

Disadvantages:
    - Requires GPU for speed advantage (CPU mode is slower than sklearn)
    - Memory limited: full distance matrix must fit in GPU memory
    - No KD-tree/Ball-tree optimization (brute force only)
    - More complex setup than sklearn
    - Overkill for small datasets
"""

# --- Standard library imports ---
import logging  # Structured logging
import warnings  # Warning suppression
from typing import Any, Dict, Tuple  # Type annotations

# --- Third-party imports ---
import numpy as np  # Numerical computing
import torch  # PyTorch for GPU-accelerated computation
import optuna  # Bayesian HPO

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
# GPU-Accelerated KNN Classifier
# ---------------------------------------------------------------------------

class KNNPyTorch:
    """K-Nearest Neighbors classifier using PyTorch for GPU acceleration.

    Key design decisions:
    - Uses torch.cdist for flexible, optimized pairwise distance computation
    - Uses torch.topk for efficient K-nearest selection without full sort
    - Stores training data as GPU tensors for fast repeated access
    - Supports uniform and distance-weighted voting

    WHY PyTorch for KNN: While KNN has no learnable parameters, the distance
    computation and neighbor selection are embarrassingly parallel operations
    that benefit enormously from GPU acceleration.
    """

    def __init__(
        self,
        n_neighbors: int = 5,      # Number of nearest neighbors
        weights: str = "uniform",   # Voting: "uniform" or "distance"
        metric: str = "euclidean",  # Distance metric
        p: float = 2.0,            # Minkowski p parameter
    ):
        """Initialize KNN with PyTorch backend.

        Args:
            n_neighbors: K value for neighbor count.
            weights: Voting scheme.
            metric: Distance metric name.
            p: Minkowski p parameter.
        """
        self.n_neighbors = n_neighbors  # Store K
        self.weights = weights          # Store voting scheme
        self.metric = metric            # Store metric name
        self.p = p                      # Store Minkowski p
        self.X_train_tensor = None      # Training features (torch tensor)
        self.y_train_tensor = None      # Training labels (torch tensor)
        self.n_classes = 0              # Number of classes
        self.device = torch.device(     # Auto-detect GPU
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNPyTorch":
        """Store training data as GPU tensors.

        WHY tensors on GPU: Moving data to GPU once avoids repeated CPU->GPU
        transfers during prediction. All distance computations happen on GPU.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self for method chaining.
        """
        # Convert numpy arrays to PyTorch tensors and move to device.
        # WHY float32: Standard GPU precision. float64 is 2x slower on GPU.
        self.X_train_tensor = torch.FloatTensor(X).to(self.device)  # Features on GPU
        self.y_train_tensor = torch.LongTensor(y.astype(int)).to(self.device)  # Labels on GPU
        self.n_classes = len(np.unique(y))  # Count unique classes

        logger.info(
            f"KNN (PyTorch) fit: {X.shape[0]} samples, {X.shape[1]} features, "
            f"device={self.device}"
        )
        return self

    def _compute_distances(self, X_query: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances using torch.cdist or manual computation.

        torch.cdist computes batched pairwise distances efficiently:
            D[i,j] = ||X_query[i] - X_train[j]||_p

        WHY torch.cdist: It's PyTorch's built-in optimized function for
        pairwise distances, supporting L1, L2, and Lp norms. On GPU,
        it uses cuBLAS for the matrix multiplication components.

        Args:
            X_query: Query tensor of shape (n_queries, n_features).

        Returns:
            Distance matrix of shape (n_queries, n_train).
        """
        if self.metric == "euclidean":
            # Use torch.cdist with p=2 for Euclidean distance.
            # WHY cdist: Optimized for GPU, handles numerical stability internally.
            distances = torch.cdist(X_query, self.X_train_tensor, p=2.0)

        elif self.metric == "manhattan":
            # Use torch.cdist with p=1 for Manhattan distance.
            distances = torch.cdist(X_query, self.X_train_tensor, p=1.0)

        elif self.metric == "minkowski":
            # Use torch.cdist with arbitrary p for Minkowski distance.
            distances = torch.cdist(X_query, self.X_train_tensor, p=self.p)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return distances  # Shape: (n_queries, n_train)

    def _find_k_nearest(
        self, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find K nearest neighbors using torch.topk.

        torch.topk efficiently finds the K smallest elements without full sort.
        On GPU, this uses parallel reduction (much faster than sequential sort).

        WHY topk with largest=False: We want the K SMALLEST distances
        (closest neighbors), not the largest.

        Args:
            distances: Pairwise distance matrix (n_queries, n_train).

        Returns:
            Tuple of (knn_distances, knn_indices), both (n_queries, K).
        """
        # torch.topk finds K smallest distances and their indices.
        # WHY largest=False: We want nearest (smallest distance) neighbors.
        # WHY sorted=True: Neighbors are returned in order of increasing distance.
        knn_distances, knn_indices = torch.topk(
            distances,                     # Distance matrix to search
            k=self.n_neighbors,            # Number of neighbors
            dim=1,                         # Search along training sample dimension
            largest=False,                 # Find K SMALLEST (nearest)
            sorted=True,                   # Sort results by distance
        )

        return knn_distances, knn_indices

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using GPU-accelerated KNN.

        Pipeline: numpy -> tensor -> distances -> topk -> vote -> numpy.

        Args:
            X: Query features of shape (n_queries, n_features).

        Returns:
            Predicted labels of shape (n_queries,).
        """
        # Convert query data to GPU tensor.
        X_query = torch.FloatTensor(X).to(self.device)  # Move queries to GPU

        with torch.no_grad():  # No gradient tracking needed for KNN
            # Compute pairwise distances.
            distances = self._compute_distances(X_query)  # (n_queries, n_train)

            # Find K nearest neighbors.
            knn_distances, knn_indices = self._find_k_nearest(distances)  # (n_queries, K)

            # Get labels of nearest neighbors.
            knn_labels = self.y_train_tensor[knn_indices]  # (n_queries, K)

            # Perform voting.
            n_queries = X_query.shape[0]
            predictions = torch.zeros(n_queries, dtype=torch.long, device=self.device)

            for i in range(n_queries):  # Vote for each query
                if self.weights == "uniform":
                    # Uniform voting: count occurrences of each class.
                    counts = torch.bincount(knn_labels[i], minlength=self.n_classes)
                    predictions[i] = torch.argmax(counts)  # Majority class

                elif self.weights == "distance":
                    # Distance-weighted voting.
                    dists = knn_distances[i]  # Distances to K neighbors
                    inv_weights = 1.0 / (dists + 1e-10)  # Inverse distance weights

                    # Accumulate weighted votes.
                    weighted_counts = torch.zeros(self.n_classes, device=self.device)
                    for j in range(self.n_neighbors):
                        weighted_counts[knn_labels[i, j]] += inv_weights[j]

                    predictions[i] = torch.argmax(weighted_counts)  # Highest weight

        return predictions.cpu().numpy()  # Move back to CPU and convert to numpy

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Query features of shape (n_queries, n_features).

        Returns:
            Probability matrix of shape (n_queries, n_classes).
        """
        X_query = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            distances = self._compute_distances(X_query)
            knn_distances, knn_indices = self._find_k_nearest(distances)
            knn_labels = self.y_train_tensor[knn_indices]

            n_queries = X_query.shape[0]
            proba = torch.zeros(n_queries, self.n_classes, device=self.device)

            for i in range(n_queries):
                if self.weights == "uniform":
                    counts = torch.bincount(knn_labels[i], minlength=self.n_classes).float()
                    proba[i] = counts / self.n_neighbors
                elif self.weights == "distance":
                    dists = knn_distances[i]
                    inv_weights = 1.0 / (dists + 1e-10)
                    for j in range(self.n_neighbors):
                        proba[i, knn_labels[i, j]] += inv_weights[j]
                    total = proba[i].sum()
                    if total > 0:
                        proba[i] /= total

        return proba.cpu().numpy()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate, split, and standardize synthetic classification data.

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

    # Standardize for distance-based methods.
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
    p: float = 2.0,
) -> KNNPyTorch:
    """Create and fit a PyTorch KNN classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_neighbors: Number of neighbors (K).
        weights: Voting scheme.
        metric: Distance metric.
        p: Minkowski p parameter.

    Returns:
        Fitted KNNPyTorch model.
    """
    logger.info(
        f"Training KNN (PyTorch): k={n_neighbors}, weights={weights}, metric={metric}"
    )

    model = KNNPyTorch(
        n_neighbors=n_neighbors, weights=weights, metric=metric, p=p
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    logger.info(f"Training accuracy: {train_acc:.4f}")

    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: KNNPyTorch,
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
    model: KNNPyTorch,
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
    """Optuna objective for PyTorch KNN HPO.

    Returns:
        Validation F1 score.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
    p = trial.suggest_float("p", 1.0, 5.0) if metric == "minkowski" else 2.0

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
        "p": {"type": "uniform", "lower": 1.0, "upper": 5.0},
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
    """Compare KNN configurations (same as numpy version for consistency).

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    configs = {
        "k1_euclidean_uniform": {
            "params": {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"},
            "reasoning": (
                "K=1 nearest neighbor. Maximum variance, zero bias on training data. "
                "Tests if GPU acceleration helps with brute-force single-neighbor lookup. "
                "Expected: overfitting on noisy data, perfect on clean data."
            ),
        },
        "k5_euclidean_distance": {
            "params": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
            "reasoning": (
                "K=5 with distance-weighted Euclidean. Standard balanced configuration. "
                "Distance weighting gives closer neighbors more influence. "
                "Expected: good generalization, reduced noise sensitivity vs K=1."
            ),
        },
        "k20_manhattan_distance": {
            "params": {"n_neighbors": 20, "weights": "distance", "metric": "manhattan"},
            "reasoning": (
                "K=20 with Manhattan distance (L1 norm). Large K for smooth boundaries. "
                "L1 is more robust to outliers and works better in high dimensions. "
                "torch.cdist supports L1 natively on GPU."
            ),
        },
        "k10_minkowski_p3": {
            "params": {"n_neighbors": 10, "weights": "distance", "metric": "minkowski", "p": 3.0},
            "reasoning": (
                "K=10 with Minkowski p=3. Non-standard p value between L2 and L-inf. "
                "Emphasizes larger per-component differences more than L2. "
                "Tests torch.cdist with arbitrary p on GPU."
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
    """Demonstrate PyTorch KNN on wine quality classification.

    Domain: Wine quality assessment.
    Features: fixed_acidity, residual_sugar, alcohol_content, pH_level
    Target: quality_class (0 = standard, 1 = premium)
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Wine Quality Classification (PyTorch KNN)")
    logger.info("=" * 60)

    np.random.seed(42)
    n_samples = 600

    # Wine chemistry features.
    fixed_acidity = np.random.normal(8.0, 1.5, n_samples).clip(4.0, 14.0)
    residual_sugar = np.random.exponential(2.5, n_samples).clip(0.5, 20.0)
    alcohol_content = np.random.normal(10.5, 1.2, n_samples).clip(8.0, 15.0)
    pH_level = np.random.normal(3.3, 0.15, n_samples).clip(2.7, 4.0)

    X = np.column_stack([fixed_acidity, residual_sugar, alcohol_content, pH_level])
    feature_names = ["fixed_acidity", "residual_sugar", "alcohol_content", "pH_level"]

    # Quality labels.
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

    model = train(X_train, y_train, n_neighbors=7, weights="distance")
    validate(model, X_val, y_val)
    test(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete PyTorch KNN pipeline."""
    logger.info("=" * 60)
    logger.info("K-Nearest Neighbors - PyTorch (GPU-Accelerated) Implementation")
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
