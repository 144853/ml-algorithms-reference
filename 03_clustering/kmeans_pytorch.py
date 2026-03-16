"""
K-Means Clustering - PyTorch (GPU-Accelerated) Implementation
==============================================================

Theory & Mathematics:
    K-Means minimises the Within-Cluster Sum of Squares (inertia):

        J = sum_{k=1}^{K} sum_{x_i in C_k} ||x_i - mu_k||_2^2

    This implementation uses PyTorch tensors for all distance computations,
    enabling GPU acceleration on CUDA-capable hardware.  The algorithm is
    identical to Lloyd's iteration (assign -> update -> repeat) but the
    assignment step leverages batched matrix operations on the GPU.

    Distance computation (vectorised):
        ||x_i - mu_k||^2  =  ||x_i||^2  -  2 * x_i . mu_k  +  ||mu_k||^2

    The outer-product trick makes the (N, K) distance matrix a single matrix
    multiply plus two broadcasts -- this is extremely fast on a GPU even for
    large N and moderate K.

    Initialisation:
        - random   : Uniform random selection of K data points.
        - kmeans++ : D^2 weighted sampling (implemented with torch operations).

    Convergence:
        Same monotonic decrease guarantee as standard Lloyd's.

    Complexity:
        Time  : O(N * K * d * I)  -- each step is GPU-parallel in N and K.
        Space : O(N * K) for the distance matrix  (GPU memory bound).

Business Use Cases:
    - Real-time customer segmentation on large-scale streaming data.
    - Image colour quantisation with millions of pixels.
    - Embedding clustering (BERT, CLIP) on GPU where embeddings already reside.
    - Feature engineering at scale in deep-learning pipelines.

Advantages:
    - Orders-of-magnitude speed-up over CPU for large N.
    - Seamless integration with PyTorch pipelines (gradients not needed but
      tensors stay on device, avoiding CPU-GPU transfers).
    - Easy to batch across multiple initialisations in parallel.

Disadvantages:
    - GPU memory limits the maximum N*K distance matrix.
    - Overhead for small datasets (CPU may be faster).
    - Still inherits fundamental K-Means limitations (spherical clusters, K).

Hyperparameters:
    - n_clusters : int    - Number of clusters K.
    - init       : str    - 'random' or 'kmeans++'.
    - max_iter   : int    - Maximum Lloyd iterations.
    - tol        : float  - Convergence tolerance.
    - n_init     : int    - Number of restarts.
"""

# -- Standard library imports --
import logging  # Structured logging for tracking pipeline stages
import warnings  # Suppress non-critical deprecation warnings
from typing import Any, Dict, Optional, Tuple  # Type annotations for function signatures

# -- Third-party imports --
import numpy as np  # CPU-side array operations, random number generation, metrics input
import optuna  # Bayesian hyperparameter optimisation framework
import ray  # Distributed computing for parallel HPO trials
import torch  # PyTorch: GPU-accelerated tensor computation library
from ray import tune  # Ray's hyperparameter tuning interface
from sklearn.datasets import make_blobs  # Synthetic Gaussian blob generator
from sklearn.metrics import (
    adjusted_rand_score,  # Chance-adjusted label agreement [-1, 1]
    calinski_harabasz_score,  # Between/within cluster variance ratio (higher=better)
    davies_bouldin_score,  # Average cluster similarity (lower=better)
    normalized_mutual_info_score,  # Normalised mutual information [0, 1]
    silhouette_score,  # Cluster cohesion vs separation [-1, 1]
)
from sklearn.model_selection import train_test_split  # Data splitting utility

# -- Configure structured logging --
logging.basicConfig(
    level=logging.INFO,  # Show INFO level and above
    format="%(asctime)s | %(levelname)-8s | %(message)s",  # Timestamped format
)
logger = logging.getLogger(__name__)  # Module-level logger

# Suppress FutureWarning to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning)

# Select the best available device: GPU (CUDA) if available, otherwise CPU
# WHY: GPU provides massive speedup for the matrix-multiply-heavy distance computation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("PyTorch device: %s", DEVICE)


# ---------------------------------------------------------------------------
# K-Means with PyTorch tensors
# ---------------------------------------------------------------------------

class KMeansPyTorch:
    """K-Means clustering using PyTorch for GPU-accelerated distance computation.

    Parameters
    ----------
    n_clusters : int
    init : str
        'random' or 'kmeans++'.
    max_iter : int
    tol : float
    n_init : int
    random_state : int or None
    device : torch.device
    """

    def __init__(
        self,
        n_clusters: int = 5,
        init: str = "kmeans++",
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int = 42,
        device: torch.device = DEVICE,
    ) -> None:
        # Store hyperparameters
        self.n_clusters = n_clusters  # K: number of clusters
        self.init = init  # Initialisation strategy
        self.max_iter = max_iter  # Max Lloyd iterations
        self.tol = tol  # Convergence tolerance
        self.n_init = n_init  # Independent restarts
        self.random_state = random_state  # Seed
        self.device = device  # GPU or CPU

        # Fitted attributes
        self.centroids_: Optional[torch.Tensor] = None  # (K, d) on device
        self.labels_: Optional[np.ndarray] = None  # (N,) on CPU
        self.inertia_: float = np.inf  # WCSS of best run
        self.n_iter_: int = 0  # Iterations in best run

    # -- distance computation ------------------------------------------------

    @staticmethod
    def _pairwise_sq_distances(
        X: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Compute squared Euclidean distances between X and C.

        Returns tensor of shape (N, K).
        Uses expansion: ||x-c||^2 = ||x||^2 - 2*x.c + ||c||^2
        WHY: Avoids creating (N, K, d) intermediate; uses fast matmul instead.
        """
        XX = (X * X).sum(dim=1, keepdim=True)   # (N, 1): ||x_i||^2
        CC = (C * C).sum(dim=1, keepdim=True).T  # (1, K): ||c_k||^2
        XC = X @ C.T                             # (N, K): x_i . c_k
        dists = XX - 2.0 * XC + CC  # (N, K): broadcast addition
        # Clamp to zero for numerical safety (tiny negative values from floating point)
        return torch.clamp(dists, min=0.0)

    # -- initialisation ------------------------------------------------------

    def _init_random(
        self, X: torch.Tensor, rng: np.random.RandomState
    ) -> torch.Tensor:
        """Select K random data points as initial centroids.
        WHY: Simplest init; fast but may place centroids in same cluster.
        """
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx].clone()  # Clone to avoid aliasing

    def _init_kmeanspp(
        self, X: torch.Tensor, rng: np.random.RandomState
    ) -> torch.Tensor:
        """K-Means++ init: spreads centroids using D^2 weighted sampling.
        WHY: O(log K)-competitive; dramatically reduces bad initialisations.
        """
        n = len(X)
        centroids = torch.empty(
            self.n_clusters, X.shape[1], dtype=X.dtype, device=X.device
        )
        # First centroid: uniform random
        centroids[0] = X[rng.randint(n)]

        for k in range(1, self.n_clusters):
            # Compute squared distances to nearest existing centroid
            dists = self._pairwise_sq_distances(X, centroids[:k])  # (N, k)
            min_dists, _ = dists.min(dim=1)  # (N,): min distance to any centroid
            # Convert to probability distribution (D^2 weighting)
            probs = min_dists / min_dists.sum()
            # Transfer to CPU for numpy random choice (GPU random choice not trivial)
            probs_np = probs.cpu().numpy()
            idx = rng.choice(n, p=probs_np)
            centroids[k] = X[idx]

        return centroids

    # -- core ----------------------------------------------------------------

    def _assign(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Assignment step: assign each point to nearest centroid.
        Returns label tensor (N,) of cluster indices.
        """
        dists = self._pairwise_sq_distances(X, C)  # (N, K)
        return dists.argmin(dim=1)  # Index of closest centroid

    def _update(
        self, X: torch.Tensor, labels: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Update step: recompute centroids as cluster means.
        WHY: Mean minimises WCSS for fixed assignments.
        """
        new_C = torch.zeros_like(C)
        for k in range(self.n_clusters):
            mask = labels == k  # Boolean mask for cluster k members
            if mask.any():
                new_C[k] = X[mask].mean(dim=0)  # Mean of members
            else:
                # Empty cluster: reinitialise to random point
                new_C[k] = X[torch.randint(len(X), (1,))]
        return new_C

    def _inertia(
        self, X: torch.Tensor, labels: torch.Tensor, C: torch.Tensor
    ) -> float:
        """Compute WCSS (inertia) for the current assignment.
        WHY: Used to compare multiple runs and select the best one.
        """
        dists = self._pairwise_sq_distances(X, C)  # (N, K)
        # Sum the distance of each point to its assigned centroid
        return float(dists[torch.arange(len(X)), labels].sum().item())

    def _fit_single(
        self, X: torch.Tensor, rng: np.random.RandomState
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        """Single K-Means run. Returns (centroids, labels, inertia, n_iter)."""
        # Initialise centroids
        if self.init == "kmeans++":
            C = self._init_kmeanspp(X, rng)
        else:
            C = self._init_random(X, rng)

        # Lloyd's iteration: assign -> update -> check convergence
        for it in range(1, self.max_iter + 1):
            labels = self._assign(X, C)  # Assign points to nearest centroid
            new_C = self._update(X, labels, C)  # Recompute centroids
            # Maximum centroid shift across all clusters
            shift = float(torch.sqrt(((new_C - C) ** 2).sum(dim=1)).max().item())
            C = new_C
            if shift < self.tol:  # Convergence check
                break

        inertia = self._inertia(X, labels, C)
        return C, labels, inertia, it

    def fit(self, X: np.ndarray) -> "KMeansPyTorch":
        """Fit model. Accepts numpy array, converts to tensor internally.
        WHY numpy input: maintains consistency with sklearn API.
        """
        # Convert numpy array to float32 tensor on the target device
        # WHY float32: sufficient precision for clustering, saves GPU memory vs float64
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        rng = np.random.RandomState(self.random_state)

        best_inertia = np.inf
        for _ in range(self.n_init):
            seed = rng.randint(0, 2**31)
            C, labels, inertia, iters = self._fit_single(
                X_t, np.random.RandomState(seed)
            )
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids_ = C
                self.labels_ = labels.cpu().numpy()  # Transfer labels to CPU
                self.inertia_ = inertia
                self.n_iter_ = iters

        logger.info(
            "KMeansPyTorch fitted: K=%d  inertia=%.4f  iters=%d  device=%s",
            self.n_clusters, self.inertia_, self.n_iter_, self.device,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        labels = self._assign(X_t, self.centroids_)
        return labels.cpu().numpy()  # Return as numpy array


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1500,
    n_clusters: int = 5,
    n_features: int = 2,
    cluster_std: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic blob data for K-Means."""
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_clusters,
        cluster_std=cluster_std, random_state=random_state,
    )
    logger.info("Generated %d samples, %d clusters, %dD.", n_samples, n_clusters, n_features)
    return X, y


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, **hyperparams) -> KMeansPyTorch:
    """Fit KMeansPyTorch model with sensible defaults."""
    defaults = {
        "n_clusters": 5, "init": "kmeans++", "max_iter": 300,
        "tol": 1e-4, "n_init": 10, "random_state": 42, "device": DEVICE,
    }
    defaults.update(hyperparams)
    model = KMeansPyTorch(**defaults)
    model.fit(X_train)
    return model


def _compute_metrics(
    X: np.ndarray, labels_pred: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Clustering quality metrics (internal and optional external)."""
    n_labels = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    metrics: Dict[str, float] = {}

    if n_labels >= 2:
        metrics["silhouette_score"] = float(silhouette_score(X, labels_pred))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels_pred))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, labels_pred))
    else:
        logger.warning("Only %d cluster(s); internal metrics skipped.", n_labels)

    if labels_true is not None:
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(labels_true, labels_pred))
        metrics["normalised_mutual_info"] = float(normalized_mutual_info_score(labels_true, labels_pred))

    return metrics


def validate(X_val: np.ndarray, model: KMeansPyTorch,
             labels_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Validate on validation set."""
    labels_pred = model.predict(X_val)
    metrics = _compute_metrics(X_val, labels_pred, labels_true)
    logger.info("Validation metrics: %s", metrics)
    return metrics


def test(X_test: np.ndarray, model: KMeansPyTorch,
         labels_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Evaluate on holdout test set."""
    labels_pred = model.predict(X_test)
    metrics = _compute_metrics(X_test, labels_pred, labels_true)
    logger.info("Test metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray,
                     y_train_true: Optional[np.ndarray] = None,
                     y_val_true: Optional[np.ndarray] = None) -> float:
    """Optuna objective maximising silhouette score."""
    n_clusters = trial.suggest_int("n_clusters", 2, 15)
    init = trial.suggest_categorical("init", ["kmeans++", "random"])
    max_iter = trial.suggest_int("max_iter", 100, 500, step=50)
    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
    n_init = trial.suggest_int("n_init", 1, 10)

    model = train(X_train, n_clusters=n_clusters, init=init,
                  max_iter=max_iter, tol=tol, n_init=n_init)
    metrics = validate(X_val, model, y_val_true)
    return metrics.get("silhouette_score", -1.0)


def run_optuna(X_train: np.ndarray, X_val: np.ndarray,
               y_train_true: Optional[np.ndarray] = None,
               y_val_true: Optional[np.ndarray] = None,
               n_trials: int = 30) -> optuna.Study:
    """Run Optuna study."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="kmeans_pytorch")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, y_train_true, y_val_true),
        n_trials=n_trials,
    )
    logger.info("Optuna best: value=%.4f  params=%s", study.best_trial.value, study.best_trial.params)
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(X_train: np.ndarray, X_val: np.ndarray,
                    y_train_true: Optional[np.ndarray] = None,
                    y_val_true: Optional[np.ndarray] = None,
                    num_samples: int = 20) -> Any:
    """Ray Tune hyperparameter search."""
    def _trainable(config: dict) -> None:
        # Force CPU inside Ray workers to avoid GPU contention across parallel trials
        cfg = {k: v for k, v in config.items()}
        cfg["device"] = torch.device("cpu")
        model = train(X_train, **cfg)
        metrics = validate(X_val, model, y_val_true)
        tune.report(
            silhouette_score=metrics.get("silhouette_score", -1.0),
            calinski_harabasz_score=metrics.get("calinski_harabasz_score", 0.0),
            davies_bouldin_score=metrics.get("davies_bouldin_score", 999.0),
        )

    search_space = {
        "n_clusters": tune.randint(2, 16), "init": tune.choice(["kmeans++", "random"]),
        "max_iter": tune.choice([100, 200, 300, 400, 500]),
        "tol": tune.loguniform(1e-6, 1e-2), "n_init": tune.randint(1, 11),
        "random_state": 42,
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    tuner = tune.Tuner(
        _trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="silhouette_score", mode="max", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="silhouette_score", mode="max")
    logger.info("Ray Tune best: %s  silhouette=%.4f", best.config, best.metrics["silhouette_score"])
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison Study
# ---------------------------------------------------------------------------

def compare_parameter_sets() -> None:
    """Compare K-Means parameter configurations using the PyTorch implementation.

    Tests k=3/5/10 and init strategies to show silhouette score changes.
    All distance computations happen on the selected device (GPU/CPU).
    """
    logger.info("=" * 70)
    logger.info("PARAMETER COMPARISON STUDY (PyTorch)")
    logger.info("=" * 70)

    # Generate test data with 5 known clusters
    X, y_true = make_blobs(n_samples=1500, centers=5, n_features=2,
                           cluster_std=1.0, random_state=42)

    configs = [
        # (k, description, best_for)
        (3, "UNDER-CLUSTERING",
         "Coarse grouping; executive summaries. Merges real clusters."),
        (5, "CORRECT K",
         "When true k is known. Expected best silhouette and ARI."),
        (10, "OVER-CLUSTERING",
         "Feature engineering; sub-cluster discovery. Fragments natural groups."),
    ]

    results = {}
    for k, desc, best_for in configs:
        logger.info("-" * 40)
        logger.info("Config: k=%d (%s)", k, desc)
        logger.info("  Best for: %s", best_for)
        # Use our PyTorch implementation
        model = KMeansPyTorch(n_clusters=k, init="kmeans++", n_init=10,
                              random_state=42, device=DEVICE)
        model.fit(X)
        sil = silhouette_score(X, model.labels_)
        ari = adjusted_rand_score(y_true, model.labels_)
        logger.info("  Silhouette=%.4f  ARI=%.4f  Inertia=%.1f", sil, ari, model.inertia_)
        results[k] = sil

    # Init comparison
    logger.info("-" * 40)
    logger.info("Init comparison: kmeans++ vs random (k=5)")
    for init_m in ["kmeans++", "random"]:
        m = KMeansPyTorch(n_clusters=5, init=init_m, n_init=10, random_state=42, device=DEVICE)
        m.fit(X)
        sil = silhouette_score(X, m.labels_)
        logger.info("  init='%s': Silhouette=%.4f  Inertia=%.1f", init_m, sil, m.inertia_)

    logger.info("SUMMARY: k=3:%.4f  k=5:%.4f  k=10:%.4f", results[3], results[5], results[10])


# ---------------------------------------------------------------------------
# Real-World Use Case: Customer Segmentation
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Customer segmentation demo using the PyTorch GPU-accelerated K-Means.

    Domain: Retail customer segmentation with annual_income, spending_score, age.
    Demonstrates that the PyTorch implementation produces equivalent results
    to sklearn/numpy versions while leveraging GPU acceleration.
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Customer Segmentation (PyTorch)")
    logger.info("=" * 70)

    # Generate realistic synthetic customer data
    rng = np.random.RandomState(42)

    # Segment 1: High income, high spenders (VIP loyalists)
    n1 = 200
    seg1 = np.column_stack([rng.normal(85, 10, n1), rng.normal(75, 8, n1), rng.normal(35, 7, n1)])

    # Segment 2: High income, low spenders (savers)
    n2 = 200
    seg2 = np.column_stack([rng.normal(90, 12, n2), rng.normal(25, 7, n2), rng.normal(50, 8, n2)])

    # Segment 3: Low income, high spenders (aspirational)
    n3 = 200
    seg3 = np.column_stack([rng.normal(30, 8, n3), rng.normal(70, 10, n3), rng.normal(28, 5, n3)])

    # Segment 4: Low income, low spenders (budget-conscious)
    n4 = 300
    seg4 = np.column_stack([rng.normal(35, 10, n4), rng.normal(30, 10, n4), rng.normal(45, 12, n4)])

    X_raw = np.vstack([seg1, seg2, seg3, seg4])
    y_true = np.array([0]*n1 + [1]*n2 + [2]*n3 + [3]*n4)

    logger.info("Dataset: %d customers, features: [income, spending, age]", len(X_raw))

    # Standardise features
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_raw)

    # Optimal k search
    logger.info("Searching for optimal k...")
    best_k, best_sil = 2, -1.0
    for k in range(2, 8):
        km = KMeansPyTorch(n_clusters=k, init="kmeans++", n_init=10,
                           random_state=42, device=DEVICE)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        logger.info("  k=%d: silhouette=%.4f", k, sil)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    logger.info("Optimal k=%d (silhouette=%.4f)", best_k, best_sil)

    # Final model
    final = KMeansPyTorch(n_clusters=best_k, init="kmeans++", n_init=10,
                          random_state=42, device=DEVICE)
    final.fit(X_scaled)

    # Analyse segments
    logger.info("Discovered segments:")
    for seg_id in range(best_k):
        mask = final.labels_ == seg_id
        seg_data = X_raw[mask]
        logger.info(
            "  Seg %d: n=%d, income=$%.0fk, spending=%.0f, age=%.0f",
            seg_id, mask.sum(), seg_data[:, 0].mean(),
            seg_data[:, 1].mean(), seg_data[:, 2].mean(),
        )

    ari = adjusted_rand_score(y_true, final.labels_)
    logger.info("ARI vs true segments: %.4f", ari)
    logger.info("Demo complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline for PyTorch K-Means."""
    logger.info("=" * 70)
    logger.info("K-Means Clustering - PyTorch Pipeline (device=%s)", DEVICE)
    logger.info("=" * 70)

    # 1. Data
    X, y = generate_data(n_samples=1500, n_clusters=5)

    # 2. Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    logger.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # 3. Baseline
    logger.info("-" * 35 + " Baseline " + "-" * 35)
    baseline = train(X_train, n_clusters=5)
    validate(X_val, baseline, y_val)
    test(X_test, baseline, y_test)

    # 4. Optuna
    logger.info("-" * 35 + " Optuna " + "-" * 35)
    study = run_optuna(X_train, X_val, y_train, y_val, n_trials=30)

    # 5. Ray Tune
    logger.info("-" * 35 + " Ray Tune " + "-" * 35)
    ray_results = ray_tune_search(X_train, X_val, y_train, y_val, num_samples=20)

    # 6. Final evaluation
    best_params_optuna = study.best_trial.params
    best_ray = ray_results.get_best_result(metric="silhouette_score", mode="max")
    best_params_ray = {
        k: v for k, v in best_ray.config.items()
        if k in ("n_clusters", "init", "max_iter", "tol", "n_init")
    }

    if study.best_trial.value >= best_ray.metrics["silhouette_score"]:
        best_params = best_params_optuna
        logger.info("Using Optuna best (sil=%.4f).", study.best_trial.value)
    else:
        best_params = best_params_ray
        logger.info("Using Ray Tune best (sil=%.4f).", best_ray.metrics["silhouette_score"])

    logger.info("-" * 35 + " Final " + "-" * 35)
    final_model = train(X_train, **best_params)
    final_metrics = test(X_test, final_model, y_test)
    logger.info("Final test metrics: %s", final_metrics)

    # 7. Parameter comparison and real-world demo
    compare_parameter_sets()
    real_world_demo()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
