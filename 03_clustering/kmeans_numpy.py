"""
K-Means Clustering - NumPy (From Scratch) Implementation
=========================================================

Theory & Mathematics:
    K-Means partitions N data points {x_1, ..., x_N} in R^d into K disjoint
    clusters C_1, ..., C_K by minimising the Within-Cluster Sum of Squares:

        J(C, mu) = sum_{k=1}^{K} sum_{x_i in C_k} ||x_i - mu_k||_2^2

    This is an NP-hard combinatorial problem; Lloyd's algorithm provides a
    greedy, iterative heuristic that converges to a local minimum.

    Lloyd's Algorithm:
        1. INITIALISE centroids mu_1, ..., mu_K.
        2. ASSIGNMENT: For every point x_i, assign it to the cluster whose
           centroid is closest:
               c_i = argmin_k ||x_i - mu_k||_2^2
        3. UPDATE: Recompute each centroid as the arithmetic mean of its
           assigned points:
               mu_k = (1 / |C_k|) * sum_{x_i in C_k} x_i
        4. Repeat 2-3 until convergence (centroid shift < tol) or max_iter.

    Initialisation strategies implemented here:
        - Random:    Select K data points uniformly at random as centroids.
        - K-Means++: Arthur & Vassilvitskii (2007).  Pick the first centroid
                     uniformly at random; pick each subsequent centroid with
                     probability proportional to D(x)^2, where D(x) is the
                     distance from x to the nearest already-chosen centroid.
                     Expected approximation ratio: O(log K).

    Convergence:
        J is bounded below (>= 0) and each step is non-increasing, so Lloyd's
        algorithm is guaranteed to converge.  However, it can converge to a
        local minimum that is arbitrarily worse than the global optimum.

    Complexity:
        Time  : O(N * K * d * I)   per run
        Space : O(N * d + K * d)

Business Use Cases:
    - Customer segmentation based on behavioural features.
    - Image colour quantisation (reduce colour palette).
    - Pre-processing step for more complex algorithms.
    - Anomaly detection via distance to nearest centroid.

Advantages:
    - Straightforward to implement and understand.
    - Efficient for large N when K is small.
    - Guaranteed convergence.

Disadvantages:
    - Must pre-specify K.
    - Sensitive to initialisation (k-means++ helps).
    - Assumes spherical, isotropic clusters.
    - Outlier-sensitive (centroids get pulled).
    - Only discovers convex boundaries.

Hyperparameters:
    - n_clusters : int    - Number of clusters K.
    - init       : str    - 'random' or 'kmeans++'.
    - max_iter   : int    - Maximum number of Lloyd iterations.
    - tol        : float  - Convergence tolerance on centroid movement.
    - n_init     : int    - Number of independent restarts (best is kept).
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline progress tracking
import warnings  # Suppresses non-critical warnings from dependencies
from typing import Any, Dict, Optional, Tuple  # Type annotations for clarity

# -- Third-party imports --
import numpy as np  # Core numerical library for array operations and linear algebra
import optuna  # Bayesian hyperparameter optimisation framework
import ray  # Distributed computing framework for parallel HPO
from ray import tune  # Ray's hyperparameter tuning module
from sklearn.datasets import make_blobs  # Generates synthetic Gaussian blobs for testing
from sklearn.metrics import (
    adjusted_rand_score,  # Chance-adjusted agreement between true and predicted labels
    calinski_harabasz_score,  # Between/within cluster dispersion ratio (higher=better)
    davies_bouldin_score,  # Average cluster similarity ratio (lower=better)
    normalized_mutual_info_score,  # Information-theoretic clustering quality [0,1]
    silhouette_score,  # Cluster cohesion vs separation measure [-1,1]
)
from sklearn.model_selection import train_test_split  # Splits data into train/val/test

# -- Configure structured logging --
# WHY this format: timestamps help track timing, level helps filter severity
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)  # Module-level logger

# Suppress FutureWarning from sklearn to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# K-Means from scratch
# ---------------------------------------------------------------------------

class KMeansNumpy:
    """Pure-NumPy K-Means implementation (Lloyd's algorithm).

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    init : str
        Initialisation method: 'random' or 'kmeans++'.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance (max centroid shift).
    n_init : int
        Number of independent runs; the best (lowest inertia) is kept.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        init: str = "kmeans++",
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        # Store all hyperparameters as instance attributes
        self.n_clusters = n_clusters  # K: how many clusters to discover
        self.init = init  # Initialisation strategy: 'random' or 'kmeans++'
        self.max_iter = max_iter  # Maximum Lloyd iterations before forced stop
        self.tol = tol  # Stop when largest centroid shift is below this
        self.n_init = n_init  # Number of independent runs to mitigate bad init
        self.random_state = random_state  # Seed for reproducibility

        # Fitted attributes -- populated after calling fit()
        self.centroids_: Optional[np.ndarray] = None  # (K, d) centroid positions
        self.labels_: Optional[np.ndarray] = None  # (N,) cluster assignments
        self.inertia_: float = np.inf  # WCSS of the best run (lower = better)
        self.n_iter_: int = 0  # Number of iterations in the best run

    # -- initialisation methods -----------------------------------------------

    def _init_random(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Select K points uniformly at random from X as initial centroids.

        WHY random init: Simplest approach. Fast but can lead to poor convergence
        if unlucky points are chosen (e.g., two centroids in the same cluster).
        """
        # Choose K unique indices from [0, N) without replacement
        # WHY replace=False: ensures we pick K distinct data points as centroids
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        # Return copies of the selected points (avoid aliasing with original data)
        return X[idx].copy()

    def _init_kmeanspp(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """K-Means++ initialisation (Arthur & Vassilvitskii, 2007).

        WHY k-means++: Spreads initial centroids apart by choosing each new
        centroid with probability proportional to its squared distance from
        the nearest existing centroid. This yields O(log K)-competitive
        solutions in expectation, dramatically reducing bad initialisations.
        """
        n = len(X)  # Total number of data points
        # Pre-allocate centroid array
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)

        # First centroid: chosen uniformly at random from all data points
        # WHY uniform: no information about cluster structure yet
        centroids[0] = X[rng.randint(n)]

        # Remaining centroids: chosen with D^2 weighting
        for k in range(1, self.n_clusters):
            # Compute squared distances from every point to every existing centroid
            # X[:, None, :] has shape (N, 1, d), centroids[None, :k, :] has shape (1, k, d)
            # Broadcasting gives (N, k, d), sum over d gives (N, k) distances
            dists = np.min(
                np.sum((X[:, None, :] - centroids[None, :k, :]) ** 2, axis=2),
                axis=1,  # Take min distance across existing centroids -> (N,)
            )
            # Convert distances to probabilities (proportional to D^2)
            # WHY D^2: points far from all centroids are more likely to be chosen,
            # ensuring good spread of initial centroids
            probs = dists / dists.sum()
            # Sample next centroid according to these probabilities
            centroids[k] = X[rng.choice(n, p=probs)]
        return centroids

    # -- core algorithm -------------------------------------------------------

    @staticmethod
    def _compute_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between all points and centroids.

        Returns shape (N, K) where element [i, k] = ||x_i - mu_k||^2.

        WHY expansion trick: Instead of computing (X - C)^2 directly (which would
        require a (N, K, d) intermediate), we use the algebraic expansion:
            ||x - c||^2 = ||x||^2 - 2*x.c + ||c||^2
        This reduces memory from O(N*K*d) to O(N*K) and leverages fast matrix
        multiplication (BLAS-optimised) for the x.c term.
        """
        # ||x_i||^2 for each point: shape (N, 1)
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        # ||c_k||^2 for each centroid: shape (K, 1)
        CC = np.sum(centroids ** 2, axis=1, keepdims=True)
        # x_i . c_k for all pairs: shape (N, K) via matrix multiplication
        XC = X @ centroids.T
        # Combine: ||x-c||^2 = ||x||^2 - 2*x.c + ||c||^2, broadcast to (N, K)
        return XX - 2 * XC + CC.T

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assignment step: assign each point to its nearest centroid.

        Returns label array of shape (N,) where label[i] is the cluster index.
        WHY argmin: each point goes to the centroid with minimum squared distance.
        """
        dists = self._compute_distances(X, centroids)  # (N, K)
        return np.argmin(dists, axis=1)  # Index of closest centroid for each point

    def _update(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Update step: recompute centroids as the mean of assigned points.

        WHY mean: the centroid that minimises WCSS for a fixed assignment is the
        arithmetic mean of the cluster members (proof via calculus/derivatives).
        """
        new_centroids = np.empty_like(centroids)  # Allocate new centroid array
        for k in range(self.n_clusters):
            # Extract all points assigned to cluster k
            members = X[labels == k]
            if len(members) > 0:
                # Compute mean of all members as the new centroid
                new_centroids[k] = members.mean(axis=0)
            else:
                # Handle empty cluster: reinitialise to a random data point
                # WHY: Empty clusters can occur with bad initialisation; without
                # this fix, the centroid would be undefined (NaN)
                new_centroids[k] = X[np.random.randint(len(X))]
        return new_centroids

    def _compute_inertia(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> float:
        """Compute Within-Cluster Sum of Squares (WCSS/inertia).

        WHY inertia: this is the objective function K-Means minimises.
        Used to compare different runs and select the best one.
        """
        dists = self._compute_distances(X, centroids)  # (N, K) squared distances
        # Sum the squared distance of each point to its assigned centroid
        return float(np.sum(dists[np.arange(len(X)), labels]))

    def _fit_single(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Execute a single K-Means run (one initialisation).

        Returns (centroids, labels, inertia, n_iter).
        WHY separate method: allows n_init runs to each call this independently.
        """
        # Initialise centroids using the selected method
        if self.init == "kmeans++":
            centroids = self._init_kmeanspp(X, rng)
        else:
            centroids = self._init_random(X, rng)

        # Lloyd's iteration loop: assign -> update -> check convergence
        for it in range(1, self.max_iter + 1):
            labels = self._assign(X, centroids)  # Step 1: assign points to nearest centroid
            new_centroids = self._update(X, labels, centroids)  # Step 2: recompute centroids
            # Compute maximum centroid shift (Euclidean distance) across all centroids
            # WHY max shift: if the largest centroid moved less than tol, all have converged
            shift = np.max(np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)))
            centroids = new_centroids  # Update centroids for next iteration
            if shift < self.tol:
                # Convergence achieved: centroids are stable
                break

        # Compute final inertia for this run
        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, it

    def fit(self, X: np.ndarray) -> "KMeansNumpy":
        """Run K-Means n_init times and keep the best result (lowest inertia).

        WHY multiple runs: K-Means can converge to local minima. Running
        multiple times with different initialisations and keeping the best
        result significantly improves the chance of finding a good solution.
        """
        # Create a master RNG from the global seed
        rng = np.random.RandomState(self.random_state)

        best_inertia = np.inf  # Track the best (lowest) inertia seen so far
        for run in range(self.n_init):
            # Generate a unique seed for this run from the master RNG
            seed = rng.randint(0, 2**31)
            # Execute one complete K-Means run
            c, l, inertia, iters = self._fit_single(X, np.random.RandomState(seed))
            # Keep this run's results if its inertia is the best so far
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids_ = c  # Best centroids
                self.labels_ = l  # Best labels
                self.inertia_ = inertia  # Best inertia
                self.n_iter_ = iters  # Iterations used in best run

        logger.info(
            "KMeansNumpy fitted: K=%d  inertia=%.4f  best_iters=%d",
            self.n_clusters,
            self.inertia_,
            self.n_iter_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest centroid from the fitted model.

        WHY separate predict: allows scoring new data without re-fitting,
        essential for validation/test evaluation.
        """
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._assign(X, self.centroids_)


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
    """Generate synthetic blob data for K-Means.

    WHY make_blobs: creates isotropic Gaussian clusters that match K-Means
    assumptions, providing an ideal testbed for algorithm validation.
    """
    X, y = make_blobs(
        n_samples=n_samples,  # Total points to generate
        n_features=n_features,  # Dimensionality of each point
        centers=n_clusters,  # Number of cluster centres
        cluster_std=cluster_std,  # Standard deviation of each cluster
        random_state=random_state,  # Reproducibility seed
    )
    logger.info(
        "Generated %d samples, %d clusters, %dD.",
        n_samples,
        n_clusters,
        n_features,
    )
    return X, y


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, **hyperparams) -> KMeansNumpy:
    """Fit the from-scratch KMeansNumpy model.

    WHY wrapper function: provides a consistent interface that matches
    the sklearn version, making it easy to swap implementations.
    """
    # Sensible defaults matching the sklearn version
    defaults = {
        "n_clusters": 5,  # Match synthetic data structure
        "init": "kmeans++",  # Better initialisation than random
        "max_iter": 300,  # Enough iterations for convergence
        "tol": 1e-4,  # Convergence threshold
        "n_init": 10,  # Multiple restarts for robustness
        "random_state": 42,  # Reproducibility
    }
    defaults.update(hyperparams)  # Override with user-specified params
    model = KMeansNumpy(**defaults)  # Instantiate the from-scratch model
    model.fit(X_train)  # Run Lloyd's algorithm n_init times
    return model


def _compute_metrics(
    X: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute internal and (optionally) external clustering metrics.

    WHY both internal and external: internal metrics (silhouette, CH, DB) work
    without ground truth and are used for model selection; external metrics
    (ARI, NMI) require ground truth and measure how well we recovered true labels.
    """
    # Count actual clusters (exclude noise label -1 if present)
    n_labels = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    metrics: Dict[str, float] = {}

    if n_labels >= 2:
        # Silhouette: how well each point fits its cluster vs nearest neighbour cluster
        metrics["silhouette_score"] = float(silhouette_score(X, labels_pred))
        # Calinski-Harabasz: ratio of between-cluster to within-cluster variance
        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(X, labels_pred)
        )
        # Davies-Bouldin: average similarity between each cluster and its most similar one
        metrics["davies_bouldin_score"] = float(
            davies_bouldin_score(X, labels_pred)
        )
    else:
        logger.warning("Only %d cluster(s) found; internal metrics skipped.", n_labels)

    if labels_true is not None:
        # ARI: chance-adjusted pairwise agreement measure
        metrics["adjusted_rand_index"] = float(
            adjusted_rand_score(labels_true, labels_pred)
        )
        # NMI: normalised information-theoretic agreement
        metrics["normalised_mutual_info"] = float(
            normalized_mutual_info_score(labels_true, labels_pred)
        )

    return metrics


def validate(
    X_val: np.ndarray,
    model: KMeansNumpy,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Validate model on a validation set by predicting and computing metrics."""
    labels_pred = model.predict(X_val)  # Assign val points to learned centroids
    metrics = _compute_metrics(X_val, labels_pred, labels_true)
    logger.info("Validation metrics: %s", metrics)
    return metrics


def test(
    X_test: np.ndarray,
    model: KMeansNumpy,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate model on a holdout test set (final, unbiased evaluation)."""
    labels_pred = model.predict(X_test)  # Assign test points to learned centroids
    metrics = _compute_metrics(X_test, labels_pred, labels_true)
    logger.info("Test metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
) -> float:
    """Optuna objective maximising validation silhouette score."""
    # Suggest hyperparameters from the search space
    n_clusters = trial.suggest_int("n_clusters", 2, 15)
    init = trial.suggest_categorical("init", ["kmeans++", "random"])
    max_iter = trial.suggest_int("max_iter", 100, 500, step=50)
    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
    n_init = trial.suggest_int("n_init", 1, 10)

    # Train with suggested params and evaluate on validation set
    model = train(
        X_train,
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
    )
    metrics = validate(X_val, model, y_val_true)
    return metrics.get("silhouette_score", -1.0)


def run_optuna(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
    n_trials: int = 30,
) -> optuna.Study:
    """Run Optuna study to find optimal hyperparameters."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna log noise
    study = optuna.create_study(direction="maximize", study_name="kmeans_numpy")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, y_train_true, y_val_true),
        n_trials=n_trials,
    )
    logger.info(
        "Optuna best: value=%.4f  params=%s",
        study.best_trial.value,
        study.best_trial.params,
    )
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
    num_samples: int = 20,
) -> Any:
    """Ray Tune hyperparameter search for distributed HPO."""
    def _trainable(config: dict) -> None:
        model = train(X_train, **config)
        metrics = validate(X_val, model, y_val_true)
        tune.report(
            silhouette_score=metrics.get("silhouette_score", -1.0),
            calinski_harabasz_score=metrics.get("calinski_harabasz_score", 0.0),
            davies_bouldin_score=metrics.get("davies_bouldin_score", 999.0),
        )

    search_space = {
        "n_clusters": tune.randint(2, 16),
        "init": tune.choice(["kmeans++", "random"]),
        "max_iter": tune.choice([100, 200, 300, 400, 500]),
        "tol": tune.loguniform(1e-6, 1e-2),
        "n_init": tune.randint(1, 11),
        "random_state": 42,
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="silhouette_score",
            mode="max",
            num_samples=num_samples,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="silhouette_score", mode="max")
    logger.info(
        "Ray Tune best: %s  silhouette=%.4f",
        best.config,
        best.metrics["silhouette_score"],
    )
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison Study
# ---------------------------------------------------------------------------

def compare_parameter_sets() -> None:
    """Compare different K-Means parameter configurations and explain trade-offs.

    Tests k=3, k=5, k=10 and init='kmeans++' vs 'random' to show how each
    parameter affects clustering quality on the from-scratch implementation.
    """
    logger.info("=" * 70)
    logger.info("PARAMETER COMPARISON STUDY (NumPy from-scratch)")
    logger.info("=" * 70)

    # Generate data with 5 true clusters for comparison baseline
    X, y_true = make_blobs(n_samples=1500, centers=5, n_features=2,
                           cluster_std=1.0, random_state=42)

    # ---- k=3: Under-clustering ----
    # WHY: With fewer clusters than reality, K-Means merges distinct groups.
    # BEST FOR: High-level executive summaries where 3 tiers suffice (e.g., gold/silver/bronze).
    # RISK: Loses important distinctions between merged clusters.
    logger.info("-" * 40)
    logger.info("Config 1: k=3 (UNDER-CLUSTERING)")
    logger.info("  Best for: Coarse segmentation, broad category assignment.")
    model_k3 = KMeansNumpy(n_clusters=3, init="kmeans++", n_init=10, random_state=42)
    model_k3.fit(X)
    sil_k3 = silhouette_score(X, model_k3.labels_)
    ari_k3 = adjusted_rand_score(y_true, model_k3.labels_)
    logger.info("  Silhouette=%.4f  ARI=%.4f  Inertia=%.1f", sil_k3, ari_k3, model_k3.inertia_)

    # ---- k=5: Correct k ----
    # WHY: Matches true cluster count. Should maximise silhouette and ARI.
    # BEST FOR: When you have used elbow method or domain knowledge to select k.
    logger.info("-" * 40)
    logger.info("Config 2: k=5 (CORRECT K)")
    logger.info("  Best for: When true number of clusters is known or estimated.")
    model_k5 = KMeansNumpy(n_clusters=5, init="kmeans++", n_init=10, random_state=42)
    model_k5.fit(X)
    sil_k5 = silhouette_score(X, model_k5.labels_)
    ari_k5 = adjusted_rand_score(y_true, model_k5.labels_)
    logger.info("  Silhouette=%.4f  ARI=%.4f  Inertia=%.1f", sil_k5, ari_k5, model_k5.inertia_)

    # ---- k=10: Over-clustering ----
    # WHY: Splits natural clusters into fragments. Inertia drops but silhouette may worsen.
    # BEST FOR: Feature engineering where fine-grained cluster IDs are used as features.
    # RISK: Over-segmentation reduces interpretability.
    logger.info("-" * 40)
    logger.info("Config 3: k=10 (OVER-CLUSTERING)")
    logger.info("  Best for: Feature engineering, discovering sub-cluster patterns.")
    model_k10 = KMeansNumpy(n_clusters=10, init="kmeans++", n_init=10, random_state=42)
    model_k10.fit(X)
    sil_k10 = silhouette_score(X, model_k10.labels_)
    ari_k10 = adjusted_rand_score(y_true, model_k10.labels_)
    logger.info("  Silhouette=%.4f  ARI=%.4f  Inertia=%.1f", sil_k10, ari_k10, model_k10.inertia_)

    # ---- init='random' vs 'kmeans++' ----
    # WHY: k-means++ spreads centroids; random may cluster them together.
    logger.info("-" * 40)
    logger.info("Config 4: init='random' vs init='kmeans++'")
    for init_method in ["kmeans++", "random"]:
        m = KMeansNumpy(n_clusters=5, init=init_method, n_init=10, random_state=42)
        m.fit(X)
        sil = silhouette_score(X, m.labels_)
        logger.info("  init='%s': Silhouette=%.4f  Inertia=%.1f  Iters=%d",
                     init_method, sil, m.inertia_, m.n_iter_)

    # ---- Summary ----
    logger.info("-" * 40)
    logger.info("SUMMARY: k=3 sil=%.4f | k=5 sil=%.4f | k=10 sil=%.4f",
                sil_k3, sil_k5, sil_k10)


# ---------------------------------------------------------------------------
# Real-World Use Case: Customer Segmentation
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate from-scratch K-Means on a customer segmentation scenario.

    Domain: Retail company segmenting customers by annual_income, spending_score, age.
    Uses the NumPy from-scratch implementation to show it produces similar results
    to the sklearn version while giving full control over the algorithm.
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Customer Segmentation (NumPy from-scratch)")
    logger.info("=" * 70)

    # -- Generate realistic synthetic customer data --
    rng = np.random.RandomState(42)

    # Segment 1: High income, high spenders (VIP loyalists)
    n1 = 200
    seg1 = np.column_stack([rng.normal(85, 10, n1), rng.normal(75, 8, n1), rng.normal(35, 7, n1)])

    # Segment 2: High income, low spenders (savers)
    n2 = 200
    seg2 = np.column_stack([rng.normal(90, 12, n2), rng.normal(25, 7, n2), rng.normal(50, 8, n2)])

    # Segment 3: Low income, high spenders (aspirational buyers)
    n3 = 200
    seg3 = np.column_stack([rng.normal(30, 8, n3), rng.normal(70, 10, n3), rng.normal(28, 5, n3)])

    # Segment 4: Low income, low spenders (budget-conscious)
    n4 = 300
    seg4 = np.column_stack([rng.normal(35, 10, n4), rng.normal(30, 10, n4), rng.normal(45, 12, n4)])

    # Combine all segments
    X_raw = np.vstack([seg1, seg2, seg3, seg4])
    y_true = np.array([0]*n1 + [1]*n2 + [2]*n3 + [3]*n4)

    feature_names = ["annual_income ($k)", "spending_score", "age"]
    logger.info("Dataset: %d customers, features: %s", len(X_raw), feature_names)

    # Standardise features for distance-based clustering
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_raw)

    # Find optimal k via silhouette analysis
    logger.info("Searching for optimal k...")
    best_k, best_sil = 2, -1.0
    for k in range(2, 8):
        km = KMeansNumpy(n_clusters=k, init="kmeans++", n_init=10, random_state=42)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        logger.info("  k=%d: silhouette=%.4f", k, sil)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    logger.info("Optimal k=%d (silhouette=%.4f)", best_k, best_sil)

    # Train final model
    final = KMeansNumpy(n_clusters=best_k, init="kmeans++", n_init=10, random_state=42)
    final.fit(X_scaled)

    # Analyse discovered segments using original (unscaled) features
    logger.info("Discovered segments:")
    for seg_id in range(best_k):
        mask = final.labels_ == seg_id
        seg_data = X_raw[mask]
        logger.info(
            "  Seg %d: n=%d, avg_income=$%.0fk, avg_spending=%.0f, avg_age=%.0f",
            seg_id, mask.sum(), seg_data[:, 0].mean(), seg_data[:, 1].mean(), seg_data[:, 2].mean(),
        )

    ari = adjusted_rand_score(y_true, final.labels_)
    logger.info("ARI vs true segments: %.4f", ari)
    logger.info("Customer segmentation demo complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline for from-scratch K-Means."""
    logger.info("=" * 70)
    logger.info("K-Means Clustering - NumPy (From Scratch) Pipeline")
    logger.info("=" * 70)

    # 1. Data
    X, y = generate_data(n_samples=1500, n_clusters=5)

    # 2. Split: 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
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

    # 6. Best model evaluation
    best_params_optuna = study.best_trial.params
    best_ray = ray_results.get_best_result(metric="silhouette_score", mode="max")
    best_params_ray = {
        k: v for k, v in best_ray.config.items()
        if k in ("n_clusters", "init", "max_iter", "tol", "n_init")
    }

    if study.best_trial.value >= best_ray.metrics["silhouette_score"]:
        best_params = best_params_optuna
        logger.info("Optuna wins (sil=%.4f).", study.best_trial.value)
    else:
        best_params = best_params_ray
        logger.info("Ray wins (sil=%.4f).", best_ray.metrics["silhouette_score"])

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
