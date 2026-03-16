"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- PyTorch (GPU-Accelerated Distance Matrix) Implementation
====================================================================

Theory & Mathematics:
    DBSCAN discovers clusters of arbitrary shape by examining the local
    density around each point.  Given two parameters -- eps (neighbourhood
    radius) and min_samples (minimum density) -- the algorithm classifies
    every point as:

        Core point:   |N_eps(p)| >= min_samples
        Border point: not core, but within eps of a core point
        Noise point:  neither core nor border  (label = -1)

    Clusters are formed by connecting core points whose eps-neighbourhoods
    overlap, and absorbing all border points reachable from those cores.

    This implementation accelerates the most expensive step -- computing the
    full pairwise Euclidean distance matrix -- with PyTorch tensor operations
    on GPU (or CPU if CUDA is unavailable).  The cluster expansion logic
    itself runs on CPU using a BFS queue, since it is inherently sequential.

    Distance matrix (N x N):
        ||x_i - x_j||^2  =  ||x_i||^2  -  2 * x_i . x_j  +  ||x_j||^2

    This is computed as a single batched matrix multiply plus two broadcasts,
    which is extremely fast on GPU.  The resulting distance matrix is then
    transferred to CPU (as a boolean adjacency matrix thresholded at eps)
    for the sequential cluster expansion.

    Algorithm Steps:
        1. Compute pairwise distance matrix on GPU  -> O(N^2 * d) on GPU.
        2. Threshold at eps to get boolean adjacency -> on GPU.
        3. Count neighbours per point               -> on GPU.
        4. Identify core points                     -> on GPU.
        5. Transfer adjacency + core mask to CPU.
        6. BFS cluster expansion on CPU              -> O(N^2) worst case.

    Complexity:
        Time  : O(N^2 * d) for distances (GPU-parallel), O(N^2) for expansion.
        Space : O(N^2) for the distance / adjacency matrix (GPU memory bound).

Business Use Cases:
    - Large-scale anomaly detection where distance computation is the
      bottleneck (e.g., network traffic, sensor arrays).
    - Geospatial event clustering with millions of records.
    - Embedding-space outlier detection (BERT / CLIP vectors on GPU).
    - Real-time dense point-cloud segmentation (LiDAR, depth cameras).

Advantages:
    - GPU acceleration makes O(N^2) distance computation practical for
      much larger N than pure CPU.
    - Seamless integration with deep learning pipelines where embeddings
      are already on GPU.
    - Produces the same results as CPU DBSCAN (deterministic for core/noise).

Disadvantages:
    - GPU memory limits maximum N (N^2 matrix).
    - Cluster expansion is still sequential on CPU.
    - Sensitive to eps and min_samples choice.
    - Poor on varying-density data.

Hyperparameters:
    - eps         : float  - Neighbourhood radius.
    - min_samples : int    - Minimum neighbours for core status.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

# logging provides structured, levelled output (INFO / WARNING / ERROR)
# for tracing every pipeline step with timestamps.
import logging

# warnings lets us suppress FutureWarning noise from scikit-learn / NumPy
# so our logs remain focused on algorithmic messages.
import warnings

# deque is a double-ended queue with O(1) append/popleft, ideal for the
# BFS cluster expansion where we repeatedly pop from the front.
from collections import deque

# Type hints make function signatures self-documenting and enable static
# analysis tools to catch bugs before runtime.
from typing import Any, Dict, Optional, Tuple

# NumPy is used for the CPU-side BFS expansion and metrics computation.
# WHY: the BFS is inherently sequential, so GPU acceleration would not help.
import numpy as np

# Optuna performs Bayesian hyperparameter optimisation using TPE,
# converging faster than grid or random search on the (eps, min_samples) space.
import optuna

# Ray is a distributed computing framework for parallel trial execution.
import ray

# PyTorch provides GPU-accelerated tensor operations for the O(N^2 * d)
# distance matrix computation, which is the computational bottleneck.
import torch

# tune is Ray's hyperparameter search module.
from ray import tune

# make_blobs and make_moons generate test data: moons for non-convex shapes,
# blobs for compact Gaussian clusters.
from sklearn.datasets import make_blobs, make_moons

# Five complementary clustering metrics:
# - adjusted_rand_score: chance-corrected label agreement (external)
# - calinski_harabasz_score: between/within-cluster variance ratio
# - davies_bouldin_score: average cluster similarity (lower = better)
# - normalized_mutual_info_score: entropy-based label agreement (external)
# - silhouette_score: intra-cluster cohesion vs inter-cluster separation
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# train_test_split creates reproducible data partitions for train/val/test.
from sklearn.model_selection import train_test_split

# StandardScaler z-normalises features to mean=0, std=1 so that eps has
# a consistent meaning across features.
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------

# Configure root logger with timestamp, severity, and message format.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# Module-level logger for attributable log messages.
logger = logging.getLogger(__name__)

# Suppress FutureWarning clutter from downstream libraries.
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# Device selection
# ------------------------------------------------------------------

# Select GPU if available, otherwise fall back to CPU.
# WHY: the pairwise distance computation is the bottleneck and benefits
# enormously from GPU parallelism (thousands of CUDA cores computing
# distances simultaneously vs sequential CPU cores).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log which device was selected so the user knows whether GPU is active.
logger.info("PyTorch device: %s", DEVICE)

# ------------------------------------------------------------------
# Label constants
# ------------------------------------------------------------------

# NOISE = -1 is the standard DBSCAN convention for outlier points
# that belong to no cluster.
NOISE = -1

# UNCLASSIFIED = -2 distinguishes unvisited points from noise points
# during the BFS expansion phase.
UNCLASSIFIED = -2


# ---------------------------------------------------------------------------
# DBSCAN with PyTorch-accelerated distances
# ---------------------------------------------------------------------------

class DBSCANPyTorch:
    """DBSCAN with GPU-accelerated pairwise distance computation.

    The pairwise Euclidean distance matrix and neighbourhood determination
    are computed on GPU using PyTorch.  The cluster expansion BFS runs on
    CPU with NumPy arrays.

    Parameters
    ----------
    eps : float
        Maximum distance for neighbourhood membership.
    min_samples : int
        Minimum points in eps-neighbourhood for core status.
    device : torch.device
        Compute device for distance matrix.
    """

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 5,
        device: torch.device = DEVICE,
    ) -> None:
        # eps defines the neighbourhood radius for density estimation.
        # WHY: smaller eps = tighter clusters, larger eps = more merging.
        self.eps = eps

        # min_samples is the density threshold for core point status.
        # WHY: higher values require denser regions to form clusters,
        # producing fewer but more robust clusters.
        self.min_samples = min_samples

        # device controls whether distance computation runs on GPU or CPU.
        # WHY: GPU can compute the O(N^2) distance matrix orders of magnitude
        # faster than CPU via thousands of parallel CUDA cores.
        self.device = device

        # Fitted attributes initialised to None until fit() is called.
        # WHY: allows checking whether the model has been fitted.
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0

    # -- GPU distance computation -------------------------------------------

    @staticmethod
    def _pairwise_distances_gpu(
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Euclidean distance matrix (N, N) on GPU.

        Uses ||x-y||^2 = ||x||^2 - 2*x.y + ||y||^2.
        """
        # Compute squared L2 norms for each row.  keepdim=True gives (N, 1)
        # for broadcasting in the next step.
        # WHY: pre-computing norms once avoids redundant O(N*d) work.
        XX = (X * X).sum(dim=1, keepdim=True)  # (N, 1)

        # Expand using the algebraic identity: one BLAS matmul + two broadcasts.
        # WHY: this is the GPU's sweet spot -- massive matrix multiply
        # executed in parallel across thousands of CUDA cores.
        D_sq = XX - 2.0 * (X @ X.T) + XX.T     # (N, N)

        # Clamp negative values (floating-point rounding artefacts) to zero.
        # WHY: sqrt of negative values produces NaN; clamping prevents this
        # while keeping the mathematical result correct.
        D_sq = torch.clamp(D_sq, min=0.0)

        # Return Euclidean distances (element-wise square root).
        # WHY: DBSCAN is defined in terms of L2 distance, not squared distance.
        return torch.sqrt(D_sq)

    def _compute_adjacency_and_core(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (adjacency, is_core) computed on GPU, transferred to CPU.

        adjacency : bool ndarray (N, N) - True where dist <= eps.
        is_core   : bool ndarray (N,)   - True for core points.
        """
        # Convert NumPy array to a PyTorch tensor on the selected device.
        # WHY: float32 is sufficient for distance computation and uses half
        # the GPU memory of float64, allowing larger datasets.
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Compute the full N x N distance matrix on GPU.
        # WHY: this is the O(N^2 * d) step that benefits most from GPU.
        dist_matrix = self._pairwise_distances_gpu(X_t)

        # Threshold distances at eps to get a boolean adjacency matrix.
        # WHY: DBSCAN only cares whether two points are within eps, not
        # the exact distance; boolean matrix uses less memory.
        adjacency_t = dist_matrix <= self.eps         # (N, N) bool

        # Count neighbours per point (sum of True values in each row).
        # WHY: a point is core if its neighbour count >= min_samples.
        neighbour_counts = adjacency_t.sum(dim=1)     # (N,) includes self

        # Identify core points using the density threshold.
        is_core_t = neighbour_counts >= self.min_samples

        # Transfer results from GPU to CPU for the sequential BFS expansion.
        # WHY: BFS is inherently sequential (each step depends on the previous),
        # so it cannot benefit from GPU parallelism.
        adjacency = adjacency_t.cpu().numpy()
        is_core = is_core_t.cpu().numpy()

        # Free GPU memory explicitly to avoid out-of-memory errors on
        # subsequent operations.
        # WHY: the N^2 distance matrix can be very large (e.g. 10000 points
        # = 400 MB in float32); releasing it immediately is important.
        del X_t, dist_matrix, adjacency_t, neighbour_counts, is_core_t
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return adjacency, is_core

    # -- CPU cluster expansion -----------------------------------------------

    @staticmethod
    def _expand_cluster(
        labels: np.ndarray,
        adjacency: np.ndarray,
        point_idx: int,
        cluster_id: int,
        is_core: np.ndarray,
    ) -> None:
        """BFS expansion from a core point."""
        # Assign the seed core point to the current cluster.
        # WHY: the seed is guaranteed to be a core point by the caller.
        labels[point_idx] = cluster_id

        # Get initial neighbours from the pre-computed adjacency matrix.
        # WHY: np.where on a boolean row is much faster than recomputing distances.
        neighbours = np.where(adjacency[point_idx])[0]

        # Initialise BFS queue with all neighbours of the seed.
        queue: deque = deque(neighbours.tolist())

        # Process queue until empty.
        while queue:
            q = queue.popleft()

            # Absorb noise points as border points of this cluster.
            if labels[q] == NOISE:
                labels[q] = cluster_id

            # Skip already-classified points to avoid infinite loops.
            if labels[q] != UNCLASSIFIED:
                continue

            # Classify this unclassified point into the current cluster.
            labels[q] = cluster_id

            # If q is a core point, extend the frontier with its neighbours.
            # WHY: only core points propagate the cluster boundary.
            if is_core[q]:
                q_neighbours = np.where(adjacency[q])[0]
                queue.extend(q_neighbours.tolist())

    # -- fit -----------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCANPyTorch":
        """Run DBSCAN with GPU-accelerated distances.

        Parameters
        ----------
        X : ndarray of shape (N, d)

        Returns
        -------
        self
        """
        # Get dataset size for label array initialisation.
        n = len(X)

        # Compute adjacency matrix and core-point mask on GPU, then transfer
        # to CPU for the sequential BFS expansion.
        # WHY: this hybrid approach uses GPU for the heavy lifting (O(N^2*d)
        # distances) and CPU for the sequential part (BFS expansion).
        adjacency, is_core = self._compute_adjacency_and_core(X)

        # Initialise all labels to UNCLASSIFIED.
        labels = np.full(n, UNCLASSIFIED, dtype=int)

        # Cluster counter starts at 0 and increments for each new cluster.
        cluster_id = 0

        # Main DBSCAN loop: visit each point exactly once.
        for i in range(n):
            # Skip already-classified points.
            if labels[i] != UNCLASSIFIED:
                continue

            # Non-core points are tentatively labelled as noise.
            # WHY: they may later be absorbed as border points.
            if not is_core[i]:
                labels[i] = NOISE
                continue

            # Core point seeds a new cluster via BFS expansion.
            self._expand_cluster(labels, adjacency, i, cluster_id, is_core)
            cluster_id += 1

        # Store fitted attributes.
        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        self.n_clusters_ = cluster_id

        # Log summary including device information.
        n_noise = int((labels == NOISE).sum())
        logger.info(
            "DBSCANPyTorch fitted: eps=%.4f  min_samples=%d  "
            "clusters=%d  noise=%d  device=%s",
            self.eps,
            self.min_samples,
            self.n_clusters_,
            n_noise,
            self.device,
        )
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit and return labels."""
        self.fit(X)
        return self.labels_

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters."""
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
        }


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1500,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mixed moons + blobs data for DBSCAN.

    Returns
    -------
    X : ndarray (n, 2) - standardised
    y : ndarray (n,)   - ground-truth labels
    """
    # Split total samples between moons (non-convex) and blobs (spherical).
    # WHY: tests DBSCAN on both shape types simultaneously.
    n_moons = n_samples // 2
    n_blobs = n_samples - n_moons

    # Generate interleaving half-circle clusters with low noise.
    # WHY: non-convex shapes that K-Means cannot handle but DBSCAN excels at.
    X_moons, y_moons = make_moons(
        n_samples=n_moons, noise=0.08, random_state=random_state
    )

    # Generate three spherical clusters at specified centres.
    # WHY: provides a mix of easy (compact) and challenging (non-convex) shapes.
    X_blobs, y_blobs = make_blobs(
        n_samples=n_blobs,
        centers=[[2.5, 2.5], [-2.0, 3.0], [3.5, -1.0]],
        cluster_std=0.35,
        random_state=random_state,
    )

    # Offset blob labels to avoid collision with moon labels.
    y_blobs += y_moons.max() + 1

    # Combine into a single dataset.
    X = np.vstack([X_moons, X_blobs])
    y = np.concatenate([y_moons, y_blobs])

    # Standardise features for consistent eps interpretation.
    # WHY: without standardisation, features with larger scales dominate
    # the Euclidean distance, making eps hard to interpret.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logger.info(
        "Generated %d samples (%d moons + %d blobs), %d true clusters.",
        len(X),
        n_moons,
        n_blobs,
        len(set(y)),
    )
    return X, y


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, **hyperparams) -> Tuple[DBSCANPyTorch, np.ndarray]:
    """Fit DBSCANPyTorch on training data.

    Returns (model, labels).
    """
    # Merge caller-provided hyperparameters with sensible defaults.
    # WHY: enables both manual and automated (Optuna/Ray) usage.
    defaults: Dict[str, Any] = {
        "eps": 0.3,
        "min_samples": 5,
        "device": DEVICE,
    }
    defaults.update(hyperparams)

    # Create and fit a fresh model.
    # WHY: DBSCAN is transductive, so each fit creates new labels.
    model = DBSCANPyTorch(**defaults)
    labels = model.fit_predict(X_train)
    return model, labels


def _compute_metrics(
    X: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Internal and external clustering metrics."""
    # Count true clusters (exclude noise label -1).
    unique = set(labels_pred)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_non_noise = int((labels_pred != -1).sum())

    metrics: Dict[str, float] = {
        "n_clusters": float(n_clusters),
        "n_noise": float((labels_pred == -1).sum()),
    }

    # Internal metrics require at least 2 clusters and 2 non-noise points.
    if n_clusters >= 2 and n_non_noise >= 2:
        mask = labels_pred != -1
        # Silhouette: cohesion vs separation, ranges [-1, +1].
        metrics["silhouette_score"] = float(
            silhouette_score(X[mask], labels_pred[mask])
        )
        # Calinski-Harabasz: between/within variance ratio (higher = better).
        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(X[mask], labels_pred[mask])
        )
        # Davies-Bouldin: cluster similarity index (lower = better).
        metrics["davies_bouldin_score"] = float(
            davies_bouldin_score(X[mask], labels_pred[mask])
        )
    else:
        logger.warning(
            "Clusters=%d, non-noise=%d; internal metrics skipped.",
            n_clusters,
            n_non_noise,
        )

    # External metrics when ground-truth is available.
    if labels_true is not None:
        metrics["adjusted_rand_index"] = float(
            adjusted_rand_score(labels_true, labels_pred)
        )
        metrics["normalised_mutual_info"] = float(
            normalized_mutual_info_score(labels_true, labels_pred)
        )

    return metrics


def validate(
    X_val: np.ndarray,
    model: DBSCANPyTorch,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Validate: re-fit DBSCAN with same params on validation data."""
    # Re-fit because DBSCAN is transductive (no predict for unseen data).
    params = model.get_params()
    val_model = DBSCANPyTorch(**params)
    labels_pred = val_model.fit_predict(X_val)
    metrics = _compute_metrics(X_val, labels_pred, labels_true)
    logger.info("Validation metrics: %s", metrics)
    return metrics


def test(
    X_test: np.ndarray,
    model: DBSCANPyTorch,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate on holdout test set (re-fits with same hyperparameters)."""
    params = model.get_params()
    test_model = DBSCANPyTorch(**params)
    labels_pred = test_model.fit_predict(X_test)
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
    """Optuna objective maximising silhouette score."""
    # Log-uniform sampling for eps because sensitivity is multiplicative.
    eps = trial.suggest_float("eps", 0.05, 2.0, log=True)
    min_samples = trial.suggest_int("min_samples", 2, 20)

    model, _ = train(X_train, eps=eps, min_samples=min_samples)
    metrics = validate(X_val, model, y_val_true)
    return metrics.get("silhouette_score", -1.0)


def run_optuna(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
    n_trials: int = 40,
) -> optuna.Study:
    """Run Optuna study."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="dbscan_pytorch")
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
    """Ray Tune hyperparameter search."""
    def _trainable(config: dict) -> None:
        # Force CPU in Ray workers to avoid GPU contention between
        # parallel trials that would cause out-of-memory errors.
        # WHY: Ray spawns multiple worker processes; if each tried to
        # allocate an N^2 matrix on the same GPU, memory would be exhausted.
        cfg = {k: v for k, v in config.items()}
        cfg["device"] = torch.device("cpu")
        model, _ = train(X_train, **cfg)
        metrics = validate(X_val, model, y_val_true)
        tune.report(
            silhouette_score=metrics.get("silhouette_score", -1.0),
            n_clusters=metrics.get("n_clusters", 0),
            n_noise=metrics.get("n_noise", 0),
        )

    search_space = {
        "eps": tune.loguniform(0.05, 2.0),
        "min_samples": tune.randint(2, 21),
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
# Compare parameter sets
# ---------------------------------------------------------------------------

def compare_parameter_sets() -> None:
    """Compare different DBSCAN hyperparameter configurations on GPU.

    Tests eps = 0.3 vs 0.5 vs 1.0 and min_samples = 3 vs 5 vs 10 to
    demonstrate how each setting affects clustering results.  Includes
    PyTorch-specific notes about GPU memory and device management.
    """
    logger.info("=" * 70)
    logger.info("COMPARE PARAMETER SETS (DBSCAN PyTorch GPU-accelerated)")
    logger.info("=" * 70)

    # Generate a moderate dataset for comparison.
    # WHY: 800 samples keeps GPU memory usage low while providing
    # meaningful cluster structure.
    X, y = generate_data(n_samples=800, random_state=42)

    configs = [
        {
            "eps": 0.3,
            "min_samples": 5,
            "label": "TIGHT (eps=0.3, min_samples=5)",
            "best_for": "High-density data with well-separated clusters",
            "risk": "Over-segmentation: fragments natural clusters, excessive noise",
            "why": "Small eps demands very close points. GPU computes the NxN distance "
                   "matrix in parallel, so the bottleneck is memory not speed.",
        },
        {
            "eps": 0.5,
            "min_samples": 5,
            "label": "MODERATE (eps=0.5, min_samples=5)",
            "best_for": "General-purpose clustering on standardised data",
            "risk": "May merge nearby clusters if their edges overlap within 0.5",
            "why": "Balanced eps captures most natural clusters. GPU acceleration "
                   "makes this configuration run identically fast as tight or loose.",
        },
        {
            "eps": 1.0,
            "min_samples": 5,
            "label": "LOOSE (eps=1.0, min_samples=5)",
            "best_for": "Sparse data or finding few large clusters",
            "risk": "Under-segmentation: distinct clusters merge into one mega-cluster",
            "why": "Large eps treats distant points as neighbours. The adjacency matrix "
                   "becomes mostly True, so GPU memory is used but most entries are 1.",
        },
        {
            "eps": 0.5,
            "min_samples": 3,
            "label": "LOW-DENSITY (eps=0.5, min_samples=3)",
            "best_for": "Small or sparse datasets with few points per cluster",
            "risk": "Noise sensitivity: random fluctuations become spurious clusters",
            "why": "Only 3 neighbours needed for core status. GPU handles the distance "
                   "matrix identically; the change only affects the boolean core mask.",
        },
        {
            "eps": 0.5,
            "min_samples": 10,
            "label": "HIGH-DENSITY (eps=0.5, min_samples=10)",
            "best_for": "Large datasets where only robust dense clusters matter",
            "risk": "Many points labelled as noise, especially in sparse regions",
            "why": "Requiring 10 neighbours filters out thin structures. GPU is "
                   "unaffected by min_samples since it only changes the threshold.",
        },
    ]

    for cfg in configs:
        logger.info("-" * 60)
        logger.info("Config: %s", cfg["label"])
        logger.info("  BEST FOR : %s", cfg["best_for"])
        logger.info("  RISK     : %s", cfg["risk"])
        logger.info("  WHY      : %s", cfg["why"])

        model = DBSCANPyTorch(eps=cfg["eps"], min_samples=cfg["min_samples"])
        labels = model.fit_predict(X)
        metrics = _compute_metrics(X, labels, y)

        logger.info("  Clusters found  : %d", int(metrics["n_clusters"]))
        logger.info("  Noise points    : %d", int(metrics["n_noise"]))
        if "silhouette_score" in metrics:
            logger.info("  Silhouette      : %.4f", metrics["silhouette_score"])
        if "adjusted_rand_index" in metrics:
            logger.info("  ARI             : %.4f", metrics["adjusted_rand_index"])

    logger.info("=" * 70)
    logger.info("Parameter comparison complete.")


# ---------------------------------------------------------------------------
# Real-world demo: geographic delivery location clustering
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate GPU-accelerated DBSCAN on geographic delivery clustering.

    Scenario: A logistics company clusters delivery locations in the NYC
    metro area to optimise delivery routes.  GPU acceleration enables
    processing larger datasets than the NumPy version.

    Synthetic data simulates four dense delivery areas plus scattered noise:
      1. Downtown Manhattan  (~300 deliveries)
      2. Midtown Manhattan   (~200 deliveries)
      3. Brooklyn            (~250 deliveries)
      4. Jersey City         (~150 deliveries)
      5. Random noise        (~50 scattered deliveries)
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Geographic Delivery Clustering (GPU)")
    logger.info("=" * 70)

    rng = np.random.RandomState(42)

    # Generate synthetic delivery locations for four NYC zones plus noise.
    # WHY: realistic lat/lon coordinates with known cluster structure
    # let us evaluate DBSCAN's accuracy against ground truth.
    n_downtown = 300
    downtown = rng.normal(loc=[40.710, -74.000], scale=[0.005, 0.005], size=(n_downtown, 2))

    n_midtown = 200
    midtown = rng.normal(loc=[40.755, -73.985], scale=[0.007, 0.007], size=(n_midtown, 2))

    n_brooklyn = 250
    brooklyn = rng.normal(loc=[40.680, -73.970], scale=[0.010, 0.010], size=(n_brooklyn, 2))

    n_jersey = 150
    jersey = rng.normal(loc=[40.720, -74.045], scale=[0.008, 0.008], size=(n_jersey, 2))

    n_noise = 50
    noise_lat = rng.uniform(40.60, 40.85, size=(n_noise, 1))
    noise_lon = rng.uniform(-74.10, -73.90, size=(n_noise, 1))
    noise_pts = np.hstack([noise_lat, noise_lon])

    X_raw = np.vstack([downtown, midtown, brooklyn, jersey, noise_pts])

    y_true = np.concatenate([
        np.full(n_downtown, 0),
        np.full(n_midtown, 1),
        np.full(n_brooklyn, 2),
        np.full(n_jersey, 3),
        np.full(n_noise, -1),
    ])

    feature_names = ["latitude", "longitude"]
    logger.info("Generated %d delivery locations with %d zones + %d noise. Device: %s",
                len(X_raw), 4, n_noise, DEVICE)
    logger.info("Features: %s", feature_names)

    # Standardise features for consistent eps interpretation.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Search for optimal eps.
    logger.info("Searching for optimal eps via silhouette score...")
    best_eps = 0.3
    best_sil = -1.0

    for eps_candidate in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        model = DBSCANPyTorch(eps=eps_candidate, min_samples=5)
        labels = model.fit_predict(X_scaled)
        n_clusters = model.n_clusters_
        n_noise_found = int((labels == NOISE).sum())

        if n_clusters >= 2 and (labels != -1).sum() >= 2:
            mask = labels != -1
            sil = silhouette_score(X_scaled[mask], labels[mask])
            logger.info("  eps=%.2f -> clusters=%d, noise=%d, silhouette=%.4f",
                        eps_candidate, n_clusters, n_noise_found, sil)
            if sil > best_sil:
                best_sil = sil
                best_eps = eps_candidate
        else:
            logger.info("  eps=%.2f -> clusters=%d, noise=%d (silhouette N/A)",
                        eps_candidate, n_clusters, n_noise_found)

    logger.info("Best eps: %.2f (silhouette=%.4f)", best_eps, best_sil)

    # Fit final model with best eps.
    final_model = DBSCANPyTorch(eps=best_eps, min_samples=5)
    final_labels = final_model.fit_predict(X_scaled)

    # Analyse delivery zones.
    n_zones = final_model.n_clusters_
    logger.info("Discovered %d delivery zones.", n_zones)

    for zone_id in range(n_zones):
        zone_mask = final_labels == zone_id
        zone_points = X_raw[zone_mask]
        centroid_lat = zone_points[:, 0].mean()
        centroid_lon = zone_points[:, 1].mean()
        spread_lat = zone_points[:, 0].std()
        spread_lon = zone_points[:, 1].std()

        logger.info(
            "  Zone %d: %d deliveries | centroid=(%.4f, %.4f) | "
            "spread=(lat=%.4f, lon=%.4f)",
            zone_id, zone_mask.sum(), centroid_lat, centroid_lon,
            spread_lat, spread_lon,
        )

    n_noise_final = int((final_labels == NOISE).sum())
    logger.info("  Noise (unzoned deliveries): %d", n_noise_final)

    # External validation.
    ari = adjusted_rand_score(y_true, final_labels)
    nmi = normalized_mutual_info_score(y_true, final_labels)
    logger.info("External validation: ARI=%.4f, NMI=%.4f", ari, nmi)

    logger.info("=" * 70)
    logger.info("Real-world demo complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline for PyTorch DBSCAN."""
    logger.info("=" * 70)
    logger.info("DBSCAN Clustering - PyTorch Pipeline (device=%s)", DEVICE)
    logger.info("=" * 70)

    # 1. Data
    X, y = generate_data(n_samples=1000)

    # 2. Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # 3. Baseline
    logger.info("-" * 35 + " Baseline " + "-" * 35)
    baseline_model, _ = train(X_train, eps=0.3, min_samples=5)
    validate(X_val, baseline_model, y_val)
    test(X_test, baseline_model, y_test)

    # 4. Optuna
    logger.info("-" * 35 + " Optuna " + "-" * 35)
    study = run_optuna(X_train, X_val, y_train, y_val, n_trials=40)

    # 5. Ray Tune
    logger.info("-" * 35 + " Ray Tune " + "-" * 35)
    ray_results = ray_tune_search(X_train, X_val, y_train, y_val, num_samples=20)

    # 6. Final evaluation
    best_params_optuna = study.best_trial.params
    best_ray = ray_results.get_best_result(metric="silhouette_score", mode="max")
    best_params_ray = {
        k: v for k, v in best_ray.config.items()
        if k in ("eps", "min_samples")
    }

    if study.best_trial.value >= best_ray.metrics["silhouette_score"]:
        best_params = best_params_optuna
        logger.info("Using Optuna best (sil=%.4f).", study.best_trial.value)
    else:
        best_params = best_params_ray
        logger.info("Using Ray Tune best (sil=%.4f).", best_ray.metrics["silhouette_score"])

    logger.info("-" * 35 + " Final " + "-" * 35)
    final_model, _ = train(X_train, **best_params)
    final_metrics = test(X_test, final_model, y_test)
    logger.info("Final test metrics: %s", final_metrics)

    # 7. Parameter comparison and real-world demo
    compare_parameter_sets()
    real_world_demo()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
