"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- NumPy (From Scratch) Implementation
====================================================================

Theory & Mathematics:
    DBSCAN groups together points that are closely packed (high density
    regions) and marks points in low-density regions as outliers (noise).
    It was introduced by Ester et al. (1996, KDD).

    Definitions (given parameters eps and min_samples):
        eps-neighbourhood of point p:
            N_eps(p) = { q in D : dist(p, q) <= eps }

        Core point:   |N_eps(p)| >= min_samples.
        Border point: |N_eps(p)| < min_samples  AND  p in N_eps(q)
                      for some core point q.
        Noise point:  Neither core nor border.  Labelled -1.

    Density-Reachability (directional):
        p is directly density-reachable from q if:
            1) q is a core point, AND
            2) p in N_eps(q).
        p is density-reachable from q if there is a chain
            q = p_1 -> p_2 -> ... -> p_n = p
        where each p_{i+1} is directly density-reachable from p_i.

    Density-Connectivity (symmetric):
        p and q are density-connected if there exists a point o such that
        both p and q are density-reachable from o.

    A cluster is a maximal set of density-connected points.

    Algorithm (from scratch):
        1. Compute the full pairwise distance matrix D (N x N).
        2. For each point i, find its eps-neighbourhood:
               neighbours[i] = { j : D[i, j] <= eps }
        3. Identify core points: { i : |neighbours[i]| >= min_samples }.
        4. For each unvisited core point, expand cluster:
           a. Create new cluster C.
           b. Use a seed queue initialised with the core point's neighbours.
           c. Pop point q from queue:
              - If q is unvisited, mark visited.
              - If q is a core point, add its neighbours to the queue.
              - If q has no cluster, assign it to C.
           d. Repeat until queue is empty.
        5. Any point still without a cluster is NOISE (-1).

    Complexity:
        Time  : O(N^2) for full distance matrix.
        Space : O(N^2) for distance matrix storage.

    This from-scratch implementation computes the full distance matrix
    explicitly with NumPy broadcasting, which is straightforward and fast
    for moderate N but memory-limited for very large datasets.

Business Use Cases:
    - Anomaly / fraud detection (noise = anomalies).
    - Geospatial data clustering (crime hotspots, ride clusters).
    - Natural language processing (topic detection in embedding space).
    - Biological data (gene expression clustering).

Advantages:
    - Does not require K.
    - Finds arbitrarily shaped clusters.
    - Identifies outliers naturally.
    - Deterministic for core and noise points.

Disadvantages:
    - O(N^2) time and space without spatial indexing.
    - Sensitive to eps and min_samples.
    - Poor on datasets with widely varying densities.
    - Border points assigned non-deterministically (order-dependent).

Hyperparameters:
    - eps         : float  - Radius of the neighbourhood.
    - min_samples : int    - Minimum neighbours for core status.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

# logging provides structured, levelled output (INFO / WARNING / ERROR)
# so we can trace every step of the pipeline instead of using bare print().
import logging

# warnings lets us silence FutureWarning noise from scikit-learn / NumPy
# so our log output stays clean and focused on algorithmic messages.
import warnings

# deque is a double-ended queue with O(1) append / popleft -- ideal for
# BFS cluster expansion where we repeatedly pop from the front.
from collections import deque

# Type hints make function signatures self-documenting and enable static
# analysis tools (mypy, pyright) to catch bugs before runtime.
from typing import Any, Dict, Optional, Tuple

# NumPy is the core numerical library; every computation in this
# from-scratch implementation (distance matrix, boolean masks,
# neighbour counting) is done via vectorised NumPy operations.
import numpy as np

# Optuna performs Bayesian hyperparameter optimisation using a Tree of
# Parzen Estimators (TPE) sampler, converging faster than grid or random search.
import optuna

# Ray is a distributed computing framework; we use it here so that
# hyperparameter trials can run in parallel across CPU cores.
import ray

# tune is Ray's hyperparameter search module that manages trial
# scheduling, early stopping, and result aggregation.
from ray import tune

# make_blobs generates isotropic Gaussian clusters useful for testing
# centroid-based algorithms; make_moons generates non-convex crescent shapes
# that specifically challenge algorithms like K-Means while being natural
# for DBSCAN.
from sklearn.datasets import make_blobs, make_moons

# These five metrics assess clustering quality from different angles:
# - adjusted_rand_score: chance-corrected label agreement (external)
# - calinski_harabasz_score: ratio of between- to within-cluster variance
# - davies_bouldin_score: average cluster similarity (lower is better)
# - normalized_mutual_info_score: information-theoretic agreement (external)
# - silhouette_score: how similar a point is to its own cluster vs neighbours
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# train_test_split creates reproducible, stratified data partitions so
# that train/val/test sets all contain representative class proportions.
from sklearn.model_selection import train_test_split

# StandardScaler z-normalises each feature to mean=0, std=1 so that
# the Euclidean distance used by DBSCAN treats all features equally
# rather than being dominated by high-variance features.
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------

# Configure the root logger with a timestamp, severity level, and message
# format so that every log line is traceable to a specific time and severity.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# Create a module-level logger named after this file so that log messages
# are attributable to this specific module in multi-module projects.
logger = logging.getLogger(__name__)

# Suppress FutureWarning messages from downstream libraries (sklearn, NumPy)
# because they clutter output without affecting correctness.
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# Label constants
# ------------------------------------------------------------------

# NOISE = -1 is the standard DBSCAN convention for points that belong
# to no cluster (low-density outliers).
NOISE = -1

# UNCLASSIFIED = -2 distinguishes points that have not yet been visited
# from points already labelled as noise, which is critical during BFS
# expansion to know whether a point needs processing.
UNCLASSIFIED = -2


# ---------------------------------------------------------------------------
# DBSCAN from scratch
# ---------------------------------------------------------------------------

class DBSCANNumpy:
    """Pure-NumPy DBSCAN implementation.

    Parameters
    ----------
    eps : float
        Maximum distance between two samples for them to be considered as
        in the same neighbourhood.
    min_samples : int
        Minimum number of points in a neighbourhood for a point to qualify
        as a core point.
    """

    def __init__(self, eps: float = 0.3, min_samples: int = 5) -> None:
        # eps defines the radius of the neighbourhood ball around each point.
        # WHY: this is the fundamental density parameter -- smaller eps means
        # tighter, denser clusters; larger eps merges distant points together.
        self.eps = eps

        # min_samples is the minimum number of points (including the point
        # itself) within the eps-ball for a point to qualify as a core point.
        # WHY: higher min_samples demands denser regions, producing fewer
        # but more robust clusters and labelling more points as noise.
        self.min_samples = min_samples

        # labels_ will hold the cluster assignment for each point after
        # fitting: non-negative integers for clusters, -1 for noise.
        # WHY: initialised to None so that calling code can check whether
        # the model has been fitted before accessing labels.
        self.labels_: Optional[np.ndarray] = None

        # core_sample_indices_ records the row indices of all core points.
        # WHY: downstream analysis often needs to distinguish core vs border
        # points (e.g. for robust centroid estimation or visualisation).
        self.core_sample_indices_: Optional[np.ndarray] = None

        # n_clusters_ counts the number of clusters discovered (excluding noise).
        # WHY: provides a quick summary of the clustering result without
        # needing to compute np.unique on the labels array.
        self.n_clusters_: int = 0

    # -- helper methods ------------------------------------------------------

    @staticmethod
    def _pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Compute full Euclidean distance matrix (N, N).

        Uses the expansion ||x-y||^2 = ||x||^2 - 2*x.y + ||y||^2.
        """
        # Compute squared norms for every row vector.  keepdims=True gives
        # shape (N, 1) so that broadcasting works in the next step.
        # WHY: pre-computing norms once avoids repeating O(N*d) work for
        # every pair of points.
        XX = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)

        # Expand ||x_i - x_j||^2 = ||x_i||^2 - 2 * x_i . x_j + ||x_j||^2
        # using matrix multiplication X @ X.T for the dot-product term.
        # WHY: this algebraic expansion turns an O(N^2 * d) naive loop into
        # a single BLAS-accelerated matrix multiply plus two broadcasts,
        # which is dramatically faster on modern CPUs.
        D_sq = XX - 2.0 * (X @ X.T) + XX.T          # (N, N)

        # Floating-point rounding can produce tiny negative values on the
        # diagonal; clamp to zero so that sqrt does not produce NaN.
        # WHY: numerical safety is essential when operating on squared
        # differences that should be exactly zero for identical vectors.
        np.maximum(D_sq, 0.0, out=D_sq)              # numerical clamp

        # Take element-wise square root to get Euclidean distances.
        # WHY: DBSCAN is defined in terms of distance, not squared distance,
        # so we need the actual L2 norm for comparison with eps.
        return np.sqrt(D_sq)

    def _region_query(self, dist_matrix: np.ndarray, point_idx: int) -> np.ndarray:
        """Return indices of all points within eps of point_idx."""
        # np.where returns a tuple; [0] extracts the 1-D array of indices
        # where the condition is True (distance <= eps).
        # WHY: this is the fundamental neighbourhood lookup that DBSCAN
        # uses for both core-point identification and cluster expansion.
        return np.where(dist_matrix[point_idx] <= self.eps)[0]

    def _expand_cluster(
        self,
        labels: np.ndarray,
        dist_matrix: np.ndarray,
        point_idx: int,
        neighbours: np.ndarray,
        cluster_id: int,
        is_core: np.ndarray,
    ) -> None:
        """Expand cluster starting from a core point.

        Uses a FIFO queue (BFS) to reach all density-connected points.
        """
        # Assign the seed core point to the current cluster.
        # WHY: the seed is guaranteed to be a core point (checked by the
        # caller) so it belongs to the cluster being formed.
        labels[point_idx] = cluster_id

        # Initialise BFS queue with all neighbours of the seed point.
        # WHY: BFS (breadth-first search) ensures we discover all
        # density-reachable points layer by layer; deque gives O(1) popleft.
        queue: deque = deque(neighbours.tolist())

        # Process queue until empty -- each iteration considers one
        # candidate point for inclusion in the current cluster.
        while queue:
            # Pop the next candidate from the front of the queue (FIFO order).
            # WHY: FIFO ensures breadth-first traversal, which is the
            # standard expansion order for DBSCAN.
            q = queue.popleft()

            # If this point was previously labelled as noise, absorb it
            # into the current cluster as a border point.
            # WHY: a noise point that falls within eps of a core point is
            # density-reachable and therefore belongs to the cluster.
            if labels[q] == NOISE:
                labels[q] = cluster_id

            # If the point was already classified (either as part of this
            # cluster or another), skip further processing.
            # WHY: revisiting already-classified points would cause infinite
            # loops or incorrectly reassign points between clusters.
            if labels[q] != UNCLASSIFIED:
                continue

            # Mark this unclassified point as belonging to the current cluster.
            # WHY: it has been reached from a core point through a chain of
            # density-reachable steps, so it is part of this cluster.
            labels[q] = cluster_id

            # If q is itself a core point, its neighbours may extend the
            # cluster further -- add them to the queue.
            # WHY: only core points propagate the cluster; border points
            # are absorbed but do not extend the frontier.
            if is_core[q]:
                q_neighbours = self._region_query(dist_matrix, q)
                queue.extend(q_neighbours.tolist())

    # -- fit -----------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCANNumpy":
        """Run the DBSCAN algorithm on dataset X.

        Parameters
        ----------
        X : ndarray of shape (N, d)

        Returns
        -------
        self
        """
        # Get the number of data points.
        # WHY: needed to initialise the labels array and iterate over all points.
        n = len(X)

        # Compute the full N x N pairwise Euclidean distance matrix.
        # WHY: DBSCAN needs to know the distance between every pair of
        # points to determine eps-neighbourhoods; this is the O(N^2) step.
        dist_matrix = self._pairwise_distances(X)

        # Count how many points fall within eps of each point (including itself).
        # WHY: a point is a core point if and only if its neighbourhood count
        # meets or exceeds min_samples.
        neighbour_counts = np.sum(dist_matrix <= self.eps, axis=1)  # includes self

        # Boolean mask identifying which points are core points.
        # WHY: core points are the building blocks of clusters -- only core
        # points can propagate cluster membership to their neighbours.
        is_core = neighbour_counts >= self.min_samples

        # Initialise all labels to UNCLASSIFIED (-2).
        # WHY: we need to distinguish between "not yet visited" (UNCLASSIFIED),
        # "visited but not in any cluster" (NOISE), and "assigned to cluster k".
        labels = np.full(n, UNCLASSIFIED, dtype=int)

        # cluster_id is a counter that increments each time a new cluster is formed.
        # WHY: clusters are numbered 0, 1, 2, ... in discovery order.
        cluster_id = 0

        # Iterate over every point in the dataset.
        # WHY: the main DBSCAN loop visits each point exactly once to decide
        # whether it starts a new cluster, joins an existing one, or is noise.
        for i in range(n):
            # Skip points that have already been assigned a label.
            # WHY: once a point has been processed (either classified or
            # marked as noise then absorbed), there is nothing more to do.
            if labels[i] != UNCLASSIFIED:
                continue

            # Find all points within eps of point i.
            # WHY: this is the region query that determines whether point i
            # has enough neighbours to be a core point.
            neighbours = self._region_query(dist_matrix, i)

            # If point i is not a core point, tentatively label it as noise.
            # WHY: non-core, unvisited points are initially noise; they may
            # later be absorbed as border points by a neighbouring cluster.
            if not is_core[i]:
                labels[i] = NOISE
                continue

            # Point i is a core point -- start a new cluster from it.
            # WHY: every unvisited core point seeds a new cluster because
            # it has sufficient local density to anchor a cluster.
            self._expand_cluster(labels, dist_matrix, i, neighbours, cluster_id, is_core)

            # Increment the cluster counter for the next discovered cluster.
            cluster_id += 1

        # Store final cluster labels as a fitted attribute.
        # WHY: downstream code accesses model.labels_ to retrieve results,
        # following the scikit-learn convention for transductive estimators.
        self.labels_ = labels

        # Record indices of all core points for downstream analysis.
        # WHY: knowing which points are core vs border is useful for
        # understanding cluster structure and stability.
        self.core_sample_indices_ = np.where(is_core)[0]

        # Record the total number of clusters discovered.
        # WHY: provides a quick summary without needing to recompute
        # np.unique on the labels array.
        self.n_clusters_ = cluster_id

        # Count noise points for logging.
        # WHY: the noise count is a key diagnostic -- too many noise points
        # suggests eps is too small or min_samples is too large.
        n_noise = int((labels == NOISE).sum())

        # Log a summary of the fitting result.
        logger.info(
            "DBSCANNumpy fitted: eps=%.4f  min_samples=%d  clusters=%d  noise=%d",
            self.eps,
            self.min_samples,
            self.n_clusters_,
            n_noise,
        )
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit and return labels."""
        # Call fit() then return the computed labels array.
        # WHY: many callers want the labels immediately without separately
        # accessing model.labels_, so this one-liner saves a step.
        self.fit(X)
        return self.labels_

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters (sklearn-style interface)."""
        # Return the two DBSCAN hyperparameters as a dictionary.
        # WHY: enables validate/test functions to create a fresh model with
        # the same hyperparameters, which is necessary because DBSCAN is
        # transductive (no predict method for unseen data).
        return {"eps": self.eps, "min_samples": self.min_samples}


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
    # Split the total sample count between moons and blobs.
    # WHY: moons test DBSCAN's ability to handle non-convex shapes,
    # while blobs provide compact Gaussian clusters for comparison.
    n_moons = n_samples // 2
    n_blobs = n_samples - n_moons

    # Generate two interleaving half-circle (moon) clusters with low noise.
    # WHY: make_moons produces non-convex shapes that K-Means fails on
    # but DBSCAN handles naturally, demonstrating DBSCAN's key advantage.
    X_moons, y_moons = make_moons(
        n_samples=n_moons, noise=0.08, random_state=random_state
    )

    # Generate three spherical blob clusters at specified centres.
    # WHY: blobs provide well-separated, convex clusters that serve as
    # an easy baseline; the combination with moons creates a mixed-difficulty
    # dataset that tests both DBSCAN strengths (shape flexibility) and
    # potential weaknesses (varying density).
    X_blobs, y_blobs = make_blobs(
        n_samples=n_blobs,
        centers=[[2.5, 2.5], [-2.0, 3.0], [3.5, -1.0]],
        cluster_std=0.35,
        random_state=random_state,
    )

    # Offset blob labels so they don't collide with moon labels.
    # WHY: make_moons labels are {0, 1} and make_blobs labels are {0, 1, 2};
    # adding max(moon_labels) + 1 gives unique labels {2, 3, 4} for blobs.
    y_blobs += y_moons.max() + 1

    # Stack the two datasets vertically into a single feature matrix.
    # WHY: DBSCAN processes a single dataset, so we need to combine the
    # separate moon and blob arrays into one unified matrix.
    X = np.vstack([X_moons, X_blobs])

    # Concatenate the label arrays to match the combined feature matrix.
    # WHY: ground-truth labels are needed for external validation metrics
    # (ARI, NMI) even though DBSCAN itself does not use them.
    y = np.concatenate([y_moons, y_blobs])

    # Z-normalise features to mean=0, std=1 using StandardScaler.
    # WHY: DBSCAN's eps parameter defines a distance threshold; without
    # standardisation, features with larger scales would dominate the
    # distance computation and make eps interpretation inconsistent.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Log dataset statistics for reproducibility and debugging.
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

def train(X_train: np.ndarray, **hyperparams) -> Tuple[DBSCANNumpy, np.ndarray]:
    """Fit DBSCANNumpy on training data.

    Returns (model, labels).
    """
    # Set default hyperparameters, then override with any caller-provided values.
    # WHY: this pattern lets us define sensible defaults while still allowing
    # Optuna or manual callers to override specific parameters.
    defaults: Dict[str, Any] = {"eps": 0.3, "min_samples": 5}
    defaults.update(hyperparams)

    # Create a fresh DBSCANNumpy instance with the merged hyperparameters.
    # WHY: DBSCAN is transductive (labels are computed during fit, not predict),
    # so we always create a new model rather than reusing a fitted one.
    model = DBSCANNumpy(**defaults)

    # Fit the model and get cluster labels in one call.
    # WHY: fit_predict is more convenient than calling fit() then accessing
    # model.labels_ separately.
    labels = model.fit_predict(X_train)
    return model, labels


def _compute_metrics(
    X: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Internal and external clustering metrics."""
    # Count unique labels; subtract 1 if noise (-1) is present because
    # noise is not a real cluster.
    # WHY: the number of clusters is a key summary statistic that tells us
    # whether the algorithm is over- or under-segmenting the data.
    unique = set(labels_pred)
    n_clusters = len(unique) - (1 if -1 in unique else 0)

    # Count non-noise points for metric computation.
    # WHY: internal metrics (silhouette, CH, DB) are undefined for noise
    # points, so we need to know how many valid points remain.
    n_non_noise = int((labels_pred != -1).sum())

    # Initialise metrics dictionary with cluster and noise counts.
    metrics: Dict[str, float] = {
        "n_clusters": float(n_clusters),
        "n_noise": float((labels_pred == -1).sum()),
    }

    # Compute internal metrics only if we have at least 2 clusters and
    # at least 2 non-noise points.
    # WHY: silhouette_score and other internal metrics require at least
    # 2 clusters; computing them with fewer would raise an error.
    if n_clusters >= 2 and n_non_noise >= 2:
        # Create a boolean mask to exclude noise points from metrics.
        # WHY: noise points have no cluster assignment, so including them
        # would distort cluster-quality measurements.
        mask = labels_pred != -1

        # Silhouette score measures how well each point fits its cluster
        # vs the nearest neighbouring cluster; ranges from -1 to +1.
        # WHY: the primary internal metric for DBSCAN; higher values
        # indicate better-defined, more separated clusters.
        metrics["silhouette_score"] = float(
            silhouette_score(X[mask], labels_pred[mask])
        )

        # Calinski-Harabasz index: ratio of between-cluster to within-cluster
        # dispersion; higher is better.
        # WHY: provides a complementary perspective to silhouette, especially
        # useful for spherical clusters.
        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(X[mask], labels_pred[mask])
        )

        # Davies-Bouldin index: average similarity ratio between each cluster
        # and its most similar cluster; lower is better.
        # WHY: penalises clusters that are close together or spread out,
        # helping detect under-segmentation.
        metrics["davies_bouldin_score"] = float(
            davies_bouldin_score(X[mask], labels_pred[mask])
        )
    else:
        # Warn when internal metrics cannot be computed.
        # WHY: this usually means the hyperparameters are badly chosen
        # (e.g. eps too large making everything one cluster, or too small
        # making everything noise).
        logger.warning(
            "Clusters=%d, non-noise=%d; internal metrics skipped.",
            n_clusters,
            n_non_noise,
        )

    # Compute external metrics if ground-truth labels are available.
    # WHY: external metrics (ARI, NMI) measure agreement between predicted
    # and true labels, which is the gold standard for evaluation.
    if labels_true is not None:
        # ARI (Adjusted Rand Index): chance-corrected version of the Rand
        # index; ranges from -1 (worse than random) to 1 (perfect match).
        # WHY: robust to different numbers of clusters and cluster sizes.
        metrics["adjusted_rand_index"] = float(
            adjusted_rand_score(labels_true, labels_pred)
        )

        # NMI (Normalised Mutual Information): information-theoretic measure
        # of how much information the predicted labels share with the true
        # labels; ranges from 0 (no mutual info) to 1 (perfect agreement).
        # WHY: complements ARI with an entropy-based perspective.
        metrics["normalised_mutual_info"] = float(
            normalized_mutual_info_score(labels_true, labels_pred)
        )

    return metrics


def validate(
    X_val: np.ndarray,
    model: DBSCANNumpy,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Validate: re-fit DBSCAN with same params on validation data."""
    # Extract the hyperparameters from the trained model.
    # WHY: DBSCAN is transductive (no predict() for new data), so we must
    # re-fit a new model with the same eps and min_samples on validation data.
    params = model.get_params()

    # Create a fresh model with the same hyperparameters and fit on val data.
    # WHY: this simulates how the model would perform on unseen data while
    # respecting DBSCAN's transductive nature.
    val_model = DBSCANNumpy(**params)
    labels_pred = val_model.fit_predict(X_val)

    # Compute and return all applicable metrics.
    metrics = _compute_metrics(X_val, labels_pred, labels_true)
    logger.info("Validation metrics: %s", metrics)
    return metrics


def test(
    X_test: np.ndarray,
    model: DBSCANNumpy,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate on holdout test set (re-fits with same hyperparameters)."""
    # Same re-fit pattern as validate() but on the test partition.
    # WHY: provides the final, unbiased performance estimate using
    # hyperparameters selected during validation.
    params = model.get_params()
    test_model = DBSCANNumpy(**params)
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
    # Sample eps on a log scale because the sensitivity of DBSCAN to eps
    # is multiplicative rather than additive.
    # WHY: log-uniform sampling gives equal probability to each order of
    # magnitude (e.g. 0.05-0.5 gets as many trials as 0.5-2.0).
    eps = trial.suggest_float("eps", 0.05, 2.0, log=True)

    # Sample min_samples as an integer in [2, 20].
    # WHY: min_samples controls the minimum density threshold; values
    # below 2 are degenerate, and above 20 is overly strict for most
    # datasets of moderate size.
    min_samples = trial.suggest_int("min_samples", 2, 20)

    # Train the model with the sampled hyperparameters.
    model, _ = train(X_train, eps=eps, min_samples=min_samples)

    # Evaluate on validation data and return the silhouette score.
    # WHY: silhouette is a robust internal metric that does not require
    # ground-truth labels, making it suitable for unsupervised tuning.
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
    # Reduce Optuna's own logging to WARNING to avoid cluttering output
    # with per-trial INFO messages.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create a study that maximises the objective (silhouette score).
    # WHY: "maximize" because higher silhouette means better-defined clusters.
    study = optuna.create_study(direction="maximize", study_name="dbscan_numpy")

    # Run the optimisation for n_trials iterations.
    # WHY: 40 trials is enough for TPE to converge on a 2-parameter space
    # (eps, min_samples) while keeping runtime reasonable for O(N^2) DBSCAN.
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, y_train_true, y_val_true),
        n_trials=n_trials,
    )

    # Log the best result found.
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
        # Train and validate with the sampled hyperparameters.
        # WHY: Ray calls this function once per trial with a different
        # config dict each time.
        model, _ = train(X_train, **config)
        metrics = validate(X_val, model, y_val_true)

        # Report metrics back to Ray Tune for tracking and comparison.
        tune.report(
            silhouette_score=metrics.get("silhouette_score", -1.0),
            n_clusters=metrics.get("n_clusters", 0),
            n_noise=metrics.get("n_noise", 0),
        )

    # Define the search space -- same ranges as Optuna for consistency.
    search_space = {
        "eps": tune.loguniform(0.05, 2.0),
        "min_samples": tune.randint(2, 21),
    }

    # Initialise Ray if not already running; ignore_reinit_error=True
    # makes this safe to call multiple times.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Create a Tuner that maximises silhouette score over num_samples trials.
    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="silhouette_score",
            mode="max",
            num_samples=num_samples,
        ),
    )

    # Run all trials.
    results = tuner.fit()

    # Extract and log the best result.
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
    """Compare different DBSCAN hyperparameter configurations.

    Tests eps = 0.3 vs 0.5 vs 1.0 and min_samples = 3 vs 5 vs 10 to
    show how each setting affects the number of clusters, noise points,
    and silhouette score.  Each configuration is annotated with its
    BEST-FOR scenario, RISK, and WHY.
    """
    logger.info("=" * 70)
    logger.info("COMPARE PARAMETER SETS (DBSCAN NumPy from scratch)")
    logger.info("=" * 70)

    # Generate a moderate-size dataset for comparison.
    # WHY: 800 samples keeps O(N^2) distance computation fast while
    # providing enough data for meaningful cluster structure.
    X, y = generate_data(n_samples=800, random_state=42)

    # ---- Configuration definitions ----------------------------------------
    # Each configuration is a dict with eps, min_samples, a human-readable
    # label, and detailed annotations explaining the parameter choice.
    configs = [
        {
            "eps": 0.3,
            "min_samples": 5,
            "label": "TIGHT (eps=0.3, min_samples=5)",
            "best_for": "High-density data with well-separated clusters",
            "risk": "Over-segmentation: may split natural clusters into fragments "
                    "and produce excessive noise",
            "why": "Small eps demands that points be very close to form a cluster. "
                   "This finds only the densest cores and labels everything else noise.",
        },
        {
            "eps": 0.5,
            "min_samples": 5,
            "label": "MODERATE (eps=0.5, min_samples=5)",
            "best_for": "General-purpose clustering on standardised data",
            "risk": "May merge nearby clusters if their edges are within 0.5 units",
            "why": "A balanced eps that captures most natural clusters in z-normalised "
                   "data without being too aggressive or too conservative.",
        },
        {
            "eps": 1.0,
            "min_samples": 5,
            "label": "LOOSE (eps=1.0, min_samples=5)",
            "best_for": "Sparse data or when you want few, large clusters",
            "risk": "Under-segmentation: distinct clusters merge into one mega-cluster "
                    "because the neighbourhood radius is too generous",
            "why": "Large eps treats distant points as neighbours, absorbing most data "
                   "into a single cluster. Useful only when clusters are very spread out.",
        },
        {
            "eps": 0.5,
            "min_samples": 3,
            "label": "LOW-DENSITY (eps=0.5, min_samples=3)",
            "best_for": "Small or sparse datasets where clusters have few points",
            "risk": "Noise sensitivity: even random fluctuations may be labelled as clusters",
            "why": "Lowering min_samples to 3 means only 3 nearby points are needed to "
                   "form a core, so even thin filaments become clusters.",
        },
        {
            "eps": 0.5,
            "min_samples": 10,
            "label": "HIGH-DENSITY (eps=0.5, min_samples=10)",
            "best_for": "Large datasets where you want only robust, dense clusters",
            "risk": "May label too many points as noise, especially in sparser regions",
            "why": "Requiring 10 neighbours within eps ensures only genuinely dense "
                   "regions form clusters, filtering out thin or small structures.",
        },
    ]

    # ---- Run each configuration -------------------------------------------
    for cfg in configs:
        logger.info("-" * 60)
        logger.info("Config: %s", cfg["label"])
        logger.info("  BEST FOR : %s", cfg["best_for"])
        logger.info("  RISK     : %s", cfg["risk"])
        logger.info("  WHY      : %s", cfg["why"])

        # Create and fit a fresh model with this configuration.
        model = DBSCANNumpy(eps=cfg["eps"], min_samples=cfg["min_samples"])
        labels = model.fit_predict(X)

        # Compute metrics for this configuration.
        metrics = _compute_metrics(X, labels, y)

        # Log key results.
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
    """Demonstrate DBSCAN on a realistic geographic delivery clustering problem.

    Scenario: A logistics company wants to group delivery locations in the
    NYC metro area into zones for route optimisation.  Locations are defined
    by (latitude, longitude).  DBSCAN is ideal here because:
      - Delivery zones have irregular shapes (following roads, coastlines).
      - Outlier deliveries (remote suburbs) should be flagged, not forced
        into a zone.
      - The number of zones is not known in advance.

    Synthetic data simulates four dense delivery areas plus scattered noise:
      1. Downtown Manhattan  (~300 deliveries)
      2. Midtown Manhattan   (~200 deliveries)
      3. Brooklyn            (~250 deliveries)
      4. Jersey City         (~150 deliveries)
      5. Random noise        (~50 scattered deliveries)
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Geographic Delivery Location Clustering")
    logger.info("=" * 70)

    # Fix random seed for reproducibility.
    rng = np.random.RandomState(42)

    # --- Generate synthetic delivery locations ---

    # Downtown Manhattan: tight cluster near (40.710, -74.000).
    # WHY: represents a high-density commercial zone with many deliveries
    # in a small area -- exactly what DBSCAN should identify as a core cluster.
    n_downtown = 300
    downtown = rng.normal(loc=[40.710, -74.000], scale=[0.005, 0.005], size=(n_downtown, 2))

    # Midtown Manhattan: slightly looser cluster near (40.755, -73.985).
    # WHY: another distinct zone separated from downtown by ~5 km; tests
    # whether DBSCAN correctly separates two nearby but distinct clusters.
    n_midtown = 200
    midtown = rng.normal(loc=[40.755, -73.985], scale=[0.007, 0.007], size=(n_midtown, 2))

    # Brooklyn: broader cluster near (40.680, -73.970).
    # WHY: a more spread-out residential area with lower delivery density;
    # tests DBSCAN's ability to handle clusters with different densities.
    n_brooklyn = 250
    brooklyn = rng.normal(loc=[40.680, -73.970], scale=[0.010, 0.010], size=(n_brooklyn, 2))

    # Jersey City: across the Hudson, near (40.720, -74.045).
    # WHY: geographically separated from Manhattan by a river, so DBSCAN
    # should clearly separate this from the Manhattan clusters.
    n_jersey = 150
    jersey = rng.normal(loc=[40.720, -74.045], scale=[0.008, 0.008], size=(n_jersey, 2))

    # Noise: random scattered deliveries across the metro area.
    # WHY: simulates one-off deliveries to suburban or unusual locations;
    # DBSCAN should label these as noise rather than forcing them into zones.
    n_noise = 50
    noise_lat = rng.uniform(40.60, 40.85, size=(n_noise, 1))
    noise_lon = rng.uniform(-74.10, -73.90, size=(n_noise, 1))
    noise_pts = np.hstack([noise_lat, noise_lon])

    # Combine all delivery locations into a single array.
    X_raw = np.vstack([downtown, midtown, brooklyn, jersey, noise_pts])

    # Create ground-truth labels for evaluation.
    y_true = np.concatenate([
        np.full(n_downtown, 0),   # Downtown = zone 0
        np.full(n_midtown, 1),    # Midtown = zone 1
        np.full(n_brooklyn, 2),   # Brooklyn = zone 2
        np.full(n_jersey, 3),     # Jersey City = zone 3
        np.full(n_noise, -1),     # Noise = -1
    ])

    # Feature names for interpretability.
    feature_names = ["latitude", "longitude"]

    logger.info("Generated %d delivery locations with %d true zones + %d noise points.",
                len(X_raw), 4, n_noise)
    logger.info("Features: %s", feature_names)

    # --- Standardise features ---
    # WHY: lat and lon have similar scales (~40 and ~-74 respectively), but
    # StandardScaler ensures eps has a consistent meaning across features
    # and removes the mean offset.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # --- Find optimal eps using silhouette search ---
    logger.info("Searching for optimal eps via silhouette score...")

    best_eps = 0.3
    best_sil = -1.0

    # Test a range of eps values to find the one that maximises silhouette.
    # WHY: eps is the most sensitive DBSCAN parameter; a systematic search
    # over a reasonable range is more reliable than guessing.
    for eps_candidate in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        model = DBSCANNumpy(eps=eps_candidate, min_samples=5)
        labels = model.fit_predict(X_scaled)
        n_clusters = model.n_clusters_
        n_noise_found = int((labels == NOISE).sum())

        # Only compute silhouette if we have at least 2 clusters and
        # enough non-noise points.
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

    # --- Fit final model with best eps ---
    final_model = DBSCANNumpy(eps=best_eps, min_samples=5)
    final_labels = final_model.fit_predict(X_scaled)

    # --- Analyse delivery zones ---
    n_zones = final_model.n_clusters_
    logger.info("Discovered %d delivery zones.", n_zones)

    for zone_id in range(n_zones):
        # Get all points belonging to this zone.
        zone_mask = final_labels == zone_id
        zone_points = X_raw[zone_mask]  # use original lat/lon for interpretation

        # Compute zone centroid in original coordinates.
        centroid_lat = zone_points[:, 0].mean()
        centroid_lon = zone_points[:, 1].mean()

        # Compute zone spread (std dev in each coordinate).
        spread_lat = zone_points[:, 0].std()
        spread_lon = zone_points[:, 1].std()

        logger.info(
            "  Zone %d: %d deliveries | centroid=(%.4f, %.4f) | "
            "spread=(lat=%.4f, lon=%.4f)",
            zone_id,
            zone_mask.sum(),
            centroid_lat,
            centroid_lon,
            spread_lat,
            spread_lon,
        )

    # Report noise (unassigned) deliveries.
    n_noise_final = int((final_labels == NOISE).sum())
    logger.info("  Noise (unzoned deliveries): %d", n_noise_final)

    # --- External validation metrics ---
    # Compute ARI and NMI against ground truth.
    ari = adjusted_rand_score(y_true, final_labels)
    nmi = normalized_mutual_info_score(y_true, final_labels)
    logger.info("External validation: ARI=%.4f, NMI=%.4f", ari, nmi)

    logger.info("=" * 70)
    logger.info("Real-world demo complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline for from-scratch DBSCAN."""
    logger.info("=" * 70)
    logger.info("DBSCAN Clustering - NumPy (From Scratch) Pipeline")
    logger.info("=" * 70)

    # 1. Data -- generate mixed moons + blobs dataset.
    # WHY: smaller sample size (1000) because our from-scratch implementation
    # computes a full O(N^2) distance matrix, which is memory-intensive.
    X, y = generate_data(n_samples=1000)

    # 2. Split into train (60%), val (20%), test (20%).
    # WHY: separate partitions prevent data leakage and allow us to tune
    # hyperparameters on validation while reserving test for final evaluation.
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

    # 3. Baseline model with default hyperparameters.
    logger.info("-" * 35 + " Baseline " + "-" * 35)
    baseline_model, _ = train(X_train, eps=0.3, min_samples=5)
    validate(X_val, baseline_model, y_val)
    test(X_test, baseline_model, y_test)

    # 4. Optuna hyperparameter optimisation.
    logger.info("-" * 35 + " Optuna " + "-" * 35)
    study = run_optuna(X_train, X_val, y_train, y_val, n_trials=40)

    # 5. Ray Tune distributed hyperparameter search.
    logger.info("-" * 35 + " Ray Tune " + "-" * 35)
    ray_results = ray_tune_search(X_train, X_val, y_train, y_val, num_samples=20)

    # 6. Final evaluation with the best hyperparameters.
    best_params_optuna = study.best_trial.params
    best_ray = ray_results.get_best_result(metric="silhouette_score", mode="max")
    best_params_ray = {
        k: v for k, v in best_ray.config.items()
        if k in ("eps", "min_samples")
    }

    # Pick the winner between Optuna and Ray Tune based on silhouette score.
    if study.best_trial.value >= best_ray.metrics["silhouette_score"]:
        best_params = best_params_optuna
        logger.info("Optuna wins (sil=%.4f).", study.best_trial.value)
    else:
        best_params = best_params_ray
        logger.info("Ray Tune wins (sil=%.4f).", best_ray.metrics["silhouette_score"])

    # Train and evaluate the final model with the winning hyperparameters.
    logger.info("-" * 35 + " Final " + "-" * 35)
    final_model, _ = train(X_train, **best_params)
    final_metrics = test(X_test, final_model, y_test)
    logger.info("Final test metrics: %s", final_metrics)

    # 7. Run the parameter comparison and real-world demo.
    compare_parameter_sets()
    real_world_demo()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
