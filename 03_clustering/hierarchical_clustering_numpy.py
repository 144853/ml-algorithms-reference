"""
Hierarchical Clustering - NumPy From-Scratch Implementation
============================================================

COMPLETE ML TUTORIAL: This file implements agglomerative hierarchical
clustering from scratch using NumPy. The algorithm builds a dendrogram by
iteratively merging the two closest clusters, with support for four linkage
methods: single, complete, average, and Ward's.

Theory & Mathematics:
    Agglomerative hierarchical clustering is a bottom-up (agglomerative)
    approach that starts with N singleton clusters and merges them iteratively.

    Algorithm:
        1. Initialise: each data point is its own cluster.
        2. Compute the pairwise distance matrix D (N x N).
        3. Find the two closest clusters (i, j) = argmin D[i, j].
        4. Merge clusters i and j into a new cluster.
        5. Update the distance matrix using the chosen linkage criterion.
        6. Repeat steps 3-5 until the desired number of clusters is reached
           (or a single cluster remains).

    Linkage Methods:

    Single Linkage: d(A, B) = min_{a in A, b in B} d(a, b)
        - Nearest-neighbour distance between clusters.
        - Lance-Williams update: d(AB, C) = min(d(A,C), d(B,C))
        - Produces elongated, chain-like clusters.
        - Time: O(n^2) with minimum spanning tree approach.

    Complete Linkage: d(A, B) = max_{a in A, b in B} d(a, b)
        - Farthest-neighbour distance between clusters.
        - Lance-Williams update: d(AB, C) = max(d(A,C), d(B,C))
        - Produces compact, spherical clusters.

    Average Linkage (UPGMA): d(A, B) = (1/(|A|*|B|)) * sum d(a,b)
        - Mean pairwise distance between all cross-cluster pairs.
        - Lance-Williams update: d(AB, C) = (|A|*d(A,C) + |B|*d(B,C)) / (|A|+|B|)
        - Compromise between single and complete.

    Ward's Method: Minimises within-cluster variance increase.
        - d(A, B) = sqrt((2*|A|*|B|)/(|A|+|B|)) * ||c_A - c_B||
        - Lance-Williams: d(AB, C) = sqrt(
            ((|A|+|C|)*d(A,C)^2 + (|B|+|C|)*d(B,C)^2 - |C|*d(A,B)^2) / (|A|+|B|+|C|)
          )
        - Produces balanced, compact clusters.
        - Requires Euclidean distance.

    Linkage Matrix (for dendrogram):
        Each merge is recorded as a row [cluster_i, cluster_j, distance, size].
        This (N-1 x 4) matrix encodes the full dendrogram.

    Complexity:
        Time: O(n^3) for the naive implementation (n^2 distance matrix updates).
        Space: O(n^2) for the distance matrix.

Hyperparameters:
    - n_clusters (int): Target number of clusters.
    - linkage (str): 'single', 'complete', 'average', 'ward'.

Business Use Cases:
    - Customer segmentation by purchase behaviour.
    - Document topic hierarchy construction.
    - Biological taxonomy / gene clustering.

Advantages:
    - Full dendrogram: choose any k by cutting at different heights.
    - Deterministic: no randomness, always same result.
    - Multiple linkage methods for different cluster shapes.

Disadvantages:
    - O(n^3) time complexity: very slow for n > 10,000.
    - Greedy: cannot undo a bad merge.
    - Single linkage: prone to chaining artefacts.
"""

# -- Standard library imports --
import logging  # Structured logging
import time  # Timing
import warnings  # Warning suppression
from functools import partial  # Argument binding

# -- Third-party imports --
import numpy as np  # Core numerical library
import optuna  # Bayesian HPO
import ray  # Distributed computing
from ray import tune  # Ray's tuning module
from sklearn.datasets import make_blobs  # Synthetic blob data
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Hierarchical Clustering From Scratch
# ---------------------------------------------------------------------------

class HierarchicalClusteringNumpy:
    """Agglomerative hierarchical clustering from scratch in NumPy.

    Implements single, complete, average, and Ward's linkage methods.
    Builds a full dendrogram (linkage matrix) and allows cutting at any k.
    """

    def __init__(self, n_clusters=3, linkage="ward"):
        """Initialise hierarchical clustering.

        Args:
            n_clusters: Target number of clusters when cutting the dendrogram.
            linkage: Linkage criterion ('single', 'complete', 'average', 'ward').
        """
        self.n_clusters = n_clusters  # Target cluster count
        self.linkage = linkage  # Linkage method

        # Results (populated during fit)
        self.labels_ = None  # Cluster labels for each data point
        self.linkage_matrix_ = None  # (N-1 x 4) dendrogram encoding
        self.n_samples_ = 0  # Number of data points

    def _compute_distance_matrix(self, X):
        """Compute the full pairwise Euclidean distance matrix.

        Uses vectorised computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

        Args:
            X: Data matrix, shape (n, d).

        Returns:
            Distance matrix, shape (n, n). D[i,j] = ||x_i - x_j||.
        """
        # Compute squared distances using the expansion formula
        sq_norms = np.sum(X ** 2, axis=1)  # ||x_i||^2 for each point
        sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)  # Squared distances
        sq_dist = np.maximum(sq_dist, 0.0)  # Clamp negatives from floating-point errors
        dist = np.sqrt(sq_dist)  # Euclidean distances (not squared)
        return dist  # Shape: (n, n)

    def _update_distance(self, D, sizes, i, j, method):
        """Update the distance matrix after merging clusters i and j.

        Uses the Lance-Williams formula appropriate for each linkage method.

        Args:
            D: Current distance matrix (modified in-place conceptually).
            sizes: Array of cluster sizes.
            i: Index of first cluster being merged.
            j: Index of second cluster being merged.
            method: Linkage method string.

        Returns:
            Array of new distances from the merged cluster to all others.
        """
        n = D.shape[0]  # Current number of active clusters
        ni = sizes[i]  # Size of cluster i
        nj = sizes[j]  # Size of cluster j
        new_dists = np.zeros(n)  # Distances from merged cluster to each other

        for k in range(n):  # For each other cluster
            if k == i or k == j:  # Skip the two being merged
                continue

            nk = sizes[k]  # Size of cluster k

            if method == "single":
                # Single linkage: minimum of the two distances
                new_dists[k] = min(D[i, k], D[j, k])

            elif method == "complete":
                # Complete linkage: maximum of the two distances
                new_dists[k] = max(D[i, k], D[j, k])

            elif method == "average":
                # Average (UPGMA): weighted average by cluster sizes
                new_dists[k] = (ni * D[i, k] + nj * D[j, k]) / (ni + nj)

            elif method == "ward":
                # Ward's method: Lance-Williams formula for Ward
                # Minimises the increase in total within-cluster variance
                n_total = ni + nj + nk  # Combined size
                new_dists[k] = np.sqrt(
                    ((ni + nk) * D[i, k] ** 2
                     + (nj + nk) * D[j, k] ** 2
                     - nk * D[i, j] ** 2) / n_total
                )

        return new_dists  # Distances from merged cluster (i+j) to all others

    def fit(self, X):
        """Fit hierarchical clustering by building the full dendrogram.

        Algorithm:
            1. Compute pairwise distance matrix.
            2. Find the closest pair of clusters.
            3. Merge them, record in linkage matrix.
            4. Update distance matrix using Lance-Williams formula.
            5. Repeat until one cluster remains.
            6. Cut dendrogram at n_clusters to assign labels.

        Args:
            X: Data matrix, shape (n, d).

        Returns:
            self: Fitted instance with labels_ and linkage_matrix_.
        """
        n = X.shape[0]  # Number of data points
        self.n_samples_ = n  # Store for later use

        # Step 1: Compute pairwise distance matrix
        D = self._compute_distance_matrix(X)  # Shape: (n, n)

        # Track cluster membership and sizes
        # active[i] = True if cluster i has not been merged into another
        active = np.ones(n, dtype=bool)  # All clusters start as active

        # sizes[i] = number of points in cluster i
        sizes = np.ones(n, dtype=int)  # Each cluster starts with 1 point

        # cluster_members[i] = list of original point indices in cluster i
        cluster_members = {i: [i] for i in range(n)}  # Singleton clusters

        # Linkage matrix: each row records [cluster_a, cluster_b, distance, new_size]
        linkage_matrix = np.zeros((n - 1, 4))  # N-1 merges to reach 1 cluster

        # Next available cluster ID for merged clusters
        next_cluster_id = n  # New clusters get IDs n, n+1, n+2, ...

        # Map from current index to cluster ID (for linkage matrix)
        cluster_ids = np.arange(n)  # Initially, cluster i has ID i

        # Step 2-5: Iteratively merge closest pairs
        for step in range(n - 1):  # N-1 merges total
            # Set inactive clusters' distances to infinity
            D_masked = D.copy()  # Work on a copy to preserve original
            for idx in range(n):
                if not active[idx]:
                    D_masked[idx, :] = np.inf
                    D_masked[:, idx] = np.inf

            # Set diagonal to infinity to prevent self-merges
            np.fill_diagonal(D_masked, np.inf)

            # Find the closest pair of active clusters
            min_idx = np.argmin(D_masked)  # Flat index of minimum
            i, j = np.unravel_index(min_idx, D_masked.shape)  # Convert to 2D indices
            merge_dist = D_masked[i, j]  # Distance at which they merge

            # Record the merge in the linkage matrix
            # Convention: smaller cluster ID first
            id_i = cluster_ids[i]  # Cluster ID for cluster i
            id_j = cluster_ids[j]  # Cluster ID for cluster j
            new_size = sizes[i] + sizes[j]  # Size of merged cluster
            linkage_matrix[step] = [
                min(id_i, id_j),  # First cluster (smaller ID)
                max(id_i, id_j),  # Second cluster (larger ID)
                merge_dist,  # Distance at merge
                new_size,  # Combined size
            ]

            # Update distances from merged cluster to all others
            new_dists = self._update_distance(D, sizes, i, j, self.linkage)

            # Merge j into i (keep i, deactivate j)
            D[i, :] = new_dists  # Update row i with new distances
            D[:, i] = new_dists  # Update column i (symmetric matrix)
            D[i, i] = 0.0  # Self-distance is zero

            # Update cluster metadata
            sizes[i] = new_size  # Update size
            cluster_ids[i] = next_cluster_id  # Assign new cluster ID
            cluster_members[next_cluster_id] = (
                cluster_members[cluster_ids[i] if step == 0 else cluster_ids[i]]
                if next_cluster_id not in cluster_members
                else cluster_members[next_cluster_id]
            )
            # Merge member lists
            old_members_i = []
            old_members_j = []
            for cid, members in cluster_members.items():
                if cid == id_i:
                    old_members_i = members
                if cid == id_j:
                    old_members_j = members
            cluster_members[next_cluster_id] = old_members_i + old_members_j

            # Deactivate cluster j
            active[j] = False  # Cluster j is no longer active

            next_cluster_id += 1  # Increment cluster ID counter

        # Store the linkage matrix
        self.linkage_matrix_ = linkage_matrix

        # Step 6: Cut dendrogram at n_clusters
        self.labels_ = self._cut_dendrogram(linkage_matrix, n, self.n_clusters)

        logger.info("Hierarchical(numpy) fitted  n_clusters=%d  linkage=%s",
                     self.n_clusters, self.linkage)
        return self

    def _cut_dendrogram(self, Z, n, k):
        """Cut the dendrogram to obtain k clusters.

        Walk through the linkage matrix in reverse: the last (n-k) merges
        determine which points belong to which of the k clusters.

        Args:
            Z: Linkage matrix, shape (n-1, 4).
            n: Number of original data points.
            k: Desired number of clusters.

        Returns:
            Array of cluster labels, shape (n,).
        """
        # Start with each point in its own cluster
        # The last (k-1) merges in Z should NOT be performed to get k clusters
        # So we perform the first (n-k) merges

        # Use union-find to track cluster membership
        parent = list(range(2 * n - 1))  # Parent array for union-find

        def find(x):
            """Find root of x with path compression."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path compression
                x = parent[x]
            return x

        # Perform the first (n - k) merges
        for step in range(n - k):
            c1 = int(Z[step, 0])  # First cluster in this merge
            c2 = int(Z[step, 1])  # Second cluster in this merge
            new_id = n + step  # New cluster ID for this merge

            # Union: set parent of c1 and c2 to the new cluster
            parent[find(c1)] = new_id
            parent[find(c2)] = new_id

        # Assign cluster labels: find the root for each original point
        roots = [find(i) for i in range(n)]  # Root cluster for each point

        # Map unique roots to consecutive labels 0, 1, ..., k-1
        unique_roots = list(set(roots))
        root_to_label = {root: label for label, root in enumerate(unique_roots)}
        labels = np.array([root_to_label[r] for r in roots])

        return labels  # Cluster labels for each data point

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=400, n_features=5, n_clusters=4, random_state=42):
    """Generate synthetic blob data for clustering.

    WHY n_samples=400: O(n^3) from-scratch implementation is slow for large n.
    """
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features,
        centers=n_clusters, cluster_std=1.0,
        random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Fit hierarchical clustering from scratch."""
    model = HierarchicalClusteringNumpy(
        n_clusters=hp.get("n_clusters", 3),
        linkage=hp.get("linkage", "ward"),
    )
    model.fit(X_train)
    return model


def _clustering_metrics(X, labels_pred, labels_true=None):
    """Compute clustering quality metrics."""
    n_unique = len(np.unique(labels_pred))
    metrics = {}

    if 2 <= n_unique <= len(X) - 1:
        metrics["silhouette"] = silhouette_score(X, labels_pred)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels_pred)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels_pred)
    else:
        metrics["silhouette"] = 0.0
        metrics["calinski_harabasz"] = 0.0
        metrics["davies_bouldin"] = 999.0

    if labels_true is not None:
        metrics["adjusted_rand"] = adjusted_rand_score(labels_true, labels_pred)
        metrics["nmi"] = normalized_mutual_info_score(labels_true, labels_pred)
    else:
        metrics["adjusted_rand"] = 0.0
        metrics["nmi"] = 0.0

    return metrics


def validate(model, X_val, y_val):
    """Evaluate clustering by re-fitting on validation data."""
    val_model = HierarchicalClusteringNumpy(
        n_clusters=model.n_clusters, linkage=model.linkage,
    )
    labels = val_model.fit_predict(X_val)
    m = _clustering_metrics(X_val, labels, y_val)
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    """Evaluate clustering on test data."""
    test_model = HierarchicalClusteringNumpy(
        n_clusters=model.n_clusters, linkage=model.linkage,
    )
    labels = test_model.fit_predict(X_test)
    m = _clustering_metrics(X_test, labels, y_test)
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare linkage methods across different cluster counts."""
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Hierarchical Clustering (NumPy from scratch)")
    print("=" * 80)
    print("\nCompares linkage methods: single, complete, average, ward.\n")

    linkages = ["single", "complete", "average", "ward"]
    results = {}

    for link in linkages:
        name = f"linkage={link}"

        start_time = time.time()
        model = HierarchicalClusteringNumpy(n_clusters=4, linkage=link)
        labels = model.fit_predict(X_train)
        train_time = time.time() - start_time

        metrics = _clustering_metrics(X_train, labels, y_train)
        metrics["train_time"] = train_time
        results[name] = metrics

        print(f"  {name:<25} Sil={metrics['silhouette']:.4f}  "
              f"ARI={metrics['adjusted_rand']:.4f}  "
              f"NMI={metrics['nmi']:.4f}  "
              f"Time={train_time:.2f}s")

    print(f"\n{'=' * 85}")
    print(f"{'Linkage':<25} {'Sil':>8} {'ARI':>8} {'NMI':>8} {'CH':>10} {'DB':>8}")
    print("-" * 85)
    for name, m in results.items():
        print(f"{name:<25} {m['silhouette']:>8.4f} {m['adjusted_rand']:>8.4f} "
              f"{m['nmi']:>8.4f} {m['calinski_harabasz']:>10.1f} {m['davies_bouldin']:>8.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Ward's produces compact, balanced clusters (usually best for spherical data).")
    print("  2. Single linkage can chain together distant points (elongated clusters).")
    print("  3. Complete linkage resists chaining but may split natural clusters.")
    print("  4. Average is a compromise between single and complete.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Segmentation
# ---------------------------------------------------------------------------

def real_world_demo():
    """Customer segmentation by purchase behaviour (from scratch)."""
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Segmentation (Hierarchical NumPy)")
    print("=" * 80)

    np.random.seed(42)

    # Generate customer purchase data
    budget = np.column_stack([
        np.random.normal(200, 80, 80),   # Annual spend
        np.random.normal(3, 1.5, 80),    # Frequency
        np.random.normal(90, 30, 80),    # Recency
    ])
    regular = np.column_stack([
        np.random.normal(800, 200, 80),
        np.random.normal(12, 3, 80),
        np.random.normal(30, 15, 80),
    ])
    premium = np.column_stack([
        np.random.normal(2500, 500, 60),
        np.random.normal(25, 5, 60),
        np.random.normal(7, 5, 60),
    ])
    churned = np.column_stack([
        np.random.normal(500, 200, 60),
        np.random.normal(1, 0.5, 60),
        np.random.normal(180, 40, 60),
    ])

    X = np.vstack([budget, regular, premium, churned])
    true_labels = np.array([0]*80 + [1]*80 + [2]*60 + [3]*60)
    segment_names = ["Budget", "Regular", "Premium", "Churned"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nDataset: {X.shape[0]} customers, 3 features")

    for link in ["ward", "complete", "average", "single"]:
        model = HierarchicalClusteringNumpy(n_clusters=4, linkage=link)
        labels = model.fit_predict(X_scaled)
        metrics = _clustering_metrics(X_scaled, labels, true_labels)

        print(f"\n  {link:<10} Sil={metrics['silhouette']:.4f}  "
              f"ARI={metrics['adjusted_rand']:.4f}  NMI={metrics['nmi']:.4f}")

        # Show cluster composition
        for cid in range(4):
            mask = labels == cid
            composition = {seg_names: int(np.sum(true_labels[mask] == sid))
                           for sid, seg_names in enumerate(segment_names)
                           if np.sum(true_labels[mask] == sid) > 0}
            print(f"    Cluster {cid} (n={np.sum(mask)}): {composition}")

    print(f"\nCONCLUSION: Ward's linkage best recovers customer segments.")

    final_model = HierarchicalClusteringNumpy(n_clusters=4, linkage="ward")
    final_model.fit(X_scaled)
    return final_model, _clustering_metrics(X_scaled, final_model.labels_, true_labels)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "n_clusters": trial.suggest_int("n_clusters", 2, 8),
        "linkage": trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"]),
    }
    model = HierarchicalClusteringNumpy(**hp)
    labels = model.fit_predict(X_train)
    return _clustering_metrics(X_train, labels, y_train)["silhouette"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=15):
    study = optuna.create_study(direction="maximize", study_name="hierarchical_numpy")
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val),
        n_trials=n_trials, show_progress_bar=True,
    )
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    model = HierarchicalClusteringNumpy(**config)
    labels = model.fit_predict(X_train)
    metrics = _clustering_metrics(X_train, labels, y_train)
    tune.report(**metrics)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "n_clusters": tune.randint(2, 9),
        "linkage": tune.choice(["ward", "complete", "average", "single"]),
    }

    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    tuner = tune.Tuner(
        trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="silhouette", mode="max", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="silhouette", mode="max")
    logger.info("Ray best: %s  sil=%.6f", best.config, best.metrics["silhouette"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Execute the full hierarchical clustering (NumPy) pipeline."""
    print("=" * 70)
    print("Hierarchical Clustering - NumPy From Scratch")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=12)
    print(f"Best params     : {study.best_trial.params}")
    print(f"Best silhouette : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=8)
    ray_best = ray_results.get_best_result(metric="silhouette", mode="max")
    print(f"Best config     : {ray_best.config}")
    print(f"Best silhouette : {ray_best.metrics['silhouette']:.6f}")

    best_params = study.best_trial.params
    test_model = HierarchicalClusteringNumpy(**best_params)
    test_labels = test_model.fit_predict(X_test)
    test_metrics = _clustering_metrics(X_test, test_labels, y_test)

    print(f"\n--- Test ---")
    for k, v in test_metrics.items():
        print(f"  {k:25s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
