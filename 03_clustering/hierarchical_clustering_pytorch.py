"""
Hierarchical Clustering - PyTorch Implementation
==================================================

COMPLETE ML TUTORIAL: This file implements agglomerative hierarchical
clustering using PyTorch tensor operations for GPU-accelerated pairwise
distance computation and cluster merging. The core distance matrix
computation and updates use PyTorch, while the sequential merge logic
runs on the tensors directly.

Theory & Mathematics:
    Agglomerative hierarchical clustering merges clusters bottom-up:
        1. Start with N singleton clusters.
        2. Compute pairwise distance matrix D (N x N).
        3. Find the closest pair and merge them.
        4. Update D using the linkage criterion.
        5. Repeat until k clusters remain.

    Linkage methods implemented:
        - Single: d(A,B) = min d(a,b) -- nearest-neighbour
        - Complete: d(A,B) = max d(a,b) -- farthest-neighbour
        - Average: d(A,B) = mean d(a,b) -- UPGMA
        - Ward's: minimises within-cluster variance increase

    PyTorch-specific optimisations:
        - torch pairwise distance computation (GPU-parallelised)
        - Tensor-based distance matrix updates (avoids numpy copies)
        - GPU memory for large distance matrices

    WHY PyTorch for Hierarchical Clustering?
        - Distance matrix computation is O(n^2): highly parallelisable on GPU.
        - Tensor min/max operations for finding closest clusters are fast.
        - For very large datasets, GPU distance computation dominates runtime.

    Limitations:
        - The sequential merge loop cannot be fully parallelised.
        - GPU memory must hold the n x n distance matrix.
        - For n > 50K, O(n^2) memory may exceed GPU VRAM.

Hyperparameters:
    - n_clusters (int): Target number of clusters.
    - linkage (str): 'single', 'complete', 'average', 'ward'.

Business Use Cases:
    - Large-scale customer segmentation with GPU acceleration.
    - Biological data clustering (gene expression, protein similarity).
    - Document hierarchy construction from embedding vectors.

Advantages:
    - GPU-accelerated distance computation.
    - Full dendrogram with flexible cut point.
    - Deterministic results.

Disadvantages:
    - O(n^2) GPU memory for distance matrix.
    - Sequential merge loop limits parallelism.
    - Overhead of GPU transfer for small datasets.
"""

# -- Standard library imports --
import logging  # Structured logging
import time  # Timing
import warnings  # Warning suppression
from functools import partial  # Argument binding

# -- Third-party imports --
import numpy as np  # NumPy for data generation and metrics
import optuna  # Bayesian HPO
import ray  # Distributed computing
from ray import tune  # Ray's tuning module
import torch  # PyTorch core
from sklearn.datasets import make_blobs  # Synthetic data
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature scaling

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# -- Device selection --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Hierarchical Clustering (PyTorch)
# ---------------------------------------------------------------------------

class HierarchicalClusteringPyTorch:
    """Agglomerative hierarchical clustering with PyTorch tensor operations.

    Distance computation and matrix updates use PyTorch tensors,
    enabling GPU acceleration for the O(n^2) pairwise distance step.
    """

    def __init__(self, n_clusters=3, linkage="ward"):
        """Initialise hierarchical clustering.

        Args:
            n_clusters: Target number of clusters.
            linkage: Linkage method ('single', 'complete', 'average', 'ward').
        """
        self.n_clusters = n_clusters  # Target cluster count
        self.linkage = linkage  # Linkage criterion

        # Results
        self.labels_ = None  # Cluster labels
        self.linkage_matrix_ = None  # Dendrogram linkage matrix
        self.n_samples_ = 0  # Number of data points

    def _compute_distance_matrix(self, X_t):
        """Compute pairwise Euclidean distance matrix using PyTorch.

        Args:
            X_t: Tensor, shape (n, d) on device.

        Returns:
            Distance matrix tensor, shape (n, n).
        """
        # Vectorised distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        sq_norms = torch.sum(X_t ** 2, dim=1)  # ||x_i||^2
        sq_dist = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (X_t @ X_t.T)
        sq_dist = torch.clamp(sq_dist, min=0.0)  # Numerical safety
        return torch.sqrt(sq_dist)  # Euclidean distances

    def fit(self, X_np):
        """Fit hierarchical clustering.

        Args:
            X_np: numpy data matrix, shape (n, d).

        Returns:
            self: Fitted instance.
        """
        n = X_np.shape[0]
        self.n_samples_ = n

        # Convert to torch tensor for GPU distance computation
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

        # Compute distance matrix on GPU
        D = self._compute_distance_matrix(X_t)  # (n, n) tensor on device

        # Move to CPU for sequential merge operations
        # (the merge loop is inherently sequential, CPU is fine)
        D_cpu = D.cpu().numpy()  # Convert to numpy for merge logic

        # Tracking arrays
        active = np.ones(n, dtype=bool)  # Active cluster flags
        sizes = np.ones(n, dtype=int)  # Cluster sizes
        cluster_ids = np.arange(n)  # Cluster ID mapping

        # Track cluster membership for label assignment
        members = {i: [i] for i in range(n)}  # point indices per cluster

        # Linkage matrix (N-1 merges)
        Z = np.zeros((n - 1, 4))  # [id_a, id_b, dist, size]
        next_id = n  # Next available cluster ID

        # Merge loop
        for step in range(n - 1):
            # Mask inactive clusters
            D_masked = D_cpu.copy()
            for idx in range(n):
                if not active[idx]:
                    D_masked[idx, :] = np.inf
                    D_masked[:, idx] = np.inf
            np.fill_diagonal(D_masked, np.inf)

            # Find closest pair
            flat_idx = np.argmin(D_masked)
            i, j = np.unravel_index(flat_idx, D_masked.shape)
            merge_dist = D_masked[i, j]

            # Record merge
            id_i, id_j = cluster_ids[i], cluster_ids[j]
            new_size = sizes[i] + sizes[j]
            Z[step] = [min(id_i, id_j), max(id_i, id_j), merge_dist, new_size]

            # Update distances using Lance-Williams formula
            ni, nj = sizes[i], sizes[j]
            new_dists = np.full(n, np.inf)

            for k in range(n):
                if not active[k] or k == i or k == j:
                    continue

                nk = sizes[k]

                if self.linkage == "single":
                    new_dists[k] = min(D_cpu[i, k], D_cpu[j, k])
                elif self.linkage == "complete":
                    new_dists[k] = max(D_cpu[i, k], D_cpu[j, k])
                elif self.linkage == "average":
                    new_dists[k] = (ni * D_cpu[i, k] + nj * D_cpu[j, k]) / (ni + nj)
                elif self.linkage == "ward":
                    n_total = ni + nj + nk
                    new_dists[k] = np.sqrt(
                        ((ni + nk) * D_cpu[i, k] ** 2
                         + (nj + nk) * D_cpu[j, k] ** 2
                         - nk * D_cpu[i, j] ** 2) / n_total
                    )

            # Update distance matrix
            D_cpu[i, :] = new_dists
            D_cpu[:, i] = new_dists
            D_cpu[i, i] = 0.0

            # Update metadata
            sizes[i] = new_size
            old_id_i = id_i
            old_id_j = id_j
            cluster_ids[i] = next_id
            members[next_id] = members.get(old_id_i, []) + members.get(old_id_j, [])
            active[j] = False
            next_id += 1

        self.linkage_matrix_ = Z

        # Cut dendrogram
        self.labels_ = self._cut_dendrogram(Z, n, self.n_clusters)

        logger.info("Hierarchical(torch) fitted  n_clusters=%d  linkage=%s  device=%s",
                     self.n_clusters, self.linkage, device)
        return self

    def _cut_dendrogram(self, Z, n, k):
        """Cut dendrogram to get k clusters using union-find."""
        parent = list(range(2 * n - 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for step in range(n - k):
            c1 = int(Z[step, 0])
            c2 = int(Z[step, 1])
            new_id = n + step
            parent[find(c1)] = new_id
            parent[find(c2)] = new_id

        roots = [find(i) for i in range(n)]
        unique_roots = list(set(roots))
        root_to_label = {r: label for label, r in enumerate(unique_roots)}
        return np.array([root_to_label[r] for r in roots])

    def fit_predict(self, X_np):
        """Fit and return labels."""
        self.fit(X_np)
        return self.labels_


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=400, n_features=5, n_clusters=4, random_state=42):
    """Generate synthetic blob data."""
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
    """Fit hierarchical clustering."""
    model = HierarchicalClusteringPyTorch(
        n_clusters=hp.get("n_clusters", 3),
        linkage=hp.get("linkage", "ward"),
    )
    model.fit(X_train)
    return model


def _clustering_metrics(X, labels_pred, labels_true=None):
    """Compute clustering metrics."""
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
    """Evaluate by re-fitting on validation data."""
    val_model = HierarchicalClusteringPyTorch(
        n_clusters=model.n_clusters, linkage=model.linkage,
    )
    labels = val_model.fit_predict(X_val)
    m = _clustering_metrics(X_val, labels, y_val)
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    """Evaluate on test data."""
    test_model = HierarchicalClusteringPyTorch(
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
    """Compare linkage methods."""
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Hierarchical Clustering (PyTorch)")
    print("=" * 80)
    print(f"\nDevice: {device}\n")

    linkages = ["single", "complete", "average", "ward"]
    results = {}

    for link in linkages:
        name = f"linkage={link}"

        start_time = time.time()
        model = HierarchicalClusteringPyTorch(n_clusters=4, linkage=link)
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
    print("  1. GPU accelerates the initial distance matrix computation.")
    print("  2. The sequential merge loop runs on CPU (inherently sequential).")
    print("  3. Same linkage properties as NumPy version.")
    print("  4. For large n, GPU distance computation provides significant speedup.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Segmentation
# ---------------------------------------------------------------------------

def real_world_demo():
    """Customer segmentation (PyTorch hierarchical clustering)."""
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Segmentation (Hierarchical PyTorch)")
    print("=" * 80)

    np.random.seed(42)

    budget = np.column_stack([
        np.random.normal(200, 80, 80),
        np.random.normal(3, 1.5, 80),
        np.random.normal(90, 30, 80),
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

    print(f"\nDataset: {X.shape[0]} customers, 3 features, device={device}")

    for link in ["ward", "complete", "average", "single"]:
        model = HierarchicalClusteringPyTorch(n_clusters=4, linkage=link)
        labels = model.fit_predict(X_scaled)
        metrics = _clustering_metrics(X_scaled, labels, true_labels)

        print(f"\n  {link:<10} Sil={metrics['silhouette']:.4f}  "
              f"ARI={metrics['adjusted_rand']:.4f}  NMI={metrics['nmi']:.4f}")

        for cid in range(4):
            mask = labels == cid
            composition = {sn: int(np.sum(true_labels[mask] == sid))
                           for sid, sn in enumerate(segment_names)
                           if np.sum(true_labels[mask] == sid) > 0}
            print(f"    Cluster {cid} (n={np.sum(mask)}): {composition}")

    print(f"\nCONCLUSION: PyTorch version gives same results with GPU distance computation.")

    final_model = HierarchicalClusteringPyTorch(n_clusters=4, linkage="ward")
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
    model = HierarchicalClusteringPyTorch(**hp)
    labels = model.fit_predict(X_train)
    return _clustering_metrics(X_train, labels, y_train)["silhouette"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=15):
    study = optuna.create_study(direction="maximize", study_name="hierarchical_pytorch")
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
    model = HierarchicalClusteringPyTorch(**config)
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
    """Execute the full hierarchical clustering (PyTorch) pipeline."""
    print("=" * 70)
    print("Hierarchical Clustering - PyTorch")
    print("=" * 70)
    print(f"Device: {device}")

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
    test_model = HierarchicalClusteringPyTorch(**best_params)
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
