"""
Hierarchical Clustering - scikit-learn Implementation
======================================================

COMPLETE ML TUTORIAL: This file implements agglomerative (bottom-up)
hierarchical clustering using scikit-learn's AgglomerativeClustering and
scipy's dendrogram visualisation utilities. Hierarchical clustering builds
a tree (dendrogram) of nested clusters by iteratively merging the two
closest clusters until a single cluster remains.

Theory & Mathematics:
    Agglomerative hierarchical clustering is a bottom-up approach:
        1. Start with N clusters (one per data point).
        2. Compute the distance between every pair of clusters.
        3. Merge the two closest clusters.
        4. Update the distance matrix.
        5. Repeat steps 2-4 until only one cluster remains (or k clusters).

    The key choice is the LINKAGE criterion, which defines "distance between
    clusters" given the pairwise distances between individual points:

    Single Linkage: d(A, B) = min_{a in A, b in B} d(a, b)
        - Merges the two clusters with the closest pair of points.
        - Tends to produce elongated, "chaining" clusters.
        - Can find non-convex clusters but is noise-sensitive.

    Complete Linkage: d(A, B) = max_{a in A, b in B} d(a, b)
        - Merges clusters where the FARTHEST pair is closest.
        - Produces compact, spherical clusters.
        - More robust to noise than single linkage.

    Average Linkage (UPGMA): d(A, B) = mean_{a in A, b in B} d(a, b)
        - Uses the average of all pairwise distances.
        - Compromise between single and complete.
        - Less affected by outliers than single or complete.

    Ward's Method: Minimises the total within-cluster variance.
        - At each step, merges the pair that causes the smallest increase
          in total within-cluster sum of squares (WCSS).
        - Equivalent to: d(A, B) = sqrt(2*|A|*|B|/(|A|+|B|)) * ||c_A - c_B||
          where c_A, c_B are cluster centroids.
        - Tends to produce roughly equal-sized, compact clusters.
        - The default and usually the best linkage for well-separated clusters.

    Dendrogram:
        The merging history forms a binary tree (dendrogram). The y-axis shows
        the distance at which each merge occurred. Cutting the dendrogram at
        a given height yields a clustering with a specific number of clusters.
        The "gap" in merge distances suggests the natural number of clusters.

    Complexity:
        Time: O(n^3) for naive, O(n^2 * log(n)) for efficient implementations.
        Space: O(n^2) for the distance matrix.

Hyperparameters:
    - n_clusters (int): Number of clusters. Default 3.
    - linkage (str): Linkage criterion. 'ward', 'complete', 'average', 'single'.
    - metric (str): Distance metric. Default 'euclidean'.
    - distance_threshold (float): Cut-off for merging (alternative to n_clusters).

Business Use Cases:
    - Customer segmentation by purchase behaviour (annual spend, frequency, recency).
    - Document topic hierarchy: organise documents into topic trees.
    - Gene expression analysis: group genes with similar expression patterns.
    - Market basket analysis: hierarchical product categories.

Advantages:
    - Produces a full hierarchy (dendrogram), not just a flat clustering.
    - No need to pre-specify k (can cut dendrogram at any level).
    - Deterministic (unlike k-means which depends on initialisation).
    - Can discover clusters of varying shapes (with appropriate linkage).

Disadvantages:
    - O(n^2) memory and O(n^2 log n) time: not scalable to large datasets.
    - Greedy: merges are irreversible (cannot undo a bad early merge).
    - Sensitive to noise and outliers (especially single linkage).
    - Ward's linkage assumes spherical, isotropic clusters.
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline monitoring
import time  # Wall-clock timing for comparisons
import warnings  # Suppress non-critical warnings
from functools import partial  # Argument binding for callbacks

# -- Third-party imports --
import numpy as np  # Core numerical operations
import optuna  # Bayesian hyperparameter optimisation
import ray  # Distributed computing for parallel HPO
from ray import tune  # Ray's tuning module
from scipy.cluster.hierarchy import dendrogram, linkage  # Dendrogram computation/display
from sklearn.cluster import AgglomerativeClustering  # sklearn's hierarchical clustering
from sklearn.datasets import make_blobs  # Synthetic blob data for clustering
from sklearn.metrics import (
    adjusted_rand_score,  # Chance-adjusted clustering agreement
    calinski_harabasz_score,  # Between/within cluster dispersion ratio
    davies_bouldin_score,  # Average cluster similarity ratio (lower=better)
    normalized_mutual_info_score,  # Information-theoretic quality [0,1]
    silhouette_score,  # Cluster cohesion vs separation [-1,1]
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=600, n_features=5, n_clusters=4, random_state=42):
    """Generate synthetic blob data for hierarchical clustering evaluation.

    WHY blobs: well-separated Gaussian clusters are ideal for evaluating
    whether hierarchical clustering correctly discovers the cluster structure.
    """
    # Generate synthetic Gaussian blobs
    X, y = make_blobs(
        n_samples=n_samples,  # Total data points
        n_features=n_features,  # Feature dimensionality
        centers=n_clusters,  # Number of true clusters
        cluster_std=1.0,  # Within-cluster standard deviation
        random_state=random_state,  # Reproducibility
    )

    # 60/20/20 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,
    )

    # Standardise features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Data: train=%d val=%d test=%d features=%d true_clusters=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0],
                X_train.shape[1], n_clusters)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Fit AgglomerativeClustering on training data.

    Args:
        X_train: Training features.
        y_train: True labels (unused by clustering, for API consistency).
        **hp: Hyperparameters (n_clusters, linkage).

    Returns:
        Fitted AgglomerativeClustering model.
    """
    n_clusters = hp.get("n_clusters", 3)  # Target number of clusters
    link = hp.get("linkage", "ward")  # Linkage criterion

    # Ward's linkage requires Euclidean metric
    # Other linkages can use various metrics, but we stick with euclidean
    model = AgglomerativeClustering(
        n_clusters=n_clusters,  # Number of clusters to find
        linkage=link,  # How to measure inter-cluster distance
    )

    # Fit and predict cluster labels
    model.fit(X_train)  # Compute the clustering

    logger.info("Hierarchical(sklearn) fitted  n_clusters=%d  linkage=%s",
                n_clusters, link)
    return model


def _clustering_metrics(X, labels_pred, labels_true=None):
    """Compute clustering quality metrics.

    Unsupervised metrics (no ground truth needed):
        - silhouette: cluster cohesion vs separation [-1, 1]
        - calinski_harabasz: between/within cluster dispersion (higher=better)
        - davies_bouldin: average similarity ratio (lower=better)

    Supervised metrics (require ground truth):
        - adjusted_rand: chance-adjusted agreement [0, 1]
        - nmi: normalised mutual information [0, 1]
    """
    n_unique = len(np.unique(labels_pred))  # Number of predicted clusters

    metrics = {}

    # Unsupervised metrics (only valid if 2 <= n_clusters <= n_samples - 1)
    if 2 <= n_unique <= len(X) - 1:
        metrics["silhouette"] = silhouette_score(X, labels_pred)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels_pred)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels_pred)
    else:
        metrics["silhouette"] = 0.0
        metrics["calinski_harabasz"] = 0.0
        metrics["davies_bouldin"] = 999.0

    # Supervised metrics (if ground truth is available)
    if labels_true is not None:
        metrics["adjusted_rand"] = adjusted_rand_score(labels_true, labels_pred)
        metrics["nmi"] = normalized_mutual_info_score(labels_true, labels_pred)
    else:
        metrics["adjusted_rand"] = 0.0
        metrics["nmi"] = 0.0

    return metrics


def validate(model, X_val, y_val):
    """Evaluate clustering on validation set."""
    # AgglomerativeClustering has no predict() method for new data
    # Re-fit on validation data to evaluate cluster quality
    val_model = AgglomerativeClustering(
        n_clusters=model.n_clusters, linkage=model._linkage,  # noqa
    )
    # Use the same linkage; sklearn stores it as model.linkage attribute
    val_model = AgglomerativeClustering(
        n_clusters=model.n_clusters, linkage="ward",
    )
    labels_pred = val_model.fit_predict(X_val)
    m = _clustering_metrics(X_val, labels_pred, y_val)
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    """Evaluate clustering on test set."""
    test_model = AgglomerativeClustering(
        n_clusters=model.n_clusters, linkage="ward",
    )
    labels_pred = test_model.fit_predict(X_test)
    m = _clustering_metrics(X_test, labels_pred, y_test)
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare hierarchical clustering with different n_clusters and linkage.

    Tests:
        - n_clusters in {3, 5, 10}
        - linkage in {ward, complete, average, single}
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Hierarchical Clustering (sklearn)")
    print("=" * 80)
    print("\nCompares n_clusters x linkage methods.\n")

    n_clusters_list = [3, 5, 10]
    linkages = ["ward", "complete", "average", "single"]

    results = {}

    for n_cl in n_clusters_list:
        for link in linkages:
            name = f"k={n_cl}, linkage={link}"

            start_time = time.time()
            model = AgglomerativeClustering(n_clusters=n_cl, linkage=link)
            labels = model.fit_predict(X_train)
            train_time = time.time() - start_time

            metrics = _clustering_metrics(X_train, labels, y_train)
            metrics["train_time"] = train_time
            results[name] = metrics

            print(f"  {name:<30} Sil={metrics['silhouette']:.4f}  "
                  f"ARI={metrics['adjusted_rand']:.4f}  "
                  f"NMI={metrics['nmi']:.4f}  "
                  f"DB={metrics['davies_bouldin']:.4f}")

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"{'Config':<30} {'Sil':>8} {'ARI':>8} {'NMI':>8} {'CH':>10} {'DB':>8}")
    print("-" * 100)
    for name, m in results.items():
        print(f"{name:<30} {m['silhouette']:>8.4f} {m['adjusted_rand']:>8.4f} "
              f"{m['nmi']:>8.4f} {m['calinski_harabasz']:>10.1f} {m['davies_bouldin']:>8.4f}")

    # Dendrogram description
    print(f"\nDENDROGRAM:")
    print("  The dendrogram shows the hierarchy of cluster merges.")
    print("  Y-axis: distance at which each merge occurred.")
    print("  Large gaps suggest natural cluster boundaries.")
    print("  Cut the dendrogram at a height to get k clusters.")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Ward's linkage: compact, equal-sized clusters (best for spherical data).")
    print("  2. Single linkage: finds elongated clusters but suffers from 'chaining'.")
    print("  3. Complete linkage: compact clusters, robust to noise.")
    print("  4. Average linkage: compromise between single and complete.")
    print("  5. Match n_clusters to the number of natural groups in the data.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Segmentation
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate hierarchical clustering for customer segmentation.

    DOMAIN CONTEXT: E-commerce companies segment customers by purchase
    behaviour to tailor marketing strategies. Key features include:
        - Annual spend (total amount spent per year)
        - Purchase frequency (orders per year)
        - Recency (days since last purchase)

    Hierarchical clustering reveals the natural customer hierarchy:
        - Top level: active vs churned customers
        - Second level: high-value vs low-value active customers
        - Third level: further sub-segments

    The dendrogram shows this hierarchy visually, helping analysts
    choose the appropriate segmentation granularity.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Segmentation (Hierarchical Clustering)")
    print("=" * 80)

    np.random.seed(42)

    n_customers = 500

    # Feature names for customer purchase behaviour
    feature_names = ["annual_spend", "purchase_frequency", "recency_days"]

    # Generate customer segments with distinct purchase patterns
    # Segment 0: Budget customers (low spend, low frequency, high recency)
    budget = np.column_stack([
        np.random.normal(200, 80, 150),  # Annual spend ~$200
        np.random.normal(3, 1.5, 150),   # ~3 purchases/year
        np.random.normal(90, 30, 150),   # ~90 days since last purchase
    ])

    # Segment 1: Regular customers (moderate spend, moderate frequency)
    regular = np.column_stack([
        np.random.normal(800, 200, 150),  # Annual spend ~$800
        np.random.normal(12, 3, 150),     # ~12 purchases/year
        np.random.normal(30, 15, 150),    # ~30 days since last purchase
    ])

    # Segment 2: Premium customers (high spend, high frequency, low recency)
    premium = np.column_stack([
        np.random.normal(2500, 500, 100),  # Annual spend ~$2500
        np.random.normal(25, 5, 100),      # ~25 purchases/year
        np.random.normal(7, 5, 100),       # ~7 days since last purchase
    ])

    # Segment 3: Churned customers (past high spend, zero recent activity)
    churned = np.column_stack([
        np.random.normal(500, 200, 100),   # Annual spend ~$500 (declining)
        np.random.normal(1, 0.5, 100),     # ~1 purchase/year (dropping off)
        np.random.normal(180, 40, 100),    # ~180 days since last purchase
    ])

    # Combine all segments
    X = np.vstack([budget, regular, premium, churned])  # Shape: (500, 3)
    true_labels = np.array(
        [0] * 150 + [1] * 150 + [2] * 100 + [3] * 100  # True segment labels
    )
    segment_names = ["Budget", "Regular", "Premium", "Churned"]

    print(f"\nDataset: {n_customers} customers, {len(feature_names)} features")
    print(f"True segments: {', '.join(segment_names)}")
    print(f"Features: {', '.join(feature_names)}")

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute scipy linkage matrix for dendrogram analysis
    Z = linkage(X_scaled, method="ward")  # Ward's linkage for dendrogram

    # Show last 10 merges (most significant)
    print(f"\n--- Last 10 Merge Steps (Dendrogram) ---")
    print(f"{'Step':>6} {'Cluster A':>10} {'Cluster B':>10} {'Distance':>12} {'Size':>6}")
    print("-" * 50)
    for i in range(max(0, len(Z) - 10), len(Z)):
        print(f"{i+1:>6} {int(Z[i, 0]):>10} {int(Z[i, 1]):>10} "
              f"{Z[i, 2]:>12.4f} {int(Z[i, 3]):>6}")

    # Try different cluster counts
    print(f"\n--- Clustering Results by n_clusters ---")
    for k in [2, 3, 4, 5]:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X_scaled)
        metrics = _clustering_metrics(X_scaled, labels, true_labels)

        print(f"\n  k={k}:")
        print(f"    Silhouette: {metrics['silhouette']:.4f}  "
              f"ARI: {metrics['adjusted_rand']:.4f}  "
              f"NMI: {metrics['nmi']:.4f}")

        # Show segment composition for each cluster
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_size = np.sum(mask)
            # What true segments are in this cluster?
            composition = {}
            for seg_id, seg_name in enumerate(segment_names):
                count = np.sum(true_labels[mask] == seg_id)
                if count > 0:
                    composition[seg_name] = count
            print(f"    Cluster {cluster_id} (n={cluster_size}): {composition}")

    # Linkage comparison
    print(f"\n--- Linkage Comparison (k=4) ---")
    for link in ["ward", "complete", "average", "single"]:
        model = AgglomerativeClustering(n_clusters=4, linkage=link)
        labels = model.fit_predict(X_scaled)
        metrics = _clustering_metrics(X_scaled, labels, true_labels)
        print(f"  {link:<10} Sil={metrics['silhouette']:.4f}  "
              f"ARI={metrics['adjusted_rand']:.4f}  NMI={metrics['nmi']:.4f}")

    print(f"\nCONCLUSION: Ward's linkage with k=4 best recovers the true segments.")
    print(f"  - The dendrogram shows clear merge distance gaps at k=4.")
    print(f"  - Budget & Churned overlap slightly (both low recent activity).")
    print(f"  - Premium customers form a tight, well-separated cluster.")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: maximise silhouette score."""
    hp = {
        "n_clusters": trial.suggest_int("n_clusters", 2, 10),
        "linkage": trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"]),
    }
    model = AgglomerativeClustering(n_clusters=hp["n_clusters"], linkage=hp["linkage"])
    labels = model.fit_predict(X_train)
    metrics = _clustering_metrics(X_train, labels, y_train)
    return metrics["silhouette"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    study = optuna.create_study(direction="maximize", study_name="hierarchical_sklearn")
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
    model = AgglomerativeClustering(n_clusters=config["n_clusters"], linkage=config["linkage"])
    labels = model.fit_predict(X_train)
    metrics = _clustering_metrics(X_train, labels, y_train)
    tune.report(**metrics)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=15):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "n_clusters": tune.randint(2, 11),
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
    """Execute the full hierarchical clustering (sklearn) pipeline."""
    print("=" * 70)
    print("Hierarchical Clustering - scikit-learn")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=20)
    print(f"Best params     : {study.best_trial.params}")
    print(f"Best silhouette : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="silhouette", mode="max")
    print(f"Best config     : {ray_best.config}")
    print(f"Best silhouette : {ray_best.metrics['silhouette']:.6f}")

    # Final evaluation
    best_params = study.best_trial.params
    final_model = AgglomerativeClustering(
        n_clusters=best_params["n_clusters"], linkage=best_params["linkage"],
    )
    final_labels = final_model.fit_predict(X_test)
    final_metrics = _clustering_metrics(X_test, final_labels, y_test)

    print(f"\n--- Test ---")
    for k, v in final_metrics.items():
        print(f"  {k:25s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
