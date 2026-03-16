"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- scikit-learn Implementation
====================================================================

Theory & Mathematics:
    DBSCAN is a density-based clustering algorithm that discovers clusters of
    arbitrary shape and identifies noise points.  Unlike K-Means it does NOT
    require the number of clusters to be specified in advance.

    Key concepts:
        - eps-neighbourhood: N_eps(x) = {y in D : dist(x, y) <= eps}
        - Core point:   A point x is a core point if |N_eps(x)| >= min_samples.
        - Border point: Not a core point but lies within the eps-neighbourhood
                        of at least one core point.
        - Noise point:  Neither core nor border (label = -1).

    Algorithm:
        1. For each unvisited point p:
           a. Mark p as visited.
           b. Retrieve N_eps(p).
           c. If |N_eps(p)| < min_samples, mark p as NOISE (may later become
              a border point if reached from a core point).
           d. Otherwise p is a CORE point.  Create a new cluster C:
              - Add p to C.
              - For each point q in N_eps(p):
                  * If q is not yet visited, mark visited and retrieve N_eps(q).
                    If |N_eps(q)| >= min_samples, merge N_eps(q) into the seed
                    set (q is also a core point).
                  * If q is not yet a member of any cluster, add q to C.

    Density-reachability:
        - p is directly density-reachable from q if q is a core point and
          p in N_eps(q).
        - p is density-reachable from q if there is a chain of directly
          density-reachable points from q to p.
        - p and q are density-connected if there exists a point o such that
          both p and q are density-reachable from o.
        A cluster is a maximal set of density-connected points.

    Complexity:
        Naive  : O(N^2) distance computations.
        With spatial index (ball tree / kd-tree): O(N log N) expected.

Business Use Cases:
    - Anomaly / fraud detection (noise points are anomalies).
    - Geospatial clustering (GPS traces, store locations).
    - Network intrusion detection.
    - Image segmentation on non-convex shapes.

Advantages:
    - No need to specify K.
    - Discovers clusters of arbitrary shape.
    - Robust to outliers (labels them as noise).
    - Only two intuitive hyperparameters: eps and min_samples.

Disadvantages:
    - Sensitive to eps and min_samples (poor choices produce bad results).
    - Struggles with clusters of varying density.
    - Not fully deterministic for border points (order-dependent assignment).
    - O(N^2) memory if no spatial index / with precomputed distances.

Hyperparameters:
    - eps         : float  - Maximum distance for neighbourhood membership.
    - min_samples : int    - Minimum points in eps-neighbourhood for core status.
    - metric      : str    - Distance metric ('euclidean', 'manhattan', ...).
    - algorithm   : str    - Nearest-neighbour algorithm ('auto', 'ball_tree',
                             'kd_tree', 'brute').
"""

# -- Standard library imports --
import logging  # Structured pipeline logging
import warnings  # Suppress non-critical warnings
from typing import Any, Dict, Optional, Tuple  # Type annotations

# -- Third-party imports --
import numpy as np  # Numerical array operations
import optuna  # Bayesian hyperparameter optimisation
import ray  # Distributed computing framework
from ray import tune  # Ray's HPO interface
from sklearn.cluster import DBSCAN  # Scikit-learn's optimised DBSCAN
from sklearn.datasets import make_blobs, make_moons  # Synthetic data generators
from sklearn.metrics import (
    adjusted_rand_score,  # Chance-adjusted label agreement
    calinski_harabasz_score,  # Between/within cluster ratio
    davies_bouldin_score,  # Average cluster similarity
    normalized_mutual_info_score,  # Normalised information-theoretic agreement
    silhouette_score,  # Cluster cohesion vs separation
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1500,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a mixed dataset with non-convex shapes and blobs.

    Combines make_moons (two interleaving half circles) with make_blobs
    (isotropic Gaussian) to test DBSCAN's ability to handle varied geometry.
    WHY mixed data: K-Means fails on non-convex shapes, but DBSCAN handles them.
    """
    # Half the samples as moon shapes to test non-convex clustering
    n_moons = n_samples // 2
    n_blobs = n_samples - n_moons

    # Create two interleaving half circles with slight noise
    # WHY noise=0.08: adds realism without making clusters overlap
    X_moons, y_moons = make_moons(n_samples=n_moons, noise=0.08, random_state=random_state)

    # Create three Gaussian blobs at fixed positions
    # WHY fixed centres: provides predictable test data with known structure
    X_blobs, y_blobs = make_blobs(
        n_samples=n_blobs,
        centers=[[2.5, 2.5], [-2.0, 3.0], [3.5, -1.0]],  # 3 well-separated centres
        cluster_std=0.35,  # Tight clusters for clear separation
        random_state=random_state,
    )
    # Offset moon labels so they don't clash with blob labels
    # WHY: moons have labels 0,1 and blobs have labels 0,1,2 -- offsetting avoids overlap
    y_blobs += y_moons.max() + 1

    # Combine all data into a single dataset
    X = np.vstack([X_moons, X_blobs])  # Stack feature matrices vertically
    y = np.concatenate([y_moons, y_blobs])  # Concatenate label vectors

    # Standard-scale for consistent eps interpretation
    # WHY: DBSCAN's eps parameter is distance-based; without scaling, features with
    # larger ranges would dominate neighbourhood calculations
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Zero mean, unit variance per feature

    logger.info(
        "Generated %d samples (%d moons + %d blobs), %d ground-truth clusters.",
        len(X), n_moons, n_blobs, len(set(y)),
    )
    return X, y


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, **hyperparams) -> Tuple[DBSCAN, np.ndarray]:
    """Fit DBSCAN on *X_train*.

    DBSCAN is transductive (no separate predict step for unseen data in the
    standard formulation), so we return both the fitted estimator and the
    training labels.
    WHY return labels: DBSCAN does not have a predict() method for new data.
    """
    # Sensible defaults for the mixed moons+blobs dataset
    defaults: Dict[str, Any] = {
        "eps": 0.3,  # Neighbourhood radius -- controls cluster tightness
        "min_samples": 5,  # Min neighbours for core point status
        "metric": "euclidean",  # Standard distance metric
        "algorithm": "auto",  # Let sklearn choose the best spatial index
    }
    defaults.update(hyperparams)  # Override with user params

    # Instantiate and fit DBSCAN
    model = DBSCAN(**defaults)
    labels = model.fit_predict(X_train)  # Fit and get labels in one step

    # Count clusters and noise points for logging
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())  # Noise points are labelled -1
    logger.info(
        "DBSCAN fitted: eps=%.4f  min_samples=%d  clusters=%d  noise=%d",
        defaults["eps"], defaults["min_samples"], n_clusters, n_noise,
    )
    return model, labels


def _compute_metrics(
    X: np.ndarray, labels_pred: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Internal & external clustering metrics.

    Internal metrics are only meaningful when at least 2 clusters exist and
    not all points are noise.
    WHY exclude noise from internal metrics: noise points have no cluster,
    so including them would skew silhouette and other scores.
    """
    unique = set(labels_pred)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_non_noise = int((labels_pred != -1).sum())
    # Always report cluster count and noise count
    metrics: Dict[str, float] = {
        "n_clusters": float(n_clusters),
        "n_noise": float((labels_pred == -1).sum()),
    }

    if n_clusters >= 2 and n_non_noise >= 2:
        # Only evaluate internal metrics on non-noise points
        mask = labels_pred != -1
        metrics["silhouette_score"] = float(silhouette_score(X[mask], labels_pred[mask]))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X[mask], labels_pred[mask]))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(X[mask], labels_pred[mask]))
    else:
        logger.warning("Clusters=%d, non-noise=%d; internal metrics skipped.", n_clusters, n_non_noise)

    if labels_true is not None:
        # External metrics compare against ground truth (include all points)
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(labels_true, labels_pred))
        metrics["normalised_mutual_info"] = float(normalized_mutual_info_score(labels_true, labels_pred))

    return metrics


def validate(X_val: np.ndarray, model: DBSCAN,
             labels_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Validate by re-running DBSCAN on validation data with same params.

    WHY refit: DBSCAN is transductive -- no built-in predict(). We refit with
    the same hyperparameters to check if they generalise.
    """
    params = model.get_params()  # Extract eps, min_samples, etc.
    val_model = DBSCAN(**params)  # Fresh DBSCAN with same config
    labels_pred = val_model.fit_predict(X_val)  # Fit on validation data
    metrics = _compute_metrics(X_val, labels_pred, labels_true)
    logger.info("Validation metrics: %s", metrics)
    return metrics


def test(X_test: np.ndarray, model: DBSCAN,
         labels_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Evaluate on holdout test set (re-fits with same hyperparameters)."""
    params = model.get_params()
    test_model = DBSCAN(**params)
    labels_pred = test_model.fit_predict(X_test)
    metrics = _compute_metrics(X_test, labels_pred, labels_true)
    logger.info("Test metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray,
                     y_train_true: Optional[np.ndarray] = None,
                     y_val_true: Optional[np.ndarray] = None) -> float:
    """Optuna objective maximising silhouette score on validation set."""
    # WHY log scale for eps: eps values differ by orders of magnitude (0.05 to 2.0)
    eps = trial.suggest_float("eps", 0.05, 2.0, log=True)
    # WHY 2-20 for min_samples: too low catches noise as clusters, too high misses small clusters
    min_samples = trial.suggest_int("min_samples", 2, 20)

    model, _ = train(X_train, eps=eps, min_samples=min_samples)
    metrics = validate(X_val, model, y_val_true)
    return metrics.get("silhouette_score", -1.0)


def run_optuna(X_train: np.ndarray, X_val: np.ndarray,
               y_train_true: Optional[np.ndarray] = None,
               y_val_true: Optional[np.ndarray] = None,
               n_trials: int = 40) -> optuna.Study:
    """Run Optuna study."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="dbscan_sklearn")
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
        model, _ = train(X_train, **config)
        metrics = validate(X_val, model, y_val_true)
        tune.report(silhouette_score=metrics.get("silhouette_score", -1.0),
                     n_clusters=metrics.get("n_clusters", 0),
                     n_noise=metrics.get("n_noise", 0))

    search_space = {"eps": tune.loguniform(0.05, 2.0), "min_samples": tune.randint(2, 21)}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    tuner = tune.Tuner(_trainable, param_space=search_space,
                       tune_config=tune.TuneConfig(metric="silhouette_score", mode="max",
                                                    num_samples=num_samples))
    results = tuner.fit()
    best = results.get_best_result(metric="silhouette_score", mode="max")
    logger.info("Ray Tune best: %s  silhouette=%.4f", best.config, best.metrics["silhouette_score"])
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison Study
# ---------------------------------------------------------------------------

def compare_parameter_sets() -> None:
    """Compare DBSCAN parameter configurations and explain their effects.

    Tests eps=0.3/0.5/1.0 and min_samples=3/5/10 to show how each parameter
    changes cluster count, noise ratio, and silhouette score.
    """
    logger.info("=" * 70)
    logger.info("PARAMETER COMPARISON STUDY (DBSCAN sklearn)")
    logger.info("=" * 70)

    # Generate mixed data (moons + blobs)
    X, y_true = generate_data(n_samples=1500)

    # ---- eps comparison (with fixed min_samples=5) ----

    # eps=0.3: TIGHT neighbourhoods
    # WHY: Small eps requires points to be very close to be neighbours.
    # Creates many small, dense clusters. Points between clusters become noise.
    # BEST FOR: High-density data where you want fine-grained clusters.
    # RISK: Over-fragments data; too many noise points if data is spread out.
    logger.info("-" * 40)
    logger.info("Config 1: eps=0.3 (TIGHT neighbourhoods)")
    logger.info("  Best for: Dense data, fine-grained clusters, anomaly-sensitive detection.")
    logger.info("  Risk: Over-fragmentation; many noise points on sparse data.")
    m1 = DBSCAN(eps=0.3, min_samples=5).fit(X)
    n1 = len(set(m1.labels_)) - (1 if -1 in m1.labels_ else 0)
    noise1 = (m1.labels_ == -1).sum()
    logger.info("  Clusters=%d  Noise=%d (%.1f%%)", n1, noise1, 100*noise1/len(X))

    # eps=0.5: MODERATE neighbourhoods
    # WHY: Balanced eps that captures most natural cluster structure.
    # BEST FOR: General-purpose clustering on standardised data.
    logger.info("-" * 40)
    logger.info("Config 2: eps=0.5 (MODERATE neighbourhoods)")
    logger.info("  Best for: General-purpose clustering on standardised data.")
    m2 = DBSCAN(eps=0.5, min_samples=5).fit(X)
    n2 = len(set(m2.labels_)) - (1 if -1 in m2.labels_ else 0)
    noise2 = (m2.labels_ == -1).sum()
    logger.info("  Clusters=%d  Noise=%d (%.1f%%)", n2, noise2, 100*noise2/len(X))

    # eps=1.0: LOOSE neighbourhoods
    # WHY: Large eps merges nearby clusters into single clusters.
    # BEST FOR: Finding broad, high-level groupings when data is sparse.
    # RISK: Merges distinct clusters; almost no noise points detected.
    logger.info("-" * 40)
    logger.info("Config 3: eps=1.0 (LOOSE neighbourhoods)")
    logger.info("  Best for: Sparse data, broad groupings.")
    logger.info("  Risk: Merges distinct clusters together; few noise points.")
    m3 = DBSCAN(eps=1.0, min_samples=5).fit(X)
    n3 = len(set(m3.labels_)) - (1 if -1 in m3.labels_ else 0)
    noise3 = (m3.labels_ == -1).sum()
    logger.info("  Clusters=%d  Noise=%d (%.1f%%)", n3, noise3, 100*noise3/len(X))

    # ---- min_samples comparison (with fixed eps=0.5) ----

    # min_samples=3: LOW density threshold
    # WHY: More points qualify as core points, fewer as noise.
    # BEST FOR: Small datasets or when you expect loose cluster boundaries.
    # RISK: Noisy data gets absorbed into clusters instead of being flagged.
    logger.info("-" * 40)
    logger.info("Config 4: min_samples=3 vs 5 vs 10 (with eps=0.5)")
    for ms in [3, 5, 10]:
        m = DBSCAN(eps=0.5, min_samples=ms).fit(X)
        nc = len(set(m.labels_)) - (1 if -1 in m.labels_ else 0)
        nn = (m.labels_ == -1).sum()
        # Compute silhouette only if >= 2 clusters and non-noise points exist
        mask = m.labels_ != -1
        sil = silhouette_score(X[mask], m.labels_[mask]) if nc >= 2 and mask.sum() >= 2 else -1.0
        logger.info("  min_samples=%2d: clusters=%d  noise=%d  silhouette=%.4f", ms, nc, nn, sil)

    logger.info("-" * 40)
    logger.info("KEY INSIGHT: eps controls cluster granularity (size), min_samples controls")
    logger.info("  noise sensitivity (how many neighbours needed for core status).")
    logger.info("  Rule of thumb: min_samples >= dimensionality + 1.")


# ---------------------------------------------------------------------------
# Real-World Use Case: Geographic Delivery Location Clustering
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate DBSCAN on geographic clustering of delivery locations.

    Domain Context:
        A logistics company wants to group delivery locations into geographic
        clusters for route optimisation. Each delivery has:
            - latitude: geographic latitude of delivery address
            - longitude: geographic longitude of delivery address

        The goal is to discover natural delivery zones that can be assigned
        to specific delivery drivers or vehicles. Unlike K-Means, DBSCAN:
            - Does not require specifying the number of zones
            - Finds zones of arbitrary shape (e.g., along a highway)
            - Identifies isolated deliveries as noise (outliers) that need
              special handling (long-distance single-stop routes)

        The synthetic data simulates a metropolitan area with dense urban
        clusters and scattered suburban/rural deliveries.
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Geographic Delivery Clustering with DBSCAN")
    logger.info("=" * 70)

    rng = np.random.RandomState(42)

    # -- Generate realistic delivery location data --

    # Cluster 1: Dense downtown core (many deliveries, tight area)
    # WHY tight std: downtown areas have high delivery density
    n1 = 300
    downtown_lat = rng.normal(40.7580, 0.005, n1)  # NYC-like latitude
    downtown_lon = rng.normal(-73.9855, 0.005, n1)  # NYC-like longitude

    # Cluster 2: Midtown business district (moderate density)
    n2 = 200
    midtown_lat = rng.normal(40.7831, 0.008, n2)
    midtown_lon = rng.normal(-73.9712, 0.008, n2)

    # Cluster 3: Brooklyn residential (spread out)
    n3 = 250
    brooklyn_lat = rng.normal(40.6782, 0.012, n3)
    brooklyn_lon = rng.normal(-73.9442, 0.012, n3)

    # Cluster 4: Jersey City (across the river, distinct zone)
    n4 = 150
    jersey_lat = rng.normal(40.7178, 0.006, n4)
    jersey_lon = rng.normal(-74.0431, 0.006, n4)

    # Noise: Scattered suburban/rural deliveries (isolated addresses)
    # WHY: These represent hard-to-reach single-stop deliveries
    n_noise = 50
    noise_lat = rng.uniform(40.5, 41.0, n_noise)
    noise_lon = rng.uniform(-74.2, -73.7, n_noise)

    # Combine all locations
    all_lat = np.concatenate([downtown_lat, midtown_lat, brooklyn_lat, jersey_lat, noise_lat])
    all_lon = np.concatenate([downtown_lon, midtown_lon, brooklyn_lon, jersey_lon, noise_lon])
    X_geo = np.column_stack([all_lat, all_lon])  # (N, 2) feature matrix
    y_true = np.array([0]*n1 + [1]*n2 + [2]*n3 + [3]*n4 + [-1]*n_noise)

    feature_names = ["latitude", "longitude"]
    logger.info("Dataset: %d delivery locations, features: %s", len(X_geo), feature_names)
    logger.info("  True zones: 4 clusters + %d isolated deliveries", n_noise)

    # Standardise for distance-based clustering
    # WHY: lat/lon have different scales; standardisation ensures equal weighting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_geo)

    # -- Cluster with DBSCAN --
    # WHY DBSCAN for geo data: delivery zones have irregular shapes (following roads,
    # neighbourhoods), and isolated deliveries are natural noise points.
    logger.info("Running DBSCAN with eps=0.3, min_samples=5...")
    db_model = DBSCAN(eps=0.3, min_samples=5)
    labels = db_model.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_found = (labels == -1).sum()
    logger.info("Discovered %d delivery zones, %d isolated deliveries", n_clusters, n_noise_found)

    # Analyse each zone using original coordinates
    for zone_id in sorted(set(labels)):
        mask = labels == zone_id
        zone_data = X_geo[mask]
        if zone_id == -1:
            logger.info("  NOISE (isolated): %d deliveries, spread across lat=[%.3f, %.3f]",
                        mask.sum(), zone_data[:, 0].min(), zone_data[:, 0].max())
        else:
            logger.info(
                "  Zone %d: %d deliveries, centre=(%.4f, %.4f), radius=%.4f deg",
                zone_id, mask.sum(),
                zone_data[:, 0].mean(), zone_data[:, 1].mean(),
                np.std(zone_data, axis=0).mean(),  # Average std as proxy for zone size
            )

    # Evaluate against known zones
    ari = adjusted_rand_score(y_true, labels)
    logger.info("ARI vs true zones: %.4f", ari)
    logger.info("Geographic delivery clustering demo complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full DBSCAN (sklearn) pipeline."""
    logger.info("=" * 70)
    logger.info("DBSCAN Clustering - scikit-learn Pipeline")
    logger.info("=" * 70)

    # 1. Data
    X, y = generate_data(n_samples=1500)

    # 2. Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    logger.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # 3. Baseline
    logger.info("-" * 35 + " Baseline " + "-" * 35)
    baseline_model, baseline_labels = train(X_train, eps=0.3, min_samples=5)
    val_metrics = validate(X_val, baseline_model, y_val)
    test_metrics = test(X_test, baseline_model, y_test)

    # 4. Optuna
    logger.info("-" * 35 + " Optuna " + "-" * 35)
    study = run_optuna(X_train, X_val, y_train, y_val, n_trials=40)

    # 5. Ray Tune
    logger.info("-" * 35 + " Ray Tune " + "-" * 35)
    ray_results = ray_tune_search(X_train, X_val, y_train, y_val, num_samples=20)

    # 6. Final evaluation
    best_params_optuna = study.best_trial.params
    best_ray = ray_results.get_best_result(metric="silhouette_score", mode="max")
    best_params_ray = {k: v for k, v in best_ray.config.items() if k in ("eps", "min_samples")}

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
