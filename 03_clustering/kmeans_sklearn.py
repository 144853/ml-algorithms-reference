"""
K-Means Clustering - scikit-learn Implementation
=================================================

Theory & Mathematics:
    K-Means is a centroid-based partitioning algorithm that divides a dataset of
    N data points into K non-overlapping clusters. The algorithm minimises the
    Within-Cluster Sum of Squares (WCSS), also known as inertia:

        J = sum_{k=1}^{K} sum_{x_i in C_k} ||x_i - mu_k||^2

    where mu_k is the centroid (mean) of cluster C_k.

    Lloyd's Algorithm (the standard K-Means procedure):
        1. Initialisation  - Select K initial centroids (randomly, or via
           k-means++ which spreads them out proportionally to squared distances).
        2. Assignment step  - Assign every point to the nearest centroid using
           Euclidean distance.
        3. Update step      - Recompute each centroid as the mean of all points
           currently assigned to that cluster.
        4. Repeat steps 2-3 until centroids converge (change < tolerance) or
           the maximum number of iterations is reached.

    Convergence is guaranteed (WCSS is monotonically non-increasing), but the
    solution depends on initialisation and may be a local minimum. Running the
    algorithm multiple times with different seeds (n_init) mitigates this.

    Initialisation methods:
        - random    : Pick K points uniformly at random from the dataset.
        - k-means++ : Sequentially pick centroids so that each new centroid is
                      far from existing ones (probability proportional to D^2).
                      This gives O(log K)-competitive solutions in expectation.

    Time complexity : O(N * K * d * I)  where d = dimensionality, I = iterations
    Space complexity: O(N * d + K * d)

Business Use Cases:
    - Customer segmentation (RFM analysis, behavioural groups)
    - Image compression (colour quantisation)
    - Document clustering (bag-of-words / TF-IDF features)
    - Feature engineering (cluster ID as a categorical feature)
    - Market basket grouping

Advantages:
    - Simple, interpretable, and fast for moderate-size data.
    - Scales well with the MiniBatchKMeans variant.
    - Guaranteed convergence.
    - Works very well when clusters are spherical and equally sized.

Disadvantages:
    - Must specify K in advance.
    - Sensitive to initialisation (mitigated by k-means++ and n_init).
    - Assumes spherical, equally-sized clusters (fails on elongated shapes).
    - Sensitive to outliers (centroids are pulled towards them).
    - Only finds convex cluster boundaries.

Hyperparameters (scikit-learn KMeans):
    - n_clusters : int        - Number of clusters K.
    - init       : str        - 'k-means++' or 'random'.
    - n_init     : int        - Number of independent runs (best kept).
    - max_iter   : int        - Maximum Lloyd iterations per run.
    - tol        : float      - Relative tolerance for convergence.
    - algorithm  : str        - 'lloyd' or 'elkan'.
"""

# -- Standard library imports for logging, type hints, and warning suppression --
import logging  # Provides structured logging for tracking pipeline progress
import warnings  # Allows us to suppress non-critical warnings (e.g., FutureWarning)
from typing import Any, Dict, Optional, Tuple  # Type annotations for function signatures

# -- Third-party numerical and ML imports --
import numpy as np  # Core numerical library for array operations and random generation
import optuna  # Bayesian hyperparameter optimisation framework
import ray  # Distributed computing framework for scaling HPO across workers
from ray import tune  # Ray's hyperparameter tuning module
from sklearn.cluster import KMeans  # Scikit-learn's optimised K-Means implementation
from sklearn.datasets import make_blobs  # Generates isotropic Gaussian blobs for testing
from sklearn.metrics import (
    adjusted_rand_score,  # Measures agreement between true and predicted labels, adjusted for chance
    calinski_harabasz_score,  # Ratio of between-cluster to within-cluster dispersion (higher = better)
    davies_bouldin_score,  # Average similarity ratio of each cluster with its most similar cluster (lower = better)
    normalized_mutual_info_score,  # Information-theoretic measure of clustering quality, normalised to [0,1]
    silhouette_score,  # Measures how similar points are to own cluster vs. nearest cluster [-1, 1]
)
from sklearn.model_selection import train_test_split  # Splits data with stratification support

# -- Configure logging to show timestamps, severity level, and messages --
# WHY: Structured logging helps track pipeline progress and debug issues
# in production. The format string includes time, level, and message.
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above (WARNING, ERROR, CRITICAL)
    format="%(asctime)s | %(levelname)-8s | %(message)s",  # Consistent, parseable format
)
# Create a module-level logger so all functions in this file share the same logger name
logger = logging.getLogger(__name__)

# Suppress FutureWarning messages from sklearn and other libraries
# WHY: These warnings are about upcoming API changes that do not affect correctness
# of our current code, and they clutter the output during pipeline runs.
warnings.filterwarnings("ignore", category=FutureWarning)

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
    """Generate synthetic blob data suitable for K-Means.

    Parameters
    ----------
    n_samples : int
        Total number of data points.
    n_clusters : int
        Number of isotropic Gaussian blobs.
    n_features : int
        Dimensionality of each data point.
    cluster_std : float
        Standard deviation of the blobs.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)  ground-truth labels
    """
    # Generate isotropic Gaussian blobs using sklearn's make_blobs utility.
    # WHY make_blobs: It creates well-separated spherical clusters that are ideal
    # for validating K-Means, which assumes spherical cluster geometry.
    # The random_state ensures identical data across runs for reproducibility.
    X, y = make_blobs(
        n_samples=n_samples,  # Total points distributed evenly across clusters
        n_features=n_features,  # Dimensionality of each point (2D for easy visualisation)
        centers=n_clusters,  # Number of blob centres to generate
        cluster_std=cluster_std,  # Controls how spread out each cluster is
        random_state=random_state,  # Seed for reproducible results
    )
    # Log the dataset summary for pipeline visibility
    logger.info(
        "Generated %d samples with %d clusters in %dD space.",
        n_samples,
        n_clusters,
        n_features,
    )
    return X, y  # Return features and ground-truth labels


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, **hyperparams) -> KMeans:
    """Fit a KMeans model on *X_train* and return the fitted estimator.

    Parameters
    ----------
    X_train : ndarray of shape (n, d)
    **hyperparams : forwarded to ``sklearn.cluster.KMeans``.

    Returns
    -------
    model : fitted KMeans instance
    """
    # Define sensible default hyperparameters for K-Means
    # WHY these defaults: n_clusters=5 matches our synthetic data, k-means++ gives
    # better initialisation than random, n_init=10 runs the algorithm 10 times with
    # different seeds and keeps the best result, max_iter=300 is enough for convergence,
    # tol=1e-4 stops early when centroids barely move, random_state=42 for reproducibility.
    defaults = {
        "n_clusters": 5,  # Number of clusters to find
        "init": "k-means++",  # Smart initialisation that spreads centroids apart
        "n_init": 10,  # Run 10 times with different seeds, keep best
        "max_iter": 300,  # Maximum Lloyd iterations before stopping
        "tol": 1e-4,  # Stop when centroid movement is below this threshold
        "random_state": 42,  # Fixed seed for reproducibility
    }
    # Override defaults with any user-specified hyperparameters
    # WHY: This pattern allows the function to work with both default and tuned params
    defaults.update(hyperparams)

    # Instantiate the KMeans estimator with the merged hyperparameters
    model = KMeans(**defaults)

    # Fit the model: runs Lloyd's algorithm n_init times and keeps the best result
    # WHY fit on training data only: prevents data leakage from validation/test sets
    model.fit(X_train)

    # Log key fitting results: cluster count, inertia (WCSS), and iterations used
    # WHY log inertia: it is the primary objective K-Means minimises, useful for debugging
    logger.info(
        "KMeans fitted: K=%d  inertia=%.4f  iterations=%d",
        defaults["n_clusters"],
        model.inertia_,  # Within-Cluster Sum of Squares (lower = tighter clusters)
        model.n_iter_,  # Number of Lloyd iterations in the best run
    )
    return model


def _compute_metrics(
    X: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return a dictionary of clustering quality metrics.

    Internal metrics (always computed when >= 2 clusters found):
        - silhouette_score
        - calinski_harabasz_score
        - davies_bouldin_score

    External metrics (computed when *labels_true* is provided):
        - adjusted_rand_index
        - normalised_mutual_info
    """
    # Count the number of distinct clusters found, excluding noise label -1
    # WHY exclude -1: some algorithms (like DBSCAN) mark noise as -1; K-Means does not
    # produce -1 but we keep the logic generic for consistency across clustering files.
    n_labels = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)

    # Initialise an empty dictionary to accumulate all computed metrics
    metrics: Dict[str, float] = {}

    # Internal metrics require at least 2 clusters to be meaningful
    # WHY: Silhouette score is undefined with only 1 cluster (no inter-cluster comparison)
    if n_labels >= 2:
        # Silhouette score: measures how similar a point is to its own cluster
        # vs neighboring clusters. Range: [-1, 1]. Higher is better.
        # WHY: Unlike inertia (which always decreases with more clusters), silhouette
        # score penalizes over-clustering, making it ideal for selecting optimal k.
        # A score near 0 means the point is on the boundary between clusters.
        # Negative scores indicate the point is likely in the wrong cluster.
        metrics["silhouette_score"] = float(silhouette_score(X, labels_pred))

        # Calinski-Harabasz index: ratio of between-cluster dispersion to
        # within-cluster dispersion. Higher values indicate denser, better-separated clusters.
        # WHY: Fast to compute, no ground truth needed, good for comparing different k values.
        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(X, labels_pred)
        )

        # Davies-Bouldin index: average similarity ratio of each cluster with its
        # most similar cluster. Lower values indicate better clustering.
        # WHY: Penalises clusters that are close together or have high intra-cluster scatter.
        metrics["davies_bouldin_score"] = float(
            davies_bouldin_score(X, labels_pred)
        )
    else:
        # Warn if we found fewer than 2 clusters -- metrics would be meaningless
        logger.warning("Only %d cluster(s) found; internal metrics skipped.", n_labels)

    # External metrics require ground-truth labels and measure agreement
    if labels_true is not None:
        # Adjusted Rand Index: measures pairwise agreement between true and predicted labels,
        # adjusted for chance. Range: [-1, 1], 1 = perfect agreement, 0 = random.
        # WHY: Chance-adjusted, so it does not inflate with more clusters like raw accuracy.
        metrics["adjusted_rand_index"] = float(
            adjusted_rand_score(labels_true, labels_pred)
        )

        # Normalised Mutual Information: information-theoretic measure of how much
        # information the predicted labels share with true labels. Range: [0, 1].
        # WHY: Unlike ARI, NMI is symmetric and normalised, making it comparable across datasets.
        metrics["normalised_mutual_info"] = float(
            normalized_mutual_info_score(labels_true, labels_pred)
        )

    return metrics


def validate(
    X_val: np.ndarray,
    model: KMeans,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Predict cluster assignments for *X_val* and compute quality metrics.

    Parameters
    ----------
    X_val : ndarray of shape (n, d)
    model : fitted KMeans
    labels_true : optional ground-truth labels

    Returns
    -------
    metrics : dict
    """
    # Assign each validation point to the nearest centroid learned during training
    # WHY predict (not fit_predict): we want to evaluate generalisation of the
    # learned centroids, not refit on validation data
    labels_pred = model.predict(X_val)

    # Compute all applicable metrics on the validation set predictions
    metrics = _compute_metrics(X_val, labels_pred, labels_true)

    # Log metrics for pipeline monitoring
    logger.info("Validation metrics: %s", metrics)
    return metrics


def test(
    X_test: np.ndarray,
    model: KMeans,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate the model on a holdout test set.

    Parameters
    ----------
    X_test : ndarray of shape (n, d)
    model : fitted KMeans
    labels_true : optional ground-truth labels

    Returns
    -------
    metrics : dict
    """
    # Assign each test point to the nearest centroid from the trained model
    # WHY separate test function: keeps validation and test evaluation distinct
    # for proper ML workflow (tune on val, report on test)
    labels_pred = model.predict(X_test)

    # Compute metrics on the held-out test set
    metrics = _compute_metrics(X_test, labels_pred, labels_true)

    # Log final test metrics for end-of-pipeline reporting
    logger.info("Test metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter optimisation - Optuna
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
) -> float:
    """Optuna objective that maximises silhouette score on the validation set.

    Search space:
        - n_clusters : 2 .. 15
        - init       : k-means++ | random
        - max_iter   : 100 .. 500
        - tol        : 1e-6 .. 1e-2  (log scale)
    """
    # Suggest number of clusters: range 2-15 covers most practical scenarios
    # WHY 2-15: below 2 is meaningless, above 15 risks over-segmentation for most datasets
    n_clusters = trial.suggest_int("n_clusters", 2, 15)

    # Suggest initialisation method: k-means++ generally outperforms random
    # WHY include random: sometimes random finds better local optima by chance
    init = trial.suggest_categorical("init", ["k-means++", "random"])

    # Suggest max iterations: higher allows convergence on difficult data
    # WHY step=50: reduces search space granularity without losing meaningful differences
    max_iter = trial.suggest_int("max_iter", 100, 500, step=50)

    # Suggest convergence tolerance on log scale: spans multiple orders of magnitude
    # WHY log scale: tol values like 1e-6 and 1e-2 differ by 4 orders of magnitude,
    # so uniform sampling would waste trials on the narrow high end
    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)

    # Train model with the suggested hyperparameters
    model = train(
        X_train,
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        tol=tol,
    )

    # Evaluate on validation set; return silhouette score as the objective
    # WHY silhouette: it is the most widely used internal metric that balances
    # cluster cohesion and separation, and works without ground truth labels
    metrics = validate(X_val, model, y_val_true)
    return metrics.get("silhouette_score", -1.0)  # Return -1 if metric unavailable


def run_optuna(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
    n_trials: int = 30,
) -> optuna.Study:
    """Create and run an Optuna study.

    Returns
    -------
    study : completed Optuna study
    """
    # Reduce Optuna's own logging verbosity to avoid cluttering our pipeline output
    # WHY WARNING level: we only want to see Optuna messages if something goes wrong
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create a study that MAXIMISES the objective (silhouette score: higher = better)
    # WHY maximize: silhouette score ranges from -1 to 1, with 1 being optimal
    study = optuna.create_study(direction="maximize", study_name="kmeans_sklearn")

    # Run the optimisation for n_trials iterations
    # WHY lambda wrapper: optuna_objective needs extra args (data), but study.optimize
    # expects a callable that takes only a Trial object
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train, X_val, y_train_true, y_val_true
        ),
        n_trials=n_trials,  # Number of hyperparameter configurations to try
    )

    # Log the best result found across all trials
    logger.info(
        "Optuna best trial: value=%.4f  params=%s",
        study.best_trial.value,
        study.best_trial.params,
    )
    return study


# ---------------------------------------------------------------------------
# Hyperparameter optimisation - Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_true: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
    num_samples: int = 20,
) -> Any:
    """Run a Ray Tune hyperparameter search.

    Returns
    -------
    results : ray.tune.ResultGrid
    """
    # Define the trainable function that Ray Tune will call for each trial
    # WHY inner function: captures X_train, X_val, y_val_true via closure
    def _trainable(config: dict) -> None:
        # Train K-Means with the config suggested by Ray Tune
        model = train(X_train, **config)
        # Validate and report metrics back to Ray Tune's scheduler
        metrics = validate(X_val, model, y_val_true)
        # Report all metrics so Ray can track them; silhouette_score is the primary objective
        tune.report(
            silhouette_score=metrics.get("silhouette_score", -1.0),
            calinski_harabasz_score=metrics.get("calinski_harabasz_score", 0.0),
            davies_bouldin_score=metrics.get("davies_bouldin_score", 999.0),
        )

    # Define the search space for Ray Tune
    # WHY these ranges: match the Optuna search space for fair comparison
    search_space = {
        "n_clusters": tune.randint(2, 16),  # Uniform integer sampling [2, 15]
        "init": tune.choice(["k-means++", "random"]),  # Categorical choice
        "max_iter": tune.choice([100, 200, 300, 400, 500]),  # Discrete set of values
        "tol": tune.loguniform(1e-6, 1e-2),  # Log-uniform for spanning orders of magnitude
        "random_state": 42,  # Fixed seed for reproducibility within each trial
    }

    # Initialise Ray runtime if not already running
    # WHY ignore_reinit_error: prevents crash if Ray was already initialised in this process
    # WHY log_to_driver=False: suppresses verbose Ray worker logs
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Create and configure the Tuner
    tuner = tune.Tuner(
        _trainable,  # The function Ray will call for each trial
        param_space=search_space,  # Hyperparameter distributions to sample from
        tune_config=tune.TuneConfig(
            metric="silhouette_score",  # Metric to optimise
            mode="max",  # Maximise silhouette score
            num_samples=num_samples,  # Total number of trials to run
        ),
    )

    # Execute the hyperparameter search across all trials
    results = tuner.fit()

    # Extract and log the best result
    best = results.get_best_result(metric="silhouette_score", mode="max")
    logger.info(
        "Ray Tune best config: %s  silhouette=%.4f",
        best.config,
        best.metrics["silhouette_score"],
    )
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison Study
# ---------------------------------------------------------------------------

def compare_parameter_sets() -> None:
    """Compare different K-Means parameter configurations and explain trade-offs.

    This function systematically evaluates multiple hyperparameter combinations
    to demonstrate how each parameter affects clustering quality. It helps
    practitioners build intuition for parameter selection in real projects.

    Configurations tested:
        1. k=3 vs k=5 vs k=10 (cluster count)
        2. init='k-means++' vs init='random' (initialisation strategy)
        3. Combined: show how silhouette score changes across configurations
    """
    logger.info("=" * 70)
    logger.info("PARAMETER COMPARISON STUDY")
    logger.info("=" * 70)

    # Generate a dataset with 5 true clusters so we can see the effect of under/over-clustering
    # WHY 5 clusters: provides a meaningful baseline -- k=3 will under-cluster,
    # k=5 should match, and k=10 will over-cluster
    X, y_true = make_blobs(
        n_samples=1500,  # Enough points for stable silhouette estimates
        centers=5,  # True number of clusters in the data
        n_features=2,  # 2D for easy interpretation
        cluster_std=1.0,  # Moderate spread within each cluster
        random_state=42,  # Reproducibility
    )

    # ---- Configuration 1: k=3 (under-clustering) ----
    # WHY k=3: When k is less than the true number of clusters, K-Means is forced
    # to merge distinct groups. This is common when domain knowledge is limited.
    # BEST FOR: Coarse segmentation when you want broad, high-level groupings
    # (e.g., "low / medium / high value customers" in marketing).
    # TRADE-OFF: Loses granular distinctions between merged clusters.
    logger.info("-" * 40)
    logger.info("Config 1: k=3 (UNDER-CLUSTERING)")
    logger.info("  Scenario: Fewer clusters than true structure.")
    logger.info("  Best for: Coarse segmentation, executive-level summaries.")
    logger.info("  Risk: Merges distinct groups, losing important distinctions.")
    model_k3 = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    labels_k3 = model_k3.fit_predict(X)  # Fit and get labels in one step
    sil_k3 = silhouette_score(X, labels_k3)  # Measure clustering quality
    ari_k3 = adjusted_rand_score(y_true, labels_k3)  # Compare to ground truth
    logger.info("  Silhouette=%.4f  ARI=%.4f  Inertia=%.1f", sil_k3, ari_k3, model_k3.inertia_)

    # ---- Configuration 2: k=5 (correct k) ----
    # WHY k=5: Matches the true number of clusters in the synthetic data.
    # This should yield the best silhouette score and highest ARI.
    # BEST FOR: When you have domain knowledge or have used the elbow method / silhouette
    # analysis to determine the correct number of segments.
    # TRADE-OFF: Requires knowing or estimating k accurately beforehand.
    logger.info("-" * 40)
    logger.info("Config 2: k=5 (CORRECT K)")
    logger.info("  Scenario: Matches the true number of clusters.")
    logger.info("  Best for: When elbow method or domain knowledge guides k selection.")
    logger.info("  Expectation: Highest silhouette and ARI scores.")
    model_k5 = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
    labels_k5 = model_k5.fit_predict(X)
    sil_k5 = silhouette_score(X, labels_k5)
    ari_k5 = adjusted_rand_score(y_true, labels_k5)
    logger.info("  Silhouette=%.4f  ARI=%.4f  Inertia=%.1f", sil_k5, ari_k5, model_k5.inertia_)

    # ---- Configuration 3: k=10 (over-clustering) ----
    # WHY k=10: When k exceeds the true number of clusters, K-Means splits
    # natural clusters into fragments. Inertia will be lower (always decreases with k)
    # but silhouette often drops because fragments are not well separated.
    # BEST FOR: Feature engineering (many fine-grained cluster IDs as features),
    # or when clusters have sub-structure you want to capture.
    # TRADE-OFF: Over-segmentation makes clusters less interpretable.
    logger.info("-" * 40)
    logger.info("Config 3: k=10 (OVER-CLUSTERING)")
    logger.info("  Scenario: More clusters than true structure.")
    logger.info("  Best for: Feature engineering, capturing sub-cluster structure.")
    logger.info("  Risk: Over-segmentation makes clusters hard to interpret.")
    model_k10 = KMeans(n_clusters=10, init="k-means++", n_init=10, random_state=42)
    labels_k10 = model_k10.fit_predict(X)
    sil_k10 = silhouette_score(X, labels_k10)
    ari_k10 = adjusted_rand_score(y_true, labels_k10)
    logger.info("  Silhouette=%.4f  ARI=%.4f  Inertia=%.1f", sil_k10, ari_k10, model_k10.inertia_)

    # ---- Configuration 4: init='random' vs 'k-means++' ----
    # WHY compare init methods: k-means++ is generally superior because it spreads
    # initial centroids apart, but random init can occasionally find better local optima
    # with enough restarts (n_init).
    # BEST FOR random: Very fast single-run scenarios where n_init=1 is acceptable.
    # BEST FOR k-means++: Default in production -- more reliable convergence with fewer restarts.
    logger.info("-" * 40)
    logger.info("Config 4: init='random' vs init='k-means++'")
    logger.info("  k-means++: Spreads centroids proportional to D^2 distance.")
    logger.info("  random: Picks K points uniformly at random from data.")
    for init_method in ["k-means++", "random"]:
        model_init = KMeans(n_clusters=5, init=init_method, n_init=10, random_state=42)
        labels_init = model_init.fit_predict(X)
        sil_init = silhouette_score(X, labels_init)
        logger.info(
            "  init='%s': Silhouette=%.4f  Inertia=%.1f  Iters=%d",
            init_method, sil_init, model_init.inertia_, model_init.n_iter_,
        )

    # ---- Summary table ----
    logger.info("-" * 40)
    logger.info("SUMMARY: Silhouette scores across configurations")
    logger.info("  k=3  (under):   %.4f", sil_k3)
    logger.info("  k=5  (correct): %.4f", sil_k5)
    logger.info("  k=10 (over):    %.4f", sil_k10)
    logger.info(
        "  Conclusion: k=5 should have the highest silhouette, confirming it matches "
        "the true cluster count. Inertia always decreases with k, so use silhouette "
        "or elbow method instead of inertia alone for k selection."
    )


# ---------------------------------------------------------------------------
# Real-World Use Case: Customer Segmentation
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate K-Means on a realistic customer segmentation scenario.

    Domain Context:
        A retail company wants to segment its customer base to tailor marketing
        campaigns. Each customer is described by three features:
            - annual_income: yearly income in thousands of dollars
            - spending_score: a score (1-100) assigned by the store based on
              purchasing behaviour and frequency
            - age: customer age in years

        The goal is to discover natural groupings such as:
            - High income, high spenders (VIP / loyalty program targets)
            - Low income, high spenders (credit risk / impulse buyers)
            - Young, moderate spenders (growth segment for future campaigns)
            - Older, low spenders (retention / re-engagement candidates)

    The synthetic data mimics patterns observed in real customer datasets.
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Customer Segmentation with K-Means")
    logger.info("=" * 70)

    # -- Generate realistic synthetic customer data --
    # WHY synthetic: we simulate realistic distributions for each segment so the
    # demo is self-contained and does not require external data files.
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility

    # Segment 1: High income, high spenders (affluent loyalists)
    # WHY these values: annual income ~$85k, spending score ~75, age ~35
    # These represent the most valuable customers for premium targeting.
    n_seg1 = 200  # 200 customers in this segment
    seg1_income = rng.normal(85, 10, n_seg1)  # Mean $85k, std $10k
    seg1_spending = rng.normal(75, 8, n_seg1)  # High spending score
    seg1_age = rng.normal(35, 7, n_seg1)  # Mid-career professionals

    # Segment 2: High income, low spenders (savers / investment-minded)
    # WHY: These customers have money but are not spending it -- potential for
    # targeted promotions or premium product recommendations.
    n_seg2 = 200
    seg2_income = rng.normal(90, 12, n_seg2)  # High income
    seg2_spending = rng.normal(25, 7, n_seg2)  # But low spending
    seg2_age = rng.normal(50, 8, n_seg2)  # Older, more conservative

    # Segment 3: Low income, high spenders (aspirational / credit-risk)
    # WHY: These customers spend beyond their means -- important for credit
    # risk assessment and targeted financing offers.
    n_seg3 = 200
    seg3_income = rng.normal(30, 8, n_seg3)  # Lower income
    seg3_spending = rng.normal(70, 10, n_seg3)  # High spending
    seg3_age = rng.normal(28, 5, n_seg3)  # Younger demographic

    # Segment 4: Low income, low spenders (budget-conscious)
    # WHY: The largest segment in many real datasets -- requires value-oriented
    # marketing and discount strategies.
    n_seg4 = 300
    seg4_income = rng.normal(35, 10, n_seg4)  # Low-moderate income
    seg4_spending = rng.normal(30, 10, n_seg4)  # Low spending
    seg4_age = rng.normal(45, 12, n_seg4)  # Mixed age range

    # Combine all segments into a single dataset
    # WHY vstack: vertically stacks arrays to create the full feature matrix
    annual_income = np.concatenate([seg1_income, seg2_income, seg3_income, seg4_income])
    spending_score = np.concatenate([seg1_spending, seg2_spending, seg3_spending, seg4_spending])
    age = np.concatenate([seg1_age, seg2_age, seg3_age, seg4_age])

    # Stack features into a (n_customers, 3) matrix with named features
    # WHY column_stack: creates a 2D array from 1D arrays, one column per feature
    X_customers = np.column_stack([annual_income, spending_score, age])

    # Create ground truth labels for evaluation
    # WHY: Allows us to measure how well K-Means recovers the known segments
    y_true = np.array([0]*n_seg1 + [1]*n_seg2 + [2]*n_seg3 + [3]*n_seg4)

    # Feature names for interpretability in log output
    feature_names = ["annual_income ($k)", "spending_score (1-100)", "age (years)"]

    logger.info("Dataset: %d customers, features: %s", len(X_customers), feature_names)
    logger.info(
        "Feature ranges: income=[%.0f, %.0f], spending=[%.0f, %.0f], age=[%.0f, %.0f]",
        annual_income.min(), annual_income.max(),
        spending_score.min(), spending_score.max(),
        age.min(), age.max(),
    )

    # -- Standardise features before clustering --
    # WHY: K-Means uses Euclidean distance. Without standardisation, features with
    # larger ranges (like income in $k) would dominate the distance calculation,
    # drowning out features with smaller ranges (like spending score).
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_customers)  # Zero mean, unit variance per feature

    # -- Find optimal k using silhouette analysis --
    # WHY silhouette: it balances cohesion and separation and does not require ground truth
    logger.info("Searching for optimal k using silhouette analysis...")
    best_k, best_sil = 2, -1.0  # Initialise with worst possible silhouette
    for k in range(2, 8):  # Test k from 2 to 7
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        logger.info("  k=%d: silhouette=%.4f", k, sil)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    logger.info("Optimal k=%d (silhouette=%.4f)", best_k, best_sil)

    # -- Train final model with optimal k --
    final_model = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
    final_labels = final_model.fit_predict(X_scaled)

    # -- Analyse discovered segments --
    logger.info("-" * 40)
    logger.info("Discovered customer segments:")
    for seg_id in range(best_k):
        # Extract customers belonging to this segment
        mask = final_labels == seg_id
        seg_data = X_customers[mask]  # Use original (unscaled) data for interpretability
        logger.info(
            "  Segment %d: n=%d, avg_income=$%.0fk, avg_spending=%.0f, avg_age=%.0f",
            seg_id, mask.sum(),
            seg_data[:, 0].mean(),  # Average annual income
            seg_data[:, 1].mean(),  # Average spending score
            seg_data[:, 2].mean(),  # Average age
        )

    # -- Evaluate against known segments --
    ari = adjusted_rand_score(y_true, final_labels)
    nmi = normalized_mutual_info_score(y_true, final_labels)
    logger.info("Comparison to known segments: ARI=%.4f, NMI=%.4f", ari, nmi)
    logger.info("Customer segmentation demo complete.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full K-Means (sklearn) pipeline.

    Steps:
        1. Generate synthetic data.
        2. Train/val/test split.
        3. Baseline training & evaluation.
        4. Optuna hyperparameter search.
        5. Ray Tune hyperparameter search.
        6. Retrain with best hyperparameters and evaluate on test set.
        7. Parameter comparison study.
        8. Real-world customer segmentation demo.
    """
    logger.info("=" * 70)
    logger.info("K-Means Clustering - scikit-learn Pipeline")
    logger.info("=" * 70)

    # 1. Data generation -- create synthetic blobs with known structure
    X, y = generate_data(n_samples=1500, n_clusters=5, n_features=2)

    # 2. Split: 60% train, 20% val, 20% test
    # WHY 60/20/20: standard split that gives enough data for training while
    # reserving separate sets for hyperparameter tuning (val) and final evaluation (test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42  # 40% goes to temp (val + test)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42  # Split temp 50/50 into val and test
    )
    logger.info(
        "Split sizes - train: %d  val: %d  test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # 3. Baseline -- train with default hyperparameters to establish a reference point
    # WHY baseline first: gives us a performance floor to compare optimised models against
    logger.info("-" * 40 + " Baseline " + "-" * 40)
    baseline_model = train(X_train, n_clusters=5)
    val_metrics = validate(X_val, baseline_model, y_val)
    test_metrics = test(X_test, baseline_model, y_test)

    # 4. Optuna -- Bayesian hyperparameter optimisation
    # WHY Optuna: uses Tree-structured Parzen Estimators (TPE) which is more
    # sample-efficient than random search, requiring fewer trials to find good params
    logger.info("-" * 40 + " Optuna " + "-" * 40)
    study = run_optuna(X_train, X_val, y_train, y_val, n_trials=30)
    best_params_optuna = study.best_trial.params

    # 5. Ray Tune -- distributed hyperparameter search
    # WHY Ray Tune: can parallelise across multiple CPU cores / machines,
    # making it faster for expensive model training
    logger.info("-" * 40 + " Ray Tune " + "-" * 40)
    ray_results = ray_tune_search(X_train, X_val, y_train, y_val, num_samples=20)
    best_ray = ray_results.get_best_result(metric="silhouette_score", mode="max")
    # Extract only the hyperparameters we care about (filter out Ray internals)
    best_params_ray = {
        k: v
        for k, v in best_ray.config.items()
        if k in ("n_clusters", "init", "max_iter", "tol")
    }

    # 6. Final evaluation with best params (pick whichever study scored higher)
    # WHY compare both: Optuna and Ray Tune use different search strategies,
    # so comparing them gives us the best of both approaches
    optuna_score = study.best_trial.value
    ray_score = best_ray.metrics["silhouette_score"]
    if optuna_score >= ray_score:
        best_params = best_params_optuna
        logger.info("Using Optuna best params (silhouette=%.4f).", optuna_score)
    else:
        best_params = best_params_ray
        logger.info("Using Ray Tune best params (silhouette=%.4f).", ray_score)

    logger.info("-" * 40 + " Final Evaluation " + "-" * 40)
    # Retrain on training data with the best hyperparameters found
    final_model = train(X_train, **best_params)
    # Evaluate on the held-out test set (never seen during HPO)
    final_test = test(X_test, final_model, y_test)
    logger.info("Final test metrics: %s", final_test)

    # 7. Parameter comparison study
    compare_parameter_sets()

    # 8. Real-world demo
    real_world_demo()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
