"""
t-SNE (t-distributed Stochastic Neighbour Embedding) - scikit-learn Implementation
====================================================================================

COMPLETE ML TUTORIAL: This file implements t-SNE using scikit-learn for
non-linear dimensionality reduction and visualisation. Unlike PCA which
preserves global variance structure, t-SNE excels at preserving LOCAL
neighbourhood structure, making it the go-to method for visualising
high-dimensional clusters in 2D or 3D.

Theory & Mathematics:
    t-SNE converts high-dimensional pairwise similarities into probabilities
    and finds a low-dimensional embedding that preserves these probabilities.

    Step 1 -- High-dimensional Affinities (Gaussian kernel):
        For each pair (i, j), compute conditional probability:
            p_{j|i} = exp(-||x_i - x_j||^2 / (2 * sigma_i^2))
                      / sum_{k != i} exp(-||x_i - x_k||^2 / (2 * sigma_i^2))

        The bandwidth sigma_i is chosen so that the effective number of
        neighbours (perplexity) matches a user-specified value:
            Perp(P_i) = 2^{H(P_i)} where H is Shannon entropy.

        Symmetrise: p_{ij} = (p_{j|i} + p_{i|j}) / (2*N)

    Step 2 -- Low-dimensional Affinities (Student-t kernel):
        In the embedding space, use a Student-t distribution with 1 degree
        of freedom (Cauchy distribution):
            q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Z
        where Z = sum_{k != l} (1 + ||y_k - y_l||^2)^{-1}

        WHY Student-t (not Gaussian): The heavy tails of Student-t allow
        distant points in high-dim to be mapped EVEN FURTHER apart in
        low-dim, solving the "crowding problem" that plagues SNE.

    Step 3 -- Minimise KL Divergence:
        KL(P || Q) = sum_{i != j} p_{ij} * log(p_{ij} / q_{ij})

        This is minimised via gradient descent with momentum.
        Early exaggeration (multiplying P by 4) for the first 250 iterations
        helps form tight clusters before fine-tuning positions.

    Gradient:
        dC/dy_i = 4 * sum_j (p_{ij} - q_{ij}) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^{-1}

Hyperparameters:
    - perplexity (float): Effective number of neighbours. Default 30.
        Low (5-10): focuses on very local structure, small tight clusters.
        High (30-50): considers more neighbours, larger-scale structure.
    - learning_rate (float): Step size for gradient descent. Default 200.
    - n_iter (int): Number of gradient descent iterations. Default 1000.
    - n_components (int): Embedding dimensions (usually 2 or 3). Default 2.
    - early_exaggeration (float): Factor for P in early iterations. Default 12.
    - metric (str): Distance metric for pairwise distances. Default 'euclidean'.

Business Use Cases:
    - Customer segment visualisation: see how customer groups separate in 2D.
    - Text document clustering: visualise topic clusters from TF-IDF features.
    - Image embedding exploration: visualise CNN feature spaces.
    - Anomaly visualisation: spot outlier points isolated from clusters.

Advantages:
    - Excellent at revealing cluster structure in high-dimensional data.
    - Handles non-linear relationships (unlike PCA).
    - Works well with up to ~10,000 data points.
    - The Student-t kernel solves the crowding problem.

Disadvantages:
    - Stochastic: different runs give different embeddings.
    - Computationally expensive: O(n^2) for exact, O(n*log(n)) for Barnes-Hut.
    - Sensitive to hyperparameters (especially perplexity and learning_rate).
    - Cannot be used for new data without re-fitting (no transform method).
    - Distances in the embedding are NOT meaningful (only neighbourhood is).
    - Global structure is NOT preserved (cluster positions are arbitrary).
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline monitoring
import time  # Wall-clock timing for speed comparisons
from functools import partial  # Binds arguments for callbacks

# -- Third-party imports --
import numpy as np  # Core numerical operations
import optuna  # Bayesian hyperparameter optimisation
import ray  # Distributed computing for parallel HPO
from ray import tune  # Ray's tuning module
from sklearn.datasets import make_classification  # Synthetic data generation
from sklearn.manifold import TSNE  # scikit-learn's t-SNE implementation
from sklearn.metrics import silhouette_score  # Cluster quality in embedding space
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.neighbors import KNeighborsClassifier  # k-NN for embedding quality
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=800, n_features=30, n_classes=5, random_state=42):
    """Generate synthetic high-dimensional data with distinct clusters.

    WHY 30 features: high enough that clusters are not visible without
    dimensionality reduction, but small enough for t-SNE to run quickly.
    WHY 5 classes: enough clusters to make the visualisation interesting.

    Returns numpy arrays and labels for all three splits.
    """
    X, y = make_classification(
        n_samples=n_samples,  # Total data points
        n_features=n_features,  # Feature dimensionality
        n_informative=15,  # Features that actually separate classes
        n_redundant=5,  # Correlated duplicates of informative features
        n_classes=n_classes,  # Number of distinct clusters/classes
        n_clusters_per_class=1,  # One cluster per class for clear separation
        class_sep=2.0,  # Separation between class centroids
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

    logger.info("Data: train=%d val=%d test=%d features=%d classes=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0],
                X_train.shape[1], len(np.unique(y_train)))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Fit t-SNE on training data and return the embedding.

    NOTE: t-SNE does not have a transform method -- it produces an embedding
    for the data it was fitted on, but cannot project new points. The model
    returned here is the fitted TSNE object, and the embedding is accessed
    via model.embedding_.

    Args:
        X_train: Training data, shape (n_samples, n_features).
        y_train: Labels (unused by t-SNE, kept for API consistency).
        **hp: Hyperparameters (perplexity, learning_rate, n_iter, etc.).

    Returns:
        Fitted TSNE object (access embedding via .embedding_).
    """
    perplexity = hp.get("perplexity", 30)  # Effective number of neighbours
    learning_rate = hp.get("learning_rate", 200)  # Gradient descent step size
    n_iter = hp.get("n_iter", 1000)  # Total gradient descent iterations
    n_components = hp.get("n_components", 2)  # Embedding dimensionality
    early_exaggeration = hp.get("early_exaggeration", 12.0)  # P multiplier in early iters

    # Ensure perplexity is valid (must be < n_samples)
    max_perp = X_train.shape[0] - 1  # Maximum valid perplexity
    perplexity = min(perplexity, max_perp)  # Clamp to valid range

    model = TSNE(
        n_components=n_components,  # Embedding dimensions (2 for visualisation)
        perplexity=perplexity,  # Controls neighbourhood size
        learning_rate=learning_rate,  # Gradient descent step size
        n_iter=n_iter,  # Number of optimisation iterations
        early_exaggeration=early_exaggeration,  # P exaggeration factor
        random_state=42,  # Reproducible embeddings
        init="pca",  # Initialise with PCA for more stable results
        method="barnes_hut",  # O(n*log(n)) approximation (vs O(n^2) exact)
    )

    # fit_transform computes the embedding (there is no separate transform)
    embedding = model.fit_transform(X_train)  # Shape: (n_samples, n_components)

    logger.info("t-SNE fitted  perplexity=%.1f  lr=%.1f  n_iter=%d  kl_div=%.4f",
                perplexity, learning_rate, n_iter, model.kl_divergence_)
    return model  # Return fitted object (embedding in .embedding_)


def _embedding_quality(embedding, labels):
    """Evaluate embedding quality using silhouette score and k-NN accuracy.

    Silhouette score: measures cluster cohesion vs separation in 2D embedding.
    k-NN accuracy: measures whether nearby points in 2D have the same class.
    """
    metrics = {}

    # Silhouette score in the embedding space
    # WHY: good embeddings should have well-separated clusters
    if len(np.unique(labels)) > 1:  # Need at least 2 classes
        sil = silhouette_score(embedding, labels)  # Range [-1, 1]
        metrics["silhouette"] = sil
    else:
        metrics["silhouette"] = 0.0

    # k-NN accuracy: train k-NN on the embedding, measure classification accuracy
    # WHY: if neighbours in 2D match in class, the embedding preserves local structure
    knn = KNeighborsClassifier(n_neighbors=5)  # 5-NN classifier
    knn.fit(embedding, labels)  # Fit on the 2D embedding
    knn_acc = knn.score(embedding, labels)  # In-sample accuracy
    metrics["knn_accuracy"] = knn_acc

    # KL divergence (from the fitted model) measures how well Q approximates P
    metrics["kl_divergence"] = 0.0  # Will be filled by caller if available

    return metrics


def validate(model, X_val, y_val):
    """Evaluate t-SNE embedding quality.

    NOTE: t-SNE cannot transform new data. We evaluate the training embedding
    quality and report the KL divergence as the primary metric.
    """
    # Use the training embedding for quality metrics
    embedding = model.embedding_  # The 2D embedding from fit_transform

    # We need the corresponding training labels -- use y_val as a proxy
    # In practice, t-SNE evaluation uses the training embedding + labels
    # Here we report KL divergence and training embedding quality
    m = {
        "kl_divergence": model.kl_divergence_,  # KL(P||Q) from the optimisation
        "n_iter": model.n_iter_,  # Actual iterations run
    }

    # Compute embedding quality on the TRAINING data (t-SNE has no transform)
    logger.info("Validation (train embedding): kl_div=%.4f", m["kl_divergence"])
    return m


def test(model, X_test, y_test):
    """Report t-SNE final metrics (KL divergence from training)."""
    m = {
        "kl_divergence": model.kl_divergence_,
        "n_iter": model.n_iter_,
    }
    logger.info("Test: kl_div=%.4f", m["kl_divergence"])
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare t-SNE across perplexity, learning_rate, and n_iter.

    Tests:
        - perplexity in {5, 30, 50}: local vs global neighbourhood
        - learning_rate in {10, 200, 500}: convergence speed
        - n_iter in {250, 1000}: optimisation thoroughness

    Expected patterns:
        - perplexity=5: very local clusters, may fragment large groups
        - perplexity=30: balanced (default), good general choice
        - perplexity=50: considers more neighbours, smoother embedding
        - learning_rate too low: slow convergence, may not finish
        - learning_rate too high: instability, poor embedding
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: t-SNE (sklearn)")
    print("=" * 80)
    print("\nPerplexity controls local vs global structure; lr controls convergence.\n")

    configs = {
        "perp=5, lr=200, iter=1000": {"perplexity": 5, "learning_rate": 200, "n_iter": 1000},
        "perp=30, lr=200, iter=1000": {"perplexity": 30, "learning_rate": 200, "n_iter": 1000},
        "perp=50, lr=200, iter=1000": {"perplexity": 50, "learning_rate": 200, "n_iter": 1000},
        "perp=30, lr=10, iter=1000": {"perplexity": 30, "learning_rate": 10, "n_iter": 1000},
        "perp=30, lr=500, iter=1000": {"perplexity": 30, "learning_rate": 500, "n_iter": 1000},
        "perp=30, lr=200, iter=250": {"perplexity": 30, "learning_rate": 200, "n_iter": 250},
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        # Compute embedding quality metrics
        embedding = model.embedding_
        quality = _embedding_quality(embedding, y_train)
        quality["kl_divergence"] = model.kl_divergence_
        quality["train_time"] = train_time
        results[name] = quality

        print(f"  {name:<35} KL={quality['kl_divergence']:.4f}  "
              f"Sil={quality['silhouette']:.4f}  "
              f"kNN={quality['knn_accuracy']:.4f}  "
              f"Time={train_time:.2f}s")

    # Summary table
    print(f"\n{'=' * 95}")
    print(f"{'Config':<35} {'KL Div':>8} {'Silhouette':>12} {'kNN Acc':>10} {'Time':>8}")
    print("-" * 95)
    for name, m in results.items():
        print(f"{name:<35} {m['kl_divergence']:>8.4f} {m['silhouette']:>12.4f} "
              f"{m['knn_accuracy']:>10.4f} {m['train_time']:>8.2f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. perplexity=5: very local structure, tight small clusters.")
    print("  2. perplexity=30: balanced default, good for most datasets.")
    print("  3. perplexity=50: smoother, larger-scale structure visible.")
    print("  4. learning_rate=200: standard default, usually works well.")
    print("  5. n_iter=250: too few iterations, embedding may be under-optimised.")
    print("  6. Always try multiple perplexity values to find the best view.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Segment Visualisation
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate t-SNE for visualising customer segments in 2D.

    DOMAIN CONTEXT: E-commerce companies collect dozens of behavioural features
    per customer (purchase history, browsing patterns, demographics). t-SNE
    projects this high-dimensional customer data to 2D, revealing natural
    customer segments that inform marketing strategies.

    This demo simulates:
        - 1000 customers with 25 behavioural features
        - 5 true customer segments (budget, mid-range, premium, VIP, churned)
        - Features include spending, frequency, recency, engagement metrics
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Segment Visualisation (t-SNE)")
    print("=" * 80)

    np.random.seed(42)

    n_customers = 1000
    n_features = 25
    n_segments = 5

    segment_names = ["Budget", "Mid-Range", "Premium", "VIP", "Churned"]

    # Generate customer features
    X = np.random.randn(n_customers, n_features)
    segment_labels = np.random.randint(0, n_segments, n_customers)

    # Add segment-specific patterns
    for seg in range(n_segments):
        mask = segment_labels == seg
        segment_shift = np.random.randn(n_features) * 3
        X[mask] += segment_shift

    # Add feature correlations (spending patterns)
    X[:, 1] += 0.6 * X[:, 0]  # Total spend ~ frequency
    X[:, 2] += 0.4 * X[:, 1]  # Avg basket ~ total spend
    X[:, 5] += 0.5 * X[:, 3]  # Web visits ~ app usage

    print(f"\nDataset: {n_customers} customers, {n_features} features, {n_segments} segments")
    print(f"Segments: {', '.join(segment_names)}")

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run t-SNE with different perplexity values
    print(f"\n--- t-SNE Embeddings with Different Perplexity ---")
    for perp in [5, 30, 50]:
        model = TSNE(
            n_components=2, perplexity=perp,
            learning_rate=200, n_iter=1000,
            random_state=42, init="pca",
        )
        embedding = model.fit_transform(X_scaled)
        quality = _embedding_quality(embedding, segment_labels)

        print(f"\n  Perplexity={perp}:")
        print(f"    KL divergence: {model.kl_divergence_:.4f}")
        print(f"    Silhouette score: {quality['silhouette']:.4f}")
        print(f"    k-NN accuracy: {quality['knn_accuracy']:.4f}")

        # Show segment centroids in 2D
        print(f"    Segment centroids (2D):")
        for seg in range(n_segments):
            mask = segment_labels == seg
            centroid = np.mean(embedding[mask], axis=0)
            print(f"      {segment_names[seg]:>10}: ({centroid[0]:>7.2f}, {centroid[1]:>7.2f})")

    print(f"\nCONCLUSION: t-SNE reveals customer segment structure in 2D.")
    print(f"  - Different perplexity values show local vs global structure.")
    print(f"  - Silhouette score indicates cluster separation quality.")
    print(f"  - Use this visualisation to inform marketing segmentation strategy.")
    print(f"  - CAUTION: distances between clusters are NOT meaningful in t-SNE.")

    return model, quality


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: maximise silhouette score of t-SNE embedding."""
    hp = {
        "perplexity": trial.suggest_float("perplexity", 5, 50),
        "learning_rate": trial.suggest_float("learning_rate", 10, 1000, log=True),
        "n_iter": trial.suggest_int("n_iter", 500, 2000, step=250),
        "early_exaggeration": trial.suggest_float("early_exaggeration", 4, 20),
    }
    model = train(X_train, y_train, **hp)
    quality = _embedding_quality(model.embedding_, y_train)
    return quality["silhouette"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    """Run Optuna to find best t-SNE hyperparameters."""
    study = optuna.create_study(direction="maximize", study_name="tsne_sklearn")
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
    """Ray trainable for t-SNE."""
    model = train(X_train, y_train, **config)
    quality = _embedding_quality(model.embedding_, y_train)
    quality["kl_divergence"] = model.kl_divergence_
    tune.report(**quality)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10):
    """Run Ray Tune search for t-SNE."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "perplexity": tune.uniform(5, 50),
        "learning_rate": tune.loguniform(10, 1000),
        "n_iter": tune.choice([500, 1000, 1500]),
        "early_exaggeration": tune.uniform(4, 20),
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
    """Execute the full t-SNE (sklearn) pipeline."""
    print("=" * 70)
    print("t-SNE - scikit-learn")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=15)
    print(f"Best params     : {study.best_trial.params}")
    print(f"Best silhouette : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=8)
    ray_best = ray_results.get_best_result(metric="silhouette", mode="max")
    print(f"Best config     : {ray_best.config}")
    print(f"Best silhouette : {ray_best.metrics['silhouette']:.6f}")

    # Final model with best params
    model = train(X_train, y_train, **study.best_trial.params)
    quality = _embedding_quality(model.embedding_, y_train)

    print(f"\n--- Final Embedding Quality ---")
    print(f"  KL divergence : {model.kl_divergence_:.6f}")
    print(f"  Silhouette    : {quality['silhouette']:.6f}")
    print(f"  k-NN accuracy : {quality['knn_accuracy']:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
