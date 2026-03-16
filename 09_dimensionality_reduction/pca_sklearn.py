"""
Principal Component Analysis (PCA) - scikit-learn Implementation
================================================================

COMPLETE ML TUTORIAL: This file implements PCA using scikit-learn's optimised
implementation. PCA is the foundational dimensionality reduction technique that
finds the directions of maximum variance in high-dimensional data and projects
the data onto a lower-dimensional subspace spanned by these directions.

Theory & Mathematics:
    PCA performs a linear transformation that maps data from R^d to R^k (k < d)
    while maximising the retained variance.

    Given centred data matrix X (n x d), PCA solves:

        maximise  w^T * Cov(X) * w
        subject to  ||w|| = 1

    The solution is the eigenvector corresponding to the largest eigenvalue
    of the covariance matrix C = (1/(n-1)) * X^T @ X.

    The top-k eigenvectors form the projection matrix W (d x k), and the
    reduced representation is: X_reduced = X_centred @ W.

    Equivalently, PCA can be computed via the Singular Value Decomposition:
        X_centred = U @ S @ V^T
    where the columns of V are the principal components (eigenvectors of C),
    and the singular values s_i relate to eigenvalues by: lambda_i = s_i^2 / (n-1).

    Explained Variance:
        The proportion of variance captured by the i-th component is:
            explained_variance_ratio_i = lambda_i / sum(lambda_j)
        A scree plot visualises these ratios to help choose k.

    Reconstruction Error:
        E = ||X - X_reconstructed||^2 = sum_{i=k+1}^{d} lambda_i
        This is the variance lost by dropping the last (d-k) components.

    Whitening:
        When whiten=True, PCA rescales each component to have unit variance:
            X_white = X_reduced / sqrt(lambda_i)
        This is useful for algorithms that assume spherical data (e.g., ICA, GMM).

Hyperparameters:
    - n_components (int or float): Number of components to keep.
        - int: exact number of components (e.g., 5)
        - float in (0, 1): fraction of variance to retain (e.g., 0.95 = 95%)
    - whiten (bool): Whether to whiten the output. Default False.
    - svd_solver (str): Algorithm for SVD. 'auto', 'full', 'arpack', 'randomized'.

Business Use Cases:
    - Customer demographics: reduce 20+ survey features to 3-5 key dimensions.
    - Image compression: retain dominant pixel patterns, discard noise.
    - Noise reduction: project data onto top components to remove noise.
    - Visualisation: project high-dim data to 2D/3D for plotting.
    - Feature engineering: create uncorrelated features for downstream models.

Advantages:
    - Linear, interpretable transformation (loadings show feature contributions).
    - Fast and scalable (SVD is O(min(n,d)^2 * max(n,d))).
    - Optimal linear dimensionality reduction (maximises retained variance).
    - No hyperparameter tuning needed beyond choosing n_components.

Disadvantages:
    - Only captures linear relationships (non-linear structure is missed).
    - Sensitive to feature scaling (must standardise first).
    - Components are orthogonal directions, not always meaningful features.
    - Does not consider class labels (unsupervised; LDA is the supervised variant).
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline progress
import time  # Wall-clock timing for speed comparisons
from functools import partial  # Binds fixed arguments for callbacks

# -- Third-party imports --
import numpy as np  # Core numerical operations
import optuna  # Bayesian hyperparameter optimisation
import ray  # Distributed computing for parallel HPO
from ray import tune  # Ray's hyperparameter tuning module
from sklearn.datasets import make_classification  # Synthetic classification data
from sklearn.decomposition import PCA  # scikit-learn's PCA implementation
from sklearn.linear_model import LogisticRegression  # Downstream classifier for evaluation
from sklearn.metrics import accuracy_score, f1_score  # Classification metrics
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Module-scoped logger


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=20, n_informative=10,
                  n_classes=3, random_state=42):
    """Generate synthetic high-dimensional classification data for PCA evaluation.

    WHY classification: PCA quality is evaluated by both reconstruction error
    AND downstream classifier accuracy after dimensionality reduction.

    Args:
        n_samples: Total number of data points.
        n_features: Original feature dimensionality (to be reduced by PCA).
        n_informative: Number of truly informative features.
        n_classes: Number of target classes for downstream classification.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Generate multi-class data with known informative/redundant features
    # WHY n_redundant=5: creates correlated features that PCA should merge
    X, y = make_classification(
        n_samples=n_samples,  # Total data points
        n_features=n_features,  # Total feature dimensions
        n_informative=n_informative,  # Truly predictive features
        n_redundant=5,  # Linearly correlated features (PCA will capture these)
        n_classes=n_classes,  # Number of target classes
        n_clusters_per_class=1,  # One cluster per class for clean separation
        random_state=random_state,  # Reproducibility seed
    )

    # Two-stage split: 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state,  # 60% train
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,  # 20/20 val/test
    )

    # Standardise features: PCA is sensitive to scale (larger-variance features dominate)
    # WHY StandardScaler: PCA operates on variance; without scaling, features with
    # larger numeric ranges would dominate the principal components
    scaler = StandardScaler()  # Zero-mean, unit-variance normalisation
    X_train = scaler.fit_transform(X_train)  # Fit on train, transform
    X_val = scaler.transform(X_val)  # Apply train statistics to val
    X_test = scaler.transform(X_test)  # Apply train statistics to test

    logger.info("Data: train=%d val=%d test=%d features=%d classes=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0],
                X_train.shape[1], len(np.unique(y_train)))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Fit PCA on training data and return the fitted PCA transformer.

    PCA is unsupervised (ignores y_train), but we accept y_train for API
    consistency and downstream classifier evaluation.

    Args:
        X_train: Training feature matrix, shape (n_samples, n_features).
        y_train: Training labels (unused by PCA, used by downstream classifier).
        **hp: Hyperparameters (n_components, whiten, svd_solver).

    Returns:
        Fitted PCA object.
    """
    # Extract PCA hyperparameters with defaults
    n_components = hp.get("n_components", 5)  # Number of components or variance ratio
    whiten = hp.get("whiten", False)  # Whether to rescale components to unit variance
    svd_solver = hp.get("svd_solver", "auto")  # SVD algorithm: auto, full, arpack, randomized

    # Create and fit PCA
    # WHY fit on train only: prevents data leakage from val/test distributions
    pca = PCA(
        n_components=n_components,  # Components to keep (int or float for variance ratio)
        whiten=whiten,  # Scale components to unit variance if True
        svd_solver=svd_solver,  # Algorithm for computing SVD
    )
    pca.fit(X_train)  # Compute principal components from training data only

    # Log PCA summary
    total_var = sum(pca.explained_variance_ratio_)  # Total variance captured
    logger.info("PCA fitted  n_components=%s  explained_var=%.4f  whiten=%s",
                pca.n_components_, total_var, whiten)
    return pca  # Return fitted PCA transformer


def _reconstruction_error(pca, X):
    """Compute mean squared reconstruction error.

    Reconstruction: X -> project to k-dim -> reconstruct back to d-dim.
    Error = mean(||X - X_reconstructed||^2) measures information lost by PCA.
    """
    X_reduced = pca.transform(X)  # Project to low-dim space (n x k)
    X_reconstructed = pca.inverse_transform(X_reduced)  # Reconstruct back to d-dim
    # Compute mean squared error between original and reconstructed data
    error = np.mean((X - X_reconstructed) ** 2)  # Average squared difference
    return error  # Scalar: lower is better


def _downstream_accuracy(pca, X_train, y_train, X_eval, y_eval):
    """Evaluate PCA quality via downstream classifier accuracy.

    WHY: Reconstruction error alone does not tell us if the retained components
    are useful for the actual task. A downstream classifier measures whether
    PCA preserved the discriminative information.
    """
    # Transform both train and eval data using the fitted PCA
    X_train_r = pca.transform(X_train)  # Reduce training data
    X_eval_r = pca.transform(X_eval)  # Reduce evaluation data

    # Train a simple logistic regression on the PCA-reduced features
    # WHY LogisticRegression: fast, interpretable, well-suited for evaluating
    # whether PCA components contain enough discriminative information
    clf = LogisticRegression(max_iter=1000, random_state=42)  # Simple classifier
    clf.fit(X_train_r, y_train)  # Train on reduced features
    y_pred = clf.predict(X_eval_r)  # Predict on reduced evaluation features

    # Compute classification metrics
    acc = accuracy_score(y_eval, y_pred)  # Overall accuracy
    f1 = f1_score(y_eval, y_pred, average="weighted")  # Weighted F1 score
    return {"accuracy": acc, "f1": f1}  # Return metric dict


def validate(model, X_val, y_val, X_train=None, y_train=None):
    """Evaluate PCA on validation set using reconstruction error and downstream accuracy."""
    # Compute reconstruction error
    recon_err = _reconstruction_error(model, X_val)  # Info lost by PCA

    # Compute downstream classifier accuracy (requires training data)
    if X_train is not None and y_train is not None:
        clf_metrics = _downstream_accuracy(model, X_train, y_train, X_val, y_val)
    else:
        clf_metrics = {"accuracy": 0.0, "f1": 0.0}

    # Combine all metrics
    m = {
        "reconstruction_error": recon_err,  # Mean squared reconstruction error
        "explained_variance": sum(model.explained_variance_ratio_),  # Total variance retained
        "accuracy": clf_metrics["accuracy"],  # Downstream classifier accuracy
        "f1": clf_metrics["f1"],  # Downstream classifier F1 score
    }
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test, X_train=None, y_train=None):
    """Evaluate PCA on test set."""
    recon_err = _reconstruction_error(model, X_test)
    if X_train is not None and y_train is not None:
        clf_metrics = _downstream_accuracy(model, X_train, y_train, X_test, y_test)
    else:
        clf_metrics = {"accuracy": 0.0, "f1": 0.0}

    m = {
        "reconstruction_error": recon_err,
        "explained_variance": sum(model.explained_variance_ratio_),
        "accuracy": clf_metrics["accuracy"],
        "f1": clf_metrics["f1"],
    }
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare PCA with different n_components and whiten settings.

    Tests:
        - n_components=2: minimum for 2D visualisation
        - n_components=5: moderate reduction
        - n_components=10: retain most information
        - n_components=0.95: automatic selection to retain 95% variance
        - whiten=True vs False: effect of component scaling

    Expected patterns:
        - More components = lower reconstruction error, higher accuracy
        - 0.95 variance ratio auto-selects the right number of components
        - Whitening may help classifiers that assume spherical data
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: PCA (sklearn) - Components and Whitening")
    print("=" * 80)
    print("\nCompares reconstruction error and downstream accuracy.\n")

    configs = {
        "n=2, whiten=False": {"n_components": 2, "whiten": False},
        "n=2, whiten=True": {"n_components": 2, "whiten": True},
        "n=5, whiten=False": {"n_components": 5, "whiten": False},
        "n=5, whiten=True": {"n_components": 5, "whiten": True},
        "n=10, whiten=False": {"n_components": 10, "whiten": False},
        "n=10, whiten=True": {"n_components": 10, "whiten": True},
        "var=0.95, whiten=False": {"n_components": 0.95, "whiten": False},
        "var=0.95, whiten=True": {"n_components": 0.95, "whiten": True},
    }

    results = {}

    for name, params in configs.items():
        start_time = time.time()
        pca = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(pca, X_val, y_val, X_train, y_train)
        metrics["train_time"] = train_time
        metrics["n_actual_components"] = pca.n_components_
        results[name] = metrics

        print(f"  {name:<30} Components={pca.n_components_:>3}  "
              f"Var={metrics['explained_variance']:.4f}  "
              f"ReconErr={metrics['reconstruction_error']:.6f}  "
              f"Acc={metrics['accuracy']:.4f}")

    # Summary table
    print(f"\n{'=' * 95}")
    print(f"{'Config':<30} {'NC':>4} {'ExpVar':>8} {'ReconErr':>10} {'Accuracy':>10} {'F1':>8}")
    print("-" * 95)
    for name, m in results.items():
        print(f"{name:<30} {m['n_actual_components']:>4} {m['explained_variance']:>8.4f} "
              f"{m['reconstruction_error']:>10.6f} {m['accuracy']:>10.4f} {m['f1']:>8.4f}")

    # Scree plot description
    print(f"\nSCREE PLOT INTERPRETATION:")
    print("  A scree plot shows explained_variance_ratio for each component.")
    print("  Look for the 'elbow' where adding more components gives diminishing returns.")
    print("  Components before the elbow capture signal; after the elbow, mostly noise.")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. More components = lower reconstruction error (more info retained).")
    print("  2. n_components=0.95 auto-selects enough components for 95% variance.")
    print("  3. Whitening normalises component scales (helps some classifiers).")
    print("  4. The 'elbow' in variance ratios guides optimal component count.")
    print("  5. Downstream accuracy plateaus once key discriminative info is captured.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Demographics
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate PCA for customer demographics dimensionality reduction.

    DOMAIN CONTEXT: Marketing teams collect 20+ demographic and behavioural
    features per customer (age, income, spending categories, web activity, etc.).
    PCA reduces these to a handful of key components that capture the main
    axes of customer variation, enabling:
        - Customer segmentation in reduced space
        - Visualisation of customer clusters in 2D/3D
        - Noise reduction for downstream predictive models
        - Identification of key demographic patterns

    This demo simulates 20 customer features and reduces them to key components,
    then evaluates whether the components preserve customer segment information.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Demographics Dimensionality Reduction")
    print("=" * 80)

    np.random.seed(42)

    n_customers = 1500  # Customer base size
    n_features = 20  # Number of demographic/behavioural features
    n_segments = 4  # True customer segments (e.g., budget, mid, premium, luxury)

    # Feature names for interpretability
    feature_names = [
        "age", "income", "education_years", "household_size", "credit_score",
        "monthly_spend_groceries", "monthly_spend_dining", "monthly_spend_travel",
        "monthly_spend_electronics", "monthly_spend_clothing",
        "web_visits_per_month", "app_usage_hours", "email_open_rate",
        "social_media_hours", "loyalty_points",
        "years_as_customer", "returns_per_year", "support_tickets",
        "avg_order_value", "purchase_frequency"
    ]

    # Generate customer segments with correlated features
    segment_labels = np.random.randint(0, n_segments, n_customers)  # True segment labels
    X = np.random.randn(n_customers, n_features)  # Base random features

    # Add segment-specific patterns (making segments distinguishable)
    for seg in range(n_segments):
        mask = segment_labels == seg  # Boolean mask for this segment
        n_seg = np.sum(mask)  # Number of customers in this segment
        # Each segment has distinct patterns across features
        segment_shift = np.random.randn(n_features) * 2  # Segment-level feature means
        X[mask] += segment_shift  # Shift this segment's features

    # Add correlations between related features
    X[:, 1] += 0.5 * X[:, 0]  # Income correlates with age
    X[:, 4] += 0.3 * X[:, 1]  # Credit score correlates with income
    X[:, 5] += 0.4 * X[:, 1]  # Grocery spend correlates with income
    X[:, 18] += 0.6 * X[:, 1]  # Avg order value correlates with income
    X[:, 11] += 0.3 * X[:, 10]  # App usage correlates with web visits

    print(f"\nDataset: {n_customers} customers, {n_features} features, {n_segments} segments")
    print(f"Features: {', '.join(feature_names[:5])}... (20 total)")
    print(f"Challenge: Reduce 20 features to key components while preserving segments")

    # Split and standardise
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, segment_labels, test_size=0.3, random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Fit PCA with different component counts
    for n_comp in [2, 5, 10]:
        pca = PCA(n_components=n_comp)
        pca.fit(X_train_s)

        # Evaluate
        recon_err = _reconstruction_error(pca, X_test_s)
        clf_metrics = _downstream_accuracy(pca, X_train_s, y_train, X_test_s, y_test)
        total_var = sum(pca.explained_variance_ratio_)

        print(f"\n  n_components={n_comp}:")
        print(f"    Explained variance: {total_var:.4f}")
        print(f"    Reconstruction error: {recon_err:.6f}")
        print(f"    Downstream accuracy: {clf_metrics['accuracy']:.4f}")

        # Show top-loading features for first 2 components
        if n_comp >= 2:
            for pc_idx in range(min(2, n_comp)):
                loadings = pca.components_[pc_idx]
                top_indices = np.argsort(np.abs(loadings))[::-1][:3]
                top_features = [(feature_names[i], loadings[i]) for i in top_indices]
                print(f"    PC{pc_idx+1} top loadings: ", end="")
                print(", ".join(f"{name}={val:.3f}" for name, val in top_features))

    # Auto-select components for 95% variance
    pca_auto = PCA(n_components=0.95)
    pca_auto.fit(X_train_s)
    auto_metrics = _downstream_accuracy(pca_auto, X_train_s, y_train, X_test_s, y_test)
    print(f"\n  Auto-selected (95% variance): {pca_auto.n_components_} components")
    print(f"    Accuracy: {auto_metrics['accuracy']:.4f}")

    print(f"\nCONCLUSION: PCA reduces 20 customer features to ~{pca_auto.n_components_} key dimensions")
    print(f"  while retaining 95% of the variance and maintaining segment separability.")

    return pca_auto, auto_metrics


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimisation
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: maximise downstream accuracy after PCA."""
    hp = {
        "n_components": trial.suggest_int("n_components", 2, min(15, X_train.shape[1])),
        "whiten": trial.suggest_categorical("whiten", [True, False]),
    }
    pca = train(X_train, y_train, **hp)
    m = validate(pca, X_val, y_val, X_train, y_train)
    return m["accuracy"]  # Maximise downstream accuracy


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Run Optuna to find optimal PCA configuration."""
    study = optuna.create_study(direction="maximize", study_name="pca_sklearn")
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val),
        n_trials=n_trials, show_progress_bar=True,
    )
    logger.info("Optuna best: %s  val_acc=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    """Ray trainable for PCA."""
    pca = train(X_train, y_train, **config)
    m = validate(pca, X_val, y_val, X_train, y_train)
    tune.report(**m)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    """Run Ray Tune parallel search for optimal PCA config."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "n_components": tune.randint(2, min(16, X_train.shape[1] + 1)),
        "whiten": tune.choice([True, False]),
    }

    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    tuner = tune.Tuner(
        trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="accuracy", mode="max", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")
    logger.info("Ray best: %s  acc=%.6f", best.config, best.metrics["accuracy"])
    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full PCA (sklearn) pipeline."""
    print("=" * 70)
    print("Principal Component Analysis (PCA) - scikit-learn")
    print("=" * 70)

    # Generate data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Parameter comparison
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Real-world demo
    real_world_demo()

    # Optuna
    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=20)
    print(f"Best params   : {study.best_trial.params}")
    print(f"Best accuracy : {study.best_trial.value:.6f}")

    # Ray Tune
    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config   : {ray_best.config}")
    print(f"Best accuracy : {ray_best.metrics['accuracy']:.6f}")

    # Final evaluation
    pca = train(X_train, y_train, **study.best_trial.params)
    print(f"\nComponents: {pca.n_components_}")
    print(f"Explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    print("\n--- Validation ---")
    for k, v in validate(pca, X_val, y_val, X_train, y_train).items():
        print(f"  {k:25s}: {v:.6f}")

    print("\n--- Test ---")
    for k, v in test(pca, X_test, y_test, X_train, y_train).items():
        print(f"  {k:25s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
