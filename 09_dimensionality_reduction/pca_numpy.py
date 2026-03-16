"""
Principal Component Analysis (PCA) - NumPy From-Scratch Implementation
======================================================================

COMPLETE ML TUTORIAL: This file implements PCA from scratch in NumPy using
two approaches: eigendecomposition of the covariance matrix and Singular
Value Decomposition (SVD). Both methods are mathematically equivalent but
have different computational properties.

Theory & Mathematics:
    PCA finds the directions (principal components) of maximum variance in
    the data and projects the data onto these directions.

    Approach 1 -- Eigendecomposition:
        1. Centre the data: X_c = X - mean(X, axis=0)
        2. Compute the covariance matrix: C = (1/(n-1)) * X_c^T @ X_c
           WHY (n-1): Bessel's correction for unbiased variance estimation
        3. Eigendecomposition: C = V * Lambda * V^T
           where Lambda = diag(lambda_1, ..., lambda_d) are eigenvalues
           and V = [v_1, ..., v_d] are the corresponding eigenvectors
        4. Sort eigenvalues in descending order: lambda_1 >= lambda_2 >= ...
        5. Select top-k eigenvectors: W = [v_1, ..., v_k]  (d x k matrix)
        6. Project: X_reduced = X_c @ W  (n x k matrix)

    Approach 2 -- SVD:
        1. Centre the data: X_c = X - mean(X, axis=0)
        2. Compute SVD: X_c = U @ S @ V^T
           where U (n x n), S (n x d diagonal), V (d x d) are orthogonal
        3. The columns of V are the principal components (same as eigenvectors of C)
        4. The singular values s_i relate to eigenvalues by: lambda_i = s_i^2 / (n-1)
        5. Project: X_reduced = X_c @ V[:, :k]  (or equivalently U[:, :k] @ S[:k, :k])

    WHY two approaches?
        - Eigendecomposition: intuitive, directly gives eigenvalues/vectors.
          Cost: O(d^3) for the eigen solve + O(n*d^2) for covariance.
          Best when d < n (fewer features than samples).
        - SVD: numerically more stable, avoids forming X^T @ X.
          Cost: O(min(n,d)^2 * max(n,d)) for the SVD.
          Best when d >> n or when numerical precision matters.

    Explained Variance Ratio:
        ratio_i = lambda_i / sum(lambda_j)
        Measures the proportion of total variance captured by component i.

    Reconstruction:
        X_reconstructed = X_reduced @ W^T + mean(X)
        Error = ||X - X_reconstructed||^2 = sum_{i=k+1}^{d} lambda_i

Hyperparameters:
    - n_components (int): Number of principal components to retain.
    - method (str): 'eig' for eigendecomposition, 'svd' for SVD approach.

Business Use Cases:
    - Customer demographics: compress 20+ features into key dimensions.
    - Signal processing: denoise by retaining only top components.
    - Visualisation: project high-dim data to 2D/3D for human inspection.
    - Pre-processing: create uncorrelated features for downstream models.

Advantages:
    - Optimal linear dimensionality reduction (maximises retained variance).
    - Interpretable: loadings show how original features contribute.
    - Fast: eigendecomposition/SVD are well-optimised in LAPACK.
    - No iteration or convergence concerns (closed-form solution).

Disadvantages:
    - Only captures linear relationships (misses non-linear structure).
    - Sensitive to feature scaling (must standardise first).
    - All components are used in reconstruction (no sparsity in loadings).
    - Unsupervised: does not use class labels to guide dimensionality reduction.
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline monitoring
import time  # Wall-clock timing for method comparison
from functools import partial  # Binds arguments for Optuna callbacks

# -- Third-party imports --
import numpy as np  # Core numerical library for all array operations
import optuna  # Bayesian hyperparameter optimisation
import ray  # Distributed computing for parallel HPO
from ray import tune  # Ray's tuning module
from sklearn.datasets import make_classification  # Synthetic classification data
from sklearn.linear_model import LogisticRegression  # Downstream classifier for evaluation
from sklearn.metrics import accuracy_score, f1_score  # Classification metrics
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Module-scoped logger


# ---------------------------------------------------------------------------
# PCA from Scratch
# ---------------------------------------------------------------------------

class PCANumpy:
    """PCA implemented from scratch using eigendecomposition or SVD.

    Supports two mathematically equivalent approaches:
        - 'eig': eigendecomposition of the covariance matrix (intuitive)
        - 'svd': singular value decomposition of the centred data (numerically stable)
    """

    def __init__(self, n_components=2, method="eig"):
        """Initialise PCA with the desired number of components and method.

        Args:
            n_components: Number of principal components to retain. Must be <= n_features.
            method: 'eig' for eigendecomposition, 'svd' for SVD approach.
        """
        # n_components: how many principal components to keep
        # WHY: k < d reduces dimensionality while retaining maximum variance
        self.n_components = n_components  # Target dimensionality

        # method: which algorithm to use for computing principal components
        # WHY two methods: eig is intuitive, svd is more numerically stable
        self.method = method  # 'eig' or 'svd'

        # Learned parameters (populated during fit)
        self.mean_ = None  # Mean of each feature (for centring)
        self.components_ = None  # Principal component vectors (k x d)
        self.eigenvalues_ = None  # Eigenvalues (variance along each component)
        self.explained_variance_ratio_ = None  # Proportion of variance per component

    def fit(self, X):
        """Fit PCA by computing principal components from the data.

        The algorithm:
            1. Centre the data by subtracting the feature means
            2. Compute covariance matrix (eig) or perform SVD
            3. Sort components by explained variance (descending)
            4. Retain top-k components

        Args:
            X: Data matrix, shape (n_samples, n_features).

        Returns:
            self: The fitted PCA instance.
        """
        n, d = X.shape  # n = samples, d = features

        # Step 1: Centre the data by subtracting the mean of each feature
        # WHY: PCA operates on variance around the mean; without centring,
        # the first component would point toward the mean instead of the
        # direction of maximum variance
        self.mean_ = np.mean(X, axis=0)  # Mean vector, shape (d,)
        X_centred = X - self.mean_  # Centred data, shape (n, d)

        if self.method == "eig":
            # -- Approach 1: Eigendecomposition of the Covariance Matrix --

            # Step 2a: Compute the covariance matrix
            # C = (1/(n-1)) * X_c^T @ X_c
            # WHY (n-1): Bessel's correction gives unbiased variance estimation
            # WHY X^T @ X (not X @ X^T): we want a (d x d) matrix, not (n x n)
            cov_matrix = (X_centred.T @ X_centred) / (n - 1)  # Shape: (d, d)

            # Step 2b: Eigendecomposition of the symmetric covariance matrix
            # np.linalg.eigh is for symmetric/Hermitian matrices (covariance IS symmetric)
            # WHY eigh (not eig): eigh guarantees real eigenvalues for symmetric matrices
            # and is faster and more numerically stable than the general eig
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # Shapes: (d,), (d, d)

            # Step 3: Sort eigenvalues in DESCENDING order (eigh returns ascending)
            # WHY descending: the largest eigenvalue corresponds to the direction
            # of maximum variance, which is the first principal component
            sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices for descending sort
            eigenvalues = eigenvalues[sorted_indices]  # Reorder eigenvalues
            eigenvectors = eigenvectors[:, sorted_indices]  # Reorder eigenvectors

            # Step 4: Retain only the top-k eigenvectors and eigenvalues
            self.eigenvalues_ = eigenvalues[:self.n_components]  # Top-k eigenvalues
            self.components_ = eigenvectors[:, :self.n_components].T  # Top-k eigenvectors (k x d)

        elif self.method == "svd":
            # -- Approach 2: Singular Value Decomposition --

            # Step 2: Compute the full SVD of the centred data
            # X_c = U @ diag(S) @ V^T
            # full_matrices=False: economy SVD (only compute min(n,d) components)
            # WHY SVD: avoids forming X^T @ X (more numerically stable)
            # WHY full_matrices=False: saves memory when n >> d or d >> n
            U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)  # Economy SVD

            # The rows of Vt are the principal components (eigenvectors of C)
            # The singular values relate to eigenvalues: lambda_i = s_i^2 / (n-1)
            eigenvalues = (S ** 2) / (n - 1)  # Convert singular values to eigenvalues

            # Step 3: SVD already returns components in descending order of S
            # So no sorting is needed (unlike eigendecomposition)

            # Step 4: Retain top-k components
            self.eigenvalues_ = eigenvalues[:self.n_components]  # Top-k eigenvalues
            self.components_ = Vt[:self.n_components]  # Top-k right singular vectors (k x d)

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'eig' or 'svd'.")

        # Compute explained variance ratio for each component
        # total_variance = sum of ALL eigenvalues (not just the top-k)
        if self.method == "eig":
            total_variance = np.sum(eigenvalues)  # Sum of all eigenvalues from eigh
        else:
            total_variance = np.sum((S ** 2) / (n - 1))  # Sum from all singular values

        # Ratio = how much of the total variance each component explains
        self.explained_variance_ratio_ = self.eigenvalues_ / total_variance  # Shape: (k,)

        logger.info("PCA(%s) fitted  n_components=%d  explained_var=%.4f",
                     self.method, self.n_components,
                     np.sum(self.explained_variance_ratio_))
        return self  # Return fitted instance

    def transform(self, X):
        """Project data onto the principal components (reduce dimensionality).

        X_reduced = (X - mean) @ components^T

        Args:
            X: Data matrix, shape (n_samples, n_features).

        Returns:
            Reduced data, shape (n_samples, n_components).
        """
        # Centre the data using the training mean
        X_centred = X - self.mean_  # Subtract the mean learned during fit

        # Project onto the top-k principal components
        # components_ is (k x d), so transpose to (d x k) for matrix multiply
        X_reduced = X_centred @ self.components_.T  # Shape: (n, k)

        return X_reduced  # Reduced-dimension representation

    def inverse_transform(self, X_reduced):
        """Reconstruct data from the reduced representation.

        X_reconstructed = X_reduced @ components + mean

        Args:
            X_reduced: Reduced data, shape (n_samples, n_components).

        Returns:
            Reconstructed data, shape (n_samples, n_features).
        """
        # Project back to original space and add the mean back
        X_reconstructed = X_reduced @ self.components_ + self.mean_  # Shape: (n, d)
        return X_reconstructed  # Approximate reconstruction of original data

    def fit_transform(self, X):
        """Fit PCA and transform the data in one step."""
        self.fit(X)  # Compute principal components
        return self.transform(X)  # Project data


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=20, n_informative=10,
                  n_classes=3, random_state=42):
    """Generate synthetic classification data for PCA evaluation."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=5,
        n_classes=n_classes, n_clusters_per_class=1,
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
    """Fit PCA on training data."""
    n_components = hp.get("n_components", 5)  # Target dimensionality
    method = hp.get("method", "eig")  # eig or svd

    pca = PCANumpy(n_components=n_components, method=method)
    pca.fit(X_train)  # Compute principal components from training data
    return pca


def _reconstruction_error(pca, X):
    """Mean squared reconstruction error."""
    X_reduced = pca.transform(X)  # Project to k-dim
    X_reconstructed = pca.inverse_transform(X_reduced)  # Reconstruct to d-dim
    return np.mean((X - X_reconstructed) ** 2)  # MSE between original and reconstructed


def _downstream_accuracy(pca, X_train, y_train, X_eval, y_eval):
    """Downstream classifier accuracy on PCA-reduced features."""
    X_train_r = pca.transform(X_train)  # Reduce training data
    X_eval_r = pca.transform(X_eval)  # Reduce evaluation data

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_r, y_train)
    y_pred = clf.predict(X_eval_r)

    return {
        "accuracy": accuracy_score(y_eval, y_pred),
        "f1": f1_score(y_eval, y_pred, average="weighted"),
    }


def validate(model, X_val, y_val, X_train=None, y_train=None):
    """Evaluate PCA on validation set."""
    recon_err = _reconstruction_error(model, X_val)
    clf_metrics = (
        _downstream_accuracy(model, X_train, y_train, X_val, y_val)
        if X_train is not None else {"accuracy": 0.0, "f1": 0.0}
    )
    m = {
        "reconstruction_error": recon_err,
        "explained_variance": float(np.sum(model.explained_variance_ratio_)),
        "accuracy": clf_metrics["accuracy"],
        "f1": clf_metrics["f1"],
    }
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test, X_train=None, y_train=None):
    """Evaluate PCA on test set."""
    recon_err = _reconstruction_error(model, X_test)
    clf_metrics = (
        _downstream_accuracy(model, X_train, y_train, X_test, y_test)
        if X_train is not None else {"accuracy": 0.0, "f1": 0.0}
    )
    m = {
        "reconstruction_error": recon_err,
        "explained_variance": float(np.sum(model.explained_variance_ratio_)),
        "accuracy": clf_metrics["accuracy"],
        "f1": clf_metrics["f1"],
    }
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare PCA with different n_components and methods (eig vs svd).

    Tests n_components in {2, 5, 10} crossed with method in {eig, svd}.
    Both methods should give identical results (modulo numerical precision),
    but may differ in speed.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: PCA (NumPy) - Components x Method")
    print("=" * 80)
    print("\nCompares eigendecomposition vs SVD approaches.\n")

    configs = {}
    for nc in [2, 5, 10]:
        for method in ["eig", "svd"]:
            name = f"n={nc}, method={method}"
            configs[name] = {"n_components": nc, "method": method}

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        pca = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(pca, X_val, y_val, X_train, y_train)
        metrics["train_time"] = train_time
        results[name] = metrics

        print(f"  {name:<25} Var={metrics['explained_variance']:.4f}  "
              f"ReconErr={metrics['reconstruction_error']:.6f}  "
              f"Acc={metrics['accuracy']:.4f}  Time={train_time:.4f}s")

    # Summary
    print(f"\n{'=' * 90}")
    print(f"{'Config':<25} {'ExpVar':>8} {'ReconErr':>10} {'Accuracy':>10} {'Time':>8}")
    print("-" * 90)
    for name, m in results.items():
        print(f"{name:<25} {m['explained_variance']:>8.4f} "
              f"{m['reconstruction_error']:>10.6f} {m['accuracy']:>10.4f} "
              f"{m['train_time']:>8.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Eigendecomposition and SVD give identical results (mathematically equivalent).")
    print("  2. SVD is often faster for wide data (d >> n) and more numerically stable.")
    print("  3. Eigendecomposition is faster for tall data (n >> d).")
    print("  4. More components = lower reconstruction error, higher accuracy.")
    print("  5. Choose method based on data shape: eig for d < n, svd for d >= n.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Demographics
# ---------------------------------------------------------------------------

def real_world_demo():
    """PCA for customer demographics dimensionality reduction (from scratch)."""
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Demographics (PCA NumPy)")
    print("=" * 80)

    np.random.seed(42)

    n_customers = 1500
    n_features = 20
    n_segments = 4

    feature_names = [
        "age", "income", "education_years", "household_size", "credit_score",
        "monthly_spend_groceries", "monthly_spend_dining", "monthly_spend_travel",
        "monthly_spend_electronics", "monthly_spend_clothing",
        "web_visits_per_month", "app_usage_hours", "email_open_rate",
        "social_media_hours", "loyalty_points",
        "years_as_customer", "returns_per_year", "support_tickets",
        "avg_order_value", "purchase_frequency"
    ]

    segment_labels = np.random.randint(0, n_segments, n_customers)
    X = np.random.randn(n_customers, n_features)

    for seg in range(n_segments):
        mask = segment_labels == seg
        segment_shift = np.random.randn(n_features) * 2
        X[mask] += segment_shift

    # Add feature correlations
    X[:, 1] += 0.5 * X[:, 0]  # Income ~ age
    X[:, 4] += 0.3 * X[:, 1]  # Credit ~ income
    X[:, 5] += 0.4 * X[:, 1]  # Grocery spend ~ income
    X[:, 18] += 0.6 * X[:, 1]  # Avg order ~ income
    X[:, 11] += 0.3 * X[:, 10]  # App ~ web visits

    print(f"\nDataset: {n_customers} customers, {n_features} features, {n_segments} segments")

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

    # Compare eig vs svd with different component counts
    for method in ["eig", "svd"]:
        print(f"\n  Method: {method}")
        for n_comp in [2, 5, 10]:
            pca = PCANumpy(n_components=n_comp, method=method)
            pca.fit(X_train_s)

            recon_err = _reconstruction_error(pca, X_test_s)
            clf_metrics = _downstream_accuracy(pca, X_train_s, y_train, X_test_s, y_test)
            total_var = float(np.sum(pca.explained_variance_ratio_))

            print(f"    n={n_comp}: Var={total_var:.4f}  "
                  f"ReconErr={recon_err:.6f}  Acc={clf_metrics['accuracy']:.4f}")

            # Show top loadings for first 2 components
            if n_comp >= 2:
                for pc_idx in range(2):
                    loadings = pca.components_[pc_idx]
                    top_indices = np.argsort(np.abs(loadings))[::-1][:3]
                    top_feats = [(feature_names[i], loadings[i]) for i in top_indices]
                    print(f"      PC{pc_idx+1}: ",
                          ", ".join(f"{n}={v:.3f}" for n, v in top_feats))

    print(f"\nCONCLUSION: Both eig and svd give identical results.")
    print(f"  PCA effectively reduces 20 customer features to key dimensions.")

    # Return final model
    final_pca = PCANumpy(n_components=5, method="svd")
    final_pca.fit(X_train_s)
    return final_pca, _downstream_accuracy(final_pca, X_train_s, y_train, X_test_s, y_test)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: maximise downstream accuracy."""
    hp = {
        "n_components": trial.suggest_int("n_components", 2, min(15, X_train.shape[1])),
        "method": trial.suggest_categorical("method", ["eig", "svd"]),
    }
    pca = train(X_train, y_train, **hp)
    m = validate(pca, X_val, y_val, X_train, y_train)
    return m["accuracy"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Run Optuna optimisation."""
    study = optuna.create_study(direction="maximize", study_name="pca_numpy")
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
    """Ray trainable for PCA."""
    pca = train(X_train, y_train, **config)
    m = validate(pca, X_val, y_val, X_train, y_train)
    tune.report(**m)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    """Run Ray Tune search."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "n_components": tune.randint(2, min(16, X_train.shape[1] + 1)),
        "method": tune.choice(["eig", "svd"]),
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
# Main
# ---------------------------------------------------------------------------

def main():
    """Execute the full PCA (NumPy) pipeline."""
    print("=" * 70)
    print("Principal Component Analysis (PCA) - NumPy From Scratch")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=20)
    print(f"Best params   : {study.best_trial.params}")
    print(f"Best accuracy : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config   : {ray_best.config}")
    print(f"Best accuracy : {ray_best.metrics['accuracy']:.6f}")

    pca = train(X_train, y_train, **study.best_trial.params)

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
