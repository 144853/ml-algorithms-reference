"""
Principal Component Analysis (PCA) - PyTorch Implementation
============================================================

COMPLETE ML TUTORIAL: This file implements PCA using PyTorch's linear algebra
operations (torch.linalg.eigh and torch.linalg.svd), enabling GPU-accelerated
dimensionality reduction for large datasets.

Theory & Mathematics:
    PCA finds the top-k directions of maximum variance in the data.
    PyTorch provides two approaches:

    Approach 1 -- Eigendecomposition (torch.linalg.eigh):
        1. Centre data: X_c = X - mean(X)
        2. Covariance: C = (1/(n-1)) * X_c^T @ X_c
        3. Eigendecomposition: eigenvalues, eigenvectors = torch.linalg.eigh(C)
        4. Sort descending, take top-k
        5. Project: X_reduced = X_c @ top_k_eigenvectors

    Approach 2 -- SVD (torch.linalg.svd):
        1. Centre data: X_c = X - mean(X)
        2. SVD: U, S, Vt = torch.linalg.svd(X_c, full_matrices=False)
        3. Components = Vt[:k], eigenvalues = S^2 / (n-1)
        4. Project: X_reduced = X_c @ Vt[:k].T

    Why PyTorch?
        - GPU acceleration: torch operations run on CUDA, making PCA
          feasible for datasets with millions of samples or features.
        - Ecosystem integration: PCA as a preprocessing step in a
          PyTorch training pipeline (no numpy<->torch conversion).
        - Batch processing: can process multiple datasets in parallel.

    GPU Speedup Considerations:
        - SVD benefits most from GPU (highly parallelisable).
        - Data transfer to/from GPU has overhead; only worth it for large datasets.
        - For small datasets (< 10K samples, < 100 features), NumPy is faster.

Hyperparameters:
    - n_components (int): Number of principal components to retain.
    - method (str): 'eig' or 'svd'.

Business Use Cases:
    - Large-scale customer analytics with GPU-accelerated preprocessing.
    - Real-time dimensionality reduction in production ML pipelines.
    - Image feature compression before deep learning classification.

Advantages:
    - GPU-accelerated for large datasets (millions of samples).
    - Seamless integration with PyTorch training pipelines.
    - Both eig and svd approaches available.

Disadvantages:
    - GPU memory limits the maximum dataset size.
    - Data transfer overhead for small datasets.
    - More complex setup than sklearn for simple use cases.
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline monitoring
import time  # Wall-clock timing for method/device comparison
from functools import partial  # Binds arguments for callbacks

# -- Third-party imports --
import numpy as np  # NumPy for data generation and metrics
import optuna  # Bayesian hyperparameter optimisation
import ray  # Distributed computing for parallel HPO
from ray import tune  # Ray's tuning module
import torch  # PyTorch core: tensors, autograd, GPU
from sklearn.datasets import make_classification  # Synthetic data
from sklearn.linear_model import LogisticRegression  # Downstream classifier
from sklearn.metrics import accuracy_score, f1_score  # Classification metrics
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature scaling

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -- Device selection --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# PCA (PyTorch)
# ---------------------------------------------------------------------------

class PCAPyTorch:
    """PCA implemented using PyTorch's linear algebra operations.

    Supports eigendecomposition (eigh) and SVD approaches, both
    GPU-acceleratable for large datasets.
    """

    def __init__(self, n_components=2, method="eig"):
        """Initialise PCA.

        Args:
            n_components: Number of principal components to retain.
            method: 'eig' for eigendecomposition, 'svd' for SVD.
        """
        self.n_components = n_components  # Target dimensionality
        self.method = method  # Algorithm choice

        # Learned parameters (populated during fit)
        self.mean_ = None  # Feature means (tensor on device)
        self.components_ = None  # Principal components (k x d tensor)
        self.eigenvalues_ = None  # Eigenvalues (k tensor)
        self.explained_variance_ratio_ = None  # Variance ratios (numpy array)

    def fit(self, X_np):
        """Fit PCA on numpy data, using PyTorch operations internally.

        Args:
            X_np: Input data as numpy array, shape (n, d).

        Returns:
            self: Fitted PCA instance.
        """
        # Convert numpy to torch tensor on the selected device (CPU or GPU)
        X = torch.tensor(X_np, dtype=torch.float32, device=device)  # Shape: (n, d)
        n, d = X.shape  # n = samples, d = features

        # Step 1: Centre the data
        self.mean_ = torch.mean(X, dim=0)  # Mean per feature, shape (d,)
        X_c = X - self.mean_  # Centred data, shape (n, d)

        if self.method == "eig":
            # -- Eigendecomposition approach --

            # Compute covariance matrix on GPU
            cov = (X_c.T @ X_c) / (n - 1)  # Shape: (d, d) -- symmetric

            # Eigendecomposition of symmetric matrix
            # torch.linalg.eigh returns eigenvalues in ASCENDING order
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # (d,), (d, d)

            # Reverse to descending order (largest eigenvalue first)
            idx = torch.argsort(eigenvalues, descending=True)  # Descending indices
            eigenvalues = eigenvalues[idx]  # Reorder eigenvalues
            eigenvectors = eigenvectors[:, idx]  # Reorder eigenvectors

            # Keep top-k
            self.eigenvalues_ = eigenvalues[:self.n_components]  # Top-k eigenvalues
            self.components_ = eigenvectors[:, :self.n_components].T  # (k, d)

            # Total variance for ratio computation
            total_var = torch.sum(eigenvalues).item()  # Sum of all eigenvalues

        elif self.method == "svd":
            # -- SVD approach --

            # Economy SVD: X_c = U @ diag(S) @ Vt
            U, S, Vt = torch.linalg.svd(X_c, full_matrices=False)  # Economy SVD

            # Convert singular values to eigenvalues
            eigenvalues = (S ** 2) / (n - 1)  # lambda_i = s_i^2 / (n-1)

            # SVD returns in descending order (no sorting needed)
            self.eigenvalues_ = eigenvalues[:self.n_components]  # Top-k
            self.components_ = Vt[:self.n_components]  # Top-k right singular vectors (k, d)

            total_var = torch.sum(eigenvalues).item()  # Total variance

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute explained variance ratio (as numpy for compatibility)
        self.explained_variance_ratio_ = (
            self.eigenvalues_.cpu().numpy() / total_var  # Proportion per component
        )

        logger.info("PCA(torch/%s) fitted  n_components=%d  explained_var=%.4f  device=%s",
                     self.method, self.n_components,
                     float(np.sum(self.explained_variance_ratio_)), device)
        return self

    def transform(self, X_np):
        """Project data onto principal components.

        Args:
            X_np: Input numpy array, shape (n, d).

        Returns:
            Reduced numpy array, shape (n, k).
        """
        X = torch.tensor(X_np, dtype=torch.float32, device=device)  # To tensor
        X_c = X - self.mean_  # Centre using training mean
        X_reduced = X_c @ self.components_.T  # Project to k-dim, shape (n, k)
        return X_reduced.cpu().numpy()  # Convert back to numpy

    def inverse_transform(self, X_reduced_np):
        """Reconstruct from reduced representation.

        Args:
            X_reduced_np: Reduced numpy array, shape (n, k).

        Returns:
            Reconstructed numpy array, shape (n, d).
        """
        X_r = torch.tensor(X_reduced_np, dtype=torch.float32, device=device)  # To tensor
        X_recon = X_r @ self.components_ + self.mean_  # Reconstruct and add mean
        return X_recon.cpu().numpy()  # Back to numpy

    def fit_transform(self, X_np):
        """Fit and transform in one step."""
        self.fit(X_np)
        return self.transform(X_np)


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
    pca = PCAPyTorch(
        n_components=hp.get("n_components", 5),
        method=hp.get("method", "eig"),
    )
    pca.fit(X_train)
    return pca


def _reconstruction_error(pca, X):
    """Mean squared reconstruction error."""
    X_reduced = pca.transform(X)
    X_recon = pca.inverse_transform(X_reduced)
    return float(np.mean((X - X_recon) ** 2))


def _downstream_accuracy(pca, X_train, y_train, X_eval, y_eval):
    """Downstream classifier accuracy after PCA."""
    X_train_r = pca.transform(X_train)
    X_eval_r = pca.transform(X_eval)
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
    """Compare PCA configurations: n_components x method."""
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: PCA (PyTorch) - Components x Method")
    print("=" * 80)
    print(f"\nDevice: {device}\n")

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

    print(f"\n{'=' * 90}")
    print(f"{'Config':<25} {'ExpVar':>8} {'ReconErr':>10} {'Accuracy':>10} {'Time':>8}")
    print("-" * 90)
    for name, m in results.items():
        print(f"{name:<25} {m['explained_variance']:>8.4f} "
              f"{m['reconstruction_error']:>10.6f} {m['accuracy']:>10.4f} "
              f"{m['train_time']:>8.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Both eig and svd give identical results on PyTorch.")
    print("  2. GPU acceleration shines for large datasets (>100K samples).")
    print("  3. For small datasets, CPU may be faster due to transfer overhead.")
    print("  4. torch.linalg.eigh/svd match numpy results to float32 precision.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Demographics
# ---------------------------------------------------------------------------

def real_world_demo():
    """PCA for customer demographics (PyTorch, GPU-ready)."""
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Demographics (PCA PyTorch)")
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

    X[:, 1] += 0.5 * X[:, 0]
    X[:, 4] += 0.3 * X[:, 1]
    X[:, 5] += 0.4 * X[:, 1]
    X[:, 18] += 0.6 * X[:, 1]
    X[:, 11] += 0.3 * X[:, 10]

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

    for method in ["eig", "svd"]:
        print(f"\n  Method: {method}")
        for n_comp in [2, 5, 10]:
            pca = PCAPyTorch(n_components=n_comp, method=method)
            pca.fit(X_train_s)

            recon_err = _reconstruction_error(pca, X_test_s)
            clf_m = _downstream_accuracy(pca, X_train_s, y_train, X_test_s, y_test)
            total_var = float(np.sum(pca.explained_variance_ratio_))

            print(f"    n={n_comp}: Var={total_var:.4f}  "
                  f"ReconErr={recon_err:.6f}  Acc={clf_m['accuracy']:.4f}")

    final_pca = PCAPyTorch(n_components=5, method="svd")
    final_pca.fit(X_train_s)
    final_metrics = _downstream_accuracy(final_pca, X_train_s, y_train, X_test_s, y_test)

    print(f"\nFinal model: 5 components via SVD on {device}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")

    return final_pca, final_metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "n_components": trial.suggest_int("n_components", 2, min(15, X_train.shape[1])),
        "method": trial.suggest_categorical("method", ["eig", "svd"]),
    }
    pca = train(X_train, y_train, **hp)
    return validate(pca, X_val, y_val, X_train, y_train)["accuracy"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    study = optuna.create_study(direction="maximize", study_name="pca_pytorch")
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
    pca = train(X_train, y_train, **config)
    m = validate(pca, X_val, y_val, X_train, y_train)
    tune.report(**m)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
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
    """Execute the full PCA (PyTorch) pipeline."""
    print("=" * 70)
    print("Principal Component Analysis (PCA) - PyTorch")
    print("=" * 70)
    print(f"Device: {device}")

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
