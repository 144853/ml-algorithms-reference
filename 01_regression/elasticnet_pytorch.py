"""
ElasticNet Regression - PyTorch Implementation
================================================

COMPLETE ML TUTORIAL: This file implements ElasticNet regression using PyTorch's
nn.Linear module with a custom combined L1+L2 penalty added to the MSE loss.
This demonstrates how to implement regularised linear models in a deep learning
framework, enabling GPU acceleration and integration with PyTorch ecosystems.

Theory & Mathematics:
    ElasticNet minimises the composite objective:

        L(w) = (1/n) * ||y - Xw - b||^2
             + alpha * l1_ratio * ||w||_1
             + alpha * (1 - l1_ratio) * (1/2) * ||w||_2^2

    In PyTorch, we implement this as:
        loss = MSE_loss + alpha * (l1_ratio * L1_norm + (1 - l1_ratio) * L2_norm)

    where L1_norm = sum(|w|) and L2_norm = sum(w^2).

    Unlike the NumPy coordinate descent approach, here we use gradient-based
    optimisation (SGD or Adam). The L1 norm is not differentiable at zero,
    but PyTorch's autograd computes subgradients automatically, and optimisers
    like Adam handle this well in practice.

    Why PyTorch for ElasticNet?
        - GPU acceleration for very large datasets (millions of samples)
        - Easy integration with neural network pipelines (e.g., as a regularised
          output layer in a deep network)
        - Automatic differentiation handles the L1 subgradient automatically
        - Familiar API for deep learning practitioners

    Trade-offs vs Coordinate Descent:
        - Coordinate descent (NumPy) converges to exact solution more reliably
        - Gradient descent may not produce exactly zero weights (only near-zero)
        - PyTorch approach is more flexible and extensible to non-linear models
        - GPU speedup only matters for very large datasets (>100K samples)

Hyperparameters:
    - alpha (float): Overall regularisation strength. Default 1.0.
    - l1_ratio (float): L1/L2 blend. 0=Ridge, 1=Lasso. Default 0.5.
    - lr (float): Learning rate for gradient descent. Default 0.01.
    - epochs (int): Number of training passes over the data. Default 1000.

Business Use Cases:
    - Genomics: Gene expression prediction with correlated gene groups.
    - Finance: Multi-factor models with correlated risk factors.
    - NLP: Text regression with correlated n-gram features.

Advantages:
    - GPU-accelerated for large datasets.
    - Integrates seamlessly with PyTorch pipelines.
    - Automatic differentiation handles L1 subgradient.
    - Flexible: easy to extend to non-linear variants.

Disadvantages:
    - Gradient descent may not produce exact zeros (unlike coordinate descent).
    - Requires learning rate tuning (additional hyperparameter).
    - Slower convergence than coordinate descent for small/medium datasets.
    - More code complexity than sklearn for a simple linear model.
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline progress tracking
import time  # Wall-clock timing for training speed comparisons
from functools import partial  # Binds fixed arguments for Optuna callbacks

# -- Third-party imports --
import numpy as np  # NumPy for data generation, metrics computation, and array ops
import optuna  # Bayesian hyperparameter optimisation with TPE sampler
import ray  # Distributed computing for parallel hyperparameter search
from ray import tune  # Ray's hyperparameter tuning module
import torch  # PyTorch core: tensors, autograd, GPU support
import torch.nn as nn  # Neural network modules (nn.Linear for linear regression)
from sklearn.datasets import make_regression  # Synthetic regression data generation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Metrics
from sklearn.model_selection import train_test_split  # Reproducible data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure module-level logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Module-scoped logger

# -- Device selection: use GPU if available, otherwise CPU --
# WHY: PyTorch can leverage CUDA GPUs for parallel tensor operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-detect GPU


# ---------------------------------------------------------------------------
# ElasticNet Model (PyTorch)
# ---------------------------------------------------------------------------

class ElasticNetPyTorch(nn.Module):
    """ElasticNet regression implemented as a PyTorch nn.Module.

    Uses nn.Linear for the linear transformation (y = Xw + b) and adds
    a custom combined L1+L2 penalty to the MSE loss during training.

    The model is a single linear layer with no activation function,
    making it mathematically equivalent to regularised linear regression.
    """

    def __init__(self, n_features, alpha=1.0, l1_ratio=0.5):
        """Initialise the ElasticNet model.

        Args:
            n_features: Number of input features (determines weight vector size).
            alpha: Overall regularisation strength.
            l1_ratio: Blend between L1 (sparsity) and L2 (shrinkage).
        """
        super().__init__()  # Initialise nn.Module base class

        # nn.Linear implements y = x @ W^T + b (a single linear layer)
        # WHY nn.Linear: provides learnable weights and bias with autograd support
        self.linear = nn.Linear(n_features, 1)  # 1 output for regression

        # Store regularisation parameters as instance attributes
        self.alpha = alpha  # Overall penalty strength
        self.l1_ratio = l1_ratio  # L1/L2 blend (0=Ridge, 1=Lasso)

    def forward(self, x):
        """Forward pass: compute linear prediction.

        Args:
            x: Input tensor, shape (batch_size, n_features).

        Returns:
            Predictions tensor, shape (batch_size, 1).
        """
        # Apply the linear transformation: y_hat = x @ W^T + b
        return self.linear(x)  # Shape: (batch_size, 1)

    def elastic_penalty(self):
        """Compute the combined L1+L2 penalty on the weights.

        penalty = alpha * (l1_ratio * L1 + (1 - l1_ratio) * L2)

        where L1 = sum(|w|) and L2 = sum(w^2).

        Returns:
            Scalar tensor representing the total ElasticNet penalty.
        """
        # Get the weight tensor from the linear layer (excludes bias)
        weights = self.linear.weight  # Shape: (1, n_features)

        # Compute L1 norm: sum of absolute values (induces sparsity)
        l1_term = torch.sum(torch.abs(weights))  # Scalar: total L1 penalty

        # Compute L2 norm squared: sum of squared values (induces shrinkage)
        l2_term = torch.sum(weights ** 2)  # Scalar: total L2 penalty

        # Combine L1 and L2 with the blend ratio and overall strength
        penalty = self.alpha * (
            self.l1_ratio * l1_term  # L1 component (sparsity)
            + (1.0 - self.l1_ratio) * l2_term  # L2 component (shrinkage)
        )
        return penalty  # Scalar tensor: total regularisation penalty

    @property
    def n_nonzero(self):
        """Count features with non-negligible weights (above threshold).

        WHY threshold 1e-6: gradient descent does not produce exact zeros
        like coordinate descent. Weights below this threshold are effectively zero.
        """
        # Detach from computation graph, move to CPU, convert to numpy
        w = self.linear.weight.detach().cpu().numpy().flatten()  # Shape: (n_features,)
        return int(np.sum(np.abs(w) > 1e-6))  # Count weights above threshold


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=20, noise=0.1, random_state=42):
    """Generate synthetic regression data with 60/20/20 split and scaling.

    Returns numpy arrays (not tensors) for compatibility with metrics functions.
    Tensors are created inside train() for GPU transfer.
    """
    # Generate synthetic data with half the features being informative
    X, y = make_regression(
        n_samples=n_samples,  # Total number of data points
        n_features=n_features,  # Feature dimensionality
        n_informative=max(1, n_features // 2),  # Half are predictive
        noise=noise,  # Gaussian noise on target
        random_state=random_state,  # Seed for reproducibility
    )

    # Two-stage split for exact 60/20/20 proportions
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state,  # 60% train
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,  # 20/20 val/test
    )

    # Standardise features: fit on train only to prevent leakage
    scaler = StandardScaler()  # Zero-mean, unit-variance normalisation
    X_train = scaler.fit_transform(X_train)  # Learn and apply to train
    X_val = scaler.transform(X_val)  # Apply train stats to val
    X_test = scaler.transform(X_test)  # Apply train stats to test

    # Log dataset summary
    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train ElasticNet model using PyTorch gradient descent with L1+L2 penalty.

    The training loop:
        1. Forward pass: compute predictions
        2. Compute loss = MSE + ElasticNet penalty
        3. Backward pass: compute gradients via autograd
        4. Update weights using the chosen optimiser
    """
    # Extract hyperparameters with defaults
    alpha = hp.get("alpha", 1.0)  # Overall regularisation strength
    l1_ratio = hp.get("l1_ratio", 0.5)  # L1/L2 blend
    lr = hp.get("lr", 0.01)  # Learning rate for gradient descent
    epochs = hp.get("epochs", 1000)  # Number of training epochs
    n_features = X_train.shape[1]  # Number of input features

    # Create the model and move to device (CPU or GPU)
    model = ElasticNetPyTorch(n_features, alpha=alpha, l1_ratio=l1_ratio)
    model = model.to(device)  # Transfer model parameters to GPU if available

    # Convert numpy arrays to PyTorch tensors on the target device
    # WHY float32: standard precision for neural network training
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)  # Features
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)  # Target (n,1)

    # Define the base loss function (MSE without regularisation)
    mse_loss = nn.MSELoss()  # Mean squared error loss

    # Choose optimiser: Adam adapts learning rate per-parameter
    # WHY Adam: handles L1 subgradients well, adapts to loss landscape
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)  # Adaptive learning rate

    # Training loop
    model.train()  # Set model to training mode (enables dropout/batchnorm if present)
    for epoch in range(epochs):  # Each epoch = one pass through the data
        # Forward pass: compute predictions
        y_pred = model(X_t)  # Shape: (n, 1) -- linear predictions

        # Compute total loss = MSE + ElasticNet penalty
        base_loss = mse_loss(y_pred, y_t)  # Data fidelity term
        penalty = model.elastic_penalty()  # Regularisation penalty
        total_loss = base_loss + penalty  # Combined objective

        # Backward pass: compute gradients of total_loss w.r.t. all parameters
        optimiser.zero_grad()  # Clear accumulated gradients from previous step
        total_loss.backward()  # Backpropagate to compute new gradients
        optimiser.step()  # Update parameters using Adam rule

    # Log training summary
    logger.info("ElasticNet(pytorch) trained  alpha=%.4f  l1_ratio=%.2f  epochs=%d  nonzero=%d",
                alpha, l1_ratio, epochs, model.n_nonzero)
    return model  # Return the trained model


def _metrics(y_true, y_pred):
    """Compute standard regression metrics from numpy arrays."""
    return {
        "mse": mean_squared_error(y_true, y_pred),  # Mean squared error
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),  # Root MSE
        "mae": mean_absolute_error(y_true, y_pred),  # Mean absolute error
        "r2": r2_score(y_true, y_pred),  # R-squared (variance explained)
    }


def _predict_numpy(model, X):
    """Generate predictions and convert from tensor to numpy array."""
    model.eval()  # Set model to evaluation mode (disables dropout etc.)
    with torch.no_grad():  # Disable gradient computation for inference speed
        X_t = torch.tensor(X, dtype=torch.float32, device=device)  # Convert to tensor
        y_pred = model(X_t).cpu().numpy().flatten()  # Predict, move to CPU, flatten
    return y_pred  # Return numpy array of predictions


def validate(model, X_val, y_val):
    """Evaluate model on validation set."""
    y_pred = _predict_numpy(model, X_val)  # Generate predictions
    m = _metrics(y_val, y_pred)  # Compute all regression metrics
    logger.info("Validation: %s", m)  # Log for monitoring
    return m  # Return metrics dict


def test(model, X_test, y_test):
    """Evaluate model on held-out test set."""
    y_pred = _predict_numpy(model, X_test)  # Generate test predictions
    m = _metrics(y_test, y_pred)  # Compute final metrics
    logger.info("Test: %s", m)  # Log results
    return m  # Return metrics dict


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare ElasticNet across alpha and l1_ratio to show L1 vs L2 dominance.

    Tests a 3x3 grid:
        - alpha in {0.1, 1.0, 10.0}: light to heavy regularisation
        - l1_ratio in {0.1, 0.5, 0.9}: Ridge-like to Lasso-like
    """
    print("\n" + "=" * 80)  # Visual separator
    print("PARAMETER COMPARISON: ElasticNet (PyTorch) - Alpha x L1 Ratio Grid")
    print("=" * 80)
    print("\nShows how alpha and l1_ratio control sparsity and performance.\n")

    n_features = X_train.shape[1]  # Total features for sparsity reporting

    # Define the parameter grid
    alphas = [0.1, 1.0, 10.0]  # Regularisation strengths
    l1_ratios = [0.1, 0.5, 0.9]  # L1/L2 blend values

    results = {}  # Store results for all configurations

    for alpha in alphas:  # Iterate over regularisation strengths
        for l1_ratio in l1_ratios:  # Iterate over blend values
            name = f"alpha={alpha}, l1_ratio={l1_ratio}"  # Config name

            start_time = time.time()  # Start timing
            # Train with 2000 epochs for adequate convergence
            model = train(X_train, y_train, alpha=alpha, l1_ratio=l1_ratio, epochs=2000)
            train_time = time.time() - start_time  # Training duration

            metrics = validate(model, X_val, y_val)  # Evaluate on validation
            metrics["n_nonzero"] = model.n_nonzero  # Track sparsity
            metrics["train_time"] = train_time  # Track speed
            results[name] = metrics  # Store results

            print(f"  {name}: NonZero={model.n_nonzero}/{n_features}  "
                  f"MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}")

    # Summary table
    print(f"\n{'=' * 85}")
    print(f"{'Configuration':<35} {'Alpha':>6} {'L1R':>5} "
          f"{'NZ':>4} {'MSE':>10} {'R2':>10}")
    print("-" * 85)

    for name, m in results.items():
        parts = name.split(", ")
        alpha_val = float(parts[0].split("=")[1])
        l1r_val = float(parts[1].split("=")[1])
        print(f"{name:<35} {alpha_val:>6.1f} {l1r_val:>5.1f} "
              f"{m['n_nonzero']:>4} {m['mse']:>10.4f} {m['r2']:>10.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. l1_ratio=0.1 (L2 dominant): keeps most features, just shrinks weights.")
    print("  2. l1_ratio=0.9 (L1 dominant): more aggressive weight reduction.")
    print("  3. Note: PyTorch gradient descent produces near-zero, not exact-zero weights.")
    print("  4. For exact sparsity, use the NumPy coordinate descent implementation.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Gene Expression Analysis
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate ElasticNet for gene expression analysis with correlated features.

    Same scenario as the NumPy version: predicting clinical outcomes from
    correlated gene expression data, demonstrating the grouping effect.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Gene Expression Analysis (ElasticNet PyTorch)")
    print("=" * 80)

    np.random.seed(42)  # Reproducible data

    # Dataset parameters
    n_patients = 500  # Patient cohort size
    n_genes = 100  # Genes measured
    n_pathways = 5  # Correlated gene groups
    genes_per_pathway = 4  # Genes per pathway
    n_informative = n_pathways * genes_per_pathway  # 20 truly predictive genes

    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]  # Gene labels

    # Generate base expression matrix
    X = np.random.randn(n_patients, n_genes)  # Independent expression levels

    # Add within-pathway correlations
    for p in range(n_pathways):  # For each biological pathway
        start_idx = p * genes_per_pathway
        end_idx = start_idx + genes_per_pathway
        shared_signal = np.random.randn(n_patients) * 2.0  # Shared co-regulation signal
        for g in range(start_idx, end_idx):
            X[:, g] += shared_signal  # Add pathway-level correlation

    # True coefficients: only pathway genes matter
    true_coefs = np.zeros(n_genes)
    for p in range(n_pathways):
        start_idx = p * genes_per_pathway
        end_idx = start_idx + genes_per_pathway
        base_effect = np.random.uniform(2, 8) * np.random.choice([-1, 1])
        for g in range(start_idx, end_idx):
            true_coefs[g] = base_effect + np.random.normal(0, 0.5)

    # Generate clinical outcome
    noise = np.random.normal(0, 3, n_patients)
    y = X @ true_coefs + noise

    print(f"\nDataset: {n_patients} patients, {n_genes} genes")
    print(f"Ground truth: {n_informative} genes in {n_pathways} correlated pathways")

    # Split and scale
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train ElasticNet (balanced blend)
    en_model = train(X_train_s, y_train, alpha=0.5, l1_ratio=0.5, epochs=3000, lr=0.01)
    en_metrics = _metrics(y_test, _predict_numpy(en_model, X_test_s))
    en_selected = en_model.n_nonzero

    # Train Lasso-like (high l1_ratio)
    lasso_model = train(X_train_s, y_train, alpha=0.5, l1_ratio=0.99, epochs=3000, lr=0.01)
    lasso_metrics = _metrics(y_test, _predict_numpy(lasso_model, X_test_s))
    lasso_selected = lasso_model.n_nonzero

    print(f"\n--- ElasticNet vs Lasso-like Comparison ---")
    print(f"{'Metric':<20} {'ElasticNet':>12} {'Lasso-like':>12}")
    print("-" * 48)
    print(f"{'Genes selected':<20} {en_selected:>12} {lasso_selected:>12}")
    print(f"{'RMSE':<20} {en_metrics['rmse']:>12.4f} {lasso_metrics['rmse']:>12.4f}")
    print(f"{'R2':<20} {en_metrics['r2']:>12.4f} {lasso_metrics['r2']:>12.4f}")

    # Pathway coverage analysis
    print(f"\n--- Pathway Coverage ---")
    print(f"{'Pathway':<12} {'True':>6} {'EN':>6} {'Lasso':>6}")
    print("-" * 35)

    en_w = en_model.linear.weight.detach().cpu().numpy().flatten()  # EN weights as numpy
    lasso_w = lasso_model.linear.weight.detach().cpu().numpy().flatten()  # Lasso weights

    for p in range(n_pathways):
        start_idx = p * genes_per_pathway
        end_idx = start_idx + genes_per_pathway
        en_count = int(np.sum(np.abs(en_w[start_idx:end_idx]) > 1e-6))
        lasso_count = int(np.sum(np.abs(lasso_w[start_idx:end_idx]) > 1e-6))
        print(f"Pathway {p + 1:<4} {genes_per_pathway:>6} {en_count:>6} {lasso_count:>6}")

    print(f"\nNote: PyTorch gradient descent produces near-zero weights, not exact zeros.")
    print(f"For exact sparsity, use the NumPy coordinate descent implementation.")

    return en_model, en_metrics


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimisation
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: minimise validation MSE."""
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
        "epochs": trial.suggest_int("epochs", 500, 3000, step=500),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Run Optuna Bayesian optimisation."""
    study = optuna.create_study(direction="minimize", study_name="elasticnet_pytorch")
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
    """Ray Tune trainable function."""
    model = train(X_train, y_train, **config)
    metrics = validate(model, X_val, y_val)
    tune.report(**metrics)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    """Run Ray Tune parallel hyperparameter search."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "alpha": tune.loguniform(1e-5, 10.0),
        "l1_ratio": tune.uniform(0.01, 0.99),
        "lr": tune.loguniform(1e-4, 0.1),
        "epochs": tune.choice([500, 1000, 2000, 3000]),
    }

    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    tuner = tune.Tuner(
        trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="mse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full ElasticNet (PyTorch) pipeline."""
    print("=" * 70)
    print("ElasticNet Regression - PyTorch (nn.Linear + L1+L2 Penalty)")
    print("=" * 70)
    print(f"Device: {device}")

    # Generate data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Parameter comparison
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Real-world demo
    real_world_demo()

    # Optuna
    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=25)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    # Ray Tune
    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    # Final evaluation
    model = train(X_train, y_train, **study.best_trial.params)
    print(f"\nNon-zero coefficients: {model.n_nonzero}/{X_train.shape[1]}")

    print("\n--- Validation ---")
    for k, v in validate(model, X_val, y_val).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n--- Test ---")
    for k, v in test(model, X_test, y_test).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
