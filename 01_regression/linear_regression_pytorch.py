"""
Linear Regression - PyTorch Implementation
===========================================

COMPLETE ML TUTORIAL: This file teaches linear regression using PyTorch,
demonstrating how the same simple model (y = Xw + b) is expressed in a
deep learning framework. This is the foundation for understanding how
neural networks work -- a linear layer is the simplest possible neural network.

Theory & Mathematics:
    Linear Regression fits the model:

        y = X @ w + b

    by minimising Mean Squared Error:

        L(w, b) = (1/n) * ||y - (Xw + b)||^2

    In this PyTorch implementation the model is expressed as a single
    ``nn.Linear`` layer.  An ``optim.SGD`` (or Adam) optimiser performs
    gradient-based parameter updates.  Mini-batch training via
    ``DataLoader`` allows scaling to large datasets.

    Gradient update rule (SGD):
        w := w - lr * dL/dw
        b := b - lr * dL/db

    PyTorch's autograd computes the gradients automatically.

Business Use Cases:
    - Real-time prediction services where PyTorch already powers the stack
    - GPU-accelerated regression on very large datasets
    - Building block inside larger neural-network architectures
    - Transfer learning pipelines that start with a linear head

Advantages:
    - Seamless GPU acceleration for large-scale data
    - Native integration with PyTorch ecosystem (DataLoader, schedulers)
    - Easy to extend to non-linear models by stacking layers
    - Automatic differentiation eliminates manual gradient derivation

Disadvantages:
    - Heavyweight dependency for a simple linear model
    - Requires tuning learning rate, batch size, and epochs
    - Mini-batch SGD introduces noise compared to closed-form OLS
    - Slower than scikit-learn / NumPy for small datasets

Hyperparameters:
    - lr (float): Learning rate. Default 0.01.
    - epochs (int): Number of training epochs. Default 100.
    - batch_size (int): Mini-batch size. Default 32.
    - optimizer (str): "sgd" or "adam". Default "adam".
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# logging: Structured diagnostic output for tracking training progress.
import logging

# time: Wall-clock measurement for benchmarking different configurations.
import time

# functools.partial: Pre-fills function arguments for Optuna compatibility.
from functools import partial

# numpy: Used for data manipulation and metric computation.
# WHY numpy alongside PyTorch: sklearn metrics expect numpy arrays, and
# data generation uses numpy. PyTorch tensors are for GPU computation.
import numpy as np

# optuna: Bayesian hyperparameter optimisation.
import optuna

# ray: Distributed hyperparameter search framework.
import ray

# torch: The core PyTorch library for tensor operations and autograd.
# WHY PyTorch for linear regression: Overkill for this model alone, but
# learning the PyTorch training loop pattern transfers directly to
# complex neural networks. Consider this "training wheels" for deep learning.
import torch

# torch.nn: Neural network building blocks (layers, loss functions).
# nn.Linear is the fundamental building block: y = Wx + b
# nn.MSELoss computes mean squared error loss.
import torch.nn as nn

# ray.tune: Distributed hyperparameter search.
from ray import tune

# sklearn utilities for data generation, evaluation, and preprocessing.
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DataLoader: Handles mini-batch iteration, shuffling, and parallel data loading.
# TensorDataset: Wraps numpy arrays as PyTorch tensors for DataLoader compatibility.
# WHY DataLoader: Efficient memory usage (loads one batch at a time) and
# automatic shuffling (prevents the model from memorising sample order).
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DEVICE: Automatically selects GPU if available, otherwise CPU.
# WHY: GPU can accelerate matrix operations 10-100x for large data.
# For small datasets, CPU is often faster (GPU has transfer overhead).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# PyTorch Model
# ---------------------------------------------------------------------------

class LinearRegressionModel(nn.Module):
    """Single-layer linear regression model in PyTorch.

    This is the simplest possible neural network: one linear layer
    with n_features inputs and 1 output. Mathematically identical
    to ordinary linear regression: y = Xw + b.

    WHY nn.Module: PyTorch's module system provides:
    - Automatic parameter tracking (model.parameters())
    - Integration with optimisers (they iterate over parameters)
    - Save/load functionality (torch.save/load)
    - GPU transfer (model.to(device))
    """

    def __init__(self, n_features):
        # super().__init__() initialises the nn.Module base class.
        # WHY required: Sets up the parameter registry and hook system
        # that PyTorch uses internally for autograd and serialisation.
        super().__init__()

        # nn.Linear(in_features, out_features): Creates a linear transformation y = Wx + b.
        # Parameters: weight matrix W of shape (1, n_features) and bias vector b of shape (1,).
        # WHY (n_features, 1): n_features inputs, 1 output (regression predicts a single value).
        # nn.Linear automatically initialises weights using Kaiming uniform distribution.
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        """Forward pass: compute predictions from input features.

        Args:
            x: Input tensor of shape (batch_size, n_features).

        Returns:
            Predictions of shape (batch_size,).
        """
        # self.linear(x) computes x @ W.T + b, returning shape (batch_size, 1).
        # .squeeze(-1) removes the last dimension: (batch_size, 1) -> (batch_size,).
        # WHY squeeze: Our target y is 1D (batch_size,), so predictions must match.
        # Without squeeze, the loss function would broadcast incorrectly.
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data and split into train/val/test.

    Same methodology as the NumPy and sklearn versions: 60/20/20 split
    with StandardScaler fitted only on training data.
    """
    # Generate linear regression data with known ground truth.
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(1, n_features // 2), noise=noise,
        random_state=random_state,
    )

    # Three-way split: train (60%), validation (20%), test (20%).
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # Feature scaling is CRITICAL for gradient-based training.
    # WHY: Without scaling, features with large magnitudes dominate the
    # gradient updates. A feature in range [0, 1000000] would make the
    # loss landscape extremely elongated in that dimension, causing
    # zigzagging convergence and requiring tiny learning rates.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


def _to_loader(X, y, batch_size=32, shuffle=True):
    """Convert numpy arrays to a PyTorch DataLoader.

    WHY DataLoader: Instead of feeding the entire dataset at once, DataLoader
    yields mini-batches. This is essential for:
    1. Memory efficiency: Only one batch is in memory at a time
    2. Better generalisation: Mini-batch noise acts as implicit regularisation
    3. GPU utilisation: Batches can be transferred to GPU while the next is prepared

    Args:
        X, y: Numpy arrays of features and targets.
        batch_size: Number of samples per mini-batch.
        shuffle: Whether to randomise order each epoch.
            WHY True for training: Prevents the model from learning order-dependent
            patterns. WHY False for evaluation: Ensures reproducible metrics.
    """
    # Convert numpy arrays to PyTorch tensors.
    # dtype=torch.float32: Neural networks use 32-bit floats by default.
    # WHY not float64: 32-bit is 2x faster on GPU and sufficient precision
    # for most ML tasks. float64 is rarely needed.
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train the PyTorch linear regression model.

    This function demonstrates the standard PyTorch training loop:
    1. Create model
    2. Create loss function and optimiser
    3. For each epoch, for each batch:
       a. Forward pass (compute predictions)
       b. Compute loss
       c. Backward pass (compute gradients via autograd)
       d. Update parameters (optimiser step)
       e. Zero gradients (prevent accumulation)

    Args:
        X_train, y_train: Training data as numpy arrays.
        **hp: Hyperparameters.

    Returns:
        Trained PyTorch model.
    """
    # Extract hyperparameters with sensible defaults.
    lr = hp.get("lr", 0.01)             # Learning rate: step size for parameter updates
    epochs = hp.get("epochs", 100)       # Number of full passes through the training data
    batch_size = hp.get("batch_size", 32) # Samples per mini-batch
    opt_name = hp.get("optimizer", "adam") # Optimiser algorithm

    # Get number of features from the data shape.
    n_features = X_train.shape[1]

    # Instantiate the model and move it to the compute device (CPU or GPU).
    # .to(DEVICE): Transfers model parameters to GPU memory if available.
    # WHY move model first: All tensors in computation must be on the same device.
    model = LinearRegressionModel(n_features).to(DEVICE)

    # MSELoss: Computes (1/n) * sum((y_pred - y_true)^2).
    # WHY MSE: Standard loss for regression. It penalises large errors
    # quadratically, encouraging the model to avoid big mistakes.
    criterion = nn.MSELoss()

    # Choose optimiser based on configuration.
    if opt_name == "sgd":
        # SGD: Stochastic Gradient Descent. Simple and well-understood.
        # WHY use SGD: Good baseline, easy to reason about. With proper
        # learning rate, converges reliably for convex problems (like linear regression).
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        # Adam: Adaptive Moment Estimation. Maintains per-parameter learning rates.
        # WHY Adam is default: Adapts learning rate for each parameter based on
        # gradient history. More robust to learning rate choice than SGD.
        # Less sensitive to initial lr, making it easier for beginners.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create DataLoader for mini-batch iteration.
    loader = _to_loader(X_train, y_train, batch_size=batch_size)

    # Set model to training mode.
    # WHY: Enables training-specific behaviours like dropout and batch norm.
    # For this simple linear model it has no effect, but it is good practice
    # to always call .train() before training and .eval() before evaluation.
    model.train()

    # Main training loop: iterate over the dataset `epochs` times.
    for epoch in range(epochs):
        # Track cumulative loss for this epoch.
        epoch_loss = 0.0

        # Iterate over mini-batches.
        for X_batch, y_batch in loader:
            # Move batch tensors to the same device as the model.
            # WHY: PyTorch requires all tensors in a computation to be
            # on the same device. This transfers data from CPU to GPU if needed.
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            # STEP 1: Zero out accumulated gradients from the previous step.
            # WHY: PyTorch ACCUMULATES gradients by default (useful for
            # gradient accumulation with small batches). If we do not zero
            # them, gradients from all previous batches would be summed,
            # making the updates incorrect.
            optimizer.zero_grad()

            # STEP 2: Forward pass -- compute model predictions.
            preds = model(X_batch)

            # STEP 3: Compute loss.
            loss = criterion(preds, y_batch)

            # STEP 4: Backward pass -- compute gradients via autograd.
            # WHY autograd: PyTorch builds a computational graph during the
            # forward pass and uses reverse-mode automatic differentiation
            # to compute dL/dw and dL/db. This is exact (not numerical
            # approximation) and efficient (O(1) per parameter).
            loss.backward()

            # STEP 5: Update parameters using the computed gradients.
            # For SGD: w := w - lr * dL/dw
            # For Adam: uses adaptive per-parameter learning rates.
            optimizer.step()

            # Accumulate loss weighted by batch size for accurate epoch average.
            # WHY * X_batch.size(0): The last batch may be smaller than batch_size.
            # Weighting by actual batch size ensures correct average.
            epoch_loss += loss.item() * X_batch.size(0)

        # Compute average loss across all samples in this epoch.
        epoch_loss /= len(loader.dataset)

        # Log progress at regular intervals to avoid console spam.
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.debug("Epoch %d/%d  loss=%.6f", epoch + 1, epochs, epoch_loss)

    logger.info("Training complete  final_loss=%.6f", epoch_loss)
    return model


def _evaluate(model, X, y):
    """Evaluate model on a dataset without gradient computation.

    WHY separate function: Avoids code duplication between validate() and test().
    """
    # Set model to evaluation mode. Disables dropout and switches batch norm
    # to use running statistics instead of batch statistics.
    model.eval()

    # torch.no_grad(): Disables gradient tracking during evaluation.
    # WHY: 1. Saves memory (no computational graph stored)
    #       2. Speeds up computation (~20% faster)
    #       3. Semantically correct (we are not training)
    with torch.no_grad():
        # Convert numpy array to tensor, send to device, get predictions.
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        # .cpu().numpy(): Move predictions back to CPU and convert to numpy
        # for sklearn metrics compatibility.
        preds = model(X_t).cpu().numpy()

    return {
        "mse": mean_squared_error(y, preds),
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
    }


def validate(model, X_val, y_val):
    """Evaluate on validation set (used for hyperparameter tuning)."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation: %s", metrics)
    return metrics


def test(model, X_test, y_test):
    """Evaluate on test set (used once for final performance estimate)."""
    metrics = _evaluate(model, X_test, y_test)
    logger.info("Test: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare different optimizer/learning rate/batch size configurations.

    This function demonstrates how PyTorch training hyperparameters affect
    convergence speed, final accuracy, and training time for linear regression.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Optimizer, Learning Rate & Batch Size Trade-offs")
    print("=" * 80)

    configs = {
        "SGD, Low LR (Conservative)": {
            "lr": 0.001,
            "epochs": 200,
            "batch_size": 32,
            "optimizer": "sgd",
            # WHY: Small learning rate with SGD converges slowly but reliably.
            # TRADE-OFF: Safe but requires many epochs. Good for unstable data.
        },
        "SGD, High LR (Aggressive)": {
            "lr": 0.1,
            "epochs": 50,
            "batch_size": 32,
            "optimizer": "sgd",
            # WHY: Large learning rate with SGD converges fast on well-scaled data.
            # TRADE-OFF: May oscillate or diverge if data is noisy. Risky but fast.
        },
        "Adam, Default (Balanced)": {
            "lr": 0.01,
            "epochs": 100,
            "batch_size": 32,
            "optimizer": "adam",
            # WHY: Adam with moderate lr is the most common default in practice.
            # TRADE-OFF: Slightly more compute per step than SGD (maintains
            # moving averages), but much less sensitive to lr choice.
        },
        "Adam, Large Batch (Efficient)": {
            "lr": 0.01,
            "epochs": 100,
            "batch_size": 128,
            "optimizer": "adam",
            # WHY: Larger batches give more stable gradient estimates.
            # TRADE-OFF: Fewer parameter updates per epoch. May need more epochs
            # or higher lr to compensate. Uses more memory per step.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        metrics["train_time"] = train_time
        results[name] = metrics

        print(f"\n  {name}:")
        print(f"    MSE: {metrics['mse']:.6f}  R2: {metrics['r2']:.6f}  Time: {train_time:.3f}s")

    # Summary table
    print(f"\n{'=' * 85}")
    print(f"{'Configuration':<35} {'MSE':>10} {'R2':>10} {'Time(s)':>10}")
    print("-" * 85)
    for name, metrics in results.items():
        print(f"{name:<35} {metrics['mse']:>10.4f} {metrics['r2']:>10.4f} {metrics['train_time']:>10.3f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Adam is more forgiving of learning rate choices than SGD.")
    print("  2. Large batch sizes give smoother but potentially slower convergence.")
    print("  3. SGD with high LR is risky -- check for divergence in the loss curve.")
    print("  4. For linear regression, all configs should reach similar final MSE.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Used Car Price Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate PyTorch linear regression on used car price prediction.

    DOMAIN CONTEXT: Used car pricing is a common real-world regression task.
    Platforms like AutoTrader, Carvana, and KBB use ML models to estimate
    fair market values. Linear regression provides a good baseline.

    WHY used cars: Features like mileage, age, and engine size have
    approximately linear relationships with price (before feature engineering),
    making linear regression reasonable as a starting model.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Used Car Price Prediction (PyTorch)")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 2000

    # Simulate realistic used car features.
    # Mileage: Exponentially distributed (most cars have moderate mileage).
    mileage = np.random.exponential(40000, n_samples).clip(1000, 200000)

    # Age in years: 0-20 years, uniformly distributed.
    age_years = np.random.uniform(0, 20, n_samples)

    # Engine size in litres: 1.0 to 6.0L, normally distributed around 2.5L.
    engine_litres = np.random.normal(2.5, 0.8, n_samples).clip(1.0, 6.0)

    # Horsepower: Correlated with engine size (larger engines = more HP).
    horsepower = engine_litres * 60 + np.random.normal(0, 30, n_samples)
    horsepower = horsepower.clip(80, 500)

    # Number of previous owners: 1-5.
    prev_owners = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.3, 0.35, 0.2, 0.1, 0.05])

    # Fuel efficiency (MPG): inversely related to engine size.
    mpg = 45 - engine_litres * 5 + np.random.normal(0, 3, n_samples)
    mpg = mpg.clip(10, 60)

    feature_names = ["mileage", "age_years", "engine_litres", "horsepower", "prev_owners", "mpg"]
    X = np.column_stack([mileage, age_years, engine_litres, horsepower, prev_owners, mpg])

    # True pricing model.
    true_coefs = np.array([
        -0.08,      # -$0.08 per mile (higher mileage = lower price)
        -1500.0,    # -$1500 per year of age
        5000.0,     # +$5000 per litre of engine size
        50.0,       # +$50 per horsepower
        -2000.0,    # -$2000 per previous owner
        200.0,      # +$200 per MPG (fuel-efficient cars are valued)
    ])

    base_price = 35000
    noise = np.random.normal(0, 3000, n_samples)
    y = base_price + X @ true_coefs + noise
    y = y.clip(1000, None)  # Prices cannot be negative

    print(f"\nDataset: {n_samples} used cars with {len(feature_names)} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Mean price: ${y.mean():,.0f}")

    # Standard ML pipeline.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train with PyTorch.
    model = train(X_train_s, y_train, lr=0.01, epochs=200, batch_size=64, optimizer="adam")

    # Evaluate.
    test_metrics = _evaluate(model, X_test_s, y_test)

    print(f"\n--- Model Performance ---")
    print(f"  RMSE: ${test_metrics['rmse']:,.2f}  (average pricing error)")
    print(f"  MAE:  ${test_metrics['mae']:,.2f}")
    print(f"  R2:   {test_metrics['r2']:.4f}")

    # Show learned weights.
    learned_w = model.linear.weight.detach().cpu().numpy().flatten()
    learned_b = model.linear.bias.detach().cpu().numpy().item()

    print(f"\n--- Learned Feature Weights (Standardised Scale) ---")
    print(f"{'Feature':<15} {'Weight':>10}")
    print("-" * 28)
    for i, name in enumerate(feature_names):
        print(f"{name:<15} {learned_w[i]:>10.2f}")
    print(f"{'Intercept':<15} {learned_b:>10.2f}")

    # Sample predictions.
    print(f"\n--- Sample Predictions ---")
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_s[:8], dtype=torch.float32).to(DEVICE)).cpu().numpy()
    print(f"{'Actual':>12} {'Predicted':>12} {'Error':>10}")
    print("-" * 36)
    for i in range(8):
        print(f"${y_test[i]:>10,.0f} ${preds[i]:>10,.0f} ${preds[i]-y_test[i]:>8,.0f}")

    return model, test_metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: find best lr, epochs, batch_size, optimizer."""
    hp = {
        "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
        "epochs": trial.suggest_int("epochs", 50, 300, step=50),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam"]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    """Run Optuna study for PyTorch linear regression."""
    study = optuna.create_study(direction="minimize", study_name="linreg_pytorch")
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
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


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10):
    """Launch Ray Tune distributed hyperparameter search."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "lr": tune.loguniform(1e-4, 0.1),
        "epochs": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([16, 32, 64]),
        "optimizer": tune.choice(["sgd", "adam"]),
    }
    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    )
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="mse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete PyTorch linear regression tutorial pipeline."""
    print("=" * 70)
    print("Linear Regression - PyTorch")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Educational comparisons.
    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    # Hyperparameter search.
    print("\n--- Optuna Hyperparameter Optimisation ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=15)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune Hyperparameter Search ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=8)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    # Final model.
    best_p = study.best_trial.params
    model = train(X_train, y_train, **best_p)

    print("\n--- Validation Results ---")
    for k, v in validate(model, X_val, y_val).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n--- Test Results ---")
    for k, v in test(model, X_test, y_test).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
