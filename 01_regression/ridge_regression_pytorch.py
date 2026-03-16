"""
Ridge Regression - PyTorch Implementation (L2 Regularisation)
=============================================================

COMPLETE ML TUTORIAL: This file implements Ridge regression in PyTorch,
demonstrating two equivalent approaches to L2 regularisation:
1. Using the optimizer's weight_decay parameter (implicit)
2. Explicitly computing the L2 penalty and adding it to the loss

Theory & Mathematics:
    Ridge Regression minimises the L2-regularised MSE loss:

        L(w, b) = (1/n)||y - (Xw + b)||^2  +  alpha * ||w||^2

    In PyTorch, L2 regularisation is most commonly implemented via the
    ``weight_decay`` parameter in optimisers such as ``Adam`` or ``SGD``.
    Setting ``weight_decay = 2 * alpha`` in the optimiser is mathematically
    equivalent to adding alpha * ||w||^2 to the loss.

Hyperparameters:
    - alpha (float): L2 regularisation strength. Default 0.01.
    - lr (float): Learning rate. Default 0.01.
    - epochs (int): Number of training epochs. Default 100.
    - batch_size (int): Mini-batch size. Default 32.
    - optimizer (str): "sgd" or "adam". Default "adam".
    - use_weight_decay (bool): Use optimizer weight_decay vs explicit penalty.
"""

import logging
import time
from functools import partial

import numpy as np
import optuna
import ray
import torch
import torch.nn as nn
from ray import tune
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Select GPU if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RidgeModel(nn.Module):
    """Single linear layer for Ridge regression.

    Identical architecture to linear regression -- the "Ridge" part is
    implemented via weight_decay in the optimiser OR an explicit L2 penalty
    in the loss function, NOT in the model architecture.
    """

    def __init__(self, n_features):
        super().__init__()
        # nn.Linear(in, out): y = xW^T + b.
        # For regression with 1 output, out=1.
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        # squeeze(-1): Remove the output dimension (batch, 1) -> (batch,).
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic data with 60/20/20 split and scaling."""
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(1, n_features // 2), noise=noise,
        random_state=random_state,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


def _to_loader(X, y, batch_size=32, shuffle=True):
    """Convert numpy arrays to a PyTorch DataLoader for mini-batch training."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train Ridge regression with PyTorch.

    Supports two L2 regularisation approaches:
    1. weight_decay in optimizer (use_weight_decay=True): The optimizer
       subtracts weight_decay * w from each parameter update.
       Setting weight_decay = 2 * alpha is equivalent to alpha * ||w||^2.
    2. Explicit L2 penalty (use_weight_decay=False): We compute
       alpha * sum(w^2) and add it to the loss manually.
    """
    # Extract hyperparameters.
    alpha = hp.get("alpha", 0.01)
    lr = hp.get("lr", 0.01)
    epochs = hp.get("epochs", 100)
    batch_size = hp.get("batch_size", 32)
    opt_name = hp.get("optimizer", "adam")

    # use_weight_decay: Which L2 implementation to use.
    # WHY two approaches: weight_decay is simpler but less flexible.
    # Explicit L2 lets you apply different regularisation to different layers.
    use_wd = hp.get("use_weight_decay", True)

    n_features = X_train.shape[1]
    model = RidgeModel(n_features).to(DEVICE)

    # MSELoss: The data-fitting term of the Ridge objective.
    criterion = nn.MSELoss()

    # Set weight_decay only if using the implicit approach.
    # WHY 2.0 * alpha: PyTorch's weight_decay subtracts wd * w from each update,
    # which is equivalent to adding (wd/2) * ||w||^2 to the loss.
    # So to get alpha * ||w||^2 in the loss, we need wd = 2 * alpha.
    wd = 2.0 * alpha if use_wd else 0.0

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    loader = _to_loader(X_train, y_train, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()

            preds = model(X_b)
            loss = criterion(preds, y_b)

            # If NOT using weight_decay, add explicit L2 penalty.
            if not use_wd:
                # Sum of squared weights across all parameters.
                # WHY model.parameters(): This includes both weights AND bias.
                # In a more careful implementation, you would exclude the bias
                # from regularisation (as in the NumPy version).
                l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                loss = loss + alpha * l2_reg

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)

        epoch_loss /= len(loader.dataset)

    logger.info("Training done  epochs=%d  final_loss=%.6f  alpha=%.4f", epochs, epoch_loss, alpha)
    return model


def _evaluate(model, X, y):
    """Evaluate model without gradient tracking."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return {
        "mse": mean_squared_error(y, preds),
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
    }


def validate(model, X_val, y_val):
    m = _evaluate(model, X_val, y_val)
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    m = _evaluate(model, X_test, y_test)
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare different alpha values and L2 implementation strategies."""
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: PyTorch Ridge - Alpha & Weight Decay Strategies")
    print("=" * 80)

    configs = {
        "Low Alpha (0.001) + weight_decay": {
            "alpha": 0.001, "lr": 0.01, "epochs": 100,
            "batch_size": 32, "optimizer": "adam", "use_weight_decay": True,
            # WHY: Minimal regularisation. Nearly equivalent to unregularised.
        },
        "Moderate Alpha (0.1) + weight_decay": {
            "alpha": 0.1, "lr": 0.01, "epochs": 100,
            "batch_size": 32, "optimizer": "adam", "use_weight_decay": True,
            # WHY: Good general-purpose setting. Moderate shrinkage.
        },
        "High Alpha (1.0) + weight_decay": {
            "alpha": 1.0, "lr": 0.01, "epochs": 100,
            "batch_size": 32, "optimizer": "adam", "use_weight_decay": True,
            # WHY: Strong regularisation. Useful for noisy data.
        },
        "Moderate Alpha (0.1) + explicit L2": {
            "alpha": 0.1, "lr": 0.01, "epochs": 100,
            "batch_size": 32, "optimizer": "adam", "use_weight_decay": False,
            # WHY: Compare explicit L2 vs weight_decay at same alpha.
            # Should give similar results but explicit is more flexible.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        metrics["train_time"] = train_time

        w = model.linear.weight.detach().cpu().numpy().flatten()
        metrics["weight_norm"] = np.sqrt(np.sum(w ** 2))
        results[name] = metrics

        print(f"\n  {name}:")
        print(f"    MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}  ||w||={metrics['weight_norm']:.4f}")

    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<42} {'MSE':>10} {'R2':>10} {'||w||':>8}")
    print("-" * 75)
    for name, m in results.items():
        print(f"{name:<42} {m['mse']:>10.4f} {m['r2']:>10.4f} {m['weight_norm']:>8.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. weight_decay and explicit L2 give similar results at same alpha.")
    print("  2. weight_decay is simpler; explicit L2 offers per-layer control.")
    print("  3. Higher alpha -> smaller ||w|| -> more regularised model.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Energy Consumption Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate Ridge regression for building energy consumption prediction.

    DOMAIN CONTEXT: Building energy management systems predict electricity
    usage based on environmental and building characteristics. Features like
    temperature, humidity, and occupancy are often correlated (e.g., hot
    days increase both temperature and cooling energy). Ridge regression
    handles these correlations gracefully.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Building Energy Consumption Prediction (PyTorch)")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 2000

    # Outdoor temperature (Celsius).
    outdoor_temp = np.random.normal(20, 10, n_samples).clip(-10, 45)

    # Humidity (%), correlated with temperature in summer.
    humidity = 40 + 0.5 * outdoor_temp + np.random.normal(0, 10, n_samples)
    humidity = humidity.clip(10, 100)

    # Building floor area (sq meters).
    floor_area = np.random.lognormal(7, 0.5, n_samples).clip(50, 5000)

    # Number of occupants.
    n_occupants = np.random.poisson(floor_area / 30, n_samples).clip(1, 200)

    # Hour of day (0-23).
    hour = np.random.randint(0, 24, n_samples).astype(float)

    # Day type: weekday (1) vs weekend (0).
    is_weekday = np.random.binomial(1, 5 / 7, n_samples).astype(float)

    feature_names = ["outdoor_temp", "humidity", "floor_area",
                     "n_occupants", "hour", "is_weekday"]
    X = np.column_stack([outdoor_temp, humidity, floor_area,
                         n_occupants, hour, is_weekday])

    true_coefs = np.array([
        50.0,       # 50 kWh per degree C (heating/cooling)
        10.0,       # 10 kWh per % humidity (dehumidification)
        2.0,        # 2 kWh per sq meter
        15.0,       # 15 kWh per occupant
        -5.0,       # -5 kWh per hour (night = less usage)
        200.0,      # 200 kWh more on weekdays
    ])

    base_energy = 500
    noise = np.random.normal(0, 300, n_samples)
    y = base_energy + X @ true_coefs + noise
    y = y.clip(100, None)

    print(f"\nDataset: {n_samples} hourly readings")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Note: humidity is correlated with outdoor_temp")
    print(f"Energy range: {y.min():.0f} - {y.max():.0f} kWh")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train(X_train_s, y_train, alpha=0.1, lr=0.01, epochs=200,
                  batch_size=64, optimizer="adam", use_weight_decay=True)

    metrics = _evaluate(model, X_test_s, y_test)
    print(f"\n--- Performance ---")
    print(f"  RMSE: {metrics['rmse']:.2f} kWh  R2: {metrics['r2']:.4f}")

    w = model.linear.weight.detach().cpu().numpy().flatten()
    print(f"\n--- Feature Weights ---")
    for i, name in enumerate(feature_names):
        print(f"  {name:<15}: {w[i]:>8.2f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
        "epochs": trial.suggest_int("epochs", 50, 300, step=50),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam"]),
        "use_weight_decay": trial.suggest_categorical("use_weight_decay", [True, False]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    study = optuna.create_study(direction="minimize", study_name="ridge_pytorch")
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
    model = train(X_train, y_train, **config)
    metrics = validate(model, X_val, y_val)
    tune.report(**metrics)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "alpha": tune.loguniform(1e-5, 10.0),
        "lr": tune.loguniform(1e-4, 0.1),
        "epochs": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([16, 32, 64]),
        "optimizer": tune.choice(["sgd", "adam"]),
        "use_weight_decay": tune.choice([True, False]),
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
    print("=" * 70)
    print("Ridge Regression - PyTorch (L2 Regularisation)")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=15)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=8)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    model = train(X_train, y_train, **study.best_trial.params)

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
