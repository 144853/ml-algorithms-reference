"""
Lasso Regression - PyTorch Implementation (L1 Regularisation)
=============================================================

COMPLETE ML TUTORIAL: This file implements Lasso regression in PyTorch by
explicitly adding an L1 penalty to the loss. Unlike Ridge (L2), L1
regularisation is NOT natively supported by PyTorch optimisers' weight_decay
parameter, so we must compute it manually.

Theory & Mathematics:
    Lasso minimises the L1-regularised MSE loss:

        L(w, b) = (1/n)||y - (Xw + b)||^2  +  alpha * ||w||_1

    PyTorch's weight_decay implements L2 ONLY. For L1, we add the penalty
    explicitly: loss = mse_loss + alpha * sum(|w|)

    Note: SGD-based Lasso does not produce exact zeros (unlike coordinate
    descent). A post-hoc thresholding step is applied to achieve sparsity.

Hyperparameters:
    - alpha (float): L1 regularisation strength. Default 0.01.
    - lr (float): Learning rate. Default 0.01.
    - epochs (int): Training epochs. Default 200.
    - batch_size (int): Mini-batch size. Default 32.
    - optimizer (str): "sgd" or "adam". Default "adam".
    - threshold (float): Post-training sparsification threshold. Default 1e-4.
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LassoModel(nn.Module):
    """Linear model with L1 penalty support for Lasso regression.

    The model architecture is identical to linear regression (one nn.Linear layer).
    The "Lasso" behaviour comes from adding an L1 penalty to the loss function
    during training, not from the architecture itself.
    """

    def __init__(self, n_features):
        super().__init__()
        # Single linear layer: y = xW^T + b
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def l1_penalty(self):
        """Compute the L1 norm of the weight matrix (excluding bias).

        WHY exclude bias: The bias captures the mean of y and should not
        be regularised. Penalising the bias would bias predictions toward 0.

        Returns:
            Scalar tensor: sum of absolute values of all weights.
        """
        return self.linear.weight.abs().sum()

    def sparsify(self, threshold=1e-4):
        """Zero out weights below the threshold for approximate sparsity.

        WHY this is needed: SGD-based optimisation does not produce exact
        zeros (the L1 gradient at zero is undefined, and PyTorch uses
        subgradient sign(w) which oscillates around zero). This post-hoc
        step thresholds near-zero weights to exactly zero.

        Args:
            threshold: Weights with |w| < threshold are set to 0.

        Returns:
            Number of weights that were zeroed out.
        """
        with torch.no_grad():
            # Find weights that are effectively zero (below threshold).
            mask = self.linear.weight.abs() < threshold
            # Set those weights to exactly zero.
            self.linear.weight[mask] = 0.0
        n_zero = int(mask.sum().item())
        return n_zero


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic data with 60/20/20 split and standard scaling."""
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
    """Convert numpy arrays to PyTorch DataLoader."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train Lasso model with PyTorch using explicit L1 penalty.

    The L1 penalty is computed each forward pass and added to the MSE loss:
        total_loss = MSE + alpha * sum(|w|)

    After training, near-zero weights are thresholded to exactly zero.
    """
    alpha = hp.get("alpha", 0.01)
    lr = hp.get("lr", 0.01)
    epochs = hp.get("epochs", 200)
    batch_size = hp.get("batch_size", 32)
    opt_name = hp.get("optimizer", "adam")

    # threshold: For post-training sparsification.
    # WHY needed: SGD never produces exact zeros, so we threshold after training.
    # WHY 1e-4 default: Weights this small contribute negligibly to predictions.
    threshold = hp.get("threshold", 1e-4)

    n_features = X_train.shape[1]
    model = LassoModel(n_features).to(DEVICE)
    criterion = nn.MSELoss()

    # NOTE: No weight_decay here! L1 is added explicitly to the loss.
    # weight_decay in PyTorch implements L2, not L1.
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = _to_loader(X_train, y_train, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()

            preds = model(X_b)

            # The TOTAL loss = MSE (data fitting) + alpha * L1 (regularisation).
            # WHY add not multiply: alpha scales the L1 term relative to MSE.
            # High alpha -> stronger sparsity pressure.
            loss = criterion(preds, y_b) + alpha * model.l1_penalty()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)
        epoch_loss /= len(loader.dataset)

    # Post-training sparsification: zero out near-zero weights.
    n_zero = model.sparsify(threshold=threshold)
    n_total = n_features
    logger.info("Trained  alpha=%.4f  zeros=%d/%d  epochs=%d  loss=%.6f",
                alpha, n_zero, n_total, epochs, epoch_loss)
    return model


def _evaluate(model, X, y):
    """Evaluate model without gradient computation."""
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
    """Compare alpha values and thresholding strategies for PyTorch Lasso."""
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: PyTorch Lasso - Alpha & Threshold Trade-offs")
    print("=" * 80)

    n_features = X_train.shape[1]

    configs = {
        "Low Alpha (0.001)": {
            "alpha": 0.001, "lr": 0.01, "epochs": 200,
            "batch_size": 32, "optimizer": "adam", "threshold": 1e-4,
            # WHY: Minimal L1. Most weights stay non-zero.
        },
        "Moderate Alpha (0.01)": {
            "alpha": 0.01, "lr": 0.01, "epochs": 200,
            "batch_size": 32, "optimizer": "adam", "threshold": 1e-4,
            # WHY: Balanced sparsity. Good default.
        },
        "High Alpha (0.1)": {
            "alpha": 0.1, "lr": 0.01, "epochs": 200,
            "batch_size": 32, "optimizer": "adam", "threshold": 1e-4,
            # WHY: Strong L1 pressure. Many weights approach zero.
        },
        "Moderate Alpha, Aggressive Threshold": {
            "alpha": 0.01, "lr": 0.01, "epochs": 200,
            "batch_size": 32, "optimizer": "adam", "threshold": 0.01,
            # WHY: Same alpha but larger threshold. More weights zeroed out.
            # TRADE-OFF: Threshold is a post-hoc hack. True sparsity comes from alpha.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        w = model.linear.weight.detach().cpu().numpy().flatten()
        n_nz = int(np.sum(w != 0))
        metrics["n_nonzero"] = n_nz
        metrics["train_time"] = train_time
        results[name] = metrics

        print(f"\n  {name}: NonZero={n_nz}/{n_features}  MSE={metrics['mse']:.6f}")

    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<40} {'NZ':>4} {'MSE':>10} {'R2':>10}")
    print("-" * 68)
    for name, m in results.items():
        print(f"{name:<40} {m['n_nonzero']:>4} {m['mse']:>10.4f} {m['r2']:>10.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. PyTorch Lasso uses thresholding for sparsity (approximate, not exact).")
    print("  2. Higher alpha drives more weights toward zero before thresholding.")
    print("  3. The threshold parameter controls the post-hoc sparsification cutoff.")
    print("  4. For exact sparsity, use sklearn Lasso (coordinate descent).")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Stock Return Factor Analysis
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate PyTorch Lasso for identifying stock return factors.

    DOMAIN CONTEXT: In quantitative finance, factor models predict stock
    returns based on many potential factors (value, momentum, size, quality,
    volatility, etc.). Most factors are noise; Lasso identifies which ones
    actually predict returns.

    WHY Lasso: With dozens of candidate factors, most are irrelevant.
    Lasso automatically selects the significant factors, producing a
    parsimonious model that is less likely to overfit to historical data.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Stock Return Factor Analysis (PyTorch Lasso)")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 1200  # Monthly returns for 100 stocks over 12 months

    # Define potential factors (some real, some noise).
    factor_names = [
        "value_pe_ratio", "momentum_12m", "market_cap_log",
        "earnings_growth", "dividend_yield", "volatility_60d",
        "debt_to_equity", "roa", "analyst_rating",
        "social_sentiment", "insider_buying", "sector_beta"
    ]

    X = np.column_stack([
        np.random.normal(0, 1, n_samples) for _ in factor_names
    ])

    # True factors: only 4 out of 12 actually predict returns.
    true_coefs = np.array([
        0.3,    # Value: stocks with low P/E outperform
        0.5,    # Momentum: recent winners keep winning
        -0.2,   # Size: smaller companies outperform (small cap premium)
        0.4,    # Earnings growth: growing companies outperform
        0.0,    # Dividend: no predictive power (noise)
        0.0,    # Volatility: no predictive power
        0.0,    # Debt: no predictive power
        0.0,    # ROA: no predictive power
        0.0,    # Analyst: no predictive power
        0.0,    # Social: no predictive power
        0.0,    # Insider: no predictive power
        0.0,    # Beta: no predictive power
    ])

    # Stock returns are notoriously noisy (low signal-to-noise ratio).
    noise = np.random.normal(0, 2, n_samples)  # High noise!
    y = X @ true_coefs + noise  # Monthly return in %

    print(f"\nDataset: {n_samples} stock-month observations")
    print(f"Factors: {len(factor_names)} candidates, only {int(np.sum(true_coefs != 0))} are real")
    print(f"Return range: {y.min():.2f}% to {y.max():.2f}%")
    print(f"Signal-to-noise ratio: LOW (realistic for finance)")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train(X_train_s, y_train, alpha=0.05, lr=0.01, epochs=300,
                  batch_size=64, optimizer="adam", threshold=0.01)

    metrics = _evaluate(model, X_test_s, y_test)
    w = model.linear.weight.detach().cpu().numpy().flatten()

    print(f"\n--- Factor Selection Results ---")
    print(f"  R2: {metrics['r2']:.4f} (Note: low R2 is normal for financial returns)")
    print(f"\n{'Factor':<22} {'Lasso Weight':>12} {'True Coef':>10} {'Selected':>10}")
    print("-" * 58)
    for i, name in enumerate(factor_names):
        sel = "YES" if w[i] != 0 else "no"
        print(f"{name:<22} {w[i]:>12.4f} {true_coefs[i]:>10.2f} {sel:>10}")

    n_selected = int(np.sum(w != 0))
    print(f"\n  Factors selected: {n_selected}/{len(factor_names)}")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-5, 5.0, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
        "epochs": trial.suggest_int("epochs", 50, 400, step=50),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam"]),
        "threshold": trial.suggest_float("threshold", 1e-6, 1e-2, log=True),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    study = optuna.create_study(direction="minimize", study_name="lasso_pytorch")
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
        "alpha": tune.loguniform(1e-5, 5.0),
        "lr": tune.loguniform(1e-4, 0.1),
        "epochs": tune.choice([100, 200, 300]),
        "batch_size": tune.choice([16, 32, 64]),
        "optimizer": tune.choice(["sgd", "adam"]),
        "threshold": tune.loguniform(1e-6, 1e-2),
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
    print("Lasso Regression - PyTorch (L1 Regularisation)")
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

    best_p = study.best_trial.params
    model = train(X_train, y_train, **best_p)

    w = model.linear.weight.detach().cpu().numpy().flatten()
    n_nonzero = int(np.sum(w != 0))
    print(f"\nNon-zero weights: {n_nonzero}/{len(w)}")

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
