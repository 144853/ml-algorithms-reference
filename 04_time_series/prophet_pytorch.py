"""
Prophet-style Decomposition Model - PyTorch Implementation
============================================================

Theory & Mathematics:
    This module implements a Prophet-inspired decomposable time series model
    in PyTorch where all components are learnable:

        y(t) = g(t) + s(t) + epsilon

    1. Trend g(t) - Learnable Piecewise Linear:
       g(t) = (k + sum_j delta_j * a_j(t)) * t + (m + sum_j gamma_j * a_j(t))

       where:
       - k (nn.Parameter): base growth rate
       - delta (nn.Parameter of size S): slope changes at S changepoints
       - m (nn.Parameter): base offset
       - gamma is computed from delta to ensure continuity
       - a_j(t) = sigmoid(steepness * (t - tau_j)) for differentiable changepoints

    2. Seasonality s(t) - Learnable Fourier Coefficients:
       s(t) = sum_{n=1}^{N} (a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P))

       where a_n, b_n are nn.Parameters, P is the period.
       Multiple seasonalities are supported (weekly + yearly).

    3. Training:
       - Input: time indices t and corresponding values y
       - Loss: MSE(y, g(t) + s(t))
       - Optimiser: Adam
       - Regularisation: L2 on delta (changepoint magnitudes)

    4. Differentiable Changepoints:
       Unlike Prophet's hard indicator a(t) in {0,1}, we use a smooth
       sigmoid: a(t) = sigmoid(beta * (t - tau_j)) where beta controls sharpness.
       This allows gradient flow through the changepoint mechanism.

Business Use Cases:
    - High-frequency time series where GPU acceleration helps
    - Scenarios requiring custom loss functions (quantile, asymmetric)
    - Transfer learning: pretrain components on related series
    - Embedding in larger neural network pipelines

Advantages:
    - End-to-end differentiable
    - GPU acceleration
    - Custom loss functions (quantile, Huber, etc.)
    - Can be embedded in larger models
    - Interpretable decomposition maintained

Disadvantages:
    - Requires careful learning rate tuning
    - No built-in uncertainty quantification (would need MC dropout)
    - More complex setup than Prophet library
    - Risk of overfitting without regularisation

Key Hyperparameters:
    - n_changepoints (int): Number of trend changepoints
    - n_fourier_weekly (int): Fourier order for weekly seasonality
    - n_fourier_yearly (int): Fourier order for yearly seasonality
    - lr (float): Learning rate
    - epochs (int): Training epochs
    - changepoint_reg (float): L2 penalty on changepoint deltas
    - sigmoid_steepness (float): Sharpness of differentiable changepoints

References:
    - Taylor, S.J. & Letham, B. (2018). Forecasting at Scale.
    - Triebe, O. et al. (2021). NeuralProphet.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
import ray
from ray import tune

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_points: int = 1000, freq: str = "D", seasonal_period: int = 7,
    trend_slope: float = 0.02, noise_std: float = 0.5, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    t = np.arange(n_points, dtype=np.float64)
    trend = trend_slope * t
    weekly = 2.0 * np.sin(2 * np.pi * t / seasonal_period)
    yearly = 1.5 * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, noise_std, n_points)
    values = 10.0 + trend + weekly + yearly + noise
    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


def _compute_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mask = actual != 0
    mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100 if mask.any() else np.inf
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    denom = np.where(denom == 0, 1e-10, denom)
    smape = np.mean(np.abs(errors) / denom) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


# ---------------------------------------------------------------------------
# PyTorch Prophet-style Model
# ---------------------------------------------------------------------------

class ProphetNet(nn.Module):
    """Prophet-style decomposition with learnable parameters."""

    def __init__(
        self,
        n_changepoints: int = 25,
        changepoint_times: torch.Tensor = None,
        n_fourier_weekly: int = 3,
        n_fourier_yearly: int = 10,
        sigmoid_steepness: float = 50.0,
    ):
        super().__init__()
        self.n_changepoints = n_changepoints
        self.n_fourier_weekly = n_fourier_weekly
        self.n_fourier_yearly = n_fourier_yearly
        self.sigmoid_steepness = sigmoid_steepness

        # Trend parameters
        self.k = nn.Parameter(torch.tensor(0.01))  # base slope
        self.m = nn.Parameter(torch.tensor(0.0))    # base intercept
        self.deltas = nn.Parameter(torch.zeros(n_changepoints))  # slope changes

        # Changepoint locations (not learnable)
        if changepoint_times is not None:
            self.register_buffer("cp_times", changepoint_times)
        else:
            self.register_buffer("cp_times", torch.linspace(0, 1, n_changepoints))

        # Fourier coefficients for weekly seasonality (2 per Fourier term)
        self.weekly_coeffs = nn.Parameter(
            torch.zeros(2 * n_fourier_weekly)
        ) if n_fourier_weekly > 0 else None

        # Fourier coefficients for yearly seasonality
        self.yearly_coeffs = nn.Parameter(
            torch.zeros(2 * n_fourier_yearly)
        ) if n_fourier_yearly > 0 else None

    def _trend(self, t: torch.Tensor) -> torch.Tensor:
        """Compute piecewise linear trend with differentiable changepoints."""
        # Soft indicator using sigmoid
        # a_j(t) = sigmoid(steepness * (t - tau_j))
        # Shape: (batch, n_changepoints)
        t_expanded = t.unsqueeze(-1)  # (batch, 1)
        cp_expanded = self.cp_times.unsqueeze(0)  # (1, n_cp)
        a = torch.sigmoid(self.sigmoid_steepness * (t_expanded - cp_expanded))

        # Slope at each point: k + sum(delta_j * a_j(t))
        slope = self.k + (a * self.deltas.unsqueeze(0)).sum(dim=-1)

        # Gamma for continuity: gamma_j = -delta_j * tau_j
        gammas = -self.deltas * self.cp_times
        intercept = self.m + (a * gammas.unsqueeze(0)).sum(dim=-1)

        return slope * t + intercept

    def _seasonality(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Fourier-based seasonality."""
        result = torch.zeros_like(t)

        # Weekly seasonality
        if self.weekly_coeffs is not None and self.n_fourier_weekly > 0:
            for n in range(1, self.n_fourier_weekly + 1):
                idx = 2 * (n - 1)
                result = result + (
                    self.weekly_coeffs[idx] * torch.cos(2 * np.pi * n * t / 7.0)
                    + self.weekly_coeffs[idx + 1] * torch.sin(2 * np.pi * n * t / 7.0)
                )

        # Yearly seasonality
        if self.yearly_coeffs is not None and self.n_fourier_yearly > 0:
            for n in range(1, self.n_fourier_yearly + 1):
                idx = 2 * (n - 1)
                result = result + (
                    self.yearly_coeffs[idx] * torch.cos(2 * np.pi * n * t / 365.25)
                    + self.yearly_coeffs[idx + 1] * torch.sin(2 * np.pi * n * t / 365.25)
                )

        return result

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Predict y(t) = g(t) + s(t)."""
        return self._trend(t) + self._seasonality(t)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    n_changepoints: int = 25,
    n_fourier_weekly: int = 3,
    n_fourier_yearly: int = 10,
    changepoint_range: float = 0.8,
    lr: float = 0.05,
    epochs: int = 300,
    changepoint_reg: float = 0.1,
    sigmoid_steepness: float = 50.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train Prophet-style PyTorch model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(train_data)
    t = torch.arange(n, dtype=torch.float32, device=device)
    y = torch.tensor(train_data, dtype=torch.float32, device=device)

    # Place changepoints
    cp_range = int(n * changepoint_range)
    cp_indices = torch.linspace(0, cp_range - 1, n_changepoints, device=device)

    model = ProphetNet(
        n_changepoints=n_changepoints,
        changepoint_times=cp_indices,
        n_fourier_weekly=n_fourier_weekly,
        n_fourier_yearly=n_fourier_yearly,
        sigmoid_steepness=sigmoid_steepness,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        pred = model(t)
        loss = criterion(pred, y)

        # Regularise changepoint deltas
        reg_loss = changepoint_reg * torch.sum(model.deltas ** 2)
        total_loss = loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, "
                  f"Reg: {reg_loss.item():.6f}")

    return {
        "model": model,
        "n_train": n,
        "train_loss": best_loss,
    }


def _predict(result: Dict, t_values: np.ndarray) -> np.ndarray:
    model = result["model"]
    device = next(model.parameters()).device
    model.eval()
    t = torch.tensor(t_values, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(t).cpu().numpy()
    return pred


def validate(result: Dict, val_data: np.ndarray) -> Dict[str, float]:
    n_train = result["n_train"]
    t_val = np.arange(n_train, n_train + len(val_data), dtype=np.float64)
    predictions = _predict(result, t_val)
    return _compute_metrics(val_data, predictions)


def test(result: Dict, test_data: np.ndarray, val_len: int = 0) -> Dict[str, float]:
    n_train = result["n_train"]
    t_test = np.arange(n_train + val_len,
                       n_train + val_len + len(test_data), dtype=np.float64)
    predictions = _predict(result, t_test)
    return _compute_metrics(test_data, predictions)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    n_cp = trial.suggest_int("n_changepoints", 5, 50)
    n_fw = trial.suggest_int("n_fourier_weekly", 1, 10)
    n_fy = trial.suggest_int("n_fourier_yearly", 3, 20)
    lr = trial.suggest_float("lr", 1e-3, 0.2, log=True)
    epochs = trial.suggest_int("epochs", 100, 500, step=100)
    reg = trial.suggest_float("changepoint_reg", 0.001, 1.0, log=True)

    try:
        result = train(train_data, n_changepoints=n_cp, n_fourier_weekly=n_fw,
                       n_fourier_yearly=n_fy, lr=lr, epochs=epochs,
                       changepoint_reg=reg, verbose=False)
        metrics = validate(result, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=20):
    study = optuna.create_study(direction="minimize", study_name="prophet_pytorch")
    study.optimize(lambda t: optuna_objective(t, train_data, val_data),
                   n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


def ray_tune_search(train_data, val_data, num_samples=20):
    def _trainable(config):
        try:
            result = train(train_data, n_changepoints=config["n_cp"],
                           n_fourier_weekly=config["n_fw"], n_fourier_yearly=config["n_fy"],
                           lr=config["lr"], epochs=config["epochs"],
                           changepoint_reg=config["reg"], verbose=False)
            metrics = validate(result, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "n_cp": tune.randint(5, 50), "n_fw": tune.randint(1, 10),
        "n_fy": tune.randint(3, 20), "lr": tune.loguniform(1e-3, 0.2),
        "epochs": tune.choice([100, 200, 300, 400]),
        "reg": tune.loguniform(0.001, 1.0),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    tuner = tune.Tuner(
        _trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Prophet-style Decomposition - PyTorch Implementation")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=15)
    bp = study.best_params

    print("\n--- Training Best ---")
    result = train(train_data, n_changepoints=bp["n_changepoints"],
                   n_fourier_weekly=bp["n_fourier_weekly"],
                   n_fourier_yearly=bp["n_fourier_yearly"],
                   lr=bp["lr"], epochs=bp["epochs"],
                   changepoint_reg=bp["changepoint_reg"])

    m = result["model"]
    print(f"Base slope: {m.k.item():.6f}")
    print(f"Base intercept: {m.m.item():.4f}")

    print("\n--- Validation ---")
    for k, v in validate(result, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Test ---")
    for k, v in test(result, test_data, val_len=len(val_data)).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Ray Tune ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
