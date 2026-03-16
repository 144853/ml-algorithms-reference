"""
Seasonal Time Series Model - PyTorch Implementation
=====================================================

Theory & Mathematics:
    This module implements a SARIMA-inspired neural model using PyTorch that
    captures both seasonal and non-seasonal autoregressive/moving-average
    patterns through learnable parameters.

    Architecture:
        The model has two parallel pathways that are summed:

        1. Non-seasonal pathway:
           AR_pred = sum_{i=1..p} phi_i * y_{t-i}  +  sum_{j=1..q} theta_j * eps_{t-j}

        2. Seasonal pathway:
           SAR_pred = sum_{k=1..P} Phi_k * y_{t-k*s}  +  sum_{l=1..Q} Theta_l * eps_{t-l*s}

        3. Combined:
           y_hat_t = bias + AR_pred + SAR_pred

        All coefficients (phi, theta, Phi, Theta, bias) are nn.Parameter objects
        optimised via gradient descent.

    Training:
        - Preprocessing: differencing (both regular and seasonal)
        - Sequences are windowed with lookback = max(p, P*s, q, Q*s)
        - Loss: MSE
        - Optimiser: Adam

    Inference:
        - Multi-step ahead forecasting with residuals set to 0 for future steps
        - Undifferencing to recover original scale

Business Use Cases:
    - Retail demand with day-of-week and seasonal effects
    - Energy grid load forecasting
    - Tourism and hospitality demand
    - Agricultural yield prediction with seasonal patterns

Advantages:
    - GPU-accelerated training
    - Easily extensible with neural layers
    - Can learn complex coefficient interactions
    - Regularisation via weight decay, dropout
    - Batch training for efficiency

Disadvantages:
    - Requires more data than classical SARIMA
    - More hyperparameters to tune
    - No closed-form confidence intervals
    - Sequential residual computation limits parallelism
    - Risk of overfitting on short series

Key Hyperparameters:
    - p, q: Non-seasonal AR/MA orders
    - P, Q: Seasonal AR/MA orders
    - s: Seasonal period
    - d, D: Differencing orders (preprocessing)
    - lr: Learning rate
    - epochs: Training epochs
    - weight_decay: L2 regularisation

References:
    - Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. (2015). Time Series Analysis.
    - Salinas, D. et al. (2020). DeepAR.
"""

import numpy as np  # NumPy for data generation, metrics, and array operations
import pandas as pd  # Pandas for date handling in the real-world demo
import warnings  # Suppress warnings during HPO sweeps
from typing import Dict, Tuple, Any  # Type hints for function signatures

import torch  # PyTorch core: tensors and autograd
import torch.nn as nn  # Neural network module: nn.Parameter, nn.Module
import torch.optim as optim  # Optimizers: Adam with weight decay

import optuna  # Bayesian HPO with TPE sampler
import ray  # Ray distributed computing for parallel HPO
from ray import tune  # Ray Tune search space definitions

# Suppress warnings during automated search
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_points: int = 1000, freq: str = "D", seasonal_period: int = 7,
    trend_slope: float = 0.02, noise_std: float = 0.5, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic time series with trend, dual seasonality, and noise."""
    np.random.seed(seed)  # Reproducible random state
    t = np.arange(n_points, dtype=np.float64)  # Float64 time index

    # Linear trend for non-stationarity (handled by d parameter)
    trend = trend_slope * t

    # Primary seasonality: period=seasonal_period, strong amplitude
    seas1 = 3.0 * np.sin(2 * np.pi * t / seasonal_period)

    # Harmonic seasonality: period=seasonal_period/2, tests higher-frequency capture
    seas2 = 1.5 * np.cos(2 * np.pi * t / (seasonal_period / 2))

    # Gaussian noise for realistic randomness
    noise = np.random.normal(0, noise_std, n_points)

    # Combine with positive baseline
    values = 10.0 + trend + seas1 + seas2 + noise

    # Chronological split: never shuffle time series
    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


def _compute_metrics(actual, predicted):
    """Compute MAE, RMSE, MAPE, SMAPE forecast evaluation metrics."""
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
# PyTorch Seasonal ARIMA Model
# ---------------------------------------------------------------------------

class SeasonalARIMANet(nn.Module):
    """Seasonal ARIMA-like model with learnable seasonal and non-seasonal coefficients.

    This model has four groups of learnable parameters:
        - ar_coeffs: Non-seasonal AR at lags 1, 2, ..., p
        - sar_coeffs: Seasonal AR at lags s, 2s, ..., Ps
        - ma_coeffs: Non-seasonal MA at lags 1, 2, ..., q
        - sma_coeffs: Seasonal MA at lags s, 2s, ..., Qs

    WHY PyTorch over classical SARIMA estimation:
        - Gradient descent can optimize all 4 coefficient groups jointly
        - Easy to add regularization (weight decay) to prevent overfitting
        - Can be extended with nonlinear layers for hybrid models
        - GPU acceleration for large-scale seasonal forecasting
    """

    def __init__(self, p: int = 1, q: int = 1, P: int = 1, Q: int = 1, s: int = 7):
        """Initialize with learnable AR/MA coefficients for both seasonal and non-seasonal.

        Args:
            p: Non-seasonal AR order (lags 1..p)
            q: Non-seasonal MA order (lags 1..q)
            P: Seasonal AR order (lags s, 2s, ..., Ps)
            Q: Seasonal MA order (lags s, 2s, ..., Qs)
            s: Seasonal period (7=weekly, 12=monthly)
        """
        super().__init__()
        self.p = p
        self.q = q
        self.P = P
        self.Q = Q
        self.s = s

        # Non-seasonal AR coefficients: small random init to avoid exploding gradients
        self.ar_coeffs = nn.Parameter(torch.randn(p) * 0.01) if p > 0 else None

        # Non-seasonal MA coefficients
        self.ma_coeffs = nn.Parameter(torch.randn(q) * 0.01) if q > 0 else None

        # Seasonal AR coefficients: operate at multiples of s
        self.sar_coeffs = nn.Parameter(torch.randn(P) * 0.01) if P > 0 else None

        # Seasonal MA coefficients: operate at multiples of s
        self.sma_coeffs = nn.Parameter(torch.randn(Q) * 0.01) if Q > 0 else None

        # Bias/constant term initialized to zero
        self.bias = nn.Parameter(torch.zeros(1))

        # Minimum lookback window must cover all lag requirements
        # WHY max over all: The input sequence must be long enough for both
        # the longest AR lag and the longest MA lag to have valid indices
        self.lookback = max(p, q, P * s, Q * s, 1)

    def forward(self, y_seq: torch.Tensor, res_seq: torch.Tensor) -> torch.Tensor:
        """Predict next value from past values and past residuals.

        Args:
            y_seq: (batch, lookback) past differenced values.
            res_seq: (batch, lookback) past residuals.

        Returns:
            (batch,) predictions on the differenced scale.
        """
        batch = y_seq.shape[0]
        pred = self.bias.expand(batch)  # Start with bias

        # Non-seasonal AR: sum phi_i * y_{t-i} for i = 1..p
        if self.ar_coeffs is not None and self.p > 0:
            for i in range(self.p):
                idx = y_seq.shape[1] - 1 - i  # Most recent = highest index
                if idx >= 0:
                    pred = pred + self.ar_coeffs[i] * y_seq[:, idx]

        # Seasonal AR: sum Phi_k * y_{t-k*s} for k = 1..P
        if self.sar_coeffs is not None and self.P > 0:
            for k in range(self.P):
                idx = y_seq.shape[1] - (k + 1) * self.s  # Seasonal lag index
                if idx >= 0:
                    pred = pred + self.sar_coeffs[k] * y_seq[:, idx]

        # Non-seasonal MA: sum theta_j * eps_{t-j} for j = 1..q
        if self.ma_coeffs is not None and self.q > 0:
            for j in range(self.q):
                idx = res_seq.shape[1] - 1 - j
                if idx >= 0:
                    pred = pred + self.ma_coeffs[j] * res_seq[:, idx]

        # Seasonal MA: sum Theta_l * eps_{t-l*s} for l = 1..Q
        if self.sma_coeffs is not None and self.Q > 0:
            for l in range(self.Q):
                idx = res_seq.shape[1] - (l + 1) * self.s
                if idx >= 0:
                    pred = pred + self.sma_coeffs[l] * res_seq[:, idx]

        return pred


def _difference(series, d, D, s):
    """Apply seasonal then regular differencing as preprocessing."""
    result = series.copy()
    # Seasonal differencing first: z_t = y_t - y_{t-s}
    for _ in range(D):
        result = result[s:] - result[:-s]
    # Then regular differencing: w_t = z_t - z_{t-1}
    for _ in range(d):
        result = np.diff(result)
    return result


def _undifference(forecasts, original, d, D, s):
    """Undo regular then seasonal differencing (reverse order of application)."""
    result = forecasts.copy()

    # Undo regular differencing (cumulative sum from last seasonally-differenced value)
    if d > 0:
        seas_diff = original.copy()
        for _ in range(D):
            seas_diff = seas_diff[s:] - seas_diff[:-s]
        last = seas_diff[-1]  # Anchor for cumulative sum
        for _ in range(d):
            result = np.cumsum(np.concatenate([[last], result]))[1:]

    # Undo seasonal differencing (add back seasonal lag values)
    if D > 0:
        tail = original[-s:]  # Last s values of original series
        restored = np.zeros(len(result))
        for i in range(len(result)):
            # For first season, use original tail; after that, use restored values
            restored[i] = result[i] + (tail[i % s] if i < s else restored[i - s])
        result = restored

    return result


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    p: int = 1, d: int = 1, q: int = 1,
    P: int = 1, D: int = 1, Q: int = 1, s: int = 7,
    lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train seasonal ARIMA PyTorch model via gradient descent.

    The training loop processes each timestep sequentially because the MA
    component requires residuals from previous timesteps, creating a
    sequential dependency that prevents full batching.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply differencing as preprocessing to achieve stationarity
    z = _difference(train_data, d, D, s)

    # Initialize model with specified orders
    model = SeasonalARIMANet(p=p, q=q, P=P, Q=Q, s=s).to(device)

    # Adam optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # MSE loss for regression
    criterion = nn.MSELoss()

    lookback = model.lookback  # Minimum history needed for all lag indices
    z_t = torch.tensor(z, dtype=torch.float32, device=device)  # Differenced series tensor

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        # Re-initialize residuals each epoch with current model parameters
        residuals = torch.zeros(len(z), device=device)

        for t in range(lookback, len(z)):
            # Extract lookback-sized history windows
            y_hist = z_t[t - lookback:t].unsqueeze(0)  # (1, lookback)
            r_hist = residuals[t - lookback:t].unsqueeze(0).detach()  # Detach to stop BPTT

            # Forward pass: predict differenced value at time t
            pred = model(y_hist, r_hist).squeeze()
            target = z_t[t]

            loss = criterion(pred, target)
            residuals[t] = (target - pred).detach()  # Record residual for future MA
            total_loss += loss.item()
            count += 1

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / max(count, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return {
        "model": model,
        "d": d, "D": D, "s": s,
        "original_series": train_data,
        "differenced_series": z,
        "residuals": residuals.detach().cpu().numpy(),
        "train_loss": best_loss,
    }


def _forecast(result, n_steps):
    """Generate multi-step recursive forecasts, then undifference."""
    model = result["model"]
    device = next(model.parameters()).device
    model.eval()

    z = list(result["differenced_series"])
    res = list(result["residuals"])
    lookback = model.lookback

    with torch.no_grad():
        for _ in range(n_steps):
            y_hist = torch.tensor(z[-lookback:], dtype=torch.float32, device=device).unsqueeze(0)
            r_hist = torch.tensor(res[-lookback:], dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(y_hist, r_hist).item()
            z.append(pred)
            res.append(0.0)  # Future residuals are zero (expected value)

    diff_fc = np.array(z[-n_steps:])
    return _undifference(diff_fc, result["original_series"], result["d"], result["D"], result["s"])


def validate(result, val_data):
    """Forecast validation period and compute metrics."""
    forecasts = _forecast(result, len(val_data))
    return _compute_metrics(val_data, forecasts)


def test(result, test_data, val_len=0):
    """Forecast through validation + test, evaluate test portion."""
    all_fc = _forecast(result, val_len + len(test_data))
    return _compute_metrics(test_data, all_fc[val_len:])


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    """Optuna objective: minimize validation RMSE over SARIMA + lr + epochs."""
    p = trial.suggest_int("p", 0, 3)
    d = trial.suggest_int("d", 0, 1)
    q = trial.suggest_int("q", 0, 3)
    P = trial.suggest_int("P", 0, 2)
    D = trial.suggest_int("D", 0, 1)
    Q = trial.suggest_int("Q", 0, 2)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 50, 150, step=50)

    try:
        result = train(train_data, p=p, d=d, q=q, P=P, D=D, Q=Q, s=7,
                       lr=lr, epochs=epochs, verbose=False)
        metrics = validate(result, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=20):
    """Run Optuna Bayesian HPO search."""
    study = optuna.create_study(direction="minimize", study_name="sarima_pytorch")
    study.optimize(
        lambda t: optuna_objective(t, train_data, val_data),
        n_trials=n_trials, show_progress_bar=True,
    )
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


def ray_tune_search(train_data, val_data, num_samples=20):
    """Ray Tune distributed HPO search."""
    def _trainable(config):
        try:
            result = train(train_data, p=config["p"], d=config["d"], q=config["q"],
                           P=config["P"], D=config["D"], Q=config["Q"], s=7,
                           lr=config["lr"], epochs=config["epochs"], verbose=False)
            metrics = validate(result, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "p": tune.randint(0, 4), "d": tune.randint(0, 2),
        "q": tune.randint(0, 4), "P": tune.randint(0, 3),
        "D": tune.randint(0, 2), "Q": tune.randint(0, 3),
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": tune.choice([50, 100, 150]),
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
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(train_data: np.ndarray, val_data: np.ndarray) -> None:
    """Compare different SARIMA-PyTorch configurations and explain tradeoffs.

    Parameter Selection Reasoning:
        - (1,1,1)(1,1,1,7) lr=0.01: Minimal seasonal model, moderate lr.
          Fastest training, least overfitting risk. The "airline model" baseline.

        - (2,1,1)(1,1,1,7) lr=0.005: Extended non-seasonal AR with slower lr.
          More parameters need gentler optimization to converge properly.
          Best for: series with consecutive-day dependence beyond lag 1.

        - (1,1,0)(1,1,0,7) lr=0.01: Pure AR, no MA components at all.
          Tests whether MA adds value or just adds optimization difficulty.
          Best for: Series where values predict well without error correction.

        - (0,1,2)(0,1,1,7) lr=0.01: Pure MA, no AR components.
          All prediction comes from error correction at daily and weekly scales.
          Best for: highly noisy series where past errors are more informative.

        - (1,1,1)(1,1,1,7) lr=0.001: Same structure as baseline but 10x lower lr.
          Tests whether slower optimization finds better minima.
    """
    configs = [
        {"label": "SARIMA(1,1,1)(1,1,1,7) lr=0.01 - Baseline",
         "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "lr": 0.01, "epochs": 80},
        {"label": "SARIMA(2,1,1)(1,1,1,7) lr=0.005 - Extended AR",
         "p": 2, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "lr": 0.005, "epochs": 100},
        {"label": "SARIMA(1,1,0)(1,1,0,7) lr=0.01 - Pure AR",
         "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "lr": 0.01, "epochs": 80},
        {"label": "SARIMA(0,1,2)(0,1,1,7) lr=0.01 - Pure MA",
         "p": 0, "d": 1, "q": 2, "P": 0, "D": 1, "Q": 1, "lr": 0.01, "epochs": 80},
        {"label": "SARIMA(1,1,1)(1,1,1,7) lr=0.001 - Low lr",
         "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "lr": 0.001, "epochs": 120},
    ]

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: Seasonal PyTorch ARIMA configurations")
    print("=" * 70)

    results_summary = []
    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            result = train(train_data, p=cfg["p"], d=cfg["d"], q=cfg["q"],
                           P=cfg["P"], D=cfg["D"], Q=cfg["Q"], s=7,
                           lr=cfg["lr"], epochs=cfg["epochs"], verbose=False)
            metrics = validate(result, val_data)
            print(f"  Training loss: {result['train_loss']:.6f}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            results_summary.append({"config": cfg["label"], "RMSE": metrics["RMSE"]})
        except Exception as e:
            print(f"  FAILED: {e}")
            results_summary.append({"config": cfg["label"], "RMSE": float("inf")})

    print("\n" + "-" * 70)
    print("RANKING (by validation RMSE):")
    results_summary.sort(key=lambda x: x["RMSE"])
    for i, r in enumerate(results_summary, 1):
        print(f"  {i}. {r['config']}: RMSE={r['RMSE']:.4f}")

    print("\nKey Takeaways:")
    print("  - Learning rate interacts with model complexity (more params = lower lr needed)")
    print("  - Pure AR is faster to train but may miss MA structure")
    print("  - Seasonal differencing (D=1) is crucial for periodic patterns")


# ---------------------------------------------------------------------------
# Real-World Demo: Weekly Retail Foot Traffic Forecasting
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate seasonal PyTorch ARIMA on retail foot traffic forecasting.

    Domain: Retail / Shopping Centres
    Task: Forecast daily foot traffic for staffing and promotional planning

    Synthetic data characteristics:
        - Strong weekly seasonality (weekends > weekdays)
        - Growth trend (store becoming more popular)
        - Holiday spikes (Black Friday, Christmas)
        - Weather-related noise
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Daily Retail Foot Traffic Forecasting")
    print("=" * 70)

    np.random.seed(99)

    n_days = 730  # 2 years of daily data
    t = np.arange(n_days, dtype=np.float64)

    # Base: 500 visitors/day for a small retail store
    base = 500.0

    # Growth: 8% annual growth (store becoming established)
    daily_growth = 1.08 ** (1 / 365)
    trend = base * (daily_growth ** t - 1)

    # Weekly seasonality: weekends ~50% more traffic
    weekly = 0.25 * base * np.sin(2 * np.pi * t / 7)

    # Annual seasonality: holiday season boost in November-December
    annual = 0.10 * base * np.sin(2 * np.pi * (t - 60) / 365)

    # Noise: 12% variation from weather, promotions, events
    noise = np.random.normal(0, 0.06 * base, n_days)

    traffic = base + trend + weekly + annual + noise
    traffic = np.maximum(np.round(traffic), 0).astype(float)

    print(f"\nDataset: {n_days} days of foot traffic data")
    print(f"Traffic range: {traffic.min():.0f} - {traffic.max():.0f} visitors/day")
    print(f"Mean: {traffic.mean():.0f} visitors/day")

    n_train = int(0.80 * n_days)
    n_val = int(0.10 * n_days)
    train_t = traffic[:n_train]
    val_t = traffic[n_train:n_train + n_val]
    test_t = traffic[n_train + n_val:]

    print(f"Train: {len(train_t)} | Val: {len(val_t)} | Test: {len(test_t)}")

    # Train best model
    print("\n--- Training SARIMA(1,1,1)(1,1,1,7) ---")
    try:
        result = train(train_t, p=1, d=1, q=1, P=1, D=1, Q=1, s=7,
                       lr=0.01, epochs=80, verbose=True)

        val_metrics = validate(result, val_t)
        print(f"\nValidation RMSE: {val_metrics['RMSE']:.1f} visitors/day")
        print(f"Validation MAPE: {val_metrics['MAPE']:.2f}%")

        test_metrics = test(result, test_t, val_len=len(val_t))
        print(f"\nTest RMSE: {test_metrics['RMSE']:.1f} visitors/day")
        print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")

        print("\nBusiness Interpretation:")
        print(f"  Average daily forecast error: {test_metrics['MAE']:.0f} visitors")
        print(f"  This allows staffing to within +/- {test_metrics['MAE']:.0f} customers")
        print(f"  For promotional planning, {test_metrics['MAPE']:.1f}% MAPE is "
              f"{'acceptable' if test_metrics['MAPE'] < 15 else 'marginal'}")
    except Exception as e:
        print(f"Training failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Full pipeline: data -> HPO -> train -> validate -> test -> compare -> demo."""
    print("=" * 70)
    print("Seasonal ARIMA - PyTorch Implementation")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=1000, seasonal_period=7)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=15)
    bp = study.best_params

    print("\n--- Training Best Model ---")
    result = train(train_data, p=bp["p"], d=bp["d"], q=bp["q"],
                   P=bp["P"], D=bp["D"], Q=bp["Q"], s=7,
                   lr=bp["lr"], epochs=bp["epochs"])

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

    compare_parameter_sets(train_data, val_data)
    real_world_demo()

    print("\nDone.")


if __name__ == "__main__":
    main()
