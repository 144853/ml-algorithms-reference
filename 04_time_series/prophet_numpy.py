"""
Prophet-style Decomposition - NumPy From-Scratch Implementation
================================================================

Theory & Mathematics:
    This module implements Prophet-style time series decomposition and forecasting
    from scratch using NumPy. The model decomposes the series into:

        y(t) = g(t) + s(t) + h(t) + epsilon_t

    1. Trend g(t) - Piecewise Linear:
       The trend is modelled as a piecewise linear function with changepoints:

       g(t) = (k + a(t)^T * delta) * t + (m + a(t)^T * gamma)

       where:
       - k: base growth rate (slope)
       - delta: vector of slope changes at changepoints [s_1, s_2, ..., s_S]
       - m: base offset (intercept)
       - gamma: adjustments to ensure continuity: gamma_j = -s_j * tau_j
         (tau_j is the changepoint time)
       - a(t): indicator vector where a(t)_j = 1 if t >= tau_j

       Estimation:
       - Changepoints are placed at evenly spaced quantiles of the training data
       - Slopes between changepoints estimated via linear regression on segments
       - Regularisation via L2 penalty on delta (changepoint_prior_scale)

    2. Seasonality s(t) - Fourier Series:
       s(t) = sum_{n=1}^{N} (a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P))

       where:
       - P: period (7 for weekly, 365.25 for yearly)
       - N: number of Fourier terms (controls smoothness)
       - a_n, b_n: Fourier coefficients estimated via least squares

       Multiple seasonalities can be stacked (weekly + yearly).

    3. Holiday Effects h(t):
       h(t) = sum_i kappa_i * indicator(t in holiday_i window)

       Each holiday has a window [lower, upper] around the date, and an
       associated coefficient kappa estimated via regression.

    4. Fitting Process:
       a) Estimate trend by fitting piecewise linear to the data
       b) Remove trend: detrended = y - g(t)
       c) Estimate seasonality via Fourier regression on detrended
       d) Remove seasonality: residual = detrended - s(t)
       e) Optionally estimate holiday effects on residual
       f) Iterate for refinement

Business Use Cases:
    - Business forecasting without external dependencies
    - Custom deployment environments without Stan/Prophet
    - Educational: understand Prophet internals
    - Embedded systems with limited library support

Advantages:
    - Full control and transparency
    - No heavy dependencies (just NumPy)
    - Piecewise linear trend is interpretable
    - Fourier seasonality is flexible and efficient
    - Easy to extend with custom components

Disadvantages:
    - No Bayesian uncertainty quantification
    - Simpler changepoint detection than Prophet
    - No automatic cross-validation utilities
    - Manual holiday specification required
    - Iterative decomposition is approximate

Key Hyperparameters:
    - n_changepoints (int): Number of trend changepoints
    - changepoint_range (float): Fraction of data for changepoint placement
    - n_fourier_weekly (int): Fourier order for weekly seasonality
    - n_fourier_yearly (int): Fourier order for yearly seasonality
    - regularisation_strength (float): L2 penalty on changepoint deltas

References:
    - Taylor, S.J. & Letham, B. (2018). Forecasting at Scale.
    - Harvey, A.C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass, field

import optuna
import ray
from ray import tune

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_points: int = 1000,
    freq: str = "D",
    seasonal_period: int = 7,
    trend_slope: float = 0.02,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic time series.

    Returns:
        (train_values, val_values, test_values, all_times)
        where times are in [0, n_points) normalised.
    """
    np.random.seed(seed)
    t = np.arange(n_points, dtype=np.float64)
    trend = trend_slope * t
    weekly = 2.0 * np.sin(2 * np.pi * t / seasonal_period)
    yearly = 1.5 * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, noise_std, n_points)
    values = 10.0 + trend + weekly + yearly + noise

    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)
    return (values[:n_train], values[n_train:n_train + n_val],
            values[n_train + n_val:], t)


def _compute_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    min_len = min(len(actual), len(predicted))
    actual, predicted = actual[:min_len], predicted[:min_len]
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
# Prophet-style Model Components
# ---------------------------------------------------------------------------

@dataclass
class ProphetNumpyModel:
    """Container for the fitted decomposition model."""
    # Trend parameters
    k: float = 0.0  # base slope
    m: float = 0.0  # base intercept
    changepoint_times: np.ndarray = field(default_factory=lambda: np.array([]))
    deltas: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas: np.ndarray = field(default_factory=lambda: np.array([]))

    # Seasonality parameters
    fourier_coeffs_weekly: np.ndarray = field(default_factory=lambda: np.array([]))
    fourier_coeffs_yearly: np.ndarray = field(default_factory=lambda: np.array([]))
    n_fourier_weekly: int = 3
    n_fourier_yearly: int = 10
    weekly_period: float = 7.0
    yearly_period: float = 365.25

    # Training time range
    t_min: float = 0.0
    t_max: float = 1.0
    n_train: int = 0


def _make_fourier_features(t: np.ndarray, period: float, n_terms: int) -> np.ndarray:
    """Create Fourier features matrix.

    Returns matrix of shape (len(t), 2*n_terms) with columns:
    [cos(2pi*1*t/P), sin(2pi*1*t/P), cos(2pi*2*t/P), sin(2pi*2*t/P), ...]
    """
    features = np.zeros((len(t), 2 * n_terms))
    for n in range(1, n_terms + 1):
        features[:, 2 * (n - 1)] = np.cos(2 * np.pi * n * t / period)
        features[:, 2 * (n - 1) + 1] = np.sin(2 * np.pi * n * t / period)
    return features


def _piecewise_linear(t: np.ndarray, k: float, m: float,
                      changepoint_times: np.ndarray,
                      deltas: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Evaluate piecewise linear trend at times t."""
    n = len(t)
    trend = np.zeros(n)
    for i in range(n):
        slope = k
        intercept = m
        for j in range(len(changepoint_times)):
            if t[i] >= changepoint_times[j]:
                slope += deltas[j]
                intercept += gammas[j]
        trend[i] = slope * t[i] + intercept
    return trend


def _fit_trend(t: np.ndarray, y: np.ndarray, n_changepoints: int = 25,
               changepoint_range: float = 0.8,
               reg_strength: float = 0.1) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit piecewise linear trend.

    Returns:
        (k, m, changepoint_times, deltas, gammas)
    """
    n = len(t)
    if n_changepoints == 0 or n < 3:
        # Simple linear regression
        A = np.column_stack([t, np.ones(n)])
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        return coeffs[0], coeffs[1], np.array([]), np.array([]), np.array([])

    # Place changepoints at quantiles in first changepoint_range of data
    cp_range = int(n * changepoint_range)
    cp_indices = np.linspace(0, cp_range - 1, n_changepoints + 2, dtype=int)[1:-1]
    cp_times = t[cp_indices]

    # Build design matrix for piecewise linear
    # Columns: [t, 1, (t - tau_1)_+, (t - tau_2)_+, ...]
    n_cp = len(cp_times)
    A = np.zeros((n, 2 + n_cp))
    A[:, 0] = t  # slope
    A[:, 1] = 1.0  # intercept
    for j in range(n_cp):
        A[:, 2 + j] = np.maximum(0, t - cp_times[j])

    # Regularised least squares
    reg = reg_strength * np.eye(A.shape[1])
    reg[0, 0] = 0  # don't regularise base slope
    reg[1, 1] = 0  # don't regularise intercept
    coeffs = np.linalg.lstsq(A.T @ A + reg, A.T @ y, rcond=None)[0]

    k = coeffs[0]
    m = coeffs[1]
    deltas = coeffs[2:]

    # Compute gammas for continuity
    gammas = -deltas * cp_times

    return k, m, cp_times, deltas, gammas


def _fit_seasonality(t: np.ndarray, y_detrended: np.ndarray,
                     period: float, n_fourier: int,
                     reg_strength: float = 0.1) -> np.ndarray:
    """Fit Fourier seasonality via regularised least squares.

    Returns:
        Fourier coefficients of shape (2 * n_fourier,).
    """
    if n_fourier == 0:
        return np.array([])

    X = _make_fourier_features(t, period, n_fourier)
    reg = reg_strength * np.eye(X.shape[1])
    coeffs = np.linalg.lstsq(X.T @ X + reg, X.T @ y_detrended, rcond=None)[0]
    return coeffs


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    n_changepoints: int = 25,
    changepoint_range: float = 0.8,
    n_fourier_weekly: int = 3,
    n_fourier_yearly: int = 10,
    reg_strength: float = 0.1,
    n_iterations: int = 3,
) -> ProphetNumpyModel:
    """Fit Prophet-style decomposition model.

    Steps (iterated n_iterations times):
        1. Fit piecewise linear trend
        2. Detrend
        3. Fit weekly Fourier seasonality
        4. Fit yearly Fourier seasonality
        5. Compute residuals, use as refined target
    """
    n = len(train_data)
    t = np.arange(n, dtype=np.float64)
    y = train_data.copy()

    model = ProphetNumpyModel(
        n_fourier_weekly=n_fourier_weekly,
        n_fourier_yearly=n_fourier_yearly,
        t_min=0.0, t_max=float(n - 1), n_train=n,
    )

    for iteration in range(n_iterations):
        # 1. Fit trend
        k, m, cp_times, deltas, gammas = _fit_trend(
            t, y, n_changepoints, changepoint_range, reg_strength
        )
        model.k = k
        model.m = m
        model.changepoint_times = cp_times
        model.deltas = deltas
        model.gammas = gammas

        # 2. Detrend
        trend = _piecewise_linear(t, k, m, cp_times, deltas, gammas)
        detrended = train_data - trend

        # 3. Fit weekly seasonality
        weekly_coeffs = _fit_seasonality(t, detrended, 7.0, n_fourier_weekly, reg_strength)
        model.fourier_coeffs_weekly = weekly_coeffs

        # 4. Remove weekly, fit yearly
        weekly_component = np.zeros(n)
        if len(weekly_coeffs) > 0:
            X_w = _make_fourier_features(t, 7.0, n_fourier_weekly)
            weekly_component = X_w @ weekly_coeffs

        residual_for_yearly = detrended - weekly_component
        yearly_coeffs = _fit_seasonality(t, residual_for_yearly, 365.25,
                                         n_fourier_yearly, reg_strength)
        model.fourier_coeffs_yearly = yearly_coeffs

        # 5. Compute full seasonal
        yearly_component = np.zeros(n)
        if len(yearly_coeffs) > 0:
            X_y = _make_fourier_features(t, 365.25, n_fourier_yearly)
            yearly_component = X_y @ yearly_coeffs

        # Refined target for next iteration
        y = train_data - weekly_component - yearly_component

    return model


def _predict(model: ProphetNumpyModel, t: np.ndarray) -> np.ndarray:
    """Generate predictions at times t."""
    # Trend
    trend = _piecewise_linear(t, model.k, model.m, model.changepoint_times,
                              model.deltas, model.gammas)

    # Weekly seasonality
    weekly = np.zeros(len(t))
    if len(model.fourier_coeffs_weekly) > 0:
        X_w = _make_fourier_features(t, model.weekly_period, model.n_fourier_weekly)
        weekly = X_w @ model.fourier_coeffs_weekly

    # Yearly seasonality
    yearly = np.zeros(len(t))
    if len(model.fourier_coeffs_yearly) > 0:
        X_y = _make_fourier_features(t, model.yearly_period, model.n_fourier_yearly)
        yearly = X_y @ model.fourier_coeffs_yearly

    return trend + weekly + yearly


def validate(model: ProphetNumpyModel, val_data: np.ndarray) -> Dict[str, float]:
    n_train = model.n_train
    t_val = np.arange(n_train, n_train + len(val_data), dtype=np.float64)
    predictions = _predict(model, t_val)
    return _compute_metrics(val_data, predictions)


def test(model: ProphetNumpyModel, test_data: np.ndarray,
         val_len: int = 0) -> Dict[str, float]:
    n_train = model.n_train
    t_test = np.arange(n_train + val_len,
                       n_train + val_len + len(test_data), dtype=np.float64)
    predictions = _predict(model, t_test)
    return _compute_metrics(test_data, predictions)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    n_cp = trial.suggest_int("n_changepoints", 5, 50)
    cp_range = trial.suggest_float("changepoint_range", 0.5, 0.95)
    n_fw = trial.suggest_int("n_fourier_weekly", 1, 10)
    n_fy = trial.suggest_int("n_fourier_yearly", 3, 20)
    reg = trial.suggest_float("reg_strength", 0.001, 10.0, log=True)

    try:
        model = train(train_data, n_changepoints=n_cp, changepoint_range=cp_range,
                       n_fourier_weekly=n_fw, n_fourier_yearly=n_fy,
                       reg_strength=reg)
        metrics = validate(model, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=30):
    study = optuna.create_study(direction="minimize", study_name="prophet_numpy")
    study.optimize(lambda t: optuna_objective(t, train_data, val_data),
                   n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data, val_data, num_samples=20):
    def _trainable(config):
        try:
            model = train(train_data, n_changepoints=config["n_cp"],
                           changepoint_range=config["cp_range"],
                           n_fourier_weekly=config["n_fw"],
                           n_fourier_yearly=config["n_fy"],
                           reg_strength=config["reg"])
            metrics = validate(model, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "n_cp": tune.randint(5, 50),
        "cp_range": tune.uniform(0.5, 0.95),
        "n_fw": tune.randint(1, 10),
        "n_fy": tune.randint(3, 20),
        "reg": tune.loguniform(0.001, 10.0),
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
    print("Prophet-style Decomposition - NumPy From-Scratch")
    print("=" * 70)

    train_data, val_data, test_data, all_t = generate_data(n_points=1000)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=25)
    bp = study.best_params

    print("\n--- Training Best Model ---")
    best_model = train(
        train_data,
        n_changepoints=bp["n_changepoints"],
        changepoint_range=bp["changepoint_range"],
        n_fourier_weekly=bp["n_fourier_weekly"],
        n_fourier_yearly=bp["n_fourier_yearly"],
        reg_strength=bp["reg_strength"],
    )
    print(f"Base slope (k): {best_model.k:.6f}")
    print(f"Base intercept (m): {best_model.m:.4f}")
    print(f"Changepoints: {len(best_model.changepoint_times)}")

    print("\n--- Validation ---")
    for k, v in validate(best_model, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Test ---")
    for k, v in test(best_model, test_data, val_len=len(val_data)).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Ray Tune ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
