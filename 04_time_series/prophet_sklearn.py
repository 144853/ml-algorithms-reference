"""
Prophet - Facebook Prophet Library Implementation
===================================================

Theory & Mathematics:
    Prophet is a decomposable time series model developed by Facebook (Meta)
    that fits an additive (or multiplicative) model:

        y(t) = g(t) + s(t) + h(t) + epsilon_t

    1. Trend g(t) - Piecewise Linear or Logistic Growth:
       Linear: g(t) = (k + a(t)^T * delta) * t + (m + a(t)^T * gamma)
       Logistic: g(t) = C(t) / (1 + exp(-(k + a(t)^T * delta) * (t - (m + a(t)^T * gamma))))

       where:
       - k: base growth rate
       - delta: vector of rate adjustments at changepoints
       - m: offset
       - gamma: adjustments to keep function continuous
       - a(t): indicator vector for changepoints
       - C(t): carrying capacity (logistic growth)

    2. Seasonality s(t) - Fourier Series:
       s(t) = sum_{n=1}^{N} (a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P))

       where P is the period (365.25 for yearly, 7 for weekly) and N controls
       smoothness (higher N = more flexible seasonality).

    3. Holiday Effects h(t):
       h(t) = Z(t) * kappa
       where Z(t) is an indicator matrix for holidays and kappa are
       holiday-specific coefficients.

    4. Fitting:
       Prophet uses Stan (Bayesian framework) for MAP estimation.
       Uncertainty intervals are generated via posterior sampling.

Business Use Cases:
    - Business metric forecasting (revenue, users, engagement)
    - Capacity planning
    - Marketing campaign impact analysis (via holiday effects)
    - Anomaly detection (flagging deviations from forecast)
    - Goal setting and budgeting

Advantages:
    - Handles missing data gracefully
    - Robust to outliers and trend changes
    - Automatic changepoint detection
    - Built-in holiday effects
    - Interpretable decomposition (trend, seasonality, holidays)
    - Works out of the box with minimal tuning
    - Uncertainty quantification

Disadvantages:
    - Assumes additive (or log-multiplicative) components
    - Cannot capture complex nonlinear interactions
    - Not designed for multivariate time series
    - Stan compilation can be slow on first run
    - Less flexible than deep learning models
    - Struggles with very high-frequency data

Key Hyperparameters:
    - changepoint_prior_scale: Flexibility of trend (0.001 - 0.5)
    - seasonality_prior_scale: Strength of seasonality (0.01 - 10)
    - holidays_prior_scale: Strength of holiday effects
    - seasonality_mode: 'additive' or 'multiplicative'
    - changepoint_range: Proportion of history for changepoints (default 0.8)
    - n_changepoints: Number of potential changepoints (default 25)
    - yearly_seasonality / weekly_seasonality / daily_seasonality: auto/True/False/int

References:
    - Taylor, S.J. & Letham, B. (2018). Forecasting at Scale. The American Statistician.
    - https://facebook.github.io/prophet/
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Any

import optuna
import ray
from ray import tune

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic time series in Prophet format (ds, y columns).

    Returns:
        Tuple of (train_df, val_df, test_df) with 'ds' and 'y' columns.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_points, freq=freq)
    t = np.arange(n_points, dtype=np.float64)

    trend = trend_slope * t
    seasonality = 2.0 * np.sin(2 * np.pi * t / seasonal_period)
    yearly_seasonality = 1.5 * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, noise_std, n_points)

    values = 10.0 + trend + seasonality + yearly_seasonality + noise

    df = pd.DataFrame({"ds": dates, "y": values})

    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
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
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: pd.DataFrame,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: str = "additive",
    n_changepoints: int = 25,
    yearly_seasonality: Any = "auto",
    weekly_seasonality: Any = "auto",
    daily_seasonality: Any = False,
) -> Any:
    """Fit Prophet model on training data.

    Args:
        train_data: DataFrame with 'ds' and 'y' columns.
        changepoint_prior_scale: Flexibility of trend changepoints.
        seasonality_prior_scale: Strength of seasonality fitting.
        seasonality_mode: 'additive' or 'multiplicative'.
        n_changepoints: Number of potential changepoints.

    Returns:
        Fitted Prophet model.
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Run: pip install prophet")

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        n_changepoints=n_changepoints,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )
    model.fit(train_data)
    return model


def validate(model, val_data: pd.DataFrame) -> Dict[str, float]:
    """Generate predictions for validation dates and compute metrics."""
    future = val_data[["ds"]].copy()
    forecast = model.predict(future)
    return _compute_metrics(val_data["y"].values, forecast["yhat"].values)


def test(model, test_data: pd.DataFrame) -> Dict[str, float]:
    """Generate predictions for test dates and compute metrics."""
    future = test_data[["ds"]].copy()
    forecast = model.predict(future)
    return _compute_metrics(test_data["y"].values, forecast["yhat"].values)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, train_data: pd.DataFrame,
                     val_data: pd.DataFrame) -> float:
    cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
    sps = trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True)
    mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
    n_cp = trial.suggest_int("n_changepoints", 5, 50)

    try:
        model = train(
            train_data,
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            seasonality_mode=mode,
            n_changepoints=n_cp,
        )
        metrics = validate(model, val_data)
        return metrics["RMSE"]
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=20):
    study = optuna.create_study(direction="minimize", study_name="prophet_sklearn")
    study.optimize(
        lambda t: optuna_objective(t, train_data, val_data),
        n_trials=n_trials, show_progress_bar=True,
    )
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data, val_data, num_samples=20):
    def _trainable(config):
        try:
            model = train(
                train_data,
                changepoint_prior_scale=config["cps"],
                seasonality_prior_scale=config["sps"],
                seasonality_mode=config["mode"],
                n_changepoints=config["n_cp"],
            )
            metrics = validate(model, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "cps": tune.loguniform(0.001, 0.5),
        "sps": tune.loguniform(0.01, 10.0),
        "mode": tune.choice(["additive", "multiplicative"]),
        "n_cp": tune.randint(5, 50),
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
    print("Prophet - Facebook Prophet Library Implementation")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    if not PROPHET_AVAILABLE:
        print("\nProphet is not installed. Install with: pip install prophet")
        print("Exiting.")
        return

    # Optuna HPO
    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=15)
    bp = study.best_params

    # Train best
    print("\n--- Training Best Model ---")
    best_model = train(
        train_data,
        changepoint_prior_scale=bp["changepoint_prior_scale"],
        seasonality_prior_scale=bp["seasonality_prior_scale"],
        seasonality_mode=bp["seasonality_mode"],
        n_changepoints=bp["n_changepoints"],
    )

    # Validate
    print("\n--- Validation ---")
    for k, v in validate(best_model, val_data).items():
        print(f"  {k}: {v:.4f}")

    # Test
    print("\n--- Test ---")
    for k, v in test(best_model, test_data).items():
        print(f"  {k}: {v:.4f}")

    # Ray Tune
    print("\n--- Ray Tune ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
