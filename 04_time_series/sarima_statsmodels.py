"""
SARIMA (Seasonal ARIMA) - Statsmodels Implementation
=====================================================

Theory & Mathematics:
    SARIMA extends ARIMA by adding seasonal components. The full model is
    denoted SARIMA(p, d, q)(P, D, Q, s) where:

    Non-seasonal part: ARIMA(p, d, q)
        - p: Non-seasonal AR order
        - d: Non-seasonal differencing order
        - q: Non-seasonal MA order

    Seasonal part: (P, D, Q, s)
        - P: Seasonal AR order (captures AR at seasonal lags)
        - D: Seasonal differencing order
        - Q: Seasonal MA order
        - s: Seasonal period (e.g., 7 for weekly, 12 for monthly, 365 for yearly)

    Full SARIMA equation (using backshift operator B):
        phi(B) * Phi(B^s) * (1 - B)^d * (1 - B^s)^D * y_t
            = c + theta(B) * Theta(B^s) * epsilon_t

    where:
        phi(B) = 1 - phi_1*B - ... - phi_p*B^p          (non-seasonal AR)
        Phi(B^s) = 1 - Phi_1*B^s - ... - Phi_P*B^{Ps}   (seasonal AR)
        theta(B) = 1 + theta_1*B + ... + theta_q*B^q      (non-seasonal MA)
        Theta(B^s) = 1 + Theta_1*B^s + ... + Theta_Q*B^{Qs}  (seasonal MA)
        (1 - B)^d = non-seasonal differencing
        (1 - B^s)^D = seasonal differencing

    Seasonal Differencing Example (D=1, s=7):
        z_t = y_t - y_{t-7}    (removes weekly pattern)

    Combined Differencing (d=1, D=1, s=7):
        z_t = (1-B)(1-B^7) y_t = y_t - y_{t-1} - y_{t-7} + y_{t-8}

Business Use Cases:
    - Retail sales with weekly/monthly/yearly seasonality
    - Electricity demand with daily/seasonal patterns
    - Tourism demand forecasting
    - Agricultural production forecasting
    - Hospital admission forecasting

Advantages:
    - Explicitly models seasonality
    - Well-established statistical theory
    - AIC/BIC for model selection
    - Confidence intervals for forecasts
    - Handles both trend and seasonal non-stationarity

Disadvantages:
    - Computationally expensive for large seasonal periods
    - Can only handle one seasonality pattern
    - Many parameters to tune (7 parameters + seasonal period)
    - Assumes periodic seasonality with fixed period
    - Linear model limitations remain

Key Hyperparameters:
    - (p, d, q): Non-seasonal orders
    - (P, D, Q, s): Seasonal orders and period
    - trend: Trend specification

References:
    - Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. (2015). Time Series Analysis, 5th ed.
    - Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice.
"""

import numpy as np  # NumPy for numerical computations and metric calculations
import pandas as pd  # Pandas for DatetimeIndex-based time series required by SARIMAX
import warnings  # Suppress convergence warnings during HPO sweeps
from typing import Dict, Tuple, Any  # Type hints for function signatures

import optuna  # Bayesian hyperparameter optimization framework
import ray  # Ray distributed computing for parallel HPO execution
from ray import tune  # Ray Tune search space definitions and trial management

# SARIMAX is statsmodels' Seasonal ARIMA with eXogenous variables implementation
# WHY SARIMAX over ARIMA: Even without exogenous variables, SARIMAX handles
# seasonal orders (P, D, Q, s) which plain ARIMA does not support
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress convergence and numerical warnings during automated HPO sweeps
# WHY: Many SARIMA order combinations fail to converge or produce numerical
# instabilities; these warnings are expected and not actionable during search
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
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Generate synthetic time series with trend, multiple seasonalities, and noise.

    Components:
        - Linear trend: simulates steady growth over time
        - Primary sinusoidal seasonality (period = seasonal_period)
        - Secondary harmonic seasonality (period = seasonal_period / 2)
        - Gaussian noise for realistic randomness

    WHY two seasonal components: Real-world seasonal data often has harmonics
    (e.g., weekly pattern may have both 7-day and 3.5-day frequency components).
    This tests whether SARIMA can capture complex seasonal structure.
    """
    # Set random seed for reproducibility across runs
    np.random.seed(seed)

    # Create DatetimeIndex for proper time series handling by statsmodels
    # WHY: SARIMAX requires a DatetimeIndex with a recognized frequency for
    # automatic period detection and forecast date alignment
    dates = pd.date_range(start="2020-01-01", periods=n_points, freq=freq)

    # Float64 time index for numerical operations
    t = np.arange(n_points, dtype=np.float64)

    # Linear trend: models steady growth (e.g., increasing customer base)
    # WHY: The 'd' differencing parameter handles trend removal
    trend = trend_slope * t

    # Primary seasonality: period = seasonal_period, amplitude = 3.0
    # WHY: The (P, D, Q, s) seasonal component in SARIMA is designed to capture this;
    # amplitude of 3.0 makes the seasonal pattern dominate over noise
    seasonality_1 = 3.0 * np.sin(2 * np.pi * t / seasonal_period)

    # Secondary (harmonic) seasonality: period = seasonal_period/2, amplitude = 1.5
    # WHY: Real seasonal patterns are rarely pure sinusoids; the harmonic adds
    # complexity that tests the model's ability to capture non-sinusoidal seasonality
    seasonality_2 = 1.5 * np.cos(2 * np.pi * t / (seasonal_period / 2))

    # Gaussian white noise with specified standard deviation
    # WHY: Simulates unpredictable real-world variations; noise_std controls
    # the signal-to-noise ratio
    noise = np.random.normal(0, noise_std, n_points)

    # Combine all components with positive baseline
    values = 10.0 + trend + seasonality_1 + seasonality_2 + noise

    # Wrap in pandas Series with DatetimeIndex for statsmodels compatibility
    series = pd.Series(values, index=dates, name="value")

    # Chronological 70/15/15 split
    # WHY: Time series must be split in temporal order to prevent data leakage
    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)

    train = series.iloc[:n_train]
    val = series.iloc[n_train : n_train + n_val]
    test = series.iloc[n_train + n_val :]

    return train, val, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and SMAPE forecast evaluation metrics."""
    # Convert to float64 for numerical precision
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Forecast errors: positive = under-prediction, negative = over-prediction
    errors = actual - predicted

    # MAE: average absolute error in original units (interpretable, robust)
    mae = np.mean(np.abs(errors))

    # RMSE: penalises large errors quadratically (good optimization target)
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE: scale-free percentage error; mask zeros to avoid division by zero
    mask = actual != 0
    mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100 if mask.any() else np.inf

    # SMAPE: symmetric percentage error bounded [0, 200%]
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    denom = np.where(denom == 0, 1e-10, denom)
    smape = np.mean(np.abs(errors) / denom) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: pd.Series,
    p: int = 1, d: int = 1, q: int = 1,
    P: int = 1, D: int = 1, Q: int = 1, s: int = 7,
    trend: str = "t",
) -> Any:
    """Fit a SARIMA model using statsmodels SARIMAX.

    Args:
        train_data: Training time series with DatetimeIndex.
        p, d, q: Non-seasonal ARIMA orders.
            p = AR lags for short-range dependence
            d = differencing to remove trend
            q = MA lags for short-range error smoothing
        P, D, Q: Seasonal ARIMA orders.
            P = AR at seasonal lags (e.g., same weekday last week)
            D = seasonal differencing to remove seasonal pattern
            Q = MA at seasonal lags for seasonal error smoothing
        s: Seasonal period (7=weekly, 12=monthly, 52=yearly for weekly data).
        trend: Trend specification ('n','c','t','ct').

    Returns:
        Fitted SARIMAX results object with parameters, diagnostics, and forecast.
    """
    # Create SARIMAX model with specified orders
    # WHY SARIMAX: The 'X' in SARIMAX stands for eXogenous variables; even without
    # exogenous regressors, SARIMAX is the standard statsmodels class for SARIMA
    model = SARIMAX(
        train_data,
        order=(p, d, q),                    # Non-seasonal component
        seasonal_order=(P, D, Q, s),         # Seasonal component
        trend=trend,                          # Trend specification
        enforce_stationarity=False,           # Allow parameter search to explore freely
        enforce_invertibility=False,          # Allow non-invertible MA during HPO
    )
    # WHY enforce_stationarity=False: During HPO, we want to explore all parameter
    # combinations without constraining the optimization. If the AR polynomial roots
    # are inside the unit circle, the model is non-stationary, but we let the
    # validation RMSE determine whether such a model is useful.

    # Fit via Maximum Likelihood Estimation
    # WHY disp=False: Suppress iteration output during fitting (too verbose for HPO)
    # WHY maxiter=200: Allow enough iterations for MLE convergence on complex models
    fitted = model.fit(disp=False, maxiter=200)

    return fitted


def validate(model, val_data: pd.Series) -> Dict[str, float]:
    """Forecast the validation period and compute metrics.

    WHY out-of-sample: Validation metrics guide hyperparameter selection
    without contaminating the test set.
    """
    # Generate point forecasts for the validation horizon
    # WHY .forecast: Produces true out-of-sample predictions beyond training data
    forecast = model.forecast(steps=len(val_data))

    return _compute_metrics(val_data.values, forecast.values)


def test(model, test_data: pd.Series, val_len: int = 0) -> Dict[str, float]:
    """Forecast through validation + test periods and compute test metrics.

    WHY forecast through val: SARIMA forecasts are sequential; we cannot skip
    to the test period without first forecasting through validation.
    """
    total = val_len + len(test_data)
    full_forecast = model.forecast(steps=total)
    test_forecast = full_forecast.iloc[val_len:]  # Extract only test portion

    return _compute_metrics(test_data.values, test_forecast.values)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, train_data: pd.Series,
                     val_data: pd.Series) -> float:
    """Optuna objective: minimize validation RMSE over SARIMA hyperparameters.

    WHY this search space: SARIMA has 7 structural parameters (p,d,q,P,D,Q,s)
    plus trend. The ranges are chosen based on common practical bounds:
    - p,q in [0,3]: Higher orders rarely improve short-term series
    - d in [0,2]: d>2 is almost never needed
    - P,Q in [0,2]: Seasonal orders above 2 are rare
    - D in {0,1}: Seasonal differencing of order 2 is extremely rare
    """
    # Non-seasonal orders
    p = trial.suggest_int("p", 0, 3)  # AR order
    d = trial.suggest_int("d", 0, 2)  # Differencing
    q = trial.suggest_int("q", 0, 3)  # MA order

    # Seasonal orders
    P = trial.suggest_int("P", 0, 2)  # Seasonal AR
    D = trial.suggest_int("D", 0, 1)  # Seasonal differencing
    Q = trial.suggest_int("Q", 0, 2)  # Seasonal MA

    # Seasonal period (fixed at 7 for this dataset)
    s = trial.suggest_categorical("s", [7])

    # Trend type
    trend = trial.suggest_categorical("trend", ["n", "c", "t", "ct"])

    try:
        model = train(train_data, p=p, d=d, q=q, P=P, D=D, Q=Q, s=s, trend=trend)
        metrics = validate(model, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        # Return penalty for invalid/failed SARIMA configurations
        return 1e10


def run_optuna(train_data, val_data, n_trials=30):
    """Run Optuna Bayesian hyperparameter search for best SARIMA configuration."""
    study = optuna.create_study(direction="minimize", study_name="sarima_statsmodels")
    study.optimize(
        lambda t: optuna_objective(t, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data, val_data, num_samples=20):
    """Ray Tune distributed hyperparameter search for SARIMA."""
    def _trainable(config):
        """Inner function called by Ray for each trial."""
        try:
            model = train(
                train_data, p=config["p"], d=config["d"], q=config["q"],
                P=config["P"], D=config["D"], Q=config["Q"],
                s=config["s"], trend=config["trend"],
            )
            metrics = validate(model, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    # Define search space matching Optuna ranges
    search_space = {
        "p": tune.randint(0, 4),
        "d": tune.randint(0, 3),
        "q": tune.randint(0, 4),
        "P": tune.randint(0, 3),
        "D": tune.randint(0, 2),
        "Q": tune.randint(0, 3),
        "s": tune.choice([7]),
        "trend": tune.choice(["n", "c", "t", "ct"]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    print(f"Ray Tune Best config: {best.config}")
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(train_data: pd.Series, val_data: pd.Series) -> None:
    """Compare different SARIMA parameter configurations and explain tradeoffs.

    Parameter Selection Reasoning:
        - (1,1,1)(1,1,1,7): Minimal seasonal model. One AR lag, one MA lag,
          one seasonal AR, one seasonal MA. The "airline model" - simple but
          effective for many seasonal series.
          Best for: Simple weekly seasonal patterns with trend.

        - (2,1,1)(1,1,1,7): Extended non-seasonal AR. Two AR lags capture
          day-over-day momentum while maintaining simple seasonal structure.
          Best for: Series where consecutive days have strong autocorrelation.

        - (1,1,1)(2,1,0,7): Extended seasonal AR, no seasonal MA. Two seasonal
          AR lags mean "this week" and "two weeks ago" predict "next week".
          Best for: Series where the same weekday matters for 2+ weeks back.

        - (0,1,2)(0,1,1,7): No AR at all, only MA components. This is a pure
          error-correction model: predictions based entirely on recent forecast
          errors at both daily and weekly scales.
          Best for: Highly noisy series where error correction outperforms
          autoregressive memory.

        - (1,0,0)(1,0,0,7) no differencing: Stationary model that assumes no
          trend and no evolving seasonality. Tests whether the series is already
          stationary (no differencing needed).
          Best for: Detrended/pre-processed series or series around a constant mean.
    """
    configs = [
        {"label": "SARIMA(1,1,1)(1,1,1,7) - Airline model",
         "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 7, "trend": "c"},
        {"label": "SARIMA(2,1,1)(1,1,1,7) - Extended AR",
         "p": 2, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 7, "trend": "c"},
        {"label": "SARIMA(1,1,1)(2,1,0,7) - Extended SAR, no SMA",
         "p": 1, "d": 1, "q": 1, "P": 2, "D": 1, "Q": 0, "s": 7, "trend": "c"},
        {"label": "SARIMA(0,1,2)(0,1,1,7) - Pure MA model",
         "p": 0, "d": 1, "q": 2, "P": 0, "D": 1, "Q": 1, "s": 7, "trend": "n"},
        {"label": "SARIMA(1,0,0)(1,0,0,7) - Stationary (no diff)",
         "p": 1, "d": 0, "q": 0, "P": 1, "D": 0, "Q": 0, "s": 7, "trend": "c"},
    ]

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: SARIMA order effects on seasonal forecasting")
    print("=" * 70)

    results_summary = []

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            model = train(train_data, p=cfg["p"], d=cfg["d"], q=cfg["q"],
                          P=cfg["P"], D=cfg["D"], Q=cfg["Q"],
                          s=cfg["s"], trend=cfg["trend"])
            metrics = validate(model, val_data)

            # Display AIC/BIC for in-sample fit quality assessment
            print(f"  AIC: {model.aic:.2f}  |  BIC: {model.bic:.2f}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            results_summary.append({
                "config": cfg["label"], "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"], "AIC": model.aic,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            results_summary.append({
                "config": cfg["label"], "RMSE": float("inf"),
                "MAE": float("inf"), "AIC": float("inf"),
            })

    # Rank by validation RMSE
    print("\n" + "-" * 70)
    print("RANKING (by validation RMSE):")
    results_summary.sort(key=lambda x: x["RMSE"])
    for i, r in enumerate(results_summary, 1):
        print(f"  {i}. {r['config']}: RMSE={r['RMSE']:.4f}, AIC={r['AIC']:.1f}")

    print("\nKey Takeaways:")
    print("  - The airline model (1,1,1)(1,1,1,s) is often a strong baseline")
    print("  - Seasonal differencing (D=1) is crucial for seasonal patterns")
    print("  - Adding more seasonal AR lags helps when multi-week patterns exist")
    print("  - Pure MA models work for error-driven series but miss autoregressive structure")
    print("  - Skipping differencing fails badly when trend or seasonality is present")


# ---------------------------------------------------------------------------
# Real-World Demo: Electricity Demand Forecasting
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate SARIMA on a realistic electricity demand forecasting scenario.

    Domain: Energy / Utilities
    Task: Forecast hourly electricity demand for grid management

    This demo generates synthetic data mimicking electricity consumption patterns:
        - Daily seasonality (peak at noon/evening, trough at night)
        - Weekly seasonality (weekdays higher than weekends)
        - Gradual upward trend (population growth, electrification)
        - Temperature-correlated noise (hot/cold extremes increase demand)

    Business Context:
        Electric utilities need accurate demand forecasts for:
        - Generation scheduling (commit power plants hours ahead)
        - Grid stability management (balance supply and demand)
        - Energy market bidding (buy/sell electricity contracts)
        - Infrastructure investment planning (new substations, transmission lines)
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Weekly Electricity Demand Forecasting")
    print("=" * 70)

    # ---- Generate domain-specific synthetic data ----
    np.random.seed(42)

    # 3 years of weekly data (156 weeks)
    # WHY weekly: Aggregated to weekly for SARIMA tractability with s=52 annual seasonality
    n_weeks = 156
    dates = pd.date_range(start="2021-01-04", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks, dtype=np.float64)

    # Base weekly demand: 10,000 MWh (typical for a small utility service area)
    base_demand = 10000.0

    # Growth trend: 2% annual demand growth
    # WHY: Population growth and electrification drive steady demand increases
    weekly_growth = (1.02 ** (1 / 52))  # Convert annual rate to weekly
    trend = base_demand * (weekly_growth ** t - 1)

    # Annual seasonality: summer and winter peaks (U-shaped)
    # WHY: Electricity demand peaks in summer (AC) and winter (heating), with
    # lower demand in spring/fall shoulder seasons
    # Use cos with phase shift so peaks align with January and July
    annual_seasonality = 0.15 * base_demand * np.cos(2 * np.pi * t / 52)

    # Noise: ~5% random variation from weather and behavioral unpredictability
    noise = np.random.normal(0, 0.05 * base_demand, n_weeks)

    # Combine components
    weekly_demand = base_demand + trend + annual_seasonality + noise
    weekly_demand = np.maximum(weekly_demand, 0)  # Demand cannot be negative

    # Create pandas Series with DatetimeIndex
    series = pd.Series(weekly_demand, index=dates, name="demand_mwh")

    print(f"\nDataset: {n_weeks} weeks of electricity demand data")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Demand range: {weekly_demand.min():,.0f} - {weekly_demand.max():,.0f} MWh")
    print(f"Mean weekly demand: {weekly_demand.mean():,.0f} MWh")

    # Split: 80% train, 10% val, 10% test
    n_train = int(0.80 * n_weeks)
    n_val = int(0.10 * n_weeks)

    train_series = series.iloc[:n_train]
    val_series = series.iloc[n_train:n_train + n_val]
    test_series = series.iloc[n_train + n_val:]

    print(f"Train: {len(train_series)} weeks | Val: {len(val_series)} weeks | "
          f"Test: {len(test_series)} weeks")

    # ---- Train and evaluate SARIMA models ----
    # Using s=52 for annual seasonality in weekly data
    # WHY s=52: For weekly observations, 52 weeks = 1 year
    print("\n--- Model Comparison ---")

    # Note: s=52 requires substantial data; with only 156 weeks we use small P,D,Q
    model_configs = [
        {"label": "SARIMA(1,1,1)(1,0,1,52)", "p": 1, "d": 1, "q": 1,
         "P": 1, "D": 0, "Q": 1, "s": 52, "trend": "c"},
        {"label": "SARIMA(2,1,1)(1,0,0,52)", "p": 2, "d": 1, "q": 1,
         "P": 1, "D": 0, "Q": 0, "s": 52, "trend": "c"},
        {"label": "ARIMA(2,1,1) no seasonal", "p": 2, "d": 1, "q": 1,
         "P": 0, "D": 0, "Q": 0, "s": 52, "trend": "c"},
    ]

    best_model = None
    best_rmse = float("inf")
    best_label = ""

    for cfg in model_configs:
        try:
            model = train(train_series, p=cfg["p"], d=cfg["d"], q=cfg["q"],
                          P=cfg["P"], D=cfg["D"], Q=cfg["Q"],
                          s=cfg["s"], trend=cfg["trend"])
            metrics = validate(model, val_series)
            print(f"\n  {cfg['label']}:")
            print(f"    AIC: {model.aic:.1f} | BIC: {model.bic:.1f}")
            print(f"    Val RMSE: {metrics['RMSE']:,.0f} MWh")
            print(f"    Val MAPE: {metrics['MAPE']:.2f}%")

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model = model
                best_label = cfg["label"]
        except Exception as e:
            print(f"\n  {cfg['label']}: FAILED - {e}")

    # ---- Final evaluation ----
    if best_model is not None:
        print(f"\n--- Best Model: {best_label} ---")
        test_metrics = test(best_model, test_series, val_len=len(val_series))
        print(f"  Test RMSE: {test_metrics['RMSE']:,.0f} MWh")
        print(f"  Test MAE:  {test_metrics['MAE']:,.0f} MWh")
        print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")

        print("\n  Business Interpretation:")
        avg_demand = np.mean(test_series.values)
        print(f"    Average weekly demand: {avg_demand:,.0f} MWh")
        print(f"    Average forecast error: {test_metrics['MAE']:,.0f} MWh "
              f"({test_metrics['MAPE']:.1f}% of average)")
        print(f"    For a 10,000 MWh week, error is ~{test_metrics['MAE']:,.0f} MWh")
        acceptable = test_metrics["MAPE"] < 10
        print(f"    Forecast quality: {'Acceptable' if acceptable else 'Needs improvement'} "
              f"(target: <10% MAPE for utility planning)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Full pipeline: data -> HPO -> train -> validate -> test -> compare -> demo."""
    print("=" * 70)
    print("SARIMA - Statsmodels Implementation")
    print("=" * 70)

    # Generate synthetic data with weekly seasonality
    train_data, val_data, test_data = generate_data(n_points=1000, seasonal_period=7)
    print(f"\nData splits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    # Optuna HPO
    print("\n--- Optuna Hyperparameter Search ---")
    study = run_optuna(train_data, val_data, n_trials=25)
    bp = study.best_params

    # Train with best parameters
    print("\n--- Training with Best Parameters ---")
    best_model = train(
        train_data,
        p=bp["p"], d=bp["d"], q=bp["q"],
        P=bp["P"], D=bp["D"], Q=bp["Q"],
        s=bp.get("s", 7), trend=bp["trend"],
    )
    print(f"Non-seasonal order: ({bp['p']}, {bp['d']}, {bp['q']})")
    print(f"Seasonal order: ({bp['P']}, {bp['D']}, {bp['Q']}, {bp.get('s', 7)})")
    print(f"AIC: {best_model.aic:.2f}, BIC: {best_model.bic:.2f}")

    # Validate
    print("\n--- Validation Metrics ---")
    val_metrics = validate(best_model, val_data)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test
    print("\n--- Test Metrics ---")
    test_metrics = test(best_model, test_data, val_len=len(val_data))
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Ray Tune
    print("\n--- Ray Tune Search ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    # Parameter comparison
    compare_parameter_sets(train_data, val_data)

    # Real-world demo
    real_world_demo()

    print("\nDone.")


if __name__ == "__main__":
    main()
