"""
Seasonal ARIMA Concepts - NumPy From-Scratch Implementation
============================================================

Theory & Mathematics:
    This module implements seasonal time series decomposition and forecasting
    from scratch using NumPy, inspired by the SARIMA framework.

    The approach breaks down into several components:

    1. Seasonal Differencing:
       z_t = y_t - y_{t-s}
       This removes a seasonal pattern of period s. For a weekly pattern with
       daily data, s=7.

    2. Regular Differencing:
       After seasonal differencing, we may apply regular differencing:
       w_t = z_t - z_{t-1}
       to remove trend from the seasonally-adjusted series.

    3. Seasonal AR (SAR) Coefficients:
       The seasonal AR component captures autoregression at seasonal lags:
       y_t = Phi_1 * y_{t-s} + Phi_2 * y_{t-2s} + ...
       Estimated via the Yule-Walker equations at seasonal lags.

    4. Non-seasonal AR:
       Standard AR estimated via Yule-Walker on the seasonally-differenced series.

    5. Seasonal MA (SMA):
       Captures moving average at seasonal lags. Estimated from residual
       autocovariance at seasonal lags.

    6. Combined Forecast:
       Forecast = SAR_part + AR_part + SMA_part + constant
       Undo differencing to return to original scale.

    7. Seasonal Decomposition (STL-like):
       As an alternative approach, we also implement additive decomposition:
       y_t = T_t + S_t + R_t
       where T_t = trend (moving average), S_t = seasonal (period averages),
       R_t = residual.

Business Use Cases:
    - Retail sales with weekly patterns
    - Energy demand with daily/weekly/annual cycles
    - Website traffic with day-of-week effects
    - Call centre volume forecasting

Advantages:
    - Full transparency - every step visible
    - No library dependencies beyond NumPy
    - Decomposition provides interpretable components
    - Educational: understand SARIMA internals

Disadvantages:
    - Approximate estimation (not full MLE)
    - No confidence intervals
    - Single seasonality only
    - Less numerically robust than library implementations

Key Hyperparameters:
    - p (int): Non-seasonal AR order
    - d (int): Non-seasonal differencing
    - q (int): Non-seasonal MA order
    - P (int): Seasonal AR order
    - D (int): Seasonal differencing
    - Q (int): Seasonal MA order
    - s (int): Seasonal period

References:
    - Cleveland, R.B. et al. (1990). STL: A Seasonal-Trend Decomposition.
    - Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. (2015). Time Series Analysis.
"""

import numpy as np  # NumPy is the sole computation backend for this from-scratch implementation
import pandas as pd  # Pandas for date handling in the real-world demo only
import warnings  # Suppress warnings during automated HPO sweeps
from typing import Dict, Tuple, Any, Optional, List  # Type annotations for function signatures
from dataclasses import dataclass, field  # Dataclass for clean model state management

import optuna  # Bayesian hyperparameter optimization with TPE sampler
import ray  # Ray for distributed parallel HPO execution
from ray import tune  # Ray Tune search space definitions

# Suppress numerical warnings during HPO (near-singular matrices are expected)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic time series with trend, dual seasonality, and noise."""
    # Reproducible random state
    np.random.seed(seed)

    # Float64 time index for precise numerical operations
    t = np.arange(n_points, dtype=np.float64)

    # Linear trend simulating steady growth (handled by d parameter)
    trend = trend_slope * t

    # Primary seasonality: period = seasonal_period, strong amplitude
    # WHY amplitude=3.0: Makes the seasonal pattern clearly dominant over noise,
    # so the seasonal AR/MA components have a clear signal to learn
    seasonality_1 = 3.0 * np.sin(2 * np.pi * t / seasonal_period)

    # Secondary harmonic: period = seasonal_period/2, weaker amplitude
    # WHY: Real seasonal patterns often have harmonics (e.g., mid-week dip
    # in addition to weekend pattern); this tests higher-frequency capture
    seasonality_2 = 1.5 * np.cos(2 * np.pi * t / (seasonal_period / 2))

    # Gaussian white noise for realistic randomness
    noise = np.random.normal(0, noise_std, n_points)

    # Combine with positive baseline
    values = 10.0 + trend + seasonality_1 + seasonality_2 + noise

    # Chronological 70/15/15 split (never shuffle time series)
    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, SMAPE forecast evaluation metrics."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    # Handle length mismatches from differencing edge effects
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
# Seasonal ARIMA from scratch
# ---------------------------------------------------------------------------

@dataclass
class SARIMAModel:
    """Container for fitted Seasonal ARIMA model parameters and state.

    Stores all coefficients (AR, SAR, MA, SMA), the differenced series for
    forecasting, and the original series for undifferencing back to original scale.
    """
    p: int = 1   # Non-seasonal AR order
    d: int = 1   # Non-seasonal differencing order
    q: int = 1   # Non-seasonal MA order
    P: int = 1   # Seasonal AR order (operates at lags s, 2s, ...)
    D: int = 1   # Seasonal differencing order
    Q: int = 1   # Seasonal MA order (operates at lags s, 2s, ...)
    s: int = 7   # Seasonal period (7=weekly for daily data)
    # Non-seasonal AR coefficients [phi_1, ..., phi_p]
    ar_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    # Seasonal AR coefficients [Phi_1, ..., Phi_P] at lags s, 2s, ...
    sar_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    # Non-seasonal MA coefficients [theta_1, ..., theta_q]
    ma_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    # Seasonal MA coefficients [Theta_1, ..., Theta_Q] at lags s, 2s, ...
    sma_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    # Constant/intercept term
    constant: float = 0.0
    # Fitted residuals for MA component forecasting
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    # Original training series (needed for undifferencing)
    original_series: np.ndarray = field(default_factory=lambda: np.array([]))
    # Fully differenced series (seasonal + regular differencing applied)
    differenced_series: np.ndarray = field(default_factory=lambda: np.array([]))


def _seasonal_difference(x: np.ndarray, s: int, D: int) -> np.ndarray:
    """Apply seasonal differencing D times with period s.

    For D=1, s=7: z_t = y_t - y_{t-7} (removes weekly pattern)
    For D=2, s=7: apply twice (removes evolving weekly pattern)

    WHY seasonal differencing: Removes the repeating seasonal pattern from the
    data, leaving only trend and irregular components for the AR/MA to model.
    """
    result = x.copy()
    for _ in range(D):
        # Subtract the value from s steps ago: z_t = y_t - y_{t-s}
        # WHY: If the same-weekday values are similar, this difference is small,
        # effectively removing the weekly pattern
        result = result[s:] - result[:-s]
    return result


def _regular_difference(x: np.ndarray, d: int) -> np.ndarray:
    """Apply regular (non-seasonal) differencing d times.

    For d=1: z_t = y_t - y_{t-1} (removes linear trend)
    WHY after seasonal: Apply seasonal differencing first, then regular, so that
    trend is removed from the seasonally-adjusted series.
    """
    result = x.copy()
    for _ in range(d):
        result = np.diff(result)
    return result


def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute sample autocorrelation for lags 0 through max_lag.

    rho(k) = sum_t (x_t - mean)(x_{t-k} - mean) / (n * var(x))
    """
    n = len(x)
    mean = np.mean(x)
    centered = x - mean
    var = np.sum(centered ** 2) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag < n:
            acf[lag] = np.sum(centered[:n - lag] * centered[lag:]) / (n * var)
    return acf


def _yule_walker(x: np.ndarray, order: int) -> np.ndarray:
    """Solve Yule-Walker equations for AR coefficients.

    R * phi = r, where R is the Toeplitz autocorrelation matrix.
    WHY: Yule-Walker provides closed-form AR estimates with stationarity guarantee.
    """
    if order == 0:
        return np.array([])

    acf = _autocorrelation(x, order)

    # Build Toeplitz matrix R[i,j] = acf[|i-j|]
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = acf[abs(i - j)]
    r = acf[1:order + 1]

    try:
        return np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(R, r, rcond=None)[0]


def _seasonal_yule_walker(x: np.ndarray, P: int, s: int) -> np.ndarray:
    """Estimate seasonal AR coefficients at seasonal lags s, 2s, ..., Ps.

    WHY separate from regular Yule-Walker: The seasonal AR coefficients operate
    at multiples of the seasonal period s, not at consecutive lags. We build
    a Toeplitz matrix from autocorrelations at seasonal lags only.
    """
    if P == 0:
        return np.array([])

    # Need autocorrelation up to lag P*s
    max_lag = P * s
    acf = _autocorrelation(x, max_lag)

    # Build P x P Toeplitz matrix using autocorrelations at seasonal lags
    # R[i,j] = acf[|i-j| * s] for seasonal lag structure
    R = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            lag = abs(i - j) * s  # Seasonal lag spacing
            R[i, j] = acf[lag] if lag < len(acf) else 0.0

    # RHS: autocorrelations at seasonal lags s, 2s, ..., Ps
    r = np.array([acf[k * s] if k * s < len(acf) else 0.0 for k in range(1, P + 1)])

    try:
        return np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(R, r, rcond=None)[0]


def _estimate_ma_from_residuals(residuals: np.ndarray, q: int) -> np.ndarray:
    """Estimate MA coefficients from residual autocorrelations.

    Simple approximation: theta_k = -acf(k)
    WHY: For an MA(q) process, the theoretical autocorrelation at lag k equals
    -theta_k (for k <= q). This gives a quick estimate without full MLE.
    """
    if q == 0:
        return np.array([])
    acf = _autocorrelation(residuals, q)
    return -acf[1:q + 1]


def _estimate_seasonal_ma(residuals: np.ndarray, Q: int, s: int) -> np.ndarray:
    """Estimate seasonal MA coefficients at seasonal lags s, 2s, ..., Qs.

    WHY: Same approximation as regular MA but at seasonal lags.
    Theta_k = -acf(k*s) for k = 1, ..., Q.
    """
    if Q == 0:
        return np.array([])
    max_lag = Q * s
    acf = _autocorrelation(residuals, max_lag)
    return np.array([-acf[k * s] if k * s < len(acf) else 0.0 for k in range(1, Q + 1)])


def train(
    train_data: np.ndarray,
    p: int = 1, d: int = 1, q: int = 1,
    P: int = 1, D: int = 1, Q: int = 1, s: int = 7,
) -> SARIMAModel:
    """Fit seasonal ARIMA model from scratch.

    Steps:
        1. Seasonal differencing (D times, period s) to remove seasonal pattern
        2. Regular differencing (d times) to remove trend
        3. Estimate seasonal AR coefficients via seasonal Yule-Walker
        4. Estimate non-seasonal AR coefficients via Yule-Walker
        5. Compute residuals from AR + SAR model
        6. Estimate MA and SMA coefficients from residual autocorrelations
    """
    model = SARIMAModel(p=p, d=d, q=q, P=P, D=D, Q=Q, s=s)
    model.original_series = train_data.copy()

    # Step 1: Seasonal differencing to remove periodic pattern
    z = _seasonal_difference(train_data, s, D)

    # Step 2: Regular differencing to remove trend from seasonally-adjusted series
    z = _regular_difference(z, d)
    model.differenced_series = z

    # Guard: check we have enough data after differencing
    if len(z) < max(p, P * s, 2) + 1:
        model.ar_coeffs = np.zeros(p)
        model.sar_coeffs = np.zeros(P)
        model.ma_coeffs = np.zeros(q)
        model.sma_coeffs = np.zeros(Q)
        model.residuals = np.zeros(len(z))
        return model

    # Step 3: Estimate seasonal AR coefficients
    model.sar_coeffs = _seasonal_yule_walker(z, P, s)

    # Step 4: Estimate non-seasonal AR coefficients
    model.ar_coeffs = _yule_walker(z, p)

    # Step 5: Estimate constant from differenced series mean
    # c = mean(z) * (1 - sum(ar) - sum(sar))
    model.constant = np.mean(z) * (1.0 - np.sum(model.ar_coeffs) - np.sum(model.sar_coeffs))

    # Step 6: Compute residuals from combined AR + SAR model
    n = len(z)
    residuals = np.zeros(n)
    start = max(p, P * s)  # Need at least this many past values
    for t in range(start, n):
        pred = model.constant
        # Non-seasonal AR: sum phi_i * z_{t-i}
        for i in range(p):
            pred += model.ar_coeffs[i] * z[t - 1 - i]
        # Seasonal AR: sum Phi_k * z_{t-k*s}
        for k in range(P):
            idx = t - (k + 1) * s
            if idx >= 0:
                pred += model.sar_coeffs[k] * z[idx]
        residuals[t] = z[t] - pred
    model.residuals = residuals

    # Step 7: Estimate MA and SMA from residuals
    model.ma_coeffs = _estimate_ma_from_residuals(residuals[start:], q)
    model.sma_coeffs = _estimate_seasonal_ma(residuals[start:], Q, s)

    return model


def _forecast_differenced(model: SARIMAModel, n_steps: int) -> np.ndarray:
    """Generate forecasts on the fully differenced scale."""
    z = list(model.differenced_series)
    res = list(model.residuals)

    for _ in range(n_steps):
        pred = model.constant
        n = len(z)

        # Non-seasonal AR component
        for i in range(model.p):
            if n - 1 - i >= 0:
                pred += model.ar_coeffs[i] * z[n - 1 - i]

        # Seasonal AR component (at seasonal lag multiples)
        for k in range(model.P):
            idx = n - (k + 1) * model.s
            if idx >= 0:
                pred += model.sar_coeffs[k] * z[idx]

        # Non-seasonal MA component
        for j in range(model.q):
            if len(res) - 1 - j >= 0:
                pred += model.ma_coeffs[j] * res[len(res) - 1 - j]

        # Seasonal MA component
        for k in range(model.Q):
            idx = len(res) - (k + 1) * model.s
            if idx >= 0:
                pred += model.sma_coeffs[k] * res[idx]

        z.append(pred)
        res.append(0.0)  # Future residuals assumed zero

    return np.array(z[-n_steps:])


def _undo_differencing(diff_forecasts: np.ndarray, original: np.ndarray,
                       d: int, D: int, s: int) -> np.ndarray:
    """Undo regular and seasonal differencing to recover original scale.

    Order: undo regular differencing first, then seasonal differencing.
    WHY this order: Differencing was applied as seasonal-then-regular,
    so undoing is regular-then-seasonal (reverse order).
    """
    result = diff_forecasts.copy()

    # Undo regular differencing (cumulative sum from last seasonally-differenced value)
    if d > 0:
        seasonal_diffed = _seasonal_difference(original, s, D)
        last_val = seasonal_diffed[-1]
        for _ in range(d):
            result = np.cumsum(np.concatenate([[last_val], result]))[1:]

    # Undo seasonal differencing (add back seasonal lag values)
    if D > 0:
        tail = original[-s:].copy()  # Last s values of original series
        restored = np.zeros(len(result))
        for i in range(len(result)):
            if i < s:
                # Use original tail values as anchors for first season
                restored[i] = result[i] + tail[i]
            else:
                # After first season, use previously restored values
                restored[i] = result[i] + restored[i - s]
        result = restored

    return result


def validate(model: SARIMAModel, val_data: np.ndarray) -> Dict[str, float]:
    """Validate: forecast validation period and compute metrics."""
    diff_fc = _forecast_differenced(model, len(val_data))
    forecasts = _undo_differencing(diff_fc, model.original_series, model.d, model.D, model.s)
    return _compute_metrics(val_data, forecasts)


def test(model: SARIMAModel, test_data: np.ndarray,
         val_data: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Test: forecast through validation + test periods, evaluate test portion."""
    val_len = len(val_data) if val_data is not None else 0
    total = val_len + len(test_data)
    diff_fc = _forecast_differenced(model, total)
    all_fc = _undo_differencing(diff_fc, model.original_series, model.d, model.D, model.s)
    return _compute_metrics(test_data, all_fc[val_len:])


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    """Optuna objective: minimize validation RMSE for SARIMA orders."""
    p = trial.suggest_int("p", 0, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 3)
    P = trial.suggest_int("P", 0, 2)
    D = trial.suggest_int("D", 0, 1)
    Q = trial.suggest_int("Q", 0, 2)

    try:
        model = train(train_data, p=p, d=d, q=q, P=P, D=D, Q=Q, s=7)
        metrics = validate(model, val_data)
        rmse = metrics["RMSE"]
        return rmse if np.isfinite(rmse) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=30):
    """Run Optuna Bayesian HPO search."""
    study = optuna.create_study(direction="minimize", study_name="sarima_numpy")
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
    """Ray Tune distributed HPO search."""
    def _trainable(config):
        try:
            m = train(train_data, p=config["p"], d=config["d"], q=config["q"],
                      P=config["P"], D=config["D"], Q=config["Q"], s=7)
            metrics = validate(m, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "p": tune.randint(0, 4), "d": tune.randint(0, 3),
        "q": tune.randint(0, 4), "P": tune.randint(0, 3),
        "D": tune.randint(0, 2), "Q": tune.randint(0, 3),
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
    """Compare different SARIMA parameter configurations and explain tradeoffs.

    Parameter Selection Reasoning:
        - (1,1,1)(1,1,1,7): The classic "airline model" - minimal but effective.
          One AR lag, one MA lag, one seasonal AR, one seasonal MA, with both
          regular and seasonal differencing. Captures basic trend + seasonality.

        - (2,1,0)(1,1,0,7): Pure AR model (no MA). Two non-seasonal lags plus
          one seasonal lag. Tests whether past values alone predict well enough.
          Best for: Series dominated by autoregressive structure.

        - (0,1,2)(0,1,1,7): Pure MA model. Only error-correction, no AR memory.
          Best for: Noisy series where recent errors are more informative than
          past values.

        - (1,1,1)(2,1,1,7): Extended seasonal AR with P=2. This means "same
          weekday last week" AND "same weekday two weeks ago" both predict
          "same weekday next week".
          Best for: Series with multi-week seasonal memory.

        - (1,0,1)(1,0,1,7): No differencing (d=0, D=0). Assumes stationarity.
          Best for: Pre-processed or naturally stationary seasonal series.
    """
    configs = [
        {"label": "SARIMA(1,1,1)(1,1,1,7) - Airline model",
         "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1},
        {"label": "SARIMA(2,1,0)(1,1,0,7) - Pure AR",
         "p": 2, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0},
        {"label": "SARIMA(0,1,2)(0,1,1,7) - Pure MA",
         "p": 0, "d": 1, "q": 2, "P": 0, "D": 1, "Q": 1},
        {"label": "SARIMA(1,1,1)(2,1,1,7) - Extended SAR",
         "p": 1, "d": 1, "q": 1, "P": 2, "D": 1, "Q": 1},
        {"label": "SARIMA(1,0,1)(1,0,1,7) - No differencing",
         "p": 1, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 1},
    ]

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: Seasonal ARIMA order effects")
    print("=" * 70)

    results_summary = []
    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            model = train(train_data, p=cfg["p"], d=cfg["d"], q=cfg["q"],
                          P=cfg["P"], D=cfg["D"], Q=cfg["Q"], s=7)
            metrics = validate(model, val_data)
            print(f"  AR: {model.ar_coeffs}  |  SAR: {model.sar_coeffs}")
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
    print("  - Seasonal differencing (D=1) is critical for removing repeating patterns")
    print("  - The airline model (1,1,1)(1,1,1,s) is a strong baseline for most seasonal series")
    print("  - Skipping differencing fails when trend or seasonality is present")


# ---------------------------------------------------------------------------
# Real-World Demo: Hospital Admissions Forecasting
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate SARIMA on a realistic hospital admissions forecasting scenario.

    Domain: Healthcare
    Task: Forecast daily hospital admissions for staffing and resource planning

    This demo generates synthetic data mimicking hospital admission patterns:
        - Weekly seasonality (more admissions on weekdays, fewer on weekends)
        - Gradual upward trend (aging population, hospital expansion)
        - Winter surge (respiratory illness season)
        - Random variation from emergency cases

    Business Context:
        Hospital administrators need admission forecasts for:
        - Nurse and doctor staffing schedules (plan shifts days ahead)
        - Bed capacity management (avoid overcrowding)
        - Supply chain (medications, surgical supplies, meals)
        - Budget planning and resource allocation
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Daily Hospital Admissions Forecasting")
    print("=" * 70)

    np.random.seed(55)

    # 2 years of daily data (730 days)
    n_days = 730
    t = np.arange(n_days, dtype=np.float64)

    # Base daily admissions: 50 patients/day (typical community hospital)
    base = 50.0

    # Growth trend: 3% annual increase
    daily_growth = 1.03 ** (1 / 365)
    trend = base * (daily_growth ** t - 1)

    # Weekly seasonality: weekdays have ~20% more admissions than weekends
    # WHY: Elective admissions happen Monday-Friday; emergency-only on weekends
    weekly = 0.10 * base * np.sin(2 * np.pi * t / 7 + np.pi / 4)

    # Annual seasonality: winter surge (+15%) from respiratory illness
    annual = 0.08 * base * np.cos(2 * np.pi * (t - 30) / 365)

    # Random noise: ~15% variation from emergency/trauma cases
    noise = np.random.normal(0, 0.08 * base, n_days)

    admissions = base + trend + weekly + annual + noise
    admissions = np.maximum(np.round(admissions), 0).astype(float)

    print(f"\nDataset: {n_days} days of hospital admission data")
    print(f"Admissions range: {admissions.min():.0f} - {admissions.max():.0f} patients/day")
    print(f"Mean daily admissions: {admissions.mean():.1f}")

    # Split: 80/10/10
    n_train = int(0.80 * n_days)
    n_val = int(0.10 * n_days)
    train_adm = admissions[:n_train]
    val_adm = admissions[n_train:n_train + n_val]
    test_adm = admissions[n_train + n_val:]

    print(f"Train: {len(train_adm)} days | Val: {len(val_adm)} days | "
          f"Test: {len(test_adm)} days")

    # Compare models with s=7 (weekly seasonality)
    print("\n--- Model Comparison ---")
    configs = [
        {"label": "SARIMA(1,1,1)(1,1,1,7)", "p": 1, "d": 1, "q": 1,
         "P": 1, "D": 1, "Q": 1},
        {"label": "SARIMA(2,1,0)(1,1,0,7)", "p": 2, "d": 1, "q": 0,
         "P": 1, "D": 1, "Q": 0},
    ]

    best_model = None
    best_rmse = float("inf")
    best_label = ""

    for cfg in configs:
        try:
            model = train(train_adm, p=cfg["p"], d=cfg["d"], q=cfg["q"],
                          P=cfg["P"], D=cfg["D"], Q=cfg["Q"], s=7)
            metrics = validate(model, val_adm)
            print(f"\n  {cfg['label']}:")
            print(f"    Val RMSE: {metrics['RMSE']:.1f} patients/day")
            print(f"    Val MAPE: {metrics['MAPE']:.2f}%")
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model = model
                best_label = cfg["label"]
        except Exception as e:
            print(f"\n  {cfg['label']}: FAILED - {e}")

    if best_model is not None:
        print(f"\n--- Best Model: {best_label} ---")
        test_metrics = test(best_model, test_adm, val_adm)
        print(f"  Test RMSE: {test_metrics['RMSE']:.1f} patients/day")
        print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")

        print("\n  Business Interpretation:")
        print(f"    Average forecast error: {test_metrics['MAE']:.1f} patients/day")
        print(f"    For a hospital admitting ~{admissions.mean():.0f} patients/day,")
        print(f"    the model is off by ~{test_metrics['MAE']:.0f} patients on average.")
        print(f"    This helps staffing within +/- {test_metrics['MAE']:.0f} nurses/beds.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Full pipeline: data -> HPO -> train -> validate -> test -> compare -> demo."""
    print("=" * 70)
    print("Seasonal ARIMA - NumPy From-Scratch Implementation")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=1000, seasonal_period=7)
    print(f"\nData splits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=25)
    bp = study.best_params

    print("\n--- Training Best Model ---")
    best_model = train(train_data, p=bp["p"], d=bp["d"], q=bp["q"],
                       P=bp["P"], D=bp["D"], Q=bp["Q"], s=7)
    print(f"AR coeffs: {best_model.ar_coeffs}")
    print(f"SAR coeffs: {best_model.sar_coeffs}")

    print("\n--- Validation ---")
    for k, v in validate(best_model, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Test ---")
    for k, v in test(best_model, test_data, val_data).items():
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
