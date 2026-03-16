"""
ARIMA (AutoRegressive Integrated Moving Average) - Statsmodels Implementation
==============================================================================

Theory & Mathematics:
    ARIMA(p, d, q) is one of the most widely used classical time series models.
    It combines three components:

    1. AR(p) - AutoRegressive: The current value depends linearly on p past values.
       y_t = c + phi_1*y_{t-1} + phi_2*y_{t-2} + ... + phi_p*y_{t-p} + epsilon_t

    2. I(d) - Integrated: The series is differenced d times to achieve stationarity.
       For d=1: y'_t = y_t - y_{t-1}
       For d=2: y''_t = y'_t - y'_{t-1}

    3. MA(q) - Moving Average: The current value depends on q past forecast errors.
       y_t = c + epsilon_t + theta_1*epsilon_{t-1} + ... + theta_q*epsilon_{t-q}

    Combined ARIMA(p,d,q):
       (1 - phi_1*B - ... - phi_p*B^p)(1-B)^d * y_t
           = c + (1 + theta_1*B + ... + theta_q*B^q) * epsilon_t

    where B is the backshift operator: B*y_t = y_{t-1}

    Parameter Estimation:
    - Maximum Likelihood Estimation (MLE) via conditional or exact methods
    - CSS (Conditional Sum of Squares) for initial estimates
    - Information criteria (AIC, BIC) for model selection

    Stationarity & Invertibility:
    - AR polynomial roots must lie outside the unit circle (stationarity)
    - MA polynomial roots must lie outside the unit circle (invertibility)

Business Use Cases:
    - Demand forecasting for inventory management
    - Short-term stock price prediction
    - Energy consumption forecasting
    - Sales forecasting and revenue planning
    - Macroeconomic indicator prediction (GDP, inflation)
    - Web traffic forecasting

Advantages:
    - Well-understood statistical foundations with confidence intervals
    - Works well for stationary or trend-stationary series
    - Interpretable parameters (AR coefficients, MA coefficients)
    - Information criteria (AIC/BIC) for automatic model selection
    - Handles linear trends through differencing
    - Efficient for short to medium time series

Disadvantages:
    - Assumes linearity in relationships
    - Cannot capture complex nonlinear patterns
    - Sensitive to outliers and structural breaks
    - Requires stationarity (or differencing to achieve it)
    - Not ideal for multiple seasonality patterns
    - Struggles with very long-term dependencies
    - Univariate by default (ARIMAX for exogenous variables)

Key Hyperparameters:
    - p (int): Order of the AutoRegressive component (0-5 typical)
    - d (int): Degree of differencing (0-2 typical)
    - q (int): Order of the Moving Average component (0-5 typical)
    - trend (str): 'n' (no trend), 'c' (constant), 't' (linear), 'ct' (both)

References:
    - Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis: Forecasting and Control.
    - Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd ed.
"""

import numpy as np  # NumPy for numerical array operations and statistical computations
import pandas as pd  # Pandas for time-indexed Series which statsmodels ARIMA requires
import warnings  # Warnings module to suppress convergence warnings during HPO sweeps
from typing import Dict, Tuple, Any, Optional  # Type hints for function signatures

import optuna  # Optuna for Bayesian hyperparameter optimization (TPE sampler by default)
import ray  # Ray distributed computing framework for parallel HPO
from ray import tune  # Ray Tune provides search spaces and trial management

# statsmodels ARIMA class implements Box-Jenkins methodology with MLE estimation
from statsmodels.tsa.arima.model import ARIMA
# Augmented Dickey-Fuller test for checking stationarity before modeling
from statsmodels.tsa.stattools import adfuller

# Suppress statsmodels convergence warnings that clutter output during HPO sweeps
# WHY: During Optuna/Ray searches, many ARIMA orders will fail to converge, and
# the warnings are not actionable in an automated search context
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
    """Generate synthetic time series with trend, seasonality, and noise.

    Components:
        - Linear trend: trend_slope * t
        - Sinusoidal seasonality: sin(2*pi*t / seasonal_period)
        - Gaussian noise: N(0, noise_std)

    Returns:
        Tuple of (train, val, test) as pd.Series with DatetimeIndex.
        Split is 70/15/15 chronological.
    """
    # Set random seed for reproducibility across runs
    # WHY: Ensures identical data splits for fair comparison between parameter configs
    np.random.seed(seed)

    # Create a DatetimeIndex starting from 2020-01-01 with the specified frequency
    # WHY: statsmodels ARIMA expects a DatetimeIndex for proper time series handling,
    # enabling automatic frequency detection and forecast date alignment
    dates = pd.date_range(start="2020-01-01", periods=n_points, freq=freq)

    # Integer time index for computing trend and seasonal components
    # WHY: float64 avoids integer overflow for large n_points and ensures smooth
    # multiplication with trend_slope
    t = np.arange(n_points, dtype=np.float64)

    # Linear trend component: simulates a steadily increasing baseline
    # WHY: Real-world time series (sales, GDP) often have an underlying growth trend
    # that ARIMA handles via differencing (the "I" component)
    trend = trend_slope * t

    # Sinusoidal seasonality: periodic oscillation with amplitude 2.0
    # WHY: Models repeating weekly/monthly patterns; the 2*pi/period formula creates
    # exactly one full cycle per seasonal_period timesteps
    seasonality = 2.0 * np.sin(2 * np.pi * t / seasonal_period)

    # Gaussian white noise: random perturbations around the signal
    # WHY: Simulates measurement error and unpredictable fluctuations;
    # noise_std controls signal-to-noise ratio (higher = harder forecasting)
    noise = np.random.normal(0, noise_std, n_points)

    # Combine all components: baseline of 10.0 keeps values positive (realistic for
    # business metrics like sales or demand which cannot be negative)
    values = 10.0 + trend + seasonality + noise

    # Wrap in pandas Series with DatetimeIndex for statsmodels compatibility
    # WHY: ARIMA model requires pd.Series with proper time index for
    # frequency inference and out-of-sample forecast date generation
    series = pd.Series(values, index=dates, name="value")

    # Chronological train/val/test split: 70% train, 15% validation, 15% test
    # WHY: Time series must be split chronologically (never randomly) to prevent
    # data leakage from future observations into training
    n_train = int(0.70 * n_points)  # 700 points for training
    n_val = int(0.15 * n_points)    # 150 points for validation

    train = series.iloc[:n_train]           # First 70% for model fitting
    val = series.iloc[n_train : n_train + n_val]  # Next 15% for hyperparameter selection
    test = series.iloc[n_train + n_val :]   # Final 15% for unbiased evaluation

    return train, val, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and SMAPE.

    Each metric captures a different aspect of forecast quality:
    - MAE: Average absolute error in original units (robust to outliers)
    - RMSE: Penalises large errors more heavily (sensitive to outliers)
    - MAPE: Percentage error relative to actual values (scale-independent)
    - SMAPE: Symmetric percentage error (handles near-zero actuals better)
    """
    # Convert to float64 arrays for numerical precision in metric calculations
    # WHY: Mixed types (int/float32) can cause subtle precision issues in division
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Element-wise forecast errors: positive = under-prediction, negative = over-prediction
    errors = actual - predicted

    # MAE (Mean Absolute Error): L1 loss, robust to outliers
    # WHY: Preferred when all errors should be weighted equally regardless of magnitude;
    # commonly used in business reporting because it is in the same units as the target
    mae = np.mean(np.abs(errors))

    # RMSE (Root Mean Squared Error): sqrt of L2 loss, penalises large errors
    # WHY: Differentiable everywhere (unlike MAE), making it suitable as an optimization
    # objective; also penalises large forecast misses disproportionately
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE (Mean Absolute Percentage Error): scale-free metric for cross-series comparison
    # WHY: Allows comparison across series with different scales (e.g., dollars vs units);
    # mask avoids division by zero where actual values are exactly 0
    mask = actual != 0  # Boolean mask to exclude zero actual values
    mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100 if mask.any() else np.inf

    # SMAPE (Symmetric MAPE): bounded [0, 200%], treats over/under prediction symmetrically
    # WHY: Regular MAPE is asymmetric (over-predictions penalised less than under-predictions);
    # SMAPE corrects this by using average of actual and predicted in denominator
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0  # Symmetric denominator
    denom = np.where(denom == 0, 1e-10, denom)  # Prevent division by zero
    smape = np.mean(np.abs(errors) / denom) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(train_data: pd.Series, p: int = 2, d: int = 1, q: int = 2,
          trend: str = "t") -> ARIMA:
    """Fit an ARIMA model on training data.

    Args:
        train_data: Training time series with DatetimeIndex.
        p: AR order - number of lagged values used for prediction.
            Higher p captures longer autoregressive memory.
        d: Differencing order - number of times to difference the series.
            d=1 removes linear trend, d=2 removes quadratic trend.
        q: MA order - number of lagged forecast errors used.
            Higher q smooths out more noise in residuals.
        trend: Trend specification ('n','c','t','ct').
            'n' = no trend, 'c' = constant/intercept only,
            't' = linear time trend, 'ct' = constant + linear trend.

    Returns:
        Fitted ARIMA results object with parameters, diagnostics, and forecast methods.
    """
    # Instantiate statsmodels ARIMA with specified order (p,d,q) and trend type
    # WHY: statsmodels handles differencing internally when d>0, applies MLE for
    # parameter estimation, and provides AIC/BIC for model comparison
    model = ARIMA(train_data, order=(p, d, q), trend=trend)

    # Fit the model using Maximum Likelihood Estimation (MLE)
    # WHY: MLE finds parameters that maximize the likelihood of observing the training
    # data under the ARIMA model; this is the standard estimation method because it is
    # statistically efficient (achieves the Cramer-Rao lower bound asymptotically)
    fitted = model.fit()

    return fitted


def validate(model, val_data: pd.Series) -> Dict[str, float]:
    """Generate forecasts for the validation period and compute metrics.

    Uses out-of-sample forecasting (forecast horizon = len(val_data)).
    WHY: Validation metrics guide hyperparameter selection (model order, trend type)
    without contaminating the test set which is reserved for final evaluation.
    """
    # Generate point forecasts for exactly len(val_data) steps ahead
    # WHY: .forecast() produces out-of-sample predictions beyond the training period;
    # these are true forecasts (not in-sample fitted values) so they honestly assess
    # how well the model generalises to unseen future data
    forecast = model.forecast(steps=len(val_data))

    # Compare forecasts against actual validation values
    # WHY: .values extracts the underlying numpy array from pandas Series,
    # ensuring consistent array types for metric computation
    return _compute_metrics(val_data.values, forecast.values)


def test(model, test_data: pd.Series, val_len: int = 0) -> Dict[str, float]:
    """Generate forecasts for the test period and compute metrics.

    Args:
        model: Fitted ARIMA results.
        test_data: Test time series.
        val_len: Length of validation set (to skip past it in forecasting).
            WHY: Since the model was trained only on training data, we must forecast
            through the entire validation period to reach the test period.
    """
    # Forecast through both validation and test periods
    # WHY: ARIMA forecasts are sequential - to get predictions at test time indices,
    # we must first forecast through the validation period; we cannot skip ahead
    total_steps = val_len + len(test_data)
    full_forecast = model.forecast(steps=total_steps)

    # Slice out only the test period forecasts (skip validation period)
    # WHY: We already evaluated validation metrics separately; here we need only
    # the test portion for unbiased final performance assessment
    test_forecast = full_forecast.iloc[val_len:]

    return _compute_metrics(test_data.values, test_forecast.values)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Optuna
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    train_data: pd.Series,
    val_data: pd.Series,
) -> float:
    """Optuna objective: minimize validation RMSE.

    WHY RMSE as objective: RMSE penalises large errors quadratically, which is
    desirable for demand/sales forecasting where large misses are costly.
    Optuna's TPE sampler efficiently explores the discrete (p,d,q) space.
    """
    # Suggest AR order p in range [0, 5]
    # WHY: p > 5 rarely improves ARIMA on typical business series and increases
    # parameter count, risking overfitting; p=0 allows pure MA or random walk models
    p = trial.suggest_int("p", 0, 5)

    # Suggest differencing order d in range [0, 2]
    # WHY: d=0 for stationary series, d=1 for linear trend, d=2 for quadratic trend;
    # d > 2 is almost never needed and can over-difference (introduce spurious patterns)
    d = trial.suggest_int("d", 0, 2)

    # Suggest MA order q in range [0, 5]
    # WHY: q controls how many past forecast errors influence current prediction;
    # similar to p, q > 5 is rarely beneficial for most real-world series
    q = trial.suggest_int("q", 0, 5)

    # Suggest trend type from four options
    # WHY: Trend specification interacts with differencing - e.g., trend='t' with d=1
    # can cause redundancy (both model a linear trend), so the search explores
    # combinations to find the best fit
    trend = trial.suggest_categorical("trend", ["n", "c", "t", "ct"])

    try:
        # Train ARIMA with suggested hyperparameters
        model = train(train_data, p=p, d=d, q=q, trend=trend)
        # Evaluate on validation set
        metrics = validate(model, val_data)
        # Return RMSE as the objective to minimize
        return metrics["RMSE"]
    except Exception:
        # Return infinity for invalid ARIMA orders (e.g., non-invertible MA polynomial)
        # WHY: Some (p,d,q) combinations cause numerical issues or fail to converge;
        # returning inf tells Optuna this configuration is invalid without crashing
        return float("inf")


def run_optuna(
    train_data: pd.Series,
    val_data: pd.Series,
    n_trials: int = 50,
) -> optuna.Study:
    """Run Optuna hyperparameter search.

    WHY Optuna over grid search: Optuna uses Tree-structured Parzen Estimator (TPE)
    which learns which hyperparameter regions are promising and focuses sampling there,
    finding good solutions in far fewer trials than exhaustive grid search.
    """
    # Create a study that minimizes the objective (lower RMSE = better forecast)
    # WHY: direction="minimize" because RMSE should be as small as possible
    study = optuna.create_study(direction="minimize", study_name="arima_statsmodels")

    # Run the optimization for n_trials
    # WHY: Lambda wraps the objective to pass train/val data without making them
    # part of the trial's parameter space
    study.optimize(
        lambda trial: optuna_objective(trial, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,  # Visual feedback during potentially long searches
    )

    # Report best results found
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(
    train_data: pd.Series,
    val_data: pd.Series,
    num_samples: int = 20,
) -> Any:
    """Ray Tune hyperparameter search.

    WHY Ray Tune: Provides distributed parallel trial execution across CPUs/GPUs.
    For ARIMA (CPU-bound), this means running multiple (p,d,q) configurations
    simultaneously, reducing wall-clock time proportional to available cores.
    """

    def _trainable(config: Dict) -> None:
        """Inner training function that Ray Tune calls for each trial.

        WHY a nested function: Captures train_data and val_data from the
        enclosing scope, avoiding serialization of large datasets in config dict.
        """
        try:
            # Train ARIMA with this trial's hyperparameter configuration
            model = train(
                train_data,
                p=config["p"],
                d=config["d"],
                q=config["q"],
                trend=config["trend"],
            )
            # Evaluate on validation set and report metrics to Ray Tune
            metrics = validate(model, val_data)
            # ray.train.report sends metrics back to the Tune controller
            # WHY: Ray Tune uses these reported metrics to track trial performance
            # and determine the best configuration
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            # Report very large loss for failed configurations
            # WHY: Unlike Optuna which accepts return values, Ray Tune requires
            # explicit reporting; inf signals a failed trial
            ray.train.report({"rmse": float("inf"), "mae": float("inf")})

    # Define the hyperparameter search space
    # WHY: Same ranges as Optuna for fair comparison between optimization frameworks
    search_space = {
        "p": tune.randint(0, 6),              # AR order: uniform random integer [0, 5]
        "d": tune.randint(0, 3),              # Differencing: [0, 2]
        "q": tune.randint(0, 6),              # MA order: [0, 5]
        "trend": tune.choice(["n", "c", "t", "ct"]),  # Trend type: categorical choice
    }

    # Initialize Ray runtime if not already running
    # WHY: Ray needs a running cluster (even single-node) to distribute trials;
    # ignore_reinit_error=True prevents crashes if Ray was initialized elsewhere
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    # Create and run the Tuner
    # WHY: Tuner is Ray Tune's high-level API that manages trial scheduling,
    # resource allocation, and result collection
    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="rmse",           # Metric to optimize
            mode="min",              # Minimize RMSE (lower = better)
            num_samples=num_samples, # Total number of trial configurations to evaluate
        ),
    )

    # Execute all trials (potentially in parallel across CPUs)
    results = tuner.fit()

    # Extract and display the best result
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    print(f"Ray Tune Best config: {best.config}")
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(train_data: pd.Series, val_data: pd.Series) -> None:
    """Compare different ARIMA parameter configurations and explain tradeoffs.

    This function trains ARIMA models with several representative (p,d,q) orders
    and trend settings, then compares their validation performance to illustrate
    how each hyperparameter affects model behavior.

    Parameter Selection Reasoning:
        - (1,1,1): Minimal model. AR(1) captures first-order autocorrelation,
          d=1 removes linear trend, MA(1) smooths one-step residuals.
          Best for: simple series with weak temporal dependence.

        - (2,1,2): Balanced model. AR(2) captures two-step memory (e.g., the
          effect of the value two days ago), MA(2) smooths two-step errors.
          Best for: moderately complex series with medium-range dependence.

        - (5,1,0): Pure AR model. AR(5) captures up to 5-step memory without
          any MA component. Equivalent to fitting a linear regression on 5 lags
          of the differenced series.
          Best for: series where past values strongly predict future values
          (autoregressive dominant), like temperature or slowly-varying processes.

        - (0,1,3): Pure MA model. No autoregressive memory, only 3 past errors
          used. Equivalent to exponential smoothing variants.
          Best for: series dominated by random shocks where the current value is
          best predicted by recent forecast errors (moving-average dominant).

        - (3,1,1) with trend='ct': AR(3) plus constant+linear trend. The explicit
          trend parameter handles drift in the differenced series.
          Best for: series with strong persistent trend that differencing alone
          does not fully remove.
    """
    # Define parameter configurations with descriptive labels
    # WHY: Each config tests a different modeling philosophy (AR-heavy, MA-heavy,
    # balanced, trend-augmented) to show how the choice affects accuracy
    configs = [
        {"label": "ARIMA(1,1,1) - Minimal",
         "p": 1, "d": 1, "q": 1, "trend": "n"},
        {"label": "ARIMA(2,1,2) - Balanced",
         "p": 2, "d": 1, "q": 2, "trend": "c"},
        {"label": "ARIMA(5,1,0) - Pure AR (long memory)",
         "p": 5, "d": 1, "q": 0, "trend": "c"},
        {"label": "ARIMA(0,1,3) - Pure MA (shock-driven)",
         "p": 0, "d": 1, "q": 3, "trend": "c"},
        {"label": "ARIMA(3,1,1) trend=ct - Trend-augmented",
         "p": 3, "d": 1, "q": 1, "trend": "ct"},
    ]

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: How ARIMA order affects forecast accuracy")
    print("=" * 70)

    # Store results for final summary table
    results_summary = []

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            # Fit the ARIMA model with this specific configuration
            model = train(train_data, p=cfg["p"], d=cfg["d"],
                          q=cfg["q"], trend=cfg["trend"])

            # Evaluate on validation data
            metrics = validate(model, val_data)

            # Display AIC/BIC alongside forecast metrics
            # WHY: AIC/BIC measure in-sample fit quality penalised by model complexity;
            # comparing them with out-of-sample RMSE reveals whether in-sample criteria
            # actually predict out-of-sample performance
            print(f"  AIC: {model.aic:.2f}  |  BIC: {model.bic:.2f}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            results_summary.append({
                "config": cfg["label"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "AIC": model.aic,
                "BIC": model.bic,
            })
        except Exception as e:
            # Some configurations may fail (e.g., non-invertible polynomials)
            # WHY: Not all (p,d,q) combinations produce valid models; reporting the
            # failure teaches which configurations are unstable
            print(f"  FAILED: {e}")
            results_summary.append({
                "config": cfg["label"],
                "RMSE": float("inf"),
                "MAE": float("inf"),
                "AIC": float("inf"),
                "BIC": float("inf"),
            })

    # Print ranked summary
    print("\n" + "-" * 70)
    print("RANKING (by validation RMSE):")
    # Sort by RMSE ascending (best first)
    results_summary.sort(key=lambda x: x["RMSE"])
    for i, r in enumerate(results_summary, 1):
        print(f"  {i}. {r['config']}: RMSE={r['RMSE']:.4f}, "
              f"MAE={r['MAE']:.4f}, AIC={r['AIC']:.1f}")

    print("\nKey Takeaways:")
    print("  - Lower AIC/BIC does not always mean lower out-of-sample RMSE")
    print("  - AR-dominant models work well when past values have strong predictive power")
    print("  - MA-dominant models work well when the series is driven by random shocks")
    print("  - Adding trend='ct' helps when differencing alone leaves residual drift")
    print("  - Simpler models (1,1,1) often generalise better than complex ones")


# ---------------------------------------------------------------------------
# Real-World Demo: Monthly Airline Passenger Forecasting
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate ARIMA on a realistic airline passenger forecasting scenario.

    Domain: Aviation / Airlines
    Task: Forecast monthly passenger counts for capacity planning

    This demo generates synthetic data mimicking the classic Box-Jenkins airline
    passenger dataset characteristics:
        - Strong upward trend (growing air travel market)
        - Annual seasonality (summer peaks, winter troughs)
        - Multiplicative noise (variance grows with level)

    Business Context:
        Airlines need accurate passenger forecasts 3-12 months ahead for:
        - Fleet scheduling and aircraft allocation
        - Staffing decisions (pilots, cabin crew, ground staff)
        - Revenue management and pricing strategy
        - Airport gate and terminal capacity planning
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Monthly Airline Passenger Forecasting")
    print("=" * 70)

    # ---- Generate domain-specific synthetic data ----
    np.random.seed(123)

    # 10 years of monthly data (120 months)
    # WHY: Airline forecasting typically uses 5-10 years of history to capture
    # annual seasonality patterns reliably
    n_months = 120
    dates = pd.date_range(start="2014-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months, dtype=np.float64)

    # Base passenger count: 100,000 passengers/month starting point
    # WHY: Realistic order of magnitude for a mid-size regional airport
    base = 100_000.0

    # Growth trend: 2% compound monthly growth rate (approx 27% annual growth)
    # WHY: Air travel has historically grown 5-7% annually; we use a higher rate
    # to make the trend clearly visible in the data
    growth = base * (1.02 ** (t / 12))  # Monthly compounding at annual rate

    # Annual seasonality: summer peak (+30%), winter trough (-20%)
    # WHY: Air travel peaks in June-August (vacations) and dips in January-February;
    # the sinusoidal pattern with phase shift aligns peaks with summer months
    seasonal_amplitude = 0.25  # +/- 25% seasonal variation
    # Phase shift of -pi/2 so the peak occurs around month 6 (July)
    seasonality = seasonal_amplitude * np.sin(2 * np.pi * t / 12 - np.pi / 2)

    # Multiplicative noise: variance proportional to the level
    # WHY: Passenger counts have heteroscedastic noise - a 5% error on 200k passengers
    # is much larger in absolute terms than on 100k passengers
    noise = np.random.normal(0, 0.05, n_months)  # 5% coefficient of variation

    # Combine components multiplicatively (realistic for count data)
    # WHY: Multiplicative composition means seasonality scales with the trend,
    # matching real-world airline data patterns
    passengers = growth * (1 + seasonality) * (1 + noise)
    passengers = np.round(passengers).astype(int)  # Passengers are integers

    # Create named DataFrame for clarity
    df = pd.DataFrame({
        "date": dates,
        "passengers": passengers,
        "trend_component": growth,
        "seasonal_factor": 1 + seasonality,
    })

    # Create pandas Series with DatetimeIndex (required by statsmodels ARIMA)
    series = pd.Series(passengers.astype(float), index=dates, name="passengers")

    print(f"\nDataset: {n_months} months of airline passenger data")
    print(f"Date range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    print(f"Passenger range: {passengers.min():,} - {passengers.max():,}")
    print(f"Mean passengers: {passengers.mean():,.0f}")

    # Split: 80% train, 10% validation, 10% test (last 12 months held out)
    # WHY: For monthly data, we want at least 2-3 full seasonal cycles in training
    # and at least one full cycle in test for reliable seasonal evaluation
    n_train = int(0.80 * n_months)  # 96 months (8 years)
    n_val = int(0.10 * n_months)    # 12 months (1 year)

    train_series = series.iloc[:n_train]
    val_series = series.iloc[n_train:n_train + n_val]
    test_series = series.iloc[n_train + n_val:]

    print(f"\nTrain: {len(train_series)} months | Val: {len(val_series)} months | "
          f"Test: {len(test_series)} months")

    # ---- Stationarity Assessment ----
    # Run ADF test to determine if differencing is needed
    # WHY: ARIMA requires stationarity; the ADF test formally checks for unit roots.
    # If p-value > 0.05, the series is likely non-stationary and needs differencing.
    print("\n--- Stationarity Check ---")
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(train_series, autolag="AIC")
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  p-value: {adf_pvalue:.4f}")
    print(f"  Stationary: {'Yes' if adf_pvalue < 0.05 else 'No - differencing needed'}")

    # Also check first difference for stationarity
    diff_series = train_series.diff().dropna()
    adf_diff = adfuller(diff_series, autolag="AIC")
    print(f"  After d=1 differencing - ADF p-value: {adf_diff[1]:.4f}")

    # ---- Train multiple ARIMA configurations ----
    # WHY: Compare how different orders handle the trend+seasonality pattern
    print("\n--- Model Comparison ---")
    model_configs = [
        {"label": "ARIMA(1,1,1)", "p": 1, "d": 1, "q": 1, "trend": "c"},
        {"label": "ARIMA(2,1,2)", "p": 2, "d": 1, "q": 2, "trend": "c"},
        {"label": "ARIMA(1,1,0)", "p": 1, "d": 1, "q": 0, "trend": "c"},
    ]

    best_model = None
    best_rmse = float("inf")

    for cfg in model_configs:
        try:
            # Fit model on training data
            model = train(train_series, p=cfg["p"], d=cfg["d"],
                          q=cfg["q"], trend=cfg["trend"])

            # Validate on held-out validation set
            metrics = validate(model, val_series)

            print(f"\n  {cfg['label']}:")
            print(f"    AIC: {model.aic:.1f} | BIC: {model.bic:.1f}")
            print(f"    Val RMSE: {metrics['RMSE']:,.0f} passengers")
            print(f"    Val MAPE: {metrics['MAPE']:.2f}%")

            # Track the best model by validation RMSE
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model = model
                best_label = cfg["label"]
        except Exception as e:
            print(f"\n  {cfg['label']}: FAILED - {e}")

    # ---- Final Evaluation on Test Set ----
    if best_model is not None:
        print(f"\n--- Best Model: {best_label} ---")
        test_metrics = test(best_model, test_series, val_len=len(val_series))
        print(f"  Test RMSE: {test_metrics['RMSE']:,.0f} passengers")
        print(f"  Test MAE:  {test_metrics['MAE']:,.0f} passengers")
        print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")

        print("\n  Business Interpretation:")
        print(f"    The model predicts monthly passenger counts within "
              f"{test_metrics['MAPE']:.1f}% on average.")
        print(f"    For a month expecting ~{int(passengers[-1]):,} passengers, "
              f"the typical error is ~{int(test_metrics['MAE']):,} passengers.")
        print(f"    This level of accuracy is {'acceptable' if test_metrics['MAPE'] < 10 else 'marginal'} "
              f"for airline capacity planning (target: <10% MAPE).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline: data generation -> HPO -> train -> validate -> test.

    Execution flow:
        1. Generate synthetic time series data
        2. Check stationarity via ADF test
        3. Run Optuna HPO to find best (p,d,q) and trend
        4. Train final model with best hyperparameters
        5. Evaluate on validation set
        6. Evaluate on held-out test set
        7. Run Ray Tune for comparison
        8. Compare different parameter configurations
        9. Run real-world airline passenger demo
    """
    print("=" * 70)
    print("ARIMA - Statsmodels Implementation")
    print("=" * 70)

    # 1. Generate synthetic data with trend + seasonality + noise
    # WHY: Controlled synthetic data lets us verify the model works correctly
    # before applying it to real-world data where ground truth is unknown
    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nData splits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    # 2. Stationarity check using Augmented Dickey-Fuller test
    # WHY: The ADF test detects unit roots - if the p-value > 0.05, the series
    # is non-stationary and differencing (d >= 1) is likely needed
    adf_result = adfuller(train_data, autolag="AIC")
    print(f"\nADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    print(f"Stationary: {'Yes' if adf_result[1] < 0.05 else 'No (differencing needed)'}")

    # 3. Optuna HPO to find the best ARIMA configuration
    # WHY: Instead of manually trying (p,d,q) combinations, Optuna efficiently
    # searches the space using Bayesian optimization (TPE sampler)
    print("\n--- Optuna Hyperparameter Search ---")
    study = run_optuna(train_data, val_data, n_trials=30)
    best_params = study.best_params

    # 4. Train with best hyperparameters found by Optuna
    print("\n--- Training with Best Parameters ---")
    best_model = train(
        train_data,
        p=best_params["p"],
        d=best_params["d"],
        q=best_params["q"],
        trend=best_params["trend"],
    )
    # Display the selected model order and information criteria
    # WHY: AIC/BIC help assess model complexity vs fit tradeoff;
    # lower values indicate better balance between fit and parsimony
    print(f"Model order: ({best_params['p']}, {best_params['d']}, {best_params['q']})")
    print(f"AIC: {best_model.aic:.2f}, BIC: {best_model.bic:.2f}")

    # 5. Validate: evaluate on validation set
    print("\n--- Validation Metrics ---")
    val_metrics = validate(best_model, val_data)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 6. Test: evaluate on held-out test set for unbiased performance estimate
    # WHY: Test metrics are the most honest assessment since the test data was
    # never used for model selection or hyperparameter tuning
    print("\n--- Test Metrics ---")
    test_metrics = test(best_model, test_data, val_len=len(val_data))
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 7. Ray Tune (optional, runs if Ray is available)
    # WHY: Demonstrates an alternative HPO framework that supports distributed
    # parallel search across multiple workers
    print("\n--- Ray Tune Search ---")
    try:
        ray_results = ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    # 8. Compare specific parameter configurations
    # WHY: Educational - shows how different (p,d,q) choices affect forecast quality
    compare_parameter_sets(train_data, val_data)

    # 9. Real-world demo with airline passenger data
    # WHY: Demonstrates practical application with domain-specific data generation,
    # business-relevant metrics interpretation, and actionable insights
    real_world_demo()

    print("\nDone.")


if __name__ == "__main__":
    main()
