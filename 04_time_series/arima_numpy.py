"""
ARIMA (AutoRegressive Integrated Moving Average) - NumPy From-Scratch Implementation
======================================================================================

Theory & Mathematics:
    This module implements ARIMA(p, d, q) from scratch using only NumPy.

    1. Differencing (I component, order d):
       Transforms non-stationary series to stationary by computing successive
       differences. For d=1: z_t = y_t - y_{t-1}.

    2. AR Coefficients via Yule-Walker Equations:
       The Yule-Walker equations relate the autocorrelation function to the AR
       coefficients:
           R * phi = r
       where:
           R = Toeplitz autocorrelation matrix of size p x p
           phi = [phi_1, phi_2, ..., phi_p]^T  (AR coefficients)
           r = [gamma(1), gamma(2), ..., gamma(p)]^T  (autocorrelation vector)

       Solving: phi = R^{-1} * r

    3. MA Coefficients via Innovations Algorithm:
       The innovations algorithm recursively computes MA coefficients by
       minimising one-step-ahead prediction errors:
           - Compute one-step-ahead prediction errors (innovations)
           - Estimate theta coefficients from the covariance structure
           - Iterate until convergence

    4. Combined Forecast:
       y_hat_t = c + sum_{i=1..p} phi_i * y_{t-i}
                   + sum_{j=1..q} theta_j * epsilon_{t-j}
       where epsilon_t = y_t - y_hat_t are the residuals.

    5. Inverse Differencing:
       After forecasting on the differenced series, reverse the differencing
       to recover the original scale.

Business Use Cases:
    - Demand forecasting in retail and supply chain
    - Economic indicator prediction
    - Energy load forecasting
    - Website traffic prediction

Advantages:
    - Full transparency into algorithm internals
    - No external dependencies beyond NumPy
    - Educational value for understanding ARIMA mechanics
    - Can be customised for special data structures

Disadvantages:
    - Less numerically stable than optimised library implementations
    - No confidence intervals (would require bootstrap)
    - Slower than compiled library code
    - Innovations algorithm is an approximation for MA estimation

Key Hyperparameters:
    - p (int): AR order
    - d (int): Differencing order
    - q (int): MA order
    - include_constant (bool): Whether to include a constant/intercept

References:
    - Brockwell, P.J. & Davis, R.A. (2002). Introduction to Time Series and Forecasting.
    - Hamilton, J.D. (1994). Time Series Analysis.
"""

import numpy as np  # NumPy for all numerical operations - this is our only computation dependency
import pandas as pd  # Pandas only used for date handling in the real-world demo
import warnings  # Suppress warnings during automated HPO sweeps
from typing import Dict, Tuple, Any, Optional, List  # Type hints for function signatures
from dataclasses import dataclass, field  # Dataclass for clean model state containers

import optuna  # Optuna Bayesian HPO framework (TPE sampler)
import ray  # Ray distributed computing for parallel HPO
from ray import tune  # Ray Tune search space definitions

# Suppress all warnings globally
# WHY: During Optuna/Ray sweeps, many ARIMA orders will produce numerical warnings
# (e.g., near-singular matrices in Yule-Walker) that clutter output without being actionable
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
    """Generate synthetic time series with trend, seasonality, and noise.

    Returns:
        Tuple of (train, val, test) as 1-D numpy arrays.
        Split is 70/15/15 chronological.
    """
    # Set random seed for reproducibility across runs
    # WHY: Ensures identical data splits for fair comparison between parameter configs
    np.random.seed(seed)

    # Create integer time index as float64 for numerical precision
    # WHY: float64 prevents integer overflow and enables smooth multiplication
    # with trend_slope for large n_points values
    t = np.arange(n_points, dtype=np.float64)

    # Linear trend: simulates steady growth over time (e.g., increasing demand)
    # WHY: trend_slope * t creates a linearly increasing baseline; this is exactly
    # the type of non-stationarity that ARIMA's differencing (d >= 1) is designed to handle
    trend = trend_slope * t

    # Sinusoidal seasonality: periodic oscillation with period = seasonal_period
    # WHY: The 2*pi/seasonal_period formula creates exactly one full sine cycle per
    # seasonal_period timesteps; amplitude of 2.0 makes seasonality clearly visible
    # relative to the noise level
    seasonality = 2.0 * np.sin(2 * np.pi * t / seasonal_period)

    # Gaussian white noise: random perturbations around the signal
    # WHY: Simulates real-world measurement error and unpredictable fluctuations;
    # noise_std=0.5 gives a moderate signal-to-noise ratio for testing
    noise = np.random.normal(0, noise_std, n_points)

    # Combine components: baseline + trend + seasonality + noise
    # WHY: The constant 10.0 ensures all values are positive (realistic for
    # business metrics like sales or demand counts)
    values = 10.0 + trend + seasonality + noise

    # Chronological split: 70% train, 15% val, 15% test
    # WHY: Time series MUST be split chronologically (never randomly shuffled)
    # to prevent data leakage from future observations into the training set
    n_train = int(0.70 * n_points)  # 700 points for model fitting
    n_val = int(0.15 * n_points)    # 150 points for hyperparameter selection

    train = values[:n_train]                   # First 70% for training
    val = values[n_train : n_train + n_val]    # Next 15% for validation
    test = values[n_train + n_val :]           # Final 15% for unbiased testing

    return train, val, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and SMAPE forecast evaluation metrics.

    Each metric captures a different dimension of forecast quality:
    - MAE: Average error magnitude in original units (interpretable, robust)
    - RMSE: Emphasises large errors (useful when big misses are costly)
    - MAPE: Percentage error for scale-free comparison across series
    - SMAPE: Symmetric version of MAPE, bounded [0, 200%]
    """
    # Convert inputs to float64 arrays for consistent numerical precision
    # WHY: Input arrays might be int or float32; float64 prevents precision loss
    # during division operations in MAPE/SMAPE calculations
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Element-wise errors: actual - predicted
    # WHY: Positive error = model under-predicted, negative = over-predicted
    errors = actual - predicted

    # MAE: Mean Absolute Error - average of |error| across all forecasts
    # WHY: The L1 metric is robust to outliers and in the same units as the target;
    # preferred in business reporting for its intuitive interpretation
    mae = np.mean(np.abs(errors))

    # RMSE: Root Mean Squared Error - sqrt(mean(error^2))
    # WHY: The L2 metric penalises large errors quadratically, making it sensitive
    # to outlier forecasts; commonly used as an optimization objective because it
    # is differentiable everywhere (unlike MAE at zero)
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE: Mean Absolute Percentage Error
    # WHY: Scale-independent metric enabling comparison across different series;
    # we mask out zero actual values to avoid division by zero
    mask = actual != 0  # Boolean array: True where actual is nonzero
    mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100 if mask.any() else np.inf

    # SMAPE: Symmetric MAPE - uses average of |actual| and |predicted| in denominator
    # WHY: Standard MAPE is asymmetric (over-predictions penalised less than under-predictions);
    # SMAPE corrects this bias and is bounded [0%, 200%] by construction
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0  # Symmetric denominator
    denom = np.where(denom == 0, 1e-10, denom)  # Replace zeros with tiny value
    smape = np.mean(np.abs(errors) / denom) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


# ---------------------------------------------------------------------------
# ARIMA from scratch
# ---------------------------------------------------------------------------

@dataclass
class ARIMAModel:
    """Container for a fitted ARIMA model (from scratch).

    Stores all model parameters and state needed for forecasting:
    - AR/MA coefficients estimated during training
    - Residual history for MA component in forecasting
    - Differencing metadata for undifferencing forecasts back to original scale
    """
    p: int = 2                  # AR order: how many past values to use
    d: int = 1                  # Differencing order: how many times to difference
    q: int = 2                  # MA order: how many past residuals to use
    include_constant: bool = True  # Whether the model includes an intercept term
    # AR coefficients [phi_1, phi_2, ..., phi_p] estimated via Yule-Walker
    ar_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    # MA coefficients [theta_1, theta_2, ..., theta_q] estimated via innovations algorithm
    ma_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    # Constant/intercept term c in the ARIMA equation
    constant: float = 0.0
    # Residuals from the fitted model (needed for MA forecasting component)
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    # Stored differencing history for reversing the differencing operation
    diff_history: List[np.ndarray] = field(default_factory=list)
    # Last d+1 values from original series needed for undifferencing
    original_last_values: List[float] = field(default_factory=list)
    # The differenced series used for AR/MA estimation
    differenced_series: np.ndarray = field(default_factory=lambda: np.array([]))


def _difference(series: np.ndarray, d: int) -> Tuple[np.ndarray, List[float]]:
    """Apply differencing d times. Returns differenced series and initial values.

    Differencing transforms non-stationary series to stationary:
        d=1: z_t = y_t - y_{t-1} (removes linear trend)
        d=2: z_t = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) (removes quadratic trend)

    WHY: ARIMA assumes stationarity in the differenced series. Without differencing,
    a trending series would have a time-varying mean, violating the AR model assumptions.
    """
    # Store initial values at each differencing level for later undifferencing
    # WHY: When we reverse differencing to get forecasts on the original scale,
    # we need these anchor values to reconstruct the cumulative sums
    initial_values = []

    # Make a copy to avoid modifying the original data
    # WHY: np.diff returns a new array, but we want to preserve the first value
    # at each level before it is lost to differencing
    result = series.copy()

    for _ in range(d):
        # Save the first value before this round of differencing
        # WHY: Each differencing level loses one data point from the front;
        # we store it so we can reverse the operation later
        initial_values.append(result[0])

        # Apply first-order differencing: z_t = y_t - y_{t-1}
        # WHY: np.diff computes z[i] = result[i+1] - result[i], producing
        # a series that is shorter by 1 element; this removes one order of trend
        result = np.diff(result)

    return result, initial_values


def _undifference(forecasts: np.ndarray, last_values: List[float],
                  d: int, original_tail: np.ndarray) -> np.ndarray:
    """Reverse differencing to recover original scale.

    WHY: Forecasts are generated on the differenced scale (z_hat). To get predictions
    in the original units (y_hat), we need to cumulatively sum the differenced forecasts
    starting from the last observed original value.

    For d=1: y_hat_t = y_{T} + sum_{i=1..t} z_hat_i
    where y_T is the last observed value and z_hat are the differenced forecasts.
    """
    result = forecasts.copy()  # Avoid modifying the input array

    for i in range(d - 1, -1, -1):
        # For each level of differencing (innermost first), cumulatively sum
        # WHY: Differencing is subtraction, so its inverse is cumulative summation.
        # We prepend the anchor value (last observed at this level) and take cumsum.
        if i == 0:
            # Anchor is the last value of the original (undifferenced) series
            anchor = original_tail[-1]
        else:
            # For higher-order differencing, use the same anchor strategy
            # WHY: In practice, d > 1 is rare (d=2 already handles quadratic trends)
            anchor = original_tail[-1]

        # Prepend anchor, compute cumulative sum, then remove the anchor from output
        # WHY: cumsum([anchor, z1, z2, ...]) = [anchor, anchor+z1, anchor+z1+z2, ...]
        # We take [1:] to skip the anchor itself, leaving [anchor+z1, anchor+z1+z2, ...]
        result = np.cumsum(np.concatenate([[anchor], result]))[1:]

    return result


def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation for lags 0..max_lag.

    The autocorrelation at lag k is defined as:
        rho(k) = Cov(x_t, x_{t-k}) / Var(x_t)
               = E[(x_t - mu)(x_{t-k} - mu)] / E[(x_t - mu)^2]

    WHY: Autocorrelation reveals the temporal dependence structure of the series.
    High autocorrelation at lag k means the value k steps ago is a good predictor
    of the current value - this directly determines the AR order p.
    """
    n = len(x)

    # Compute the sample mean of the series
    # WHY: We need to center the data (subtract mean) before computing covariances;
    # autocorrelation is defined in terms of deviations from the mean
    mean = np.mean(x)

    # Center the series by subtracting the mean
    # WHY: Centering ensures we measure covariance rather than raw cross-products;
    # without centering, the autocorrelation would be biased by the mean level
    x_centered = x - mean

    # Compute variance (denominator for normalizing covariances to correlations)
    # WHY: Dividing by n (not n-1) gives the population variance estimator,
    # consistent with the standard autocorrelation formula in time series analysis
    var = np.sum(x_centered ** 2) / n

    # Handle degenerate case where series has zero variance
    # WHY: A constant series has no temporal structure; return zero autocorrelation
    # to avoid division by zero in the normalization step
    if var == 0:
        return np.zeros(max_lag + 1)

    # Compute autocorrelation for each lag from 0 to max_lag
    # WHY: We need rho(0) through rho(p) for the Yule-Walker equations;
    # rho(0) = 1 by definition (series is perfectly correlated with itself)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        # Cross-product of x_t and x_{t-lag}, summed over all valid time indices
        # WHY: x_centered[:n-lag] aligns with x_centered[lag:] to compute the
        # lagged cross-product; dividing by (n * var) normalizes to [-1, 1]
        acf[lag] = np.sum(x_centered[:n - lag] * x_centered[lag:]) / (n * var)

    return acf


def _yule_walker(x: np.ndarray, p: int) -> np.ndarray:
    """Estimate AR coefficients using Yule-Walker equations.

    Solves the linear system R * phi = r where:
        R = Toeplitz autocorrelation matrix: R[i,j] = rho(|i-j|)
        r = autocorrelation vector: r[k] = rho(k) for k=1..p
        phi = AR coefficients to solve for

    WHY Yule-Walker over OLS regression:
        The Yule-Walker method exploits the Toeplitz structure of the autocorrelation
        matrix, which is theoretically guaranteed to be positive semi-definite. This
        ensures the resulting AR coefficients satisfy stationarity conditions.
        OLS regression on lagged values would give similar estimates but without
        the stationarity guarantee.
    """
    # Handle the trivial case where no AR component is requested
    if p == 0:
        return np.array([])

    # Compute autocorrelation values up to lag p
    # WHY: We need rho(0) through rho(p); rho(0)=1 goes on the diagonal of R,
    # and rho(1) through rho(p) form both the off-diagonals of R and the RHS vector r
    acf = _autocorrelation(x, p)

    # Build the Toeplitz autocorrelation matrix R[i,j] = rho(|i-j|)
    # WHY: The Toeplitz structure means R is symmetric with constant diagonals;
    # R[0,0] = R[1,1] = ... = rho(0) = 1 (the main diagonal)
    # R[0,1] = R[1,0] = rho(1), etc.
    R = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            # |i-j| gives the lag for this matrix element
            # WHY: The autocorrelation between x_t and x_{t-k} depends only on
            # the absolute lag |k|, which is the defining property of stationarity
            R[i, j] = acf[abs(i - j)]

    # Right-hand side vector: autocorrelations at lags 1 through p
    # WHY: The Yule-Walker equations state that the expected autocorrelation at
    # lag k equals the weighted sum of autocorrelations at neighboring lags,
    # with weights being the AR coefficients
    r = acf[1 : p + 1]

    try:
        # Solve the linear system R * phi = r using LU decomposition
        # WHY: np.linalg.solve is faster and more numerically stable than
        # computing R^{-1} explicitly; it uses partial pivoting for stability
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        # Fallback to least-squares if R is singular (happens with very short series)
        # WHY: A singular R matrix means the autocorrelations are linearly dependent,
        # which can occur with very short series or series with exact periodicity;
        # lstsq finds the minimum-norm solution even when R is rank-deficient
        phi = np.linalg.lstsq(R, r, rcond=None)[0]

    return phi


def _estimate_ma_innovations(residuals: np.ndarray, q: int,
                             max_iter: int = 50) -> np.ndarray:
    """Estimate MA coefficients using the innovations algorithm.

    The innovations algorithm recursively estimates one-step prediction
    coefficients from the autocovariance structure of the residuals.

    WHY the innovations algorithm:
        After removing the AR component, the residuals should approximate a
        MA(q) process. The innovations algorithm estimates MA coefficients by
        recursively computing optimal linear predictors from the autocovariance
        function. It is simpler to implement from scratch than full MLE while
        giving reasonable MA estimates for moderate q values.

    Mathematical basis:
        Let gamma(k) be the autocovariance of residuals at lag k.
        The algorithm computes prediction coefficients theta[i,j] and
        prediction variances v[i] recursively:
            theta[i, i-k] = (gamma(i-k) - sum_j theta[k,k-j]*theta[i,i-j]*v[j]) / v[k]
            v[i] = gamma(0) - sum_j theta[i,i-j]^2 * v[j]
    """
    # No MA component requested
    if q == 0:
        return np.array([])

    n = len(residuals)

    # Need at least q+1 residual observations to estimate q MA coefficients
    # WHY: With fewer observations than parameters, the system is underdetermined
    if n < q + 1:
        return np.zeros(q)

    # Compute autocovariances of the residuals up to lag q
    # WHY: The innovations algorithm expresses MA coefficients in terms of
    # autocovariances; gamma(0) is the variance, gamma(k) for k>0 are covariances
    mean_r = np.mean(residuals)  # Mean of residuals (should be near zero)
    centered = residuals - mean_r  # Center residuals for covariance computation

    gamma = np.zeros(q + 1)
    for lag in range(q + 1):
        # Autocovariance at lag k: (1/n) * sum_t (r_t - mean)(r_{t-k} - mean)
        # WHY: Using n (not n-1) in denominator for consistency with the
        # autocorrelation function used in the AR estimation above
        gamma[lag] = np.sum(centered[:n - lag] * centered[lag:]) / n

    # Initialize the innovations algorithm arrays
    # theta[i,j] stores prediction coefficient at step i for innovation j
    # v[i] stores the prediction error variance at step i
    theta = np.zeros((n, n))
    v = np.zeros(n)

    # Initial prediction variance is just the series variance
    # WHY: With no past observations, the best prediction is the mean,
    # and the prediction error variance equals the series variance
    v[0] = gamma[0]

    # Iterate the innovations recursion
    # WHY: max_iter limits computation for very long series where we only need
    # the last row of theta; the recursion converges quickly for typical MA processes
    for i in range(1, min(n, max_iter)):
        for k in range(max(0, i - q), i):
            # Compute the sum of previously estimated coefficient products
            # WHY: This sum accounts for the correlation between past innovations;
            # each term theta[k,k-j]*theta[i,i-j]*v[j] measures how much of the
            # covariance at this lag is already explained by previous coefficients
            s = 0.0
            for j in range(max(0, k - q), k):
                if v[j] > 0:  # Guard against zero variance (degenerate case)
                    s += theta[k, k - j] * theta[i, i - j] * v[j]

            # Update theta coefficient for lag (i-k)
            lag_idx = i - k
            if lag_idx <= q:
                # theta[i, i-k] = (gamma(i-k) - accumulated_sum) / v[k]
                # WHY: This is the innovations recursion formula; it computes
                # the optimal linear predictor coefficient for this lag
                theta[i, i - k] = (gamma[lag_idx] - s) / v[k] if v[k] > 0 else 0.0

        # Update prediction error variance
        # WHY: As more coefficients are estimated, the prediction variance decreases;
        # v[i] = gamma(0) - sum of squared theta terms times their variances
        s2 = 0.0
        for j in range(max(0, i - q), i):
            s2 += theta[i, i - j] ** 2 * v[j]
        v[i] = gamma[0] - s2

    # Extract MA coefficients from the last computed row of theta
    # WHY: The innovations algorithm fills theta row by row; the last row contains
    # the converged estimates. We extract theta[row, row-1], theta[row, row-2], etc.
    # which correspond to MA coefficients theta_1, theta_2, ..., theta_q
    row = min(n - 1, max_iter - 1)
    ma_coeffs = np.array([theta[row, row - j] for j in range(1, q + 1)])

    return ma_coeffs


def _compute_residuals(x: np.ndarray, ar_coeffs: np.ndarray,
                       constant: float) -> np.ndarray:
    """Compute residuals from AR model: residual_t = x_t - (c + sum phi_i * x_{t-i}).

    WHY: Residuals represent the portion of the series NOT explained by the AR model.
    If the residuals show autocorrelation, an MA component is needed to model
    the remaining temporal structure (this is what the innovations algorithm does).
    """
    p = len(ar_coeffs)  # Number of AR lags
    n = len(x)          # Length of the differenced series

    # Initialize residual array (first p residuals are zero since we lack sufficient history)
    # WHY: We cannot compute residuals for the first p timesteps because we need
    # p past values for the AR prediction, which are not available at the start
    residuals = np.zeros(n)

    for t in range(p, n):
        # Compute AR prediction for time t: pred = c + phi_1*x_{t-1} + ... + phi_p*x_{t-p}
        pred = constant
        for i in range(p):
            # ar_coeffs[0] multiplies the most recent value x_{t-1}
            # ar_coeffs[1] multiplies x_{t-2}, and so on
            # WHY: This ordering matches the Yule-Walker solution where phi_1
            # corresponds to the lag-1 autocorrelation
            pred += ar_coeffs[i] * x[t - 1 - i]

        # Residual is the difference between actual and predicted
        # WHY: If the AR model is perfect, residuals would be white noise;
        # any remaining structure will be captured by the MA component
        residuals[t] = x[t] - pred

    return residuals


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(train_data: np.ndarray, p: int = 2, d: int = 1, q: int = 2,
          include_constant: bool = True) -> ARIMAModel:
    """Train ARIMA model from scratch.

    Steps:
        1. Difference the series d times to achieve stationarity
        2. Estimate AR coefficients via Yule-Walker equations
        3. Compute AR residuals (what the AR model cannot explain)
        4. Estimate MA coefficients via innovations algorithm on residuals
        5. Estimate the constant term from the differenced series mean

    Returns:
        Fitted ARIMAModel dataclass containing all parameters for forecasting.
    """
    # Initialize the model container with specified orders
    model = ARIMAModel(p=p, d=d, q=q, include_constant=include_constant)

    # Step 1: Apply differencing d times to remove trend
    # WHY: The AR estimation via Yule-Walker assumes stationarity; differencing
    # removes polynomial trends (d=1 for linear, d=2 for quadratic)
    z, initial_values = _difference(train_data, d)
    model.original_last_values = initial_values  # Save for undifferencing
    model.differenced_series = z                 # Save for forecasting

    # Step 2: Estimate AR coefficients using Yule-Walker equations
    # WHY: Yule-Walker provides closed-form AR coefficient estimates by solving
    # a linear system involving the autocorrelation structure of the series
    ar_coeffs = _yule_walker(z, p)
    model.ar_coeffs = ar_coeffs

    # Step 3: Estimate the constant term
    # WHY: The constant c in the ARIMA equation adjusts for the mean of the
    # differenced series. Formula: c = mean(z) * (1 - sum(phi_i))
    # This ensures the model's expected value matches the observed mean
    if include_constant and len(z) > 0:
        mean_z = np.mean(z)  # Mean of the differenced series
        # Adjust constant for the AR coefficients
        # WHY: For an AR(p) model with mean mu, the intercept c = mu * (1 - sum(phi))
        # This relationship comes from taking expectations of the AR equation
        model.constant = mean_z * (1.0 - np.sum(ar_coeffs)) if len(ar_coeffs) > 0 else mean_z
    else:
        model.constant = 0.0

    # Step 4: Compute residuals from the AR model
    # WHY: Residuals = actual - AR_prediction. If these residuals have temporal
    # structure, the MA component will capture it
    residuals = _compute_residuals(z, ar_coeffs, model.constant)
    model.residuals = residuals

    # Step 5: Estimate MA coefficients from the residuals
    # WHY: The innovations algorithm finds theta coefficients that model the
    # autocorrelation structure remaining in the AR residuals
    ma_coeffs = _estimate_ma_innovations(residuals, q)
    model.ma_coeffs = ma_coeffs

    # Store the tail of the original series for undifferencing forecasts
    # WHY: When reversing differencing, we need the last d+1 values as anchors
    # for cumulative summation back to the original scale
    model.diff_history = list(train_data[-(d + 1):]) if d > 0 else []

    return model


def _forecast_n_steps(model: ARIMAModel, n_steps: int,
                      original_tail: np.ndarray) -> np.ndarray:
    """Generate n-step ahead forecasts on the differenced scale,
    then undifference to get forecasts on the original scale.

    WHY two phases:
        1. Forecasting on the differenced scale uses the AR/MA equations directly
        2. Undifferencing converts back to original units for interpretability
    """
    # Unpack model parameters for readability
    z = model.differenced_series  # Historical differenced values
    p, q = model.p, model.q      # AR and MA orders
    ar, ma = model.ar_coeffs, model.ma_coeffs  # Estimated coefficients
    c = model.constant            # Intercept term

    # Initialize history lists with copies of training data
    # WHY: We append to these lists as we generate forecasts, creating a
    # "rolling history" that the AR/MA components reference
    z_history = list(z)      # Differenced values history (for AR)
    r_history = list(model.residuals)  # Residual history (for MA)

    forecasts_diff = []  # Will store forecasts on the differenced scale

    for step in range(n_steps):
        # Start with the constant term
        pred = c

        # AR component: weighted sum of p most recent differenced values
        # WHY: phi_i * z_{t-i} captures the autoregressive relationship -
        # past values predict future values with exponentially decaying influence
        for i in range(p):
            idx = len(z_history) - 1 - i  # Index into history (most recent first)
            if idx >= 0:
                pred += ar[i] * z_history[idx]

        # MA component: weighted sum of q most recent residuals
        # WHY: theta_j * epsilon_{t-j} captures the moving-average relationship -
        # past forecast errors improve future predictions by correcting systematic bias
        for j in range(q):
            idx = len(r_history) - 1 - j  # Index into residual history
            if idx >= 0:
                pred += ma[j] * r_history[idx]

        forecasts_diff.append(pred)

        # Append forecast to history for use in subsequent steps
        # WHY: Multi-step forecasting is recursive - each forecast becomes part of
        # the history for predicting the next step
        z_history.append(pred)

        # Future residuals are assumed to be zero (best guess for unknown errors)
        # WHY: We have no information about future errors, so the expected value
        # of epsilon_{t+k} is zero; this causes MA influence to decay over the
        # forecast horizon, which is a known limitation of ARIMA forecasting
        r_history.append(0.0)

    forecasts_diff = np.array(forecasts_diff)

    # Undifference to recover forecasts on the original scale
    # WHY: The model operates on differenced data; we must reverse the differencing
    # to produce forecasts in the original units (e.g., dollars, passengers)
    if model.d > 0:
        result = forecasts_diff.copy()
        last_val = original_tail[-1]  # Last observed value on original scale

        for _ in range(model.d):
            # Cumulatively sum from the last observed value
            # WHY: If z_hat are differenced forecasts, then y_hat = cumsum(last_val, z_hat)
            result = np.cumsum(np.concatenate([[last_val], result]))[1:]
            last_val = result[-1] if len(result) > 0 else last_val

        # For d=1 (most common case), simplify to direct cumulative sum
        # WHY: The nested loop above handles d>=2, but for d=1 the computation
        # is just cumsum([original_last, diffs])[1:], which we compute cleanly
        if model.d == 1:
            result = np.cumsum(np.concatenate([[original_tail[-1]], forecasts_diff]))[1:]
        return result
    else:
        return forecasts_diff


def validate(model: ARIMAModel, val_data: np.ndarray,
             train_data: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Validate model on held-out validation data.

    WHY train_data is needed: For undifferencing forecasts, we need the tail
    of the original training series as the anchor for cumulative summation.
    """
    if train_data is None:
        # Reconstruct approximate tail from stored differencing history
        # WHY: If original training data is not available, use the stored tail
        # as a fallback; this is less accurate for d > 1
        tail = np.array(model.diff_history) if model.diff_history else np.array([0.0])
    else:
        tail = train_data

    # Generate forecasts for len(val_data) steps and compare to actuals
    forecasts = _forecast_n_steps(model, len(val_data), tail)
    return _compute_metrics(val_data, forecasts)


def test(model: ARIMAModel, test_data: np.ndarray,
         train_data: Optional[np.ndarray] = None,
         val_data: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Test model on held-out test data.

    WHY we forecast through val first: Since the model knows only training data,
    reaching the test period requires forecasting through the entire validation
    period first. We cannot skip ahead because each forecast depends on all
    previous forecasts (recursive multi-step prediction).
    """
    if train_data is None:
        tail = np.array(model.diff_history) if model.diff_history else np.array([0.0])
    else:
        tail = train_data

    # Total forecast horizon includes both validation and test periods
    val_len = len(val_data) if val_data is not None else 0
    total_steps = val_len + len(test_data)

    # Generate all forecasts, then extract only the test portion
    all_forecasts = _forecast_n_steps(model, total_steps, tail)
    test_forecasts = all_forecasts[val_len:]  # Skip validation period

    return _compute_metrics(test_data, test_forecasts)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, train_data: np.ndarray,
                     val_data: np.ndarray) -> float:
    """Optuna objective: minimize validation RMSE.

    WHY RMSE: Penalises large errors quadratically, appropriate for forecasting
    tasks where large misses are disproportionately costly (e.g., inventory shortages).
    """
    # Suggest ARIMA orders within reasonable ranges
    p = trial.suggest_int("p", 0, 5)  # AR order: [0, 5]
    d = trial.suggest_int("d", 0, 2)  # Differencing: [0, 2]
    q = trial.suggest_int("q", 0, 5)  # MA order: [0, 5]

    try:
        # Train and evaluate with suggested hyperparameters
        model = train(train_data, p=p, d=d, q=q)
        metrics = validate(model, val_data, train_data)
        rmse = metrics["RMSE"]
        # Return RMSE if finite, else penalty value
        # WHY: Non-finite RMSE indicates numerical failure; 1e10 tells Optuna
        # this configuration is very bad without crashing the search
        return rmse if np.isfinite(rmse) else 1e10
    except Exception:
        # Return large penalty for any failure
        return 1e10


def run_optuna(train_data: np.ndarray, val_data: np.ndarray,
               n_trials: int = 50) -> optuna.Study:
    """Run Optuna hyperparameter search for best ARIMA orders.

    WHY Optuna: Uses Tree-structured Parzen Estimator (TPE) which learns from
    previous trials to focus sampling on promising hyperparameter regions,
    finding good solutions faster than random or grid search.
    """
    # Create study with minimization objective
    study = optuna.create_study(direction="minimize", study_name="arima_numpy")

    # Run optimization with progress bar for user feedback
    study.optimize(
        lambda trial: optuna_objective(trial, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Display best results
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data: np.ndarray, val_data: np.ndarray,
                    num_samples: int = 20) -> Any:
    """Run Ray Tune distributed hyperparameter search.

    WHY Ray Tune: Provides parallel trial execution across CPU cores.
    Each ARIMA configuration runs independently, so parallelism gives
    near-linear speedup proportional to available cores.
    """
    def _trainable(config):
        """Inner function called by Ray for each trial configuration."""
        try:
            model = train(train_data, p=config["p"], d=config["d"], q=config["q"])
            metrics = validate(model, val_data, train_data)
            # Report metrics back to the Ray Tune controller
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    # Define search space matching Optuna ranges for fair comparison
    search_space = {
        "p": tune.randint(0, 6),  # AR order: uniform random [0, 5]
        "d": tune.randint(0, 3),  # Differencing: [0, 2]
        "q": tune.randint(0, 6),  # MA order: [0, 5]
    }

    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    # Create and execute the tuner
    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()

    # Report best result
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    print(f"Ray Tune Best config: {best.config}")
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(train_data: np.ndarray, val_data: np.ndarray) -> None:
    """Compare different ARIMA(p,d,q) parameter configurations and explain tradeoffs.

    This function demonstrates how each hyperparameter choice affects the model:

    Parameter Selection Reasoning:
        - (1,1,1): Minimal model. AR(1) captures first-order dependence (today
          predicts tomorrow), d=1 removes linear trend, MA(1) smooths one residual.
          Best for: simple series with short memory and clear trend.

        - (2,1,2): Balanced model. AR(2) captures two-step memory (values from
          2 steps ago still matter), MA(2) adjusts for two steps of past errors.
          Best for: moderately complex series with medium-range dependence.

        - (5,1,0): Pure AR model with long memory. Five lagged values predict
          the next, with no MA smoothing. This is like fitting a 5th-order linear
          regression on the differenced series.
          Best for: autoregressive-dominant series where past values are the primary
          signal (e.g., temperature, slowly-varying physical processes).

        - (0,1,3): Pure MA model. No autoregressive memory, only recent forecast
          errors drive predictions. Related to exponential smoothing methods.
          Best for: shock-driven series where random events dominate and
          past values have weak direct influence.

        - (3,0,1) with d=0: No differencing. Assumes the series is already
          stationary. AR(3) + MA(1) models stationary dynamics directly.
          Best for: series that fluctuate around a constant mean with no trend.
    """
    configs = [
        {"label": "ARIMA(1,1,1) - Minimal", "p": 1, "d": 1, "q": 1},
        {"label": "ARIMA(2,1,2) - Balanced", "p": 2, "d": 1, "q": 2},
        {"label": "ARIMA(5,1,0) - Pure AR (long memory)", "p": 5, "d": 1, "q": 0},
        {"label": "ARIMA(0,1,3) - Pure MA (shock-driven)", "p": 0, "d": 1, "q": 3},
        {"label": "ARIMA(3,0,1) - Stationary (no diff)", "p": 3, "d": 0, "q": 1},
    ]

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: How ARIMA order affects forecast accuracy")
    print("=" * 70)

    results_summary = []

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            model = train(train_data, p=cfg["p"], d=cfg["d"], q=cfg["q"])
            metrics = validate(model, val_data, train_data)

            # Display AR and MA coefficients to show model structure
            print(f"  AR coeffs: {model.ar_coeffs}")
            print(f"  MA coeffs: {model.ma_coeffs}")
            print(f"  Constant: {model.constant:.4f}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            results_summary.append({"config": cfg["label"], "RMSE": metrics["RMSE"],
                                    "MAE": metrics["MAE"]})
        except Exception as e:
            print(f"  FAILED: {e}")
            results_summary.append({"config": cfg["label"], "RMSE": float("inf"),
                                    "MAE": float("inf")})

    # Rank configurations by validation RMSE
    print("\n" + "-" * 70)
    print("RANKING (by validation RMSE):")
    results_summary.sort(key=lambda x: x["RMSE"])
    for i, r in enumerate(results_summary, 1):
        print(f"  {i}. {r['config']}: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}")

    print("\nKey Takeaways:")
    print("  - Higher AR order captures more temporal memory but risks overfitting")
    print("  - MA component is most useful when residuals show autocorrelation")
    print("  - Differencing (d=1) is essential for trended series")
    print("  - The simplest adequate model often generalises best (parsimony principle)")


# ---------------------------------------------------------------------------
# Real-World Demo: Retail Daily Sales Forecasting
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate ARIMA on a realistic retail daily sales forecasting scenario.

    Domain: Retail / E-commerce
    Task: Forecast daily sales revenue for inventory and staffing decisions

    This demo generates synthetic data mimicking typical retail sales patterns:
        - Upward growth trend (expanding customer base)
        - Weekly seasonality (weekday vs weekend shopping patterns)
        - Holiday effects (Black Friday, Christmas spikes)
        - Random noise (weather, promotions, competitor actions)

    Business Context:
        Retailers need accurate daily sales forecasts for:
        - Inventory replenishment decisions (avoid stockouts and overstock)
        - Staff scheduling (more cashiers on busy days)
        - Cash flow management and financial planning
        - Promotional campaign timing and budget allocation
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Retail Daily Sales Forecasting")
    print("=" * 70)

    # ---- Generate domain-specific synthetic data ----
    np.random.seed(42)

    # 2 years of daily data (730 days)
    # WHY: Daily retail data needs at least 1-2 years to capture annual patterns
    n_days = 730
    t = np.arange(n_days, dtype=np.float64)

    # Base daily revenue: $50,000
    # WHY: Realistic for a medium-size retail store or small e-commerce operation
    base_revenue = 50000.0

    # Growth trend: 5% annual revenue growth
    # WHY: Typical for a growing retail business; compound daily growth rate
    daily_growth = (1.05 ** (1 / 365))  # Convert annual to daily rate
    trend = base_revenue * (daily_growth ** t - 1)  # Revenue increase from growth

    # Weekly seasonality: weekends are ~40% higher revenue
    # WHY: Retail shopping peaks on weekends (Saturday/Sunday) and dips mid-week;
    # sin wave with period 7 approximates this pattern
    weekly = 0.20 * base_revenue * np.sin(2 * np.pi * t / 7)

    # Annual seasonality: holiday season boost in December
    # WHY: November-December typically accounts for 25-30% of annual retail revenue;
    # we model this as a broad annual sinusoidal with peak around day 335 (December 1)
    annual = 0.15 * base_revenue * np.sin(2 * np.pi * (t - 60) / 365)

    # Random noise: ~10% coefficient of variation
    # WHY: Daily sales have significant randomness from weather, foot traffic,
    # and individual customer behavior
    noise = np.random.normal(0, 0.10 * base_revenue, n_days)

    # Combine all components
    daily_sales = base_revenue + trend + weekly + annual + noise

    # Ensure non-negative sales (can't have negative revenue)
    daily_sales = np.maximum(daily_sales, 0)

    print(f"\nDataset: {n_days} days of daily sales data")
    print(f"Revenue range: ${daily_sales.min():,.0f} - ${daily_sales.max():,.0f}")
    print(f"Mean daily sales: ${daily_sales.mean():,.0f}")

    # Split: 80% train, 10% val, 10% test
    n_train = int(0.80 * n_days)
    n_val = int(0.10 * n_days)

    train_sales = daily_sales[:n_train]
    val_sales = daily_sales[n_train:n_train + n_val]
    test_sales = daily_sales[n_train + n_val:]

    print(f"Train: {len(train_sales)} days | Val: {len(val_sales)} days | "
          f"Test: {len(test_sales)} days")

    # ---- Compare ARIMA configurations for daily retail data ----
    print("\n--- Model Comparison ---")
    configs = [
        {"label": "ARIMA(1,1,1)", "p": 1, "d": 1, "q": 1},
        {"label": "ARIMA(3,1,2)", "p": 3, "d": 1, "q": 2},
        {"label": "ARIMA(5,1,0)", "p": 5, "d": 1, "q": 0},
    ]

    best_model = None
    best_rmse = float("inf")
    best_label = ""

    for cfg in configs:
        try:
            model = train(train_sales, p=cfg["p"], d=cfg["d"], q=cfg["q"])
            metrics = validate(model, val_sales, train_sales)
            print(f"\n  {cfg['label']}:")
            print(f"    Val RMSE: ${metrics['RMSE']:,.0f}")
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
        test_metrics = test(best_model, test_sales, train_sales, val_sales)
        print(f"  Test RMSE: ${test_metrics['RMSE']:,.0f}")
        print(f"  Test MAE:  ${test_metrics['MAE']:,.0f}")
        print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")

        print("\n  Business Interpretation:")
        print(f"    Average daily forecast error: ${test_metrics['MAE']:,.0f}")
        avg_sales = np.mean(test_sales)
        print(f"    As % of average daily sales (${avg_sales:,.0f}): "
              f"{test_metrics['MAPE']:.1f}%")
        print(f"    Forecast quality: {'Good' if test_metrics['MAPE'] < 15 else 'Needs improvement'} "
              f"(target: <15% MAPE for daily retail)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline: data generation -> HPO -> train -> validate -> test.

    Execution flow:
        1. Generate synthetic time series data
        2. Run Optuna HPO to find best (p,d,q) orders
        3. Train final model with best hyperparameters
        4. Report AR/MA coefficients for interpretability
        5. Evaluate on validation set
        6. Evaluate on held-out test set
        7. Run Ray Tune for comparison
        8. Compare parameter configurations
        9. Run real-world retail sales demo
    """
    print("=" * 70)
    print("ARIMA - NumPy From-Scratch Implementation")
    print("=" * 70)

    # 1. Generate synthetic data
    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nData splits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    # 2. Optuna HPO
    print("\n--- Optuna Hyperparameter Search ---")
    study = run_optuna(train_data, val_data, n_trials=30)
    best_params = study.best_params

    # 3. Train with best params
    print("\n--- Training with Best Parameters ---")
    best_model = train(
        train_data,
        p=best_params["p"],
        d=best_params["d"],
        q=best_params["q"],
    )
    # Display learned coefficients for interpretability
    print(f"AR coefficients: {best_model.ar_coeffs}")
    print(f"MA coefficients: {best_model.ma_coeffs}")
    print(f"Constant: {best_model.constant:.4f}")

    # 4. Validate
    print("\n--- Validation Metrics ---")
    val_metrics = validate(best_model, val_data, train_data)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5. Test
    print("\n--- Test Metrics ---")
    test_metrics = test(best_model, test_data, train_data, val_data)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 6. Ray Tune
    print("\n--- Ray Tune Search ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    # 7. Parameter comparison
    compare_parameter_sets(train_data, val_data)

    # 8. Real-world demo
    real_world_demo()

    print("\nDone.")


if __name__ == "__main__":
    main()
