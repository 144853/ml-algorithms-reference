# ARIMA (AutoRegressive Integrated Moving Average) - Complete Guide with Stock Market Applications

## Overview

ARIMA is one of the most widely used classical statistical methods for time series forecasting. It combines three fundamental components: AutoRegression (AR), which models the relationship between an observation and a number of lagged observations; Integration (I), which represents the differencing of raw observations to make the time series stationary; and Moving Average (MA), which models the dependency between an observation and a residual error from a moving average model applied to lagged observations. Together, these components form a powerful framework denoted as ARIMA(p, d, q).

In the context of stock market forecasting, ARIMA has been a foundational tool for decades. While modern deep learning approaches have gained popularity, ARIMA remains relevant for short-term price forecasting, volatility modeling, and as a baseline model against which more complex methods are compared. Its interpretability and well-understood statistical properties make it a go-to choice for quantitative analysts who need transparent models with confidence intervals.

ARIMA works best with univariate time series data where the underlying process can be approximated as a linear combination of past values and past forecast errors. For stock prices, this means ARIMA is most effective when applied to returns or differenced prices rather than raw price levels, since raw stock prices are typically non-stationary (they exhibit trends and changing variance over time).

## How It Works - The Math Behind It

### Step 1: Stationarity and Differencing (the "I" in ARIMA)

A stationary time series has constant mean, variance, and autocovariance over time. Raw stock prices are almost never stationary, so we apply differencing of order d:

```
First difference (d=1):  y'_t = y_t - y_{t-1}
Second difference (d=2): y''_t = y'_t - y'_{t-1} = y_t - 2*y_{t-1} + y_{t-2}
```

For stock prices, first differencing (d=1) typically suffices, effectively converting prices to returns. We verify stationarity using the Augmented Dickey-Fuller (ADF) test where the null hypothesis is that the series has a unit root (non-stationary).

### Step 2: AutoRegressive Component (the "AR" in ARIMA)

The AR(p) component models the current value as a linear combination of p past values:

```
AR(p): y_t = c + phi_1 * y_{t-1} + phi_2 * y_{t-2} + ... + phi_p * y_{t-p} + epsilon_t
```

Where:
- `c` is a constant (intercept)
- `phi_1, phi_2, ..., phi_p` are the autoregressive coefficients
- `epsilon_t` is white noise (error term)
- `p` is the order of the AR component

The parameter p is typically identified by examining the Partial Autocorrelation Function (PACF), where significant spikes indicate the appropriate lag order.

### Step 3: Moving Average Component (the "MA" in ARIMA)

The MA(q) component models the current value as a linear combination of q past forecast errors:

```
MA(q): y_t = mu + epsilon_t + theta_1 * epsilon_{t-1} + theta_2 * epsilon_{t-2} + ... + theta_q * epsilon_{t-q}
```

Where:
- `mu` is the mean of the series
- `theta_1, theta_2, ..., theta_q` are the moving average coefficients
- `epsilon_t, epsilon_{t-1}, ...` are current and past error terms
- `q` is the order of the MA component

The parameter q is identified by examining the Autocorrelation Function (ACF), where significant spikes indicate the appropriate order.

### Step 4: Full ARIMA Model

Combining all three components, the ARIMA(p, d, q) model for the differenced series is:

```
y'_t = c + phi_1*y'_{t-1} + ... + phi_p*y'_{t-p} + epsilon_t + theta_1*epsilon_{t-1} + ... + theta_q*epsilon_{t-q}
```

### Step 5: Parameter Estimation

Parameters are estimated using Maximum Likelihood Estimation (MLE). The log-likelihood for a Gaussian ARIMA model is:

```
log L = -n/2 * log(2*pi) - n/2 * log(sigma^2) - 1/(2*sigma^2) * sum(epsilon_t^2)
```

Model selection uses information criteria:
- AIC = 2k - 2*log(L), where k is the number of parameters
- BIC = k*log(n) - 2*log(L), where n is the number of observations

Lower AIC/BIC values indicate better model fit with appropriate complexity penalization.

### Step 6: Forecasting

Forecasts are generated recursively. For a one-step-ahead forecast:

```
y_hat_{t+1} = c + phi_1*y_t + phi_2*y_{t-1} + ... + theta_1*epsilon_t + theta_2*epsilon_{t-1} + ...
```

Confidence intervals widen as the forecast horizon increases, reflecting growing uncertainty.

## Stock Market Use Case: Short-Term Stock Price Forecasting for Intraday Trading

### The Problem

A quantitative trading desk needs to forecast the closing price of a large-cap technology stock (e.g., a stock similar to AAPL) over the next 5 trading days. The model must provide point forecasts with confidence intervals to support position sizing decisions. The desk requires a transparent, interpretable model that can be quickly validated by risk management.

### Stock Market Features (Input Data)

| Feature | Description | Role in ARIMA |
|---------|-------------|---------------|
| Close Price | Daily closing price | Primary series (differenced) |
| Log Returns | ln(P_t / P_{t-1}) | Alternative primary series |
| Volume | Daily trading volume | External regressor (ARIMAX) |
| Bid-Ask Spread | Daily average spread | Liquidity indicator |
| VIX | Market volatility index | External regressor (ARIMAX) |
| 10Y Treasury Yield | Risk-free rate proxy | Macro indicator |
| S&P 500 Returns | Market benchmark returns | Market factor |
| Trading Days | Business days calendar | Calendar adjustment |

### Example Data Structure

```python
import numpy as np

# Simulate 252 trading days (1 year) of stock price data
np.random.seed(42)
n_days = 252

# Generate realistic stock prices using geometric Brownian motion
mu_annual = 0.08       # 8% annual drift
sigma_annual = 0.25    # 25% annual volatility
dt = 1/252             # Daily time step

daily_returns = np.random.normal(mu_annual * dt, sigma_annual * np.sqrt(dt), n_days)
initial_price = 150.0
prices = initial_price * np.cumprod(1 + daily_returns)

# Generate corresponding volume data
base_volume = 50_000_000
volume = base_volume + np.random.normal(0, 5_000_000, n_days)
volume = np.abs(volume).astype(int)

# Create date index (trading days only)
dates = np.arange(n_days)

print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
print(f"Mean daily return: {daily_returns.mean()*100:.4f}%")
print(f"Daily volatility: {daily_returns.std()*100:.4f}%")
print(f"Final price: ${prices[-1]:.2f}")
```

### The Model in Action

```python
import numpy as np

# Step 1: Compute log returns (differencing for stationarity)
log_prices = np.log(prices)
log_returns = np.diff(log_prices)  # First difference of log prices

# Step 2: Verify stationarity via simple mean-reversion check
rolling_mean = np.convolve(log_returns, np.ones(20)/20, mode='valid')
print(f"Return series mean: {log_returns.mean():.6f}")
print(f"Return series std:  {log_returns.std():.6f}")

# Step 3: Compute ACF and PACF manually
def compute_acf(series, max_lag=20):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    acf_values = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        cov = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
        acf_values[lag] = cov / var
    return acf_values

acf = compute_acf(log_returns, max_lag=20)
print("\nACF values (first 10 lags):")
for i in range(11):
    print(f"  Lag {i:2d}: {acf[i]:.4f}")

# Step 4: Simple AR(1) model implementation
# y_t = c + phi * y_{t-1} + epsilon_t
y = log_returns[1:]
y_lag = log_returns[:-1]

# OLS estimation of AR(1) coefficients
X = np.column_stack([np.ones(len(y_lag)), y_lag])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
c_hat, phi_hat = beta[0], beta[1]

print(f"\nAR(1) coefficients:")
print(f"  Intercept (c): {c_hat:.6f}")
print(f"  AR(1) coeff (phi): {phi_hat:.6f}")

# Step 5: Compute residuals
residuals = y - (c_hat + phi_hat * y_lag)
sigma_hat = np.std(residuals)

# Step 6: Generate 5-day forecasts
n_forecast = 5
forecasts = np.zeros(n_forecast)
last_return = log_returns[-1]

for i in range(n_forecast):
    if i == 0:
        forecasts[i] = c_hat + phi_hat * last_return
    else:
        forecasts[i] = c_hat + phi_hat * forecasts[i-1]

# Convert log return forecasts to price forecasts
last_price = prices[-1]
forecast_prices = np.zeros(n_forecast)
for i in range(n_forecast):
    if i == 0:
        forecast_prices[i] = last_price * np.exp(forecasts[i])
    else:
        forecast_prices[i] = forecast_prices[i-1] * np.exp(forecasts[i])

# Confidence intervals (95%)
z_95 = 1.96
ci_width = np.zeros(n_forecast)
for i in range(n_forecast):
    ci_width[i] = z_95 * sigma_hat * np.sqrt(i + 1)

upper_prices = np.zeros(n_forecast)
lower_prices = np.zeros(n_forecast)
for i in range(n_forecast):
    cumulative_return = np.sum(forecasts[:i+1])
    upper_prices[i] = last_price * np.exp(cumulative_return + ci_width[i])
    lower_prices[i] = last_price * np.exp(cumulative_return - ci_width[i])

print(f"\n5-Day Price Forecast from ${last_price:.2f}:")
print(f"{'Day':<6} {'Forecast':>10} {'Lower 95%':>12} {'Upper 95%':>12}")
print("-" * 42)
for i in range(n_forecast):
    print(f"  {i+1:<4} ${forecast_prices[i]:>8.2f}  ${lower_prices[i]:>10.2f}  ${upper_prices[i]:>10.2f}")
```

## Advantages

1. **Statistical rigor and interpretability**: ARIMA models are grounded in well-established statistical theory. Each coefficient has a clear interpretation, and significance tests can be applied to each parameter. This is critical in finance where regulators and risk managers demand model transparency.

2. **Built-in stationarity handling**: The integration component (differencing) directly addresses the non-stationarity inherent in stock prices. By working with returns rather than levels, ARIMA naturally handles trending behavior without requiring separate detrending steps.

3. **Confidence intervals for risk management**: ARIMA provides analytical confidence intervals for forecasts, which widen naturally with the forecast horizon. This is invaluable for position sizing, stop-loss placement, and Value-at-Risk calculations in portfolio management.

4. **Low computational requirements**: ARIMA models train in seconds even on years of daily data. This makes them ideal for real-time trading systems that need to re-estimate models frequently throughout the trading day or across hundreds of securities.

5. **Well-established model selection criteria**: AIC and BIC provide principled approaches to selecting model order (p, d, q). Auto-ARIMA algorithms can automatically search the parameter space, reducing the need for manual tuning compared to deep learning approaches.

6. **Strong baseline performance**: ARIMA consistently provides competitive short-term forecasts for financial time series. Many studies have shown that simple ARIMA models are difficult to beat for 1-5 day ahead forecasts, making them an essential benchmark for any forecasting system.

7. **Extensibility to ARIMAX**: The framework extends naturally to include exogenous variables (ARIMAX), allowing incorporation of market indicators, macroeconomic data, or sector-specific factors while maintaining the interpretable linear structure.

## Disadvantages

1. **Linear assumption limitation**: ARIMA assumes linear relationships between lagged values and current values. Stock markets exhibit well-documented nonlinear behaviors such as volatility clustering, leverage effects, and regime changes that ARIMA cannot capture.

2. **Poor long-horizon forecasts**: ARIMA forecasts converge rapidly to the unconditional mean as the horizon extends. For stock prices, this means forecasts beyond 5-10 days become essentially flat, providing no useful directional information for medium-term trading strategies.

3. **Inability to model volatility dynamics**: ARIMA models the conditional mean but not the conditional variance. Stock returns exhibit strong GARCH effects (volatility clustering), meaning periods of high volatility tend to persist. Separate GARCH models must be used alongside ARIMA for volatility forecasting.

4. **Stationarity requirement**: The differencing operation required for stationarity can remove important long-term trend information. For stocks in strong trends (growth stocks, bear markets), excessive differencing may eliminate the signal that matters most for medium-term predictions.

5. **Sensitivity to outliers**: Stock markets experience extreme events (flash crashes, earnings surprises, geopolitical shocks) that create outliers. ARIMA parameter estimates based on MLE are sensitive to these outliers, potentially distorting forecasts until the outlier data point ages out of the estimation window.

6. **Univariate limitation**: Standard ARIMA processes only a single time series. In reality, stock prices are influenced by multiple correlated factors (sector movements, macro indicators, sentiment). While ARIMAX partially addresses this, it lacks the flexibility of multivariate models like VAR or neural networks.

7. **No regime awareness**: Stock markets alternate between bull and bear regimes, each with different statistical properties. ARIMA uses a single set of parameters for the entire series, making it unable to adapt to regime changes without explicit regime-switching extensions.

## When to Use in Stock Market

- Short-term price forecasting (1-5 day horizon) for liquid, large-cap stocks
- As a benchmark model against which more complex algorithms are compared
- When interpretability and model transparency are required by compliance or risk management
- For generating confidence intervals needed in portfolio risk calculations
- When computational resources are limited or speed is critical (high-frequency rebalancing)
- For preprocessing: removing linear dependencies before feeding residuals into nonlinear models
- When the stock exhibits strong autocorrelation in returns (momentum or mean-reversion patterns)
- In ensemble models where ARIMA captures the linear component of the signal

## When NOT to Use in Stock Market

- For long-term investment decisions (horizons beyond 2 weeks) where mean-reversion dominates
- When modeling highly volatile penny stocks or illiquid securities with frequent gaps
- For capturing complex nonlinear patterns like earnings announcement reactions
- When the goal is to model volatility itself (use GARCH-family models instead)
- For multi-asset portfolio optimization requiring cross-asset dependencies
- During market regime transitions (crisis onset, recovery periods) where parameters shift rapidly
- When the primary signal comes from alternative data (sentiment, satellite imagery, web traffic)
- For high-frequency trading where tick-level microstructure effects dominate

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| p (AR order) | Number of lagged observations | 0-5 | Start with p=1-2; use PACF to guide selection |
| d (Differencing) | Number of differences for stationarity | 0-2 | d=1 for prices (returns); d=0 if already using returns |
| q (MA order) | Number of lagged forecast errors | 0-5 | Start with q=1-2; use ACF to guide selection |
| Trend | Constant, linear trend, or none | c, t, ct, n | Use 'c' for returns with non-zero mean; 'n' otherwise |
| Estimation method | MLE, CSS, or CSS-MLE | - | MLE for most cases; CSS-MLE for numerical stability |
| Information criterion | Model selection metric | AIC, BIC | BIC for more parsimonious models; AIC if prediction is primary |
| Training window | Number of observations for estimation | 60-500 days | 252 days (1 year) for daily data; expand for stability |
| Forecast horizon | Steps ahead to predict | 1-20 | 1-5 days for reliable forecasts; accuracy drops quickly beyond |

## Stock Market Performance Tips

1. **Always work with returns, not prices**: Apply log transformation then first differencing. Log returns have better statistical properties (approximate normality, time-additivity) than simple returns for ARIMA modeling.

2. **Use rolling window estimation**: Re-estimate ARIMA parameters using a rolling window (e.g., 252 trading days) rather than an expanding window. This allows the model to adapt to changing market dynamics and prevents ancient data from dominating parameter estimates.

3. **Check for structural breaks**: Use the CUSUM test or Chow test to detect regime changes in the estimation window. If a structural break is detected, only use data after the break for estimation.

4. **Combine with GARCH for volatility**: Fit ARIMA to the mean equation and GARCH to the residuals. The ARIMA-GARCH combination captures both return dynamics and volatility clustering, providing better confidence intervals.

5. **Validate with out-of-sample testing**: Never trust in-sample fit metrics alone. Use walk-forward validation where you re-estimate the model at each step and forecast one step ahead, then compare accumulated out-of-sample forecasts to actuals.

6. **Handle earnings announcements carefully**: Exclude or flag earnings announcement days. The abnormal returns on these days can distort ARIMA parameter estimates. Consider using dummy variables or separate models for earnings periods.

7. **Monitor residual diagnostics**: Check that residuals are white noise (no remaining autocorrelation) using the Ljung-Box test. If residuals show patterns, the model order may be insufficient.

## Comparison with Other Algorithms

| Feature | ARIMA | SARIMA | Prophet | LSTM | TFT |
|---------|-------|--------|---------|------|-----|
| Handles seasonality | No (use SARIMA) | Yes | Yes | Yes | Yes |
| Nonlinear patterns | No | No | Partially | Yes | Yes |
| Multivariate inputs | Limited (ARIMAX) | Limited | Yes (regressors) | Yes | Yes |
| Interpretability | High | High | High | Low | Medium |
| Training speed | Very fast | Fast | Fast | Slow | Very slow |
| Data requirements | Low (50+ obs) | Medium (2+ seasons) | Medium (2+ years) | High (1000+ obs) | Very high |
| Forecast uncertainty | Analytical CI | Analytical CI | Simulated CI | Manual/MC dropout | Quantile outputs |
| Best horizon | 1-5 days | 1 season | Weeks-months | Days-weeks | Days-months |
| Stock market strength | Short-term returns | Seasonal patterns | Trend decomposition | Complex patterns | Multi-horizon |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete ARIMA Implementation for Stock Price Forecasting
# Using only NumPy - No external dependencies
# ============================================================

np.random.seed(123)

# --- Generate Realistic Stock Price Data ---
n_trading_days = 504  # ~2 years of trading data
mu = 0.0003           # Daily drift (~7.5% annually)
sigma = 0.015         # Daily volatility (~24% annually)

# Add some autocorrelation to returns (momentum effect)
raw_returns = np.random.normal(mu, sigma, n_trading_days)
returns = np.zeros(n_trading_days)
returns[0] = raw_returns[0]
for t in range(1, n_trading_days):
    returns[t] = 0.05 * returns[t-1] + raw_returns[t]  # Slight AR(1) component

# Convert to prices
initial_price = 175.0
prices = initial_price * np.cumprod(1 + returns)

# Split into train/test
train_size = 480
test_size = n_trading_days - train_size
train_prices = prices[:train_size]
test_prices = prices[train_size:]

print("=" * 60)
print("ARIMA Stock Price Forecasting")
print("=" * 60)
print(f"Total observations: {n_trading_days}")
print(f"Training set: {train_size} days")
print(f"Test set:     {test_size} days")
print(f"Price range:  ${prices.min():.2f} - ${prices.max():.2f}")

# --- Step 1: Transform to Log Returns ---
train_log_prices = np.log(train_prices)
train_log_returns = np.diff(train_log_prices)

print(f"\nLog Returns Statistics:")
print(f"  Mean:     {train_log_returns.mean():.6f}")
print(f"  Std Dev:  {train_log_returns.std():.6f}")
print(f"  Skewness: {((train_log_returns - train_log_returns.mean())**3).mean() / train_log_returns.std()**3:.4f}")
print(f"  Kurtosis: {((train_log_returns - train_log_returns.mean())**4).mean() / train_log_returns.std()**4:.4f}")

# --- Step 2: Compute ACF and PACF ---
def compute_acf(series, max_lag=20):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    acf_vals = []
    for lag in range(max_lag + 1):
        if var == 0:
            acf_vals.append(0.0)
        else:
            cov = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
            acf_vals.append(cov / var)
    return np.array(acf_vals)

def compute_pacf(series, max_lag=10):
    """Compute PACF using Durbin-Levinson algorithm."""
    acf_vals = compute_acf(series, max_lag)
    pacf_vals = np.zeros(max_lag + 1)
    pacf_vals[0] = 1.0
    pacf_vals[1] = acf_vals[1]

    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf_vals[1]

    for k in range(2, max_lag + 1):
        num = acf_vals[k] - sum(phi[k-1, j] * acf_vals[k-j] for j in range(1, k))
        den = 1.0 - sum(phi[k-1, j] * acf_vals[j] for j in range(1, k))
        if abs(den) < 1e-10:
            pacf_vals[k] = 0.0
            continue
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
        pacf_vals[k] = phi[k, k]

    return pacf_vals

acf_vals = compute_acf(train_log_returns, 15)
pacf_vals = compute_pacf(train_log_returns, 10)

print(f"\nACF (first 5 lags):  {[f'{v:.4f}' for v in acf_vals[1:6]]}")
print(f"PACF (first 5 lags): {[f'{v:.4f}' for v in pacf_vals[1:6]]}")

# Significance threshold
sig_threshold = 1.96 / np.sqrt(len(train_log_returns))
print(f"95% significance threshold: +/- {sig_threshold:.4f}")

# --- Step 3: Fit AR(1) Model via OLS ---
def fit_ar(series, p):
    """Fit AR(p) model using OLS."""
    n = len(series)
    y = series[p:]
    X = np.ones((n - p, p + 1))
    for i in range(p):
        X[:, i + 1] = series[p - i - 1:n - i - 1]

    # OLS: beta = (X'X)^(-1) X'y
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    sigma2 = np.var(residuals, ddof=p + 1)

    # AIC and BIC
    n_obs = len(y)
    k = p + 2  # p AR coeffs + intercept + sigma2
    log_likelihood = -n_obs/2 * np.log(2*np.pi) - n_obs/2 * np.log(sigma2) - n_obs/2
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n_obs) - 2 * log_likelihood

    return {
        'intercept': beta[0],
        'ar_coeffs': beta[1:],
        'sigma2': sigma2,
        'residuals': residuals,
        'aic': aic,
        'bic': bic,
        'n_obs': n_obs
    }

# --- Step 4: Model Selection ---
print(f"\nModel Selection (AR models):")
print(f"{'Model':<10} {'AIC':>12} {'BIC':>12} {'Sigma':>10}")
print("-" * 46)
best_bic = np.inf
best_p = 0
for p in range(0, 6):
    if p == 0:
        # AR(0) is just the mean
        mean_val = np.mean(train_log_returns)
        resid = train_log_returns - mean_val
        sigma2 = np.var(resid)
        n_obs = len(train_log_returns)
        log_l = -n_obs/2 * np.log(2*np.pi) - n_obs/2 * np.log(sigma2) - n_obs/2
        aic = 2*2 - 2*log_l
        bic = 2*np.log(n_obs) - 2*log_l
        print(f"AR(0)     {aic:>12.2f} {bic:>12.2f} {np.sqrt(sigma2):>10.6f}")
        if bic < best_bic:
            best_bic = bic
            best_p = 0
    else:
        result = fit_ar(train_log_returns, p)
        print(f"AR({p})     {result['aic']:>12.2f} {result['bic']:>12.2f} {np.sqrt(result['sigma2']):>10.6f}")
        if result['bic'] < best_bic:
            best_bic = result['bic']
            best_p = p

print(f"\nBest model by BIC: AR({best_p})")

# --- Step 5: Fit Best Model and Forecast ---
if best_p == 0:
    intercept = np.mean(train_log_returns)
    ar_coeffs = np.array([])
    resid = train_log_returns - intercept
    sigma = np.std(resid)
else:
    best_model = fit_ar(train_log_returns, best_p)
    intercept = best_model['intercept']
    ar_coeffs = best_model['ar_coeffs']
    sigma = np.sqrt(best_model['sigma2'])

print(f"\nFitted Model Parameters:")
print(f"  Intercept: {intercept:.8f}")
for i, coeff in enumerate(ar_coeffs):
    print(f"  AR({i+1}):     {coeff:.8f}")
print(f"  Sigma:     {sigma:.8f}")

# --- Step 6: Walk-Forward Forecasting ---
forecast_returns = np.zeros(test_size)
actual_returns = np.diff(np.log(prices[train_size-1:]))  # includes connection point

all_returns = np.concatenate([train_log_returns, np.zeros(test_size)])

for t in range(test_size):
    idx = train_size - 1 + t
    if best_p == 0:
        forecast_returns[t] = intercept
    else:
        pred = intercept
        for j in range(best_p):
            pred += ar_coeffs[j] * all_returns[idx - j - 1]
        forecast_returns[t] = pred
    all_returns[idx] = actual_returns[t]

# Convert to price forecasts
last_train_price = prices[train_size - 1]
forecast_prices = np.zeros(test_size)
actual_test_prices = prices[train_size:]

for t in range(test_size):
    if t == 0:
        forecast_prices[t] = last_train_price * np.exp(forecast_returns[t])
    else:
        forecast_prices[t] = actual_test_prices[t-1] * np.exp(forecast_returns[t])

# --- Step 7: Evaluation Metrics ---
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def direction_accuracy(actual_ret, forecast_ret):
    correct = np.sum(np.sign(actual_ret) == np.sign(forecast_ret))
    return correct / len(actual_ret) * 100

dir_acc = direction_accuracy(actual_returns[:test_size], forecast_returns)

print(f"\n{'='*60}")
print(f"Out-of-Sample Forecast Evaluation ({test_size} days)")
print(f"{'='*60}")
print(f"RMSE:                ${rmse(actual_test_prices, forecast_prices):.4f}")
print(f"MAE:                 ${mae(actual_test_prices, forecast_prices):.4f}")
print(f"MAPE:                {mape(actual_test_prices, forecast_prices):.4f}%")
print(f"Direction Accuracy:  {dir_acc:.2f}%")

# Show first 10 forecasts
print(f"\nFirst 10 Day Forecasts:")
print(f"{'Day':<6} {'Actual':>10} {'Forecast':>10} {'Error':>10} {'Direction':>10}")
print("-" * 50)
for i in range(min(10, test_size)):
    actual = actual_test_prices[i]
    forecast = forecast_prices[i]
    error = actual - forecast
    direction = "Correct" if np.sign(actual_returns[i]) == np.sign(forecast_returns[i]) else "Wrong"
    print(f"  {i+1:<4} ${actual:>8.2f}  ${forecast:>8.2f}  ${error:>8.2f}  {direction:>10}")
```

## Key Takeaways

1. ARIMA is the foundational time series model for stock market forecasting, providing a statistically rigorous framework for modeling linear dependencies in return series.

2. The model excels at short-term forecasting (1-5 days) where linear autocorrelation in returns can be exploited, but forecasts quickly converge to the mean for longer horizons.

3. Always work with stationary series (log returns) rather than raw prices. Use the ADF test to verify stationarity and select the minimum differencing order needed.

4. Model selection via AIC/BIC helps balance model complexity against fit. For stock returns, parsimonious models (low p and q) typically outperform complex ones out of sample.

5. ARIMA should be paired with GARCH models for a complete picture of both return dynamics and volatility clustering in stock markets.

6. Use walk-forward validation rather than simple train/test splits. Re-estimating parameters at each step better simulates real-world trading conditions.

7. ARIMA serves as an essential baseline. Any more complex model (LSTM, Transformer) should demonstrate clear superiority over ARIMA before being deployed in production trading systems.

8. Direction accuracy (predicting up vs. down correctly) is often more important than point forecast accuracy for trading strategies. Even modest directional accuracy above 50% can be profitable with proper position sizing.
