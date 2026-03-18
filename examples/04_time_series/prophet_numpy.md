# Prophet - Complete Guide with Stock Market Applications

## Overview

Prophet is a time series forecasting framework originally developed by Facebook (now Meta) for business forecasting at scale. It is based on a decomposable additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. The model is expressed as y(t) = g(t) + s(t) + h(t) + epsilon(t), where g(t) is the trend function, s(t) models seasonality, h(t) captures holiday and event effects, and epsilon(t) is the error term. Prophet uses a Bayesian framework with Stan for parameter estimation.

In the stock market context, Prophet offers a compelling approach for decomposing stock price movements into interpretable components: long-term trend (secular bull or bear markets), seasonal patterns (January Effect, quarterly earnings cycles, day-of-week effects), and the impact of specific events (Fed meetings, earnings announcements, market holidays). This decomposition is valuable for portfolio managers who need to separate predictable patterns from idiosyncratic movements when making allocation decisions.

Prophet's design philosophy emphasizes analyst-friendly forecasting: it handles missing data gracefully, is robust to outliers, and allows domain experts to inject knowledge through configurable seasonality, known changepoints, and holiday schedules. For stock market applications, this means quantitative analysts can encode known market events (FOMC meetings, options expiration, index rebalancing dates) directly into the model, combining statistical learning with domain expertise in a principled way.

## How It Works - The Math Behind It

### The Decomposable Model

Prophet decomposes the time series into three main components:

```
y(t) = g(t) + s(t) + h(t) + epsilon(t)
```

### Component 1: Trend Function g(t)

Prophet offers two trend models:

**Piecewise Linear Trend (default):**
```
g(t) = (k + a(t)^T * delta) * t + (m + a(t)^T * gamma)
```

Where:
- `k` is the base growth rate
- `delta` is a vector of rate adjustments at changepoints
- `m` is the offset parameter
- `gamma` is set to maintain continuity: gamma_j = -s_j * delta_j
- `a(t)` is a vector of indicator functions: a_j(t) = 1 if t >= s_j (changepoint j)
- `s_j` are the changepoint locations

**Logistic Growth Trend (for bounded growth):**
```
g(t) = C(t) / (1 + exp(-(k + a(t)^T * delta) * (t - (m + a(t)^T * gamma))))
```

Where C(t) is the carrying capacity (price ceiling/floor), which can vary over time.

### Component 2: Seasonality Function s(t)

Seasonality is modeled using Fourier series:

```
s(t) = sum_{n=1}^{N} [a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P)]
```

Where:
- `P` is the period (P=365.25 for yearly, P=7 for weekly)
- `N` is the number of Fourier terms (controls smoothness)
- `a_n, b_n` are the Fourier coefficients to be estimated

For yearly seasonality, P=365.25 and N=10 by default (20 parameters).
For weekly seasonality, P=7 and N=3 by default (6 parameters).

Higher N allows more complex seasonal shapes but risks overfitting.

### Component 3: Holiday/Event Effects h(t)

Events are modeled as indicator variables with windows:

```
h(t) = Z(t) * kappa
```

Where:
- `Z(t)` is an indicator matrix for holidays/events
- `kappa ~ Normal(0, nu^2)` are the holiday effects (regularized)
- Each holiday gets a window [lower_window, upper_window] for multi-day effects

### Prior Specifications

Prophet uses the following priors (Bayesian regularization):

```
k ~ Normal(0, 5)              # Growth rate
delta ~ Laplace(0, tau)        # Changepoint magnitudes (sparse via Laplace prior)
sigma ~ Normal(0, 0.5)         # Observation noise
beta ~ Normal(0, sigma_beta)   # Seasonality coefficients
kappa ~ Normal(0, nu)          # Holiday effects
```

The Laplace prior on delta encourages sparsity in changepoints, meaning only a few changepoints will have large effects while most will be near zero.

### Changepoint Detection

Prophet places potential changepoints uniformly across the first 80% of the training data:

```
s_j = j * T / (S + 1), for j = 1, ..., S
```

Where S is the number of potential changepoints (default: 25). The model then learns which changepoints are significant through the sparse prior on delta.

### Uncertainty Quantification

Prophet generates prediction intervals through:

1. **Trend uncertainty**: Simulating future changepoints based on historical changepoint frequency and magnitude
2. **Observation noise**: Adding Gaussian noise based on estimated sigma
3. **Full posterior**: Sampling from the posterior distribution of all parameters

## Stock Market Use Case: Multi-Component Stock Price Decomposition and Event-Driven Forecasting

### The Problem

A fundamental equity analyst covers a portfolio of 20 large-cap technology stocks. She needs to decompose each stock's price movement into: (1) a secular growth trend reflecting the company's fundamental trajectory, (2) seasonal patterns tied to product launch cycles, earnings announcements, and sector rotation, and (3) event-driven impacts from FOMC meetings, index rebalancing, and company-specific catalysts. The decomposition helps her distinguish between predictable price movements and genuine alpha opportunities.

### Stock Market Features (Input Data)

| Feature | Description | Prophet Component |
|---------|-------------|-------------------|
| Date (ds) | Trading date | Time index |
| Close Price (y) | Daily adjusted close price | Target variable |
| FOMC Meeting | Federal Reserve meeting dates | Holiday/event regressor |
| Earnings Date | Quarterly earnings announcement | Holiday/event regressor |
| Options Expiration | Monthly/quarterly expiration | Holiday/event regressor |
| Index Rebalancing | S&P 500 / Russell rebalancing | Holiday/event regressor |
| Ex-Dividend Date | Dividend payment dates | Holiday/event regressor |
| VIX Level | Market volatility regime | Additional regressor |
| Sector ETF Flow | Sector-level fund flows | Additional regressor |
| 10Y Yield Change | Interest rate changes | Additional regressor |
| Trading Volume | Daily volume | Floor/cap for logistic trend |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)

# Generate 5 years of daily stock price data with multiple components
n_years = 5
n_trading_days = 252 * n_years

# Component 1: Piecewise linear trend with changepoints
base_growth_rate = 0.0004  # Daily growth rate (~10% annual)
trend = np.zeros(n_trading_days)
trend[0] = np.log(150.0)  # Starting price $150

# Define changepoints (regime changes)
changepoints = {
    200: 0.0008,   # Acceleration (strong earnings)
    500: -0.0002,  # Slowdown (market correction)
    700: 0.0006,   # Recovery rally
    900: -0.0004,  # Bear market onset
    1050: 0.0010,  # Strong recovery
}

current_rate = base_growth_rate
for t in range(1, n_trading_days):
    if t in changepoints:
        current_rate += changepoints[t]
    trend[t] = trend[t-1] + current_rate

# Component 2: Seasonal patterns
day_of_year = np.arange(n_trading_days) % 252
seasonal = np.zeros(n_trading_days)

# Yearly seasonality (Fourier series approximation)
for n in range(1, 6):
    seasonal += 0.003/n * np.cos(2 * np.pi * n * day_of_year / 252)
    seasonal += 0.002/n * np.sin(2 * np.pi * n * day_of_year / 252)

# Component 3: Event effects (earnings bumps every ~63 days)
events = np.zeros(n_trading_days)
for t in range(0, n_trading_days, 63):  # Quarterly earnings
    event_effect = np.random.normal(0.02, 0.01)  # Earnings surprise
    for d in range(min(3, n_trading_days - t)):
        events[t + d] += event_effect * (1 - d * 0.3)

# Component 4: Noise
noise = np.random.normal(0, 0.012, n_trading_days)

# Combine all components
log_prices = trend + seasonal + events + noise
prices = np.exp(log_prices)

print(f"Stock Price Data Summary:")
print(f"  Period: {n_years} years ({n_trading_days} trading days)")
print(f"  Start price:  ${prices[0]:.2f}")
print(f"  End price:    ${prices[-1]:.2f}")
print(f"  Total return: {(prices[-1]/prices[0] - 1)*100:.1f}%")
print(f"  Max price:    ${prices.max():.2f}")
print(f"  Min price:    ${prices.min():.2f}")
```

### The Model in Action

```python
import numpy as np

# Prophet-style decomposition using NumPy
# Implementing core concepts: trend + seasonality + events

# --- Step 1: Piecewise Linear Trend Estimation ---
def fit_piecewise_trend(y, n_changepoints=10):
    """Fit a piecewise linear trend using OLS with changepoints."""
    n = len(y)
    t = np.arange(n, dtype=float) / n  # Normalize time to [0, 1]

    # Place changepoints uniformly in first 80% of data
    cp_indices = np.linspace(0, int(0.8 * n), n_changepoints + 2)[1:-1].astype(int)
    cp_times = t[cp_indices]

    # Build design matrix: [1, t, (t - s_1)+, (t - s_2)+, ...]
    X = np.ones((n, n_changepoints + 2))
    X[:, 1] = t
    for j, s_j in enumerate(cp_times):
        X[:, j + 2] = np.maximum(0, t - s_j)

    # Ridge regression (L2 regularization for stability)
    lambda_reg = 0.1
    XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    trend = X @ beta

    # Identify significant changepoints
    rate_changes = beta[2:]
    significant = np.abs(rate_changes) > np.std(rate_changes)

    return trend, cp_indices, rate_changes, significant

# Work with log prices
log_prices = np.log(prices)

# Fit trend
trend_hat, cp_idx, rate_chg, sig_mask = fit_piecewise_trend(log_prices, n_changepoints=15)

print("Trend Estimation:")
print(f"  Significant changepoints found: {sig_mask.sum()}")
print(f"  Changepoint locations (significant):")
for i, (idx, chg) in enumerate(zip(cp_idx[sig_mask], rate_chg[sig_mask])):
    direction = "acceleration" if chg > 0 else "deceleration"
    print(f"    Day {idx}: rate change = {chg:.6f} ({direction})")

# --- Step 2: Fourier Seasonality Estimation ---
def fit_fourier_seasonality(residuals, period=252, n_harmonics=5):
    """Fit Fourier series seasonality."""
    n = len(residuals)
    t = np.arange(n, dtype=float)

    # Build Fourier design matrix
    X = np.ones((n, 2 * n_harmonics + 1))
    for h in range(1, n_harmonics + 1):
        X[:, 2*h - 1] = np.cos(2 * np.pi * h * t / period)
        X[:, 2*h] = np.sin(2 * np.pi * h * t / period)

    # Ridge regression
    lambda_reg = 1.0
    XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
    Xty = X.T @ residuals
    beta = np.linalg.solve(XtX, Xty)

    seasonal = X @ beta

    return seasonal, beta

# Remove trend and fit seasonality
detrended = log_prices - trend_hat
seasonal_hat, season_coeffs = fit_fourier_seasonality(detrended, period=252, n_harmonics=6)

print(f"\nSeasonality Estimation:")
print(f"  Fourier harmonics used: 6")
print(f"  Seasonal amplitude: {(seasonal_hat.max() - seasonal_hat.min())*100:.2f}%")

# Seasonal pattern by month (approximate)
monthly_seasonal = np.zeros(12)
days_per_month = 21  # approximate trading days per month
for m in range(12):
    start = m * days_per_month
    end = start + days_per_month
    indices = np.arange(start, min(end, 252))
    monthly_seasonal[m] = np.mean(seasonal_hat[indices % 252])

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
print(f"\n  Monthly Seasonal Effects:")
for m in range(12):
    bar = "+" * max(0, int(monthly_seasonal[m] * 1000)) + \
          "-" * max(0, int(-monthly_seasonal[m] * 1000))
    print(f"    {month_names[m]}: {monthly_seasonal[m]*100:>+6.3f}% {bar}")

# --- Step 3: Event Detection ---
def detect_events(residuals, threshold=2.0):
    """Detect significant events (outliers) in residuals."""
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    z_scores = (residuals - mean_r) / std_r

    event_mask = np.abs(z_scores) > threshold
    event_indices = np.where(event_mask)[0]
    event_magnitudes = residuals[event_mask]

    return event_indices, event_magnitudes, z_scores

residuals = log_prices - trend_hat - seasonal_hat
event_idx, event_mag, z_scores = detect_events(residuals, threshold=2.5)

print(f"\nEvent Detection:")
print(f"  Total events detected (|z| > 2.5): {len(event_idx)}")
print(f"  Positive events (jumps): {np.sum(event_mag > 0)}")
print(f"  Negative events (drops): {np.sum(event_mag < 0)}")

if len(event_idx) > 0:
    print(f"\n  Top 5 Events by Magnitude:")
    sorted_idx = np.argsort(np.abs(event_mag))[::-1][:5]
    for i in sorted_idx:
        day = event_idx[i]
        mag = event_mag[i]
        direction = "UP" if mag > 0 else "DOWN"
        print(f"    Day {day}: {direction} {abs(mag)*100:.2f}% (z={z_scores[day]:.2f})")

# --- Step 4: Forecasting ---
print(f"\n--- Forecasting Next 21 Trading Days ---")

n_forecast = 21
train_size = n_trading_days - n_forecast
train_log = log_prices[:train_size]

# Re-fit on training data
trend_tr, _, _, _ = fit_piecewise_trend(train_log, n_changepoints=15)
detrended_tr = train_log - trend_tr
seasonal_tr, s_coeffs = fit_fourier_seasonality(detrended_tr, period=252, n_harmonics=6)

# Extrapolate trend
t_norm = np.arange(n_trading_days, dtype=float) / train_size
# Simple linear extrapolation from last trend slope
last_slope = trend_tr[-1] - trend_tr[-2]
trend_forecast = np.zeros(n_forecast)
for i in range(n_forecast):
    trend_forecast[i] = trend_tr[-1] + last_slope * (i + 1)

# Extrapolate seasonality
t_future = np.arange(train_size, train_size + n_forecast, dtype=float)
X_future = np.ones((n_forecast, 2 * 6 + 1))
for h in range(1, 7):
    X_future[:, 2*h - 1] = np.cos(2 * np.pi * h * t_future / 252)
    X_future[:, 2*h] = np.sin(2 * np.pi * h * t_future / 252)
seasonal_forecast = X_future @ s_coeffs

# Combined forecast
log_forecast = trend_forecast + seasonal_forecast
price_forecast = np.exp(log_forecast)

# Actual values
actual_prices = prices[train_size:]

# Uncertainty (based on residual std)
residual_std = np.std(train_log - trend_tr - seasonal_tr)
ci_factor = 1.96

print(f"\n{'Day':<6} {'Forecast':>10} {'Actual':>10} {'Error%':>10} {'95% CI':>24}")
print("-" * 64)
for i in range(n_forecast):
    fc = price_forecast[i]
    ac = actual_prices[i]
    err = (fc - ac) / ac * 100
    ci_lo = np.exp(log_forecast[i] - ci_factor * residual_std * np.sqrt(i+1))
    ci_hi = np.exp(log_forecast[i] + ci_factor * residual_std * np.sqrt(i+1))
    print(f"  {i+1:<4} ${fc:>8.2f}  ${ac:>8.2f}  {err:>+8.2f}%  [${ci_lo:.2f}, ${ci_hi:.2f}]")

# Evaluation
rmse = np.sqrt(np.mean((price_forecast - actual_prices)**2))
mape = np.mean(np.abs((price_forecast - actual_prices) / actual_prices)) * 100
print(f"\nForecast Evaluation:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")
```

## Advantages

1. **Intuitive decomposition for equity analysis**: Prophet decomposes stock prices into trend, seasonality, and event components that align naturally with how fundamental analysts think. The trend reflects the company's growth trajectory, seasonality captures predictable cycles, and events model earnings surprises and macro shocks.

2. **Robust handling of missing data and outliers**: Stock market data frequently has gaps (holidays, halted trading) and outliers (flash crashes, gap openings). Prophet handles both gracefully without requiring imputation or data cleaning, making it practical for large-scale automated forecasting.

3. **Configurable domain knowledge integration**: Analysts can specify known events (FOMC meetings, earnings dates, index rebalancing) as holidays with configurable windows. This allows the model to learn event-specific effects while maintaining the overall seasonal structure, blending data-driven learning with expert knowledge.

4. **Automatic changepoint detection**: Prophet automatically identifies structural breaks in the trend, which correspond to regime changes in stock markets (onset of bear markets, sector rotations, fundamental shifts in company growth). The analyst can inspect these changepoints and validate them against known market events.

5. **Scalable to many securities**: Prophet was designed to forecast thousands of time series automatically. A portfolio manager covering 500 stocks can generate decompositions and forecasts for the entire universe without manual tuning, then focus attention on stocks where the decomposition reveals interesting patterns.

6. **Uncertainty quantification through Bayesian inference**: Prophet provides principled prediction intervals through posterior sampling. For stock market applications, these intervals reflect uncertainty in both the trend trajectory and seasonal patterns, supporting risk-aware decision making.

7. **Flexible seasonality specification**: Prophet can model multiple overlapping seasonal patterns simultaneously using Fourier series of different periods. For stocks, this means capturing daily (intraday patterns), weekly (day-of-week effects), monthly (options expiration), and annual (tax-loss selling, January Effect) cycles in a single model.

## Disadvantages

1. **Not designed for financial returns**: Prophet was built for business metrics (daily active users, revenue, page views) that tend to be positive and trend-following. Stock returns are noisy, mean-reverting, and can be negative, making Prophet's trend extrapolation potentially misleading for medium-term price forecasts.

2. **Trend extrapolation risk**: Prophet extrapolates the last detected trend linearly (or logistically) into the future. For stocks, this can be dangerous: a stock trending up at 20% annually will not continue indefinitely, and Prophet lacks fundamental constraints to prevent unrealistic extrapolations.

3. **Limited volatility modeling**: Prophet models the mean of the series but not its variance. Stock market volatility is itself a time-varying process (GARCH effects, volatility clustering) that Prophet cannot capture, making its confidence intervals potentially unreliable during high-volatility regimes.

4. **Overfitting to seasonal patterns**: With many Fourier terms and event indicators, Prophet can overfit to historical seasonal patterns in stock data that may not persist. The model may find spurious weekly or monthly patterns in what is essentially random noise.

5. **No cross-sectional information**: Prophet is a univariate model that processes each stock independently. It cannot capture cross-asset correlations, sector momentum, or market-wide regime changes that are essential for portfolio-level forecasting.

6. **Inefficient market assumption**: Prophet implicitly assumes that historical patterns (trend, seasonality, events) will continue into the future. In relatively efficient stock markets, predictable patterns are quickly arbitraged away, limiting Prophet's out-of-sample forecasting power for liquid, large-cap stocks.

7. **Changepoint sensitivity**: The number and placement of changepoints significantly affects forecasts. Too few changepoints may miss genuine regime changes; too many may overfit to noise. For volatile stocks, this sensitivity can lead to unstable forecasts across different training windows.

## When to Use in Stock Market

- Decomposing stock price movements into interpretable trend, seasonal, and event components for equity research reports
- Identifying and quantifying the impact of known events (earnings, FOMC, rebalancing) on individual stock prices
- Generating long-term trend projections for valuation exercises (with appropriate fundamental constraints)
- Automated forecasting across large stock universes where manual model tuning is impractical
- Detecting regime changes (changepoints) in stock price trends for systematic trading strategies
- Modeling seasonal patterns in sector ETFs or thematic portfolios
- Creating visual decomposition dashboards for portfolio managers and clients
- Forecasting trading volume or bid-ask spread (non-price metrics) which may follow more predictable patterns

## When NOT to Use in Stock Market

- For short-term tactical trading (1-5 day horizon) where ARIMA or machine learning models are more appropriate
- When forecasting volatile small-cap or penny stocks with erratic behavior and frequent regime changes
- For high-frequency or intraday price forecasting where microstructure effects dominate
- When the primary goal is volatility forecasting (use GARCH or realized volatility models instead)
- For portfolio optimization requiring multi-asset correlation modeling
- When the stock has limited history (less than 2 years) insufficient for seasonal estimation
- For derivative pricing where option-implied information is more relevant than historical patterns
- When the fundamental story has changed dramatically (M&A, restructuring) invalidating historical patterns

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| growth | Trend type: linear or logistic | 'linear', 'logistic' | 'linear' for most stocks; 'logistic' for bounded metrics |
| n_changepoints | Number of potential trend changepoints | 10-50 | 25 for daily data (5+ years); increase for volatile stocks |
| changepoint_prior_scale | Flexibility of trend changes (tau) | 0.001-0.5 | 0.05 default; lower for stable blue-chips, higher for growth stocks |
| seasonality_prior_scale | Strength of seasonal components | 0.01-10 | 1.0 default; lower if seasonal overfitting suspected |
| holidays_prior_scale | Strength of holiday/event effects | 0.01-10 | 10.0 for earnings effects; 1.0 for minor events |
| yearly_seasonality | Number of Fourier terms for yearly | 3-20 | 6-10 for stocks; captures major seasonal patterns without overfitting |
| weekly_seasonality | Number of Fourier terms for weekly | 1-6 | 3 for daily data; captures day-of-week effects |
| seasonality_mode | Additive or multiplicative | 'additive', 'multiplicative' | 'multiplicative' for prices; 'additive' for returns |
| interval_width | Width of prediction interval | 0.8-0.99 | 0.95 for risk management; 0.80 for trading signals |
| mcmc_samples | Number of MCMC samples (0 = MAP) | 0, 300-1000 | 300+ for uncertainty quantification; 0 for speed |

## Stock Market Performance Tips

1. **Use log-transformed prices**: Apply log transformation before fitting Prophet. This converts multiplicative patterns to additive ones, stabilizes variance, and ensures price forecasts remain positive when exponentiated back.

2. **Create a financial calendar**: Build a comprehensive holiday DataFrame including market closures, half-days, options expirations, FOMC meetings, and company-specific earnings dates. Each event type should have its own entry with appropriate lower/upper windows.

3. **Limit the forecast horizon**: Prophet works best for decomposition and short-to-medium term forecasts. For stocks, limit forecasts to 1-3 months and refit the model monthly. Long-term stock forecasts are inherently unreliable regardless of the method.

4. **Add regressors judiciously**: Include market-level regressors (VIX, sector ETF returns, interest rates) as additional regressors, but be cautious about overfitting. These regressors must also be forecastable for out-of-sample predictions.

5. **Validate changepoints against market events**: After fitting, inspect the detected changepoints and verify they correspond to known market events (COVID crash, Fed pivot, earnings surprise). Unexplained changepoints may indicate overfitting.

6. **Use cross-validation with appropriate gaps**: When cross-validating, use a gap between training and validation to prevent look-ahead bias. For daily stock data, a gap of 5-10 trading days is typical to account for autocorrelation in features.

7. **Compare with naive forecasts**: Always benchmark Prophet against simple baselines (random walk, seasonal naive, drift model). For efficient stock markets, beating these baselines is non-trivial and validates that the model captures genuine patterns.

## Comparison with Other Algorithms

| Feature | Prophet | ARIMA | SARIMA | LSTM | TFT |
|---------|---------|-------|--------|------|-----|
| Trend modeling | Piecewise linear/logistic | None (differencing) | None (differencing) | Implicit | Implicit |
| Seasonality | Fourier (multiple) | None | Single period | Learned | Learned |
| Event effects | Built-in holidays | No | No | Manual encoding | Manual encoding |
| Changepoints | Automatic detection | No | No | No | No |
| Missing data | Handled natively | Requires imputation | Requires imputation | Requires imputation | Requires imputation |
| Interpretability | Very high (decomposition) | High | High | Low | Medium (attention) |
| Scalability | Excellent (designed for scale) | Good | Good | Moderate | Poor |
| Nonlinear patterns | Limited | No | No | Yes | Yes |
| Multi-horizon | Yes | Recursive | Recursive | Yes | Yes (native) |
| Stock market fit | Moderate (decomposition) | Good (short-term) | Good (seasonal) | Good (patterns) | Best (multi-horizon) |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete Prophet-Style Decomposition for Stock Market Data
# Trend + Seasonality + Events using NumPy
# ============================================================

np.random.seed(2024)

# --- Generate Realistic Stock Data with Known Components ---
n_days = 1260  # 5 years of trading days
t = np.arange(n_days, dtype=float)

# TRUE COMPONENTS (for validation)
# Trend: piecewise linear with 4 regime changes
true_trend = np.zeros(n_days)
rates = [0.0003, 0.0008, -0.0003, 0.0006, 0.0002]
breakpoints = [0, 250, 500, 750, 1000]
current_val = np.log(200.0)
rate_idx = 0
for i in range(n_days):
    if rate_idx < len(breakpoints) - 1 and i >= breakpoints[rate_idx + 1]:
        rate_idx += 1
    true_trend[i] = current_val
    current_val += rates[rate_idx]

# Seasonality: annual cycle (Fourier)
true_seasonal = (0.008 * np.sin(2*np.pi*t/252) +
                 0.005 * np.cos(2*np.pi*t/252) +
                 0.003 * np.sin(4*np.pi*t/252) +
                 0.002 * np.cos(4*np.pi*t/252))

# Events: quarterly earnings (every 63 days)
true_events = np.zeros(n_days)
earnings_days = np.arange(42, n_days, 63)
for ed in earnings_days:
    surprise = np.random.normal(0.015, 0.008)
    true_events[ed] = surprise
    if ed + 1 < n_days:
        true_events[ed + 1] = surprise * 0.3
    if ed + 2 < n_days:
        true_events[ed + 2] = surprise * 0.1

# Noise
noise = np.random.normal(0, 0.012, n_days)

# Combined series
log_prices = true_trend + true_seasonal + true_events + noise
prices = np.exp(log_prices)

# Train/test split
train_n = 1200
test_n = n_days - train_n
train_lp = log_prices[:train_n]
test_lp = log_prices[train_n:]

print("=" * 65)
print("Prophet-Style Stock Price Decomposition")
print("=" * 65)
print(f"Training: {train_n} days, Testing: {test_n} days")
print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")

# ============================================================
# STEP 1: TREND ESTIMATION - Piecewise Linear with L1 Penalty
# ============================================================
print(f"\n--- Step 1: Trend Estimation ---")

def fit_trend_with_changepoints(y, n_cp=20, lambda_l1=0.5):
    """Fit piecewise linear trend with regularized changepoints."""
    n = len(y)
    t_norm = np.arange(n, dtype=float) / n

    # Changepoint locations (first 80%)
    cp_positions = np.linspace(0.05, 0.8, n_cp)

    # Design matrix
    X = np.ones((n, 2 + n_cp))
    X[:, 1] = t_norm
    for j, cp in enumerate(cp_positions):
        X[:, 2 + j] = np.maximum(0, t_norm - cp)

    # Iteratively reweighted least squares for approximate L1
    weights = np.ones(2 + n_cp)
    for iteration in range(10):
        W = np.diag(np.concatenate([[0, 0], lambda_l1 / (np.abs(weights[2:]) + 1e-6)]))
        beta = np.linalg.solve(X.T @ X + W, X.T @ y)
        weights = beta.copy()

    trend = X @ beta

    # Identify significant changepoints
    cp_magnitudes = np.abs(beta[2:])
    threshold = np.percentile(cp_magnitudes, 70)
    significant = cp_magnitudes > threshold

    return trend, beta, cp_positions, significant

trend_est, trend_beta, cp_pos, cp_sig = fit_trend_with_changepoints(
    train_lp, n_cp=20, lambda_l1=0.3
)

print(f"Changepoints detected: {cp_sig.sum()} significant out of 20")
sig_days = (cp_pos[cp_sig] * train_n).astype(int)
print(f"Significant changepoint days: {sig_days.tolist()}")
print(f"Trend RMSE vs true: {np.sqrt(np.mean((trend_est - true_trend[:train_n])**2)):.6f}")

# ============================================================
# STEP 2: SEASONALITY ESTIMATION - Fourier Series
# ============================================================
print(f"\n--- Step 2: Seasonality Estimation ---")

detrended = train_lp - trend_est

def fit_fourier_season(residuals, periods_and_orders):
    """Fit multiple Fourier seasonalities."""
    n = len(residuals)
    t = np.arange(n, dtype=float)

    # Build combined design matrix
    cols = [np.ones(n)]
    for period, order in periods_and_orders:
        for h in range(1, order + 1):
            cols.append(np.cos(2 * np.pi * h * t / period))
            cols.append(np.sin(2 * np.pi * h * t / period))

    X = np.column_stack(cols)

    # Ridge regression
    lambda_reg = 5.0
    beta = np.linalg.solve(X.T @ X + lambda_reg * np.eye(X.shape[1]), X.T @ residuals)

    seasonal = X @ beta
    return seasonal, beta, X.shape[1]

# Annual (252 trading days) + Weekly (5 trading days)
season_est, season_beta, n_season_params = fit_fourier_season(
    detrended,
    periods_and_orders=[(252, 6), (5, 2)]  # 6 annual + 2 weekly harmonics
)

print(f"Seasonal parameters: {n_season_params}")
print(f"Seasonal amplitude: {(season_est.max() - season_est.min())*100:.3f}%")
print(f"Seasonality RMSE vs true: {np.sqrt(np.mean((season_est - true_seasonal[:train_n])**2)):.6f}")

# Monthly averages of estimated seasonality
print(f"\nMonthly Seasonal Pattern (estimated):")
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in range(12):
    start = m * 21
    end = start + 21
    avg = np.mean(season_est[start:end])
    true_avg = np.mean(true_seasonal[start:end])
    print(f"  {month_names[m]}: estimated={avg*100:>+6.3f}%, true={true_avg*100:>+6.3f}%")

# ============================================================
# STEP 3: EVENT DETECTION
# ============================================================
print(f"\n--- Step 3: Event Detection ---")

residuals = train_lp - trend_est - season_est
resid_std = np.std(residuals)
resid_mean = np.mean(residuals)

# Detect outliers as events
z_scores = (residuals - resid_mean) / resid_std
event_mask = np.abs(z_scores) > 2.0
n_events = np.sum(event_mask)

print(f"Residual std: {resid_std:.6f}")
print(f"Events detected (|z| > 2.0): {n_events}")
print(f"True earnings events in training: {len(earnings_days[earnings_days < train_n])}")

# Check overlap with true earnings days
detected_days = np.where(event_mask)[0]
true_event_days = earnings_days[earnings_days < train_n]
matches = 0
for td in true_event_days:
    if any(abs(detected_days - td) <= 1):
        matches += 1
print(f"Earnings events correctly detected: {matches}/{len(true_event_days)}")

# ============================================================
# STEP 4: FORECASTING
# ============================================================
print(f"\n--- Step 4: {test_n}-Day Forecast ---")

# Extrapolate trend
last_slope = trend_est[-1] - trend_est[-2]
trend_fc = np.array([trend_est[-1] + last_slope * (i+1) for i in range(test_n)])

# Extrapolate seasonality
t_fc = np.arange(train_n, train_n + test_n, dtype=float)
cols_fc = [np.ones(test_n)]
for period, order in [(252, 6), (5, 2)]:
    for h in range(1, order + 1):
        cols_fc.append(np.cos(2 * np.pi * h * t_fc / period))
        cols_fc.append(np.sin(2 * np.pi * h * t_fc / period))
X_fc = np.column_stack(cols_fc)
season_fc = X_fc @ season_beta

# Combined forecast
forecast_lp = trend_fc + season_fc
forecast_prices = np.exp(forecast_lp)
actual_prices = np.exp(test_lp)

# Uncertainty bands
ci_mult = np.array([1.96 * resid_std * np.sqrt(min(i+1, 20)) for i in range(test_n)])
upper = np.exp(forecast_lp + ci_mult)
lower = np.exp(forecast_lp - ci_mult)

# Print forecast
print(f"\n{'Day':<6} {'Forecast':>10} {'Actual':>10} {'Error%':>8} {'In CI?':>8}")
print("-" * 46)
for i in range(test_n):
    fc = forecast_prices[i]
    ac = actual_prices[i]
    err = (fc - ac) / ac * 100
    in_ci = "Yes" if lower[i] <= ac <= upper[i] else "No"
    print(f"  {i+1:<4} ${fc:>8.2f}  ${ac:>8.2f}  {err:>+6.2f}%  {in_ci:>6}")

# ============================================================
# STEP 5: EVALUATION
# ============================================================
print(f"\n--- Comprehensive Evaluation ---")

rmse = np.sqrt(np.mean((forecast_prices - actual_prices)**2))
mae = np.mean(np.abs(forecast_prices - actual_prices))
mape = np.mean(np.abs((forecast_prices - actual_prices) / actual_prices)) * 100

# Direction accuracy (using returns)
fc_returns = np.diff(np.log(np.concatenate([[np.exp(train_lp[-1])], forecast_prices])))
ac_returns = np.diff(np.log(np.concatenate([[np.exp(train_lp[-1])], actual_prices])))
dir_acc = np.mean(np.sign(fc_returns) == np.sign(ac_returns)) * 100

# CI coverage
coverage = np.mean((actual_prices >= lower) & (actual_prices <= upper)) * 100

# Naive benchmark (random walk)
naive_forecast = np.full(test_n, np.exp(train_lp[-1]))
naive_rmse = np.sqrt(np.mean((naive_forecast - actual_prices)**2))

print(f"{'Metric':<30} {'Prophet':>12} {'Random Walk':>12}")
print("-" * 56)
print(f"{'RMSE':<30} ${rmse:>10.2f}  ${naive_rmse:>10.2f}")
print(f"{'MAE':<30} ${mae:>10.2f}  {'N/A':>12}")
print(f"{'MAPE':<30} {mape:>10.2f}%  {'N/A':>12}")
print(f"{'Direction Accuracy':<30} {dir_acc:>10.1f}%  {'N/A':>12}")
print(f"{'95% CI Coverage':<30} {coverage:>10.1f}%  {'N/A':>12}")

# Component variance decomposition
total_var = np.var(train_lp)
trend_var = np.var(trend_est)
season_var = np.var(season_est)
resid_var = np.var(residuals)

print(f"\nVariance Decomposition:")
print(f"  Trend:       {trend_var/total_var*100:>6.1f}%")
print(f"  Seasonality: {season_var/total_var*100:>6.1f}%")
print(f"  Residual:    {resid_var/total_var*100:>6.1f}%")
```

## Key Takeaways

1. Prophet excels at decomposing stock price movements into interpretable components (trend, seasonality, events), making it a powerful exploratory tool for equity research even when pure forecasting accuracy is modest.

2. The Fourier-based seasonality captures smooth seasonal patterns well but may miss sharp, localized effects like specific event days. Complement with explicit event indicators for earnings dates and macro events.

3. Automatic changepoint detection identifies regime shifts in stock price trends, which can be validated against known market events (policy changes, earnings surprises, sector rotations) to build narrative around price movements.

4. Prophet is most valuable in stock markets for its scalability: it can automatically generate decompositions and forecasts across hundreds of securities, flagging stocks where seasonal patterns or trend changes deserve analyst attention.

5. Treat Prophet forecasts as one signal among many. The trend extrapolation is inherently uncertain for stocks, and the model should be combined with fundamental analysis, risk models, and alternative data for investment decisions.

6. Always benchmark against simple baselines (random walk, seasonal naive). In efficient stock markets, Prophet's value may come more from its decomposition insights than from its raw forecasting accuracy.

7. Use Prophet's prediction intervals cautiously for risk management. They assume stationary residual variance, which is often violated in stock markets during stress periods. Augment with scenario analysis and stress testing.

8. The model works best when applied to log-transformed prices with multiplicative seasonality, capturing the percentage-based nature of stock returns and ensuring positive price forecasts.
