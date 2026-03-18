# SARIMA (Seasonal ARIMA) - Complete Guide with Stock Market Applications

## Overview

SARIMA (Seasonal AutoRegressive Integrated Moving Average) extends the standard ARIMA model by adding seasonal components to capture repeating patterns at fixed intervals. Denoted as ARIMA(p, d, q)(P, D, Q)[m], it combines both non-seasonal parameters (p, d, q) and seasonal parameters (P, D, Q) operating at a seasonal period m. This dual structure allows SARIMA to simultaneously model short-term dynamics and longer-term cyclical behavior.

In stock markets, seasonal patterns are well-documented phenomena. The "January Effect" refers to historically higher returns in January as investors reinvest after year-end tax-loss selling. The "Sell in May" anomaly suggests returns from May through October are systematically lower than November through April. Quarterly earnings cycles create predictable volatility patterns every 13 weeks. Day-of-week effects show different return distributions on Mondays versus Fridays. SARIMA provides a principled statistical framework for modeling these recurring patterns.

While the efficient market hypothesis suggests that known seasonal patterns should be arbitraged away, empirical evidence shows that many seasonal effects persist, particularly in smaller-cap stocks and emerging markets. SARIMA helps quantify the strength of these patterns, measure their statistical significance, and generate forecasts that account for where we are in the seasonal cycle. It is particularly valuable for risk management, where understanding predictable volatility patterns (such as increased variance around earnings dates) directly impacts hedging strategies and option pricing.

## How It Works - The Math Behind It

### The Full SARIMA Model: ARIMA(p, d, q)(P, D, Q)[m]

The model combines non-seasonal and seasonal components through a multiplicative structure:

```
Phi_P(B^m) * phi_p(B) * nabla^d * nabla_m^D * y_t = Theta_Q(B^m) * theta_q(B) * epsilon_t
```

Where:
- `B` is the backshift operator: B*y_t = y_{t-1}
- `B^m` is the seasonal backshift: B^m * y_t = y_{t-m}
- `nabla^d = (1 - B)^d` is non-seasonal differencing
- `nabla_m^D = (1 - B^m)^D` is seasonal differencing

### Non-Seasonal Components

**AR(p) - AutoRegressive:**
```
phi_p(B) = 1 - phi_1*B - phi_2*B^2 - ... - phi_p*B^p
```

**MA(q) - Moving Average:**
```
theta_q(B) = 1 + theta_1*B + theta_2*B^2 + ... + theta_q*B^q
```

### Seasonal Components

**Seasonal AR(P):**
```
Phi_P(B^m) = 1 - Phi_1*B^m - Phi_2*B^{2m} - ... - Phi_P*B^{Pm}
```
This captures how the value m periods ago (same point in the seasonal cycle) influences the current value.

**Seasonal MA(Q):**
```
Theta_Q(B^m) = 1 + Theta_1*B^m + Theta_2*B^{2m} + ... + Theta_Q*B^{Qm}
```
This captures how the forecast error m periods ago influences the current value.

**Seasonal Differencing:**
```
nabla_m^D * y_t = (1 - B^m)^D * y_t
```
For D=1: `y_t - y_{t-m}` (year-over-year or quarter-over-quarter change)

### Example: ARIMA(1,1,1)(1,1,1)[4] for Quarterly Stock Data

Expanding the model for quarterly data (m=4):

```
(1 - Phi_1*B^4)(1 - phi_1*B)(1-B)(1-B^4)*y_t = (1 + Theta_1*B^4)(1 + theta_1*B)*epsilon_t
```

Step by step:
1. Seasonal difference: z_t = y_t - y_{t-4} (removes quarterly pattern)
2. Non-seasonal difference: w_t = z_t - z_{t-1} (removes trend)
3. AR components: w_t depends on w_{t-1} (short-term) and w_{t-4} (seasonal)
4. MA components: w_t depends on epsilon_{t-1} (short-term) and epsilon_{t-4} (seasonal)
5. Cross terms: w_{t-5} appears from the multiplicative interaction of AR terms

### Parameter Estimation

Parameters are estimated via Maximum Likelihood Estimation (MLE). The likelihood function for the Gaussian case:

```
L(params | y) = (2*pi*sigma^2)^{-n/2} * exp(-1/(2*sigma^2) * sum(epsilon_t^2))
```

Where epsilon_t are the one-step-ahead prediction errors computed via the Kalman filter or innovations algorithm.

### Seasonal Decomposition

Before fitting SARIMA, it is useful to decompose the series:

```
y_t = T_t + S_t + R_t  (Additive)
y_t = T_t * S_t * R_t  (Multiplicative)
```

Where T_t is trend, S_t is seasonal component, and R_t is residual. For stock returns, additive decomposition is typically appropriate since returns are already on a relative scale.

## Stock Market Use Case: Quarterly Earnings Cycle and Seasonal Volatility Patterns

### The Problem

A portfolio risk manager at an asset management firm needs to model the seasonal volatility patterns in a diversified equity portfolio. Historically, the portfolio shows increased volatility during earnings seasons (January, April, July, October), reduced volatility in summer months (June-August), and a year-end rally pattern (November-December). The risk manager needs to forecast volatility 3 months ahead to set appropriate risk limits, adjust hedge ratios, and plan options-based protection strategies.

### Stock Market Features (Input Data)

| Feature | Description | Seasonal Relevance |
|---------|-------------|-------------------|
| Monthly Returns | Monthly log returns of the stock | Primary series |
| Monthly Realized Volatility | Standard deviation of daily returns within month | Volatility seasonality |
| Earnings Month Flag | Binary indicator for earnings months | Quarterly cycle (m=4) |
| January Indicator | Binary flag for January | January Effect |
| Quarter-End Flag | End of fiscal quarter | Rebalancing flows |
| Tax-Loss Selling Period | November-December flag | Year-end patterns |
| Options Expiration Week | Monthly/quarterly expiration | Volatility spikes |
| Trading Days in Month | Number of trading days | Volume normalization |
| VIX Monthly Average | Market-wide volatility | Regime context |
| Sector Rotation Index | Relative strength across sectors | Cross-sector seasonality |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)

# Generate 10 years of monthly stock returns with seasonal patterns
n_years = 10
n_months = n_years * 12
months = np.tile(np.arange(1, 13), n_years)  # 1-12 repeated

# Base return parameters
base_mu = 0.008  # ~10% annual return
base_sigma = 0.04  # ~14% annual volatility

# Seasonal return adjustments (monthly effects)
seasonal_return = np.array([
    0.012,   # January: strong (January effect)
    0.002,   # February: slightly positive
    0.001,   # March: neutral
    0.005,   # April: moderate (earnings optimism)
    -0.002,  # May: "Sell in May"
    -0.001,  # June: summer lull
    0.003,   # July: earnings season
    -0.003,  # August: summer doldrums
    -0.005,  # September: historically weakest month
    0.004,   # October: recovery/volatility
    0.006,   # November: pre-holiday rally
    0.008,   # December: Santa rally
])

# Seasonal volatility adjustments
seasonal_vol = np.array([
    1.3,  # January: high vol (earnings + new year flows)
    1.0,  # February: normal
    1.1,  # March: quarter-end rebalancing
    1.2,  # April: earnings season
    0.9,  # May: declining activity
    0.8,  # June: low summer vol
    1.2,  # July: earnings season
    0.7,  # August: vacation season, lowest vol
    1.1,  # September: return from summer
    1.4,  # October: historically volatile
    1.0,  # November: normal
    0.9,  # December: low vol, holiday trading
])

# Generate returns with seasonal patterns
returns = np.zeros(n_months)
for t in range(n_months):
    m = months[t] - 1  # 0-indexed month
    mu_t = base_mu + seasonal_return[m]
    sigma_t = base_sigma * seasonal_vol[m]
    returns[t] = np.random.normal(mu_t, sigma_t)

# Convert to prices
initial_price = 100.0
prices = initial_price * np.cumprod(1 + returns)

print("Monthly Return Seasonal Averages:")
for m in range(12):
    month_returns = returns[months == m + 1]
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    print(f"  {month_names[m]}: mean={month_returns.mean()*100:+.2f}%, "
          f"std={month_returns.std()*100:.2f}%")
```

### The Model in Action

```python
import numpy as np

# Using the returns data generated above
# SARIMA(1,0,1)(1,1,1)[12] for monthly stock returns

# Step 1: Seasonal differencing (D=1, m=12)
# Remove annual seasonal pattern
seasonal_diff = returns[12:] - returns[:-12]
n = len(seasonal_diff)

print(f"Original series length: {len(returns)}")
print(f"After seasonal differencing: {n}")
print(f"Seasonally differenced mean: {seasonal_diff.mean():.6f}")
print(f"Seasonally differenced std:  {seasonal_diff.std():.6f}")

# Step 2: Compute seasonal ACF
def compute_acf(series, max_lag=36):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    acf_vals = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if n - lag < 1:
            break
        cov = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
        acf_vals[lag] = cov / var if var > 0 else 0
    return acf_vals

acf = compute_acf(seasonal_diff, max_lag=36)

# Check for remaining seasonal correlation
print(f"\nACF at seasonal lags:")
for lag in [1, 2, 3, 6, 12, 24, 36]:
    if lag < len(acf):
        print(f"  Lag {lag:2d}: {acf[lag]:+.4f}")

# Step 3: Fit seasonal AR(1) component at lag 12
# After seasonal differencing, check lag-12 correlation
y = seasonal_diff[12:]
y_seasonal_lag = seasonal_diff[:-12]

# Simple seasonal AR(1) fit
n_fit = len(y)
Phi_1 = np.sum(y * y_seasonal_lag) / np.sum(y_seasonal_lag**2)
print(f"\nSeasonal AR(1) coefficient (Phi_1): {Phi_1:.4f}")

# Step 4: Also fit non-seasonal AR(1) on residuals
residuals_seasonal = y - Phi_1 * y_seasonal_lag
y_ns = residuals_seasonal[1:]
y_ns_lag = residuals_seasonal[:-1]
phi_1 = np.sum(y_ns * y_ns_lag) / np.sum(y_ns_lag**2)
print(f"Non-seasonal AR(1) coefficient (phi_1): {phi_1:.4f}")

# Step 5: Final residuals
final_residuals = y_ns - phi_1 * y_ns_lag
sigma = np.std(final_residuals)
print(f"Residual std: {sigma:.6f}")

# Step 6: Forecast next 12 months
train_end = n_months - 12
train_returns = returns[:train_end]
test_returns = returns[train_end:]

# Re-fit on training data
sd_train = train_returns[12:] - train_returns[:-12]
y_tr = sd_train[12:]
y_tr_sl = sd_train[:-12]
Phi_1_fit = np.sum(y_tr * y_tr_sl) / np.sum(y_tr_sl**2)

res_tr = y_tr - Phi_1_fit * y_tr_sl
y_ns_tr = res_tr[1:]
y_ns_lag_tr = res_tr[:-1]
phi_1_fit = np.sum(y_ns_tr * y_ns_lag_tr) / np.sum(y_ns_lag_tr**2)

sigma_fit = np.std(y_ns_tr - phi_1_fit * y_ns_lag_tr)

print(f"\nModel fitted on training data:")
print(f"  Phi_1 (seasonal AR): {Phi_1_fit:.4f}")
print(f"  phi_1 (non-seasonal AR): {phi_1_fit:.4f}")
print(f"  sigma: {sigma_fit:.6f}")

# Generate 12-month ahead forecasts
forecasts = np.zeros(12)
last_12_returns = train_returns[-12:]  # For seasonal reference
last_24_returns = train_returns[-24:]  # For seasonal differenced reference

for h in range(12):
    # Seasonal reference: return from 12 months ago
    seasonal_ref = last_12_returns[h]

    # Seasonal differenced value from 12 months ago
    if h < len(sd_train) and len(sd_train) >= 12:
        sd_ref = sd_train[-(12 - h)] if h < 12 else 0
    else:
        sd_ref = 0

    # Forecast the seasonally differenced value
    sd_forecast = Phi_1_fit * sd_ref

    # Add back the seasonal level
    forecasts[h] = seasonal_ref + sd_forecast

print(f"\n12-Month Forecast vs Actual:")
print(f"{'Month':<8} {'Forecast':>10} {'Actual':>10} {'Error':>10}")
print("-" * 40)
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
start_month = months[train_end]
for h in range(12):
    m_idx = (start_month - 1 + h) % 12
    print(f"{month_names[m_idx]:<8} {forecasts[h]*100:>+9.2f}% {test_returns[h]*100:>+9.2f}% "
          f"{(forecasts[h]-test_returns[h])*100:>+9.2f}%")

# Evaluation
rmse = np.sqrt(np.mean((forecasts - test_returns)**2))
mae = np.mean(np.abs(forecasts - test_returns))
dir_acc = np.mean(np.sign(forecasts) == np.sign(test_returns)) * 100

print(f"\nForecast Evaluation:")
print(f"  RMSE:               {rmse*100:.4f}%")
print(f"  MAE:                {mae*100:.4f}%")
print(f"  Direction Accuracy: {dir_acc:.1f}%")
```

## Advantages

1. **Captures documented seasonal anomalies**: SARIMA directly models well-known stock market seasonalities such as the January Effect, quarterly earnings cycles, and the Halloween indicator. By explicitly parameterizing these patterns, it provides statistically testable evidence for their presence and strength.

2. **Multiplicative seasonal structure**: The multiplicative combination of seasonal and non-seasonal components allows SARIMA to capture interaction effects. For example, the impact of a positive earnings surprise in Q1 may differ from Q3 due to seasonal modulation, and SARIMA naturally accommodates this.

3. **Analytical forecasting framework**: Like ARIMA, SARIMA provides analytical point forecasts and confidence intervals. For seasonal forecasting (e.g., predicting next January's return based on historical January patterns), the confidence intervals correctly account for both seasonal and non-seasonal uncertainty.

4. **Handles calendar effects naturally**: By choosing the appropriate seasonal period m, SARIMA can model various calendar effects: m=12 for monthly seasonality, m=4 for quarterly patterns, m=5 for day-of-week effects in daily data, and m=252 for annual patterns in daily data.

5. **Decomposition insight**: The seasonal decomposition implicit in SARIMA helps portfolio managers understand what portion of a stock's return is attributable to seasonal factors versus idiosyncratic movements. This separation is valuable for timing entry and exit points around seasonal windows.

6. **Robust to missing seasonal cycles**: SARIMA gracefully handles cases where seasonal patterns vary in strength across cycles. The seasonal parameters are estimated across all available cycles, producing robust average seasonal effects rather than overfitting to any single year.

7. **Compatible with risk management frameworks**: Seasonal volatility forecasts from SARIMA integrate naturally into VaR models and stress testing frameworks, allowing risk managers to set time-varying risk limits that reflect predictable seasonal changes in market volatility.

## Disadvantages

1. **Requires sufficient seasonal cycles**: SARIMA needs multiple complete seasonal cycles for reliable estimation. For annual seasonality (m=12 with monthly data), this means at least 3-4 years of data. For newer stocks or recently IPO'd companies, insufficient history limits seasonal modeling accuracy.

2. **Fixed seasonal period assumption**: SARIMA assumes the seasonal period m is constant and known. In stock markets, seasonal patterns can shift: earnings dates change, regulatory calendars evolve, and market microstructure effects (like options expiration) can move. The rigid seasonal period cannot accommodate these shifts.

3. **Large parameter space**: SARIMA(p,d,q)(P,D,Q)[m] has up to 7 parameters to select plus the seasonal period. With m=12 (monthly data), the model can have terms at lags up to 12*P + p, leading to complex estimation challenges and potential overfitting.

4. **Weakening seasonal effects in efficient markets**: As more market participants become aware of seasonal patterns and trade on them, these patterns tend to weaken over time. SARIMA estimated on historical data may overweight seasonal effects that have since been arbitraged away.

5. **Linear seasonal assumption**: SARIMA models seasonal effects as additive or multiplicative linear patterns. In reality, stock market seasonality often interacts nonlinearly with other factors such as market regime, volatility environment, and macroeconomic conditions.

6. **Computational cost with high-frequency data**: When m is large (e.g., m=252 for daily data with annual seasonality), the state space dimension grows substantially, making estimation slow and numerically challenging. This limits SARIMA's practical application to lower-frequency data.

7. **Difficulty with multiple seasonal patterns**: Stock markets can exhibit multiple overlapping seasonal patterns (day-of-week, month-of-year, earnings cycle). Standard SARIMA handles only a single seasonal period. While extensions like double-seasonal SARIMA exist, they are complex and prone to overfitting.

## When to Use in Stock Market

- Modeling monthly or quarterly return patterns for strategic asset allocation timing
- Forecasting earnings-season volatility for options trading strategies
- Identifying and quantifying calendar anomalies (January Effect, Sell in May) in specific stocks or sectors
- Setting seasonal risk limits for portfolio managers based on historical volatility patterns
- Mean-reversion trading strategies where seasonal deviations from trend create entry opportunities
- Sector rotation strategies driven by predictable quarterly flows (tax selling, mutual fund window dressing)
- Commodity-linked stocks with inherent supply/demand seasonality (energy, agriculture)
- REITs and dividend stocks with known seasonal distribution patterns

## When NOT to Use in Stock Market

- For daily or intraday forecasting where seasonal period m would be very large (252 or more)
- When trading recently listed stocks with less than 3 years of price history
- For stocks experiencing fundamental regime changes (M&A, sector reclassification) that invalidate historical seasonal patterns
- When the primary signal is event-driven (clinical trial results, regulatory decisions) rather than calendar-driven
- For cryptocurrency markets where seasonal patterns are less established and structural shifts are common
- When cross-sectional (multi-stock) analysis is more important than time-series patterns
- For high-frequency strategies where microstructure effects dominate seasonal patterns
- When seasonal patterns have been demonstrably weakened by widespread adoption of seasonal trading strategies

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| p (AR order) | Non-seasonal autoregressive order | 0-3 | p=1-2; captures short-term momentum/mean-reversion |
| d (Differencing) | Non-seasonal differencing order | 0-1 | d=0 for returns; d=1 for prices |
| q (MA order) | Non-seasonal moving average order | 0-3 | q=1; captures short-term shock persistence |
| P (Seasonal AR) | Seasonal autoregressive order | 0-2 | P=1; captures year-over-year correlation |
| D (Seasonal diff) | Seasonal differencing order | 0-1 | D=1 for strong seasonality; D=0 if seasonality is weak |
| Q (Seasonal MA) | Seasonal moving average order | 0-2 | Q=1; captures seasonal shock effects |
| m (Seasonal period) | Length of seasonal cycle | 4, 12, 52, 252 | m=12 for monthly; m=4 for quarterly; m=252 for daily |
| Trend | Deterministic trend component | c, t, ct, n | 'c' if returns have non-zero mean; 'n' for mean-zero returns |
| Training window | Historical data for estimation | 3m-10m periods | At least 4 complete seasonal cycles |

## Stock Market Performance Tips

1. **Test for seasonality before modeling it**: Use the Kruskal-Wallis test or seasonal subseries plots to verify that statistically significant seasonal patterns exist. Blindly fitting SARIMA to non-seasonal data leads to overfitting and poor forecasts.

2. **Use multiplicative decomposition for volatility**: When modeling seasonal volatility patterns, multiplicative decomposition (where seasonal effects scale with the level) is more appropriate since volatility tends to be proportional to price levels.

3. **Account for changing market regimes**: Fit SARIMA on rolling windows and track whether seasonal coefficients are stable over time. If the January Effect coefficient is declining, it may indicate the anomaly is being arbitraged away.

4. **Beware of data mining bias**: With 12 months, 4 quarters, and 5 weekdays, there are many potential seasonal patterns to test. Apply Bonferroni corrections or use holdout validation to avoid discovering spurious seasonal effects.

5. **Combine seasonal forecasts with fundamental analysis**: Use SARIMA seasonal forecasts as one input among many. A strong seasonal sell signal in September combined with deteriorating fundamentals is more actionable than either signal alone.

6. **Adjust for varying month lengths and holidays**: Trading days vary across months and years. Normalize returns by the number of trading days in each month to avoid confusing calendar effects with genuine seasonal patterns.

7. **Consider regime-dependent seasonality**: The January Effect tends to be stronger following years with large tax-loss selling opportunities (after market declines). Building separate SARIMA models for bull and bear market regimes can improve seasonal forecasts.

## Comparison with Other Algorithms

| Feature | SARIMA | ARIMA | Prophet | LSTM | TFT |
|---------|--------|-------|---------|------|-----|
| Seasonal modeling | Explicit (core strength) | None | Automatic (Fourier) | Learned | Learned |
| Multiple seasonalities | Limited (one period) | None | Yes (additive) | Yes | Yes |
| Calendar effects | Through m parameter | No | Holiday regressors | Manual encoding | Manual encoding |
| Interpretability | High (seasonal coefficients) | High | High (decomposition) | Low | Medium |
| Data efficiency | Moderate (needs 3+ cycles) | High | Moderate (2+ years) | Low | Low |
| Nonlinear seasonality | No | No | Partially | Yes | Yes |
| Forecast horizon | Up to 2 seasonal cycles | Short-term only | Multiple seasonal cycles | Flexible | Multi-horizon |
| Volatility seasonality | Separate model needed | No | Partial | Yes | Yes |
| Stock market best use | Calendar anomalies | Short-term returns | Trend + holidays | Complex patterns | Multi-horizon |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete SARIMA Implementation for Seasonal Stock Returns
# Capturing Quarterly Earnings Cycle and Monthly Effects
# Using only NumPy
# ============================================================

np.random.seed(2024)

# --- Generate 8 Years of Monthly Stock Returns with Seasonality ---
n_years = 8
n_months = n_years * 12
month_indices = np.tile(np.arange(12), n_years)  # 0-11 repeated

# Monthly seasonal return effects (documented anomalies)
monthly_effects = np.array([
    0.015,   # Jan: January Effect (strong)
    0.003,   # Feb: post-January normalization
    0.001,   # Mar: quarter-end rebalancing
    0.006,   # Apr: earnings optimism
    -0.003,  # May: "Sell in May"
    -0.002,  # Jun: summer lull begins
    0.004,   # Jul: mid-year earnings
    -0.004,  # Aug: low liquidity
    -0.008,  # Sep: historically worst month
    0.005,   # Oct: bargain hunting after Sep dip
    0.007,   # Nov: pre-holiday rally
    0.010,   # Dec: Santa Claus rally
])

# Monthly seasonal volatility multipliers
monthly_vol = np.array([
    1.30, 1.05, 1.10, 1.20, 0.95, 0.80,
    1.15, 0.75, 1.15, 1.40, 1.00, 0.85
])

# Generate returns
base_return = 0.006
base_vol = 0.042
returns = np.zeros(n_months)
for t in range(n_months):
    m = month_indices[t]
    mu = base_return + monthly_effects[m]
    sigma = base_vol * monthly_vol[m]
    # Add slight AR(1) component
    if t > 0:
        returns[t] = 0.08 * returns[t-1] + np.random.normal(mu, sigma)
    else:
        returns[t] = np.random.normal(mu, sigma)

# Also add year-over-year correlation
for t in range(12, n_months):
    returns[t] += 0.1 * returns[t-12]

# Convert to prices
prices = 100.0 * np.cumprod(1 + returns)

# Train/test split: last 12 months for testing
train_n = n_months - 12
train_returns = returns[:train_n]
test_returns = returns[train_n:]

print("=" * 65)
print("SARIMA Stock Return Seasonal Analysis and Forecasting")
print("=" * 65)
print(f"Total months: {n_months} ({n_years} years)")
print(f"Training:     {train_n} months ({train_n//12} years)")
print(f"Testing:      12 months")
print(f"Price range:  ${prices.min():.2f} - ${prices.max():.2f}")
print(f"Final price:  ${prices[-1]:.2f}")

# --- Step 1: Seasonal Decomposition ---
print(f"\n--- Seasonal Decomposition ---")

# Estimate seasonal component (average of each month)
seasonal_component = np.zeros(12)
for m in range(12):
    mask = month_indices[:train_n] == m
    seasonal_component[m] = np.mean(train_returns[mask])

# Deseasonalized returns
deseasonalized = np.zeros(train_n)
for t in range(train_n):
    deseasonalized[t] = train_returns[t] - seasonal_component[month_indices[t]]

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

print(f"\nEstimated Monthly Seasonal Effects:")
print(f"{'Month':<6} {'Effect':>8} {'Avg Vol':>10} {'Count':>6}")
print("-" * 32)
for m in range(12):
    mask = month_indices[:train_n] == m
    vol = np.std(train_returns[mask])
    count = np.sum(mask)
    print(f"{month_names[m]:<6} {seasonal_component[m]*100:>+7.3f}% {vol*100:>9.3f}% {count:>5d}")

# --- Step 2: Test for Seasonality ---
print(f"\n--- Seasonality Test ---")

# Kruskal-Wallis style test (simplified F-test)
grand_mean = np.mean(train_returns)
ss_between = 0
ss_within = 0
for m in range(12):
    mask = month_indices[:train_n] == m
    group = train_returns[mask]
    n_group = len(group)
    ss_between += n_group * (np.mean(group) - grand_mean) ** 2
    ss_within += np.sum((group - np.mean(group)) ** 2)

k = 12  # number of groups
n_total = train_n
f_stat = (ss_between / (k - 1)) / (ss_within / (n_total - k))
print(f"F-statistic for monthly seasonality: {f_stat:.4f}")
print(f"(Higher values indicate stronger seasonal patterns)")

# --- Step 3: Seasonal Differencing ---
print(f"\n--- SARIMA Model Fitting ---")

# Seasonal difference (D=1, m=12)
sd = train_returns[12:] - train_returns[:-12]
print(f"After seasonal differencing: {len(sd)} observations")
print(f"Mean: {sd.mean():.6f}, Std: {sd.std():.6f}")

# Compute ACF of seasonally differenced series
def compute_acf(series, max_lag):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    acf_vals = np.zeros(max_lag + 1)
    for lag in range(min(max_lag + 1, n)):
        cov = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
        acf_vals[lag] = cov / var
    return acf_vals

acf_sd = compute_acf(sd, 24)
print(f"\nACF of seasonally differenced series:")
print(f"  Lag  1: {acf_sd[1]:+.4f}")
print(f"  Lag  2: {acf_sd[2]:+.4f}")
print(f"  Lag 12: {acf_sd[12]:+.4f}")
print(f"  Lag 13: {acf_sd[13]:+.4f}")

# --- Step 4: Fit SARIMA(1,0,0)(1,1,0)[12] ---
# After seasonal differencing, fit AR(1) + Seasonal AR(1)

# Build regression matrix
# sd_t = c + phi_1 * sd_{t-1} + Phi_1 * sd_{t-12} + phi_1*Phi_1 * sd_{t-13} + e_t
min_lag = 13  # max(1, 12, 13)
y = sd[min_lag:]
n_y = len(y)

X = np.ones((n_y, 4))  # intercept, lag-1, lag-12, lag-13
X[:, 1] = sd[min_lag - 1:-1]     # lag 1
X[:, 2] = sd[min_lag - 12:len(sd) - 12]  # lag 12
X[:, 3] = sd[min_lag - 13:len(sd) - 13]  # lag 13

# OLS fit
beta = np.linalg.lstsq(X, y, rcond=None)[0]
residuals = y - X @ beta
sigma_resid = np.std(residuals)

c_hat = beta[0]
phi_1_hat = beta[1]
Phi_1_hat = beta[2]
interaction = beta[3]

print(f"\nSARIMA(1,0,0)(1,1,0)[12] Coefficients:")
print(f"  Intercept:            {c_hat:.6f}")
print(f"  phi_1 (AR1):          {phi_1_hat:.6f}")
print(f"  Phi_1 (SAR1):         {Phi_1_hat:.6f}")
print(f"  phi_1*Phi_1 (cross):  {interaction:.6f}")
print(f"  Residual sigma:       {sigma_resid:.6f}")

# Model diagnostics
n_params = 5  # 4 coefficients + sigma
log_l = -n_y/2 * np.log(2*np.pi) - n_y/2 * np.log(sigma_resid**2) - n_y/2
aic = 2 * n_params - 2 * log_l
bic = n_params * np.log(n_y) - 2 * log_l
print(f"  AIC: {aic:.2f}")
print(f"  BIC: {bic:.2f}")

# Residual diagnostics
acf_resid = compute_acf(residuals, 12)
print(f"\nResidual ACF (should be near zero):")
for lag in [1, 2, 3, 6, 12]:
    print(f"  Lag {lag:2d}: {acf_resid[lag]:+.4f}")

# --- Step 5: 12-Month Forecast ---
print(f"\n--- 12-Month Ahead Forecast ---")

# Extend the seasonally differenced series for forecasting
all_sd = np.concatenate([sd, np.zeros(12)])
all_returns = np.concatenate([train_returns, np.zeros(12)])

forecasts = np.zeros(12)
for h in range(12):
    idx = len(sd) + h

    # Compute seasonal differenced forecast
    lag1 = all_sd[idx - 1]
    lag12 = all_sd[idx - 12] if idx >= 12 else 0
    lag13 = all_sd[idx - 13] if idx >= 13 else 0

    sd_forecast = c_hat + phi_1_hat * lag1 + Phi_1_hat * lag12 + interaction * lag13
    all_sd[idx] = sd_forecast

    # Convert back: return_t = sd_t + return_{t-12}
    forecasts[h] = sd_forecast + all_returns[train_n - 12 + h]
    all_returns[train_n + h] = test_returns[h]  # Use actual for next step

# Confidence intervals
z_95 = 1.96
ci_width = np.array([z_95 * sigma_resid * np.sqrt(h + 1) for h in range(12)])

# Results
print(f"\n{'Month':<8} {'Forecast':>10} {'Actual':>10} {'Error':>10} {'95% CI':>20}")
print("-" * 62)
start_m = month_indices[train_n]
for h in range(12):
    m_idx = (start_m + h) % 12
    err = forecasts[h] - test_returns[h]
    ci_lo = (forecasts[h] - ci_width[h]) * 100
    ci_hi = (forecasts[h] + ci_width[h]) * 100
    print(f"{month_names[m_idx]:<8} {forecasts[h]*100:>+9.3f}% {test_returns[h]*100:>+9.3f}% "
          f"{err*100:>+9.3f}% [{ci_lo:>+7.2f}, {ci_hi:>+7.2f}]")

# --- Step 6: Comprehensive Evaluation ---
print(f"\n--- Forecast Evaluation ---")

rmse = np.sqrt(np.mean((forecasts - test_returns)**2))
mae = np.mean(np.abs(forecasts - test_returns))
dir_acc = np.mean(np.sign(forecasts) == np.sign(test_returns)) * 100

# Seasonal naive benchmark: just use same month from last year
naive_forecasts = returns[train_n - 12:train_n]
naive_rmse = np.sqrt(np.mean((naive_forecasts - test_returns)**2))
naive_mae = np.mean(np.abs(naive_forecasts - test_returns))
naive_dir = np.mean(np.sign(naive_forecasts) == np.sign(test_returns)) * 100

print(f"{'Metric':<25} {'SARIMA':>10} {'Seasonal Naive':>15}")
print("-" * 52)
print(f"{'RMSE':<25} {rmse*100:>9.4f}% {naive_rmse*100:>14.4f}%")
print(f"{'MAE':<25} {mae*100:>9.4f}% {naive_mae*100:>14.4f}%")
print(f"{'Direction Accuracy':<25} {dir_acc:>9.1f}% {naive_dir:>14.1f}%")

# Improvement over naive
if naive_rmse > 0:
    improvement = (1 - rmse / naive_rmse) * 100
    print(f"\nRMSE improvement over seasonal naive: {improvement:+.1f}%")

# --- Step 7: Seasonal Pattern Stability Analysis ---
print(f"\n--- Seasonal Pattern Stability ---")
print(f"(Comparing first half vs second half of training data)")

half = train_n // 2
print(f"\n{'Month':<6} {'First Half':>12} {'Second Half':>12} {'Stable?':>10}")
print("-" * 44)
for m in range(12):
    mask1 = month_indices[:half] == m
    mask2 = (month_indices[half:train_n] == m)
    avg1 = np.mean(train_returns[:half][mask1])
    avg2 = np.mean(train_returns[half:train_n][mask2])
    diff = abs(avg1 - avg2)
    stable = "Yes" if diff < 0.01 else "No"
    print(f"{month_names[m]:<6} {avg1*100:>+11.3f}% {avg2*100:>+11.3f}% {stable:>10}")
```

## Key Takeaways

1. SARIMA extends ARIMA with explicit seasonal components, making it the natural choice for modeling calendar-based patterns in stock markets such as the January Effect, quarterly earnings cycles, and seasonal volatility changes.

2. The model is specified as ARIMA(p,d,q)(P,D,Q)[m] where the seasonal period m must be chosen based on the frequency of the data and the target seasonal pattern (m=12 for monthly, m=4 for quarterly).

3. Always test for the statistical significance of seasonal patterns before fitting SARIMA. Many perceived seasonal effects in stock markets are weak or have diminished over time as they became widely known and traded upon.

4. Seasonal differencing (D=1) removes the seasonal pattern from the data, analogous to how first differencing removes a trend. This is essential when the seasonal pattern has a stochastic rather than deterministic nature.

5. For stock markets, SARIMA is most useful at monthly or quarterly frequencies where seasonal patterns are strongest. At daily frequency, the large seasonal period (m=252) makes estimation impractical.

6. Compare SARIMA forecasts against a seasonal naive benchmark (same month last year) to verify the model adds value. Beating the naive seasonal benchmark is a minimum requirement before deployment.

7. Monitor seasonal pattern stability over time. A weakening January Effect or diminishing September dip may indicate that the anomaly is being arbitraged away, requiring model recalibration or alternative approaches.

8. SARIMA works best as part of a broader modeling framework: use it to capture predictable seasonal components, then layer additional models for non-seasonal signals, fundamental factors, and volatility dynamics.
