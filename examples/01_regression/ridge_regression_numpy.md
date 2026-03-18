# Ridge Regression - Complete Guide with Stock Market Applications

## Overview

Ridge Regression, also known as Tikhonov regularization or L2-regularized linear regression, is an extension of ordinary linear regression that adds a penalty term proportional to the squared magnitude of the weight coefficients. This penalty discourages large weights, effectively shrinking them toward zero without setting them exactly to zero. The result is a model that is more robust to multicollinearity and less prone to overfitting.

In stock market analysis, Ridge Regression is particularly valuable because financial features are notoriously correlated. Technical indicators like SMA_5, SMA_10, SMA_20, EMA_12, and EMA_26 all track the same underlying price trend and are highly collinear. Ordinary least squares becomes unstable in the presence of such multicollinearity, producing coefficients with large variance that flip signs between training runs. Ridge Regression stabilizes these estimates by trading a small amount of bias for a significant reduction in variance.

The algorithm introduces a single hyperparameter, alpha (also called lambda), that controls the strength of the regularization penalty. When alpha is zero, Ridge Regression reduces to ordinary linear regression. As alpha increases, coefficients are shrunk more aggressively toward zero, reducing model complexity. Finding the optimal alpha through cross-validation is a key step in applying Ridge Regression effectively to financial data.

## How It Works - The Math Behind It

### The Ridge Objective Function

Ridge Regression modifies the ordinary least squares cost function by adding an L2 penalty:

```
J(w) = (1 / 2m) * sum_{i=1}^{m} (y_hat_i - y_i)^2 + alpha * sum_{j=1}^{n} w_j^2
```

In matrix notation:

```
J(w) = (1 / 2m) * ||X @ w - y||^2 + alpha * ||w||^2
```

The first term measures prediction accuracy (data fit), and the second term penalizes large weights (model complexity). The hyperparameter `alpha` controls the trade-off between the two.

### Closed-Form Solution

Taking the gradient of J(w) and setting it to zero yields the Ridge solution:

```
w* = (X^T X + alpha * I)^(-1) X^T y
```

where `I` is the identity matrix (with the bias term excluded from regularization). The key insight is that adding `alpha * I` to `X^T X` ensures the matrix is always invertible, even when features are perfectly collinear. This is what makes Ridge Regression numerically stable.

### Gradient Descent Formulation

For iterative optimization:

```
gradient = (1/m) * X^T @ (X @ w - y) + 2 * alpha * w
w = w - learning_rate * gradient
```

The gradient has an extra term `2 * alpha * w` compared to ordinary linear regression. This term continuously pushes weights toward zero, counteracting the data-driven updates that might inflate weights.

### Why It Works: Bias-Variance Trade-off

- **Without regularization**: The model can assign arbitrarily large (positive or negative) weights to correlated features, resulting in high variance (the weights change dramatically with small data changes)
- **With Ridge penalty**: Large weights are penalized, forcing the model to distribute weight more evenly among correlated features, reducing variance at the cost of a small increase in bias
- **Geometric interpretation**: Ridge Regression constrains the weight vector to lie within an L2 ball of radius proportional to 1/alpha. The solution is the point on this ball closest to the unconstrained OLS solution.

### Effect on Correlated Features

When two features are highly correlated (e.g., SMA_5 and SMA_10):
- OLS might assign weights like +50 and -48 (large, opposing, unstable)
- Ridge would assign weights like +1.2 and +1.0 (small, similar, stable)
- Both give similar predictions, but Ridge coefficients are more meaningful

## Stock Market Use Case: Predicting Stock Returns with Many Correlated Features (Multicollinearity in Market Data)

### The Problem

A portfolio manager wants to predict weekly stock returns using a comprehensive set of 25+ technical and fundamental indicators. Many of these indicators are derived from the same underlying price and volume data, creating severe multicollinearity. For instance, SMA_5, SMA_10, SMA_20, SMA_50, EMA_12, EMA_26, and VWAP are all weighted averages of recent prices. Using ordinary linear regression on this feature set produces wildly unstable coefficient estimates that change dramatically when a few data points are added or removed.

The manager needs a model that: (1) uses all available indicators without manually removing redundant ones, (2) provides stable, reliable coefficients that do not fluctuate, and (3) generalizes well to unseen market conditions. Ridge Regression is the ideal solution because it handles multicollinearity by shrinking correlated coefficients proportionally.

### Stock Market Features (Input Data)

| Feature Group | Features | Correlation Issue |
|---------------|----------|-------------------|
| **Price Averages** | SMA_5, SMA_10, SMA_20, SMA_50, EMA_12, EMA_26 | All track price trend, r > 0.95 |
| **Momentum** | RSI_14, MACD, MACD_Signal, Stochastic_K, Stochastic_D | Measure similar momentum, r > 0.7 |
| **Volatility** | ATR_14, Bollinger_Width, Volatility_10, Volatility_20 | All measure price dispersion, r > 0.8 |
| **Volume** | Volume, OBV, Volume_SMA_20, Relative_Volume | Volume-derived metrics, r > 0.6 |
| **Price Structure** | High-Low Range, Open-Close Range, Body/Shadow Ratio | Intraday price structure, r > 0.5 |
| **Fundamental** | P/E Ratio, Market Cap, Dividend Yield | Lower correlation with technicals |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_samples = 1000

# Generate correlated stock features
# Start with base price series
base_price = 200.0
log_returns = np.random.normal(0.0002, 0.015, n_samples)
prices = base_price * np.exp(np.cumsum(log_returns))

# Moving averages (highly correlated group)
def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan)
    for i in range(window-1, len(arr)):
        result[i] = np.mean(arr[i-window+1:i+1])
    return result

sma_5 = rolling_mean(prices, 5)
sma_10 = rolling_mean(prices, 10)
sma_20 = rolling_mean(prices, 20)
sma_50 = rolling_mean(prices, 50)

# EMA (correlated with SMA)
def ema(data, span):
    alpha = 2 / (span + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

ema_12 = ema(prices, 12)
ema_26 = ema(prices, 26)

# MACD (derived from EMA)
macd_line = ema_12 - ema_26
macd_signal = ema(macd_line, 9)
macd_histogram = macd_line - macd_signal

# Volatility indicators (correlated group)
daily_returns = np.diff(prices, prepend=prices[0]) / prices
vol_10 = np.array([np.std(daily_returns[max(0,i-9):i+1]) for i in range(n_samples)])
vol_20 = np.array([np.std(daily_returns[max(0,i-19):i+1]) for i in range(n_samples)])

# Bollinger Bands
bb_upper = sma_20 + 2 * vol_20 * prices
bb_lower = sma_20 - 2 * vol_20 * prices
bb_width = (bb_upper - bb_lower) / sma_20

# Volume features (correlated group)
volume = np.random.lognormal(17, 0.5, n_samples)
volume_sma20 = rolling_mean(volume, 20)
relative_vol = volume / np.where(volume_sma20 > 0, volume_sma20, 1)

# RSI
def compute_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    rsi = np.full(len(prices), 50.0)
    for i in range(period, len(deltas)):
        avg_g = np.mean(gains[i-period:i])
        avg_l = np.mean(losses[i-period:i])
        rs = avg_g / (avg_l + 1e-10)
        rsi[i+1] = 100 - 100 / (1 + rs)
    return rsi

rsi_14 = compute_rsi(prices)

# Build feature matrix (start from day 50 for valid indicators)
start = 55

feature_names = [
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'MACD_Histogram',
    'RSI_14',
    'Volatility_10', 'Volatility_20', 'BB_Width',
    'Volume_M', 'Relative_Volume',
    'Daily_Return', 'Price_SMA20_Ratio',
    'High_Low_Range'
]

high_prices = prices * (1 + np.abs(np.random.normal(0, 0.008, n_samples)))
low_prices = prices * (1 - np.abs(np.random.normal(0, 0.008, n_samples)))
hl_range = (high_prices - low_prices) / prices

X = np.column_stack([
    sma_5[start:-1], sma_10[start:-1], sma_20[start:-1], sma_50[start:-1],
    ema_12[start:-1], ema_26[start:-1],
    macd_line[start:-1], macd_signal[start:-1], macd_histogram[start:-1],
    rsi_14[start:-1],
    vol_10[start:-1] * 100, vol_20[start:-1] * 100, bb_width[start:-1],
    volume[start:-1] / 1e6, relative_vol[start:-1],
    daily_returns[start:-1] * 100,
    prices[start:-1] / sma_20[start:-1],
    hl_range[start:-1] * 100
])

# Target: Weekly return (5-day forward return)
forward_returns = (prices[start+5:] - prices[start:-5]) / prices[start:-5]
min_len = min(len(X), len(forward_returns))
X = X[:min_len]
y = forward_returns[:min_len] * 100  # Percentage returns

# Remove NaN rows
valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X, y = X[valid], y[valid]

print(f"Feature matrix: {X.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Target: Weekly forward return (%)")
```

### The Model in Action

```python
class RidgeRegressionNumpy:
    """Ridge Regression from scratch with L2 regularization."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        # Add bias column (not regularized)
        X_b = np.column_stack([np.ones(m), X])

        # Ridge solution: (X^T X + alpha * I')^(-1) X^T y
        # I' has 0 in top-left (don't regularize bias)
        identity = np.eye(n + 1)
        identity[0, 0] = 0  # Don't regularize bias

        self.theta = np.linalg.solve(
            X_b.T @ X_b + self.alpha * identity,
            X_b.T @ y
        )
        self.bias = self.theta[0]
        self.weights = self.theta[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

# Demonstration of multicollinearity effect
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

mu, sigma = X_train.mean(0), X_train.std(0)
sigma[sigma == 0] = 1
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma

# Compare OLS vs Ridge
from numpy.linalg import cond
print(f"Condition number of X^T X: {cond(X_train_s.T @ X_train_s):.1f}")
print("(Values > 30 indicate multicollinearity)\n")

# OLS (alpha=0 equivalent)
ols = RidgeRegressionNumpy(alpha=0.001)
ols.fit(X_train_s, y_train)

# Ridge with optimal alpha
ridge = RidgeRegressionNumpy(alpha=10.0)
ridge.fit(X_train_s, y_train)

print("Weight comparison (OLS vs Ridge):")
print(f"{'Feature':25s} {'OLS':>10s} {'Ridge':>10s}")
print("-" * 47)
for i, name in enumerate(feature_names):
    print(f"{name:25s} {ols.weights[i]:10.4f} {ridge.weights[i]:10.4f}")
```

Sample Output:
```
Condition number of X^T X: 4521.7
(Values > 30 indicate multicollinearity)

Weight comparison (OLS vs Ridge):
Feature                          OLS      Ridge
-----------------------------------------------
SMA_5                        12.4531     0.8234
SMA_10                      -15.2187     0.5128
SMA_20                        8.7643     0.3891
SMA_50                       -4.1298     0.2104
EMA_12                       11.8921     0.7543
EMA_26                       -9.3214     0.4219
MACD                          0.4521     0.3215
MACD_Signal                   0.3187     0.2891
MACD_Histogram                0.1342     0.1198
RSI_14                       -0.2341    -0.1876
Volatility_10                -0.8912    -0.5432
Volatility_20                 0.7234    -0.4198
BB_Width                     -0.3421    -0.2156
Volume_M                      0.0892     0.0654
Relative_Volume               0.1234     0.0987
Daily_Return                  0.5643     0.4321
Price_SMA20_Ratio             0.3215     0.2876
High_Low_Range               -0.1987    -0.1543
```

## Advantages

1. **Handles multicollinearity gracefully.** When stock market features are highly correlated (as is almost always the case with technical indicators), OLS produces unstable, high-variance weights. Ridge Regression stabilizes these by shrinking correlated coefficients, ensuring that adding SMA_50 alongside SMA_20 does not destabilize the entire model. This is perhaps its single most important advantage for financial applications.

2. **Guaranteed unique solution.** By adding `alpha * I` to `X^T X`, Ridge ensures the matrix is always invertible, even when features outnumber samples or when perfect collinearity exists. In stock market analysis, where analysts may engineer more features than there are trading days in the training window, this guarantee is essential.

3. **Smooth bias-variance trade-off.** The alpha parameter provides continuous control over model complexity. Small alpha values yield models close to OLS (low bias, high variance), while large alpha values produce simpler models (higher bias, lower variance). This smooth control allows practitioners to find the sweet spot for their specific stock prediction task.

4. **Keeps all features in the model.** Unlike Lasso, Ridge never sets coefficients exactly to zero. All features contribute to predictions, which is desirable when you believe all technical indicators carry some signal, even if their individual contributions are small. In financial markets, where weak signals from many sources can combine into meaningful predictions, this property is valuable.

5. **Computationally efficient.** The closed-form solution only requires solving a linear system, which is numerically stable and fast. For a trading system that needs to retrain models across hundreds of stocks daily, Ridge Regression's speed is a practical advantage over iterative methods.

6. **Robust to small perturbations in data.** OLS weights can change dramatically when a few training samples are added or removed. Ridge weights are much more stable, meaning the model's predictions and trading signals are consistent from one retraining cycle to the next. This stability reduces unnecessary portfolio turnover driven by model instability rather than genuine market signals.

## Disadvantages

1. **Does not perform feature selection.** Ridge shrinks all coefficients but never eliminates them. In stock market settings where some indicators are pure noise, including irrelevant features adds a small amount of unnecessary complexity and can slightly degrade predictions. If true sparsity exists (only 5 out of 50 indicators matter), Lasso or ElasticNet would be more appropriate.

2. **Requires choosing the alpha hyperparameter.** The regularization strength must be selected via cross-validation, adding a model selection step. In stock markets where the optimal alpha may change as market conditions evolve, this parameter must be periodically re-tuned, adding operational complexity.

3. **Still assumes linearity.** Like ordinary linear regression, Ridge cannot capture non-linear relationships between features and target. Stock market relationships are often non-linear (e.g., RSI signals mean different things above 70 vs. below 30). Ridge Regression misses these patterns entirely.

4. **Biased estimator by design.** The regularization penalty introduces bias in exchange for reduced variance. In scenarios where the true relationship is well-captured by a linear model with large coefficients, Ridge will systematically underestimate the effect, potentially leading to muted trading signals that miss profitable opportunities.

5. **Coefficients are harder to interpret than OLS.** While Ridge coefficients are more stable, they are biased toward zero and depend on the choice of alpha. Comparing coefficient magnitudes across different alpha values or different model runs requires careful normalization and interpretation.

6. **Feature scaling is mandatory.** The L2 penalty penalizes all weights equally, so features on different scales will receive unequal effective regularization. Price-based features (hundreds of dollars) must be standardized alongside percentage-based features (0-100) to ensure fair penalization. Failing to scale features is a common and serious mistake.

## When to Use in Stock Market

- When your feature set contains **many correlated technical indicators** (multiple moving averages, overlapping momentum signals)
- When you want to use **all available features** without manual feature selection
- When you need **stable, reproducible coefficients** across retraining cycles
- When building **factor models** where you want controlled exposure to many risk factors
- When the number of features is **large relative to the sample size** (e.g., 50 indicators with 200 training days)
- When you observe that **OLS coefficients are unstable** or have implausibly large magnitudes
- As a **regularized baseline** that sits between OLS (no regularization) and Lasso (sparse regularization)

## When NOT to Use in Stock Market

- When you need **automatic feature selection** to identify the few indicators that truly matter (use Lasso)
- When you suspect that **most features are irrelevant noise** and want them zeroed out
- When relationships between indicators and returns are **strongly non-linear** (use tree-based models or neural networks)
- When **exact coefficient interpretation** is critical and bias from regularization is unacceptable
- When the feature set is already **small and uncorrelated** (OLS will suffice)
- For **ultra-high-frequency data** where the feature space changes rapidly and retraining alpha is impractical

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| `alpha` | L2 regularization strength | 0.01 - 1000 | Cross-validate; start with 1.0 - 100.0 for financial data |
| `fit_intercept` | Include bias term | True/False | True; returns have non-zero mean |
| `normalize` | Standardize features | True/False | True (mandatory for proper regularization) |
| `solver` | Solution method | 'cholesky' / 'svd' | 'svd' for near-singular systems; 'cholesky' for speed |
| `tol` | Convergence tolerance (iterative solvers) | 1e-6 - 1e-3 | 1e-4 for financial precision |

### Alpha Selection Strategy for Financial Data

```
alpha too small (0.001):  ~= OLS, unstable coefficients, overfitting
alpha just right (1-100): Stable coefficients, good generalization
alpha too large (10000):  All coefficients near zero, underfitting
```

Use time-series cross-validation with expanding or rolling windows to find optimal alpha. Typical financial datasets work well with alpha in the range 1 to 100 after feature standardization.

## Stock Market Performance Tips

1. **Compute the condition number first.** Before applying Ridge, check `np.linalg.cond(X.T @ X)`. If it is > 30, multicollinearity is present and Ridge is warranted. If it is > 1000, Ridge is strongly recommended.

2. **Use rolling-window cross-validation for alpha.** Financial time series are non-stationary, so the optimal alpha changes over time. Use a 5-fold time-series CV with expanding windows to select alpha, and re-select every month.

3. **Standardize features using training data only.** Compute mean and standard deviation on the training set and apply the same transformation to the test set. Using test set statistics introduces information leakage.

4. **Consider alpha schedules.** In volatile market periods, stronger regularization (higher alpha) prevents overfitting to noisy data. In calm periods, weaker regularization captures more signal. Adaptive alpha based on market volatility can improve performance.

5. **Monitor weight stability.** Track the L2 norm of the weight vector over time. If it spikes despite regularization, the feature set may have changed character (e.g., a regime shift), and the model needs attention.

6. **Combine Ridge with target engineering.** Instead of predicting raw returns, predict risk-adjusted returns (Sharpe ratio) or excess returns relative to a benchmark. This often produces more stable and actionable predictions.

7. **Group correlated features before analysis.** While Ridge handles correlation internally, understanding feature groups helps interpret results. If all moving average coefficients are similar, it confirms Ridge is properly distributing weight across the correlated group.

## Comparison with Other Algorithms

| Criteria | Ridge Regression | Linear Regression | Lasso Regression | ElasticNet | MLP Regressor |
|----------|-----------------|------------------|-----------------|------------|---------------|
| **Regularization** | L2 (squared) | None | L1 (absolute) | L1 + L2 | Weight decay (L2) |
| **Feature Selection** | No (shrinks all) | No | Yes (zeros out) | Yes (zeros out) | No |
| **Multicollinearity** | Excellent | Poor | Fair | Good | N/A |
| **Coefficient Stability** | High | Low (if collinear) | Moderate | Moderate | N/A |
| **Interpretability** | Good | Excellent | Good (sparse) | Good | Poor |
| **Solution Uniqueness** | Always unique | May not exist | May not be unique | May not be unique | Multiple local optima |
| **Typical Stock R2** | 0.65 - 0.90 | 0.60 - 0.85 | 0.60 - 0.88 | 0.65 - 0.90 | 0.70 - 0.95 |
| **Best For** | Correlated features | Small, clean data | Sparse feature sets | Mixed scenarios | Complex patterns |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# COMPLETE RIDGE REGRESSION EXAMPLE: Weekly Return Prediction
# with Multicollinear Features
# ============================================================

np.random.seed(42)

# --- Step 1: Generate Realistic Multicollinear Stock Data ---
n_days = 1200
base_price = 200.0

# Simulate price path with realistic properties
log_rets = np.random.normal(0.0003, 0.016, n_days)
log_prices = np.log(base_price) + np.cumsum(log_rets)
close = np.exp(log_prices)

# OHLV data
open_p = close * (1 + np.random.normal(0, 0.003, n_days))
high_p = np.maximum(open_p, close) * (1 + np.abs(np.random.normal(0, 0.007, n_days)))
low_p = np.minimum(open_p, close) * (1 - np.abs(np.random.normal(0, 0.007, n_days)))
volume = np.random.lognormal(17.2, 0.4, n_days)

# --- Step 2: Compute MANY Correlated Technical Indicators ---
def rolling_stat(arr, window, func='mean'):
    result = np.full_like(arr, np.nan)
    for i in range(window-1, len(arr)):
        chunk = arr[i-window+1:i+1]
        if func == 'mean':
            result[i] = np.mean(chunk)
        elif func == 'std':
            result[i] = np.std(chunk)
        elif func == 'max':
            result[i] = np.max(chunk)
        elif func == 'min':
            result[i] = np.min(chunk)
    return result

def compute_ema(data, span):
    alpha = 2 / (span + 1)
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

# Moving averages (GROUP 1 - highly correlated)
sma_5 = rolling_stat(close, 5)
sma_10 = rolling_stat(close, 10)
sma_20 = rolling_stat(close, 20)
sma_50 = rolling_stat(close, 50)
ema_12 = compute_ema(close, 12)
ema_26 = compute_ema(close, 26)

# MACD family (GROUP 2 - derived from EMAs)
macd = ema_12 - ema_26
macd_signal = compute_ema(macd, 9)
macd_hist = macd - macd_signal

# Momentum (GROUP 3)
daily_ret = np.diff(close, prepend=close[0]) / close
ret_5d = (close[5:] - close[:-5]) / close[:-5]
ret_5d = np.concatenate([np.zeros(5), ret_5d])

def compute_rsi(p, period=14):
    d = np.diff(p)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    rsi_out = np.full(len(p), 50.0)
    for i in range(period, len(d)):
        ag = np.mean(g[i-period:i])
        al = np.mean(l[i-period:i])
        rs = ag / (al + 1e-10)
        rsi_out[i+1] = 100 - 100/(1+rs)
    return rsi_out

rsi_14 = compute_rsi(close, 14)
rsi_7 = compute_rsi(close, 7)

# Volatility (GROUP 4 - correlated)
vol_5 = np.array([np.std(daily_ret[max(0,i-4):i+1]) for i in range(n_days)])
vol_10 = np.array([np.std(daily_ret[max(0,i-9):i+1]) for i in range(n_days)])
vol_20 = np.array([np.std(daily_ret[max(0,i-19):i+1]) for i in range(n_days)])
atr_14 = rolling_stat(high_p - low_p, 14)

# Bollinger Bands
bb_mid = sma_20
bb_std = rolling_stat(close, 20, func='std')
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

# Volume indicators (GROUP 5)
vol_sma10 = rolling_stat(volume, 10)
vol_sma20 = rolling_stat(volume, 20)
rel_vol = volume / (vol_sma20 + 1e-10)

# Price structure
hl_range = (high_p - low_p) / close * 100
oc_range = np.abs(open_p - close) / close * 100
price_sma20 = close / (sma_20 + 1e-10)

# --- Step 3: Build Feature Matrix (25 features, many correlated) ---
start = 55
end = n_days - 5  # Need 5-day forward return

feature_names = [
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI_14', 'RSI_7',
    'Vol_5%', 'Vol_10%', 'Vol_20%', 'ATR_14',
    'BB_Width', 'BB_Position',
    'Volume_M', 'Rel_Volume',
    'Daily_Ret%', 'Ret_5d%',
    'HL_Range%', 'OC_Range%',
    'Price/SMA20', 'Close'
]

X = np.column_stack([
    sma_5[start:end], sma_10[start:end], sma_20[start:end], sma_50[start:end],
    ema_12[start:end], ema_26[start:end],
    macd[start:end], macd_signal[start:end], macd_hist[start:end],
    rsi_14[start:end], rsi_7[start:end],
    vol_5[start:end]*100, vol_10[start:end]*100, vol_20[start:end]*100, atr_14[start:end],
    bb_width[start:end], bb_position[start:end],
    volume[start:end]/1e6, rel_vol[start:end],
    daily_ret[start:end]*100, ret_5d[start:end]*100,
    hl_range[start:end], oc_range[start:end],
    price_sma20[start:end], close[start:end]
])

# Target: 5-day forward return
y = ((close[start+5:end+5] - close[start:end]) / close[start:end]) * 100

valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X, y = X[valid], y[valid]

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Target: 5-day forward return (%)")

# --- Step 4: Time-Series Train/Test Split ---
split = int(len(X) * 0.75)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Standardize
mu = X_train.mean(axis=0)
sig = X_train.std(axis=0)
sig[sig == 0] = 1
X_train_s = (X_train - mu) / sig
X_test_s = (X_test - mu) / sig

# --- Step 5: Check Multicollinearity ---
corr_matrix = np.corrcoef(X_train_s.T)
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if abs(corr_matrix[i,j]) > 0.8:
            high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i,j]))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"\nHighly correlated feature pairs (|r| > 0.8): {len(high_corr_pairs)}")
for f1, f2, r in high_corr_pairs[:10]:
    print(f"  {f1:15s} <-> {f2:15s}  r = {r:.4f}")

cond_num = np.linalg.cond(X_train_s.T @ X_train_s)
print(f"\nCondition number: {cond_num:.1f}")
print(f"Multicollinearity severity: {'SEVERE' if cond_num > 1000 else 'MODERATE' if cond_num > 30 else 'LOW'}")

# --- Step 6: Ridge Regression Implementation ---
class RidgeRegressionComplete:
    """Complete Ridge Regression with cross-validation support."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.column_stack([np.ones(m), X])
        I = np.eye(n + 1)
        I[0, 0] = 0  # Don't regularize intercept

        self.theta = np.linalg.solve(
            X_b.T @ X_b + self.alpha * I,
            X_b.T @ y
        )
        self.bias = self.theta[0]
        self.weights = self.theta[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def weight_norm(self):
        return np.sqrt(np.sum(self.weights ** 2))


def time_series_cv(X, y, alphas, n_folds=5):
    """Time-series cross-validation for alpha selection."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    results = {}

    for alpha in alphas:
        fold_scores = []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 2)
            test_start = train_end
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start:
                continue

            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[test_start:test_end], y[test_start:test_end]

            model = RidgeRegressionComplete(alpha=alpha)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            mse = np.mean((y_te - y_pred) ** 2)
            fold_scores.append(mse)

        results[alpha] = np.mean(fold_scores)

    return results

# --- Step 7: Cross-Validate Alpha ---
alphas = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
cv_results = time_series_cv(X_train_s, y_train, alphas)

print("\n" + "="*50)
print("  Cross-Validation Results")
print("="*50)
for alpha, mse in sorted(cv_results.items()):
    marker = " <-- best" if mse == min(cv_results.values()) else ""
    print(f"  alpha = {alpha:8.3f}  |  CV MSE = {mse:.6f}{marker}")

best_alpha = min(cv_results, key=cv_results.get)
print(f"\nBest alpha: {best_alpha}")

# --- Step 8: Train Final Models ---
# OLS baseline
ols_model = RidgeRegressionComplete(alpha=0.001)
ols_model.fit(X_train_s, y_train)

# Ridge with best alpha
ridge_model = RidgeRegressionComplete(alpha=best_alpha)
ridge_model.fit(X_train_s, y_train)

# --- Step 9: Evaluate and Compare ---
def full_evaluate(y_true, y_pred, name):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot

    # Direction accuracy
    direction = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  RMSE:               {rmse:.4f}%")
    print(f"  MAE:                {mae:.4f}%")
    print(f"  R2:                 {r2:.6f}")
    print(f"  Direction Accuracy: {direction:.1f}%")
    return r2

y_pred_ols = ols_model.predict(X_test_s)
y_pred_ridge = ridge_model.predict(X_test_s)

r2_ols = full_evaluate(y_test, y_pred_ols, f"OLS (alpha=0.001)")
r2_ridge = full_evaluate(y_test, y_pred_ridge, f"Ridge (alpha={best_alpha})")

print(f"\nR2 improvement: {(r2_ridge - r2_ols):.6f}")
print(f"Weight norm - OLS:   {ols_model.weight_norm():.4f}")
print(f"Weight norm - Ridge: {ridge_model.weight_norm():.4f}")
print(f"Weight norm reduction: {(1 - ridge_model.weight_norm()/ols_model.weight_norm())*100:.1f}%")

# --- Step 10: Coefficient Stability Analysis ---
print(f"\n{'='*50}")
print(f"  Coefficient Analysis")
print(f"{'='*50}")
print(f"{'Feature':20s} {'OLS':>10s} {'Ridge':>10s} {'|Change|':>10s}")
print("-" * 52)
for i in range(len(feature_names)):
    change = abs(ridge_model.weights[i] - ols_model.weights[i])
    print(f"{feature_names[i]:20s} {ols_model.weights[i]:10.4f} {ridge_model.weights[i]:10.4f} {change:10.4f}")

# --- Step 11: Regularization Path ---
print(f"\n{'='*50}")
print(f"  Regularization Path (Weight Norms)")
print(f"{'='*50}")
path_alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
for a in path_alphas:
    m = RidgeRegressionComplete(alpha=a)
    m.fit(X_train_s, y_train)
    y_p = m.predict(X_test_s)
    test_mse = np.mean((y_test - y_p)**2)
    print(f"  alpha={a:>8.3f}  |  ||w||={m.weight_norm():8.4f}  |  Test MSE={test_mse:.6f}")

# --- Step 12: Simulated Trading Performance ---
print(f"\n{'='*50}")
print(f"  Simulated Trading Strategy")
print(f"{'='*50}")

# Long when predicted positive, short when predicted negative
actual_rets = y_test / 100
pred_direction = np.sign(y_pred_ridge)

strategy_rets = pred_direction * actual_rets
cum_strategy = np.cumprod(1 + strategy_rets / 5)  # Divide by 5 for daily
cum_buyhold = np.cumprod(1 + actual_rets / 5)

total_strat = (cum_strategy[-1] - 1) * 100
total_bh = (cum_buyhold[-1] - 1) * 100

sharpe_strat = np.mean(strategy_rets) / (np.std(strategy_rets) + 1e-10) * np.sqrt(52)
sharpe_bh = np.mean(actual_rets) / (np.std(actual_rets) + 1e-10) * np.sqrt(52)

print(f"  Ridge Strategy Return: {total_strat:.2f}%")
print(f"  Buy-and-Hold Return:   {total_bh:.2f}%")
print(f"  Ridge Strategy Sharpe: {sharpe_strat:.3f}")
print(f"  Buy-and-Hold Sharpe:   {sharpe_bh:.3f}")
```

## Key Takeaways

- Ridge Regression is the **go-to algorithm when multicollinearity is present** in your stock market features, which is almost always the case with technical indicators
- The **L2 penalty shrinks coefficients toward zero** without eliminating them, keeping all features in the model
- **Alpha selection via time-series cross-validation** is critical; the optimal value depends on the specific stock, feature set, and market regime
- The **condition number of X^T X** is a quick diagnostic; values above 30 indicate Ridge is warranted
- Ridge produces **more stable coefficients** than OLS, leading to more consistent trading signals and less unnecessary portfolio turnover
- **Always standardize features** before applying Ridge; the penalty treats all weights equally, so features must be on comparable scales
- Ridge Regression **guarantees a unique solution** even when features outnumber samples, making it safe for high-dimensional financial feature engineering
- Compare Ridge to OLS as a **diagnostic tool**: if Ridge significantly outperforms OLS, multicollinearity was distorting OLS predictions
- The **regularization path** (weights vs. alpha plot) reveals how features group together as regularization increases, providing insight into the correlation structure of your indicators
