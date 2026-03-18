# Linear Regression - Complete Guide with Stock Market Applications

## Overview

Linear Regression is one of the most fundamental and widely-used algorithms in machine learning and statistics. At its core, it models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data. The algorithm assumes that the target variable can be expressed as a weighted sum of input features plus a bias term.

In the context of stock market analysis, Linear Regression serves as a baseline predictive model for forecasting prices, returns, and other financial metrics. While financial markets are inherently noisy and non-linear, Linear Regression provides interpretable coefficients that help traders and analysts understand the marginal impact of each technical indicator on price movements. Its simplicity makes it an excellent starting point before exploring more complex models.

The algorithm works by minimizing the sum of squared residuals (differences between predicted and actual values), a method known as Ordinary Least Squares (OLS). This optimization has a closed-form solution, making training extremely fast even on large datasets. However, the linearity assumption means it cannot capture complex non-linear relationships without feature engineering.

## How It Works - The Math Behind It

### The Linear Model

The hypothesis function for Linear Regression with `n` features is:

```
y_hat = w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n
```

In matrix notation, this is compactly written as:

```
y_hat = X @ w
```

where `X` is the design matrix (with a column of ones prepended for the bias), `w` is the weight vector (including the bias `w_0`), and `y_hat` is the vector of predictions.

### Cost Function (Mean Squared Error)

The objective is to minimize the Mean Squared Error (MSE):

```
J(w) = (1 / 2m) * sum_{i=1}^{m} (y_hat_i - y_i)^2
```

where `m` is the number of training samples. The factor of `1/2` is included for mathematical convenience when taking derivatives.

### Closed-Form Solution (Normal Equation)

The optimal weights can be found analytically:

```
w* = (X^T X)^(-1) X^T y
```

This gives the exact solution in one step without iteration. However, matrix inversion is O(n^3), so for very high-dimensional data, iterative methods like gradient descent are preferred.

### Gradient Descent Alternative

For large datasets, we iteratively update weights using gradients:

```
gradient = (1/m) * X^T @ (X @ w - y)
w = w - learning_rate * gradient
```

The gradient points in the direction of steepest ascent, so we subtract it to descend toward the minimum. The learning rate controls the step size.

### Step-by-Step Learning Process

1. **Initialize weights** to zeros or small random values
2. **Forward pass**: Compute predictions using the current weights
3. **Compute loss**: Calculate MSE between predictions and actual values
4. **Compute gradients**: Partial derivatives of loss with respect to each weight
5. **Update weights**: Adjust weights in the opposite direction of gradients
6. **Repeat** steps 2-5 until convergence (or use Normal Equation for one-step solution)

## Stock Market Use Case: Predicting Next-Day Stock Closing Price from Technical Indicators

### The Problem

A quantitative trader wants to predict tomorrow's closing price for a given stock using today's technical indicators. By accurately forecasting the closing price, the trader can decide whether to go long (buy) or short (sell) the stock. The model uses historical price data and technical analysis features computed from the last several trading days.

This is a regression problem because the target variable (next-day closing price) is continuous. The trader needs not just the direction but the magnitude of expected price change to size positions appropriately and set stop-loss orders.

### Stock Market Features (Input Data)

| Feature | Description | Typical Range | Relevance |
|---------|-------------|---------------|-----------|
| `open_price` | Opening price of the day | $10 - $500+ | Indicates overnight sentiment |
| `high_price` | Highest price during the day | $10 - $500+ | Shows intraday demand |
| `low_price` | Lowest price during the day | $10 - $500+ | Shows intraday supply pressure |
| `close_price` | Closing price of the day | $10 - $500+ | Most important price reference |
| `volume` | Number of shares traded | 1M - 100M+ | Indicates conviction behind moves |
| `sma_5` | 5-day Simple Moving Average | $10 - $500+ | Short-term trend indicator |
| `sma_20` | 20-day Simple Moving Average | $10 - $500+ | Medium-term trend indicator |
| `rsi_14` | 14-day Relative Strength Index | 0 - 100 | Overbought/oversold signal |
| `macd` | MACD line value | -10 to +10 | Momentum indicator |
| `bollinger_upper` | Upper Bollinger Band | $10 - $500+ | Upper volatility boundary |
| `bollinger_lower` | Lower Bollinger Band | $10 - $500+ | Lower volatility boundary |
| `daily_return` | Percentage daily return | -10% to +10% | Recent price momentum |
| `volatility_10` | 10-day rolling volatility | 0.5% - 5% | Recent risk measure |
| `vwap` | Volume-Weighted Average Price | $10 - $500+ | Fair value benchmark |

### Example Data Structure

```python
import numpy as np

# Simulate 500 trading days of stock market data
np.random.seed(42)
n_samples = 500

# Generate realistic stock price data starting at $150
base_price = 150.0
daily_changes = np.random.normal(0.0005, 0.02, n_samples)
prices = base_price * np.cumprod(1 + daily_changes)

# Feature engineering: Technical indicators
open_prices = prices * (1 + np.random.normal(0, 0.005, n_samples))
high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
close_prices = prices
volume = np.random.lognormal(17, 0.5, n_samples).astype(int)  # ~24M avg

# Moving averages
sma_5 = np.convolve(close_prices, np.ones(5)/5, mode='same')
sma_20 = np.convolve(close_prices, np.ones(20)/20, mode='same')

# RSI (simplified calculation)
deltas = np.diff(close_prices, prepend=close_prices[0])
gains = np.where(deltas > 0, deltas, 0)
losses = np.where(deltas < 0, -deltas, 0)
avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
rs = avg_gain / (avg_loss + 1e-10)
rsi_14 = 100 - (100 / (1 + rs))

# Daily returns and volatility
daily_returns = np.diff(close_prices, prepend=close_prices[0]) / close_prices
volatility_10 = np.array([np.std(daily_returns[max(0,i-10):i+1]) for i in range(n_samples)])

# Combine all features into a matrix
# Each row = one trading day, each column = one feature
X = np.column_stack([
    open_prices,
    high_prices,
    low_prices,
    close_prices,
    volume / 1e6,        # Scale volume to millions
    sma_5,
    sma_20,
    rsi_14,
    daily_returns * 100,  # Percentage returns
    volatility_10 * 100   # Percentage volatility
])

# Target: Next-day closing price (shift by 1)
y = np.roll(close_prices, -1)

# Remove last sample (no future target available)
X = X[:-1]
y = y[:-1]

feature_names = [
    'Open', 'High', 'Low', 'Close', 'Volume_M',
    'SMA_5', 'SMA_20', 'RSI_14', 'Daily_Return_%', 'Volatility_10_%'
]

print(f"Feature matrix shape: {X.shape}")   # (499, 10)
print(f"Target vector shape: {y.shape}")     # (499,)
print(f"Feature names: {feature_names}")
```

### The Model in Action

```python
class LinearRegressionNumpy:
    """Linear Regression from scratch using NumPy."""

    def __init__(self, method='normal_equation'):
        self.method = method
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        # Add bias column
        X_b = np.column_stack([np.ones(m), X])

        if self.method == 'normal_equation':
            # Closed-form solution: w = (X^T X)^(-1) X^T y
            self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = self.theta[0]
        self.weights = self.theta[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

# Train/test split (80/20, preserving time order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Feature scaling (fit on train, transform both)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# Train the model
model = LinearRegressionNumpy(method='normal_equation')
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("=== Stock Price Prediction Results ===")
print(f"RMSE:  ${rmse:.2f}")
print(f"MAE:   ${mae:.2f}")
print(f"R2:    {r2:.4f}")
print(f"\nSample predictions vs actuals:")
for i in range(5):
    print(f"  Predicted: ${y_pred[i]:.2f} | Actual: ${y_test[i]:.2f} | Error: ${abs(y_pred[i]-y_test[i]):.2f}")
```

Sample Output:
```
=== Stock Price Prediction Results ===
RMSE:  $2.34
MAE:   $1.87
R2:    0.9812

Sample predictions vs actuals:
  Predicted: $158.32 | Actual: $157.89 | Error: $0.43
  Predicted: $159.14 | Actual: $160.02 | Error: $0.88
  Predicted: $160.55 | Actual: $159.71 | Error: $0.84
  Predicted: $161.23 | Actual: $161.98 | Error: $0.75
  Predicted: $162.01 | Actual: $161.45 | Error: $0.56
```

## Advantages

1. **Extreme simplicity and interpretability.** Each weight coefficient directly tells you how much a one-unit change in the corresponding feature affects the predicted stock price. For example, if the weight for RSI is -0.15, a 1-point increase in RSI predicts a $0.15 decrease in the next day's price. This transparency is critical in financial settings where traders need to understand and justify their decisions.

2. **Very fast training with the Normal Equation.** The closed-form solution computes the optimal weights in a single matrix operation, making it orders of magnitude faster than iterative methods. For stock market applications where models need to be retrained frequently (e.g., daily with new data), this speed is a significant practical advantage.

3. **No hyperparameters to tune.** Unlike regularized models or neural networks, basic Linear Regression has no regularization strength, learning rate, or architecture choices. This eliminates the risk of overfitting due to poor hyperparameter selection, though the model itself may still underfit complex relationships.

4. **Strong mathematical guarantees under assumptions.** When the Gauss-Markov assumptions hold (linearity, independence, homoscedasticity, no perfect multicollinearity), OLS provides the Best Linear Unbiased Estimator (BLUE). While stock market data often violates these assumptions, understanding these guarantees helps practitioners know when to trust the model.

5. **Excellent baseline for comparison.** In quantitative finance, every complex model should be compared against a Linear Regression baseline. If a deep learning model only marginally outperforms Linear Regression on stock prediction, the added complexity may not be justified given the extra computational cost and reduced interpretability.

6. **Feature importance is built in.** The magnitude and sign of each coefficient reveal feature importance after standardization. Traders can quickly identify which technical indicators have the strongest predictive power for a given stock, enabling more targeted analysis and strategy development.

## Disadvantages

1. **Cannot capture non-linear relationships.** Stock market dynamics are inherently non-linear. Relationships between technical indicators and future prices often involve thresholds, interactions, and regime changes that a linear model completely misses. For example, RSI might be irrelevant between 40-60 but highly predictive below 30 or above 70.

2. **Sensitive to outliers.** Extreme market events (flash crashes, earnings surprises, black swan events) create outliers that disproportionately influence the squared error loss function. A single day with a 10% price swing can significantly distort the learned weights, making predictions unreliable during volatile periods.

3. **Assumes feature independence (no multicollinearity).** Technical indicators are often highly correlated (e.g., SMA_5 and SMA_20 both track price trends). Multicollinearity inflates the variance of coefficient estimates, making individual weights unreliable and unstable. The model may still predict well overall, but interpreting individual feature effects becomes misleading.

4. **No built-in regularization.** Without regularization, the model is prone to overfitting when the number of features approaches or exceeds the number of samples. In stock market analysis, where analysts may compute dozens of technical indicators from limited historical data, this can be a serious problem.

5. **Assumes constant relationships over time.** Markets are non-stationary. The relationship between technical indicators and future prices changes over different market regimes (bull market, bear market, high volatility, low volatility). A Linear Regression model trained on bull market data will perform poorly when the market regime shifts.

6. **Poor handling of categorical and seasonal effects.** Market behavior changes with day-of-week effects, earnings seasons, and macroeconomic cycles. While these can be encoded as dummy variables, Linear Regression treats them as having a constant additive effect, which may not reflect reality.

## When to Use in Stock Market

- As a **baseline model** before trying more complex algorithms for any prediction task
- When you need **interpretable coefficients** to understand which indicators drive predictions
- For **intraday or short-term predictions** where relationships are approximately linear
- When working with **limited training data** where complex models would overfit
- For **portfolio construction** where factor exposures need to be estimated (factor models)
- When **computational speed** is critical, such as real-time trading systems with frequent retraining
- For **screening and filtering** stocks based on predicted returns in a large universe

## When NOT to Use in Stock Market

- When predicting **highly non-linear phenomena** like option prices or volatility smiles
- During **regime changes** where historical relationships break down (e.g., 2008 financial crisis)
- When the feature space has **high multicollinearity** (use Ridge or Lasso instead)
- For **high-frequency trading** where microstructure effects dominate and relationships are complex
- When **outlier robustness** is critical (consider Huber regression or quantile regression)
- For **classification tasks** like buy/sell signals (use logistic regression or classifiers instead)
- When you have **many more features than samples** (underdetermined system, needs regularization)

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| `learning_rate` | Step size for gradient descent | 0.001 - 0.1 | Start with 0.01; not needed for Normal Equation |
| `n_iterations` | Number of gradient descent steps | 100 - 10000 | 1000-5000; not needed for Normal Equation |
| `fit_intercept` | Whether to include bias term | True/False | Always True for stock prices (non-zero baseline) |
| `normalize` | Standardize features before fitting | True/False | True; financial features have vastly different scales |
| `method` | Solution method | 'normal_eq' / 'gd' | Normal Equation for < 10K features, GD otherwise |

## Stock Market Performance Tips

1. **Always standardize features.** Price-based features (in dollars) and percentage-based features (returns, RSI) have vastly different scales. Without standardization, the model will be dominated by large-scale features like absolute prices while ignoring potentially informative scaled features.

2. **Use time-series cross-validation.** Never use random train/test splits with stock data. Always split chronologically and use walk-forward validation (train on past, test on future). This prevents look-ahead bias that would give unrealistically optimistic results.

3. **Engineer stationary features.** Raw stock prices are non-stationary (trending). Use returns, log-returns, or price ratios instead of raw prices. Moving average crossover signals (SMA_5 / SMA_20 ratio) are more stationary than the raw averages themselves.

4. **Check for multicollinearity.** Calculate the Variance Inflation Factor (VIF) for each feature. Drop or combine features with VIF > 10. High, Low, and Close prices will almost certainly be collinear, so consider using derived features like the High-Low range instead.

5. **Include lagged features.** Adding 1-day, 2-day, and 5-day lagged values of key indicators can capture short-term momentum effects that a single snapshot misses.

6. **Monitor residual autocorrelation.** If residuals are autocorrelated (check with Durbin-Watson test), the model is missing temporal patterns. Consider adding autoregressive features or switching to a time-series model.

7. **Retrain periodically.** Market relationships drift over time. Retrain the model weekly or monthly with a rolling window to keep it current. A 252-day (one trading year) rolling window is a common starting point.

## Comparison with Other Algorithms

| Criteria | Linear Regression | Ridge Regression | Lasso Regression | ElasticNet | MLP Regressor |
|----------|------------------|-----------------|-----------------|------------|---------------|
| **Non-linearity** | None | None | None | None | High |
| **Interpretability** | Excellent | Good | Good (sparse) | Good | Poor |
| **Multicollinearity Handling** | Poor | Excellent | Moderate | Good | N/A |
| **Feature Selection** | None | None (shrinks) | Built-in (zeros) | Built-in | None |
| **Training Speed** | Very Fast | Fast | Fast | Fast | Slow |
| **Overfitting Risk** | Moderate | Low | Low | Low | High |
| **Typical Stock R2** | 0.60 - 0.85 | 0.65 - 0.90 | 0.60 - 0.88 | 0.65 - 0.90 | 0.70 - 0.95 |
| **Best For** | Baseline | Correlated features | Sparse models | Mixed | Complex patterns |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# COMPLETE LINEAR REGRESSION EXAMPLE: Stock Price Prediction
# ============================================================

np.random.seed(42)

# --- Step 1: Generate Realistic Stock Market Data ---
n_days = 600
base_price = 150.0

# Simulate daily log returns with slight positive drift
log_returns = np.random.normal(0.0003, 0.018, n_days)
log_prices = np.log(base_price) + np.cumsum(log_returns)
close_prices = np.exp(log_prices)

# Derive OHLV data
open_prices = close_prices * (1 + np.random.normal(0, 0.003, n_days))
high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
volume = np.random.lognormal(17.0, 0.4, n_days).astype(int)

# --- Step 2: Compute Technical Indicators ---
def sma(data, window):
    result = np.full_like(data, np.nan)
    for i in range(window - 1, len(data)):
        result[i] = np.mean(data[i - window + 1:i + 1])
    return result

def compute_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    rsi = np.full(len(prices), 50.0)
    for i in range(period, len(deltas)):
        avg_g = np.mean(gains[i-period:i])
        avg_l = np.mean(losses[i-period:i])
        if avg_l == 0:
            rsi[i+1] = 100.0
        else:
            rs = avg_g / avg_l
            rsi[i+1] = 100 - (100 / (1 + rs))
    return rsi

sma_5 = sma(close_prices, 5)
sma_10 = sma(close_prices, 10)
sma_20 = sma(close_prices, 20)
rsi_14 = compute_rsi(close_prices, 14)
daily_ret = np.diff(close_prices, prepend=close_prices[0]) / close_prices

# Rolling volatility (10-day)
vol_10 = np.array([
    np.std(daily_ret[max(0, i-9):i+1]) if i >= 9 else np.std(daily_ret[:i+1])
    for i in range(n_days)
])

# Price relative to SMA (momentum signal)
price_sma20_ratio = close_prices / np.where(sma_20 > 0, sma_20, close_prices)

# High-Low range as percentage of close
hl_range_pct = (high_prices - low_prices) / close_prices * 100

# Volume relative to 20-day average
vol_sma20 = sma(volume.astype(float), 20)
relative_volume = volume / np.where(vol_sma20 > 0, vol_sma20, volume.astype(float))

# --- Step 3: Build Feature Matrix ---
# Start from day 20 to ensure all indicators are valid
start_idx = 25
end_idx = n_days - 1  # Reserve last day for target

X = np.column_stack([
    daily_ret[start_idx:end_idx] * 100,          # Daily return %
    rsi_14[start_idx:end_idx],                     # RSI
    (close_prices[start_idx:end_idx] - sma_5[start_idx:end_idx]),  # Price - SMA5
    (sma_5[start_idx:end_idx] - sma_20[start_idx:end_idx]),        # SMA5 - SMA20
    price_sma20_ratio[start_idx:end_idx],          # Price / SMA20
    vol_10[start_idx:end_idx] * 100,               # Volatility %
    hl_range_pct[start_idx:end_idx],               # HL Range %
    relative_volume[start_idx:end_idx],            # Relative Volume
    close_prices[start_idx:end_idx],               # Current Close
])

# Target: Next-day closing price
y = close_prices[start_idx + 1:end_idx + 1]

feature_names = [
    'Daily_Return_%', 'RSI_14', 'Price_minus_SMA5', 'SMA5_minus_SMA20',
    'Price_SMA20_Ratio', 'Volatility_10_%', 'HL_Range_%',
    'Relative_Volume', 'Current_Close'
]

# Remove any rows with NaN
valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X = X[valid_mask]
y = y[valid_mask]

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {feature_names}")

# --- Step 4: Train/Test Split (Time-Series) ---
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# --- Step 5: Feature Scaling ---
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1  # Prevent division by zero

X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma

# --- Step 6: Linear Regression Implementation ---
class LinearRegressionFromScratch:
    """
    Linear Regression using Normal Equation and Gradient Descent.
    """

    def __init__(self, method='normal_equation', lr=0.01, n_iters=1000):
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape

        if self.method == 'normal_equation':
            X_b = np.column_stack([np.ones(m), X])
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]

        elif self.method == 'gradient_descent':
            self.weights = np.zeros(n)
            self.bias = 0.0

            for iteration in range(self.n_iters):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y

                # Gradients
                dw = (1/m) * X.T @ error
                db = (1/m) * np.sum(error)

                # Update
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # Track loss
                loss = np.mean(error ** 2)
                self.loss_history.append(loss)

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def get_coefficients(self):
        return self.weights, self.bias

# --- Step 7: Train Both Methods ---
# Method 1: Normal Equation
model_ne = LinearRegressionFromScratch(method='normal_equation')
model_ne.fit(X_train_s, y_train)

# Method 2: Gradient Descent
model_gd = LinearRegressionFromScratch(method='gradient_descent', lr=0.01, n_iters=2000)
model_gd.fit(X_train_s, y_train)

# --- Step 8: Evaluate ---
def evaluate(y_true, y_pred, label):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  RMSE:          ${rmse:.4f}")
    print(f"  MAE:           ${mae:.4f}")
    print(f"  R2 Score:      {r2:.6f}")
    print(f"  MAPE:          {mape:.4f}%")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

y_pred_ne = model_ne.predict(X_test_s)
y_pred_gd = model_gd.predict(X_test_s)

results_ne = evaluate(y_test, y_pred_ne, "Normal Equation Results")
results_gd = evaluate(y_test, y_pred_gd, "Gradient Descent Results")

# --- Step 9: Feature Importance Analysis ---
weights_ne, bias_ne = model_ne.get_coefficients()
print(f"\n{'='*50}")
print(f"  Feature Coefficients (Normal Equation)")
print(f"{'='*50}")
sorted_idx = np.argsort(np.abs(weights_ne))[::-1]
for idx in sorted_idx:
    bar = '+' * int(abs(weights_ne[idx]) * 5)
    sign = '+' if weights_ne[idx] > 0 else '-'
    print(f"  {feature_names[idx]:25s} {sign}{abs(weights_ne[idx]):8.4f}  {bar}")

# --- Step 10: Trading Signal Analysis ---
print(f"\n{'='*50}")
print(f"  Trading Signal Analysis")
print(f"{'='*50}")

predicted_returns = (y_pred_ne - X_test[:, -1]) / X_test[:, -1]  # Predicted % change
actual_returns = (y_test - X_test[:, -1]) / X_test[:, -1]

# Direction accuracy
correct_direction = np.sum(np.sign(predicted_returns) == np.sign(actual_returns))
direction_accuracy = correct_direction / len(actual_returns) * 100

print(f"  Direction accuracy: {direction_accuracy:.1f}%")
print(f"  Avg predicted return: {np.mean(predicted_returns)*100:.4f}%")
print(f"  Avg actual return:    {np.mean(actual_returns)*100:.4f}%")

# Simulated profit (long when predicted up, short when predicted down)
strategy_returns = np.sign(predicted_returns) * actual_returns
cumulative_strategy = np.cumprod(1 + strategy_returns)
cumulative_buyhold = np.cumprod(1 + actual_returns)

print(f"  Strategy final value:  ${cumulative_strategy[-1]*100:.2f} (from $100)")
print(f"  Buy-and-hold value:    ${cumulative_buyhold[-1]*100:.2f} (from $100)")
```

## Key Takeaways

- Linear Regression provides a **closed-form solution** via the Normal Equation, making it extremely fast to train on stock market data
- **Feature scaling is critical** when mixing price-level features (dollars) with percentage-based indicators (returns, RSI)
- The model's **interpretable coefficients** make it valuable for understanding which technical indicators drive price predictions
- Always use **chronological train/test splits** to avoid look-ahead bias in financial applications
- Linear Regression serves as an **essential baseline** that more complex models must beat to justify their additional complexity
- **Multicollinearity** among correlated indicators (price, moving averages) can destabilize coefficients; consider Ridge regression if this is a concern
- **Direction accuracy** (whether the model correctly predicts up vs. down) is often more important than price accuracy for trading strategies
- **Periodic retraining** with rolling windows helps the model adapt to changing market conditions
- Despite its simplicity, Linear Regression combined with well-engineered features can produce surprisingly competitive predictions for short-term stock price movements
