# Logistic Regression - Complete Guide with Stock Market Applications

## Overview

Logistic Regression is one of the most fundamental classification algorithms in machine learning, and despite its name, it is used for classification rather than regression. At its core, Logistic Regression models the probability that a given input belongs to a particular class by applying the logistic (sigmoid) function to a linear combination of input features. In the stock market context, it serves as a powerful tool for generating binary Buy/Sell signals based on technical indicators and fundamental data.

The algorithm works by finding the optimal decision boundary that separates two classes in feature space. Unlike linear regression, which predicts continuous values, Logistic Regression squashes its output through the sigmoid function to produce values between 0 and 1, which can be interpreted as probabilities. For stock trading, this means the model can output the probability that a stock will go up (Buy signal) or go down (Sell signal), giving traders not just a binary decision but a confidence measure.

Logistic Regression remains popular in quantitative finance because of its interpretability, computational efficiency, and strong theoretical foundation. Portfolio managers and risk analysts favor it because each feature's coefficient directly indicates how that feature influences the Buy/Sell decision, making it easy to explain trading decisions to stakeholders and regulators.

## How It Works - The Math Behind It

### The Sigmoid Function

The core of Logistic Regression is the sigmoid (logistic) function that maps any real-valued number to a value between 0 and 1:

```
sigma(z) = 1 / (1 + e^(-z))
```

Where `z` is the linear combination of inputs:

```
z = w0 + w1*x1 + w2*x2 + ... + wn*xn = W^T * X
```

In stock market terms:
- `x1, x2, ..., xn` are features like RSI, MACD, volume changes, price momentum
- `w1, w2, ..., wn` are learned weights that determine each feature's importance
- `w0` is the bias/intercept term
- The output `sigma(z)` represents P(Buy | features)

### The Cost Function (Log-Loss / Binary Cross-Entropy)

The model is trained by minimizing the log-loss cost function:

```
J(W) = -(1/m) * SUM[y_i * log(h(x_i)) + (1 - y_i) * log(1 - h(x_i))]
```

Where:
- `m` = number of training samples (historical trading days)
- `y_i` = actual label (1 for Buy, 0 for Sell)
- `h(x_i)` = predicted probability = sigma(W^T * x_i)

### Gradient Descent Update Rule

The weights are updated iteratively using gradient descent:

```
w_j = w_j - alpha * (1/m) * SUM[(h(x_i) - y_i) * x_ij]
```

Where `alpha` is the learning rate controlling step size.

### Step-by-Step Learning Process

1. **Initialize weights** to small random values or zeros
2. **Forward pass**: Compute z = W^T * X, then apply sigmoid to get predictions
3. **Compute loss**: Calculate log-loss between predictions and actual Buy/Sell labels
4. **Backward pass**: Compute gradients of loss with respect to each weight
5. **Update weights**: Adjust weights in the direction that reduces loss
6. **Repeat** steps 2-5 until convergence or max iterations reached

### Regularization

To prevent overfitting on noisy stock data, regularization is added:

- **L2 (Ridge)**: `J(W) + lambda * SUM(w_j^2)` - shrinks all weights, good for correlated features like overlapping technical indicators
- **L1 (Lasso)**: `J(W) + lambda * SUM(|w_j|)` - produces sparse weights, useful for feature selection among many indicators
- **Elastic Net**: Combination of L1 and L2

## Stock Market Use Case: Buy/Sell Signal Classification

### The Problem

You are a quantitative analyst at a hedge fund managing a portfolio of S&P 500 stocks. Each day, you need to decide whether to Buy or Sell each stock based on a combination of technical indicators and market conditions. You have 5 years of historical daily data (approximately 1,260 trading days per stock) with labeled outcomes: if the stock price increased by more than 0.5% the next day, the label is "Buy" (1); otherwise, it is "Sell" (0). The goal is to build a model that generates reliable daily Buy/Sell signals with associated confidence probabilities.

### Stock Market Features (Input Data)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| RSI_14 | 14-day Relative Strength Index | Numerical | 0 - 100 |
| MACD_Signal | MACD minus Signal line | Numerical | -5 to 5 |
| SMA_20_Cross | Price relative to 20-day SMA (%) | Numerical | -10% to 10% |
| SMA_50_Cross | Price relative to 50-day SMA (%) | Numerical | -20% to 20% |
| Volume_Ratio | Today's volume / 20-day avg volume | Numerical | 0.1 to 5.0 |
| Daily_Return | Previous day's return (%) | Numerical | -10% to 10% |
| Volatility_20 | 20-day rolling standard deviation | Numerical | 0.5% to 5% |
| BB_Position | Position within Bollinger Bands (%) | Numerical | -50% to 150% |
| ATR_14 | 14-day Average True Range | Numerical | 0.5 to 10 |
| OBV_Change | On-Balance Volume change (%) | Numerical | -20% to 20% |
| PE_Ratio | Price-to-Earnings ratio | Numerical | 5 to 60 |
| Market_Return | S&P 500 return (market factor) | Numerical | -5% to 5% |

### Example Data Structure

```python
import numpy as np

# Simulated stock market data for 1000 trading days
np.random.seed(42)
n_samples = 1000

# Generate realistic stock market features
stock_data = {
    'RSI_14': np.random.uniform(20, 80, n_samples),
    'MACD_Signal': np.random.normal(0, 1.5, n_samples),
    'SMA_20_Cross': np.random.normal(0, 3, n_samples),
    'SMA_50_Cross': np.random.normal(0, 5, n_samples),
    'Volume_Ratio': np.random.lognormal(0, 0.4, n_samples),
    'Daily_Return': np.random.normal(0.05, 1.5, n_samples),
    'Volatility_20': np.random.uniform(0.8, 4.0, n_samples),
    'BB_Position': np.random.normal(50, 30, n_samples),
    'ATR_14': np.random.uniform(1, 6, n_samples),
    'OBV_Change': np.random.normal(0, 5, n_samples),
    'PE_Ratio': np.random.uniform(8, 45, n_samples),
    'Market_Return': np.random.normal(0.03, 1.0, n_samples),
}

# Feature matrix
X = np.column_stack(list(stock_data.values()))

# Generate Buy/Sell labels based on a realistic relationship
z = (0.02 * (stock_data['RSI_14'] - 50) +
     0.3 * stock_data['MACD_Signal'] +
     0.15 * stock_data['SMA_20_Cross'] +
     0.1 * stock_data['Volume_Ratio'] +
     0.05 * stock_data['Daily_Return'] +
     -0.1 * stock_data['Volatility_20'] +
     0.2 * stock_data['Market_Return'] +
     np.random.normal(0, 0.5, n_samples))

y = (z > 0).astype(int)  # 1 = Buy, 0 = Sell

print(f"Dataset shape: {X.shape}")
print(f"Buy signals: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Sell signals: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
```

### The Model in Action

```python
# After training, the model produces outputs like:
# Day 1: Features -> z = 1.35 -> sigma(1.35) = 0.794 -> BUY (79.4% confidence)
# Day 2: Features -> z = -0.82 -> sigma(-0.82) = 0.306 -> SELL (69.4% confidence)
# Day 3: Features -> z = 0.12 -> sigma(0.12) = 0.530 -> BUY (53.0% confidence, weak signal)

# The coefficient for RSI_14 is 0.023:
#   -> Each 1-unit increase in RSI increases log-odds of Buy by 0.023
#   -> Stocks with higher RSI are slightly more likely to get Buy signals

# The coefficient for Volatility_20 is -0.15:
#   -> Higher volatility decreases Buy probability
#   -> Model is more cautious in volatile markets
```

## Advantages

1. **High Interpretability for Trading Decisions**: Each feature coefficient directly tells you how much influence that indicator has on the Buy/Sell decision. A portfolio manager can explain to clients exactly why the model recommended buying AAPL: "RSI was oversold at 28, MACD crossover was positive, and volume was 1.5x above average."

2. **Probability Outputs Enable Position Sizing**: Unlike many classifiers that only output a class label, Logistic Regression naturally produces calibrated probabilities. A 90% Buy signal can justify a larger position size than a 55% Buy signal, enabling sophisticated risk-adjusted portfolio construction.

3. **Computationally Efficient for Real-Time Trading**: Training and prediction are extremely fast, making it suitable for intraday or even real-time trading systems. A Logistic Regression model can score thousands of stocks in milliseconds, which is critical for large-scale systematic trading strategies.

4. **Robust to Noisy Financial Data with Regularization**: Stock market data is inherently noisy, and Logistic Regression with L1/L2 regularization handles this well. L1 regularization can automatically select the most important indicators from a large set of candidates, while L2 prevents any single indicator from dominating the signal.

5. **Strong Baseline Model for Benchmarking**: In quantitative finance, Logistic Regression serves as an excellent baseline. If a complex neural network only marginally outperforms Logistic Regression, the added complexity may not justify the reduced interpretability and increased overfitting risk.

6. **Well-Calibrated Probabilities for Risk Management**: The probabilistic outputs are naturally well-calibrated, meaning a predicted 70% probability of a Buy signal genuinely corresponds to approximately 70% historical accuracy. This is critical for accurate Value-at-Risk (VaR) calculations and risk budgeting.

7. **Feature Importance is Built-In**: The magnitude and sign of each weight directly indicate feature importance and direction of influence. This allows quants to validate that the model learned sensible relationships (e.g., positive MACD should increase Buy probability).

## Disadvantages

1. **Assumes Linear Decision Boundary**: Logistic Regression can only learn linear relationships between features and the log-odds of the outcome. Stock markets exhibit many nonlinear patterns (e.g., RSI being bearish both at extremes of 90 and 10), which Logistic Regression cannot capture without manual feature engineering.

2. **Cannot Capture Complex Feature Interactions**: The model treats each feature independently unless interaction terms are explicitly created. It will miss that "high volume + MACD crossover" together might be far more predictive than either feature alone, a common pattern in technical analysis.

3. **Sensitive to Feature Scaling**: Features must be standardized before training; otherwise, features with larger scales (like PE_Ratio ranging 5-60) will dominate over features with smaller scales (like Daily_Return ranging -0.1 to 0.1). Forgetting to scale is a common mistake in financial ML pipelines.

4. **Limited by Binary Classification for Multi-Class Markets**: Standard Logistic Regression handles only two classes. If you want to classify markets into "Strong Buy / Buy / Hold / Sell / Strong Sell," you need multinomial logistic regression or one-vs-rest approaches, which add complexity.

5. **Vulnerable to Multicollinearity Among Technical Indicators**: Many technical indicators are highly correlated (RSI and Stochastic, SMA_20 and SMA_50). This multicollinearity inflates variance of coefficient estimates, making individual feature weights unreliable even if overall predictions are reasonable.

6. **Struggles with Non-Stationary Market Data**: Financial markets change regimes over time (bull markets, bear markets, high-volatility periods). A Logistic Regression model trained on bull market data may perform poorly in a bear market because the linear decision boundary shifts with market conditions.

7. **Prone to Overfitting on Small Datasets**: With limited historical data (e.g., fewer than 500 trading days), the model can overfit to noise, especially when the number of features is large relative to the number of samples. This is particularly problematic for newly listed stocks or niche markets.

## When to Use in Stock Market

- When you need a transparent, explainable model for regulatory compliance or client reporting
- For generating initial Buy/Sell signals as a baseline before exploring complex models
- When probability estimates are needed for position sizing and risk management
- For high-frequency scoring where prediction speed is critical
- When working with well-engineered, relatively linear technical indicators
- As a component in ensemble strategies where interpretability of individual models matters
- For binary outcome prediction (stock goes up vs. down, beats earnings vs. misses)

## When NOT to Use in Stock Market

- When the relationship between indicators and price movement is highly nonlinear
- For multi-class market regime classification (use multinomial or other algorithms)
- When complex feature interactions are known to be important (e.g., sector rotation strategies)
- For time-series prediction where temporal dependencies matter (use RNNs or LSTMs instead)
- When the dataset is extremely imbalanced (e.g., predicting rare crash events)
- For high-dimensional alternative data (satellite imagery, social media text) without dimensionality reduction

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|---------------------------|
| Learning Rate (alpha) | Step size for gradient descent | 0.001 - 0.1 | Start with 0.01; decrease if loss oscillates |
| Regularization Strength (lambda) | Controls overfitting penalty | 0.001 - 10.0 | Use 0.1-1.0 for noisy stock data |
| Regularization Type | L1, L2, or Elastic Net | - | L1 for feature selection among many indicators; L2 when all features are relevant |
| Max Iterations | Training convergence limit | 100 - 10000 | 1000 is usually sufficient for stock data |
| Decision Threshold | Probability cutoff for Buy/Sell | 0.3 - 0.7 | 0.5 is default; raise to 0.6+ for conservative signals |
| Class Weight | Handling imbalanced Buy/Sell | balanced or custom | Use "balanced" if Buy/Sell ratio is skewed |
| Solver | Optimization algorithm | lbfgs, saga, liblinear | lbfgs for L2; saga for L1 with large datasets |
| Feature Scaling | Standardization method | StandardScaler, MinMax | StandardScaler is preferred for financial features |

## Stock Market Performance Tips

1. **Feature Engineering Matters More Than Model Complexity**: Create interaction terms like RSI * Volume_Ratio or MACD * Market_Return to help Logistic Regression capture nonlinear patterns. Polynomial features of degree 2 can significantly boost performance.

2. **Use Walk-Forward Validation**: Never use standard cross-validation for time-series stock data. Use walk-forward (expanding or sliding window) validation to avoid lookahead bias. Train on 2017-2019, validate on 2020, then retrain on 2017-2020 for 2021 predictions.

3. **Adjust Decision Threshold Based on Trading Costs**: The default 0.5 threshold may not be optimal when transaction costs exist. If your round-trip cost is 0.2%, you might need a threshold of 0.55+ for Buy signals to ensure expected returns exceed costs.

4. **Retrain Periodically to Handle Regime Changes**: Market conditions evolve. Retrain the model monthly or quarterly with a rolling window of 2-3 years of data to keep the model adapted to current market regimes.

5. **Monitor Feature Coefficient Stability**: If weights change dramatically between retraining periods, it signals instability. Stable coefficients across time suggest robust, generalizable patterns.

6. **Combine with Risk Filters**: Use the model's probability output as one input alongside risk filters like maximum drawdown limits, position size constraints, and correlation checks before executing trades.

## Comparison with Other Algorithms

| Criteria | Logistic Regression | Random Forest | XGBoost | SVM | Neural Network |
|----------|-------------------|---------------|---------|-----|---------------|
| Interpretability | Excellent | Moderate | Low | Low | Very Low |
| Training Speed | Very Fast | Fast | Moderate | Slow | Slow |
| Prediction Speed | Very Fast | Fast | Fast | Moderate | Fast (after training) |
| Handles Nonlinearity | Poor | Excellent | Excellent | Good | Excellent |
| Overfitting Risk | Low | Moderate | High | Moderate | High |
| Feature Interactions | Manual only | Automatic | Automatic | Kernel-based | Automatic |
| Probability Calibration | Excellent | Moderate | Poor (needs calibration) | Poor | Moderate |
| Best For (Stock Market) | Baseline signals, risk models | Feature-rich prediction | Competition-grade models | Market state classification | Complex pattern recognition |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Logistic Regression from Scratch for Buy/Sell Signals
# ============================================================

class StockLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.1, reg_type='l2'):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        m = len(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        # Add regularization
        if self.reg_type == 'l2':
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        elif self.reg_type == 'l1':
            loss += (self.lambda_reg / m) * np.sum(np.abs(self.weights))
        return loss

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.losses = []

        for i in range(self.n_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # Compute gradients
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            # Add regularization gradient
            if self.reg_type == 'l2':
                dw += (self.lambda_reg / m) * self.weights
            elif self.reg_type == 'l1':
                dw += (self.lambda_reg / m) * np.sign(self.weights)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

            if i % 200 == 0:
                accuracy = np.mean((y_pred >= 0.5) == y)
                print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# Generate Synthetic Stock Market Data
# ============================================================
np.random.seed(42)
n_days = 1200  # ~5 years of trading data

# Technical indicators
rsi = np.random.uniform(15, 85, n_days)
macd_signal = np.random.normal(0, 1.2, n_days)
sma20_cross = np.random.normal(0, 2.5, n_days)
sma50_cross = np.random.normal(0, 4.0, n_days)
volume_ratio = np.random.lognormal(0, 0.35, n_days)
daily_return = np.random.normal(0.04, 1.3, n_days)
volatility = np.random.uniform(0.8, 3.5, n_days)
bb_position = np.random.normal(50, 25, n_days)
atr = np.random.uniform(1, 5, n_days)
obv_change = np.random.normal(0, 4, n_days)
pe_ratio = np.random.uniform(10, 40, n_days)
market_return = np.random.normal(0.03, 0.9, n_days)

# Build feature matrix
X = np.column_stack([rsi, macd_signal, sma20_cross, sma50_cross,
                     volume_ratio, daily_return, volatility, bb_position,
                     atr, obv_change, pe_ratio, market_return])

feature_names = ['RSI_14', 'MACD_Signal', 'SMA_20_Cross', 'SMA_50_Cross',
                 'Volume_Ratio', 'Daily_Return', 'Volatility_20', 'BB_Position',
                 'ATR_14', 'OBV_Change', 'PE_Ratio', 'Market_Return']

# Generate labels: Buy (1) if next-day return > 0.5%
true_signal = (0.015 * (rsi - 50) + 0.25 * macd_signal + 0.12 * sma20_cross +
               0.08 * volume_ratio + 0.04 * daily_return - 0.08 * volatility +
               0.18 * market_return + np.random.normal(0, 0.6, n_days))
y = (true_signal > 0).astype(int)

print(f"Total trading days: {n_days}")
print(f"Buy signals: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Sell signals: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

# ============================================================
# Standardize Features
# ============================================================
X_mean = X[:900].mean(axis=0)
X_std = X[:900].std(axis=0)
X_scaled = (X - X_mean) / X_std

# Walk-forward split: Train on first 900 days, test on last 300
X_train, X_test = X_scaled[:900], X_scaled[900:]
y_train, y_test = y[:900], y[900:]

print(f"\nTrain set: {X_train.shape[0]} days")
print(f"Test set: {X_test.shape[0]} days")

# ============================================================
# Train Model
# ============================================================
model = StockLogisticRegression(learning_rate=0.01, n_iterations=1000,
                                 lambda_reg=0.1, reg_type='l2')
model.fit(X_train, y_train)

# ============================================================
# Evaluate on Test Set
# ============================================================
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test, threshold=0.5)

accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
f1 = 2 * precision * recall / (precision + recall)

print(f"\n--- Test Set Performance ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ============================================================
# Feature Importance (Coefficient Analysis)
# ============================================================
print(f"\n--- Feature Coefficients (Trading Signal Importance) ---")
coef_importance = sorted(zip(feature_names, model.weights),
                         key=lambda x: abs(x[1]), reverse=True)
for name, coef in coef_importance:
    direction = "BULLISH" if coef > 0 else "BEARISH"
    print(f"  {name:20s}: {coef:+.4f} ({direction})")

# ============================================================
# Simulated Trading Results
# ============================================================
print(f"\n--- Simulated Trading on Test Period ---")
daily_returns_test = daily_return[900:]

# Strategy: go long when Buy signal, flat when Sell signal
strategy_returns = y_pred * daily_returns_test / 100
buy_and_hold_returns = daily_returns_test / 100

total_strategy = np.sum(strategy_returns) * 100
total_buyhold = np.sum(buy_and_hold_returns) * 100
n_trades = np.sum(np.diff(y_pred) != 0)

print(f"Strategy Total Return: {total_strategy:.2f}%")
print(f"Buy & Hold Return: {total_buyhold:.2f}%")
print(f"Number of Trades: {n_trades}")
print(f"Days in Market: {y_pred.sum()} / {len(y_pred)}")

# Show sample predictions with confidence
print(f"\n--- Sample Daily Predictions ---")
print(f"{'Day':>5} {'Probability':>12} {'Signal':>8} {'Actual':>8} {'Correct':>8}")
for i in range(10):
    prob = y_pred_proba[i]
    signal = "BUY" if y_pred[i] == 1 else "SELL"
    actual = "BUY" if y_test[i] == 1 else "SELL"
    correct = "Yes" if y_pred[i] == y_test[i] else "No"
    print(f"{i+1:>5} {prob:>12.4f} {signal:>8} {actual:>8} {correct:>8}")
```

## Key Takeaways

- Logistic Regression provides a transparent, interpretable baseline for Buy/Sell signal generation in stock markets, making it ideal for regulatory-compliant trading strategies
- The sigmoid function naturally produces probability outputs that enable position sizing and risk management
- Feature coefficients directly indicate each indicator's bullish or bearish influence on the signal
- L1 regularization is particularly useful for selecting the most relevant technical indicators from a large candidate set
- Walk-forward validation is essential to avoid lookahead bias when backtesting on historical stock data
- The model works best as a linear signal generator; combine with nonlinear models in ensembles for improved performance
- Decision threshold tuning based on transaction costs can significantly improve real-world trading profitability
- Despite its simplicity, Logistic Regression often serves as a surprisingly hard-to-beat baseline in financial ML
