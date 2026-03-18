# AdaBoost (Adaptive Boosting) - Complete Guide with Stock Market Applications

## Overview

AdaBoost, short for Adaptive Boosting, is an ensemble learning algorithm that combines multiple weak classifiers to form a single strong classifier. The core idea is elegantly simple: train a sequence of simple models (often decision stumps -- single-level decision trees), and after each round, increase the weight of misclassified samples so the next weak learner focuses on the examples the previous ones got wrong. The final prediction is a weighted vote of all the weak classifiers, where better-performing classifiers receive higher voting weight.

In the context of stock market trading, AdaBoost is particularly useful for amplifying weak trading signals that individually have only marginal predictive power. A single technical indicator like RSI or a moving average crossover might be only slightly better than random at predicting market direction. However, AdaBoost can combine dozens of these weak signals into a composite classifier that captures complex, non-linear relationships between indicators and future price movements. This makes it a natural fit for systematic trading strategies where no single indicator is reliable on its own.

The algorithm's adaptive nature is especially valuable in financial markets because it automatically identifies which market conditions are hardest to classify and allocates more modeling effort to those ambiguous periods. For example, during transitional phases between bull and bear markets, AdaBoost will iteratively up-weight those difficult-to-classify trading days, forcing subsequent weak learners to specialize in exactly those challenging regimes.

## How It Works - The Math Behind It

### Step 1: Initialize Sample Weights

Given a training set of N samples, initialize uniform weights:

```
w_i^(1) = 1/N for i = 1, 2, ..., N
```

Each training sample (each trading day or observation) starts with equal importance.

### Step 2: For Each Boosting Round t = 1, 2, ..., T

**a) Train weak classifier h_t(x) using weighted samples:**

The weak learner minimizes the weighted classification error:

```
epsilon_t = sum(w_i^(t) * I(h_t(x_i) != y_i)) / sum(w_i^(t))
```

where `I()` is the indicator function (1 if the prediction is wrong, 0 otherwise).

**b) Compute the classifier weight (alpha):**

```
alpha_t = 0.5 * ln((1 - epsilon_t) / epsilon_t)
```

If `epsilon_t < 0.5` (better than random), then `alpha_t > 0` and the classifier gets a positive vote. A classifier with lower error gets a higher alpha, meaning its vote counts more in the final ensemble.

**c) Update sample weights:**

```
w_i^(t+1) = w_i^(t) * exp(-alpha_t * y_i * h_t(x_i))
```

Then normalize so weights sum to 1:

```
w_i^(t+1) = w_i^(t+1) / sum(w_j^(t+1))
```

Misclassified samples (where `y_i * h_t(x_i) < 0`) get their weights increased, while correctly classified samples get their weights decreased.

### Step 3: Final Strong Classifier

```
H(x) = sign(sum(alpha_t * h_t(x)) for t = 1 to T)
```

The final prediction is the sign of the weighted sum of all weak classifier outputs.

### Exponential Loss Function

AdaBoost implicitly minimizes the exponential loss:

```
L = sum(exp(-y_i * F(x_i)))
```

where `F(x) = sum(alpha_t * h_t(x))`. This loss function penalizes misclassified examples exponentially, which drives the algorithm to focus on hard-to-classify samples but also makes it sensitive to noisy labels -- a critical consideration in financial data where labels (e.g., "up day" vs. "down day") can be inherently noisy.

### Learning Rate (Shrinkage)

A learning rate parameter `eta` can be introduced to control the contribution of each weak learner:

```
F_t(x) = F_{t-1}(x) + eta * alpha_t * h_t(x)
```

Smaller values of `eta` (e.g., 0.01 to 0.1) require more boosting rounds but often produce better generalization, which is crucial for out-of-sample trading performance.

## Stock Market Use Case: Ensemble Boosting for Weak Trading Signals

### The Problem

A quantitative trading firm wants to predict whether a stock will close higher or lower than its opening price on any given day. They have access to a large collection of technical indicators, each of which provides only a weak directional signal. The goal is to combine these weak signals into a robust daily trading classifier that can generate actionable buy/sell signals.

### Stock Market Features (Input Data)

| Feature | Description | Type | Example Value |
|---------|-------------|------|---------------|
| RSI_14 | 14-day Relative Strength Index | Continuous | 62.4 |
| MACD_signal | MACD minus Signal line | Continuous | 0.85 |
| BB_position | Price position within Bollinger Bands (0 to 1) | Continuous | 0.72 |
| Volume_ratio | Today's volume / 20-day average volume | Continuous | 1.35 |
| ATR_14 | 14-day Average True Range (normalized) | Continuous | 0.023 |
| SMA_cross_20_50 | 1 if SMA20 > SMA50, else -1 | Binary | 1 |
| Price_vs_SMA200 | (Price - SMA200) / SMA200 | Continuous | 0.045 |
| Stoch_K | Stochastic %K oscillator | Continuous | 78.2 |
| ADX_14 | Average Directional Index | Continuous | 28.5 |
| OBV_slope | Slope of On-Balance Volume (5-day) | Continuous | 0.0012 |
| Gap_pct | Overnight gap percentage | Continuous | 0.003 |
| Prev_day_return | Previous day's return | Continuous | -0.008 |
| Momentum_10 | 10-day price momentum | Continuous | 0.032 |
| VWAP_deviation | Deviation from VWAP | Continuous | -0.005 |

**Target Variable:** `direction` -- +1 if close > open (up day), -1 if close <= open (down day)

### Example Data Structure

```python
import numpy as np
import pandas as pd

# Simulated stock market feature data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'RSI_14': np.random.uniform(20, 80, n_samples),
    'MACD_signal': np.random.normal(0, 1.5, n_samples),
    'BB_position': np.random.uniform(0, 1, n_samples),
    'Volume_ratio': np.random.lognormal(0, 0.3, n_samples),
    'ATR_14': np.random.uniform(0.005, 0.05, n_samples),
    'SMA_cross_20_50': np.random.choice([-1, 1], n_samples),
    'Price_vs_SMA200': np.random.normal(0, 0.05, n_samples),
    'Stoch_K': np.random.uniform(10, 90, n_samples),
    'ADX_14': np.random.uniform(10, 50, n_samples),
    'OBV_slope': np.random.normal(0, 0.002, n_samples),
    'Gap_pct': np.random.normal(0, 0.01, n_samples),
    'Prev_day_return': np.random.normal(0, 0.015, n_samples),
    'Momentum_10': np.random.normal(0, 0.04, n_samples),
    'VWAP_deviation': np.random.normal(0, 0.008, n_samples),
})

# Generate target with weak signal from features
signal = (0.3 * (data['RSI_14'] > 50).astype(float) +
          0.2 * np.sign(data['MACD_signal']) +
          0.15 * data['SMA_cross_20_50'] +
          0.1 * np.sign(data['Momentum_10']) +
          np.random.normal(0, 0.5, n_samples))  # noise
data['direction'] = np.where(signal > 0, 1, -1)

print(f"Dataset shape: {data.shape}")
print(f"Class distribution: {dict(zip(*np.unique(data['direction'], return_counts=True)))}")
print(f"\nFeature summary:")
print(data.describe().round(4))
```

### The Model in Action

```python
# AdaBoost decision process for a single trading day
# Features: RSI=65, MACD_signal=1.2, BB_position=0.8, Volume_ratio=1.4

# Weak Learner 1 (Decision Stump): RSI > 55 => BUY
#   alpha_1 = 0.42, prediction = +1 (BUY)

# Weak Learner 2 (Decision Stump): MACD_signal > 0.5 => BUY
#   alpha_2 = 0.38, prediction = +1 (BUY)

# Weak Learner 3 (Decision Stump): BB_position > 0.85 => SELL
#   alpha_3 = 0.35, prediction = -1 (SELL)

# Weak Learner 4 (Decision Stump): Volume_ratio > 1.3 => BUY
#   alpha_4 = 0.31, prediction = +1 (BUY)

# Final: sign(0.42*1 + 0.38*1 + 0.35*(-1) + 0.31*1) = sign(0.76) = +1 => BUY
```

## Advantages

1. **Amplifies weak trading signals effectively.** Individual technical indicators like RSI or MACD are only marginally predictive on their own. AdaBoost systematically combines these weak signals into a composite prediction that captures multivariate patterns no single indicator can detect. This mirrors how experienced traders mentally synthesize multiple chart readings.

2. **Automatic feature importance ranking.** The algorithm naturally assigns higher alpha weights to weak learners built on the most informative features. This reveals which technical indicators are most predictive for a given stock or market, providing insights that can guide further strategy development and feature engineering.

3. **Resistant to overfitting with proper regularization.** Compared to other complex models, AdaBoost with shallow decision stumps has low model complexity per weak learner. The ensemble builds complexity gradually, and with a learning rate applied, the model generalizes well to unseen market data. Empirical studies have shown AdaBoost can maintain out-of-sample performance for extended periods.

4. **Handles mixed feature types without preprocessing.** Stock market data often combines continuous indicators (RSI, MACD) with binary signals (moving average crossovers, candlestick patterns). AdaBoost handles both seamlessly since decision stumps naturally split on any feature type. This eliminates the need for extensive feature normalization or encoding.

5. **Interpretable ensemble structure.** Unlike black-box models, each weak learner in AdaBoost is a simple rule (e.g., "if RSI > 55, predict UP"). Traders and portfolio managers can inspect the ensemble to understand what conditions drive predictions, which is critical for regulatory compliance and risk management in institutional trading.

6. **Computationally efficient for real-time trading.** Training is sequential but each weak learner is extremely fast to train. Prediction is a simple weighted sum of stump outputs, making it suitable for intraday or even tick-level trading systems where latency matters. The model can be retrained daily with minimal computational overhead.

7. **Adapts to changing market difficulty.** By up-weighting misclassified samples, AdaBoost naturally dedicates more modeling capacity to ambiguous market periods (e.g., range-bound markets or regime transitions). This adaptive focus is valuable because the hardest-to-predict market days are often the most important for risk management.

## Disadvantages

1. **Highly sensitive to noisy labels in financial data.** Stock market direction labels are inherently noisy -- a stock might close up by 0.01% on a day with mixed signals. AdaBoost's exponential loss function aggressively up-weights these noisy, misclassified samples, causing subsequent weak learners to chase noise rather than signal. This can degrade performance significantly without careful label filtering.

2. **Vulnerable to outlier trading days.** Extreme market events (flash crashes, earnings surprises, circuit breakers) create outlier samples that receive enormous weight after being misclassified. The algorithm may devote many weak learners to fitting these rare events at the expense of normal market behavior, leading to a model that performs well on extremes but poorly on typical trading days.

3. **Sequential training prevents parallelization.** Unlike Random Forest or bagging methods, AdaBoost's weak learners must be trained sequentially since each depends on the previous round's weight updates. For a trading desk that needs to retrain models across thousands of stocks daily, this sequential bottleneck can be a practical limitation compared to easily parallelizable alternatives.

4. **Limited to binary classification in standard form.** The classic AdaBoost algorithm is designed for binary outcomes (up/down). Extending it to multi-class problems (e.g., strong buy/buy/hold/sell/strong sell) requires variants like AdaBoost.M1 or SAMME, which add complexity and may not perform as reliably. Many trading applications need more nuanced predictions than simple direction.

5. **Performance degrades in non-stationary markets.** Financial markets are fundamentally non-stationary -- the statistical relationships between features and outcomes change over time. AdaBoost trained on historical data may learn patterns that no longer hold in current market regimes. Without regular retraining and regime detection, the model's edge can decay rapidly.

6. **No built-in probability calibration.** AdaBoost's raw output scores are not well-calibrated probabilities. For position sizing and risk management in trading, you need reliable probability estimates (e.g., "70% chance of an up day"). Additional calibration steps (Platt scaling, isotonic regression) are required, adding complexity to the pipeline.

7. **Greedy optimization may miss global patterns.** The stage-wise additive approach means AdaBoost greedily selects the best weak learner at each round given current weights. It may fail to discover complex multi-feature interactions that would be apparent with a global optimization approach. In markets, some predictive patterns involve simultaneous conditions across multiple indicators.

## When to Use in Stock Market

- When you have many weak technical indicators that individually show marginal edge but might be collectively powerful
- For daily or weekly trading signal generation where the signal-to-noise ratio is moderate
- When interpretability of the trading model is important for compliance or risk oversight
- When combining signals from heterogeneous sources (technical, fundamental ratios, sentiment scores)
- For stocks with sufficient trading history (1000+ daily observations) to support ensemble training
- When computational resources are limited and you need an efficient, low-latency prediction model
- For building base models in a larger model stacking or blending framework

## When NOT to Use in Stock Market

- When label noise is extremely high (e.g., predicting tiny intraday price changes where direction is essentially random)
- For high-frequency trading where microsecond latency is critical and even simple model evaluation is too slow
- When you need calibrated probability outputs for Kelly criterion position sizing without additional calibration steps
- During extreme market volatility when outlier samples will dominate the weight distribution
- When the dataset is very small (fewer than 200 trading days), as the ensemble will overfit
- For multi-class market state classification without using appropriate multi-class AdaBoost variants
- When features have strong multicollinear relationships that cause decision stumps to split redundantly

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Recommendation |
|----------------|-------------|---------------|----------------------------|
| n_estimators | Number of weak learners (boosting rounds) | 50 - 500 | Start with 100, increase to 300 for complex feature sets. Monitor validation set performance for overfitting. |
| learning_rate | Shrinkage factor applied to each weak learner | 0.01 - 1.0 | Use 0.05 - 0.1 for financial data. Lower values require more estimators but generalize better to future market conditions. |
| base_estimator | Type of weak learner | Decision stump (max_depth=1) | Decision stumps are standard. max_depth=2 can capture two-way feature interactions (e.g., RSI + Volume) but increases overfitting risk. |
| algorithm | SAMME or SAMME.R | - | SAMME.R (real-valued) generally performs better and converges faster. Use SAMME.R unless weak learners cannot output probability estimates. |
| random_state | Random seed for reproducibility | Any integer | Always set for reproducible backtesting results. Use multiple seeds to assess model stability. |

## Stock Market Performance Tips

1. **Filter noisy labels aggressively.** Consider only trading days where the absolute return exceeds a threshold (e.g., |return| > 0.2%) as training samples. This removes near-zero return days where the "up" or "down" label is essentially random, preventing AdaBoost from wasting capacity on unfittable noise.

2. **Use time-series cross-validation.** Never use random k-fold cross-validation for financial data. Instead, use expanding window or walk-forward validation: train on data up to date T, validate on T+1 to T+k, then advance the window. This respects the temporal structure of market data and provides realistic performance estimates.

3. **Apply feature importance pruning.** After initial training, examine the feature importances (based on how often each feature is selected and its associated alpha). Remove features with very low importance, then retrain. This reduces noise and improves out-of-sample performance.

4. **Monitor the weight distribution during training.** If a few samples accumulate extreme weights (>10x the mean), they are likely outliers or mislabeled points. Consider capping weights or removing those samples to prevent the ensemble from overfitting to anomalies.

5. **Combine with a regime detection overlay.** Train separate AdaBoost models for different market regimes (trending vs. mean-reverting, high vs. low volatility). Use a regime classifier to select the appropriate model. This addresses non-stationarity more effectively than a single model.

6. **Retrain regularly but not too frequently.** Weekly or monthly retraining with an expanding training window is usually optimal for daily trading models. Daily retraining can introduce instability, while quarterly retraining may miss regime changes.

7. **Ensemble the ensemble.** Use AdaBoost predictions as features in a meta-model (stacking). Combine AdaBoost output with predictions from Random Forest, gradient boosting, and linear models for a more robust final trading signal.

## Comparison with Other Algorithms

| Criterion | AdaBoost | Random Forest | Gradient Boosting (XGBoost) | Logistic Regression | SVM |
|-----------|----------|---------------|-----------------------------|---------------------|-----|
| Handling weak signals | Excellent -- designed for this | Good | Excellent | Poor -- needs strong linear signals | Moderate |
| Noise sensitivity | High (exponential loss) | Low (bagging averages noise) | Moderate (customizable loss) | Low | Moderate |
| Training speed | Moderate (sequential) | Fast (parallelizable) | Moderate (sequential) | Very fast | Slow for large N |
| Interpretability | High (inspect stumps) | Moderate (feature importance) | Low (complex trees) | High (coefficients) | Low (kernel space) |
| Overfitting risk | Moderate | Low | High without regularization | Low | Moderate |
| Handles non-linearity | Yes (ensemble of stumps) | Yes (deep trees) | Yes (deep trees) | No (linear only) | Yes (kernel trick) |
| Probability calibration | Poor (needs calibration) | Good (natural probabilities) | Moderate | Excellent (logistic output) | Poor (needs calibration) |
| Market regime adaptability | Moderate | Moderate | Good (with retraining) | Poor | Poor |
| Feature preprocessing | Minimal | Minimal | Minimal | Requires scaling | Requires scaling |
| Best stock market use | Combining indicator signals | General classification | High-accuracy prediction | Simple trend following | Market state classification |

## Real-World Stock Market Example

```python
import numpy as np
from collections import Counter

# =============================================================================
# AdaBoost from Scratch for Stock Market Direction Prediction
# =============================================================================

class DecisionStump:
    """A single-level decision tree (decision stump) as the weak learner."""

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1  # 1 or -1
        self.alpha = None

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = np.unique(feature_values)

            # Sample thresholds to speed up training
            if len(thresholds) > 50:
                thresholds = np.percentile(feature_values,
                                           np.linspace(0, 100, 50))

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    error = np.sum(sample_weights[predictions != y])

                    if error < min_error:
                        min_error = error
                        self.feature_idx = feature_i
                        self.threshold = threshold
                        self.polarity = polarity

        return min_error

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X[:, self.feature_idx]

        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1

        return predictions


class AdaBoostClassifier:
    """AdaBoost classifier for stock market direction prediction."""

    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.stump_errors = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, 1.0 / n_samples)

        for t in range(self.n_estimators):
            stump = DecisionStump()
            error = stump.fit(X, y, sample_weights)

            # Clip error to avoid division by zero or log(0)
            error = np.clip(error, 1e-10, 1.0 - 1e-10)

            # Compute classifier weight
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            stump.alpha = alpha

            # Get predictions and update weights
            predictions = stump.predict(X)
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            # Cap extreme weights to prevent outlier domination
            max_weight = 10.0 / n_samples
            sample_weights = np.minimum(sample_weights, max_weight)
            sample_weights /= np.sum(sample_weights)

            self.stumps.append(stump)
            self.stump_errors.append(error)

        return self

    def predict(self, X):
        stump_preds = np.array([
            stump.alpha * stump.predict(X) for stump in self.stumps
        ])
        return np.sign(np.sum(stump_preds, axis=0))

    def decision_function(self, X):
        """Raw decision scores (not calibrated probabilities)."""
        stump_preds = np.array([
            stump.alpha * stump.predict(X) for stump in self.stumps
        ])
        return np.sum(stump_preds, axis=0)

    def feature_importances(self, feature_names=None):
        """Compute feature importance based on selection frequency and alpha."""
        n_features = max(s.feature_idx for s in self.stumps) + 1
        importances = np.zeros(n_features)

        for stump in self.stumps:
            importances[stump.feature_idx] += abs(stump.alpha)

        importances /= importances.sum()

        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return importances


# =============================================================================
# Generate Realistic Stock Market Data
# =============================================================================

def generate_stock_features(n_days=1500, seed=42):
    """Generate synthetic but realistic stock market feature data."""
    np.random.seed(seed)

    # Simulate a stock price path
    returns = np.random.normal(0.0003, 0.015, n_days)
    prices = 100 * np.cumprod(1 + returns)
    volumes = np.random.lognormal(15, 0.5, n_days)

    # Calculate technical indicators
    features = {}

    # RSI (14-day)
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
    rs = avg_gain / avg_loss
    features['RSI_14'] = 100 - (100 / (1 + rs))

    # Simple Moving Averages
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
    sma_50 = np.convolve(prices, np.ones(50)/50, mode='same')
    sma_200 = np.convolve(prices, np.ones(200)/200, mode='same')
    features['SMA_cross_20_50'] = np.where(sma_20 > sma_50, 1, -1).astype(float)
    features['Price_vs_SMA200'] = (prices - sma_200) / sma_200

    # MACD
    ema_12 = np.convolve(prices, np.ones(12)/12, mode='same')
    ema_26 = np.convolve(prices, np.ones(26)/26, mode='same')
    macd = ema_12 - ema_26
    signal_line = np.convolve(macd, np.ones(9)/9, mode='same')
    features['MACD_signal'] = macd - signal_line

    # Bollinger Band position
    bb_std = np.array([np.std(prices[max(0,i-20):i+1]) for i in range(n_days)])
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_range = np.where(bb_range == 0, 1e-10, bb_range)
    features['BB_position'] = (prices - bb_lower) / bb_range

    # Volume ratio
    vol_sma_20 = np.convolve(volumes, np.ones(20)/20, mode='same')
    vol_sma_20 = np.where(vol_sma_20 == 0, 1e-10, vol_sma_20)
    features['Volume_ratio'] = volumes / vol_sma_20

    # ATR (normalized)
    high = prices * (1 + np.random.uniform(0, 0.02, n_days))
    low = prices * (1 - np.random.uniform(0, 0.02, n_days))
    tr = high - low
    features['ATR_14'] = np.convolve(tr, np.ones(14)/14, mode='same') / prices

    # Momentum
    features['Momentum_10'] = np.zeros(n_days)
    features['Momentum_10'][10:] = (prices[10:] - prices[:-10]) / prices[:-10]

    # Previous day return
    features['Prev_day_return'] = np.zeros(n_days)
    features['Prev_day_return'][1:] = returns[1:]

    # Gap percentage (simulated)
    features['Gap_pct'] = np.random.normal(0, 0.005, n_days)

    # ADX (simplified)
    features['ADX_14'] = np.random.uniform(10, 50, n_days)

    # Stochastic K
    features['Stoch_K'] = np.random.uniform(10, 90, n_days)

    # Create feature matrix
    feature_names = list(features.keys())
    X = np.column_stack([features[f] for f in feature_names])

    # Target: next day direction (up=1, down=-1)
    y = np.zeros(n_days)
    y[:-1] = np.where(returns[1:] > 0, 1, -1)
    y[-1] = 1  # placeholder for last day

    # Remove warmup period
    warmup = 200
    X = X[warmup:]
    y = y[warmup:]

    return X, y, feature_names


# =============================================================================
# Training and Evaluation
# =============================================================================

# Generate data
X, y, feature_names = generate_stock_features(n_days=1500)
print(f"Dataset: {X.shape[0]} trading days, {X.shape[1]} features")
print(f"Class balance: {Counter(y)}")

# Time-series train/test split (no shuffling!)
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTraining: {len(X_train)} days | Testing: {len(X_test)} days")

# Train AdaBoost
model = AdaBoostClassifier(n_estimators=150, learning_rate=0.05)
model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Accuracy
train_acc = np.mean(train_preds == y_train)
test_acc = np.mean(test_preds == y_test)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# Feature importances
importances = model.feature_importances(feature_names)
print("\nFeature Importances:")
for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = '#' * int(imp * 50)
    print(f"  {name:20s}: {imp:.4f} {bar}")

# Simulated trading performance
decision_scores = model.decision_function(X_test)
test_returns = np.random.normal(0.0003, 0.015, len(X_test))  # simulated

# Strategy: trade in predicted direction with confidence scaling
positions = np.sign(decision_scores)
strategy_returns = positions * test_returns
cumulative_strategy = np.cumprod(1 + strategy_returns)
cumulative_buyhold = np.cumprod(1 + test_returns)

print(f"\n--- Trading Performance (Test Period) ---")
print(f"Strategy cumulative return: {(cumulative_strategy[-1] - 1)*100:.2f}%")
print(f"Buy & Hold cumulative return: {(cumulative_buyhold[-1] - 1)*100:.2f}%")
print(f"Strategy Sharpe (annualized): "
      f"{np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns):.2f}")
print(f"Buy & Hold Sharpe (annualized): "
      f"{np.sqrt(252) * np.mean(test_returns) / np.std(test_returns):.2f}")
print(f"Max drawdown (strategy): "
      f"{(1 - cumulative_strategy / np.maximum.accumulate(cumulative_strategy)).max()*100:.2f}%")

# Error rate per boosting round
print(f"\nBoosting rounds error rates (first 10): "
      f"{[f'{e:.4f}' for e in model.stump_errors[:10]]}")
print(f"Boosting rounds error rates (last 10): "
      f"{[f'{e:.4f}' for e in model.stump_errors[-10:]]}")
```

## Key Takeaways

1. **AdaBoost excels at combining weak trading signals** that individually have marginal predictive power into a strong composite classifier through iterative reweighting of misclassified samples.

2. **The exponential loss function is a double-edged sword** -- it drives powerful learning but makes the algorithm sensitive to label noise and outliers, both of which are prevalent in financial data. Label filtering and weight capping are essential preprocessing steps.

3. **Feature importance from AdaBoost provides actionable trading insight** by revealing which technical indicators are most predictive for a given stock or market, enabling more focused strategy development.

4. **Time-series cross-validation is mandatory** when applying AdaBoost to financial data. Standard random cross-validation produces misleadingly optimistic performance estimates due to temporal autocorrelation.

5. **Regular retraining is necessary** because financial markets are non-stationary. A model trained on data from a low-volatility bull market will likely underperform during a high-volatility bear market. Monthly or quarterly retraining with an expanding window is recommended.

6. **Decision score calibration is important for trading applications.** Raw AdaBoost scores should be calibrated to probabilities for proper position sizing, or alternatively, use score magnitude as a confidence indicator for trade filtering -- only take positions when the ensemble has high agreement.

7. **AdaBoost works best as part of a larger system.** Its predictions can serve as one input to a meta-model or be combined with risk management overlays, position sizing algorithms, and regime detection modules for a complete trading strategy.
