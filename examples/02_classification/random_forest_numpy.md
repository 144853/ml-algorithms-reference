# Random Forest - Complete Guide with Stock Market Applications

## Overview

Random Forest is an ensemble learning method that constructs multiple Decision Trees during training and outputs the class that is the mode (majority vote) of the individual trees' predictions. Introduced by Leo Breiman in 2001, it combines two powerful ideas: bagging (Bootstrap Aggregating) and random feature selection at each split. In the stock market context, Random Forest is one of the most reliable algorithms for predicting stock movement direction (up/down) because it dramatically reduces the overfitting problem that plagues single Decision Trees while maintaining the ability to capture complex, nonlinear relationships between technical indicators and price movements.

The algorithm works by training each tree on a random bootstrap sample of the data (sampling with replacement) and considering only a random subset of features at each split point. This dual randomization ensures that the individual trees are decorrelated, meaning they make different errors on different data points. When aggregated through majority voting, these diverse but individually weak predictors combine to form a strong classifier. For stock trading, this means the model is robust to noise and less likely to overfit to spurious patterns in historical market data.

Random Forest has become a workhorse in quantitative finance because it offers an excellent balance between predictive power, robustness, and computational feasibility. It handles high-dimensional feature sets (dozens of technical indicators), requires minimal hyperparameter tuning compared to gradient boosting methods, and provides built-in estimates of feature importance and prediction uncertainty through out-of-bag (OOB) error estimation.

## How It Works - The Math Behind It

### Bootstrap Aggregating (Bagging)

Given training set D of size n, create B bootstrap samples:
```
D_b = {(x_i, y_i)} drawn with replacement from D, |D_b| = n
```

Each bootstrap sample contains approximately 63.2% unique observations (the rest are duplicates). The ~36.8% of observations NOT in a bootstrap sample are called "out-of-bag" (OOB) samples and serve as a built-in validation set.

### Random Feature Selection

At each split in each tree, instead of considering all p features, randomly select a subset of size m:
```
m = sqrt(p)  for classification (common choice)
m = p/3      for regression
```

For 12 stock market features: m = sqrt(12) ~ 3-4 features per split.

### Ensemble Prediction

For classification (Buy/Sell):
```
y_hat = mode(h_1(x), h_2(x), ..., h_B(x))
```

Probability estimate:
```
P(Buy | x) = (1/B) * SUM(I(h_b(x) = Buy))
```

Where h_b is the b-th tree and I() is the indicator function.

### Out-of-Bag (OOB) Error Estimation

For each observation x_i, predict using only trees where x_i was NOT in the bootstrap sample:
```
OOB_prediction(x_i) = mode(h_b(x_i) for all b where x_i not in D_b)
OOB_error = (1/n) * SUM(I(OOB_prediction(x_i) != y_i))
```

This provides an unbiased estimate of generalization error without a separate validation set.

### Feature Importance (Mean Decrease in Impurity)

```
Importance(feature_j) = SUM over all trees, all nodes using feature_j:
    (n_node / n_total) * (impurity - n_left/n_node * impurity_left - n_right/n_node * impurity_right)
```

### Permutation Importance (More Robust)

```
Importance(feature_j) = OOB_accuracy_original - OOB_accuracy_after_permuting_feature_j
```

### Step-by-Step Process

1. **Generate B bootstrap samples** from the training data (e.g., B=500 trees)
2. **For each bootstrap sample**, grow an unpruned Decision Tree:
   - At each node, randomly select m features from p total
   - Find the best split among those m features
   - Split and continue until stopping criteria met
3. **No pruning**: Individual trees are grown deep (low bias, high variance)
4. **Aggregate predictions** by majority voting across all B trees
5. **Estimate uncertainty** using the vote distribution (e.g., 340/500 vote Buy = 68% confidence)

## Stock Market Use Case: Stock Movement Direction Prediction (Up/Down)

### The Problem

You are a quantitative portfolio manager running a systematic long/short equity strategy across 500 stocks. Each day, you need to predict whether each stock's price will move Up (return > 0) or Down (return <= 0) over the next trading day. You have 4 years of historical daily data with 15 technical and fundamental features per stock. The challenge is building a model robust enough to handle market noise, regime changes, and the inherently low signal-to-noise ratio of daily stock returns. Random Forest is chosen for its robustness and ability to handle high-dimensional feature spaces without overfitting.

### Stock Market Features (Input Data)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| RSI_14 | 14-day Relative Strength Index | Numerical | 0 - 100 |
| MACD_Line | MACD line value | Numerical | -5 to 5 |
| MACD_Signal | Signal line value | Numerical | -5 to 5 |
| SMA_Cross_20_50 | SMA20 / SMA50 ratio | Numerical | 0.85 - 1.15 |
| Volume_Ratio_5 | 5-day volume / 20-day volume | Numerical | 0.3 - 3.0 |
| Momentum_10 | 10-day price momentum (%) | Numerical | -15% to 15% |
| Volatility_20 | 20-day rolling volatility (%) | Numerical | 0.5% to 6% |
| BB_Pct | Bollinger Band %B | Numerical | -0.5 to 1.5 |
| ATR_14 | Average True Range (14-day) | Numerical | 0.5 to 10 |
| OBV_Slope | On-Balance Volume slope (5-day) | Numerical | -1M to 1M |
| Advance_Decline | Market breadth ratio | Numerical | 0.3 to 3.0 |
| VIX_Level | Market fear index | Numerical | 10 to 80 |
| Sector_Momentum | Sector relative strength (%) | Numerical | -10% to 10% |
| Earnings_Surprise | Last earnings surprise (%) | Numerical | -20% to 20% |
| Short_Interest | Short interest ratio | Numerical | 0.5 to 30 |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_stocks = 500
n_days = 1000  # per stock
n_samples = n_stocks * n_days  # Total: 500,000 data points

# In practice, you would flatten stock-day pairs
# Here we simulate aggregated features
n_total = 5000  # Simplified for demonstration

features = {
    'RSI_14': np.random.uniform(15, 85, n_total),
    'MACD_Line': np.random.normal(0, 1.5, n_total),
    'MACD_Signal': np.random.normal(0, 1.2, n_total),
    'SMA_Cross': np.random.normal(1.0, 0.05, n_total),
    'Volume_Ratio': np.random.lognormal(0, 0.3, n_total),
    'Momentum_10': np.random.normal(0.2, 4, n_total),
    'Volatility_20': np.random.uniform(0.8, 4.5, n_total),
    'BB_Pct': np.random.normal(0.5, 0.3, n_total),
    'ATR_14': np.random.uniform(1, 8, n_total),
    'OBV_Slope': np.random.normal(0, 100000, n_total),
    'Advance_Decline': np.random.lognormal(0, 0.3, n_total),
    'VIX_Level': np.random.uniform(12, 45, n_total),
    'Sector_Momentum': np.random.normal(0, 3, n_total),
    'Earnings_Surprise': np.random.normal(1, 5, n_total),
    'Short_Interest': np.random.exponential(5, n_total),
}

X = np.column_stack(list(features.values()))
feature_names = list(features.keys())

# Generate Up/Down labels with realistic complexity
signal = (0.01 * (features['RSI_14'] - 50) +
          0.2 * features['MACD_Line'] +
          0.1 * features['Momentum_10'] +
          2.0 * (features['SMA_Cross'] - 1.0) +
          -0.05 * features['Volatility_20'] +
          0.15 * features['Sector_Momentum'] +
          -0.01 * features['VIX_Level'] +
          0.02 * features['Earnings_Surprise'] +
          np.random.normal(0, 0.8, n_total))

y = (signal > 0).astype(int)  # 1 = Up, 0 = Down

print(f"Dataset: {n_total} stock-day observations")
print(f"Up days: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Down days: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
```

### The Model in Action

```python
# Random Forest with 500 trees, each seeing different data and features:
#
# Tree 1 (bootstrap sample 1, features: RSI, MACD_Line, VIX, Momentum_10):
#   -> If RSI < 40 AND Momentum_10 > 2: UP
#
# Tree 2 (bootstrap sample 2, features: Volume_Ratio, SMA_Cross, Sector_Mom):
#   -> If SMA_Cross > 1.02 AND Sector_Mom > 0: UP
#
# Tree 3 (bootstrap sample 3, features: BB_Pct, Volatility, Earnings):
#   -> If Earnings_Surprise > 3 AND Volatility < 2.5: UP
#
# ...Tree 500...
#
# Final Prediction: 312/500 trees vote UP (62.4%) => UP
# Confidence: 62.4% (moderate - consider half position)
```

## Advantages

1. **Dramatically Reduces Overfitting Compared to Single Trees**: By averaging hundreds of decorrelated trees, Random Forest cancels out individual tree noise. A single tree might overfit to a spurious pattern like "buy when RSI = 42.3," but across 500 trees, only genuine patterns survive averaging.

2. **Robust to Noisy Stock Market Data**: Financial data has a very low signal-to-noise ratio. Random Forest handles this well because the aggregation of many weak predictions produces a strong signal, following the "wisdom of crowds" principle.

3. **Built-in Feature Importance for Indicator Selection**: The algorithm naturally ranks which technical indicators matter most for predicting stock movements. This helps quants focus on the most informative features and discard noise-contributing indicators.

4. **Out-of-Bag Error Provides Free Validation**: The OOB error estimate gives a reliable measure of model performance without sacrificing training data for a validation set. This is valuable when historical stock data is limited.

5. **Handles High-Dimensional Feature Spaces**: With dozens of technical indicators, fundamental ratios, and market factors, Random Forest efficiently selects relevant features at each split without being overwhelmed by dimensionality.

6. **Parallelizable Training**: Each tree is independent and can be trained on a separate CPU core. For large-scale stock prediction across hundreds of stocks, this enables practical training times.

7. **Minimal Hyperparameter Tuning Required**: Unlike gradient boosting methods that require careful tuning of learning rate, max depth, and regularization, Random Forest works well with default parameters (500 trees, max_features=sqrt(n)).

8. **Prediction Confidence from Vote Distribution**: The fraction of trees voting for each class provides a natural confidence measure. A 90% Buy vote justifies a larger position than a 52% Buy vote.

## Disadvantages

1. **Less Interpretable Than Single Decision Trees**: While feature importances are available, you cannot easily extract "if-then" trading rules from a Random Forest with 500 trees. This makes it harder to explain trading decisions to non-technical stakeholders.

2. **Cannot Extrapolate Beyond Training Data Range**: Random Forest predictions are bounded by the range of training data. If VIX spikes to 80 during a crisis but training data only saw VIX up to 45, the model cannot properly handle this extreme condition.

3. **Memory Intensive for Large Forests**: Storing 500+ deep trees requires significant memory, especially when each tree has thousands of nodes. This can be a constraint for real-time trading systems processing thousands of stocks.

4. **Slower Prediction Than Linear Models**: Although each prediction is fast (tree traversal), aggregating predictions across 500 trees is slower than a single matrix multiplication in Logistic Regression. This matters for ultra-high-frequency trading.

5. **Biased Feature Importance with Correlated Features**: When technical indicators are correlated (e.g., RSI and Stochastic), the importance is split between them, making each appear less important than it truly is. Permutation importance partially addresses this.

6. **Struggles with Highly Imbalanced Classes**: If only 5% of stock-day pairs have a "Strong Buy" signal, the majority voting mechanism tends to favor the majority class. Class weighting or oversampling can help but adds complexity.

7. **No Built-in Handling of Temporal Dependencies**: Like individual Decision Trees, Random Forest treats each observation independently and cannot learn sequential patterns like "three consecutive up days followed by a reversal."

## When to Use in Stock Market

- When you need a robust, general-purpose model for stock direction prediction across many stocks
- For feature importance analysis to identify which technical indicators are most predictive
- When training data is limited and OOB error estimation is needed as a free validation method
- As a strong baseline before experimenting with more complex gradient boosting methods
- When model stability is important (small data changes should not drastically change predictions)
- For strategies where moderate interpretability (feature importance) is sufficient
- When you have many features and need implicit feature selection

## When NOT to Use in Stock Market

- When maximum predictive accuracy is needed and you are willing to tune extensively (use XGBoost/LightGBM)
- When transparent trading rules are required for compliance (use single Decision Trees or Logistic Regression)
- For ultra-high-frequency trading where prediction latency of < 1ms is required
- When the dataset has extreme class imbalance (rare event prediction like crash detection)
- When temporal dependencies are critical to the strategy (use RNNs/LSTMs)
- When memory constraints are severe (embedded or edge deployment)

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|---------------------------|
| n_estimators | Number of trees | 100 - 2000 | 500 is a good balance; diminishing returns above 1000 |
| max_depth | Maximum tree depth | None, 5 - 30 | 8-15; None (unlimited) can work with enough data |
| max_features | Features per split | sqrt(p), log2(p), 0.3*p | sqrt(p) for diversity; try 0.3*p if performance is low |
| min_samples_split | Min samples to split | 2 - 50 | 10-30 for noisy stock data |
| min_samples_leaf | Min samples per leaf | 1 - 30 | 5-20 to prevent memorizing individual trading days |
| bootstrap | Use bootstrap sampling | True/False | True (essential for Random Forest behavior) |
| oob_score | Compute OOB error | True/False | True for validation without data sacrifice |
| class_weight | Handle imbalance | balanced, custom | "balanced" when Up/Down ratio is skewed |
| max_samples | Bootstrap sample size | 0.5 - 1.0 | 0.7-0.8 can improve diversity |
| n_jobs | Parallel CPU cores | -1, 1-N | -1 (use all cores) for large datasets |

## Stock Market Performance Tips

1. **Use Enough Trees for Stable Predictions**: Start with 500 trees and increase until OOB error stabilizes. For stock prediction, 500-1000 trees typically provides a good convergence of the vote distribution.

2. **Feature Engineering Is Still Critical**: Random Forest does not create features. Engineer meaningful indicators like RSI divergence, volume profile changes, and cross-asset momentum before feeding to the model.

3. **Monitor OOB Error Across Time**: Calculate OOB error on a rolling basis. If OOB error increases over time, it signals regime change and the need for model retraining.

4. **Use Prediction Confidence for Position Sizing**: Instead of binary Buy/Sell, use the vote fraction (e.g., 75% of trees vote Buy) to scale position sizes. Higher confidence = larger position.

5. **Implement Walk-Forward Retraining**: Retrain the model monthly with a rolling 3-year window. This ensures the forest adapts to evolving market conditions while maintaining enough training data.

6. **Combine Permutation and Impurity-Based Feature Importance**: Use both methods to get a robust view of which indicators matter. Features important by both methods are more likely to be genuinely predictive.

## Comparison with Other Algorithms

| Criteria | Random Forest | Decision Tree | XGBoost | LightGBM | Logistic Regression |
|----------|--------------|--------------|---------|----------|-------------------|
| Overfitting Risk | Low | Very High | Moderate | Moderate | Low |
| Accuracy | High | Moderate | Very High | Very High | Moderate |
| Training Speed | Moderate | Fast | Slow | Fast | Very Fast |
| Interpretability | Moderate | Excellent | Low | Low | Excellent |
| Hyperparameter Sensitivity | Low | Moderate | High | High | Low |
| Feature Interactions | Automatic | Automatic | Automatic | Automatic | Manual |
| Handles Missing Data | Some implementations | Some | Yes | Yes | No |
| Best For (Stock Market) | Robust direction prediction | Rule generation | Competition-grade | Large-scale prediction | Baseline signals |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Random Forest from Scratch for Stock Direction Prediction
# ============================================================

class SimpleDecisionTree:
    """Simplified decision tree used as a base learner in Random Forest."""

    def __init__(self, max_depth=10, min_samples_split=10, min_samples_leaf=5,
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 1 - p**2 - (1-p)**2

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feat = None
        best_thresh = None

        parent_gini = self._gini(y)

        if self.max_features:
            feat_idx = np.random.choice(n_features, min(self.max_features, n_features),
                                        replace=False)
        else:
            feat_idx = np.arange(n_features)

        for f in feat_idx:
            values = np.unique(X[:, f])
            if len(values) > 30:
                values = np.percentile(X[:, f], np.linspace(10, 90, 30))

            for thresh in values:
                left = y[X[:, f] <= thresh]
                right = y[X[:, f] > thresh]

                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                gain = parent_gini - (len(left)/n_samples * self._gini(left) +
                                      len(right)/n_samples * self._gini(right))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        n_up = np.sum(y == 1)
        n_down = np.sum(y == 0)

        if (depth >= self.max_depth or len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'leaf': True, 'value': np.array([n_down, n_up]),
                    'prediction': 1 if n_up >= n_down else 0}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {'leaf': True, 'value': np.array([n_down, n_up]),
                    'prediction': 1 if n_up >= n_down else 0}

        left_mask = X[:, feat] <= thresh
        return {
            'leaf': False,
            'feature': feat,
            'threshold': thresh,
            'left': self._build(X[left_mask], y[left_mask], depth + 1),
            'right': self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict_counts(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class StockRandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=10,
                 min_samples_leaf=5, max_features='sqrt', max_samples=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.trees = []
        self.oob_indices = []
        self.feature_importances_ = None

    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.feature_names = feature_names or [f"F{i}" for i in range(n_features)]
        self.trees = []
        self.oob_indices = []

        if self.max_features == 'sqrt':
            mf = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            mf = int(np.log2(n_features))
        else:
            mf = self.max_features or n_features

        sample_size = self.max_samples or n_samples
        if isinstance(sample_size, float):
            sample_size = int(sample_size * n_samples)

        oob_predictions = np.zeros((n_samples, 2))
        oob_counts = np.zeros(n_samples)

        for b in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, size=sample_size, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(boot_idx))

            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.oob_indices.append(oob_idx)

            # OOB predictions
            if len(oob_idx) > 0:
                oob_preds = tree.predict_counts(X[oob_idx])
                oob_predictions[oob_idx] += oob_preds
                oob_counts[oob_idx] += 1

            if (b + 1) % 50 == 0:
                # Calculate current OOB accuracy
                valid = oob_counts > 0
                if valid.sum() > 0:
                    oob_classes = np.argmax(oob_predictions[valid], axis=1)
                    oob_acc = np.mean(oob_classes == y[valid])
                    print(f"Trees: {b+1}/{self.n_estimators}, OOB Accuracy: {oob_acc:.4f}")

        # Final OOB score
        valid = oob_counts > 0
        if valid.sum() > 0:
            oob_classes = np.argmax(oob_predictions[valid], axis=1)
            self.oob_score_ = np.mean(oob_classes == y[valid])
            print(f"\nFinal OOB Accuracy: {self.oob_score_:.4f}")

        return self

    def predict_proba(self, X):
        all_counts = np.zeros((len(X), 2))
        for tree in self.trees:
            counts = tree.predict_counts(X)
            # Normalize each tree's counts to probabilities
            row_sums = counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            all_counts += counts / row_sums
        return all_counts / self.n_estimators

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


# ============================================================
# Generate Synthetic Stock Market Data
# ============================================================
np.random.seed(42)
n_total = 3000

rsi = np.random.uniform(10, 90, n_total)
macd_line = np.random.normal(0, 1.5, n_total)
macd_signal = np.random.normal(0, 1.2, n_total)
sma_cross = np.random.normal(1.0, 0.04, n_total)
volume_ratio = np.random.lognormal(0, 0.3, n_total)
momentum = np.random.normal(0.1, 4, n_total)
volatility = np.random.uniform(0.8, 5, n_total)
bb_pct = np.random.normal(0.5, 0.25, n_total)
atr = np.random.uniform(1, 7, n_total)
obv_slope = np.random.normal(0, 80000, n_total)
adv_decline = np.random.lognormal(0, 0.25, n_total)
vix = np.random.uniform(12, 45, n_total)
sector_mom = np.random.normal(0, 3, n_total)
earnings = np.random.normal(1, 5, n_total)
short_interest = np.random.exponential(4, n_total)

X = np.column_stack([rsi, macd_line, macd_signal, sma_cross, volume_ratio,
                     momentum, volatility, bb_pct, atr, obv_slope,
                     adv_decline, vix, sector_mom, earnings, short_interest])

feature_names = ['RSI_14', 'MACD_Line', 'MACD_Signal', 'SMA_Cross', 'Volume_Ratio',
                 'Momentum_10', 'Volatility_20', 'BB_Pct', 'ATR_14', 'OBV_Slope',
                 'Adv_Decline', 'VIX', 'Sector_Mom', 'Earnings_Surp', 'Short_Interest']

# Complex nonlinear signal
signal = (0.02 * (rsi - 50) * (1 + 0.5 * np.sign(macd_line)) +
          0.15 * macd_line +
          0.08 * momentum +
          3.0 * (sma_cross - 1.0) +
          -0.04 * volatility +
          0.12 * sector_mom +
          -0.008 * vix +
          0.03 * earnings +
          -0.01 * short_interest +
          np.random.normal(0, 0.7, n_total))

y = (signal > 0).astype(int)

print(f"Total samples: {n_total}")
print(f"Up: {y.sum()} ({y.mean()*100:.1f}%), Down: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

# Walk-forward split
train_end = 2200
X_train, X_test = X[:train_end], X[train_end:]
y_train, y_test = y[:train_end], y[train_end:]

print(f"Train: {train_end} | Test: {n_total - train_end}")

# ============================================================
# Train Random Forest
# ============================================================
print("\n=== Training Random Forest ===")
rf = StockRandomForest(n_estimators=200, max_depth=8, min_samples_split=15,
                        min_samples_leaf=8, max_features='sqrt')
rf.fit(X_train, y_train, feature_names)

# ============================================================
# Evaluate
# ============================================================
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)

print(f"\n=== TEST SET PERFORMANCE ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confidence-based analysis
print(f"\n=== CONFIDENCE-BASED ANALYSIS ===")
high_conf = y_proba[:, 1] > 0.65
low_conf = (y_proba[:, 1] >= 0.45) & (y_proba[:, 1] <= 0.55)
if high_conf.sum() > 0:
    print(f"High confidence predictions (>65%): {high_conf.sum()}")
    print(f"  Accuracy: {np.mean(y_pred[high_conf] == y_test[high_conf]):.4f}")
if low_conf.sum() > 0:
    print(f"Low confidence predictions (45-55%): {low_conf.sum()}")
    print(f"  Accuracy: {np.mean(y_pred[low_conf] == y_test[low_conf]):.4f}")

# Simulated trading
print(f"\n=== SIMULATED TRADING ===")
daily_returns = momentum[train_end:] / 10 / 100
strategy_returns = np.where(y_pred == 1, daily_returns, -daily_returns)
cumul = np.cumsum(strategy_returns)
print(f"Strategy Return: {cumul[-1]*100:.2f}%")
print(f"Buy & Hold Return: {np.sum(daily_returns)*100:.2f}%")
print(f"Signals: {y_pred.sum()} Up / {(1-y_pred).sum()} Down out of {len(y_pred)}")

# Sample predictions
print(f"\n=== SAMPLE PREDICTIONS ===")
print(f"{'Day':>5} {'P(Up)':>8} {'Signal':>8} {'Actual':>8} {'Correct':>8}")
for i in range(12):
    signal_str = "UP" if y_pred[i] == 1 else "DOWN"
    actual_str = "UP" if y_test[i] == 1 else "DOWN"
    correct = "Yes" if y_pred[i] == y_test[i] else "No"
    print(f"{i+1:>5} {y_proba[i,1]:>8.3f} {signal_str:>8} {actual_str:>8} {correct:>8}")
```

## Key Takeaways

- Random Forest combines hundreds of decorrelated Decision Trees to produce robust stock direction predictions that are far less prone to overfitting than single trees
- The dual randomization (bootstrap sampling + random feature selection) ensures diversity among trees, critical for handling noisy financial data
- Out-of-bag error provides a free validation metric without sacrificing training data, which is especially valuable when historical stock data is limited
- Vote distribution across trees provides natural prediction confidence for position sizing in portfolio management
- Feature importance rankings help identify which technical indicators genuinely predict stock movements versus which are noise
- Walk-forward retraining is essential: retrain monthly or quarterly with rolling windows to adapt to changing market regimes
- Random Forest works well as a strong baseline; if it is significantly outperformed by gradient boosting methods, the additional accuracy often justifies the added complexity
- High-confidence predictions (where most trees agree) are substantially more accurate and should be weighted more heavily in trading strategies
