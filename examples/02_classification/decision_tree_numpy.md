# Decision Tree - Complete Guide with Stock Market Applications

## Overview

A Decision Tree is a non-parametric supervised learning algorithm that partitions the feature space into regions by making a series of hierarchical, binary decisions based on feature values. Each internal node represents a decision rule (e.g., "Is RSI > 70?"), each branch represents the outcome of that decision, and each leaf node represents a final classification (Buy, Sell, or Hold). In the stock market context, Decision Trees excel at generating interpretable trading rules that mirror how human traders think about market conditions.

The algorithm constructs its tree by recursively finding the best feature and threshold to split the data at each node, maximizing the purity of the resulting child nodes. For stock market classification, this means the tree automatically discovers threshold-based rules like "If MACD > 0 AND Volume > 1.5x average AND RSI < 70, then Buy." These rules are easy to verify, backtest, and communicate to portfolio managers and compliance teams.

Decision Trees are particularly appealing in quantitative finance because they naturally handle nonlinear relationships and feature interactions without requiring manual feature engineering. A tree can learn that "RSI > 70 is bearish UNLESS volume is also spiking," capturing conditional logic that linear models miss entirely. However, single Decision Trees are prone to overfitting on noisy stock market data, which is why they are often used as building blocks for ensemble methods like Random Forest and Gradient Boosting.

## How It Works - The Math Behind It

### Splitting Criteria

At each node, the tree evaluates every feature and every possible threshold to find the split that best separates the classes. The two most common criteria are:

**Gini Impurity:**
```
Gini(t) = 1 - SUM(p_i^2)
```
Where `p_i` is the proportion of class `i` at node `t`. For a binary Buy/Sell problem:
```
Gini(t) = 1 - p_buy^2 - p_sell^2
```
A pure node (all Buy or all Sell) has Gini = 0. A perfectly mixed node has Gini = 0.5.

**Information Gain (Entropy):**
```
Entropy(t) = -SUM(p_i * log2(p_i))
```
Information Gain for a split:
```
IG = Entropy(parent) - SUM((n_child / n_parent) * Entropy(child))
```

### Finding the Best Split

For each candidate split on feature `j` at threshold `s`:
```
Quality(j, s) = Impurity(parent) - (n_left/n * Impurity(left) + n_right/n * Impurity(right))
```

The algorithm searches over all features and all unique thresholds to find:
```
(j*, s*) = argmax Quality(j, s)
```

### Step-by-Step Tree Building Process

1. **Start with all data at root**: All 1,000 trading days in one node
2. **Evaluate all possible splits**: For each of 12 features, try every unique value as a threshold
3. **Select best split**: Choose the feature and threshold with highest information gain
4. **Create child nodes**: Divide data into left (condition true) and right (condition false)
5. **Recurse**: Repeat steps 2-4 for each child node
6. **Stop when**: Maximum depth reached, minimum samples per leaf reached, or node is pure

### Prediction

For a new trading day, traverse the tree from root to leaf following the decision rules:
```
IF RSI_14 <= 35.2:
    IF MACD_Signal > 0.5:
        IF Volume_Ratio > 1.3:
            PREDICT: BUY (confidence: 82%)
        ELSE:
            PREDICT: BUY (confidence: 61%)
    ELSE:
        PREDICT: SELL (confidence: 73%)
ELSE:
    ...continue traversal...
```

### Pruning

To prevent overfitting on noisy stock data:
- **Pre-pruning**: Limit max_depth, min_samples_split, min_samples_leaf during construction
- **Post-pruning (Cost-Complexity)**: Grow full tree, then prune branches that do not improve validation performance. The cost-complexity parameter alpha controls aggressiveness:
```
Cost_alpha(T) = R(T) + alpha * |T|
```
Where R(T) is misclassification rate and |T| is number of leaves.

## Stock Market Use Case: Trading Rule Generation from Market Conditions

### The Problem

You are a systematic trader who wants to generate explicit, rule-based trading strategies from historical market data. Rather than relying on black-box models, you need transparent rules that can be audited, explained to investors, and easily implemented in a trading engine. Your goal is to classify each trading day into a Buy or Sell action based on a combination of technical indicators, price action, and market conditions. The Decision Tree will automatically discover optimal threshold values and rule combinations.

### Stock Market Features (Input Data)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| RSI_14 | 14-day Relative Strength Index | Numerical | 0 - 100 |
| MACD_Histogram | MACD histogram value | Numerical | -3 to 3 |
| Price_vs_SMA20 | Price distance from 20-day SMA (%) | Numerical | -15% to 15% |
| Price_vs_SMA50 | Price distance from 50-day SMA (%) | Numerical | -25% to 25% |
| Volume_Spike | Volume / 20-day average volume | Numerical | 0.2 to 5.0 |
| Prev_5D_Return | Past 5-day cumulative return (%) | Numerical | -10% to 10% |
| Volatility_10 | 10-day realized volatility (%) | Numerical | 0.5% to 6% |
| BB_Width | Bollinger Band width (%) | Numerical | 1% to 15% |
| ATR_Pct | ATR as percentage of price | Numerical | 0.5% to 5% |
| Stochastic_K | Stochastic %K oscillator | Numerical | 0 - 100 |
| ADX | Average Directional Index | Numerical | 0 - 80 |
| Market_Trend | S&P 500 above 200-day SMA (binary) | Categorical | 0 or 1 |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_samples = 1200

# Generate realistic technical indicator data
stock_features = {
    'RSI_14': np.random.uniform(15, 85, n_samples),
    'MACD_Histogram': np.random.normal(0, 0.8, n_samples),
    'Price_vs_SMA20': np.random.normal(0, 3, n_samples),
    'Price_vs_SMA50': np.random.normal(0, 5, n_samples),
    'Volume_Spike': np.random.lognormal(0, 0.4, n_samples),
    'Prev_5D_Return': np.random.normal(0.1, 2.5, n_samples),
    'Volatility_10': np.random.uniform(0.8, 4.5, n_samples),
    'BB_Width': np.random.uniform(2, 12, n_samples),
    'ATR_Pct': np.random.uniform(0.8, 3.5, n_samples),
    'Stochastic_K': np.random.uniform(5, 95, n_samples),
    'ADX': np.random.uniform(10, 60, n_samples),
    'Market_Trend': np.random.binomial(1, 0.6, n_samples).astype(float),
}

X = np.column_stack(list(stock_features.values()))
feature_names = list(stock_features.keys())

# Generate labels using nonlinear rules (what a tree should discover)
y = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    if stock_features['RSI_14'][i] < 35 and stock_features['MACD_Histogram'][i] > 0:
        y[i] = 1  # Buy: oversold with positive momentum
    elif stock_features['RSI_14'][i] > 65 and stock_features['MACD_Histogram'][i] < 0:
        y[i] = 0  # Sell: overbought with negative momentum
    elif stock_features['Market_Trend'][i] == 1 and stock_features['Price_vs_SMA20'][i] > 0:
        y[i] = 1  # Buy: uptrend confirmed
    elif stock_features['Volatility_10'][i] > 3.5:
        y[i] = 0  # Sell: too volatile
    else:
        y[i] = np.random.binomial(1, 0.5)  # Noise in ambiguous cases

# Add some noise to make it realistic
noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
y[noise_idx] = 1 - y[noise_idx]

print(f"Dataset: {n_samples} trading days, {len(feature_names)} features")
print(f"Buy signals: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Sell signals: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
```

### The Model in Action

```
Trading Rule Tree (simplified):
                    [RSI_14 <= 34.8?]
                   /                  \
            YES (oversold)        NO
              /                      \
    [MACD_Hist > 0.12?]       [RSI_14 > 65.3?]
       /           \               /           \
   BUY (85%)   SELL (58%)   [MACD < -0.3?]    [Market_Trend=1?]
                                /    \              /       \
                           SELL(79%) HOLD      BUY(68%)  SELL(55%)

Interpretation:
- Rule 1: RSI < 35 AND MACD > 0 => BUY (oversold with momentum reversal)
- Rule 2: RSI > 65 AND MACD < -0.3 => SELL (overbought losing momentum)
- Rule 3: Non-extreme RSI AND bullish market trend => BUY (trend following)
```

## Advantages

1. **Generates Human-Readable Trading Rules**: The tree structure directly produces "if-then" rules that traders can understand, verify, and manually override. A rule like "If RSI < 30 and Volume > 2x average, BUY" can be validated against trading intuition before deployment.

2. **Captures Nonlinear Relationships Naturally**: Unlike Logistic Regression, Decision Trees can learn that RSI at 20 is bullish (oversold bounce) while RSI at 80 is bearish (overbought), without requiring manual feature transformations.

3. **Handles Feature Interactions Automatically**: The hierarchical structure naturally captures interactions like "MACD crossover is only a Buy signal when the market is in an uptrend (Market_Trend = 1)," which would require explicit interaction terms in linear models.

4. **No Feature Scaling Required**: Decision Trees are invariant to monotonic transformations of features. You can mix raw RSI (0-100 scale) with ATR_Pct (0.5-5% scale) without standardization.

5. **Handles Mixed Data Types**: The algorithm works with both numerical indicators (RSI, MACD) and categorical features (Market_Trend, Sector) without encoding or special preprocessing.

6. **Fast Training and Prediction**: A single Decision Tree trains in seconds even on large datasets, and prediction is simply traversing a tree, making it suitable for real-time trading signal generation.

7. **Built-in Feature Selection**: The tree automatically selects the most informative features. Features that never appear in any split can be safely removed from the trading strategy, simplifying the system.

## Disadvantages

1. **Highly Prone to Overfitting on Market Data**: Without proper pruning, a Decision Tree will memorize every noise pattern in historical stock data, creating overly specific rules that do not generalize. A tree might learn "If RSI = 42.3 AND MACD = 0.178, BUY" which is pure noise.

2. **Unstable to Small Data Changes**: Small changes in the training data can produce completely different tree structures. If you add or remove 50 trading days, the entire rule set may change, making strategy validation unreliable.

3. **Axis-Aligned Splits Are Suboptimal for Diagonal Boundaries**: Decision Trees only split along one feature at a time (perpendicular to feature axes). If the true decision boundary is a linear combination like "0.3*RSI + 0.7*MACD > threshold," the tree needs many splits to approximate it.

4. **Greedy Construction Leads to Suboptimal Trees**: The algorithm makes locally optimal splits without considering global tree structure. The best first split may not lead to the best overall tree, potentially missing more profitable trading rule combinations.

5. **Poor Extrapolation Beyond Training Data Range**: If the training data has RSI values between 20 and 80 but the test data encounters RSI = 10 during a crash, the tree cannot extrapolate and will apply the nearest learned rule, which may be inappropriate.

6. **Binary Splits Create Artificial Discontinuities**: A threshold at RSI = 30 treats RSI = 29.9 and RSI = 30.1 completely differently, despite being nearly identical. This creates fragile trading rules at boundary values.

7. **Cannot Capture Temporal Dependencies**: Decision Trees treat each trading day independently and cannot learn patterns like "three consecutive days of declining volume precedes a breakout." They ignore the sequential nature of stock data.

## When to Use in Stock Market

- When you need transparent, auditable trading rules for compliance or investor reporting
- For exploratory analysis to discover which technical indicators matter most and at what thresholds
- As a feature selection tool to identify which indicators carry information about future returns
- When the trading strategy involves known conditional logic (e.g., "only buy in uptrends when oversold")
- As building blocks for ensemble methods like Random Forest or Gradient Boosted Trees
- For initial prototyping of rule-based trading systems before production optimization

## When NOT to Use in Stock Market

- When predictive accuracy is the primary goal (use ensembles instead)
- For high-frequency trading where model stability is critical across small data changes
- When the underlying relationships are primarily linear (Logistic Regression will be simpler and more stable)
- For time-series forecasting where temporal order matters
- When the dataset is very small (fewer than 200 trading days), as the tree will overfit
- For strategies requiring smooth position sizing (the tree only outputs discrete classes)

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|---------------------------|
| max_depth | Maximum tree depth | 2 - 20 | 3-5 for interpretable rules; 6-10 for accuracy |
| min_samples_split | Minimum samples to split a node | 2 - 100 | 20-50 (at least 1 month of data) |
| min_samples_leaf | Minimum samples in a leaf | 1 - 50 | 10-30 to avoid memorizing noise |
| max_features | Number of features to consider per split | sqrt(n), log2(n), all | sqrt(n) to reduce overfitting |
| criterion | Split quality measure | gini, entropy | gini is faster; entropy may find slightly different rules |
| ccp_alpha | Cost-complexity pruning parameter | 0.0 - 0.05 | 0.005-0.02 for noisy stock data |
| class_weight | Handle imbalanced Buy/Sell | balanced, custom | Use "balanced" if signal distribution is skewed |
| max_leaf_nodes | Maximum number of terminal nodes | 5 - 100 | 8-20 for a manageable rule set |

## Stock Market Performance Tips

1. **Limit Tree Depth for Robustness**: In stock markets, deeper trees almost always overfit. A depth of 3-5 captures meaningful patterns while avoiding noise. Each additional level exponentially increases overfitting risk on financial data.

2. **Use Large Minimum Leaf Sizes**: Require at least 20-30 samples per leaf (approximately 1-2 months of trading days). This ensures each trading rule is based on a statistically meaningful number of observations.

3. **Extract and Validate Rules Manually**: After training, extract the decision rules and verify them against known trading principles. If the tree learns "Sell when PE < 5," investigate whether this makes economic sense.

4. **Walk-Forward Validation with Rule Stability Analysis**: Not only measure accuracy across time periods, but also check whether the tree discovers similar rules in each training window. Stable rules are more likely to be genuine patterns.

5. **Use Cost-Complexity Pruning**: Post-pruning with cross-validation selects the simplest tree that explains the data well, automatically finding the right balance between complexity and generalization.

6. **Combine with Domain Constraints**: Impose maximum depth and minimum samples constraints based on trading domain knowledge, not just cross-validation metrics.

## Comparison with Other Algorithms

| Criteria | Decision Tree | Logistic Regression | Random Forest | XGBoost | SVM |
|----------|--------------|-------------------|---------------|---------|-----|
| Interpretability | Excellent (rules) | Excellent (coefficients) | Low | Low | Low |
| Handles Nonlinearity | Good | Poor | Excellent | Excellent | Good (kernel) |
| Overfitting Risk | Very High | Low | Low | Moderate | Moderate |
| Stability | Very Low | High | High | High | High |
| Feature Interactions | Automatic | Manual | Automatic | Automatic | Kernel-based |
| Training Speed | Very Fast | Very Fast | Moderate | Moderate | Slow |
| Feature Scaling Needed | No | Yes | No | No | Yes |
| Best For (Stock Market) | Rule generation | Baseline signals | Robust prediction | Max accuracy | Market regimes |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Decision Tree from Scratch for Trading Rule Generation
# ============================================================

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 value=None, n_samples=None, impurity=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # Class distribution [n_sell, n_buy]
        self.n_samples = n_samples
        self.impurity = impurity


class StockDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=10,
                 max_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.root = None
        self.feature_importances_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 1 - p**2 - (1-p)**2

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        p = np.mean(y)
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

    def _impurity(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        return self._entropy(y)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        parent_impurity = self._impurity(y)

        # Select features to consider
        if self.max_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)

        for feat_idx in feature_indices:
            thresholds = np.unique(X[:, feat_idx])
            # Sample thresholds if too many unique values
            if len(thresholds) > 50:
                thresholds = np.percentile(X[:, feat_idx], np.linspace(5, 95, 50))

            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                gain = parent_impurity - (n_left/n_samples * left_impurity +
                                          n_right/n_samples * right_impurity)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_buy = np.sum(y == 1)
        n_sell = np.sum(y == 0)

        node = Node(
            value=np.array([n_sell, n_buy]),
            n_samples=n_samples,
            impurity=self._impurity(y)
        )

        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return node

        feat_idx, threshold, gain = self._best_split(X, y)

        if feat_idx is None or gain <= 0:
            return node

        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        node.feature_idx = feat_idx
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y, feature_names=None):
        self.n_features = X.shape[1]
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(self.n_features)]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._build_tree(X, y)
        self._compute_feature_importance(self.root)
        # Normalize
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _compute_feature_importance(self, node):
        if node.feature_idx is not None:
            # Weighted impurity reduction
            n = node.n_samples
            left_n = node.left.n_samples
            right_n = node.right.n_samples
            importance = (n * node.impurity -
                         left_n * node.left.impurity -
                         right_n * node.right.impurity)
            self.feature_importances_[node.feature_idx] += importance
            self._compute_feature_importance(node.left)
            self._compute_feature_importance(node.right)

    def _predict_single(self, x, node):
        if node.feature_idx is None:
            # Leaf node: return majority class
            return np.argmax(node.value)
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def predict_proba(self, X):
        probas = []
        for x in X:
            node = self.root
            while node.feature_idx is not None:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            total = node.value.sum()
            probas.append(node.value[1] / total)  # P(Buy)
        return np.array(probas)

    def print_rules(self, node=None, depth=0, prefix=""):
        if node is None:
            node = self.root

        indent = "  " * depth

        if node.feature_idx is None:
            # Leaf node
            total = node.value.sum()
            buy_pct = node.value[1] / total * 100
            action = "BUY" if node.value[1] > node.value[0] else "SELL"
            print(f"{indent}{prefix}=> {action} (confidence: {buy_pct:.1f}% buy, "
                  f"n={node.n_samples})")
            return

        feat_name = self.feature_names[node.feature_idx]
        print(f"{indent}{prefix}[{feat_name} <= {node.threshold:.2f}?]")
        self.print_rules(node.left, depth + 1, "YES -> ")
        self.print_rules(node.right, depth + 1, "NO  -> ")


# ============================================================
# Generate Synthetic Stock Market Data
# ============================================================
np.random.seed(42)
n_days = 1500

# Technical indicators
rsi = np.random.uniform(12, 88, n_days)
macd_hist = np.random.normal(0, 0.8, n_days)
price_sma20 = np.random.normal(0, 3.5, n_days)
price_sma50 = np.random.normal(0, 5.5, n_days)
volume_spike = np.random.lognormal(0, 0.4, n_days)
prev_5d_return = np.random.normal(0.1, 2.5, n_days)
volatility = np.random.uniform(0.8, 5, n_days)
bb_width = np.random.uniform(2, 12, n_days)
atr_pct = np.random.uniform(0.8, 3.5, n_days)
stoch_k = np.random.uniform(5, 95, n_days)
adx = np.random.uniform(10, 60, n_days)
market_trend = np.random.binomial(1, 0.6, n_days).astype(float)

X = np.column_stack([rsi, macd_hist, price_sma20, price_sma50, volume_spike,
                     prev_5d_return, volatility, bb_width, atr_pct, stoch_k,
                     adx, market_trend])

feature_names = ['RSI_14', 'MACD_Histogram', 'Price_vs_SMA20', 'Price_vs_SMA50',
                 'Volume_Spike', 'Prev_5D_Return', 'Volatility_10', 'BB_Width',
                 'ATR_Pct', 'Stochastic_K', 'ADX', 'Market_Trend']

# Generate labels using rule-based logic
y = np.zeros(n_days, dtype=int)
for i in range(n_days):
    if rsi[i] < 30 and macd_hist[i] > 0.2:
        y[i] = 1
    elif rsi[i] > 70 and macd_hist[i] < -0.2:
        y[i] = 0
    elif market_trend[i] == 1 and price_sma20[i] > 1.0 and adx[i] > 25:
        y[i] = 1
    elif volatility[i] > 4.0 and prev_5d_return[i] < -2:
        y[i] = 0
    else:
        y[i] = np.random.binomial(1, 0.48)

noise_idx = np.random.choice(n_days, size=int(0.08 * n_days), replace=False)
y[noise_idx] = 1 - y[noise_idx]

print(f"Dataset: {n_days} trading days, {len(feature_names)} features")
print(f"Buy: {y.sum()} ({y.mean()*100:.1f}%), Sell: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

# ============================================================
# Walk-Forward Split
# ============================================================
train_size = 1100
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nTrain: {train_size} days | Test: {n_days - train_size} days")

# ============================================================
# Train Decision Tree
# ============================================================
tree = StockDecisionTree(max_depth=4, min_samples_split=30, min_samples_leaf=15)
tree.fit(X_train, y_train, feature_names)

# ============================================================
# Print Discovered Trading Rules
# ============================================================
print("\n=== DISCOVERED TRADING RULES ===")
tree.print_rules()

# ============================================================
# Evaluate
# ============================================================
y_pred = tree.predict(X_test)
y_proba = tree.predict_proba(X_test)

accuracy = np.mean(y_pred == y_test)
buy_mask = y_pred == 1
precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)

print(f"\n=== TEST SET PERFORMANCE ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ============================================================
# Feature Importance
# ============================================================
print(f"\n=== FEATURE IMPORTANCE (Trading Signal Relevance) ===")
importance_order = np.argsort(tree.feature_importances_)[::-1]
for idx in importance_order:
    bar = "#" * int(tree.feature_importances_[idx] * 50)
    print(f"  {feature_names[idx]:20s}: {tree.feature_importances_[idx]:.4f} {bar}")

# ============================================================
# Simulated Trading
# ============================================================
print(f"\n=== SIMULATED TRADING RESULTS ===")
test_returns = prev_5d_return[train_size:] / 5  # approximate daily returns
strategy_returns = y_pred * test_returns / 100
buyhold_returns = test_returns / 100

cumulative_strategy = np.cumsum(strategy_returns)
cumulative_buyhold = np.cumsum(buyhold_returns)

print(f"Strategy Return:    {cumulative_strategy[-1]*100:.2f}%")
print(f"Buy & Hold Return:  {cumulative_buyhold[-1]*100:.2f}%")
print(f"Days in Market:     {y_pred.sum()} / {len(y_pred)}")
print(f"Number of Trades:   {np.sum(np.abs(np.diff(y_pred)))}")

# Show sample predictions
print(f"\n=== SAMPLE PREDICTIONS ===")
print(f"{'Day':>5} {'P(Buy)':>8} {'Signal':>8} {'Actual':>8} {'Correct':>8}")
for i in range(15):
    signal = "BUY" if y_pred[i] == 1 else "SELL"
    actual = "BUY" if y_test[i] == 1 else "SELL"
    correct = "Yes" if y_pred[i] == y_test[i] else "No"
    print(f"{i+1:>5} {y_proba[i]:>8.3f} {signal:>8} {actual:>8} {correct:>8}")
```

## Key Takeaways

- Decision Trees automatically discover threshold-based trading rules from historical data, producing transparent "if-then" strategies that can be audited and explained
- The tree structure naturally handles nonlinear relationships and feature interactions without manual feature engineering
- Single Decision Trees are highly prone to overfitting on noisy stock data; always use aggressive pruning (max_depth=3-5, min_samples_leaf=15+)
- Trees do not require feature scaling, making them easy to use with mixed technical indicators at different scales
- Feature importance scores reveal which indicators carry the most information about future price movements
- Stability is a major concern: validate that the tree discovers similar rules across different training windows
- Decision Trees serve as excellent building blocks for Random Forests and Gradient Boosting ensembles, which address the overfitting problem
- Walk-forward validation is essential to ensure discovered rules generalize to unseen market conditions
