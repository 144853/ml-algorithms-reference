# LightGBM - Complete Guide with Stock Market Applications

## Overview

LightGBM (Light Gradient Boosting Machine), developed by Microsoft Research in 2017, is a gradient boosting framework that uses tree-based learning algorithms optimized for speed and efficiency on large-scale datasets. LightGBM introduced two key innovations: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), which allow it to train significantly faster than XGBoost while achieving comparable or better accuracy. In the stock market context, LightGBM excels at large-scale market regime classification where millions of data points across thousands of stocks need to be processed efficiently.

The algorithm's leaf-wise tree growth strategy (as opposed to XGBoost's level-wise approach) builds deeper, more focused trees by always splitting the leaf with the highest loss reduction. This results in lower loss with fewer splits, making it particularly effective when dealing with complex market regime transitions that require deep, specialized decision paths. For classifying whether the market is in a "Bull," "Bear," "Sideways," or "High Volatility" regime, LightGBM's ability to model subtle distinctions in feature patterns across regimes is invaluable.

LightGBM has become the preferred gradient boosting tool in many quantitative finance teams due to its ability to handle datasets with millions of rows and hundreds of features in minutes rather than hours. Its native support for categorical features, built-in cross-validation with early stopping, and memory-efficient histogram-based algorithms make it practical for production trading systems that require frequent model retraining.

## How It Works - The Math Behind It

### Histogram-Based Algorithm

Instead of sorting features and scanning all split points (like XGBoost's exact algorithm), LightGBM discretizes continuous features into histogram bins:

```
For each feature:
    1. Divide feature values into k bins (default k=255)
    2. Accumulate gradient statistics per bin
    3. Find best split by scanning k bins instead of n data points

Complexity: O(k) per feature instead of O(n * log(n))
```

This is dramatically faster when n >> k, which is always the case for large stock datasets.

### Gradient-based One-Side Sampling (GOSS)

GOSS keeps all instances with large gradients (poorly predicted, most informative) and randomly samples from instances with small gradients:

```
1. Sort instances by absolute gradient |g_i|
2. Keep top a% instances (large gradients)
3. Randomly sample b% from remaining instances
4. Amplify sampled small-gradient instances by factor (1-a)/b
```

For market regime classification: data points near regime transitions (high gradient) are always kept, while data points in the middle of a clear regime (low gradient, already well-predicted) are downsampled.

### Exclusive Feature Bundling (EFB)

Many features are mutually exclusive (rarely non-zero simultaneously). EFB bundles such features to reduce dimensionality:

```
1. Construct a feature conflict graph
2. Features that are rarely non-zero simultaneously are bundled
3. Original features are reconstructed from bundles during split finding
```

In stock data: if you have indicator flags like "RSI_oversold," "RSI_overbought," "RSI_neutral," these are mutually exclusive and can be bundled.

### Leaf-Wise Tree Growth

Unlike XGBoost's level-wise (breadth-first) growth:

```
Level-wise (XGBoost):
    Split ALL leaves at each level before going deeper
    Produces balanced trees

Leaf-wise (LightGBM):
    Always split the leaf with MAXIMUM loss reduction
    Produces unbalanced but more accurate trees
    Risk: can overfit with too many leaves
```

For regime classification: leaf-wise growth can create deep paths for rare but important regimes (e.g., crash conditions) while keeping simple paths for common regimes.

### Optimal Split Finding

Same as gradient boosting, using first and second order gradients:
```
Gain = 0.5 * [G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
```

But computed over histogram bins rather than exact values.

### Step-by-Step Process

1. **Bin continuous features** into 255 discrete bins using quantile binning
2. **Initialize predictions** to the base rate (e.g., most common regime)
3. **For each boosting round**:
   a. Compute gradients and Hessians for all samples
   b. Apply GOSS: keep top-gradient samples, subsample rest
   c. Apply EFB: bundle mutually exclusive features
   d. Build tree leaf-wise: always split the leaf with highest gain
   e. Update predictions with learning rate shrinkage
4. **Early stopping** when validation metric stops improving

## Stock Market Use Case: Large-Scale Market Regime Classification

### The Problem

You manage a multi-strategy hedge fund that allocates capital among different strategies based on the current market regime. You need to classify each stock-day observation into one of four market regimes: "Bull" (strong uptrend), "Bear" (strong downtrend), "Sideways" (range-bound), or "Volatile" (high uncertainty, large swings). You have 10 years of daily data across 2,000 stocks, resulting in over 5 million observations with 20 features each. The model must be retrained weekly, so training speed is critical. Different strategies are deployed depending on the regime: momentum strategies in Bull regimes, mean-reversion in Sideways, defensive positioning in Bear, and options-based strategies in Volatile regimes.

### Stock Market Features (Input Data)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| Return_5D | 5-day rolling return (%) | Numerical | -20% to 20% |
| Return_20D | 20-day rolling return (%) | Numerical | -40% to 40% |
| Return_60D | 60-day rolling return (%) | Numerical | -60% to 60% |
| Volatility_10D | 10-day realized vol (annualized %) | Numerical | 5% to 100% |
| Volatility_60D | 60-day realized vol (annualized %) | Numerical | 5% to 80% |
| Vol_Regime_Ratio | Vol_10D / Vol_60D (vol expansion/contraction) | Numerical | 0.3 to 3.0 |
| Trend_Strength | ADX value | Numerical | 5 to 80 |
| SMA_20_50_Cross | SMA20/SMA50 ratio | Numerical | 0.85 to 1.15 |
| SMA_50_200_Cross | SMA50/SMA200 ratio | Numerical | 0.80 to 1.20 |
| RSI_14 | Relative Strength Index | Numerical | 5 to 95 |
| MACD_Histogram | MACD histogram | Numerical | -5 to 5 |
| BB_Width | Bollinger Band width (%) | Numerical | 1% to 20% |
| Volume_Trend | 5-day avg vol / 20-day avg vol | Numerical | 0.3 to 3.0 |
| Drawdown_20D | Max drawdown in last 20 days (%) | Numerical | -30% to 0% |
| Up_Days_Pct_20 | % of up days in last 20 days | Numerical | 0% to 100% |
| Skewness_20D | 20-day return skewness | Numerical | -3 to 3 |
| Kurtosis_20D | 20-day return kurtosis | Numerical | 1 to 15 |
| Market_Beta | Rolling 60-day beta to S&P 500 | Numerical | -1 to 3 |
| Sector_Rel_Strength | Stock sector vs market return | Numerical | -10% to 10% |
| VIX_Level | Current VIX level | Numerical | 10 to 80 |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_samples = 100000  # Large-scale dataset

# Generate features
features = {
    'Return_5D': np.random.normal(0.2, 3, n_samples),
    'Return_20D': np.random.normal(0.8, 6, n_samples),
    'Return_60D': np.random.normal(2.0, 12, n_samples),
    'Volatility_10D': np.random.lognormal(2.8, 0.5, n_samples),
    'Volatility_60D': np.random.lognormal(2.7, 0.4, n_samples),
    'Vol_Regime_Ratio': np.random.lognormal(0, 0.3, n_samples),
    'Trend_Strength': np.random.uniform(8, 65, n_samples),
    'SMA_20_50': np.random.normal(1.0, 0.04, n_samples),
    'SMA_50_200': np.random.normal(1.0, 0.06, n_samples),
    'RSI_14': np.random.uniform(10, 90, n_samples),
    'MACD_Hist': np.random.normal(0, 1.2, n_samples),
    'BB_Width': np.random.lognormal(1.5, 0.5, n_samples),
    'Volume_Trend': np.random.lognormal(0, 0.3, n_samples),
    'Drawdown_20D': -np.random.exponential(3, n_samples),
    'Up_Days_Pct': np.random.uniform(20, 80, n_samples),
    'Skewness_20D': np.random.normal(0, 0.8, n_samples),
    'Kurtosis_20D': np.random.exponential(2, n_samples) + 1,
    'Market_Beta': np.random.normal(1.0, 0.4, n_samples),
    'Sector_Rel': np.random.normal(0, 3, n_samples),
    'VIX_Level': np.random.lognormal(2.8, 0.4, n_samples),
}

X = np.column_stack(list(features.values()))
feature_names = list(features.keys())

# Generate 4-class regime labels
# 0=Bull, 1=Bear, 2=Sideways, 3=Volatile
y = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    r20 = features['Return_20D'][i]
    vol = features['Volatility_10D'][i]
    adx = features['Trend_Strength'][i]

    if r20 > 4 and adx > 25:
        y[i] = 0  # Bull
    elif r20 < -4 and adx > 25:
        y[i] = 1  # Bear
    elif vol > 30 or features['VIX_Level'][i] > 30:
        y[i] = 3  # Volatile
    else:
        y[i] = 2  # Sideways

# Add noise
noise = np.random.choice(n_samples, size=int(0.12 * n_samples), replace=False)
y[noise] = np.random.randint(0, 4, len(noise))

regime_names = ['Bull', 'Bear', 'Sideways', 'Volatile']
for r in range(4):
    print(f"  {regime_names[r]}: {np.sum(y==r)} ({np.mean(y==r)*100:.1f}%)")
```

### The Model in Action

```python
# LightGBM classifies market regimes with high granularity:
#
# Day 1: Return_20D=+8.2%, ADX=42, Vol_10D=14%, VIX=15
#   -> Leaf-wise tree path: Strong return + High trend + Low vol
#   -> Classification: BULL (probability: [0.82, 0.03, 0.08, 0.07])
#   -> Action: Deploy momentum strategy, increase equity exposure
#
# Day 2: Return_20D=-12.5%, ADX=55, Vol_10D=35%, VIX=42
#   -> Path: Large negative return + High trend + High vol + High VIX
#   -> Classification: BEAR (probability: [0.02, 0.78, 0.05, 0.15])
#   -> Action: Deploy defensive strategy, reduce equity, hedge with puts
#
# Day 3: Return_20D=+1.2%, ADX=12, Vol_10D=45%, VIX=35
#   -> Path: Small return + Low trend + High vol
#   -> Classification: VOLATILE (probability: [0.08, 0.10, 0.12, 0.70])
#   -> Action: Deploy options strategy, reduce directional exposure
```

## Advantages

1. **Dramatically Faster Training Than XGBoost**: Histogram binning, GOSS, and EFB make LightGBM 5-20x faster than XGBoost on large datasets. For a 5-million-row market regime dataset, training completes in minutes instead of hours, enabling weekly retraining.

2. **Memory Efficient for Large-Scale Financial Data**: The histogram-based algorithm uses O(n_bins * n_features) memory instead of O(n_data * n_features), making it feasible to process decades of multi-stock data on a single machine.

3. **Leaf-Wise Growth Captures Complex Regime Patterns**: By always splitting the most promising leaf, LightGBM can develop deep, specialized paths for rare but important regimes (e.g., market crashes) without wasting capacity on already well-classified common regimes.

4. **Native Categorical Feature Support**: LightGBM can handle categorical features (like sector, exchange, market cap category) without one-hot encoding, finding optimal splits for categorical variables directly. This preserves information and reduces dimensionality.

5. **GOSS Preserves Hard-to-Classify Boundary Cases**: In regime classification, the most important data points are those near regime transitions (e.g., is the market shifting from Bull to Volatile?). GOSS automatically preserves these high-gradient boundary cases while downsampling easy interior cases.

6. **Excellent Multi-Class Support**: LightGBM natively handles multi-class classification (Bull/Bear/Sideways/Volatile) without requiring one-vs-rest decomposition, simplifying the pipeline and improving performance.

7. **Built-in Cross-Validation and Early Stopping**: The framework provides integrated CV with early stopping, making it straightforward to find the optimal number of boosting rounds without manual implementation.

## Disadvantages

1. **Leaf-Wise Growth Can Overfit on Small Datasets**: The aggressive leaf-wise strategy can produce very deep, unbalanced trees that overfit when data is limited. For individual stock analysis with fewer than 5,000 observations, max_depth must be carefully limited.

2. **Sensitive to num_leaves Hyperparameter**: Unlike max_depth which is intuitive, LightGBM's primary complexity control is num_leaves. Setting it too high causes overfitting; too low limits model capacity. Finding the right value requires careful tuning.

3. **Histogram Approximation Loses Precision**: Binning continuous features into 255 bins loses information about exact values. For features with important precise thresholds (e.g., RSI = 30 vs. RSI = 32), the binning may miss the optimal split point.

4. **GOSS Introduces Stochasticity**: The random sampling of low-gradient instances means results vary between runs (even with the same hyperparameters). This can make model comparison and reproducibility challenging in systematic trading research.

5. **Less Interpretable Than XGBoost for Simple Models**: The unbalanced leaf-wise trees are harder to visualize and interpret than XGBoost's balanced level-wise trees. For regime classification, explaining why a specific day was classified as "Bear" requires SHAP analysis.

6. **Requires Careful Learning Rate / num_leaves Balance**: LightGBM is more sensitive to the interaction between learning rate and num_leaves than XGBoost is to learning rate and max_depth. This increases the hyperparameter search space.

## When to Use in Stock Market

- When the dataset is large (100,000+ observations) and training speed is critical for frequent retraining
- For multi-class regime classification where native multi-class support simplifies the pipeline
- When features include categorical variables (sector, exchange, country) that benefit from native handling
- For production systems requiring weekly or daily model retraining on large cross-sectional stock data
- When memory efficiency is a constraint (processing data for thousands of stocks simultaneously)
- As a faster alternative to XGBoost with comparable accuracy

## When NOT to Use in Stock Market

- With very small datasets (fewer than 5,000 observations) where leaf-wise growth will overfit
- When exact feature thresholds matter (histogram binning may miss precise critical levels)
- When full reproducibility is required across runs (GOSS introduces randomness)
- For simple binary classification problems where XGBoost or Random Forest suffice
- When maximum interpretability is needed (balanced trees from XGBoost are easier to analyze)

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|---------------------------|
| num_leaves | Maximum leaves per tree | 15 - 256 | 31-63 for regimes; higher for large data |
| learning_rate | Shrinkage factor | 0.01 - 0.3 | 0.03-0.1 with early stopping |
| n_estimators | Number of boosting rounds | 100 - 5000 | 500-2000 with early stopping |
| max_depth | Tree depth limit (-1=unlimited) | -1, 3-15 | 6-10; use with num_leaves for control |
| min_child_samples | Min samples per leaf | 5 - 100 | 20-50 for noisy regime labels |
| subsample (bagging_fraction) | Row sampling rate | 0.5 - 1.0 | 0.7-0.9 |
| colsample_bytree | Feature sampling rate | 0.5 - 1.0 | 0.6-0.8 |
| reg_alpha | L1 regularization | 0 - 10 | 0.1-1.0 for feature selection |
| reg_lambda | L2 regularization | 0 - 10 | 1-5 for regime stability |
| max_bin | Histogram bins per feature | 63 - 512 | 255 default; 127 for faster training |
| min_gain_to_split | Min gain for split | 0 - 1.0 | 0.01-0.1 to prevent noise splits |
| class_weight | Multi-class weighting | balanced | "balanced" for skewed regime distribution |

## Stock Market Performance Tips

1. **Set num_leaves = 2^(max_depth) - 1**: This aligns leaf-wise and level-wise growth. For max_depth=6, use num_leaves=63. Going higher than this risks overfitting.

2. **Use Feature Fraction with Regime Classification**: Set colsample_bytree=0.6-0.7 to prevent the model from over-relying on a single dominant indicator (like VIX for Volatile regime detection).

3. **Apply Class Weights for Rare Regimes**: Bear and Volatile regimes are typically less frequent than Bull and Sideways. Use class_weight="balanced" or custom weights to ensure the model learns rare regime patterns.

4. **Monitor Per-Class Metrics**: Overall accuracy can be misleading if one regime dominates. Track precision, recall, and F1 for each regime separately, especially Bear and Volatile which have the highest portfolio impact.

5. **Use Regime Transition Features**: Engineer features that capture regime transitions (e.g., "days since last regime change," "regime duration") to help the model identify turning points.

6. **Retrain Weekly with Rolling Window**: Market regimes shift over months. Use a rolling 3-5 year training window, retrained weekly, to keep the model adapted to current conditions.

## Comparison with Other Algorithms

| Criteria | LightGBM | XGBoost | Random Forest | CatBoost | SVM |
|----------|----------|---------|--------------|----------|-----|
| Training Speed (large data) | Very Fast | Moderate | Moderate | Slow | Very Slow |
| Memory Efficiency | Excellent | Good | Poor | Good | Poor |
| Multi-Class Support | Native | Native | Native | Native | OvR needed |
| Categorical Features | Native | Encoding needed | Encoding needed | Native | Encoding needed |
| Accuracy | Very High | Very High | High | Very High | Moderate |
| Overfitting Control | Good (with tuning) | Very Good | Excellent | Very Good | Good |
| Tree Growth | Leaf-wise | Level-wise | Independent | Symmetric | N/A |
| Best For (Stock Market) | Large-scale regimes | HFT signals | Robust baseline | Mixed feature types | Small data regimes |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Simplified LightGBM-style Classifier for Market Regime Detection
# ============================================================

class HistogramBin:
    """Bins continuous features into discrete histograms."""

    def __init__(self, max_bins=64):
        self.max_bins = max_bins
        self.bin_edges = None

    def fit(self, X):
        n_features = X.shape[1]
        self.bin_edges = []
        for f in range(n_features):
            percentiles = np.linspace(0, 100, self.max_bins + 1)
            edges = np.percentile(X[:, f], percentiles)
            edges = np.unique(edges)
            self.bin_edges.append(edges)
        return self

    def transform(self, X):
        X_binned = np.zeros_like(X, dtype=int)
        for f in range(X.shape[1]):
            X_binned[:, f] = np.digitize(X[:, f], self.bin_edges[f]) - 1
            X_binned[:, f] = np.clip(X_binned[:, f], 0, len(self.bin_edges[f]) - 2)
        return X_binned


class LGBMTree:
    """Leaf-wise tree for gradient boosting (simplified LightGBM tree)."""

    def __init__(self, num_leaves=31, min_child_samples=20, lambda_reg=1.0,
                 colsample=0.7):
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.lambda_reg = lambda_reg
        self.colsample = colsample
        self.tree = None

    def _leaf_weight(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.lambda_reg)

    def _split_gain(self, g_left, h_left, g_right, h_right):
        gl, hl = np.sum(g_left), np.sum(h_left)
        gr, hr = np.sum(g_right), np.sum(h_right)
        return 0.5 * (gl**2 / (hl + self.lambda_reg) +
                       gr**2 / (hr + self.lambda_reg) -
                       (gl + gr)**2 / (hl + hr + self.lambda_reg))

    def _find_best_split(self, X, g, h, indices, feat_subset):
        best_gain = 0
        best_feat = None
        best_thresh = None

        for f in feat_subset:
            unique_vals = np.unique(X[indices, f])
            for thresh in unique_vals[:-1]:
                left = indices[X[indices, f] <= thresh]
                right = indices[X[indices, f] > thresh]

                if len(left) < self.min_child_samples or len(right) < self.min_child_samples:
                    continue

                gain = self._split_gain(g[left], h[left], g[right], h[right])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def fit(self, X, g, h):
        n_features = X.shape[1]
        n_selected = max(1, int(n_features * self.colsample))

        # Leaf-wise growth: maintain a list of leaves to potentially split
        all_indices = np.arange(len(g))
        feat_subset = np.random.choice(n_features, n_selected, replace=False)

        root = {
            'leaf': True,
            'indices': all_indices,
            'weight': self._leaf_weight(g[all_indices], h[all_indices]),
        }
        leaves = [root]
        n_leaves = 1

        while n_leaves < self.num_leaves:
            # Find leaf with best potential split
            best_leaf_gain = 0
            best_leaf_idx = -1
            best_split_info = None

            for li, leaf in enumerate(leaves):
                if not leaf['leaf'] or len(leaf['indices']) < 2 * self.min_child_samples:
                    continue

                feat, thresh, gain = self._find_best_split(
                    X, g, h, leaf['indices'], feat_subset)

                if gain > best_leaf_gain:
                    best_leaf_gain = gain
                    best_leaf_idx = li
                    best_split_info = (feat, thresh)

            if best_leaf_idx == -1:
                break

            # Split the best leaf
            leaf = leaves[best_leaf_idx]
            feat, thresh = best_split_info
            idx = leaf['indices']
            left_idx = idx[X[idx, feat] <= thresh]
            right_idx = idx[X[idx, feat] > thresh]

            leaf['leaf'] = False
            leaf['feature'] = feat
            leaf['threshold'] = thresh

            left_node = {
                'leaf': True,
                'indices': left_idx,
                'weight': self._leaf_weight(g[left_idx], h[left_idx]),
            }
            right_node = {
                'leaf': True,
                'indices': right_idx,
                'weight': self._leaf_weight(g[right_idx], h[right_idx]),
            }

            leaf['left'] = left_node
            leaf['right'] = right_node
            leaves.append(left_node)
            leaves.append(right_node)
            n_leaves += 1

        self.tree = root
        return self

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['weight']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])


class StockLightGBM:
    """Simplified LightGBM for multi-class market regime classification."""

    def __init__(self, n_estimators=200, learning_rate=0.05, num_leaves=31,
                 min_child_samples=20, lambda_reg=1.0, subsample=0.8,
                 colsample=0.7, max_bins=64):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.lambda_reg = lambda_reg
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.trees = []
        self.n_classes = None
        self.binner = None

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=30):
        n_samples = len(y)
        self.n_classes = len(np.unique(y))

        # Bin features for speed
        self.binner = HistogramBin(self.max_bins)
        self.binner.fit(X)
        X_binned = self.binner.transform(X)
        if X_val is not None:
            X_val_binned = self.binner.transform(X_val)

        # One-hot encode labels
        Y_onehot = np.eye(self.n_classes)[y]

        # Initialize raw predictions
        raw_pred = np.zeros((n_samples, self.n_classes))
        if X_val is not None:
            raw_pred_val = np.zeros((len(y_val), self.n_classes))
            best_val_acc = 0
            best_round = 0

        self.trees = []

        for t in range(self.n_estimators):
            probs = self._softmax(raw_pred)
            round_trees = []

            # Subsample
            n_sub = int(n_samples * self.subsample)
            sub_idx = np.random.choice(n_samples, n_sub, replace=False)

            for c in range(self.n_classes):
                # Gradients for class c (softmax cross-entropy)
                g = probs[:, c] - Y_onehot[:, c]
                h = probs[:, c] * (1 - probs[:, c])
                h = np.maximum(h, 1e-6)

                tree = LGBMTree(
                    num_leaves=self.num_leaves,
                    min_child_samples=self.min_child_samples,
                    lambda_reg=self.lambda_reg,
                    colsample=self.colsample
                )
                tree.fit(X_binned[sub_idx], g[sub_idx], h[sub_idx])
                round_trees.append(tree)

                raw_pred[:, c] += self.lr * tree.predict(X_binned)
                if X_val is not None:
                    raw_pred_val[:, c] += self.lr * tree.predict(X_val_binned)

            self.trees.append(round_trees)

            if X_val is not None and (t + 1) % 20 == 0:
                val_pred = np.argmax(self._softmax(raw_pred_val), axis=1)
                val_acc = np.mean(val_pred == y_val)
                train_pred = np.argmax(self._softmax(raw_pred), axis=1)
                train_acc = np.mean(train_pred == y)
                print(f"Round {t+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_round = t
                elif t - best_round >= early_stopping_rounds:
                    print(f"Early stopping at round {t+1}. Best: {best_round+1}")
                    self.trees = self.trees[:best_round + 1]
                    break

        return self

    def predict_proba(self, X):
        X_binned = self.binner.transform(X)
        raw_pred = np.zeros((len(X), self.n_classes))
        for round_trees in self.trees:
            for c, tree in enumerate(round_trees):
                raw_pred[:, c] += self.lr * tree.predict(X_binned)
        return self._softmax(raw_pred)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# Generate Large-Scale Market Regime Data
# ============================================================
np.random.seed(42)
n_total = 30000

return_5d = np.random.normal(0.2, 3, n_total)
return_20d = np.random.normal(0.8, 6, n_total)
return_60d = np.random.normal(2, 12, n_total)
vol_10d = np.random.lognormal(2.7, 0.5, n_total)
vol_60d = np.random.lognormal(2.6, 0.4, n_total)
vol_ratio = vol_10d / vol_60d
adx = np.random.uniform(8, 65, n_total)
sma_20_50 = np.random.normal(1.0, 0.04, n_total)
sma_50_200 = np.random.normal(1.0, 0.06, n_total)
rsi = np.random.uniform(10, 90, n_total)
macd = np.random.normal(0, 1.2, n_total)
bb_width = np.random.lognormal(1.5, 0.5, n_total)
vol_trend = np.random.lognormal(0, 0.3, n_total)
drawdown = -np.random.exponential(3, n_total)
up_pct = np.random.uniform(20, 80, n_total)
skew = np.random.normal(0, 0.8, n_total)
kurt = np.random.exponential(2, n_total) + 1
beta = np.random.normal(1.0, 0.4, n_total)
sector_rel = np.random.normal(0, 3, n_total)
vix = np.random.lognormal(2.8, 0.4, n_total)

X = np.column_stack([return_5d, return_20d, return_60d, vol_10d, vol_60d,
                     vol_ratio, adx, sma_20_50, sma_50_200, rsi,
                     macd, bb_width, vol_trend, drawdown, up_pct,
                     skew, kurt, beta, sector_rel, vix])

feature_names = ['Return_5D', 'Return_20D', 'Return_60D', 'Vol_10D', 'Vol_60D',
                 'Vol_Ratio', 'ADX', 'SMA_20_50', 'SMA_50_200', 'RSI',
                 'MACD', 'BB_Width', 'Vol_Trend', 'Drawdown', 'Up_Pct',
                 'Skewness', 'Kurtosis', 'Beta', 'Sector_Rel', 'VIX']

# 4-class regime labels
y = np.zeros(n_total, dtype=int)
for i in range(n_total):
    if return_20d[i] > 5 and adx[i] > 25 and sma_20_50[i] > 1.01:
        y[i] = 0  # Bull
    elif return_20d[i] < -5 and adx[i] > 25 and sma_20_50[i] < 0.99:
        y[i] = 1  # Bear
    elif vol_10d[i] > 30 or vix[i] > 30 or vol_ratio[i] > 1.5:
        y[i] = 3  # Volatile
    else:
        y[i] = 2  # Sideways

noise = np.random.choice(n_total, int(0.1 * n_total), replace=False)
y[noise] = np.random.randint(0, 4, len(noise))

regime_names = ['Bull', 'Bear', 'Sideways', 'Volatile']
print("Market Regime Distribution:")
for r in range(4):
    print(f"  {regime_names[r]:>10}: {np.sum(y==r):>6} ({np.mean(y==r)*100:.1f}%)")

# Temporal split
train_end = 22000
val_end = 26000
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"\nTrain: {train_end} | Val: {val_end-train_end} | Test: {n_total-val_end}")

# ============================================================
# Train LightGBM
# ============================================================
print("\n=== Training LightGBM ===")
lgbm = StockLightGBM(n_estimators=200, learning_rate=0.08, num_leaves=31,
                      min_child_samples=25, lambda_reg=2.0, subsample=0.8,
                      colsample=0.7, max_bins=64)
lgbm.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=30)

# ============================================================
# Evaluate
# ============================================================
y_pred = lgbm.predict(X_test)
y_proba = lgbm.predict_proba(X_test)

overall_acc = np.mean(y_pred == y_test)
print(f"\n=== TEST PERFORMANCE ===")
print(f"Overall Accuracy: {overall_acc:.4f}")

print(f"\nPer-Regime Performance:")
print(f"{'Regime':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
for r in range(4):
    tp = np.sum((y_pred == r) & (y_test == r))
    fp = np.sum((y_pred == r) & (y_test != r))
    fn = np.sum((y_pred != r) & (y_test == r))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    support = np.sum(y_test == r)
    print(f"{regime_names[r]:>12} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")

# Confusion matrix
print(f"\nConfusion Matrix:")
print(f"{'':>12} {'Pred Bull':>10} {'Pred Bear':>10} {'Pred Side':>10} {'Pred Vol':>10}")
for r in range(4):
    row = [np.sum((y_test == r) & (y_pred == c)) for c in range(4)]
    print(f"{regime_names[r]:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

# Strategy allocation based on regime
print(f"\n=== REGIME-BASED STRATEGY ALLOCATION ===")
strategy_map = {0: 'Momentum Long', 1: 'Defensive/Short', 2: 'Mean Reversion', 3: 'Options Straddle'}
for i in range(10):
    regime = regime_names[y_pred[i]]
    confidence = y_proba[i, y_pred[i]]
    strategy = strategy_map[y_pred[i]]
    actual = regime_names[y_test[i]]
    print(f"Day {i+1}: {regime} ({confidence:.1%} conf) -> {strategy} | Actual: {actual}")
```

## Key Takeaways

- LightGBM's histogram-based algorithm and GOSS make it 5-20x faster than XGBoost for large-scale market regime classification while maintaining comparable accuracy
- Leaf-wise tree growth creates specialized deep paths for rare but important regimes (Bear, Volatile) while keeping simple paths for common regimes
- Native categorical feature support eliminates the need for one-hot encoding of sector, exchange, and market cap category variables
- For multi-class regime classification, monitor per-class precision and recall separately, as overall accuracy can mask poor performance on critical rare regimes
- The num_leaves parameter is the primary complexity control; set it to approximately 2^(max_depth)-1 to balance model capacity and overfitting risk
- Weekly retraining with 3-5 year rolling windows keeps the model adapted to evolving market conditions
- GOSS naturally preserves data points near regime transitions (high gradient) which are the most informative for classification accuracy
- Use class weights to handle imbalanced regime distributions, ensuring the model learns Bear and Volatile patterns despite their lower frequency
