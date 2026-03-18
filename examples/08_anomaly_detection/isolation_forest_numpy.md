# Isolation Forest - Complete Guide with Stock Market Applications

## Overview

Isolation Forest is an unsupervised anomaly detection algorithm based on the principle that anomalies are few and different -- they are easier to isolate than normal observations. Unlike traditional density-based or distance-based methods that model normality and flag deviations, Isolation Forest directly targets anomalies by measuring how quickly a data point can be isolated through random partitioning. Points that require fewer random splits to isolate are more anomalous.

In stock market applications, Isolation Forest excels at detecting flash crashes, abnormal trading volumes, unusual price movements, market manipulation patterns, and other rare events that deviate from normal market behavior. Financial markets generate vast amounts of high-dimensional data where anomalies are inherently rare (<1% of observations) but critically important -- a flash crash lasting minutes can wipe out billions in market value, and detecting manipulation early can prevent regulatory penalties and protect investors.

The algorithm constructs an ensemble of random binary trees (isolation trees) by recursively splitting the data along randomly selected features at random split points. Anomalies, being isolated and distinct, tend to be separated near the root of the trees (short path lengths), while normal points buried in dense regions require many splits (long path lengths). The anomaly score is computed from the average path length across all trees, normalized by the expected path length for the dataset size, producing a score between 0 and 1 where values close to 1 indicate anomalies.

## How It Works - The Math Behind It

### Isolation Tree Construction

For a dataset `X` with `n` samples and `d` features:

1. Randomly select a feature `q` from `{1, ..., d}`
2. Randomly select a split value `p` between `min(X_q)` and `max(X_q)`
3. Partition data into left (`X_q < p`) and right (`X_q >= p`)
4. Recurse until: node has one sample, or tree reaches height limit `l = ceil(log2(n))`

```
IsolationTree(X, height, height_limit):
    if height >= height_limit or |X| <= 1:
        return ExternalNode(size=|X|)
    q = random_feature()
    p = random_split(min(X_q), max(X_q))
    X_left = {x in X : x_q < p}
    X_right = {x in X : x_q >= p}
    return InternalNode(
        left = IsolationTree(X_left, height+1, height_limit),
        right = IsolationTree(X_right, height+1, height_limit),
        split_feature = q,
        split_value = p
    )
```

### Path Length

The path length `h(x)` for a point `x` is the number of edges traversed from root to the external node containing `x`. For external nodes with size `> 1`, add an adjustment:

```
c(n) = 2 * H(n-1) - 2*(n-1)/n
```

where `H(i)` is the harmonic number: `H(i) = ln(i) + 0.5772156649` (Euler-Mascheroni constant).

This `c(n)` is the average path length of unsuccessful search in a Binary Search Tree, used to normalize the score.

### Anomaly Score

The anomaly score for a point `x` across `T` trees:

```
E[h(x)] = (1/T) * sum_{t=1}^{T} h_t(x)

s(x, n) = 2^(-E[h(x)] / c(n))
```

Score interpretation:
- `s(x) -> 1`: anomaly (short average path length)
- `s(x) -> 0.5`: normal (average path length matches expected)
- `s(x) -> 0`: very dense, deep in the tree

### Contamination and Threshold

Given an expected contamination ratio `epsilon`:

```
threshold = quantile(s(X), 1 - epsilon)
anomaly if s(x) > threshold
```

### Subsampling

Isolation Forest uses subsampling (`psi` samples per tree) to:
- Reduce computational cost: O(T * psi * log(psi)) vs O(T * n * log(n))
- Reduce swamping: prevents anomalies from being masked by nearby normal points
- Reduce masking: prevents clusters of anomalies from appearing normal

Typical `psi = 256` regardless of dataset size.

## Stock Market Use Case: Detecting Flash Crashes and Abnormal Trading Patterns

### The Problem

A stock exchange's surveillance team monitors real-time trading data across 3,000 listed stocks. They need to:
- Detect flash crashes (sudden, extreme price drops followed by recovery) within minutes
- Flag abnormal trading volume spikes that may indicate insider trading or market manipulation
- Identify unusual bid-ask spread widening that signals liquidity crises
- Detect spoofing patterns (large orders placed and quickly canceled)
- Operate in real-time with minimal false positives to avoid unnecessary trading halts

### Stock Market Features (Input Data)

| Feature Name | Description | Example Value | Anomaly Signal |
|---|---|---|---|
| return_1min | 1-minute log return | -0.085 | Flash crash |
| return_5min | 5-minute log return | -0.12 | Sustained drop |
| volume_zscore | Volume relative to 20-day avg (z-score) | 8.5 | Volume spike |
| spread_zscore | Bid-ask spread relative to avg | 4.2 | Liquidity dry-up |
| trade_count_ratio | Number of trades vs avg | 3.5 | Unusual activity |
| order_cancel_rate | Fraction of orders canceled in window | 0.85 | Spoofing |
| price_impact | Price move per unit volume | 0.008 | Low liquidity |
| depth_imbalance | (bid_depth - ask_depth) / total_depth | -0.65 | Orderbook imbalance |
| volatility_ratio | Current vol / 30-day avg vol | 5.2 | Volatility spike |
| cross_stock_corr | Correlation with sector index | 0.15 | Decoupled movement |
| large_trade_pct | % of volume from large trades | 0.70 | Block trading |
| momentum_break | Deviation from 20-period momentum | -3.5 | Trend reversal |

### Example Data Structure

```python
import numpy as np

# Real-time market surveillance data structure
np.random.seed(42)

n_observations = 50000   # 50K one-minute bars across many stocks
n_features = 12

feature_names = [
    'return_1min', 'return_5min', 'volume_zscore', 'spread_zscore',
    'trade_count_ratio', 'order_cancel_rate', 'price_impact',
    'depth_imbalance', 'volatility_ratio', 'cross_stock_corr',
    'large_trade_pct', 'momentum_break'
]

# Normal market data (99% of observations)
n_normal = int(n_observations * 0.99)
X_normal = np.column_stack([
    np.random.randn(n_normal) * 0.002,          # return_1min (small)
    np.random.randn(n_normal) * 0.004,          # return_5min
    np.random.randn(n_normal) * 1.0,            # volume_zscore (~N(0,1))
    np.abs(np.random.randn(n_normal)) * 0.5,    # spread_zscore (positive)
    1 + np.random.randn(n_normal) * 0.3,        # trade_count_ratio (~1)
    np.random.beta(2, 10, n_normal),             # order_cancel_rate (~0.17)
    np.abs(np.random.randn(n_normal)) * 0.001,  # price_impact (small)
    np.random.randn(n_normal) * 0.2,            # depth_imbalance
    1 + np.abs(np.random.randn(n_normal)) * 0.3, # volatility_ratio (~1)
    0.6 + np.random.randn(n_normal) * 0.15,     # cross_stock_corr (~0.6)
    np.random.beta(5, 5, n_normal),              # large_trade_pct (~0.5)
    np.random.randn(n_normal) * 0.5,            # momentum_break (~0)
])

# Anomalous events (1%): flash crashes, volume spikes, spoofing
n_anomaly = n_observations - n_normal

# Flash crash anomalies
n_flash = n_anomaly // 3
X_flash = np.column_stack([
    np.random.randn(n_flash) * 0.01 - 0.05,     # large negative return
    np.random.randn(n_flash) * 0.02 - 0.08,     # sustained drop
    np.abs(np.random.randn(n_flash)) * 3 + 5,   # massive volume
    np.abs(np.random.randn(n_flash)) * 2 + 3,   # wide spreads
    np.abs(np.random.randn(n_flash)) * 1 + 3,   # many trades
    np.random.beta(2, 3, n_flash),                # moderate cancellation
    np.abs(np.random.randn(n_flash)) * 0.005 + 0.005, # high impact
    np.random.randn(n_flash) * 0.3 - 0.4,       # sell-side imbalance
    np.abs(np.random.randn(n_flash)) * 2 + 4,   # high vol ratio
    np.random.randn(n_flash) * 0.3 + 0.2,       # low correlation
    np.random.beta(8, 2, n_flash),                # large trades dominate
    np.random.randn(n_flash) * 1.0 - 3,         # momentum broken
])

# Spoofing anomalies
n_spoof = n_anomaly // 3
X_spoof = np.column_stack([
    np.random.randn(n_spoof) * 0.003,            # normal returns
    np.random.randn(n_spoof) * 0.005,            # normal 5min returns
    np.abs(np.random.randn(n_spoof)) * 2 + 2,   # elevated volume
    np.abs(np.random.randn(n_spoof)) * 1 + 1,   # somewhat wide spreads
    np.abs(np.random.randn(n_spoof)) * 1 + 2,   # more trades
    np.random.beta(1, 1.5, n_spoof) * 0.3 + 0.7, # very high cancel rate!
    np.abs(np.random.randn(n_spoof)) * 0.002,    # low impact
    np.random.randn(n_spoof) * 0.4,              # alternating imbalance
    1 + np.abs(np.random.randn(n_spoof)) * 0.5,  # slightly elevated vol
    0.5 + np.random.randn(n_spoof) * 0.2,        # normal correlation
    np.random.beta(3, 7, n_spoof),                # small trades dominate
    np.random.randn(n_spoof) * 0.8,              # some momentum deviation
])

# Volume anomalies
n_vol = n_anomaly - n_flash - n_spoof
X_vol = np.column_stack([
    np.random.randn(n_vol) * 0.005 + 0.01,      # small positive returns
    np.random.randn(n_vol) * 0.008 + 0.015,     # positive 5min returns
    np.abs(np.random.randn(n_vol)) * 4 + 7,     # extreme volume!!
    np.abs(np.random.randn(n_vol)) * 0.5,        # normal spreads
    np.abs(np.random.randn(n_vol)) * 2 + 4,     # many trades
    np.random.beta(2, 8, n_vol),                  # normal cancel rate
    np.abs(np.random.randn(n_vol)) * 0.003 + 0.003, # moderate impact
    np.random.randn(n_vol) * 0.2 + 0.3,         # buy-side imbalance
    1 + np.abs(np.random.randn(n_vol)) * 0.5,    # somewhat elevated vol
    0.5 + np.random.randn(n_vol) * 0.2,          # normal correlation
    np.random.beta(7, 3, n_vol),                  # large trades
    np.random.randn(n_vol) * 0.5 + 1,           # positive momentum break
])

X = np.vstack([X_normal, X_flash, X_spoof, X_vol])
y_true = np.concatenate([
    np.zeros(n_normal),
    np.ones(n_flash),
    np.ones(n_spoof) * 2,
    np.ones(n_vol) * 3
])

# Shuffle
perm = np.random.permutation(n_observations)
X = X[perm]
y_true = y_true[perm]

print(f"Dataset shape: {X.shape}")
print(f"Normal: {np.sum(y_true == 0)}, Flash crashes: {np.sum(y_true == 1)}, "
      f"Spoofing: {np.sum(y_true == 2)}, Volume anomalies: {np.sum(y_true == 3)}")
```

### The Model in Action

Isolation Forest detects market anomalies through an elegant process:

1. **Subsampling**: From the 50,000 observations, randomly sample 256 points for each tree. This small subsample size is crucial -- it prevents anomalies from being masked by surrounding normal points and keeps computation fast.

2. **Random tree construction**: For each tree, recursively partition the 256 points by randomly selecting a feature (e.g., volume_zscore) and a split value. Flash crash observations with extreme values (-0.05 return, 8x volume) are isolated in just 2-3 splits, while normal observations deep in the center of the distribution require 7-8 splits.

3. **Path length computation**: For a new observation (e.g., a suspicious 1-minute bar), traverse each of the 100 trees and record the depth at which the point is isolated. Average across all trees.

4. **Anomaly scoring**: Convert the average path length to a score between 0 and 1. A flash crash with average path length of 3 out of a maximum of ~8 receives a score near 0.85, clearly anomalous. A normal observation with path length 7 receives a score near 0.45.

5. **Alert generation**: Apply the contamination threshold to flag the top 1% of scores as anomalies. Cross-reference with the anomaly type indicators to classify as flash crash, spoofing, or volume anomaly for targeted response.

## Advantages

1. **No assumptions about the distribution of normal trading data.** Unlike statistical methods that assume normality or Gaussian distributions, Isolation Forest works with any data distribution. This is critical for financial data which exhibits fat tails, skewness, and complex multimodal distributions.

2. **Linear time complexity with subsampling.** With a fixed subsample size (typically 256), training time is O(T * psi * log(psi)) regardless of dataset size. This makes Isolation Forest viable for real-time market surveillance processing millions of observations per day.

3. **Handles high-dimensional feature spaces naturally.** Random feature selection at each split means the algorithm scales well with dimensionality. Market surveillance data with 12+ features per observation does not cause curse-of-dimensionality issues that plague distance-based methods.

4. **Does not require labeled anomaly data for training.** Training is fully unsupervised -- no need for manually labeled flash crashes or manipulation events. This is valuable because financial anomalies are rare, diverse, and evolving, making labeled training sets incomplete by definition.

5. **Robust to irrelevant features.** If some features are uninformative for detecting anomalies, the random feature selection naturally reduces their influence. The anomaly will be isolated quickly along the informative dimensions regardless of noise dimensions.

6. **Interpretable through path analysis.** By examining which features appear in the isolation path for a flagged anomaly, analysts can understand what makes the observation unusual (e.g., "isolated primarily on volume_zscore and order_cancel_rate" suggests spoofing).

7. **Effective for diverse anomaly types simultaneously.** A single Isolation Forest can detect flash crashes (extreme returns), spoofing (high cancel rates), and volume anomalies (extreme volume) without needing separate models -- the random partitioning isolates any type of deviation quickly.

## Disadvantages

1. **Axis-parallel splits miss correlated anomalies.** Isolation Forest splits along one feature at a time, struggling to detect anomalies that are only anomalous in combination (e.g., normal volume + normal price move but unusual for that specific stock). Correlated feature anomalies require more splits and may receive lower scores.

2. **Uniform random splitting is not optimal.** The random feature and split selection does not consider feature relevance. In a 12-feature market surveillance system, splits on less informative features (e.g., cross_stock_corr) waste tree depth that could be used to isolate anomalies along more discriminative dimensions.

3. **Score calibration across different market conditions.** The anomaly score depends on the training data distribution. If trained during a calm period, normal volatility during an earnings season may trigger false positives. Conversely, training during volatile periods may normalize genuine anomalies.

4. **Contamination parameter requires domain expertise.** The fraction of expected anomalies (contamination rate) must be specified in advance. In financial markets, the anomaly rate varies -- calm periods may see <0.1% anomalies while crisis periods may see 5%+. A fixed contamination rate is suboptimal.

5. **No natural way to classify anomaly types.** Isolation Forest only provides an anomaly score, not a category. Distinguishing between flash crashes, spoofing, and fat-finger errors requires post-hoc analysis or additional classification models.

6. **Sensitivity to subsampling randomness.** With subsamples of only 256 points, there is variance across runs. Rare but genuine anomaly patterns may not be well-represented in many subsamples, leading to inconsistent scoring. This is mitigated by using more trees but increases computation.

7. **Limited effectiveness for clustered anomalies.** If anomalies form a dense cluster (e.g., coordinated spoofing across multiple participants), they may not be quickly isolated because the cluster occupies a region of the feature space. The algorithm is designed for isolated, individual anomalies.

## When to Use in Stock Market

- Real-time market surveillance for detecting flash crashes and unusual price moves
- Identifying potential insider trading through abnormal volume patterns pre-announcement
- Spoofing and layering detection based on order placement and cancellation patterns
- Fat-finger error detection (grossly mispriced orders)
- Pre-trade risk checks flagging unusual order characteristics
- Post-trade surveillance for regulatory compliance (MiFID II, Dodd-Frank)
- Detecting unusual cross-asset correlations during market stress events

## When NOT to Use in Stock Market

- When you need to classify the type of anomaly (use supervised classification)
- For detecting known, well-defined patterns (use rule-based systems)
- When anomalies are clustered rather than isolated (use DBSCAN or LOF)
- For sequential pattern detection where temporal ordering matters (use HMMs or sequence models)
- When the anomaly rate exceeds 10-15% (the "few and different" assumption breaks)
- For high-frequency tick-level data requiring sub-millisecond detection (latency too high)
- When interpretability requirements demand exact decision rules (use decision trees)

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Guidance |
|---|---|---|---|
| `n_trees` | Number of isolation trees in ensemble | 50 to 500 | 100-200 for production; more trees reduce score variance |
| `subsample_size` | Points sampled per tree | 64 to 512 | 256 default works well; reduce for faster inference |
| `max_depth` | Maximum tree height | auto (ceil(log2(psi))) | Use auto; capping too early reduces discrimination |
| `contamination` | Expected fraction of anomalies | 0.001 to 0.05 | 0.01 for general surveillance; 0.005 for low false-positive |
| `n_features` | Features considered per split | 1 to d | 1 (standard); consider sqrt(d) for Extended Isolation Forest |
| `random_state` | Random seed for reproducibility | - | Set for reproducibility; average scores across seeds for production |
| `bootstrap` | Sample with replacement | True / False | False (standard); True for additional variance |

## Stock Market Performance Tips

1. **Feature engineering is critical for market surveillance.** Raw prices and volumes are less informative than derived features: z-scores, ratios to moving averages, cancel rates, and orderbook imbalances. Pre-compute these at the appropriate time scale for your surveillance window.

2. **Use rolling windows for training.** Retrain the forest periodically (weekly or monthly) on recent data to adapt to changing market conditions. What counts as "normal" volume during earnings season differs from a quiet summer period.

3. **Ensemble multiple forests trained on different time scales.** Run separate Isolation Forests on 1-minute, 5-minute, and 30-minute features. An anomaly detected across multiple time scales is more likely genuine than one appearing at only one scale.

4. **Implement a two-stage detection pipeline.** Use Isolation Forest as a fast first-pass filter to flag the top 5% of observations, then apply more expensive, specialized models (rule-based checks, supervised classifiers) to the flagged subset.

5. **Calibrate the contamination rate per market regime.** Use VIX or realized volatility to dynamically adjust the contamination threshold. During high-volatility periods, increase the threshold to reduce false positives from legitimate but extreme market moves.

6. **Track anomaly score distributions over time.** Monitor the distribution of anomaly scores for drift. A gradual increase in average scores may indicate changing market dynamics that require retraining.

7. **Combine with domain-specific rules.** Augment Isolation Forest with hard rules for known anomaly patterns (e.g., "flag if order size > 10x average daily volume"). The forest catches novel anomalies while rules catch known patterns reliably.

## Comparison with Other Algorithms

| Aspect | Isolation Forest | LOF | One-Class SVM | Autoencoder | DBSCAN | Statistical Tests |
|---|---|---|---|---|---|---|
| Training Speed | Fast (linear) | Slow (O(n^2)) | Slow (O(n^2+)) | Medium | Fast | Very fast |
| Inference Speed | Fast | Slow (needs full data) | Fast | Fast | Slow | Very fast |
| High Dimensions | Good | Poor | Medium | Good | Poor | Poor |
| Unsupervised | Yes | Yes | Yes | Yes | Yes | No (assumptions) |
| Anomaly Types | General | Local density | Global | Reconstruction | Cluster-based | Distribution-based |
| Interpretability | Moderate (paths) | Low | Low | Moderate (residuals) | High (clusters) | High |
| Best For (Finance) | General screening | Local outliers | Known normal class | Complex patterns | Cluster anomalies | Simple features |

## Real-World Stock Market Example

```python
import numpy as np

# ================================================================
# Isolation Forest: Market Anomaly Detection from Scratch
# ================================================================

class IsolationTree:
    """Single isolation tree for anomaly detection."""

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.is_leaf = False

    def fit(self, X, depth=0):
        """Build the isolation tree."""
        n_samples, n_features = X.shape
        self.size = n_samples

        if depth >= self.max_depth or n_samples <= 1:
            self.is_leaf = True
            return self

        # Random feature and split value
        self.split_feature = np.random.randint(n_features)
        feat_values = X[:, self.split_feature]
        feat_min, feat_max = np.min(feat_values), np.max(feat_values)

        if feat_min == feat_max:
            self.is_leaf = True
            return self

        self.split_value = np.random.uniform(feat_min, feat_max)

        # Partition
        left_mask = feat_values < self.split_value
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            self.is_leaf = True
            return self

        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[right_mask], depth + 1)

        return self

    def path_length(self, x, depth=0):
        """Compute path length for a single point."""
        if self.is_leaf:
            return depth + _c(self.size)

        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        else:
            return self.right.path_length(x, depth + 1)


def _c(n):
    """Average path length of unsuccessful search in BST."""
    if n <= 1:
        return 0
    if n == 2:
        return 1
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class IsolationForest:
    """Isolation Forest ensemble for anomaly detection."""

    def __init__(self, n_trees=100, subsample_size=256, contamination=0.01):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.contamination = contamination
        self.trees = []
        self.threshold = None

    def fit(self, X):
        """Build the isolation forest."""
        n_samples = X.shape[0]
        psi = min(self.subsample_size, n_samples)
        max_depth = int(np.ceil(np.log2(psi)))

        self.trees = []
        for t in range(self.n_trees):
            # Subsample
            idx = np.random.choice(n_samples, psi, replace=False)
            X_sub = X[idx]

            # Build tree
            tree = IsolationTree(max_depth).fit(X_sub)
            self.trees.append(tree)

        # Compute threshold on training data
        scores = self.score_samples(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))

        return self

    def score_samples(self, X):
        """Compute anomaly scores for all samples."""
        n_samples = X.shape[0]
        avg_path_lengths = np.zeros(n_samples)

        for tree in self.trees:
            for i in range(n_samples):
                avg_path_lengths[i] += tree.path_length(X[i])

        avg_path_lengths /= self.n_trees

        # Normalize: score = 2^(-E[h(x)] / c(psi))
        c_n = _c(self.subsample_size)
        scores = np.power(2, -avg_path_lengths / c_n)

        return scores

    def predict(self, X):
        """Predict anomalies: 1 = anomaly, 0 = normal."""
        scores = self.score_samples(X)
        return (scores >= self.threshold).astype(int)

    def get_feature_importance(self, X, feature_names=None):
        """Estimate feature importance based on split frequency in anomaly paths."""
        n_features = X.shape[1]
        importance = np.zeros(n_features)

        # Count feature usage in early splits (depth 0-2) across trees
        for tree in self.trees:
            self._count_splits(tree, importance, depth=0, max_depth=3)

        importance = importance / np.sum(importance)

        if feature_names:
            return list(zip(feature_names, importance))
        return importance

    def _count_splits(self, node, counts, depth, max_depth):
        """Recursively count feature splits."""
        if node.is_leaf or depth >= max_depth:
            return
        weight = 1.0 / (depth + 1)  # Weight early splits higher
        counts[node.split_feature] += weight
        if node.left:
            self._count_splits(node.left, counts, depth + 1, max_depth)
        if node.right:
            self._count_splits(node.right, counts, depth + 1, max_depth)


# ================================================================
# Generate Stock Market Surveillance Data
# ================================================================

np.random.seed(42)

n_total = 20000
n_features = 12
contamination_rate = 0.015

feature_names = [
    'return_1min', 'return_5min', 'volume_zscore', 'spread_zscore',
    'trade_count_ratio', 'order_cancel_rate', 'price_impact',
    'depth_imbalance', 'volatility_ratio', 'cross_stock_corr',
    'large_trade_pct', 'momentum_break'
]

# Normal market behavior
n_normal = int(n_total * (1 - contamination_rate))
X_normal = np.column_stack([
    np.random.randn(n_normal) * 0.002,
    np.random.randn(n_normal) * 0.004,
    np.random.randn(n_normal) * 1.0,
    np.abs(np.random.randn(n_normal)) * 0.5,
    1 + np.random.randn(n_normal) * 0.3,
    np.random.beta(2, 10, n_normal),
    np.abs(np.random.randn(n_normal)) * 0.001,
    np.random.randn(n_normal) * 0.2,
    1 + np.abs(np.random.randn(n_normal)) * 0.3,
    0.6 + np.random.randn(n_normal) * 0.15,
    np.random.beta(5, 5, n_normal),
    np.random.randn(n_normal) * 0.5,
])

# Flash crash anomalies
n_flash = int(n_total * contamination_rate * 0.4)
X_flash = np.column_stack([
    -0.05 + np.random.randn(n_flash) * 0.015,
    -0.08 + np.random.randn(n_flash) * 0.02,
    5 + np.abs(np.random.randn(n_flash)) * 3,
    3 + np.abs(np.random.randn(n_flash)) * 2,
    3 + np.abs(np.random.randn(n_flash)) * 1,
    np.random.beta(3, 5, n_flash),
    0.005 + np.abs(np.random.randn(n_flash)) * 0.005,
    -0.4 + np.random.randn(n_flash) * 0.2,
    4 + np.abs(np.random.randn(n_flash)) * 2,
    0.2 + np.random.randn(n_flash) * 0.2,
    0.7 + np.random.beta(5, 2, n_flash) * 0.3,
    -3 + np.random.randn(n_flash) * 1,
])

# Spoofing anomalies
n_spoof = int(n_total * contamination_rate * 0.35)
X_spoof = np.column_stack([
    np.random.randn(n_spoof) * 0.003,
    np.random.randn(n_spoof) * 0.005,
    2 + np.abs(np.random.randn(n_spoof)) * 2,
    1 + np.abs(np.random.randn(n_spoof)) * 1,
    2 + np.abs(np.random.randn(n_spoof)) * 1,
    0.75 + np.random.beta(2, 5, n_spoof) * 0.25,
    np.abs(np.random.randn(n_spoof)) * 0.002,
    np.random.randn(n_spoof) * 0.4,
    1.2 + np.abs(np.random.randn(n_spoof)) * 0.5,
    0.5 + np.random.randn(n_spoof) * 0.2,
    np.random.beta(3, 7, n_spoof),
    np.random.randn(n_spoof) * 0.8,
])

# Volume spike anomalies
n_vol = n_total - n_normal - n_flash - n_spoof
X_vol = np.column_stack([
    0.01 + np.random.randn(n_vol) * 0.005,
    0.015 + np.random.randn(n_vol) * 0.008,
    7 + np.abs(np.random.randn(n_vol)) * 4,
    np.abs(np.random.randn(n_vol)) * 0.5,
    4 + np.abs(np.random.randn(n_vol)) * 2,
    np.random.beta(2, 8, n_vol),
    0.003 + np.abs(np.random.randn(n_vol)) * 0.003,
    0.3 + np.random.randn(n_vol) * 0.2,
    1.5 + np.abs(np.random.randn(n_vol)) * 0.5,
    0.5 + np.random.randn(n_vol) * 0.2,
    np.random.beta(7, 3, n_vol),
    1 + np.random.randn(n_vol) * 0.5,
])

X = np.vstack([X_normal, X_flash, X_spoof, X_vol])
y_true = np.concatenate([
    np.zeros(n_normal),
    np.ones(n_flash),
    2 * np.ones(n_spoof),
    3 * np.ones(n_vol)
])
y_binary = (y_true > 0).astype(int)

# Shuffle
perm = np.random.permutation(n_total)
X = X[perm]
y_true = y_true[perm]
y_binary = y_binary[perm]

print(f"Dataset: {X.shape}")
print(f"Normal: {np.sum(y_true == 0)}, Anomalies: {np.sum(y_true > 0)}")
print(f"  Flash crashes: {np.sum(y_true == 1)}")
print(f"  Spoofing: {np.sum(y_true == 2)}")
print(f"  Volume spikes: {np.sum(y_true == 3)}")

# ================================================================
# Train Isolation Forest
# ================================================================

print("\n=== Training Isolation Forest ===")
iforest = IsolationForest(n_trees=100, subsample_size=256, contamination=0.02)
iforest.fit(X)

# ================================================================
# Evaluate Detection Performance
# ================================================================

scores = iforest.score_samples(X)
predictions = iforest.predict(X)

# Binary classification metrics
tp = np.sum((predictions == 1) & (y_binary == 1))
fp = np.sum((predictions == 1) & (y_binary == 0))
fn = np.sum((predictions == 0) & (y_binary == 1))
tn = np.sum((predictions == 0) & (y_binary == 0))

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\n=== Detection Performance ===")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ================================================================
# Score Distribution Analysis
# ================================================================

print(f"\n=== Anomaly Score Distribution ===")
print(f"{'Category':<20} {'Mean Score':>12} {'Median':>10} {'Max':>10}")
print("-" * 52)

categories = ['Normal', 'Flash Crash', 'Spoofing', 'Volume Spike']
for cat_id, cat_name in enumerate(categories):
    mask = y_true == cat_id
    if np.sum(mask) > 0:
        cat_scores = scores[mask]
        print(f"{cat_name:<20} {np.mean(cat_scores):>12.4f} "
              f"{np.median(cat_scores):>10.4f} {np.max(cat_scores):>10.4f}")

# ================================================================
# Per-Anomaly-Type Detection Rate
# ================================================================

print(f"\n=== Detection Rate by Anomaly Type ===")
for cat_id, cat_name in enumerate(['Flash Crash', 'Spoofing', 'Volume Spike']):
    mask = y_true == (cat_id + 1)
    detected = np.sum(predictions[mask] == 1)
    total = np.sum(mask)
    print(f"  {cat_name}: {detected}/{total} detected ({detected/total:.1%})")

# ================================================================
# Feature Importance
# ================================================================

print(f"\n=== Feature Importance (Split Frequency) ===")
importances = iforest.get_feature_importance(X, feature_names)
importances_sorted = sorted(importances, key=lambda x: x[1], reverse=True)
for name, imp in importances_sorted:
    bar = "#" * int(imp * 100)
    print(f"  {name:<25s}: {imp:.4f} {bar}")

# ================================================================
# Top Anomalies Deep Dive
# ================================================================

print(f"\n=== Top 10 Most Anomalous Observations ===")
top_10_idx = np.argsort(scores)[-10:][::-1]
print(f"{'Rank':<6} {'Score':>8} {'Type':<15} {'Key Features'}")
print("-" * 70)

for rank, idx in enumerate(top_10_idx, 1):
    anomaly_type = categories[int(y_true[idx])]
    # Find the most extreme feature values
    z_scores = np.abs((X[idx] - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8))
    top_feat = np.argsort(z_scores)[-3:][::-1]
    feat_str = ", ".join([f"{feature_names[f]}={X[idx,f]:.4f}" for f in top_feat])
    print(f"{rank:<6} {scores[idx]:>8.4f} {anomaly_type:<15} {feat_str}")

# ================================================================
# Threshold Sensitivity Analysis
# ================================================================

print(f"\n=== Threshold Sensitivity ===")
print(f"{'Contamination':>15} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 55)

for cont in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
    thresh = np.percentile(scores, 100 * (1 - cont))
    preds = (scores >= thresh).astype(int)
    tp_c = np.sum((preds == 1) & (y_binary == 1))
    fp_c = np.sum((preds == 1) & (y_binary == 0))
    fn_c = np.sum((preds == 0) & (y_binary == 1))
    prec = tp_c / (tp_c + fp_c + 1e-8)
    rec = tp_c / (tp_c + fn_c + 1e-8)
    f1_c = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"{cont:>15.3f} {thresh:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1_c:>10.4f}")

# ================================================================
# Real-Time Monitoring Simulation
# ================================================================

print(f"\n=== Real-Time Monitoring Simulation (last 100 observations) ===")
recent = X[-100:]
recent_scores = iforest.score_samples(recent)
recent_preds = (recent_scores >= iforest.threshold).astype(int)

n_alerts = np.sum(recent_preds)
print(f"Observations processed: 100")
print(f"Alerts generated: {n_alerts}")
print(f"Alert rate: {n_alerts/100:.1%}")

if n_alerts > 0:
    alert_idx = np.where(recent_preds == 1)[0]
    print(f"\nAlert details:")
    for i in alert_idx[:5]:
        print(f"  Observation {i}: score={recent_scores[i]:.4f}, "
              f"return_1min={recent[i,0]:.4f}, volume_z={recent[i,2]:.2f}, "
              f"cancel_rate={recent[i,5]:.2f}")
```

## Key Takeaways

1. **Isolation Forest is ideal for market surveillance** because it directly models anomalousness rather than normality, efficiently handling the extreme class imbalance (>99% normal) typical of market data without requiring labeled anomaly examples.

2. **Subsampling is both a feature and a necessity** -- the small subsample size (256) prevents masking effects where anomalies influence the model and enables the linear scaling needed for real-time deployment.

3. **Feature engineering determines detection quality** -- raw market data should be transformed into z-scores, ratios, and rates that highlight deviations from normal behavior. The algorithm itself is straightforward; the features encode the domain knowledge.

4. **The contamination rate should be dynamically adjusted** based on market regime. Fixed thresholds produce too many false positives during volatile periods and miss anomalies during calm periods.

5. **A two-stage pipeline is practical** -- use Isolation Forest as a fast first-pass filter to flag suspicious observations, then apply domain-specific rules and more expensive models to classify and investigate the flagged events.

6. **Different anomaly types have different detection rates** -- flash crashes with extreme feature values are easiest to detect, while subtle manipulation patterns (spoofing) with fewer extreme features require more trees and better features for reliable detection.
