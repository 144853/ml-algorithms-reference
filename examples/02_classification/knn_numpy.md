# K-Nearest Neighbors (KNN) - Complete Guide with Stock Market Applications

## Overview

K-Nearest Neighbors (KNN) is one of the simplest yet most intuitive machine learning algorithms. It is a non-parametric, instance-based (lazy) learning algorithm that makes predictions based on the similarity between new observations and stored training examples. Rather than learning an explicit model during training, KNN memorizes the entire training dataset and defers all computation to prediction time. When a new data point needs to be classified, KNN finds the K most similar (nearest) training examples in the feature space and assigns the majority class among those neighbors.

For stock market applications, KNN's core principle -- "similar market conditions lead to similar outcomes" -- aligns naturally with how experienced traders think. Technical analysts often look for historical patterns that resemble the current market setup, reasoning that if five previous instances of a particular chart pattern all preceded a rally, the current occurrence is likely to do the same. KNN formalizes this pattern-matching intuition by computing distances between the current feature vector (today's technical indicators) and all historical feature vectors, then examining what happened after the most similar historical days.

The algorithm is particularly powerful for identifying similar historical stock patterns because it makes no assumptions about the functional form of the relationship between features and outcomes. Unlike linear models that assume a specific mathematical structure, KNN can capture arbitrary decision boundaries shaped entirely by the local structure of the training data. This flexibility is valuable in financial markets where the mapping from indicators to outcomes is complex, regime-dependent, and difficult to specify parametrically. However, this flexibility comes at the cost of computational expense at prediction time and sensitivity to the curse of dimensionality, both of which require careful management in financial applications.

## How It Works - The Math Behind It

### Distance Metrics

The choice of distance metric defines what "similar" means for historical pattern matching.

**Euclidean Distance (L2 norm):**
```
d(x, y) = sqrt(sum((x_i - y_i)^2))  for i = 1 to n
```

**Manhattan Distance (L1 norm):**
```
d(x, y) = sum(|x_i - y_i|)  for i = 1 to n
```

**Minkowski Distance (generalized):**
```
d(x, y) = (sum(|x_i - y_i|^p))^(1/p)
```

**Cosine Distance (for direction-based similarity):**
```
d(x, y) = 1 - (x . y) / (||x|| * ||y||)
```

**Mahalanobis Distance (correlation-aware):**
```
d(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```
where S is the covariance matrix of the feature space. This accounts for correlations between features, which is important for stock market indicators that are often highly correlated (e.g., RSI and Stochastic oscillator).

### Classification Rule

Given a query point `x_q`, find the K nearest neighbors `N_K(x_q)`:

```
y_hat = argmax_c sum(I(y_i == c))  for all x_i in N_K(x_q)
```

where `I()` is the indicator function. The predicted class is the majority vote among the K neighbors.

### Distance-Weighted Voting

Rather than equal votes, weight each neighbor inversely by distance:

```
y_hat = argmax_c sum(w_i * I(y_i == c))  for all x_i in N_K(x_q)

w_i = 1 / (d(x_q, x_i) + epsilon)
```

where `epsilon` prevents division by zero. Closer neighbors (more similar historical patterns) have more influence on the prediction.

### Probability Estimation

KNN naturally provides probability estimates:

```
P(C = c | x_q) = sum(w_i * I(y_i == c)) / sum(w_i)  for all x_i in N_K(x_q)
```

### Optimal K Selection

The choice of K controls the bias-variance trade-off:
- **Small K (e.g., 1-3):** Low bias, high variance. The model is sensitive to individual noisy neighbors (noisy trading days).
- **Large K (e.g., 50-100):** High bias, low variance. The model smooths over local patterns, potentially missing short-lived market anomalies.
- **K = sqrt(N):** A common heuristic starting point, where N is the number of training samples.

### Feature Scaling

Because KNN is distance-based, feature scaling is critical:

```
x_i_scaled = (x_i - mu_i) / sigma_i     (standardization)
x_i_scaled = (x_i - min_i) / (max_i - min_i)  (min-max normalization)
```

Without scaling, features with larger magnitudes (e.g., volume in millions) dominate the distance calculation over features with smaller ranges (e.g., RSI from 0 to 100).

## Stock Market Use Case: Pattern Matching for Similar Historical Stock Patterns

### The Problem

A quantitative analyst wants to find historical trading days that most closely resemble today's market conditions and use the outcomes of those similar days to predict tomorrow's price movement. The system should identify the K most similar historical patterns across multiple technical indicators and aggregate their subsequent returns to generate a trading signal with a confidence measure.

### Stock Market Features (Input Data)

| Feature | Description | Type | Example Value |
|---------|-------------|------|---------------|
| RSI_14 | 14-day Relative Strength Index | Continuous | 58.3 |
| MACD_hist | MACD histogram value | Continuous | 0.42 |
| BB_pct | Bollinger Band percentile (0-1) | Continuous | 0.65 |
| Volume_zscore | Volume z-score (vs. 20-day mean) | Continuous | 1.2 |
| Return_5d | 5-day cumulative return | Continuous | 0.018 |
| Return_20d | 20-day cumulative return | Continuous | 0.045 |
| Volatility_10d | 10-day realized volatility | Continuous | 0.015 |
| Volatility_ratio | Short/long term vol ratio | Continuous | 1.15 |
| Trend_slope | Linear regression slope of last 20 prices | Continuous | 0.003 |
| ATR_pct | ATR as percentage of price | Continuous | 0.018 |
| Stoch_K | Stochastic %K | Continuous | 72.5 |
| OBV_zscore | On-Balance Volume z-score | Continuous | 0.85 |
| Gap_pct | Overnight gap as percentage | Continuous | 0.002 |
| Candle_body_ratio | Candle body / total range | Continuous | 0.65 |
| High_low_range | (High - Low) / Close | Continuous | 0.022 |
| Days_since_high | Days since 52-week high | Count | 15 |

**Target Variable:** `next_day_direction` -- +1 (up day) or -1 (down day)

### Example Data Structure

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_days = 2500  # ~10 years of trading data

# Simulate a realistic stock price path
daily_returns = np.random.normal(0.0003, 0.015, n_days)
prices = 100 * np.cumprod(1 + daily_returns)
volumes = np.random.lognormal(15, 0.4, n_days)
highs = prices * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
lows = prices * (1 - np.abs(np.random.normal(0, 0.008, n_days)))

# Calculate features
features = {}
features['RSI_14'] = np.random.uniform(20, 80, n_days)  # simplified
features['MACD_hist'] = np.random.normal(0, 0.5, n_days)
features['BB_pct'] = np.random.uniform(0, 1, n_days)
features['Volume_zscore'] = (volumes - np.convolve(volumes, np.ones(20)/20,
                             mode='same')) / np.std(volumes)
features['Return_5d'] = np.convolve(daily_returns, np.ones(5), mode='same')
features['Return_20d'] = np.convolve(daily_returns, np.ones(20), mode='same')
features['Volatility_10d'] = np.array([np.std(daily_returns[max(0,i-10):i+1])
                              for i in range(n_days)])
features['Volatility_ratio'] = np.random.normal(1.0, 0.2, n_days)
features['Trend_slope'] = np.random.normal(0, 0.003, n_days)
features['ATR_pct'] = (highs - lows) / prices
features['Stoch_K'] = np.random.uniform(10, 90, n_days)
features['OBV_zscore'] = np.random.normal(0, 1, n_days)
features['Gap_pct'] = np.random.normal(0, 0.005, n_days)
features['Candle_body_ratio'] = np.random.uniform(0.1, 0.9, n_days)
features['High_low_range'] = (highs - lows) / prices
features['Days_since_high'] = np.random.randint(0, 252, n_days)

feature_names = list(features.keys())
X = np.column_stack([features[f] for f in feature_names])

# Target: next day direction
y = np.zeros(n_days)
y[:-1] = np.where(daily_returns[1:] > 0, 1, -1)
y[-1] = 1

print(f"Dataset: {X.shape[0]} trading days, {X.shape[1]} features")
print(f"Features: {feature_names}")
print(f"Class distribution: Up={np.sum(y==1)}, Down={np.sum(y==-1)}")
```

### The Model in Action

```python
# KNN pattern matching for a single trading day
# Today's features: RSI=62, MACD_hist=0.3, BB_pct=0.7, Vol_z=1.1, ...

# Step 1: Compute distance to ALL historical days
# d(today, day_1) = sqrt((62-45)^2 + (0.3-(-0.5))^2 + ... ) = 4.82
# d(today, day_2) = sqrt((62-58)^2 + (0.3-0.4)^2 + ...)     = 1.23
# d(today, day_3) = sqrt((62-71)^2 + (0.3-0.1)^2 + ...)     = 3.45
# ...
# d(today, day_N) = ...

# Step 2: Sort by distance and select K=7 nearest neighbors
# Neighbor 1: day_487, distance=0.85, next_day=+1 (UP),    weight=1.18
# Neighbor 2: day_1203, distance=1.23, next_day=+1 (UP),   weight=0.81
# Neighbor 3: day_892, distance=1.45, next_day=-1 (DOWN),  weight=0.69
# Neighbor 4: day_2010, distance=1.52, next_day=+1 (UP),   weight=0.66
# Neighbor 5: day_156, distance=1.78, next_day=+1 (UP),    weight=0.56
# Neighbor 6: day_1567, distance=1.95, next_day=-1 (DOWN), weight=0.51
# Neighbor 7: day_734, distance=2.10, next_day=+1 (UP),    weight=0.48

# Step 3: Weighted vote
# UP weight:   1.18 + 0.81 + 0.66 + 0.56 + 0.48 = 3.69
# DOWN weight: 0.69 + 0.51 = 1.20

# Step 4: Prediction
# P(UP) = 3.69 / (3.69 + 1.20) = 0.755
# P(DOWN) = 1.20 / (3.69 + 1.20) = 0.245
# Prediction: UP with 75.5% confidence => BUY signal
```

## Advantages

1. **Intuitive pattern-matching aligns with trader reasoning.** The idea that "similar past conditions lead to similar outcomes" is how experienced technical analysts think. KNN formalizes this intuition with precise distance calculations, making it easy for traders to understand, trust, and incorporate into their workflow. When the model predicts "UP," you can show the K most similar historical days and their outcomes as justification.

2. **No training phase required -- instantly deployable.** KNN stores the entire training set without any model fitting. This means new historical data can be added instantly without retraining. When today's market closes, the day's features are simply appended to the dataset and immediately available for future queries. This is a significant operational advantage for live trading systems.

3. **Naturally provides probability estimates with interpretable confidence.** The proportion of neighbors in each class gives a natural probability estimate. Furthermore, the average distance to neighbors indicates how "unprecedented" the current market condition is -- if all neighbors are very far away, the current pattern has no close historical analog, and the model's prediction should be treated with less confidence.

4. **Captures complex, non-linear decision boundaries.** KNN makes no assumptions about the functional form of the relationship between features and outcomes. It can model arbitrary decision surfaces, including ones that are locally variable -- different regions of the feature space can have completely different decision rules. This flexibility is valuable in markets where the signal-to-outcome mapping varies by regime.

5. **Effective for anomaly detection.** If the average distance to the K nearest neighbors is unusually large, the current market condition is unlike anything in the historical database. This anomaly detection capability can serve as an early warning system for unprecedented market conditions (e.g., flash crashes, liquidity crises) where historical analogy breaks down and caution is warranted.

6. **Easily extended with custom domain-specific distance metrics.** The distance function is the core of KNN and can be customized for financial applications. A finance-specific distance metric might weight recent historical days more than distant ones (temporal weighting), weight volatile features differently from stable ones, or use Mahalanobis distance to account for feature correlations. This extensibility allows incorporating domain expertise directly into the similarity calculation.

7. **Provides a complete set of analogous historical scenarios.** Beyond the prediction itself, KNN returns the actual historical days that form the basis for the prediction. This enables rich scenario analysis: what was the average return, maximum drawdown, holding period, and risk-reward ratio of the K nearest historical analogs? This information supports more informed position sizing and risk management than a simple directional prediction.

## Disadvantages

1. **Prediction time is O(N * d) for each query -- slow for large datasets.** At prediction time, KNN must compute the distance between the query point and every training example. With 5,000+ training days and 16 features, this is manageable, but scaling to tick-level data or multi-asset universes makes brute-force KNN impractical. Approximate nearest neighbor methods (KD-trees, ball trees, locality-sensitive hashing) can reduce this to O(d * log N) but add implementation complexity.

2. **Curse of dimensionality severely degrades performance.** As the number of features increases, distances between points become increasingly uniform in high-dimensional space. With 16+ features, the nearest neighbor and the farthest neighbor may have nearly identical distances, making the concept of "nearest" meaningless. Dimensionality reduction (PCA, feature selection) is essential before applying KNN to stock market data.

3. **Feature scaling dominates the outcome and must be chosen carefully.** Because KNN is purely distance-based, the choice of scaling method (standardization, normalization, rank-based) dramatically affects which neighbors are selected. Two different scaling choices applied to the same data can produce entirely different predictions. There is no universal best scaling for financial features, and this decision requires domain expertise and empirical validation.

4. **Sensitive to noise and irrelevant features.** Every feature contributes equally to the distance calculation (unless weighted). Irrelevant or noisy features add random noise to the distance, potentially causing irrelevant historical days to appear as nearest neighbors. Feature selection or feature weighting is critical but adds a model selection layer that partially undermines KNN's simplicity.

5. **Memory-intensive -- must store entire training set.** Unlike parametric models that compress training data into parameters, KNN stores all N training samples. For a multi-stock, multi-feature system processing decades of history, memory requirements grow linearly with data size. Compression techniques (prototypes, condensed nearest neighbors) can reduce storage but may lose important edge cases.

6. **Cannot extrapolate beyond the training data range.** KNN predictions are always interpolations within the convex hull of the training data. If the current market conditions involve feature values outside the historical range (e.g., unprecedented volatility or interest rates at historic highs), KNN will predict based on the closest known examples, which may be poor analogs. It fundamentally cannot handle truly novel market conditions.

7. **Equal weighting of all historical periods ignores market evolution.** Standard KNN treats a trading day from 2005 as equally relevant as one from 2024. But markets evolve -- algorithmic trading has changed microstructure, new instruments affect correlations, and regulatory changes alter dynamics. Without temporal weighting, KNN may match today's pattern to an outdated historical analog that is no longer predictively relevant.

## When to Use in Stock Market

- For finding historical analogs to the current market environment ("what happened the last 10 times RSI was 65 with declining volume?")
- When interpretability is paramount -- showing clients the K most similar historical days provides intuitive justification
- For scenario analysis and stress testing -- examining the range of outcomes from similar historical periods
- When the dataset is moderate-sized (500-10,000 observations) with well-curated features (5-10 after dimensionality reduction)
- As a complementary signal alongside model-based predictions (ensemble diversification)
- For detecting unprecedented market conditions (anomaly detection based on neighbor distances)
- When the trading strategy needs to be updated instantly as new data arrives (no retraining needed)

## When NOT to Use in Stock Market

- For high-frequency trading or tick-level prediction where prediction latency is critical
- When the feature space is high-dimensional (>15 features) without prior dimensionality reduction
- For multi-asset universe-level prediction (computing distances across thousands of stocks is expensive)
- When market structure has changed significantly (features from pre-2010 data may be poor analogs for post-2020 markets)
- For long-horizon predictions where local pattern matching is less reliable than macroeconomic models
- When the training dataset is very small (<200 observations) and neighbors would be sparse and unreliable
- When a clear, deployable model file is needed (KNN requires shipping the entire training dataset)

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Recommendation |
|----------------|-------------|---------------|----------------------------|
| K | Number of nearest neighbors | 3 - 50 | Start with sqrt(N). For daily stock data with ~2000 samples, try K=15-30. Odd K avoids ties in binary classification. Optimize via walk-forward validation. |
| distance_metric | Distance function | euclidean, manhattan, cosine, mahalanobis | Start with Euclidean on standardized features. Test Mahalanobis to account for feature correlations. Cosine distance is useful when feature magnitudes are less important than relative patterns. |
| weights | Neighbor weighting scheme | uniform, distance, custom | Always use distance-weighted voting for financial data. Closer analogs should have more influence. Custom time-decay weighting is even better: recent similar days weighted more than distant ones. |
| feature_scaling | Preprocessing method | standard, minmax, robust, rank | Robust scaling (using median and IQR) handles the fat tails common in financial distributions. Rank-based scaling is the most robust option for heavily skewed features like volume. |
| n_features | Number of features after selection | 5 - 12 | More is not better for KNN. Use 6-10 features after PCA or feature selection. Prefer independent, diversified features (one volatility, one trend, one volume, one momentum indicator). |
| leaf_size | KD-tree leaf size (for fast lookup) | 20 - 50 | Only matters for efficiency. Default 30 is fine for datasets under 50,000 points. Larger values for higher-dimensional data. |
| temporal_window | Historical data lookback | 1-10 years | 3-5 years provides sufficient history while maintaining relevance. Older data may reflect obsolete market dynamics. |

## Stock Market Performance Tips

1. **Reduce dimensionality before applying KNN.** Use PCA to reduce the feature set to 5-8 principal components, or manually select a diverse set of features covering different market aspects (trend, volatility, volume, momentum). Verify that the curse of dimensionality is not degrading performance by checking the ratio of the nearest to farthest neighbor distance -- if it is close to 1.0, your feature space is too high-dimensional.

2. **Apply temporal weighting to distances.** Modify the distance calculation to penalize historical days that are far in calendar time: `d_modified(x_q, x_i) = d(x_q, x_i) * (1 + lambda * |t_q - t_i|)` where `lambda` controls the temporal decay rate. This ensures that recent analogs are preferred over distant ones, adapting to evolving market structure.

3. **Use adaptive K based on neighbor density.** In dense regions of the feature space (common market conditions), use larger K for more stable predictions. In sparse regions (unusual market conditions), reduce K to avoid pulling in dissimilar neighbors. Alternatively, use a radius-based approach: include all neighbors within a fixed distance rather than a fixed count.

4. **Exclude forward-looking features and prevent leakage.** Ensure that features computed at time t use only data available up to time t. Rolling calculations must use trailing windows only. This is a common pitfall in financial KNN implementations that leads to spectacular but unrealistic backtesting results.

5. **Examine the neighbor set for qualitative coherence.** Periodically inspect the K nearest neighbors returned by the model. Do they correspond to qualitatively similar market conditions? If the model matches a low-volatility trending day with a high-volatility mean-reverting day purely due to numerical proximity in scaled feature space, the feature set or distance metric needs refinement.

6. **Combine KNN predictions with a staleness filter.** If the nearest neighbors are all more than 3 years old, the current market condition may have no relevant modern analog. In this case, flag the prediction as "low confidence" and either abstain from trading or fall back to a parametric model that can extrapolate.

7. **Use leave-one-out cross-validation for K selection.** For time-series data, this means: for each day t, find the K nearest neighbors excluding day t itself, predict day t's outcome, and measure accuracy across all days. This is computationally expensive but provides the most reliable estimate of optimal K for financial data.

## Comparison with Other Algorithms

| Criterion | KNN | Random Forest | SVM | Logistic Regression | Neural Network | AdaBoost |
|-----------|-----|---------------|-----|---------------------|----------------|----------|
| Training speed | None (lazy) | Fast | Moderate | Very fast | Slow | Moderate |
| Prediction speed | Slow (O(N*d)) | Fast | Fast | Very fast | Fast | Fast |
| Memory requirement | High (stores all data) | Moderate | Low (support vectors) | Very low | Moderate | Low |
| Interpretability | High (show neighbors) | Moderate | Low | High | Very low | Moderate |
| Non-linearity | Yes (inherent) | Yes | Yes (kernel) | No | Yes | Yes (ensemble) |
| Feature scaling needed | Yes (critical) | No | Yes | Yes | Yes | No |
| Handles high dimensions | Poorly | Well | Well | Well | Well | Well |
| Probability calibration | Good (natural) | Good | Poor | Excellent | Moderate | Poor |
| Online learning | Trivial (add points) | Must retrain | Must retrain | SGD variant | Must retrain | Must retrain |
| Anomaly detection | Excellent (distance-based) | Poor | Moderate | Poor | Moderate | Poor |
| Best stock market use | Historical pattern matching | General classification | Regime detection | Simple trends | Complex patterns | Weak signal boosting |

## Real-World Stock Market Example

```python
import numpy as np
from collections import Counter

# =============================================================================
# KNN from Scratch for Historical Stock Pattern Matching
# =============================================================================

class StandardScaler:
    """Feature standardization."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class KNNClassifier:
    """K-Nearest Neighbors for stock market pattern matching."""

    def __init__(self, k=15, weights='distance', metric='euclidean',
                 temporal_decay=0.0):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.temporal_decay = temporal_decay
        self.X_train = None
        self.y_train = None
        self.train_indices = None  # for temporal weighting

    def fit(self, X, y, time_indices=None):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.train_indices = (time_indices if time_indices is not None
                             else np.arange(len(X)))
        return self

    def _compute_distances(self, X_query, query_time_idx=None):
        """Compute distances from query points to all training points."""
        if self.metric == 'euclidean':
            # Vectorized Euclidean distance
            sq_dists = (np.sum(X_query**2, axis=1, keepdims=True) +
                       np.sum(self.X_train**2, axis=1) -
                       2 * X_query @ self.X_train.T)
            distances = np.sqrt(np.maximum(sq_dists, 0))

        elif self.metric == 'manhattan':
            distances = np.zeros((X_query.shape[0], self.X_train.shape[0]))
            for i in range(X_query.shape[0]):
                distances[i] = np.sum(np.abs(X_query[i] - self.X_train), axis=1)

        elif self.metric == 'cosine':
            query_norms = np.sqrt(np.sum(X_query**2, axis=1, keepdims=True))
            train_norms = np.sqrt(np.sum(self.X_train**2, axis=1, keepdims=True))
            query_norms[query_norms == 0] = 1
            train_norms[train_norms == 0] = 1
            X_query_norm = X_query / query_norms
            X_train_norm = self.X_train / train_norms
            cosine_sim = X_query_norm @ X_train_norm.T
            distances = 1 - cosine_sim

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Apply temporal decay if configured
        if self.temporal_decay > 0 and query_time_idx is not None:
            time_diffs = np.abs(query_time_idx[:, np.newaxis] -
                               self.train_indices[np.newaxis, :])
            temporal_penalty = 1 + self.temporal_decay * time_diffs
            distances *= temporal_penalty

        return distances

    def _get_neighbors(self, distances, exclude_self=False):
        """Get K nearest neighbor indices and distances."""
        if exclude_self:
            # Set self-distance to infinity
            np.fill_diagonal(distances, np.inf)

        k_nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]

        # Sort the K nearest by distance
        k_distances = np.zeros((distances.shape[0], self.k))
        for i in range(distances.shape[0]):
            sorted_order = np.argsort(distances[i, k_nearest_indices[i]])
            k_nearest_indices[i] = k_nearest_indices[i][sorted_order]
            k_distances[i] = distances[i, k_nearest_indices[i]]

        return k_nearest_indices, k_distances

    def _compute_weights(self, distances):
        """Compute neighbor weights."""
        if self.weights == 'uniform':
            return np.ones_like(distances)
        elif self.weights == 'distance':
            return 1.0 / (distances + 1e-10)
        else:
            raise ValueError(f"Unknown weight scheme: {self.weights}")

    def predict(self, X, query_time_idx=None):
        distances = self._compute_distances(X, query_time_idx)
        neighbor_indices, neighbor_distances = self._get_neighbors(distances)
        weights = self._compute_weights(neighbor_distances)

        classes = np.unique(self.y_train)
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            neighbor_labels = self.y_train[neighbor_indices[i]]
            class_weights = {}
            for c in classes:
                mask = neighbor_labels == c
                class_weights[c] = weights[i][mask].sum()
            predictions[i] = max(class_weights, key=class_weights.get)

        return predictions

    def predict_proba(self, X, query_time_idx=None):
        distances = self._compute_distances(X, query_time_idx)
        neighbor_indices, neighbor_distances = self._get_neighbors(distances)
        weights = self._compute_weights(neighbor_distances)

        classes = np.unique(self.y_train)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        probas = np.zeros((X.shape[0], len(classes)))

        for i in range(X.shape[0]):
            neighbor_labels = self.y_train[neighbor_indices[i]]
            total_weight = weights[i].sum()
            for c in classes:
                mask = neighbor_labels == c
                probas[i, class_to_idx[c]] = weights[i][mask].sum() / total_weight

        return probas

    def get_neighbors_detail(self, x_query, query_time_idx=None):
        """Return detailed information about nearest neighbors."""
        x_query = x_query.reshape(1, -1)
        if query_time_idx is not None:
            query_time_idx = np.array([query_time_idx])

        distances = self._compute_distances(x_query, query_time_idx)
        neighbor_indices, neighbor_distances = self._get_neighbors(distances)

        details = []
        for j in range(self.k):
            idx = neighbor_indices[0][j]
            details.append({
                'train_index': idx,
                'distance': neighbor_distances[0][j],
                'label': self.y_train[idx],
                'features': self.X_train[idx],
                'time_index': self.train_indices[idx]
            })
        return details

    def average_neighbor_distance(self, X, query_time_idx=None):
        """Compute average distance to K neighbors (anomaly score)."""
        distances = self._compute_distances(X, query_time_idx)
        neighbor_indices, neighbor_distances = self._get_neighbors(distances)
        return neighbor_distances.mean(axis=1)


# =============================================================================
# Feature Engineering for Stock Pattern Matching
# =============================================================================

def generate_stock_pattern_data(n_days=3000, seed=42):
    """Generate realistic stock data with technical features."""
    np.random.seed(seed)

    # Simulate stock with regime changes
    returns = np.zeros(n_days)
    volatility = 0.015
    for t in range(1, n_days):
        # Varying volatility regime
        if t % 500 < 200:
            volatility = 0.012  # low vol
        elif t % 500 < 350:
            volatility = 0.020  # high vol
        else:
            volatility = 0.015  # normal vol
        returns[t] = np.random.normal(0.0003, volatility)

    prices = 100 * np.cumprod(1 + returns)
    volumes = np.random.lognormal(15, 0.4, n_days)
    highs = prices * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.008, n_days)))

    # Technical features
    features = {}

    # RSI (14-day approximation)
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
    avg_loss[avg_loss == 0] = 1e-10
    features['RSI_14'] = 100 - (100 / (1 + avg_gain / avg_loss))

    # Returns over different periods
    features['Return_5d'] = np.zeros(n_days)
    features['Return_5d'][5:] = (prices[5:] - prices[:-5]) / prices[:-5]
    features['Return_20d'] = np.zeros(n_days)
    features['Return_20d'][20:] = (prices[20:] - prices[:-20]) / prices[:-20]

    # Volatility
    features['Volatility_10d'] = np.array([
        np.std(returns[max(0,i-10):i+1]) if i >= 10 else 0.015
        for i in range(n_days)
    ])
    vol_20 = np.array([
        np.std(returns[max(0,i-20):i+1]) if i >= 20 else 0.015
        for i in range(n_days)
    ])
    vol_60 = np.array([
        np.std(returns[max(0,i-60):i+1]) if i >= 60 else 0.015
        for i in range(n_days)
    ])
    vol_60[vol_60 == 0] = 1e-10
    features['Volatility_ratio'] = vol_20 / vol_60

    # Bollinger Band position
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
    bb_std = np.array([np.std(prices[max(0,i-20):i+1]) for i in range(n_days)])
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_range[bb_range == 0] = 1e-10
    features['BB_pct'] = np.clip((prices - bb_lower) / bb_range, 0, 1)

    # Volume z-score
    vol_sma = np.convolve(volumes, np.ones(20)/20, mode='same')
    vol_std = np.array([np.std(volumes[max(0,i-20):i+1]) for i in range(n_days)])
    vol_std[vol_std == 0] = 1e-10
    features['Volume_zscore'] = (volumes - vol_sma) / vol_std

    # Trend slope (simplified)
    features['Trend_slope'] = np.zeros(n_days)
    for i in range(20, n_days):
        x = np.arange(20)
        y_vals = prices[i-20:i]
        features['Trend_slope'][i] = np.polyfit(x, y_vals, 1)[0] / prices[i]

    # High-Low range
    features['High_low_range'] = (highs - lows) / prices

    # MACD histogram (simplified)
    ema_12 = np.convolve(prices, np.ones(12)/12, mode='same')
    ema_26 = np.convolve(prices, np.ones(26)/26, mode='same')
    macd = ema_12 - ema_26
    signal = np.convolve(macd, np.ones(9)/9, mode='same')
    features['MACD_hist'] = (macd - signal) / prices

    feature_names = list(features.keys())
    X = np.column_stack([features[f] for f in feature_names])

    # Target: next day direction
    y = np.zeros(n_days)
    y[:-1] = np.where(returns[1:] > 0, 1, -1)
    y[-1] = 1

    # Remove warmup period
    warmup = 60
    return X[warmup:], y[warmup:], feature_names, returns[warmup:], prices[warmup:]


# =============================================================================
# Training and Evaluation
# =============================================================================

# Generate data
X, y, feature_names, returns, prices = generate_stock_pattern_data(n_days=3000)
n = len(X)

print("=" * 65)
print("KNN Historical Stock Pattern Matching")
print("=" * 65)
print(f"\nDataset: {n} trading days, {len(feature_names)} features")
print(f"Features: {feature_names}")
print(f"Class distribution: Up={int(np.sum(y==1))}, Down={int(np.sum(y==-1))}")

# Time-series split
split = int(n * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
returns_test = returns[split+1:split+1+len(X_test)]  # next-day returns
if len(returns_test) < len(X_test):
    returns_test = np.append(returns_test, [0] * (len(X_test) - len(returns_test)))

time_train = np.arange(split)
time_test = np.arange(split, split + len(X_test))

print(f"\nTrain: {len(X_train)} days | Test: {len(X_test)} days")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Test different K values ---
print("\n--- K Value Optimization ---")
k_values = [5, 10, 15, 20, 30, 50]
best_k = 5
best_acc = 0

for k in k_values:
    knn = KNNClassifier(k=k, weights='distance', metric='euclidean')
    knn.fit(X_train_scaled, y_train, time_train)
    preds = knn.predict(X_test_scaled, time_test)
    acc = np.mean(preds == y_test)
    if acc > best_acc:
        best_acc = acc
        best_k = k
    print(f"  K={k:>3d}: Test accuracy = {acc:.4f}")

print(f"\nBest K = {best_k} (accuracy = {best_acc:.4f})")

# --- Train final model with best K ---
print(f"\n--- Final Model (K={best_k}, distance-weighted, temporal decay) ---")
knn_final = KNNClassifier(k=best_k, weights='distance', metric='euclidean',
                          temporal_decay=0.001)
knn_final.fit(X_train_scaled, y_train, time_train)

# Predictions with probabilities
test_preds = knn_final.predict(X_test_scaled, time_test)
test_probas = knn_final.predict_proba(X_test_scaled, time_test)

train_preds = knn_final.predict(X_train_scaled, time_train)
train_acc = np.mean(train_preds == y_train)
test_acc = np.mean(test_preds == y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# --- Confidence analysis ---
print("\n--- Confidence-Stratified Performance ---")
max_proba = test_probas.max(axis=1)
confidence_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
print(f"{'Confidence':>12} {'Accuracy':>10} {'Count':>8} {'Pct':>6}")
for low, high in confidence_bins:
    mask = (max_proba >= low) & (max_proba < high)
    if mask.sum() > 0:
        acc = np.mean(test_preds[mask] == y_test[mask])
        pct = mask.sum() / len(y_test) * 100
        print(f"  [{low:.1f}, {high:.1f}): {acc:>10.4f} {mask.sum():>8d} {pct:>5.1f}%")

# --- Anomaly detection ---
print("\n--- Anomaly Detection (Unprecedented Market Conditions) ---")
avg_distances = knn_final.average_neighbor_distance(X_test_scaled, time_test)
anomaly_threshold = np.percentile(avg_distances, 95)
anomalous_days = avg_distances > anomaly_threshold
print(f"Anomaly threshold (95th pctl): {anomaly_threshold:.4f}")
print(f"Anomalous days: {anomalous_days.sum()} / {len(X_test)} "
      f"({anomalous_days.sum()/len(X_test)*100:.1f}%)")

# Accuracy on normal vs anomalous days
if anomalous_days.sum() > 0 and (~anomalous_days).sum() > 0:
    normal_acc = np.mean(test_preds[~anomalous_days] == y_test[~anomalous_days])
    anomal_acc = np.mean(test_preds[anomalous_days] == y_test[anomalous_days])
    print(f"Accuracy on normal days:    {normal_acc:.4f}")
    print(f"Accuracy on anomalous days: {anomal_acc:.4f}")

# --- Trading simulation ---
print("\n--- Pattern-Based Trading Simulation ---")
confidence_threshold = 0.60
positions = np.zeros(len(y_test))

for i in range(len(y_test)):
    if anomalous_days[i]:
        positions[i] = 0  # No trade on unprecedented days
    elif max_proba[i] > confidence_threshold:
        direction = 1 if test_preds[i] == 1 else -1
        # Scale by confidence minus threshold
        positions[i] = direction * (max_proba[i] - 0.5) * 2
    else:
        positions[i] = 0  # No trade when uncertain

strategy_returns = positions * returns_test
active_days = np.sum(positions != 0)

cum_strategy = np.cumprod(1 + strategy_returns)
cum_buyhold = np.cumprod(1 + returns_test)

print(f"Active trading days: {active_days} / {len(y_test)} "
      f"({active_days/len(y_test)*100:.1f}%)")
print(f"Strategy cumulative return: {(cum_strategy[-1]-1)*100:.2f}%")
print(f"Buy & Hold cumulative return: {(cum_buyhold[-1]-1)*100:.2f}%")

if active_days > 0:
    active_returns = strategy_returns[positions != 0]
    sharpe = np.sqrt(252) * np.mean(active_returns) / np.std(active_returns)
    print(f"Strategy Sharpe (active days): {sharpe:.2f}")

    max_dd = (1 - cum_strategy / np.maximum.accumulate(cum_strategy)).max()
    print(f"Max drawdown: {max_dd*100:.2f}%")

# --- Inspect nearest neighbors for a sample day ---
print("\n--- Example: Nearest Neighbors for Test Day 0 ---")
neighbors = knn_final.get_neighbors_detail(X_test_scaled[0],
                                           query_time_idx=time_test[0])
print(f"Query day features (scaled): {X_test_scaled[0][:4].round(3)}...")
print(f"True next-day direction: {'UP' if y_test[0] == 1 else 'DOWN'}")
print(f"\n{'Rank':>5} {'Train Day':>10} {'Distance':>10} {'Direction':>10}")
for rank, n_info in enumerate(neighbors[:10], 1):
    direction = 'UP' if n_info['label'] == 1 else 'DOWN'
    print(f"{rank:>5} {int(n_info['train_index']):>10} "
          f"{n_info['distance']:>10.4f} {direction:>10}")

# --- Compare distance metrics ---
print("\n--- Distance Metric Comparison ---")
for metric in ['euclidean', 'manhattan', 'cosine']:
    knn_metric = KNNClassifier(k=best_k, weights='distance', metric=metric)
    knn_metric.fit(X_train_scaled, y_train, time_train)
    preds = knn_metric.predict(X_test_scaled, time_test)
    acc = np.mean(preds == y_test)
    print(f"  {metric:>12}: Test accuracy = {acc:.4f}")

print("\n--- Feature Importance (Leave-One-Out) ---")
baseline_acc = test_acc
for f_idx, f_name in enumerate(feature_names):
    X_test_modified = X_test_scaled.copy()
    X_test_modified[:, f_idx] = 0  # Zero out feature
    preds_mod = knn_final.predict(X_test_modified, time_test)
    acc_mod = np.mean(preds_mod == y_test)
    importance = baseline_acc - acc_mod
    bar = '#' * int(max(0, importance) * 500)
    print(f"  {f_name:>20}: drop = {importance:+.4f} {bar}")
```

## Key Takeaways

1. **KNN is the algorithm that most naturally captures the trader's intuition of pattern matching.** When a trader says "the last five times the market looked like this, it rallied," they are performing a mental KNN classification. The algorithm formalizes this intuition with precise distance calculations and provides transparent justification through the actual historical analogs.

2. **Feature engineering and dimensionality reduction are more important than algorithm tuning for KNN.** The quality of the feature set determines the quality of the "similarity" definition. Use 6-10 diverse, well-scaled features covering different market aspects. PCA or manual feature selection is essential to avoid the curse of dimensionality.

3. **Distance-weighted voting consistently outperforms uniform voting** for financial data. Closer historical analogs (more similar market conditions) should have more influence on the prediction. Additionally, temporal weighting that favors recent analogs helps adapt to evolving market dynamics.

4. **KNN's anomaly detection capability is a unique and underutilized feature.** When the average distance to the K nearest neighbors is unusually large, the current market condition has no close historical precedent. This is a valuable early warning signal for unprecedented events where all model-based predictions should be treated with skepticism.

5. **Confidence-based position sizing significantly improves trading performance.** Only trade when the KNN prediction has high confidence (strong agreement among neighbors) and abstain during ambiguous periods. This naturally reduces trading frequency but improves the win rate and risk-adjusted returns.

6. **KNN requires no retraining when new data arrives** -- simply append new observations to the training set. This makes it operationally simple for live trading systems but means the training set grows over time, requiring periodic pruning or approximate nearest neighbor methods for efficiency.

7. **KNN works best as a complementary signal alongside parametric models.** Its non-parametric, local nature captures patterns that global models miss, while parametric models handle extrapolation and high-dimensional patterns better. Combining KNN pattern-matching scores with model-based predictions in an ensemble often produces superior results.
