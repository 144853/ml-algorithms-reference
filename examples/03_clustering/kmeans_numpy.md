# K-Means Clustering - Complete Guide with Stock Market Applications

## Overview

K-Means clustering is one of the most widely used unsupervised learning algorithms in machine learning. It partitions a dataset into K distinct, non-overlapping groups (clusters) where each data point belongs to the cluster with the nearest centroid. The algorithm works by iteratively assigning data points to clusters and updating cluster centers until convergence is reached. In the stock market context, K-Means is invaluable for grouping stocks that exhibit similar trading behavior, enabling portfolio managers to identify natural groupings that may not align with traditional sector classifications.

The algorithm operates on the principle of minimizing within-cluster variance, also known as inertia. Starting with K randomly initialized centroids, K-Means alternates between two steps: assigning each data point to the nearest centroid (assignment step) and recalculating centroids as the mean of all points assigned to each cluster (update step). This process repeats until the assignments no longer change or a maximum number of iterations is reached. The simplicity and scalability of K-Means make it particularly attractive for financial applications where datasets can contain thousands of stocks with dozens of features.

In stock market analysis, K-Means clustering helps traders and portfolio managers discover hidden patterns in trading behavior. Rather than relying solely on industry classifications (technology, healthcare, finance), K-Means can reveal that certain stocks from different industries actually trade very similarly in terms of volatility, momentum, and volume patterns. This insight is crucial for building truly diversified portfolios and understanding cross-sector correlations that traditional classification misses.

## How It Works - The Math Behind It

### Objective Function

K-Means minimizes the Within-Cluster Sum of Squares (WCSS), also called inertia:

```
J = sum_{i=1}^{K} sum_{x in C_i} ||x - mu_i||^2
```

Where:
- `K` is the number of clusters
- `C_i` is the set of points in cluster i
- `mu_i` is the centroid (mean) of cluster i
- `||x - mu_i||^2` is the squared Euclidean distance between point x and centroid mu_i

### Step-by-Step Process

**Step 1: Initialization**
Randomly select K data points as initial centroids, or use K-Means++ for smarter initialization:
```
mu_1, mu_2, ..., mu_K = random selection from dataset X
```

**Step 2: Assignment Step**
For each data point x_n, assign it to the nearest centroid:
```
c_n = argmin_{k} ||x_n - mu_k||^2
```

**Step 3: Update Step**
Recalculate each centroid as the mean of all points assigned to it:
```
mu_k = (1 / |C_k|) * sum_{x_n in C_k} x_n
```

**Step 4: Convergence Check**
Repeat Steps 2 and 3 until:
- Centroids no longer move significantly: `||mu_k^(t+1) - mu_k^(t)|| < epsilon`
- Or maximum iterations are reached

### K-Means++ Initialization

Standard K-Means is sensitive to initialization. K-Means++ selects initial centroids that are spread apart:

1. Choose the first centroid uniformly at random from the data points
2. For each subsequent centroid, choose a data point with probability proportional to its squared distance from the nearest existing centroid:
```
P(x) = D(x)^2 / sum_{x' in X} D(x')^2
```
3. Repeat until K centroids are chosen

### Distance Metric

The standard Euclidean distance for d-dimensional data:
```
d(x, y) = sqrt(sum_{j=1}^{d} (x_j - y_j)^2)
```

For stock market features with different scales, data must be standardized first:
```
x_standardized = (x - mean(x)) / std(x)
```

## Stock Market Use Case: Grouping Stocks by Trading Behavior

### The Problem

A quantitative hedge fund manages a portfolio of 50 stocks across multiple sectors. The fund wants to group these stocks not by their industry classification, but by their actual trading behavior -- volatility patterns, return profiles, and volume characteristics. This behavioral clustering helps identify stocks that truly move differently from each other, enabling better diversification and risk management. Traditional sector classifications often fail because a high-growth tech stock may trade more like a biotech stock than a stable enterprise software company.

### Stock Market Features (Input Data)

| Feature | Description | Range | Why It Matters |
|---------|-------------|-------|----------------|
| annualized_volatility | Standard deviation of daily returns * sqrt(252) | 0.10 - 0.80 | Measures price fluctuation intensity |
| avg_daily_return | Mean daily percentage return | -0.002 - 0.005 | Indicates growth trajectory |
| sharpe_ratio | Risk-adjusted return metric | -0.5 - 3.0 | Combines return and risk profile |
| avg_daily_volume_millions | Average shares traded per day in millions | 0.5 - 200 | Reflects liquidity and interest |
| beta | Sensitivity to market movements | 0.3 - 2.5 | Shows market correlation |
| max_drawdown | Largest peak-to-trough decline | -0.60 - -0.05 | Measures downside risk |
| momentum_30d | 30-day price momentum | -0.15 - 0.25 | Captures trend strength |
| volume_volatility | Std deviation of daily volume | 0.1 - 5.0 | Shows trading consistency |

### Example Data Structure

```python
import numpy as np

# Synthetic stock market data: 50 stocks, 8 features
np.random.seed(42)
n_stocks = 50

# Feature columns:
# [volatility, avg_return, sharpe, volume_mm, beta, max_dd, momentum_30d, vol_volatility]

# Cluster 1: Blue-chip stable stocks (low vol, moderate return, high volume)
blue_chips = np.column_stack([
    np.random.normal(0.18, 0.03, 15),    # low volatility
    np.random.normal(0.001, 0.0005, 15),  # moderate return
    np.random.normal(1.8, 0.3, 15),       # good sharpe
    np.random.normal(50, 20, 15),          # high volume
    np.random.normal(0.9, 0.15, 15),      # beta near 1
    np.random.normal(-0.15, 0.05, 15),    # small drawdown
    np.random.normal(0.03, 0.02, 15),     # low momentum
    np.random.normal(0.8, 0.2, 15),       # stable volume
])

# Cluster 2: Growth/momentum stocks (high vol, high return, high beta)
growth_stocks = np.column_stack([
    np.random.normal(0.45, 0.08, 20),     # high volatility
    np.random.normal(0.003, 0.001, 20),   # higher return
    np.random.normal(1.2, 0.4, 20),       # moderate sharpe
    np.random.normal(15, 8, 20),           # moderate volume
    np.random.normal(1.6, 0.3, 20),       # high beta
    np.random.normal(-0.35, 0.1, 20),     # larger drawdown
    np.random.normal(0.12, 0.05, 20),     # strong momentum
    np.random.normal(2.0, 0.5, 20),       # volatile volume
])

# Cluster 3: Defensive/dividend stocks (very low vol, low return, low beta)
defensive_stocks = np.column_stack([
    np.random.normal(0.12, 0.02, 15),     # very low volatility
    np.random.normal(0.0005, 0.0003, 15), # low return
    np.random.normal(1.0, 0.3, 15),       # low sharpe
    np.random.normal(8, 4, 15),            # low volume
    np.random.normal(0.5, 0.1, 15),       # low beta
    np.random.normal(-0.08, 0.03, 15),    # tiny drawdown
    np.random.normal(0.01, 0.01, 15),     # minimal momentum
    np.random.normal(0.4, 0.15, 15),      # very stable volume
])

# Combine all stocks
stock_data = np.vstack([blue_chips, growth_stocks, defensive_stocks])
stock_names = [f"STOCK_{i+1}" for i in range(n_stocks)]
feature_names = [
    'volatility', 'avg_return', 'sharpe_ratio', 'volume_mm',
    'beta', 'max_drawdown', 'momentum_30d', 'vol_volatility'
]

print(f"Dataset shape: {stock_data.shape}")
print(f"Features: {feature_names}")
```

### The Model in Action

```python
import numpy as np

def standardize(X):
    """Standardize features to zero mean and unit variance."""
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1  # prevent division by zero
    return (X - means) / stds, means, stds

def kmeans_plus_plus_init(X, K):
    """K-Means++ initialization for better starting centroids."""
    n_samples, n_features = X.shape
    centroids = np.zeros((K, n_features))

    # Choose first centroid randomly
    idx = np.random.randint(0, n_samples)
    centroids[0] = X[idx]

    for k in range(1, K):
        # Compute distances to nearest centroid
        distances = np.min([
            np.sum((X - centroids[j]) ** 2, axis=1)
            for j in range(k)
        ], axis=0)

        # Choose next centroid with probability proportional to distance^2
        probs = distances / distances.sum()
        idx = np.random.choice(n_samples, p=probs)
        centroids[k] = X[idx]

    return centroids

def kmeans(X, K, max_iters=300, tol=1e-6):
    """
    K-Means clustering implementation from scratch.

    Parameters:
        X: numpy array of shape (n_samples, n_features)
        K: number of clusters
        max_iters: maximum iterations
        tol: convergence tolerance

    Returns:
        labels: cluster assignments
        centroids: final centroid positions
        inertia: within-cluster sum of squares
        history: list of inertia values per iteration
    """
    n_samples, n_features = X.shape
    history = []

    # Initialize centroids using K-Means++
    centroids = kmeans_plus_plus_init(X, K)
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iters):
        # Assignment step: assign each point to nearest centroid
        distances = np.zeros((n_samples, K))
        for k in range(K):
            distances[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
        new_labels = np.argmin(distances, axis=1)

        # Compute inertia (WCSS)
        inertia = sum(
            np.sum(distances[new_labels == k, k])
            for k in range(K)
        )
        history.append(inertia)

        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # Update step: recalculate centroids
        old_centroids = centroids.copy()
        for k in range(K):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)

        # Check centroid movement
        centroid_shift = np.sqrt(np.sum((centroids - old_centroids) ** 2))
        if centroid_shift < tol:
            break

    return labels, centroids, inertia, history

def elbow_method(X, max_k=10):
    """Run K-Means for different K values to find optimal number of clusters."""
    inertias = []
    for k in range(1, max_k + 1):
        _, _, inertia, _ = kmeans(X, k)
        inertias.append(inertia)
    return inertias

def silhouette_score(X, labels):
    """Compute silhouette score to evaluate clustering quality."""
    n_samples = len(X)
    unique_labels = np.unique(labels)
    scores = np.zeros(n_samples)

    for i in range(n_samples):
        # a(i) = average distance to points in same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean(np.sqrt(np.sum((same_cluster - X[i]) ** 2, axis=1)))
        else:
            a_i = 0

        # b(i) = minimum average distance to points in other clusters
        b_i = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = X[labels == label]
                avg_dist = np.mean(np.sqrt(np.sum((other_cluster - X[i]) ** 2, axis=1)))
                b_i = min(b_i, avg_dist)

        # Silhouette coefficient for point i
        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    return np.mean(scores)

# Standardize the data
X_std, means, stds = standardize(stock_data)

# Run K-Means with K=3
labels, centroids, inertia, history = kmeans(X_std, K=3)

# Evaluate clustering
sil_score = silhouette_score(X_std, labels)

print(f"Final inertia (WCSS): {inertia:.2f}")
print(f"Silhouette score: {sil_score:.4f}")
print(f"Converged in {len(history)} iterations")
print(f"\nCluster distribution:")
for k in range(3):
    count = np.sum(labels == k)
    print(f"  Cluster {k}: {count} stocks")

# Interpret centroids in original scale
original_centroids = centroids * stds + means
print(f"\nCluster centroids (original scale):")
for k in range(3):
    print(f"\n  Cluster {k}:")
    for j, feat in enumerate(feature_names):
        print(f"    {feat}: {original_centroids[k, j]:.4f}")
```

## Advantages

1. **Computational Efficiency for Large Stock Universes**: K-Means has a time complexity of O(n * K * d * i) where n is the number of stocks, K is clusters, d is features, and i is iterations. This makes it feasible to cluster thousands of stocks across global markets in seconds, enabling real-time portfolio rebalancing and intraday trading behavior analysis.

2. **Intuitive Interpretability for Portfolio Managers**: Each cluster centroid represents the "average stock" in that group, making it straightforward for portfolio managers to understand what defines each cluster. A centroid with high volatility, high beta, and strong momentum clearly represents growth stocks, which is immediately actionable for investment decisions.

3. **Scalability Across Market Regimes**: K-Means can be efficiently re-run as market conditions change. During a market crash, stock behavior shifts dramatically, and K-Means can quickly re-cluster the universe to reflect new trading patterns, helping risk managers identify which stocks have changed their behavioral group.

4. **Natural Integration with Portfolio Construction**: The cluster assignments directly map to portfolio allocation decisions. By ensuring holdings are spread across different behavioral clusters (not just sectors), portfolio managers achieve genuine diversification that accounts for how stocks actually trade rather than how they are classified.

5. **Feature Flexibility**: K-Means works with any numerical features, allowing analysts to cluster stocks by fundamental data (P/E ratios, revenue growth), technical indicators (RSI, MACD), or a combination. This flexibility means the same algorithm can serve different investment strategies.

6. **Deterministic Results with K-Means++**: When using K-Means++ initialization with a fixed random seed, results are reproducible. This is crucial for regulatory compliance in finance, where investment decisions based on quantitative models must be auditable and repeatable.

7. **Easy Monitoring of Cluster Stability**: By tracking which stocks change clusters over time, analysts can detect regime changes in the market. A blue-chip stock suddenly clustering with high-volatility growth stocks may signal fundamental changes in that company's risk profile.

## Disadvantages

1. **Pre-specified K is Problematic for Markets**: The number of clusters K must be chosen in advance, but the natural number of stock behavioral groups changes with market conditions. During calm markets there might be 3-4 distinct groups, but during crises, stocks tend to correlate heavily, reducing natural clusters to 2. The elbow method and silhouette scores help but are not definitive.

2. **Sensitivity to Outliers**: Penny stocks, meme stocks, or stocks experiencing extreme events (earnings surprises, FDA announcements) can significantly distort cluster centroids. A single stock with 500% monthly return can pull an entire cluster centroid, misrepresenting the behavior of other stocks in that cluster.

3. **Assumes Spherical Clusters**: K-Means assumes clusters are roughly spherical and equally sized in feature space. Stock market behavior often forms elongated or irregularly shaped clusters (e.g., a continuum from low-vol to high-vol rather than distinct groups), which K-Means cannot capture well.

4. **Scale Sensitivity Requires Careful Normalization**: Features like volume (millions of shares) and daily return (small decimals) operate on vastly different scales. Without proper standardization, volume would dominate the distance calculations, and subtle but important return differences would be ignored. The choice of normalization method itself affects results.

5. **Static Snapshot Misses Temporal Dynamics**: K-Means clusters stocks based on a snapshot of their features, but stock behavior is inherently dynamic. A stock's volatility and momentum change continuously, meaning cluster assignments can become stale quickly. Frequent re-clustering is needed but introduces instability in downstream trading systems.

6. **Cannot Handle Non-Numeric Market Data**: Important stock characteristics like sector, exchange listing, index membership, and market cap category (small/mid/large) are categorical and cannot be directly used in K-Means. Encoding these as numeric values introduces arbitrary distance relationships that can distort clustering results.

7. **Local Optima Risk**: K-Means can converge to local optima where the clustering is suboptimal. In practice, this means running the algorithm multiple times with different initializations and selecting the best result, which adds computational overhead and non-determinism to the clustering process.

## When to Use in Stock Market

- **Portfolio diversification**: Group stocks by behavioral similarity to ensure true diversification beyond sector labels
- **Risk bucketing**: Classify stocks into risk categories (conservative, moderate, aggressive) based on volatility and drawdown metrics
- **Peer group analysis**: Identify which stocks genuinely behave like peers for relative valuation comparisons
- **Market regime detection**: Track how cluster compositions change over time to detect shifts in market behavior
- **Factor exposure grouping**: Cluster stocks by their sensitivity to common factors (value, momentum, quality, size)
- **Trading strategy segmentation**: Group stocks to apply different trading strategies (mean reversion for stable clusters, momentum for trending clusters)
- **Index construction**: Build custom indices by selecting representative stocks from each behavioral cluster
- **Watchlist organization**: Automatically organize large watchlists into meaningful behavioral groups

## When NOT to Use in Stock Market

- **When clusters are non-spherical**: If stock behaviors form elongated or chain-like patterns, use DBSCAN or spectral clustering instead
- **When outlier detection is the goal**: K-Means forces every stock into a cluster; use DBSCAN or isolation forests for anomaly detection
- **When the number of groups is truly unknown**: If you have no prior belief about how many behavioral groups exist, start with hierarchical clustering to explore the dendrogram
- **When temporal patterns matter**: K-Means ignores the order of observations; use time-series clustering methods for pattern recognition in price sequences
- **When categorical features dominate**: If sector, exchange, and other categorical features are primary, use K-Modes or K-Prototypes instead
- **When real-time streaming is needed**: K-Means requires the full dataset; use online/mini-batch variants for streaming market data
- **When cluster sizes are highly imbalanced**: If you expect one large group of "normal" stocks and several small groups of unusual ones, K-Means will tend to split the large group

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| K (n_clusters) | Number of clusters | 2 - 15 | Start with 3-5; use elbow method and silhouette score. Consider 3 (conservative/moderate/aggressive) or 5 (add income and speculative) |
| init | Initialization method | random, k-means++ | Always use k-means++ for stock data to avoid poor initialization with outlier stocks |
| max_iter | Maximum iterations | 100 - 1000 | 300 is usually sufficient; increase if using many features or large universes |
| n_init | Number of random restarts | 5 - 20 | Use 10+ for production systems; each run may find different groupings |
| tol | Convergence tolerance | 1e-4 to 1e-8 | 1e-6 provides good balance between accuracy and speed for daily clustering |
| random_state | Random seed | any integer | Fix for reproducibility in backtesting; vary for robustness checks |

## Stock Market Performance Tips

1. **Feature Engineering Matters More Than Algorithm Tuning**: Spend time creating meaningful features. Combine raw volatility with relative volatility (vs. sector average), use rolling windows of different lengths, and include both absolute and risk-adjusted return metrics. Good features make K-Means far more effective than parameter tuning.

2. **Standardize Using Robust Scalers**: Market data often has extreme outliers (earnings jumps, flash crashes). Use median and interquartile range for standardization instead of mean and standard deviation to prevent outliers from distorting the scaling.

3. **Run Multiple Initializations**: Always run K-Means at least 10 times with different random seeds and select the result with lowest inertia. Stock market data has enough noise that different initializations can produce meaningfully different clusterings.

4. **Monitor Cluster Stability**: Track what percentage of stocks change clusters week-over-week. If more than 20-30% of stocks change clusters, either your features are too noisy or market regime is shifting, both of which require attention.

5. **Use Domain Knowledge to Validate**: After clustering, verify that the clusters make financial sense. If a utility stock clusters with speculative biotech stocks, investigate whether the features need adjustment or if that stock genuinely has unusual trading characteristics.

6. **Combine with Dimensionality Reduction**: Before clustering, apply PCA to reduce correlated features. This removes noise from redundant features and often produces cleaner, more stable clusters.

7. **Time-Window Selection**: Use 60-90 trading days of data for feature calculation. Too short (< 20 days) captures noise; too long (> 252 days) blends different market regimes.

## Comparison with Other Algorithms

| Feature | K-Means | DBSCAN | Hierarchical | Gaussian Mixture |
|---------|---------|--------|--------------|-----------------|
| Requires K specified | Yes | No | Optional (cut dendrogram) | Yes |
| Handles outliers | Poor | Excellent | Moderate | Moderate |
| Cluster shapes | Spherical only | Arbitrary | Arbitrary | Elliptical |
| Scalability | Excellent | Good | Poor for large datasets | Good |
| Probabilistic membership | No | No | No | Yes |
| Stock market suitability | Good for broad grouping | Good for anomaly detection | Good for taxonomy | Best for overlapping behaviors |
| Interpretability | High (centroids) | Moderate | High (dendrogram) | Moderate (distributions) |
| Computation time | O(nKdi) | O(n log n) | O(n^2 log n) | O(nK d^2 i) |
| Handles noise | Poor | Excellent | Moderate | Good |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete K-Means Stock Clustering Pipeline
# ============================================================

np.random.seed(42)

# --- Generate Synthetic Stock Universe ---
def generate_stock_universe(n_stocks=100):
    """Generate realistic stock market feature data."""
    stocks = {}
    feature_list = []

    # Blue-chip / Large-cap stable (30 stocks)
    for i in range(30):
        features = {
            'volatility': np.random.normal(0.20, 0.04),
            'avg_daily_return': np.random.normal(0.0008, 0.0003),
            'sharpe_ratio': np.random.normal(1.5, 0.3),
            'volume_mm': np.random.normal(40, 15),
            'beta': np.random.normal(0.95, 0.12),
            'max_drawdown': np.random.normal(-0.18, 0.05),
            'momentum_30d': np.random.normal(0.02, 0.015),
            'vol_volatility': np.random.normal(0.7, 0.2),
        }
        stocks[f"BLUE_{i+1}"] = features
        feature_list.append(list(features.values()))

    # Growth / High-beta stocks (35 stocks)
    for i in range(35):
        features = {
            'volatility': np.random.normal(0.50, 0.10),
            'avg_daily_return': np.random.normal(0.0025, 0.001),
            'sharpe_ratio': np.random.normal(0.9, 0.4),
            'volume_mm': np.random.normal(12, 7),
            'beta': np.random.normal(1.7, 0.25),
            'max_drawdown': np.random.normal(-0.40, 0.12),
            'momentum_30d': np.random.normal(0.10, 0.05),
            'vol_volatility': np.random.normal(2.5, 0.7),
        }
        stocks[f"GROWTH_{i+1}"] = features
        feature_list.append(list(features.values()))

    # Defensive / Low-vol dividend stocks (20 stocks)
    for i in range(20):
        features = {
            'volatility': np.random.normal(0.12, 0.02),
            'avg_daily_return': np.random.normal(0.0004, 0.0002),
            'sharpe_ratio': np.random.normal(0.8, 0.25),
            'volume_mm': np.random.normal(6, 3),
            'beta': np.random.normal(0.45, 0.10),
            'max_drawdown': np.random.normal(-0.08, 0.03),
            'momentum_30d': np.random.normal(0.005, 0.008),
            'vol_volatility': np.random.normal(0.3, 0.1),
        }
        stocks[f"DEFENSIVE_{i+1}"] = features
        feature_list.append(list(features.values()))

    # Speculative / Meme-like stocks (15 stocks)
    for i in range(15):
        features = {
            'volatility': np.random.normal(0.80, 0.15),
            'avg_daily_return': np.random.normal(0.004, 0.003),
            'sharpe_ratio': np.random.normal(0.5, 0.5),
            'volume_mm': np.random.normal(80, 40),
            'beta': np.random.normal(2.2, 0.4),
            'max_drawdown': np.random.normal(-0.55, 0.15),
            'momentum_30d': np.random.normal(0.15, 0.10),
            'vol_volatility': np.random.normal(4.0, 1.0),
        }
        stocks[f"SPEC_{i+1}"] = features
        feature_list.append(list(features.values()))

    X = np.array(feature_list)
    names = list(stocks.keys())
    feat_names = list(list(stocks.values())[0].keys())
    return X, names, feat_names

# --- K-Means Implementation ---
def standardize(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1
    return (X - means) / stds, means, stds

def kmeans_pp_init(X, K):
    n = X.shape[0]
    centroids = [X[np.random.randint(n)]]
    for _ in range(1, K):
        dists = np.array([
            min(np.sum((x - c) ** 2) for c in centroids) for x in X
        ])
        probs = dists / dists.sum()
        centroids.append(X[np.random.choice(n, p=probs)])
    return np.array(centroids)

def run_kmeans(X, K, max_iters=300, tol=1e-6):
    n = X.shape[0]
    centroids = kmeans_pp_init(X, K)
    labels = np.zeros(n, dtype=int)
    inertia_history = []

    for it in range(max_iters):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]).T
        new_labels = np.argmin(dists, axis=1)
        inertia = sum(dists[i, new_labels[i]] for i in range(n))
        inertia_history.append(inertia)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for k in range(K):
            mask = labels == k
            if np.sum(mask) > 0:
                centroids[k] = X[mask].mean(axis=0)

    return labels, centroids, inertia, inertia_history

def compute_silhouette(X, labels):
    n = len(X)
    unique = np.unique(labels)
    scores = np.zeros(n)
    for i in range(n):
        same = X[labels == labels[i]]
        a = np.mean(np.sqrt(np.sum((same - X[i]) ** 2, axis=1))) if len(same) > 1 else 0
        b = min(
            np.mean(np.sqrt(np.sum((X[labels == l] - X[i]) ** 2, axis=1)))
            for l in unique if l != labels[i]
        )
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return np.mean(scores)

# --- Run Pipeline ---
X_raw, stock_names, feature_names = generate_stock_universe()
X_std, means, stds = standardize(X_raw)

# Find optimal K using elbow method
print("=== Elbow Method ===")
for k in range(2, 8):
    _, _, inertia, _ = run_kmeans(X_std, k)
    sil = compute_silhouette(X_std, run_kmeans(X_std, k)[0])
    print(f"  K={k}: Inertia={inertia:.1f}, Silhouette={sil:.4f}")

# Run with optimal K=4
print("\n=== K-Means with K=4 ===")
labels, centroids, inertia, history = run_kmeans(X_std, K=4)

# Analyze clusters
print(f"\nFinal Inertia: {inertia:.2f}")
print(f"Converged in {len(history)} iterations")
print(f"Silhouette Score: {compute_silhouette(X_std, labels):.4f}")

# Cluster profiles in original scale
orig_centroids = centroids * stds + means
print("\n=== Cluster Profiles ===")
for k in range(4):
    members = [stock_names[i] for i in range(len(labels)) if labels[i] == k]
    print(f"\nCluster {k} ({len(members)} stocks):")
    print(f"  Sample members: {members[:5]}")
    for j, feat in enumerate(feature_names):
        print(f"  {feat:>20s}: {orig_centroids[k, j]:>10.4f}")

# Portfolio diversification check
print("\n=== Portfolio Diversification Analysis ===")
for k in range(4):
    count = np.sum(labels == k)
    pct = count / len(labels) * 100
    avg_vol = np.mean(X_raw[labels == k, 0])
    avg_beta = np.mean(X_raw[labels == k, 4])
    print(f"Cluster {k}: {count:>3d} stocks ({pct:>5.1f}%) | "
          f"Avg Vol: {avg_vol:.3f} | Avg Beta: {avg_beta:.2f}")
```

## Key Takeaways

1. **K-Means groups stocks by behavioral similarity**, not industry classification, revealing hidden relationships between stocks from different sectors that trade in similar patterns.

2. **Feature standardization is mandatory** for stock market data because features like volume and daily returns operate on completely different scales. Skipping standardization produces meaningless clusters dominated by the largest-scale feature.

3. **The optimal K depends on the investment objective**. For broad risk bucketing, K=3 (conservative/moderate/aggressive) works well. For detailed portfolio construction, K=5-7 captures more nuanced behavioral groups.

4. **K-Means++ initialization** significantly improves results for stock data by avoiding the problem of initial centroids being placed among outlier stocks.

5. **Cluster stability over time** is as important as cluster quality at a single point. Stocks that frequently change clusters may warrant special attention as their risk profile is unstable.

6. **Combine K-Means with the silhouette score** to validate that clusters are well-separated. In stock markets, a silhouette score above 0.3 indicates meaningful behavioral groups.

7. **Re-cluster periodically** (weekly or monthly) to capture evolving market dynamics. The clusters from a bull market will not accurately represent stock behavior during a bear market.

8. **Use cluster membership as a feature** in downstream models. A stock's cluster assignment can serve as a categorical input to return prediction models, encoding behavioral similarity information.
