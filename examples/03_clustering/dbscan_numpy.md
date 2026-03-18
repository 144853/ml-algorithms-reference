# DBSCAN Clustering - Complete Guide with Stock Market Applications

## Overview

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that discovers clusters of arbitrary shape by identifying regions of high point density separated by regions of low density. Unlike K-Means, DBSCAN does not require the number of clusters to be specified in advance and can identify points that do not belong to any cluster, labeling them as noise. This makes it exceptionally well-suited for stock market analysis where detecting anomalous trading days and unusual market behavior is just as important as grouping normal patterns.

The algorithm works by defining two parameters: epsilon (eps), the maximum distance between two points to be considered neighbors, and minPts, the minimum number of points required to form a dense region. A point is classified as a core point if it has at least minPts neighbors within its eps-neighborhood, a border point if it is within eps of a core point but has fewer than minPts neighbors, or a noise point if it is neither. Clusters are formed by connecting core points that are within eps distance of each other, along with their border points.

In stock market contexts, DBSCAN excels at identifying unusual trading day clusters -- groups of days with similar anomalous behavior such as flash crashes, short squeezes, or coordinated selloffs. Traditional clustering methods would force these anomalous days into normal clusters, masking their unusual nature. DBSCAN naturally separates them as noise or as small distinct clusters, providing risk managers with clear signals about market stress events.

## How It Works - The Math Behind It

### Core Definitions

**Eps-Neighborhood**: The set of all points within distance eps of a point p:
```
N_eps(p) = {q in D : dist(p, q) <= eps}
```

**Core Point**: A point p is a core point if:
```
|N_eps(p)| >= minPts
```

**Directly Density-Reachable**: A point q is directly density-reachable from p if:
```
q in N_eps(p) AND |N_eps(p)| >= minPts
```

**Density-Reachable**: A point q is density-reachable from p if there exists a chain of points p1, p2, ..., pn where p1 = p and pn = q, and each pi+1 is directly density-reachable from pi.

**Density-Connected**: Two points p and q are density-connected if there exists a point o such that both p and q are density-reachable from o.

### Step-by-Step Algorithm

```
1. For each unvisited point p in dataset:
   a. Mark p as visited
   b. Find all neighbors N_eps(p) within distance eps
   c. If |N_eps(p)| < minPts:
      - Mark p as NOISE (may later become border point)
   d. Else (p is a core point):
      - Create new cluster C
      - Add p to C
      - Create seed set S = N_eps(p)
      - For each point q in S:
        - If q is not yet visited:
          - Mark q as visited
          - Find N_eps(q)
          - If |N_eps(q)| >= minPts:
            - Add N_eps(q) to S (expand cluster)
        - If q is not yet member of any cluster:
          - Add q to C
```

### Distance Metrics

**Euclidean Distance** (default):
```
d(p, q) = sqrt(sum_{i=1}^{d} (p_i - q_i)^2)
```

**Manhattan Distance** (useful for financial features with different scales):
```
d(p, q) = sum_{i=1}^{d} |p_i - q_i|
```

**Mahalanobis Distance** (accounts for feature correlations):
```
d(p, q) = sqrt((p - q)^T * S^{-1} * (p - q))
```
Where S is the covariance matrix.

### Complexity

- **Time Complexity**: O(n^2) in the worst case, O(n log n) with spatial indexing (e.g., KD-tree)
- **Space Complexity**: O(n) for storing labels and visited status

## Stock Market Use Case: Detecting Unusual Trading Day Clusters and Market Anomaly Groups

### The Problem

A market surveillance team at a major exchange needs to identify unusual trading days that may indicate market manipulation, flash crashes, or systemic stress events. Each trading day is characterized by multiple market-wide metrics: aggregate volatility, volume spikes, breadth indicators, and cross-asset correlations. Normal trading days should cluster together, while anomalous days should be flagged as noise or form small unusual clusters. The team needs an algorithm that does not force every day into a "normal" category and can detect anomalies of varying types automatically.

### Stock Market Features (Input Data)

| Feature | Description | Typical Range | Anomaly Indicator |
|---------|-------------|---------------|-------------------|
| vix_level | VIX index closing value | 12 - 80 | > 35 signals extreme fear |
| market_return | S&P 500 daily return | -0.12 to 0.10 | |return| > 0.03 is unusual |
| advance_decline_ratio | Advancing vs declining stocks | 0.2 - 5.0 | < 0.3 or > 4.0 extreme |
| volume_ratio | Volume vs 20-day average | 0.5 - 4.0 | > 2.5 signals panic/euphoria |
| sector_dispersion | Cross-sector return std | 0.002 - 0.05 | > 0.03 signals rotation |
| high_low_ratio | New highs / (new highs + new lows) | 0.0 - 1.0 | < 0.1 or > 0.9 extreme |
| correlation_spike | Avg pairwise stock correlation | 0.1 - 0.9 | > 0.7 signals systemic risk |
| put_call_ratio | Put volume / Call volume | 0.5 - 2.0 | > 1.5 signals hedging rush |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_days = 500  # ~2 years of trading days

# Normal trading days (vast majority)
normal_days = np.column_stack([
    np.random.normal(16, 3, 430),          # VIX: calm markets
    np.random.normal(0.0004, 0.008, 430),  # small positive returns
    np.random.normal(1.5, 0.4, 430),       # balanced breadth
    np.random.normal(1.0, 0.2, 430),       # normal volume
    np.random.normal(0.008, 0.003, 430),   # low sector dispersion
    np.random.normal(0.55, 0.15, 430),     # balanced high-low
    np.random.normal(0.35, 0.08, 430),     # normal correlation
    np.random.normal(0.85, 0.15, 430),     # normal put/call
])

# Market crash days (rare cluster)
crash_days = np.column_stack([
    np.random.normal(45, 8, 15),           # VIX spike
    np.random.normal(-0.04, 0.015, 15),    # large negative returns
    np.random.normal(0.25, 0.08, 15),      # almost all stocks down
    np.random.normal(3.0, 0.5, 15),        # massive volume
    np.random.normal(0.035, 0.008, 15),    # high dispersion
    np.random.normal(0.08, 0.04, 15),      # almost no new highs
    np.random.normal(0.75, 0.06, 15),      # correlated selling
    np.random.normal(1.6, 0.2, 15),        # heavy put buying
])

# Melt-up / euphoria days (rare cluster)
euphoria_days = np.column_stack([
    np.random.normal(11, 1.5, 20),         # very low VIX
    np.random.normal(0.025, 0.008, 20),    # large positive returns
    np.random.normal(4.0, 0.6, 20),        # almost all stocks up
    np.random.normal(2.2, 0.4, 20),        # elevated volume
    np.random.normal(0.005, 0.002, 20),    # low dispersion (all rising)
    np.random.normal(0.92, 0.04, 20),      # almost all new highs
    np.random.normal(0.50, 0.05, 20),      # moderate correlation
    np.random.normal(0.55, 0.08, 20),      # low put/call
])

# Flash crash events (individual anomalies)
flash_crashes = np.column_stack([
    np.random.normal(60, 10, 5),           # extreme VIX
    np.random.normal(-0.08, 0.02, 5),      # massive drops
    np.random.normal(0.10, 0.03, 5),       # almost nothing advancing
    np.random.normal(3.8, 0.3, 5),         # extreme volume
    np.random.normal(0.04, 0.01, 5),       # very high dispersion
    np.random.normal(0.02, 0.01, 5),       # no new highs
    np.random.normal(0.85, 0.04, 5),       # extreme correlation
    np.random.normal(1.9, 0.1, 5),         # panic put buying
])

# Combine and shuffle
trading_data = np.vstack([normal_days, crash_days, euphoria_days, flash_crashes])
true_labels = np.array(
    [0]*430 + [1]*15 + [2]*20 + [3]*5
)

# Shuffle
shuffle_idx = np.random.permutation(len(trading_data))
trading_data = trading_data[shuffle_idx]
true_labels = true_labels[shuffle_idx]

day_labels = [f"Day_{i+1}" for i in range(len(trading_data))]
feature_names = [
    'vix', 'market_return', 'adv_dec_ratio', 'volume_ratio',
    'sector_dispersion', 'high_low_ratio', 'correlation', 'put_call_ratio'
]

print(f"Dataset: {trading_data.shape[0]} trading days, {trading_data.shape[1]} features")
print(f"True distribution: Normal={np.sum(true_labels==0)}, "
      f"Crash={np.sum(true_labels==1)}, Euphoria={np.sum(true_labels==2)}, "
      f"Flash={np.sum(true_labels==3)}")
```

### The Model in Action

```python
import numpy as np

def standardize(X):
    """Standardize features using median and IQR for robustness."""
    medians = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1
    return (X - medians) / iqr, medians, iqr

def euclidean_distances(X):
    """Compute pairwise Euclidean distance matrix."""
    n = X.shape[0]
    # Using broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = np.sum(X ** 2, axis=1)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T
    dist_sq = np.maximum(dist_sq, 0)  # numerical stability
    return np.sqrt(dist_sq)

def dbscan(X, eps, min_pts):
    """
    DBSCAN clustering implementation from scratch.

    Parameters:
        X: numpy array of shape (n_samples, n_features)
        eps: maximum neighborhood distance
        min_pts: minimum points to form dense region

    Returns:
        labels: cluster assignments (-1 for noise)
        n_clusters: number of clusters found
        core_mask: boolean mask of core points
    """
    n = X.shape[0]
    labels = np.full(n, -1)  # -1 = unassigned / noise
    visited = np.zeros(n, dtype=bool)
    core_mask = np.zeros(n, dtype=bool)

    # Precompute distance matrix
    dist_matrix = euclidean_distances(X)

    # Identify core points
    for i in range(n):
        neighbors = np.where(dist_matrix[i] <= eps)[0]
        if len(neighbors) >= min_pts:
            core_mask[i] = True

    cluster_id = -1

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = np.where(dist_matrix[i] <= eps)[0]

        if len(neighbors) < min_pts:
            # Noise point (may become border later)
            continue

        # Start new cluster
        cluster_id += 1
        labels[i] = cluster_id

        # Expand cluster
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = np.where(dist_matrix[q] <= eps)[0]

                if len(q_neighbors) >= min_pts:
                    # q is a core point, expand
                    for n_idx in q_neighbors:
                        if n_idx not in seed_set:
                            seed_set.append(n_idx)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

    n_clusters = cluster_id + 1
    return labels, n_clusters, core_mask

def estimate_eps(X, k=4):
    """
    Estimate eps using k-distance graph method.
    Sort k-th nearest neighbor distances and look for the 'knee'.
    """
    dist_matrix = euclidean_distances(X)
    k_distances = []
    for i in range(len(X)):
        sorted_dists = np.sort(dist_matrix[i])
        k_distances.append(sorted_dists[k])  # k-th nearest neighbor

    k_distances = np.sort(k_distances)
    return k_distances

# Standardize the trading data
X_std, medians, iqrs = standardize(trading_data)

# Estimate eps using k-distance graph
k_dists = estimate_eps(X_std, k=5)
print("K-distance values at various percentiles:")
for p in [80, 85, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(k_dists, p):.3f}")

# Run DBSCAN
eps_value = np.percentile(k_dists, 90)
labels, n_clusters, core_mask = dbscan(X_std, eps=eps_value, min_pts=5)

# Results
n_noise = np.sum(labels == -1)
print(f"\nDBSCAN Results:")
print(f"  Epsilon: {eps_value:.3f}")
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
print(f"  Core points: {np.sum(core_mask)}")

for c in range(n_clusters):
    mask = labels == c
    count = np.sum(mask)
    print(f"\n  Cluster {c} ({count} days):")
    for j, feat in enumerate(feature_names):
        cluster_mean = np.mean(trading_data[mask, j])
        print(f"    {feat:>20s}: {cluster_mean:.4f}")

# Analyze noise points (anomalies)
noise_mask = labels == -1
if np.sum(noise_mask) > 0:
    print(f"\n  Noise / Anomalous Days ({np.sum(noise_mask)} days):")
    for j, feat in enumerate(feature_names):
        noise_mean = np.mean(trading_data[noise_mask, j])
        normal_mean = np.mean(trading_data[~noise_mask, j])
        print(f"    {feat:>20s}: Noise={noise_mean:.4f} vs Normal={normal_mean:.4f}")
```

## Advantages

1. **Automatic Anomaly Detection**: DBSCAN naturally identifies noise points -- trading days that do not fit any cluster pattern. In market surveillance, these are exactly the days that warrant investigation: flash crashes, short squeezes, or insider-trading-driven moves. No other clustering algorithm provides this anomaly detection capability as a built-in feature.

2. **No Need to Specify Number of Clusters**: Unlike K-Means, DBSCAN discovers the number of clusters automatically from the data. This is crucial for market analysis because the number of distinct market regimes is genuinely unknown and changes over time. Forcing an analyst to guess K introduces bias.

3. **Arbitrary Cluster Shapes**: Stock market behavior often forms non-spherical clusters. For example, "stressed market" days may form an elongated cluster along the VIX-correlation axis. DBSCAN captures these natural shapes without assuming spherical geometry, producing more accurate groupings of market conditions.

4. **Robust to Outliers by Design**: Extreme market events (circuit breakers triggered, overnight gaps of 10%+) are automatically classified as noise rather than distorting cluster centroids. This prevents a single Black Monday event from pulling the "normal market decline" cluster toward extreme values, which would happen with K-Means.

5. **Density-Based Intuition Matches Market Reality**: Financial markets have dense regions of "normal" behavior and sparse regions of "unusual" behavior. DBSCAN's density-based approach aligns perfectly with this structure, making the algorithm's assumptions a natural fit for market data.

6. **Hierarchical Density Structure**: By varying epsilon, analysts can explore market behavior at different granularity levels -- tight eps reveals fine-grained trading patterns (distinguishing mild corrections from sharp selloffs), while loose eps reveals broad regimes (bull vs. bear).

7. **No Centroid Bias**: DBSCAN does not compute cluster means/centroids, so clusters are defined by their actual point distribution rather than a single summary statistic. This preserves the full richness of market regime descriptions.

## Disadvantages

1. **Epsilon Selection is Critical and Difficult**: The eps parameter fundamentally determines what DBSCAN considers "nearby." In stock market data, the appropriate neighborhood distance depends on feature scaling, market volatility regime, and the specific time period. Too small an eps labels everything as noise; too large merges distinct market regimes. The k-distance graph helps but still requires subjective judgment.

2. **Varying Density Problem**: Stock market data often has clusters of varying density. Normal trading days form a very dense cluster, while crisis periods form sparser clusters. DBSCAN with a single eps may correctly cluster normal days but split crisis days into many small clusters or label them as noise. HDBSCAN addresses this but adds complexity.

3. **High-Dimensional Sensitivity**: With many market features (10+ indicators), the concept of distance becomes less meaningful due to the curse of dimensionality. In high dimensions, all points tend to be roughly equidistant, making DBSCAN's distance threshold less effective. Dimensionality reduction (PCA) before DBSCAN is often necessary.

4. **Computational Cost for Large Datasets**: Without spatial indexing, DBSCAN requires computing an O(n^2) distance matrix. For tick-level or minute-level market data with millions of points, this becomes prohibitively expensive. Even with KD-trees, performance degrades in high dimensions.

5. **Non-Deterministic Border Points**: When a border point is equidistant from two clusters, its assignment depends on processing order, introducing non-determinism. In a market surveillance context, this means the same trading day might be classified differently across runs, complicating reproducibility requirements.

6. **Sensitivity to Feature Scaling**: DBSCAN uses a single distance threshold for all feature dimensions simultaneously. If VIX (range 10-80) and daily return (range -0.05 to 0.05) are not properly scaled, VIX dominates the distance calculation entirely, and DBSCAN effectively clusters only on VIX level.

7. **Difficulty with Gradual Transitions**: Markets often transition gradually between regimes (slow shift from bull to bear). DBSCAN may connect these transitional days to the wrong cluster or label them as noise, since there is no clear density gap during gradual transitions.

## When to Use in Stock Market

- **Market anomaly detection**: Identifying unusual trading days that deviate from normal market patterns (flash crashes, melt-ups, short squeezes)
- **Regime discovery**: Finding natural market regimes without predefining how many exist
- **Surveillance and compliance**: Detecting clusters of suspicious trading activity across multiple securities
- **Event clustering**: Grouping market events by their characteristic signatures (earnings seasons, FOMC announcements, geopolitical shocks)
- **Outlier trading session identification**: Flagging individual sessions where market microstructure breaks down
- **Cross-asset anomaly detection**: Finding unusual days across equities, bonds, commodities, and currencies simultaneously
- **Order flow anomaly detection**: Clustering order book patterns to detect manipulation or unusual institutional activity
- **Tail risk analysis**: Identifying and characterizing the different types of extreme market events

## When NOT to Use in Stock Market

- **When you need exactly K groups**: If portfolio construction requires exactly 5 risk buckets, use K-Means. DBSCAN may find 3 or 7 clusters depending on parameters
- **When all data must be assigned**: DBSCAN labels outliers as noise (-1). If every trading day must belong to a regime for a systematic strategy, noise points create gaps
- **When clusters have very different densities**: Calm market clusters are much denser than crisis clusters, causing DBSCAN to underperform with a single eps. Consider HDBSCAN instead
- **When working with very high-dimensional data**: With 50+ technical indicators, distance-based clustering degrades. Apply PCA first to reduce dimensions
- **When computational speed is critical**: For real-time intraday analysis with millions of data points, DBSCAN's O(n^2) complexity may be too slow
- **When interpretability is paramount**: K-Means centroids are easier to explain to non-technical stakeholders than density-based region descriptions
- **When incremental updates are needed**: Adding new trading days requires re-running DBSCAN from scratch; K-Means allows simple nearest-centroid assignment

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| eps (epsilon) | Maximum neighborhood distance | 0.3 - 3.0 (standardized) | Use k-distance graph with k=2*n_features. For 8 features, use k=16 and look for the knee in the sorted distances |
| min_pts (min_samples) | Minimum points for dense region | 3 - 20 | Rule of thumb: 2 * n_features. For 8 market features, start with min_pts=16. Increase for noisier data |
| metric | Distance function | euclidean, manhattan | Euclidean after standardization works well. Manhattan is more robust to individual feature outliers |
| algorithm | Nearest neighbor method | brute, kd_tree, ball_tree | brute for < 5000 points; kd_tree for larger datasets with < 20 features |

### Parameter Selection Guidelines for Market Data

**Eps Selection via K-Distance Graph**:
1. Compute the distance to the k-th nearest neighbor for each point (k = min_pts)
2. Sort these distances in ascending order
3. Plot and find the "elbow" -- the point where distance starts increasing sharply
4. Set eps at or slightly above this elbow value

**MinPts Heuristic**:
- `min_pts >= d + 1` where d is the number of features (minimum)
- `min_pts = 2 * d` is a good starting point
- Increase min_pts for noisier data or when you want fewer, more robust clusters
- For stock market daily data with 8 features, min_pts between 10-20 works well

## Stock Market Performance Tips

1. **Use Robust Scaling**: Standardize using median and IQR instead of mean and std. Market data contains extreme outliers that inflate standard deviation, causing most data points to cluster near zero after z-score normalization.

2. **Pre-filter Extreme Outliers**: Before running DBSCAN, winsorize extreme values (e.g., VIX > 80) at the 99.5th percentile. These extreme events will be noise regardless, but they distort the distance matrix and affect eps estimation for the remaining data.

3. **Apply PCA Before DBSCAN**: Reduce correlated market features to 3-5 principal components. This removes redundant information (e.g., VIX and put/call ratio are correlated) and makes the distance metric more meaningful in the reduced space.

4. **Use Rolling Windows**: Apply DBSCAN to rolling windows (e.g., 252 trading days) rather than the entire history. Market structure evolves, and what constitutes "dense" changes across decades. Rolling windows capture the current regime structure.

5. **Validate with Known Events**: After running DBSCAN, check whether known market events (2020 COVID crash, 2021 meme stock mania) appear as noise or distinct clusters. If they do not, adjust eps downward or min_pts upward.

6. **Layer Multiple Eps Values**: Run DBSCAN at multiple eps values and analyze how clusters split and merge. This multi-scale analysis reveals hierarchical structure in market behavior.

7. **Feature Selection Matters**: Not all market indicators add value for clustering. Remove redundant features (e.g., if including VIX, do not also include VIX percentile rank). Use correlation analysis to identify and remove collinear features.

## Comparison with Other Algorithms

| Feature | DBSCAN | K-Means | Hierarchical | Isolation Forest |
|---------|--------|---------|--------------|-----------------|
| Anomaly detection | Built-in (noise points) | None | None | Primary purpose |
| Number of clusters | Automatic | Must specify K | Optional (cut height) | N/A |
| Cluster shapes | Arbitrary | Spherical only | Arbitrary | N/A |
| Handles varying density | Poor (single eps) | N/A | Good | Excellent |
| Scalability | O(n^2) or O(n log n) | O(nKdi) | O(n^2 log n) | O(n log n) |
| Deterministic | Nearly (border points vary) | No (random init) | Yes | No (random subsampling) |
| Stock market fit | Anomaly + regime detection | Risk bucketing | Taxonomy building | Pure anomaly detection |
| Parameter sensitivity | High (eps critical) | Moderate (K choice) | Low (linkage choice) | Low |
| Interpretability | Moderate | High | High (dendrogram) | Low |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete DBSCAN Market Anomaly Detection Pipeline
# ============================================================

np.random.seed(42)

# --- Generate 3 Years of Synthetic Trading Day Data ---
def generate_market_data(n_days=750):
    """Generate realistic daily market condition data."""
    data = []
    labels = []

    # Normal bull market days (500 days)
    for _ in range(500):
        day = [
            np.random.normal(14, 2.5),          # VIX
            np.random.normal(0.0006, 0.006),     # market return
            np.random.normal(1.6, 0.35),         # adv/dec ratio
            np.random.normal(1.0, 0.15),         # volume ratio
            np.random.normal(0.007, 0.002),      # sector dispersion
            np.random.normal(0.60, 0.12),        # high/low ratio
            np.random.normal(0.30, 0.07),        # correlation
            np.random.normal(0.80, 0.12),        # put/call
        ]
        data.append(day)
        labels.append('normal_bull')

    # Normal bear market days (120 days)
    for _ in range(120):
        day = [
            np.random.normal(24, 4),
            np.random.normal(-0.005, 0.008),
            np.random.normal(0.7, 0.2),
            np.random.normal(1.3, 0.25),
            np.random.normal(0.012, 0.004),
            np.random.normal(0.35, 0.10),
            np.random.normal(0.45, 0.08),
            np.random.normal(1.05, 0.15),
        ]
        data.append(day)
        labels.append('normal_bear')

    # Correction/crash cluster (25 days)
    for _ in range(25):
        day = [
            np.random.normal(38, 6),
            np.random.normal(-0.03, 0.012),
            np.random.normal(0.22, 0.06),
            np.random.normal(2.8, 0.4),
            np.random.normal(0.028, 0.006),
            np.random.normal(0.07, 0.03),
            np.random.normal(0.72, 0.05),
            np.random.normal(1.5, 0.18),
        ]
        data.append(day)
        labels.append('crash')

    # Rally/euphoria cluster (30 days)
    for _ in range(30):
        day = [
            np.random.normal(10.5, 1.2),
            np.random.normal(0.022, 0.006),
            np.random.normal(4.2, 0.5),
            np.random.normal(1.9, 0.3),
            np.random.normal(0.004, 0.001),
            np.random.normal(0.93, 0.03),
            np.random.normal(0.42, 0.05),
            np.random.normal(0.52, 0.06),
        ]
        data.append(day)
        labels.append('euphoria')

    # Extreme individual anomalies (5 days - flash crashes etc.)
    anomalies = [
        [72, -0.09, 0.05, 4.2, 0.05, 0.01, 0.92, 2.1],  # flash crash
        [65, -0.07, 0.08, 3.9, 0.048, 0.02, 0.88, 1.95], # severe selloff
        [8, 0.05, 6.0, 3.5, 0.002, 0.98, 0.55, 0.35],    # extreme rally
        [55, -0.06, 0.12, 3.5, 0.042, 0.03, 0.82, 1.8],  # panic day
        [9, 0.04, 5.5, 0.3, 0.001, 0.95, 0.15, 0.40],    # low-vol melt-up
    ]
    for a in anomalies:
        data.append(a)
        labels.append('extreme_anomaly')

    data = np.array(data)
    labels = np.array(labels)

    # Shuffle
    idx = np.random.permutation(len(data))
    return data[idx], labels[idx]

# --- DBSCAN Implementation ---
def robust_standardize(X):
    medians = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1
    return (X - medians) / iqr, medians, iqr

def pairwise_distances(X):
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(D, 0))

def dbscan_cluster(X, eps, min_pts):
    n = X.shape[0]
    dist_mat = pairwise_distances(X)
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    is_core = np.zeros(n, dtype=bool)

    # Find core points
    for i in range(n):
        if np.sum(dist_mat[i] <= eps) >= min_pts:
            is_core[i] = True

    cluster_id = -1
    for i in range(n):
        if visited[i] or not is_core[i]:
            continue
        visited[i] = True
        cluster_id += 1
        labels[i] = cluster_id

        seeds = list(np.where(dist_mat[i] <= eps)[0])
        idx = 0
        while idx < len(seeds):
            q = seeds[idx]
            if not visited[q]:
                visited[q] = True
                if is_core[q]:
                    new_neighbors = np.where(dist_mat[q] <= eps)[0]
                    for nn in new_neighbors:
                        if nn not in seeds:
                            seeds.append(nn)
            if labels[q] == -1:
                labels[q] = cluster_id
            idx += 1

    return labels, cluster_id + 1, is_core

def k_distance_graph(X, k):
    dist_mat = pairwise_distances(X)
    k_dists = np.sort(dist_mat, axis=1)[:, k]
    return np.sort(k_dists)

# --- Run Pipeline ---
market_data, true_labels = generate_market_data()
X_std, meds, iqrs = robust_standardize(market_data)

# Estimate eps
k = 16  # 2 * n_features
k_dists = k_distance_graph(X_std, k)

# Find knee point (simple method: maximum second derivative)
diffs = np.diff(k_dists)
diffs2 = np.diff(diffs)
knee_idx = np.argmax(diffs2) + 2
eps_estimate = k_dists[knee_idx]

print(f"=== Parameter Estimation ===")
print(f"  k for distance graph: {k}")
print(f"  Estimated eps (knee): {eps_estimate:.3f}")
print(f"  Using min_pts: {k}")

# Run DBSCAN
db_labels, n_clusters, core_mask = dbscan_cluster(X_std, eps=eps_estimate, min_pts=k)

print(f"\n=== DBSCAN Results ===")
print(f"  Clusters found: {n_clusters}")
print(f"  Core points: {np.sum(core_mask)}")
print(f"  Border points: {np.sum((~core_mask) & (db_labels >= 0))}")
print(f"  Noise points: {np.sum(db_labels == -1)}")

feature_names = [
    'VIX', 'Market_Return', 'Adv_Dec', 'Volume_Ratio',
    'Sector_Disp', 'HighLow', 'Correlation', 'PutCall'
]

# Cluster profiles
print(f"\n=== Cluster Profiles ===")
for c in range(n_clusters):
    mask = db_labels == c
    n_in = np.sum(mask)
    true_comp = {l: np.sum(true_labels[mask] == l) for l in np.unique(true_labels[mask])}
    print(f"\nCluster {c} ({n_in} days):")
    print(f"  True label composition: {true_comp}")
    for j, feat in enumerate(feature_names):
        print(f"  {feat:>15s}: mean={np.mean(market_data[mask, j]):>8.4f}  "
              f"std={np.std(market_data[mask, j]):>7.4f}")

# Noise analysis
noise_mask = db_labels == -1
if np.sum(noise_mask) > 0:
    print(f"\n=== Noise Points Analysis ({np.sum(noise_mask)} days) ===")
    noise_true = {l: np.sum(true_labels[noise_mask] == l)
                  for l in np.unique(true_labels[noise_mask])}
    print(f"  True label composition: {noise_true}")

    for j, feat in enumerate(feature_names):
        noise_vals = market_data[noise_mask, j]
        all_vals = market_data[:, j]
        z_scores = (noise_vals - np.mean(all_vals)) / np.std(all_vals)
        print(f"  {feat:>15s}: mean={np.mean(noise_vals):>8.4f}  "
              f"avg |z-score|={np.mean(np.abs(z_scores)):>5.2f}")

# Detection accuracy for anomalies
true_anomalies = (true_labels == 'crash') | (true_labels == 'extreme_anomaly')
detected_anomalies = db_labels == -1
tp = np.sum(true_anomalies & detected_anomalies)
fp = np.sum(~true_anomalies & detected_anomalies)
fn = np.sum(true_anomalies & ~detected_anomalies)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n=== Anomaly Detection Performance ===")
print(f"  True anomalies (crash + extreme): {np.sum(true_anomalies)}")
print(f"  Detected as noise: {np.sum(detected_anomalies)}")
print(f"  True positives: {tp}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
```

## Key Takeaways

1. **DBSCAN is the go-to algorithm for market anomaly detection** because it naturally separates unusual trading days as noise points without requiring a pre-specified number of clusters or forcing every observation into a group.

2. **Epsilon selection is the single most important decision** when applying DBSCAN to market data. Use the k-distance graph method with k = 2 * number_of_features as a starting point, and validate by checking whether known market events are correctly identified.

3. **Robust standardization is essential** for market data. Use median and IQR instead of mean and standard deviation to prevent extreme market events from distorting the feature scaling.

4. **DBSCAN reveals market regimes automatically**, discovering the natural number of distinct market states (bull, bear, crisis, euphoria) without analyst bias about how many regimes should exist.

5. **Noise points are the most valuable output** in a market surveillance context. These are the trading days that warrant human investigation -- they do not fit any normal pattern and may indicate manipulation, systemic stress, or structural market changes.

6. **Combine DBSCAN with PCA** for best results. Reducing correlated market indicators to principal components before clustering improves distance metric quality and speeds up computation.

7. **Re-run periodically with updated parameters** because market microstructure evolves. The density characteristics that defined "normal" in 2020 differ from those in 2025. Adaptive eps estimation using recent data windows keeps the algorithm calibrated.

8. **Use DBSCAN cluster labels as features** in downstream models. A stock's cluster assignment on a given day can inform trading strategy selection, position sizing, and risk management decisions.
