# t-SNE (t-Distributed Stochastic Neighbor Embedding) - Complete Guide with Stock Market Applications

## Overview

t-SNE (t-distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique specifically designed for visualizing high-dimensional data in 2D or 3D. Unlike PCA, which preserves global variance structure, t-SNE focuses on preserving local neighborhood relationships -- points that are close in the high-dimensional space remain close in the low-dimensional embedding, while distant points are allowed to spread apart. This makes t-SNE exceptionally effective at revealing clusters, groupings, and hidden structures that linear methods like PCA cannot capture.

The algorithm works in two phases. First, it converts pairwise distances in the high-dimensional space into conditional probabilities that represent similarities: nearby points get high probability, distant points get low probability. Second, it defines a similar probability distribution in the low-dimensional space using a Student's t-distribution (which has heavier tails than a Gaussian) and minimizes the Kullback-Leibler divergence between the two distributions using gradient descent. The t-distribution in the low-dimensional space solves the "crowding problem" -- in lower dimensions, there is less room to accommodate moderately distant points, and the heavy-tailed t-distribution allows these points to spread further apart.

In stock market analysis, t-SNE is used to visualize how stocks naturally group in a high-dimensional feature space defined by returns, fundamentals, or technical indicators. It can reveal that stocks from different GICS sectors actually cluster together based on trading behavior, uncover hidden sub-groups within sectors, and identify outlier stocks that do not belong to any natural grouping. While t-SNE is primarily a visualization tool (not suitable for feature engineering or downstream modeling), the visual insights it provides are invaluable for generating hypotheses about market structure.

## How It Works - The Math Behind It

### Step 1: Compute Pairwise Similarities in High-Dimensional Space

For each pair of points (x_i, x_j), compute the conditional probability that x_i would pick x_j as its neighbor under a Gaussian centered at x_i:

```
p_{j|i} = exp(-||x_i - x_j||^2 / (2 * sigma_i^2)) / sum_{k != i} exp(-||x_i - x_k||^2 / (2 * sigma_i^2))
```

Symmetrize the probabilities:
```
p_{ij} = (p_{j|i} + p_{i|j}) / (2 * n)
```

### Step 2: Perplexity and Bandwidth (sigma)

The bandwidth sigma_i for each point is set such that the perplexity of the conditional distribution P_i matches a user-specified target:

```
Perplexity(P_i) = 2^{H(P_i)}
```

Where H(P_i) is the Shannon entropy:
```
H(P_i) = -sum_j p_{j|i} * log2(p_{j|i})
```

Sigma_i is found via binary search to match the target perplexity. Higher perplexity = larger effective neighborhood = more focus on global structure.

### Step 3: Initialize Low-Dimensional Embedding

Initialize the low-dimensional points y_1, ..., y_n randomly, typically from a small Gaussian:
```
y_i ~ N(0, 10^{-4} * I)
```

### Step 4: Compute Similarities in Low-Dimensional Space

Use a Student's t-distribution with one degree of freedom (Cauchy distribution):
```
q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / sum_{k != l} (1 + ||y_k - y_l||^2)^{-1}
```

The t-distribution's heavy tails allow moderate distances in high-D to map to larger distances in low-D, solving the crowding problem.

### Step 5: Minimize KL Divergence

Minimize the KL divergence between P and Q:
```
KL(P || Q) = sum_{i != j} p_{ij} * log(p_{ij} / q_{ij})
```

Gradient:
```
dC/dy_i = 4 * sum_j (p_{ij} - q_{ij}) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^{-1}
```

Update rule with momentum:
```
y_i^(t+1) = y_i^(t) + eta * (dC/dy_i) + alpha * (y_i^(t) - y_i^(t-1))
```

### Early Exaggeration

During the first ~250 iterations, multiply all p_{ij} by a factor (typically 4-12). This forces clusters to move apart quickly in the early optimization, creating better-separated groups.

### Complexity

- **Time Complexity**: O(n^2) per iteration (naive), O(n log n) with Barnes-Hut approximation
- **Space Complexity**: O(n^2) for pairwise probabilities

## Stock Market Use Case: Visualizing Stock Market Sectors and Hidden Groupings

### The Problem

An equity research team covers 200 stocks across 11 GICS sectors. They want to visualize how these stocks relate to each other based on their actual trading behavior (returns, volatility, volume patterns) rather than their official sector classification. The goal is to discover: (1) which official sectors truly contain similar stocks, (2) whether there are hidden sub-groups within sectors, (3) which stocks are mislabeled by traditional classification, and (4) whether certain stocks from different sectors cluster together (e.g., fintech stocks clustering with tech rather than financials). This visualization will guide research coverage and portfolio construction decisions.

### Stock Market Features (Input Data)

| Feature | Description | Dimension Category |
|---------|-------------|-------------------|
| return_1m | 1-month trailing return | Return profile |
| return_3m | 3-month trailing return | Return profile |
| return_12m | 12-month trailing return | Return profile |
| volatility_30d | 30-day realized volatility | Risk |
| volatility_90d | 90-day realized volatility | Risk |
| max_drawdown_6m | 6-month max drawdown | Risk |
| beta | Market beta (vs S&P 500) | Risk |
| avg_volume_ratio | Volume vs 60-day average | Liquidity |
| volume_volatility | Std dev of daily volume | Liquidity |
| rsi_14 | 14-day RSI | Momentum |
| macd_signal | MACD minus signal line | Momentum |
| pe_ratio | Price-to-earnings | Fundamental |
| pb_ratio | Price-to-book | Fundamental |
| dividend_yield | Annual dividend yield | Fundamental |
| market_cap_log | Log of market cap | Size |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)

def generate_stock_universe():
    """Generate 200 stocks across sectors with realistic feature profiles."""

    sectors = {
        'Technology': {
            'n': 35,
            'profile': [0.02, 0.06, 0.20, 0.28, 0.25, -0.15, 1.3,
                        1.2, 1.5, 55, 0.3, 35, 7.0, 0.005, 5.0]
        },
        'Healthcare': {
            'n': 25,
            'profile': [0.01, 0.04, 0.12, 0.30, 0.28, -0.18, 0.9,
                        0.9, 1.2, 50, 0.1, 25, 4.0, 0.015, 4.5]
        },
        'Financials': {
            'n': 30,
            'profile': [0.008, 0.03, 0.10, 0.22, 0.20, -0.12, 1.1,
                        1.1, 1.3, 48, 0.15, 12, 1.2, 0.03, 4.8]
        },
        'Consumer_Disc': {
            'n': 25,
            'profile': [0.015, 0.05, 0.15, 0.25, 0.23, -0.14, 1.2,
                        1.0, 1.4, 52, 0.2, 22, 5.0, 0.01, 4.3]
        },
        'Industrials': {
            'n': 20,
            'profile': [0.01, 0.035, 0.11, 0.20, 0.18, -0.10, 1.0,
                        0.8, 1.0, 50, 0.12, 18, 3.0, 0.02, 4.2]
        },
        'Energy': {
            'n': 15,
            'profile': [0.005, 0.02, 0.08, 0.35, 0.32, -0.25, 1.4,
                        1.3, 1.8, 45, -0.1, 10, 1.5, 0.04, 4.0]
        },
        'Utilities': {
            'n': 15,
            'profile': [-0.002, 0.01, 0.05, 0.12, 0.11, -0.06, 0.4,
                        0.7, 0.6, 48, 0.05, 17, 1.8, 0.04, 4.1]
        },
        'Real_Estate': {
            'n': 15,
            'profile': [0.003, 0.015, 0.07, 0.18, 0.16, -0.10, 0.7,
                        0.8, 0.8, 46, 0.08, 20, 2.0, 0.045, 3.8]
        },
        'Staples': {
            'n': 10,
            'profile': [0.005, 0.02, 0.08, 0.13, 0.12, -0.07, 0.5,
                        0.7, 0.5, 49, 0.06, 20, 4.0, 0.03, 4.5]
        },
        'Materials': {
            'n': 10,
            'profile': [0.008, 0.03, 0.10, 0.25, 0.22, -0.15, 1.1,
                        0.9, 1.2, 47, 0.10, 14, 2.5, 0.025, 4.0]
        },
    }

    # Noise scales for each feature
    noise_scales = [0.02, 0.04, 0.10, 0.06, 0.05, 0.06, 0.25,
                    0.3, 0.4, 8, 0.15, 12, 2.0, 0.01, 0.5]

    all_data = []
    all_names = []
    all_sectors = []

    for sector_name, config in sectors.items():
        for i in range(config['n']):
            profile = np.array(config['profile'])
            noise = np.array(noise_scales) * np.random.randn(15)
            stock = profile + noise
            all_data.append(stock)
            all_names.append(f"{sector_name[:4]}_{i+1}")
            all_sectors.append(sector_name)

    return np.array(all_data), all_names, all_sectors

feature_names = [
    'return_1m', 'return_3m', 'return_12m', 'vol_30d', 'vol_90d',
    'max_dd_6m', 'beta', 'vol_ratio', 'vol_volatility', 'rsi_14',
    'macd_signal', 'pe_ratio', 'pb_ratio', 'div_yield', 'log_mcap'
]

stock_data, stock_names, stock_sectors = generate_stock_universe()
print(f"Stock universe: {stock_data.shape[0]} stocks, {stock_data.shape[1]} features")
print(f"Sectors: {len(set(stock_sectors))}")
for sector in sorted(set(stock_sectors)):
    count = sum(1 for s in stock_sectors if s == sector)
    print(f"  {sector}: {count} stocks")
```

### The Model in Action

```python
import numpy as np

def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma, mu, sigma

def compute_pairwise_distances(X):
    """Compute squared Euclidean distance matrix."""
    sq_norms = np.sum(X ** 2, axis=1)
    D = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    return np.maximum(D, 0)

def binary_search_sigma(distances_i, target_perplexity, tol=1e-5, max_iter=50):
    """
    Find sigma for a single point via binary search to match target perplexity.
    """
    lo, hi = 1e-10, 1e4
    sigma = 1.0

    for _ in range(max_iter):
        # Compute conditional probabilities
        exp_d = np.exp(-distances_i / (2 * sigma ** 2))
        sum_exp = np.sum(exp_d)

        if sum_exp == 0:
            sigma *= 2
            continue

        p = exp_d / sum_exp

        # Compute entropy
        p_safe = np.maximum(p, 1e-12)
        entropy = -np.sum(p * np.log2(p_safe))
        perplexity = 2 ** entropy

        if abs(perplexity - target_perplexity) < tol:
            break

        if perplexity > target_perplexity:
            hi = sigma
            sigma = (lo + hi) / 2
        else:
            lo = sigma
            sigma = (lo + hi) / 2

    return sigma, p

def compute_joint_probabilities(X, perplexity=30):
    """
    Compute symmetric joint probability matrix P for high-dimensional data.
    """
    n = X.shape[0]
    D = compute_pairwise_distances(X)
    P = np.zeros((n, n))

    for i in range(n):
        # Distances from point i to all others (excluding self)
        dists = D[i].copy()
        dists[i] = np.inf  # exclude self

        sigma, p_i = binary_search_sigma(dists, perplexity)
        p_i[i] = 0  # no self-similarity
        P[i] = p_i

    # Symmetrize
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    return P

def compute_q_distribution(Y):
    """
    Compute the Student-t joint probability matrix Q for low-dimensional embedding.
    """
    D = compute_pairwise_distances(Y)
    Q = 1.0 / (1.0 + D)
    np.fill_diagonal(Q, 0)
    Q_sum = np.sum(Q)
    if Q_sum == 0:
        Q_sum = 1
    Q = Q / Q_sum
    Q = np.maximum(Q, 1e-12)
    return Q, D

def tsne(X, n_components=2, perplexity=30, n_iter=1000,
         learning_rate=200, early_exaggeration=12.0,
         early_exag_iters=250, momentum_init=0.5, momentum_final=0.8):
    """
    t-SNE implementation from scratch.

    Parameters:
        X: high-dimensional data (n_samples, n_features)
        n_components: output dimensions (usually 2)
        perplexity: effective number of neighbors (5-50)
        n_iter: number of gradient descent iterations
        learning_rate: step size for gradient descent
        early_exaggeration: factor to multiply P during early iterations
        early_exag_iters: number of early exaggeration iterations
        momentum_init: momentum during early exaggeration
        momentum_final: momentum after early exaggeration

    Returns:
        Y: low-dimensional embedding (n_samples, n_components)
        kl_history: KL divergence per iteration
    """
    n = X.shape[0]

    # Step 1: Compute joint probabilities in high-D
    print("  Computing pairwise similarities...")
    P = compute_joint_probabilities(X, perplexity)

    # Step 2: Initialize embedding
    Y = np.random.randn(n, n_components) * 1e-4
    Y_prev = Y.copy()

    kl_history = []

    # Step 3: Gradient descent
    print("  Running gradient descent...")
    for iteration in range(n_iter):
        # Apply early exaggeration
        if iteration < early_exag_iters:
            P_used = P * early_exaggeration
            momentum = momentum_init
        else:
            P_used = P
            momentum = momentum_final

        # Compute Q distribution
        Q, D_low = compute_q_distribution(Y)

        # Compute KL divergence
        kl = np.sum(P_used * np.log(P_used / Q))
        kl_history.append(kl)

        # Compute gradients
        PQ_diff = P_used - Q
        inv_dist = 1.0 / (1.0 + D_low)
        np.fill_diagonal(inv_dist, 0)

        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum(
                (PQ_diff[i, :] * inv_dist[i, :])[:, None] * diff,
                axis=0
            )

        # Update with momentum
        Y_new = Y - learning_rate * grad + momentum * (Y - Y_prev)
        Y_prev = Y.copy()
        Y = Y_new

        # Center the embedding
        Y = Y - np.mean(Y, axis=0)

        if iteration % 100 == 0:
            print(f"    Iteration {iteration}: KL divergence = {kl:.4f}")

    return Y, kl_history

# Standardize and run t-SNE
X_std, _, _ = standardize(stock_data)

# Apply PCA first to reduce noise (recommended for t-SNE)
from_pca = X_std @ np.linalg.eigh(np.cov(X_std.T))[1][:, -10:]  # top 10 PCA components

print("Running t-SNE on stock data...")
Y, kl_hist = tsne(from_pca, n_components=2, perplexity=30, n_iter=800,
                   learning_rate=200)

# Analyze the embedding
print(f"\n=== t-SNE Embedding Results ===")
print(f"Final KL divergence: {kl_hist[-1]:.4f}")
print(f"Embedding shape: {Y.shape}")

# Check sector separation
sector_list = sorted(set(stock_sectors))
print(f"\n=== Sector Centroid Locations ===")
for sector in sector_list:
    mask = np.array([s == sector for s in stock_sectors])
    centroid = np.mean(Y[mask], axis=0)
    spread = np.std(Y[mask], axis=0)
    print(f"  {sector:>15s}: center=({centroid[0]:>6.2f}, {centroid[1]:>6.2f}), "
          f"spread=({spread[0]:>5.2f}, {spread[1]:>5.2f})")
```

## Advantages

1. **Reveals Hidden Market Structure That Linear Methods Miss**: t-SNE can uncover nonlinear relationships between stocks that PCA cannot capture. For example, it can show that small-cap biotech stocks form a distinct cluster separate from large-cap pharma, even though both are in "Healthcare." These nonlinear groupings are invisible in PCA projections.

2. **Exceptional Cluster Visualization**: t-SNE produces visualizations where natural clusters are clearly separated with visible gaps between them. When applied to stock market data, the resulting 2D plot immediately shows which groups of stocks behave similarly, making it an invaluable exploratory tool for equity research teams.

3. **Preserves Local Neighborhood Structure**: Stocks that are close in the high-dimensional feature space (genuine behavioral peers) remain close in the t-SNE embedding. This means the visualization faithfully represents which stocks truly behave alike, providing reliable peer group identification at a glance.

4. **Handles High-Dimensional Data Effectively**: While PCA struggles to project 15+ dimensions into 2D meaningfully, t-SNE excels at this task. The 15 trading features of 200 stocks can be compressed into a single, interpretable 2D plot that captures the essential grouping structure.

5. **Robust to Feature Scale Differences**: After standardization, t-SNE handles mixed-scale features (returns, ratios, volumes) effectively because it operates on pairwise distances rather than on the features directly. The distance-based approach naturally adapts to the local density of each region.

6. **Identifies Outlier and Misclassified Stocks**: Stocks that appear isolated in the t-SNE plot, far from their official sector cluster, are candidates for reclassification. A fintech company classified as "Financials" that appears near the Technology cluster provides actionable insight for more accurate sector assignment.

7. **Perplexity Parameter Controls Resolution**: By varying perplexity, analysts can explore stock relationships at different scales. Low perplexity (5-10) reveals fine-grained sub-groups within sectors. High perplexity (40-50) shows broad sector-level separation. Multiple runs at different perplexities provide a multi-scale view.

## Disadvantages

1. **Not Suitable for Downstream Modeling**: t-SNE embeddings are optimized for visualization, not for use as features in predictive models. The coordinates are not stable across runs (different random seeds produce different layouts), making them useless as inputs to trading models. Use PCA for feature engineering instead.

2. **Computationally Expensive**: The O(n^2) pairwise distance computation and iterative optimization make t-SNE slow for large stock universes. Embedding 5,000 stocks with 20 features takes minutes to hours, making it impractical for daily or intraday analysis. Barnes-Hut approximation helps but is still much slower than PCA.

3. **Random Initialization Leads to Different Results**: Each run of t-SNE with a different random seed produces a different embedding. Clusters may appear in different locations, and their relative positions can change. This non-determinism makes it unreliable for quantitative comparisons across time periods.

4. **Distances Between Clusters Are Meaningless**: In a t-SNE plot, the distance between two clusters does not reflect their actual dissimilarity. Technology and Healthcare clusters might appear close in one run and far apart in another, even though their actual distance has not changed. Only within-cluster and immediate neighborhood relationships are meaningful.

5. **Global Structure Is Not Preserved**: t-SNE focuses on local structure at the expense of global structure. The overall arrangement of sectors in the 2D plot may not reflect their true relationships. PCA is better for understanding global relationships between sectors.

6. **Perplexity Sensitivity**: The choice of perplexity significantly affects the visualization. Too low a perplexity fragments clusters into meaningless sub-groups. Too high a perplexity merges distinct groups into blobs. Finding the right perplexity requires experimentation and domain knowledge, and no single value works for all datasets.

7. **Cannot Handle New Stocks Without Re-running**: When a new stock is added to the universe, the entire t-SNE must be re-run. There is no way to project a new point into an existing embedding. This limitation makes t-SNE impractical for any application requiring incremental updates.

## When to Use in Stock Market

- **Exploratory data analysis**: Visualizing the overall structure of a stock universe before building quantitative models
- **Sector classification validation**: Checking whether GICS sector assignments align with actual trading behavior
- **Hidden sub-group discovery**: Finding natural sub-sectors within broad classifications (e.g., discovering that "Technology" contains distinct sub-clusters of SaaS, hardware, and semiconductors)
- **Outlier identification**: Spotting stocks that do not belong to any natural cluster and may warrant special research attention
- **Cross-sector relationship discovery**: Identifying stocks from different sectors that behave similarly (fintech in financials behaving like tech)
- **Presentation and communication**: Creating compelling visual summaries of market structure for research reports and client presentations
- **Regime change visualization**: Comparing t-SNE plots from different time periods to see how market structure has evolved
- **Feature quality assessment**: Verifying that your chosen features produce meaningful stock groupings before using them in models

## When NOT to Use in Stock Market

- **Feature engineering for models**: t-SNE coordinates are not stable or meaningful as input features. Use PCA instead
- **Quantitative distance measurement**: Distances in t-SNE plots do not correspond to actual dissimilarity. Use correlation or Euclidean distance directly
- **Large stock universes (>5000 stocks)**: Computational cost is prohibitive. Consider UMAP as a faster alternative with similar visualization quality
- **Real-time or production systems**: Non-deterministic results and slow computation make t-SNE unsuitable for automated trading systems
- **Time-series analysis**: t-SNE operates on a single snapshot and has no temporal awareness. Use time-series specific methods for pattern recognition over time
- **When global structure matters**: If understanding how sectors relate to each other (not just within-sector groupings) is the primary goal, PCA or MDS preserves global structure better
- **Comparative analysis across time**: Different t-SNE runs produce different layouts, making it impossible to directly compare stock positions between two time periods

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| perplexity | Effective number of neighbors | 5 - 50 | Start with 30 for 200 stocks. Use 15-20 for <100 stocks, 40-50 for >500 stocks. Rule of thumb: perplexity < n/3 |
| n_iter | Number of optimization iterations | 500 - 5000 | 1000 for exploration, 2000-3000 for publication-quality plots. Watch KL divergence for convergence |
| learning_rate | Gradient descent step size | 10 - 1000 | 200 is default and usually works. Increase if KL divergence is not decreasing |
| early_exaggeration | P multiplier in early iterations | 4 - 20 | 12 (default) works well. Increase for clearer cluster separation |
| n_components | Output dimensions | 2 or 3 | Use 2 for plots, 3 if interactive 3D visualization is available |
| init | Initialization method | random, pca | PCA initialization is more stable and converges faster |
| metric | Distance function | euclidean, cosine | Euclidean after standardization for fundamental data; cosine for return-based similarity |

### Perplexity Selection Guide

| Stock Universe Size | Suggested Perplexity Range | Notes |
|--------------------|-----------------------------|-------|
| 50 - 100 stocks | 10 - 20 | Small universe; low perplexity captures fine structure |
| 100 - 300 stocks | 20 - 40 | Medium universe; balanced local/global |
| 300 - 1000 stocks | 30 - 50 | Large universe; higher perplexity for stable results |
| 1000+ stocks | 40 - 100 | Very large; consider UMAP instead |

## Stock Market Performance Tips

1. **Always Apply PCA Before t-SNE**: Reduce the feature space to 10-30 dimensions with PCA before running t-SNE. This removes noise from redundant features, speeds up computation significantly, and often produces cleaner visualizations. Going from 50 raw features to 15 PCA components before t-SNE is standard practice.

2. **Run Multiple Times with Different Seeds**: Due to non-deterministic initialization, run t-SNE 5-10 times with different random seeds. Only trust patterns that appear consistently across runs. If a cluster appears in every run, it represents genuine structure; if it appears in only some runs, it may be an artifact.

3. **Try Multiple Perplexity Values**: Run t-SNE with perplexity = 10, 20, 30, 50 and compare results. Clusters that are stable across perplexity values represent robust market structure. Clusters that appear only at specific perplexity values may be artifacts of parameter choice.

4. **Color Points by Known Labels**: After running t-SNE, color points by sector, market cap, or other known attributes. This reveals whether the algorithm has discovered meaningful structure. If stocks of the same sector cluster together, the features capture sector-relevant information.

5. **Use Interactive Visualizations**: Static t-SNE plots lose information because you cannot identify individual stocks. Use interactive tools (plotly, bokeh) that allow hovering over points to see stock names, enabling analysts to investigate specific clustering decisions.

6. **Standardize Features Carefully**: Use z-score standardization for normally distributed features and rank-based standardization for skewed features (P/E ratios, market cap). Mixed standardization approaches improve the quality of pairwise distances.

7. **Compare t-SNE with PCA Side by Side**: Show both visualizations to stakeholders. PCA preserves global structure (how sectors relate to each other) while t-SNE reveals local structure (which stocks within a sector are most similar). Together, they provide a complete picture.

## Comparison with Other Algorithms

| Feature | t-SNE | PCA | UMAP | MDS |
|---------|-------|-----|------|-----|
| Preserves | Local structure | Global variance | Local + some global | Global distances |
| Linearity | Nonlinear | Linear | Nonlinear | Linear (classical) |
| Speed | Slow O(n^2) | Very fast O(nd^2) | Fast O(n^1.14) | Slow O(n^3) |
| Deterministic | No | Yes | No | Yes |
| Suitable as features | No | Yes | Somewhat | Somewhat |
| Cluster separation | Excellent | Moderate | Excellent | Moderate |
| Global structure | Poor | Excellent | Good | Excellent |
| New point projection | No | Yes | Yes | No |
| Stock market use | Visualization only | Feature reduction | Visualization + features | Visualization |
| Scalability | <5000 stocks | Unlimited | <100K stocks | <2000 stocks |
| Interactive exploration | Excellent | Limited | Excellent | Limited |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete t-SNE Stock Market Visualization Pipeline
# ============================================================

np.random.seed(42)

# --- Generate Comprehensive Stock Universe ---
def create_stock_universe(n_stocks=200):
    """Create a diverse stock universe with realistic feature profiles."""

    sector_configs = {
        'Tech_SaaS': {
            'n': 20, 'base': [0.025, 0.08, 0.25, 0.30, 0.27, -0.16, 1.3,
                              1.1, 1.4, 58, 0.35, 45, 10, 0.002, 5.2]
        },
        'Tech_Hardware': {
            'n': 15, 'base': [0.015, 0.05, 0.18, 0.25, 0.22, -0.13, 1.2,
                              1.0, 1.2, 52, 0.20, 20, 4, 0.015, 4.8]
        },
        'Biotech': {
            'n': 18, 'base': [0.01, 0.03, 0.10, 0.40, 0.38, -0.25, 0.8,
                              0.8, 1.5, 48, 0.05, 80, 6, 0.0, 3.5]
        },
        'Big_Pharma': {
            'n': 12, 'base': [0.005, 0.02, 0.08, 0.18, 0.16, -0.10, 0.7,
                              0.9, 0.8, 50, 0.10, 15, 3, 0.03, 5.0]
        },
        'Banks': {
            'n': 18, 'base': [0.008, 0.03, 0.12, 0.24, 0.22, -0.14, 1.2,
                              1.0, 1.1, 48, 0.15, 10, 1.0, 0.035, 4.8]
        },
        'Fintech': {
            'n': 10, 'base': [0.02, 0.06, 0.20, 0.35, 0.30, -0.20, 1.4,
                              1.2, 1.6, 55, 0.30, 60, 8, 0.0, 4.2]
        },
        'Retail': {
            'n': 15, 'base': [0.01, 0.04, 0.12, 0.22, 0.20, -0.12, 1.0,
                              0.9, 1.1, 50, 0.12, 18, 3.5, 0.02, 4.3]
        },
        'E_Commerce': {
            'n': 12, 'base': [0.02, 0.07, 0.22, 0.32, 0.28, -0.18, 1.3,
                              1.1, 1.4, 54, 0.28, 50, 8, 0.0, 4.8]
        },
        'Oil_Gas': {
            'n': 12, 'base': [0.005, 0.015, 0.06, 0.35, 0.32, -0.22, 1.3,
                              1.2, 1.8, 44, -0.10, 8, 1.2, 0.05, 4.2]
        },
        'Renewables': {
            'n': 10, 'base': [0.015, 0.05, 0.15, 0.38, 0.35, -0.20, 1.1,
                              1.0, 1.5, 52, 0.20, 70, 5, 0.0, 3.5]
        },
        'Utilities': {
            'n': 12, 'base': [-0.002, 0.01, 0.05, 0.12, 0.11, -0.05, 0.35,
                              0.7, 0.5, 47, 0.04, 18, 1.8, 0.045, 4.0]
        },
        'REITs': {
            'n': 12, 'base': [0.003, 0.015, 0.06, 0.18, 0.16, -0.08, 0.6,
                              0.75, 0.7, 46, 0.06, 22, 2.0, 0.05, 3.8]
        },
        'Industrials': {
            'n': 15, 'base': [0.01, 0.035, 0.11, 0.20, 0.18, -0.10, 1.0,
                              0.85, 1.0, 50, 0.12, 16, 3, 0.02, 4.3]
        },
        'Staples': {
            'n': 9, 'base': [0.005, 0.02, 0.07, 0.13, 0.12, -0.06, 0.5,
                              0.7, 0.5, 49, 0.06, 22, 4, 0.028, 4.6]
        },
    }

    noise_scales = np.array([0.015, 0.03, 0.08, 0.05, 0.04, 0.05, 0.2,
                              0.25, 0.35, 7, 0.12, 10, 2.0, 0.008, 0.4])

    all_data, all_names, all_sectors, all_subsectors = [], [], [], []

    # GICS-like sector mapping
    gics_map = {
        'Tech_SaaS': 'Technology', 'Tech_Hardware': 'Technology',
        'Biotech': 'Healthcare', 'Big_Pharma': 'Healthcare',
        'Banks': 'Financials', 'Fintech': 'Financials',
        'Retail': 'Cons_Disc', 'E_Commerce': 'Cons_Disc',
        'Oil_Gas': 'Energy', 'Renewables': 'Energy',
        'Utilities': 'Utilities', 'REITs': 'Real_Estate',
        'Industrials': 'Industrials', 'Staples': 'Cons_Staples',
    }

    for sub_sector, config in sector_configs.items():
        for i in range(config['n']):
            base = np.array(config['base'])
            stock = base + noise_scales * np.random.randn(15)
            all_data.append(stock)
            all_names.append(f"{sub_sector[:5]}_{i+1}")
            all_sectors.append(gics_map[sub_sector])
            all_subsectors.append(sub_sector)

    X = np.array(all_data)
    return X, all_names, all_sectors, all_subsectors

# --- t-SNE Implementation ---
def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma, mu, sigma

def pca_reduce(X_std, n_components=10):
    """Reduce dimensions with PCA before t-SNE."""
    cov = np.cov(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx[:n_components]]
    var_explained = np.sum(eigenvalues[idx[:n_components]]) / np.sum(eigenvalues)
    return X_std @ W, var_explained

def compute_high_d_probs(X, perplexity=30):
    """Compute symmetric pairwise probabilities in high-D."""
    n = X.shape[0]
    D = np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T
    D = np.maximum(D, 0)
    P = np.zeros((n, n))

    for i in range(n):
        dists = D[i].copy()
        dists[i] = np.inf

        # Binary search for sigma
        lo, hi, sigma = 1e-10, 1e4, 1.0
        for _ in range(50):
            exp_d = np.exp(-dists / (2 * sigma ** 2))
            exp_d[i] = 0
            s = np.sum(exp_d)
            if s == 0:
                sigma *= 2
                continue
            p = exp_d / s
            H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
            perp = 2 ** H
            if abs(perp - perplexity) < 0.5:
                break
            if perp > perplexity:
                hi = sigma
            else:
                lo = sigma
            sigma = (lo + hi) / 2

        P[i] = p

    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    return P

def run_tsne(X, n_components=2, perplexity=30, n_iter=1000,
             lr=200, exag=12.0, exag_iters=250):
    """Complete t-SNE pipeline."""
    n = X.shape[0]

    # High-D probabilities
    P = compute_high_d_probs(X, perplexity)

    # Initialize
    Y = np.random.randn(n, n_components) * 1e-4
    Y_prev = Y.copy()
    kl_hist = []

    for it in range(n_iter):
        P_use = P * exag if it < exag_iters else P
        mom = 0.5 if it < exag_iters else 0.8

        # Q distribution
        D_low = np.sum(Y ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * Y @ Y.T
        D_low = np.maximum(D_low, 0)
        Q = 1.0 / (1.0 + D_low)
        np.fill_diagonal(Q, 0)
        Q_sum = max(np.sum(Q), 1e-12)
        Q = Q / Q_sum
        Q = np.maximum(Q, 1e-12)

        # KL divergence
        kl = np.sum(P_use * np.log(P_use / Q))
        kl_hist.append(kl)

        # Gradient
        PQ = P_use - Q
        inv_d = 1.0 / (1.0 + D_low)
        np.fill_diagonal(inv_d, 0)

        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ[i] * inv_d[i])[:, None] * diff, axis=0)

        # Update
        Y_new = Y - lr * grad + mom * (Y - Y_prev)
        Y_prev = Y.copy()
        Y = Y_new - np.mean(Y_new, axis=0)

        if it % 200 == 0:
            print(f"    Iter {it:>4d}: KL = {kl:.4f}")

    return Y, kl_hist

# --- Run Full Pipeline ---
X_raw, names, sectors, subsectors = create_stock_universe()
X_std, _, _ = standardize(X_raw)

feature_names = [
    'ret_1m', 'ret_3m', 'ret_12m', 'vol_30d', 'vol_90d',
    'max_dd', 'beta', 'vol_ratio', 'vol_vol', 'rsi',
    'macd', 'pe', 'pb', 'div_yield', 'log_mcap'
]

print(f"Stock universe: {len(names)} stocks, {len(feature_names)} features")
print(f"Sectors: {len(set(sectors))}, Sub-sectors: {len(set(subsectors))}")

# Step 1: PCA pre-processing
X_pca, pca_var = pca_reduce(X_std, n_components=10)
print(f"\nPCA: {X_std.shape[1]} -> {X_pca.shape[1]} dimensions ({pca_var*100:.1f}% variance)")

# Step 2: Run t-SNE
print("\nRunning t-SNE (perplexity=30)...")
Y, kl_hist = run_tsne(X_pca, n_components=2, perplexity=30, n_iter=800, lr=200)

# --- Analysis ---
print(f"\n{'='*60}")
print("t-SNE ANALYSIS RESULTS")
print(f"{'='*60}")

# Sector centroids in t-SNE space
print(f"\n--- GICS Sector Centroids ---")
for sector in sorted(set(sectors)):
    mask = np.array([s == sector for s in sectors])
    center = np.mean(Y[mask], axis=0)
    spread = np.mean(np.sqrt(np.sum((Y[mask] - center) ** 2, axis=1)))
    print(f"  {sector:>15s} ({np.sum(mask):>3d} stocks): "
          f"center=({center[0]:>6.1f}, {center[1]:>6.1f}), "
          f"avg_spread={spread:.2f}")

# Sub-sector centroids
print(f"\n--- Sub-sector Centroids ---")
for sub in sorted(set(subsectors)):
    mask = np.array([s == sub for s in subsectors])
    center = np.mean(Y[mask], axis=0)
    spread = np.mean(np.sqrt(np.sum((Y[mask] - center) ** 2, axis=1)))
    parent = [sectors[i] for i in range(len(subsectors)) if subsectors[i] == sub][0]
    print(f"  {sub:>15s} [{parent:>12s}] ({np.sum(mask):>2d} stocks): "
          f"center=({center[0]:>6.1f}, {center[1]:>6.1f}), spread={spread:.2f}")

# Cross-sector proximity analysis
print(f"\n--- Cross-Sector Proximity Analysis ---")
print("  (Which sub-sectors from different GICS sectors cluster closest?)")
sub_centroids = {}
for sub in set(subsectors):
    mask = np.array([s == sub for s in subsectors])
    sub_centroids[sub] = np.mean(Y[mask], axis=0)

# Find closest cross-sector pairs
cross_pairs = []
sub_list = sorted(sub_centroids.keys())
for i in range(len(sub_list)):
    for j in range(i+1, len(sub_list)):
        s1, s2 = sub_list[i], sub_list[j]
        parent1 = [sectors[k] for k in range(len(subsectors)) if subsectors[k] == s1][0]
        parent2 = [sectors[k] for k in range(len(subsectors)) if subsectors[k] == s2][0]
        if parent1 != parent2:
            dist = np.sqrt(np.sum((sub_centroids[s1] - sub_centroids[s2]) ** 2))
            cross_pairs.append((s1, s2, parent1, parent2, dist))

cross_pairs.sort(key=lambda x: x[4])
print("  Closest cross-sector sub-sector pairs:")
for s1, s2, p1, p2, d in cross_pairs[:5]:
    print(f"    {s1} [{p1}] <-> {s2} [{p2}]: distance = {d:.2f}")

# Outlier detection: stocks far from their sector centroid
print(f"\n--- Potential Sector Misclassifications ---")
sector_centroids = {}
for sector in set(sectors):
    mask = np.array([s == sector for s in sectors])
    sector_centroids[sector] = np.mean(Y[mask], axis=0)

outlier_scores = []
for i in range(len(names)):
    own_centroid = sector_centroids[sectors[i]]
    own_dist = np.sqrt(np.sum((Y[i] - own_centroid) ** 2))

    # Distance to nearest other sector centroid
    min_other_dist = np.inf
    nearest_sector = None
    for other_sector, other_centroid in sector_centroids.items():
        if other_sector != sectors[i]:
            d = np.sqrt(np.sum((Y[i] - other_centroid) ** 2))
            if d < min_other_dist:
                min_other_dist = d
                nearest_sector = other_sector

    if min_other_dist < own_dist:
        outlier_scores.append((names[i], sectors[i], subsectors[i],
                              nearest_sector, own_dist, min_other_dist))

outlier_scores.sort(key=lambda x: x[4] - x[5], reverse=True)
print(f"  Stocks closer to another sector than their own ({len(outlier_scores)} found):")
for name, own_sec, own_sub, near_sec, own_d, near_d in outlier_scores[:8]:
    print(f"    {name:>10s} ({own_sub} in {own_sec}): "
          f"closer to {near_sec} ({near_d:.2f} vs {own_d:.2f})")

# Convergence analysis
print(f"\n--- Convergence ---")
print(f"  Initial KL: {kl_hist[0]:.4f}")
print(f"  Final KL: {kl_hist[-1]:.4f}")
print(f"  KL reduction: {(1 - kl_hist[-1]/kl_hist[0])*100:.1f}%")

# Trustworthiness metric (simplified)
print(f"\n--- Embedding Quality ---")
# Check if nearest neighbors in high-D are preserved in low-D
k_neighbors = 10
D_high = np.sum(X_pca ** 2, axis=1)[:, None] + np.sum(X_pca ** 2, axis=1)[None, :] - 2 * X_pca @ X_pca.T
D_low = np.sum(Y ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * Y @ Y.T

preserved = 0
total = 0
for i in range(len(Y)):
    high_nn = np.argsort(D_high[i])[:k_neighbors+1][1:]  # exclude self
    low_nn = np.argsort(D_low[i])[:k_neighbors+1][1:]
    preserved += len(set(high_nn) & set(low_nn))
    total += k_neighbors

trust = preserved / total
print(f"  Neighborhood preservation (k={k_neighbors}): {trust*100:.1f}%")
print(f"  (>70% = good embedding quality)")
```

## Key Takeaways

1. **t-SNE is the premier tool for visualizing stock market structure** in 2D. It reveals clusters, sub-groups, and outliers that are invisible in PCA projections, making it invaluable for exploratory analysis and hypothesis generation.

2. **Always apply PCA before t-SNE** to reduce the feature space to 10-30 dimensions. This removes noise, speeds up computation by 5-10x, and typically produces cleaner visualizations.

3. **t-SNE is a visualization tool, not a modeling tool**. Never use t-SNE coordinates as features in trading models. The embeddings are non-deterministic, non-metric, and unstable across runs. Use PCA for feature engineering.

4. **Run t-SNE multiple times with different random seeds** and only trust patterns that appear consistently. A cluster that appears in every run represents genuine market structure; one that appears intermittently is likely an artifact.

5. **Perplexity controls the scale of structure revealed**. Low perplexity (10-15) shows fine-grained sub-groups within sectors. High perplexity (40-50) shows broad sector-level separation. Try multiple values for a complete picture.

6. **Distances between clusters in t-SNE are meaningless**. Do not interpret the gap between Technology and Healthcare clusters as a measure of their dissimilarity. Only local neighborhood relationships are preserved.

7. **Cross-sector clustering provides actionable insights**. When t-SNE shows Fintech stocks clustering with Technology rather than Financials, this suggests that Fintech stocks should be treated as tech-like for risk management and portfolio construction purposes.

8. **Combine t-SNE with domain knowledge for maximum value**. The visual output is most useful when analysts overlay sector labels, market cap sizes, or other metadata to interpret what the algorithm has discovered.
