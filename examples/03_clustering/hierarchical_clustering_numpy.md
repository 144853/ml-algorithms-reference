# Hierarchical Clustering - Complete Guide with Stock Market Applications

## Overview

Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy of clusters by either progressively merging smaller clusters into larger ones (agglomerative, bottom-up) or recursively splitting larger clusters into smaller ones (divisive, top-down). The result is a tree-like structure called a dendrogram that shows the complete clustering history at every level of granularity. In stock market analysis, this hierarchical view is exceptionally valuable because it reveals not just which stocks are similar, but the degree and structure of their similarity -- creating a taxonomy of stocks by fundamental characteristics.

Agglomerative hierarchical clustering, the more common variant, starts with each data point as its own cluster and iteratively merges the two closest clusters until all points form a single cluster. The key decision is how to measure distance between clusters (linkage criterion): single linkage uses the minimum distance between any two points in the clusters, complete linkage uses the maximum, average linkage uses the mean, and Ward's method minimizes the total within-cluster variance. Each linkage produces different dendrogram shapes and is suited to different data structures.

For stock market taxonomy building, hierarchical clustering is uniquely powerful because portfolio managers need to understand relationships at multiple levels. At the highest level, stocks might split into "growth" and "value." Within growth, they might further divide into "high-momentum tech" and "emerging biotech." This multi-resolution view is impossible with flat clustering methods like K-Means, making hierarchical clustering the natural choice for building stock classification taxonomies based on fundamental and behavioral data.

## How It Works - The Math Behind It

### Agglomerative Algorithm

```
1. Initialize: Each data point is its own cluster: C = {{x_1}, {x_2}, ..., {x_n}}
2. Compute pairwise distance matrix D between all clusters
3. While |C| > 1:
   a. Find the two closest clusters: (C_i, C_j) = argmin_{i,j} D(C_i, C_j)
   b. Merge C_i and C_j into new cluster C_new = C_i ∪ C_j
   c. Remove C_i and C_j from C, add C_new
   d. Update distance matrix D for the new cluster
   e. Record the merge (linkage matrix entry)
4. Return linkage matrix (dendrogram)
```

### Linkage Criteria

**Single Linkage** (nearest neighbor):
```
D(C_i, C_j) = min_{x in C_i, y in C_j} d(x, y)
```
Tends to create elongated, chain-like clusters.

**Complete Linkage** (farthest neighbor):
```
D(C_i, C_j) = max_{x in C_i, y in C_j} d(x, y)
```
Tends to create compact, spherical clusters.

**Average Linkage** (UPGMA):
```
D(C_i, C_j) = (1 / (|C_i| * |C_j|)) * sum_{x in C_i} sum_{y in C_j} d(x, y)
```
Compromise between single and complete linkage.

**Ward's Linkage** (minimum variance):
```
D(C_i, C_j) = sqrt((2 * |C_i| * |C_j|) / (|C_i| + |C_j|)) * ||mu_i - mu_j||
```
Where mu_i and mu_j are cluster centroids. Minimizes the increase in total within-cluster variance upon merging.

### Lance-Williams Update Formula

Efficiently update distances after merging clusters C_i and C_j into C_k:
```
D(C_k, C_l) = alpha_i * D(C_i, C_l) + alpha_j * D(C_j, C_l)
              + beta * D(C_i, C_j) + gamma * |D(C_i, C_l) - D(C_j, C_l)|
```

Where coefficients depend on the linkage method:

| Linkage | alpha_i | alpha_j | beta | gamma |
|---------|---------|---------|------|-------|
| Single | 0.5 | 0.5 | 0 | -0.5 |
| Complete | 0.5 | 0.5 | 0 | 0.5 |
| Average | n_i/(n_i+n_j) | n_j/(n_i+n_j) | 0 | 0 |
| Ward | (n_l+n_i)/(n_l+n_k) | (n_l+n_j)/(n_l+n_k) | -n_l/(n_l+n_k) | 0 |

### Dendrogram and Cutting

The dendrogram records every merge as a tuple: `(cluster_i, cluster_j, distance, n_elements)`.

To obtain flat clusters, cut the dendrogram at a chosen height h or specify a desired number of clusters k:
```
clusters = cut_tree(dendrogram, height=h)
  OR
clusters = cut_tree(dendrogram, n_clusters=k)
```

### Complexity

- **Time Complexity**: O(n^3) naive, O(n^2 log n) with priority queues
- **Space Complexity**: O(n^2) for the distance matrix

## Stock Market Use Case: Building a Taxonomy of Stocks by Fundamental Similarity

### The Problem

An asset management firm wants to build a proprietary stock classification system based on actual fundamental characteristics rather than standard GICS sector codes. The traditional sector classification (Technology, Healthcare, Finance, etc.) groups companies by what they do, not by how they are financially structured. A profitable, low-debt technology company may have more in common financially with a consumer staples company than with a cash-burning tech startup. The firm needs a hierarchical taxonomy that reveals multi-level relationships: broad categories (growth vs. value), mid-level groups (large-cap growth vs. small-cap growth), and fine-grained peer groups for relative valuation.

### Stock Market Features (Input Data)

| Feature | Description | Range | Financial Meaning |
|---------|-------------|-------|-------------------|
| pe_ratio | Price-to-Earnings ratio | 5 - 100+ | Valuation: how much investors pay per dollar of earnings |
| pb_ratio | Price-to-Book ratio | 0.5 - 20 | Asset-based valuation metric |
| dividend_yield | Annual dividend / price | 0.0 - 0.08 | Income return to shareholders |
| debt_to_equity | Total debt / shareholders equity | 0.0 - 5.0 | Financial leverage and risk |
| roe | Return on equity | -0.10 - 0.50 | Profitability and efficiency |
| revenue_growth_yoy | Year-over-year revenue growth | -0.20 - 1.00 | Business expansion rate |
| profit_margin | Net income / revenue | -0.20 - 0.40 | Operating efficiency |
| market_cap_log | Log10 of market cap in millions | 2.0 - 6.0 | Company size |
| free_cash_flow_yield | FCF / market cap | -0.05 - 0.15 | Cash generation relative to price |
| earnings_volatility | Std dev of quarterly EPS growth | 0.05 - 2.0 | Earnings predictability |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)

# Generate 60 stocks across 4 fundamental archetypes
def generate_fundamental_data():
    stocks = {}

    # Large-cap Value (stable, profitable, dividends)
    for i in range(15):
        stocks[f"LCV_{i+1}"] = [
            np.random.normal(15, 3),        # low PE
            np.random.normal(2.0, 0.5),     # low PB
            np.random.normal(0.035, 0.01),  # decent dividend
            np.random.normal(0.8, 0.3),     # moderate debt
            np.random.normal(0.18, 0.04),   # good ROE
            np.random.normal(0.05, 0.03),   # slow growth
            np.random.normal(0.15, 0.04),   # good margins
            np.random.normal(4.8, 0.3),     # large cap
            np.random.normal(0.07, 0.02),   # good FCF yield
            np.random.normal(0.2, 0.08),    # low earnings vol
        ]

    # Large-cap Growth (expensive, fast-growing, reinvesting)
    for i in range(15):
        stocks[f"LCG_{i+1}"] = [
            np.random.normal(45, 12),       # high PE
            np.random.normal(8.0, 2.5),     # high PB
            np.random.normal(0.005, 0.005), # little/no dividend
            np.random.normal(0.5, 0.25),    # lower debt
            np.random.normal(0.25, 0.08),   # high ROE
            np.random.normal(0.25, 0.10),   # fast growth
            np.random.normal(0.20, 0.06),   # good margins
            np.random.normal(5.2, 0.3),     # large cap
            np.random.normal(0.02, 0.01),   # low FCF yield
            np.random.normal(0.5, 0.15),    # moderate earnings vol
        ]

    # Small-cap Growth (speculative, high growth, unprofitable)
    for i in range(15):
        stocks[f"SCG_{i+1}"] = [
            np.random.normal(80, 25),       # very high PE (or N/A)
            np.random.normal(5.0, 2.0),     # high PB
            np.random.normal(0.0, 0.001),   # no dividend
            np.random.normal(1.2, 0.5),     # moderate debt
            np.random.normal(0.02, 0.08),   # low/negative ROE
            np.random.normal(0.45, 0.20),   # very fast growth
            np.random.normal(0.02, 0.08),   # thin/negative margins
            np.random.normal(3.2, 0.4),     # small cap
            np.random.normal(-0.01, 0.02),  # negative FCF
            np.random.normal(1.2, 0.3),     # high earnings vol
        ]

    # Income/Defensive (utilities, REITs - high yield, low growth)
    for i in range(15):
        stocks[f"INC_{i+1}"] = [
            np.random.normal(18, 4),        # moderate PE
            np.random.normal(1.5, 0.4),     # low PB
            np.random.normal(0.055, 0.015), # high dividend
            np.random.normal(1.8, 0.6),     # higher debt (utilities)
            np.random.normal(0.10, 0.03),   # modest ROE
            np.random.normal(0.02, 0.02),   # minimal growth
            np.random.normal(0.12, 0.04),   # decent margins
            np.random.normal(4.0, 0.4),     # mid cap
            np.random.normal(0.05, 0.015),  # good FCF yield
            np.random.normal(0.15, 0.05),   # very stable earnings
        ]

    names = list(stocks.keys())
    data = np.array(list(stocks.values()))
    return data, names

stock_data, stock_names = generate_fundamental_data()
feature_names = [
    'PE', 'PB', 'DivYield', 'D/E', 'ROE', 'RevGrowth',
    'ProfitMargin', 'LogMktCap', 'FCFYield', 'EarningsVol'
]

print(f"Dataset: {stock_data.shape[0]} stocks, {stock_data.shape[1]} fundamental features")
```

### The Model in Action

```python
import numpy as np

def standardize(X):
    """Z-score standardization."""
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1
    return (X - means) / stds, means, stds

def pairwise_distances(X):
    """Compute Euclidean distance matrix."""
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(D, 0))

def agglomerative_clustering(X, linkage='ward'):
    """
    Agglomerative hierarchical clustering from scratch.

    Parameters:
        X: numpy array (n_samples, n_features)
        linkage: 'single', 'complete', 'average', or 'ward'

    Returns:
        Z: linkage matrix (n-1, 4) with columns:
           [cluster_i, cluster_j, distance, n_elements]
    """
    n = X.shape[0]
    # Initialize: each point is a cluster
    clusters = {i: [i] for i in range(n)}
    centroids = {i: X[i].copy() for i in range(n)}
    sizes = {i: 1 for i in range(n)}

    # Distance matrix (condensed form stored as full for simplicity)
    dist = pairwise_distances(X)
    np.fill_diagonal(dist, np.inf)

    Z = np.zeros((n - 1, 4))
    next_id = n

    for step in range(n - 1):
        # Find closest pair
        active = sorted(clusters.keys())
        min_dist = np.inf
        merge_i, merge_j = -1, -1

        for idx_a in range(len(active)):
            for idx_b in range(idx_a + 1, len(active)):
                i, j = active[idx_a], active[idx_b]
                if dist[i, j] < min_dist:
                    min_dist = dist[i, j]
                    merge_i, merge_j = i, j

        # Record merge
        Z[step] = [merge_i, merge_j, min_dist,
                   sizes[merge_i] + sizes[merge_j]]

        # Create new cluster
        new_members = clusters[merge_i] + clusters[merge_j]
        clusters[next_id] = new_members
        sizes[next_id] = len(new_members)

        # Compute new centroid (for Ward's method)
        ni, nj = sizes[merge_i], sizes[merge_j]
        centroids[next_id] = (ni * centroids[merge_i] + nj * centroids[merge_j]) / (ni + nj)

        # Update distances to new cluster
        # Expand distance matrix
        new_size = dist.shape[0] + 1
        new_dist = np.full((new_size, new_size), np.inf)
        new_dist[:dist.shape[0], :dist.shape[1]] = dist

        for k in clusters.keys():
            if k == next_id or k == merge_i or k == merge_j:
                continue
            if linkage == 'single':
                d = min(dist[merge_i, k] if merge_i < dist.shape[0] and k < dist.shape[1] else np.inf,
                        dist[merge_j, k] if merge_j < dist.shape[0] and k < dist.shape[1] else np.inf)
            elif linkage == 'complete':
                d = max(dist[merge_i, k], dist[merge_j, k])
            elif linkage == 'average':
                d = (ni * dist[merge_i, k] + nj * dist[merge_j, k]) / (ni + nj)
            elif linkage == 'ward':
                nk = sizes[k]
                nt = ni + nj + nk
                d = np.sqrt(
                    ((ni + nk) * dist[merge_i, k] ** 2 +
                     (nj + nk) * dist[merge_j, k] ** 2 -
                     nk * dist[merge_i, merge_j] ** 2) / nt
                )
            else:
                raise ValueError(f"Unknown linkage: {linkage}")

            new_dist[next_id, k] = d
            new_dist[k, next_id] = d

        dist = new_dist

        # Remove merged clusters
        del clusters[merge_i]
        del clusters[merge_j]

        next_id += 1

    return Z

def cut_dendrogram(Z, n_clusters):
    """Cut dendrogram to get flat cluster assignments."""
    n = int(Z[-1, 3])
    labels = np.arange(n)

    # Process merges in reverse order, stopping when we have n_clusters
    merges_to_undo = len(Z) - n_clusters + 1

    # Forward pass: assign cluster labels
    cluster_map = {i: i for i in range(n)}
    next_label = n

    for step in range(len(Z)):
        if step >= len(Z) - n_clusters + 1:
            break
        i, j = int(Z[step, 0]), int(Z[step, 1])
        # Merge: all points in cluster i and j get new label
        for point in range(n):
            if cluster_map[point] == i or cluster_map[point] == j:
                cluster_map[point] = next_label
        next_label += 1

    # Remap to 0, 1, ..., n_clusters-1
    unique_labels = sorted(set(cluster_map.values()))
    remap = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([remap[cluster_map[i]] for i in range(n)])

    return labels

def cophenetic_correlation(X, Z):
    """
    Compute cophenetic correlation coefficient.
    Measures how well the dendrogram preserves pairwise distances.
    """
    n = X.shape[0]
    original_dist = pairwise_distances(X)

    # Build cophenetic distance matrix from linkage
    coph_dist = np.zeros((n, n))
    # For each merge, the cophenetic distance between merged points
    # is the merge distance
    cluster_members = {i: [i] for i in range(n)}
    next_id = n

    for step in range(len(Z)):
        i, j = int(Z[step, 0]), int(Z[step, 1])
        d = Z[step, 2]

        members_i = cluster_members.get(i, [i] if i < n else [])
        members_j = cluster_members.get(j, [j] if j < n else [])

        for mi in members_i:
            for mj in members_j:
                coph_dist[mi, mj] = d
                coph_dist[mj, mi] = d

        cluster_members[next_id] = members_i + members_j
        next_id += 1

    # Correlation between original and cophenetic distances
    mask = np.triu_indices(n, k=1)
    orig_vals = original_dist[mask]
    coph_vals = coph_dist[mask]

    correlation = np.corrcoef(orig_vals, coph_vals)[0, 1]
    return correlation

# Run hierarchical clustering
X_std, means, stds = standardize(stock_data)
Z = agglomerative_clustering(X_std, linkage='ward')

# Cophenetic correlation
coph_corr = cophenetic_correlation(X_std, Z)
print(f"Cophenetic correlation: {coph_corr:.4f}")

# Cut at different levels
for k in [2, 3, 4, 6]:
    labels = cut_dendrogram(Z, n_clusters=k)
    print(f"\n=== {k} Clusters ===")
    for c in range(k):
        members = [stock_names[i] for i in range(len(labels)) if labels[i] == c]
        print(f"  Cluster {c} ({len(members)}): {members[:6]}...")
```

## Advantages

1. **Complete Hierarchical Taxonomy**: Unlike flat clustering methods, hierarchical clustering reveals the full tree of relationships. At the top level, stocks split into growth vs. value. Drilling deeper reveals sub-categories like large-cap growth vs. small-cap growth. This multi-resolution view is exactly what portfolio managers need for building classification systems.

2. **No Need to Pre-specify Number of Clusters**: The dendrogram contains all possible clusterings from 1 to n clusters. Analysts can explore the tree at any resolution and choose the cut level that best serves their investment objective. Different desks at the same firm can use different cuts from the same dendrogram.

3. **Visual Interpretability via Dendrograms**: The dendrogram provides an intuitive visual representation of stock relationships. Portfolio managers can immediately see which stocks are most similar (merge early) and which are most different (merge late). This visual tool is powerful for communicating complex relationships to non-quantitative stakeholders.

4. **Ward's Linkage Produces Balanced Clusters**: Ward's method minimizes variance increases, naturally producing clusters of similar size. For portfolio construction, this is desirable because it prevents one giant cluster of "everything else" alongside tiny niche clusters, which is a common problem with single linkage.

5. **Deterministic Results**: Given the same data, hierarchical clustering always produces the same dendrogram. This reproducibility is essential for regulatory compliance and audit trails in financial institutions where investment decisions must be traceable.

6. **Reveals Merger Distance Information**: The height at which two stocks or clusters merge in the dendrogram directly quantifies their dissimilarity. This distance information can be used to weight portfolio allocations -- stocks that merge early (very similar) should have inversely correlated positions, while those merging late can have independent positions.

7. **Flexible Post-Hoc Analysis**: Since the complete hierarchy is computed once, analysts can perform multiple analyses without re-running the algorithm. Need 3 peer groups? Cut at 3. Need 10 for detailed analysis? Cut at 10. The computational cost is paid once upfront.

## Disadvantages

1. **Quadratic Space Complexity**: The O(n^2) distance matrix becomes prohibitive for large stock universes. Clustering 5,000 stocks requires storing 25 million distance values. For global equity universes (10,000+ stocks), this memory requirement exceeds practical limits and requires sampling or approximate methods.

2. **Cubic Time Complexity**: The naive O(n^3) algorithm is slow for large datasets. Even with O(n^2 log n) optimizations, clustering 5,000 stocks takes significantly longer than K-Means. For daily re-clustering in quantitative trading systems, this latency may be unacceptable.

3. **Irreversible Merge Decisions**: Once two clusters are merged, they cannot be separated in subsequent steps. If an early merge is suboptimal (perhaps due to one noisy feature), all downstream merges are affected. A single bad decision propagates through the entire tree, and there is no mechanism for correction.

4. **Sensitivity to Linkage Choice**: Different linkage methods produce dramatically different dendrograms from the same data. Single linkage creates chaining effects (one outlier stock connecting unrelated groups), complete linkage may split natural groups, and Ward's assumes spherical clusters. The analyst must choose the right linkage for their purpose, and the choice is often non-obvious.

5. **Outlier Handling Depends on Linkage**: With single linkage, a single outlier stock (e.g., a recently IPO'd company with extreme fundamentals) can chain together unrelated clusters. With complete linkage, outliers form singleton clusters but inflate the distances for legitimate groupings. No linkage handles outliers perfectly.

6. **Static Hierarchy Does Not Adapt**: Stock fundamentals change over time as companies grow, take on debt, or change strategies. The hierarchical clustering produces a fixed taxonomy that does not update incrementally -- any change requires complete recomputation of the dendrogram.

7. **Cutting Height is Subjective**: While the dendrogram contains all clusterings, choosing where to cut still requires human judgment. Different cut heights produce different numbers of clusters, and there is no single objective criterion for the "right" number of stock groups. The gap statistic and silhouette methods help but do not eliminate subjectivity.

## When to Use in Stock Market

- **Building proprietary classification systems**: Creating multi-level stock taxonomies that replace or supplement GICS sector codes with fundamental-based groupings
- **Peer group identification**: Finding the true fundamental peers of a specific stock for relative valuation (comparing P/E ratios only within the correct peer group)
- **Portfolio hierarchy construction**: Organizing a portfolio into nested levels (broad asset classes, sub-sectors, peer groups) for layered risk management
- **Factor taxonomy creation**: Organizing factor exposures into a hierarchy to understand which factors are related and which are independent
- **Merger and acquisition analysis**: Identifying which companies are most similar to acquisition targets for comparable transaction analysis
- **Index construction**: Building hierarchical index structures where sub-indices correspond to dendrogram branches
- **Risk decomposition**: Breaking portfolio risk into hierarchical components (market risk, sector risk, stock-specific risk) aligned with the dendrogram structure

## When NOT to Use in Stock Market

- **Large stock universes (>5000 stocks)**: The O(n^2) memory and O(n^3) time make full hierarchical clustering impractical. Use K-Means or mini-batch approaches and reserve hierarchical for focused analyses
- **Real-time clustering needs**: Hierarchical clustering cannot incorporate new stocks incrementally. If stocks are added or removed daily, the entire dendrogram must be recomputed
- **When clusters have very different densities**: If large-cap and small-cap stocks exist in fundamentally different density regions, single and average linkage may produce poor results
- **When noise is prevalent**: Unlike DBSCAN, hierarchical clustering has no concept of noise points. Outlier stocks will be forced into the hierarchy somewhere, potentially distorting the tree
- **When only flat clusters are needed**: If the hierarchy itself provides no value and only the final cluster assignments matter, K-Means is simpler and faster
- **Streaming data**: Cannot handle online/incremental updates to the stock universe

## Hyperparameters Guide

| Parameter | Description | Options | Stock Market Recommendation |
|-----------|-------------|---------|----------------------------|
| linkage | How to measure cluster distance | single, complete, average, ward | Ward's for balanced fundamental groups; average for general use; avoid single (chaining risk) |
| metric | Point distance function | euclidean, manhattan, correlation | Correlation distance for returns-based clustering; euclidean for fundamentals after standardization |
| n_clusters | Number of flat clusters (cut level) | 2 - 20 | Start with 4-6 for broad groups; 8-12 for detailed peer groups; examine dendrogram gaps |
| cut_height | Height at which to cut dendrogram | depends on data | Cut at the largest gap in merge distances for natural clusters |

### Linkage Selection Guide for Finance

| Linkage | Best For | Pitfall | Stock Market Notes |
|---------|----------|---------|-------------------|
| Ward | Balanced, compact groups | Assumes spherical clusters | Best for fundamental taxonomy, produces even-sized peer groups |
| Complete | Tight, well-separated groups | Sensitive to outliers | Good for identifying distinct archetypes (value, growth, income) |
| Average | General-purpose grouping | May produce irregular shapes | Robust default choice when unsure about cluster structure |
| Single | Finding connected components | Chaining effect | Avoid for stocks; one outlier can chain unrelated companies |

## Stock Market Performance Tips

1. **Use Correlation Distance for Returns-Based Clustering**: When clustering stocks by return patterns, use `1 - abs(correlation)` as the distance metric instead of Euclidean distance. Stocks with correlation of 0.9 and -0.9 are both highly related (just in opposite directions) and should cluster near each other for risk management purposes.

2. **Winsorize Extreme Fundamentals Before Clustering**: P/E ratios for loss-making companies can be negative or infinite. Cap extreme values at the 5th and 95th percentiles before computing distances to prevent a few outliers from dominating the hierarchy.

3. **Log-Transform Skewed Features**: Features like market cap, trading volume, and P/E ratios are heavily right-skewed. Apply log transformation before standardization to make the distance metric more meaningful across the full range.

4. **Validate with Cophenetic Correlation**: The cophenetic correlation coefficient measures how well the dendrogram preserves the original pairwise distances. Values above 0.7 indicate a good hierarchical representation. If the cophenetic correlation is low, try a different linkage method.

5. **Use the Gap Statistic for Cut Height**: Generate random reference datasets, compute their dendrograms, and compare gap statistics to determine the optimal number of clusters. This provides a more objective cut point than visual inspection alone.

6. **Combine Fundamental and Technical Features Thoughtfully**: When mixing fundamental data (P/E, debt ratios) with technical data (momentum, volatility), normalize each group separately to prevent one category from dominating. Consider giving equal total weight to each feature category.

7. **Recompute Monthly, Not Daily**: Fundamental data changes quarterly, so daily recomputation is wasteful. Monthly recomputation captures earnings updates while keeping computation manageable.

## Comparison with Other Algorithms

| Feature | Hierarchical | K-Means | DBSCAN | Spectral |
|---------|-------------|---------|--------|----------|
| Output | Full dendrogram | Flat clusters | Flat clusters + noise | Flat clusters |
| Requires K upfront | No | Yes | No | Yes |
| Multi-resolution | Yes (cut anywhere) | No | No (vary eps) | No |
| Time complexity | O(n^2 log n) | O(nKdi) | O(n^2) | O(n^3) |
| Space complexity | O(n^2) | O(nK) | O(n^2) | O(n^2) |
| Handles outliers | Poor | Poor | Excellent | Moderate |
| Deterministic | Yes | No | Mostly | No |
| Incremental updates | No | Easy | No | No |
| Stock taxonomy | Excellent | Poor | N/A | Moderate |
| Interpretability | Excellent (dendrogram) | Good (centroids) | Moderate | Poor |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete Hierarchical Stock Taxonomy Pipeline
# ============================================================

np.random.seed(42)

# --- Generate Comprehensive Stock Fundamental Data ---
def create_stock_universe():
    """Create 80 stocks with realistic fundamental profiles."""
    stocks = []
    names = []
    true_categories = []

    archetypes = {
        'mega_cap_value': {
            'n': 12, 'prefix': 'MCV',
            'params': {
                'pe': (14, 2), 'pb': (2.5, 0.5), 'div_yield': (0.03, 0.008),
                'de': (0.7, 0.2), 'roe': (0.20, 0.03), 'rev_growth': (0.04, 0.02),
                'profit_margin': (0.18, 0.03), 'log_mcap': (5.5, 0.2),
                'fcf_yield': (0.06, 0.015), 'earn_vol': (0.15, 0.04)
            }
        },
        'large_cap_growth': {
            'n': 15, 'prefix': 'LCG',
            'params': {
                'pe': (40, 10), 'pb': (9, 3), 'div_yield': (0.003, 0.003),
                'de': (0.4, 0.2), 'roe': (0.28, 0.07), 'rev_growth': (0.22, 0.08),
                'profit_margin': (0.22, 0.05), 'log_mcap': (5.0, 0.3),
                'fcf_yield': (0.02, 0.01), 'earn_vol': (0.45, 0.12)
            }
        },
        'small_cap_growth': {
            'n': 13, 'prefix': 'SCG',
            'params': {
                'pe': (75, 20), 'pb': (5, 2), 'div_yield': (0.0, 0.001),
                'de': (1.0, 0.4), 'roe': (0.03, 0.06), 'rev_growth': (0.40, 0.15),
                'profit_margin': (0.01, 0.06), 'log_mcap': (3.0, 0.3),
                'fcf_yield': (-0.01, 0.02), 'earn_vol': (1.1, 0.3)
            }
        },
        'income_defensive': {
            'n': 12, 'prefix': 'INC',
            'params': {
                'pe': (17, 3), 'pb': (1.5, 0.3), 'div_yield': (0.05, 0.012),
                'de': (1.5, 0.4), 'roe': (0.10, 0.02), 'rev_growth': (0.02, 0.01),
                'profit_margin': (0.14, 0.03), 'log_mcap': (4.2, 0.3),
                'fcf_yield': (0.05, 0.01), 'earn_vol': (0.12, 0.03)
            }
        },
        'deep_value': {
            'n': 10, 'prefix': 'DV',
            'params': {
                'pe': (8, 2), 'pb': (0.8, 0.2), 'div_yield': (0.04, 0.01),
                'de': (1.2, 0.5), 'roe': (0.08, 0.03), 'rev_growth': (-0.02, 0.03),
                'profit_margin': (0.08, 0.03), 'log_mcap': (3.5, 0.4),
                'fcf_yield': (0.09, 0.02), 'earn_vol': (0.3, 0.1)
            }
        },
        'speculative': {
            'n': 10, 'prefix': 'SPEC',
            'params': {
                'pe': (90, 30), 'pb': (12, 4), 'div_yield': (0.0, 0.0),
                'de': (2.0, 0.8), 'roe': (-0.05, 0.10), 'rev_growth': (0.60, 0.25),
                'profit_margin': (-0.10, 0.08), 'log_mcap': (2.8, 0.4),
                'fcf_yield': (-0.03, 0.02), 'earn_vol': (1.5, 0.4)
            }
        },
        'quality_compounder': {
            'n': 8, 'prefix': 'QC',
            'params': {
                'pe': (28, 5), 'pb': (6, 1.5), 'div_yield': (0.01, 0.005),
                'de': (0.3, 0.1), 'roe': (0.35, 0.05), 'rev_growth': (0.12, 0.04),
                'profit_margin': (0.28, 0.04), 'log_mcap': (5.0, 0.2),
                'fcf_yield': (0.04, 0.01), 'earn_vol': (0.2, 0.05)
            }
        },
    }

    for cat_name, config in archetypes.items():
        for i in range(config['n']):
            p = config['params']
            stock = [
                max(1, np.random.normal(*p['pe'])),
                max(0.1, np.random.normal(*p['pb'])),
                max(0, np.random.normal(*p['div_yield'])),
                max(0, np.random.normal(*p['de'])),
                np.random.normal(*p['roe']),
                np.random.normal(*p['rev_growth']),
                np.random.normal(*p['profit_margin']),
                np.random.normal(*p['log_mcap']),
                np.random.normal(*p['fcf_yield']),
                max(0.01, np.random.normal(*p['earn_vol'])),
            ]
            stocks.append(stock)
            names.append(f"{config['prefix']}_{i+1}")
            true_categories.append(cat_name)

    return np.array(stocks), names, true_categories

# --- Hierarchical Clustering Implementation ---
def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma, mu, sigma

def dist_matrix(X):
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(D, 0))

def ward_clustering(X):
    """Ward's linkage hierarchical clustering."""
    n = X.shape[0]
    # Track active clusters
    active = set(range(n))
    members = {i: [i] for i in range(n)}
    centroids = {i: X[i].copy() for i in range(n)}
    cluster_sizes = {i: 1 for i in range(n)}

    # Initial distance matrix using Ward's criterion
    D = dist_matrix(X)
    np.fill_diagonal(D, np.inf)

    # Extend D as we create new clusters
    Z = []
    next_id = n

    for step in range(n - 1):
        # Find minimum distance among active clusters
        min_d = np.inf
        best_i, best_j = -1, -1
        active_list = sorted(active)

        for ai in range(len(active_list)):
            for aj in range(ai + 1, len(active_list)):
                ci, cj = active_list[ai], active_list[aj]
                if ci < D.shape[0] and cj < D.shape[1] and D[ci, cj] < min_d:
                    min_d = D[ci, cj]
                    best_i, best_j = ci, cj

        # Record merge
        ni, nj = cluster_sizes[best_i], cluster_sizes[best_j]
        Z.append([best_i, best_j, min_d, ni + nj])

        # New cluster
        new_centroid = (ni * centroids[best_i] + nj * centroids[best_j]) / (ni + nj)
        members[next_id] = members[best_i] + members[best_j]
        centroids[next_id] = new_centroid
        cluster_sizes[next_id] = ni + nj

        # Update distances using Ward's formula
        new_row = np.full(D.shape[1] + 1, np.inf)
        new_col = np.full(D.shape[0] + 1, np.inf)

        # Expand D
        D_new = np.full((D.shape[0] + 1, D.shape[1] + 1), np.inf)
        D_new[:D.shape[0], :D.shape[1]] = D

        for k in active:
            if k == best_i or k == best_j:
                continue
            nk = cluster_sizes[k]
            nt = ni + nj + nk
            d_ik = D[best_i, k] if best_i < D.shape[0] and k < D.shape[1] else np.inf
            d_jk = D[best_j, k] if best_j < D.shape[0] and k < D.shape[1] else np.inf
            d_ij = min_d

            ward_d = np.sqrt(
                ((ni + nk) * d_ik ** 2 +
                 (nj + nk) * d_jk ** 2 -
                 nk * d_ij ** 2) / nt
            )
            D_new[next_id, k] = ward_d
            D_new[k, next_id] = ward_d

        D = D_new
        active.discard(best_i)
        active.discard(best_j)
        active.add(next_id)
        next_id += 1

    return np.array(Z)

def get_flat_clusters(Z, n_samples, n_clusters):
    """Extract flat cluster labels from linkage matrix."""
    labels = np.arange(n_samples)
    cluster_map = {i: {i} for i in range(n_samples)}
    next_id = n_samples

    n_merges = len(Z) - n_clusters + 1
    for step in range(n_merges):
        ci, cj = int(Z[step, 0]), int(Z[step, 1])
        new_members = cluster_map.get(ci, {ci}) | cluster_map.get(cj, {cj})
        cluster_map[next_id] = new_members
        if ci in cluster_map: del cluster_map[ci]
        if cj in cluster_map: del cluster_map[cj]
        next_id += 1

    # Assign labels
    label_map = {}
    for label_id, (cluster_key, point_set) in enumerate(sorted(cluster_map.items())):
        for p in point_set:
            label_map[p] = label_id

    return np.array([label_map[i] for i in range(n_samples)])

# --- Run Pipeline ---
X_raw, stock_names, true_cats = create_stock_universe()
X_std, mu, sigma = standardize(X_raw)

feature_names = ['PE', 'PB', 'DivYield', 'D/E', 'ROE', 'RevGrowth',
                 'ProfMargin', 'LogMcap', 'FCFYield', 'EarnVol']

print(f"Stock universe: {len(stock_names)} stocks")
print(f"Features: {len(feature_names)}")

# Run Ward's hierarchical clustering
print("\nRunning Ward's hierarchical clustering...")
Z = ward_clustering(X_std)

# Analyze at multiple levels
print("\n" + "=" * 60)
print("HIERARCHICAL TAXONOMY")
print("=" * 60)

for n_clust in [2, 4, 7]:
    labels = get_flat_clusters(Z, len(stock_names), n_clust)
    print(f"\n--- {n_clust} Clusters (Level {'Broad' if n_clust==2 else 'Mid' if n_clust==4 else 'Detailed'}) ---")

    for c in range(n_clust):
        mask = labels == c
        member_names = [stock_names[i] for i in range(len(labels)) if labels[i] == c]
        true_comp = {}
        for i in range(len(labels)):
            if labels[i] == c:
                cat = true_cats[i]
                true_comp[cat] = true_comp.get(cat, 0) + 1

        print(f"\n  Cluster {c} ({np.sum(mask)} stocks):")
        print(f"    Composition: {true_comp}")
        print(f"    Key fundamentals:")
        for j, feat in enumerate(feature_names):
            val = np.mean(X_raw[mask, j])
            print(f"      {feat:>12s}: {val:>8.3f}")

# Dendrogram statistics
print(f"\n=== Dendrogram Statistics ===")
merge_distances = Z[:, 2]
print(f"  Min merge distance: {np.min(merge_distances):.4f}")
print(f"  Max merge distance: {np.max(merge_distances):.4f}")
print(f"  Mean merge distance: {np.mean(merge_distances):.4f}")

# Identify largest gaps (natural cluster boundaries)
gaps = np.diff(merge_distances)
top_gaps = np.argsort(gaps)[-5:][::-1]
print(f"\n  Largest gaps in merge distances (natural cut points):")
for g in top_gaps:
    n_clusters_at_cut = len(Z) - g
    print(f"    Cut at height {merge_distances[g]:.3f}-{merge_distances[g+1]:.3f} "
          f"-> {n_clusters_at_cut} clusters (gap={gaps[g]:.3f})")

# Peer group analysis for a specific stock
print(f"\n=== Peer Group Analysis ===")
target_idx = 0  # First stock
labels_detailed = get_flat_clusters(Z, len(stock_names), 7)
target_cluster = labels_detailed[target_idx]
peers = [stock_names[i] for i in range(len(labels_detailed))
         if labels_detailed[i] == target_cluster and i != target_idx]

print(f"  Target stock: {stock_names[target_idx]}")
print(f"  Fundamental peers ({len(peers)}): {peers}")
print(f"  Target PE: {X_raw[target_idx, 0]:.1f}")
peer_pes = [X_raw[i, 0] for i in range(len(labels_detailed))
            if labels_detailed[i] == target_cluster and i != target_idx]
print(f"  Peer avg PE: {np.mean(peer_pes):.1f} (range: {np.min(peer_pes):.1f}-{np.max(peer_pes):.1f})")
```

## Key Takeaways

1. **Hierarchical clustering creates a multi-level stock taxonomy** that reveals both broad categories (growth vs. value) and fine-grained peer groups from a single computation. This hierarchical view is impossible with flat clustering methods.

2. **Ward's linkage is generally best for stock fundamentals** because it produces balanced, compact clusters. Avoid single linkage for financial data as it creates chaining effects where one unusual stock connects unrelated groups.

3. **The dendrogram is both an analytical tool and a communication tool**. Use it to explore relationships between stocks at multiple levels and to explain clustering results to non-technical stakeholders.

4. **Feature preprocessing is critical**: log-transform skewed features (market cap, volume), winsorize extreme values (P/E ratios), and standardize before computing distances. Raw fundamental data produces poor hierarchies.

5. **Use cophenetic correlation to validate** that the dendrogram faithfully represents the original pairwise distances between stocks. Values above 0.7 indicate a reliable hierarchy.

6. **The largest gaps in merge distances** indicate natural cluster boundaries. Cutting the dendrogram at these gaps produces the most stable, well-separated stock groups.

7. **Hierarchical clustering excels at peer group identification** for relative valuation. By finding a stock's cluster at a fine-grained level, analysts get a set of true fundamental peers -- not just companies in the same sector.

8. **Recompute quarterly to align with earnings cycles**. Since fundamental data updates quarterly, monthly or weekly recomputation adds noise without new information. Quarterly recomputation captures genuine changes in company characteristics.
