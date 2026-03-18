# Collaborative Filtering - Complete Guide with Stock Market Applications

## Overview

Collaborative Filtering (CF) is a recommendation technique that makes predictions about a user's interests by collecting preference information from many users. The core assumption is that users who agreed in the past will agree in the future -- if two investors have similar portfolio compositions, they are likely to find value in similar stocks. CF does not require any information about the items themselves (unlike content-based filtering); it relies purely on the patterns of user-item interactions.

In the stock market context, collaborative filtering can be applied to recommend stocks to investors based on the collective wisdom of similar investors. Rather than analyzing financial fundamentals or technical indicators, CF identifies investors with similar holding patterns and suggests stocks that similar investors hold but the target investor does not. This approach captures latent preferences and investment philosophies that may not be easily codified through traditional financial analysis -- for example, ESG-conscious investing patterns or sector rotation strategies that emerge from portfolio data.

There are two primary approaches: user-based CF (find similar investors, recommend their stocks) and item-based CF (find similar stocks based on co-holding patterns, recommend related stocks). User-based CF answers "what do investors like you hold?" while item-based CF answers "what other stocks are held alongside the ones you already own?" Both approaches use similarity metrics like cosine similarity or Pearson correlation to measure closeness in the investor-stock interaction space.

## How It Works - The Math Behind It

### User-Item Interaction Matrix

The foundation of CF is the interaction matrix `R` where:

```
R[i][j] = rating/weight of stock j in investor i's portfolio
```

For stock markets, `R[i][j]` can represent:
- Portfolio weight (0 to 1)
- Binary holding indicator (0 or 1)
- Conviction score (scaled by position size relative to typical holding)
- Return-adjusted holding (weight * holding period return)

### User-Based Collaborative Filtering

1. **Compute user similarity** using cosine similarity:

```
sim(u, v) = (R_u . R_v) / (||R_u|| * ||R_v||)
           = sum_j (R[u][j] * R[v][j]) / (sqrt(sum_j R[u][j]^2) * sqrt(sum_j R[v][j]^2))
```

Or Pearson correlation:

```
sim(u, v) = sum_j (R[u][j] - R_u_bar)(R[v][j] - R_v_bar)
            / (sqrt(sum_j (R[u][j] - R_u_bar)^2) * sqrt(sum_j (R[v][j] - R_v_bar)^2))
```

2. **Predict rating** for user `u` on stock `j`:

```
pred(u, j) = R_u_bar + sum_{v in N(u)} sim(u, v) * (R[v][j] - R_v_bar)
                       / sum_{v in N(u)} |sim(u, v)|
```

where `N(u)` is the set of `k` nearest neighbors of user `u`.

### Item-Based Collaborative Filtering

1. **Compute item similarity** using cosine similarity between stock column vectors:

```
sim(i, j) = (R_i . R_j) / (||R_i|| * ||R_j||)
```

where `R_i` is the column vector of all investor ratings for stock `i`.

2. **Predict rating** for user `u` on stock `j`:

```
pred(u, j) = sum_{i in S(j)} sim(i, j) * R[u][i]
             / sum_{i in S(j)} |sim(i, j)|
```

where `S(j)` is the set of stocks most similar to `j` that user `u` has rated.

### Adjusted Cosine Similarity

For item-based CF, adjusted cosine removes user-level bias:

```
sim_adj(i, j) = sum_u (R[u][i] - R_u_bar)(R[u][j] - R_u_bar)
                / (sqrt(sum_u (R[u][i] - R_u_bar)^2) * sqrt(sum_u (R[u][j] - R_u_bar)^2))
```

### Top-K Recommendation

After computing predicted scores for all unrated items, recommend the top-K stocks:

```
recommend(u) = argsort(pred(u, :))[-K:]  for all j where R[u][j] = 0
```

## Stock Market Use Case: Recommending Stocks Based on Similar Investor Portfolios

### The Problem

A wealth management platform serves 10,000 investors who collectively hold positions in 2,000 stocks. The platform wants to:
- Suggest new stock ideas to investors based on similar investors' portfolios
- Help investors discover stocks they might have overlooked
- Identify stocks commonly held together for diversification analysis
- Power a "investors who hold X also hold Y" feature
- The system should work without requiring fundamental analysis, relying purely on collective portfolio patterns

### Stock Market Features (Input Data)

| Feature Name | Description | Data Type | Example Value |
|---|---|---|---|
| investor_id | Unique identifier for each investor | Integer | 4523 |
| stock_ticker | Stock symbol | String | AAPL |
| portfolio_weight | Weight of stock in portfolio (%) | Float | 8.5 |
| holding_days | Number of days held | Integer | 365 |
| conviction_score | Derived from weight and holding period | Float | 0.72 |
| sector | GICS sector classification | String | Technology |
| market_cap | Market capitalization bucket | String | Large-Cap |
| investor_style | Self-reported investment style | String | Growth |
| account_value | Total portfolio value bucket | String | $100K-$500K |
| num_holdings | Total stocks in portfolio | Integer | 25 |

### Example Data Structure

```python
import numpy as np

# Simulated investor-stock portfolio data
np.random.seed(42)

n_investors = 2000
n_stocks = 500
sparsity = 0.05  # Each investor holds ~5% of universe (25 stocks)

# Stock tickers
sectors = ['Tech', 'Healthcare', 'Finance', 'Energy', 'Consumer',
           'Industrial', 'Utilities', 'Materials', 'RealEstate', 'Telecom', 'Staples']
stock_tickers = [f"STK_{i:04d}" for i in range(n_stocks)]
stock_sectors = [sectors[i % len(sectors)] for i in range(n_stocks)]

# Create investor-stock interaction matrix
# Value = portfolio weight (0 means not held)
R = np.zeros((n_investors, n_stocks))

# Create investor archetypes (clusters of similar investors)
n_archetypes = 15
archetype_profiles = np.random.dirichlet(np.ones(n_stocks) * 0.1, n_archetypes)

for investor in range(n_investors):
    # Each investor is a noisy version of an archetype
    archetype = investor % n_archetypes
    base_profile = archetype_profiles[archetype]

    # Select top stocks from noisy profile
    noise = np.random.exponential(0.1, n_stocks)
    noisy_profile = base_profile + noise

    # Select ~25 stocks
    n_holdings = np.random.randint(15, 40)
    top_stocks = np.argsort(noisy_profile)[-n_holdings:]

    # Assign portfolio weights (sum to 1)
    weights = np.random.dirichlet(np.ones(n_holdings) * 2)
    R[investor, top_stocks] = weights

print(f"Interaction matrix shape: {R.shape}")
print(f"Sparsity: {1 - np.count_nonzero(R) / R.size:.2%}")
print(f"Avg holdings per investor: {np.mean(np.sum(R > 0, axis=1)):.1f}")
print(f"Avg investors per stock: {np.mean(np.sum(R > 0, axis=0)):.1f}")
```

### The Model in Action

Collaborative filtering processes the investor-stock matrix through these steps:

1. **Matrix construction**: Build the 2,000 x 500 investor-stock matrix where each cell contains the portfolio weight. Non-held stocks have weight 0.

2. **Similarity computation**: For user-based CF, compute pairwise cosine similarity between all investor portfolio vectors. For item-based CF, compute similarity between all stock columns. The result is a similarity matrix showing which investors (or stocks) are most alike.

3. **Neighbor selection**: For each target investor, identify the k most similar investors (e.g., k=50). Filter to those who hold stocks the target investor does not.

4. **Score aggregation**: For each candidate stock not in the target portfolio, aggregate the similarity-weighted portfolio weights from neighboring investors. Stocks held by many similar investors with high conviction receive the highest recommendation scores.

5. **Ranking and filtering**: Sort candidate stocks by predicted score, apply any sector or risk filters, and present the top-N recommendations.

## Advantages

1. **Captures latent investment philosophies without explicit modeling.** CF discovers that growth investors tend to cluster together without anyone defining what "growth investing" means algorithmically. It picks up on subtle patterns like ESG preferences, sector rotation strategies, or risk appetite that emerge from collective behavior.

2. **No financial domain knowledge required for the algorithm.** Unlike fundamental analysis, CF treats stocks as opaque items and investors as preference profiles. This makes it robust to market structural changes -- when new sectors emerge or traditional metrics break down, CF adapts based on what investors actually do rather than what models predict.

3. **Serendipitous discovery of overlooked stocks.** By surfacing stocks held by similar investors, CF can introduce an investor to companies outside their usual screening criteria. A tech-focused investor might discover a fintech company classified under financials that their peers already hold.

4. **Scalable to large investor universes.** Once the similarity matrix is computed, recommendations for any individual investor can be generated in milliseconds. This makes CF suitable for real-time recommendation on platforms serving millions of users.

5. **Naturally adapts to market regime changes.** As investors rebalance their portfolios in response to changing conditions, the interaction matrix updates and recommendations shift accordingly. No explicit regime detection is needed -- the collective behavior of investors encodes the current market environment.

6. **Complementary to fundamental and technical analysis.** CF provides an orthogonal signal to traditional stock analysis. Combining CF recommendations with fundamental screens and technical filters creates a multi-perspective system that catches opportunities each approach might miss individually.

## Disadvantages

1. **Cold start problem for new stocks and new investors.** When a stock IPOs, no investors hold it yet, so CF cannot recommend it. Similarly, a new investor with no portfolio history receives generic recommendations. This is especially problematic in rapidly evolving markets with frequent IPOs and new market participants.

2. **Popularity bias favors large-cap, widely-held stocks.** Stocks held by many investors (like AAPL, MSFT) appear similar to everything and dominate recommendations. Smaller, less widely-held stocks that might offer higher alpha are systematically underrepresented in CF recommendations.

3. **Scalability of similarity computation.** Computing pairwise similarity for 10,000 investors is O(n^2 * d), where d is the number of stocks. For very large platforms, this requires approximation techniques (locality-sensitive hashing, random projections) that introduce accuracy trade-offs.

4. **Herding risk from self-reinforcing recommendations.** If many investors follow CF recommendations, they may concentrate into the same positions, creating artificial demand and potential systemic risk. When unwinding occurs, the correlated positions amplify selling pressure.

5. **Cannot explain recommendations in financial terms.** CF says "investors like you hold stock X" but cannot articulate why -- is it the valuation, growth prospects, dividend yield, or sector rotation? This lack of financial reasoning makes it hard for investors to evaluate whether the recommendation fits their specific thesis.

6. **Sensitive to portfolio reporting frequency and accuracy.** CF relies on up-to-date portfolio data. If holdings are reported with delay (e.g., quarterly 13F filings), recommendations are based on stale information. In fast-moving markets, a stock recommended based on last quarter's holdings may no longer be attractive.

7. **Data sparsity in the investor-stock matrix.** With 2,000 stocks and each investor holding 25, the matrix is 98.75% zeros. Sparse matrices make similarity computation noisy and can lead to unreliable neighbor selection, especially for investors with unique or concentrated portfolios.

## When to Use in Stock Market

- Building "investors like you also hold" recommendation features for brokerage platforms
- Discovering stocks commonly co-held for portfolio construction insights
- Identifying investor clusters and archetype portfolios for market research
- Augmenting fundamental screening with behavioral signals from similar investors
- Creating watchlist suggestions for retail investor platforms
- Cross-selling opportunities for wealth management advisors

## When NOT to Use in Stock Market

- Timing entry/exit decisions (CF has no temporal signal)
- When portfolio data is stale or infrequently updated (quarterly 13F filings are too slow)
- For very concentrated portfolios (hedge funds with 5-10 positions lack enough overlap)
- When the investor universe is small (<100 investors) and sparsity is extreme
- For derivative or options strategies where the interaction matrix is fundamentally different
- When regulatory constraints require explainable recommendations backed by financial analysis
- Real-time trading signals requiring sub-second latency

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Guidance |
|---|---|---|---|
| `k` (neighbors) | Number of similar investors/stocks to consider | 10 to 100 | 30-50 for user-based; 20-30 for item-based |
| `similarity_metric` | Cosine, Pearson, or adjusted cosine | - | Cosine for binary holdings; Pearson for weighted portfolios |
| `min_common` | Minimum overlapping holdings for valid similarity | 3 to 10 | At least 5 common stocks to ensure meaningful similarity |
| `threshold` | Minimum similarity score to consider a neighbor | 0.0 to 0.5 | 0.1 for user-based; 0.2 for item-based |
| `normalization` | Mean-centering or z-score normalization | - | Mean-center portfolio weights to remove position-size bias |
| `weighting` | Uniform or similarity-weighted aggregation | - | Similarity-weighted produces better rankings |
| `top_n` | Number of recommendations to return | 5 to 20 | 10 recommendations with sector diversity filter |

## Stock Market Performance Tips

1. **Use conviction scores instead of raw portfolio weights.** Combine position size with holding period to create a conviction metric: `conviction = weight * log(1 + holding_days / 30)`. This better reflects true investor preference than snapshot weights.

2. **Apply sector-aware diversification to recommendations.** Post-filter recommendations to ensure sector diversity. If the top 10 recommendations are all tech stocks, force inclusion of other sectors to avoid concentrating risk.

3. **Implement time-decay for stale holdings.** Weight recent portfolio changes more heavily than long-standing positions. An investor who just added a stock has stronger signal than one who passively holds an inherited position.

4. **Segment investors before computing similarities.** Compute similarities within investor segments (e.g., growth vs. value, retail vs. institutional) to produce more relevant recommendations. A retail investor's neighbors should be other retail investors, not hedge funds.

5. **Combine user-based and item-based approaches.** Use item-based CF for the "related stocks" feature and user-based CF for personalized recommendations. The hybrid captures both stock-level co-holding patterns and investor-level preference patterns.

6. **Refresh the similarity matrix regularly.** Recompute similarities weekly or after major market events (earnings season, FOMC decisions) when portfolio compositions change significantly.

## Comparison with Other Algorithms

| Aspect | User-Based CF | Item-Based CF | Matrix Factorization | Content-Based | Knowledge Graph |
|---|---|---|---|---|---|
| Data Required | Portfolio holdings | Portfolio holdings | Portfolio holdings | Stock features | Feature + relationships |
| Cold Start (new stock) | Partial | Yes | Yes | No | No |
| Cold Start (new investor) | Yes | Partial | Yes | No | No |
| Scalability | O(n_users^2) | O(n_items^2) | O(k * n_entries) | O(n_items * features) | Varies |
| Interpretability | "Similar investors hold X" | "Stocks like Y" | Low (latent factors) | "Based on features" | "Based on relationships" |
| Serendipity | High | Medium | Medium | Low | Medium |
| Popularity Bias | High | Medium | Tunable | Low | Low |
| Best For (Finance) | Personalized picks | Related stocks | Latent factor discovery | Feature-based screening | Complex reasoning |

## Real-World Stock Market Example

```python
import numpy as np

# ================================================================
# Collaborative Filtering: Stock Recommendation System
# User-Based and Item-Based CF from scratch
# ================================================================

class StockCollaborativeFilter:
    """
    Collaborative Filtering for stock recommendations based on
    investor portfolio holdings.
    """

    def __init__(self, method='user', k_neighbors=30, min_common=5,
                 similarity='cosine'):
        self.method = method          # 'user' or 'item'
        self.k_neighbors = k_neighbors
        self.min_common = min_common
        self.similarity_type = similarity
        self.sim_matrix = None
        self.R = None
        self.R_mean = None

    def cosine_similarity(self, A):
        """Compute pairwise cosine similarity."""
        norms = np.sqrt(np.sum(A ** 2, axis=1, keepdims=True)) + 1e-8
        A_norm = A / norms
        return A_norm @ A_norm.T

    def pearson_similarity(self, A):
        """Compute pairwise Pearson correlation."""
        A_centered = A - np.mean(A, axis=1, keepdims=True)
        norms = np.sqrt(np.sum(A_centered ** 2, axis=1, keepdims=True)) + 1e-8
        A_norm = A_centered / norms
        return A_norm @ A_norm.T

    def compute_similarity(self, A):
        """Compute similarity matrix based on chosen metric."""
        if self.similarity_type == 'cosine':
            return self.cosine_similarity(A)
        elif self.similarity_type == 'pearson':
            return self.pearson_similarity(A)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity_type}")

    def fit(self, R):
        """
        Fit the collaborative filter.
        R: investor-stock interaction matrix (n_investors x n_stocks)
        """
        self.R = R.copy()
        self.R_mean = np.zeros(R.shape[0])

        # Compute per-user mean (only over rated items)
        for i in range(R.shape[0]):
            rated = R[i] > 0
            if np.sum(rated) > 0:
                self.R_mean[i] = np.mean(R[i, rated])

        if self.method == 'user':
            self.sim_matrix = self.compute_similarity(R)
            # Zero out self-similarity
            np.fill_diagonal(self.sim_matrix, 0)
        elif self.method == 'item':
            self.sim_matrix = self.compute_similarity(R.T)
            np.fill_diagonal(self.sim_matrix, 0)

        return self

    def predict_user_based(self, user_id, stock_id):
        """Predict rating using user-based CF."""
        # Find users who rate this stock
        rated_mask = self.R[:, stock_id] > 0

        # Get similarities with these users
        sims = self.sim_matrix[user_id].copy()
        sims[~rated_mask] = 0  # Only consider users who hold this stock
        sims[user_id] = 0       # Exclude self

        # Check minimum common holdings
        common = np.sum((self.R[user_id] > 0) & (self.R > 0), axis=1)
        sims[common < self.min_common] = 0

        # Select top-k neighbors
        top_k_idx = np.argsort(sims)[-self.k_neighbors:]
        top_k_sims = sims[top_k_idx]

        if np.sum(np.abs(top_k_sims)) < 1e-8:
            return self.R_mean[user_id]

        # Weighted average of neighbor ratings
        numerator = np.sum(top_k_sims * (self.R[top_k_idx, stock_id] - self.R_mean[top_k_idx]))
        denominator = np.sum(np.abs(top_k_sims))

        return self.R_mean[user_id] + numerator / denominator

    def predict_item_based(self, user_id, stock_id):
        """Predict rating using item-based CF."""
        # Find stocks the user holds
        held_mask = self.R[user_id] > 0

        # Get similarities of target stock with held stocks
        sims = self.sim_matrix[stock_id].copy()
        sims[~held_mask] = 0
        sims[stock_id] = 0

        # Select top-k similar stocks
        top_k_idx = np.argsort(sims)[-self.k_neighbors:]
        top_k_sims = sims[top_k_idx]

        if np.sum(np.abs(top_k_sims)) < 1e-8:
            return self.R_mean[user_id]

        # Weighted average
        numerator = np.sum(top_k_sims * self.R[user_id, top_k_idx])
        denominator = np.sum(np.abs(top_k_sims))

        return numerator / denominator

    def predict(self, user_id, stock_id):
        """Predict rating based on method."""
        if self.method == 'user':
            return self.predict_user_based(user_id, stock_id)
        else:
            return self.predict_item_based(user_id, stock_id)

    def recommend(self, user_id, n_recommendations=10, exclude_held=True):
        """Generate top-N stock recommendations for an investor."""
        n_stocks = self.R.shape[1]
        scores = np.zeros(n_stocks)

        for stock_id in range(n_stocks):
            if exclude_held and self.R[user_id, stock_id] > 0:
                scores[stock_id] = -np.inf  # Exclude already held
            else:
                scores[stock_id] = self.predict(user_id, stock_id)

        # Return top-N stock indices and scores
        top_n_idx = np.argsort(scores)[-n_recommendations:][::-1]
        return [(idx, scores[idx]) for idx in top_n_idx]

    def find_similar_investors(self, user_id, n=10):
        """Find the most similar investors."""
        if self.method != 'user':
            sim = self.cosine_similarity(self.R)
        else:
            sim = self.sim_matrix

        sims = sim[user_id].copy()
        sims[user_id] = -np.inf
        top_idx = np.argsort(sims)[-n:][::-1]
        return [(idx, sims[idx]) for idx in top_idx]

    def find_similar_stocks(self, stock_id, n=10):
        """Find stocks most commonly co-held."""
        if self.method != 'item':
            sim = self.cosine_similarity(self.R.T)
        else:
            sim = self.sim_matrix

        sims = sim[stock_id].copy()
        sims[stock_id] = -np.inf
        top_idx = np.argsort(sims)[-n:][::-1]
        return [(idx, sims[idx]) for idx in top_idx]


# ================================================================
# Generate Stock Market Data
# ================================================================

np.random.seed(42)

n_investors = 2000
n_stocks = 500

# Define investor archetypes with characteristic holding patterns
archetypes = {
    'Tech Growth': np.random.dirichlet(np.ones(n_stocks) * 0.05),
    'Value Dividend': np.random.dirichlet(np.ones(n_stocks) * 0.05),
    'Balanced': np.random.dirichlet(np.ones(n_stocks) * 0.1),
    'Small Cap': np.random.dirichlet(np.ones(n_stocks) * 0.05),
    'Sector Rotator': np.random.dirichlet(np.ones(n_stocks) * 0.08),
    'Index Hugger': np.random.dirichlet(np.ones(n_stocks) * 0.2),
    'Concentrated': np.random.dirichlet(np.ones(n_stocks) * 0.02),
    'Global Macro': np.random.dirichlet(np.ones(n_stocks) * 0.07),
}

archetype_names = list(archetypes.keys())
archetype_profiles = list(archetypes.values())

# Assign each investor to an archetype with noise
investor_archetypes = []
R = np.zeros((n_investors, n_stocks))

for i in range(n_investors):
    arch_idx = i % len(archetype_names)
    investor_archetypes.append(archetype_names[arch_idx])

    base = archetype_profiles[arch_idx].copy()
    noise = np.random.exponential(0.05, n_stocks)
    profile = base + noise

    # Select holdings (15-40 stocks)
    n_holdings = np.random.randint(15, 40)
    top_stocks = np.argsort(profile)[-n_holdings:]

    # Assign weights summing to 1
    weights = np.random.dirichlet(np.ones(n_holdings) * 3)
    R[i, top_stocks] = weights

# Create stock metadata
stock_tickers = [f"STK_{i:04d}" for i in range(n_stocks)]
sectors = ['Tech', 'Healthcare', 'Finance', 'Energy', 'Consumer',
           'Industrial', 'Utilities', 'Materials', 'RealEstate', 'Telecom']
stock_sectors = [sectors[i % len(sectors)] for i in range(n_stocks)]

print(f"Investor-Stock Matrix: {R.shape}")
print(f"Non-zero entries: {np.count_nonzero(R)}")
print(f"Sparsity: {1 - np.count_nonzero(R) / R.size:.2%}")
print(f"Avg holdings per investor: {np.mean(np.sum(R > 0, axis=1)):.1f}")
print(f"Avg investors per stock: {np.mean(np.sum(R > 0, axis=0)):.1f}")

# ================================================================
# Train-Test Split
# ================================================================

# Hold out 20% of each investor's holdings for evaluation
R_train = R.copy()
R_test = np.zeros_like(R)

for i in range(n_investors):
    held_stocks = np.where(R[i] > 0)[0]
    if len(held_stocks) > 5:
        n_test = max(1, len(held_stocks) // 5)
        test_stocks = np.random.choice(held_stocks, n_test, replace=False)
        R_test[i, test_stocks] = R[i, test_stocks]
        R_train[i, test_stocks] = 0

print(f"\nTrain non-zeros: {np.count_nonzero(R_train)}")
print(f"Test non-zeros: {np.count_nonzero(R_test)}")

# ================================================================
# Train User-Based CF
# ================================================================

print("\n=== User-Based Collaborative Filtering ===")
user_cf = StockCollaborativeFilter(method='user', k_neighbors=30,
                                    min_common=3, similarity='cosine')
user_cf.fit(R_train)

# Evaluate: for each test entry, predict and compute error
user_preds = []
user_actuals = []
for i in range(min(200, n_investors)):  # Evaluate subset for speed
    test_stocks = np.where(R_test[i] > 0)[0]
    for j in test_stocks:
        pred = user_cf.predict(i, j)
        user_preds.append(pred)
        user_actuals.append(R_test[i, j])

user_preds = np.array(user_preds)
user_actuals = np.array(user_actuals)
user_rmse = np.sqrt(np.mean((user_preds - user_actuals) ** 2))
user_mae = np.mean(np.abs(user_preds - user_actuals))
print(f"User-Based RMSE: {user_rmse:.6f}")
print(f"User-Based MAE: {user_mae:.6f}")

# ================================================================
# Train Item-Based CF
# ================================================================

print("\n=== Item-Based Collaborative Filtering ===")
item_cf = StockCollaborativeFilter(method='item', k_neighbors=20,
                                    min_common=3, similarity='cosine')
item_cf.fit(R_train)

item_preds = []
item_actuals = []
for i in range(min(200, n_investors)):
    test_stocks = np.where(R_test[i] > 0)[0]
    for j in test_stocks:
        pred = item_cf.predict(i, j)
        item_preds.append(pred)
        item_actuals.append(R_test[i, j])

item_preds = np.array(item_preds)
item_actuals = np.array(item_actuals)
item_rmse = np.sqrt(np.mean((item_preds - item_actuals) ** 2))
item_mae = np.mean(np.abs(item_preds - item_actuals))
print(f"Item-Based RMSE: {item_rmse:.6f}")
print(f"Item-Based MAE: {item_mae:.6f}")

# ================================================================
# Generate Recommendations
# ================================================================

print("\n=== Sample Recommendations ===")
target_investor = 0
print(f"\nInvestor {target_investor} (Archetype: {investor_archetypes[target_investor]})")
print(f"Current holdings: {np.sum(R_train[target_investor] > 0)} stocks")

# Top held stocks
held_idx = np.argsort(R_train[target_investor])[::-1]
print(f"\nTop 5 current holdings:")
for idx in held_idx[:5]:
    if R_train[target_investor, idx] > 0:
        print(f"  {stock_tickers[idx]} ({stock_sectors[idx]}): "
              f"weight={R_train[target_investor, idx]:.4f}")

# User-based recommendations
print(f"\nUser-Based Recommendations:")
user_recs = user_cf.recommend(target_investor, n_recommendations=10)
for rank, (stock_idx, score) in enumerate(user_recs, 1):
    print(f"  {rank}. {stock_tickers[stock_idx]} ({stock_sectors[stock_idx]}): "
          f"score={score:.4f}")

# Item-based recommendations
print(f"\nItem-Based Recommendations:")
item_recs = item_cf.recommend(target_investor, n_recommendations=10)
for rank, (stock_idx, score) in enumerate(item_recs, 1):
    print(f"  {rank}. {stock_tickers[stock_idx]} ({stock_sectors[stock_idx]}): "
          f"score={score:.4f}")

# ================================================================
# Find Similar Investors
# ================================================================

print(f"\n=== Similar Investors to Investor {target_investor} ===")
similar_investors = user_cf.find_similar_investors(target_investor, n=5)
for inv_id, sim_score in similar_investors:
    common = np.sum((R_train[target_investor] > 0) & (R_train[inv_id] > 0))
    print(f"  Investor {inv_id} (Archetype: {investor_archetypes[inv_id]}): "
          f"similarity={sim_score:.4f}, common_holdings={common}")

# ================================================================
# Find Related Stocks
# ================================================================

print(f"\n=== Stocks Similar to {stock_tickers[0]} ({stock_sectors[0]}) ===")
similar_stocks = item_cf.find_similar_stocks(0, n=10)
for stock_idx, sim_score in similar_stocks:
    co_holders = np.sum((R_train[:, 0] > 0) & (R_train[:, stock_idx] > 0))
    print(f"  {stock_tickers[stock_idx]} ({stock_sectors[stock_idx]}): "
          f"similarity={sim_score:.4f}, co-held by {co_holders} investors")

# ================================================================
# Evaluation Metrics
# ================================================================

print(f"\n=== Hit Rate Analysis ===")
hit_at_10 = 0
hit_at_20 = 0
ndcg_at_10 = 0
n_eval = 0

for i in range(min(200, n_investors)):
    test_stocks = set(np.where(R_test[i] > 0)[0])
    if len(test_stocks) == 0:
        continue

    recs = user_cf.recommend(i, n_recommendations=20)
    rec_stocks = [r[0] for r in recs]

    # Hit rate
    hits_10 = len(set(rec_stocks[:10]) & test_stocks)
    hits_20 = len(set(rec_stocks[:20]) & test_stocks)
    hit_at_10 += 1 if hits_10 > 0 else 0
    hit_at_20 += 1 if hits_20 > 0 else 0

    # NDCG@10
    dcg = 0
    for rank, stock_idx in enumerate(rec_stocks[:10]):
        if stock_idx in test_stocks:
            dcg += 1.0 / np.log2(rank + 2)
    ideal_dcg = sum(1.0 / np.log2(k + 2) for k in range(min(len(test_stocks), 10)))
    ndcg_at_10 += dcg / (ideal_dcg + 1e-8)
    n_eval += 1

print(f"Hit Rate @10: {hit_at_10 / n_eval:.2%}")
print(f"Hit Rate @20: {hit_at_20 / n_eval:.2%}")
print(f"NDCG @10: {ndcg_at_10 / n_eval:.4f}")

# ================================================================
# Summary
# ================================================================

print(f"\n=== Model Comparison Summary ===")
print(f"{'Metric':<20} {'User-Based':>12} {'Item-Based':>12}")
print(f"{'-'*44}")
print(f"{'RMSE':<20} {user_rmse:>12.6f} {item_rmse:>12.6f}")
print(f"{'MAE':<20} {user_mae:>12.6f} {item_mae:>12.6f}")
```

## Key Takeaways

1. **Collaborative filtering captures collective investor wisdom** without requiring any financial domain knowledge -- it discovers patterns purely from portfolio composition data, identifying investment styles and stock relationships that emerge from behavior rather than fundamental analysis.

2. **User-based CF answers "what do investors like you hold?"** while item-based CF answers "what stocks are commonly held alongside yours?" -- both perspectives provide valuable and often complementary stock discovery signals.

3. **The cold start problem is a real limitation in stock markets** -- newly IPO'd stocks and new investors without history cannot be served by pure CF, requiring hybrid approaches that combine CF with content-based features.

4. **Popularity bias must be actively managed** to prevent recommendations from degenerating into a list of the most widely held large-cap stocks. Techniques like inverse-popularity weighting and sector diversification filters help surface more unique opportunities.

5. **Portfolio weight provides richer signal than binary holding indicators** -- using conviction scores that combine position size and holding duration captures investor preference intensity, not just presence.

6. **Regular recomputation of the similarity matrix is essential** as portfolios evolve, especially around earnings seasons, index rebalancings, and major macro events that trigger widespread portfolio adjustments.
