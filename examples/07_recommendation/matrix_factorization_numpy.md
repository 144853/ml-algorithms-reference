# Matrix Factorization - Complete Guide with Stock Market Applications

## Overview

Matrix Factorization (MF) is a dimensionality reduction and collaborative filtering technique that decomposes a large, sparse interaction matrix into the product of two lower-rank matrices. Each row of one matrix represents a user (investor) in a low-dimensional latent space, and each row of the other represents an item (stock) in the same space. The dot product of an investor's latent vector with a stock's latent vector approximates the original interaction value, effectively filling in missing entries and revealing hidden structure in the data.

In the stock market, matrix factorization discovers latent factors that explain why certain investors hold certain stocks. These factors might correspond to interpretable concepts like "growth vs. value orientation," "risk appetite," "sector preference," or "momentum sensitivity" -- though they emerge purely from data without explicit labeling. By decomposing the investor-stock preference matrix, MF simultaneously learns investor profiles and stock characteristics in a shared latent space, enabling predictions about which stocks an investor would prefer and which investors would be interested in a given stock.

The most common approach is regularized SVD or Alternating Least Squares (ALS), which handles the extreme sparsity of portfolio data gracefully. Unlike full SVD, these methods only fit to observed entries (stocks actually held), avoiding the problem of treating missing entries as zero preferences. The resulting low-rank approximation captures the dominant patterns while filtering out noise, making it particularly effective for the noisy, high-dimensional investor-stock interaction data found in financial markets.

## How It Works - The Math Behind It

### Matrix Decomposition

Given an interaction matrix `R` of shape `(m x n)` where `m` = investors and `n` = stocks:

```
R ≈ P @ Q^T
```

where:
- `P` is the investor factor matrix of shape `(m x k)`
- `Q` is the stock factor matrix of shape `(n x k)`
- `k` is the number of latent factors (k << min(m, n))

Each entry is approximated as:

```
R_hat[i][j] = P[i] . Q[j] = sum_{f=1}^{k} P[i][f] * Q[j][f]
```

### Objective Function with Regularization

Minimize the regularized squared error over observed entries:

```
L = sum_{(i,j) in Omega} (R[i][j] - P[i] . Q[j] - b_i - c_j - mu)^2
    + lambda_P * ||P||_F^2 + lambda_Q * ||Q||_F^2
    + lambda_b * (sum_i b_i^2 + sum_j c_j^2)
```

where:
- `Omega` is the set of observed entries
- `mu` is the global mean
- `b_i` is the investor bias (tendency to over/underweight)
- `c_j` is the stock bias (popularity effect)
- `lambda` terms control regularization strength
- `||.||_F` is the Frobenius norm

### Stochastic Gradient Descent (SGD)

For each observed entry `(i, j)`:

```
e_ij = R[i][j] - (mu + b_i + c_j + P[i] . Q[j])

P[i] = P[i] + lr * (e_ij * Q[j] - lambda_P * P[i])
Q[j] = Q[j] + lr * (e_ij * P[i] - lambda_Q * Q[j])
b_i  = b_i  + lr * (e_ij - lambda_b * b_i)
c_j  = c_j  + lr * (e_ij - lambda_b * c_j)
```

### Alternating Least Squares (ALS)

Fix `Q` and solve for each `P[i]`:

```
P[i] = (Q_i^T Q_i + lambda_P * I)^{-1} Q_i^T R[i]
```

where `Q_i` is the submatrix of `Q` for stocks rated by investor `i`.

Then fix `P` and solve for each `Q[j]`:

```
Q[j] = (P_j^T P_j + lambda_Q * I)^{-1} P_j^T R[:, j]
```

where `P_j` is the submatrix of `P` for investors who rate stock `j`.

### Non-Negative Matrix Factorization (NMF)

When entries are non-negative (portfolio weights), enforce `P >= 0` and `Q >= 0`:

```
P[i][f] = P[i][f] * (R @ Q)[i][f] / (P @ Q^T @ Q)[i][f]
Q[j][f] = Q[j][f] * (R^T @ P)[j][f] / (Q @ P^T @ P)[j][f]
```

NMF produces purely additive decompositions where latent factors represent parts rather than contrasts, making them more interpretable for portfolio analysis.

### Singular Value Decomposition (SVD)

The truncated SVD provides the optimal rank-k approximation:

```
R ≈ U_k @ Sigma_k @ V_k^T
```

where `U_k`, `Sigma_k`, `V_k` contain the top-k singular vectors/values. The investor factors are `P = U_k @ sqrt(Sigma_k)` and stock factors are `Q = V_k @ sqrt(Sigma_k)`.

## Stock Market Use Case: Discovering Latent Factors in Stock-Investor Preference Matrices

### The Problem

A prime brokerage aggregates portfolio data from 5,000 institutional investors holding positions in 3,000 stocks. The firm wants to:
- Discover the hidden factors that explain investor-stock preferences
- Predict which stocks an investor is likely to add next
- Group stocks by latent characteristics beyond traditional sector classification
- Identify investor archetypes based on their latent factor profiles
- Build a recommendation engine for the firm's stock research distribution

### Stock Market Features (Input Data)

| Feature Name | Description | Data Type | Example Value |
|---|---|---|---|
| investor_id | Unique institutional investor ID | Integer | 1247 |
| stock_id | Unique stock identifier | Integer | 892 |
| portfolio_weight | Normalized position weight | Float | 0.035 |
| conviction_score | Weight * holding_period factor | Float | 0.68 |
| active_weight | Weight relative to benchmark | Float | 0.015 |
| turnover_rate | Trading frequency in stock | Float | 0.25 |
| investor_aum | Assets under management (bucket) | String | $1B-$10B |
| investor_type | Hedge fund / Mutual fund / Pension | String | Hedge Fund |
| stock_sector | GICS sector | String | Technology |
| stock_market_cap | Market capitalization ($B) | Float | 125.5 |

### Example Data Structure

```python
import numpy as np

# Simulated institutional investor-stock preference data
np.random.seed(42)

n_investors = 5000
n_stocks = 3000
n_latent_factors = 10
sparsity_rate = 0.98  # 98% sparse (each investor holds ~60 stocks)

# Ground truth: investors and stocks live in a latent factor space
true_P = np.random.randn(n_investors, n_latent_factors) * 0.5  # investor factors
true_Q = np.random.randn(n_stocks, n_latent_factors) * 0.5     # stock factors

# True interaction matrix (low-rank)
R_true = true_P @ true_Q.T

# Convert to non-negative preferences (portfolio-like)
R_true = np.maximum(R_true, 0)

# Sparsify: each investor only holds a subset of stocks
R_observed = np.zeros((n_investors, n_stocks))
for i in range(n_investors):
    # Higher latent scores = more likely to hold
    probs = R_true[i] / (np.sum(R_true[i]) + 1e-8)
    n_holdings = np.random.randint(20, 80)
    held = np.random.choice(n_stocks, n_holdings, replace=False, p=probs)

    # Portfolio weights
    weights = np.random.dirichlet(np.ones(n_holdings) * 2)
    R_observed[i, held] = weights

# Add noise
noise = np.random.randn(*R_observed.shape) * 0.005
R_observed = np.maximum(R_observed + noise * (R_observed > 0), 0)

print(f"Interaction matrix: {R_observed.shape}")
print(f"Non-zero entries: {np.count_nonzero(R_observed):,}")
print(f"Sparsity: {1 - np.count_nonzero(R_observed) / R_observed.size:.2%}")
print(f"Avg holdings per investor: {np.mean(np.sum(R_observed > 0, axis=1)):.1f}")
```

### The Model in Action

Matrix factorization processes the sparse investor-stock matrix through iterative optimization:

1. **Initialization**: Randomly initialize investor factor matrix P (5000 x k) and stock factor matrix Q (3000 x k) where k=10 latent factors. Also initialize investor biases, stock biases, and the global mean.

2. **SGD iteration**: For each observed entry (investor i holds stock j with weight w_ij), compute the prediction error and update the latent vectors for investor i and stock j to reduce the error. Regularization prevents overfitting.

3. **Convergence**: After 30-50 epochs over all observed entries, the factor matrices converge. The product P @ Q^T now approximates the full matrix, including predictions for unobserved (investor, stock) pairs.

4. **Factor interpretation**: Each of the k latent factors captures a dimension of the investor-stock preference space. By examining which stocks load heavily on each factor, we can interpret what the factor represents (e.g., "tech growth," "defensive dividend," "small-cap value").

5. **Recommendation**: For investor i, compute P[i] @ Q.T to get predicted scores for all stocks. Rank the unobserved stocks by predicted score and recommend the top-N.

## Advantages

1. **Discovers interpretable latent factors from portfolio data.** The learned factors often align with meaningful financial concepts -- one factor might capture growth vs. value orientation, another risk appetite, another sector preference. These emergent factors provide insights beyond traditional categorization.

2. **Handles extreme sparsity effectively.** With investors holding 20-80 stocks out of 3,000, the matrix is >97% sparse. MF gracefully fills in missing entries by leveraging the low-rank structure, making predictions even for investor-stock pairs with no direct observation.

3. **Computational efficiency through dimensionality reduction.** Instead of storing and computing with a 5000 x 3000 matrix, MF works with two much smaller matrices (5000 x 10 and 3000 x 10). This enables fast prediction, similarity computation, and clustering in the latent space.

4. **Bias terms capture systematic effects.** Investor biases capture individual tendencies (concentrated vs. diversified), while stock biases capture popularity effects (widely-held blue chips vs. niche small-caps). Separating these from the latent interaction allows cleaner factor discovery.

5. **Flexible framework supporting various extensions.** MF can incorporate temporal dynamics (time-varying factors), side information (stock fundamentals, investor demographics), implicit feedback (trading activity without explicit ratings), and constraints (non-negativity, orthogonality).

6. **Simultaneous investor profiling and stock characterization.** The dual factor matrices provide a unified framework for both investor clustering (similar latent profiles) and stock similarity (similar factor loadings), enabling rich analytics from a single decomposition.

7. **Robust to noise through regularization.** Financial data is inherently noisy -- portfolio weights fluctuate due to rebalancing, reporting timing, and market movements. Regularized MF filters out this noise by constraining the model to a low-rank representation.

## Disadvantages

1. **Latent factors are not guaranteed to be interpretable.** While factors sometimes align with financial concepts, they can also be abstract mixtures that defy interpretation. This limits the usefulness of MF for generating actionable investment insights.

2. **Cold start problem for new stocks and investors.** A newly IPO'd stock has no observed interactions, so its factor vector cannot be estimated. Similarly, a new investor requires a minimum holding history before meaningful recommendations can be generated.

3. **Static model does not capture temporal dynamics.** Standard MF produces a single decomposition for the entire observation period, unable to model how investor preferences and stock characteristics evolve over time as market regimes change.

4. **Sensitive to the choice of latent dimensionality k.** Too few factors underfit (missing important patterns), too many overfit (capturing noise). The optimal k depends on the true complexity of the investor-stock interaction space, which is unknown.

5. **Non-convex optimization with local minima.** SGD and ALS for MF are non-convex, meaning different initializations can lead to different solutions. While regularization and multiple restarts help, there is no guarantee of finding the global optimum.

6. **Missing-not-at-random assumption violation.** MF assumes missing entries are missing at random, but in reality, investors actively choose not to hold certain stocks (informative missingness). A zero in the matrix could mean "not interested" or "haven't evaluated yet" -- conflating these biases the model.

7. **Popularity bias amplification.** Widely-held stocks (like S&P 500 components) contribute disproportionately to the loss function, leading to factor vectors that primarily explain popular stock preferences while underrepresenting small-cap or niche stock patterns.

## When to Use in Stock Market

- Building recommendation systems for brokerage platforms suggesting new stock ideas
- Discovering latent investment themes and factors across a large investor universe
- Grouping stocks by hidden characteristics beyond sector/industry classification
- Profiling institutional investors based on their revealed preferences
- Predicting which stocks an investor will add or increase in their next rebalancing
- Dimensionality reduction of large investor-stock datasets before downstream analysis
- Constructing factor portfolios based on learned latent factors

## When NOT to Use in Stock Market

- When temporal dynamics are critical (use temporal MF or RNNs instead)
- For real-time trading signals requiring sub-second latency
- When explainability in financial terms is required for compliance
- With very few investors (<50) where the matrix is too small for meaningful factorization
- When interactions are binary (held/not held) without magnitude -- use implicit feedback methods
- For options or derivatives where the interaction structure is fundamentally different
- When missing entries are highly informative (investor actively avoids certain stocks)

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Guidance |
|---|---|---|---|
| `n_factors` (k) | Number of latent factors | 5 to 100 | 10-30 for investor portfolios; validate via held-out RMSE |
| `learning_rate` | SGD step size | 0.001 to 0.05 | Start at 0.01; reduce if training loss oscillates |
| `lambda_reg` | Regularization strength | 0.01 to 0.5 | Higher for sparse data; 0.05-0.1 typical |
| `n_epochs` | Number of training iterations | 20 to 200 | Monitor validation loss; early stop around 50-100 |
| `bias` | Include user/item biases | True / False | True for portfolio weight prediction; captures AUM effects |
| `init_scale` | Initial factor magnitude | 0.01 to 0.1 | 0.05 typical; smaller for more factors |
| `method` | SGD vs ALS | - | ALS for implicit feedback; SGD for explicit weights |
| `n_negatives` | Negative samples for implicit | 1 to 10 | 4-5 negative samples per positive for balanced training |

## Stock Market Performance Tips

1. **Use conviction scores as interaction values.** Raw portfolio weights are noisy due to price drift. Compute conviction as `weight * log(1 + holding_quarters)` to emphasize deliberate, persistent positions over passive price-driven weight changes.

2. **Apply temporal weighting to recent interactions.** Weight recent portfolio compositions more heavily than older snapshots. Use exponential decay: `weight_t = base_weight * exp(-lambda * (T - t))` where T is the current period.

3. **Validate using time-based holdout.** Hold out the most recent quarter of portfolio data for testing, not random entries. This simulates the real use case of predicting future holdings from historical patterns.

4. **Experiment with non-negative MF for portfolio data.** Since portfolio weights are inherently non-negative, NMF produces additive factors that are more naturally interpretable as "portfolio components" or "investment themes."

5. **Combine with side information using hybrid models.** Augment the latent factors with stock fundamentals (PE, sector, market cap) and investor attributes (AUM, style, type) to address the cold start problem and improve generalization.

6. **Regularize more aggressively for stocks with few holders.** Small-cap stocks held by few investors have unreliable factor estimates. Increase regularization proportional to the inverse of the number of interactions to prevent overfitting.

7. **Interpret factors by examining top-loading stocks and investors.** For each latent factor, list the top-10 stocks and investors with the highest loadings. This often reveals interpretable themes like "tech growth," "defensive yield," or "emerging market exposure."

## Comparison with Other Algorithms

| Aspect | Matrix Factorization | Collaborative Filtering | Autoencoders | PCA | Topic Models (LDA) |
|---|---|---|---|---|---|
| Handles Sparsity | Excellent | Good | Good | Poor (needs imputation) | Good |
| Scalability | O(k * nnz * epochs) | O(n^2 * d) | O(n * d * hidden) | O(n * d^2) | O(n * d * k) |
| Non-linearity | No | No | Yes | No | No |
| Interpretability | Moderate (factors) | High (neighbors) | Low | High (components) | High (topics) |
| Cold Start | Problem | Problem | Partial (side info) | No (needs full data) | No |
| Bias Modeling | Built-in | External | Implicit | No | No |
| Best For (Finance) | Latent factors | Neighbor-based recs | Complex patterns | Dense data reduction | Thematic discovery |

## Real-World Stock Market Example

```python
import numpy as np

# ================================================================
# Matrix Factorization: Discovering Latent Investment Factors
# Implements SGD-based MF with biases from scratch
# ================================================================

class MatrixFactorization:
    """
    Matrix Factorization with biases using SGD.
    Decomposes investor-stock preference matrix into latent factors.
    """

    def __init__(self, n_factors=10, learning_rate=0.01, reg_lambda=0.05,
                 n_epochs=50, init_scale=0.05):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg_lambda
        self.n_epochs = n_epochs
        self.init_scale = init_scale

        self.P = None  # Investor factors
        self.Q = None  # Stock factors
        self.b_u = None  # Investor biases
        self.b_i = None  # Stock biases
        self.mu = 0.0  # Global mean

    def fit(self, R, val_data=None):
        """
        Fit the model using SGD on observed entries.
        R: sparse interaction matrix (n_investors x n_stocks)
        val_data: list of (user, item, rating) tuples for validation
        """
        n_users, n_items = R.shape

        # Initialize
        self.P = np.random.randn(n_users, self.n_factors) * self.init_scale
        self.Q = np.random.randn(n_items, self.n_factors) * self.init_scale
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        # Global mean of observed entries
        observed = R[R > 0]
        self.mu = np.mean(observed) if len(observed) > 0 else 0.0

        # Get observed indices
        obs_users, obs_items = np.where(R > 0)
        obs_ratings = R[obs_users, obs_items]
        n_obs = len(obs_ratings)

        train_losses = []
        val_losses = []

        for epoch in range(self.n_epochs):
            # Shuffle observed entries
            perm = np.random.permutation(n_obs)
            total_loss = 0.0

            for idx in perm:
                u = obs_users[idx]
                i = obs_items[idx]
                r = obs_ratings[idx]

                # Prediction
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
                error = r - pred

                # SGD updates
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])

                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u_old - self.reg * self.Q[i])

                total_loss += error ** 2

            train_rmse = np.sqrt(total_loss / n_obs)
            train_losses.append(train_rmse)

            # Validation
            if val_data is not None and epoch % 5 == 0:
                val_rmse = self.evaluate(val_data)
                val_losses.append(val_rmse)
                print(f"Epoch {epoch+1:3d}/{self.n_epochs}: "
                      f"Train RMSE={train_rmse:.6f}, Val RMSE={val_rmse:.6f}")
            elif epoch % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.n_epochs}: Train RMSE={train_rmse:.6f}")

            # Learning rate decay
            self.lr *= 0.99

        return train_losses, val_losses

    def predict(self, user_id, item_id):
        """Predict preference score."""
        return (self.mu + self.b_u[user_id] + self.b_i[item_id] +
                self.P[user_id] @ self.Q[item_id])

    def predict_all(self, user_id):
        """Predict scores for all items for a given user."""
        return self.mu + self.b_u[user_id] + self.b_i + self.P[user_id] @ self.Q.T

    def evaluate(self, test_data):
        """Compute RMSE on test data."""
        errors = []
        for u, i, r in test_data:
            pred = self.predict(u, i)
            errors.append((r - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def recommend(self, user_id, R_observed, n_recs=10):
        """Recommend top-N stocks not currently held."""
        scores = self.predict_all(user_id)
        held_mask = R_observed[user_id] > 0
        scores[held_mask] = -np.inf
        top_idx = np.argsort(scores)[-n_recs:][::-1]
        return [(idx, scores[idx]) for idx in top_idx]

    def get_factor_loadings(self, factor_idx, top_n=10, item_names=None):
        """Get top items loading on a specific latent factor."""
        loadings = self.Q[:, factor_idx]
        top_positive = np.argsort(loadings)[-top_n:][::-1]
        top_negative = np.argsort(loadings)[:top_n]

        results = {
            'positive': [(i, loadings[i], item_names[i] if item_names else i)
                         for i in top_positive],
            'negative': [(i, loadings[i], item_names[i] if item_names else i)
                         for i in top_negative]
        }
        return results

    def investor_similarity(self, user_id, top_n=10):
        """Find similar investors in latent space."""
        diffs = self.P - self.P[user_id]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        distances[user_id] = np.inf
        closest = np.argsort(distances)[:top_n]
        return [(idx, distances[idx]) for idx in closest]

    def stock_similarity(self, item_id, top_n=10):
        """Find similar stocks in latent space."""
        diffs = self.Q - self.Q[item_id]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        distances[item_id] = np.inf
        closest = np.argsort(distances)[:top_n]
        return [(idx, distances[idx]) for idx in closest]


# ================================================================
# Generate Stock Market Data
# ================================================================

np.random.seed(42)

n_investors = 3000
n_stocks = 1000
n_true_factors = 8

# Stock metadata
sectors = ['Tech', 'Healthcare', 'Finance', 'Energy', 'Consumer',
           'Industrial', 'Utilities', 'Materials', 'RealEstate', 'Telecom']
stock_names = [f"STK_{i:04d}" for i in range(n_stocks)]
stock_sectors = [sectors[i % len(sectors)] for i in range(n_stocks)]
stock_market_caps = np.random.lognormal(mean=2.5, sigma=1.5, size=n_stocks)

# Investor metadata
investor_types = ['HedgeFund', 'MutualFund', 'Pension', 'Endowment', 'Family Office']
investor_labels = [investor_types[i % len(investor_types)] for i in range(n_investors)]

# True latent factors
true_P = np.abs(np.random.randn(n_investors, n_true_factors) * 0.3)
true_Q = np.abs(np.random.randn(n_stocks, n_true_factors) * 0.3)

# Build sparse interaction matrix
R = np.zeros((n_investors, n_stocks))
true_scores = true_P @ true_Q.T

for i in range(n_investors):
    scores = true_scores[i]
    probs = scores / (np.sum(scores) + 1e-8)

    n_holdings = np.random.randint(20, 80)
    holdings = np.random.choice(n_stocks, n_holdings, replace=False, p=probs)

    weights = np.random.dirichlet(np.ones(n_holdings) * 2)
    R[i, holdings] = weights

print(f"Interaction matrix: {R.shape}")
print(f"Sparsity: {1 - np.count_nonzero(R) / R.size:.2%}")
print(f"Non-zero entries: {np.count_nonzero(R):,}")

# ================================================================
# Train-Test Split
# ================================================================

# Hold out 20% of interactions for testing
obs_users, obs_items = np.where(R > 0)
obs_ratings = R[obs_users, obs_items]
n_obs = len(obs_ratings)

perm = np.random.permutation(n_obs)
n_train = int(0.8 * n_obs)

train_idx = perm[:n_train]
test_idx = perm[n_train:]

R_train = np.zeros_like(R)
R_train[obs_users[train_idx], obs_items[train_idx]] = obs_ratings[train_idx]

test_data = list(zip(obs_users[test_idx], obs_items[test_idx], obs_ratings[test_idx]))

print(f"Training entries: {n_train:,}")
print(f"Test entries: {len(test_data):,}")

# ================================================================
# Train Matrix Factorization Model
# ================================================================

print("\n=== Training Matrix Factorization ===")
mf = MatrixFactorization(
    n_factors=15,
    learning_rate=0.01,
    reg_lambda=0.05,
    n_epochs=30,
    init_scale=0.05
)

train_losses, val_losses = mf.fit(R_train, val_data=test_data[:1000])

# Final evaluation
test_rmse = mf.evaluate(test_data)
print(f"\nFinal Test RMSE: {test_rmse:.6f}")

# ================================================================
# Latent Factor Analysis
# ================================================================

print("\n=== Latent Factor Analysis ===")
for f in range(min(5, mf.n_factors)):
    print(f"\n--- Factor {f+1} ---")
    loadings = mf.get_factor_loadings(f, top_n=5, item_names=stock_names)

    print("  High loading stocks (positive):")
    for idx, loading, name in loadings['positive']:
        print(f"    {name} ({stock_sectors[idx]}): {loading:.4f}")

    print("  High loading stocks (negative):")
    for idx, loading, name in loadings['negative']:
        print(f"    {name} ({stock_sectors[idx]}): {loading:.4f}")

# Factor variance explained
factor_norms = np.sum(mf.Q ** 2, axis=0)
total_norm = np.sum(factor_norms)
print(f"\n=== Factor Importance (by Q-norm) ===")
sorted_factors = np.argsort(factor_norms)[::-1]
for rank, f in enumerate(sorted_factors):
    pct = factor_norms[f] / total_norm * 100
    bar = "#" * int(pct * 2)
    print(f"  Factor {f+1}: {pct:5.1f}% {bar}")

# ================================================================
# Recommendations
# ================================================================

print("\n=== Recommendations for Sample Investors ===")
for inv_id in [0, 100, 500]:
    print(f"\nInvestor {inv_id} ({investor_labels[inv_id]}):")
    print(f"  Current holdings: {np.sum(R[inv_id] > 0)} stocks")

    # Top current holdings
    top_held = np.argsort(R[inv_id])[-5:][::-1]
    print(f"  Top holdings: {[f'{stock_names[j]}({stock_sectors[j]})' for j in top_held if R[inv_id, j] > 0]}")

    # Recommendations
    recs = mf.recommend(inv_id, R, n_recs=10)
    print(f"  Recommendations:")
    for rank, (stock_idx, score) in enumerate(recs, 1):
        print(f"    {rank}. {stock_names[stock_idx]} ({stock_sectors[stock_idx]}): "
              f"score={score:.4f}")

# ================================================================
# Investor Clustering via Latent Factors
# ================================================================

print("\n=== Investor Clustering ===")

# K-means in latent space
def simple_kmeans(X, k=5, max_iter=50):
    n = X.shape[0]
    centroids = X[np.random.choice(n, k, replace=False)]
    for _ in range(max_iter):
        dists = np.sqrt(np.sum((X[:, np.newaxis] - centroids[np.newaxis]) ** 2, axis=2))
        labels = np.argmin(dists, axis=1)
        for c in range(k):
            mask = labels == c
            if np.sum(mask) > 0:
                centroids[c] = np.mean(X[mask], axis=0)
    return labels, centroids

cluster_labels, centroids = simple_kmeans(mf.P, k=6)

for c in range(6):
    mask = cluster_labels == c
    n_members = np.sum(mask)
    type_counts = {}
    for i in np.where(mask)[0]:
        t = investor_labels[i]
        type_counts[t] = type_counts.get(t, 0) + 1

    top_type = max(type_counts, key=type_counts.get)
    avg_holdings = np.mean(np.sum(R[mask] > 0, axis=1))

    print(f"  Cluster {c}: {n_members} investors, "
          f"dominant type={top_type}, avg holdings={avg_holdings:.0f}")

# ================================================================
# Stock Similarity Analysis
# ================================================================

print("\n=== Stock Similarity in Latent Space ===")
target_stock = 0
print(f"Stocks similar to {stock_names[target_stock]} ({stock_sectors[target_stock]}):")
similar = mf.stock_similarity(target_stock, top_n=10)
for stock_idx, dist in similar:
    same_sector = "SAME" if stock_sectors[stock_idx] == stock_sectors[target_stock] else ""
    print(f"  {stock_names[stock_idx]} ({stock_sectors[stock_idx]}): "
          f"distance={dist:.4f} {same_sector}")

# ================================================================
# Compare Different Factor Counts
# ================================================================

print("\n=== Factor Count Comparison ===")
print(f"{'Factors':>10} {'Test RMSE':>12} {'Train Time':>12}")
print("-" * 36)

for k in [5, 10, 15, 20, 30]:
    mf_k = MatrixFactorization(n_factors=k, learning_rate=0.01,
                                reg_lambda=0.05, n_epochs=20, init_scale=0.05)
    import time
    start = time.time()
    mf_k.fit(R_train)
    elapsed = time.time() - start
    rmse_k = mf_k.evaluate(test_data[:1000])
    print(f"{k:>10} {rmse_k:>12.6f} {elapsed:>10.1f}s")

# ================================================================
# Investor Profile in Latent Space
# ================================================================

print("\n=== Investor Latent Profiles ===")
inv_id = 0
profile = mf.P[inv_id]
print(f"Investor {inv_id} factor profile:")
for f in range(mf.n_factors):
    bar_len = int(abs(profile[f]) * 50)
    direction = "+" if profile[f] > 0 else "-"
    bar = "#" * bar_len
    print(f"  Factor {f+1:2d}: {profile[f]:+.4f} {direction}{bar}")

# Find investor's nearest neighbor
similar_inv = mf.investor_similarity(inv_id, top_n=5)
print(f"\nMost similar investors:")
for sim_id, dist in similar_inv:
    overlap = np.sum((R[inv_id] > 0) & (R[sim_id] > 0))
    print(f"  Investor {sim_id} ({investor_labels[sim_id]}): "
          f"distance={dist:.4f}, portfolio overlap={overlap} stocks")
```

## Key Takeaways

1. **Matrix factorization discovers hidden investment themes** by decomposing the investor-stock preference matrix into low-rank factors that capture the dominant patterns in how institutional investors allocate across stocks.

2. **The latent factors provide a dual representation** -- investor profiles and stock characteristics in the same space -- enabling both personalized recommendations and stock similarity analysis from a single model.

3. **Bias terms are essential for financial data** where investor-level effects (portfolio concentration) and stock-level effects (popularity) are strong and should be modeled separately from the latent interaction.

4. **The number of latent factors controls the trade-off** between model capacity and generalization. For stock market applications, 10-30 factors typically capture the major investment themes without overfitting.

5. **Non-negative matrix factorization produces more interpretable factors** for portfolio data where all interactions are non-negative, yielding additive components that can be read as investment themes.

6. **Regular retraining is necessary** as investor preferences evolve, especially around major market events, earnings seasons, and regime transitions that shift the latent factor structure.
