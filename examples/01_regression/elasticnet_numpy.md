# ElasticNet Regression - Complete Guide with Stock Market Applications

## Overview

ElasticNet Regression is a regularized linear regression technique that combines the penalties of both Lasso (L1) and Ridge (L2) regression into a single model. By blending these two regularization strategies, ElasticNet inherits the feature selection capability of Lasso while maintaining Ridge's ability to handle correlated features gracefully. The mixing parameter (alpha or l1_ratio) controls the balance between the two penalties, giving practitioners fine-grained control over model complexity.

In stock market applications, ElasticNet is particularly valuable for predicting portfolio returns when the feature space includes dozens or hundreds of correlated technical indicators, fundamental ratios, and macroeconomic variables. Financial data is notorious for multicollinearity -- for example, moving averages of different windows, overlapping momentum indicators, and sector-correlated fundamentals all tend to move together. Pure Lasso would arbitrarily drop one of a group of correlated features, while pure Ridge would keep all of them with shrunken coefficients. ElasticNet strikes a balance, grouping correlated features and selecting among groups.

The algorithm is solved via coordinate descent or proximal gradient methods, making it efficient even for high-dimensional financial datasets. Its regularization prevents overfitting to noise in historical market data, a critical concern given the low signal-to-noise ratio prevalent in financial time series. The resulting sparse, interpretable models are well-suited for risk management and factor-based investing strategies.

## How It Works - The Math Behind It

### The ElasticNet Objective Function

ElasticNet minimizes the following objective:

```
J(w) = (1 / 2m) * sum_{i=1}^{m} (y_i - X_i @ w)^2
       + lambda * [ (1 - l1_ratio) / 2 * ||w||_2^2 + l1_ratio * ||w||_1 ]
```

where:
- `m` is the number of training samples
- `X_i` is the feature vector for sample `i`
- `w` is the weight vector
- `lambda` (or `alpha`) is the overall regularization strength
- `l1_ratio` (or `rho`) controls the mix between L1 and L2 penalties
- `||w||_1 = sum |w_j|` is the L1 norm (Lasso penalty)
- `||w||_2^2 = sum w_j^2` is the squared L2 norm (Ridge penalty)

### Special Cases

When `l1_ratio = 1`, ElasticNet reduces to Lasso:
```
J(w) = (1 / 2m) * ||y - Xw||^2 + lambda * ||w||_1
```

When `l1_ratio = 0`, ElasticNet reduces to Ridge:
```
J(w) = (1 / 2m) * ||y - Xw||^2 + (lambda / 2) * ||w||_2^2
```

### Coordinate Descent Solution

ElasticNet is typically solved using coordinate descent. For each weight `w_j`, holding all others fixed:

```
w_j = S(rho_j, lambda * l1_ratio) / (1 + lambda * (1 - l1_ratio))
```

where `S` is the soft-thresholding operator:

```
S(z, gamma) = sign(z) * max(|z| - gamma, 0)
```

and `rho_j` is the partial residual for feature `j`:

```
rho_j = (1/m) * sum_{i=1}^{m} x_{ij} * (y_i - y_hat_i^{(-j)})
```

Here `y_hat_i^{(-j)}` denotes the prediction without the contribution of feature `j`.

### Gradient Computation

The gradient of the smooth part (MSE + L2) is:

```
grad_j = -(1/m) * X_j^T @ (y - X @ w) + lambda * (1 - l1_ratio) * w_j
```

The L1 term contributes a subgradient, handled by the soft-thresholding operator.

### Regularization Path

The regularization path traces solutions for a sequence of `lambda` values from `lambda_max` (where all coefficients are zero) down to a small fraction:

```
lambda_max = max_j | (1/m) * X_j^T @ y | / l1_ratio
```

This allows efficient warm-starting of the coordinate descent across the path.

## Stock Market Use Case: Predicting Portfolio Returns with Mixed Regularization

### The Problem

A quantitative hedge fund wants to predict monthly returns for a portfolio of 50 large-cap stocks using a rich feature set of 200+ financial indicators. Many features are highly correlated (e.g., 10-day, 20-day, and 50-day moving averages), and the fund needs a model that:
- Selects the most informative features automatically
- Handles multicollinearity without unstable coefficient estimates
- Remains interpretable for regulatory reporting and risk oversight
- Generalizes to unseen market conditions without overfitting

### Stock Market Features (Input Data)

| Feature Category | Feature Name | Description | Example Value |
|---|---|---|---|
| Price Momentum | RSI_14 | 14-day Relative Strength Index | 62.5 |
| Price Momentum | MACD_signal | MACD minus signal line crossover | 1.23 |
| Price Momentum | momentum_20d | 20-day price momentum (%) | 3.8 |
| Moving Averages | SMA_10 | 10-day simple moving average | 152.30 |
| Moving Averages | SMA_50 | 50-day simple moving average | 148.75 |
| Moving Averages | EMA_20 | 20-day exponential moving average | 150.90 |
| Volatility | ATR_14 | 14-day Average True Range | 4.52 |
| Volatility | bollinger_width | Bollinger Band width | 0.085 |
| Volatility | hist_vol_30d | 30-day historical volatility | 0.22 |
| Volume | OBV | On-Balance Volume (normalized) | 1.15 |
| Volume | volume_ratio | Current to avg volume ratio | 1.32 |
| Volume | VWAP_deviation | Price deviation from VWAP | -0.45 |
| Fundamental | PE_ratio | Price-to-Earnings ratio | 18.5 |
| Fundamental | PB_ratio | Price-to-Book ratio | 3.2 |
| Fundamental | dividend_yield | Annual dividend yield (%) | 2.1 |
| Fundamental | debt_to_equity | Debt-to-Equity ratio | 0.85 |
| Macro | treasury_10y | 10-year Treasury yield | 4.25 |
| Macro | VIX | CBOE Volatility Index | 18.3 |
| Macro | dollar_index | US Dollar Index | 104.5 |
| Sector | sector_momentum | Sector-relative momentum | 0.015 |

### Example Data Structure

```python
import numpy as np

# Simulated monthly stock data for ElasticNet training
np.random.seed(42)
n_samples = 500     # 500 months of historical data
n_features = 200    # 200 financial indicators

# Generate correlated feature blocks (mimicking real financial data)
def generate_correlated_block(n, size, base_correlation=0.8):
    """Generate a block of correlated features like moving averages."""
    base = np.random.randn(n, 1)
    noise = np.random.randn(n, size) * np.sqrt(1 - base_correlation)
    return base * np.sqrt(base_correlation) + noise

# Build feature matrix with correlated groups
momentum_features = generate_correlated_block(n_samples, 30)    # momentum indicators
ma_features = generate_correlated_block(n_samples, 25)          # moving averages
volatility_features = generate_correlated_block(n_samples, 20)  # volatility measures
volume_features = generate_correlated_block(n_samples, 25)      # volume indicators
fundamental_features = generate_correlated_block(n_samples, 40) # fundamental ratios
macro_features = generate_correlated_block(n_samples, 30)       # macro variables
sector_features = generate_correlated_block(n_samples, 30)      # sector indicators

X = np.hstack([
    momentum_features, ma_features, volatility_features,
    volume_features, fundamental_features, macro_features,
    sector_features
])

# True signal: only ~15 features actually drive returns
true_weights = np.zeros(n_features)
true_weights[0] = 0.05    # RSI
true_weights[5] = 0.03    # MACD
true_weights[30] = -0.02  # SMA spread
true_weights[55] = 0.04   # ATR
true_weights[80] = -0.03  # PE ratio
true_weights[100] = 0.02  # Treasury yield
true_weights[120] = 0.015 # VIX sensitivity
# ... a few more non-zero weights

# Monthly portfolio returns (target)
y = X @ true_weights + np.random.randn(n_samples) * 0.02  # noisy returns

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"True non-zero weights: {np.sum(true_weights != 0)}")
print(f"Feature correlation range: {np.min(np.corrcoef(X.T)):.2f} to {np.max(np.corrcoef(X.T) - np.eye(n_features)):.2f}")
```

### The Model in Action

ElasticNet processes the 200-feature stock dataset by:

1. **Standardizing features**: Each financial indicator is z-scored so that regularization treats all features equally regardless of their natural scale (e.g., PE ratio ~18 vs. VIX ~18 vs. volume ratio ~1.3).

2. **Coordinate descent iteration**: The algorithm cycles through each of the 200 features, computing the partial residual and applying the soft-thresholding operator. Features with weak signal (most of the 200) get driven to zero by the L1 penalty, while correlated groups of informative features share the coefficient mass thanks to the L2 penalty.

3. **Convergence**: After 50-200 iterations, the algorithm converges. The resulting model has ~15-25 non-zero coefficients, corresponding to the most predictive indicators. Correlated features within each informative group receive similar, non-zero coefficients rather than one being arbitrarily selected.

4. **Prediction**: New monthly data is standardized using training statistics, then multiplied by the sparse weight vector to produce a predicted portfolio return.

## Advantages

1. **Automatic feature selection in high-dimensional financial data.** ElasticNet's L1 component drives irrelevant indicator coefficients to exactly zero, effectively performing variable selection. This is critical when working with 200+ technical indicators where most add noise rather than signal, producing a parsimonious model that focuses on truly predictive features.

2. **Robust handling of multicollinearity among financial indicators.** Stock market features are notoriously correlated -- overlapping moving averages, related momentum signals, and sector-linked fundamentals all co-move. The L2 component ensures that correlated features share coefficient mass rather than exhibiting the unstable, high-variance estimates that ordinary least squares would produce.

3. **Grouped feature selection preserves domain knowledge.** Unlike pure Lasso which randomly drops one feature from a correlated group, ElasticNet tends to select or exclude entire groups together. In a stock market context, if 10-day and 20-day moving averages are both relevant, ElasticNet is more likely to keep both rather than arbitrarily dropping one, aligning with the trader's intuition that multiple timeframe signals matter.

4. **Built-in overfitting protection for noisy financial data.** Financial returns have an extremely low signal-to-noise ratio, often below 5%. ElasticNet's dual regularization prevents the model from fitting to spurious patterns in historical data, leading to better out-of-sample performance compared to unregularized regression on market data.

5. **Efficient regularization path computation.** The coordinate descent algorithm with warm-starting allows efficient computation of the entire regularization path, enabling practitioners to quickly evaluate model complexity trade-offs. This is valuable in production trading systems where models need retraining daily or weekly with updated market data.

6. **Interpretable coefficients for regulatory compliance.** In regulated financial environments, model interpretability is often required. ElasticNet's linear coefficients directly quantify the marginal impact of each indicator on predicted returns, making it straightforward to explain model decisions to risk committees, auditors, and regulators.

7. **Tunable complexity via mixing parameter.** The `l1_ratio` parameter provides a continuous knob between pure feature selection (Lasso) and pure coefficient shrinkage (Ridge). Portfolio managers can tune this to match their prior beliefs about feature sparsity in different market regimes or asset classes.

## Disadvantages

1. **Assumes linear relationships between features and returns.** Stock market dynamics are fundamentally non-linear -- regime changes, volatility clustering, and feedback loops all violate the linearity assumption. ElasticNet cannot capture interaction effects like "RSI is only predictive when VIX is elevated" without explicit feature engineering.

2. **Two hyperparameters require careful tuning.** Both `lambda` (regularization strength) and `l1_ratio` (mixing parameter) must be tuned, typically via cross-validation. In financial applications where data is serially correlated, standard k-fold cross-validation is inappropriate, requiring specialized time-series CV schemes that reduce effective sample size.

3. **Cannot model time-varying relationships.** Financial markets exhibit non-stationarity -- the predictive power of indicators changes over time as market regimes shift. ElasticNet fits a single static set of coefficients, unable to adapt to changing dynamics without periodic retraining and window selection.

4. **Sensitive to feature scaling in financial data.** All features must be standardized before applying ElasticNet, and the choice of scaling method (z-score, min-max, robust) can significantly affect results. Financial features with heavy tails (e.g., volume spikes, return outliers) can distort standard z-score normalization.

5. **Limited capacity for capturing tail risk.** Stock market returns have fat tails and are not normally distributed. ElasticNet's squared loss function treats all prediction errors equally, failing to properly penalize large errors that correspond to market crashes or extreme events -- precisely the scenarios where accurate prediction matters most.

6. **Ignores sequential structure of market data.** ElasticNet treats each observation independently, ignoring the temporal ordering of financial data. It cannot model autocorrelation, momentum persistence, or mean-reversion patterns without engineered lag features, missing important information embedded in the time series structure.

## When to Use in Stock Market

- Building factor models with many potentially correlated alpha signals
- Monthly or quarterly return prediction where feature count exceeds sample count
- Initial feature selection before feeding into more complex models (ensemble, neural nets)
- Multi-asset portfolio return forecasting with shared macro factors
- Regulatory-compliant models requiring transparent, interpretable coefficients
- Cross-sectional stock ranking models with hundreds of fundamental and technical features
- Risk factor attribution where you need to identify which factors drive portfolio variance

## When NOT to Use in Stock Market

- High-frequency trading where non-linear microstructure effects dominate
- Market regime detection requiring discrete state identification
- Extreme event prediction where tail behavior is critical (use quantile regression instead)
- When you have very few features (<10) and no multicollinearity (use simpler OLS or Ridge)
- Intraday price prediction requiring sequence modeling (use RNNs or Transformers)
- When feature interactions are known to be important (use tree-based methods or polynomial features)
- Options pricing or derivatives modeling requiring non-linear payoff structures

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Guidance |
|---|---|---|---|
| `alpha` / `lambda` | Overall regularization strength | 1e-4 to 10 | Higher values for noisier data; use CV with purged time-series splits |
| `l1_ratio` | Balance between L1 and L2 penalty | 0.1 to 0.9 | Use 0.5-0.7 when you suspect many irrelevant features with some correlated groups |
| `max_iter` | Maximum coordinate descent iterations | 1000 to 10000 | Increase if convergence warnings appear; financial data may need more iterations |
| `tol` | Convergence tolerance | 1e-4 to 1e-6 | Tighter tolerance for production models; looser for exploration |
| `fit_intercept` | Whether to fit intercept term | True / False | True when predicting raw returns; False if features are centered and returns demeaned |
| `warm_start` | Reuse previous solution | True / False | True for sequential retraining in production; speeds up daily model updates |
| `selection` | Coordinate descent variable order | 'cyclic' / 'random' | 'random' often converges faster with many correlated financial features |

## Stock Market Performance Tips

1. **Use purged time-series cross-validation.** Standard k-fold CV leaks future information through overlapping windows. Use purged walk-forward validation with an embargo period (e.g., 5 trading days) between train and test sets to get realistic performance estimates.

2. **Standardize features using rolling windows.** Instead of computing mean and standard deviation on the entire training set, use rolling z-scores (e.g., 252-day rolling) to account for changing market conditions. This prevents look-ahead bias and handles non-stationarity.

3. **Engineer interaction features for key indicators.** Since ElasticNet cannot model interactions natively, create cross-features like `RSI * VIX` or `momentum * volatility` to capture known non-linear effects before fitting the model.

4. **Apply winsorization to handle outliers.** Clip extreme values at the 1st and 99th percentiles before standardization. Financial data contains outliers (flash crashes, earnings surprises) that can disproportionately influence the least-squares objective.

5. **Retrain on a schedule aligned with your prediction horizon.** For monthly return prediction, retrain monthly with an expanding or rolling window. Stale models degrade as market dynamics evolve.

6. **Monitor coefficient stability over time.** Track how ElasticNet coefficients change across retraining periods. Rapidly changing coefficients suggest instability and potential overfitting to recent noise.

7. **Combine with residual analysis.** Examine prediction residuals for patterns (autocorrelation, heteroscedasticity) that indicate model misspecification, and consider ensemble approaches to capture what ElasticNet misses.

## Comparison with Other Algorithms

| Aspect | ElasticNet | Lasso | Ridge | Random Forest | XGBoost |
|---|---|---|---|---|---|
| Feature Selection | Yes (grouped) | Yes (arbitrary) | No | Implicit | Implicit |
| Multicollinearity Handling | Excellent | Poor | Good | Good | Good |
| Non-linear Relationships | No | No | No | Yes | Yes |
| Interpretability | High | High | High | Medium | Low |
| Training Speed | Fast | Fast | Fast | Medium | Medium |
| Overfitting Risk (small n) | Low | Low | Low | Medium | High |
| Handling Outliers | Poor | Poor | Poor | Good | Good |
| Hyperparameter Count | 2 | 1 | 1 | 3-5 | 6-10 |
| Best For (Finance) | Factor models | Sparse signals | Correlated factors | Non-linear signals | Complex patterns |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# ElasticNet from Scratch: Predicting Monthly Portfolio Returns
# ============================================================

class ElasticNetRegression:
    """
    ElasticNet Regression using coordinate descent.
    Combines L1 (Lasso) and L2 (Ridge) regularization.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha          # Overall regularization strength
        self.l1_ratio = l1_ratio    # Mix: 1=Lasso, 0=Ridge
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.intercept = 0.0
        self.mean_X = None
        self.std_X = None
        self.mean_y = None

    def _soft_threshold(self, z, gamma):
        """Soft-thresholding operator for L1 proximal step."""
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

    def fit(self, X, y):
        """Fit ElasticNet using coordinate descent."""
        m, n = X.shape

        # Standardize features
        self.mean_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0) + 1e-8
        X_std = (X - self.mean_X) / self.std_X

        # Center target
        self.mean_y = np.mean(y)
        y_centered = y - self.mean_y

        # Initialize weights
        w = np.zeros(n)

        # Precompute X^T X diagonal and X^T y
        X_sq = np.sum(X_std ** 2, axis=0) / m

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)

        for iteration in range(self.max_iter):
            w_old = w.copy()

            for j in range(n):
                # Compute partial residual
                residual = y_centered - X_std @ w + X_std[:, j] * w[j]
                rho_j = np.dot(X_std[:, j], residual) / m

                # Coordinate update with soft thresholding
                w[j] = self._soft_threshold(rho_j, l1_penalty) / (X_sq[j] + l2_penalty)

            # Check convergence
            if np.max(np.abs(w - w_old)) < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

        # Transform weights back to original scale
        self.weights = w / self.std_X
        self.intercept = self.mean_y - np.dot(self.mean_X, self.weights)

        return self

    def predict(self, X):
        """Predict returns for new data."""
        return X @ self.weights + self.intercept

    def get_selected_features(self, feature_names=None, threshold=1e-6):
        """Return features with non-zero coefficients."""
        mask = np.abs(self.weights) > threshold
        indices = np.where(mask)[0]
        if feature_names is not None:
            return [(feature_names[i], self.weights[i]) for i in indices]
        return [(i, self.weights[i]) for i in indices]


# ============================================================
# Generate Realistic Stock Market Data
# ============================================================

np.random.seed(42)
n_months = 360        # 30 years of monthly data
n_stocks = 50         # Portfolio of 50 stocks
n_features = 120      # 120 financial indicators

feature_names = (
    [f"momentum_{i}d" for i in [5, 10, 20, 60, 120, 252]] +
    [f"SMA_{i}" for i in [5, 10, 20, 50, 100, 200]] +
    [f"EMA_{i}" for i in [5, 10, 20, 50, 100, 200]] +
    [f"volatility_{i}d" for i in [5, 10, 20, 60, 120, 252]] +
    ["RSI_14", "RSI_28", "MACD", "MACD_signal", "MACD_hist"] +
    ["ATR_14", "ATR_20", "bollinger_upper", "bollinger_lower", "bollinger_width"] +
    ["OBV", "volume_ratio", "VWAP_dev", "accumulation_dist", "money_flow"] +
    [f"PE_ratio", "PB_ratio", "PS_ratio", "EV_EBITDA", "dividend_yield"] +
    ["debt_equity", "current_ratio", "ROE", "ROA", "profit_margin"] +
    ["revenue_growth", "earnings_growth", "book_value_growth"] +
    ["beta", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"] +
    ["treasury_2y", "treasury_5y", "treasury_10y", "yield_spread"] +
    ["VIX", "dollar_index", "oil_price", "gold_price", "copper_price"] +
    ["GDP_growth", "CPI", "unemployment", "PMI", "consumer_sentiment"] +
    [f"sector_momentum_{i}" for i in range(11)] +
    [f"market_cap_decile_{i}" for i in range(10)] +
    [f"style_factor_{i}" for i in range(5)]
)
feature_names = feature_names[:n_features]

# Create correlated feature blocks
def make_block(n, size, corr=0.7):
    base = np.random.randn(n, 1)
    noise = np.random.randn(n, size) * np.sqrt(1 - corr)
    return base * np.sqrt(corr) + noise

blocks = [
    make_block(n_months, 18, 0.85),   # momentum + MAs (highly correlated)
    make_block(n_months, 18, 0.80),   # more MAs and EMAs
    make_block(n_months, 11, 0.60),   # volatility indicators
    make_block(n_months, 10, 0.65),   # volume indicators
    make_block(n_months, 13, 0.50),   # fundamental ratios
    make_block(n_months, 13, 0.45),   # risk metrics and macro
    make_block(n_months, 15, 0.55),   # economic indicators
    make_block(n_months, 22, 0.40),   # sector and style factors
]

X = np.hstack(blocks)[:, :n_features]

# True signal: sparse, only ~12 features matter
true_w = np.zeros(n_features)
true_w[0] = 0.004    # momentum_5d
true_w[4] = 0.003    # momentum_120d
true_w[12] = -0.002  # EMA_5
true_w[24] = 0.005   # RSI_14
true_w[29] = 0.003   # ATR_14
true_w[34] = -0.004  # OBV
true_w[39] = -0.003  # PE_ratio
true_w[45] = 0.002   # ROE
true_w[50] = 0.004   # beta
true_w[55] = -0.003  # treasury_10y
true_w[59] = 0.006   # VIX
true_w[70] = 0.002   # GDP_growth

# Monthly portfolio returns
noise = np.random.randn(n_months) * 0.015  # realistic noise level
y = X @ true_w + noise

# ============================================================
# Train-Test Split (time-series aware)
# ============================================================

train_size = int(0.7 * n_months)
embargo = 6  # 6-month embargo period

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size + embargo:]
y_test = y[train_size + embargo:]

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"True non-zero features: {np.sum(true_w != 0)}")

# ============================================================
# Cross-Validation for Hyperparameter Tuning
# ============================================================

def purged_walk_forward_cv(X, y, n_splits=5, embargo=3):
    """Generate train/test indices with purging and embargo."""
    n = len(X)
    fold_size = n // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)
        if test_end > test_start:
            splits.append((
                np.arange(0, train_end),
                np.arange(test_start, test_end)
            ))
    return splits

best_score = -np.inf
best_params = {}

alphas = [0.001, 0.01, 0.05, 0.1, 0.5]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

print("\nHyperparameter search:")
for alpha in alphas:
    for l1_ratio in l1_ratios:
        cv_scores = []
        splits = purged_walk_forward_cv(X_train, y_train, n_splits=3, embargo=3)

        for train_idx, val_idx in splits:
            model = ElasticNetRegression(alpha=alpha, l1_ratio=l1_ratio, max_iter=500, tol=1e-4)
            model.fit(X_train[train_idx], y_train[train_idx])
            y_pred = model.predict(X_train[val_idx])

            # R-squared
            ss_res = np.sum((y_train[val_idx] - y_pred) ** 2)
            ss_tot = np.sum((y_train[val_idx] - np.mean(y_train[val_idx])) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            cv_scores.append(r2)

        mean_score = np.mean(cv_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}

print(f"Best params: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
print(f"Best CV R-squared: {best_score:.4f}")

# ============================================================
# Train Final Model
# ============================================================

final_model = ElasticNetRegression(
    alpha=best_params['alpha'],
    l1_ratio=best_params['l1_ratio'],
    max_iter=2000,
    tol=1e-6
)
final_model.fit(X_train, y_train)

# ============================================================
# Evaluate on Test Set
# ============================================================

y_pred_test = final_model.predict(X_test)

# Metrics
mse = np.mean((y_test - y_pred_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred_test))
ss_res = np.sum((y_test - y_pred_test) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - ss_res / (ss_tot + 1e-8)

# Information Coefficient (correlation of predictions with actuals)
ic = np.corrcoef(y_test, y_pred_test)[0, 1]

print(f"\n=== Test Set Results ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R-squared: {r2:.4f}")
print(f"Information Coefficient: {ic:.4f}")

# ============================================================
# Feature Selection Analysis
# ============================================================

selected = final_model.get_selected_features(feature_names)
print(f"\n=== Feature Selection ===")
print(f"Total features: {n_features}")
print(f"Selected features: {len(selected)}")
print(f"Sparsity: {1 - len(selected)/n_features:.1%}")
print(f"\nTop selected features by |coefficient|:")

selected_sorted = sorted(selected, key=lambda x: abs(x[1]), reverse=True)
for name, coef in selected_sorted[:15]:
    direction = "+" if coef > 0 else "-"
    print(f"  {direction} {name}: {coef:.6f}")

# ============================================================
# Compare with Pure Lasso and Pure Ridge
# ============================================================

print(f"\n=== Model Comparison ===")

lasso = ElasticNetRegression(alpha=best_params['alpha'], l1_ratio=0.99, max_iter=2000, tol=1e-6)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_r2 = 1 - np.sum((y_test - lasso_pred)**2) / (ss_tot + 1e-8)
lasso_selected = len(lasso.get_selected_features())

ridge = ElasticNetRegression(alpha=best_params['alpha'], l1_ratio=0.01, max_iter=2000, tol=1e-6)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_r2 = 1 - np.sum((y_test - ridge_pred)**2) / (ss_tot + 1e-8)
ridge_selected = len(ridge.get_selected_features())

print(f"{'Model':<15} {'R-squared':>10} {'Features':>10}")
print(f"{'-'*35}")
print(f"{'ElasticNet':<15} {r2:>10.4f} {len(selected):>10}")
print(f"{'Lasso':<15} {lasso_r2:>10.4f} {lasso_selected:>10}")
print(f"{'Ridge':<15} {ridge_r2:>10.4f} {ridge_selected:>10}")

# ============================================================
# Trading Signal Generation
# ============================================================

print(f"\n=== Trading Signal Example ===")
# Use predictions as trading signals
predicted_returns = y_pred_test
signal = np.where(predicted_returns > 0.005, 1,      # Long
         np.where(predicted_returns < -0.005, -1, 0))  # Short or flat

n_long = np.sum(signal == 1)
n_short = np.sum(signal == -1)
n_flat = np.sum(signal == 0)

# Simulated strategy returns
strategy_returns = signal * y_test
cumulative_return = np.sum(strategy_returns)
hit_rate = np.mean((signal * y_test) > 0)

print(f"Long signals: {n_long}, Short signals: {n_short}, Flat: {n_flat}")
print(f"Strategy cumulative return: {cumulative_return:.4f}")
print(f"Hit rate: {hit_rate:.2%}")
print(f"Annualized Sharpe (approx): {np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(12):.2f}")
```

## Key Takeaways

1. **ElasticNet bridges the gap between Lasso and Ridge**, making it the go-to regularized regression for stock market applications where features are both numerous and correlated -- a near-universal condition in quantitative finance.

2. **The L1 component provides automatic feature selection**, reducing hundreds of financial indicators to a manageable set of truly predictive signals, while the L2 component ensures that correlated indicators within a group are handled stably.

3. **Proper cross-validation is essential** -- standard k-fold CV is inappropriate for financial time series. Use purged walk-forward validation with embargo periods to avoid lookahead bias and get realistic performance estimates.

4. **ElasticNet serves as an excellent baseline and feature selector** for more complex downstream models. The selected features can be fed into gradient boosting or neural network models that can capture non-linear relationships.

5. **Coefficient interpretability is a major practical advantage** in regulated financial environments where model transparency is required for compliance, risk oversight, and stakeholder communication.

6. **Monitor model stability over time** -- retraining on rolling windows and tracking coefficient changes helps detect regime shifts and model degradation before they impact portfolio performance.
