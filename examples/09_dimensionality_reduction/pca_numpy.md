# PCA (Principal Component Analysis) - Complete Guide with Stock Market Applications

## Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms a set of correlated variables into a smaller set of uncorrelated variables called principal components. Each principal component is a linear combination of the original features, ordered by the amount of variance it explains. The first principal component captures the most variance in the data, the second captures the most remaining variance orthogonal to the first, and so on. In stock market analysis, PCA is invaluable for distilling dozens of correlated technical indicators into a handful of independent market factors that capture the essential information.

PCA works by computing the eigenvectors and eigenvalues of the data's covariance matrix. The eigenvectors define the directions of the new feature space (principal components), while the eigenvalues quantify how much variance each direction captures. By retaining only the top-k components that explain most of the variance (typically 80-95%), analysts can dramatically reduce the dimensionality of their feature space while preserving the signal. This is particularly important in quantitative finance where models with too many correlated inputs suffer from multicollinearity, overfitting, and numerical instability.

For stock market technical analysis, traders commonly use 20-50 indicators (RSI, MACD, Bollinger Bands, moving averages, volume indicators, etc.), many of which are highly correlated because they are derived from the same underlying price and volume data. PCA reduces these to a small number of independent "market factors" -- perhaps 3-5 components that represent momentum, volatility, trend, volume, and mean-reversion. These factors are more stable, less noisy, and more suitable as inputs to trading models than the raw indicators.

## How It Works - The Math Behind It

### Step 1: Standardize the Data

For features with different scales (common in technical indicators):
```
x_standardized = (x - mean(x)) / std(x)
```

This ensures each feature contributes equally to the variance calculation.

### Step 2: Compute the Covariance Matrix

For a standardized data matrix X of shape (n_samples, d_features):
```
C = (1 / (n - 1)) * X^T * X
```

The covariance matrix C is a d x d symmetric matrix where:
- Diagonal entries: variance of each feature
- Off-diagonal entries: covariance between feature pairs

### Step 3: Eigendecomposition

Solve the eigenvalue equation:
```
C * v_i = lambda_i * v_i
```

Where:
- `v_i` is the i-th eigenvector (principal component direction)
- `lambda_i` is the i-th eigenvalue (variance explained by that component)

### Step 4: Sort and Select Components

Sort eigenvalues in descending order: `lambda_1 >= lambda_2 >= ... >= lambda_d`

The proportion of variance explained by component i:
```
PVE_i = lambda_i / sum_{j=1}^{d} lambda_j
```

Cumulative variance explained:
```
CVE_k = sum_{i=1}^{k} PVE_i
```

Select k components such that CVE_k >= threshold (e.g., 0.90 for 90%).

### Step 5: Project Data

Transform original data to the reduced space:
```
Z = X * W_k
```

Where W_k is the d x k matrix of the top-k eigenvectors.

### Reconstruction

Approximate the original data from the reduced representation:
```
X_reconstructed = Z * W_k^T
```

Reconstruction error measures information loss:
```
error = ||X - X_reconstructed||_F^2 / ||X||_F^2
```

### Alternative: SVD-Based PCA

For numerical stability and efficiency, PCA is often computed via Singular Value Decomposition:
```
X = U * S * V^T
```

Where:
- Columns of V are the principal component directions
- Diagonal of S contains singular values (sqrt of eigenvalues * (n-1))
- The top-k columns of V give the same result as eigendecomposition

## Stock Market Use Case: Reducing Correlated Technical Indicators to Principal Market Factors

### The Problem

A quantitative trading firm uses 15 technical indicators to generate trading signals for S&P 500 stocks. However, these indicators are highly correlated -- RSI and Stochastic Oscillator both measure momentum, Bollinger Band Width and ATR both measure volatility, and various moving average slopes all capture trend. When fed directly into a machine learning model, these correlated features cause multicollinearity (unstable coefficients), overfitting (too many degrees of freedom), and noise amplification. The firm needs to reduce these 15 indicators to a smaller set of uncorrelated "market factors" that capture the essential information for predicting returns.

### Stock Market Features (Input Data)

| Feature | Category | Description | Correlation Group |
|---------|----------|-------------|-------------------|
| RSI_14 | Momentum | Relative Strength Index (14-day) | Momentum cluster |
| Stoch_K | Momentum | Stochastic %K oscillator | Momentum cluster |
| Williams_R | Momentum | Williams %R oscillator | Momentum cluster |
| MACD_signal | Trend | MACD minus signal line | Trend cluster |
| SMA_slope_20 | Trend | Slope of 20-day SMA | Trend cluster |
| EMA_slope_50 | Trend | Slope of 50-day EMA | Trend cluster |
| ADX | Trend | Average Directional Index | Trend strength |
| BB_width | Volatility | Bollinger Band width | Volatility cluster |
| ATR_14 | Volatility | Average True Range (14-day) | Volatility cluster |
| hist_vol_20 | Volatility | 20-day historical volatility | Volatility cluster |
| OBV_slope | Volume | Slope of On-Balance Volume | Volume cluster |
| VWAP_deviation | Volume | Price deviation from VWAP | Volume cluster |
| volume_ratio | Volume | Volume vs 20-day average | Volume cluster |
| MFI | Volume/Momentum | Money Flow Index | Mixed |
| CCI | Momentum/Trend | Commodity Channel Index | Mixed |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_days = 500  # Trading days (~2 years)
n_indicators = 15

# Generate correlated technical indicators
# Start with 4 latent factors: momentum, trend, volatility, volume
n_factors = 4

# Latent factors
momentum_factor = np.random.randn(n_days)
trend_factor = np.random.randn(n_days)
volatility_factor = np.random.randn(n_days)
volume_factor = np.random.randn(n_days)

# Construct indicators as combinations of factors + noise
noise_level = 0.3

indicators = np.column_stack([
    # Momentum indicators (RSI, Stoch_K, Williams_R)
    0.8 * momentum_factor + 0.1 * trend_factor + noise_level * np.random.randn(n_days),
    0.75 * momentum_factor + 0.15 * trend_factor + noise_level * np.random.randn(n_days),
    0.7 * momentum_factor + 0.2 * volatility_factor + noise_level * np.random.randn(n_days),

    # Trend indicators (MACD_signal, SMA_slope, EMA_slope, ADX)
    0.15 * momentum_factor + 0.8 * trend_factor + noise_level * np.random.randn(n_days),
    0.1 * momentum_factor + 0.85 * trend_factor + noise_level * np.random.randn(n_days),
    0.1 * momentum_factor + 0.75 * trend_factor + 0.1 * volatility_factor + noise_level * np.random.randn(n_days),
    0.2 * momentum_factor + 0.6 * trend_factor + 0.2 * volatility_factor + noise_level * np.random.randn(n_days),

    # Volatility indicators (BB_width, ATR, hist_vol)
    0.1 * trend_factor + 0.85 * volatility_factor + noise_level * np.random.randn(n_days),
    0.05 * trend_factor + 0.80 * volatility_factor + noise_level * np.random.randn(n_days),
    0.9 * volatility_factor + noise_level * np.random.randn(n_days),

    # Volume indicators (OBV_slope, VWAP_dev, volume_ratio)
    0.15 * momentum_factor + 0.8 * volume_factor + noise_level * np.random.randn(n_days),
    0.1 * momentum_factor + 0.1 * trend_factor + 0.75 * volume_factor + noise_level * np.random.randn(n_days),
    0.85 * volume_factor + noise_level * np.random.randn(n_days),

    # Mixed indicators (MFI, CCI)
    0.4 * momentum_factor + 0.2 * trend_factor + 0.3 * volume_factor + noise_level * np.random.randn(n_days),
    0.35 * momentum_factor + 0.35 * trend_factor + 0.1 * volatility_factor + noise_level * np.random.randn(n_days),
])

indicator_names = [
    'RSI_14', 'Stoch_K', 'Williams_R',
    'MACD_signal', 'SMA_slope_20', 'EMA_slope_50', 'ADX',
    'BB_width', 'ATR_14', 'Hist_vol_20',
    'OBV_slope', 'VWAP_dev', 'Volume_ratio',
    'MFI', 'CCI'
]

print(f"Technical indicator matrix: {indicators.shape}")
print(f"Number of indicators: {len(indicator_names)}")

# Show correlation structure
corr_matrix = np.corrcoef(indicators.T)
print(f"\nCorrelation matrix statistics:")
print(f"  Mean absolute correlation: {np.mean(np.abs(corr_matrix[np.triu_indices(n_indicators, k=1)])):.3f}")
print(f"  Max correlation: {np.max(corr_matrix[np.triu_indices(n_indicators, k=1)]):.3f}")
print(f"  Pairs with |corr| > 0.7: {np.sum(np.abs(corr_matrix[np.triu_indices(n_indicators, k=1)]) > 0.7)}")
```

### The Model in Action

```python
import numpy as np

def standardize(X):
    """Standardize to zero mean, unit variance."""
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)
    stds[stds == 0] = 1
    return (X - means) / stds, means, stds

def pca(X, n_components=None):
    """
    PCA implementation from scratch using eigendecomposition.

    Parameters:
        X: data matrix (n_samples, n_features), should be standardized
        n_components: number of components to retain (None = all)

    Returns:
        components: principal component directions (n_components, n_features)
        explained_variance: variance explained by each component
        explained_variance_ratio: proportion of variance explained
        transformed: projected data (n_samples, n_components)
    """
    n_samples, n_features = X.shape

    # Step 1: Compute covariance matrix
    cov_matrix = np.cov(X.T)

    # Step 2: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalue (descending)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Step 3: Select components
    if n_components is None:
        n_components = n_features

    components = eigenvectors[:, :n_components].T  # (n_components, n_features)
    explained_variance = eigenvalues[:n_components]

    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = explained_variance / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Step 4: Project data
    transformed = X @ components.T  # (n_samples, n_components)

    return {
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'transformed': transformed,
        'n_components': n_components,
    }

def pca_svd(X, n_components=None):
    """
    PCA via SVD (more numerically stable).
    """
    n_samples, n_features = X.shape

    # SVD decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Eigenvalues from singular values
    eigenvalues = (S ** 2) / (n_samples - 1)

    if n_components is None:
        n_components = min(n_samples, n_features)

    components = Vt[:n_components]
    explained_variance = eigenvalues[:n_components]
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = explained_variance / total_var

    transformed = X @ components.T

    return {
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': np.cumsum(explained_variance_ratio),
        'transformed': transformed,
        'n_components': n_components,
    }

def select_n_components(explained_variance_ratio, threshold=0.90):
    """Select number of components to explain threshold variance."""
    cumulative = np.cumsum(explained_variance_ratio)
    n = np.argmax(cumulative >= threshold) + 1
    return n

def reconstruct(transformed, components, means, stds):
    """Reconstruct original data from PCA components."""
    reconstructed_std = transformed @ components
    return reconstructed_std * stds + means

def interpret_components(components, feature_names, top_n=5):
    """Interpret each principal component by its top feature loadings."""
    interpretations = []
    for i, comp in enumerate(components):
        # Sort features by absolute loading
        sorted_idx = np.argsort(np.abs(comp))[::-1]
        top_features = [(feature_names[j], comp[j]) for j in sorted_idx[:top_n]]
        interpretations.append(top_features)
    return interpretations

# Standardize indicators
X_std, means, stds = standardize(indicators)

# Run PCA
result = pca(X_std)

# Display variance explained
print("=== Variance Explained by Each Component ===")
for i in range(len(result['explained_variance_ratio'])):
    pve = result['explained_variance_ratio'][i]
    cve = result['cumulative_variance_ratio'][i]
    bar = '#' * int(pve * 50)
    print(f"  PC{i+1:>2d}: {pve*100:>6.2f}%  (cumulative: {cve*100:>6.2f}%)  {bar}")

# Select optimal number of components
n_opt = select_n_components(result['explained_variance_ratio'], threshold=0.90)
print(f"\nOptimal components for 90% variance: {n_opt}")

# Re-run with optimal components
result_opt = pca(X_std, n_components=n_opt)

# Interpret components
print(f"\n=== Component Interpretation ===")
interpretations = interpret_components(
    result_opt['components'], indicator_names, top_n=4
)
factor_labels = ['Momentum', 'Trend', 'Volatility', 'Volume', 'Mixed']

for i, interp in enumerate(interpretations):
    label = factor_labels[i] if i < len(factor_labels) else 'Unknown'
    print(f"\n  PC{i+1} (suggested: {label} factor, "
          f"explains {result_opt['explained_variance_ratio'][i]*100:.1f}%):")
    for feat, loading in interp:
        direction = '+' if loading > 0 else '-'
        print(f"    {direction}{feat:>15s}: {abs(loading):.3f}")

# Reconstruction error
X_reconstructed_std = result_opt['transformed'] @ result_opt['components']
recon_error = np.mean((X_std - X_reconstructed_std) ** 2)
print(f"\nReconstruction MSE: {recon_error:.4f}")
print(f"Dimensions reduced: {X_std.shape[1]} -> {n_opt}")
```

## Advantages

1. **Eliminates Multicollinearity in Technical Indicators**: Many technical indicators are derived from the same price/volume data and are highly correlated. PCA produces orthogonal (uncorrelated) components, solving multicollinearity problems that plague regression-based trading models. A model using 4 principal components is far more stable than one using 15 correlated indicators.

2. **Noise Reduction and Signal Extraction**: The lower-variance principal components often represent noise rather than signal. By discarding these (keeping only top components), PCA acts as a natural denoiser. In stock markets where signal-to-noise ratio is inherently low, this noise reduction can meaningfully improve model performance.

3. **Dramatic Dimensionality Reduction**: Reducing 15-50 technical indicators to 3-5 principal components reduces model complexity by an order of magnitude. This means faster training, less overfitting, and more robust out-of-sample performance -- all critical for live trading systems.

4. **Interpretable Market Factors**: When applied to technical indicators, the principal components often align with intuitive market concepts. The first component might capture "momentum," the second "volatility regime," and the third "volume trend." These interpretable factors help traders understand what their models are actually trading on.

5. **Stable Feature Space**: Individual technical indicators can be noisy and unstable, but the principal components they produce are more stable because they aggregate information across multiple indicators. This stability makes them better inputs for machine learning models that are sensitive to feature distribution shifts.

6. **Computational Efficiency for Downstream Models**: Training a random forest or neural network on 5 features instead of 50 is dramatically faster. For strategies that require frequent model retraining (daily or intraday), this speedup is operationally significant.

7. **Universal Applicability**: PCA works with any set of numerical indicators and makes no assumptions about the relationship between features and returns. It is a preprocessing step that improves almost any downstream analysis -- clustering, classification, regression, or anomaly detection.

## Disadvantages

1. **Linear Assumption Limits Capture of Market Dynamics**: PCA only captures linear relationships between features. If two indicators have a nonlinear relationship (e.g., volatility's quadratic relationship with returns), PCA will not capture it efficiently. Nonlinear dimensionality reduction methods like autoencoders or kernel PCA may be needed for complex market dynamics.

2. **Components Are Linear Combinations, Not Individual Features**: Each principal component is a weighted sum of all original indicators, making it harder to map back to specific trading signals. A portfolio manager cannot easily say "we're buying because RSI is oversold" -- instead it is "PC1 is at -2 standard deviations," which is less intuitive for discretionary overlay.

3. **Variance Does Not Equal Predictive Power**: PCA maximizes variance explained, but the most variable component is not necessarily the most predictive of future returns. A low-variance component capturing subtle mean-reversion signals might be more profitable than the high-variance momentum component. PCA has no concept of the target variable.

4. **Sensitivity to Outliers and Market Crashes**: Extreme market events (flash crashes, circuit breakers) create outlier data points that disproportionately influence the covariance matrix and hence the principal components. A single day with 10% market drop can rotate the entire component space, making historical components inconsistent with current ones.

5. **Stationarity Assumption is Violated**: PCA assumes the covariance structure of the data is stable over time. In stock markets, correlations between indicators change with market regimes (bull vs. bear, high vs. low volatility). Components computed during a calm market may be misleading during a crisis.

6. **Information Loss is Inevitable**: Discarding lower-ranked components loses some information. While this is usually noise, it could occasionally contain subtle but tradeable signals. There is no guarantee that the discarded variance is purely noise.

7. **Choosing the Number of Components is Subjective**: While the 90% variance threshold is common, the optimal number of components depends on the downstream application. For return prediction, fewer components (more regularization) might be better. For clustering, more components might be needed. This requires experimentation.

## When to Use in Stock Market

- **Before machine learning model training**: Reduce 20+ correlated technical indicators to 4-6 uncorrelated factors to prevent overfitting and multicollinearity
- **Factor model construction**: Identify the principal drivers of cross-sectional stock returns (market factor, size, value, momentum)
- **Risk decomposition**: Decompose portfolio variance into independent risk factors to understand where risk is concentrated
- **Feature engineering**: Create orthogonal features from correlated inputs for any downstream task (return prediction, volatility forecasting, regime detection)
- **Correlation structure analysis**: Understand the dominant patterns in how technical indicators co-move and whether this structure is changing over time
- **Data compression for storage**: Store 4-5 principal components instead of 50 indicators for historical backtesting databases
- **Pre-processing for clustering**: Apply PCA before K-Means or DBSCAN to remove correlated noise and improve cluster quality

## When NOT to Use in Stock Market

- **When nonlinear relationships dominate**: If indicator interactions are nonlinear (e.g., RSI effectiveness depends on volatility regime), PCA misses these patterns. Use autoencoders or kernel PCA instead
- **When feature interpretability is critical**: If the trading strategy requires explaining exactly which indicator triggered a signal, PCA's linear combinations are too opaque. Use feature selection (LASSO) instead
- **When the target variable is known**: If predicting returns, supervised methods (Partial Least Squares, supervised feature selection) that account for the target are more appropriate than unsupervised PCA
- **When features have very different importance**: PCA treats all variance equally, but some indicators might be known to be more important. Weighted or supervised dimensionality reduction is better
- **When data is sparse or categorical**: PCA requires dense numerical data. For binary trading signals or categorical features, use multiple correspondence analysis or factor analysis for mixed data
- **When real-time adaptation is needed**: Standard PCA requires batch recomputation when new data arrives. For streaming applications, use incremental PCA

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| n_components | Number of components to retain | 2 - 10 | Use 90% variance threshold as starting point; typically 3-5 for 15-20 indicators |
| variance_threshold | Cumulative variance to retain | 0.80 - 0.95 | 0.90 for return prediction (more regularization); 0.95 for risk analysis (preserve more info) |
| standardize | Whether to standardize first | Yes/No | Always Yes for technical indicators with different scales |
| method | Eigendecomposition vs SVD | eig, svd | SVD for numerical stability with many features |
| window_size | Rolling window for dynamic PCA | 60 - 252 days | 126 days (6 months) balances stability and adaptivity |

## Stock Market Performance Tips

1. **Use Rolling PCA for Non-Stationary Markets**: Compute PCA on a rolling window (e.g., 126 trading days) rather than the entire history. The covariance structure of market indicators changes with regime, and rolling PCA adapts the components to the current market environment.

2. **Standardize, Do Not Just Center**: Always standardize (z-score) before PCA when working with technical indicators. ATR (measured in price units) and RSI (0-100 scale) have vastly different variances, and PCA on unstandardized data will be dominated by the highest-variance feature.

3. **Monitor Component Stability**: Track the loadings of each principal component over time. If the "momentum factor" suddenly starts loading heavily on volatility indicators, the market regime has changed and models may need recalibration.

4. **Use the Scree Plot to Choose Components**: Plot eigenvalues and look for the "elbow" where they drop sharply. In practice, 3-5 components typically suffice for 15-20 technical indicators, explaining 85-95% of variance.

5. **Verify Orthogonality of Transformed Features**: After PCA, confirm that the correlation matrix of the transformed features is diagonal (all off-diagonal elements near zero). This validates that multicollinearity has been eliminated.

6. **Combine PCA with Domain Knowledge**: After computing components, examine the loadings to assign meaningful labels (momentum, volatility, trend). If a component does not have a clear interpretation, it may represent noise and can potentially be dropped.

7. **Consider Robust PCA for Market Data**: Standard PCA is sensitive to outliers. Use trimmed or robust covariance estimators (minimum covariance determinant) for the covariance matrix to reduce the influence of extreme market days.

## Comparison with Other Algorithms

| Feature | PCA | Factor Analysis | Autoencoders | t-SNE | UMAP |
|---------|-----|-----------------|-------------|-------|------|
| Linearity | Linear | Linear | Nonlinear | Nonlinear | Nonlinear |
| Preserves | Variance | Latent factors | Reconstruction | Local structure | Local + Global |
| Invertible | Yes | Yes | Yes | No | Approximately |
| Interpretability | Moderate (loadings) | High (factors) | Low | None | Low |
| Speed | Very fast | Fast | Slow | Very slow | Moderate |
| Scales to many features | Excellent | Good | Good | Poor (>50) | Good |
| Supervised version | Partial Least Squares | None | Supervised AE | None | None |
| Stock market fit | Feature reduction | Factor models | Complex patterns | Visualization only | Visualization |
| Handles non-Gaussian | Yes | No (assumes Gaussian) | Yes | Yes | Yes |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete PCA Pipeline for Technical Indicator Reduction
# ============================================================

np.random.seed(42)

# --- Generate Realistic Technical Indicator Data ---
def generate_technical_indicators(n_days=500, n_stocks=50):
    """
    Generate synthetic technical indicator data for multiple stocks.
    Indicators are constructed from latent market factors with noise.
    """
    # Latent market factors
    momentum = np.random.randn(n_days)
    trend = np.random.randn(n_days)
    volatility = np.abs(np.random.randn(n_days))
    volume = np.random.randn(n_days)
    mean_reversion = np.random.randn(n_days)

    noise = 0.25

    # 15 technical indicators as functions of latent factors
    data = np.column_stack([
        # Momentum group
        50 + 15 * (0.8*momentum + 0.1*trend + noise*np.random.randn(n_days)),  # RSI_14
        50 + 20 * (0.75*momentum + 0.15*trend + noise*np.random.randn(n_days)), # Stoch_K
        -50 + 20 * (0.7*momentum + 0.1*volatility + noise*np.random.randn(n_days)), # Williams_R
        0.5 * (0.35*momentum + 0.6*trend + noise*np.random.randn(n_days)),      # CCI normalized

        # Trend group
        0.3 * (0.15*momentum + 0.8*trend + noise*np.random.randn(n_days)),      # MACD_signal
        0.001 * (0.1*momentum + 0.85*trend + noise*np.random.randn(n_days)),    # SMA_slope_20
        0.0008 * (0.1*momentum + 0.75*trend + noise*np.random.randn(n_days)),   # EMA_slope_50
        25 + 8 * (0.2*momentum + 0.6*trend + 0.2*volatility + noise*np.random.randn(n_days)), # ADX

        # Volatility group
        0.03 + 0.015 * (0.05*trend + 0.85*volatility + noise*np.random.randn(n_days)), # BB_width
        2.0 + 0.8 * (0.05*trend + 0.8*volatility + noise*np.random.randn(n_days)),     # ATR_14
        0.15 + 0.06 * (0.9*volatility + noise*np.random.randn(n_days)),                 # Hist_vol_20

        # Volume group
        1000 * (0.15*momentum + 0.8*volume + noise*np.random.randn(n_days)),    # OBV_slope
        0.5 * (0.1*momentum + 0.1*trend + 0.75*volume + noise*np.random.randn(n_days)), # VWAP_dev
        1.0 + 0.3 * (0.85*volume + noise*np.random.randn(n_days)),             # Volume_ratio

        # Mixed
        50 + 12 * (0.4*momentum + 0.2*volume + 0.2*mean_reversion + noise*np.random.randn(n_days)), # MFI
    ])

    indicator_names = [
        'RSI_14', 'Stoch_K', 'Williams_R', 'CCI',
        'MACD_signal', 'SMA_slope_20', 'EMA_slope_50', 'ADX',
        'BB_width', 'ATR_14', 'Hist_vol_20',
        'OBV_slope', 'VWAP_dev', 'Volume_ratio',
        'MFI'
    ]

    return data, indicator_names

# --- PCA Implementation ---
def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma, mu, sigma

def compute_pca(X_std):
    """Full PCA via eigendecomposition."""
    n, d = X_std.shape
    cov = np.cov(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = np.sum(eigenvalues)
    pve = eigenvalues / total_var
    cve = np.cumsum(pve)

    return eigenvalues, eigenvectors, pve, cve

def project(X_std, eigenvectors, k):
    """Project data onto top-k components."""
    W = eigenvectors[:, :k]
    return X_std @ W, W

def reconstruction_error(X_std, X_projected, W):
    """Compute reconstruction error."""
    X_recon = X_projected @ W.T
    mse = np.mean((X_std - X_recon) ** 2)
    return mse

def rolling_pca(X_std, window=126, n_components=4):
    """Rolling window PCA to track component evolution."""
    n_days = X_std.shape[0]
    loadings_history = []
    variance_history = []

    for end in range(window, n_days, 21):  # every ~month
        start = end - window
        X_window = X_std[start:end]

        eigenvalues, eigenvectors, pve, _ = compute_pca(X_window)
        loadings_history.append(eigenvectors[:, :n_components].copy())
        variance_history.append(pve[:n_components].copy())

    return loadings_history, variance_history

def component_stability(loadings_history):
    """Measure how stable components are over time."""
    n_periods = len(loadings_history)
    if n_periods < 2:
        return []

    stabilities = []
    for t in range(1, n_periods):
        # Absolute cosine similarity between consecutive loadings
        for k in range(loadings_history[t].shape[1]):
            cos_sim = np.abs(np.dot(
                loadings_history[t][:, k],
                loadings_history[t-1][:, k]
            ))
            stabilities.append(cos_sim)

    return np.mean(stabilities)

# --- Run Pipeline ---
indicators, indicator_names = generate_technical_indicators(n_days=500)
X_std, means, stds = standardize(indicators)

print(f"Input: {indicators.shape[0]} days x {indicators.shape[1]} indicators")

# Full PCA
eigenvalues, eigenvectors, pve, cve = compute_pca(X_std)

print(f"\n=== Eigenvalue Spectrum ===")
for i in range(len(pve)):
    bar = '#' * int(pve[i] * 60)
    marker = ' <-- 90% threshold' if i > 0 and cve[i] >= 0.90 and cve[i-1] < 0.90 else ''
    print(f"  PC{i+1:>2d}: eigenvalue={eigenvalues[i]:>6.3f}  "
          f"variance={pve[i]*100:>5.2f}%  "
          f"cumulative={cve[i]*100:>5.2f}%  {bar}{marker}")

# Select components
n_components = np.argmax(cve >= 0.90) + 1
print(f"\nSelected {n_components} components (explain {cve[n_components-1]*100:.1f}% variance)")
print(f"Dimension reduction: {indicators.shape[1]} -> {n_components} ({(1 - n_components/indicators.shape[1])*100:.0f}% reduction)")

# Project and analyze
X_projected, W = project(X_std, eigenvectors, n_components)
recon_err = reconstruction_error(X_std, X_projected, W)
print(f"Reconstruction MSE: {recon_err:.4f}")

# Interpret components
print(f"\n=== Component Interpretation ===")
factor_suggestions = {
    0: 'Momentum Factor',
    1: 'Trend Factor',
    2: 'Volatility Factor',
    3: 'Volume Factor',
    4: 'Mean-Reversion Factor',
}

for i in range(n_components):
    loadings = eigenvectors[:, i]
    sorted_idx = np.argsort(np.abs(loadings))[::-1]

    suggested = factor_suggestions.get(i, 'Unknown')
    print(f"\nPC{i+1} ({suggested}, explains {pve[i]*100:.1f}% variance):")
    print(f"  Top indicator loadings:")
    for j in sorted_idx[:5]:
        sign = '+' if loadings[j] > 0 else '-'
        print(f"    {sign} {indicator_names[j]:>15s}: {abs(loadings[j]):.3f}")

# Verify orthogonality
corr_projected = np.corrcoef(X_projected.T)
print(f"\n=== Orthogonality Check ===")
print(f"Max off-diagonal correlation: {np.max(np.abs(corr_projected - np.eye(n_components))):.6f}")
print(f"(should be ~0, confirming uncorrelated components)")

# Rolling PCA stability
print(f"\n=== Rolling PCA Stability ===")
loadings_hist, var_hist = rolling_pca(X_std, window=126, n_components=n_components)
stability = component_stability(loadings_hist)
print(f"Average component stability (cosine similarity): {stability:.4f}")
print(f"(> 0.8 = stable components, < 0.5 = unstable/regime shift)")

# Compare model with and without PCA
print(f"\n=== Feature Correlation Before vs After PCA ===")
corr_before = np.corrcoef(X_std.T)
off_diag_before = corr_before[np.triu_indices(X_std.shape[1], k=1)]
print(f"Before PCA: avg |correlation| = {np.mean(np.abs(off_diag_before)):.3f}")
print(f"            max |correlation| = {np.max(np.abs(off_diag_before)):.3f}")
print(f"            pairs with |r| > 0.7: {np.sum(np.abs(off_diag_before) > 0.7)}")

corr_after = np.corrcoef(X_projected.T)
off_diag_after = corr_after[np.triu_indices(n_components, k=1)]
print(f"After PCA:  avg |correlation| = {np.mean(np.abs(off_diag_after)):.6f}")
print(f"            max |correlation| = {np.max(np.abs(off_diag_after)):.6f}")
print(f"            pairs with |r| > 0.7: {np.sum(np.abs(off_diag_after) > 0.7)}")

# Practical output: reduced feature matrix for downstream modeling
print(f"\n=== Ready for Downstream Modeling ===")
print(f"Original features: {X_std.shape}")
print(f"PCA-reduced features: {X_projected.shape}")
print(f"Sample of transformed data (first 5 days):")
for d in range(5):
    vals = '  '.join([f"PC{j+1}={X_projected[d,j]:>6.3f}" for j in range(n_components)])
    print(f"  Day {d+1}: {vals}")
```

## Key Takeaways

1. **PCA transforms correlated technical indicators into uncorrelated market factors**, eliminating multicollinearity that plagues trading models. This is its single most valuable contribution to quantitative finance.

2. **Typically 3-5 principal components capture 85-95% of the variance** in 15-20 technical indicators. This dramatic reduction (70-80% fewer features) makes models faster, more stable, and less prone to overfitting.

3. **The principal components often map to intuitive market concepts**: momentum (PC1), trend (PC2), volatility (PC3), and volume (PC4). Examining the loadings confirms this interpretation and builds confidence in the reduced features.

4. **Always standardize before applying PCA** to stock market indicators. Features with different scales (RSI: 0-100 vs. MACD: -2 to +2) would otherwise bias the components toward high-variance features regardless of their informational content.

5. **Rolling PCA is essential for non-stationary markets**. The covariance structure of indicators changes with market regime (bull, bear, crisis). Recompute PCA periodically (monthly) to keep components aligned with current market dynamics.

6. **PCA is unsupervised and variance-maximizing, not prediction-maximizing**. The highest-variance component is not necessarily the most useful for return prediction. Consider supervised alternatives (PLS) when the goal is specifically forecasting.

7. **Use reconstruction error to validate** that the reduced representation preserves essential information. If reconstruction MSE is low, the dropped components contained mainly noise.

8. **PCA is a preprocessing step, not a model**. Its value comes from feeding cleaner, uncorrelated features into downstream models (random forests, neural networks, clustering algorithms) that benefit from reduced dimensionality.
