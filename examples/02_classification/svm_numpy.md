# Support Vector Machine (SVM) - Complete Guide with Stock Market Applications

## Overview

Support Vector Machines (SVMs) are powerful supervised learning algorithms that find the optimal hyperplane separating different classes in feature space. The key insight behind SVM is the concept of maximum margin classification: rather than finding just any decision boundary that separates the classes, SVM finds the boundary that maximizes the geometric distance (margin) between itself and the nearest data points from each class. These closest points are called support vectors, and they alone determine the position of the decision boundary.

What makes SVM particularly elegant for stock market applications is the kernel trick, which allows the algorithm to implicitly map input features into a higher-dimensional space where linearly inseparable classes become separable. For market state classification -- distinguishing between bull, bear, and sideways regimes -- the relationships between technical indicators and market states are rarely linear. The kernel trick enables SVM to capture complex, non-linear decision boundaries without explicitly computing the high-dimensional feature representations, making it computationally tractable even with many features.

In financial markets, SVM has found widespread adoption for regime classification tasks. Markets cycle through distinct states (trending up, trending down, range-bound), and correctly identifying the current regime is critical for strategy selection. An SVM trained on volatility metrics, trend indicators, and market breadth measures can classify the current market environment, allowing a trading system to switch between trend-following and mean-reversion strategies accordingly. The maximum margin property provides a natural confidence measure -- points far from the decision boundary represent clear regime classifications, while points near the boundary indicate ambiguous transitional periods.

## How It Works - The Math Behind It

### Linear SVM: The Optimization Problem

Given training data `{(x_i, y_i)}` where `x_i` is the feature vector and `y_i in {-1, +1}` is the class label, SVM solves:

```
minimize    (1/2) * ||w||^2
subject to  y_i * (w . x_i + b) >= 1  for all i
```

where `w` is the weight vector (normal to the hyperplane) and `b` is the bias term. The constraint ensures all points are correctly classified with a margin of at least 1/||w||.

### Soft Margin SVM (Handling Noisy Financial Data)

Real market data is never perfectly separable. The soft margin formulation introduces slack variables:

```
minimize    (1/2) * ||w||^2 + C * sum(xi_i)
subject to  y_i * (w . x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i
```

The parameter `C` controls the trade-off between margin width and classification errors. Larger `C` penalizes misclassifications more heavily (tighter fit), while smaller `C` allows more violations for a wider margin (better generalization).

### The Kernel Trick

Instead of explicitly mapping features to a high-dimensional space via function `phi(x)`, the kernel trick computes inner products directly:

```
K(x_i, x_j) = phi(x_i) . phi(x_j)
```

Common kernels for financial applications:

**Linear Kernel:**
```
K(x_i, x_j) = x_i . x_j
```

**Radial Basis Function (RBF/Gaussian) Kernel:**
```
K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
```

**Polynomial Kernel:**
```
K(x_i, x_j) = (gamma * x_i . x_j + r)^d
```

### The Dual Formulation

The optimization is typically solved in its dual form:

```
maximize    sum(alpha_i) - (1/2) * sum(sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j)))
subject to  0 <= alpha_i <= C  for all i
            sum(alpha_i * y_i) = 0
```

The decision function becomes:

```
f(x) = sign(sum(alpha_i * y_i * K(x_i, x)) + b)
```

Only support vectors (where `alpha_i > 0`) contribute to the decision function, making prediction efficient.

### Multi-Class Extension (One-vs-One)

For classifying three market states (bull/bear/sideways), SVM uses the one-vs-one approach:

- Train 3 binary classifiers: bull-vs-bear, bull-vs-sideways, bear-vs-sideways
- For a new sample, each classifier votes for one class
- The class with the most votes wins

Total classifiers needed: `K*(K-1)/2` where K is the number of classes (3 classifiers for 3 states).

### Hinge Loss Interpretation

SVM minimizes the hinge loss:

```
L = (1/N) * sum(max(0, 1 - y_i * f(x_i))) + (lambda/2) * ||w||^2
```

This loss is zero when the sample is correctly classified with margin >= 1, and increases linearly with the violation magnitude. The flat region of zero loss beyond the margin gives SVM its sparsity property -- most training points do not influence the decision boundary.

## Stock Market Use Case: Classifying Market States (Bull/Bear/Sideways)

### The Problem

A portfolio management firm needs to automatically classify the current market regime to select appropriate trading strategies. During bull markets, they deploy momentum and trend-following strategies. During bear markets, they shift to defensive positions and short-selling. During sideways markets, they use mean-reversion and options strategies. Accurate regime classification with a confidence measure is critical for smooth strategy transitions.

### Stock Market Features (Input Data)

| Feature | Description | Type | Example Value |
|---------|-------------|------|---------------|
| Return_20d | 20-day cumulative return | Continuous | 0.045 |
| Return_60d | 60-day cumulative return | Continuous | 0.12 |
| Volatility_20d | 20-day realized volatility (annualized) | Continuous | 0.18 |
| Volatility_ratio | 20-day vol / 60-day vol | Continuous | 1.25 |
| Trend_strength | ADX (Average Directional Index) | Continuous | 32.5 |
| Breadth_advance | Advance-decline ratio | Continuous | 1.45 |
| VIX_level | CBOE Volatility Index | Continuous | 18.5 |
| VIX_change_10d | 10-day VIX change | Continuous | -2.3 |
| Yield_curve_slope | 10Y minus 2Y Treasury yield | Continuous | 0.85 |
| Put_call_ratio | Total put/call options ratio | Continuous | 0.72 |
| SMA50_above_SMA200 | 1 if SMA50 > SMA200, else 0 | Binary | 1 |
| New_highs_ratio | New 52-week highs / (highs + lows) | Continuous | 0.68 |
| Sector_dispersion | Cross-sector return dispersion | Continuous | 0.035 |
| Credit_spread | High-yield minus investment-grade spread | Continuous | 3.45 |

**Target Variable:** `market_state` -- 0 (Bear), 1 (Sideways), 2 (Bull)

### Example Data Structure

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1200

# Simulate market regime data
# Create clusters representing three market states
n_bull = n_samples // 3
n_bear = n_samples // 3
n_sideways = n_samples - 2 * (n_samples // 3)

# Bull market characteristics
bull_data = {
    'Return_20d': np.random.normal(0.04, 0.02, n_bull),
    'Return_60d': np.random.normal(0.10, 0.04, n_bull),
    'Volatility_20d': np.random.normal(0.14, 0.03, n_bull),
    'Volatility_ratio': np.random.normal(0.9, 0.15, n_bull),
    'Trend_strength': np.random.normal(30, 8, n_bull),
    'Breadth_advance': np.random.normal(1.5, 0.3, n_bull),
    'VIX_level': np.random.normal(15, 3, n_bull),
    'VIX_change_10d': np.random.normal(-1, 2, n_bull),
    'Yield_curve_slope': np.random.normal(1.2, 0.4, n_bull),
    'Put_call_ratio': np.random.normal(0.65, 0.1, n_bull),
    'SMA50_above_SMA200': np.random.binomial(1, 0.85, n_bull).astype(float),
    'New_highs_ratio': np.random.normal(0.72, 0.1, n_bull),
    'Sector_dispersion': np.random.normal(0.025, 0.01, n_bull),
    'Credit_spread': np.random.normal(3.0, 0.5, n_bull),
}

# Bear market characteristics
bear_data = {
    'Return_20d': np.random.normal(-0.05, 0.03, n_bear),
    'Return_60d': np.random.normal(-0.12, 0.05, n_bear),
    'Volatility_20d': np.random.normal(0.28, 0.06, n_bear),
    'Volatility_ratio': np.random.normal(1.3, 0.2, n_bear),
    'Trend_strength': np.random.normal(35, 10, n_bear),
    'Breadth_advance': np.random.normal(0.6, 0.2, n_bear),
    'VIX_level': np.random.normal(28, 6, n_bear),
    'VIX_change_10d': np.random.normal(3, 4, n_bear),
    'Yield_curve_slope': np.random.normal(0.2, 0.5, n_bear),
    'Put_call_ratio': np.random.normal(1.1, 0.2, n_bear),
    'SMA50_above_SMA200': np.random.binomial(1, 0.2, n_bear).astype(float),
    'New_highs_ratio': np.random.normal(0.15, 0.1, n_bear),
    'Sector_dispersion': np.random.normal(0.055, 0.015, n_bear),
    'Credit_spread': np.random.normal(5.5, 1.0, n_bear),
}

# Sideways market characteristics
sideways_data = {
    'Return_20d': np.random.normal(0.0, 0.015, n_sideways),
    'Return_60d': np.random.normal(0.01, 0.03, n_sideways),
    'Volatility_20d': np.random.normal(0.16, 0.03, n_sideways),
    'Volatility_ratio': np.random.normal(1.0, 0.1, n_sideways),
    'Trend_strength': np.random.normal(15, 5, n_sideways),
    'Breadth_advance': np.random.normal(1.0, 0.15, n_sideways),
    'VIX_level': np.random.normal(17, 3, n_sideways),
    'VIX_change_10d': np.random.normal(0, 2, n_sideways),
    'Yield_curve_slope': np.random.normal(0.8, 0.3, n_sideways),
    'Put_call_ratio': np.random.normal(0.85, 0.1, n_sideways),
    'SMA50_above_SMA200': np.random.binomial(1, 0.5, n_sideways).astype(float),
    'New_highs_ratio': np.random.normal(0.45, 0.15, n_sideways),
    'Sector_dispersion': np.random.normal(0.03, 0.008, n_sideways),
    'Credit_spread': np.random.normal(3.8, 0.6, n_sideways),
}

feature_names = list(bull_data.keys())
X = np.vstack([
    np.column_stack([bull_data[f] for f in feature_names]),
    np.column_stack([bear_data[f] for f in feature_names]),
    np.column_stack([sideways_data[f] for f in feature_names]),
])
y = np.array([2]*n_bull + [0]*n_bear + [1]*n_sideways)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: Bull={np.sum(y==2)}, Bear={np.sum(y==0)}, "
      f"Sideways={np.sum(y==1)}")
```

### The Model in Action

```python
# SVM decision process for market state classification
# Today's features: Return_20d=0.03, Volatility_20d=0.15, VIX=16, ADX=28

# Step 1: Feature scaling (critical for SVM)
#   z_return = (0.03 - mean) / std = 1.2
#   z_vol = (0.15 - mean) / std = -0.8
#   z_vix = (16 - mean) / std = -0.6
#   z_adx = (28 - mean) / std = 0.5

# Step 2: One-vs-One Classification
#   Bull-vs-Bear:     f(x) = +2.1  => BULL  (high confidence)
#   Bull-vs-Sideways: f(x) = +0.8  => BULL  (moderate confidence)
#   Bear-vs-Sideways: f(x) = -1.5  => SIDEWAYS (clear separation)

# Step 3: Voting
#   Bull: 2 votes | Bear: 0 votes | Sideways: 1 vote
#   Result: BULL MARKET regime => Deploy momentum strategies

# Confidence: Distance to nearest decision boundary = 0.8
# Interpretation: Moderate confidence -- near sideways boundary,
#                 reduce position sizing as precaution
```

## Advantages

1. **Maximum margin provides a natural confidence measure for regime transitions.** The distance of a data point from the SVM decision boundary directly indicates classification confidence. Points close to the boundary represent transitional market periods where the regime is ambiguous, providing a natural signal to reduce position sizes or blend strategies from both regimes. This geometric interpretation is more intuitive than probability scores from other models.

2. **Effective in high-dimensional feature spaces without overfitting.** SVM's regularization through margin maximization means it handles many features well relative to sample size. In stock market applications, you might have dozens of technical and fundamental features but only hundreds of labeled regime periods. SVM's structural risk minimization prevents overfitting in this high-dimension, low-sample scenario better than most other classifiers.

3. **Kernel trick captures non-linear regime boundaries.** Market states are not linearly separable in raw feature space -- the transition from bull to bear involves complex interactions between volatility, trend, breadth, and credit metrics. The RBF kernel maps these features into a space where regime boundaries become linear, capturing the non-linear relationships between market indicators and regimes without explicit feature engineering.

4. **Sparse solution enables efficient prediction.** Only support vectors (typically 10-30% of training data) contribute to the decision function. This means the model stores and computes with only the most informative historical market observations -- the boundary cases between regimes. For real-time regime classification, this sparsity keeps prediction latency low.

5. **Robust to moderate class imbalance.** Market regimes are not equally distributed -- bull markets tend to last longer than bear markets. SVM can be adjusted with class-specific C parameters to compensate for imbalanced regime frequencies, ensuring the model does not simply default to predicting the majority class.

6. **Strong theoretical guarantees on generalization.** SVM's generalization bounds depend on the margin and the number of support vectors, not the dimensionality of the feature space. This provides theoretical assurance that the model will perform well on unseen market data if the training margin is large, which is more than most black-box models can offer.

7. **Deterministic predictions (no randomness in training).** Unlike Random Forest or neural networks, SVM optimization is a convex quadratic program with a unique global solution. This means identical training data always produces the identical model, ensuring fully reproducible backtesting results -- a critical requirement for auditable trading systems.

## Disadvantages

1. **Feature scaling is mandatory and scale-sensitive.** SVM's distance-based computations mean that features on different scales dominate arbitrarily. Volatility (0.10-0.40) and VIX (10-80) differ by orders of magnitude, and without careful standardization, VIX will overwhelm the distance calculations. Choosing between standard scaling, min-max scaling, or robust scaling is a non-trivial decision that can significantly impact regime classification accuracy.

2. **Training time scales poorly with dataset size.** SVM training is O(n^2) to O(n^3) in memory and computation, where n is the number of training samples. For a regime classifier trained on 20+ years of daily data across multiple markets, the dataset can exceed 100,000 samples, making standard SVM implementations prohibitively slow. Approximate methods (SGD-SVM, kernel approximation) sacrifice some accuracy.

3. **Kernel and hyperparameter selection requires extensive tuning.** The choice of kernel (linear, RBF, polynomial) and hyperparameters (C, gamma) dramatically affects performance. A grid search over these parameters using time-series cross-validation is computationally expensive, and the optimal settings may change as market dynamics evolve, requiring periodic re-tuning.

4. **No inherent probability outputs.** SVM produces decision function scores, not probabilities. For portfolio allocation that requires probability estimates of each regime, additional calibration (Platt scaling or isotonic regression) is needed. These calibration methods add complexity and may not be well-calibrated in the tails -- precisely where regime transitions occur.

5. **Difficulty with regime transition dynamics.** SVM treats each observation independently and does not model temporal dynamics. Market regimes have momentum -- a bear market is more likely to continue than to instantly switch to a bull market. SVM cannot capture this state persistence without additional temporal features or a hidden Markov model overlay.

6. **Sensitive to label definition.** How you define and label market regimes significantly affects model quality. A 20% decline might be labeled "bear" by one definition but "correction within a bull" by another. SVM will learn whatever labeling scheme is provided, and poor regime definitions produce poor classifiers regardless of the algorithm's power.

7. **Black-box nature of kernel-space decisions.** While the original features are interpretable (VIX, volatility, returns), the decision boundary in kernel space is not. It is difficult to explain why the SVM classified a particular day as "sideways" rather than "bull" in terms that a portfolio manager can act on. This opacity can create trust issues in institutional settings.

## When to Use in Stock Market

- For market regime classification where clear geometric separation between states exists in feature space
- When you have a moderate-sized dataset (500-50,000 observations) with many features relative to samples
- When deterministic, reproducible classification results are required for audit trails
- For binary or few-class problems (bull/bear, risk-on/risk-off) where one-vs-one is efficient
- When you need a confidence measure based on distance from the decision boundary for position sizing
- When the feature set includes a mix of volatility, trend, and breadth indicators that interact non-linearly
- For strategy selection systems that switch between trading strategies based on detected regime

## When NOT to Use in Stock Market

- For very large datasets (100,000+ observations) where training time becomes impractical without approximation
- When you need well-calibrated probability outputs for direct use in Kelly criterion or mean-variance optimization
- For high-frequency applications where feature distributions shift rapidly within a single training window
- When model interpretability and explainability are strict regulatory requirements (prefer decision trees or logistic regression)
- For online learning scenarios where the model must update incrementally with each new observation
- When the number of market states is large (>5) making one-vs-one combinatorially expensive
- When temporal dependencies between successive observations are critical to the classification task

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Recommendation |
|----------------|-------------|---------------|----------------------------|
| C | Regularization parameter (misclassification penalty) | 0.01 - 1000 | Start with 1.0. For noisy market data, use smaller C (0.1-1.0) for wider margins. Increase for clear regime definitions. |
| kernel | Kernel function type | linear, rbf, poly | RBF is the default choice for market regime classification. Try linear first as a baseline -- if it works well, the problem may be simpler than expected. |
| gamma | RBF kernel coefficient (1/(2*sigma^2)) | 'scale', 0.001 - 10 | Use 'scale' (1/(n_features * X.var())) as default. Smaller gamma = smoother boundaries = less overfitting. Critical to tune via cross-validation. |
| class_weight | Per-class penalty multiplier | None, 'balanced', custom dict | Use 'balanced' to compensate for unequal regime durations. Bull markets last longer, so without balancing, SVM may under-predict bear regimes. |
| degree | Polynomial kernel degree | 2 - 5 | Only relevant for polynomial kernel. degree=2 captures pairwise feature interactions. Higher degrees overfit on financial data. |
| tol | Convergence tolerance | 1e-4 to 1e-2 | Default 1e-3 is fine. Increase to 1e-2 if training is slow on large datasets, at minimal accuracy cost. |
| max_iter | Maximum training iterations | 1000 - 100000 | Increase if convergence warnings appear. Financial data may need more iterations due to overlapping classes. |

## Stock Market Performance Tips

1. **Standardize features rigorously.** Use rolling z-score normalization: for each feature at time t, subtract the mean and divide by the standard deviation computed over the trailing N days (e.g., 252 trading days). This prevents look-ahead bias and adapts to changing feature distributions over time.

2. **Engineer temporal features explicitly.** Since SVM does not model time dynamics, add features that capture regime momentum: consecutive days above/below key thresholds, rolling return percentiles, days since last regime change, and volatility acceleration (change in volatility).

3. **Use walk-forward validation with regime-aware splits.** Ensure that each validation fold contains complete regime transitions, not just samples from the middle of a regime. This provides realistic evaluation of the model's ability to detect regime changes.

4. **Implement a multi-scale approach.** Train separate SVMs on features computed over different lookback windows (20-day, 60-day, 120-day). A voting or stacking scheme across time scales produces more robust regime classification than any single time scale.

5. **Create a transition buffer zone.** Rather than hard regime labels, define a "transition" class for periods within N days of a regime change. This prevents the SVM from trying to precisely classify inherently ambiguous transitional periods, improving accuracy on clear regime periods.

6. **Combine with position sizing based on margin distance.** Scale portfolio exposure by the normalized distance from the decision boundary. When the SVM is highly confident in a bull regime (large positive margin), take full positions. Near the boundary, reduce exposure proportionally. This natural risk management is a unique advantage of SVM.

7. **Monitor support vector turnover.** After retraining, compare the new support vectors to the previous model's. High turnover suggests the market's regime structure is changing, warranting closer manual review before deploying the updated model.

## Comparison with Other Algorithms

| Criterion | SVM (RBF) | Random Forest | Logistic Regression | K-Nearest Neighbors | Neural Network |
|-----------|-----------|---------------|---------------------|---------------------|----------------|
| Regime separation quality | Excellent (maximum margin) | Good | Moderate (linear only) | Good (local patterns) | Excellent |
| Training speed | Slow for large N | Fast | Very fast | No training needed | Slow |
| Prediction speed | Fast (sparse) | Moderate | Very fast | Slow (all-point comparison) | Fast |
| Feature scaling required | Yes (critical) | No | Yes | Yes (critical) | Yes |
| Handles non-linearity | Yes (kernel) | Yes (tree splits) | No | Yes (inherently) | Yes |
| Interpretability | Low (kernel space) | Moderate | High | Low | Very low |
| Probability calibration | Poor (needs calibration) | Good | Excellent | Moderate | Moderate |
| Robustness to noise | Good (margin regularization) | Very good (averaging) | Moderate | Poor (noise in neighbors) | Moderate |
| Multi-class handling | One-vs-one (combinatorial) | Natural | Natural (softmax) | Natural | Natural (softmax) |
| Best market use case | Regime classification | General classification | Trend direction | Pattern matching | Complex regime detection |

## Real-World Stock Market Example

```python
import numpy as np
from collections import Counter

# =============================================================================
# SVM from Scratch for Market State Classification
# =============================================================================

class StandardScaler:
    """Feature standardization for SVM."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0  # prevent division by zero
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class BinarySVM:
    """Binary SVM classifier using SMO-like gradient descent."""

    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 lr=0.001, n_iters=1000):
        self.C = C
        self.kernel_type = kernel
        self.gamma_param = gamma
        self.lr = lr
        self.n_iters = n_iters
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.gamma = None

    def _compute_kernel(self, X1, X2):
        """Compute the kernel matrix between X1 and X2."""
        if self.kernel_type == 'linear':
            return X1 @ X2.T
        elif self.kernel_type == 'rbf':
            sq_dists = (np.sum(X1**2, axis=1, keepdims=True) +
                       np.sum(X2**2, axis=1) -
                       2 * X1 @ X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Set gamma
        if self.gamma_param == 'scale':
            self.gamma = 1.0 / (n_features * np.var(X))
        else:
            self.gamma = self.gamma_param

        # Compute kernel matrix
        K = self._compute_kernel(X, X)

        # Initialize alphas
        self.alphas = np.zeros(n_samples)
        self.b = 0

        # Sub-gradient descent on dual
        for iteration in range(self.n_iters):
            # Decision values
            decision = (self.alphas * y) @ K + self.b

            # Compute margins
            margins = y * decision

            # Find violating samples
            for i in range(n_samples):
                if margins[i] < 1:
                    # Misclassified or within margin
                    self.alphas[i] += self.lr * (1 - margins[i])
                    self.b += self.lr * y[i] * (1 - margins[i])

                # Project alphas to [0, C]
                self.alphas[i] = np.clip(self.alphas[i], 0, self.C)

        # Identify support vectors
        sv_mask = self.alphas > 1e-7
        self.support_vectors_idx = np.where(sv_mask)[0]

        return self

    def decision_function(self, X):
        K = self._compute_kernel(X, self.X_train)
        return (self.alphas * self.y_train) @ K.T + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class MultiClassSVM:
    """One-vs-One multi-class SVM for market state classification."""

    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 lr=0.001, n_iters=500):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.lr = lr
        self.n_iters = n_iters
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Train one classifier for each pair of classes
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i, class_j = self.classes[i], self.classes[j]

                # Select samples from both classes
                mask = (y == class_i) | (y == class_j)
                X_pair = X[mask]
                y_pair = np.where(y[mask] == class_i, 1, -1).astype(float)

                # Train binary SVM
                svm = BinarySVM(C=self.C, kernel=self.kernel,
                               gamma=self.gamma, lr=self.lr,
                               n_iters=self.n_iters)
                svm.fit(X_pair, y_pair)

                self.classifiers[(class_i, class_j)] = svm

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))

        for (class_i, class_j), svm in self.classifiers.items():
            predictions = svm.predict(X)
            i_idx = np.where(self.classes == class_i)[0][0]
            j_idx = np.where(self.classes == class_j)[0][0]

            votes[predictions > 0, i_idx] += 1
            votes[predictions <= 0, j_idx] += 1

        # Return class with most votes
        return self.classes[np.argmax(votes, axis=1)]

    def decision_scores(self, X):
        """Get average decision function scores per class."""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        counts = np.zeros(len(self.classes))

        for (class_i, class_j), svm in self.classifiers.items():
            df = svm.decision_function(X)
            i_idx = np.where(self.classes == class_i)[0][0]
            j_idx = np.where(self.classes == class_j)[0][0]

            scores[:, i_idx] += df
            scores[:, j_idx] -= df
            counts[i_idx] += 1
            counts[j_idx] += 1

        counts[counts == 0] = 1
        return scores / counts


# =============================================================================
# Generate Realistic Market Regime Data
# =============================================================================

def generate_market_regime_data(n_days=2000, seed=42):
    """Generate synthetic market data with regime labels."""
    np.random.seed(seed)

    # Simulate regime sequences (block structure)
    regime_lengths = []
    regimes = []
    current = 0
    total = 0
    while total < n_days:
        length = np.random.randint(30, 180)  # regimes last 1-6 months
        if total + length > n_days:
            length = n_days - total
        regime_lengths.append(length)
        regimes.extend([current] * length)
        total += length
        current = (current + 1) % 3  # cycle through 0=bear, 1=sideways, 2=bull

    regimes = np.array(regimes[:n_days])

    # Generate features conditioned on regime
    features = {}
    n = n_days

    # Bear regime parameters
    bear_params = {'ret20': (-0.06, 0.03), 'ret60': (-0.15, 0.05),
                   'vol': (0.30, 0.08), 'vix': (30, 8), 'adx': (35, 10)}
    # Sideways regime parameters
    side_params = {'ret20': (0.00, 0.015), 'ret60': (0.01, 0.03),
                   'vol': (0.15, 0.03), 'vix': (16, 3), 'adx': (14, 5)}
    # Bull regime parameters
    bull_params = {'ret20': (0.04, 0.02), 'ret60': (0.12, 0.04),
                   'vol': (0.13, 0.03), 'vix': (14, 3), 'adx': (30, 8)}

    params_list = [bear_params, side_params, bull_params]

    # Generate features based on regime
    features['Return_20d'] = np.zeros(n)
    features['Return_60d'] = np.zeros(n)
    features['Volatility_20d'] = np.zeros(n)
    features['VIX_level'] = np.zeros(n)
    features['Trend_strength'] = np.zeros(n)

    for regime_id in range(3):
        mask = regimes == regime_id
        count = mask.sum()
        p = params_list[regime_id]
        features['Return_20d'][mask] = np.random.normal(p['ret20'][0], p['ret20'][1], count)
        features['Return_60d'][mask] = np.random.normal(p['ret60'][0], p['ret60'][1], count)
        features['Volatility_20d'][mask] = np.random.normal(p['vol'][0], p['vol'][1], count)
        features['VIX_level'][mask] = np.random.normal(p['vix'][0], p['vix'][1], count)
        features['Trend_strength'][mask] = np.random.normal(p['adx'][0], p['adx'][1], count)

    # Derived features
    features['Volatility_ratio'] = (features['Volatility_20d'] /
                                    np.maximum(np.convolve(features['Volatility_20d'],
                                              np.ones(60)/60, mode='same'), 0.01))
    features['VIX_change_10d'] = np.concatenate([[0]*10,
                                  np.diff(features['VIX_level'], n=1)[9:]])
    features['Breadth_advance'] = np.where(regimes == 2, np.random.normal(1.5, 0.3, n),
                                  np.where(regimes == 0, np.random.normal(0.6, 0.2, n),
                                           np.random.normal(1.0, 0.15, n)))
    features['Put_call_ratio'] = np.where(regimes == 2, np.random.normal(0.65, 0.1, n),
                                 np.where(regimes == 0, np.random.normal(1.1, 0.2, n),
                                          np.random.normal(0.85, 0.1, n)))
    features['SMA50_above_SMA200'] = np.where(regimes == 2,
                                     np.random.binomial(1, 0.85, n),
                                     np.where(regimes == 0,
                                              np.random.binomial(1, 0.2, n),
                                              np.random.binomial(1, 0.5, n))).astype(float)
    features['New_highs_ratio'] = np.where(regimes == 2,
                                  np.random.normal(0.7, 0.1, n),
                                  np.where(regimes == 0,
                                           np.random.normal(0.15, 0.1, n),
                                           np.random.normal(0.45, 0.15, n)))
    features['Credit_spread'] = np.where(regimes == 2,
                                np.random.normal(3.0, 0.5, n),
                                np.where(regimes == 0,
                                         np.random.normal(5.5, 1.0, n),
                                         np.random.normal(3.8, 0.6, n)))

    feature_names = list(features.keys())
    X = np.column_stack([features[f] for f in feature_names])
    y = regimes

    return X, y, feature_names


# =============================================================================
# Training and Evaluation
# =============================================================================

# Generate data
X, y, feature_names = generate_market_regime_data(n_days=2000)
regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}

print("=" * 60)
print("SVM Market Regime Classifier")
print("=" * 60)
print(f"\nDataset: {X.shape[0]} trading days, {X.shape[1]} features")
print(f"Regime distribution: {dict(Counter(y))}")
print(f"  Bear={np.sum(y==0)}, Sideways={np.sum(y==1)}, Bull={np.sum(y==2)}")

# Time-series split
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTrain: {len(X_train)} days | Test: {len(X_test)} days")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multi-class SVM
print("\nTraining SVM (One-vs-One, RBF kernel)...")
svm = MultiClassSVM(C=1.0, kernel='rbf', gamma='scale',
                    lr=0.0005, n_iters=300)
svm.fit(X_train_scaled, y_train)
print("Training complete.")

# Predictions
train_preds = svm.predict(X_train_scaled)
test_preds = svm.predict(X_test_scaled)

train_acc = np.mean(train_preds == y_train)
test_acc = np.mean(test_preds == y_test)

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# Per-class accuracy
print("\nPer-class accuracy (test):")
for cls in [0, 1, 2]:
    mask = y_test == cls
    if mask.sum() > 0:
        acc = np.mean(test_preds[mask] == cls)
        print(f"  {regime_names[cls]:>10}: {acc:.4f} ({mask.sum()} samples)")

# Confusion matrix
print("\nConfusion Matrix (Test):")
print(f"{'':>12} {'Pred Bear':>10} {'Pred Side':>10} {'Pred Bull':>10}")
for true_cls in [0, 1, 2]:
    row = []
    for pred_cls in [0, 1, 2]:
        count = np.sum((y_test == true_cls) & (test_preds == pred_cls))
        row.append(count)
    print(f"{'True '+regime_names[true_cls]:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

# Decision scores for confidence analysis
scores = svm.decision_scores(X_test_scaled)
confidence = np.max(scores, axis=1) - np.sort(scores, axis=1)[:, -2]

print("\n--- Confidence Analysis ---")
high_conf_mask = confidence > np.percentile(confidence, 75)
low_conf_mask = confidence <= np.percentile(confidence, 25)
print(f"High confidence accuracy: {np.mean(test_preds[high_conf_mask] == y_test[high_conf_mask]):.4f}")
print(f"Low confidence accuracy:  {np.mean(test_preds[low_conf_mask] == y_test[low_conf_mask]):.4f}")

# Simulated strategy performance
print("\n--- Strategy Simulation ---")
simulated_returns = np.random.normal(0.0003, 0.015, len(y_test))
# Bull => go long, Bear => go short, Sideways => small position
position_map = {2: 1.0, 0: -1.0, 1: 0.2}
positions = np.array([position_map[p] for p in test_preds])

# Scale by confidence
positions *= np.clip(confidence / np.percentile(confidence, 50), 0.3, 1.5)

strategy_ret = positions * simulated_returns
cum_strategy = np.cumprod(1 + strategy_ret)
cum_buyhold = np.cumprod(1 + simulated_returns)

print(f"Strategy return:   {(cum_strategy[-1]-1)*100:.2f}%")
print(f"Buy & Hold return: {(cum_buyhold[-1]-1)*100:.2f}%")
print(f"Strategy Sharpe:   {np.sqrt(252)*np.mean(strategy_ret)/np.std(strategy_ret):.2f}")
print(f"Number of binary classifiers: {len(svm.classifiers)}")

# Support vector analysis
total_svs = sum(len(c.support_vectors_idx) for c in svm.classifiers.values())
print(f"Total support vectors: {total_svs}")
```

## Key Takeaways

1. **SVM's maximum margin principle provides a geometrically motivated confidence measure** for regime classification. The distance from the decision boundary is a natural indicator of classification certainty, directly useful for scaling portfolio exposure during regime transitions.

2. **Feature scaling is non-negotiable for SVM.** Use rolling standardization (trailing z-scores) to avoid look-ahead bias while ensuring all features contribute equally to the distance calculations that drive the SVM decision boundary.

3. **The RBF kernel is the default choice for market regime classification** because the relationships between market indicators and regimes are inherently non-linear. However, always benchmark against a linear kernel -- if linear SVM performs comparably, the simpler model should be preferred.

4. **Multi-class regime classification via one-vs-one is effective for 3-5 states** but becomes combinatorially expensive for finer-grained classifications. Keep the number of market states small and well-defined.

5. **SVM does not model temporal dynamics.** Supplement the model with explicit temporal features (regime duration, trend persistence) and consider combining SVM with Hidden Markov Models or change-point detection for more sophisticated regime identification.

6. **Hyperparameter tuning (C, gamma) is critical and must use time-series cross-validation.** The optimal regularization depends on the noise level in your regime labels, which varies across different market eras.

7. **SVM works best as the regime classifier in a larger system** that includes strategy selection, position sizing (using margin distance), and risk management layers. The regime label itself is an input to downstream decision-making, not the final trading signal.
