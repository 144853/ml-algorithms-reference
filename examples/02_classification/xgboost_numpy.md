# XGBoost - Complete Guide with Stock Market Applications

## Overview

XGBoost (eXtreme Gradient Boosting) is an optimized implementation of gradient boosted decision trees designed for speed, performance, and scalability. Developed by Tianqi Chen in 2014, XGBoost has become the dominant algorithm in machine learning competitions and is widely adopted in quantitative finance for its superior predictive accuracy. Unlike Random Forest, which builds trees independently and averages their predictions, XGBoost builds trees sequentially where each new tree corrects the errors made by the previous ensemble, progressively refining the model's predictions.

In the stock market context, XGBoost is particularly suited for high-frequency trading (HFT) signal classification where extracting every fraction of a percent of predictive accuracy matters. The algorithm's ability to handle complex nonlinear interactions among hundreds of features -- from order book dynamics and tick-level price changes to microstructure indicators -- makes it a go-to choice for prop trading firms and HFT desks. Its built-in regularization (L1 and L2) prevents overfitting on noisy tick data, while its speed allows rapid retraining as market microstructure evolves.

What sets XGBoost apart is its second-order Taylor expansion approximation of the loss function, which provides more precise gradient information for tree construction. Combined with column subsampling, shrinkage (learning rate), and sophisticated handling of missing values, XGBoost achieves state-of-the-art classification performance on structured financial data. It also supports GPU acceleration and distributed training for large-scale applications.

## How It Works - The Math Behind It

### Gradient Boosting Framework

The model is an additive ensemble of K trees:
```
y_hat_i = SUM(k=1 to K) f_k(x_i)
```

Each tree f_k is added greedily to minimize the objective:
```
Obj = SUM(i=1 to n) L(y_i, y_hat_i) + SUM(k=1 to K) Omega(f_k)
```

Where L is the loss function and Omega is the regularization term.

### Second-Order Approximation (Key Innovation)

At step t, adding tree f_t, the objective is approximated using a second-order Taylor expansion:
```
Obj^(t) ~ SUM(i=1 to n) [g_i * f_t(x_i) + 0.5 * h_i * f_t(x_i)^2] + Omega(f_t)
```

Where:
- `g_i = dL/dy_hat` (first-order gradient) = y_hat - y_i for squared loss
- `h_i = d^2L/dy_hat^2` (second-order gradient, Hessian) = 1 for squared loss

For log-loss (classification):
- `g_i = p_i - y_i` where p_i = sigmoid(y_hat_i)
- `h_i = p_i * (1 - p_i)`

### Tree Structure Regularization

```
Omega(f) = gamma * T + 0.5 * lambda * SUM(j=1 to T) w_j^2
```

Where:
- T = number of leaves in the tree
- w_j = weight (prediction value) of leaf j
- gamma = minimum loss reduction for a split (complexity penalty)
- lambda = L2 regularization on leaf weights

### Optimal Leaf Weight

For a given tree structure, the optimal weight for leaf j is:
```
w_j* = -SUM(i in I_j) g_i / (SUM(i in I_j) h_i + lambda)
```

Where I_j is the set of data points in leaf j.

### Split Gain Formula

The gain from splitting a leaf into left and right children:
```
Gain = 0.5 * [G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
```

Where G_L, H_L are sums of gradients and Hessians for the left child, and similarly for right.

A split is only made if Gain > 0 (i.e., improvement exceeds the complexity penalty gamma).

### Learning Rate (Shrinkage)

Each new tree's contribution is shrunk by a factor eta:
```
y_hat^(t) = y_hat^(t-1) + eta * f_t(x)
```

Typical values: eta = 0.01 to 0.3. Lower values require more trees but produce better generalization.

### Step-by-Step Process

1. **Initialize predictions** (e.g., log-odds of base rate for classification)
2. **For each boosting round t = 1 to T**:
   a. Compute gradients g_i and Hessians h_i for each sample
   b. Build a regression tree to fit the gradients using the gain formula
   c. Compute optimal leaf weights w_j*
   d. Update predictions: y_hat += eta * f_t(x)
3. **Final prediction**: Sum of all tree outputs passed through sigmoid for probability

## Stock Market Use Case: High-Frequency Trading Signal Classification

### The Problem

You operate an HFT desk at a proprietary trading firm executing thousands of trades per day on NASDAQ-listed equities. For each stock at each decision point (every 100 milliseconds), you need to classify the next short-term price movement as one of: "Up" (price increases by more than 1 tick), "Down" (price decreases by more than 1 tick), or "Flat" (price stays within 1 tick). Features include order book microstructure data, trade flow indicators, and intraday technical signals. The model must be highly accurate because even a 0.1% improvement in signal quality translates to millions in annual PnL at HFT scale.

### Stock Market Features (Input Data)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| Bid_Ask_Spread | Current bid-ask spread (bps) | Numerical | 1 - 50 |
| Order_Imbalance | (Bid_vol - Ask_vol) / Total_vol | Numerical | -1 to 1 |
| Trade_Imbalance | Net buy volume / Total volume (1min) | Numerical | -1 to 1 |
| Price_Momentum_1s | 1-second price change (bps) | Numerical | -20 to 20 |
| Price_Momentum_10s | 10-second price change (bps) | Numerical | -50 to 50 |
| Price_Momentum_60s | 60-second price change (bps) | Numerical | -100 to 100 |
| Volume_Intensity | Current volume / avg volume ratio | Numerical | 0.1 to 10 |
| VWAP_Distance | Price distance from VWAP (bps) | Numerical | -200 to 200 |
| Depth_Ratio | Top 5 bid depth / Top 5 ask depth | Numerical | 0.2 to 5 |
| Trade_Size_Avg | Average trade size (last 50 trades) | Numerical | 100 - 10000 |
| Tick_Direction | Last 5 tick directions encoded | Numerical | -5 to 5 |
| Realized_Vol_1min | 1-minute realized volatility (bps) | Numerical | 1 to 100 |
| Spread_Change | Change in spread (last 10s) | Numerical | -10 to 10 |
| Queue_Position | Estimated queue position at best bid | Numerical | 0 to 50000 |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_ticks = 50000  # 50,000 decision points in a trading day

hft_features = {
    'Bid_Ask_Spread': np.random.exponential(5, n_ticks) + 1,
    'Order_Imbalance': np.random.uniform(-0.8, 0.8, n_ticks),
    'Trade_Imbalance': np.random.normal(0, 0.3, n_ticks),
    'Price_Mom_1s': np.random.normal(0, 3, n_ticks),
    'Price_Mom_10s': np.random.normal(0, 8, n_ticks),
    'Price_Mom_60s': np.random.normal(0, 20, n_ticks),
    'Volume_Intensity': np.random.lognormal(0, 0.5, n_ticks),
    'VWAP_Distance': np.random.normal(0, 40, n_ticks),
    'Depth_Ratio': np.random.lognormal(0, 0.4, n_ticks),
    'Trade_Size_Avg': np.random.lognormal(6, 0.8, n_ticks),
    'Tick_Direction': np.random.normal(0, 1.5, n_ticks),
    'Realized_Vol_1min': np.random.exponential(15, n_ticks) + 2,
    'Spread_Change': np.random.normal(0, 2, n_ticks),
    'Queue_Position': np.random.exponential(5000, n_ticks),
}

X = np.column_stack(list(hft_features.values()))
feature_names = list(hft_features.keys())

# Generate labels: Up (1), Down (0) based on microstructure signals
signal = (0.3 * hft_features['Order_Imbalance'] +
          0.25 * hft_features['Trade_Imbalance'] +
          0.05 * hft_features['Price_Mom_1s'] +
          0.02 * hft_features['Price_Mom_10s'] +
          0.15 * hft_features['Tick_Direction'] +
          -0.01 * hft_features['Bid_Ask_Spread'] +
          0.08 * np.log(hft_features['Depth_Ratio']) +
          np.random.normal(0, 0.3, n_ticks))

y = (signal > 0).astype(int)

print(f"HFT Decision Points: {n_ticks}")
print(f"Up signals: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Down signals: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
```

### The Model in Action

```python
# XGBoost builds trees sequentially, each correcting previous errors:
#
# Round 1: Tree focuses on Order_Imbalance (strongest signal)
#   -> Residual errors are large where Order_Imbalance alone is insufficient
#
# Round 2: Tree focuses on Trade_Imbalance to correct Round 1 errors
#   -> Combined model captures two aspects of order flow
#
# Round 50: Tree finds subtle interaction: Depth_Ratio * Spread_Change
#   -> Captures liquidity withdrawal pattern preceding price drops
#
# Round 200: Tree learns that high Volume_Intensity + positive Order_Imbalance
#             is more predictive than either alone
#
# Final: 300 trees combined with learning_rate=0.05
#   -> Tick-level prediction: P(Up) = 0.63 => BUY
#   -> Expected profit: 0.63 * 1 tick - 0.37 * 1 tick = 0.26 ticks per trade
```

## Advantages

1. **Superior Predictive Accuracy**: XGBoost consistently achieves the highest accuracy among tree-based methods for structured financial data. In HFT, even 0.5% accuracy improvement translates to significant PnL improvement across millions of trades.

2. **Built-in Regularization Prevents Overfitting**: The L1 (alpha) and L2 (lambda) regularization on leaf weights, combined with the complexity penalty (gamma), effectively prevents the model from memorizing noise in tick-level data, which has an extremely low signal-to-noise ratio.

3. **Second-Order Gradient Information**: Using both first and second derivatives of the loss function allows XGBoost to make more informed splits than standard gradient boosting, resulting in better convergence and more accurate tree structures.

4. **Handles Missing Values Natively**: XGBoost learns the optimal direction for missing values at each split. In HFT, missing data occurs frequently (delayed feeds, gaps in order book data), and this automatic handling eliminates the need for imputation.

5. **Column Subsampling Reduces Overfitting and Training Time**: Like Random Forest, XGBoost can sample features at each tree or each level, reducing correlation between trees and preventing any single microstructure feature from dominating.

6. **GPU Acceleration for Rapid Retraining**: XGBoost supports GPU training, enabling HFT models to be retrained intraday as market microstructure evolves. A model can be retrained on 100K ticks in seconds on a modern GPU.

7. **Early Stopping Prevents Overshoot**: The model can monitor validation loss and stop adding trees when performance starts to degrade, automatically selecting the optimal number of boosting rounds.

8. **Feature Importance with Multiple Methods**: XGBoost provides gain-based, cover-based, and frequency-based importance metrics, giving a comprehensive view of which microstructure features drive predictions.

## Disadvantages

1. **Extensive Hyperparameter Tuning Required**: XGBoost has many interacting hyperparameters (learning rate, max depth, min child weight, gamma, lambda, alpha, subsample, colsample). Finding optimal settings for HFT data requires expensive grid search or Bayesian optimization.

2. **Sequential Training Cannot Be Fully Parallelized**: While individual tree construction is parallelized, trees must be built sequentially because each tree depends on the residuals from all previous trees. This makes training slower than Random Forest for the same number of trees.

3. **Prone to Overfitting Without Careful Regularization**: The sequential error-correction nature means XGBoost can fit noise if not properly regularized. In HFT, where noise vastly exceeds signal, this requires aggressive regularization settings.

4. **Poor Probability Calibration**: XGBoost's raw probability outputs are often poorly calibrated. A predicted 0.7 probability may not correspond to a true 70% accuracy. This requires post-hoc calibration (Platt scaling or isotonic regression) for reliable position sizing.

5. **Sensitive to Feature Scale in Some Configurations**: While tree-based methods are generally scale-invariant, some XGBoost configurations (especially with L1 regularization) can be sensitive to feature scales, requiring standardization.

6. **Computationally Expensive for Real-Time Prediction at Ultra-Low Latency**: While faster than neural networks, XGBoost's prediction (traversing hundreds of trees) is slower than linear models. For sub-microsecond HFT, the model may need to be distilled into simpler structures.

7. **Difficult to Interpret Individual Predictions**: While feature importance is available at the model level, explaining why the model predicted "Up" for a specific tick is complex. SHAP values help but add computational overhead.

## When to Use in Stock Market

- When maximum predictive accuracy is the priority and you can invest time in hyperparameter tuning
- For medium-frequency to high-frequency trading where the feature set is well-defined and structured
- When the dataset is large enough (10,000+ observations) to benefit from boosting
- For capturing complex feature interactions in order book and microstructure data
- When built-in missing value handling is needed for messy real-time market data
- As the final model in a research pipeline after feature engineering with simpler models

## When NOT to Use in Stock Market

- For ultra-low latency (sub-microsecond) prediction where linear models are needed
- When full model interpretability is required for regulatory compliance
- With very small datasets (fewer than 1,000 observations) where overfitting risk is high
- When quick prototyping is needed without extensive hyperparameter tuning
- For online learning scenarios requiring continuous model updates (XGBoost is batch-oriented)
- When probability calibration is critical without post-processing (use Logistic Regression instead)

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|---------------------------|
| n_estimators | Number of boosting rounds | 100 - 5000 | 300-1000 with early stopping |
| learning_rate (eta) | Shrinkage factor per tree | 0.01 - 0.3 | 0.03-0.1; lower = more trees needed |
| max_depth | Maximum tree depth | 3 - 10 | 4-6 for HFT; 3-4 for daily signals |
| min_child_weight | Min Hessian sum in child | 1 - 100 | 10-50 for noisy financial data |
| gamma | Min loss reduction for split | 0 - 5 | 0.1-1.0 to prevent noise-fitting |
| lambda (reg_lambda) | L2 regularization | 0 - 10 | 1-5 for HFT; prevents extreme leaf weights |
| alpha (reg_alpha) | L1 regularization | 0 - 5 | 0.1-1.0 for sparse feature selection |
| subsample | Row sampling ratio | 0.5 - 1.0 | 0.7-0.8 reduces overfitting |
| colsample_bytree | Column sampling per tree | 0.5 - 1.0 | 0.6-0.8 decorrelates trees |
| scale_pos_weight | Class imbalance handling | 1 - 10 | sum(neg)/sum(pos) for imbalanced signals |

## Stock Market Performance Tips

1. **Use Early Stopping Religiously**: Set aside 15-20% of training data for validation and stop training when validation loss has not improved for 50-100 rounds. This is the single most effective way to prevent overfitting on financial data.

2. **Start with Low Learning Rate and Many Trees**: Begin with eta=0.05 and 1000+ trees with early stopping. This produces more stable models than high learning rate with few trees.

3. **Tune Regularization Before Tree Structure**: First find good values for lambda, alpha, and gamma, then optimize max_depth and min_child_weight. Regularization has a larger impact on generalization for noisy HFT data.

4. **Use Time-Based Validation Splits**: Never use random train/test splits for financial data. Use temporal splits to simulate real trading conditions and avoid lookahead bias.

5. **Calibrate Probabilities Post-Training**: Apply Platt scaling or isotonic regression to XGBoost's probability outputs before using them for position sizing.

6. **Monitor Feature Importance Stability**: Track which features are important across retraining periods. Sudden changes in feature importance may signal market regime shifts.

## Comparison with Other Algorithms

| Criteria | XGBoost | LightGBM | Random Forest | CatBoost | Logistic Regression |
|----------|---------|----------|---------------|----------|-------------------|
| Accuracy | Very High | Very High | High | Very High | Moderate |
| Training Speed | Moderate | Fast | Moderate | Slow (initial) | Very Fast |
| Memory Usage | Moderate | Low | High | Moderate | Very Low |
| Categorical Features | Needs encoding | Needs encoding | Needs encoding | Native support | Needs encoding |
| Regularization | Comprehensive | Comprehensive | Implicit (averaging) | Comprehensive | L1/L2 |
| GPU Support | Yes | Yes | Limited | Yes | No |
| Missing Values | Native | Native | Some implementations | Native | No |
| Best For (Stock Market) | HFT signals | Large-scale | Robust baseline | Mixed data types | Simple signals |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Simplified XGBoost from Scratch for HFT Signal Classification
# ============================================================

class XGBTree:
    """Single regression tree for gradient boosting."""

    def __init__(self, max_depth=4, min_child_weight=10, gamma=0.1, lambda_reg=1.0,
                 colsample=0.8):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.colsample = colsample
        self.tree = None

    def _compute_leaf_weight(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.lambda_reg)

    def _compute_gain(self, g_left, h_left, g_right, h_right):
        def score(g, h):
            return (np.sum(g) ** 2) / (np.sum(h) + self.lambda_reg)
        gain = 0.5 * (score(g_left, h_left) + score(g_right, h_right) -
                       score(np.concatenate([g_left, g_right]),
                             np.concatenate([h_left, h_right]))) - self.gamma
        return gain

    def _best_split(self, X, g, h, feature_indices):
        n_samples = len(g)
        best_gain = 0
        best_feat = None
        best_thresh = None

        for feat in feature_indices:
            sorted_idx = np.argsort(X[:, feat])
            sorted_g = g[sorted_idx]
            sorted_h = h[sorted_idx]
            sorted_vals = X[sorted_idx, feat]

            # Scan through sorted values
            g_left_sum, h_left_sum = 0.0, 0.0
            g_total, h_total = np.sum(g), np.sum(h)

            for i in range(self.min_child_weight, n_samples - self.min_child_weight):
                g_left_sum += sorted_g[i-1]
                h_left_sum += sorted_h[i-1]

                if sorted_vals[i] == sorted_vals[i-1]:
                    continue

                g_right_sum = g_total - g_left_sum
                h_right_sum = h_total - h_left_sum

                if h_left_sum < self.min_child_weight or h_right_sum < self.min_child_weight:
                    continue

                gain = 0.5 * ((g_left_sum**2 / (h_left_sum + self.lambda_reg)) +
                              (g_right_sum**2 / (h_right_sum + self.lambda_reg)) -
                              (g_total**2 / (h_total + self.lambda_reg))) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = (sorted_vals[i] + sorted_vals[i-1]) / 2

        return best_feat, best_thresh, best_gain

    def _build(self, X, g, h, depth):
        n_features = X.shape[1]
        n_selected = max(1, int(n_features * self.colsample))
        feature_indices = np.random.choice(n_features, n_selected, replace=False)

        weight = self._compute_leaf_weight(g, h)

        if depth >= self.max_depth or len(g) < 2 * self.min_child_weight:
            return {'leaf': True, 'weight': weight}

        feat, thresh, gain = self._best_split(X, g, h, feature_indices)

        if feat is None:
            return {'leaf': True, 'weight': weight}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature': feat,
            'threshold': thresh,
            'gain': gain,
            'left': self._build(X[left_mask], g[left_mask], h[left_mask], depth + 1),
            'right': self._build(X[right_mask], g[right_mask], h[right_mask], depth + 1),
        }

    def fit(self, X, g, h):
        self.tree = self._build(X, g, h, 0)
        return self

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['weight']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])


class StockXGBoost:
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=4,
                 min_child_weight=10, gamma=0.1, lambda_reg=1.0, subsample=0.8,
                 colsample=0.8):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.subsample = subsample
        self.colsample = colsample
        self.trees = []
        self.base_score = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _log_loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=50):
        n_samples = len(y)
        self.base_score = np.log(np.mean(y) / (1 - np.mean(y)))
        y_hat = np.full(n_samples, self.base_score)

        if X_val is not None:
            y_hat_val = np.full(len(y_val), self.base_score)
            best_val_loss = float('inf')
            best_round = 0

        self.trees = []

        for t in range(self.n_estimators):
            # Compute gradients and Hessians
            p = self._sigmoid(y_hat)
            g = p - y          # gradient of log-loss
            h = p * (1 - p)    # Hessian of log-loss

            # Subsample rows
            n_sub = max(1, int(n_samples * self.subsample))
            sub_idx = np.random.choice(n_samples, n_sub, replace=False)

            tree = XGBTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                lambda_reg=self.lambda_reg,
                colsample=self.colsample
            )
            tree.fit(X[sub_idx], g[sub_idx], h[sub_idx])
            self.trees.append(tree)

            # Update predictions
            y_hat += self.lr * tree.predict(X)

            # Compute training loss
            train_loss = self._log_loss(y, self._sigmoid(y_hat))

            # Validation and early stopping
            if X_val is not None:
                y_hat_val += self.lr * tree.predict(X_val)
                val_loss = self._log_loss(y_val, self._sigmoid(y_hat_val))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_round = t

                if (t + 1) % 50 == 0:
                    print(f"Round {t+1}: Train Loss={train_loss:.4f}, "
                          f"Val Loss={val_loss:.4f}")

                if t - best_round >= early_stopping_rounds:
                    print(f"Early stopping at round {t+1}. Best round: {best_round+1}")
                    self.trees = self.trees[:best_round + 1]
                    break
            else:
                if (t + 1) % 50 == 0:
                    acc = np.mean((self._sigmoid(y_hat) >= 0.5) == y)
                    print(f"Round {t+1}: Loss={train_loss:.4f}, Accuracy={acc:.4f}")

        return self

    def predict_proba(self, X):
        y_hat = np.full(len(X), self.base_score)
        for tree in self.trees:
            y_hat += self.lr * tree.predict(X)
        return self._sigmoid(y_hat)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# Generate Synthetic HFT Data
# ============================================================
np.random.seed(42)
n_ticks = 20000

order_imbalance = np.random.uniform(-0.8, 0.8, n_ticks)
trade_imbalance = np.random.normal(0, 0.3, n_ticks)
price_mom_1s = np.random.normal(0, 3, n_ticks)
price_mom_10s = np.random.normal(0, 8, n_ticks)
price_mom_60s = np.random.normal(0, 20, n_ticks)
volume_intensity = np.random.lognormal(0, 0.5, n_ticks)
vwap_dist = np.random.normal(0, 40, n_ticks)
depth_ratio = np.random.lognormal(0, 0.35, n_ticks)
spread = np.random.exponential(5, n_ticks) + 1
tick_dir = np.random.normal(0, 1.5, n_ticks)
real_vol = np.random.exponential(15, n_ticks) + 2
spread_change = np.random.normal(0, 2, n_ticks)
trade_size = np.random.lognormal(6, 0.7, n_ticks)
queue_pos = np.random.exponential(5000, n_ticks)

X = np.column_stack([order_imbalance, trade_imbalance, price_mom_1s, price_mom_10s,
                     price_mom_60s, volume_intensity, vwap_dist, depth_ratio,
                     spread, tick_dir, real_vol, spread_change, trade_size, queue_pos])

feature_names = ['Order_Imbalance', 'Trade_Imbalance', 'Price_Mom_1s', 'Price_Mom_10s',
                 'Price_Mom_60s', 'Volume_Intensity', 'VWAP_Distance', 'Depth_Ratio',
                 'Spread', 'Tick_Direction', 'Realized_Vol', 'Spread_Change',
                 'Trade_Size', 'Queue_Position']

# HFT signal with interactions
signal = (0.35 * order_imbalance +
          0.25 * trade_imbalance +
          0.04 * price_mom_1s +
          0.15 * tick_dir +
          0.10 * np.log(depth_ratio) +
          -0.008 * spread +
          0.05 * order_imbalance * trade_imbalance +  # interaction
          np.random.normal(0, 0.35, n_ticks))

y = (signal > 0).astype(int)

print(f"HFT Dataset: {n_ticks} ticks")
print(f"Up: {y.sum()} ({y.mean()*100:.1f}%) | Down: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

# Temporal split
train_end = 14000
val_end = 17000
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"Train: {train_end} | Val: {val_end-train_end} | Test: {n_ticks-val_end}")

# ============================================================
# Train XGBoost
# ============================================================
print("\n=== Training XGBoost ===")
xgb = StockXGBoost(n_estimators=500, learning_rate=0.05, max_depth=4,
                    min_child_weight=15, gamma=0.2, lambda_reg=2.0,
                    subsample=0.8, colsample=0.7)
xgb.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)

# ============================================================
# Evaluate
# ============================================================
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)

accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)

print(f"\n=== HFT TEST PERFORMANCE ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Trees used: {len(xgb.trees)}")

# HFT PnL simulation
print(f"\n=== HFT PnL SIMULATION ===")
tick_value = 0.01  # $0.01 per tick
position_size = 1000  # shares
correct_trades = np.sum(y_pred == y_test)
incorrect_trades = np.sum(y_pred != y_test)
net_ticks = correct_trades - incorrect_trades
pnl = net_ticks * tick_value * position_size

print(f"Total decisions: {len(y_test)}")
print(f"Correct: {correct_trades} | Incorrect: {incorrect_trades}")
print(f"Net ticks: {net_ticks}")
print(f"Estimated PnL: ${pnl:,.2f}")
print(f"Win rate: {correct_trades/len(y_test)*100:.1f}%")

# Sample predictions
print(f"\n=== SAMPLE TICK PREDICTIONS ===")
print(f"{'Tick':>6} {'P(Up)':>8} {'Signal':>8} {'Actual':>8}")
for i in range(10):
    sig = "UP" if y_pred[i] == 1 else "DOWN"
    act = "UP" if y_test[i] == 1 else "DOWN"
    print(f"{i+1:>6} {y_proba[i]:>8.4f} {sig:>8} {act:>8}")
```

## Key Takeaways

- XGBoost builds trees sequentially where each tree corrects errors of the previous ensemble, achieving superior accuracy on structured financial data
- The second-order gradient approximation provides more precise tree construction than standard gradient boosting, leading to better convergence
- Built-in L1/L2 regularization and the gamma complexity penalty are essential for preventing overfitting on noisy HFT data
- Early stopping with a temporal validation set is the most important technique for practical XGBoost deployment in trading
- Learning rate (eta) and number of trees have an inverse relationship: lower learning rate requires more trees but generally produces better generalization
- XGBoost naturally handles missing values by learning optimal split directions, crucial for real-time market data with feed gaps
- Probability calibration (Platt scaling) should be applied post-training if the probability outputs are used for position sizing
- For HFT applications, the model may need to be distilled into a simpler structure for ultra-low-latency production deployment
