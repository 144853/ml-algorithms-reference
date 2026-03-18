# CatBoost - Complete Guide with Stock Market Applications

## Overview

CatBoost (Categorical Boosting), developed by Yandex in 2017, is a gradient boosting library that excels at handling datasets containing categorical features alongside numerical ones. Its key innovation is the Ordered Target Statistics method for encoding categorical variables, which avoids the target leakage problem inherent in traditional target encoding approaches. In the stock market context, CatBoost is ideal for classification tasks involving mixed feature types -- such as combining numerical technical indicators (RSI, MACD, volume) with categorical attributes (sector, exchange, market cap category, analyst rating).

The algorithm introduces Ordered Boosting, which uses a permutation-driven approach to compute unbiased gradient estimates. Traditional gradient boosting computes gradients on the same data used for tree construction, leading to a subtle form of overfitting called prediction shift. CatBoost addresses this by using different permutations of the data to compute gradient estimates and build trees, resulting in more generalizable models. For stock classification, this means CatBoost produces more reliable out-of-sample predictions, which is critical when real money is at stake.

CatBoost also grows symmetric (oblivious) decision trees where the same feature and threshold is used at each level, producing balanced trees that are highly regularized by design. This architectural choice, combined with ordered boosting and native categorical handling, makes CatBoost particularly robust for financial applications where overfitting on noisy data is the primary concern.

## How It Works - The Math Behind It

### Ordered Target Statistics (Categorical Encoding)

For a categorical feature with value c, the target statistic is computed using only preceding observations in a random permutation:

```
TS(x_i) = (SUM(j<i, x_j=c) y_j + prior * count_prior) / (count(j<i, x_j=c) + count_prior)
```

Where:
- Only observations before index i in the permutation are used (prevents target leakage)
- `prior` = global average of the target variable
- `count_prior` = smoothing parameter (typically 1)

For stock classification: encoding "Sector=Technology" uses only past observations of Technology stocks to compute the average label, preventing lookahead bias.

### Ordered Boosting

Traditional boosting computes residuals r_i on the same data used to train the tree. CatBoost uses ordered boosting:

```
1. Generate a random permutation sigma of training data
2. For observation i (in permutation order):
   - Compute model prediction using ONLY trees trained on observations {sigma(1), ..., sigma(i-1)}
   - Compute residual: r_i = y_i - model_prediction_i
3. Build next tree using these unbiased residuals
```

This prevents the model from "seeing" its own predictions when computing gradients, eliminating prediction shift.

### Symmetric (Oblivious) Decision Trees

CatBoost uses oblivious trees where the same split condition is applied at each level:

```
Level 0: Split on Feature A at threshold t1
Level 1: Split on Feature B at threshold t2  (same split for BOTH children of level 0)
Level 2: Split on Feature C at threshold t3  (same split for ALL children of level 1)

Result: 2^depth leaves, each defined by a unique combination of conditions
```

For a depth-6 tree: 64 leaves, each checking 6 conditions. This is highly regularized because the number of unique split conditions equals the tree depth (not exponential).

### Gradient Computation

Same as standard gradient boosting with log-loss for classification:
```
g_i = p_i - y_i           (first derivative)
h_i = p_i * (1 - p_i)     (second derivative)
```

But computed using ordered (unbiased) predictions.

### Leaf Value Computation

```
w_leaf = -SUM(g_i in leaf) / (SUM(h_i in leaf) + lambda)
```

With L2 regularization lambda on leaf weights.

### Step-by-Step Process

1. **Encode categorical features** using ordered target statistics with random permutations
2. **Initialize predictions** to the prior (log-odds of positive class)
3. **For each boosting round**:
   a. Compute ordered residuals using a random permutation
   b. Build an oblivious (symmetric) tree of fixed depth
   c. At each level, find the single best feature-threshold pair across all nodes at that level
   d. Compute leaf weights with L2 regularization
   e. Update predictions with learning rate shrinkage
4. **Repeat** until early stopping or max iterations

## Stock Market Use Case: Classifying Stocks with Mixed Categorical and Numerical Features

### The Problem

You are building a cross-sectional stock selection model that ranks stocks daily for a long/short equity portfolio. Unlike pure technical analysis, your model incorporates fundamental, categorical, and technical features. Each stock has categorical attributes (sector, exchange, market cap category, analyst consensus) alongside numerical indicators (PE ratio, RSI, momentum). The challenge is that many traditional ML models require one-hot encoding of categorical features, which creates sparse, high-dimensional data and loses the ordinal/grouping information. CatBoost handles these mixed feature types natively, learning optimal encodings from the target variable.

### Stock Market Features (Input Data)

| Feature | Description | Type | Example Values |
|---------|-------------|------|---------------|
| Sector | GICS sector classification | Categorical | Technology, Healthcare, Financials, Energy, ... |
| Exchange | Stock exchange | Categorical | NYSE, NASDAQ, AMEX |
| Market_Cap_Cat | Market cap category | Categorical | Mega, Large, Mid, Small, Micro |
| Analyst_Rating | Consensus rating | Categorical | Strong_Buy, Buy, Hold, Sell, Strong_Sell |
| Country | Domicile country | Categorical | US, UK, JP, DE, CN, ... |
| RSI_14 | Relative Strength Index | Numerical | 0 - 100 |
| PE_Ratio | Price-to-Earnings ratio | Numerical | -50 to 200 |
| PB_Ratio | Price-to-Book ratio | Numerical | 0.5 to 30 |
| Dividend_Yield | Annual dividend yield (%) | Numerical | 0 to 12 |
| Momentum_6M | 6-month price momentum (%) | Numerical | -50% to 100% |
| Volatility_60D | 60-day annualized volatility (%) | Numerical | 5% to 100% |
| Earnings_Growth | YoY earnings growth (%) | Numerical | -100% to 200% |
| Revenue_Surprise | Last revenue surprise (%) | Numerical | -20% to 20% |
| Short_Interest | Short interest ratio (%) | Numerical | 0 to 40 |
| Institutional_Own | Institutional ownership (%) | Numerical | 0 to 100 |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)
n_stocks = 5000

# Categorical features (encoded as integers for numpy)
sectors = np.random.randint(0, 11, n_stocks)  # 11 GICS sectors
exchanges = np.random.randint(0, 3, n_stocks)  # NYSE, NASDAQ, AMEX
cap_cat = np.random.randint(0, 5, n_stocks)    # Mega, Large, Mid, Small, Micro
analyst_rating = np.random.randint(0, 5, n_stocks)  # Strong Buy to Strong Sell
country = np.random.randint(0, 10, n_stocks)   # 10 countries

# Numerical features
rsi = np.random.uniform(15, 85, n_stocks)
pe_ratio = np.random.normal(20, 10, n_stocks)
pb_ratio = np.random.lognormal(1, 0.6, n_stocks)
div_yield = np.random.exponential(2, n_stocks)
momentum_6m = np.random.normal(5, 20, n_stocks)
volatility = np.random.lognormal(2.8, 0.4, n_stocks)
earnings_growth = np.random.normal(10, 30, n_stocks)
rev_surprise = np.random.normal(1, 4, n_stocks)
short_interest = np.random.exponential(5, n_stocks)
inst_own = np.random.uniform(20, 95, n_stocks)

X = np.column_stack([sectors, exchanges, cap_cat, analyst_rating, country,
                     rsi, pe_ratio, pb_ratio, div_yield, momentum_6m,
                     volatility, earnings_growth, rev_surprise, short_interest, inst_own])

cat_features = [0, 1, 2, 3, 4]  # Indices of categorical features

feature_names = ['Sector', 'Exchange', 'Cap_Cat', 'Analyst_Rating', 'Country',
                 'RSI_14', 'PE_Ratio', 'PB_Ratio', 'Div_Yield', 'Momentum_6M',
                 'Volatility_60D', 'Earnings_Growth', 'Rev_Surprise',
                 'Short_Interest', 'Inst_Ownership']

# Label: 1 = Outperform (buy), 0 = Underperform (avoid)
signal = (0.02 * momentum_6m +
          0.01 * (rsi - 50) +
          -0.01 * pe_ratio +
          0.03 * earnings_growth / 10 +
          0.05 * rev_surprise +
          -0.02 * short_interest +
          0.5 * (analyst_rating < 2).astype(float) +  # Strong Buy/Buy boost
          -0.3 * (sectors == 8).astype(float) +  # Sector effect (e.g., Utilities drag)
          0.4 * (cap_cat <= 1).astype(float) +  # Large/Mega cap advantage
          np.random.normal(0, 0.5, n_stocks))

y = (signal > np.median(signal)).astype(int)

print(f"Stocks: {n_stocks}")
print(f"Outperform: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Underperform: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
print(f"Categorical features: {len(cat_features)}")
print(f"Numerical features: {X.shape[1] - len(cat_features)}")
```

### The Model in Action

```python
# CatBoost with ordered target statistics:
#
# Categorical encoding for "Sector=Technology":
#   - In permutation order, among previous Technology stocks:
#     80 outperformed, 60 underperformed
#   - TS = (80 + 0.5 * 1) / (140 + 1) = 0.571
#   - Technology gets a slight positive encoding
#
# Oblivious tree (depth 4):
#   Level 0: Momentum_6M > 12.5?
#   Level 1: Analyst_Rating_encoded > 0.55?
#   Level 2: PE_Ratio > 25.3?
#   Level 3: Sector_encoded > 0.48?
#
#   Leaf: Momentum>12.5, Good analyst rating, PE<25.3, Good sector
#   => OUTPERFORM (confidence: 78%)
#
# Key advantage: Sector and Analyst_Rating are encoded optimally
# without losing information from one-hot encoding
```

## Advantages

1. **Native Categorical Feature Handling Preserves Information**: One-hot encoding "Sector" into 11 binary columns loses the grouping structure and inflates dimensionality. CatBoost's ordered target statistics create a single informative encoding that captures each sector's predictive relationship with stock performance.

2. **Ordered Boosting Eliminates Prediction Shift**: Traditional gradient boosting overfits by computing gradients on the same data used for tree construction. CatBoost's ordered approach produces truly unbiased gradient estimates, leading to better out-of-sample performance -- critical when trading real capital.

3. **Symmetric Trees Provide Strong Regularization**: Oblivious trees are inherently more regularized than standard trees because the number of unique split conditions equals the depth (6 splits for a depth-6 tree with 64 leaves). This prevents the model from creating overly specific paths for individual stocks.

4. **Robust Default Hyperparameters**: CatBoost works well out of the box with minimal tuning. For stock classification with mixed features, the default settings often produce competitive results, saving significant research time.

5. **Handles High-Cardinality Categoricals Efficiently**: Features like "Country" (100+ values) or "Sub-Industry" (150+ GICS classifications) are handled natively without creating sparse one-hot vectors that overwhelm the model.

6. **Built-in Overfitting Detection**: CatBoost includes automatic overfitting detection that monitors validation loss and stops training optimally, without requiring manual early stopping configuration.

7. **GPU Training Support for Large Cross-Sections**: When classifying thousands of stocks daily, GPU acceleration enables training in seconds, making it feasible to retrain models frequently.

## Disadvantages

1. **Slower Training Than LightGBM**: The ordered boosting procedure requires multiple passes through the data with different permutations, making CatBoost 2-5x slower than LightGBM for the same number of trees. For very large datasets, this may be a practical constraint.

2. **Symmetric Trees Limit Model Expressiveness**: The requirement that all nodes at a level use the same split means the tree cannot create asymmetric paths. If Technology stocks need different rules than Healthcare stocks, the symmetric tree must waste depth levels to separate them before applying sector-specific rules.

3. **Ordered Target Statistics Are Sensitive to Permutation Order**: The categorical encoding depends on the random permutation, introducing variance. While multiple permutations are averaged, the encoding for rare categories (e.g., a sector with few stocks) may be unreliable.

4. **Memory Intensive During Training**: Storing multiple permutations and intermediate predictions for ordered boosting requires significantly more memory than standard gradient boosting, which can be a constraint for very large stock universes.

5. **Complexity of Categorical Interactions**: While CatBoost handles individual categorical features well, it does not explicitly model interactions between categoricals (e.g., "Sector=Tech AND Exchange=NASDAQ" combinations). These must be engineered manually or captured implicitly through tree depth.

6. **Less Community Tooling Than XGBoost/LightGBM**: SHAP values, model interpretation tools, and online resources are less abundant for CatBoost, making debugging and interpretation slightly harder in practice.

## When to Use in Stock Market

- When the feature set contains important categorical variables (sector, exchange, rating, country)
- For cross-sectional stock selection models with mixed fundamental and technical features
- When you want minimal preprocessing and feature engineering for categorical data
- When overfitting is a primary concern and ordered boosting's unbiased gradients are valuable
- For production systems where default hyperparameters should work reasonably well
- When model robustness is more important than maximum training speed

## When NOT to Use in Stock Market

- When all features are numerical (LightGBM or XGBoost will be faster without the overhead)
- When training speed is the top priority and the dataset is very large
- When highly asymmetric decision paths are needed (symmetric trees are a limitation)
- For tick-level HFT data where all features are continuous microstructure metrics
- When interpretability of categorical feature effects is more important than predictive accuracy

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|---------------------------|
| iterations | Number of boosting rounds | 100 - 5000 | 500-2000 with early stopping |
| learning_rate | Shrinkage factor | 0.01 - 0.3 | 0.03-0.1 |
| depth | Tree depth (symmetric tree) | 4 - 10 | 6-8 for mixed features |
| l2_leaf_reg | L2 regularization on leaves | 1 - 10 | 3-7 for noisy stock data |
| random_strength | Bagging temperature | 0 - 10 | 1-3 |
| border_count | Number of histogram bins | 32 - 255 | 128 for categorical-heavy data |
| one_hot_max_size | Max categories for one-hot | 2 - 25 | 10 (one-hot for small categories like Exchange) |
| min_data_in_leaf | Min samples per leaf | 1 - 100 | 20-50 for stock cross-sections |
| bagging_temperature | Controls Bayesian bootstrap | 0 - 10 | 1.0 for moderate randomization |
| rsm | Random subspace method (feature sampling) | 0.5 - 1.0 | 0.7-0.8 for feature diversity |
| cat_features | Indices of categorical columns | list | Must specify correctly |

## Stock Market Performance Tips

1. **Always Specify Categorical Features Explicitly**: CatBoost can auto-detect categoricals, but explicitly specifying them ensures correct encoding. Misidentifying a numerical feature as categorical (or vice versa) will degrade performance.

2. **Increase Depth for Category-Rich Data**: Symmetric trees need more depth to handle categorical interactions. With 5 categorical features, use depth 7-8 to allow enough levels for both categorical and numerical splits.

3. **Use Ordered Boosting for Small Datasets**: When the stock universe is small (fewer than 500 stocks), ordered boosting's bias reduction is most impactful. For large universes, the standard approach may suffice.

4. **Encode Temporal Information as Features**: CatBoost does not know about time. Add features like "quarter," "month," and "days_since_earnings" to help the model capture seasonal patterns in stock behavior.

5. **Cross-Validate Across Time, Not Stocks**: Use temporal folds (train on 2018-2020, validate on 2021) rather than random stock-level splits to properly evaluate out-of-sample performance.

6. **Monitor Category-Level Performance**: Track model accuracy separately for each sector and market cap category. The model may perform well overall but poorly for specific sectors where categorical encoding is less reliable.

## Comparison with Other Algorithms

| Criteria | CatBoost | XGBoost | LightGBM | Random Forest | Logistic Regression |
|----------|----------|---------|----------|--------------|-------------------|
| Categorical Handling | Native (best) | Needs encoding | Basic native | Needs encoding | Needs encoding |
| Training Speed | Moderate | Moderate | Fast | Moderate | Very Fast |
| Overfitting Control | Excellent (ordered) | Good | Good | Excellent | Good |
| Tree Structure | Symmetric | Asymmetric | Asymmetric | Asymmetric | N/A |
| Default Performance | Very Good | Good | Good | Good | Moderate |
| Prediction Shift | Eliminated | Present | Present | N/A | N/A |
| Best For (Stock Market) | Mixed feature types | HFT signals | Large-scale | Robust baseline | Linear signals |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Simplified CatBoost for Stock Classification with Mixed Features
# ============================================================

class OrderedTargetEncoder:
    """Ordered target statistics for categorical features."""

    def __init__(self, cat_indices, prior_weight=1.0):
        self.cat_indices = cat_indices
        self.prior_weight = prior_weight

    def fit_transform(self, X, y):
        """Encode categoricals using ordered target statistics."""
        X_encoded = X.copy().astype(float)
        n = len(y)
        perm = np.random.permutation(n)
        prior = np.mean(y)

        for col in self.cat_indices:
            encoded = np.zeros(n)
            # Track running sum and count per category
            cat_sum = {}
            cat_count = {}

            for idx in perm:
                cat_val = int(X[idx, col])
                s = cat_sum.get(cat_val, 0)
                c = cat_count.get(cat_val, 0)
                encoded[idx] = (s + prior * self.prior_weight) / (c + self.prior_weight)
                cat_sum[cat_val] = s + y[idx]
                cat_count[cat_val] = c + 1

            X_encoded[:, col] = encoded

        return X_encoded

    def transform(self, X, y_train, X_train):
        """Encode test data using full training statistics."""
        X_encoded = X.copy().astype(float)
        prior = np.mean(y_train)

        for col in self.cat_indices:
            cat_stats = {}
            for i in range(len(X_train)):
                cat_val = int(X_train[i, col])
                if cat_val not in cat_stats:
                    cat_stats[cat_val] = {'sum': 0, 'count': 0}
                cat_stats[cat_val]['sum'] += y_train[i]
                cat_stats[cat_val]['count'] += 1

            for i in range(len(X)):
                cat_val = int(X[i, col])
                if cat_val in cat_stats:
                    s = cat_stats[cat_val]['sum']
                    c = cat_stats[cat_val]['count']
                    X_encoded[i, col] = (s + prior * self.prior_weight) / (c + self.prior_weight)
                else:
                    X_encoded[i, col] = prior

        return X_encoded


class ObliviousTree:
    """Symmetric (oblivious) decision tree."""

    def __init__(self, depth=6, lambda_reg=3.0):
        self.depth = depth
        self.lambda_reg = lambda_reg
        self.splits = []  # (feature_idx, threshold) per level
        self.leaf_weights = None

    def _find_best_split_for_level(self, X, g, h, leaf_assignments):
        n_features = X.shape[1]
        best_gain = -np.inf
        best_feat = 0
        best_thresh = 0

        # Current leaves
        n_current_leaves = 2 ** len(self.splits) if self.splits else 1
        unique_leaves = range(n_current_leaves)

        for f in range(n_features):
            thresholds = np.percentile(X[:, f], np.linspace(10, 90, 20))
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                total_gain = 0
                valid = True

                for leaf_id in unique_leaves:
                    mask = leaf_assignments == leaf_id
                    if mask.sum() < 4:
                        valid = False
                        break

                    left_mask = mask & (X[:, f] <= thresh)
                    right_mask = mask & (X[:, f] > thresh)

                    if left_mask.sum() < 2 or right_mask.sum() < 2:
                        valid = False
                        break

                    gl = np.sum(g[left_mask])
                    hl = np.sum(h[left_mask])
                    gr = np.sum(g[right_mask])
                    hr = np.sum(h[right_mask])
                    gp = gl + gr
                    hp = hl + hr

                    gain = 0.5 * (gl**2/(hl+self.lambda_reg) +
                                  gr**2/(hr+self.lambda_reg) -
                                  gp**2/(hp+self.lambda_reg))
                    total_gain += gain

                if valid and total_gain > best_gain:
                    best_gain = total_gain
                    best_feat = f
                    best_thresh = thresh

        return best_feat, best_thresh

    def fit(self, X, g, h):
        n_samples = len(g)
        leaf_assignments = np.zeros(n_samples, dtype=int)

        self.splits = []
        for level in range(self.depth):
            feat, thresh = self._find_best_split_for_level(X, g, h, leaf_assignments)
            self.splits.append((feat, thresh))

            # Update leaf assignments: each leaf splits into two
            new_assignments = leaf_assignments * 2
            right_mask = X[:, feat] > thresh
            new_assignments[right_mask] += 1
            leaf_assignments = new_assignments

        # Compute leaf weights
        n_leaves = 2 ** self.depth
        self.leaf_weights = np.zeros(n_leaves)
        for leaf_id in range(n_leaves):
            mask = leaf_assignments == leaf_id
            if mask.sum() > 0:
                self.leaf_weights[leaf_id] = (-np.sum(g[mask]) /
                                               (np.sum(h[mask]) + self.lambda_reg))

        return self

    def predict(self, X):
        n = len(X)
        leaf_ids = np.zeros(n, dtype=int)
        for feat, thresh in self.splits:
            leaf_ids = leaf_ids * 2
            right_mask = X[:, feat] > thresh
            leaf_ids[right_mask] += 1
        return self.leaf_weights[leaf_ids]


class StockCatBoost:
    """Simplified CatBoost with ordered target encoding and oblivious trees."""

    def __init__(self, n_estimators=200, learning_rate=0.05, depth=6,
                 lambda_reg=3.0, cat_features=None, subsample=0.8):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.depth = depth
        self.lambda_reg = lambda_reg
        self.cat_features = cat_features or []
        self.subsample = subsample
        self.trees = []
        self.encoder = None
        self.base_score = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=30):
        n = len(y)
        self.base_score = np.log(np.mean(y) / (1 - np.mean(y) + 1e-10))

        # Encode categorical features
        self.encoder = OrderedTargetEncoder(self.cat_features)
        X_enc = self.encoder.fit_transform(X, y)

        if X_val is not None:
            X_val_enc = self.encoder.transform(X_val, y, X)
            raw_val = np.full(len(y_val), self.base_score)
            best_val_loss = float('inf')
            best_round = 0

        raw_pred = np.full(n, self.base_score)
        self.trees = []

        for t in range(self.n_estimators):
            p = self._sigmoid(raw_pred)
            g = p - y
            h = p * (1 - p)
            h = np.maximum(h, 1e-6)

            # Subsample
            n_sub = int(n * self.subsample)
            idx = np.random.choice(n, n_sub, replace=False)

            tree = ObliviousTree(depth=self.depth, lambda_reg=self.lambda_reg)
            tree.fit(X_enc[idx], g[idx], h[idx])
            self.trees.append(tree)

            raw_pred += self.lr * tree.predict(X_enc)

            if X_val is not None:
                raw_val += self.lr * tree.predict(X_val_enc)
                val_p = self._sigmoid(raw_val)
                val_p = np.clip(val_p, 1e-15, 1-1e-15)
                val_loss = -np.mean(y_val * np.log(val_p) + (1-y_val) * np.log(1-val_p))

                if (t+1) % 25 == 0:
                    train_acc = np.mean((p >= 0.5) == y)
                    val_acc = np.mean((val_p >= 0.5) == y_val)
                    print(f"Round {t+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_round = t
                elif t - best_round >= early_stopping:
                    print(f"Early stopping at round {t+1}. Best: {best_round+1}")
                    self.trees = self.trees[:best_round+1]
                    break

        self._X_train = X
        self._y_train = y
        return self

    def predict_proba(self, X):
        X_enc = self.encoder.transform(X, self._y_train, self._X_train)
        raw = np.full(len(X), self.base_score)
        for tree in self.trees:
            raw += self.lr * tree.predict(X_enc)
        return self._sigmoid(raw)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# Generate Stock Data with Mixed Features
# ============================================================
np.random.seed(42)
n_stocks = 8000

# Categorical features
sectors = np.random.randint(0, 11, n_stocks)
exchanges = np.random.randint(0, 3, n_stocks)
cap_cat = np.random.randint(0, 5, n_stocks)
analyst_rating = np.random.randint(0, 5, n_stocks)
country = np.random.randint(0, 10, n_stocks)

# Numerical features
rsi = np.random.uniform(15, 85, n_stocks)
pe = np.random.normal(20, 10, n_stocks)
pb = np.random.lognormal(1, 0.6, n_stocks)
div_yield = np.random.exponential(2, n_stocks)
momentum = np.random.normal(5, 20, n_stocks)
vol = np.random.lognormal(2.8, 0.4, n_stocks)
earn_growth = np.random.normal(10, 30, n_stocks)
rev_surprise = np.random.normal(1, 4, n_stocks)
short_int = np.random.exponential(5, n_stocks)
inst_own = np.random.uniform(20, 95, n_stocks)

X = np.column_stack([sectors, exchanges, cap_cat, analyst_rating, country,
                     rsi, pe, pb, div_yield, momentum,
                     vol, earn_growth, rev_surprise, short_int, inst_own])

cat_features = [0, 1, 2, 3, 4]

feature_names = ['Sector', 'Exchange', 'Cap_Cat', 'Analyst_Rating', 'Country',
                 'RSI_14', 'PE_Ratio', 'PB_Ratio', 'Div_Yield', 'Momentum_6M',
                 'Volatility', 'Earnings_Growth', 'Rev_Surprise',
                 'Short_Interest', 'Inst_Ownership']

# Labels with categorical effects
sector_effects = {0: 0.3, 1: -0.1, 2: 0.2, 3: -0.2, 4: 0.1,
                  5: -0.3, 6: 0.0, 7: 0.15, 8: -0.15, 9: 0.1, 10: -0.05}
analyst_effects = {0: 0.6, 1: 0.3, 2: 0.0, 3: -0.3, 4: -0.6}

signal = np.zeros(n_stocks)
for i in range(n_stocks):
    signal[i] = (0.02 * momentum[i] + 0.01 * (rsi[i]-50) - 0.01 * pe[i] +
                 0.03 * earn_growth[i]/10 + 0.05 * rev_surprise[i] -
                 0.02 * short_int[i] + sector_effects[sectors[i]] +
                 analyst_effects[analyst_rating[i]] +
                 0.2 * (cap_cat[i] <= 1) + np.random.normal(0, 0.5))

y = (signal > np.median(signal)).astype(int)

print(f"Stocks: {n_stocks}")
print(f"Outperform: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Underperform: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

# Sector label names
sector_names = ['Tech', 'Health', 'Finance', 'Energy', 'Materials',
                'Utilities', 'RealEstate', 'ConsDisc', 'ConsStap', 'Industrials', 'Comm']
print("\nSector distribution:")
for s in range(11):
    mask = sectors == s
    print(f"  {sector_names[s]:>12}: {mask.sum()} stocks, "
          f"outperform rate: {y[mask].mean()*100:.1f}%")

# Temporal split
train_end = 5600
val_end = 7000
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"\nTrain: {train_end} | Val: {val_end-train_end} | Test: {n_stocks-val_end}")

# ============================================================
# Train CatBoost
# ============================================================
print("\n=== Training CatBoost ===")
cb = StockCatBoost(n_estimators=300, learning_rate=0.06, depth=5,
                    lambda_reg=3.0, cat_features=cat_features, subsample=0.8)
cb.fit(X_train, y_train, X_val, y_val, early_stopping=30)

# ============================================================
# Evaluate
# ============================================================
y_pred = cb.predict(X_test)
y_proba = cb.predict_proba(X_test)

accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred==1) & (y_test==1)) / max(np.sum(y_pred==1), 1)
recall = np.sum((y_pred==1) & (y_test==1)) / max(np.sum(y_test==1), 1)
f1 = 2*precision*recall / max(precision+recall, 1e-10)

print(f"\n=== TEST PERFORMANCE ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Per-sector analysis
print(f"\n=== PER-SECTOR ACCURACY ===")
test_sectors = X_test[:, 0].astype(int)
for s in range(11):
    mask = test_sectors == s
    if mask.sum() > 0:
        sec_acc = np.mean(y_pred[mask] == y_test[mask])
        print(f"  {sector_names[s]:>12}: Accuracy={sec_acc:.4f} (n={mask.sum()})")

# Portfolio simulation
print(f"\n=== STOCK SELECTION PORTFOLIO ===")
top_quintile = y_proba >= np.percentile(y_proba, 80)
bottom_quintile = y_proba <= np.percentile(y_proba, 20)

top_accuracy = np.mean(y_test[top_quintile] == 1) if top_quintile.sum() > 0 else 0
bottom_accuracy = np.mean(y_test[bottom_quintile] == 0) if bottom_quintile.sum() > 0 else 0

print(f"Top quintile (highest P(outperform)):")
print(f"  Stocks selected: {top_quintile.sum()}")
print(f"  Actual outperform rate: {top_accuracy*100:.1f}%")
print(f"Bottom quintile (lowest P(outperform)):")
print(f"  Stocks selected: {bottom_quintile.sum()}")
print(f"  Actual underperform rate: {bottom_accuracy*100:.1f}%")
print(f"  Long-Short spread: {(top_accuracy - (1-bottom_accuracy))*100:.1f}%")
```

## Key Takeaways

- CatBoost's ordered target statistics provide the most principled approach to handling categorical features like sector, exchange, and analyst rating in stock classification
- Ordered boosting eliminates prediction shift, producing more reliable out-of-sample predictions, critical for live trading
- Symmetric (oblivious) trees act as a strong regularizer, preventing overfitting on noisy stock data at the structural level
- Always explicitly specify which features are categorical to ensure proper encoding
- CatBoost is the top choice when the feature set contains a mix of fundamental categoricals and technical numericals
- Per-sector and per-category performance monitoring is essential to ensure the model generalizes across all market segments
- Despite slower training than LightGBM, CatBoost's superior handling of categorical features and robustness often justify the additional training time for cross-sectional stock selection
- Use quintile-based portfolio analysis (long top quintile, short bottom quintile) to evaluate the model's practical value for portfolio construction
