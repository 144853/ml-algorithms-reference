# LSTM (Long Short-Term Memory) - Complete Guide with Stock Market Applications

## Overview

Long Short-Term Memory (LSTM) networks are a specialized type of Recurrent Neural Network (RNN) architecture designed to learn long-range dependencies in sequential data. Introduced by Hochreiter and Schmidhuber in 1997, LSTMs address the vanishing gradient problem that plagues standard RNNs by introducing a gating mechanism that controls the flow of information through the network. The key innovation is the cell state, a dedicated memory pathway that allows information to flow across many time steps with minimal degradation.

In stock market applications, LSTMs have become one of the most popular deep learning approaches for price prediction. Their ability to learn complex nonlinear patterns from sequential data makes them well-suited for capturing the temporal dependencies in financial time series: momentum effects, mean-reversion patterns, volatility clustering, and cross-feature interactions. Unlike linear models such as ARIMA, LSTMs can automatically discover and exploit nonlinear relationships between current prices, historical patterns, technical indicators, and volume data.

The appeal of LSTMs for stock forecasting lies in their flexibility. They can ingest multivariate input sequences (price, volume, indicators, macro data), learn hierarchical temporal features at multiple time scales, and output predictions for single or multiple future time steps. However, this flexibility comes with significant challenges: LSTMs require large amounts of training data, are prone to overfitting on noisy financial data, and their predictions lack the interpretability of classical statistical models. Understanding these trade-offs is essential for deploying LSTMs effectively in quantitative trading.

## How It Works - The Math Behind It

### The LSTM Cell Architecture

An LSTM cell contains four main components operating on the hidden state h_t and cell state C_t:

```
At each time step t, given:
  - Input: x_t (feature vector at time t)
  - Previous hidden state: h_{t-1}
  - Previous cell state: C_{t-1}
```

### Gate 1: Forget Gate (f_t)

Decides what information to discard from the cell state:

```
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
```

Where:
- `sigma` is the sigmoid function: sigma(x) = 1 / (1 + exp(-x))
- `W_f` is the weight matrix for the forget gate
- `b_f` is the bias vector
- `[h_{t-1}, x_t]` is the concatenation of previous hidden state and current input

Output range: [0, 1]. A value near 0 means "forget this," near 1 means "keep this."

For stock data: the forget gate learns to discard old price patterns that are no longer relevant (e.g., pre-earnings vs. post-earnings regime).

### Gate 2: Input Gate (i_t) and Candidate Cell State (C_tilde_t)

Decides what new information to store:

```
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)           # What to update
C_tilde_t = tanh(W_C * [h_{t-1}, x_t] + b_C)       # Candidate values
```

The input gate i_t controls how much of each candidate value to add.

For stock data: the input gate learns to incorporate new information (earnings surprises, volume spikes, trend changes) into the cell memory.

### Cell State Update

The new cell state combines forgotten old information and selected new information:

```
C_t = f_t * C_{t-1} + i_t * C_tilde_t
```

This is the crucial innovation: the cell state acts as a conveyor belt, allowing gradients to flow through time without vanishing (additive update rather than multiplicative).

### Gate 3: Output Gate (o_t)

Decides what to output based on the cell state:

```
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

The hidden state h_t is a filtered version of the cell state, containing only the information relevant for the current output.

### Full Forward Pass Summary

```
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)           # Forget gate
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)           # Input gate
C_tilde_t = tanh(W_C * [h_{t-1}, x_t] + b_C)      # Candidate
C_t = f_t * C_{t-1} + i_t * C_tilde_t              # Cell update
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)           # Output gate
h_t = o_t * tanh(C_t)                               # Hidden state
```

### Parameter Count

For an LSTM layer with input size `n` and hidden size `h`:

```
Parameters = 4 * (n * h + h * h + h) = 4 * h * (n + h + 1)
```

The factor 4 accounts for the four weight matrices (forget, input, candidate, output).

### Training: Backpropagation Through Time (BPTT)

Gradients are computed by unrolling the LSTM through the sequence length T:

```
Loss = sum_{t=1}^{T} L(y_hat_t, y_t)
```

Gradients flow backward through both the hidden state path and the cell state path. The cell state gradient:

```
dC_t/dC_{t-1} = f_t    (element-wise)
```

Since f_t is in [0, 1], this multiplicative factor can maintain gradient magnitude across time steps (unlike vanilla RNNs where repeated matrix multiplications cause vanishing/exploding gradients).

### Output Layer for Stock Prediction

For price prediction, the final hidden state feeds into a dense layer:

```
y_hat = W_out * h_T + b_out
```

For multi-step forecasting, options include:
1. **Recursive**: Feed predictions back as inputs
2. **Direct**: Separate output head for each horizon
3. **Sequence-to-sequence**: Decoder LSTM generates the full forecast sequence

## Stock Market Use Case: Multi-Feature Sequential Stock Price Pattern Learning

### The Problem

A systematic hedge fund wants to build a model that learns complex nonlinear patterns in stock price data for 5-day ahead return forecasting. The model should ingest a rich feature set including OHLCV data, technical indicators, and market regime indicators, processing 60-day lookback windows to capture medium-term momentum, mean-reversion, and volatility patterns. The fund needs the model to work across a universe of 500 mid-to-large-cap stocks.

### Stock Market Features (Input Data)

| Feature | Description | Category |
|---------|-------------|----------|
| Log Return | ln(Close_t / Close_{t-1}) | Price dynamics |
| High-Low Range | (High - Low) / Close | Intraday volatility |
| Open-Close Gap | (Open - prev Close) / prev Close | Overnight sentiment |
| Volume Ratio | Volume / 20-day avg volume | Liquidity signal |
| RSI (14-day) | Relative Strength Index | Momentum indicator |
| MACD Signal | MACD - Signal line | Trend indicator |
| Bollinger %B | Price position within Bollinger Bands | Mean-reversion signal |
| ATR (14-day) | Average True Range / Close | Volatility measure |
| OBV Change | On-Balance Volume rate of change | Volume-price confirmation |
| 5-day SMA Ratio | Close / SMA(5) | Short-term trend |
| 20-day SMA Ratio | Close / SMA(20) | Medium-term trend |
| 60-day SMA Ratio | Close / SMA(60) | Long-term trend |
| Realized Vol (20d) | 20-day rolling standard deviation of returns | Risk measure |
| VIX Level | Market volatility index | Market regime |
| Sector Relative Strength | Stock return - sector return | Relative performance |

### Example Data Structure

```python
import numpy as np

np.random.seed(42)

# Generate 1500 trading days (~6 years) of multi-feature stock data
n_days = 1500
n_features = 15

# Generate base stock prices using GBM with regime switching
mu_regimes = [0.0005, -0.0003, 0.0008]  # Bull, bear, strong bull
sigma_regimes = [0.015, 0.025, 0.012]
regime_lengths = [300, 200, 250, 150, 300, 300]

prices = np.zeros(n_days)
prices[0] = 100.0
returns = np.zeros(n_days)

idx = 0
for i, length in enumerate(regime_lengths):
    regime = i % 3
    for t in range(length):
        if idx + t + 1 >= n_days:
            break
        r = np.random.normal(mu_regimes[regime], sigma_regimes[regime])
        returns[idx + t + 1] = r
        prices[idx + t + 1] = prices[idx + t] * (1 + r)
    idx += length

# Generate feature matrix
features = np.zeros((n_days, n_features))
features[:, 0] = returns                                    # Log returns
features[:, 1] = np.abs(np.random.normal(0.02, 0.008, n_days))  # High-Low range
features[:, 2] = np.random.normal(0, 0.005, n_days)         # Open-Close gap
features[:, 3] = np.abs(np.random.normal(1.0, 0.3, n_days)) # Volume ratio

# Technical indicators (simplified)
for t in range(14, n_days):
    # RSI (simplified)
    gains = np.maximum(returns[t-13:t+1], 0)
    losses = np.maximum(-returns[t-13:t+1], 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        features[t, 4] = 100 - 100 / (1 + rs)
    else:
        features[t, 4] = 100

# SMA ratios
for t in range(60, n_days):
    features[t, 9] = prices[t] / np.mean(prices[t-4:t+1])     # 5-day SMA ratio
    features[t, 10] = prices[t] / np.mean(prices[t-19:t+1])   # 20-day SMA ratio
    features[t, 11] = prices[t] / np.mean(prices[t-59:t+1])   # 60-day SMA ratio

# Realized volatility
for t in range(20, n_days):
    features[t, 12] = np.std(returns[t-19:t+1]) * np.sqrt(252)

# Fill remaining features with correlated noise
features[:, 5] = np.random.normal(0, 0.01, n_days)   # MACD signal
features[:, 6] = np.random.normal(0.5, 0.15, n_days)  # Bollinger %B
features[:, 7] = np.abs(np.random.normal(0.02, 0.005, n_days))  # ATR
features[:, 8] = np.random.normal(0, 0.02, n_days)    # OBV change
features[:, 13] = np.abs(np.random.normal(18, 5, n_days))  # VIX
features[:, 14] = np.random.normal(0, 0.005, n_days)  # Sector relative

print(f"Feature matrix shape: ({n_days}, {n_features})")
print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
print(f"Feature names: [log_ret, hl_range, oc_gap, vol_ratio, rsi, macd, boll_b, atr, obv, sma5, sma20, sma60, rvol, vix, sector_rs]")
```

### The Model in Action

```python
import numpy as np

# LSTM implementation from scratch using NumPy

class LSTMCell:
    """Single LSTM cell implementation."""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier initialization
        scale_ih = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))

        # Weights for all 4 gates concatenated: [forget, input, candidate, output]
        self.W_ih = np.random.randn(4 * hidden_size, input_size) * scale_ih
        self.W_hh = np.random.randn(4 * hidden_size, hidden_size) * scale_hh
        self.bias = np.zeros(4 * hidden_size)

        # Initialize forget gate bias to 1 (helps with long-term memory)
        self.bias[:hidden_size] = 1.0

    def forward(self, x, h_prev, c_prev):
        """Forward pass for a single time step."""
        H = self.hidden_size

        # Combined linear transformation
        gates = self.W_ih @ x + self.W_hh @ h_prev + self.bias

        # Split into 4 gates
        f_gate = self._sigmoid(gates[:H])          # Forget gate
        i_gate = self._sigmoid(gates[H:2*H])       # Input gate
        c_cand = np.tanh(gates[2*H:3*H])           # Candidate cell
        o_gate = self._sigmoid(gates[3*H:4*H])     # Output gate

        # Cell state update
        c_new = f_gate * c_prev + i_gate * c_cand

        # Hidden state update
        h_new = o_gate * np.tanh(c_new)

        return h_new, c_new, (f_gate, i_gate, c_cand, o_gate)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTMModel:
    """LSTM model for stock price prediction."""

    def __init__(self, input_size, hidden_size, output_size=1, n_layers=1):
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # LSTM layers
        self.cells = []
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

        # Output layer
        self.W_out = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros(output_size)

    def forward(self, X_sequence):
        """
        Forward pass through the full sequence.
        X_sequence: (seq_len, input_size)
        Returns: prediction (output_size,)
        """
        seq_len = X_sequence.shape[0]
        H = self.hidden_size

        # Initialize hidden and cell states
        h_states = [np.zeros(H) for _ in range(self.n_layers)]
        c_states = [np.zeros(H) for _ in range(self.n_layers)]

        # Process sequence
        for t in range(seq_len):
            x_t = X_sequence[t]

            for layer in range(self.n_layers):
                h_states[layer], c_states[layer], _ = self.cells[layer].forward(
                    x_t, h_states[layer], c_states[layer]
                )
                x_t = h_states[layer]  # Input to next layer is hidden state of current

        # Output from final hidden state
        output = self.W_out @ h_states[-1] + self.b_out
        return output

    def predict_batch(self, X_batch):
        """Predict for a batch of sequences."""
        predictions = []
        for i in range(X_batch.shape[0]):
            pred = self.forward(X_batch[i])
            predictions.append(pred)
        return np.array(predictions)


# --- Prepare data for LSTM ---
seq_length = 60  # 60-day lookback window
forecast_horizon = 5  # Predict 5-day return

# Normalize features (z-score normalization using training stats)
start_idx = 60  # Skip first 60 days (for feature computation)
valid_features = features[start_idx:]
valid_returns = returns[start_idx:]
valid_prices = prices[start_idx:]

# Train/val/test split
n_valid = len(valid_features)
train_end = int(n_valid * 0.7)
val_end = int(n_valid * 0.85)

train_features = valid_features[:train_end]
train_mean = np.mean(train_features, axis=0)
train_std = np.std(train_features, axis=0) + 1e-8

# Normalize all features using training statistics
norm_features = (valid_features - train_mean) / train_std

# Create sequences
def create_sequences(features, returns, seq_len, horizon):
    X, y = [], []
    for i in range(seq_len, len(features) - horizon):
        X.append(features[i-seq_len:i])
        # Target: sum of next 'horizon' returns (cumulative return)
        y.append(np.sum(returns[i:i+horizon]))
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(norm_features, valid_returns, seq_length, forecast_horizon)

# Split
n_samples = len(X_all)
train_split = int(n_samples * 0.7)
val_split = int(n_samples * 0.85)

X_train, y_train = X_all[:train_split], y_all[:train_split]
X_val, y_val = X_all[train_split:val_split], y_all[train_split:val_split]
X_test, y_test = X_all[val_split:], y_all[val_split:]

print(f"\nDataset shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape},  y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# --- Initialize and run model ---
model = LSTMModel(
    input_size=n_features,
    hidden_size=32,
    output_size=1,
    n_layers=2
)

total_params = sum(
    4 * cell.hidden_size * (cell.input_size + cell.hidden_size + 1)
    for cell in model.cells
) + model.hidden_size + 1
print(f"\nModel architecture:")
print(f"  LSTM layers: 2")
print(f"  Hidden size: 32")
print(f"  Total parameters: ~{total_params}")

# Generate predictions (untrained model - demonstrating forward pass)
print(f"\nForward pass demonstration (untrained model):")
sample_pred = model.forward(X_test[0])
print(f"  Input shape: {X_test[0].shape}")
print(f"  Prediction: {sample_pred[0]:.6f}")
print(f"  Actual 5-day return: {y_test[0]:.6f}")
```

## Advantages

1. **Captures nonlinear temporal patterns**: LSTMs can learn complex nonlinear relationships between sequential inputs and outputs. In stock markets, this means capturing patterns like momentum that transitions to mean-reversion at different price levels, or volatility that responds asymmetrically to positive versus negative returns.

2. **Flexible multi-feature input**: Unlike ARIMA which processes a single series, LSTMs naturally handle multivariate input sequences. A single LSTM can simultaneously process price data, volume, technical indicators, fundamental ratios, and macro variables, learning cross-feature interactions automatically.

3. **Adaptive memory management**: The gating mechanism allows LSTMs to selectively retain or forget information across time steps. For stocks, this means the model can learn to weight recent earnings data heavily while gradually discounting stale macro information, adapting its effective memory dynamically.

4. **Long-range dependency learning**: The cell state pathway allows LSTMs to capture dependencies spanning dozens or hundreds of time steps. This is valuable for identifying long-term cycles, seasonal patterns, and regime-dependent behavior in stock price series.

5. **Automatic feature engineering**: LSTMs learn hierarchical representations of the input sequence, effectively performing automatic feature engineering. The hidden states at different time steps encode learned features that may capture patterns no human analyst would explicitly specify.

6. **Multi-step forecasting capability**: LSTMs support various multi-step forecasting strategies (recursive, direct, sequence-to-sequence). For trading applications, this enables simultaneous prediction of returns at multiple horizons (1-day, 5-day, 20-day) from a single model architecture.

7. **Transfer learning potential**: LSTM models pre-trained on a broad stock universe can be fine-tuned for individual stocks with limited data. This transfer learning approach is particularly valuable for recently listed stocks or stocks transitioning to new sectors.

## Disadvantages

1. **Requires large training datasets**: LSTMs have thousands to millions of parameters that need sufficient data to estimate reliably. For individual stocks with daily data, 5-10 years may not be enough, leading to overfitting. This is problematic for stocks with limited history or structural changes.

2. **Black-box predictions**: Unlike ARIMA or Prophet, LSTM predictions are difficult to interpret. When the model predicts a price decline, understanding why (which features, which time steps) requires additional explainability tools (attention weights, SHAP values) that add complexity.

3. **Overfitting to noise**: Stock returns have very low signal-to-noise ratios. LSTMs, being powerful function approximators, can easily memorize noise patterns in training data that don't generalize. Extensive regularization (dropout, L2, early stopping) is essential but imperfect.

4. **Computational expense**: Training LSTMs on sequences of financial data requires significant GPU resources, especially for multi-layer models with long sequence lengths. Re-training across a 500-stock universe daily is computationally demanding compared to ARIMA.

5. **Hyperparameter sensitivity**: Performance is highly sensitive to architecture choices (hidden size, number of layers, sequence length, learning rate, dropout rate). The optimal configuration varies across stocks and time periods, requiring expensive hyperparameter search.

6. **Stationarity challenges**: LSTMs trained on one market regime may fail in another. A model trained during a bull market may produce systematically biased predictions during a bear market. Unlike ARIMA with differencing, ensuring stationarity requires careful preprocessing.

7. **No native uncertainty quantification**: Standard LSTMs produce point predictions without confidence intervals. Obtaining uncertainty estimates requires additional techniques (MC dropout, ensemble methods, Bayesian LSTM) that increase complexity and computational cost.

## When to Use in Stock Market

- When the feature space is rich (10+ features) and cross-feature interactions are important
- For medium-frequency trading (daily to weekly) with sufficient historical data (5+ years)
- When nonlinear patterns are hypothesized (asymmetric volatility response, regime-dependent momentum)
- As part of an ensemble where LSTM captures nonlinear components complementing linear models
- For stocks or sectors where traditional linear models consistently underperform
- When GPU resources are available for training and the model can be updated regularly
- For pattern recognition tasks like identifying chart patterns (head-and-shoulders, flags) from raw price data
- When transfer learning from a broad stock universe can compensate for limited individual stock history

## When NOT to Use in Stock Market

- For single stocks with less than 3 years of daily data (insufficient for training)
- When model interpretability is a regulatory or compliance requirement
- For real-time high-frequency trading where inference latency matters (use simpler models)
- When the signal is primarily in fundamental or macro data without strong temporal patterns
- For stable, low-volatility securities (bonds, utilities) where simple linear models suffice
- When computational resources are limited and model re-training frequency is high
- For pure risk management applications where analytical confidence intervals are required
- When the investment process demands full attribution of model predictions to specific factors

## Hyperparameters Guide

| Parameter | Description | Typical Range | Stock Market Recommendation |
|-----------|-------------|---------------|----------------------------|
| hidden_size | Dimensionality of hidden state | 16-256 | 32-64 for individual stocks; 128-256 for universe models |
| n_layers | Number of stacked LSTM layers | 1-4 | 2 layers for most applications; 1 if data is limited |
| seq_length | Length of input sequence (lookback) | 10-120 | 60 trading days (3 months); shorter for volatile stocks |
| learning_rate | Step size for gradient descent | 1e-4 to 1e-2 | 1e-3 with Adam optimizer; use learning rate scheduling |
| dropout | Dropout rate between layers | 0.0-0.5 | 0.2-0.3 to prevent overfitting on noisy financial data |
| batch_size | Number of sequences per gradient update | 16-128 | 32-64; smaller batches add regularization effect |
| epochs | Number of training passes | 10-200 | Use early stopping with patience 10-20 epochs |
| gradient_clip | Maximum gradient norm | 0.5-5.0 | 1.0 to prevent exploding gradients during volatile periods |
| weight_decay | L2 regularization strength | 0-1e-3 | 1e-5 to 1e-4; helps prevent overfitting to noise |
| forecast_horizon | Number of steps ahead to predict | 1-20 | 5 days for swing trading; 1 day for short-term strategies |

## Stock Market Performance Tips

1. **Normalize inputs carefully**: Use z-score normalization computed only on training data. For financial features, rolling z-scores (using a rolling window of 252 days) often outperform static normalization by adapting to changing market conditions.

2. **Use return-based targets**: Predict log returns rather than price levels. Returns are approximately stationary and bounded, making them much easier for LSTMs to learn. Convert predictions back to prices only for final output.

3. **Implement proper walk-forward validation**: Use expanding or rolling window validation where the model is retrained periodically (e.g., monthly) on the most recent data. Never allow future data to leak into training features.

4. **Apply dropout and early stopping aggressively**: Financial data has very low signal-to-noise ratio. Use dropout rates of 0.2-0.3 between LSTM layers and monitor validation loss for early stopping with patience of 10-20 epochs.

5. **Consider attention mechanisms**: Adding attention layers on top of LSTM helps the model focus on the most relevant time steps in the lookback window. Self-attention can identify which historical days carry the most predictive information.

6. **Ensemble multiple models**: Train 5-10 LSTM models with different random seeds, lookback lengths, or feature subsets and average their predictions. This reduces variance and provides implicit uncertainty estimates through prediction spread.

7. **Monitor for concept drift**: Track rolling out-of-sample performance metrics. When accuracy degrades significantly, trigger model retraining. Markets undergo regime changes that can invalidate learned patterns.

## Comparison with Other Algorithms

| Feature | LSTM | ARIMA | SARIMA | Prophet | TFT |
|---------|------|-------|--------|---------|-----|
| Nonlinear patterns | Yes (core strength) | No | No | Limited | Yes |
| Multivariate input | Yes | Limited (ARIMAX) | Limited | Yes (regressors) | Yes |
| Sequence learning | Yes | Implicit (lags) | Implicit (lags) | No | Yes |
| Interpretability | Low | High | High | High | Medium (attention) |
| Data requirements | High (1000+) | Low (50+) | Medium (3+ cycles) | Medium (2+ years) | Very high |
| Training time | Slow (GPU) | Very fast | Fast | Fast | Very slow |
| Uncertainty | Manual (MC dropout) | Analytical | Analytical | Bayesian | Built-in quantiles |
| Multi-horizon | Yes (seq2seq) | Recursive | Recursive | Yes | Native |
| Feature interaction | Automatic | None | None | Limited | Automatic |
| Stock suitability | Complex patterns | Short-term | Seasonal | Decomposition | Multi-horizon |

## Real-World Stock Market Example

```python
import numpy as np

# ============================================================
# Complete LSTM Implementation for Stock Price Forecasting
# From-scratch NumPy implementation with training loop
# ============================================================

np.random.seed(2024)

# --- Generate Multi-Feature Stock Data ---
n_days = 2000
n_features = 8

# Price generation with multiple regimes
prices_gen = np.zeros(n_days)
prices_gen[0] = 150.0
returns_gen = np.zeros(n_days)

# Regime-switching parameters
regime_schedule = [(0, 400, 0.0004, 0.014),      # Steady bull
                   (400, 650, -0.0002, 0.022),    # Correction
                   (650, 1000, 0.0006, 0.011),    # Strong recovery
                   (1000, 1400, 0.0001, 0.018),   # Consolidation
                   (1400, 2000, 0.0005, 0.013)]   # New bull

for start, end, mu, sigma in regime_schedule:
    for t in range(max(1, start), min(end, n_days)):
        r = np.random.normal(mu, sigma)
        # Add slight autocorrelation (momentum)
        if t > 1:
            r += 0.05 * returns_gen[t-1]
        returns_gen[t] = r
        prices_gen[t] = prices_gen[t-1] * (1 + r)

# Build feature matrix
X_raw = np.zeros((n_days, n_features))
X_raw[:, 0] = returns_gen  # Return

for t in range(20, n_days):
    X_raw[t, 1] = np.std(returns_gen[t-19:t+1]) * np.sqrt(252)   # Realized vol
    X_raw[t, 2] = prices_gen[t] / np.mean(prices_gen[t-4:t+1])   # SMA5 ratio
    X_raw[t, 3] = prices_gen[t] / np.mean(prices_gen[t-19:t+1])  # SMA20 ratio
    X_raw[t, 4] = np.mean(returns_gen[t-4:t+1])                   # 5d momentum
    X_raw[t, 5] = np.mean(returns_gen[t-19:t+1])                  # 20d momentum

    gains = np.maximum(returns_gen[t-13:t+1], 0)
    losses = np.maximum(-returns_gen[t-13:t+1], 0)
    avg_g, avg_l = np.mean(gains), np.mean(losses)
    X_raw[t, 6] = 100 - 100/(1 + avg_g/(avg_l + 1e-10))          # RSI

X_raw[:, 7] = np.abs(np.random.normal(1.0, 0.3, n_days))          # Volume ratio

# Start from day 60 (after features stabilize)
start = 60
X_data = X_raw[start:]
y_data = returns_gen[start:]
p_data = prices_gen[start:]
n_valid = len(X_data)

# Normalize with training statistics
train_cutoff = int(n_valid * 0.7)
val_cutoff = int(n_valid * 0.85)

X_mean = np.mean(X_data[:train_cutoff], axis=0)
X_std = np.std(X_data[:train_cutoff], axis=0) + 1e-8
X_norm = (X_data - X_mean) / X_std

# Create sequences
SEQ_LEN = 30
HORIZON = 5

def make_sequences(X, y, seq_len, horizon):
    Xs, ys = [], []
    for i in range(seq_len, len(X) - horizon):
        Xs.append(X[i-seq_len:i])
        ys.append(np.sum(y[i:i+horizon]))  # Cumulative 5-day return
    return np.array(Xs), np.array(ys).reshape(-1, 1)

X_seq, y_seq = make_sequences(X_norm, y_data, SEQ_LEN, HORIZON)
n_seq = len(X_seq)

# Split
tr_end = int(n_seq * 0.7)
va_end = int(n_seq * 0.85)
X_tr, y_tr = X_seq[:tr_end], y_seq[:tr_end]
X_va, y_va = X_seq[tr_end:va_end], y_seq[tr_end:va_end]
X_te, y_te = X_seq[va_end:], y_seq[va_end:]

print("=" * 60)
print("LSTM Stock Price Forecasting System")
print("=" * 60)
print(f"Sequences: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")
print(f"Sequence length: {SEQ_LEN}, Features: {n_features}")
print(f"Forecast horizon: {HORIZON} days")

# --- LSTM Implementation ---
class SimpleLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        scale_i = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_h = np.sqrt(2.0 / (hidden_dim * 2))

        self.Wf_x = np.random.randn(hidden_dim, input_dim) * scale_i
        self.Wf_h = np.random.randn(hidden_dim, hidden_dim) * scale_h
        self.bf = np.ones(hidden_dim)

        self.Wi_x = np.random.randn(hidden_dim, input_dim) * scale_i
        self.Wi_h = np.random.randn(hidden_dim, hidden_dim) * scale_h
        self.bi = np.zeros(hidden_dim)

        self.Wc_x = np.random.randn(hidden_dim, input_dim) * scale_i
        self.Wc_h = np.random.randn(hidden_dim, hidden_dim) * scale_h
        self.bc = np.zeros(hidden_dim)

        self.Wo_x = np.random.randn(hidden_dim, input_dim) * scale_i
        self.Wo_h = np.random.randn(hidden_dim, hidden_dim) * scale_h
        self.bo = np.zeros(hidden_dim)

        self.Wy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.by = np.zeros(output_dim)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward_sequence(self, X_seq):
        """Process full sequence, return final prediction."""
        T = X_seq.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        for t in range(T):
            x = X_seq[t]
            f = self.sigmoid(self.Wf_x @ x + self.Wf_h @ h + self.bf)
            i = self.sigmoid(self.Wi_x @ x + self.Wi_h @ h + self.bi)
            c_cand = np.tanh(self.Wc_x @ x + self.Wc_h @ h + self.bc)
            o = self.sigmoid(self.Wo_x @ x + self.Wo_h @ h + self.bo)

            c = f * c + i * c_cand
            h = o * np.tanh(c)

        y = self.Wy @ h + self.by
        return y, h

    def predict_batch(self, X_batch):
        preds = []
        for i in range(len(X_batch)):
            y, _ = self.forward_sequence(X_batch[i])
            preds.append(y)
        return np.array(preds)

# Initialize model
HIDDEN_DIM = 24
model = SimpleLSTM(n_features, HIDDEN_DIM, 1)

total_params = (4 * HIDDEN_DIM * (n_features + HIDDEN_DIM + 1) +
                HIDDEN_DIM + 1)
print(f"\nModel: LSTM(input={n_features}, hidden={HIDDEN_DIM}, output=1)")
print(f"Parameters: {total_params}")

# --- Simplified Training via Numerical Gradient ---
# (For demonstration - real training uses backpropagation)
print(f"\n--- Training (Simplified SGD with Perturbation) ---")

def compute_loss(model, X_batch, y_batch):
    preds = model.predict_batch(X_batch)
    return np.mean((preds - y_batch)**2)

# Use a small subset for fast demonstration
n_demo = min(100, len(X_tr))
X_demo = X_tr[:n_demo]
y_demo = y_tr[:n_demo]

best_loss = compute_loss(model, X_demo, y_demo)
print(f"Initial loss: {best_loss:.8f}")

# Simple random search for weight adjustment (demonstration)
n_iterations = 20
lr = 0.001

for iteration in range(n_iterations):
    # Perturb output weights slightly
    noise_W = np.random.randn(*model.Wy.shape) * lr
    noise_b = np.random.randn(*model.by.shape) * lr

    model.Wy += noise_W
    model.by += noise_b

    new_loss = compute_loss(model, X_demo, y_demo)

    if new_loss < best_loss:
        best_loss = new_loss
    else:
        model.Wy -= noise_W
        model.by -= noise_b

    if (iteration + 1) % 5 == 0:
        print(f"  Iteration {iteration+1}: loss = {best_loss:.8f}")

# --- Evaluation on Test Set ---
print(f"\n--- Test Set Evaluation ---")

test_preds = model.predict_batch(X_te).flatten()
test_actual = y_te.flatten()

# Metrics
mse = np.mean((test_preds - test_actual)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test_preds - test_actual))
corr = np.corrcoef(test_preds, test_actual)[0, 1]

# Direction accuracy
dir_acc = np.mean(np.sign(test_preds) == np.sign(test_actual)) * 100

# Sharpe ratio of strategy (simplified)
positions = np.sign(test_preds)  # Long if positive, short if negative
strategy_returns = positions * test_actual
sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252/HORIZON)

print(f"MSE:                 {mse:.8f}")
print(f"RMSE:                {rmse:.6f}")
print(f"MAE:                 {mae:.6f}")
print(f"Correlation:         {corr:.4f}")
print(f"Direction Accuracy:  {dir_acc:.1f}%")
print(f"Strategy Sharpe:     {sharpe:.3f}")

# Naive baseline comparison
naive_pred = np.zeros_like(test_actual)  # Predict zero return
naive_rmse = np.sqrt(np.mean((naive_pred - test_actual)**2))
print(f"\nNaive (zero return) RMSE: {naive_rmse:.6f}")
print(f"LSTM improvement: {(1 - rmse/naive_rmse)*100:+.1f}%")

# --- Sample Predictions ---
print(f"\nSample Predictions (first 15 test periods):")
print(f"{'#':<4} {'Predicted':>12} {'Actual':>12} {'Direction':>12} {'Correct':>10}")
print("-" * 52)
for i in range(min(15, len(test_preds))):
    pred_dir = "LONG" if test_preds[i] > 0 else "SHORT"
    actual_dir = "UP" if test_actual[i] > 0 else "DOWN"
    correct = "Yes" if np.sign(test_preds[i]) == np.sign(test_actual[i]) else "No"
    print(f"{i+1:<4} {test_preds[i]:>+11.6f} {test_actual[i]:>+11.6f} "
          f"{pred_dir:>12} {correct:>10}")

# --- Hidden State Analysis ---
print(f"\n--- Hidden State Analysis ---")
# Analyze what the LSTM learned by examining hidden states over time
hidden_states = []
for i in range(len(X_te)):
    _, h = model.forward_sequence(X_te[i])
    hidden_states.append(h)
hidden_states = np.array(hidden_states)

print(f"Hidden state shape: {hidden_states.shape}")
print(f"Hidden state statistics:")
for dim in range(min(5, HIDDEN_DIM)):
    h_dim = hidden_states[:, dim]
    print(f"  Dim {dim}: mean={h_dim.mean():.4f}, std={h_dim.std():.4f}, "
          f"corr with return={np.corrcoef(h_dim, test_actual)[0,1]:.4f}")
```

## Key Takeaways

1. LSTMs are powerful sequence models that can capture complex nonlinear temporal patterns in stock data through their gating mechanism (forget, input, output gates) and persistent cell state memory.

2. The key innovation of LSTM over standard RNNs is the cell state pathway, which allows gradients to flow through long sequences without vanishing. This enables learning dependencies across 60+ trading days, capturing medium-term momentum and mean-reversion patterns.

3. LSTMs excel when the feature space is rich and multivariate. Feeding price data alongside technical indicators, volume metrics, and macro variables allows the network to learn cross-feature interactions that linear models miss.

4. Overfitting is the primary challenge for stock market LSTMs due to low signal-to-noise ratios. Aggressive regularization (dropout 0.2-0.3, early stopping, weight decay) and proper walk-forward validation are essential for reliable out-of-sample performance.

5. Data preprocessing critically impacts LSTM performance. Use z-score normalization (based on training data only), target log returns rather than prices, and ensure no information leakage from future data into features or normalization statistics.

6. Direction accuracy (predicting up vs. down correctly) is often more relevant than point forecast accuracy for trading strategies. Even modest directional accuracy above 52-53% can generate positive risk-adjusted returns with proper position sizing.

7. LSTMs should be benchmarked against simple baselines (random walk, linear regression, ARIMA) and only deployed when they demonstrate statistically significant improvement. In efficient markets, this improvement is often marginal.

8. Consider combining LSTMs with attention mechanisms (leading toward Transformer architectures) or using them as components in larger ensemble systems that include both machine learning and traditional quantitative models.
