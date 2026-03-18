# Temporal Fusion Transformer (TFT) - Complete Guide with Stock Market Applications

## Overview

The Temporal Fusion Transformer (TFT) is a state-of-the-art deep learning architecture designed for multi-horizon time series forecasting. Introduced by Google Research in 2020, it combines the strengths of recurrent neural networks for temporal processing with self-attention mechanisms for learning long-range dependencies, plus specialized gating mechanisms for adaptive feature selection. Unlike simpler forecasting models, TFT explicitly handles three types of inputs: static covariates (e.g., stock sector, exchange), known future inputs (e.g., calendar features, scheduled earnings dates), and observed past inputs (e.g., historical prices, volume).

In stock market forecasting, TFT addresses a fundamental challenge: predicting prices at multiple future horizons simultaneously (e.g., 1-day, 5-day, 20-day ahead) while incorporating heterogeneous information sources. Traditional models either predict a single step ahead or struggle to integrate static metadata with dynamic time series. TFT's variable selection networks automatically learn which inputs matter at each time step, providing interpretable attention weights that reveal which historical periods and features drive predictions -- invaluable for building trust in trading systems.

The architecture consists of several key components: Variable Selection Networks (VSN) for input feature importance, a sequence-to-sequence LSTM encoder-decoder for local temporal processing, multi-head self-attention for capturing long-range dependencies, and gated residual networks (GRN) for flexible non-linear processing with skip connections. Quantile outputs enable probabilistic forecasting, producing prediction intervals rather than point estimates -- critical for risk management in volatile markets.

## How It Works - The Math Behind It

### Variable Selection Network (VSN)

The VSN learns feature importance at each time step. For input features `x_t = [x_t^1, ..., x_t^p]`:

```
xi_t = Softmax(GRN_xi(flatten(x_t^1, ..., x_t^p)))
```

Each feature is transformed and the weighted combination is:

```
tilde_x_t = sum_{j=1}^{p} xi_t^j * GRN_j(x_t^j)
```

where `xi_t^j` is the learned importance weight for feature `j` at time `t`.

### Gated Residual Network (GRN)

The GRN provides flexible non-linear processing with optional skip connections:

```
eta_1 = W_1 * x + b_1                     (linear layer)
eta_2 = W_2 * ELU(eta_1) + b_2            (non-linear layer with context)
GLU(eta_2) = sigmoid(W_3 * eta_2) * (W_4 * eta_2)   (gated linear unit)
GRN(x) = LayerNorm(x + GLU(eta_2))        (residual + normalization)
```

The GLU gate allows the network to suppress irrelevant inputs entirely.

### Temporal Processing (LSTM Encoder-Decoder)

The encoder processes historical observations:

```
h_t, c_t = LSTM_encoder(tilde_x_t, h_{t-1}, c_{t-1})    for t = 1, ..., T
```

The decoder processes known future inputs:

```
h_t, c_t = LSTM_decoder(tilde_x_t^{future}, h_{t-1}, c_{t-1})    for t = T+1, ..., T+tau
```

where `tau` is the forecast horizon.

### Multi-Head Interpretable Attention

Self-attention captures long-range dependencies across the combined encoder-decoder sequence:

```
Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) * V
```

For interpretability, TFT uses additive attention with shared value weights:

```
InterpretableMultiHead(Q, K, V) = (1/H) * sum_{h=1}^{H} A_h * (V W_V)
```

where `A_h` are attention weights for head `h`, producing interpretable temporal attention patterns.

### Quantile Loss for Probabilistic Forecasting

Instead of MSE, TFT optimizes the quantile loss for multiple quantiles `q in {0.1, 0.5, 0.9}`:

```
QL_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
```

Total loss across all horizons and quantiles:

```
L = sum_{t=T+1}^{T+tau} sum_{q} QL_q(y_t, y_hat_t^q) / tau
```

### Temporal Self-Attention Masking

To prevent information leakage in autoregressive settings:

```
M_{ij} = 0 if i >= j, else -inf
Masked_Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k) + M) * V
```

## Stock Market Use Case: Multi-Horizon Stock Price Forecasting with Attention

### The Problem

An asset management firm wants to build a forecasting system that simultaneously predicts stock prices at 1-day, 5-day, and 20-day horizons for a universe of 500 stocks. The system must:
- Handle diverse information: historical prices, technical indicators, fundamental data, macro variables, and calendar effects
- Provide prediction intervals for risk management (not just point estimates)
- Explain which features and time periods drive predictions
- Adapt to different stock characteristics (sector, market cap, volatility profile)

### Stock Market Features (Input Data)

| Feature Type | Feature Name | Description | Input Category |
|---|---|---|---|
| Historical Price | close_return | Daily log return | Past observed |
| Historical Price | high_low_range | Intraday range (high-low)/close | Past observed |
| Historical Price | open_close_gap | Overnight gap (open-prev_close)/prev_close | Past observed |
| Technical | RSI_14 | 14-day Relative Strength Index | Past observed |
| Technical | MACD_hist | MACD histogram value | Past observed |
| Technical | bollinger_pct | Price position within Bollinger Bands | Past observed |
| Volume | log_volume | Log-transformed daily volume | Past observed |
| Volume | volume_ma_ratio | Volume / 20-day avg volume | Past observed |
| Fundamental | PE_ratio | Trailing P/E ratio (quarterly updated) | Past observed |
| Fundamental | earnings_surprise | Last earnings surprise (%) | Past observed |
| Calendar | day_of_week | Day of week (0-4) | Known future |
| Calendar | month_of_year | Month (1-12) | Known future |
| Calendar | days_to_earnings | Days until next earnings report | Known future |
| Calendar | quad_witching | Quadruple witching week flag | Known future |
| Macro | VIX | CBOE Volatility Index | Past observed |
| Macro | treasury_spread | 10Y-2Y Treasury spread | Past observed |
| Static | sector | GICS sector code | Static |
| Static | market_cap_bucket | Size bucket (small/mid/large) | Static |
| Static | exchange | NYSE / NASDAQ / AMEX | Static |
| Static | avg_daily_volume_bucket | Liquidity bucket | Static |

### Example Data Structure

```python
import numpy as np

# Multi-horizon stock forecasting data structure for TFT
np.random.seed(42)

n_stocks = 500
n_days = 1260         # 5 years of trading days
lookback = 60         # 60-day lookback window
horizons = [1, 5, 20] # Forecast horizons

# Past observed features (change over time, only available historically)
n_past_features = 10
past_feature_names = [
    'close_return', 'high_low_range', 'open_close_gap', 'RSI_14',
    'MACD_hist', 'bollinger_pct', 'log_volume', 'volume_ma_ratio',
    'VIX', 'treasury_spread'
]

# Known future features (known at prediction time for future dates)
n_future_features = 4
future_feature_names = ['day_of_week', 'month_of_year', 'days_to_earnings', 'quad_witching']

# Static features (constant per stock)
n_static_features = 4
static_feature_names = ['sector', 'market_cap_bucket', 'exchange', 'avg_volume_bucket']

# Example: Generate data for one stock
stock_past = np.random.randn(n_days, n_past_features) * 0.01
stock_future = np.column_stack([
    np.tile(np.arange(5), n_days // 5 + 1)[:n_days],          # day_of_week
    np.repeat(np.arange(1, 13), n_days // 12 + 1)[:n_days],   # month_of_year
    np.random.randint(0, 90, n_days),                           # days_to_earnings
    np.random.binomial(1, 4/52, n_days)                         # quad_witching
])
stock_static = np.array([3, 2, 1, 2])  # sector=Tech, cap=Large, exchange=NASDAQ, vol=High

# Create sliding window samples
def create_tft_samples(past, future, static, lookback, max_horizon):
    """Create TFT input samples with lookback window and forecast horizon."""
    n = len(past)
    samples = []
    for t in range(lookback, n - max_horizon):
        sample = {
            'past_observed': past[t - lookback:t],              # (lookback, n_past)
            'known_future': future[t:t + max_horizon],          # (max_horizon, n_future)
            'static': static,                                    # (n_static,)
            'targets': past[t:t + max_horizon, 0]               # close_return for horizons
        }
        samples.append(sample)
    return samples

samples = create_tft_samples(stock_past, stock_future, stock_static, lookback, max(horizons))
print(f"Total samples for one stock: {len(samples)}")
print(f"Past observed shape: {samples[0]['past_observed'].shape}")
print(f"Known future shape: {samples[0]['known_future'].shape}")
print(f"Static features shape: {samples[0]['static'].shape}")
```

### The Model in Action

The TFT processes stock data through its layered architecture:

1. **Static Covariate Encoding**: The stock's sector, market cap, and exchange are embedded and transformed through a GRN. These static context vectors condition the encoder, decoder, and attention layers, allowing the model to learn sector-specific temporal patterns.

2. **Variable Selection**: At each time step, the VSN evaluates all past features (returns, RSI, MACD, volume, etc.) and assigns importance weights. For example, during high-volatility regimes, the VSN might upweight VIX and bollinger_pct while downweighting PE_ratio.

3. **Temporal Encoding**: The LSTM encoder processes the 60-day lookback sequence, building a hidden state that captures recent market dynamics. The decoder then processes the known future inputs (calendar features, earnings schedule) conditioned on the encoder's final state.

4. **Attention Over History**: The multi-head attention mechanism identifies which historical periods are most relevant for each forecast horizon. For a 20-day prediction, it might attend to similar historical patterns 30-60 days ago, while for 1-day prediction, it focuses on the most recent 3-5 days.

5. **Quantile Output**: Three output heads produce the 10th, 50th, and 90th percentile forecasts for each horizon, giving traders both a point estimate and a confidence interval.

## Advantages

1. **Multi-horizon forecasting in a single model.** TFT predicts multiple future time steps simultaneously, which is essential for portfolio management where you need 1-day risk estimates alongside 20-day return forecasts. This is more efficient and consistent than training separate models for each horizon.

2. **Explicit handling of different input types.** Stock market data naturally falls into past observations (prices, volume), known future information (earnings dates, holidays), and static metadata (sector, exchange). TFT's architecture explicitly models these three categories, ensuring known future information is only used appropriately and static features condition temporal processing.

3. **Interpretable attention weights for trading decisions.** The attention mechanism produces human-readable importance scores showing which historical days influence predictions. Traders can inspect whether the model is attending to relevant events (earnings announcements, Fed meetings) rather than spurious patterns.

4. **Variable selection provides feature importance at each time step.** The VSN's time-varying feature weights reveal which indicators matter in different market conditions. During a sell-off, the model might shift attention from momentum to volatility features, matching intuitive trading behavior.

5. **Probabilistic outputs via quantile regression.** Rather than point predictions, TFT produces prediction intervals essential for risk management. A wide 10th-90th percentile spread signals high uncertainty, prompting traders to reduce position size or hedge more aggressively.

6. **Gating mechanisms handle irrelevant inputs gracefully.** The GLU gates can effectively shut off uninformative features or time steps, preventing noise from corrupting predictions. This is valuable when some stocks have missing data or when certain indicators become temporarily uninformative.

7. **Static covariate conditioning enables cross-sectional learning.** By conditioning temporal processing on stock-level attributes, TFT can learn that technology stocks respond differently to interest rate changes than utility stocks, enabling knowledge transfer across a diverse stock universe.

## Disadvantages

1. **High computational cost and long training times.** TFT combines LSTMs, attention, and multiple gating networks, making it significantly more expensive to train than simpler models. For a universe of 500 stocks with 5 years of daily data, training can take hours on GPU hardware, limiting iteration speed.

2. **Large number of hyperparameters to tune.** Hidden dimensions, number of attention heads, dropout rates, LSTM layers, learning rate, quantile weights, and lookback window all require tuning. The high-dimensional hyperparameter space makes optimization expensive, especially with time-series cross-validation.

3. **Requires substantial training data.** Deep learning models are data-hungry, and TFT is no exception. For individual stock prediction, 5 years of daily data (1260 samples) may be insufficient, requiring cross-sectional pooling across stocks which introduces its own challenges.

4. **Risk of overfitting to market regimes.** Despite regularization, TFT can overfit to the market regime present in training data (e.g., a prolonged bull market). When regimes change, the model's complex learned patterns may fail catastrophically, and the failure modes are harder to diagnose than in simpler models.

5. **Black-box components despite interpretability features.** While attention weights and variable selection scores aid interpretation, the underlying GRN transformations and LSTM dynamics remain opaque. Attention weights can be misleading -- high attention to a time step does not necessarily mean causal influence on the prediction.

6. **Complex implementation and maintenance.** Production deployment of TFT requires GPU infrastructure, careful data pipelines for three input types, and monitoring of many internal components. Model updates and debugging are more challenging than for linear or tree-based models.

7. **Latency constraints for real-time trading.** TFT's inference involves sequential LSTM computation and multi-head attention, making it slower than simple models. For high-frequency or latency-sensitive strategies, the inference time may be prohibitive without model distillation or approximation.

## When to Use in Stock Market

- Multi-horizon portfolio return forecasting requiring consistent predictions across time scales
- When you have diverse feature types: static stock metadata, historical observations, and known future events
- Probabilistic forecasting for Value-at-Risk and position sizing decisions
- Cross-sectional models pooling data across many stocks with heterogeneous characteristics
- When interpretability of temporal dynamics is important for model validation
- Medium-frequency strategies (daily to weekly) where inference latency is not critical
- When you have sufficient data (multiple years, hundreds of stocks) to train deep models

## When NOT to Use in Stock Market

- High-frequency trading where microsecond latency matters
- Small datasets with only a few stocks and limited history (use ARIMA or linear models)
- When model simplicity and auditability are regulatory requirements (use ElasticNet)
- Univariate time series forecasting without additional features (use simpler models)
- When compute resources are limited (no GPU access)
- Rapid prototyping where fast iteration matters more than model sophistication
- When the prediction target is binary (up/down) rather than continuous returns

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Guidance |
|---|---|---|---|
| `hidden_size` | Dimension of hidden layers and embeddings | 32 to 256 | 64-128 for individual stocks; 128-256 for cross-sectional models |
| `n_heads` | Number of attention heads | 1 to 8 | 4 heads balances interpretability and capacity |
| `dropout` | Dropout rate for regularization | 0.1 to 0.4 | Higher (0.3-0.4) for noisy daily stock data |
| `lstm_layers` | Number of LSTM encoder/decoder layers | 1 to 3 | 1-2 layers usually sufficient; more layers risk overfitting |
| `lookback` | Number of historical time steps | 20 to 252 | 60 (3 months) for daily data; 252 for annual patterns |
| `max_horizon` | Maximum forecast horizon | 1 to 60 | Match your trading strategy: 20 for monthly rebalancing |
| `learning_rate` | Optimizer learning rate | 1e-4 to 1e-2 | Start at 1e-3 with cosine annealing schedule |
| `batch_size` | Training batch size | 32 to 256 | 64-128; larger for cross-sectional models |
| `quantiles` | Quantile levels to predict | [0.1, 0.5, 0.9] | Add 0.01 and 0.99 for tail risk estimation |
| `grad_clip` | Gradient clipping norm | 0.1 to 1.0 | 0.5 to prevent exploding gradients with volatile data |

## Stock Market Performance Tips

1. **Pool data across stocks for training.** Instead of training per-stock models, use a single cross-sectional model conditioned on static features. This dramatically increases effective sample size and enables knowledge transfer between stocks.

2. **Use log-returns as targets, not raw prices.** Log-returns are approximately stationary and prevent the model from merely learning price levels. Normalize returns by rolling volatility to stabilize the target distribution.

3. **Align calendar features with trading days.** Known future inputs should reflect the trading calendar (excluding weekends and holidays). Misaligned calendar features can confuse the model and introduce noise.

4. **Implement early stopping on validation quantile loss.** Monitor the quantile loss on a held-out validation period (not random split) and stop training when it stops improving. This prevents overfitting to training market conditions.

5. **Use exponential moving average of model weights.** EMA smoothing of weights during training produces more stable predictions, reducing the impact of noisy gradient updates from volatile market data.

6. **Evaluate calibration of prediction intervals.** Check that the 10th and 90th percentile predictions actually contain ~80% of true values. Miscalibrated intervals lead to incorrect risk estimates and position sizing.

7. **Monitor attention patterns for sanity checking.** Visualize attention weights to verify the model attends to meaningful historical periods. If attention patterns appear random or uniform, the temporal component may not be adding value.

## Comparison with Other Algorithms

| Aspect | TFT | LSTM | ARIMA | XGBoost | Prophet |
|---|---|---|---|---|---|
| Multi-horizon | Native | Loop-based | Per-horizon | Per-horizon | Native |
| Feature Types | 3 types | Homogeneous | Univariate | Homogeneous | Limited |
| Probabilistic | Quantiles | Possible | Confidence intervals | Possible | Yes |
| Interpretability | Attention + VSN | Low | Parameter-based | Feature importance | Decomposition |
| Non-linearity | High | High | Limited | High | Moderate |
| Training Data Needs | High | High | Low | Medium | Low |
| Inference Speed | Slow | Moderate | Fast | Fast | Moderate |
| Cross-sectional Learning | Yes | Possible | No | Possible | No |
| Best For (Finance) | Multi-horizon forecasting | Sequence patterns | Univariate | Tabular features | Trend decomposition |

## Real-World Stock Market Example

```python
import numpy as np

# ================================================================
# Temporal Fusion Transformer (Simplified) for Stock Forecasting
# Implements core TFT components from scratch using NumPy
# ================================================================

class GatedResidualNetwork:
    """Gated Residual Network for non-linear feature transformation."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights (Xavier initialization)
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)
        self.W_gate = np.random.randn(hidden_dim, output_dim) * scale2
        self.b_gate = np.zeros(output_dim)

        # Skip connection projection if dimensions differ
        if input_dim != output_dim:
            self.W_skip = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        else:
            self.W_skip = None

    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -10, 0)) - 1))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def layer_norm(self, x, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def forward(self, x):
        """Forward pass through GRN."""
        # Primary pathway
        eta1 = x @ self.W1 + self.b1
        eta1 = self.elu(eta1)
        eta2 = eta1 @ self.W2 + self.b2

        # Gating
        gate = self.sigmoid(eta1 @ self.W_gate + self.b_gate)
        gated_output = gate * eta2

        # Skip connection
        if self.W_skip is not None:
            skip = x @ self.W_skip
        else:
            skip = x

        # Residual + layer norm
        return self.layer_norm(skip + gated_output)


class VariableSelectionNetwork:
    """Variable Selection Network for adaptive feature importance."""

    def __init__(self, n_features, input_dim, hidden_dim):
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Per-feature GRNs
        self.feature_grns = [
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim)
            for _ in range(n_features)
        ]

        # Selection weights
        total_dim = n_features * input_dim
        scale = np.sqrt(2.0 / (total_dim + n_features))
        self.W_select = np.random.randn(total_dim, n_features) * scale
        self.b_select = np.zeros(n_features)

    def softmax(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, features_list):
        """
        features_list: list of n_features arrays, each shape (batch, input_dim)
        Returns: selected features (batch, hidden_dim), importance weights (batch, n_features)
        """
        batch_size = features_list[0].shape[0]

        # Concatenate all features for selection weights
        flat = np.concatenate(features_list, axis=-1)
        selection_weights = self.softmax(flat @ self.W_select + self.b_select)

        # Transform each feature through its GRN
        transformed = []
        for i, grn in enumerate(self.feature_grns):
            transformed.append(grn.forward(features_list[i]))

        # Weighted combination
        output = np.zeros((batch_size, self.hidden_dim))
        for i in range(self.n_features):
            output += selection_weights[:, i:i+1] * transformed[i]

        return output, selection_weights


class SimplifiedTFT:
    """
    Simplified Temporal Fusion Transformer for stock forecasting.
    Core components: VSN, GRN, simplified attention, quantile output.
    """

    def __init__(self, n_past_features, n_future_features, n_static_features,
                 hidden_dim=64, n_heads=4, lookback=60, horizon=20,
                 quantiles=[0.1, 0.5, 0.9]):
        self.n_past_features = n_past_features
        self.n_future_features = n_future_features
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.lookback = lookback
        self.horizon = horizon
        self.quantiles = quantiles
        self.head_dim = hidden_dim // n_heads

        # Static covariate encoder
        self.static_grn = GatedResidualNetwork(n_static_features, hidden_dim, hidden_dim)

        # Variable selection for past inputs
        self.past_vsn = VariableSelectionNetwork(n_past_features, 1, hidden_dim)

        # Temporal processing (simplified: linear projection instead of LSTM)
        scale = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W_temporal = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_temporal = np.zeros(hidden_dim)

        # Attention weights
        self.W_Q = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_K = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_V = np.random.randn(hidden_dim, hidden_dim) * scale

        # Output projection (one per quantile per horizon step)
        n_outputs = len(quantiles) * horizon
        out_scale = np.sqrt(2.0 / (hidden_dim + n_outputs))
        self.W_out = np.random.randn(hidden_dim, n_outputs) * out_scale
        self.b_out = np.zeros(n_outputs)

    def softmax(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def attention(self, Q, K, V):
        """Multi-head attention with interpretable averaging."""
        batch, seq_len, dim = Q.shape

        # Reshape for multi-head
        Q = Q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        K = K.reshape(batch, seq_len, self.n_heads, self.head_dim)
        V = V.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # Scaled dot-product attention per head
        scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / np.sqrt(self.head_dim)

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores += mask[np.newaxis, np.newaxis, :, :]

        attn_weights = self.softmax(scores, axis=-1)

        # Apply attention to values
        out = np.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        out = out.reshape(batch, seq_len, dim)

        # Average attention weights across heads for interpretability
        avg_attn = np.mean(attn_weights, axis=1)  # (batch, seq_len, seq_len)

        return out, avg_attn

    def forward(self, past_data, static_data):
        """
        past_data: (batch, lookback, n_past_features)
        static_data: (batch, n_static_features)
        Returns: predictions (batch, horizon, n_quantiles), attention_weights, feature_importance
        """
        batch_size = past_data.shape[0]

        # 1. Static covariate encoding
        static_encoded = self.static_grn.forward(static_data)  # (batch, hidden)

        # 2. Variable selection for each time step
        all_selected = []
        all_importance = []

        for t in range(self.lookback):
            features_list = [past_data[:, t, f:f+1] for f in range(self.n_past_features)]
            selected, importance = self.past_vsn.forward(features_list)
            # Add static context
            selected = selected + static_encoded * 0.1
            all_selected.append(selected)
            all_importance.append(importance)

        temporal_input = np.stack(all_selected, axis=1)  # (batch, lookback, hidden)
        feature_importance = np.stack(all_importance, axis=1)  # (batch, lookback, n_features)

        # 3. Temporal processing
        tanh_out = np.tanh(temporal_input @ self.W_temporal + self.b_temporal)
        temporal_features = temporal_input + tanh_out  # residual

        # 4. Self-attention
        Q = temporal_features @ self.W_Q
        K = temporal_features @ self.W_K
        V = temporal_features @ self.W_V

        attn_out, attn_weights = self.attention(Q, K, V)

        # 5. Aggregate temporal representation (use last position)
        final_repr = attn_out[:, -1, :]  # (batch, hidden)

        # 6. Quantile predictions
        raw_output = final_repr @ self.W_out + self.b_out  # (batch, n_quantiles * horizon)
        predictions = raw_output.reshape(batch_size, self.horizon, len(self.quantiles))

        # Sort quantiles to ensure monotonicity
        predictions = np.sort(predictions, axis=-1)

        return predictions, attn_weights, np.mean(feature_importance, axis=1)

    def quantile_loss(self, y_true, y_pred):
        """Compute quantile loss across all horizons and quantiles."""
        total_loss = 0.0
        for q_idx, q in enumerate(self.quantiles):
            errors = y_true - y_pred[:, :, q_idx]
            total_loss += np.mean(np.maximum(q * errors, (q - 1) * errors))
        return total_loss / len(self.quantiles)


# ================================================================
# Generate Stock Market Data
# ================================================================

np.random.seed(42)

n_stocks = 100
n_days = 1260
lookback = 60
horizon = 20
n_past_features = 8

feature_names = ['close_return', 'high_low_range', 'RSI_14', 'MACD_hist',
                 'log_volume', 'volume_ratio', 'VIX', 'treasury_spread']
static_feature_names = ['sector', 'market_cap', 'exchange', 'volatility_bucket']

print("Generating stock market data...")

# Generate data for multiple stocks
all_past_data = []
all_static_data = []
all_targets = []

for stock_id in range(n_stocks):
    # Simulate correlated features with some predictive signal
    returns = np.random.randn(n_days) * 0.02
    # Add momentum effect
    for t in range(1, n_days):
        returns[t] += 0.05 * returns[t-1]

    features = np.column_stack([
        returns,                                          # close_return
        np.abs(np.random.randn(n_days) * 0.015),         # high_low_range
        50 + np.cumsum(np.random.randn(n_days) * 2),     # RSI_14 (mean-reverting)
        np.random.randn(n_days) * 0.5,                   # MACD_hist
        15 + np.random.randn(n_days) * 0.5,              # log_volume
        1 + np.random.randn(n_days) * 0.3,               # volume_ratio
        20 + np.cumsum(np.random.randn(n_days) * 0.5),   # VIX
        0.5 + np.cumsum(np.random.randn(n_days) * 0.02), # treasury_spread
    ])

    # Clip RSI and VIX to reasonable ranges
    features[:, 2] = np.clip(features[:, 2], 10, 90)
    features[:, 6] = np.clip(features[:, 6], 9, 80)

    static = np.array([
        stock_id % 11,             # sector
        stock_id % 3,              # market_cap
        stock_id % 2,              # exchange
        stock_id % 5               # volatility_bucket
    ], dtype=float)

    # Create samples
    for t in range(lookback, n_days - horizon):
        all_past_data.append(features[t-lookback:t])
        all_static_data.append(static)
        all_targets.append(features[t:t+horizon, 0])  # future returns

all_past_data = np.array(all_past_data[:5000])    # Limit for demo
all_static_data = np.array(all_static_data[:5000])
all_targets = np.array(all_targets[:5000])

print(f"Dataset shape: {all_past_data.shape}")
print(f"Static data shape: {all_static_data.shape}")
print(f"Targets shape: {all_targets.shape}")

# ================================================================
# Train-Test Split (temporal)
# ================================================================

train_size = 4000
X_past_train = all_past_data[:train_size]
X_static_train = all_static_data[:train_size]
y_train = all_targets[:train_size]

X_past_test = all_past_data[train_size:]
X_static_test = all_static_data[train_size:]
y_test = all_targets[train_size:]

print(f"\nTrain: {train_size} samples, Test: {len(y_test)} samples")

# ================================================================
# Initialize and Run TFT
# ================================================================

print("\nInitializing Temporal Fusion Transformer...")
tft = SimplifiedTFT(
    n_past_features=n_past_features,
    n_future_features=4,
    n_static_features=4,
    hidden_dim=32,
    n_heads=4,
    lookback=lookback,
    horizon=horizon,
    quantiles=[0.1, 0.5, 0.9]
)

# Forward pass on a batch
batch_size = 64
batch_past = X_past_train[:batch_size]
batch_static = X_static_train[:batch_size]
batch_targets = y_train[:batch_size]

print(f"\nRunning forward pass on batch of {batch_size}...")
predictions, attn_weights, feature_importance = tft.forward(batch_past, batch_static)

print(f"Predictions shape: {predictions.shape}")  # (batch, horizon, n_quantiles)
print(f"Attention weights shape: {attn_weights.shape}")
print(f"Feature importance shape: {feature_importance.shape}")

# Compute initial loss
loss = tft.quantile_loss(batch_targets, predictions)
print(f"Initial quantile loss: {loss:.6f}")

# ================================================================
# Analyze Feature Importance
# ================================================================

print("\n=== Variable Selection Analysis ===")
avg_importance = np.mean(feature_importance, axis=0)
sorted_idx = np.argsort(avg_importance)[::-1]

for rank, idx in enumerate(sorted_idx):
    bar = "#" * int(avg_importance[idx] * 100)
    print(f"  {rank+1}. {feature_names[idx]:<20s}: {avg_importance[idx]:.4f} {bar}")

# ================================================================
# Analyze Temporal Attention Patterns
# ================================================================

print("\n=== Temporal Attention Analysis ===")
# Average attention from last position to all historical positions
avg_attn = np.mean(attn_weights[:, -1, :], axis=0)  # average across batch
recent_attn = np.sum(avg_attn[-10:])
mid_attn = np.sum(avg_attn[-30:-10])
distant_attn = np.sum(avg_attn[:-30])

print(f"Attention to recent 10 days: {recent_attn:.4f}")
print(f"Attention to mid 11-30 days: {mid_attn:.4f}")
print(f"Attention to distant 31-60 days: {distant_attn:.4f}")

# ================================================================
# Prediction Interval Analysis
# ================================================================

print("\n=== Prediction Intervals (sample stock) ===")
sample_pred = predictions[0]  # First sample
print(f"{'Horizon':<10} {'P10':>10} {'P50 (median)':>15} {'P90':>10} {'Width':>10}")
print("-" * 55)
for h in [0, 4, 9, 19]:
    width = sample_pred[h, 2] - sample_pred[h, 0]
    print(f"Day {h+1:<5} {sample_pred[h,0]:>10.4f} {sample_pred[h,1]:>15.4f} "
          f"{sample_pred[h,2]:>10.4f} {width:>10.4f}")

# ================================================================
# Evaluate on Test Set
# ================================================================

print("\n=== Test Set Evaluation ===")
# Process in batches
test_preds = []
for i in range(0, len(X_past_test), batch_size):
    end = min(i + batch_size, len(X_past_test))
    preds, _, _ = tft.forward(X_past_test[i:end], X_static_test[i:end])
    test_preds.append(preds)

test_preds = np.concatenate(test_preds, axis=0)
test_loss = tft.quantile_loss(y_test, test_preds)
print(f"Test quantile loss: {test_loss:.6f}")

# Evaluate median predictions (P50) at different horizons
for h_idx, h in enumerate([1, 5, 20]):
    median_pred = test_preds[:, h-1, 1]  # P50
    actual = y_test[:, h-1]

    mse = np.mean((actual - median_pred) ** 2)
    direction_accuracy = np.mean(np.sign(median_pred) == np.sign(actual))
    ic = np.corrcoef(actual, median_pred)[0, 1]

    print(f"\nHorizon {h}-day:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Direction accuracy: {direction_accuracy:.2%}")
    print(f"  Information Coefficient: {ic:.4f}")

# Calibration check
print("\n=== Prediction Interval Calibration ===")
for h in [1, 5, 20]:
    actual = y_test[:, h-1]
    lower = test_preds[:, h-1, 0]  # P10
    upper = test_preds[:, h-1, 2]  # P90
    coverage = np.mean((actual >= lower) & (actual <= upper))
    print(f"  {h}-day: {coverage:.1%} coverage (target: 80%)")
```

## Key Takeaways

1. **TFT is the most sophisticated forecasting architecture for multi-horizon stock prediction**, combining temporal processing, attention mechanisms, and variable selection into a unified framework that handles the heterogeneous nature of financial data.

2. **The three-input-type design matches stock market data naturally** -- static stock attributes, historical observations, and known future events (earnings dates, rebalancing schedules) are all first-class inputs with dedicated processing pathways.

3. **Interpretability features set TFT apart from other deep learning models** -- variable selection weights and attention patterns provide actionable insights into model behavior, helping traders build trust and diagnose failures.

4. **Quantile outputs are essential for risk-aware trading** -- prediction intervals directly inform position sizing, stop-loss placement, and portfolio-level risk budgeting, going beyond simple point forecasts.

5. **Cross-sectional training is key to making TFT work for stocks** -- pooling data across hundreds of stocks with static conditioning provides the data volume deep learning requires while allowing stock-specific adaptation.

6. **TFT is best suited for medium-frequency strategies** where its computational overhead is acceptable and the multi-horizon probabilistic outputs provide meaningful value over simpler alternatives.
