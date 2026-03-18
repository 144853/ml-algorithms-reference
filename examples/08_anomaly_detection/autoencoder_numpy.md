# Autoencoder - Complete Guide with Stock Market Applications

## Overview

An Autoencoder is a neural network architecture trained to reconstruct its input through a compressed intermediate representation (bottleneck). It consists of an encoder that maps high-dimensional input to a lower-dimensional latent space, and a decoder that reconstructs the original input from the compressed representation. When used for anomaly detection, the key insight is that the autoencoder is trained on normal data, so it learns to reconstruct normal patterns well but produces high reconstruction errors for anomalous inputs that differ from the training distribution.

In stock market applications, autoencoders are particularly effective for detecting market manipulation and unusual order flow patterns. Market manipulation -- including wash trading, spoofing, layering, and pump-and-dump schemes -- produces order flow signatures that deviate from normal trading patterns. An autoencoder trained on legitimate trading data learns the manifold of normal order flow, and manipulative patterns that fall off this manifold produce elevated reconstruction errors that trigger alerts. This approach is especially powerful because manipulation tactics evolve constantly, and the autoencoder can detect novel manipulation methods without having seen them before, unlike supervised classifiers trained on historical examples.

The architecture's flexibility allows it to model complex, non-linear relationships between order flow features -- something that linear methods and even tree-based models struggle with. Variants like variational autoencoders (VAEs) provide probabilistic anomaly scores with calibrated uncertainty, while denoising autoencoders add robustness by learning to reconstruct clean signals from noisy inputs, matching the inherently noisy nature of market data.

## How It Works - The Math Behind It

### Encoder Network

The encoder maps input `x` to a latent representation `z`:

```
h_1 = activation(W_1 @ x + b_1)           (first hidden layer)
h_2 = activation(W_2 @ h_1 + b_2)         (second hidden layer)
z   = W_z @ h_2 + b_z                      (latent/bottleneck layer)
```

where the latent dimension `d_z << d_x` (input dimension).

### Decoder Network

The decoder reconstructs the input from `z`:

```
h_3 = activation(W_3 @ z + b_3)           (first decoder layer)
h_4 = activation(W_4 @ h_3 + b_4)         (second decoder layer)
x_hat = W_out @ h_4 + b_out                (reconstruction)
```

### Loss Function (Reconstruction Error)

Mean Squared Error (MSE) loss:

```
L_MSE(x, x_hat) = (1/d) * sum_{j=1}^{d} (x_j - x_hat_j)^2
```

For binary features, Binary Cross-Entropy:

```
L_BCE(x, x_hat) = -(1/d) * sum_j [x_j * log(x_hat_j) + (1-x_j) * log(1-x_hat_j)]
```

### Anomaly Score

The anomaly score for a new observation is its reconstruction error:

```
anomaly_score(x) = ||x - Decoder(Encoder(x))||^2
```

Threshold: `anomaly if anomaly_score(x) > mu + k * sigma`

where `mu` and `sigma` are the mean and standard deviation of reconstruction errors on normal training data, and `k` is typically 2-3.

### Variational Autoencoder (VAE) Extension

The VAE encoder outputs parameters of a distribution:

```
mu_z = W_mu @ h_2 + b_mu
log_var_z = W_var @ h_2 + b_var
z = mu_z + exp(0.5 * log_var_z) * epsilon,  epsilon ~ N(0, I)
```

VAE loss includes a KL divergence term:

```
L_VAE = L_reconstruction + beta * KL(q(z|x) || p(z))
KL = -0.5 * sum(1 + log_var_z - mu_z^2 - exp(log_var_z))
```

The KL term can also serve as an anomaly signal -- anomalous inputs produce latent distributions far from the prior N(0,I).

### Backpropagation and Training

Gradients for weight updates using chain rule:

```
dL/dW_out = dL/dx_hat * dh_4^T
dL/dW_4 = (W_out^T @ dL/dx_hat * activation'(h_4)) @ h_3^T
...continuing backward through all layers...
```

Weight update via Adam optimizer:

```
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
m_hat = m_t / (1 - beta_1^t)
v_hat = v_t / (1 - beta_2^t)
W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
```

## Stock Market Use Case: Detecting Market Manipulation and Unusual Order Flow Patterns

### The Problem

A regulatory body (like the SEC or FCA) monitors order flow data across all listed stocks to detect market manipulation. The challenge is immense:
- Millions of orders per day across thousands of stocks
- Manipulation tactics constantly evolve to evade rule-based detection
- False positives waste expensive investigator time
- New manipulation methods must be detected without prior examples
- Normal trading patterns vary significantly across stocks, times, and market conditions

The autoencoder approach learns the complex manifold of normal order flow and flags deviations, catching both known and novel manipulation patterns.

### Stock Market Features (Input Data)

| Feature Name | Description | Example Value | Manipulation Signal |
|---|---|---|---|
| order_rate | Orders per second (z-score) | 0.8 | Spoofing: rapid-fire orders |
| cancel_rate | Cancel-to-order ratio | 0.15 | Spoofing: high cancel rate |
| cancel_latency | Avg time before cancel (ms, log) | 5.2 | Spoofing: very fast cancels |
| trade_to_order | Trade-to-order ratio | 0.35 | Wash trading: unusual ratio |
| self_trade_rate | Self-cross rate (same entity) | 0.02 | Wash trading: self-dealing |
| price_impact_asym | Buy vs sell price impact asymmetry | 0.05 | Layering: directional pressure |
| depth_oscillation | Orderbook depth oscillation freq | 0.3 | Layering: depth manipulation |
| spread_deviation | Bid-ask spread vs normal (z-score) | 0.4 | Market making anomaly |
| volume_concentration | Top-5 participant volume share | 0.45 | Dominance/manipulation |
| tick_direction_run | Consecutive same-direction ticks | 3 | Momentum ignition |
| cross_market_lag | Price lead-lag vs related markets | 0.1 | Front-running |
| odd_lot_ratio | Fraction of odd-lot orders | 0.08 | Retail flow anomaly |
| hidden_order_pct | Hidden/iceberg order percentage | 0.12 | Institutional signal |
| time_clustering | Order arrival burstiness (CoV) | 0.9 | Algorithmic pattern |
| size_pattern_entropy | Entropy of order size distribution | 2.1 | Repeated size = manipulation |
| counterparty_diversity | Unique counterparties (normalized) | 0.7 | Wash trading: few counterparties |

### Example Data Structure

```python
import numpy as np

# Order flow feature data for autoencoder-based manipulation detection
np.random.seed(42)

n_observations = 30000   # 30K time windows (e.g., 5-minute intervals)
n_features = 16

feature_names = [
    'order_rate', 'cancel_rate', 'cancel_latency', 'trade_to_order',
    'self_trade_rate', 'price_impact_asym', 'depth_oscillation',
    'spread_deviation', 'volume_concentration', 'tick_direction_run',
    'cross_market_lag', 'odd_lot_ratio', 'hidden_order_pct',
    'time_clustering', 'size_pattern_entropy', 'counterparty_diversity'
]

# Normal order flow (97%)
n_normal = int(n_observations * 0.97)
X_normal = np.column_stack([
    np.random.randn(n_normal) * 1.0,               # order_rate (z-score)
    np.random.beta(2, 10, n_normal),                 # cancel_rate (~0.17)
    5 + np.random.randn(n_normal) * 1.5,            # cancel_latency (log ms)
    np.random.beta(5, 8, n_normal),                  # trade_to_order (~0.38)
    np.random.beta(1, 50, n_normal),                 # self_trade_rate (~0.02)
    np.random.randn(n_normal) * 0.1,                # price_impact_asym
    np.random.beta(3, 7, n_normal),                  # depth_oscillation
    np.random.randn(n_normal) * 0.5,                # spread_deviation
    np.random.beta(5, 5, n_normal),                  # volume_concentration
    np.random.poisson(3, n_normal).astype(float),    # tick_direction_run
    np.abs(np.random.randn(n_normal)) * 0.1,        # cross_market_lag
    np.random.beta(2, 20, n_normal),                 # odd_lot_ratio
    np.random.beta(3, 20, n_normal),                 # hidden_order_pct
    1 + np.abs(np.random.randn(n_normal)) * 0.3,    # time_clustering
    2 + np.random.randn(n_normal) * 0.5,            # size_pattern_entropy
    0.7 + np.random.randn(n_normal) * 0.15,         # counterparty_diversity
])

# Spoofing anomalies (1%)
n_spoof = int(n_observations * 0.01)
X_spoof = np.column_stack([
    3 + np.random.randn(n_spoof) * 1.0,              # high order rate
    0.7 + np.random.beta(5, 2, n_spoof) * 0.3,       # very high cancel rate!
    2 + np.random.randn(n_spoof) * 0.5,              # fast cancels (low latency)
    np.random.beta(2, 15, n_spoof),                    # low trade-to-order
    np.random.beta(1, 50, n_spoof),                    # normal self-trade
    np.random.randn(n_spoof) * 0.15 + 0.3,           # some price impact asym
    0.6 + np.random.beta(5, 3, n_spoof) * 0.4,       # high depth oscillation
    np.random.randn(n_spoof) * 0.8 + 1.5,            # wide spreads
    0.6 + np.random.beta(7, 3, n_spoof) * 0.4,       # concentrated volume
    np.random.poisson(3, n_spoof).astype(float),       # normal ticks
    np.abs(np.random.randn(n_spoof)) * 0.1,           # normal lag
    np.random.beta(2, 20, n_spoof),                    # normal odd lots
    np.random.beta(3, 20, n_spoof),                    # normal hidden orders
    2 + np.abs(np.random.randn(n_spoof)) * 0.5,       # bursty orders
    0.5 + np.random.randn(n_spoof) * 0.3,            # low entropy (repetitive!)
    0.7 + np.random.randn(n_spoof) * 0.15,            # normal diversity
])

# Wash trading anomalies (1%)
n_wash = int(n_observations * 0.01)
X_wash = np.column_stack([
    1 + np.random.randn(n_wash) * 0.8,               # moderate order rate
    np.random.beta(3, 10, n_wash),                     # moderate cancel rate
    5 + np.random.randn(n_wash) * 1.0,               # normal cancel latency
    0.8 + np.random.beta(8, 2, n_wash) * 0.2,        # very high trade-to-order!
    0.15 + np.random.beta(3, 5, n_wash) * 0.3,       # high self-trade rate!!
    np.random.randn(n_wash) * 0.05,                   # minimal price impact
    np.random.beta(3, 7, n_wash),                      # normal oscillation
    np.random.randn(n_wash) * 0.3,                    # normal spread
    0.7 + np.random.beta(7, 3, n_wash) * 0.3,        # concentrated
    np.random.poisson(3, n_wash).astype(float),        # normal ticks
    np.abs(np.random.randn(n_wash)) * 0.1,            # normal lag
    np.random.beta(2, 20, n_wash),                     # normal odd lots
    np.random.beta(3, 20, n_wash),                     # normal hidden
    1 + np.abs(np.random.randn(n_wash)) * 0.3,        # normal timing
    1.8 + np.random.randn(n_wash) * 0.3,             # moderate entropy
    0.2 + np.random.beta(2, 8, n_wash) * 0.3,        # low counterparty diversity!
])

# Pump-and-dump anomalies (1%)
n_pump = n_observations - n_normal - n_spoof - n_wash
X_pump = np.column_stack([
    4 + np.random.randn(n_pump) * 1.5,               # very high order rate
    np.random.beta(2, 8, n_pump),                      # normal cancel rate
    5 + np.random.randn(n_pump) * 1.5,               # normal cancel latency
    0.5 + np.random.beta(5, 5, n_pump) * 0.3,        # moderate trade-to-order
    np.random.beta(1, 30, n_pump),                     # normal self-trade
    np.random.randn(n_pump) * 0.1 + 0.5,             # strong positive impact
    np.random.beta(3, 7, n_pump),                      # normal oscillation
    np.random.randn(n_pump) * 0.4 + 0.5,             # some spread deviation
    0.6 + np.random.beta(5, 3, n_pump) * 0.4,        # concentrated
    8 + np.random.poisson(5, n_pump).astype(float),   # long tick runs!
    np.abs(np.random.randn(n_pump)) * 0.3 + 0.2,     # elevated lag
    np.random.beta(5, 10, n_pump),                     # some odd lots
    np.random.beta(2, 20, n_pump),                     # normal hidden
    3 + np.abs(np.random.randn(n_pump)) * 0.5,        # very bursty!
    1 + np.random.randn(n_pump) * 0.4,               # low entropy
    0.3 + np.random.beta(2, 8, n_pump) * 0.3,        # low diversity
])

X = np.vstack([X_normal, X_spoof, X_wash, X_pump])
y_true = np.concatenate([
    np.zeros(n_normal),
    np.ones(n_spoof),
    2 * np.ones(n_wash),
    3 * np.ones(n_pump)
])
y_binary = (y_true > 0).astype(int)

perm = np.random.permutation(n_observations)
X = X[perm]
y_true = y_true[perm]
y_binary = y_binary[perm]

print(f"Dataset: {X.shape}")
print(f"Normal: {np.sum(y_true==0)}, Spoofing: {np.sum(y_true==1)}, "
      f"Wash trading: {np.sum(y_true==2)}, Pump-and-dump: {np.sum(y_true==3)}")
```

### The Model in Action

The autoencoder processes order flow data through a compression-reconstruction pipeline:

1. **Training on normal data only**: The autoencoder is trained exclusively on legitimate order flow windows. It learns the statistical manifold of normal trading: typical cancel rates, order-to-trade ratios, timing patterns, and their complex interdependencies.

2. **Compression through bottleneck**: The encoder compresses 16 order flow features into a 4-6 dimensional latent space. This forces the network to learn the most salient patterns of normal trading, discarding noise.

3. **Reconstruction**: The decoder attempts to reconstruct the full 16-feature input from the compressed representation. For normal trading patterns, reconstruction is accurate (low error). For manipulation patterns that deviate from the learned manifold, reconstruction is poor (high error).

4. **Anomaly scoring**: Each observation receives an anomaly score equal to its reconstruction error. Spoofing with extreme cancel rates, wash trading with unusual self-trade patterns, and pump-and-dump with coordinated volume all produce elevated reconstruction errors because they lie off the normal manifold.

5. **Feature-level analysis**: By examining which features have the highest per-feature reconstruction error, analysts can identify the specific aspects of order flow that are anomalous, guiding investigation focus (e.g., "anomaly driven primarily by cancel_rate and cancel_latency" suggests spoofing).

## Advantages

1. **Detects novel manipulation methods without prior examples.** Unlike supervised classifiers that can only catch manipulation patterns present in training data, the autoencoder flags anything that deviates from normal -- including entirely new tactics. This is critical as manipulation strategies constantly evolve to evade known detection rules.

2. **Captures complex non-linear relationships in order flow.** Financial order flow features interact in complex ways -- the relationship between cancel rate and order size depends on market conditions, stock liquidity, and time of day. The neural network architecture models these non-linear dependencies that simpler methods miss.

3. **Feature-level reconstruction error provides interpretability.** By examining which features contribute most to the total reconstruction error for a flagged observation, analysts can quickly identify the suspicious aspect of the order flow. This interpretable breakdown guides investigation and supports regulatory evidence gathering.

4. **Handles high-dimensional feature spaces efficiently.** As surveillance teams add more features (orderbook depth, cross-market signals, participant network metrics), the autoencoder scales naturally. The bottleneck architecture prevents curse-of-dimensionality issues by learning a compact representation.

5. **Continuous anomaly scores enable flexible thresholding.** Rather than binary anomaly/normal classification, the reconstruction error provides a continuous severity score. This allows the surveillance team to prioritize investigations by severity and adjust alert thresholds based on available investigator capacity.

6. **Latent space enables clustering and visualization.** The compressed latent representations can be clustered and visualized (e.g., via t-SNE) to discover groups of similar anomalies, potentially revealing coordinated manipulation campaigns or identifying new manipulation archetypes.

7. **Transfer learning across stocks and markets.** An autoencoder trained on normal order flow for liquid large-cap stocks can be partially adapted to small-cap stocks or different exchanges, reducing the data requirements for expanding surveillance coverage.

## Disadvantages

1. **Requires clean normal training data.** If manipulation patterns are present in the training data (undetected historical manipulation), the autoencoder learns to reconstruct them as normal, reducing detection sensitivity. Obtaining guaranteed-clean training data is challenging for market surveillance.

2. **Reconstruction error threshold is subjective.** Choosing the anomaly threshold requires balancing false positives (wasted investigator time) and false negatives (missed manipulation). There is no principled way to set this threshold without labeled validation data.

3. **Sensitive to input distribution shift.** Market microstructure changes over time -- new exchange regulations, market structure reforms, or seasonal patterns can shift the normal distribution. Without regular retraining, the autoencoder's definition of "normal" becomes stale, increasing false positive rates.

4. **Training instability and hyperparameter sensitivity.** Neural network training involves many choices: architecture depth/width, learning rate, regularization, batch size, and training epochs. Poor choices lead to underfitting (everything looks normal) or overfitting (everything looks anomalous). Financial data's noise amplifies this sensitivity.

5. **Black-box latent representations.** While per-feature reconstruction error aids interpretation, the latent space encoding is opaque. Regulators and compliance teams may question why a specific observation was flagged, and the autoencoder cannot provide a rule-based explanation.

6. **Cannot handle sequential patterns directly.** Standard autoencoders treat each observation independently. Sequential manipulation patterns (e.g., spoofing that builds up over minutes or days) require recurrent or convolutional autoencoder variants, adding architectural complexity.

7. **Computational overhead for real-time processing.** Neural network inference, while fast on GPUs, may be slower than simple rule-based checks or tree-based models on CPUs. For markets processing millions of messages per second, this latency can be a constraint for intraday surveillance.

## When to Use in Stock Market

- Market manipulation detection where novel tactics must be caught without prior examples
- Order flow surveillance across thousands of stocks simultaneously
- Building anomaly scores for prioritizing compliance investigations
- Detecting unusual trading patterns around corporate events (earnings, M&A)
- Identifying coordinated trading activity across related instruments
- Cross-market surveillance where feature interactions are complex
- When you need both an anomaly score and feature-level explanations

## When NOT to Use in Stock Market

- When simple rule-based checks suffice (e.g., "flag if volume > 10x average")
- For labeled classification problems with ample historical examples (use supervised learning)
- When real-time sub-millisecond latency is required (use simpler models)
- For sequential pattern detection requiring memory (use LSTM/attention autoencoders)
- When training data is known to be contaminated with anomalies and cannot be cleaned
- For small feature sets (<5 features) where simpler methods are equally effective
- When complete model interpretability is a regulatory requirement

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Guidance |
|---|---|---|---|
| `latent_dim` | Bottleneck dimensionality | 2 to 32 | 4-8 for 16 features; should be 25-50% of input dim |
| `hidden_layers` | Encoder/decoder layer sizes | [64, 32] to [256, 128, 64] | Deeper for complex order flow; [64, 32] for standard |
| `activation` | Hidden layer activation | ReLU, ELU, tanh | ReLU for standard; ELU if vanishing gradients occur |
| `learning_rate` | Optimizer learning rate | 1e-4 to 1e-2 | 1e-3 with Adam; use scheduler for fine-tuning |
| `batch_size` | Training batch size | 32 to 512 | 128-256 for order flow data; larger for stable gradients |
| `epochs` | Training iterations | 50 to 500 | Use early stopping on validation reconstruction error |
| `dropout` | Dropout rate for regularization | 0.0 to 0.5 | 0.1-0.2 to prevent overfitting to normal patterns |
| `threshold_k` | Number of std devs for threshold | 2 to 4 | 3 for balanced precision/recall; 2 for high recall |
| `loss_function` | MSE vs Huber vs BCE | - | Huber for robustness to outliers in training data |
| `weight_decay` | L2 regularization on weights | 1e-5 to 1e-3 | 1e-4 to prevent memorization of training noise |

## Stock Market Performance Tips

1. **Train exclusively on clean normal data.** Filter the training set rigorously to exclude known anomalous periods (flash crash days, halt events, known manipulation cases). Even a small fraction of anomalies in training data degrades detection performance.

2. **Use per-feature reconstruction error for root cause analysis.** Instead of just the total reconstruction error, compute the error for each feature independently. This creates an "anomaly fingerprint" that helps classify the type of manipulation.

3. **Implement rolling retraining with forgetting.** Retrain the autoencoder monthly on the most recent 6-12 months of normal data. This adapts to evolving market microstructure without requiring manual intervention.

4. **Apply feature normalization carefully.** Use robust scaling (median and IQR) rather than z-scores to handle the heavy-tailed distributions common in order flow data. Re-compute normalization statistics at each retraining.

5. **Ensemble multiple autoencoders.** Train autoencoders with different architectures, random seeds, and feature subsets. Average their anomaly scores for more robust detection that is less sensitive to any single model's biases.

6. **Monitor the reconstruction error distribution for drift.** Track the mean and variance of reconstruction errors over time on a held-out normal validation set. Significant drift indicates the model is becoming stale and needs retraining.

7. **Combine autoencoder scores with domain rules.** Use the autoencoder score as one input among several, including traditional surveillance rules, statistical tests, and cross-market checks. The ensemble catches both known patterns (rules) and novel anomalies (autoencoder).

## Comparison with Other Algorithms

| Aspect | Autoencoder | Isolation Forest | One-Class SVM | LOF | VAE | Statistical Tests |
|---|---|---|---|---|---|---|
| Non-linear Patterns | Yes | Limited (axis-parallel) | Kernel-dependent | No | Yes | No |
| Feature Interactions | Captures | Ignores | Kernel-based | Local | Captures | Manual |
| Interpretability | Feature-level errors | Path analysis | Low | Low | Latent + errors | High |
| Training Speed | Moderate (GPU) | Fast | Slow (O(n^2)) | Slow | Moderate (GPU) | Very fast |
| Novel Anomalies | Excellent | Good | Moderate | Good | Excellent | Poor |
| Sequential Patterns | Requires variant | No | No | No | Requires variant | No |
| Probabilistic Score | No (MSE) | Yes (path) | Yes (distance) | Yes (density) | Yes (ELBO) | Yes (p-value) |
| Best For (Finance) | Complex manipulation | General screening | Defined normal class | Local anomalies | Probabilistic | Simple features |

## Real-World Stock Market Example

```python
import numpy as np

# ================================================================
# Autoencoder: Market Manipulation Detection from Scratch
# ================================================================

class Autoencoder:
    """
    Autoencoder for anomaly detection in order flow data.
    Trained on normal data; high reconstruction error = anomaly.
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], latent_dim=8,
                 learning_rate=0.001, dropout_rate=0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.dropout_rate = dropout_rate

        # Build symmetric architecture
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]

        self.encoder_weights = []
        self.encoder_biases = []
        self.decoder_weights = []
        self.decoder_biases = []

        # Initialize encoder weights (He initialization)
        for i in range(len(encoder_dims) - 1):
            scale = np.sqrt(2.0 / encoder_dims[i])
            W = np.random.randn(encoder_dims[i], encoder_dims[i+1]) * scale
            b = np.zeros(encoder_dims[i+1])
            self.encoder_weights.append(W)
            self.encoder_biases.append(b)

        # Initialize decoder weights
        for i in range(len(decoder_dims) - 1):
            scale = np.sqrt(2.0 / decoder_dims[i])
            W = np.random.randn(decoder_dims[i], decoder_dims[i+1]) * scale
            b = np.zeros(decoder_dims[i+1])
            self.decoder_weights.append(W)
            self.decoder_biases.append(b)

        # Adam optimizer state
        self.m_w = [[np.zeros_like(w) for w in self.encoder_weights + self.decoder_weights]]
        self.v_w = [[np.zeros_like(w) for w in self.encoder_weights + self.decoder_weights]]
        self.t = 0

        # Normalization parameters
        self.mean = None
        self.std = None

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def encode(self, x, training=False):
        """Forward pass through encoder."""
        activations = [x]
        h = x

        for i, (W, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            z = h @ W + b
            if i < len(self.encoder_weights) - 1:  # ReLU for hidden, linear for latent
                h = self.relu(z)
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1-self.dropout_rate, h.shape) / (1-self.dropout_rate)
                    h = h * mask
            else:
                h = z  # Linear activation for latent layer
            activations.append(h)

        return h, activations

    def decode(self, z, training=False):
        """Forward pass through decoder."""
        activations = [z]
        h = z

        for i, (W, b) in enumerate(zip(self.decoder_weights, self.decoder_biases)):
            z_layer = h @ W + b
            if i < len(self.decoder_weights) - 1:  # ReLU for hidden, linear for output
                h = self.relu(z_layer)
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1-self.dropout_rate, h.shape) / (1-self.dropout_rate)
                    h = h * mask
            else:
                h = z_layer  # Linear output for reconstruction
            activations.append(h)

        return h, activations

    def forward(self, x, training=False):
        """Full forward pass: encode then decode."""
        latent, enc_acts = self.encode(x, training)
        reconstruction, dec_acts = self.decode(latent, training)
        return reconstruction, latent, enc_acts, dec_acts

    def compute_loss(self, x, x_hat):
        """Mean squared error loss."""
        return np.mean((x - x_hat) ** 2)

    def backward(self, x, x_hat, enc_acts, dec_acts):
        """Backpropagation through the entire network."""
        batch_size = x.shape[0]

        # Output gradient: dL/dx_hat = 2 * (x_hat - x) / (batch_size * input_dim)
        grad = 2.0 * (x_hat - x) / (batch_size * self.input_dim)

        # Decoder gradients
        dec_grads_w = []
        dec_grads_b = []

        for i in range(len(self.decoder_weights) - 1, -1, -1):
            dW = dec_acts[i].T @ grad / batch_size
            db = np.mean(grad, axis=0)
            dec_grads_w.insert(0, dW)
            dec_grads_b.insert(0, db)

            grad = grad @ self.decoder_weights[i].T
            if i > 0:
                grad = grad * self.relu_grad(dec_acts[i])

        # Encoder gradients
        enc_grads_w = []
        enc_grads_b = []

        for i in range(len(self.encoder_weights) - 1, -1, -1):
            dW = enc_acts[i].T @ grad / batch_size
            db = np.mean(grad, axis=0)
            enc_grads_w.insert(0, dW)
            enc_grads_b.insert(0, db)

            grad = grad @ self.encoder_weights[i].T
            if i > 0:
                grad = grad * self.relu_grad(enc_acts[i])

        return enc_grads_w, enc_grads_b, dec_grads_w, dec_grads_b

    def update_weights(self, enc_gw, enc_gb, dec_gw, dec_gb, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update."""
        self.t += 1

        all_weights = self.encoder_weights + self.decoder_weights
        all_biases = self.encoder_biases + self.decoder_biases
        all_gw = enc_gw + dec_gw
        all_gb = enc_gb + dec_gb

        for i in range(len(all_weights)):
            # Weights
            if not hasattr(self, '_m_w'):
                self._m_w = [np.zeros_like(w) for w in all_weights]
                self._v_w = [np.zeros_like(w) for w in all_weights]
                self._m_b = [np.zeros_like(b) for b in all_biases]
                self._v_b = [np.zeros_like(b) for b in all_biases]

            self._m_w[i] = beta1 * self._m_w[i] + (1 - beta1) * all_gw[i]
            self._v_w[i] = beta2 * self._v_w[i] + (1 - beta2) * all_gw[i]**2
            m_hat = self._m_w[i] / (1 - beta1**self.t)
            v_hat = self._v_w[i] / (1 - beta2**self.t)
            all_weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            # Biases
            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * all_gb[i]
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * all_gb[i]**2
            m_hat_b = self._m_b[i] / (1 - beta1**self.t)
            v_hat_b = self._v_b[i] / (1 - beta2**self.t)
            all_biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # Write back
        n_enc = len(self.encoder_weights)
        self.encoder_weights = all_weights[:n_enc]
        self.encoder_biases = all_biases[:n_enc]
        self.decoder_weights = all_weights[n_enc:]
        self.decoder_biases = all_biases[n_enc:]

    def fit(self, X, epochs=100, batch_size=128, verbose=True):
        """Train the autoencoder on normal data."""
        # Normalize
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.mean) / self.std

        n_samples = X_norm.shape[0]
        losses = []

        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = X_norm[perm[start:end]]

                # Forward
                x_hat, latent, enc_acts, dec_acts = self.forward(batch, training=True)
                loss = self.compute_loss(batch, x_hat)
                epoch_loss += loss
                n_batches += 1

                # Backward
                enc_gw, enc_gb, dec_gw, dec_gb = self.backward(batch, x_hat, enc_acts, dec_acts)

                # Update
                self.update_weights(enc_gw, enc_gb, dec_gw, dec_gb)

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")

        return losses

    def reconstruction_error(self, X):
        """Compute per-sample reconstruction error."""
        X_norm = (X - self.mean) / self.std
        x_hat, _, _, _ = self.forward(X_norm, training=False)
        errors = np.mean((X_norm - x_hat) ** 2, axis=1)
        return errors

    def feature_reconstruction_error(self, X):
        """Compute per-sample, per-feature reconstruction error."""
        X_norm = (X - self.mean) / self.std
        x_hat, _, _, _ = self.forward(X_norm, training=False)
        return (X_norm - x_hat) ** 2

    def get_latent(self, X):
        """Get latent representations."""
        X_norm = (X - self.mean) / self.std
        latent, _ = self.encode(X_norm, training=False)
        return latent

    def set_threshold(self, X_normal, k=3):
        """Set anomaly threshold from normal data."""
        errors = self.reconstruction_error(X_normal)
        self.error_mean = np.mean(errors)
        self.error_std = np.std(errors)
        self.threshold = self.error_mean + k * self.error_std
        return self.threshold

    def predict(self, X):
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        errors = self.reconstruction_error(X)
        return (errors > self.threshold).astype(int)


# ================================================================
# Generate Order Flow Data
# ================================================================

np.random.seed(42)

n_total = 20000
n_features = 16

feature_names = [
    'order_rate', 'cancel_rate', 'cancel_latency', 'trade_to_order',
    'self_trade_rate', 'price_impact_asym', 'depth_oscillation',
    'spread_deviation', 'volume_concentration', 'tick_direction_run',
    'cross_market_lag', 'odd_lot_ratio', 'hidden_order_pct',
    'time_clustering', 'size_pattern_entropy', 'counterparty_diversity'
]

# Normal data (97%)
n_normal = int(n_total * 0.97)
X_normal = np.column_stack([
    np.random.randn(n_normal) * 1.0,
    np.random.beta(2, 10, n_normal),
    5 + np.random.randn(n_normal) * 1.5,
    np.random.beta(5, 8, n_normal),
    np.random.beta(1, 50, n_normal),
    np.random.randn(n_normal) * 0.1,
    np.random.beta(3, 7, n_normal),
    np.random.randn(n_normal) * 0.5,
    np.random.beta(5, 5, n_normal),
    np.random.poisson(3, n_normal).astype(float),
    np.abs(np.random.randn(n_normal)) * 0.1,
    np.random.beta(2, 20, n_normal),
    np.random.beta(3, 20, n_normal),
    1 + np.abs(np.random.randn(n_normal)) * 0.3,
    2 + np.random.randn(n_normal) * 0.5,
    0.7 + np.random.randn(n_normal) * 0.15,
])

# Spoofing (1%)
n_spoof = int(n_total * 0.01)
X_spoof = np.column_stack([
    3 + np.random.randn(n_spoof) * 1.0,
    0.7 + np.random.beta(5, 2, n_spoof) * 0.3,
    2 + np.random.randn(n_spoof) * 0.5,
    np.random.beta(2, 15, n_spoof),
    np.random.beta(1, 50, n_spoof),
    0.3 + np.random.randn(n_spoof) * 0.15,
    0.6 + np.random.beta(5, 3, n_spoof) * 0.4,
    1.5 + np.random.randn(n_spoof) * 0.8,
    0.6 + np.random.beta(7, 3, n_spoof) * 0.4,
    np.random.poisson(3, n_spoof).astype(float),
    np.abs(np.random.randn(n_spoof)) * 0.1,
    np.random.beta(2, 20, n_spoof),
    np.random.beta(3, 20, n_spoof),
    2 + np.abs(np.random.randn(n_spoof)) * 0.5,
    0.5 + np.random.randn(n_spoof) * 0.3,
    0.7 + np.random.randn(n_spoof) * 0.15,
])

# Wash trading (1%)
n_wash = int(n_total * 0.01)
X_wash = np.column_stack([
    1 + np.random.randn(n_wash) * 0.8,
    np.random.beta(3, 10, n_wash),
    5 + np.random.randn(n_wash) * 1.0,
    0.8 + np.random.beta(8, 2, n_wash) * 0.2,
    0.15 + np.random.beta(3, 5, n_wash) * 0.3,
    np.random.randn(n_wash) * 0.05,
    np.random.beta(3, 7, n_wash),
    np.random.randn(n_wash) * 0.3,
    0.7 + np.random.beta(7, 3, n_wash) * 0.3,
    np.random.poisson(3, n_wash).astype(float),
    np.abs(np.random.randn(n_wash)) * 0.1,
    np.random.beta(2, 20, n_wash),
    np.random.beta(3, 20, n_wash),
    1 + np.abs(np.random.randn(n_wash)) * 0.3,
    1.8 + np.random.randn(n_wash) * 0.3,
    0.2 + np.random.beta(2, 8, n_wash) * 0.3,
])

# Pump-and-dump (1%)
n_pump = n_total - n_normal - n_spoof - n_wash
X_pump = np.column_stack([
    4 + np.random.randn(n_pump) * 1.5,
    np.random.beta(2, 8, n_pump),
    5 + np.random.randn(n_pump) * 1.5,
    0.5 + np.random.beta(5, 5, n_pump) * 0.3,
    np.random.beta(1, 30, n_pump),
    0.5 + np.random.randn(n_pump) * 0.1,
    np.random.beta(3, 7, n_pump),
    0.5 + np.random.randn(n_pump) * 0.4,
    0.6 + np.random.beta(5, 3, n_pump) * 0.4,
    8 + np.random.poisson(5, n_pump).astype(float),
    0.2 + np.abs(np.random.randn(n_pump)) * 0.3,
    np.random.beta(5, 10, n_pump),
    np.random.beta(2, 20, n_pump),
    3 + np.abs(np.random.randn(n_pump)) * 0.5,
    1 + np.random.randn(n_pump) * 0.4,
    0.3 + np.random.beta(2, 8, n_pump) * 0.3,
])

# Full dataset
X_all = np.vstack([X_normal, X_spoof, X_wash, X_pump])
y_all = np.concatenate([
    np.zeros(n_normal), np.ones(n_spoof),
    2 * np.ones(n_wash), 3 * np.ones(n_pump)
])

perm = np.random.permutation(n_total)
X_all = X_all[perm]
y_all = y_all[perm]

print(f"Total dataset: {X_all.shape}")
print(f"Normal: {np.sum(y_all==0)}, Anomalies: {np.sum(y_all>0)}")

# ================================================================
# Train-Test Split (train on normal only)
# ================================================================

normal_mask = y_all == 0
X_normal_all = X_all[normal_mask]
n_train = int(len(X_normal_all) * 0.8)

X_train = X_normal_all[:n_train]    # Train only on normal
X_val = X_normal_all[n_train:]      # Validate on held-out normal

# Test on everything
X_test = X_all
y_test = y_all
y_test_binary = (y_all > 0).astype(int)

print(f"Training (normal only): {X_train.shape}")
print(f"Validation (normal only): {X_val.shape}")
print(f"Test (all): {X_test.shape}")

# ================================================================
# Train Autoencoder
# ================================================================

print("\n=== Training Autoencoder ===")
ae = Autoencoder(
    input_dim=n_features,
    hidden_dims=[64, 32],
    latent_dim=6,
    learning_rate=0.001,
    dropout_rate=0.1
)

losses = ae.fit(X_train, epochs=80, batch_size=128, verbose=True)

# Set threshold using validation normal data
threshold = ae.set_threshold(X_val, k=3)
print(f"\nAnomaly threshold: {threshold:.6f}")
print(f"Normal error mean: {ae.error_mean:.6f}")
print(f"Normal error std: {ae.error_std:.6f}")

# ================================================================
# Evaluate Detection Performance
# ================================================================

errors = ae.reconstruction_error(X_test)
predictions = ae.predict(X_test)

tp = np.sum((predictions == 1) & (y_test_binary == 1))
fp = np.sum((predictions == 1) & (y_test_binary == 0))
fn = np.sum((predictions == 0) & (y_test_binary == 1))
tn = np.sum((predictions == 0) & (y_test_binary == 0))

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\n=== Detection Performance ===")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ================================================================
# Per-Type Detection
# ================================================================

print(f"\n=== Detection by Manipulation Type ===")
type_names = {1: 'Spoofing', 2: 'Wash Trading', 3: 'Pump-and-Dump'}
for type_id, type_name in type_names.items():
    mask = y_all == type_id
    type_errors = errors[mask]
    detected = np.sum(predictions[mask] == 1)
    total = np.sum(mask)
    print(f"  {type_name}: {detected}/{total} detected ({detected/total:.1%}), "
          f"mean error={np.mean(type_errors):.6f}")

# ================================================================
# Feature-Level Anomaly Analysis
# ================================================================

print(f"\n=== Feature-Level Reconstruction Error (Top Anomaly) ===")
top_anomaly_idx = np.argmax(errors)
feat_errors = ae.feature_reconstruction_error(X_test[top_anomaly_idx:top_anomaly_idx+1])[0]
anomaly_type = int(y_all[top_anomaly_idx])
type_str = {0: 'Normal', 1: 'Spoofing', 2: 'Wash Trading', 3: 'Pump-and-Dump'}

print(f"True type: {type_str[anomaly_type]}")
print(f"Total error: {errors[top_anomaly_idx]:.6f}")
print(f"\nPer-feature reconstruction error:")

sorted_feat = np.argsort(feat_errors)[::-1]
for idx in sorted_feat:
    bar = "#" * int(feat_errors[idx] * 20)
    print(f"  {feature_names[idx]:<25s}: {feat_errors[idx]:.4f} {bar}")

# ================================================================
# Latent Space Analysis
# ================================================================

print(f"\n=== Latent Space Analysis ===")
latent_all = ae.get_latent(X_test)

for type_id in range(4):
    mask = y_all == type_id
    latent_subset = latent_all[mask]
    name = type_str[type_id]
    print(f"  {name}:")
    print(f"    Latent mean: [{', '.join(f'{v:.3f}' for v in np.mean(latent_subset, axis=0)[:4])}...]")
    print(f"    Latent std:  [{', '.join(f'{v:.3f}' for v in np.std(latent_subset, axis=0)[:4])}...]")

# ================================================================
# Threshold Sensitivity
# ================================================================

print(f"\n=== Threshold Sensitivity (k * sigma) ===")
print(f"{'k':>5} {'Threshold':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP Rate':>10}")
print("-" * 57)

for k in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    thresh = ae.error_mean + k * ae.error_std
    preds = (errors > thresh).astype(int)
    tp_k = np.sum((preds == 1) & (y_test_binary == 1))
    fp_k = np.sum((preds == 1) & (y_test_binary == 0))
    fn_k = np.sum((preds == 0) & (y_test_binary == 1))
    tn_k = np.sum((preds == 0) & (y_test_binary == 0))
    prec = tp_k / (tp_k + fp_k + 1e-8)
    rec = tp_k / (tp_k + fn_k + 1e-8)
    f1_k = 2 * prec * rec / (prec + rec + 1e-8)
    fp_rate = fp_k / (fp_k + tn_k + 1e-8)
    print(f"{k:>5.1f} {thresh:>12.6f} {prec:>10.4f} {rec:>10.4f} {f1_k:>10.4f} {fp_rate:>10.4f}")

# ================================================================
# Reconstruction Error Distribution
# ================================================================

print(f"\n=== Reconstruction Error Statistics ===")
print(f"{'Category':<20} {'Mean':>10} {'Median':>10} {'P95':>10} {'P99':>10} {'Max':>10}")
print("-" * 70)

for type_id in range(4):
    mask = y_all == type_id
    e = errors[mask]
    name = type_str[type_id]
    print(f"{name:<20} {np.mean(e):>10.6f} {np.median(e):>10.6f} "
          f"{np.percentile(e, 95):>10.6f} {np.percentile(e, 99):>10.6f} "
          f"{np.max(e):>10.6f}")
```

## Key Takeaways

1. **Autoencoders excel at detecting novel manipulation patterns** because they model normality rather than specific anomaly types -- any order flow that deviates from the learned normal manifold triggers elevated reconstruction error, including manipulation tactics never seen before.

2. **Training on clean normal data only is both a strength and a challenge** -- it eliminates the need for labeled manipulation examples but requires careful curation of the training set to ensure no contamination from undetected historical manipulation.

3. **Feature-level reconstruction error provides actionable interpretability** -- by examining which features contribute most to an anomaly's high total error, investigators can quickly identify the suspicious aspect (e.g., cancel rate for spoofing, self-trade rate for wash trading).

4. **The anomaly threshold is the most critical operational parameter** -- it directly controls the trade-off between catching manipulation (recall) and minimizing false alarms (precision). Dynamic threshold adjustment based on market conditions improves operational efficiency.

5. **The latent space captures meaningful structure** -- different types of manipulation occupy distinct regions of the latent space, enabling downstream clustering and classification of detected anomalies without explicit labeling.

6. **Ensemble approaches and regular retraining are essential for production deployment** -- combining autoencoder scores with domain rules and retraining on rolling windows of recent normal data maintains detection effectiveness as market microstructure evolves.
