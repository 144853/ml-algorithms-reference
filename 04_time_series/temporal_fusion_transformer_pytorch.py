"""
Temporal Fusion Transformer (TFT) - PyTorch Implementation
============================================================

Theory & Mathematics:
    The Temporal Fusion Transformer (TFT) is a state-of-the-art architecture
    for interpretable multi-horizon time series forecasting. It combines several
    specialised neural network components:

    1. Gated Residual Network (GRN):
       The fundamental building block of TFT. Provides flexible nonlinear
       processing with skip connections:

           eta_1 = W_1 @ x + b_1
           eta_2 = W_2 @ ELU(eta_1) + b_2

           GLU(a, b) = sigma(a) * b         (Gated Linear Unit)

           GRN(x, c) = LayerNorm(x + GLU(W_3 @ eta_2, W_4 @ eta_2))

       When context c is provided:
           eta_1 = W_1 @ x + W_c @ c + b_1

       The GLU gate allows the network to suppress components that are
       not needed, providing adaptive depth.

    2. Variable Selection Network (VSN):
       Learns instance-wise feature importance:

           For each feature i: xi_i = GRN_i(x_i)
           Importance weights: v = Softmax(GRN_v(flatten(x_1, ..., x_n)))
           Selected: x_tilde = sum_i v_i * xi_i

       This provides built-in interpretability about which features
       contribute to forecasts.

    3. Multi-Head Interpretable Attention:
       Modified self-attention where value projections share weights
       across heads for interpretability:

           For each head h:
               Q_h = W_Q_h @ q,  K_h = W_K_h @ k
               V = W_V @ v  (shared across heads)

           A(Q, K, V) = Softmax(Q @ K^T / sqrt(d_attn)) @ V

       The attention weights directly indicate which past time steps
       influence the forecast.

    4. Static Covariate Encoders:
       Process time-invariant features (e.g., product category, location)
       and produce four context vectors:
           - c_s: Context for variable selection
           - c_e: Context for enrichment
           - c_h: Initial hidden state for LSTM
           - c_c: Initial cell state for LSTM

    5. Temporal Processing:
       a) Past observed inputs -> Variable Selection -> Encoder LSTM
       b) Known future inputs -> Variable Selection -> Decoder LSTM
       c) Static enrichment via GRN
       d) Self-attention over all temporal positions
       e) Position-wise feed-forward via GRN
       f) Gated skip connections between layers

    6. Quantile Outputs:
       For quantile q in {0.1, 0.5, 0.9}:
           y_hat_q(t) = W_q @ output(t) + b_q

       Loss = sum_t sum_q QL(y(t), y_hat_q(t), q)
       where QL(y, y_hat, q) = max(q*(y-y_hat), (q-1)*(y-y_hat))

    Full Architecture Flow:
       Input Features
           |
       [Variable Selection Networks]  <-- Static context
           |
       [LSTM Encoder / Decoder]       <-- Static h0/c0
           |
       [Static Enrichment GRN]        <-- Static context
           |
       [Multi-Head Attention]
           |
       [Position-wise GRN]
           |
       [Quantile/Point Output]

Business Use Cases:
    - Multi-horizon retail demand forecasting (Walmart, Amazon)
    - Energy price and load forecasting
    - Financial portfolio risk assessment
    - Healthcare patient trajectory prediction
    - Supply chain optimisation with known future events (promotions)
    - Tourism demand with calendar effects

Advantages:
    - State-of-the-art accuracy on numerous benchmarks
    - Built-in interpretability:
      * Variable importance from VSN
      * Temporal patterns from attention weights
      * Regime detection from GRN gating
    - Handles heterogeneous inputs (static, past, known future)
    - Multi-horizon direct forecasting (no recursive error accumulation)
    - Quantile outputs for uncertainty estimation
    - Robust to noisy features via variable selection

Disadvantages:
    - Complex architecture (many components to implement and tune)
    - High computational cost (attention is O(T^2))
    - Requires large datasets for best performance
    - Many hyperparameters
    - Training can be unstable without careful initialisation
    - Memory-intensive for long sequences

Key Hyperparameters:
    - hidden_size (int): Core hidden dimension (typically 32-256)
    - num_heads (int): Number of attention heads (typically 4-8)
    - num_lstm_layers (int): LSTM encoder/decoder depth (1-2)
    - dropout (float): Dropout rate (0.1-0.3)
    - lookback (int): Historical context window
    - horizon (int): Forecast horizon
    - lr (float): Learning rate (1e-4 to 1e-3)
    - batch_size (int): Training batch size (32-128)
    - quantiles (list): Output quantiles (e.g., [0.1, 0.5, 0.9])

References:
    - Lim, B., Arik, S.O., Loeff, N. & Pfister, T. (2021). Temporal Fusion
      Transformers for Interpretable Multi-horizon Time Series Forecasting.
      International Journal of Forecasting, 37(4), 1748-1764.
    - Vaswani, A. et al. (2017). Attention Is All You Need.
    - Dauphin, Y.N. et al. (2017). Language Modeling with Gated Convolutional Networks.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
import ray
from ray import tune

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_points: int = 1000, freq: str = "D", seasonal_period: int = 7,
    trend_slope: float = 0.02, noise_std: float = 0.5, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic multivariate time series.

    Returns univariate target with additional engineered features:
    - value (target)
    - time_index (normalised)
    - day_of_week (sin/cos encoded)
    """
    np.random.seed(seed)
    t = np.arange(n_points, dtype=np.float64)
    trend = trend_slope * t
    seasonality = 2.0 * np.sin(2 * np.pi * t / seasonal_period)
    yearly = 1.0 * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, noise_std, n_points)
    values = 10.0 + trend + seasonality + yearly + noise

    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


def _compute_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64).flatten()
    predicted = np.asarray(predicted, dtype=np.float64).flatten()
    min_len = min(len(actual), len(predicted))
    actual, predicted = actual[:min_len], predicted[:min_len]
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mask = actual != 0
    mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100 if mask.any() else np.inf
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    denom = np.where(denom == 0, 1e-10, denom)
    smape = np.mean(np.abs(errors) / denom) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


# ---------------------------------------------------------------------------
# TFT Building Blocks
# ---------------------------------------------------------------------------

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit: sigma(Wa) * Wb.

    Allows the network to control information flow through gating.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc_gate = nn.Linear(input_size, output_size)
        self.fc_value = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc_gate(x)) * self.fc_value(x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN).

    GRN(a, c) = LayerNorm(a + GLU(dropout(W2 * ELU(W1 * a + Wc * c + b1) + b2)))

    Provides nonlinear processing with skip connections and gating.
    The optional context vector c modulates the processing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_fc = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size is not None
            else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

        # Skip connection (project if dimensions differ)
        self.skip = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else None
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x

        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        gated = self.glu(hidden)

        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network.

    Learns instance-wise feature importance and produces weighted
    combination of processed features.

    For each time step:
        weights = softmax(GRN(concat(features)))
        output = sum(weight_i * GRN_i(feature_i))
    """

    def __init__(
        self,
        input_size: int,
        n_features: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features

        # GRN for computing feature importance weights
        self.weight_grn = GatedResidualNetwork(
            input_size=input_size * n_features,
            hidden_size=hidden_size,
            output_size=n_features,
            dropout=dropout,
            context_size=context_size,
        )

        # Individual GRNs for processing each feature
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for _ in range(n_features)
        ])

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features, input_size)
            context: (batch, context_size) optional static context

        Returns:
            selected: (batch, seq_len, hidden_size)
            weights: (batch, seq_len, n_features) importance weights
        """
        batch, seq_len, n_feat, feat_dim = x.shape

        # Flatten features for weight computation
        flat = x.reshape(batch, seq_len, -1)  # (batch, seq, n_feat * feat_dim)

        # Expand context if provided
        ctx = None
        if context is not None:
            ctx = context.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute importance weights
        weights = torch.softmax(
            self.weight_grn(flat, ctx), dim=-1
        )  # (batch, seq_len, n_features)

        # Process individual features
        processed = []
        for i in range(self.n_features):
            processed.append(self.feature_grns[i](x[:, :, i, :]))
        processed = torch.stack(
            processed, dim=-1
        )  # (batch, seq_len, hidden_size, n_features)

        # Weighted combination
        weights_expanded = weights.unsqueeze(2)  # (batch, seq, 1, n_feat)
        selected = (processed * weights_expanded).sum(dim=-1)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention.

    Unlike standard multi-head attention, values share weights across
    heads, making attention weights directly interpretable.

    A_h(Q, K, V) = softmax(Q_h * K_h^T / sqrt(d_attn)) * V
    Output = W_H * (1/H * sum_h A_h)
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_attn = hidden_size // num_heads
        self.hidden_size = hidden_size

        # Per-head Q, K projections
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)

        # Shared V projection (for interpretability)
        self.W_V = nn.Linear(hidden_size, self.d_attn, bias=False)

        # Output projection
        self.W_out = nn.Linear(self.d_attn, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, q_len, hidden_size)
            key: (batch, k_len, hidden_size)
            value: (batch, v_len, hidden_size)
            mask: optional attention mask

        Returns:
            output: (batch, q_len, hidden_size)
            attention_weights: (batch, num_heads, q_len, k_len)
        """
        batch = query.shape[0]

        # Q, K projections: split into heads
        Q = self.W_Q(query).view(batch, -1, self.num_heads, self.d_attn)
        K = self.W_K(key).view(batch, -1, self.num_heads, self.d_attn)
        Q = Q.permute(0, 2, 1, 3)  # (batch, heads, q_len, d_attn)
        K = K.permute(0, 2, 1, 3)  # (batch, heads, k_len, d_attn)

        # Shared V projection
        V = self.W_V(value)  # (batch, v_len, d_attn)

        # Attention scores
        scale = np.sqrt(self.d_attn)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # (batch, heads, q_len, k_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to shared values
        # Average across heads then apply to V
        avg_weights = attn_weights.mean(dim=1)  # (batch, q_len, k_len)
        context = torch.matmul(avg_weights, V)  # (batch, q_len, d_attn)

        output = self.W_out(context)  # (batch, q_len, hidden_size)

        return output, attn_weights


# ---------------------------------------------------------------------------
# Full TFT Architecture
# ---------------------------------------------------------------------------

class TemporalFusionTransformer(nn.Module):
    """Full Temporal Fusion Transformer.

    Components:
        1. Input embedding and variable selection
        2. LSTM encoder for temporal processing
        3. Static enrichment
        4. Interpretable multi-head attention
        5. Position-wise feed-forward (GRN)
        6. Point or quantile output
    """

    def __init__(
        self,
        n_features: int = 3,
        feature_size: int = 1,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
        lookback: int = 30,
        horizon: int = 1,
        quantiles: Optional[List[float]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.lookback = lookback
        self.horizon = horizon
        self.quantiles = quantiles or [0.5]
        self.n_quantiles = len(self.quantiles)

        # Feature embedding
        self.feature_embedding = nn.Linear(feature_size, hidden_size)

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_size=hidden_size,
            n_features=n_features,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # LSTM Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # Post-LSTM gate
        self.post_lstm_glu = GatedLinearUnit(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        # Static enrichment GRN
        self.enrichment_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # Multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attn_glu = GatedLinearUnit(hidden_size, hidden_size)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Position-wise feed-forward GRN
        self.ff_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # Output layer
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon * self.n_quantiles),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: (batch, lookback, n_features) input features.

        Returns:
            predictions: (batch, horizon, n_quantiles)
            interpretability: dict with 'attention_weights' and 'variable_weights'
        """
        batch = x.shape[0]
        seq_len = x.shape[1]
        n_features = x.shape[2]

        # 1. Feature embedding
        # Reshape to (batch, seq, n_feat, 1) then embed
        x_embedded = self.feature_embedding(
            x.unsqueeze(-1)
        )  # (batch, seq, n_feat, hidden)

        # 2. Variable Selection
        selected, var_weights = self.vsn(x_embedded)
        # selected: (batch, seq, hidden)
        # var_weights: (batch, seq, n_features)

        # 3. LSTM Encoding
        lstm_out, _ = self.encoder_lstm(selected)
        # (batch, seq, hidden)

        # Post-LSTM gating with skip connection
        gated = self.post_lstm_glu(lstm_out)
        temporal = self.post_lstm_norm(selected + gated)

        # 4. Static Enrichment
        enriched = self.enrichment_grn(temporal)

        # 5. Multi-Head Attention
        attn_out, attn_weights = self.attention(enriched, enriched, enriched)
        # Gated skip connection
        attn_gated = self.attn_glu(attn_out)
        attn_output = self.attn_norm(enriched + attn_gated)

        # 6. Position-wise Feed-Forward
        ff_output = self.ff_grn(attn_output)

        # 7. Output (use last time step)
        last_output = ff_output[:, -1, :]  # (batch, hidden)
        raw_output = self.output_fc(last_output)  # (batch, horizon * n_quantiles)

        predictions = raw_output.view(batch, self.horizon, self.n_quantiles)

        interpretability = {
            "attention_weights": attn_weights.detach(),
            "variable_weights": var_weights.detach(),
        }

        return predictions, interpretability


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

class QuantileLoss(nn.Module):
    """Quantile loss (pinball loss) for quantile regression."""

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, horizon, n_quantiles)
            targets: (batch, horizon)
        """
        targets = targets.unsqueeze(-1)  # (batch, horizon, 1)
        errors = targets - predictions

        losses = []
        for i, q in enumerate(self.quantiles):
            e = errors[:, :, i]
            loss = torch.max(q * e, (q - 1) * e)
            losses.append(loss)

        return torch.stack(losses, dim=-1).mean()


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _create_features(values: np.ndarray) -> np.ndarray:
    """Create multivariate features from univariate time series.

    Features:
        1. Value itself
        2. Normalised time index
        3. Sinusoidal day-of-week encoding
    """
    n = len(values)
    t_norm = np.arange(n, dtype=np.float64) / n
    day_sin = np.sin(2 * np.pi * np.arange(n) / 7.0)

    features = np.column_stack([values, t_norm, day_sin])
    return features  # (n, 3)


def _create_sequences(features: np.ndarray, targets: np.ndarray,
                      lookback: int, horizon: int):
    """Create supervised learning sequences."""
    X, y = [], []
    for i in range(lookback, len(features) - horizon + 1):
        X.append(features[i - lookback:i])
        y.append(targets[i:i + horizon])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    lookback: int = 30,
    horizon: int = 1,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_lstm_layers: int = 1,
    dropout: float = 0.1,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    quantiles: Optional[List[float]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train Temporal Fusion Transformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale
    mean = np.mean(train_data)
    std = np.std(train_data) + 1e-8
    scaled = (train_data - mean) / std

    # Create features
    features = _create_features(scaled)
    n_features = features.shape[1]

    # Create sequences
    X, y = _create_sequences(features, scaled, lookback, horizon)

    X_tensor = torch.tensor(X, dtype=torch.float32)  # (N, lookback, n_features)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # (N, horizon)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    if quantiles is None:
        quantiles = [0.5]

    model = TemporalFusionTransformer(
        n_features=n_features,
        feature_size=1,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        lookback=lookback,
        horizon=horizon,
        quantiles=quantiles,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    if len(quantiles) > 1:
        criterion = QuantileLoss(quantiles)
    else:
        criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds, _ = model(X_batch)

            if len(quantiles) == 1:
                loss = criterion(preds.squeeze(-1), y_batch)
            else:
                loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, "
                  f"LR: {current_lr:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Store necessary info for prediction
    train_features = features[-lookback:]

    return {
        "model": model,
        "lookback": lookback,
        "horizon": horizon,
        "mean": mean,
        "std": std,
        "n_features": n_features,
        "quantiles": quantiles,
        "train_tail_features": train_features,
        "train_tail_scaled": scaled[-lookback:],
        "train_loss": best_loss,
    }


def _forecast(result: Dict, n_steps: int) -> np.ndarray:
    """Generate multi-step forecasts."""
    model = result["model"]
    device = next(model.parameters()).device
    model.eval()

    lookback = result["lookback"]
    horizon = result["horizon"]
    mean, std = result["mean"], result["std"]
    n_train = len(result["train_tail_scaled"])

    # Build feature history
    scaled_history = list(result["train_tail_scaled"])
    predictions = []

    with torch.no_grad():
        steps_done = 0
        while steps_done < n_steps:
            # Create features from current history
            vals = np.array(scaled_history[-lookback:])
            n_total = len(scaled_history)
            t_start = n_total - lookback
            t_norm = np.arange(t_start, n_total, dtype=np.float64) / max(n_total, 1)
            day_sin = np.sin(2 * np.pi * np.arange(t_start, n_total) / 7.0)
            features = np.column_stack([vals, t_norm, day_sin])

            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            preds, interpretability = model(x)

            # For point forecast (median quantile)
            if len(result["quantiles"]) == 1:
                pred_vals = preds.squeeze(-1).cpu().numpy().flatten()
            else:
                # Use median (0.5 quantile, index 1 if [0.1, 0.5, 0.9])
                median_idx = result["quantiles"].index(0.5) if 0.5 in result["quantiles"] else 0
                pred_vals = preds[:, :, median_idx].cpu().numpy().flatten()

            for p in pred_vals:
                if steps_done < n_steps:
                    predictions.append(p)
                    scaled_history.append(p)
                    steps_done += 1

    return np.array(predictions) * std + mean


def validate(result: Dict, val_data: np.ndarray) -> Dict[str, float]:
    predictions = _forecast(result, len(val_data))
    return _compute_metrics(val_data, predictions)


def test(result: Dict, test_data: np.ndarray, val_len: int = 0) -> Dict[str, float]:
    total = val_len + len(test_data)
    all_preds = _forecast(result, total)
    return _compute_metrics(test_data, all_preds[val_len:])


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    lookback = trial.suggest_int("lookback", 7, 60)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 30, 120, step=30)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Ensure hidden_size is divisible by num_heads
    while hidden_size % num_heads != 0:
        num_heads = max(1, num_heads // 2)

    try:
        result = train(
            train_data, lookback=lookback, hidden_size=hidden_size,
            num_heads=num_heads, num_lstm_layers=num_lstm_layers,
            dropout=dropout, lr=lr, epochs=epochs, batch_size=batch_size,
            verbose=False,
        )
        metrics = validate(result, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=15):
    study = optuna.create_study(direction="minimize", study_name="tft_pytorch")
    study.optimize(
        lambda t: optuna_objective(t, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data, val_data, num_samples=20):
    def _trainable(config):
        hs = config["hidden_size"]
        nh = config["num_heads"]
        while hs % nh != 0:
            nh = max(1, nh // 2)

        try:
            result = train(
                train_data, lookback=config["lookback"],
                hidden_size=hs, num_heads=nh,
                num_lstm_layers=config["num_lstm_layers"],
                dropout=config["dropout"], lr=config["lr"],
                epochs=config["epochs"], batch_size=config["batch_size"],
                verbose=False,
            )
            metrics = validate(result, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "lookback": tune.randint(7, 60),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_heads": tune.choice([2, 4, 8]),
        "num_lstm_layers": tune.randint(1, 3),
        "dropout": tune.uniform(0.0, 0.4),
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([30, 60, 90]),
        "batch_size": tune.choice([16, 32, 64]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="rmse", mode="min", num_samples=num_samples
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    print(f"Ray Tune Best config: {best.config}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Temporal Fusion Transformer - PyTorch Implementation")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nData splits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    # Optuna HPO
    print("\n--- Optuna Hyperparameter Search ---")
    study = run_optuna(train_data, val_data, n_trials=12)
    bp = study.best_params

    # Ensure hidden_size divisible by num_heads
    hs = bp["hidden_size"]
    nh = bp["num_heads"]
    while hs % nh != 0:
        nh = max(1, nh // 2)

    # Train best
    print("\n--- Training with Best Parameters ---")
    result = train(
        train_data,
        lookback=bp["lookback"],
        hidden_size=hs,
        num_heads=nh,
        num_lstm_layers=bp["num_lstm_layers"],
        dropout=bp["dropout"],
        lr=bp["lr"],
        epochs=bp["epochs"],
        batch_size=bp["batch_size"],
    )

    model = result["model"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Training loss: {result['train_loss']:.6f}")

    # Interpretability: show variable importance
    print("\n--- Variable Importance (sample) ---")
    device = next(model.parameters()).device
    sample_features = _create_features(
        (train_data[-30:] - result["mean"]) / result["std"]
    )
    sample_x = torch.tensor(sample_features, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _, interp = model(sample_x)
    avg_var_weights = interp["variable_weights"].mean(dim=(0, 1)).cpu().numpy()
    feature_names = ["value", "time_index", "day_of_week_sin"]
    for name, weight in zip(feature_names, avg_var_weights):
        print(f"  {name}: {weight:.4f}")

    # Validate
    print("\n--- Validation Metrics ---")
    val_metrics = validate(result, val_data)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test
    print("\n--- Test Metrics ---")
    test_metrics = test(result, test_data, val_len=len(val_data))
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Ray Tune
    print("\n--- Ray Tune Search ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=8)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
