"""
Temporal Fusion Transformer (TFT) - Sklearn-Compatible Wrapper
================================================================

Theory & Mathematics:
    The Temporal Fusion Transformer (TFT) is a state-of-the-art architecture
    for multi-horizon time series forecasting that combines:

    1. Variable Selection Networks (VSN):
       Learns which input features are most important at each time step.
       Uses GRN (Gated Residual Network) to produce feature weights:
           xi_t = Softmax(GRN(xi_1, xi_2, ..., xi_n))
       Selected features: x_tilde = sum_i xi_i * xi_i

    2. Gated Residual Networks (GRN):
       A flexible nonlinear processing unit:
           eta_1 = W_1 * a + b_1
           eta_2 = W_2 * ELU(eta_1) + b_2
           GRN(a) = LayerNorm(a + GLU(eta_2))
       where GLU is the Gated Linear Unit: GLU(x) = sigma(W_g*x) * (W_v*x)

    3. Multi-Head Attention:
       Interpretable self-attention over the temporal dimension:
           Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
       with multiple heads for capturing different temporal patterns.

    4. Static Covariate Encoders:
       Process time-invariant features and use them to initialise LSTM
       hidden states and provide context to other components.

    5. Temporal Processing:
       - Past inputs: processed by encoder LSTM
       - Known future inputs: processed by decoder LSTM
       - Self-attention: applied over the full temporal range
       - Gated skip connections between layers

    6. Quantile Outputs:
       Optional quantile regression for prediction intervals:
           L = sum_{q in Q} sum_t max(q*(y_t - y_hat_t), (q-1)*(y_t - y_hat_t))

    This module wraps a PyTorch TFT implementation in an sklearn-compatible
    interface with fit() and predict() methods.

Business Use Cases:
    - Multi-horizon demand forecasting with known future events
    - Energy price forecasting with weather covariates
    - Retail sales with promotions (known future) and economic indicators
    - Healthcare patient outcome prediction
    - Financial portfolio optimisation

Advantages:
    - State-of-the-art accuracy on many benchmarks
    - Built-in interpretability (variable importance, attention weights)
    - Handles multiple input types (static, past, known future)
    - Quantile outputs for uncertainty estimation
    - Sklearn-compatible API for easy integration

Disadvantages:
    - Complex architecture with many components
    - Requires substantial training data
    - Computationally expensive
    - Many hyperparameters
    - Harder to debug than simpler models

Key Hyperparameters:
    - hidden_size (int): Core hidden dimension
    - num_heads (int): Number of attention heads
    - num_layers (int): Number of stacked LSTM layers
    - dropout (float): Dropout rate
    - lookback (int): Historical lookback window
    - horizon (int): Forecast horizon
    - lr (float): Learning rate
    - batch_size (int): Training batch size

References:
    - Lim, B. et al. (2021). Temporal Fusion Transformers for Interpretable
      Multi-horizon Time Series Forecasting. International Journal of Forecasting.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

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
# TFT Building Blocks (simplified)
# ---------------------------------------------------------------------------

class GatedLinearUnit(nn.Module):
    """GLU: sigma(W_g * x) * (W_v * x)"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    """GRN: LayerNorm(x + GLU(W2 * ELU(W1 * x)))"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1,
                 context_size=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.glu = GatedLinearUnit(output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        eta1 = self.fc1(x)
        if self.context_fc is not None and context is not None:
            eta1 = eta1 + self.context_fc(context)
        eta1 = self.elu(eta1)
        eta2 = self.dropout(self.fc2(eta1))
        gated = self.glu(eta2)
        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """Select and weight input features."""
    def __init__(self, input_size, n_features, hidden_size, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.grn_flat = GatedResidualNetwork(
            input_size * n_features, hidden_size, n_features, dropout
        )
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(n_features)
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, seq_len, n_features, input_size)
        batch, seq_len = x.shape[0], x.shape[1]
        flat = x.reshape(batch, seq_len, -1)
        weights = self.softmax(self.grn_flat(flat))  # (batch, seq_len, n_features)

        processed = []
        for i in range(self.n_features):
            processed.append(self.grns[i](x[:, :, i, :]))
        processed = torch.stack(processed, dim=-1)  # (batch, seq, hidden, n_feat)

        weights_expanded = weights.unsqueeze(2)  # (batch, seq, 1, n_feat)
        selected = (processed * weights_expanded).sum(dim=-1)
        return selected, weights


class SimplifiedTFT(nn.Module):
    """Simplified Temporal Fusion Transformer for univariate time series."""

    def __init__(self, input_size=1, hidden_size=32, num_heads=4,
                 num_lstm_layers=1, dropout=0.1, lookback=30, horizon=1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size

        # Input embedding
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LSTM encoder
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # GRN for post-LSTM processing
        self.post_lstm_grn = GatedResidualNetwork(hidden_size, hidden_size,
                                                   hidden_size, dropout)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        self.attn_grn = GatedResidualNetwork(hidden_size, hidden_size,
                                              hidden_size, dropout)

        # Output
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )

    def forward(self, x):
        # x: (batch, lookback, input_size)
        batch = x.shape[0]

        # Project input
        embedded = self.input_proj(x)  # (batch, lookback, hidden_size)

        # LSTM encoding
        lstm_out, _ = self.encoder_lstm(embedded)

        # Post-LSTM GRN with skip connection
        processed = self.post_lstm_grn(lstm_out)

        # Self-attention
        attn_out, attn_weights = self.attention(processed, processed, processed)
        attn_out = self.attn_layer_norm(processed + attn_out)
        attn_out = self.attn_grn(attn_out)

        # Use last time step
        last = attn_out[:, -1, :]

        return self.output_fc(last)


# ---------------------------------------------------------------------------
# Sklearn-Compatible Wrapper
# ---------------------------------------------------------------------------

class TFTForecaster(BaseEstimator, RegressorMixin):
    """Sklearn-compatible Temporal Fusion Transformer wrapper."""

    def __init__(
        self,
        lookback: int = 30,
        horizon: int = 1,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        verbose: bool = True,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        self.model_ = None
        self.scaler_ = StandardScaler()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.lookback, len(data) - self.horizon + 1):
            X.append(data[i - self.lookback:i])
            y.append(data[i:i + self.horizon])
        return np.array(X), np.array(y)

    def fit(self, X, y=None):
        data = self.scaler_.fit_transform(X.reshape(-1, 1)).flatten()
        X_seq, y_seq = self._create_sequences(data)

        X_t = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y_seq, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = SimplifiedTFT(
            input_size=1, hidden_size=self.hidden_size,
            num_heads=self.num_heads, num_lstm_layers=self.num_lstm_layers,
            dropout=self.dropout, lookback=self.lookback, horizon=self.horizon,
        ).to(self.device_)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            if avg < best_loss:
                best_loss = avg
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg:.6f}")

        self.model_.load_state_dict(best_state)
        self._train_tail_ = data[-self.lookback:]
        return self

    def predict(self, X):
        self.model_.eval()
        n_steps = len(X)
        history = list(self._train_tail_)
        predictions = []

        with torch.no_grad():
            steps_done = 0
            while steps_done < n_steps:
                seq = np.array(history[-self.lookback:])
                xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                xt = xt.to(self.device_)
                pred = self.model_(xt).cpu().numpy().flatten()
                for p in pred:
                    if steps_done < n_steps:
                        predictions.append(p)
                        history.append(p)
                        steps_done += 1

        return self.scaler_.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(train_data, **kwargs):
    model = TFTForecaster(**kwargs)
    model.fit(train_data)
    return model


def validate(model, val_data):
    predictions = model.predict(val_data)
    return _compute_metrics(val_data, predictions)


def test(model, test_data, val_data=None):
    if val_data is not None:
        _ = model.predict(val_data)
    predictions = model.predict(test_data)
    return _compute_metrics(test_data, predictions)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    lookback = trial.suggest_int("lookback", 7, 60)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 30, 100, step=10)

    try:
        model = train(
            train_data, lookback=lookback, hidden_size=hidden_size,
            num_heads=num_heads, dropout=dropout, lr=lr,
            epochs=epochs, verbose=False,
        )
        metrics = validate(model, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=15):
    study = optuna.create_study(direction="minimize", study_name="tft_sklearn")
    study.optimize(lambda t: optuna_objective(t, train_data, val_data),
                   n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


def ray_tune_search(train_data, val_data, num_samples=20):
    def _trainable(config):
        try:
            model = train(
                train_data, lookback=config["lookback"],
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                dropout=config["dropout"], lr=config["lr"],
                epochs=config["epochs"], verbose=False,
            )
            metrics = validate(model, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "lookback": tune.randint(7, 60),
        "hidden_size": tune.choice([16, 32, 64]),
        "num_heads": tune.choice([2, 4]),
        "dropout": tune.uniform(0.0, 0.4),
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([30, 50, 80]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    tuner = tune.Tuner(
        _trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Temporal Fusion Transformer - Sklearn Wrapper")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=10)
    bp = study.best_params

    print("\n--- Training Best ---")
    best_model = train(
        train_data, lookback=bp["lookback"], hidden_size=bp["hidden_size"],
        num_heads=bp["num_heads"], dropout=bp["dropout"],
        lr=bp["lr"], epochs=bp["epochs"],
    )

    print("\n--- Validation ---")
    for k, v in validate(best_model, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Test ---")
    best_model2 = train(
        train_data, lookback=bp["lookback"], hidden_size=bp["hidden_size"],
        num_heads=bp["num_heads"], dropout=bp["dropout"],
        lr=bp["lr"], epochs=bp["epochs"], verbose=False,
    )
    for k, v in test(best_model2, test_data, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Ray Tune ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=8)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
