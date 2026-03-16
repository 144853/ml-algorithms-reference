"""
Simplified Attention-Based Time Series Model - NumPy From-Scratch
==================================================================

Theory & Mathematics:
    This module implements a simplified version of the Temporal Fusion Transformer
    (TFT) concepts from scratch using only NumPy. While a full TFT implementation
    in pure NumPy would be prohibitively complex, this captures the key ideas:

    1. Multi-Head Scaled Dot-Product Attention:
       For each head h:
           Q_h = X @ W_Q_h,  K_h = X @ W_K_h,  V_h = X @ W_V_h
           A_h = softmax(Q_h @ K_h^T / sqrt(d_k)) @ V_h

       MultiHead(X) = concat(A_1, ..., A_H) @ W_O

       where:
       - d_k = hidden_size / num_heads (dimension per head)
       - softmax is applied row-wise (over the key dimension)
       - W_Q, W_K, W_V: (hidden_size, d_k) projection matrices per head
       - W_O: (hidden_size, hidden_size) output projection

    2. Gated Residual Network (GRN):
       eta_1 = W_1 @ x + b_1
       eta_2 = W_2 @ ELU(eta_1) + b_2
       GRN(x) = LayerNorm(x + GLU(eta_2))
       GLU(a) = sigmoid(W_g @ a) * (W_v @ a)

    3. Variable Selection Network (simplified):
       For n input features, compute importance weights via a GRN:
           weights = softmax(GRN_flat(concat(features)))
           selected = sum_i weights_i * GRN_i(feature_i)

    4. Position Encoding:
       Sinusoidal position encoding for temporal ordering:
           PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
           PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    5. Forward Pass:
       Input -> Linear projection -> Position encoding -> Multi-head attention
       -> GRN post-processing -> Output projection

    6. Training:
       - Forward pass computes predictions
       - MSE loss
       - Numerical gradient estimation (finite differences) or simple SGD
       - Adam optimiser with accumulated gradients

Business Use Cases:
    - Understanding attention mechanisms for time series
    - Lightweight deployment without framework dependencies
    - Teaching attention-based forecasting concepts
    - Custom architectures in constrained environments

Advantages:
    - Complete transparency into attention computations
    - No framework dependencies
    - Educational: visualise attention weights
    - Understand gradient flow through attention

Disadvantages:
    - Very slow compared to GPU implementations
    - Limited to small models
    - Approximate gradient computation
    - No automatic differentiation

Key Hyperparameters:
    - hidden_size (int): Model dimension
    - num_heads (int): Number of attention heads
    - lookback (int): Input sequence length
    - lr (float): Learning rate
    - epochs (int): Training epochs

References:
    - Vaswani, A. et al. (2017). Attention Is All You Need.
    - Lim, B. et al. (2021). Temporal Fusion Transformers.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

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
# Activation Functions
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))


def layer_norm(x, eps=1e-5):
    """Apply layer normalisation along the last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

@dataclass
class AttentionParams:
    """Parameters for multi-head attention."""
    W_Q: list = field(default_factory=list)  # List of (d_model, d_k) per head
    W_K: list = field(default_factory=list)
    W_V: list = field(default_factory=list)
    W_O: np.ndarray = None  # (d_model, d_model)
    b_O: np.ndarray = None


@dataclass
class GRNParams:
    """Parameters for Gated Residual Network."""
    W1: np.ndarray = None
    b1: np.ndarray = None
    W2: np.ndarray = None
    b2: np.ndarray = None
    W_gate: np.ndarray = None  # For GLU gate
    W_value: np.ndarray = None  # For GLU value
    W_skip: np.ndarray = None  # Skip connection projection


@dataclass
class TFTNumpyParams:
    """All parameters for the simplified TFT."""
    # Input projection
    W_input: np.ndarray = None
    b_input: np.ndarray = None

    # Attention
    attention: AttentionParams = None

    # Post-attention GRN
    grn: GRNParams = None

    # Output
    W_out1: np.ndarray = None
    b_out1: np.ndarray = None
    W_out2: np.ndarray = None
    b_out2: np.ndarray = None


def _init_params(input_size: int, hidden_size: int, num_heads: int,
                 seed: int = 42) -> TFTNumpyParams:
    """Initialise all model parameters."""
    rng = np.random.RandomState(seed)
    d_k = hidden_size // num_heads
    scale = lambda fan_in: np.sqrt(2.0 / fan_in)

    params = TFTNumpyParams()

    # Input projection
    params.W_input = rng.randn(input_size, hidden_size) * scale(input_size)
    params.b_input = np.zeros(hidden_size)

    # Multi-head attention
    attn = AttentionParams()
    for _ in range(num_heads):
        attn.W_Q.append(rng.randn(hidden_size, d_k) * scale(hidden_size))
        attn.W_K.append(rng.randn(hidden_size, d_k) * scale(hidden_size))
        attn.W_V.append(rng.randn(hidden_size, d_k) * scale(hidden_size))
    attn.W_O = rng.randn(hidden_size, hidden_size) * scale(hidden_size)
    attn.b_O = np.zeros(hidden_size)
    params.attention = attn

    # GRN
    grn = GRNParams()
    grn.W1 = rng.randn(hidden_size, hidden_size) * scale(hidden_size)
    grn.b1 = np.zeros(hidden_size)
    grn.W2 = rng.randn(hidden_size, hidden_size) * scale(hidden_size)
    grn.b2 = np.zeros(hidden_size)
    grn.W_gate = rng.randn(hidden_size, hidden_size) * scale(hidden_size)
    grn.W_value = rng.randn(hidden_size, hidden_size) * scale(hidden_size)
    params.grn = grn

    # Output layers
    out_hidden = hidden_size // 2
    params.W_out1 = rng.randn(hidden_size, out_hidden) * scale(hidden_size)
    params.b_out1 = np.zeros(out_hidden)
    params.W_out2 = rng.randn(out_hidden, 1) * scale(out_hidden)
    params.b_out2 = np.zeros(1)

    return params


def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal positional encoding."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
    return pe


def _multi_head_attention(X: np.ndarray, params: AttentionParams,
                          num_heads: int) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-head scaled dot-product attention.

    Args:
        X: (seq_len, d_model) input.
        params: Attention parameters.
        num_heads: Number of attention heads.

    Returns:
        (output, attention_weights) where output is (seq_len, d_model).
    """
    seq_len, d_model = X.shape
    d_k = d_model // num_heads

    head_outputs = []
    all_weights = []

    for h in range(num_heads):
        Q = X @ params.W_Q[h]  # (seq_len, d_k)
        K = X @ params.W_K[h]
        V = X @ params.W_V[h]

        # Scaled dot-product attention
        scores = (Q @ K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
        weights = softmax(scores, axis=-1)  # (seq_len, seq_len)
        attn_out = weights @ V  # (seq_len, d_k)

        head_outputs.append(attn_out)
        all_weights.append(weights)

    # Concatenate heads
    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)

    # Output projection
    output = concat @ params.W_O + params.b_O

    # Average attention weights across heads
    avg_weights = np.mean(np.stack(all_weights), axis=0)

    return output, avg_weights


def _grn_forward(x: np.ndarray, params: GRNParams) -> np.ndarray:
    """GRN forward pass: LayerNorm(x + GLU(W2 * ELU(W1 * x)))."""
    eta1 = x @ params.W1 + params.b1
    eta1 = elu(eta1)
    eta2 = eta1 @ params.W2 + params.b2

    # GLU
    gate = sigmoid(eta2 @ params.W_gate)
    value = eta2 @ params.W_value
    gated = gate * value

    # Residual + LayerNorm
    out = layer_norm(x + gated)
    return out


def _forward(x_seq: np.ndarray, params: TFTNumpyParams,
             num_heads: int) -> Tuple[float, np.ndarray]:
    """Full forward pass.

    Args:
        x_seq: (seq_len, input_size) input sequence.
        params: Model parameters.
        num_heads: Number of attention heads.

    Returns:
        (prediction, attention_weights).
    """
    seq_len = x_seq.shape[0]
    hidden_size = params.W_input.shape[1]

    # Input projection
    projected = x_seq @ params.W_input + params.b_input  # (seq_len, hidden_size)

    # Add positional encoding
    pe = _positional_encoding(seq_len, hidden_size)
    projected = projected + pe

    # Multi-head attention
    attn_out, attn_weights = _multi_head_attention(projected, params.attention, num_heads)

    # Residual connection
    attn_out = layer_norm(projected + attn_out)

    # GRN
    grn_out = _grn_forward(attn_out, params.grn)

    # Take last time step
    last = grn_out[-1]  # (hidden_size,)

    # Output MLP
    h = np.maximum(0, last @ params.W_out1 + params.b_out1)  # ReLU
    y_hat = (h @ params.W_out2 + params.b_out2).item()

    return y_hat, attn_weights


# ---------------------------------------------------------------------------
# Parameter Flattening for Optimisation
# ---------------------------------------------------------------------------

def _flatten_params(params: TFTNumpyParams) -> np.ndarray:
    """Flatten all parameters into a single vector."""
    arrays = []
    arrays.append(params.W_input.flatten())
    arrays.append(params.b_input.flatten())
    for h in range(len(params.attention.W_Q)):
        arrays.append(params.attention.W_Q[h].flatten())
        arrays.append(params.attention.W_K[h].flatten())
        arrays.append(params.attention.W_V[h].flatten())
    arrays.append(params.attention.W_O.flatten())
    arrays.append(params.attention.b_O.flatten())
    arrays.append(params.grn.W1.flatten())
    arrays.append(params.grn.b1.flatten())
    arrays.append(params.grn.W2.flatten())
    arrays.append(params.grn.b2.flatten())
    arrays.append(params.grn.W_gate.flatten())
    arrays.append(params.grn.W_value.flatten())
    arrays.append(params.W_out1.flatten())
    arrays.append(params.b_out1.flatten())
    arrays.append(params.W_out2.flatten())
    arrays.append(params.b_out2.flatten())
    return np.concatenate(arrays)


def _unflatten_params(flat: np.ndarray, params: TFTNumpyParams,
                      num_heads: int) -> TFTNumpyParams:
    """Reconstruct parameters from flat vector."""
    idx = 0

    def _take(shape):
        nonlocal idx
        size = int(np.prod(shape))
        arr = flat[idx:idx + size].reshape(shape)
        idx += size
        return arr

    params.W_input = _take(params.W_input.shape)
    params.b_input = _take(params.b_input.shape)
    for h in range(num_heads):
        params.attention.W_Q[h] = _take(params.attention.W_Q[h].shape)
        params.attention.W_K[h] = _take(params.attention.W_K[h].shape)
        params.attention.W_V[h] = _take(params.attention.W_V[h].shape)
    params.attention.W_O = _take(params.attention.W_O.shape)
    params.attention.b_O = _take(params.attention.b_O.shape)
    params.grn.W1 = _take(params.grn.W1.shape)
    params.grn.b1 = _take(params.grn.b1.shape)
    params.grn.W2 = _take(params.grn.W2.shape)
    params.grn.b2 = _take(params.grn.b2.shape)
    params.grn.W_gate = _take(params.grn.W_gate.shape)
    params.grn.W_value = _take(params.grn.W_value.shape)
    params.W_out1 = _take(params.W_out1.shape)
    params.b_out1 = _take(params.b_out1.shape)
    params.W_out2 = _take(params.W_out2.shape)
    params.b_out2 = _take(params.b_out2.shape)
    return params


# ---------------------------------------------------------------------------
# Model Container
# ---------------------------------------------------------------------------

@dataclass
class TFTNumpyModel:
    params: TFTNumpyParams = None
    num_heads: int = 4
    lookback: int = 30
    hidden_size: int = 16
    scaler_mean: float = 0.0
    scaler_std: float = 1.0
    train_tail: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    lookback: int = 30,
    hidden_size: int = 16,
    num_heads: int = 4,
    lr: float = 0.001,
    epochs: int = 30,
    verbose: bool = True,
    seed: int = 42,
) -> TFTNumpyModel:
    """Train simplified TFT from scratch using stochastic gradient descent.

    Uses finite differences for gradient estimation since implementing
    full backprop through attention in pure NumPy is extremely complex.
    For efficiency, we use a subset of parameters and mini-batches.
    """
    # Scale
    mean_val = np.mean(train_data)
    std_val = np.std(train_data) + 1e-8
    data = (train_data - mean_val) / std_val

    # Create sequences
    X_seqs, y_targets = [], []
    for i in range(lookback, len(data)):
        X_seqs.append(data[i - lookback:i].reshape(-1, 1))
        y_targets.append(data[i])
    n_samples = len(X_seqs)

    # Initialise parameters
    params = _init_params(1, hidden_size, num_heads, seed)

    # Adam state
    flat = _flatten_params(params)
    n_params = len(flat)
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_step = 0

    rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        epoch_loss = 0.0
        indices = rng.permutation(n_samples)

        # Use mini-batch of sequences
        batch_size = min(32, n_samples)
        n_batches = max(1, n_samples // batch_size)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]

            # Compute loss and approximate gradient via finite differences
            # (stochastic, subset of parameters for speed)
            flat = _flatten_params(params)

            # Forward pass on batch
            batch_loss = 0.0
            for idx in batch_indices:
                y_hat, _ = _forward(X_seqs[idx], params, num_heads)
                batch_loss += (y_hat - y_targets[idx]) ** 2
            batch_loss /= len(batch_indices)

            # Approximate gradient via simultaneous perturbation (SPSA)
            delta = rng.choice([-1, 1], size=n_params)
            eps_spsa = max(0.01 / (1 + epoch * 0.1), 0.001)

            # Perturbed forward pass
            flat_plus = flat + eps_spsa * delta
            params_plus = _unflatten_params(flat_plus.copy(), params, num_heads)
            loss_plus = 0.0
            for idx in batch_indices:
                y_hat_p, _ = _forward(X_seqs[idx], params_plus, num_heads)
                loss_plus += (y_hat_p - y_targets[idx]) ** 2
            loss_plus /= len(batch_indices)

            flat_minus = flat - eps_spsa * delta
            params_minus = _unflatten_params(flat_minus.copy(), params, num_heads)
            loss_minus = 0.0
            for idx in batch_indices:
                y_hat_m, _ = _forward(X_seqs[idx], params_minus, num_heads)
                loss_minus += (y_hat_m - y_targets[idx]) ** 2
            loss_minus /= len(batch_indices)

            # SPSA gradient estimate
            grad = (loss_plus - loss_minus) / (2 * eps_spsa * delta)
            grad = np.clip(grad, -1.0, 1.0)

            # Adam update
            t_step += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            flat = flat - lr * m_hat / (np.sqrt(v_hat) + eps)

            params = _unflatten_params(flat.copy(), params, num_heads)
            epoch_loss += batch_loss

        avg_loss = epoch_loss / n_batches
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, MSE: {avg_loss:.6f}")

    model = TFTNumpyModel(
        params=params, num_heads=num_heads, lookback=lookback,
        hidden_size=hidden_size, scaler_mean=mean_val, scaler_std=std_val,
        train_tail=data[-lookback:],
    )
    return model


def _predict_n_steps(model: TFTNumpyModel, n_steps: int) -> np.ndarray:
    history = list(model.train_tail)
    predictions = []

    for _ in range(n_steps):
        seq = np.array(history[-model.lookback:]).reshape(-1, 1)
        y_hat, _ = _forward(seq, model.params, model.num_heads)
        predictions.append(y_hat)
        history.append(y_hat)

    return np.array(predictions) * model.scaler_std + model.scaler_mean


def validate(model: TFTNumpyModel, val_data: np.ndarray) -> Dict[str, float]:
    predictions = _predict_n_steps(model, len(val_data))
    return _compute_metrics(val_data, predictions)


def test(model: TFTNumpyModel, test_data: np.ndarray,
         val_data: Optional[np.ndarray] = None) -> Dict[str, float]:
    val_len = len(val_data) if val_data is not None else 0
    total = val_len + len(test_data)
    all_preds = _predict_n_steps(model, total)
    return _compute_metrics(test_data, all_preds[val_len:])


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    lookback = trial.suggest_int("lookback", 7, 40)
    hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 32])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 10, 30, step=5)

    try:
        model = train(train_data, lookback=lookback, hidden_size=hidden_size,
                       num_heads=num_heads, lr=lr, epochs=epochs, verbose=False)
        metrics = validate(model, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=10):
    study = optuna.create_study(direction="minimize", study_name="tft_numpy")
    study.optimize(lambda t: optuna_objective(t, train_data, val_data),
                   n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


def ray_tune_search(train_data, val_data, num_samples=10):
    def _trainable(config):
        try:
            model = train(train_data, lookback=config["lookback"],
                           hidden_size=config["hidden_size"],
                           num_heads=config["num_heads"],
                           lr=config["lr"], epochs=config["epochs"], verbose=False)
            metrics = validate(model, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "lookback": tune.randint(7, 40),
        "hidden_size": tune.choice([8, 16, 32]),
        "num_heads": tune.choice([2, 4]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([10, 15, 20, 25]),
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


def main():
    print("=" * 70)
    print("Attention-Based Time Series - NumPy From-Scratch (TFT-inspired)")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=500)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=8)
    bp = study.best_params

    print("\n--- Training Best ---")
    best_model = train(train_data, lookback=bp["lookback"],
                        hidden_size=bp["hidden_size"],
                        num_heads=bp["num_heads"],
                        lr=bp["lr"], epochs=bp["epochs"])

    print("\n--- Validation ---")
    for k, v in validate(best_model, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Test ---")
    for k, v in test(best_model, test_data, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Ray Tune ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=6)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
