"""
LSTM Time Series Forecasting - Sklearn Pipeline Wrapper
========================================================

Theory & Mathematics:
    Long Short-Term Memory (LSTM) networks are a type of recurrent neural
    network (RNN) designed to learn long-term dependencies via gating mechanisms.

    LSTM Cell Equations (at time step t):
        f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)          Forget gate
        i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)          Input gate
        c_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)       Candidate cell state
        c_t = f_t * c_{t-1} + i_t * c_tilde               Cell state update
        o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)          Output gate
        h_t = o_t * tanh(c_t)                              Hidden state

    where:
        - sigma: sigmoid activation function
        - *: element-wise multiplication (Hadamard product)
        - [h_{t-1}, x_t]: concatenation of previous hidden state and input
        - W_f, W_i, W_c, W_o: weight matrices for each gate
        - b_f, b_i, b_c, b_o: bias vectors for each gate

    For Time Series Forecasting:
        - Input: sliding windows of past observations (lookback window)
        - Architecture: LSTM layers -> Dense layers -> single output
        - Loss: MSE for point forecasting
        - Training: Teacher forcing (using true past values)

    This module wraps a PyTorch LSTM model in an sklearn-compatible interface
    (fit/predict methods) for integration with sklearn pipelines.

Business Use Cases:
    - Complex nonlinear time series with long memory
    - Multivariate time series forecasting
    - Financial market prediction
    - Anomaly detection in sequences
    - Natural language processing embeddings for time features

Advantages:
    - Captures long-range dependencies via gating
    - Handles nonlinear patterns
    - Can process multivariate inputs
    - Sklearn-compatible interface for easy pipeline integration
    - Well-suited for GPU acceleration

Disadvantages:
    - Requires substantial data for good performance
    - Computationally expensive to train
    - Many hyperparameters
    - Risk of overfitting on small datasets
    - Sequential nature limits parallelism

Key Hyperparameters:
    - lookback (int): Window size for input sequences
    - hidden_size (int): LSTM hidden state dimension
    - num_layers (int): Number of stacked LSTM layers
    - dropout (float): Dropout rate between LSTM layers
    - lr (float): Learning rate
    - epochs (int): Training epochs
    - batch_size (int): Training batch size

References:
    - Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
    - Gers, F.A. et al. (2000). Learning to Forget: Continual Prediction with LSTM.
"""

import numpy as np
import pandas as pd
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
    """Generate synthetic time series with trend, seasonality, and noise."""
    np.random.seed(seed)
    t = np.arange(n_points, dtype=np.float64)
    trend = trend_slope * t
    seasonality = 2.0 * np.sin(2 * np.pi * t / seasonal_period)
    noise = np.random.normal(0, noise_std, n_points)
    values = 10.0 + trend + seasonality + noise
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
# PyTorch LSTM Module
# ---------------------------------------------------------------------------

class _LSTMModule(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out).squeeze(-1)


# ---------------------------------------------------------------------------
# Sklearn-compatible LSTM Wrapper
# ---------------------------------------------------------------------------

class LSTMForecaster(BaseEstimator, RegressorMixin):
    """Sklearn-compatible LSTM forecaster for time series.

    Implements fit() and predict() methods compatible with sklearn pipelines.
    Internally uses PyTorch for the LSTM model.
    """

    def __init__(
        self,
        lookback: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loss_ = None

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for supervised learning."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y=None):
        """Fit LSTM model on time series data.

        Args:
            X: 1-D array of time series values (train data).
            y: Ignored (present for sklearn API compatibility).
        """
        # Scale data
        data = self.scaler_.fit_transform(X.reshape(-1, 1)).flatten()

        # Create sequences
        X_seq, y_seq = self._create_sequences(data)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model_ = _LSTMModule(
            input_size=1, hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout,
        ).to(self.device_)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self.train_loss_ = best_loss
        # Store the scaled training data tail for prediction
        self._train_tail_ = data[-self.lookback:]
        self._raw_train_ = X.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate forecasts.

        If X is a 1-D array of length n, generates n-step ahead forecasts
        using recursive prediction.
        """
        self.model_.eval()
        n_steps = len(X)

        # Start with the last lookback values from training
        history = list(self._train_tail_)
        predictions = []

        with torch.no_grad():
            for _ in range(n_steps):
                seq = np.array(history[-self.lookback:])
                x_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                x_t = x_t.to(self.device_)
                pred = self.model_(x_t).item()
                predictions.append(pred)
                history.append(pred)

        # Inverse transform
        preds_scaled = np.array(predictions).reshape(-1, 1)
        return self.scaler_.inverse_transform(preds_scaled).flatten()


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(train_data: np.ndarray, **kwargs) -> LSTMForecaster:
    """Train LSTM forecaster."""
    model = LSTMForecaster(**kwargs)
    model.fit(train_data)
    return model


def validate(model: LSTMForecaster, val_data: np.ndarray) -> Dict[str, float]:
    predictions = model.predict(val_data)
    return _compute_metrics(val_data, predictions)


def test(model: LSTMForecaster, test_data: np.ndarray,
         val_data: np.ndarray = None) -> Dict[str, float]:
    if val_data is not None:
        # First predict through validation to update history
        _ = model.predict(val_data)
    predictions = model.predict(test_data)
    return _compute_metrics(test_data, predictions)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    lookback = trial.suggest_int("lookback", 7, 60)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 30, 100, step=10)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    try:
        model = train(
            train_data, lookback=lookback, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, lr=lr,
            epochs=epochs, batch_size=batch_size, verbose=False,
        )
        metrics = validate(model, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=20):
    study = optuna.create_study(direction="minimize", study_name="lstm_sklearn")
    study.optimize(lambda t: optuna_objective(t, train_data, val_data),
                   n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data, val_data, num_samples=20):
    def _trainable(config):
        try:
            model = train(
                train_data, lookback=config["lookback"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                dropout=config["dropout"], lr=config["lr"],
                epochs=config["epochs"], batch_size=config["batch_size"],
                verbose=False,
            )
            metrics = validate(model, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "lookback": tune.randint(7, 60),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.randint(1, 4),
        "dropout": tune.uniform(0.0, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([30, 50, 80, 100]),
        "batch_size": tune.choice([16, 32, 64]),
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
    print("LSTM Time Series - Sklearn Pipeline Wrapper")
    print("=" * 70)

    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=15)
    bp = study.best_params

    print("\n--- Training Best ---")
    best_model = train(
        train_data, lookback=bp["lookback"], hidden_size=bp["hidden_size"],
        num_layers=bp["num_layers"], dropout=bp["dropout"],
        lr=bp["lr"], epochs=bp["epochs"], batch_size=bp["batch_size"],
    )

    print("\n--- Validation ---")
    for k, v in validate(best_model, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Test ---")
    # Re-train for clean prediction
    best_model2 = train(
        train_data, lookback=bp["lookback"], hidden_size=bp["hidden_size"],
        num_layers=bp["num_layers"], dropout=bp["dropout"],
        lr=bp["lr"], epochs=bp["epochs"], batch_size=bp["batch_size"],
        verbose=False,
    )
    for k, v in test(best_model2, test_data, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Ray Tune ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
