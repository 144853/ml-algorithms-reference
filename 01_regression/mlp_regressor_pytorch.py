"""
MLP Regressor - PyTorch Implementation
=======================================

Theory & Mathematics:
    A Multi-Layer Perceptron (MLP) Regressor is a fully-connected
    feed-forward neural network that learns non-linear mappings from input
    features to a continuous target.

    This PyTorch implementation uses ``nn.Module`` to define a configurable
    architecture with:
        - Variable number of hidden layers and units per layer
        - Choice of activation function (ReLU, Tanh, LeakyReLU, GELU)
        - Optional Batch Normalisation between layers
        - Optional Dropout for regularisation
        - L2 regularisation via weight_decay in the optimiser

    Architecture:
        Input --> [Linear -> BN -> Activation -> Dropout] x L --> Linear --> Output

    Forward Pass for hidden layer l:
        z_l = W_l @ a_{l-1} + b_l
        bn_l = BatchNorm(z_l)           (optional)
        a_l = activation(bn_l)
        a_l = Dropout(a_l, p)           (optional, training only)

    Training uses mini-batch SGD or Adam with the MSE loss:
        L = (1/n) sum (y_i - y_hat_i)^2

    PyTorch autograd handles all gradient computation automatically.

Business Use Cases:
    - Production ML pipelines with GPU acceleration
    - Non-linear regression on tabular data (pricing, demand forecasting)
    - Multi-task regression (shared backbone with multiple output heads)
    - Part of larger architectures (e.g., neural collaborative filtering)

Advantages:
    - Full GPU acceleration for large datasets
    - Flexible architecture (batch norm, dropout, custom layers)
    - Automatic differentiation eliminates manual gradient computation
    - Integration with PyTorch ecosystem (TensorBoard, ONNX, torchserve)

Disadvantages:
    - More code / boilerplate than scikit-learn for simple tasks
    - Requires careful hyperparameter tuning (lr, architecture, regularisation)
    - Risk of overfitting on small datasets without proper regularisation
    - Training stability can require learning rate scheduling

Hyperparameters:
    - hidden_sizes (list[int]): Units per hidden layer. Default [128, 64, 32].
    - activation (str): "relu", "tanh", "leaky_relu", "gelu". Default "relu".
    - lr (float): Learning rate. Default 0.001.
    - epochs (int): Training epochs. Default 200.
    - batch_size (int): Mini-batch size. Default 32.
    - optimizer (str): "adam" or "sgd". Default "adam".
    - weight_decay (float): L2 regularisation. Default 1e-5.
    - dropout (float): Dropout probability. Default 0.0.
    - use_batch_norm (bool): Whether to use batch normalisation. Default True.
"""

import logging
from functools import partial

import numpy as np
import optuna
import ray
import torch
import torch.nn as nn
from ray import tune
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPRegressorModel(nn.Module):
    """Configurable multi-layer perceptron for regression."""

    def __init__(self, input_dim, hidden_sizes=None, activation="relu",
                 dropout=0.0, use_batch_norm=True):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64, 32]
        act_cls = ACTIVATION_MAP.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(1, n_features // 2), noise=noise,
        random_state=random_state,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


def _to_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    hidden_sizes = hp.get("hidden_sizes", [128, 64, 32])
    if isinstance(hidden_sizes, tuple):
        hidden_sizes = list(hidden_sizes)
    activation = hp.get("activation", "relu")
    lr = hp.get("lr", 0.001)
    epochs = hp.get("epochs", 200)
    batch_size = hp.get("batch_size", 32)
    opt_name = hp.get("optimizer", "adam")
    weight_decay = hp.get("weight_decay", 1e-5)
    dropout = hp.get("dropout", 0.0)
    use_batch_norm = hp.get("use_batch_norm", True)

    n_features = X_train.shape[1]
    model = MLPRegressorModel(
        input_dim=n_features,
        hidden_sizes=hidden_sizes,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    ).to(DEVICE)

    criterion = nn.MSELoss()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    loader = _to_loader(X_train, y_train, batch_size=batch_size)

    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)
        epoch_loss /= len(loader.dataset)
        scheduler.step(epoch_loss)
        best_loss = min(best_loss, epoch_loss)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.debug("Epoch %d/%d  loss=%.6f  lr=%.2e",
                         epoch + 1, epochs, epoch_loss, optimizer.param_groups[0]["lr"])

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Training done  hidden=%s  params=%d  best_loss=%.6f", hidden_sizes, total_params, best_loss)
    return model


def _evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return {
        "mse": mean_squared_error(y, preds),
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
    }


def validate(model, X_val, y_val):
    m = _evaluate(model, X_val, y_val)
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    m = _evaluate(model, X_test, y_test)
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"n_units_l{i}", 16, 256, step=16))

    hp = {
        "hidden_sizes": layers,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu", "gelu"]),
        "lr": trial.suggest_float("lr", 1e-4, 0.05, log=True),
        "epochs": trial.suggest_int("epochs", 50, 300, step=50),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "use_batch_norm": trial.suggest_categorical("use_batch_norm", [True, False]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    study = optuna.create_study(direction="minimize", study_name="mlp_pytorch")
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        n_trials=n_trials, show_progress_bar=True,
    )
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    model = train(X_train, y_train, **config)
    metrics = validate(model, X_val, y_val)
    tune.report(**metrics)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "hidden_sizes": tune.choice([[64], [128, 64], [128, 64, 32], [256, 128, 64], [256, 128, 64, 32]]),
        "activation": tune.choice(["relu", "tanh", "gelu"]),
        "lr": tune.loguniform(1e-4, 0.05),
        "epochs": tune.choice([100, 200, 300]),
        "batch_size": tune.choice([16, 32, 64]),
        "optimizer": tune.choice(["adam", "sgd"]),
        "weight_decay": tune.loguniform(1e-7, 1e-2),
        "dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
        "use_batch_norm": tune.choice([True, False]),
    }
    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    )
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="mse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MLP Regressor - PyTorch")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=15)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=8)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    # Reconstruct hidden_sizes from Optuna
    best_p = dict(study.best_trial.params)
    n_layers = best_p.pop("n_layers", 1)
    layers = []
    for i in range(n_layers):
        key = f"n_units_l{i}"
        layers.append(best_p.pop(key, 64))
    best_p["hidden_sizes"] = layers

    model = train(X_train, y_train, **best_p)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nArchitecture: {layers}")
    print(f"Total parameters: {total_params}")

    print("\n--- Validation ---")
    for k, v in validate(model, X_val, y_val).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n--- Test ---")
    for k, v in test(model, X_test, y_test).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
