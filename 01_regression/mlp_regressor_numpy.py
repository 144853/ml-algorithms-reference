"""
MLP Regressor - NumPy From-Scratch Implementation
==================================================

COMPLETE ML TUTORIAL: This file builds a Multi-Layer Perceptron (MLP) neural
network from scratch using only NumPy. Every forward pass, backward pass
(backpropagation), and gradient computation is implemented manually, giving
you complete visibility into how neural networks learn.

Theory & Mathematics:
    A Multi-Layer Perceptron (MLP) is a universal function approximator
    consisting of layers of neurons connected by learnable weight matrices.

    Architecture:
        Input (n_features) --> Hidden_1 (h1) --> ... --> Hidden_L (hL) --> Output (1)

    Forward Pass (for layer l):
        z_l = W_l @ a_{l-1} + b_l          (pre-activation)
        a_l = activation(z_l)               (activation)

    Output layer (no activation for regression):
        y_hat = W_out @ a_L + b_out

    Loss Function (MSE):
        L = (1/n) sum (y_i - y_hat_i)^2

    Backpropagation:
        Output layer:
            delta_out = (2/n) * (y_hat - y)
            dW_out = delta_out @ a_L^T
        Hidden layer l (going backward):
            delta_l = (W_{l+1}^T @ delta_{l+1}) * activation'(z_l)
            dW_l = delta_l @ a_{l-1}^T

    Weight Initialisation:
        He initialisation for ReLU:  W ~ N(0, sqrt(2/fan_in))
        Xavier for tanh/sigmoid:     W ~ N(0, sqrt(2/(fan_in + fan_out)))

Hyperparameters:
    - hidden_sizes (list[int]): Units per hidden layer. Default [64, 32].
    - activation (str): "relu", "tanh", "sigmoid". Default "relu".
    - lr (float): Learning rate. Default 0.001.
    - epochs (int): Training epochs. Default 200.
    - batch_size (int): Mini-batch size. Default 32.
"""

import logging
import time
from functools import partial

import numpy as np
import optuna
import ray
from ray import tune
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation functions and their derivatives
# ---------------------------------------------------------------------------

def relu(z):
    """ReLU activation: f(z) = max(0, z).
    WHY ReLU: Avoids vanishing gradient problem (gradient is 1 for z > 0).
    Most popular activation for hidden layers in modern networks.
    """
    return np.maximum(0, z)

def relu_deriv(z):
    """ReLU derivative: 1 if z > 0, else 0.
    WHY step function: ReLU is piecewise linear; derivative is piecewise constant.
    Note: undefined at z=0, but we use 0 by convention (measure-zero set).
    """
    return (z > 0).astype(z.dtype)

def tanh(z):
    """Tanh activation: output in (-1, 1). Zero-centered.
    WHY tanh: Zero-centered outputs help subsequent layers learn faster.
    Saturates at extremes (vanishing gradient for |z| >> 0).
    """
    return np.tanh(z)

def tanh_deriv(z):
    """Tanh derivative: 1 - tanh(z)^2. Maximum = 1 at z=0, decays to 0."""
    return 1.0 - np.tanh(z) ** 2

def sigmoid(z):
    """Sigmoid activation: output in (0, 1).
    WHY clip: Prevents numerical overflow in exp() for extreme values.
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_deriv(z):
    """Sigmoid derivative: s(z) * (1 - s(z)). Maximum = 0.25 at z=0."""
    s = sigmoid(z)
    return s * (1.0 - s)

# Map activation names to (function, derivative) pairs.
ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "tanh": (tanh, tanh_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
}


# ---------------------------------------------------------------------------
# MLP Model (from scratch)
# ---------------------------------------------------------------------------

class MLPRegressorNumpy:
    """Multi-layer perceptron regressor from scratch with NumPy.

    This implementation manually computes forward passes, backward passes
    (backpropagation), and gradient updates. No deep learning framework
    is used -- just NumPy matrix operations.
    """

    def __init__(self, input_dim, hidden_sizes=None, activation="relu",
                 lr=0.001, epochs=200, batch_size=32):
        # hidden_sizes: Number of neurons in each hidden layer.
        # WHY [64, 32] default: Two layers with decreasing width is a common
        # "funnel" architecture that progressively compresses representations.
        self.hidden_sizes = hidden_sizes or [64, 32]

        self.activation_name = activation
        # Store both the activation function and its derivative.
        # WHY derivative: Needed for backpropagation to compute gradients.
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []

        # Build the network architecture.
        # layer_dims = [input_dim, h1, h2, ..., 1] for regression.
        layer_dims = [input_dim] + self.hidden_sizes + [1]

        # Initialise weights and biases for each layer.
        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]

            # Weight initialisation strategy depends on activation.
            if activation == "relu":
                # He initialisation: W ~ N(0, sqrt(2/fan_in))
                # WHY: Accounts for ReLU zeroing out ~half the neurons.
                # Using sqrt(2/fan_in) keeps the variance of activations
                # stable across layers, preventing signal from vanishing or exploding.
                std = np.sqrt(2.0 / fan_in)
            else:
                # Xavier/Glorot initialisation: W ~ N(0, sqrt(2/(fan_in + fan_out)))
                # WHY: For sigmoid/tanh, considers both input and output dimensions.
                # Keeps gradients in a reasonable range during backprop.
                std = np.sqrt(2.0 / (fan_in + fan_out))

            # Random normal initialisation with computed std.
            W = np.random.randn(fan_out, fan_in) * std

            # Bias initialised to zero.
            # WHY zero: Biases do not have the symmetry-breaking concern that
            # weights have. Starting at zero is standard.
            b = np.zeros((fan_out, 1))

            self.weights.append(W)
            self.biases.append(b)

    def _forward(self, X):
        """Forward pass through the network.

        Computes activations layer by layer, storing intermediate values
        needed for backpropagation.

        Args:
            X: Input matrix of shape (batch_size, n_features).

        Returns:
            activations: List of activation values at each layer.
            pre_activations: List of pre-activation values (before activation fn).
        """
        # Transpose X so each column is a sample: (n_features, batch_size).
        # WHY transpose: Our weight matrices are (fan_out, fan_in), so
        # W @ a gives (fan_out, batch_size) -- one output per sample.
        a = X.T
        activations = [a]           # Store input as the first "activation"
        pre_activations = [None]    # Input has no pre-activation

        for i in range(len(self.weights)):
            # z = W @ a + b (linear transformation + bias)
            z = self.weights[i] @ a + self.biases[i]
            pre_activations.append(z)

            if i < len(self.weights) - 1:
                # Hidden layers: apply activation function.
                a = self.act_fn(z)
            else:
                # Output layer: NO activation for regression.
                # WHY: Regression targets can be any real number. Applying
                # sigmoid would limit output to (0,1), tanh to (-1,1).
                a = z

            activations.append(a)

        return activations, pre_activations

    def _backward(self, activations, pre_activations, y_batch):
        """Backpropagation: compute gradients for all weights and biases.

        Uses the chain rule to propagate the error signal from the output
        layer back through all hidden layers, computing dL/dW and dL/db
        for each layer.

        Args:
            activations: From forward pass.
            pre_activations: From forward pass.
            y_batch: True target values for this batch.

        Returns:
            dW_list: Gradient of loss w.r.t. each weight matrix.
            db_list: Gradient of loss w.r.t. each bias vector.
        """
        n = y_batch.shape[0]  # Batch size
        n_layers = len(self.weights)
        dW_list = [None] * n_layers
        db_list = [None] * n_layers

        # OUTPUT LAYER gradient.
        # MSE derivative: dL/dy_hat = (2/n) * (y_hat - y)
        y_row = y_batch.reshape(1, -1)  # Shape: (1, batch_size)
        delta = (2.0 / n) * (activations[-1] - y_row)

        # Weight gradient: dW = delta @ a_prev^T / n (averaged over batch).
        dW_list[-1] = delta @ activations[-2].T / n
        # Bias gradient: db = mean of delta across batch.
        db_list[-1] = np.sum(delta, axis=1, keepdims=True) / n

        # HIDDEN LAYERS gradient (backpropagate from output toward input).
        for i in range(n_layers - 2, -1, -1):
            # Propagate error through the weight matrix of the next layer.
            # Then multiply by the derivative of this layer's activation.
            # This is the CHAIN RULE in action:
            #   delta_l = (W_{l+1}^T @ delta_{l+1}) * f'(z_l)
            delta = (self.weights[i + 1].T @ delta) * self.act_deriv(pre_activations[i + 1])

            # Compute gradients for this layer's weights and biases.
            dW_list[i] = delta @ activations[i].T / n
            db_list[i] = np.sum(delta, axis=1, keepdims=True) / n

        return dW_list, db_list

    def fit(self, X, y):
        """Train the MLP using mini-batch gradient descent.

        For each epoch:
        1. Shuffle the data (prevents order-dependent learning).
        2. Split into mini-batches.
        3. For each batch: forward pass -> compute loss -> backward pass -> update weights.
        """
        n = X.shape[0]
        self.loss_history = []

        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch.
            # WHY: Ensures each epoch sees the data in a different order,
            # which acts as a form of regularisation and prevents the model
            # from learning patterns based on data ordering.
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            epoch_loss = 0.0
            n_batches = 0

            # Process mini-batches.
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                X_b = X_shuffled[start:end]
                y_b = y_shuffled[start:end]

                # Forward pass.
                activations, pre_activations = self._forward(X_b)

                # Backward pass (compute gradients).
                dW_list, db_list = self._backward(activations, pre_activations, y_b)

                # Update weights using gradient descent.
                # w = w - lr * dw
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * dW_list[i]
                    self.biases[i] -= self.lr * db_list[i]

                # Track batch loss.
                y_pred = activations[-1].flatten()
                batch_loss = np.mean((y_b - y_pred) ** 2)
                epoch_loss += batch_loss
                n_batches += 1

            epoch_loss /= n_batches
            self.loss_history.append(epoch_loss)

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.debug("Epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, epoch_loss)

        return self

    def predict(self, X):
        """Generate predictions for new data."""
        activations, _ = self._forward(X)
        return activations[-1].flatten()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic data with target scaling for better MLP convergence."""
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

    # Scale the target variable too for MLP convergence.
    # WHY: Neural networks learn better when targets are on a similar scale
    # to activations. Large targets cause large gradients which cause instability.
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train the from-scratch MLP regressor."""
    hidden_sizes = hp.get("hidden_sizes", [64, 32])
    if isinstance(hidden_sizes, tuple):
        hidden_sizes = list(hidden_sizes)
    model = MLPRegressorNumpy(
        input_dim=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        activation=hp.get("activation", "relu"),
        lr=hp.get("lr", 0.001),
        epochs=hp.get("epochs", 200),
        batch_size=hp.get("batch_size", 32),
    )
    model.fit(X_train, y_train)
    logger.info("MLP trained  hidden=%s  activation=%s  epochs=%d  final_loss=%.6f",
                hidden_sizes, model.activation_name, model.epochs,
                model.loss_history[-1] if model.loss_history else float("nan"))
    return model


def _metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def validate(model, X_val, y_val):
    m = _metrics(y_val, model.predict(X_val))
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    m = _metrics(y_test, model.predict(X_test))
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare different MLP architectures and hyperparameters.

    Demonstrates how depth, width, activation, and learning rate affect
    the model's ability to learn non-linear patterns.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: MLP Architecture & Training Trade-offs")
    print("=" * 80)

    configs = {
        "Shallow & Narrow [32]": {
            "hidden_sizes": [32], "activation": "relu",
            "lr": 0.001, "epochs": 200, "batch_size": 32,
            # WHY: Minimal network. Fast but limited capacity.
            # USE WHEN: Simple relationships, small datasets.
        },
        "Standard [64, 32]": {
            "hidden_sizes": [64, 32], "activation": "relu",
            "lr": 0.001, "epochs": 200, "batch_size": 32,
            # WHY: Good default. Two layers can approximate complex functions.
        },
        "Deep & Wide [128, 64, 32]": {
            "hidden_sizes": [128, 64, 32], "activation": "relu",
            "lr": 0.001, "epochs": 200, "batch_size": 32,
            # WHY: More capacity. Can learn more complex patterns.
            # TRADE-OFF: Slower training, risk of overfitting.
        },
        "Tanh Activation [64, 32]": {
            "hidden_sizes": [64, 32], "activation": "tanh",
            "lr": 0.001, "epochs": 200, "batch_size": 32,
            # WHY: Compare tanh vs ReLU. Tanh is zero-centered but can
            # suffer from vanishing gradients in deep networks.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        metrics["train_time"] = train_time
        n_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
        metrics["n_params"] = n_params
        results[name] = metrics

        print(f"\n  {name}: MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}  Params={n_params}  Time={train_time:.2f}s")

    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<30} {'Params':>8} {'MSE':>10} {'R2':>10} {'Time(s)':>8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<30} {m['n_params']:>8} {m['mse']:>10.4f} {m['r2']:>10.4f} {m['train_time']:>8.2f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Deeper/wider networks have more capacity but risk overfitting.")
    print("  2. ReLU generally trains faster than tanh for deeper networks.")
    print("  3. More parameters = slower training but potentially better fit.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Air Quality Index Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate MLP for Air Quality Index (AQI) prediction.

    DOMAIN CONTEXT: AQI prediction combines meteorological data (temperature,
    humidity, wind) with pollution source data to forecast air quality.
    The relationship is NON-LINEAR: e.g., temperature inversions trap
    pollutants near the ground, creating non-linear interactions.

    WHY MLP: Linear regression cannot capture the non-linear interactions
    between weather and pollution. MLP can learn these complex mappings.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Air Quality Index (AQI) Prediction (NumPy MLP)")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 2000

    # Weather features.
    temperature = np.random.normal(20, 10, n_samples)  # Celsius
    humidity = np.random.uniform(20, 95, n_samples)     # Percentage
    wind_speed = np.random.exponential(8, n_samples).clip(0, 50)  # km/h
    pressure = np.random.normal(1013, 10, n_samples)    # hPa

    # Pollution sources.
    traffic_density = np.random.exponential(50, n_samples).clip(0, 200)
    industrial_output = np.random.exponential(30, n_samples).clip(0, 100)

    feature_names = ["temperature", "humidity", "wind_speed",
                     "pressure", "traffic_density", "industrial_output"]
    X = np.column_stack([temperature, humidity, wind_speed,
                         pressure, traffic_density, industrial_output])

    # NON-LINEAR AQI model:
    # - High temp + low wind = bad (temperature inversion traps pollutants)
    # - Wind disperses pollutants (inverse relationship)
    # - Traffic and industry are pollution sources
    y = (
        50                                              # Base AQI
        + 0.5 * traffic_density                         # Linear traffic effect
        + 0.3 * industrial_output                       # Linear industry effect
        - 3.0 * np.sqrt(wind_speed + 1)                 # Non-linear wind dispersal
        + 0.01 * temperature ** 2                       # Non-linear temp effect
        + 0.001 * traffic_density * (30 - wind_speed)   # Interaction term
        + np.random.normal(0, 10, n_samples)            # Noise
    )
    y = y.clip(0, 500)

    print(f"\nDataset: {n_samples} hourly readings")
    print(f"Features: {', '.join(feature_names)}")
    print(f"AQI range: {y.min():.0f} - {y.max():.0f}")
    print(f"Note: AQI depends NON-LINEARLY on features (temperature inversion, wind dispersal)")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Scale targets for MLP convergence.
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    model = train(X_train_s, y_train_s, hidden_sizes=[64, 32], activation="relu",
                  lr=0.001, epochs=300, batch_size=32)

    metrics = _metrics(y_test_s, model.predict(X_test_s))
    print(f"\n--- Performance (on scaled targets) ---")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")

    # Show predictions in original scale.
    preds_orig = model.predict(X_test_s) * y_std + y_mean
    rmse_orig = np.sqrt(np.mean((y_test - preds_orig) ** 2))
    print(f"\n  RMSE in AQI units: {rmse_orig:.2f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = [trial.suggest_int(f"n_units_l{i}", 16, 128, step=16) for i in range(n_layers)]

    hp = {
        "hidden_sizes": layers,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "lr": trial.suggest_float("lr", 1e-4, 0.05, log=True),
        "epochs": trial.suggest_int("epochs", 50, 300, step=50),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    study = optuna.create_study(direction="minimize", study_name="mlp_numpy")
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
        "hidden_sizes": tune.choice([[32], [64], [64, 32], [128, 64], [64, 32, 16]]),
        "activation": tune.choice(["relu", "tanh"]),
        "lr": tune.loguniform(1e-4, 0.05),
        "epochs": tune.choice([100, 200, 300]),
        "batch_size": tune.choice([16, 32, 64]),
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
    print("MLP Regressor - NumPy From Scratch")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=15)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=8)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    # Reconstruct hidden_sizes from Optuna.
    best_p = dict(study.best_trial.params)
    n_layers = best_p.pop("n_layers", 1)
    layers = [best_p.pop(f"n_units_l{i}", 64) for i in range(n_layers)]
    best_p["hidden_sizes"] = layers

    model = train(X_train, y_train, **best_p)
    n_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
    print(f"\nArchitecture: {model.hidden_sizes}")
    print(f"Total parameters: {n_params}")

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
