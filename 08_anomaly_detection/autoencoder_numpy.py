"""
Autoencoder-Based Anomaly Detection - NumPy From-Scratch Implementation
========================================================================

Theory & Mathematics:
    This module implements a fully-connected autoencoder from scratch using
    only NumPy, including both forward propagation and backpropagation for
    gradient computation and weight updates.

    Autoencoder Architecture:
        The autoencoder consists of two parts:

        Encoder:  Input(d) -> h1(d1) -> h2(d2) -> ... -> Bottleneck(k)
        Decoder:  Bottleneck(k) -> h2'(d2) -> h1'(d1) -> ... -> Output(d)

        The bottleneck dimension k << d forces the network to learn a
        compressed representation of the input.

    Forward Pass:
        For layer l with weights W_l, bias b_l, and activation f_l:
            z_l = W_l @ a_{l-1} + b_l     (pre-activation)
            a_l = f_l(z_l)                 (activation)

        Activations used:
            - Hidden layers: ReLU(z) = max(0, z)
            - Output layer: Linear (identity) for reconstruction

    Loss Function (MSE):
        L(x, x_hat) = (1/d) * sum_i (x_i - x_hat_i)^2

        This measures reconstruction quality. High loss = anomaly.

    Backpropagation:
        Output layer error:
            delta_L = (a_L - x) * f'_L(z_L)    [f'_L = 1 for linear output]

        Hidden layer error (propagated backwards):
            delta_l = (W_{l+1}^T @ delta_{l+1}) * f'_l(z_l)

        Gradient computations:
            dL/dW_l = delta_l @ a_{l-1}^T / batch_size
            dL/db_l = mean(delta_l, axis=1)

        Weight updates (SGD with momentum or Adam):
            W_l = W_l - lr * dL/dW_l
            b_l = b_l - lr * dL/db_l

    ReLU Derivative:
        f'(z) = 1 if z > 0, else 0

    Adam Optimizer (implemented from scratch):
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat_t = m_t / (1 - beta1^t)
        v_hat_t = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)

    Anomaly Scoring:
        score(x) = MSE(x, AE(x)) = (1/d) * ||x - x_hat||^2
        Threshold: tau = percentile(scores_train, 100*(1-contamination))

Business Use Cases:
    - Fraud detection in financial transactions
    - Network intrusion detection
    - Manufacturing defect detection
    - Medical anomaly detection
    - Predictive maintenance

Advantages:
    - Full understanding of the algorithm (no black-box libraries)
    - Customizable architecture and training procedure
    - Educational: demonstrates backpropagation mechanics
    - No external deep learning framework required

Disadvantages:
    - Slower than optimized libraries (PyTorch, TensorFlow)
    - No GPU acceleration
    - Manual gradient computation is error-prone
    - Limited to fully-connected architectures
    - Numerical stability requires careful implementation

Key Hyperparameters:
    - layer_dims: list of layer dimensions [input, h1, ..., bottleneck, ..., h1', output]
    - learning_rate: step size for weight updates
    - n_epochs: number of training passes
    - batch_size: mini-batch size
    - beta1, beta2: Adam momentum parameters
    - contamination: expected fraction of anomalies

References:
    - Rumelhart, D.E., Hinton, G.E. and Williams, R.J., 1986.
      Learning representations by back-propagating errors. Nature.
    - Sakurada, M. and Yairi, T., 2014. Anomaly detection using autoencoders.
      MLSDA workshop.
    - Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
      optimization. ICLR 2015.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, z)."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)


def linear(z: np.ndarray) -> np.ndarray:
    """Linear (identity) activation for output layer."""
    return z


def linear_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of linear activation: always 1."""
    return np.ones_like(z)


# ---------------------------------------------------------------------------
# Autoencoder Implementation
# ---------------------------------------------------------------------------


class AutoencoderNumpy:
    """
    Autoencoder for anomaly detection implemented from scratch with NumPy.

    The network uses ReLU activations for hidden layers and linear activation
    for the output layer. Training uses the Adam optimizer with MSE loss.

    Parameters
    ----------
    layer_dims : list of int
        Dimensions for each layer. Should be symmetric for autoencoder.
        Example: [10, 64, 32, 8, 32, 64, 10] for input_dim=10, bottleneck=8.
    learning_rate : float
        Adam optimizer learning rate.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    beta1 : float
        Adam first moment decay rate.
    beta2 : float
        Adam second moment decay rate.
    epsilon : float
        Adam numerical stability constant.
    contamination : float
        Expected fraction of anomalies for threshold.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        layer_dims: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 64,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.layer_dims = layer_dims or [10, 64, 32, 8, 32, 64, 10]
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.contamination = contamination
        self.random_state = random_state

        self.weights_: List[np.ndarray] = []
        self.biases_: List[np.ndarray] = []
        self.threshold_: float = 0.0
        self.train_losses_: List[float] = []

        # Adam optimizer state
        self._m_w: List[np.ndarray] = []
        self._v_w: List[np.ndarray] = []
        self._m_b: List[np.ndarray] = []
        self._v_b: List[np.ndarray] = []
        self._t: int = 0

    def _initialize_weights(self) -> None:
        """
        Initialize weights using He initialization and biases to zero.

        He initialization: W ~ N(0, sqrt(2/fan_in))
        This is appropriate for ReLU activations.
        """
        rng = np.random.RandomState(self.random_state)
        self.weights_ = []
        self.biases_ = []
        self._m_w = []
        self._v_w = []
        self._m_b = []
        self._v_b = []

        n_layers = len(self.layer_dims) - 1
        for i in range(n_layers):
            fan_in = self.layer_dims[i]
            fan_out = self.layer_dims[i + 1]

            # He initialization
            std = np.sqrt(2.0 / fan_in)
            W = rng.randn(fan_out, fan_in) * std
            b = np.zeros((fan_out, 1))

            self.weights_.append(W)
            self.biases_.append(b)

            # Adam moment estimates
            self._m_w.append(np.zeros_like(W))
            self._v_w.append(np.zeros_like(W))
            self._m_b.append(np.zeros_like(b))
            self._v_b.append(np.zeros_like(b))

        self._t = 0

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network.

        Parameters
        ----------
        X : np.ndarray of shape (d, batch_size)
            Input data (columns are samples).

        Returns
        -------
        activations : list of np.ndarray
            Activations at each layer (including input).
        pre_activations : list of np.ndarray
            Pre-activation values (z) at each layer.
        """
        n_layers = len(self.weights_)
        activations = [X]
        pre_activations = [None]  # No pre-activation for input

        a = X
        for i in range(n_layers):
            z = self.weights_[i] @ a + self.biases_[i]
            pre_activations.append(z)

            if i < n_layers - 1:
                # Hidden layers: ReLU
                a = relu(z)
            else:
                # Output layer: Linear
                a = linear(z)

            activations.append(a)

        return activations, pre_activations

    def _backward(
        self,
        X: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation to compute gradients.

        Parameters
        ----------
        X : np.ndarray of shape (d, batch_size)
            Original input (target for reconstruction).
        activations : list of np.ndarray
            Forward pass activations.
        pre_activations : list of np.ndarray
            Forward pass pre-activations.

        Returns
        -------
        dW_list : list of np.ndarray
            Weight gradients for each layer.
        db_list : list of np.ndarray
            Bias gradients for each layer.
        """
        n_layers = len(self.weights_)
        batch_size = X.shape[1]

        dW_list = [None] * n_layers
        db_list = [None] * n_layers

        # Output layer error: d(MSE)/d(z_L) = (a_L - x) * f'_L(z_L)
        # For linear output: f'_L = 1, so delta = (a_L - x)
        output = activations[-1]
        delta = (output - X) * linear_derivative(pre_activations[-1])

        # Compute gradients for output layer
        dW_list[-1] = (delta @ activations[-2].T) / batch_size
        db_list[-1] = np.mean(delta, axis=1, keepdims=True)

        # Propagate backwards through hidden layers
        for i in range(n_layers - 2, -1, -1):
            delta = (self.weights_[i + 1].T @ delta) * relu_derivative(pre_activations[i + 1])
            dW_list[i] = (delta @ activations[i].T) / batch_size
            db_list[i] = np.mean(delta, axis=1, keepdims=True)

        return dW_list, db_list

    def _adam_update(
        self,
        dW_list: List[np.ndarray],
        db_list: List[np.ndarray],
    ) -> None:
        """
        Update weights and biases using Adam optimizer.

        Parameters
        ----------
        dW_list : list of np.ndarray
            Weight gradients.
        db_list : list of np.ndarray
            Bias gradients.
        """
        self._t += 1

        for i in range(len(self.weights_)):
            # Weight updates
            self._m_w[i] = self.beta1 * self._m_w[i] + (1 - self.beta1) * dW_list[i]
            self._v_w[i] = self.beta2 * self._v_w[i] + (1 - self.beta2) * (dW_list[i] ** 2)

            m_hat_w = self._m_w[i] / (1 - self.beta1 ** self._t)
            v_hat_w = self._v_w[i] / (1 - self.beta2 ** self._t)

            self.weights_[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            # Bias updates
            self._m_b[i] = self.beta1 * self._m_b[i] + (1 - self.beta1) * db_list[i]
            self._v_b[i] = self.beta2 * self._v_b[i] + (1 - self.beta2) * (db_list[i] ** 2)

            m_hat_b = self._m_b[i] / (1 - self.beta1 ** self._t)
            v_hat_b = self._v_b[i] / (1 - self.beta2 ** self._t)

            self.biases_[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def fit(self, X: np.ndarray) -> "AutoencoderNumpy":
        """
        Train the autoencoder on (mostly normal) data.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Training data (rows are samples).

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape

        # Validate / adjust layer dims
        if self.layer_dims[0] != n_features:
            self.layer_dims[0] = n_features
        if self.layer_dims[-1] != n_features:
            self.layer_dims[-1] = n_features

        self._initialize_weights()

        # Transpose so columns are samples: shape (d, n)
        X_T = X.T

        rng = np.random.RandomState(self.random_state + 1)

        for epoch in range(self.n_epochs):
            # Shuffle samples
            perm = rng.permutation(n_samples)
            X_shuffled = X_T[:, perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[:, start:end]
                batch_size = X_batch.shape[1]

                # Forward pass
                activations, pre_activations = self._forward(X_batch)

                # Compute loss: MSE
                output = activations[-1]
                loss = np.mean((output - X_batch) ** 2)
                epoch_loss += loss

                # Backward pass
                dW_list, db_list = self._backward(X_batch, activations, pre_activations)

                # Adam update
                self._adam_update(dW_list, db_list)

                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses_.append(avg_loss)

        # Set threshold from training reconstruction errors
        scores = self.anomaly_score(X)
        self.threshold_ = float(np.percentile(scores, 100 * (1 - self.contamination)))

        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input through the autoencoder.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n, d)
            Reconstructed output.
        """
        X_T = X.T
        activations, _ = self._forward(X_T)
        return activations[-1].T

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error (MSE per sample) as anomaly score.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n,)
        """
        X_hat = self.reconstruct(X)
        mse = np.mean((X - X_hat) ** 2, axis=1)
        return mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Returns
        -------
        np.ndarray of shape (n,)
            1 = anomaly, 0 = normal.
        """
        scores = self.anomaly_score(X)
        return (scores >= self.threshold_).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores."""
        return self.anomaly_score(X)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 2000,
    n_features: int = 10,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic anomaly-detection data.

    Normal: multivariate Gaussian. Anomalies: uniform in [-6, 6]^d.
    Split 60/20/20 train/val/test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    rng = np.random.RandomState(random_state)

    n_normal = int(n_samples * (1 - contamination))
    n_anomaly = n_samples - n_normal

    X_normal = rng.randn(n_normal, n_features)
    y_normal = np.zeros(n_normal, dtype=int)

    X_anomaly = rng.uniform(low=-6, high=6, size=(n_anomaly, n_features))
    y_anomaly = np.ones(n_anomaly, dtype=int)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Data generated: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Contamination: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: np.ndarray, **hyperparams) -> AutoencoderNumpy:
    """
    Train autoencoder anomaly detector from scratch.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    **hyperparams
        Forwarded to AutoencoderNumpy constructor.

    Returns
    -------
    AutoencoderNumpy
    """
    n_features = X_train.shape[1]

    defaults = dict(
        layer_dims=[n_features, 64, 32, 8, 32, 64, n_features],
        learning_rate=1e-3,
        n_epochs=100,
        batch_size=64,
        contamination=0.05,
        random_state=42,
    )
    defaults.update(hyperparams)

    model = AutoencoderNumpy(**defaults)
    model.fit(X_train)
    return model


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> Dict[str, Any]:
    """Compute anomaly detection metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    avg_precision = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_precision),
        "confusion_matrix": cm,
    }


def validate(
    model: AutoencoderNumpy,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate model on validation set."""
    y_pred = model.predict(X_val)
    scores = model.anomaly_score(X_val)
    return _compute_metrics(y_val, y_pred, scores)


def test(
    model: AutoencoderNumpy,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    scores = model.anomaly_score(X_test)
    return _compute_metrics(y_test, y_pred, scores)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Optuna objective for autoencoder hyperparameter search.

    Maximizes validation F1-score.
    """
    n_features = X_train.shape[1]

    # Architecture: encoder dims then mirror for decoder
    bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [4, 8, 16])
    hidden_dim_1 = trial.suggest_categorical("hidden_dim_1", [32, 64, 128])
    hidden_dim_2 = trial.suggest_categorical("hidden_dim_2", [16, 32, 64])

    # Ensure decreasing dimensions toward bottleneck
    if hidden_dim_2 >= hidden_dim_1:
        hidden_dim_2 = hidden_dim_1 // 2
    if bottleneck_dim >= hidden_dim_2:
        bottleneck_dim = max(2, hidden_dim_2 // 2)

    layer_dims = [
        n_features,
        hidden_dim_1,
        hidden_dim_2,
        bottleneck_dim,
        hidden_dim_2,
        hidden_dim_1,
        n_features,
    ]

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    n_epochs = trial.suggest_int("n_epochs", 50, 200)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    contamination = trial.suggest_float("contamination", 0.01, 0.15)

    model = train(
        X_train,
        layer_dims=layer_dims,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        contamination=contamination,
    )

    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Hyperparameter search using Ray Tune with Optuna.

    Parameters
    ----------
    X_train, X_val, y_val : np.ndarray
        Training/validation data.
    num_samples : int
        Number of trials.

    Returns
    -------
    dict
        Best config and metrics.
    """
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    n_features = X_train.shape[1]

    def trainable(config: Dict[str, Any]) -> None:
        layer_dims = config["layer_dims"]
        model = train(
            X_train,
            layer_dims=layer_dims,
            learning_rate=config["learning_rate"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            contamination=config["contamination"],
        )
        metrics = validate(model, X_val, y_val)
        tune.report(
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
        )

    search_space = {
        "layer_dims": tune.choice([
            [n_features, 64, 32, 8, 32, 64, n_features],
            [n_features, 128, 64, 16, 64, 128, n_features],
            [n_features, 64, 16, 64, n_features],
            [n_features, 32, 8, 32, n_features],
        ]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "n_epochs": tune.randint(50, 200),
        "batch_size": tune.choice([32, 64, 128]),
        "contamination": tune.uniform(0.01, 0.15),
    }

    optuna_search = OptunaSearch(metric="f1", mode="max")

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=num_samples,
            metric="f1",
            mode="max",
        ),
        run_config=ray.train.RunConfig(verbose=0),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="f1", mode="max")
    best_config = best_result.config
    best_metrics = best_result.metrics

    ray.shutdown()

    return {
        "best_config": best_config,
        "best_f1": best_metrics.get("f1"),
        "best_roc_auc": best_metrics.get("roc_auc"),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete from-scratch autoencoder anomaly detection pipeline."""
    print("=" * 70)
    print("Autoencoder Anomaly Detection - NumPy From-Scratch Implementation")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/5] Generating synthetic anomaly detection data...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=1500, n_features=10, contamination=0.05, random_state=42
    )

    # 2. Train with defaults
    print("\n[2/5] Training autoencoder from scratch...")
    n_features = X_train.shape[1]
    model = train(
        X_train,
        layer_dims=[n_features, 64, 32, 8, 32, 64, n_features],
        learning_rate=1e-3,
        n_epochs=100,
        batch_size=64,
        contamination=0.05,
    )
    print(f"  Final training loss: {model.train_losses_[-1]:.6f}")
    print(f"  Threshold: {model.threshold_:.6f}")

    # 3. Validate
    print("\n[3/5] Validation results:")
    val_metrics = validate(model, X_val, y_val)
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")

    # 4. Optuna
    print("\n[4/5] Running Optuna hyperparameter optimization (15 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, y_val),
        n_trials=15,
        show_progress_bar=False,
    )
    print(f"  Best trial F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Retrain with best params
    bp = study.best_params.copy()
    h1 = bp.pop("hidden_dim_1")
    h2 = bp.pop("hidden_dim_2")
    bn = bp.pop("bottleneck_dim")
    if h2 >= h1:
        h2 = h1 // 2
    if bn >= h2:
        bn = max(2, h2 // 2)
    layer_dims = [n_features, h1, h2, bn, h2, h1, n_features]
    best_model = train(X_train, layer_dims=layer_dims, **bp)

    # 5. Test
    print("\n[5/5] Test results (best model):")
    test_metrics = test(best_model, X_test, y_test)
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
