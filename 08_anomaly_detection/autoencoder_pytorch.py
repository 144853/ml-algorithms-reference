"""
Autoencoder-Based Anomaly Detection - PyTorch Implementation
=============================================================

Theory & Mathematics:
    This module implements a deep autoencoder for anomaly detection using
    PyTorch with a custom ``nn.Module``. The autoencoder learns a compressed
    representation of normal data, and anomalies are detected via high
    reconstruction error.

    Architecture:
        Encoder:  Input(d) -> Dense(h1, ReLU, BN) -> Dense(h2, ReLU, BN) -> Bottleneck(k)
        Decoder:  Bottleneck(k) -> Dense(h2, ReLU, BN) -> Dense(h1, ReLU, BN) -> Output(d)

        The bottleneck (latent space) has dimension k << d, forcing the network
        to learn a compact representation of the data distribution.

    Training Objective (MSE Reconstruction Loss):
        L(theta) = (1/N) * sum_{i=1}^{N} ||x_i - D(E(x_i; theta_E); theta_D)||^2

        where E is the encoder, D is the decoder, and theta = {theta_E, theta_D}.

    Anomaly Score:
        score(x) = ||x - AE(x)||^2 = sum_j (x_j - x_hat_j)^2

        High reconstruction error indicates the input deviates from learned
        normal patterns. A threshold tau classifies points as:
            - Normal  if score(x) < tau
            - Anomaly if score(x) >= tau

    Threshold Selection Strategies:
        1. Percentile-based: tau = P_{100*(1-c)}(scores_train)
           where c is the contamination rate
        2. Adaptive: tau = mean(scores) + k * std(scores) on validation set
        3. ROC-optimal: choose tau maximizing F1 on validation set

    Regularization:
        - Batch Normalization: stabilizes training, acts as implicit regularizer
        - Dropout: prevents overfitting, especially in hidden layers
        - L2 Weight Decay: via optimizer's weight_decay parameter
        - Early Stopping: monitor validation loss to prevent overfitting

    Training Details:
        - Optimizer: Adam (adaptive learning rate)
        - Learning Rate Scheduler: ReduceLROnPlateau (reduce LR on validation plateau)
        - Gradient Clipping: prevents exploding gradients
        - Mini-batch training with shuffled data

Business Use Cases:
    - Credit card fraud detection (abnormal transaction patterns)
    - Network intrusion detection (unusual traffic behavior)
    - Manufacturing quality control (product defect detection)
    - Medical imaging anomalies (pathological scan identification)
    - Time series anomaly detection (sensor data monitoring)
    - Cybersecurity (malware and phishing detection)

Advantages:
    - GPU acceleration for large-scale datasets
    - Flexible architecture design (depth, width, skip connections)
    - Captures complex non-linear relationships
    - Can be extended to variational autoencoders (VAE) or adversarial setups
    - Rich PyTorch ecosystem (schedulers, mixed precision, distributed training)
    - Reconstruction error is intuitive and interpretable

Disadvantages:
    - Requires careful hyperparameter tuning (architecture, LR, epochs)
    - Risk of overfitting on small datasets
    - Training is more complex than traditional ML methods
    - Bottleneck size critically affects performance
    - May learn to reconstruct anomalies if contamination is high
    - Less interpretable than tree-based methods

Key Hyperparameters:
    - encoder_dims: list of encoder layer dimensions
    - latent_dim: bottleneck dimension
    - dropout_rate: dropout probability
    - learning_rate: Adam optimizer learning rate
    - weight_decay: L2 regularization strength
    - n_epochs: number of training epochs
    - batch_size: mini-batch size
    - contamination: expected fraction of anomalies
    - patience: early stopping patience

References:
    - Sakurada, M. and Yairi, T., 2014. Anomaly detection using autoencoders
      with nonlinear dimensionality reduction. MLSDA workshop.
    - An, J. and Cho, S., 2015. Variational autoencoder based anomaly detection
      using reconstruction probability. Special Lecture on IE.
    - Zhou, C. and Paffenroth, R.C., 2017. Anomaly detection with robust deep
      autoencoders. KDD 2017.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Autoencoder Neural Network
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """
    Encoder network: maps input to a lower-dimensional latent space.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Sizes of hidden layers.
    latent_dim : int
        Dimension of the bottleneck (latent space).
    dropout_rate : float
        Dropout probability for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Decoder network: maps latent representation back to input space.

    Parameters
    ----------
    latent_dim : int
        Dimension of the bottleneck input.
    hidden_dims : list of int
        Sizes of hidden layers (typically reverse of encoder).
    output_dim : int
        Dimension of the reconstructed output (same as input_dim).
    dropout_rate : float
        Dropout probability.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim

        # Output layer: linear activation for reconstruction
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    """
    Full autoencoder: encoder + decoder.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    encoder_dims : list of int
        Encoder hidden layer sizes.
    latent_dim : int
        Bottleneck dimension.
    dropout_rate : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_dims = encoder_dims or [64, 32]
        decoder_dims = list(reversed(encoder_dims))

        self.encoder = Encoder(input_dim, encoder_dims, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, decoder_dims, input_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Anomaly Detector Wrapper
# ---------------------------------------------------------------------------


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detector with PyTorch.

    Parameters
    ----------
    input_dim : int
        Number of features.
    encoder_dims : list of int
        Encoder hidden layer sizes.
    latent_dim : int
        Bottleneck dimension.
    dropout_rate : float
        Dropout probability.
    learning_rate : float
        Adam optimizer learning rate.
    weight_decay : float
        L2 regularization.
    n_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early stopping patience.
    contamination : float
        Expected proportion of anomalies.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        input_dim: int = 10,
        encoder_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims or [64, 32]
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.contamination = contamination
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = Autoencoder(
            input_dim=input_dim,
            encoder_dims=self.encoder_dims,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        ).to(DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )
        self.criterion = nn.MSELoss(reduction="mean")

        self.threshold_: float = 0.0
        self.train_losses_: List[float] = []
        self.val_losses_: List[float] = []

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> "AutoencoderAnomalyDetector":
        """
        Train the autoencoder on (mostly normal) data.

        Parameters
        ----------
        X_train : np.ndarray of shape (n, d)
            Training data.
        X_val : np.ndarray of shape (m, d), optional
            Validation data for early stopping and LR scheduling.

        Returns
        -------
        self
        """
        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        train_dataset = TensorDataset(X_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        val_loader = None
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(DEVICE)
            val_dataset = TensorDataset(X_val_t)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for (batch_x,) in train_loader:
                self.optimizer.zero_grad()
                x_hat = self.model(batch_x)
                loss = self.criterion(x_hat, batch_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.train_losses_.append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                self.val_losses_.append(val_loss)
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Set threshold from training data
        scores = self.anomaly_score(X_train)
        self.threshold_ = float(np.percentile(scores, 100 * (1 - self.contamination)))

        return self

    @torch.no_grad()
    def _evaluate_loss(self, loader: DataLoader) -> float:
        """Compute average loss over a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for (batch_x,) in loader:
            x_hat = self.model(batch_x)
            loss = self.criterion(x_hat, batch_x)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through the autoencoder."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)
        X_hat = self.model(X_t)
        return X_hat.cpu().numpy()

    @torch.no_grad()
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction error (MSE) as anomaly score.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n,)
        """
        self.model.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_scores = []
        for (batch_x,) in loader:
            x_hat = self.model(batch_x)
            mse = torch.mean((batch_x - x_hat) ** 2, dim=1)
            all_scores.append(mse.cpu().numpy())

        return np.concatenate(all_scores)

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
        """Return raw anomaly scores (higher = more anomalous)."""
        return self.anomaly_score(X)

    @torch.no_grad()
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get the latent (bottleneck) representation.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n, latent_dim)
        """
        self.model.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)
        z = self.model.encode(X_t)
        return z.cpu().numpy()


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


def train(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    **hyperparams,
) -> AutoencoderAnomalyDetector:
    """
    Train autoencoder anomaly detector.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    X_val : np.ndarray, optional
        Validation features for early stopping.
    **hyperparams
        Forwarded to AutoencoderAnomalyDetector constructor.

    Returns
    -------
    AutoencoderAnomalyDetector
    """
    input_dim = X_train.shape[1]
    defaults = dict(
        input_dim=input_dim,
        encoder_dims=[64, 32],
        latent_dim=8,
        dropout_rate=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        n_epochs=100,
        batch_size=64,
        patience=10,
        contamination=0.05,
        random_state=42,
    )
    defaults.update(hyperparams)

    model = AutoencoderAnomalyDetector(**defaults)
    model.fit(X_train, X_val)
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
    model: AutoencoderAnomalyDetector,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate model on validation set.

    Returns
    -------
    dict
        precision, recall, f1, roc_auc, average_precision, confusion_matrix.
    """
    y_pred = model.predict(X_val)
    scores = model.anomaly_score(X_val)
    return _compute_metrics(y_val, y_pred, scores)


def test(
    model: AutoencoderAnomalyDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate model on held-out test set.

    Returns
    -------
    dict
        precision, recall, f1, roc_auc, average_precision, confusion_matrix.
    """
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
    n_enc_layers = trial.suggest_int("n_enc_layers", 1, 3)
    encoder_dims = []
    prev_dim = X_train.shape[1]
    for i in range(n_enc_layers):
        dim = trial.suggest_categorical(f"enc_dim_{i}", [32, 64, 128, 256])
        encoder_dims.append(dim)
        prev_dim = dim

    latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    n_epochs = trial.suggest_int("n_epochs", 30, 150)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    contamination = trial.suggest_float("contamination", 0.01, 0.15)

    model = train(
        X_train,
        X_val=X_val,
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        batch_size=batch_size,
        contamination=contamination,
        random_state=42,
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

    def trainable(config: Dict[str, Any]) -> None:
        model = train(
            X_train,
            X_val=X_val,
            encoder_dims=config["encoder_dims"],
            latent_dim=config["latent_dim"],
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            contamination=config["contamination"],
            random_state=42,
        )
        metrics = validate(model, X_val, y_val)
        tune.report(
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
        )

    search_space = {
        "encoder_dims": tune.choice([[32, 16], [64, 32], [128, 64], [64, 32, 16]]),
        "latent_dim": tune.choice([4, 8, 16, 32]),
        "dropout_rate": tune.uniform(0.0, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "n_epochs": tune.randint(30, 150),
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
    """Run the complete PyTorch autoencoder anomaly detection pipeline."""
    print("=" * 70)
    print("Autoencoder Anomaly Detection - PyTorch Implementation")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # 1. Generate data
    print("\n[1/5] Generating synthetic anomaly detection data...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=2000, n_features=10, contamination=0.05, random_state=42
    )

    # 2. Train with defaults
    print("\n[2/5] Training autoencoder...")
    model = train(
        X_train,
        X_val=X_val,
        encoder_dims=[64, 32],
        latent_dim=8,
        dropout_rate=0.1,
        learning_rate=1e-3,
        n_epochs=100,
        batch_size=64,
        patience=10,
        contamination=0.05,
    )
    print(f"  Final training loss: {model.train_losses_[-1]:.6f}")
    if model.val_losses_:
        print(f"  Final validation loss: {model.val_losses_[-1]:.6f}")
    print(f"  Threshold: {model.threshold_:.6f}")
    print(f"  Epochs trained: {len(model.train_losses_)}")

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
    n_enc_layers = bp.pop("n_enc_layers")
    encoder_dims = [bp.pop(f"enc_dim_{i}") for i in range(n_enc_layers)]
    best_model = train(X_train, X_val=X_val, encoder_dims=encoder_dims, **bp, random_state=42)

    # 5. Test
    print("\n[5/5] Test results (best model):")
    test_metrics = test(best_model, X_test, y_test)
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Bonus: Latent space analysis
    print("\n[Bonus] Latent space analysis:")
    z_train = best_model.get_latent_representation(X_train)
    z_normal = z_train[y_train == 0]
    z_anomaly = z_train[y_train == 1]
    print(f"  Latent dim: {z_train.shape[1]}")
    print(f"  Normal latent mean norm: {np.mean(np.linalg.norm(z_normal, axis=1)):.4f}")
    if len(z_anomaly) > 0:
        print(f"  Anomaly latent mean norm: {np.mean(np.linalg.norm(z_anomaly, axis=1)):.4f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
