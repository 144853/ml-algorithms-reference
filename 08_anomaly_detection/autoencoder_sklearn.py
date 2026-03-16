"""
Autoencoder-Based Anomaly Detection - Scikit-Learn Implementation
=================================================================

Theory & Mathematics:
    An autoencoder is a neural network trained to reconstruct its input through
    a bottleneck (compressed representation). For anomaly detection, the key
    insight is:

        - The autoencoder is trained on *normal* data only
        - It learns to reconstruct normal patterns well (low reconstruction error)
        - Anomalous inputs produce *high* reconstruction error because the
          network has never seen such patterns during training

    Architecture:
        Input (d) -> Encoder -> Bottleneck (k) -> Decoder -> Output (d)
        where k << d forces the network to learn a compressed representation.

    Reconstruction Error as Anomaly Score:
        For a sample x, the anomaly score is:
            score(x) = ||x - AE(x)||^2 = sum_i (x_i - x_hat_i)^2

        where AE(x) = Decoder(Encoder(x)) is the reconstructed output.

    Threshold Selection:
        A threshold tau is chosen such that samples with score > tau are
        classified as anomalies. Common approaches:
        - Percentile-based: tau = percentile(scores_train, 100*(1-contamination))
        - Mean + k*std on training scores
        - ROC curve optimization on validation set

    This Implementation (sklearn MLPRegressor):
        We use sklearn's MLPRegressor as an autoencoder by setting the target
        equal to the input (y = X). The hidden layer sizes define the
        encoder-bottleneck-decoder architecture:

        hidden_layer_sizes = (encoder_dim, bottleneck_dim, decoder_dim)

        For example: (64, 16, 64) creates:
            Input(d) -> 64 -> 16 -> 64 -> Output(d)

        The reconstruction error (MSE per sample) serves as the anomaly score.

Business Use Cases:
    - Credit card fraud detection (normal transactions vs. fraudulent ones)
    - Network intrusion detection (normal traffic vs. attacks)
    - Manufacturing quality control (normal products vs. defects)
    - Medical imaging (normal scans vs. pathological findings)
    - Predictive maintenance (normal sensor readings vs. failures)

Advantages:
    - Simple to implement with sklearn's MLPRegressor
    - No explicit anomaly labels needed for training (semi-supervised)
    - Captures non-linear relationships via neural network
    - Reconstruction error is intuitive and interpretable
    - Easy to deploy and integrate into existing sklearn pipelines

Disadvantages:
    - MLPRegressor is not as flexible as PyTorch (limited architectures)
    - Training can be slow for large datasets
    - Sensitive to hyperparameters (architecture, learning rate)
    - May not scale well to very high-dimensional data
    - Bottleneck size must be tuned carefully
    - sklearn does not support GPU acceleration

Key Hyperparameters:
    - hidden_layer_sizes: tuple defining encoder-bottleneck-decoder widths
    - activation: activation function ('relu', 'tanh', 'logistic')
    - solver: optimization algorithm ('adam', 'sgd', 'lbfgs')
    - alpha: L2 regularization strength
    - learning_rate_init: initial learning rate
    - max_iter: maximum training iterations
    - batch_size: mini-batch size for 'adam'/'sgd'
    - contamination: fraction of anomalies for threshold

References:
    - Sakurada, M. and Yairi, T., 2014. Anomaly detection using autoencoders
      with nonlinear dimensionality reduction. MLSDA workshop.
    - An, J. and Cho, S., 2015. Variational autoencoder based anomaly detection
      using reconstruction probability. Special Lecture on IE, 2(1), 1-18.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


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

    Normal samples: multivariate Gaussian (mean=0, std=1).
    Anomalies: uniform distribution in [-6, 6]^d shifted to low-density regions.

    Split: 60% train / 20% val / 20% test.
    Training set is mostly normal (semi-supervised).

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Scaled feature matrices.
    y_train, y_val, y_test : np.ndarray
        Labels (0=normal, 1=anomaly).
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
# Autoencoder Wrapper
# ---------------------------------------------------------------------------


class AutoencoderSklearn:
    """
    Autoencoder-based anomaly detector using sklearn's MLPRegressor.

    Uses MLPRegressor with target = input (identity mapping through
    a bottleneck) to learn normal data patterns. Reconstruction error
    serves as the anomaly score.

    Parameters
    ----------
    hidden_layer_sizes : tuple
        Architecture of the network. Should be symmetric (e.g., (64, 16, 64))
        to form an encoder-bottleneck-decoder structure.
    activation : str
        Activation function ('relu', 'tanh', 'logistic').
    solver : str
        Optimizer ('adam', 'sgd', 'lbfgs').
    alpha : float
        L2 regularization.
    learning_rate_init : float
        Initial learning rate.
    max_iter : int
        Maximum number of training iterations.
    batch_size : int or 'auto'
        Mini-batch size.
    contamination : float
        Expected fraction of anomalies for threshold.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 16, 64),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        batch_size: int = 64,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.contamination = contamination
        self.random_state = random_state

        self.model_ = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray) -> "AutoencoderSklearn":
        """
        Train the autoencoder: learn to reconstruct X from X.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Training data (mostly normal).

        Returns
        -------
        self
        """
        # Target = Input (autoencoder objective)
        self.model_.fit(X, X)

        # Set threshold from training reconstruction errors
        scores = self.anomaly_score(X)
        self.threshold_ = float(np.percentile(scores, 100 * (1 - self.contamination)))

        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through the autoencoder."""
        return self.model_.predict(X)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error (MSE per sample) as anomaly score.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n,)
            MSE reconstruction error for each sample.
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
        """Return raw anomaly scores (higher = more anomalous)."""
        return self.anomaly_score(X)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: np.ndarray, **hyperparams) -> AutoencoderSklearn:
    """
    Train autoencoder anomaly detector.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    **hyperparams
        Forwarded to AutoencoderSklearn constructor.

    Returns
    -------
    AutoencoderSklearn
    """
    defaults = dict(
        hidden_layer_sizes=(64, 16, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        batch_size=64,
        contamination=0.05,
        random_state=42,
    )
    defaults.update(hyperparams)

    model = AutoencoderSklearn(**defaults)
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
    model: AutoencoderSklearn,
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
    model: AutoencoderSklearn,
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
    n_features = X_train.shape[1]

    # Architecture: symmetric encoder-bottleneck-decoder
    encoder_dim = trial.suggest_categorical("encoder_dim", [32, 64, 128])
    bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [4, 8, 16, 32])
    # Ensure bottleneck < encoder
    if bottleneck_dim >= encoder_dim:
        bottleneck_dim = max(4, encoder_dim // 4)
    decoder_dim = encoder_dim  # Symmetric architecture

    hidden_layer_sizes = (encoder_dim, bottleneck_dim, decoder_dim)

    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 500)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    contamination = trial.suggest_float("contamination", 0.01, 0.15)

    model = train(
        X_train,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
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

    def trainable(config: Dict[str, Any]) -> None:
        encoder_dim = config["encoder_dim"]
        bottleneck_dim = config["bottleneck_dim"]
        if bottleneck_dim >= encoder_dim:
            bottleneck_dim = max(4, encoder_dim // 4)
        hidden_layer_sizes = (encoder_dim, bottleneck_dim, encoder_dim)

        model = train(
            X_train,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=config["activation"],
            alpha=config["alpha"],
            learning_rate_init=config["learning_rate_init"],
            max_iter=config["max_iter"],
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
        "encoder_dim": tune.choice([32, 64, 128]),
        "bottleneck_dim": tune.choice([4, 8, 16, 32]),
        "activation": tune.choice(["relu", "tanh"]),
        "alpha": tune.loguniform(1e-6, 1e-2),
        "learning_rate_init": tune.loguniform(1e-4, 1e-2),
        "max_iter": tune.randint(100, 500),
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
    """Run the complete autoencoder anomaly detection pipeline."""
    print("=" * 70)
    print("Autoencoder Anomaly Detection - Scikit-Learn MLPRegressor")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/5] Generating synthetic anomaly detection data...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=2000, n_features=10, contamination=0.05, random_state=42
    )

    # 2. Train with defaults
    print("\n[2/5] Training autoencoder with default hyperparameters...")
    model = train(X_train, hidden_layer_sizes=(64, 16, 64), contamination=0.05)
    print(f"  Training loss: {model.model_.loss_:.6f}")
    print(f"  Threshold: {model.threshold_:.6f}")

    # 3. Validate
    print("\n[3/5] Validation results:")
    val_metrics = validate(model, X_val, y_val)
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")

    # 4. Optuna
    print("\n[4/5] Running Optuna hyperparameter optimization (20 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, y_val),
        n_trials=20,
        show_progress_bar=False,
    )
    print(f"  Best trial F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Retrain with best params
    bp = study.best_params.copy()
    enc_dim = bp.pop("encoder_dim")
    bn_dim = bp.pop("bottleneck_dim")
    if bn_dim >= enc_dim:
        bn_dim = max(4, enc_dim // 4)
    best_model = train(
        X_train,
        hidden_layer_sizes=(enc_dim, bn_dim, enc_dim),
        **bp,
    )

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
