"""
LightGBM-Inspired Boosted Neural Network - PyTorch Implementation
===================================================================

Theory & Mathematics:
    This module implements a gradient boosting ensemble where each weak
    learner is a small multi-layer perceptron (MLP) instead of a decision
    tree. This follows the same additive model framework as LightGBM but
    replaces histogram-based trees with neural networks.

    Gradient Boosting with Neural Network Weak Learners:
        Initialize: F_0(x) = log(p / (1-p)) for binary classification
        For m = 1, ..., M:
            1. Compute pseudo-residuals (negative gradient of log loss):
               r_i = y_i - sigma(F_{m-1}(x_i))
            2. Train a small MLP h_m to predict r_i from x_i:
               h_m = argmin_h (1/N) * sum (r_i - h(x_i))^2
            3. Update ensemble: F_m(x) = F_{m-1}(x) + eta * h_m(x)

    Key Differences from Tree-Based LightGBM:
        - No histogram binning needed (neural nets handle continuous inputs)
        - Each learner can model nonlinear residual patterns
        - Training uses backpropagation instead of greedy splitting
        - Regularization via weight decay and dropout
        - Can leverage GPU acceleration for large datasets

    Connection to Gradient Boosting:
        The framework is identical: additive ensemble of weak learners
        trained on the negative gradient of the loss function. Only the
        weak learner type changes from tree to neural network.

    Loss Function:
        Primary: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
        Each weak learner's loss: MSE between predicted and actual residuals
        Final ensemble: sigmoid(F_M(x)) for probability estimation

Business Use Cases:
    - Click-through rate prediction for ad tech
    - Complex tabular data where trees plateau
    - Transfer learning pipelines requiring differentiable components
    - GPU-accelerated boosting for real-time serving
    - Hybrid systems combining tree and neural network strengths

Advantages:
    - Neural networks capture complex nonlinear residuals
    - GPU acceleration for faster training and inference
    - End-to-end differentiable for integration with other neural modules
    - Flexible architecture per weak learner
    - Can handle both tabular and mixed input types

Disadvantages:
    - More hyperparameters than tree-based boosting (LR, architecture, etc.)
    - Typically slower convergence than LightGBM on tabular data
    - Sequential training limits parallelism
    - Neural networks are less interpretable than trees
    - May overfit if weak learners are too complex

Hyperparameters:
    - n_estimators: Number of boosting rounds (number of MLPs)
    - learning_rate: Shrinkage factor for each MLP's contribution
    - hidden_dim: Hidden layer size for each weak MLP
    - n_layers: Number of hidden layers per MLP
    - lr_nn: Learning rate for the MLP optimizer
    - n_epochs_per_round: Training epochs per boosting round
    - weight_decay: L2 regularization on MLP weights
    - batch_size: Mini-batch size for training
    - dropout: Dropout rate within each MLP
"""

import logging  # Standard logging for training progress tracking
import warnings  # Suppress non-critical warnings
from typing import Any, Dict, List, Optional, Tuple  # Type annotations

import numpy as np  # Numerical computing for data manipulation
import optuna  # Bayesian hyperparameter optimization
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network building blocks
from torch.utils.data import DataLoader, TensorDataset  # Data loading utilities
from sklearn.datasets import make_classification  # Synthetic data generation
from sklearn.metrics import (  # Classification evaluation metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature scaling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Weak Learner: Small MLP for Residual Prediction
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """
    A small MLP designed to predict gradient residuals.
    Kept deliberately small to maintain the weak learner property:
    each MLP corrects a small portion of the overall error.
    """

    def __init__(
        self,
        input_dim: int,  # Number of input features
        hidden_dim: int = 32,  # Hidden layer width
        n_layers: int = 1,  # Number of hidden layers
        dropout: float = 0.1,  # Dropout rate for regularization
    ) -> None:
        """Build a small MLP with configurable depth and width."""
        super().__init__()  # Initialize nn.Module base class

        layers = []  # Accumulate layers in a list
        # First hidden layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))  # Linear transformation
        layers.append(nn.ReLU())  # ReLU activation for nonlinearity
        layers.append(nn.Dropout(dropout))  # Dropout for regularization

        # Additional hidden layers if n_layers > 1
        for _ in range(n_layers - 1):  # Add more hidden layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # Hidden -> hidden
            layers.append(nn.ReLU())  # ReLU activation
            layers.append(nn.Dropout(dropout))  # Dropout

        # Output layer: hidden_dim -> 1 (single residual prediction)
        layers.append(nn.Linear(hidden_dim, 1))  # Output a single scalar

        self.net = nn.Sequential(*layers)  # Wrap all layers in Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict the residual for each input sample."""
        return self.net(x).squeeze(-1)  # (batch,) shaped output


# ---------------------------------------------------------------------------
# Boosted Neural Network Classifier
# ---------------------------------------------------------------------------

class BoostedNNClassifier:
    """
    Gradient boosting classifier using small MLPs as weak learners.
    Each MLP is trained to predict the negative gradient (residual)
    of the current ensemble's predictions.
    """

    def __init__(
        self,
        input_dim: int = 10,  # Number of features (set from data)
        n_estimators: int = 20,  # Number of boosting rounds
        learning_rate: float = 0.1,  # Shrinkage factor per MLP
        hidden_dim: int = 32,  # Hidden layer size per MLP
        n_layers: int = 1,  # Hidden layers per MLP
        lr_nn: float = 0.01,  # MLP optimizer learning rate
        n_epochs_per_round: int = 30,  # Epochs per boosting round
        weight_decay: float = 1e-4,  # L2 regularization
        batch_size: int = 64,  # Mini-batch size
        dropout: float = 0.1,  # Dropout rate
        random_state: int = 42,  # Reproducibility seed
    ) -> None:
        """Store all hyperparameters."""
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr_nn = lr_nn
        self.n_epochs_per_round = n_epochs_per_round
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.learners: List[ResidualMLP] = []  # Trained weak learners
        self.initial_pred: float = 0.0  # Initial log-odds prediction

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-z))  # Standard sigmoid

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoostedNNClassifier":
        """
        Train the boosted ensemble sequentially.
        Each round: compute residuals, train a new MLP on them, update predictions.
        """
        torch.manual_seed(self.random_state)  # Set PyTorch seed
        np.random.seed(self.random_state)  # Set NumPy seed
        self.input_dim = X.shape[1]  # Infer input dim from data
        n_samples = X.shape[0]  # Number of training samples

        # Initialize F_0 with log-odds of the positive class
        p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)  # Mean class probability
        self.initial_pred = np.log(p / (1 - p))  # Convert to log-odds
        raw_predictions = np.full(n_samples, self.initial_pred)  # Initialize all

        # Convert features to a tensor once (reused every round)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)  # Features

        self.learners = []  # Reset learner list

        for m in range(self.n_estimators):  # Train each weak learner
            # Step 1: Compute pseudo-residuals (negative gradient of log loss)
            probs = self._sigmoid(raw_predictions)  # Current predicted probabilities
            residuals = y - probs  # Residuals: y_i - sigma(F(x_i))

            # Convert residuals to tensor for MLP training
            residual_tensor = torch.tensor(
                residuals, dtype=torch.float32, device=DEVICE,
            )

            # Step 2: Train a small MLP to predict the residuals
            mlp = ResidualMLP(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                dropout=self.dropout,
            ).to(DEVICE)  # Move to GPU if available

            optimizer = torch.optim.Adam(
                mlp.parameters(),
                lr=self.lr_nn,
                weight_decay=self.weight_decay,
            )
            loss_fn = nn.MSELoss()  # MSE between predicted and actual residuals

            # Create DataLoader for mini-batch training
            dataset = TensorDataset(X_tensor, residual_tensor)  # Pair features with residuals
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
            )

            # Train the MLP for n_epochs_per_round epochs
            mlp.train()  # Set to training mode
            for epoch in range(self.n_epochs_per_round):
                for batch_X, batch_r in loader:  # Iterate over mini-batches
                    optimizer.zero_grad()  # Clear previous gradients
                    pred_residuals = mlp(batch_X)  # Forward pass
                    loss = loss_fn(pred_residuals, batch_r)  # MSE loss
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Parameter update

            # Step 3: Update ensemble predictions
            mlp.eval()  # Switch to eval mode (disables dropout)
            with torch.no_grad():  # No gradient computation
                update = mlp(X_tensor).cpu().numpy()  # Get MLP predictions

            # Additive update with learning rate shrinkage
            raw_predictions += self.learning_rate * update  # F_m = F_{m-1} + eta*h_m

            self.learners.append(mlp)  # Store the trained MLP

            if (m + 1) % 10 == 0 or m == 0:  # Log every 10 rounds
                current_loss = -np.mean(
                    y * np.log(self._sigmoid(raw_predictions) + 1e-7)
                    + (1 - y) * np.log(1 - self._sigmoid(raw_predictions) + 1e-7)
                )
                logger.info("Round %d/%d, log loss=%.4f", m + 1, self.n_estimators, current_loss)

        logger.info(
            "BoostedNN trained: %d MLPs, hidden_dim=%d, lr=%.3f",
            self.n_estimators, self.hidden_dim, self.learning_rate,
        )
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute raw log-odds by summing initial prediction and all MLPs."""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        raw = np.full(X.shape[0], self.initial_pred)  # Start with F_0
        for mlp in self.learners:  # Sum each learner's contribution
            mlp.eval()
            with torch.no_grad():
                raw += self.learning_rate * mlp(X_tensor).cpu().numpy()
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert raw predictions to class probabilities."""
        raw = self._raw_predict(X)
        p1 = self._sigmoid(raw)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by thresholding probabilities."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data and split into train/val/test."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> BoostedNNClassifier:
    """Train a BoostedNN classifier with given hyperparameters."""
    defaults = dict(
        input_dim=X_train.shape[1],
        n_estimators=20,
        learning_rate=0.1,
        hidden_dim=32,
        n_layers=1,
        lr_nn=0.01,
        n_epochs_per_round=30,
        weight_decay=1e-4,
        batch_size=64,
        dropout=0.1,
        random_state=42,
    )
    defaults.update(hyperparams)
    model = BoostedNNClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def _evaluate(model: BoostedNNClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    auc = roc_auc_score(y, y_proba[:, 1])
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: BoostedNNClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Evaluate on validation data."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: BoostedNNClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate on test data with full reporting."""
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective: suggest params, train, return val F1."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 5, 50, step=5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "lr_nn": trial.suggest_float("lr_nn", 1e-4, 0.05, log=True),
        "n_epochs_per_round": trial.suggest_int("n_epochs_per_round", 10, 50, step=10),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    model = train(X_train, y_train, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search."""
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "n_estimators": ray_tune.choice([10, 20, 30, 50]),
        "learning_rate": ray_tune.loguniform(0.01, 0.5),
        "hidden_dim": ray_tune.choice([16, 32, 64]),
        "lr_nn": ray_tune.loguniform(1e-4, 0.05),
        "n_epochs_per_round": ray_tune.choice([10, 20, 30]),
        "weight_decay": ray_tune.loguniform(1e-6, 1e-2),
        "dropout": ray_tune.uniform(0.0, 0.3),
    }

    tuner = ray_tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=ray_tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="f1", mode="max")
    logger.info("Ray Tune best config: %s", best.config)
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Compare Parameter Sets
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare 4 configurations to study the trade-off between
    ensemble size and learning rate.

    Configurations:
        1. few_fast (T=10, lr=0.1):
           Fast training with aggressive updates. Each MLP contributes
           a large correction. May overfit or be unstable.

        2. many_slow (T=50, lr=0.01):
           Many small corrections. More stable but slower convergence.
           Follows the classic boosting philosophy of many weak adjustments.

        3. moderate_balanced (T=20, lr=0.05):
           Balanced approach. Reasonable number of rounds with moderate
           learning rate. Good default for most datasets.

        4. few_slow_large (T=10, lr=0.01, hidden=64):
           Fewer but larger learners with conservative learning rate.
           Tests whether stronger learners compensate for fewer rounds.
    """
    configs = {
        "few_fast (T=10, lr=0.1)": {
            # Reasoning: Aggressive boosting with few rounds. Each MLP
            # makes a large contribution. Fast but may overshoot.
            "n_estimators": 10,
            "learning_rate": 0.1,
            "hidden_dim": 32,
        },
        "many_slow (T=50, lr=0.01)": {
            # Reasoning: Classic gradient boosting with small steps.
            # Many rounds of small corrections converge more smoothly.
            # Lower risk of overfitting per round.
            "n_estimators": 50,
            "learning_rate": 0.01,
            "hidden_dim": 32,
        },
        "moderate_balanced (T=20, lr=0.05)": {
            # Reasoning: Middle ground. 20 rounds with moderate LR
            # balances speed and accuracy. Practical default choice.
            "n_estimators": 20,
            "learning_rate": 0.05,
            "hidden_dim": 32,
        },
        "few_slow_large (T=10, lr=0.01, h=64)": {
            # Reasoning: Larger hidden dim compensates for fewer rounds.
            # Each learner has more capacity to model complex residuals.
            # Tests if learner capacity can replace ensemble depth.
            "n_estimators": 10,
            "learning_rate": 0.01,
            "hidden_dim": 64,
        },
    }

    results = {}
    logger.info("=" * 70)
    logger.info("Comparing %d BoostedNN configurations", len(configs))
    logger.info("=" * 70)

    for name, params in configs.items():
        logger.info("\n--- Config: %s ---", name)
        model = train(X_train, y_train, **params)
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("%-40s | Accuracy | F1     | AUC-ROC", "Configuration")
    logger.info("-" * 75)
    for name, metrics in results.items():
        logger.info(
            "%-40s | %.4f   | %.4f | %.4f",
            name, metrics["accuracy"], metrics["f1"], metrics["auc_roc"],
        )

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Click-Through Rate Prediction
# ---------------------------------------------------------------------------

def real_world_demo() -> Dict[str, float]:
    """
    Demonstrate BoostedNN on a simulated click-through rate prediction problem.

    Domain: Digital advertising
    Goal: Predict whether a user clicks on a displayed advertisement.

    Features:
        - user_age: User age (18-70)
        - device_type: Mobile=0, tablet=1, desktop=2
        - ad_position: Top=1, middle=2, bottom=3
        - hour_of_day: 0-23
        - day_of_week: 0-6
        - session_duration: Minutes spent on site
        - num_previous_clicks: Historical user engagement
        - page_depth: Pages visited in current session
    """
    rng = np.random.RandomState(42)
    n_samples = 1000

    user_age = rng.normal(35, 12, n_samples).clip(18, 70)
    device_type = rng.choice([0, 1, 2], n_samples, p=[0.6, 0.15, 0.25])
    ad_position = rng.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])
    hour_of_day = rng.randint(0, 24, n_samples)
    day_of_week = rng.randint(0, 7, n_samples)
    session_duration = rng.exponential(5, n_samples)
    num_previous_clicks = rng.poisson(3, n_samples)
    page_depth = rng.poisson(2, n_samples) + 1

    X = np.column_stack([
        user_age, device_type, ad_position, hour_of_day,
        day_of_week, session_duration, num_previous_clicks, page_depth,
    ])

    click_logit = (
        -3.0 - 0.01 * user_age
        + 0.3 * (device_type == 0).astype(float)
        - 0.5 * (ad_position == 3).astype(float)
        + 0.1 * np.sin(hour_of_day * np.pi / 12)
        + 0.1 * num_previous_clicks
        - 0.05 * session_duration
        + rng.normal(0, 0.3, n_samples)
    )
    click_prob = 1.0 / (1.0 + np.exp(-click_logit))
    y = (rng.random(n_samples) < click_prob).astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Click-Through Rate (BoostedNN)")
    logger.info("=" * 70)
    logger.info("Click rate: %.1f%%", 100 * np.mean(y))

    model = train(X_train, y_train, n_estimators=20, hidden_dim=32, learning_rate=0.1)
    validate(model, X_val, y_val)
    metrics = test(model, X_test, y_test)
    return metrics


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full BoostedNN pipeline."""
    logger.info("=" * 70)
    logger.info("LightGBM-Inspired Boosted Neural Network - PyTorch Implementation")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(n_samples=800, n_features=15)

    # Baseline
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, n_estimators=20, hidden_dim=32)
    validate(model, X_val, y_val)

    # Optuna HPO
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=10,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    best_model = train(X_train, y_train, **study.best_params)

    # Test
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Compare
    logger.info("\n--- Parameter Set Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Demo
    logger.info("\n--- Real-World Demo: Click-Through Rate ---")
    real_world_demo()


if __name__ == "__main__":
    main()
