"""
Gradient Boosting Concept - PyTorch Implementation (Boosted Neural Networks)
=============================================================================

Theory & Mathematics:
    This module implements the gradient boosting concept using PyTorch
    neural networks as weak learners instead of decision trees. Each
    boosting round adds a small neural network that learns to correct
    the residual errors of the current ensemble.

    Boosted Neural Networks:
        Initialize: F_0(x) = 0 (or a bias term)
        For m = 1, ..., M:
            1. Compute pseudo-residuals: r_i = y_i - sigma(F_{m-1}(x_i))
            2. Train a small neural network h_m on (X, r_i)
               using MSE loss: L = (1/N) * sum (r_i - h_m(x_i))^2
            3. Update: F_m(x) = F_{m-1}(x) + eta * h_m(x)

    This is equivalent to functional gradient descent in function space,
    where each neural network approximates the negative gradient of the
    loss with respect to the current ensemble's predictions.

    Key Differences from Tree-Based Boosting:
        - Neural networks are universal function approximators
        - Each weak learner can model nonlinear residuals
        - Training uses backpropagation instead of greedy splitting
        - Regularization via weight decay, dropout, and learning rate

    Loss Functions:
        Binary: BCEWithLogitsLoss (sigmoid + cross-entropy)
        Multiclass: CrossEntropyLoss (softmax + NLL)

    The small networks are kept deliberately weak (few parameters)
    to prevent individual learners from overfitting, similar to how
    decision stumps are used in traditional gradient boosting.

Business Use Cases:
    - Complex pattern recognition where trees underperform
    - Differentiable boosting for end-to-end learning pipelines
    - Research on neural ensemble methods
    - GPU-accelerated boosting for large datasets
    - Transfer learning with boosted fine-tuning stages

Advantages:
    - Neural networks can capture complex nonlinear residuals
    - Fully differentiable: compatible with end-to-end training
    - GPU-accelerated training and inference
    - Flexible architecture for each weak learner
    - Can be combined with representation learning

Disadvantages:
    - More hyperparameters than tree-based boosting
    - Slower convergence than XGBoost on tabular data
    - Risk of overfitting if weak learners are too complex
    - Sequential training limits parallelism
    - Harder to interpret than tree-based boosting

Hyperparameters:
    - n_estimators: Number of boosting rounds (networks)
    - learning_rate: Shrinkage factor for each network's contribution
    - hidden_dim: Hidden layer size in each weak learner
    - n_layers: Number of hidden layers per weak learner
    - lr_nn: Learning rate for the neural network optimizer
    - n_epochs_per_round: Training epochs per boosting round
    - weight_decay: L2 regularization
    - batch_size: Mini-batch size
"""

import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Weak Learner (small neural network)
# ---------------------------------------------------------------------------

class WeakLearner(nn.Module):
    """Small MLP serving as a single weak learner in the boosting ensemble."""

    def __init__(self, n_features: int, out_dim: int = 1, hidden_dim: int = 16, n_layers: int = 1) -> None:
        super().__init__()
        layers: list = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.05))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Boosted Neural Network Classifier
# ---------------------------------------------------------------------------

class BoostedNeuralClassifier:
    """Gradient boosting with neural network weak learners."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        n_estimators: int = 20,
        learning_rate: float = 0.1,
        hidden_dim: int = 16,
        n_layers: int = 1,
        lr_nn: float = 0.01,
        n_epochs_per_round: int = 50,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr_nn = lr_nn
        self.n_epochs_per_round = n_epochs_per_round
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state

        self.models: List[WeakLearner] = []
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _train_weak_learner(
        self, X: np.ndarray, residuals: np.ndarray, out_dim: int = 1,
    ) -> WeakLearner:
        """Train a single weak learner on residuals."""
        model = WeakLearner(self.n_features, out_dim, self.hidden_dim, self.n_layers).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_nn, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        r_t = torch.tensor(residuals, dtype=torch.float32)
        if r_t.ndim == 1:
            r_t = r_t.unsqueeze(1)

        loader = DataLoader(TensorDataset(X_t, r_t), batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.n_epochs_per_round):
            for X_b, r_b in loader:
                X_b, r_b = X_b.to(DEVICE), r_b.to(DEVICE)
                pred = model(X_b)
                loss = criterion(pred, r_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    @torch.no_grad()
    def _predict_weak(self, model: WeakLearner, X: np.ndarray) -> np.ndarray:
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(X_t).cpu().numpy().squeeze()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoostedNeuralClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_samples = X.shape[0]

        if self.n_classes == 2:
            # Binary classification
            p = np.mean(y)
            self.bias = np.log(p / (1 - p + 1e-15))
            raw_predictions = np.full(n_samples, self.bias, dtype=np.float64)

            for m in range(self.n_estimators):
                probs = self._sigmoid(raw_predictions)
                residuals = y - probs  # negative gradient

                weak = self._train_weak_learner(X, residuals.astype(np.float32))
                self.models.append(weak)

                update = self._predict_weak(weak, X)
                raw_predictions += self.learning_rate * update

                if (m + 1) % max(1, self.n_estimators // 5) == 0:
                    acc = accuracy_score(y, (self._sigmoid(raw_predictions) >= 0.5).astype(int))
                    logger.debug("Boosting round %d/%d train_acc=%.4f", m + 1, self.n_estimators, acc)
        else:
            # Multiclass
            y_onehot = np.eye(self.n_classes)[y]
            raw_predictions = np.zeros((n_samples, self.n_classes), dtype=np.float64)

            for m in range(self.n_estimators):
                exp_raw = np.exp(raw_predictions - raw_predictions.max(axis=1, keepdims=True))
                probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
                residuals = y_onehot - probs

                weak = self._train_weak_learner(X, residuals.astype(np.float32), out_dim=self.n_classes)
                self.models.append(weak)

                update = self._predict_weak(weak, X)
                if update.ndim == 1:
                    update = update.reshape(-1, self.n_classes)
                raw_predictions += self.learning_rate * update

        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if self.n_classes == 2:
            raw = np.full(X.shape[0], self.bias, dtype=np.float64)
            for weak in self.models:
                raw += self.learning_rate * self._predict_weak(weak, X)
            return raw
        else:
            raw = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
            for weak in self.models:
                update = self._predict_weak(weak, X)
                if update.ndim == 1:
                    update = update.reshape(-1, self.n_classes)
                raw += self.learning_rate * update
            return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_predict(X)
        if self.n_classes == 2:
            p1 = self._sigmoid(raw)
            return np.column_stack([1 - p1, p1])
        else:
            exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
            return exp_raw / exp_raw.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> BoostedNeuralClassifier:
    defaults = dict(
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        n_estimators=20,
        learning_rate=0.1,
        hidden_dim=16,
        n_layers=1,
        lr_nn=0.01,
        n_epochs_per_round=50,
        weight_decay=1e-4,
        batch_size=64,
        random_state=42,
    )
    defaults.update(hyperparams)
    model = BoostedNeuralClassifier(**defaults)
    model.fit(X_train, y_train)
    logger.info(
        "Boosted NN trained: %d rounds, lr=%.3f, hidden=%d",
        defaults["n_estimators"], defaults["learning_rate"], defaults["hidden_dim"],
    )
    return model


def _evaluate(model: BoostedNeuralClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    n_classes = len(np.unique(y))
    if n_classes == 2:
        auc = roc_auc_score(y, y_proba[:, 1])
    else:
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: BoostedNeuralClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: BoostedNeuralClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
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
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 5, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "lr_nn": trial.suggest_float("lr_nn", 1e-4, 0.05, log=True),
        "n_epochs_per_round": trial.suggest_int("n_epochs_per_round", 20, 100, step=20),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
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
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "n_estimators": ray_tune.choice([10, 15, 20, 25]),
        "learning_rate": ray_tune.loguniform(0.01, 0.5),
        "hidden_dim": ray_tune.choice([8, 16, 32]),
        "n_layers": ray_tune.randint(1, 3),
        "lr_nn": ray_tune.loguniform(1e-4, 0.05),
        "n_epochs_per_round": ray_tune.choice([30, 50, 80]),
        "batch_size": ray_tune.choice([32, 64, 128]),
        "weight_decay": ray_tune.loguniform(1e-6, 1e-2),
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 70)
    logger.info("Gradient Boosting Concept - PyTorch (Boosted Neural Networks)")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Baseline
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # Optuna
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=15,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    best_model = train(X_train, y_train, **study.best_params)

    # Test
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
