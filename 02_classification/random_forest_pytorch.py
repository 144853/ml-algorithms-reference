"""
Random Forest-Style Ensemble - PyTorch Implementation
=======================================================

Theory & Mathematics:
    Traditional Random Forests use non-differentiable decision trees, making
    them incompatible with gradient-based optimization. This implementation
    uses an ensemble of small neural networks (sub-networks) to approximate
    the Random Forest paradigm in PyTorch:

    Approach: Ensemble of Diverse Sub-Networks with Feature Bagging
        1. Create T small neural networks (each analogous to a "tree").
        2. Each sub-network receives only a random subset of features
           (feature bagging, similar to random feature selection in RF).
        3. Each sub-network is trained on a bootstrap sample of the data.
        4. Final prediction = average of all sub-network predictions.

    Sub-Network Architecture:
        Each "tree" is a small MLP:
            h1 = ReLU(X_subset @ W1 + b1)
            h2 = ReLU(h1 @ W2 + b2)
            out = h2 @ W3 + b3

    Ensemble Prediction:
        P(y|x) = (1/T) * sum_{t=1}^{T} softmax(f_t(x_subset_t))

Business Use Cases:
    - Neural ensemble models for tabular data
    - GPU-accelerated ensemble inference
    - Differentiable ensemble for end-to-end pipelines
    - Hybrid deep learning + ensemble methods

Hyperparameters:
    - n_estimators: Number of sub-networks in the ensemble
    - hidden_dim: Hidden layer dimension in each sub-network
    - n_layers: Number of hidden layers per sub-network
    - max_features_ratio: Fraction of features each sub-network sees
    - learning_rate, n_epochs, batch_size, weight_decay
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
# Select GPU if available, else CPU for all tensor operations.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Sub-network ("tree") definition
# ---------------------------------------------------------------------------

class SubNetwork(nn.Module):
    """Small MLP acting as a single 'tree' in the ensemble.

    WHY MLP instead of decision tree: Decision trees are not differentiable,
    so they can't be trained with backpropagation. Small MLPs serve as learnable,
    differentiable approximations that can capture nonlinear patterns.
    WHY small: Keeping sub-networks small (few layers, narrow hidden dim) makes
    them "weak learners" -- individually not very accurate but collectively powerful.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        layers: list = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))    # Linear transformation.
            layers.append(nn.BatchNorm1d(hidden_dim))        # Batch normalization for stable training.
            # WHY BatchNorm: Normalizes activations to prevent internal covariate shift,
            # enabling faster convergence and allowing higher learning rates.
            layers.append(nn.ReLU())                          # ReLU activation for nonlinearity.
            layers.append(nn.Dropout(0.1))                    # Light dropout for regularization.
            # WHY 0.1: Mild dropout. Higher values would weaken already-small sub-networks too much.
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_classes))           # Output layer: one score per class.
        self.net = nn.Sequential(*layers)                     # Combine all layers into a sequential model.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sub-network to produce raw class logits."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Random Forest Ensemble
# ---------------------------------------------------------------------------

class RandomForestEnsemble:
    """Ensemble of SubNetworks with feature bagging and bootstrap sampling.

    WHY this design: Mirrors the three key ideas of Random Forests:
    1. Bootstrap sampling: each sub-network trains on a different random subset of samples.
    2. Feature bagging: each sub-network sees only a random subset of features.
    3. Ensemble averaging: final prediction averages all sub-network outputs.
    """

    def __init__(self, n_features: int, n_classes: int, n_estimators: int = 10,
                 hidden_dim: int = 32, n_layers: int = 2, max_features_ratio: float = 0.7,
                 learning_rate: float = 0.01, n_epochs: int = 100, batch_size: int = 64,
                 weight_decay: float = 1e-4, random_state: int = 42) -> None:
        self.n_features = n_features          # Total number of input features.
        self.n_classes = n_classes            # Number of target classes.
        self.n_estimators = n_estimators      # Number of sub-networks (analogous to n_trees).
        self.hidden_dim = hidden_dim          # Width of hidden layers in each sub-network.
        self.n_layers = n_layers              # Depth of each sub-network.
        self.max_features_ratio = max_features_ratio  # Fraction of features each sub-network sees.
        # WHY 0.7: Each sub-network sees 70% of features, leaving 30% out.
        # This creates diversity while keeping enough features for good individual accuracy.
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.rng = np.random.RandomState(random_state)
        self.models: List[SubNetwork] = []          # Trained sub-networks.
        self.feature_subsets: List[np.ndarray] = []  # Feature indices for each sub-network.

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestEnsemble":
        """Train the ensemble by fitting each sub-network on its own bootstrap + feature subset."""
        n_samples = X.shape[0]
        n_sub_features = max(1, int(self.n_features * self.max_features_ratio))
        self.models = []
        self.feature_subsets = []

        for i in range(self.n_estimators):
            # Feature bagging: randomly select a subset of features for this sub-network.
            feat_idx = self.rng.choice(self.n_features, size=n_sub_features, replace=False)
            feat_idx.sort()  # Sort for consistent indexing.
            self.feature_subsets.append(feat_idx)

            # Bootstrap sample: sample with replacement from training data.
            boot_idx = self.rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_idx][:, feat_idx]  # Apply both bootstrap AND feature subset.
            y_boot = y[boot_idx]

            # Create and train a sub-network on this bootstrap sample with feature subset.
            model = SubNetwork(n_sub_features, self.n_classes, self.hidden_dim, self.n_layers).to(DEVICE)
            self._train_subnetwork(model, X_boot, y_boot)
            self.models.append(model)

        logger.debug("Ensemble: %d sub-networks trained", self.n_estimators)
        return self

    def _train_subnetwork(self, model: SubNetwork, X: np.ndarray, y: np.ndarray) -> None:
        """Train a single sub-network using CrossEntropyLoss and Adam optimizer."""
        criterion = nn.CrossEntropyLoss()  # Softmax + negative log-likelihood.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.n_epochs):
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(X_b)           # Forward pass: compute class logits.
                loss = criterion(logits, y_b)  # Compute cross-entropy loss.
                optimizer.zero_grad()          # Clear previous gradients.
                loss.backward()                # Compute gradients via backpropagation.
                optimizer.step()               # Update parameters.

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average softmax probabilities across all sub-networks.

        WHY averaging: Equivalent to Random Forest's majority vote but with
        soft probabilities, giving smoother and more calibrated predictions.
        """
        all_proba = []
        for model, feat_idx in zip(self.models, self.feature_subsets):
            model.eval()  # Disable dropout and use running BatchNorm stats.
            X_sub = torch.tensor(X[:, feat_idx], dtype=torch.float32).to(DEVICE)
            logits = model(X_sub).cpu().numpy()
            # Apply softmax to convert logits to probabilities.
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))  # Numerical stability.
            proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            all_proba.append(proba)
        return np.mean(all_proba, axis=0)  # Average probabilities across all sub-networks.

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Data / Train / Validate / Test
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features//2,
                               n_redundant=n_features//4, n_classes=n_classes, random_state=random_state)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    logger.info("Data generated: train=%d, val=%d, test=%d", X_train.shape[0], X_val.shape[0], X_test.shape[0])
    return X_train, X_val, X_test, y_train, y_val, y_test

def train(X_train, y_train, **hp):
    defaults = dict(n_features=X_train.shape[1], n_classes=len(np.unique(y_train)), n_estimators=10,
                    hidden_dim=32, n_layers=2, max_features_ratio=0.7, learning_rate=0.01,
                    n_epochs=100, batch_size=64, weight_decay=1e-4, random_state=42)
    defaults.update(hp)
    model = RandomForestEnsemble(**defaults)
    model.fit(X_train, y_train)
    logger.info("Ensemble trained: %d sub-networks, hidden=%d", defaults["n_estimators"], defaults["hidden_dim"])
    return model

def _evaluate(model, X, y):
    y_pred, y_proba = model.predict(X), model.predict_proba(X)
    n_classes = len(np.unique(y))
    auc = roc_auc_score(y, y_proba[:, 1]) if n_classes == 2 else roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
    return {"accuracy": accuracy_score(y, y_pred), "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0), "f1": f1_score(y, y_pred, average="weighted", zero_division=0), "auc_roc": auc}

def validate(model, X_val, y_val):
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics

def test(model, X_test, y_test):
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare ensemble configurations: size, diversity, and sub-network complexity."""
    configs = {
        "Small Ensemble (5 nets)": {"n_estimators": 5, "hidden_dim": 32, "max_features_ratio": 0.7,
            # WHY: Few sub-networks = less diversity, faster training. Tests minimum viable ensemble.
        },
        "Medium Ensemble (10 nets)": {"n_estimators": 10, "hidden_dim": 32, "max_features_ratio": 0.7,
            # WHY: Default configuration. Good balance of diversity and training time.
        },
        "Wide Sub-networks": {"n_estimators": 10, "hidden_dim": 64, "max_features_ratio": 0.5,
            # WHY: Wider hidden layers = more expressive sub-networks. Lower feature ratio
            # increases diversity but each sub-network sees fewer features.
        },
        "Deep Sub-networks": {"n_estimators": 10, "hidden_dim": 32, "n_layers": 3, "max_features_ratio": 0.7,
            # WHY: More layers = more capacity. Risk: deeper sub-networks may overfit
            # on their bootstrap samples, reducing the benefit of ensembling.
        },
    }
    print("\n" + "=" * 90)
    print("RANDOM FOREST ENSEMBLE (PyTorch) - HYPERPARAMETER COMPARISON")
    print("=" * 90)
    print(f"{'Config':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 90)
    for name, params in configs.items():
        model = train(X_train, y_train, **params)
        metrics = _evaluate(model, X_val, y_val)
        print(f"{name:<30} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f}")
    print("-" * 90)
    print("INTERPRETATION: More sub-networks usually helps. Wider/deeper nets add capacity but risk overfitting.")
    print("=" * 90 + "\n")


def real_world_demo():
    """Demonstrate PyTorch ensemble on loan approval prediction."""
    print("\n" + "=" * 90)
    print("REAL-WORLD DEMO: Loan Approval Prediction (PyTorch Ensemble)")
    print("=" * 90)
    np.random.seed(42); torch.manual_seed(42)
    n = 1500
    annual_income = np.random.lognormal(10.8, 0.5, n).clip(20000, 300000)
    credit_score = np.random.normal(680, 60, n).clip(300, 850).astype(int)
    dti = np.random.uniform(0.05, 0.6, n)
    emp_years = np.random.exponential(5, n).clip(0, 40)
    loan_amt = annual_income * np.random.uniform(0.1, 0.5, n)
    n_credit = np.random.poisson(5, n)
    score = 0.00003*annual_income + 0.01*credit_score - 3.0*dti + 0.05*emp_years - 0.000005*loan_amt - 8.0 + np.random.normal(0, 1, n)
    y = (np.random.random(n) < 1/(1+np.exp(-score))).astype(int)
    X = np.column_stack([annual_income, credit_score, dti, emp_years, loan_amt, n_credit])
    print(f"\nDataset: {n} applications, Approval rate: {y.mean():.1%}, Device: {DEVICE}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler(); X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
    model = RandomForestEnsemble(n_features=6, n_classes=2, n_estimators=8, hidden_dim=32, n_epochs=80, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te); y_proba = model.predict_proba(X_te)
    print(f"\n  Accuracy: {accuracy_score(y_te, y_pred):.4f}  F1: {f1_score(y_te, y_pred):.4f}  AUC: {roc_auc_score(y_te, y_proba[:, 1]):.4f}")
    print("\n  Key advantage: GPU-accelerated ensemble inference for real-time serving.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Hyperparameter Optimization & Main
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    params = {"n_estimators": trial.suggest_int("n_estimators", 5, 20), "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
              "n_layers": trial.suggest_int("n_layers", 1, 3), "max_features_ratio": trial.suggest_float("max_features_ratio", 0.3, 1.0),
              "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True), "n_epochs": trial.suggest_int("n_epochs", 50, 200, step=50),
              "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]), "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}
    return validate(train(X_train, y_train, **params), X_val, y_val)["f1"]

def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    import ray; from ray import tune as ray_tune
    if not ray.is_initialized(): ray.init(ignore_reinit_error=True, log_to_driver=False)
    def _t(config): metrics = validate(train(X_train, y_train, **config), X_val, y_val); ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])
    space = {"n_estimators": ray_tune.choice([5,10,15,20]), "hidden_dim": ray_tune.choice([16,32,64]), "n_layers": ray_tune.randint(1,4),
             "max_features_ratio": ray_tune.uniform(0.3,1.0), "learning_rate": ray_tune.loguniform(1e-4,0.1), "n_epochs": ray_tune.choice([50,100,150]),
             "batch_size": ray_tune.choice([32,64,128]), "weight_decay": ray_tune.loguniform(1e-6,1e-2)}
    tuner = ray_tune.Tuner(_t, param_space=space, tune_config=ray_tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"))
    best = tuner.fit().get_best_result(metric="f1", mode="max"); logger.info("Best: %s", best.config); ray.shutdown(); return best.config

def main():
    logger.info("=" * 70); logger.info("Random Forest-Style Ensemble - PyTorch Implementation"); logger.info("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()
    logger.info("\n--- Baseline ---"); model = train(X_train, y_train); validate(model, X_val, y_val)
    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()
    logger.info("\n--- Optuna ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: optuna_objective(t, X_train, y_train, X_val, y_val), n_trials=15, show_progress_bar=True)
    logger.info("Best: %s  F1: %.4f", study.best_params, study.best_value)
    best_model = train(X_train, y_train, **study.best_params)
    logger.info("\n--- Final Test ---"); test(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
