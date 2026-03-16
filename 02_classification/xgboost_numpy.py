"""
Gradient Boosted Trees - NumPy From-Scratch Implementation
============================================================

Theory & Mathematics:
    This module implements Gradient Boosting for classification from scratch
    using NumPy, following the core XGBoost / GBDT principles with simple
    decision stumps (trees of depth 1) or shallow decision trees as weak
    learners.

    Gradient Boosting Framework:
        Initialize: F_0(x) = log(p / (1-p)) for binary, where p = mean(y)
        For m = 1, ..., M:
            1. Compute pseudo-residuals (negative gradient of the loss):
               r_i = y_i - sigma(F_{m-1}(x_i))   (for log loss)
            2. Fit a decision tree h_m to the pseudo-residuals.
            3. Update: F_m(x) = F_{m-1}(x) + eta * h_m(x)

    Loss Function (Binary Cross-Entropy / Log Loss):
        L(y, F) = -[y * log(sigma(F)) + (1-y) * log(1 - sigma(F))]
        Gradient: g_i = sigma(F(x_i)) - y_i
        Hessian: h_i = sigma(F(x_i)) * (1 - sigma(F(x_i)))

    Decision Stump (Weak Learner):
        A tree of depth 1 that finds the best single split:
            - Feature j* and threshold t* that minimize the weighted sum
              of squared residuals in left/right children.
            - Leaf values: w_left, w_right

    Optimal Leaf Weight (Newton step):
        w_j = -sum(g_i in leaf j) / (sum(h_i in leaf j) + lambda)

    Regularization:
        - Learning rate (eta): shrinks each tree's contribution
        - L2 regularization (lambda): penalizes large leaf weights
        - Max depth: controls tree complexity
        - Subsampling: stochastic gradient boosting

    For multiclass (K classes):
        Train K separate boosting sequences (one per class), using softmax
        to convert raw scores to probabilities.

Business Use Cases:
    - Understanding gradient boosting internals for research
    - Custom loss function experimentation
    - Educational tool for ML courses
    - Lightweight boosting for embedded systems
    - Baseline comparison for more complex boosting libraries

Advantages:
    - Complete transparency into the boosting algorithm
    - No external ML library dependency
    - Easy to modify loss functions, split criteria, regularization
    - Demonstrates Newton's method for optimization
    - Small footprint, pure NumPy

Disadvantages:
    - Significantly slower than XGBoost/LightGBM C++ implementations
    - No histogram-based splitting or sparsity handling
    - No GPU acceleration
    - Decision stumps are very weak learners
    - Limited to binary/multiclass classification (no ranking, etc.)

Hyperparameters:
    - n_estimators: Number of boosting rounds
    - learning_rate: Shrinkage factor (eta)
    - max_depth: Maximum depth of each tree (1 = stump)
    - lambda_reg: L2 regularization on leaf weights
    - subsample: Row subsampling ratio per round
    - min_samples_leaf: Minimum samples in a leaf
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
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


# ---------------------------------------------------------------------------
# Shallow Decision Tree for Gradient Boosting
# ---------------------------------------------------------------------------

class _GBTreeNode:
    """Internal node for gradient boosting tree."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_GBTreeNode"] = None,
        right: Optional["_GBTreeNode"] = None,
        leaf_value: Optional[float] = None,
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_value = leaf_value

    def is_leaf(self) -> bool:
        return self.leaf_value is not None


class GradientBoostingTree:
    """Shallow decision tree fitted to gradients/hessians."""

    def __init__(
        self,
        max_depth: int = 3,
        lambda_reg: float = 1.0,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.RandomState(random_state)
        self.root: Optional[_GBTreeNode] = None

    def _compute_leaf_value(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        return -np.sum(gradients) / (np.sum(hessians) + self.lambda_reg)

    def _compute_gain(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        G = np.sum(gradients)
        H = np.sum(hessians)
        return (G ** 2) / (H + self.lambda_reg)

    def _find_best_split(
        self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray,
    ) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature, best_threshold = None, None

        parent_gain = self._compute_gain(gradients, hessians)

        for feat_idx in range(n_features):
            sorted_idx = np.argsort(X[:, feat_idx])
            sorted_feat = X[sorted_idx, feat_idx]
            sorted_grad = gradients[sorted_idx]
            sorted_hess = hessians[sorted_idx]

            G_left, H_left = 0.0, 0.0
            G_total = np.sum(sorted_grad)
            H_total = np.sum(sorted_hess)

            for i in range(n_samples - 1):
                G_left += sorted_grad[i]
                H_left += sorted_hess[i]
                G_right = G_total - G_left
                H_right = H_total - H_left

                if (i + 1) < self.min_samples_leaf or (n_samples - i - 1) < self.min_samples_leaf:
                    continue

                if sorted_feat[i] == sorted_feat[i + 1]:
                    continue

                gain_left = (G_left ** 2) / (H_left + self.lambda_reg)
                gain_right = (G_right ** 2) / (H_right + self.lambda_reg)
                gain = gain_left + gain_right - parent_gain

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = (sorted_feat[i] + sorted_feat[i + 1]) / 2.0

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int = 0,
    ) -> _GBTreeNode:
        # Stopping criteria
        if depth >= self.max_depth or len(gradients) < 2 * self.min_samples_leaf:
            return _GBTreeNode(leaf_value=self._compute_leaf_value(gradients, hessians))

        feat_idx, threshold, gain = self._find_best_split(X, gradients, hessians)

        if feat_idx is None or gain <= 0:
            return _GBTreeNode(leaf_value=self._compute_leaf_value(gradients, hessians))

        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1)

        return _GBTreeNode(
            feature_idx=feat_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> "GradientBoostingTree":
        self.root = self._build_tree(X, gradients, hessians)
        return self

    def _predict_single(self, x: np.ndarray, node: _GBTreeNode) -> float:
        if node.is_leaf():
            return node.leaf_value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x, self.root) for x in X])


# ---------------------------------------------------------------------------
# Gradient Boosted Trees Classifier
# ---------------------------------------------------------------------------

class GradientBoostedClassifier:
    """Gradient Boosted Trees classifier implemented from scratch."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        lambda_reg: float = 1.0,
        subsample: float = 1.0,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees: List[List[GradientBoostingTree]] = []
        self.initial_predictions: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - z.max(axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedClassifier":
        rng = np.random.RandomState(self.random_state)
        self.n_classes_ = len(np.unique(y))
        n_samples = X.shape[0]

        if self.n_classes_ == 2:
            # Binary classification
            p = np.mean(y)
            self.initial_predictions = np.full(n_samples, np.log(p / (1 - p + 1e-15)))
            raw_predictions = self.initial_predictions.copy()
            self.trees = []

            for m in range(self.n_estimators):
                # Subsample
                if self.subsample < 1.0:
                    idx = rng.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
                else:
                    idx = np.arange(n_samples)

                probs = self._sigmoid(raw_predictions[idx])
                gradients = probs - y[idx]
                hessians = probs * (1 - probs)

                tree = GradientBoostingTree(
                    max_depth=self.max_depth,
                    lambda_reg=self.lambda_reg,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=rng.randint(0, 2**31),
                )
                tree.fit(X[idx], gradients, hessians)
                self.trees.append([tree])

                update = tree.predict(X)
                raw_predictions += self.learning_rate * update
        else:
            # Multiclass (one set of trees per class)
            y_onehot = np.eye(self.n_classes_)[y]
            self.initial_predictions = np.zeros((n_samples, self.n_classes_))
            raw_predictions = self.initial_predictions.copy()
            self.trees = []

            for m in range(self.n_estimators):
                if self.subsample < 1.0:
                    idx = rng.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
                else:
                    idx = np.arange(n_samples)

                probs = self._softmax(raw_predictions[idx])
                round_trees = []

                for k in range(self.n_classes_):
                    gradients = probs[:, k] - y_onehot[idx, k]
                    hessians = probs[:, k] * (1 - probs[:, k])
                    hessians = np.maximum(hessians, 1e-8)

                    tree = GradientBoostingTree(
                        max_depth=self.max_depth,
                        lambda_reg=self.lambda_reg,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=rng.randint(0, 2**31),
                    )
                    tree.fit(X[idx], gradients, hessians)
                    round_trees.append(tree)

                    update = tree.predict(X)
                    raw_predictions[:, k] += self.learning_rate * update

                self.trees.append(round_trees)

        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        if self.n_classes_ == 2:
            p = np.mean(self.initial_predictions)
            raw = np.full(n_samples, p)
            for round_trees in self.trees:
                raw += self.learning_rate * round_trees[0].predict(X)
            return raw
        else:
            raw = np.zeros((n_samples, self.n_classes_))
            for round_trees in self.trees:
                for k, tree in enumerate(round_trees):
                    raw[:, k] += self.learning_rate * tree.predict(X)
            return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_predict(X)
        if self.n_classes_ == 2:
            p1 = self._sigmoid(raw)
            return np.column_stack([1 - p1, p1])
        else:
            return self._softmax(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Data generation
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

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> GradientBoostedClassifier:
    defaults = dict(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        lambda_reg=1.0,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    defaults.update(hyperparams)
    model = GradientBoostedClassifier(**defaults)
    model.fit(X_train, y_train)
    logger.info(
        "GBT (NumPy) trained: %d rounds, depth=%d, lr=%.3f",
        defaults["n_estimators"], defaults["max_depth"], defaults["learning_rate"],
    )
    return model


def _evaluate(model: GradientBoostedClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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


def validate(model: GradientBoostedClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: GradientBoostedClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
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
        "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "lambda_reg": trial.suggest_float("lambda_reg", 0.01, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
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
        "n_estimators": ray_tune.choice([50, 100, 150]),
        "learning_rate": ray_tune.loguniform(0.01, 0.5),
        "max_depth": ray_tune.randint(1, 6),
        "lambda_reg": ray_tune.loguniform(0.01, 10.0),
        "subsample": ray_tune.uniform(0.5, 1.0),
        "min_samples_leaf": ray_tune.randint(1, 20),
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
    logger.info("Gradient Boosted Trees - NumPy From-Scratch Implementation")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=800, n_features=15,
    )

    # Baseline
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, n_estimators=50, max_depth=3)
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
