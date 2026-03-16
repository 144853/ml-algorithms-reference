"""
Random Forest Classifier - NumPy From-Scratch Implementation
==============================================================

Theory & Mathematics:
    This module implements a Random Forest classifier entirely from scratch
    using NumPy, including the underlying Decision Tree with support for
    both Gini Impurity and Entropy (Information Gain) splitting criteria.

    Decision Tree (CART Algorithm):
        The tree recursively partitions the feature space by selecting the
        feature and threshold that maximize the reduction in impurity.

        Gini Impurity:
            G(t) = 1 - sum_{k=1}^{K} p_k^2
            where p_k is the proportion of class k at node t.

        Entropy:
            H(t) = -sum_{k=1}^{K} p_k * log2(p_k)

        Information Gain for a split:
            IG = H(parent) - [n_left/n * H(left) + n_right/n * H(right)]

    Bootstrap Aggregating (Bagging):
        For each tree t in {1, ..., T}:
            1. Draw a bootstrap sample D_t of size N from training data.
            2. At each node, consider a random subset of m features.
            3. Grow the tree to max_depth or until stopping criteria.

    Ensemble Prediction:
        y_hat = mode({h_1(x), h_2(x), ..., h_T(x)})

    Variance Reduction:
        Var(avg) = (1/T) * [rho * sigma^2 + (1-rho)/T * sigma^2]
        where rho is the pairwise correlation between trees.

Business Use Cases:
    - Interpretable ML with feature importance for regulatory compliance
    - Anomaly detection via isolation concepts
    - Gene expression analysis in bioinformatics
    - Soil and terrain classification in agriculture

Hyperparameters:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of each decision tree
    - min_samples_split: Minimum samples to attempt a split
    - min_samples_leaf: Minimum samples at a leaf node
    - max_features: Number of random features at each split
    - criterion: 'gini' or 'entropy'
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Decision Tree Node
# ---------------------------------------------------------------------------

class TreeNode:
    """Node in a decision tree.

    WHY a class: Each node stores either a split decision (feature index + threshold)
    or a leaf value (class probability distribution). Using a class with is_leaf()
    makes tree traversal clean and recursive.
    """

    def __init__(
        self,
        feature_idx: Optional[int] = None,      # Feature index to split on.
        threshold: Optional[float] = None,       # Threshold value for the split.
        left: Optional["TreeNode"] = None,       # Left child (feature <= threshold).
        right: Optional["TreeNode"] = None,      # Right child (feature > threshold).
        value: Optional[np.ndarray] = None,      # Class distribution at leaf nodes.
    ) -> None:
        self.feature_idx = feature_idx   # Which feature to split on (None for leaves).
        self.threshold = threshold       # Split threshold (None for leaves).
        self.left = left                 # Left subtree (samples where feature <= threshold).
        self.right = right               # Right subtree (samples where feature > threshold).
        self.value = value               # Normalized class distribution (only set for leaves).
        # WHY value at leaves: Storing the full class distribution enables probability
        # estimation, not just class prediction.

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has a value, no children).

        WHY: Leaf nodes return predictions; internal nodes route samples to children.
        """
        return self.value is not None


# ---------------------------------------------------------------------------
# Decision Tree Classifier (from scratch)
# ---------------------------------------------------------------------------

class DecisionTreeNumpy:
    """CART decision tree for classification, built entirely with NumPy.

    WHY from scratch: Understanding CART tree construction (finding best splits,
    computing impurity, recursive partitioning) is foundational for understanding
    all tree-based ML methods including Random Forest, XGBoost, and LightGBM.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: str = "gini",
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth             # Maximum tree depth to prevent overfitting.
        self.min_samples_split = min_samples_split  # Minimum samples to attempt a split.
        self.min_samples_leaf = min_samples_leaf    # Minimum samples in each leaf.
        self.max_features = max_features       # Random feature subset size (None = all features).
        self.criterion = criterion             # 'gini' or 'entropy' impurity measure.
        self.rng = np.random.RandomState(random_state)  # RNG for feature subsampling.
        self.root: Optional[TreeNode] = None   # Root node of the fitted tree.
        self.n_classes_: int = 0               # Number of classes (set during fitting).

    def _impurity(self, y: np.ndarray) -> float:
        """Compute node impurity using the configured criterion.

        WHY: Impurity measures how "mixed" the classes are at a node.
        A pure node (all same class) has impurity 0. We want splits that
        minimize child impurity (maximize information gain).
        """
        # bincount counts occurrences of each class; divide by total to get proportions.
        # WHY minlength: Ensures the array has n_classes elements even if some classes
        # are absent in this node (important for consistent array shapes).
        proportions = np.bincount(y, minlength=self.n_classes_) / len(y)
        if self.criterion == "gini":
            # Gini impurity: 1 - sum(p_k^2). Measures probability of misclassification.
            # WHY: Fast to compute (no logarithm) and works well in practice.
            return 1.0 - np.sum(proportions ** 2)
        else:  # entropy
            # Entropy: -sum(p_k * log2(p_k)). Measures information content.
            # WHY: Theoretically principled (from information theory) but slightly slower.
            proportions = proportions[proportions > 0]  # Avoid log(0).
            return -np.sum(proportions * np.log2(proportions))

    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute information gain from splitting y into y_left and y_right.

        WHY: Information gain = parent impurity - weighted child impurity.
        We select the split that maximizes this gain (most impurity reduction).
        """
        n = len(y)
        parent_impurity = self._impurity(y)
        # Weighted average of child impurities (weighted by fraction of samples in each child).
        child_impurity = (
            len(y_left) / n * self._impurity(y_left)
            + len(y_right) / n * self._impurity(y_right)
        )
        return parent_impurity - child_impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find the best feature and threshold for splitting this node.

        WHY: This is the core of CART -- exhaustively searching over all features
        and thresholds to find the split that maximizes information gain.
        Random feature subsampling (max_features) makes this a Random Forest split.
        """
        n_samples, n_features = X.shape
        # Don't split if we don't have enough samples.
        if n_samples < self.min_samples_split:
            return None, None

        # Random feature subset selection (the "random" in Random Forest).
        # WHY: Considering only a subset of features at each split decorrelates
        # the trees in the forest, which is the key to Random Forest's success.
        n_feat = self.max_features or n_features
        feature_indices = self.rng.choice(n_features, size=min(n_feat, n_features), replace=False)

        best_gain = -1.0
        best_feature, best_threshold = None, None

        # Try each feature in the random subset.
        for feat_idx in feature_indices:
            # Get unique values of this feature as candidate thresholds.
            thresholds = np.unique(X[:, feat_idx])
            # Subsample thresholds for efficiency when there are many unique values.
            # WHY: Testing all ~1000 thresholds is slow. 50 random ones are usually enough
            # to find a good split. This is a common optimization in tree implementations.
            if len(thresholds) > 50:
                thresholds = self.rng.choice(thresholds, size=50, replace=False)

            # Try each threshold for this feature.
            for thr in thresholds:
                # Split samples into left (<=) and right (>) groups.
                left_mask = X[:, feat_idx] <= thr
                right_mask = ~left_mask

                # Enforce minimum leaf size constraint.
                # WHY: Prevents creating leaves with too few samples (overfitting).
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Compute information gain for this split.
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                # Keep track of the best split found so far.
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thr

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build the decision tree from data.

        WHY recursive: Each call creates one node, then recursively builds its children.
        The recursion stops when a stopping criterion is met (max depth, pure node, etc.).
        """
        # Compute class distribution at this node (used for leaf predictions).
        class_counts = np.bincount(y, minlength=self.n_classes_).astype(float)

        # --- Stopping criteria ---
        # WHY multiple criteria: Each prevents different types of overfitting.
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            # Return a leaf node with the normalized class distribution.
            return TreeNode(value=class_counts / class_counts.sum())

        # Find the best split for this node.
        feat_idx, threshold = self._best_split(X, y)
        # If no valid split was found, make this a leaf.
        if feat_idx is None:
            return TreeNode(value=class_counts / class_counts.sum())

        # Split the data and recursively build child subtrees.
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return an internal node with the split decision.
        return TreeNode(feature_idx=feat_idx, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeNumpy":
        """Fit the decision tree to training data."""
        self.n_classes_ = len(np.unique(y))  # Determine number of classes.
        self.root = self._build_tree(X, y)   # Build the tree recursively.
        return self

    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> np.ndarray:
        """Traverse the tree for a single sample to get its class distribution.

        WHY recursive: Follow the split decisions from root to leaf.
        """
        if node.is_leaf():
            return node.value  # Return the class distribution at this leaf.
        # Route left or right based on the split decision.
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for all samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for all samples (argmax of probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Random Forest Classifier (from scratch)
# ---------------------------------------------------------------------------

class RandomForestNumpy:
    """Random Forest classifier built on DecisionTreeNumpy with bagging.

    WHY from scratch: Understanding how bagging + random feature selection creates
    a powerful ensemble from weak individual trees is fundamental to ML.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: str = "gini",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators       # Number of trees in the forest.
        self.max_depth = max_depth             # Max depth per tree.
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features       # Features per split (None = sqrt(n_features)).
        self.criterion = criterion             # 'gini' or 'entropy'.
        self.random_state = random_state
        self.trees: List[DecisionTreeNumpy] = []   # List of fitted trees.
        self.classes_: Optional[np.ndarray] = None  # Unique class labels.

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestNumpy":
        """Fit the Random Forest by training n_estimators trees on bootstrap samples."""
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        # Default max_features to sqrt(n_features) if not specified.
        # WHY sqrt: The standard choice for classification that balances tree diversity
        # (fewer features = more diverse) with individual tree accuracy (more features = better).
        max_features = self.max_features or int(np.sqrt(X.shape[1]))
        rng = np.random.RandomState(self.random_state)

        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sample: sample N items WITH replacement from the training data.
            # WHY with replacement: Each bootstrap sample contains ~63.2% unique samples.
            # The remaining ~36.8% are "out-of-bag" and can be used for internal validation.
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            # Create and fit a single decision tree on this bootstrap sample.
            # WHY separate random_state per tree: Ensures each tree makes different
            # random feature selections at each split, creating ensemble diversity.
            tree = DecisionTreeNumpy(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                criterion=self.criterion,
                random_state=rng.randint(0, 2**31),
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        logger.debug("Random Forest: %d trees fitted", self.n_estimators)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average class probabilities across all trees.

        WHY average: This is the "aggregating" in "bootstrap aggregating" (bagging).
        Averaging probability estimates reduces variance while maintaining low bias.
        """
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_proba, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (argmax of averaged probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000, n_features: int = 20, n_classes: int = 2, random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data with stratified train/val/test splits."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_features // 2, n_redundant=n_features // 4,
        n_classes=n_classes, random_state=random_state,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    logger.info("Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> RandomForestNumpy:
    """Train a from-scratch Random Forest."""
    defaults = dict(n_estimators=50, max_depth=10, min_samples_split=2,
                    min_samples_leaf=1, max_features=None, criterion="gini", random_state=42)
    defaults.update(hyperparams)
    model = RandomForestNumpy(**defaults)
    model.fit(X_train, y_train)
    logger.info("Random Forest (NumPy) trained: %d trees, max_depth=%s",
                defaults["n_estimators"], defaults["max_depth"])
    return model


def _evaluate(model: RandomForestNumpy, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    n_classes = len(np.unique(y))
    auc = roc_auc_score(y, y_proba[:, 1]) if n_classes == 2 else roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
    return {"accuracy": accuracy_score(y, y_pred), "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0), "f1": f1_score(y, y_pred, average="weighted", zero_division=0), "auc_roc": auc}


def validate(model: RandomForestNumpy, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: RandomForestNumpy, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val) -> None:
    """Compare RF configurations focusing on tree depth vs ensemble size trade-off."""
    configs = {
        "Few Shallow Trees": {"n_estimators": 20, "max_depth": 5, "criterion": "gini",
            # WHY: Quick baseline. Shallow trees underfit individually but train fast.
        },
        "Many Moderate Trees": {"n_estimators": 50, "max_depth": 10, "criterion": "gini",
            # WHY: Good balance. Moderate depth captures patterns without extreme overfitting.
        },
        "Deep Trees + Entropy": {"n_estimators": 30, "max_depth": 20, "criterion": "entropy",
            # WHY: Deep trees capture complex patterns. Entropy may find slightly different splits.
        },
        "Regularized (large leaf)": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 8,
            # WHY: Large min_samples_leaf prevents overfitting on noisy data.
        },
    }
    print("\n" + "=" * 90)
    print("RANDOM FOREST (NumPy) - HYPERPARAMETER COMPARISON")
    print("=" * 90)
    print(f"{'Config':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 90)
    for name, params in configs.items():
        model = train(X_train, y_train, **params)
        metrics = _evaluate(model, X_val, y_val)
        print(f"{name:<30} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f}")
    print("-" * 90)
    print("INTERPRETATION: More trees + moderate depth usually wins. Deep trees risk overfitting.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Real-World Demo: Loan Approval Prediction
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate from-scratch RF on loan approval prediction."""
    print("\n" + "=" * 90)
    print("REAL-WORLD DEMO: Loan Approval Prediction (NumPy From-Scratch)")
    print("=" * 90)
    np.random.seed(42)
    n_samples = 1500
    # Generate realistic loan features.
    annual_income = np.random.lognormal(10.8, 0.5, n_samples).clip(20000, 300000)
    credit_score = np.random.normal(680, 60, n_samples).clip(300, 850).astype(int)
    debt_to_income = np.random.uniform(0.05, 0.6, n_samples)
    employment_years = np.random.exponential(5, n_samples).clip(0, 40)
    loan_amount = annual_income * np.random.uniform(0.1, 0.5, n_samples)
    num_credit_lines = np.random.poisson(5, n_samples)

    approval_score = (0.00003 * annual_income + 0.01 * credit_score - 3.0 * debt_to_income
                      + 0.05 * employment_years - 0.000005 * loan_amount + 0.05 * num_credit_lines
                      - 8.0 + np.random.normal(0, 1.0, n_samples))
    y = (np.random.random(n_samples) < 1 / (1 + np.exp(-approval_score))).astype(int)
    X = np.column_stack([annual_income, credit_score, debt_to_income, employment_years, loan_amount, num_credit_lines])
    feature_names = ["annual_income", "credit_score", "debt_to_income", "employment_years", "loan_amount", "num_credit_lines"]

    print(f"\nDataset: {n_samples} applications, Approval rate: {y.mean():.1%}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train from-scratch RF (fewer trees than sklearn for speed).
    model = RandomForestNumpy(n_estimators=30, max_depth=10, min_samples_leaf=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\n--- Performance ---")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}  F1: {f1_score(y_test, y_pred):.4f}  AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
    print("\n--- Key Insight ---")
    print("  From-scratch RF captures nonlinear feature interactions (income x credit score)")
    print("  just like sklearn's version, demonstrating that the algorithm itself is the source")
    print("  of power, not the implementation language.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val) -> float:
    params = {"n_estimators": trial.suggest_int("n_estimators", 10, 100, step=10),
              "max_depth": trial.suggest_int("max_depth", 3, 20),
              "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
              "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
              "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])}
    return validate(train(X_train, y_train, **params), X_val, y_val)["f1"]


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    import ray
    from ray import tune as ray_tune
    if not ray.is_initialized(): ray.init(ignore_reinit_error=True, log_to_driver=False)
    def _trainable(config):
        metrics = validate(train(X_train, y_train, **config), X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])
    search_space = {"n_estimators": ray_tune.choice([10, 30, 50, 80]), "max_depth": ray_tune.randint(3, 20),
                    "min_samples_split": ray_tune.randint(2, 15), "min_samples_leaf": ray_tune.randint(1, 8),
                    "criterion": ray_tune.choice(["gini", "entropy"])}
    tuner = ray_tune.Tuner(_trainable, param_space=search_space,
                           tune_config=ray_tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"))
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
    logger.info("Random Forest - NumPy From-Scratch Implementation")
    logger.info("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(n_samples=800, n_features=15)

    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, n_estimators=30, max_depth=8)
    validate(model, X_val, y_val)

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val), n_trials=15, show_progress_bar=True)
    logger.info("Optuna best params: %s  F1: %.4f", study.best_params, study.best_value)

    best_model = train(X_train, y_train, **study.best_params)
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
