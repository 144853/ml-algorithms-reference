"""
Ordered Boosting with Oblivious Trees - NumPy From-Scratch Implementation
==========================================================================

Theory & Mathematics:
    This module implements CatBoost's core ideas from scratch using NumPy:
    ordered target encoding for categorical features and gradient boosting
    with oblivious (symmetric) decision trees.

    1. Ordered Target Encoding (Leave-One-Out with Ordering):
        The central problem with target encoding is target leakage: if we
        encode a category using the mean of the target for that category,
        we leak information about y into X, causing overfitting.

        CatBoost solves this with ordered target statistics:
        a. Randomly permute the training data: sigma = random_permutation(1..N)
        b. For each sample at position i in the permutation, compute the
           target statistic using ONLY samples that appear before position i:

           TS(x_i, c) = (sum_{j < i: x_j = c} y_j + prior * alpha) / (count_{j < i: x_j = c} + alpha)

           where:
           - c is the categorical value of feature for sample i
           - prior is the global mean of y (or a smoothing constant)
           - alpha is the smoothing parameter (prevents unstable estimates
             when few samples precede position i)

        This is analogous to leave-one-out encoding but with a strict
        ordering constraint that prevents any form of target leakage.

    2. Oblivious Decision Trees (Symmetric Trees):
        Unlike regular decision trees where each node can choose a different
        feature and threshold, oblivious trees use the SAME (feature, threshold)
        pair at every node of a given depth level:

        Level 0: All samples split on (feature_0, threshold_0)
        Level 1: All 2 nodes split on (feature_1, threshold_1)
        Level 2: All 4 nodes split on (feature_2, threshold_2)
        ...
        Level d: 2^d leaf nodes, each identified by the binary path

        Properties:
        - 2^depth leaf nodes (exactly)
        - Each leaf is identified by a binary vector of length=depth
        - The split condition at level l is: x[feature_l] <= threshold_l
        - Inference is O(depth): evaluate depth conditions, look up leaf
        - Acts as a regularizer (fewer unique splits than regular trees)

        Leaf assignment: for a sample x, its leaf index (0..2^depth-1) is:
            leaf = sum_{l=0}^{depth-1} (x[feat_l] > thresh_l) * 2^l

    3. Gradient Boosting Framework:
        Initialize: F_0(x) = log(p / (1-p)) where p = mean(y)
        For m = 1, ..., M:
            g_i = sigma(F_{m-1}(x_i)) - y_i  (gradient)
            h_i = sigma(F(x_i)) * (1 - sigma(F(x_i)))  (hessian)
            Build oblivious tree on (X, g, h)
            F_m(x) = F_{m-1}(x) + eta * tree_m(x)

        Leaf weight: w_leaf = -G_leaf / (H_leaf + lambda)

Business Use Cases:
    - Hotel booking cancellation prediction (many categorical features)
    - Customer segmentation with mixed continuous/categorical data
    - Insurance risk classification with ordinal and nominal categories
    - Recommendation systems with user/item categorical features
    - Medical diagnosis with categorical symptom codes

Advantages:
    - Ordered encoding prevents target leakage (key innovation)
    - Oblivious trees are fast at inference (binary path lookup)
    - Oblivious structure acts as built-in regularization
    - Handles categorical features without one-hot encoding
    - Transparent implementation for educational purposes

Disadvantages:
    - Oblivious trees are less flexible than asymmetric trees
    - Ordered encoding adds computational overhead during training
    - From-scratch implementation is much slower than C++ CatBoost
    - Only binary classification implemented here
    - No GPU acceleration in this NumPy version

Hyperparameters:
    - n_estimators: Number of boosting rounds
    - learning_rate: Shrinkage factor per tree
    - depth: Depth of each oblivious tree
    - lambda_reg: L2 regularization on leaf weights
    - cat_features: Indices of categorical feature columns
    - prior_smoothing: Smoothing alpha for ordered target encoding
    - random_state: Seed for reproducibility
"""

import logging  # Standard logging for progress tracking
import warnings  # Suppress non-critical warnings
from typing import Any, Dict, List, Optional, Tuple  # Type hints

import numpy as np  # Core numerical computing library
import optuna  # Bayesian hyperparameter optimization
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
from sklearn.preprocessing import StandardScaler  # Feature normalization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Ordered Target Encoding
# ---------------------------------------------------------------------------

class OrderedTargetEncoder:
    """
    Implements CatBoost's ordered target encoding for categorical features.
    For each sample, the encoding uses only information from samples that
    appear earlier in a random permutation, preventing target leakage.
    """

    def __init__(
        self,
        cat_features: List[int],  # Indices of categorical columns
        prior_smoothing: float = 1.0,  # Smoothing parameter alpha
        random_state: int = 42,  # Seed for the random permutation
    ) -> None:
        """Store encoder configuration."""
        self.cat_features = cat_features  # Which columns are categorical
        self.prior_smoothing = prior_smoothing  # Smoothing strength (alpha)
        self.random_state = random_state  # Random seed
        self.global_mean: float = 0.0  # Global target mean (prior)
        self.category_stats: Dict[int, Dict[str, Tuple[float, int]]] = {}  # Per-category stats

    def fit_transform(
        self,
        X: np.ndarray,  # Feature matrix (n_samples, n_features)
        y: np.ndarray,  # Target labels
    ) -> np.ndarray:
        """
        Fit the encoder and transform training data using ordered encoding.
        For each sample at position i in a random permutation, compute
        the target statistic using only samples that precede it.
        """
        rng = np.random.RandomState(self.random_state)  # Create RNG
        n_samples = X.shape[0]  # Number of training samples
        self.global_mean = np.mean(y)  # Compute global target mean (prior)

        # Random permutation determines the ordering for leave-one-out encoding
        perm = rng.permutation(n_samples)  # Random ordering of sample indices
        X_encoded = X.copy().astype(np.float64)  # Copy to avoid modifying original

        # Track running statistics per category for each categorical feature
        self.category_stats = {}  # Reset stats

        for feat_idx in self.cat_features:  # Process each categorical feature
            # Dictionaries to accumulate running sum and count per category
            running_sum = {}  # Sum of y values seen so far for each category
            running_count = {}  # Count of samples seen so far for each category
            self.category_stats[feat_idx] = {}  # Will store final stats

            for pos in range(n_samples):  # Process samples in permuted order
                sample_idx = perm[pos]  # Original index of this sample
                cat_value = str(X[sample_idx, feat_idx])  # Category value (as string)

                # Get running stats for this category (samples BEFORE this one)
                sum_before = running_sum.get(cat_value, 0.0)  # Sum of y for this category
                count_before = running_count.get(cat_value, 0)  # Count for this category

                # Ordered target statistic:
                # TS = (sum_before + prior * alpha) / (count_before + alpha)
                # This only uses samples that appeared BEFORE position pos
                ts = (sum_before + self.global_mean * self.prior_smoothing) / (
                    count_before + self.prior_smoothing
                )

                X_encoded[sample_idx, feat_idx] = ts  # Replace category with encoding

                # Update running statistics for subsequent samples
                running_sum[cat_value] = sum_before + y[sample_idx]  # Add this y
                running_count[cat_value] = count_before + 1  # Increment count

            # Store final accumulated stats for transform() on new data
            for cat_val in running_sum:
                self.category_stats[feat_idx][cat_val] = (
                    running_sum[cat_val],  # Total sum of y for this category
                    running_count[cat_val],  # Total count for this category
                )

        return X_encoded  # Return the encoded feature matrix

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new (validation/test) data using the learned category stats.
        For unseen data, we use all training data (no ordering needed since
        test samples never appear in training).
        """
        X_encoded = X.copy().astype(np.float64)  # Copy to avoid modification

        for feat_idx in self.cat_features:  # Encode each categorical feature
            for i in range(X.shape[0]):  # Process each sample
                cat_value = str(X[i, feat_idx])  # Get the category value

                if cat_value in self.category_stats[feat_idx]:  # Known category
                    total_sum, total_count = self.category_stats[feat_idx][cat_value]
                    # Use full training stats for test encoding
                    ts = (total_sum + self.global_mean * self.prior_smoothing) / (
                        total_count + self.prior_smoothing
                    )
                else:  # Unknown category (not seen in training)
                    ts = self.global_mean  # Fall back to the global mean

                X_encoded[i, feat_idx] = ts  # Replace with encoded value

        return X_encoded  # Return encoded matrix


# ---------------------------------------------------------------------------
# Oblivious Decision Tree
# ---------------------------------------------------------------------------

class ObliviousTree:
    """
    An oblivious (symmetric) decision tree where all nodes at the same
    depth level use the same (feature, threshold) split. This results in
    exactly 2^depth leaves, identified by a binary path of split outcomes.
    """

    def __init__(
        self,
        depth: int = 6,  # Number of split levels
        lambda_reg: float = 1.0,  # L2 regularization on leaf weights
    ) -> None:
        """Initialize the oblivious tree."""
        self.depth = depth  # Tree depth (number of levels)
        self.lambda_reg = lambda_reg  # L2 penalty on leaf values
        self.splits: List[Tuple[int, float]] = []  # (feature_idx, threshold) per level
        self.leaf_values: np.ndarray = np.array([])  # Prediction for each of 2^depth leaves

    def _compute_leaf_weight(
        self,
        gradients: np.ndarray,  # Gradients in this leaf
        hessians: np.ndarray,  # Hessians in this leaf
    ) -> float:
        """Optimal leaf weight: w = -G / (H + lambda)."""
        G = np.sum(gradients)  # Total gradient
        H = np.sum(hessians)  # Total hessian
        return -G / (H + self.lambda_reg)  # Newton step

    def _find_best_split_for_level(
        self,
        X: np.ndarray,  # Features for all samples
        gradients: np.ndarray,  # Gradients for all samples
        hessians: np.ndarray,  # Hessians for all samples
        leaf_assignments: np.ndarray,  # Current leaf index for each sample
        n_current_leaves: int,  # Number of leaves at this level
    ) -> Tuple[int, float, float]:
        """
        Find the best (feature, threshold) for this level of the oblivious tree.
        The same split is applied to ALL current leaves simultaneously.
        The gain is the total gain across all leaves when they are all split.
        """
        n_features = X.shape[1]  # Number of features to consider
        best_gain = -np.inf  # Track the best total gain
        best_feature = 0  # Best feature index
        best_threshold = 0.0  # Best threshold value

        for feat_idx in range(n_features):  # Try every feature
            # Get unique sorted values for candidate thresholds
            unique_vals = np.unique(X[:, feat_idx])  # Unique feature values
            if len(unique_vals) <= 1:  # Cannot split on a constant feature
                continue

            # Generate candidate thresholds as midpoints between consecutive values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0  # Midpoints

            # Subsample thresholds if too many (for speed)
            if len(thresholds) > 50:  # Limit candidate thresholds
                indices = np.linspace(0, len(thresholds) - 1, 50, dtype=int)
                thresholds = thresholds[indices]

            for threshold in thresholds:  # Try each candidate threshold
                total_gain = 0.0  # Accumulate gain across all current leaves
                valid_split = True  # Flag: is this split valid for all leaves?

                for leaf_id in range(n_current_leaves):  # Check each current leaf
                    leaf_mask = leaf_assignments == leaf_id  # Samples in this leaf
                    if np.sum(leaf_mask) < 2:  # Not enough samples to split
                        continue

                    leaf_X = X[leaf_mask, feat_idx]  # Feature values in this leaf
                    leaf_g = gradients[leaf_mask]  # Gradients in this leaf
                    leaf_h = hessians[leaf_mask]  # Hessians in this leaf

                    # Split the leaf into left (<=threshold) and right (>threshold)
                    left_mask = leaf_X <= threshold
                    right_mask = ~left_mask

                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue  # Degenerate split for this leaf

                    # Compute gain for this leaf's split
                    G_left = np.sum(leaf_g[left_mask])
                    H_left = np.sum(leaf_h[left_mask])
                    G_right = np.sum(leaf_g[right_mask])
                    H_right = np.sum(leaf_h[right_mask])
                    G_parent = G_left + G_right
                    H_parent = H_left + H_right

                    gain = (
                        (G_left ** 2) / (H_left + self.lambda_reg)
                        + (G_right ** 2) / (H_right + self.lambda_reg)
                        - (G_parent ** 2) / (H_parent + self.lambda_reg)
                    ) / 2.0

                    total_gain += gain  # Add this leaf's gain to the total

                if total_gain > best_gain:  # Found a better split
                    best_gain = total_gain
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def fit(
        self,
        X: np.ndarray,  # Feature matrix (n_samples, n_features)
        gradients: np.ndarray,  # Gradient for each sample
        hessians: np.ndarray,  # Hessian for each sample
    ) -> "ObliviousTree":
        """
        Build the oblivious tree level by level.
        At each level, find the best (feature, threshold) across all leaves,
        then split all leaves using that same condition.
        """
        n_samples = X.shape[0]
        self.splits = []  # Reset splits

        # All samples start in leaf 0
        leaf_assignments = np.zeros(n_samples, dtype=int)  # Current leaf index per sample

        for level in range(self.depth):  # Build one level at a time
            n_current_leaves = 2 ** level  # Number of leaves at this level

            # Find the best split for this level (applied to all leaves)
            feat_idx, threshold, gain = self._find_best_split_for_level(
                X, gradients, hessians, leaf_assignments, n_current_leaves,
            )

            self.splits.append((feat_idx, threshold))  # Store the split for this level

            # Update leaf assignments: each leaf splits into two
            # Left child: leaf_id remains the same, but shifted left by 1 bit
            # Right child: leaf_id gets a new bit set
            goes_right = (X[:, feat_idx] > threshold).astype(int)  # 1 if right
            leaf_assignments = leaf_assignments * 2 + goes_right  # Binary path update

        # Compute optimal leaf values for all 2^depth leaves
        n_leaves = 2 ** self.depth  # Total number of leaves
        self.leaf_values = np.zeros(n_leaves)  # One value per leaf

        for leaf_id in range(n_leaves):  # Compute value for each leaf
            leaf_mask = leaf_assignments == leaf_id  # Samples in this leaf
            if np.sum(leaf_mask) > 0:  # Leaf has samples
                self.leaf_values[leaf_id] = self._compute_leaf_weight(
                    gradients[leaf_mask], hessians[leaf_mask],
                )
            # else: leaf_value remains 0.0 (empty leaf)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by routing each sample through the oblivious tree.
        Since the tree is symmetric, we can compute the leaf index
        directly from the depth binary conditions without recursion.
        """
        n_samples = X.shape[0]
        leaf_ids = np.zeros(n_samples, dtype=int)  # Initialize leaf indices

        for level, (feat_idx, threshold) in enumerate(self.splits):  # Each level
            goes_right = (X[:, feat_idx] > threshold).astype(int)  # 1 if right
            leaf_ids = leaf_ids * 2 + goes_right  # Build binary path incrementally

        return self.leaf_values[leaf_ids]  # Look up leaf values by index


# ---------------------------------------------------------------------------
# CatBoost-Style Classifier (Ordered Boosting + Oblivious Trees)
# ---------------------------------------------------------------------------

class OrderedBoostingClassifier:
    """
    Gradient boosting classifier with ordered target encoding for
    categorical features and oblivious decision trees, following
    CatBoost's core principles.
    """

    def __init__(
        self,
        n_estimators: int = 100,  # Number of boosting rounds
        learning_rate: float = 0.1,  # Shrinkage per tree
        depth: int = 6,  # Oblivious tree depth
        lambda_reg: float = 1.0,  # L2 regularization
        cat_features: Optional[List[int]] = None,  # Categorical column indices
        prior_smoothing: float = 1.0,  # Smoothing for target encoding
        random_state: int = 42,  # Reproducibility seed
    ) -> None:
        """Store hyperparameters."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.lambda_reg = lambda_reg
        self.cat_features = cat_features if cat_features else []
        self.prior_smoothing = prior_smoothing
        self.random_state = random_state
        self.trees: List[ObliviousTree] = []  # Trained trees
        self.initial_pred: float = 0.0  # Initial log-odds
        self.encoder: Optional[OrderedTargetEncoder] = None  # Target encoder

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrderedBoostingClassifier":
        """
        Train ordered boosting classifier:
        1. Apply ordered target encoding to categorical features
        2. Initialize predictions with log-odds
        3. Sequentially train oblivious trees on gradients
        """
        n_samples = X.shape[0]

        # Step 1: Ordered target encoding for categorical features
        if len(self.cat_features) > 0:
            self.encoder = OrderedTargetEncoder(
                cat_features=self.cat_features,
                prior_smoothing=self.prior_smoothing,
                random_state=self.random_state,
            )
            X_encoded = self.encoder.fit_transform(X, y)  # Encode using ordered stats
        else:
            X_encoded = X.astype(np.float64)  # No categoricals to encode
            self.encoder = None

        # Step 2: Initialize with log-odds of positive class
        p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
        self.initial_pred = np.log(p / (1 - p))
        raw_predictions = np.full(n_samples, self.initial_pred)

        self.trees = []  # Reset tree list

        # Step 3: Boosting loop with oblivious trees
        for m in range(self.n_estimators):
            # Compute gradients and hessians from current predictions
            probs = self._sigmoid(raw_predictions)
            gradients = probs - y  # First-order gradient of log loss
            hessians = probs * (1 - probs)  # Second-order (diagonal Hessian)
            hessians = np.maximum(hessians, 1e-8)  # Prevent zero hessians

            # Build an oblivious tree on the gradients
            tree = ObliviousTree(depth=self.depth, lambda_reg=self.lambda_reg)
            tree.fit(X_encoded, gradients, hessians)
            self.trees.append(tree)

            # Update predictions with shrinkage
            raw_predictions += self.learning_rate * tree.predict(X_encoded)

        logger.info(
            "OrderedBoosting trained: %d trees, depth=%d, lr=%.3f, cat=%d",
            self.n_estimators, self.depth, self.learning_rate, len(self.cat_features),
        )
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute raw predictions by summing all trees."""
        if self.encoder is not None:
            X_encoded = self.encoder.transform(X)
        else:
            X_encoded = X.astype(np.float64)

        raw = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            raw += self.learning_rate * tree.predict(X_encoded)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        raw = self._raw_predict(X)
        p1 = self._sigmoid(raw)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Generate a hotel booking dataset with mixed continuous/categorical features.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, cat_feature_indices).
    """
    rng = np.random.RandomState(random_state)

    # Continuous features
    lead_time = rng.exponential(100, n_samples)
    adr = rng.normal(100, 40, n_samples).clip(20, 500)
    stays_weekend = rng.poisson(1, n_samples)
    stays_week = rng.poisson(3, n_samples)
    previous_cancellations = rng.poisson(0.3, n_samples)
    special_requests = rng.poisson(1.5, n_samples)

    # Categorical features
    country = rng.choice(
        ["PRT", "GBR", "FRA", "ESP", "DEU", "USA", "BRA", "OTHER"],
        n_samples, p=[0.3, 0.12, 0.1, 0.1, 0.08, 0.1, 0.1, 0.1],
    )
    market_segment = rng.choice(
        ["Online_TA", "Offline_TA", "Direct", "Corporate", "Groups"],
        n_samples, p=[0.45, 0.15, 0.15, 0.15, 0.10],
    )
    deposit_type = rng.choice(
        ["No_Deposit", "Non_Refund", "Refundable"],
        n_samples, p=[0.7, 0.2, 0.1],
    )

    X = np.column_stack([
        lead_time, adr, stays_weekend, stays_week,
        previous_cancellations, special_requests,
        country, market_segment, deposit_type,
    ])
    cat_features = [6, 7, 8]  # Indices of categorical columns

    # Generate cancellation labels
    cancel_logit = (
        0.005 * lead_time - 0.005 * adr
        + 0.5 * previous_cancellations - 0.3 * special_requests
        + 1.0 * (deposit_type == "Non_Refund").astype(float)
        - 0.5 * (market_segment == "Corporate").astype(float)
        + 0.3 * (country == "PRT").astype(float)
        + rng.normal(0, 0.5, n_samples)
    )
    cancel_prob = 1.0 / (1.0 + np.exp(-cancel_logit))
    y = (rng.random(n_samples) < cancel_prob).astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, cat_features=%s",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], cat_features,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, cat_features


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cat_features: Optional[List[int]] = None,
    **hyperparams: Any,
) -> OrderedBoostingClassifier:
    """Train an ordered boosting classifier."""
    defaults = dict(
        n_estimators=100,
        learning_rate=0.1,
        depth=6,
        lambda_reg=1.0,
        cat_features=cat_features if cat_features else [],
        prior_smoothing=1.0,
        random_state=42,
    )
    defaults.update(hyperparams)
    model = OrderedBoostingClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def _evaluate(model: OrderedBoostingClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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


def validate(model: OrderedBoostingClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Evaluate on validation data."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: OrderedBoostingClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
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
    cat_features: Optional[List[int]] = None,
) -> float:
    """Optuna objective: suggest params, train, return val F1."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "lambda_reg": trial.suggest_float("lambda_reg", 0.01, 10.0, log=True),
        "prior_smoothing": trial.suggest_float("prior_smoothing", 0.1, 10.0, log=True),
    }
    model = train(X_train, y_train, cat_features=cat_features, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_features: Optional[List[int]] = None,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search."""
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, cat_features=cat_features, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "n_estimators": ray_tune.choice([50, 100, 150]),
        "learning_rate": ray_tune.loguniform(0.01, 0.3),
        "depth": ray_tune.randint(3, 9),
        "lambda_reg": ray_tune.loguniform(0.01, 10.0),
        "prior_smoothing": ray_tune.loguniform(0.1, 10.0),
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
    cat_features: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare 3 configurations focusing on tree depth and learning rate.

    Configurations:
        1. shallow_fast (depth=4, lr=0.1):
           Shallow oblivious trees (16 leaves) with fast learning.
           Quick to train, good baseline.

        2. medium_moderate (depth=6, lr=0.1):
           CatBoost's default-ish depth. 64 leaves per tree.
           Balances complexity and regularization.

        3. deep_slow (depth=8, lr=0.01):
           Deep oblivious trees (256 leaves) with slow learning.
           Most complex model, needs more rounds to converge.
    """
    configs = {
        "shallow_fast (d=4, lr=0.1)": {
            # Reasoning: 4-level oblivious tree = 16 leaves. Simple model.
            # Fast to train and resistant to overfitting. Good for small data.
            "depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
        "medium_moderate (d=6, lr=0.1)": {
            # Reasoning: 6-level oblivious tree = 64 leaves. Moderate complexity.
            # CatBoost's default depth. Captures feature interactions.
            "depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
        "deep_slow (d=8, lr=0.01)": {
            # Reasoning: 8-level oblivious tree = 256 leaves. High capacity.
            # Low learning rate (0.01) prevents overfitting despite deep trees.
            # Needs more estimators to converge but may find complex patterns.
            "depth": 8,
            "learning_rate": 0.01,
            "n_estimators": 200,
        },
    }

    results = {}
    logger.info("=" * 70)
    logger.info("Comparing %d Ordered Boosting configurations", len(configs))
    logger.info("=" * 70)

    for name, params in configs.items():
        logger.info("\n--- Config: %s ---", name)
        model = train(X_train, y_train, cat_features=cat_features, **params)
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
# Real-World Demo: Hotel Booking Cancellation
# ---------------------------------------------------------------------------

def real_world_demo() -> Dict[str, float]:
    """
    Demonstrate ordered boosting on a simulated hotel booking cancellation problem.

    Domain: Hospitality / hotel revenue management
    Goal: Predict whether a hotel reservation will be cancelled.

    Features:
        - lead_time: Days between booking and arrival
        - adr: Average daily rate ($)
        - stays_weekend: Weekend night stays
        - stays_week: Weekday night stays
        - previous_cancellations: Prior cancellation count
        - special_requests: Number of special requests
        - country: Country of origin (categorical)
        - market_segment: Booking channel (categorical)
        - deposit_type: Deposit type (categorical)
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Hotel Booking Cancellation (Ordered Boosting)")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test, cat_features = generate_data(
        n_samples=1000, random_state=42,
    )

    feature_names = [
        "lead_time", "adr", "stays_weekend", "stays_week",
        "previous_cancellations", "special_requests",
        "country", "market_segment", "deposit_type",
    ]
    logger.info("Features: %s", feature_names)
    logger.info("Categorical: %s", [feature_names[i] for i in cat_features])
    logger.info("Cancellation rate: %.1f%%", 100 * np.mean(y_train))

    model = train(X_train, y_train, cat_features=cat_features, n_estimators=100, depth=6)
    validate(model, X_val, y_val)
    metrics = test(model, X_test, y_test)
    return metrics


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full ordered boosting pipeline."""
    logger.info("=" * 70)
    logger.info("Ordered Boosting with Oblivious Trees - NumPy From-Scratch")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test, cat_features = generate_data(n_samples=800)

    # Baseline
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, cat_features=cat_features, n_estimators=50, depth=4)
    validate(model, X_val, y_val)

    # Optuna HPO
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, cat_features),
        n_trials=10,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    best_model = train(X_train, y_train, cat_features=cat_features, **study.best_params)

    # Test
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Compare
    logger.info("\n--- Parameter Set Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val, cat_features)

    # Demo
    logger.info("\n--- Real-World Demo: Hotel Booking Cancellation ---")
    real_world_demo()


if __name__ == "__main__":
    main()
