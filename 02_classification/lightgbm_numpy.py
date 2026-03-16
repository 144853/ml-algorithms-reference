"""
Histogram-Based Gradient Boosting (LightGBM-Style) - NumPy From-Scratch
=========================================================================

Theory & Mathematics:
    This module implements a histogram-based gradient boosting classifier
    from scratch using NumPy, following the core principles of LightGBM.
    The key innovation is binning continuous feature values into discrete
    histograms, enabling O(#bins) split finding instead of O(#samples).

    Histogram-Based Split Finding:
        1. Bin each feature into B discrete bins (default 256):
           For feature j, compute B-1 equally spaced quantile boundaries.
           Map each value x_{ij} -> bin_id in {0, 1, ..., B-1}.

        2. For each feature and each candidate split:
           Accumulate gradient (G) and hessian (H) sums per bin.
           The split gain for splitting at bin b is:
               Gain = (G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda)
                       - (G_L + G_R)^2 / (H_L + H_R + lambda)) / 2 - gamma
           where G_L, H_L = sum of gradients/hessians in bins 0..b
                 G_R, H_R = sum of gradients/hessians in bins (b+1)..B-1
                 lambda = L2 regularization on leaf weights
                 gamma = minimum gain required to make a split

        3. Complexity: O(B * n_features) per split instead of O(N * n_features)
           This is the main speedup of LightGBM over vanilla gradient boosting.

    Leaf-Wise (Best-First) Tree Growth:
        Unlike level-wise growth (XGBoost default), LightGBM grows trees
        leaf-wise: at each step, split the leaf with the highest potential
        gain. This leads to deeper, more asymmetric trees that can reduce
        loss more aggressively, at the risk of overfitting (controlled by
        max_leaves / num_leaves).

        Algorithm:
            1. Start with root leaf
            2. While num_leaves < max_leaves:
               a. Among all current leaves, find the one with the best
                  potential split gain
               b. Split that leaf into two children
               c. num_leaves += 1 (one leaf becomes two, net +1)

    Optimal Leaf Weight (Newton step):
        w_j = -sum(g_i in leaf j) / (sum(h_i in leaf j) + lambda)
        This is derived from a second-order Taylor expansion of the loss.

    Loss Function (Binary Log Loss):
        L(y, F) = -[y * log(sigma(F)) + (1-y) * log(1 - sigma(F))]
        Gradient: g_i = sigma(F(x_i)) - y_i
        Hessian: h_i = sigma(F(x_i)) * (1 - sigma(F(x_i)))

    LightGBM-Specific Optimizations (simplified in this implementation):
        - Histogram binning: Reduces memory and computation
        - Leaf-wise growth: More aggressive loss reduction
        - Gradient-based One-Side Sampling (GOSS): Not implemented
        - Exclusive Feature Bundling (EFB): Not implemented
        - Histogram subtraction: parent - left = right (saves computation)

Business Use Cases:
    - Click-through rate (CTR) prediction for online advertising
    - Ranking and recommendation systems
    - Large-scale classification with millions of samples
    - Real-time prediction with fast inference
    - Feature-rich tabular data classification

Advantages:
    - O(#bins) split finding is much faster than O(#samples)
    - Leaf-wise growth produces deeper, more effective trees
    - Memory-efficient histogram representation
    - Natural handling of missing values (not implemented here)
    - Highly competitive accuracy on tabular data

Disadvantages:
    - Leaf-wise growth can overfit on small datasets
    - Histogram approximation loses some precision
    - More complex implementation than vanilla gradient boosting
    - Requires careful tuning of num_leaves and min_data_in_leaf
    - No GPU support in this NumPy implementation

Hyperparameters:
    - n_estimators: Number of boosting rounds
    - learning_rate: Shrinkage factor for each tree
    - num_leaves: Maximum number of leaves per tree (controls complexity)
    - max_bins: Number of histogram bins (256 is LightGBM's default)
    - lambda_reg: L2 regularization on leaf weights
    - gamma: Minimum gain to make a split
    - min_data_in_leaf: Minimum samples in a leaf node
    - subsample: Row subsampling ratio per round
"""

import logging  # Standard logging for training progress
import warnings  # Suppress non-critical sklearn warnings
from typing import Any, Dict, List, Optional, Tuple  # Type hints for documentation

import numpy as np  # Core numerical computing library
import optuna  # Bayesian hyperparameter optimization
from sklearn.datasets import make_classification  # Synthetic data generation
from sklearn.metrics import (  # Classification evaluation metrics
    accuracy_score,  # Overall prediction correctness
    classification_report,  # Detailed per-class metrics
    confusion_matrix,  # True/false positive/negative matrix
    f1_score,  # Harmonic mean of precision and recall
    precision_score,  # Positive predictive value
    recall_score,  # Sensitivity / true positive rate
    roc_auc_score,  # Area under the ROC curve
)
from sklearn.model_selection import train_test_split  # Stratified data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardization

# Configure logging with timestamp format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # Module-specific logger
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature Binning (Histogram Construction)
# ---------------------------------------------------------------------------

class FeatureBinner:
    """
    Bins continuous feature values into discrete histogram bins.
    Uses quantile-based binning to ensure roughly equal bin populations,
    which is crucial for finding good splits efficiently.
    """

    def __init__(self, max_bins: int = 256) -> None:
        """Initialize the binner with the maximum number of bins."""
        self.max_bins = max_bins  # Maximum number of discrete bins
        self.bin_edges: List[np.ndarray] = []  # Bin boundaries for each feature

    def fit(self, X: np.ndarray) -> "FeatureBinner":
        """
        Learn bin edges from training data using quantile-based binning.
        For each feature, compute (max_bins - 1) equally spaced quantiles.
        """
        self.bin_edges = []  # Reset bin edges for a fresh fit
        for j in range(X.shape[1]):  # Process each feature column independently
            # Compute quantile boundaries: divide the data into max_bins groups
            percentiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]  # Interior boundaries
            edges = np.percentile(X[:, j], percentiles)  # Quantile values
            edges = np.unique(edges)  # Remove duplicates (constant regions)
            self.bin_edges.append(edges)  # Store edges for this feature
        return self  # Allow method chaining

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map continuous feature values to their corresponding bin indices.
        Uses np.searchsorted for O(log(max_bins)) per value.
        Returns integer bin indices in [0, n_bins-1] for each feature.
        """
        X_binned = np.zeros_like(X, dtype=np.int32)  # Output matrix of bin indices
        for j in range(X.shape[1]):  # Transform each feature column
            # searchsorted returns the bin index where each value would be inserted
            X_binned[:, j] = np.searchsorted(self.bin_edges[j], X[:, j])  # Map value -> bin
        return X_binned  # Return the binned feature matrix


# ---------------------------------------------------------------------------
# Histogram-Based Leaf Node
# ---------------------------------------------------------------------------

class _HistLeafNode:
    """Node in a leaf-wise gradient boosting tree."""

    def __init__(
        self,
        indices: np.ndarray,  # Sample indices belonging to this node
        depth: int = 0,  # Current depth in the tree
        leaf_value: Optional[float] = None,  # Leaf prediction (None if internal)
        feature_idx: Optional[int] = None,  # Split feature index (None if leaf)
        bin_threshold: Optional[int] = None,  # Split bin index (None if leaf)
        left: Optional["_HistLeafNode"] = None,  # Left child (bin <= threshold)
        right: Optional["_HistLeafNode"] = None,  # Right child (bin > threshold)
        split_gain: float = 0.0,  # Gain achieved by splitting this node
    ) -> None:
        """Store node attributes."""
        self.indices = indices  # Which training samples reside in this node
        self.depth = depth  # How deep this node is in the tree
        self.leaf_value = leaf_value  # Prediction value if this is a leaf
        self.feature_idx = feature_idx  # Which feature to split on
        self.bin_threshold = bin_threshold  # Which bin boundary to split at
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.split_gain = split_gain  # Information gain from splitting

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return self.left is None and self.right is None  # Leaf has no children


# ---------------------------------------------------------------------------
# Histogram-Based Tree (Leaf-Wise Growth)
# ---------------------------------------------------------------------------

class HistogramGBTree:
    """
    A single gradient boosting tree using histogram-based split finding
    and leaf-wise (best-first) growth strategy, following LightGBM's approach.
    """

    def __init__(
        self,
        num_leaves: int = 31,  # Max leaves (LightGBM default)
        max_bins: int = 256,  # Number of histogram bins
        lambda_reg: float = 1.0,  # L2 regularization on leaf weights
        gamma: float = 0.0,  # Minimum split gain threshold
        min_data_in_leaf: int = 5,  # Minimum samples per leaf
    ) -> None:
        """Store tree hyperparameters."""
        self.num_leaves = num_leaves  # Controls tree complexity
        self.max_bins = max_bins  # Histogram resolution
        self.lambda_reg = lambda_reg  # L2 penalty on leaf weights
        self.gamma = gamma  # Minimum gain to justify a split
        self.min_data_in_leaf = min_data_in_leaf  # Prevents overfitting to few samples
        self.root: Optional[_HistLeafNode] = None  # Root of the tree
        self.n_features: int = 0  # Set during fitting

    def _compute_leaf_weight(
        self,
        gradients: np.ndarray,  # First-order gradients for samples in this leaf
        hessians: np.ndarray,  # Second-order Hessians for samples in this leaf
    ) -> float:
        """
        Compute the optimal leaf weight using Newton's method.
        w* = -G / (H + lambda) where G = sum of gradients, H = sum of hessians.
        """
        G = np.sum(gradients)  # Total gradient in this leaf
        H = np.sum(hessians)  # Total hessian in this leaf
        return -G / (H + self.lambda_reg)  # Newton step with L2 regularization

    def _build_histogram(
        self,
        X_binned: np.ndarray,  # Binned feature matrix for samples in this node
        gradients: np.ndarray,  # Gradients for samples in this node
        hessians: np.ndarray,  # Hessians for samples in this node
        feature_idx: int,  # Which feature to build the histogram for
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build gradient and hessian histograms for a single feature.
        Accumulates G and H sums for each bin, plus counts per bin.
        Returns (grad_hist, hess_hist, count_hist) each of shape (max_bins,).
        """
        # Initialize histogram arrays: one slot per possible bin value
        grad_hist = np.zeros(self.max_bins, dtype=np.float64)  # Gradient sum per bin
        hess_hist = np.zeros(self.max_bins, dtype=np.float64)  # Hessian sum per bin
        count_hist = np.zeros(self.max_bins, dtype=np.int32)  # Sample count per bin

        # Accumulate gradients, hessians, and counts into their bins
        bins = X_binned[:, feature_idx]  # Bin index for each sample in this feature
        for b, g, h in zip(bins, gradients, hessians):  # Iterate over all samples
            grad_hist[b] += g  # Add gradient to the appropriate bin
            hess_hist[b] += h  # Add hessian to the appropriate bin
            count_hist[b] += 1  # Increment count for this bin

        return grad_hist, hess_hist, count_hist  # Return the three histograms

    def _find_best_split_for_node(
        self,
        X_binned: np.ndarray,  # Binned features for samples in the node
        gradients: np.ndarray,  # Gradients for node samples
        hessians: np.ndarray,  # Hessians for node samples
    ) -> Tuple[Optional[int], Optional[int], float]:
        """
        Find the best (feature, bin_threshold) split for this node.
        Iterates over all features and bins, computing the split gain formula:
            Gain = (G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                    - (G_L+G_R)^2/(H_L+H_R+lambda)) / 2 - gamma
        Returns (best_feature, best_bin, best_gain).
        """
        n_samples = X_binned.shape[0]  # Number of samples in this node
        best_gain = -np.inf  # Track the highest gain found
        best_feature = None  # Which feature gives the best split
        best_bin = None  # Which bin boundary is best

        for feat_idx in range(self.n_features):  # Try every feature
            # Build histograms: O(n_samples) per feature
            grad_hist, hess_hist, count_hist = self._build_histogram(
                X_binned, gradients, hessians, feat_idx,
            )

            # Total gradient and hessian sums for the entire node
            G_total = np.sum(grad_hist)  # Sum of all gradients
            H_total = np.sum(hess_hist)  # Sum of all hessians
            n_total = np.sum(count_hist)  # Total sample count

            # Scan through bins to find the best split point: O(max_bins)
            G_left = 0.0  # Cumulative gradient sum for left partition
            H_left = 0.0  # Cumulative hessian sum for left partition
            n_left = 0  # Cumulative count for left partition

            for b in range(self.max_bins - 1):  # Try each bin as a split point
                G_left += grad_hist[b]  # Add this bin's gradient to left sum
                H_left += hess_hist[b]  # Add this bin's hessian to left sum
                n_left += count_hist[b]  # Add this bin's count to left sum

                G_right = G_total - G_left  # Right partition gradient
                H_right = H_total - H_left  # Right partition hessian
                n_right = n_total - n_left  # Right partition count

                # Skip if either side has too few samples
                if n_left < self.min_data_in_leaf or n_right < self.min_data_in_leaf:
                    continue  # Minimum leaf size constraint not met

                # Compute the split gain using the histogram-based formula
                gain_left = (G_left ** 2) / (H_left + self.lambda_reg)  # Left term
                gain_right = (G_right ** 2) / (H_right + self.lambda_reg)  # Right term
                gain_parent = (G_total ** 2) / (H_total + self.lambda_reg)  # Parent term
                gain = (gain_left + gain_right - gain_parent) / 2.0 - self.gamma  # Net gain

                if gain > best_gain:  # Found a better split
                    best_gain = gain  # Update best gain
                    best_feature = feat_idx  # Record best feature
                    best_bin = b  # Record best bin threshold

        return best_feature, best_bin, best_gain  # Return optimal split info

    def fit(
        self,
        X_binned: np.ndarray,  # Binned feature matrix (n_samples, n_features)
        gradients: np.ndarray,  # First-order gradients for all samples
        hessians: np.ndarray,  # Second-order hessians for all samples
    ) -> "HistogramGBTree":
        """
        Build the tree using leaf-wise (best-first) growth.
        At each step, split the leaf with the highest potential gain,
        until we reach num_leaves or no beneficial split exists.
        """
        self.n_features = X_binned.shape[1]  # Store feature count
        all_indices = np.arange(X_binned.shape[0])  # All sample indices

        # Create root node containing all samples
        root_weight = self._compute_leaf_weight(gradients, hessians)
        self.root = _HistLeafNode(
            indices=all_indices,  # All samples start in the root
            depth=0,  # Root is at depth 0
            leaf_value=root_weight,  # Initial leaf prediction
        )

        # List of splittable leaves, each with (node, gain, feature, bin)
        # Find initial best split for the root
        feat, bin_t, gain = self._find_best_split_for_node(
            X_binned[all_indices], gradients[all_indices], hessians[all_indices],
        )
        # Priority queue of (gain, node, feature, bin_threshold)
        splittable = []  # List of candidate splits
        if feat is not None and gain > 0:  # Root has a valid split
            splittable.append((gain, self.root, feat, bin_t))

        current_leaves = 1  # Start with one leaf (the root)

        # Leaf-wise growth: keep splitting the best leaf until we hit num_leaves
        while current_leaves < self.num_leaves and len(splittable) > 0:
            # Sort by gain descending and pick the best candidate
            splittable.sort(key=lambda x: -x[0])  # Sort by gain (highest first)
            best_gain, best_node, best_feat, best_bin = splittable.pop(0)  # Pop best

            if best_gain <= 0:  # No beneficial split available
                break  # Stop growing the tree

            # Perform the split on the chosen leaf
            indices = best_node.indices  # Samples in this leaf
            left_mask = X_binned[indices, best_feat] <= best_bin  # Left partition
            right_mask = ~left_mask  # Right partition (complement)

            left_indices = indices[left_mask]  # Sample indices for left child
            right_indices = indices[right_mask]  # Sample indices for right child

            # Compute optimal leaf weights for the two children
            left_weight = self._compute_leaf_weight(
                gradients[left_indices], hessians[left_indices],
            )
            right_weight = self._compute_leaf_weight(
                gradients[right_indices], hessians[right_indices],
            )

            # Create child nodes
            left_child = _HistLeafNode(
                indices=left_indices,
                depth=best_node.depth + 1,
                leaf_value=left_weight,
            )
            right_child = _HistLeafNode(
                indices=right_indices,
                depth=best_node.depth + 1,
                leaf_value=right_weight,
            )

            # Convert the current leaf into an internal node
            best_node.feature_idx = best_feat  # Store split feature
            best_node.bin_threshold = best_bin  # Store split bin
            best_node.left = left_child  # Attach left child
            best_node.right = right_child  # Attach right child
            best_node.leaf_value = None  # No longer a leaf (internal node now)
            best_node.split_gain = best_gain  # Store the gain achieved

            current_leaves += 1  # One leaf split into two, net gain of 1 leaf

            # Find best splits for the two new children and add to candidates
            for child, child_indices in [(left_child, left_indices), (right_child, right_indices)]:
                if len(child_indices) >= 2 * self.min_data_in_leaf:  # Enough samples
                    f, b, g = self._find_best_split_for_node(
                        X_binned[child_indices],
                        gradients[child_indices],
                        hessians[child_indices],
                    )
                    if f is not None and g > 0:  # Valid split found
                        splittable.append((g, child, f, b))  # Add to candidates

        return self  # Return the fitted tree

    def predict(self, X_binned: np.ndarray) -> np.ndarray:
        """Predict leaf values for each sample by routing through the tree."""
        predictions = np.zeros(X_binned.shape[0])  # Initialize output array
        for i in range(X_binned.shape[0]):  # Route each sample through the tree
            predictions[i] = self._predict_single(X_binned[i], self.root)
        return predictions  # Return all predictions

    def _predict_single(self, x: np.ndarray, node: _HistLeafNode) -> float:
        """Route a single sample through the tree to find its leaf value."""
        if node.is_leaf():  # Reached a leaf node
            return node.leaf_value  # Return the leaf's prediction
        if x[node.feature_idx] <= node.bin_threshold:  # Go left
            return self._predict_single(x, node.left)  # Recurse left
        return self._predict_single(x, node.right)  # Recurse right


# ---------------------------------------------------------------------------
# LightGBM-Style Classifier
# ---------------------------------------------------------------------------

class HistGBMClassifier:
    """
    Histogram-based gradient boosting classifier following LightGBM principles.
    Uses binned features, leaf-wise tree growth, and gradient/hessian optimization.
    """

    def __init__(
        self,
        n_estimators: int = 100,  # Number of boosting rounds
        learning_rate: float = 0.1,  # Shrinkage factor per tree
        num_leaves: int = 31,  # Max leaves per tree (LightGBM default)
        max_bins: int = 256,  # Histogram bin count
        lambda_reg: float = 1.0,  # L2 regularization
        gamma: float = 0.0,  # Minimum split gain
        min_data_in_leaf: int = 5,  # Min samples per leaf
        subsample: float = 1.0,  # Row subsampling ratio
        random_state: int = 42,  # Reproducibility seed
    ) -> None:
        """Store all hyperparameters for the classifier."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_bins = max_bins
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_data_in_leaf = min_data_in_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees: List[HistogramGBTree] = []  # Trained trees
        self.initial_pred: float = 0.0  # Initial log-odds prediction
        self.binner: Optional[FeatureBinner] = None  # Feature binning transformer

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function: sigma(z) = 1/(1+exp(-z))."""
        z = np.clip(z, -500, 500)  # Prevent overflow in exp
        return 1.0 / (1.0 + np.exp(-z))  # Standard sigmoid formula

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGBMClassifier":
        """
        Train the histogram gradient boosting classifier.
        Steps: bin features, initialize predictions, then iteratively
        fit trees to the negative gradient (residuals).
        """
        rng = np.random.RandomState(self.random_state)  # Create random state
        n_samples = X.shape[0]  # Number of training samples

        # Step 1: Bin the features into discrete histograms
        self.binner = FeatureBinner(max_bins=self.max_bins)  # Create binner
        self.binner.fit(X)  # Learn bin edges from training data
        X_binned = self.binner.transform(X)  # Transform features to bin indices

        # Step 2: Initialize predictions with the log-odds of the positive class
        p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)  # Mean positive rate
        self.initial_pred = np.log(p / (1 - p))  # Convert to log-odds
        raw_predictions = np.full(n_samples, self.initial_pred)  # Initialize all samples

        self.trees = []  # Reset the tree list

        # Step 3: Boosting loop
        for m in range(self.n_estimators):  # Train each tree sequentially
            # Subsample rows if configured
            if self.subsample < 1.0:  # Stochastic gradient boosting
                idx = rng.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
            else:  # Use all samples
                idx = np.arange(n_samples)

            # Compute gradients and hessians for the current predictions
            probs = self._sigmoid(raw_predictions[idx])  # Current predicted probabilities
            gradients = probs - y[idx]  # Gradient of log loss: p - y
            hessians = probs * (1 - probs)  # Hessian of log loss: p*(1-p)
            hessians = np.maximum(hessians, 1e-8)  # Prevent zero hessians

            # Build a histogram-based tree on the residuals
            tree = HistogramGBTree(
                num_leaves=self.num_leaves,
                max_bins=self.max_bins,
                lambda_reg=self.lambda_reg,
                gamma=self.gamma,
                min_data_in_leaf=self.min_data_in_leaf,
            )
            tree.fit(X_binned[idx], gradients, hessians)  # Fit tree to gradients
            self.trees.append(tree)  # Store the trained tree

            # Update raw predictions with the tree's output (shrunk by learning_rate)
            update = tree.predict(X_binned)  # Predict for ALL samples (not just subset)
            raw_predictions += self.learning_rate * update  # Additive update

        logger.info(
            "HistGBM (NumPy) trained: %d trees, num_leaves=%d, lr=%.3f",
            self.n_estimators, self.num_leaves, self.learning_rate,
        )
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute raw log-odds predictions by summing all tree outputs."""
        X_binned = self.binner.transform(X)  # Bin the features
        raw = np.full(X.shape[0], self.initial_pred)  # Start with initial prediction
        for tree in self.trees:  # Add each tree's contribution
            raw += self.learning_rate * tree.predict(X_binned)  # Shrunk tree output
        return raw  # Return raw log-odds scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert raw predictions to class probabilities via sigmoid."""
        raw = self._raw_predict(X)  # Get raw log-odds
        p1 = self._sigmoid(raw)  # Convert to P(y=1)
        return np.column_stack([1 - p1, p1])  # Return [P(y=0), P(y=1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hard class labels by thresholding probabilities at 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)  # Threshold


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data and split into train/val/test."""
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

    # Note: We do NOT standardize features here because histogram binning
    # is invariant to monotonic feature transformations. LightGBM doesn't
    # require feature scaling.
    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> HistGBMClassifier:
    """Train a HistGBM classifier with given hyperparameters."""
    defaults = dict(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_bins=256,
        lambda_reg=1.0,
        gamma=0.0,
        min_data_in_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    defaults.update(hyperparams)
    model = HistGBMClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def _evaluate(model: HistGBMClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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


def validate(model: HistGBMClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Evaluate on validation data."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: HistGBMClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
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
        "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63]),
        "lambda_reg": trial.suggest_float("lambda_reg", 0.01, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
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
        "n_estimators": ray_tune.choice([50, 100, 150]),
        "learning_rate": ray_tune.loguniform(0.01, 0.3),
        "num_leaves": ray_tune.choice([15, 31, 63]),
        "lambda_reg": ray_tune.loguniform(0.01, 10.0),
        "gamma": ray_tune.uniform(0.0, 5.0),
        "min_data_in_leaf": ray_tune.randint(5, 50),
        "subsample": ray_tune.uniform(0.5, 1.0),
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
    Compare 4 configurations focusing on num_leaves and min_data_in_leaf.

    Configurations:
        1. few_leaves_strict (num_leaves=15, min_data=20):
           Conservative tree growth. Avoids overfitting but may underfit
           complex data. Good for small datasets.

        2. default_lgbm (num_leaves=31, min_data=5):
           LightGBM's default. Balanced complexity. Works well on most
           tabular datasets without modification.

        3. many_leaves_relaxed (num_leaves=63, min_data=5):
           Aggressive tree growth for complex patterns. Higher capacity
           but needs more data to avoid overfitting.

        4. many_leaves_strict (num_leaves=63, min_data=20):
           Complex trees but with strict leaf size constraints. Attempts
           to capture complex patterns while preventing leaf overfitting.
    """
    configs = {
        "few_leaves_strict (L=15, m=20)": {
            # Reasoning: Conservative approach. 15 leaves limit tree complexity.
            # min_data_in_leaf=20 ensures each leaf has statistical significance.
            # Best for small/medium datasets or when overfitting is a concern.
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "n_estimators": 100,
        },
        "default_lgbm (L=31, m=5)": {
            # Reasoning: LightGBM's default values. 31 leaves with 5 min samples
            # provides a good balance. The standard starting point for any task.
            "num_leaves": 31,
            "min_data_in_leaf": 5,
            "n_estimators": 100,
        },
        "many_leaves_relaxed (L=63, m=5)": {
            # Reasoning: 63 leaves allows very complex decision boundaries.
            # Low min_data_in_leaf (5) gives maximum flexibility. Works best
            # on large datasets. Risk of overfitting on small data.
            "num_leaves": 63,
            "min_data_in_leaf": 5,
            "n_estimators": 100,
        },
        "many_leaves_strict (L=63, m=20)": {
            # Reasoning: High leaf count for complexity, but strict min_data
            # to prevent individual leaf overfitting. A compromise between
            # model capacity and regularization.
            "num_leaves": 63,
            "min_data_in_leaf": 20,
            "n_estimators": 100,
        },
    }

    results = {}
    logger.info("=" * 70)
    logger.info("Comparing %d HistGBM configurations", len(configs))
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
    Demonstrate histogram gradient boosting on a simulated click-through rate
    (CTR) prediction problem for online advertising.

    Domain: Digital advertising / ad tech
    Goal: Predict whether a user will click on a displayed ad (click = 1).

    Features simulate real ad serving signals:
        - user_age: Age of the user in years
        - device_type: Device category (0=mobile, 1=tablet, 2=desktop)
        - ad_position: Position of the ad on the page (1=top, 2=middle, 3=bottom)
        - hour_of_day: Hour when the ad was shown (0-23)
        - day_of_week: Day of the week (0=Monday, 6=Sunday)
        - session_duration: How long the user has been on the site (minutes)
        - num_previous_clicks: Historical click count for this user
        - page_depth: How many pages the user has visited in this session
    """
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    n_samples = 1000  # Realistic CTR dataset size

    # Generate domain-specific features
    user_age = rng.normal(35, 12, n_samples).clip(18, 70)  # Age 18-70
    device_type = rng.choice([0, 1, 2], n_samples, p=[0.6, 0.15, 0.25])  # 60% mobile
    ad_position = rng.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])  # Position mix
    hour_of_day = rng.randint(0, 24, n_samples)  # Hour of the day
    day_of_week = rng.randint(0, 7, n_samples)  # Day of week
    session_duration = rng.exponential(5, n_samples)  # Minutes on site
    num_previous_clicks = rng.poisson(3, n_samples)  # Historical click count
    page_depth = rng.poisson(2, n_samples) + 1  # Pages visited (min 1)

    X = np.column_stack([
        user_age, device_type, ad_position, hour_of_day,
        day_of_week, session_duration, num_previous_clicks, page_depth,
    ])

    # CTR model: typical CTR is 1-5%, so we model a low base rate
    click_logit = (
        -3.0  # Low base rate (typical CTR ~5%)
        - 0.01 * user_age  # Younger users click more
        + 0.3 * (device_type == 0).astype(float)  # Mobile users click more
        - 0.5 * (ad_position == 3).astype(float)  # Bottom ads get fewer clicks
        + 0.1 * np.sin(hour_of_day * np.pi / 12)  # Time-of-day effect
        + 0.1 * num_previous_clicks  # Engaged users click more
        - 0.05 * session_duration  # Long sessions = less interest in ads
        + rng.normal(0, 0.3, n_samples)  # Noise
    )
    click_prob = 1.0 / (1.0 + np.exp(-click_logit))  # Sigmoid
    y = (rng.random(n_samples) < click_prob).astype(int)  # Binary click label

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    feature_names = [
        "user_age", "device_type", "ad_position", "hour_of_day",
        "day_of_week", "session_duration", "num_previous_clicks", "page_depth",
    ]

    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Click-Through Rate Prediction (HistGBM)")
    logger.info("=" * 70)
    logger.info("Features: %s", feature_names)
    logger.info("Click rate: %.1f%%", 100 * np.mean(y))

    model = train(X_train, y_train, n_estimators=100, num_leaves=31, min_data_in_leaf=5)
    validate(model, X_val, y_val)
    metrics = test(model, X_test, y_test)
    return metrics


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full LightGBM-style gradient boosting pipeline."""
    logger.info("=" * 70)
    logger.info("Histogram-Based Gradient Boosting (LightGBM-Style) - NumPy Implementation")
    logger.info("=" * 70)

    # Generate data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(n_samples=800, n_features=15)

    # Baseline
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, n_estimators=50, num_leaves=31)
    validate(model, X_val, y_val)

    # Optuna HPO
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

    # Compare parameter sets
    logger.info("\n--- Parameter Set Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Real-world demo
    logger.info("\n--- Real-World Demo: Click-Through Rate ---")
    real_world_demo()


if __name__ == "__main__":
    main()
