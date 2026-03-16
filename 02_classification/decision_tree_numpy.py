"""
Decision Tree Classifier - NumPy From-Scratch Implementation (CART Algorithm)
=============================================================================

Theory & Mathematics:
    A Decision Tree is a non-parametric supervised learning algorithm that learns
    simple if/then/else rules from the data by recursively partitioning the feature
    space into rectangular regions. The CART (Classification and Regression Trees)
    algorithm, introduced by Breiman et al. (1984), builds binary trees using greedy
    recursive binary splitting.

    Core Algorithm - Recursive Binary Splitting:
        1. For each node, consider ALL features and ALL possible thresholds.
        2. Select the (feature, threshold) pair that maximizes the impurity reduction.
        3. Split the data into left (feature <= threshold) and right (feature > threshold).
        4. Recurse on left and right children until a stopping criterion is met.

    Impurity Measures:

        Gini Impurity:
            Gini(S) = 1 - sum_{k=1}^{C} p_k^2
            where p_k = proportion of class k in set S, C = number of classes.
            Range: [0, 1 - 1/C]. For binary: [0, 0.5].
            Interpretation: probability that a randomly chosen sample would be
            incorrectly classified if labeled according to the class distribution.
            WHY: Computationally cheaper than entropy (no logarithm), works well
            in practice, and is the default criterion in most implementations.

        Entropy (Information Gain):
            Entropy(S) = -sum_{k=1}^{C} p_k * log2(p_k)
            Information Gain = Entropy(parent) - sum_{j} (n_j / n_parent) * Entropy(child_j)
            Range: [0, log2(C)]. For binary: [0, 1.0].
            Interpretation: measures the average amount of information (in bits) needed
            to identify the class of a randomly chosen sample.
            WHY: Rooted in information theory; tends to produce slightly more balanced
            trees because log penalizes impure nodes more heavily than Gini.

        Impurity Reduction (the objective we maximize at each split):
            Delta_I = I(parent) - (n_left/n) * I(left) - (n_right/n) * I(right)
            We pick the (feature, threshold) that maximizes Delta_I.

    Prediction:
        To predict a new sample, traverse the tree from root to leaf following the
        learned split rules. The leaf's majority class becomes the prediction.

    Pruning (Regularization):
        Without constraints, a decision tree will grow until every leaf is pure,
        perfectly memorizing the training data (overfitting). We control complexity via:
        - max_depth: limits the number of levels in the tree
        - min_samples_split: minimum samples needed to attempt a split
        - min_samples_leaf: minimum samples required in each leaf after a split

Business Use Cases:
    - Medical diagnosis: transparent decision rules for clinical workflows
    - Credit scoring: regulators require explainable models (GDPR, ECOA)
    - Customer segmentation: marketing teams can interpret rules directly
    - Root cause analysis: manufacturing defect classification
    - Triage systems: emergency room patient prioritization

Advantages:
    - Highly interpretable: can be visualized as a flowchart
    - No feature scaling required (splits based on thresholds, not distances)
    - Handles both numerical and categorical features
    - Non-parametric: no assumptions about underlying data distribution
    - Automatically performs feature selection (uninformative features are ignored)
    - Can capture non-linear relationships and feature interactions

Disadvantages:
    - High variance: small data changes can produce very different trees
    - Prone to overfitting without regularization (especially deep trees)
    - Axis-aligned splits: cannot efficiently capture diagonal boundaries
    - Greedy algorithm: locally optimal splits, not globally optimal tree
    - Biased toward features with many unique values (more candidate splits)
    - Single trees generally underperform ensembles (Random Forest, XGBoost)

Complexity:
    - Training: O(n * m * n * log(n)) in worst case, where n = samples, m = features
      (for each of n*m candidate splits, sorting n samples takes O(n log n))
    - Prediction: O(depth) per sample, where depth <= log2(n) for balanced trees
    - Space: O(nodes) for storing the tree structure
"""

# --- Standard library imports ---
# logging: provides structured output messages with severity levels (INFO, DEBUG, WARNING).
# WHY: Structured logging is superior to print() for production code; supports
# log levels, file output, and monitoring system integration.
import logging  # Standard library module for structured logging output

# warnings: used to suppress noisy convergence/deprecation warnings.
# WHY: During hyperparameter optimization, many edge-case configurations trigger
# warnings that clutter output without providing actionable information.
import warnings  # Standard library module for warning control

# typing: provides type annotations for function signatures and return types.
# WHY: Type hints serve as documentation, enable IDE autocompletion, and allow
# static analysis tools like mypy to catch type errors before runtime.
from typing import Any, Dict, List, Optional, Tuple  # Type annotation classes

# --- Third-party imports ---
# numpy: foundational numerical computing library for array operations.
# WHY: All ML data flows through numpy arrays; vectorized operations are orders
# of magnitude faster than Python loops for numerical computation.
import numpy as np  # Numerical computing library - the backbone of from-scratch ML

# optuna: Bayesian hyperparameter optimization framework using TPE (Tree-structured Parzen Estimator).
# WHY: Optuna intelligently explores the hyperparameter space using past trial results
# to focus on promising regions, converging faster than grid or random search.
import optuna  # Hyperparameter optimization framework

# sklearn metrics: each captures a different aspect of classification performance.
# WHY: A single metric (e.g., accuracy) is insufficient; precision/recall trade-offs,
# AUC-ROC threshold independence, and per-class analysis are all important.
from sklearn.metrics import (
    accuracy_score,           # Overall correctness: (TP+TN) / total
    classification_report,    # Per-class precision, recall, F1 summary table
    f1_score,                 # Harmonic mean of precision and recall
    precision_score,          # Of positive predictions, fraction correct: TP / (TP+FP)
    recall_score,             # Of actual positives, fraction found: TP / (TP+FN)
    roc_auc_score,            # Area under ROC curve; threshold-independent ranking metric
)

# train_test_split: splits data into train/val/test sets with stratification.
# WHY: We need separate sets for training (learn parameters), validation (tune
# hyperparameters), and test (final unbiased evaluation) to avoid data leakage.
from sklearn.model_selection import train_test_split  # Data splitting utility

# make_classification: generates synthetic classification datasets.
# WHY: Synthetic data lets us control difficulty, dimensionality, and class balance,
# ideal for benchmarking and understanding algorithm behavior.
from sklearn.datasets import make_classification  # Synthetic data generator

# --- Logging configuration ---
# Set up logging to display timestamps, severity level, and message content.
# WHY: level=INFO shows training progress and metrics but hides verbose DEBUG output.
logging.basicConfig(
    level=logging.INFO,  # Only show INFO and above (not DEBUG)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + level + message
)
# Create a module-level logger named after this file for identification.
# WHY: __name__ makes the logger name match the module ("decision_tree_numpy").
logger = logging.getLogger(__name__)  # Module-specific logger instance

# Suppress warnings during Optuna hyperparameter search.
# WHY: Edge-case parameter combos produce valid but warned configurations.
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings


# ---------------------------------------------------------------------------
# Tree Node Data Structure
# ---------------------------------------------------------------------------

class TreeNode:
    """Represents a single node in the decision tree.

    Each node is either an internal (decision) node with a split rule,
    or a leaf node with a class prediction.

    WHY a class instead of a dict: OOP provides clearer structure, type safety,
    and makes tree traversal code more readable. Each node cleanly encapsulates
    its split rule (feature_index, threshold) and child pointers (left, right).
    """

    def __init__(
        self,
        feature_index: Optional[int] = None,  # Index of the feature to split on
        threshold: Optional[float] = None,     # Threshold value for the split
        left: Optional["TreeNode"] = None,     # Left child (feature <= threshold)
        right: Optional["TreeNode"] = None,    # Right child (feature > threshold)
        value: Optional[int] = None,           # Predicted class (leaf nodes only)
        impurity: float = 0.0,                 # Impurity at this node (for analysis)
        n_samples: int = 0,                    # Number of samples reaching this node
    ):
        """Initialize a tree node with optional split parameters or leaf value.

        For internal nodes: feature_index, threshold, left, right are set.
        For leaf nodes: value (predicted class) is set, children are None.
        """
        self.feature_index = feature_index  # Which feature dimension to test
        self.threshold = threshold          # The cutoff value for the split
        self.left = left                    # Left subtree (samples where feature <= threshold)
        self.right = right                  # Right subtree (samples where feature > threshold)
        self.value = value                  # Class label for leaf prediction
        self.impurity = impurity            # Gini or entropy at this node
        self.n_samples = n_samples          # Count of training samples at this node

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (terminal) node.

        WHY: A leaf node has a prediction value but no children.
        This is used during prediction to know when to stop traversal.
        """
        return self.value is not None  # Leaf nodes have a class value assigned


# ---------------------------------------------------------------------------
# Decision Tree Classifier (From Scratch)
# ---------------------------------------------------------------------------

class DecisionTreeFromScratch:
    """CART Decision Tree Classifier implemented from scratch using NumPy.

    This implements the full CART algorithm:
    1. Exhaustive search over all features and thresholds
    2. Gini impurity or entropy as split criterion
    3. Recursive binary splitting
    4. Pruning via max_depth, min_samples_split, min_samples_leaf

    WHY from scratch: Understanding the internals of decision trees is essential
    for ML engineers. Building from scratch reveals how split selection works,
    why trees overfit, and how regularization parameters affect the structure.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,    # Maximum tree depth (None = unlimited)
        min_samples_split: int = 2,          # Minimum samples to attempt a split
        min_samples_leaf: int = 1,           # Minimum samples in each leaf after split
        criterion: str = "gini",             # Impurity criterion: "gini" or "entropy"
    ):
        """Initialize the decision tree with hyperparameters.

        Args:
            max_depth: Controls tree complexity. None means grow until pure leaves.
                WHY: Deeper trees memorize training data (overfit). Limiting depth
                forces the tree to learn general patterns instead.
            min_samples_split: Don't split nodes with fewer than this many samples.
                WHY: Splitting tiny nodes captures noise rather than signal.
            min_samples_leaf: Each leaf must have at least this many samples.
                WHY: Prevents creating leaves with a single sample, which is
                pure overfitting. Forces the tree to make broader generalizations.
            criterion: Which impurity measure to use for split evaluation.
                WHY: Gini is slightly faster (no log), entropy produces slightly
                more balanced trees. In practice, they give very similar results.
        """
        self.max_depth = max_depth              # Store maximum depth limit
        self.min_samples_split = min_samples_split  # Store minimum split threshold
        self.min_samples_leaf = min_samples_leaf    # Store minimum leaf size
        self.criterion = criterion              # Store impurity criterion choice
        self.root: Optional[TreeNode] = None    # Root node (set during fit)
        self.n_classes: int = 0                 # Number of unique classes (set during fit)
        self.n_features: int = 0                # Number of features (set during fit)

    def _gini_impurity(self, y: np.ndarray) -> float:
        """Compute Gini impurity for a set of labels.

        Formula: Gini(S) = 1 - sum_{k=1}^{C} p_k^2
        where p_k is the proportion of class k in set S.

        WHY Gini: It measures the probability that a randomly chosen sample
        would be misclassified if labeled according to the class distribution.
        A pure node (all same class) has Gini = 0. Maximum impurity occurs
        when all classes are equally represented.

        Args:
            y: Array of class labels for the samples in this node.

        Returns:
            Gini impurity value in range [0, 1 - 1/C].
        """
        n_samples = len(y)  # Total number of samples in this node
        if n_samples == 0:  # Empty node has zero impurity (edge case)
            return 0.0  # Return 0 for empty sets to avoid division by zero

        # Count occurrences of each class using np.bincount.
        # WHY bincount: It's the fastest way to count integer class occurrences
        # in NumPy, much faster than np.unique with return_counts.
        counts = np.bincount(y, minlength=self.n_classes)  # Class frequency array

        # Compute class probabilities by dividing counts by total samples.
        # WHY: p_k = n_k / n gives the empirical probability of each class.
        probabilities = counts / n_samples  # p_k for each class k

        # Gini = 1 - sum(p_k^2): subtract sum of squared probabilities from 1.
        # WHY squared: sum(p_k^2) is the probability of correct classification
        # under random labeling, so 1 - sum(p_k^2) is the error probability.
        gini = 1.0 - np.sum(probabilities ** 2)  # Gini impurity formula

        return gini  # Return the computed Gini impurity

    def _entropy(self, y: np.ndarray) -> float:
        """Compute entropy for a set of labels.

        Formula: H(S) = -sum_{k=1}^{C} p_k * log2(p_k)
        where p_k is the proportion of class k in set S.

        WHY Entropy: From information theory, entropy measures the average
        number of bits needed to encode the class of a random sample.
        A pure node has entropy = 0 (no uncertainty). Maximum entropy = log2(C)
        occurs when all classes are equally likely.

        Args:
            y: Array of class labels for the samples in this node.

        Returns:
            Entropy value in range [0, log2(C)].
        """
        n_samples = len(y)  # Total number of samples in this node
        if n_samples == 0:  # Empty node has zero entropy (edge case)
            return 0.0  # Return 0 for empty sets

        # Count occurrences of each class.
        counts = np.bincount(y, minlength=self.n_classes)  # Class frequency array

        # Compute class probabilities, only for non-zero classes.
        # WHY filter zeros: log2(0) is undefined; we skip classes not present.
        probabilities = counts / n_samples  # Empirical class probabilities
        # Filter out zero probabilities to avoid log(0) = -inf.
        probabilities = probabilities[probabilities > 0]  # Keep only non-zero probs

        # Entropy = -sum(p_k * log2(p_k)): negative because log of fraction is negative.
        # WHY log2: convention from information theory; measures in bits.
        entropy = -np.sum(probabilities * np.log2(probabilities))  # Shannon entropy

        return entropy  # Return the computed entropy

    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity using the selected criterion.

        WHY a dispatch method: Centralizes the criterion choice so the rest
        of the code doesn't need to check which criterion to use.

        Args:
            y: Array of class labels.

        Returns:
            Impurity value (Gini or entropy depending on self.criterion).
        """
        if self.criterion == "gini":  # Use Gini impurity
            return self._gini_impurity(y)  # Delegate to Gini computation
        elif self.criterion == "entropy":  # Use entropy
            return self._entropy(y)  # Delegate to entropy computation
        else:  # Unknown criterion
            raise ValueError(f"Unknown criterion: {self.criterion}")  # Fail loudly

    def _information_gain(
        self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """Compute information gain from a candidate split.

        Formula: IG = I(parent) - (n_left/n) * I(left) - (n_right/n) * I(right)

        WHY: Information gain measures how much a split reduces impurity.
        We want to maximize this: the split that reduces impurity the most
        creates the most homogeneous child nodes.

        Args:
            y_parent: Labels of the parent node (before split).
            y_left: Labels of the left child (feature <= threshold).
            y_right: Labels of the right child (feature > threshold).

        Returns:
            Information gain value (higher is better).
        """
        n_parent = len(y_parent)  # Total samples in the parent node
        n_left = len(y_left)      # Samples going to the left child
        n_right = len(y_right)    # Samples going to the right child

        if n_left == 0 or n_right == 0:  # Degenerate split sends all data one way
            return 0.0  # No information gained from a trivial split

        # Compute parent impurity.
        # WHY: This is the baseline impurity before the split.
        parent_impurity = self._impurity(y_parent)  # I(parent)

        # Compute weighted average of children's impurities.
        # WHY weighted: Larger child nodes should contribute more to the average
        # because they represent more of the data. This is the proper way to
        # aggregate impurity across children.
        left_weight = n_left / n_parent    # Fraction of data going left
        right_weight = n_right / n_parent  # Fraction of data going right

        # Weighted child impurity = w_L * I(left) + w_R * I(right).
        child_impurity = (
            left_weight * self._impurity(y_left)    # Weighted left impurity
            + right_weight * self._impurity(y_right)  # Weighted right impurity
        )

        # Information gain = parent impurity - weighted child impurity.
        # WHY subtraction: If children are purer than parent, gain is positive.
        gain = parent_impurity - child_impurity  # Impurity reduction

        return gain  # Return the information gain value

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best (feature, threshold) split by exhaustive search.

        For each feature, we sort the data, consider midpoints between consecutive
        unique values as candidate thresholds, and pick the one with highest
        information gain.

        WHY exhaustive search: CART guarantees finding the globally optimal split
        at each node (though the overall tree structure is only locally optimal
        due to greedy construction). We must check all features and thresholds.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).

        Returns:
            Tuple of (best_feature_index, best_threshold, best_gain).
            Returns (None, None, 0.0) if no valid split exists.
        """
        best_gain = 0.0           # Track the highest information gain found
        best_feature = None        # Track which feature gives the best split
        best_threshold = None      # Track the threshold for the best split

        n_samples, n_features = X.shape  # Get data dimensions

        # Iterate over every feature dimension.
        # WHY all features: We need to find the globally best split at this node,
        # which requires comparing all possible feature-threshold combinations.
        for feature_idx in range(n_features):  # Loop over each feature
            # Extract the column for this feature.
            feature_values = X[:, feature_idx]  # Shape: (n_samples,)

            # Get sorted unique values for candidate threshold computation.
            # WHY sorted unique: We only need to consider thresholds between
            # distinct values; thresholds within a group of identical values
            # would produce the exact same split.
            unique_values = np.unique(feature_values)  # Sorted unique feature values

            # If all values are identical, no split is possible on this feature.
            if len(unique_values) <= 1:  # Only one unique value
                continue  # Skip to next feature

            # Compute candidate thresholds as midpoints between consecutive unique values.
            # WHY midpoints: Placing the threshold exactly on a data point is ambiguous
            # (does it go left or right?). Midpoints avoid this and are the standard
            # approach in CART implementations.
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0  # Midpoints

            # Evaluate each candidate threshold for this feature.
            for threshold in thresholds:  # Loop over candidate thresholds
                # Split data into left (<=) and right (>) based on the threshold.
                # WHY <=: Convention from CART; samples equal to threshold go left.
                left_mask = feature_values <= threshold  # Boolean mask for left child
                right_mask = ~left_mask                   # Boolean mask for right child

                y_left = y[left_mask]    # Labels for left child
                y_right = y[right_mask]  # Labels for right child

                # Check min_samples_leaf constraint on both children.
                # WHY: If either child has fewer samples than min_samples_leaf,
                # this split would create an overly specific leaf node.
                if len(y_left) < self.min_samples_leaf:  # Left child too small
                    continue  # Skip this threshold
                if len(y_right) < self.min_samples_leaf:  # Right child too small
                    continue  # Skip this threshold

                # Compute information gain for this split.
                gain = self._information_gain(y, y_left, y_right)  # IG value

                # Update best split if this one is better.
                # WHY >: We want the split that maximizes information gain.
                if gain > best_gain:  # Found a better split
                    best_gain = gain              # Update best gain
                    best_feature = feature_idx    # Update best feature index
                    best_threshold = threshold    # Update best threshold

        return best_feature, best_threshold, best_gain  # Return the best split found

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build the decision tree using CART algorithm.

        This is the core recursive function that constructs the tree top-down.
        At each call, it either creates a leaf node or finds the best split
        and recurses on the two resulting subsets.

        WHY recursive: The tree structure is inherently recursive (each internal
        node has two subtrees). Recursion naturally maps to this structure.

        Args:
            X: Feature matrix for samples reaching this node.
            y: Labels for samples reaching this node.
            depth: Current depth in the tree (0 = root).

        Returns:
            A TreeNode representing this subtree.
        """
        n_samples = len(y)                 # Number of samples at this node
        n_classes_here = len(np.unique(y))  # Number of unique classes at this node

        # Compute impurity at this node (for analysis/logging purposes).
        node_impurity = self._impurity(y)  # Current node's impurity

        # --- Stopping conditions (create a leaf node) ---

        # Condition 1: Maximum depth reached.
        # WHY: Depth limiting is the primary regularization mechanism for trees.
        # Deeper trees have more capacity to memorize training data.
        if self.max_depth is not None and depth >= self.max_depth:  # Depth limit hit
            leaf_value = self._majority_class(y)  # Predict majority class
            return TreeNode(
                value=leaf_value,        # Set leaf prediction
                impurity=node_impurity,  # Record impurity for analysis
                n_samples=n_samples,     # Record sample count
            )

        # Condition 2: Node is pure (all samples belong to one class).
        # WHY: A pure node cannot be improved by further splitting.
        if n_classes_here == 1:  # Only one class present
            return TreeNode(
                value=y[0],              # All labels are the same, pick any
                impurity=0.0,            # Pure node has zero impurity
                n_samples=n_samples,     # Record sample count
            )

        # Condition 3: Too few samples to split.
        # WHY: Splitting tiny nodes captures noise, not signal.
        if n_samples < self.min_samples_split:  # Below minimum split threshold
            leaf_value = self._majority_class(y)  # Predict majority class
            return TreeNode(
                value=leaf_value,        # Set leaf prediction
                impurity=node_impurity,  # Record impurity
                n_samples=n_samples,     # Record sample count
            )

        # --- Find the best split ---
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # If no valid split was found (gain is 0 or no split possible).
        # WHY: This happens when min_samples_leaf prevents all candidate splits,
        # or when no feature can reduce impurity.
        if best_feature is None or best_gain <= 0.0:  # No beneficial split exists
            leaf_value = self._majority_class(y)  # Predict majority class
            return TreeNode(
                value=leaf_value,        # Set leaf prediction
                impurity=node_impurity,  # Record impurity
                n_samples=n_samples,     # Record sample count
            )

        # --- Perform the split and recurse ---

        # Split data into left and right subsets based on the best split.
        left_mask = X[:, best_feature] <= best_threshold  # Boolean mask for left child
        right_mask = ~left_mask                             # Boolean mask for right child

        # Recursively build left subtree with depth incremented.
        # WHY depth+1: Tracks how deep we are for the max_depth stopping criterion.
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)

        # Recursively build right subtree.
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return an internal (decision) node with the split rule and children.
        return TreeNode(
            feature_index=best_feature,   # Which feature to test
            threshold=best_threshold,     # What threshold to compare against
            left=left_child,              # Left subtree (feature <= threshold)
            right=right_child,            # Right subtree (feature > threshold)
            impurity=node_impurity,       # Impurity at this node
            n_samples=n_samples,          # Number of samples at this node
        )

    def _majority_class(self, y: np.ndarray) -> int:
        """Return the most frequent class label in y.

        WHY: When creating a leaf node, we predict the class that appears most
        often among the training samples that reached this leaf. This is the
        maximum likelihood estimate under a uniform prior.

        Args:
            y: Array of class labels.

        Returns:
            The most common class label (integer).
        """
        counts = np.bincount(y, minlength=self.n_classes)  # Count each class
        return int(np.argmax(counts))  # Return the class with highest count

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeFromScratch":
        """Build the decision tree from training data.

        WHY fit/predict API: Follows the sklearn convention, making our
        from-scratch implementation a drop-in replacement for comparisons.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self (for method chaining, following sklearn convention).
        """
        # Store dataset properties for use during tree construction.
        self.n_classes = len(np.unique(y))  # Number of unique classes
        self.n_features = X.shape[1]         # Number of feature dimensions

        # Ensure labels are integer type for np.bincount compatibility.
        # WHY: np.bincount requires non-negative integers. Converting ensures
        # compatibility regardless of the input label format.
        y = y.astype(int)  # Cast to integer type

        # Build the tree recursively starting from the root.
        # WHY: The root considers all training data; recursion handles partitioning.
        self.root = self._build_tree(X, y, depth=0)  # Construct the full tree

        return self  # Return self for method chaining

    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> int:
        """Predict the class for a single sample by traversing the tree.

        Starting from the given node (typically root), follow left or right
        based on the split rule until reaching a leaf node.

        WHY recursive: Tree traversal is naturally recursive. Each call moves
        one level deeper until a leaf is reached.

        Args:
            x: A single sample feature vector of shape (n_features,).
            node: Current node in the tree (starts at root).

        Returns:
            Predicted class label (integer).
        """
        # Base case: reached a leaf node.
        if node.is_leaf():  # Check if this is a leaf
            return node.value  # Return the leaf's predicted class

        # Decision: compare sample's feature value to node's threshold.
        # WHY <=: Matches the convention used during tree construction.
        if x[node.feature_index] <= node.threshold:  # Sample goes left
            return self._predict_sample(x, node.left)  # Recurse into left subtree
        else:  # Sample goes right
            return self._predict_sample(x, node.right)  # Recurse into right subtree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for all samples in X.

        WHY loop over samples: Each sample follows its own path through the tree,
        so prediction cannot be easily vectorized (different samples visit
        different nodes).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted class labels of shape (n_samples,).
        """
        # Apply _predict_sample to each row of X.
        # WHY list comprehension: Clean, Pythonic way to apply a function to each row.
        predictions = np.array([
            self._predict_sample(x, self.root)  # Traverse tree for each sample
            for x in X  # Iterate over each sample (row)
        ])

        return predictions  # Return all predictions as a numpy array

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for all samples in X.

        WHY: Many metrics (like AUC-ROC) require probability estimates, not just
        hard class predictions. For a basic decision tree, the "probability" is
        the class distribution at the leaf node.

        Note: This is a simplified version that returns hard 0/1 probabilities.
        A full implementation would store class distributions at each leaf.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        predictions = self.predict(X)  # Get hard predictions first
        n_samples = len(predictions)    # Number of samples

        # Create a one-hot probability matrix (hard probabilities).
        # WHY one-hot: Without storing leaf class distributions, the best we can
        # do is assign probability 1.0 to the predicted class and 0.0 to others.
        proba = np.zeros((n_samples, self.n_classes))  # Initialize with zeros
        for i, pred in enumerate(predictions):  # Set probability=1 for predicted class
            proba[i, pred] = 1.0  # One-hot encode the prediction

        return proba  # Return probability matrix


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,      # Total number of samples to generate
    n_features: int = 20,       # Total number of features
    n_classes: int = 2,         # Number of target classes (binary by default)
    random_state: int = 42,     # Random seed for reproducibility
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data and split into train/val/test.

    WHY synthetic data: Provides full control over problem difficulty, dimensionality,
    and class balance, making it ideal for algorithm benchmarking.

    Split ratios: 60% train / 20% validation / 20% test.
    WHY: Industry standard split. Train for learning, validation for tuning
    hyperparameters (used repeatedly), test for final evaluation (used once).

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) as numpy arrays.
    """
    # Generate synthetic dataset with controllable properties.
    # WHY make_classification: Creates clusters with known signal-to-noise ratio.
    X, y = make_classification(
        n_samples=n_samples,            # Total data points
        n_features=n_features,          # Feature dimensionality
        n_informative=n_features // 2,  # Half the features carry real signal
        n_redundant=n_features // 4,    # Quarter are linear combos of informative features
        n_classes=n_classes,            # Number of classes
        random_state=random_state,      # Seed for reproducibility
    )

    # First split: 60% train, 40% temp.
    # WHY stratify=y: Ensures each split preserves the class distribution.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,                           # Features and labels
        test_size=0.4,                  # 40% held out for val + test
        random_state=random_state,      # Reproducible split
        stratify=y,                     # Maintain class proportions
    )

    # Second split: 50% of temp (20% of total) for validation, 50% for test.
    # WHY separate val/test: Validation is used repeatedly during HPO,
    # test is used exactly once for final unbiased evaluation.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,                 # Split the held-out data
        test_size=0.5,                  # Equal split
        random_state=random_state,      # Reproducible split
        stratify=y_temp,                # Maintain class proportions
    )

    # Log dataset sizes for verification.
    logger.info(  # Report the split sizes
        f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test  # Return all splits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,           # Training features
    y_train: np.ndarray,           # Training labels
    max_depth: Optional[int] = 5,  # Maximum tree depth
    min_samples_split: int = 2,    # Minimum samples to attempt split
    min_samples_leaf: int = 1,     # Minimum samples per leaf
    criterion: str = "gini",       # Impurity criterion
) -> DecisionTreeFromScratch:
    """Train a from-scratch Decision Tree on the training data.

    WHY a separate train function: Separates the model creation logic from
    data generation and evaluation, following single-responsibility principle.

    Args:
        X_train: Training feature matrix.
        y_train: Training label array.
        max_depth: Tree depth limit for regularization.
        min_samples_split: Minimum samples to attempt split.
        min_samples_leaf: Minimum samples in each leaf.
        criterion: Impurity criterion ("gini" or "entropy").

    Returns:
        Trained DecisionTreeFromScratch model.
    """
    logger.info(  # Log the training configuration
        f"Training Decision Tree (from scratch): depth={max_depth}, "
        f"min_split={min_samples_split}, min_leaf={min_samples_leaf}, "
        f"criterion={criterion}"
    )

    # Instantiate the model with the specified hyperparameters.
    model = DecisionTreeFromScratch(
        max_depth=max_depth,                # Set depth limit
        min_samples_split=min_samples_split,  # Set min split threshold
        min_samples_leaf=min_samples_leaf,    # Set min leaf size
        criterion=criterion,                  # Set impurity criterion
    )

    # Fit the model to the training data.
    # WHY: This triggers the recursive tree construction process.
    model.fit(X_train, y_train)  # Build the tree

    # Compute and log training accuracy as a sanity check.
    # WHY: Training accuracy should be high (near 100% for deep trees).
    # If it's low, something is wrong with the implementation.
    train_preds = model.predict(X_train)  # Predict on training data
    train_acc = accuracy_score(y_train, train_preds)  # Compute accuracy
    logger.info(f"Training accuracy: {train_acc:.4f}")  # Log result

    return model  # Return the trained model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: DecisionTreeFromScratch,  # Trained model to evaluate
    X_val: np.ndarray,                # Validation features
    y_val: np.ndarray,                # Validation labels
) -> Dict[str, float]:
    """Evaluate the model on the validation set with comprehensive metrics.

    WHY validation: Used during hyperparameter tuning to estimate how well
    the model will perform on unseen data. Unlike the test set, the validation
    set can be used repeatedly without biasing the final evaluation.

    Args:
        model: Trained DecisionTreeFromScratch model.
        X_val: Validation feature matrix.
        y_val: Validation label array.

    Returns:
        Dictionary of metric names to values.
    """
    # Generate predictions on the validation set.
    y_pred = model.predict(X_val)  # Hard class predictions

    # Generate probability predictions for AUC-ROC.
    # WHY: AUC-ROC requires probability scores, not just class labels.
    y_proba = model.predict_proba(X_val)  # Probability matrix

    # Compute comprehensive classification metrics.
    # WHY multiple metrics: Each captures a different aspect of performance.
    metrics = {
        # Accuracy: fraction of correct predictions. Simple but misleading for imbalanced data.
        "accuracy": accuracy_score(y_val, y_pred),
        # Precision: of positive predictions, fraction correct. Important when FP cost is high.
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        # Recall: of actual positives, fraction found. Important when FN cost is high.
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        # F1: harmonic mean of precision and recall. Balanced metric.
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        # AUC-ROC: threshold-independent measure of class separability.
        "auc_roc": roc_auc_score(y_val, y_proba[:, 1]) if model.n_classes == 2 else 0.0,
    }

    # Log all metrics for monitoring.
    logger.info("Validation Metrics:")  # Header
    for name, value in metrics.items():  # Log each metric
        logger.info(f"  {name}: {value:.4f}")  # Format to 4 decimal places

    return metrics  # Return the metrics dictionary


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test(
    model: DecisionTreeFromScratch,  # Trained model to evaluate
    X_test: np.ndarray,               # Test features
    y_test: np.ndarray,               # Test labels
) -> Dict[str, float]:
    """Final evaluation on the held-out test set.

    WHY separate test function: The test set provides an unbiased estimate
    of generalization performance. It is used exactly ONCE after all
    hyperparameter tuning is complete.

    Args:
        model: Trained DecisionTreeFromScratch model.
        X_test: Test feature matrix.
        y_test: Test label array.

    Returns:
        Dictionary of metric names to values.
    """
    # Generate predictions on the test set.
    y_pred = model.predict(X_test)  # Hard class predictions
    y_proba = model.predict_proba(X_test)  # Probability predictions

    # Compute comprehensive classification metrics (same as validation).
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),  # Overall correctness
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),  # Positive predictive value
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),  # Sensitivity
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),  # Balanced metric
        "auc_roc": roc_auc_score(y_test, y_proba[:, 1]) if model.n_classes == 2 else 0.0,  # Ranking metric
    }

    # Log test results with emphasis (this is the final evaluation).
    logger.info("=" * 50)  # Visual separator for importance
    logger.info("TEST SET RESULTS (Final Evaluation):")  # Header
    for name, value in metrics.items():  # Log each metric
        logger.info(f"  {name}: {value:.4f}")  # Format to 4 decimal places
    logger.info("=" * 50)  # Visual separator

    # Print detailed classification report for per-class analysis.
    # WHY: Shows precision, recall, F1 for each class individually.
    logger.info(f"\n{classification_report(y_test, y_pred)}")  # Detailed report

    return metrics  # Return the metrics dictionary


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial) -> float:
    """Optuna objective function for Bayesian hyperparameter optimization.

    WHY Optuna: Uses Tree-structured Parzen Estimator (TPE) to intelligently
    explore the hyperparameter space. Unlike grid search, Optuna learns from
    previous trials to focus on promising regions.

    The function generates data, trains a model with suggested hyperparameters,
    evaluates on validation set, and returns the metric to optimize.

    Args:
        trial: Optuna Trial object that suggests hyperparameter values.

    Returns:
        Validation F1 score (the objective to maximize).
    """
    # Generate fresh data for each trial to avoid data leakage.
    # WHY: Using the same split across trials is fine for tuning, but fresh
    # generation with the same seed ensures consistency.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()  # Get data splits

    # Suggest hyperparameters from defined search spaces.
    # WHY suggest_int/suggest_categorical: Optuna's TPE sampler uses these
    # to build probability models of good vs bad parameter regions.

    # max_depth: controls tree complexity. Range 2-15 covers shallow to deep trees.
    max_depth = trial.suggest_int("max_depth", 2, 15)  # Tree depth limit

    # min_samples_split: minimum samples to attempt a split. Range 2-20.
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)  # Split threshold

    # min_samples_leaf: minimum samples per leaf. Range 1-20.
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)  # Leaf size minimum

    # criterion: impurity measure choice.
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])  # Criterion

    # Train model with suggested hyperparameters.
    model = train(
        X_train, y_train,
        max_depth=max_depth,                # Suggested depth
        min_samples_split=min_samples_split,  # Suggested min split
        min_samples_leaf=min_samples_leaf,    # Suggested min leaf
        criterion=criterion,                  # Suggested criterion
    )

    # Evaluate on validation set and return F1 score.
    # WHY F1: Balances precision and recall; more robust than accuracy for
    # potentially imbalanced datasets.
    metrics = validate(model, X_val, y_val)  # Compute validation metrics

    return metrics["f1"]  # Return F1 as the objective to maximize


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune hyperparameter search space configuration.

    WHY Ray Tune: Provides distributed hyperparameter search with advanced
    schedulers (ASHA, PBT) and supports running trials in parallel across
    multiple GPUs/machines. Complementary to Optuna.

    Returns:
        Dictionary defining the Ray Tune search space.
    """
    # Define the search space for Ray Tune.
    # WHY dictionary format: Ray Tune uses a config dict where each key maps
    # to a distribution from ray.tune that defines the search space.
    search_space = {
        # max_depth: integer range from 2 to 15.
        # WHY this range: depth < 2 underfits, depth > 15 likely overfits.
        "max_depth": {"type": "randint", "lower": 2, "upper": 16},

        # min_samples_split: integer range from 2 to 20.
        # WHY: Below 2 is invalid, above 20 is too restrictive for most datasets.
        "min_samples_split": {"type": "randint", "lower": 2, "upper": 21},

        # min_samples_leaf: integer range from 1 to 20.
        # WHY: 1 allows full flexibility, 20 forces broad generalizations.
        "min_samples_leaf": {"type": "randint", "lower": 1, "upper": 21},

        # criterion: categorical choice.
        # WHY: Both Gini and entropy are valid; we want to compare them.
        "criterion": {"type": "choice", "values": ["gini", "entropy"]},
    }

    logger.info("Ray Tune search space defined:")  # Log the search space
    for param, config in search_space.items():  # Log each parameter
        logger.info(f"  {param}: {config}")  # Show the configuration

    return search_space  # Return the search space configuration


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,  # Training features
    y_train: np.ndarray,  # Training labels
    X_val: np.ndarray,    # Validation features
    y_val: np.ndarray,    # Validation labels
) -> Dict[str, Dict[str, float]]:
    """Compare multiple hyperparameter configurations with reasoning.

    WHY: Systematic comparison reveals how each hyperparameter affects model
    behavior. This is more educational than blind optimization because it
    builds intuition about the algorithm.

    Configurations tested:
    1. Shallow tree (max_depth=3, gini): Underfitting baseline, high bias, low variance.
       WHY: Shows what happens when the model is too simple to capture patterns.

    2. Medium tree (max_depth=10, gini): Balanced complexity.
       WHY: Often the sweet spot between underfitting and overfitting.

    3. Deep tree (max_depth=None, gini): No depth limit, potential overfitting.
       WHY: Shows the overfitting behavior of unrestricted trees.

    4. Entropy criterion (max_depth=10, entropy): Compare criterion effect.
       WHY: Tests whether entropy's theoretical advantages translate to practice.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Dictionary mapping config names to their validation metrics.
    """
    # Define parameter configurations with reasoning for each choice.
    configs = {
        "shallow_gini_depth3": {
            "params": {
                "max_depth": 3,           # Very shallow tree
                "min_samples_split": 2,   # Default
                "min_samples_leaf": 1,    # Default
                "criterion": "gini",      # Standard criterion
            },
            "reasoning": (
                "Shallow tree (depth=3) with Gini criterion. Expected to UNDERFIT: "
                "only 3 levels of splits cannot capture complex decision boundaries. "
                "High bias, low variance. Good as a baseline for comparison."
            ),
        },
        "medium_gini_depth10": {
            "params": {
                "max_depth": 10,          # Moderate depth
                "min_samples_split": 5,   # Moderate regularization
                "min_samples_leaf": 2,    # Prevent single-sample leaves
                "criterion": "gini",      # Standard criterion
            },
            "reasoning": (
                "Medium tree (depth=10) with moderate regularization. Expected to "
                "BALANCE bias-variance: enough depth for complex patterns but not "
                "so deep as to memorize noise. Often the practical sweet spot."
            ),
        },
        "deep_gini_unlimited": {
            "params": {
                "max_depth": None,        # No depth limit
                "min_samples_split": 2,   # Minimal regularization
                "min_samples_leaf": 1,    # Allow single-sample leaves
                "criterion": "gini",      # Standard criterion
            },
            "reasoning": (
                "Unlimited depth tree with minimal regularization. Expected to "
                "OVERFIT: will grow until every leaf is pure, memorizing training "
                "data including noise. High training accuracy, lower test accuracy."
            ),
        },
        "medium_entropy_depth10": {
            "params": {
                "max_depth": 10,          # Same depth as medium_gini
                "min_samples_split": 5,   # Same regularization
                "min_samples_leaf": 2,    # Same leaf constraint
                "criterion": "entropy",   # Alternative criterion
            },
            "reasoning": (
                "Same as medium_gini but with entropy criterion. Expected to produce "
                "SIMILAR results to Gini: in practice, the criterion rarely makes a "
                "significant difference. Entropy may create slightly more balanced trees."
            ),
        },
    }

    results = {}  # Store results for each configuration

    # Train and evaluate each configuration.
    for name, config in configs.items():  # Iterate over configs
        logger.info(f"\n{'='*60}")  # Visual separator
        logger.info(f"Config: {name}")  # Config name
        logger.info(f"Reasoning: {config['reasoning']}")  # Why this config
        logger.info(f"Parameters: {config['params']}")  # Parameter values

        # Train model with this configuration.
        model = train(X_train, y_train, **config["params"])  # Train with config params

        # Evaluate on validation set.
        metrics = validate(model, X_val, y_val)  # Compute metrics

        results[name] = metrics  # Store results

    # Log comparison summary.
    logger.info(f"\n{'='*60}")  # Visual separator
    logger.info("COMPARISON SUMMARY:")  # Header
    for name, metrics in results.items():  # Log each config's key metric
        logger.info(f"  {name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    return results  # Return all results for further analysis


# ---------------------------------------------------------------------------
# Real-World Demo
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate the from-scratch Decision Tree on a medical diagnosis task.

    Domain: Cardiovascular disease risk assessment.
    Features: age, systolic_blood_pressure, cholesterol_level, bmi
    Target: heart_disease (0 = no, 1 = yes)

    WHY this domain: Medical diagnosis is a classic decision tree use case because:
    1. Interpretability is critical (doctors need to understand and trust the model)
    2. Feature interactions matter (e.g., age + high cholesterol is riskier)
    3. Both false positives and false negatives have real consequences
    4. Decision trees naturally produce clinical decision rules

    The synthetic data is generated with realistic feature ranges and known
    relationships to simulate a plausible medical scenario.
    """
    logger.info("\n" + "=" * 60)  # Visual separator
    logger.info("REAL-WORLD DEMO: Medical Diagnosis (Heart Disease)")  # Header
    logger.info("=" * 60)  # Visual separator

    np.random.seed(42)  # Set seed for reproducibility
    n_samples = 500     # Number of synthetic patients

    # Generate realistic medical features with domain-appropriate ranges.
    # WHY these ranges: Based on typical clinical reference ranges.
    age = np.random.normal(55, 15, n_samples).clip(20, 90)  # Age: mean=55, std=15, range [20,90]
    systolic_bp = np.random.normal(130, 20, n_samples).clip(90, 200)  # SBP: mean=130, std=20
    cholesterol = np.random.normal(220, 40, n_samples).clip(120, 350)  # Cholesterol: mean=220, std=40
    bmi = np.random.normal(27, 5, n_samples).clip(15, 45)  # BMI: mean=27, std=5

    # Stack features into a matrix with named columns.
    # WHY: Combines individual arrays into the standard (n_samples, n_features) format.
    X = np.column_stack([age, systolic_bp, cholesterol, bmi])  # Shape: (500, 4)
    feature_names = ["age", "systolic_blood_pressure", "cholesterol_level", "bmi"]  # Feature names

    # Create realistic target variable with domain-appropriate risk factors.
    # WHY this formula: Higher age, blood pressure, cholesterol, and BMI all
    # increase cardiovascular risk. The logistic function maps the risk score
    # to a probability, and we add noise for realism.
    risk_score = (
        0.03 * (age - 50)           # Age over 50 increases risk
        + 0.02 * (systolic_bp - 120)  # BP over 120 increases risk
        + 0.01 * (cholesterol - 200)  # Cholesterol over 200 increases risk
        + 0.05 * (bmi - 25)          # BMI over 25 increases risk
    )

    # Convert risk score to probability using logistic (sigmoid) function.
    # WHY sigmoid: Maps continuous risk scores to [0, 1] probability range.
    probability = 1.0 / (1.0 + np.exp(-risk_score))  # Sigmoid function

    # Generate binary labels from probabilities with added noise.
    # WHY noise: Real-world diagnoses have inherent uncertainty; perfect
    # separation would be unrealistic.
    noise = np.random.normal(0, 0.1, n_samples)  # Small Gaussian noise
    y = (probability + noise > 0.5).astype(int)  # Threshold at 0.5

    # Split into train/val/test sets.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y  # 60/40 split
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp  # Split remaining
    )

    # Log feature statistics.
    logger.info("\nFeature Statistics (Training Set):")  # Header
    for i, name in enumerate(feature_names):  # Log each feature's stats
        logger.info(
            f"  {name}: mean={X_train[:, i].mean():.1f}, "  # Mean value
            f"std={X_train[:, i].std():.1f}, "                # Standard deviation
            f"range=[{X_train[:, i].min():.1f}, {X_train[:, i].max():.1f}]"  # Min-max range
        )

    # Log class distribution.
    logger.info(f"\nClass distribution: {np.bincount(y_train)}")  # Count of each class
    logger.info(f"  No heart disease: {(y_train == 0).sum()}")  # Count of class 0
    logger.info(f"  Heart disease: {(y_train == 1).sum()}")  # Count of class 1

    # Train the model with moderate regularization.
    # WHY depth=5: In medical diagnosis, we want interpretable rules that
    # aren't too complex for clinicians to follow.
    model = train(
        X_train, y_train,
        max_depth=5,            # Moderate depth for interpretability
        min_samples_split=10,   # Require sufficient evidence before splitting
        min_samples_leaf=5,     # Each leaf must have at least 5 patients
        criterion="gini",       # Standard criterion
    )

    # Evaluate on validation and test sets.
    logger.info("\nValidation Results:")  # Validation header
    val_metrics = validate(model, X_val, y_val)  # Validate

    logger.info("\nTest Results:")  # Test header
    test_metrics = test(model, X_test, y_test)  # Final evaluation


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete Decision Tree (from scratch) pipeline.

    Pipeline steps:
    1. Generate synthetic data
    2. Train with default parameters
    3. Validate on validation set
    4. Compare multiple parameter configurations
    5. Run Optuna hyperparameter optimization
    6. Train best model on full training data
    7. Final evaluation on test set
    8. Run real-world medical diagnosis demo

    WHY this order: Follows the standard ML workflow of data -> baseline ->
    comparison -> optimization -> final evaluation -> real-world application.
    """
    logger.info("=" * 60)  # Visual separator
    logger.info("Decision Tree Classifier - NumPy From-Scratch Implementation")  # Title
    logger.info("=" * 60)  # Visual separator

    # --- Step 1: Generate Data ---
    logger.info("\n--- Step 1: Generating Data ---")  # Step header
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()  # Create dataset

    # --- Step 2: Train Baseline Model ---
    logger.info("\n--- Step 2: Training Baseline Model ---")  # Step header
    baseline_model = train(X_train, y_train, max_depth=5, criterion="gini")  # Default params

    # --- Step 3: Validate Baseline ---
    logger.info("\n--- Step 3: Validating Baseline ---")  # Step header
    val_metrics = validate(baseline_model, X_val, y_val)  # Evaluate on val set

    # --- Step 4: Compare Parameter Sets ---
    logger.info("\n--- Step 4: Comparing Parameter Sets ---")  # Step header
    comparison_results = compare_parameter_sets(X_train, y_train, X_val, y_val)  # Compare configs

    # --- Step 5: Optuna Hyperparameter Optimization ---
    logger.info("\n--- Step 5: Optuna Hyperparameter Optimization ---")  # Step header
    # Create Optuna study to maximize F1 score.
    # WHY maximize: Higher F1 = better model. Optuna supports both maximize and minimize.
    study = optuna.create_study(direction="maximize")  # Create study
    # Suppress Optuna's verbose logging during optimization.
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna output
    # Run 20 trials of Bayesian optimization.
    # WHY 20 trials: Enough to explore the space, not so many as to be slow
    # for a from-scratch implementation.
    study.optimize(optuna_objective, n_trials=20)  # Run optimization

    # Log best hyperparameters found.
    logger.info(f"Best trial F1: {study.best_trial.value:.4f}")  # Best F1 score
    logger.info(f"Best params: {study.best_trial.params}")  # Best parameters

    # --- Step 6: Define Ray Tune Search Space ---
    logger.info("\n--- Step 6: Ray Tune Search Space ---")  # Step header
    ray_config = ray_tune_search()  # Define search space (not executed, just defined)

    # --- Step 7: Train Best Model ---
    logger.info("\n--- Step 7: Training Best Model ---")  # Step header
    best_params = study.best_trial.params  # Extract best parameters from Optuna
    best_model = train(X_train, y_train, **best_params)  # Train with best params

    # --- Step 8: Final Test Evaluation ---
    logger.info("\n--- Step 8: Final Test Evaluation ---")  # Step header
    test_metrics = test(best_model, X_test, y_test)  # Final unbiased evaluation

    # --- Step 9: Real-World Demo ---
    logger.info("\n--- Step 9: Real-World Demo ---")  # Step header
    real_world_demo()  # Run medical diagnosis demo

    logger.info("\n--- Pipeline Complete ---")  # Completion message


# Entry point: run the main pipeline when this script is executed directly.
# WHY if __name__: Allows this module to be imported without running the pipeline,
# enabling reuse of the DecisionTreeFromScratch class in other scripts.
if __name__ == "__main__":
    main()  # Execute the full pipeline
