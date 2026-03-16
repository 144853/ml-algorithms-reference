"""
AdaBoost (SAMME Algorithm) - NumPy From-Scratch Implementation
================================================================

Theory & Mathematics:
    AdaBoost (Adaptive Boosting) is an ensemble meta-algorithm that combines
    multiple weak learners (classifiers slightly better than random guessing)
    into a strong classifier. The SAMME (Stagewise Additive Modeling using a
    Multi-class Exponential loss function) variant generalizes the original
    AdaBoost to multiclass classification, though this implementation focuses
    on binary classification.

    Core Idea:
        Instead of fitting a single complex model, AdaBoost sequentially trains
        simple models (decision stumps), where each subsequent model focuses on
        the samples that previous models misclassified. This is achieved by
        maintaining a distribution (weights) over the training samples and
        increasing the weight of misclassified samples after each round.

    SAMME Algorithm for Binary Classification:
        1. Initialize sample weights uniformly: w_i = 1/N for i = 1,...,N
        2. For t = 1, ..., T (number of boosting rounds):
            a. Train weak learner h_t on weighted training data
            b. Compute weighted error:
               epsilon_t = sum(w_i * I(y_i != h_t(x_i))) / sum(w_i)
            c. Compute learner weight (importance):
               alpha_t = 0.5 * ln((1 - epsilon_t) / epsilon_t)
            d. Update sample weights:
               w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
            e. Normalize: w_i = w_i / sum(w_j)
        3. Final prediction: H(x) = sign(sum_{t=1}^{T} alpha_t * h_t(x))

    Decision Stump (Weak Learner):
        A decision stump is a depth-1 decision tree (single split). It finds
        the best feature j* and threshold t* that minimizes the weighted
        classification error:
            j*, t* = argmin_{j,t} sum(w_i * I(y_i != stump(x_i; j, t)))

        The stump can also be generalized to depth-2 trees for slightly
        stronger weak learners while maintaining the boosting property.

    Convergence:
        The training error decreases exponentially:
            L_train <= exp(-2 * sum_{t=1}^{T} gamma_t^2)
        where gamma_t = 0.5 - epsilon_t is the edge of learner t.

    Relationship to Exponential Loss:
        AdaBoost minimizes the exponential loss function:
            L(y, F(x)) = exp(-y * F(x))
        where F(x) = sum(alpha_t * h_t(x)). This makes it sensitive to
        outliers and noisy labels, as misclassified samples receive
        exponentially increasing weights.

Business Use Cases:
    - Customer churn prediction (focus on hard-to-classify customers)
    - Fraud detection (iteratively focus on misclassified transactions)
    - Face detection (Viola-Jones algorithm uses AdaBoost with Haar features)
    - Medical diagnosis (combine multiple simple diagnostic rules)
    - Credit scoring (ensemble of simple threshold rules)

Advantages:
    - Simple and easy to implement
    - No hyperparameters to tune beyond number of rounds (T)
    - Resistant to overfitting (in practice, with sufficient data)
    - Automatically identifies difficult samples
    - Feature importance via weighted feature usage across stumps
    - Works well with decision stumps (very fast weak learners)

Disadvantages:
    - Sensitive to noisy data and outliers (exponential loss amplifies them)
    - Sequential training: cannot easily parallelize boosting rounds
    - Weak learners must be better than random (epsilon < 0.5)
    - Can overfit with very noisy labels
    - Less effective than gradient boosting on complex datasets
    - Decision stumps may be too weak for high-dimensional problems

Hyperparameters:
    - n_estimators: Number of boosting rounds (T)
    - max_depth: Depth of the weak learner tree (1 = stump)
    - random_state: Seed for reproducibility
"""

import logging  # Standard logging for tracking training progress
import warnings  # Suppress non-critical warnings for cleaner output
from typing import Any, Dict, List, Optional, Tuple  # Type hints for clarity

import numpy as np  # Core numerical library for all array operations
import optuna  # Bayesian hyperparameter optimization framework
from sklearn.datasets import make_classification  # Synthetic dataset generation
from sklearn.metrics import (  # Evaluation metrics for classification
    accuracy_score,  # Fraction of correct predictions
    classification_report,  # Detailed per-class metrics report
    confusion_matrix,  # True/false positive/negative counts
    f1_score,  # Harmonic mean of precision and recall
    precision_score,  # Fraction of true positives among predicted positives
    recall_score,  # Fraction of true positives among actual positives
    roc_auc_score,  # Area under the ROC curve
)
from sklearn.model_selection import train_test_split  # Split data into train/val/test
from sklearn.preprocessing import StandardScaler  # Zero-mean unit-variance normalization

# Configure logging to show timestamps and severity levels for debugging
logging.basicConfig(
    level=logging.INFO,  # Show INFO level and above (INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + level + message
)
logger = logging.getLogger(__name__)  # Module-level logger for this file
warnings.filterwarnings("ignore")  # Suppress sklearn convergence warnings


# ---------------------------------------------------------------------------
# Decision Stump (Weak Learner for AdaBoost)
# ---------------------------------------------------------------------------

class DecisionStump:
    """
    A decision stump is a depth-1 decision tree that makes a single split
    on one feature. It finds the feature and threshold that minimizes the
    weighted classification error, making it the ideal weak learner for
    AdaBoost.

    For a depth-2 variant, we recursively build a small tree. This class
    supports configurable max_depth for flexibility.
    """

    def __init__(self, max_depth: int = 1) -> None:
        """Initialize the stump with configurable depth."""
        self.max_depth = max_depth  # 1 = stump, 2 = small tree
        self.feature_idx: Optional[int] = None  # Index of the feature to split on
        self.threshold: Optional[float] = None  # Threshold value for the split
        self.polarity: int = 1  # Direction: 1 if left<=threshold is class -1
        self.left_value: float = -1.0  # Prediction for samples going left
        self.right_value: float = 1.0  # Prediction for samples going right
        self.left_child: Optional["DecisionStump"] = None  # Left subtree for depth > 1
        self.right_child: Optional["DecisionStump"] = None  # Right subtree for depth > 1

    def _weighted_gini(
        self,
        y: np.ndarray,  # Labels for samples in this subset
        weights: np.ndarray,  # Per-sample importance weights
    ) -> float:
        """
        Compute weighted Gini impurity for a set of samples.
        Gini = 1 - sum(p_k^2) where p_k is the weighted proportion of class k.
        Lower Gini means purer node (better split).
        """
        if len(y) == 0:  # Empty node has zero impurity (trivially pure)
            return 0.0
        total_weight = np.sum(weights)  # Sum of all sample weights in this node
        if total_weight == 0:  # Avoid division by zero if all weights are zero
            return 0.0
        classes = np.unique(y)  # Get the distinct class labels present
        gini = 1.0  # Start with maximum impurity (1.0)
        for c in classes:  # Iterate over each class
            # Proportion of weight belonging to class c
            p_c = np.sum(weights[y == c]) / total_weight  # Weighted class frequency
            gini -= p_c ** 2  # Subtract squared proportion (pure node -> gini = 0)
        return gini  # Return the weighted Gini impurity value

    def _find_best_split(
        self,
        X: np.ndarray,  # Feature matrix (n_samples, n_features)
        y: np.ndarray,  # Labels encoded as {-1, +1} for AdaBoost
        weights: np.ndarray,  # Sample weights from AdaBoost distribution
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best feature and threshold that minimizes weighted Gini.
        Iterates over all features and all midpoints between sorted values.
        Returns (best_feature_index, best_threshold, best_weighted_error).
        """
        n_samples, n_features = X.shape  # Get dimensions of the data
        best_error = np.inf  # Initialize with worst possible error
        best_feature = None  # Track which feature gives the best split
        best_threshold = None  # Track which threshold gives the best split

        for feat_idx in range(n_features):  # Try every feature as a split candidate
            # Sort samples by this feature's values for efficient threshold search
            sorted_indices = np.argsort(X[:, feat_idx])  # Indices that sort the feature
            sorted_feat = X[sorted_indices, feat_idx]  # Sorted feature values
            sorted_y = y[sorted_indices]  # Corresponding labels in sorted order
            sorted_w = weights[sorted_indices]  # Corresponding weights in sorted order

            for i in range(1, n_samples):  # Try every possible split point
                if sorted_feat[i] == sorted_feat[i - 1]:  # Skip duplicate values
                    continue  # No information gain from splitting between equal values

                # Midpoint between consecutive sorted values as the threshold
                threshold = (sorted_feat[i - 1] + sorted_feat[i]) / 2.0

                # Split samples into left (<=threshold) and right (>threshold)
                left_mask = X[:, feat_idx] <= threshold  # Boolean mask for left child
                right_mask = ~left_mask  # Boolean mask for right child (complement)

                # Skip if either side has no samples (degenerate split)
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # Cannot compute Gini for an empty partition

                # Weighted Gini for left and right children
                gini_left = self._weighted_gini(y[left_mask], weights[left_mask])
                gini_right = self._weighted_gini(y[right_mask], weights[right_mask])

                # Total weighted Gini is the weighted average of children's Gini
                w_left = np.sum(weights[left_mask])  # Total weight in left child
                w_right = np.sum(weights[right_mask])  # Total weight in right child
                w_total = w_left + w_right  # Total weight (should equal sum of all weights)
                weighted_gini = (w_left / w_total) * gini_left + (w_right / w_total) * gini_right

                if weighted_gini < best_error:  # Found a better split
                    best_error = weighted_gini  # Update the best error so far
                    best_feature = feat_idx  # Record which feature was best
                    best_threshold = threshold  # Record the optimal threshold

        return best_feature, best_threshold, best_error  # Return the optimal split info

    def fit(
        self,
        X: np.ndarray,  # Training features (n_samples, n_features)
        y: np.ndarray,  # Training labels in {-1, +1}
        weights: np.ndarray,  # AdaBoost sample weight distribution
        depth: int = 0,  # Current recursion depth (0 at root)
    ) -> "DecisionStump":
        """
        Fit the decision stump (or small tree) to the weighted training data.
        Uses recursive splitting up to max_depth. Each leaf stores the
        weighted majority class prediction.
        """
        # Base case: if we've reached max depth or all labels are the same
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            # Compute the weighted majority vote for this leaf
            # For binary {-1, +1}, sum of w_i * y_i tells which class dominates
            weighted_sum = np.sum(weights * y)  # Positive means +1 dominates
            self.left_value = np.sign(weighted_sum) if weighted_sum != 0 else 1.0
            self.right_value = self.left_value  # Leaf node: same prediction both sides
            return self  # Return self to allow method chaining

        # Find the best split using weighted Gini impurity
        feat_idx, threshold, _ = self._find_best_split(X, y, weights)

        if feat_idx is None:  # No valid split found (all features identical)
            weighted_sum = np.sum(weights * y)  # Fall back to weighted majority
            self.left_value = np.sign(weighted_sum) if weighted_sum != 0 else 1.0
            self.right_value = self.left_value  # Make this a leaf node
            return self  # Return early since we can't split further

        self.feature_idx = feat_idx  # Store the chosen feature index
        self.threshold = threshold  # Store the chosen threshold

        # Partition samples into left and right based on the split
        left_mask = X[:, feat_idx] <= threshold  # Samples going to the left child
        right_mask = ~left_mask  # Samples going to the right child

        if depth + 1 < self.max_depth:  # More splitting allowed, create subtrees
            # Recursively build left child tree on the left partition
            self.left_child = DecisionStump(max_depth=self.max_depth)
            self.left_child.fit(X[left_mask], y[left_mask], weights[left_mask], depth + 1)
            # Recursively build right child tree on the right partition
            self.right_child = DecisionStump(max_depth=self.max_depth)
            self.right_child.fit(X[right_mask], y[right_mask], weights[right_mask], depth + 1)
        else:
            # At max depth, compute weighted majority for left leaf
            w_left = np.sum(weights[left_mask] * y[left_mask])  # Weighted class sum
            self.left_value = np.sign(w_left) if w_left != 0 else 1.0  # Majority class

            # Compute weighted majority for right leaf
            w_right = np.sum(weights[right_mask] * y[right_mask])  # Weighted class sum
            self.right_value = np.sign(w_right) if w_right != 0 else 1.0  # Majority class

        return self  # Return the fitted stump/tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels {-1, +1} for each sample in X.
        Routes each sample through the tree to reach a leaf prediction.
        """
        predictions = np.zeros(X.shape[0])  # Initialize output array
        for i in range(X.shape[0]):  # Predict each sample individually
            predictions[i] = self._predict_single(X[i])  # Route through tree
        return predictions  # Return all predictions as an array

    def _predict_single(self, x: np.ndarray) -> float:
        """Route a single sample through the tree to get a prediction."""
        if self.feature_idx is None:  # This is a pure leaf node (no split)
            return self.left_value  # Return the leaf's majority class

        if x[self.feature_idx] <= self.threshold:  # Sample goes to left child
            if self.left_child is not None:  # Left subtree exists
                return self.left_child._predict_single(x)  # Recurse into left child
            return self.left_value  # Return left leaf prediction
        else:  # Sample goes to right child
            if self.right_child is not None:  # Right subtree exists
                return self.right_child._predict_single(x)  # Recurse into right child
            return self.right_value  # Return right leaf prediction


# ---------------------------------------------------------------------------
# AdaBoost Classifier (SAMME Algorithm)
# ---------------------------------------------------------------------------

class AdaBoostClassifier:
    """
    AdaBoost classifier using the SAMME algorithm for binary classification.
    Combines T weak learners (decision stumps) with learned importance weights
    (alpha_t) to form a strong ensemble classifier.
    """

    def __init__(
        self,
        n_estimators: int = 50,  # Number of boosting rounds T
        max_depth: int = 1,  # Depth of each weak learner (1 = stump)
        random_state: int = 42,  # Random seed for reproducibility
    ) -> None:
        """Store hyperparameters and initialize empty model containers."""
        self.n_estimators = n_estimators  # How many weak learners to train
        self.max_depth = max_depth  # Complexity of each weak learner
        self.random_state = random_state  # Reproducibility seed
        self.alphas: List[float] = []  # Learner importance weights (one per round)
        self.stumps: List[DecisionStump] = []  # Trained weak learners (one per round)

    def fit(
        self,
        X: np.ndarray,  # Training feature matrix (n_samples, n_features)
        y: np.ndarray,  # Training labels in {0, 1} (will be converted to {-1, +1})
    ) -> "AdaBoostClassifier":
        """
        Train the AdaBoost ensemble using the SAMME algorithm.
        Sequentially trains weak learners, each focusing on previously
        misclassified samples through sample weight updates.
        """
        n_samples = X.shape[0]  # Total number of training examples

        # Convert labels from {0, 1} to {-1, +1} for AdaBoost's sign-based math
        y_coded = np.where(y == 0, -1, 1)  # Map 0 -> -1, 1 -> +1

        # Step 1: Initialize all sample weights uniformly to 1/N
        # This means every sample is equally important at the start
        weights = np.full(n_samples, 1.0 / n_samples)  # Uniform distribution over samples

        self.alphas = []  # Reset learner weights from any previous fit
        self.stumps = []  # Reset weak learners from any previous fit

        for t in range(self.n_estimators):  # Iterate over boosting rounds
            # Step 2a: Train a weak learner on the current weighted distribution
            stump = DecisionStump(max_depth=self.max_depth)  # Create a new stump
            stump.fit(X, y_coded, weights)  # Fit the stump to weighted data

            # Step 2b: Get predictions from the current weak learner
            predictions = stump.predict(X)  # Predict {-1, +1} for all training samples

            # Step 2c: Compute the weighted classification error rate
            # epsilon = sum of weights for misclassified samples
            misclassified = (predictions != y_coded).astype(float)  # 1.0 if wrong, 0.0 if right
            epsilon = np.sum(weights * misclassified)  # Weighted error rate

            # Clip epsilon to avoid division by zero or log(0)
            epsilon = np.clip(epsilon, 1e-10, 1.0 - 1e-10)  # Keep epsilon in (0, 1)

            # If the weak learner is worse than random guessing, stop early
            if epsilon >= 0.5:  # Random guessing threshold for binary classification
                logger.warning("Stopping early at round %d: epsilon=%.4f >= 0.5", t, epsilon)
                break  # This learner provides no useful information

            # Step 2d: Compute the learner's importance weight alpha
            # alpha_t = 0.5 * ln((1 - epsilon) / epsilon)
            # Higher alpha for lower error (more accurate learners get more weight)
            alpha = 0.5 * np.log((1.0 - epsilon) / epsilon)  # Learner importance

            # Step 2e: Update sample weights
            # Correctly classified: w_i *= exp(-alpha) -> weight decreases
            # Misclassified: w_i *= exp(+alpha) -> weight increases
            # This forces the next learner to focus on hard examples
            weights *= np.exp(-alpha * y_coded * predictions)  # Exponential reweighting

            # Normalize weights so they form a proper probability distribution
            weights /= np.sum(weights)  # Ensure weights sum to 1.0

            # Store the trained stump and its importance weight
            self.alphas.append(alpha)  # Save this round's alpha
            self.stumps.append(stump)  # Save the trained weak learner

        logger.info(  # Log summary of the training run
            "AdaBoost trained: %d rounds completed, max_depth=%d",
            len(self.stumps), self.max_depth,
        )
        return self  # Return the fitted classifier for method chaining

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw weighted vote scores (not thresholded).
        F(x) = sum_{t=1}^{T} alpha_t * h_t(x)
        Positive score -> class +1, negative -> class -1.
        """
        # Accumulate weighted predictions from all weak learners
        raw_scores = np.zeros(X.shape[0])  # Initialize scores to zero
        for alpha, stump in zip(self.alphas, self.stumps):  # Iterate over ensemble
            raw_scores += alpha * stump.predict(X)  # Add weighted prediction
        return raw_scores  # Return continuous-valued ensemble scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels {0, 1} for each sample.
        Uses the sign of the weighted vote: H(x) = sign(sum(alpha_t * h_t(x))).
        """
        raw_scores = self.predict_raw(X)  # Get continuous scores from ensemble
        # Map sign back to {0, 1}: sign >= 0 -> class 1, sign < 0 -> class 0
        return (np.sign(raw_scores) > 0).astype(int)  # Convert {-1,+1} to {0,1}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate class probabilities using sigmoid of raw scores.
        This is an approximation since AdaBoost doesn't directly model
        probabilities, but sigmoid(2*F(x)) works well in practice.
        """
        raw_scores = self.predict_raw(X)  # Get raw weighted scores
        # Apply sigmoid to convert raw scores to probability of class 1
        prob_positive = 1.0 / (1.0 + np.exp(-2.0 * raw_scores))  # Sigmoid(2*F(x))
        prob_positive = np.clip(prob_positive, 1e-7, 1.0 - 1e-7)  # Numerical stability
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.column_stack([1.0 - prob_positive, prob_positive])  # Shape: (n, 2)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,  # Total number of samples to generate
    n_features: int = 20,  # Total number of features per sample
    n_classes: int = 2,  # Number of target classes (binary for AdaBoost)
    random_state: int = 42,  # Seed for reproducible data generation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data and split into train/val/test.
    Uses sklearn's make_classification with informative and redundant features.
    Returns six arrays: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # Create a synthetic dataset with controlled class separability
    X, y = make_classification(
        n_samples=n_samples,  # Total samples to generate
        n_features=n_features,  # Dimensionality of feature space
        n_informative=n_features // 2,  # Half the features carry signal
        n_redundant=n_features // 4,  # Quarter are linear combos of informative features
        n_classes=n_classes,  # Binary classification for AdaBoost
        random_state=random_state,  # Reproducibility
    )

    # Split into 60% train, 20% validation, 20% test (stratified to preserve class ratios)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,  # 60/40 first split
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,  # 50/50 of the 40%
    )

    # Standardize features to zero mean and unit variance for numerical stability
    scaler = StandardScaler()  # Fit on training data only to prevent data leakage
    X_train = scaler.fit_transform(X_train)  # Fit scaler on train, transform train
    X_val = scaler.transform(X_val)  # Transform validation using train statistics
    X_test = scaler.transform(X_test)  # Transform test using train statistics

    logger.info(  # Log the dataset dimensions for verification
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test  # Return all splits


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,  # Training features
    y_train: np.ndarray,  # Training labels
    **hyperparams: Any,  # Keyword arguments for hyperparameters
) -> AdaBoostClassifier:
    """
    Train an AdaBoost classifier with given hyperparameters.
    Merges provided hyperparams with sensible defaults.
    """
    # Define default hyperparameters that work reasonably well
    defaults = dict(
        n_estimators=50,  # 50 boosting rounds as a reasonable default
        max_depth=1,  # Decision stumps (depth-1) as default weak learner
        random_state=42,  # Fixed seed for reproducibility
    )
    defaults.update(hyperparams)  # Override defaults with any user-provided values
    model = AdaBoostClassifier(**defaults)  # Instantiate the classifier
    model.fit(X_train, y_train)  # Train on the full training set
    logger.info(  # Log the training configuration
        "AdaBoost (NumPy) trained: %d rounds, max_depth=%d",
        defaults["n_estimators"], defaults["max_depth"],
    )
    return model  # Return the trained classifier


def _evaluate(
    model: AdaBoostClassifier,  # Trained AdaBoost model
    X: np.ndarray,  # Feature matrix to evaluate on
    y: np.ndarray,  # True labels for evaluation
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for the model.
    Returns a dictionary with accuracy, precision, recall, F1, and AUC-ROC.
    """
    y_pred = model.predict(X)  # Get hard class predictions {0, 1}
    y_proba = model.predict_proba(X)  # Get probability estimates for AUC

    # Compute AUC-ROC using probability of the positive class
    auc = roc_auc_score(y, y_proba[:, 1])  # Use P(y=1) column for binary AUC

    return {  # Return all metrics in a dictionary
        "accuracy": accuracy_score(y, y_pred),  # Overall correctness
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),  # Positive predictive value
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),  # Sensitivity / true positive rate
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),  # Harmonic mean of precision & recall
        "auc_roc": auc,  # Area under the ROC curve (discrimination ability)
    }


def validate(
    model: AdaBoostClassifier,  # Trained model to evaluate
    X_val: np.ndarray,  # Validation features
    y_val: np.ndarray,  # Validation labels
) -> Dict[str, float]:
    """Evaluate the model on validation data and log metrics."""
    metrics = _evaluate(model, X_val, y_val)  # Compute all metrics on val set
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics  # Return metrics dictionary


def test(
    model: AdaBoostClassifier,  # Trained model to evaluate
    X_test: np.ndarray,  # Test features (held-out)
    y_test: np.ndarray,  # Test labels (held-out)
) -> Dict[str, float]:
    """Evaluate the model on test data with full reporting."""
    metrics = _evaluate(model, X_test, y_test)  # Compute all metrics on test set
    y_pred = model.predict(X_test)  # Get predictions for confusion matrix
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))  # Show TP/FP/TN/FN
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))  # Per-class detail
    return metrics  # Return metrics dictionary


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,  # Optuna trial object for suggesting hyperparameters
    X_train: np.ndarray,  # Training features
    y_train: np.ndarray,  # Training labels
    X_val: np.ndarray,  # Validation features for evaluation
    y_val: np.ndarray,  # Validation labels for evaluation
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    Suggests hyperparameters, trains a model, and returns validation F1.
    Optuna maximizes this value over multiple trials.
    """
    # Suggest hyperparameters from defined search ranges
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 300, step=10),  # 10 to 300 rounds
        "max_depth": trial.suggest_int("max_depth", 1, 3),  # Stump to depth-3 tree
    }

    model = train(X_train, y_train, **params)  # Train with suggested params
    metrics = validate(model, X_val, y_val)  # Evaluate on validation set
    return metrics["f1"]  # Return F1 score for Optuna to maximize


def ray_tune_search(
    X_train: np.ndarray,  # Training features
    y_train: np.ndarray,  # Training labels
    X_val: np.ndarray,  # Validation features
    y_val: np.ndarray,  # Validation labels
    num_samples: int = 20,  # Number of configurations to try
) -> Dict[str, Any]:
    """
    Run hyperparameter search using Ray Tune for distributed/parallel search.
    Returns the best configuration found.
    """
    import ray  # Distributed computing framework
    from ray import tune as ray_tune  # Ray's hyperparameter tuning module

    if not ray.is_initialized():  # Initialize Ray only if not already running
        ray.init(ignore_reinit_error=True, log_to_driver=False)  # Quiet initialization

    def _trainable(config: Dict[str, Any]) -> None:
        """Inner function that trains a model with a given config and reports metrics."""
        model = train(X_train, y_train, **config)  # Train with the config
        metrics = validate(model, X_val, y_val)  # Evaluate on validation set
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])  # Report to Ray

    # Define the search space for hyperparameters
    search_space = {
        "n_estimators": ray_tune.choice([10, 50, 100, 200, 300]),  # Discrete choices
        "max_depth": ray_tune.randint(1, 4),  # Random integer in [1, 3]
    }

    # Create and run the tuner
    tuner = ray_tune.Tuner(
        _trainable,  # Function to optimize
        param_space=search_space,  # Search space definition
        tune_config=ray_tune.TuneConfig(
            num_samples=num_samples, metric="f1", mode="max",  # Maximize F1
        ),
    )
    results = tuner.fit()  # Run the search
    best = results.get_best_result(metric="f1", mode="max")  # Get best result
    logger.info("Ray Tune best config: %s", best.config)  # Log the best config
    ray.shutdown()  # Clean up Ray resources
    return best.config  # Return the best hyperparameter configuration


# ---------------------------------------------------------------------------
# Compare Parameter Sets
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,  # Training features
    y_train: np.ndarray,  # Training labels
    X_val: np.ndarray,  # Validation features
    y_val: np.ndarray,  # Validation labels
) -> Dict[str, Dict[str, float]]:
    """
    Compare 4 different AdaBoost configurations to understand the effect
    of number of estimators and weak learner complexity.

    Configurations:
        1. few_stumps: n_estimators=10, max_depth=1
           - Minimal ensemble, very fast, likely underfitting
        2. moderate_stumps: n_estimators=50, max_depth=1
           - Standard AdaBoost with stumps, good baseline
        3. many_stumps: n_estimators=200, max_depth=1
           - Large ensemble of stumps, tests if more rounds help
        4. depth2_trees: n_estimators=50, max_depth=2
           - Stronger weak learners (depth-2 trees), fewer needed
    """
    # Define each configuration with reasoning
    configs = {
        "few_stumps (T=10, d=1)": {
            # Reasoning: Very few rounds, likely insufficient for complex boundaries.
            # Use case: quick prototyping or very simple separable data.
            "n_estimators": 10,
            "max_depth": 1,  # Classic decision stump
        },
        "moderate_stumps (T=50, d=1)": {
            # Reasoning: Standard setting, 50 stumps usually suffice for moderate complexity.
            # Each stump adds a simple axis-aligned rule. AdaBoost converges well here.
            "n_estimators": 50,
            "max_depth": 1,  # Classic decision stump
        },
        "many_stumps (T=200, d=1)": {
            # Reasoning: More rounds allow finer granularity in the decision boundary.
            # Risk of slight overfitting on noisy data but generally robust.
            "n_estimators": 200,
            "max_depth": 1,  # Classic decision stump
        },
        "depth2_trees (T=50, d=2)": {
            # Reasoning: Depth-2 trees can model feature interactions (2 features per tree).
            # Fewer rounds needed because each learner is stronger. May capture XOR-like patterns.
            "n_estimators": 50,
            "max_depth": 2,  # Small tree with 2 levels of splits
        },
    }

    results = {}  # Store metrics for each configuration
    logger.info("=" * 70)
    logger.info("Comparing %d AdaBoost configurations", len(configs))
    logger.info("=" * 70)

    for name, params in configs.items():  # Train and evaluate each config
        logger.info("\n--- Config: %s ---", name)
        model = train(X_train, y_train, **params)  # Train with this config
        metrics = validate(model, X_val, y_val)  # Evaluate on validation data
        results[name] = metrics  # Store the results

    # Log summary comparison table
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("%-35s | Accuracy | F1     | AUC-ROC", "Configuration")
    logger.info("-" * 70)
    for name, metrics in results.items():  # Print each config's results
        logger.info(
            "%-35s | %.4f   | %.4f | %.4f",
            name, metrics["accuracy"], metrics["f1"], metrics["auc_roc"],
        )

    return results  # Return all results for further analysis


# ---------------------------------------------------------------------------
# Real-World Demo: Customer Churn Prediction
# ---------------------------------------------------------------------------

def real_world_demo() -> Dict[str, float]:
    """
    Demonstrate AdaBoost on a simulated customer churn prediction problem.

    Domain: Telecom / SaaS customer retention
    Goal: Predict which customers will cancel their subscription (churn = 1).

    Features simulate real business signals:
        - tenure_months: How long the customer has been with the company
        - monthly_charges: Monthly billing amount
        - total_charges: Cumulative billing (tenure * monthly)
        - num_support_tickets: Support interactions (frustrated customers call more)
        - contract_length_months: Contract duration (longer = more committed)
        - num_products: Cross-sell count (more products = stickier customer)
        - payment_delay_days: Average payment delay (financial distress signal)
        - usage_hours_per_week: Product engagement level
    """
    rng = np.random.RandomState(42)  # Fixed seed for reproducible demo
    n_samples = 800  # Realistic dataset size for a small business

    # Generate domain-specific features with realistic distributions
    tenure_months = rng.exponential(24, n_samples)  # Most customers are newer
    monthly_charges = rng.normal(70, 25, n_samples)  # Mean $70/month, std $25
    monthly_charges = np.clip(monthly_charges, 10, 200)  # Realistic charge range
    total_charges = tenure_months * monthly_charges  # Cumulative spending
    num_support_tickets = rng.poisson(2, n_samples)  # Average 2 tickets per customer
    contract_length = rng.choice([1, 6, 12, 24], n_samples, p=[0.3, 0.3, 0.25, 0.15])  # Contract mix
    num_products = rng.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1])  # Product adoption
    payment_delay = rng.exponential(5, n_samples)  # Average 5 days late
    usage_hours = rng.gamma(3, 5, n_samples)  # Right-skewed usage distribution

    # Stack features into a matrix with named columns conceptually
    X = np.column_stack([
        tenure_months,  # Feature 0: Customer lifetime in months
        monthly_charges,  # Feature 1: Monthly billing amount
        total_charges,  # Feature 2: Cumulative revenue from customer
        num_support_tickets,  # Feature 3: Support ticket count
        contract_length,  # Feature 4: Contract duration in months
        num_products,  # Feature 5: Number of products subscribed
        payment_delay,  # Feature 6: Average payment delay in days
        usage_hours,  # Feature 7: Weekly product usage hours
    ])

    # Generate churn labels based on realistic business logic
    # High churn probability when: short tenure, many tickets, few products, low usage
    churn_logit = (
        -0.03 * tenure_months  # Longer tenure -> lower churn (loyalty effect)
        + 0.01 * monthly_charges  # Higher charges -> slightly higher churn (price sensitivity)
        + 0.15 * num_support_tickets  # More tickets -> higher churn (frustration)
        - 0.05 * contract_length  # Longer contract -> lower churn (lock-in)
        - 0.3 * num_products  # More products -> lower churn (stickiness)
        + 0.05 * payment_delay  # Late payments -> higher churn (financial stress)
        - 0.04 * usage_hours  # More usage -> lower churn (engagement)
        + rng.normal(0, 0.5, n_samples)  # Random noise to make it realistic
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))  # Sigmoid to convert to probability
    y = (rng.random(n_samples) < churn_prob).astype(int)  # Sample binary labels

    # Split into train/val/test maintaining class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp,
    )

    # Standardize features using training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on train only
    X_val = scaler.transform(X_val)  # Transform val with train stats
    X_test = scaler.transform(X_test)  # Transform test with train stats

    feature_names = [  # Human-readable feature names for interpretability
        "tenure_months", "monthly_charges", "total_charges",
        "num_support_tickets", "contract_length_months",
        "num_products", "payment_delay_days", "usage_hours_per_week",
    ]

    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Customer Churn Prediction with AdaBoost")
    logger.info("=" * 70)
    logger.info("Features: %s", feature_names)
    logger.info("Churn rate: %.1f%%", 100 * np.mean(y))

    # Train AdaBoost model tuned for churn detection
    model = train(X_train, y_train, n_estimators=100, max_depth=1)
    validate(model, X_val, y_val)
    metrics = test(model, X_test, y_test)

    return metrics  # Return test metrics


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full AdaBoost pipeline:
    1. Generate synthetic data
    2. Train baseline model
    3. Run Optuna hyperparameter optimization
    4. Compare parameter sets with reasoning
    5. Run real-world customer churn demo
    6. Final test evaluation
    """
    logger.info("=" * 70)
    logger.info("AdaBoost (SAMME) - NumPy From-Scratch Implementation")
    logger.info("=" * 70)

    # Step 1: Generate synthetic classification data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=800, n_features=15,  # 800 samples, 15 features
    )

    # Step 2: Train baseline model with default hyperparameters
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, n_estimators=50, max_depth=1)  # 50 stumps
    validate(model, X_val, y_val)  # Check validation performance

    # Step 3: Optuna hyperparameter optimization
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")  # Maximize F1 score
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=15,  # Run 15 optimization trials
        show_progress_bar=True,  # Show progress in terminal
    )
    logger.info("Optuna best params: %s", study.best_params)  # Log best configuration
    logger.info("Optuna best F1: %.4f", study.best_value)  # Log best F1 achieved

    # Step 4: Retrain with best hyperparameters
    best_model = train(X_train, y_train, **study.best_params)

    # Step 5: Final test evaluation
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Step 6: Compare different parameter configurations
    logger.info("\n--- Parameter Set Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 7: Real-world customer churn demo
    logger.info("\n--- Real-World Demo: Customer Churn ---")
    real_world_demo()


if __name__ == "__main__":
    main()  # Entry point: run the full pipeline when executed as a script
