"""
Support Vector Machine - NumPy From-Scratch Implementation
============================================================

Theory & Mathematics:
    This module implements a Linear Support Vector Machine (SVM) from scratch
    using NumPy with hinge loss and stochastic gradient descent (SGD).

    Linear SVM Optimization (Primal):
        Minimize: (1/2) * ||w||^2 + C * sum max(0, 1 - y_i * (w^T x_i + b))

        The first term is the regularization (maximizes margin).
        The second term is the hinge loss (penalizes misclassification).

    Hinge Loss:
        L_hinge(y, f(x)) = max(0, 1 - y * f(x))
        where f(x) = w^T x + b and y in {-1, +1}.

        - If y * f(x) >= 1: correctly classified with margin, loss = 0
        - If 0 < y * f(x) < 1: correctly classified but within margin
        - If y * f(x) <= 0: misclassified

    Gradient Computation:
        If y_i * (w^T x_i + b) < 1 (violates margin):
            dL/dw = w - C * y_i * x_i    (regularization + hinge gradient)
            dL/db = -C * y_i
        Else (satisfies margin):
            dL/dw = w                      (regularization only)
            dL/db = 0

    SGD Update:
        w := w - lr * dL/dw
        b := b - lr * dL/db

    Learning Rate Schedule:
        lr_t = lr_0 / (1 + decay * t)
        Decreasing learning rate for convergence.

    Multiclass Extension (One-vs-Rest):
        For K classes, train K binary SVMs. Each SVM separates class k
        from all other classes. Predict the class with the highest
        decision function value: argmax_k (w_k^T x + b_k).

    Decision Function:
        f(x) = w^T x + b
        Positive values -> class +1, negative -> class -1

Business Use Cases:
    - Text classification with bag-of-words features
    - Linearly separable binary classification tasks
    - Understanding SVM internals for educational purposes
    - Lightweight classifier for embedded systems
    - Baseline for comparing with kernel SVM

Advantages:
    - Full transparency into SVM optimization
    - No external ML library dependency
    - Easy to extend with custom loss functions
    - Demonstrates margin-based classification clearly
    - Efficient for linearly separable data

Disadvantages:
    - Only linear SVM (no kernel trick without explicit feature mapping)
    - SGD may not converge to exact solution
    - Sensitive to learning rate and C hyperparameters
    - Slower than optimized libraries (libsvm, liblinear)
    - No built-in probability estimates

Hyperparameters:
    - C: Regularization parameter (higher = less regularization)
    - learning_rate: Initial SGD learning rate
    - n_epochs: Number of passes over the training data
    - lr_decay: Learning rate decay factor
    - batch_size: Mini-batch size for SGD
"""

# ── Standard library imports ──
# logging: provides structured diagnostic output for tracking training progress
# WHY: essential for monitoring convergence and debugging in production ML pipelines
import logging

# warnings: controls display of non-critical Python warnings
# WHY: we suppress sklearn/numpy warnings to keep console output focused on our SVM metrics
import warnings

# typing: provides type annotations for function signatures
# WHY: type hints make the from-scratch code self-documenting and catch bugs early via mypy
from typing import Any, Dict, List, Optional, Tuple

# ── Third-party imports ──
# numpy: the core numerical computing library for array operations
# WHY: we implement SVM entirely in numpy to understand every gradient computation step
import numpy as np

# optuna: Bayesian hyperparameter optimization framework
# WHY: TPE sampler efficiently explores the hyperparameter space (C, learning_rate, epochs)
import optuna

# sklearn utilities: we use these ONLY for data generation, splitting, scaling, and evaluation
# WHY: building these utilities from scratch would distract from the SVM implementation itself
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

# ── Logging configuration ──
# Set up structured logging with timestamps for tracking training progress
# WHY: timestamps help correlate training events with wall-clock time and debug slowdowns
logging.basicConfig(
    level=logging.INFO,  # INFO level shows training milestones without flooding the console
    format="%(asctime)s - %(levelname)s - %(message)s",  # Include timestamp and severity level
)
logger = logging.getLogger(__name__)  # Module-level logger for consistent naming in multi-file projects

# Suppress non-critical warnings from sklearn and numpy to keep output clean
# WHY: convergence warnings and deprecation notices clutter the training output
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Linear SVM from scratch
# ---------------------------------------------------------------------------

class LinearSVMBinary:
    """
    Binary Linear SVM using hinge loss + SGD.

    This implements the primal SVM optimization problem:
        min_w (1/2)||w||^2 + C * sum max(0, 1 - y_i(w^T x_i + b))

    The first term maximizes the margin (distance between classes).
    The second term penalizes points that violate or are within the margin.
    C controls the trade-off: large C => less regularization, tighter fit.
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        lr_decay: float = 0.001,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        # C: regularization parameter controlling the penalty for margin violations
        # WHY: higher C forces the SVM to classify more training points correctly (less regularization),
        # while lower C allows more margin violations for a wider, more generalizable margin
        self.C = C

        # learning_rate: initial step size for SGD weight updates
        # WHY: too large causes oscillation around the minimum; too small causes slow convergence
        self.learning_rate = learning_rate

        # n_epochs: number of complete passes over the entire training dataset
        # WHY: more epochs allow the weights to converge closer to the optimal solution,
        # but too many epochs waste compute if the model has already converged
        self.n_epochs = n_epochs

        # lr_decay: controls how quickly the learning rate decreases over epochs
        # WHY: decaying LR helps SGD converge by taking smaller steps as we approach the minimum,
        # following the schedule: lr_t = lr_0 / (1 + decay * t)
        self.lr_decay = lr_decay

        # batch_size: number of samples per mini-batch for SGD
        # WHY: mini-batch SGD provides a balance between pure SGD (noisy, 1 sample)
        # and full-batch GD (smooth but slow); None means full-batch gradient descent
        self.batch_size = batch_size

        # random_state: seed for reproducibility of weight initialization and shuffling
        # WHY: ensures experiments are reproducible for debugging and fair comparison
        self.random_state = random_state

        # w: weight vector (one weight per feature), initialized during fit()
        # WHY: stores the learned normal vector to the separating hyperplane
        self.w: Optional[np.ndarray] = None

        # b: bias term (scalar), shifts the hyperplane away from the origin
        # WHY: without bias, the hyperplane is forced to pass through the origin,
        # which is too restrictive for most real datasets
        self.b: float = 0.0

        # losses: tracks the loss value at the end of each epoch
        # WHY: monitoring the loss trajectory helps detect convergence issues,
        # oscillation, or learning rate problems
        self.losses: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMBinary":
        """
        Train the SVM using mini-batch SGD on the hinge loss + L2 regularization.
        Labels should be {-1, +1} (standard SVM convention).
        """
        # Create a RandomState instance for reproducible weight initialization and data shuffling
        # WHY: using a local RNG avoids corrupting the global random state used by other code
        rng = np.random.RandomState(self.random_state)

        # Extract dataset dimensions: n_samples (rows) and n_features (columns)
        # WHY: n_features determines the weight vector size; n_samples is used for batch calculations
        n_samples, n_features = X.shape

        # Initialize weights to small random values drawn from N(0, 0.01)
        # WHY: small random initialization breaks symmetry while staying close to zero;
        # starting at zero would work for convex SVM but random init can help escape saddle points
        self.w = rng.randn(n_features) * 0.01

        # Initialize bias to zero
        # WHY: bias has no preferred direction; the gradient updates will adjust it appropriately
        self.b = 0.0

        # Reset the loss history for this training run
        # WHY: allows tracking convergence from the start of this fit() call
        self.losses = []

        # If batch_size is None, use the full dataset (full-batch gradient descent)
        # WHY: full-batch gives exact gradients but is slow for large datasets;
        # mini-batch provides noisy but faster updates that often generalize better
        batch_size = self.batch_size or n_samples

        # ── Main training loop: iterate over the dataset n_epochs times ──
        for epoch in range(self.n_epochs):
            # Apply learning rate decay schedule: lr_t = lr_0 / (1 + decay * epoch)
            # WHY: decreasing the step size helps SGD converge by preventing oscillation
            # around the optimum; early epochs take large exploratory steps, later epochs fine-tune
            lr = self.learning_rate / (1 + self.lr_decay * epoch)

            # Randomly shuffle sample indices at the start of each epoch
            # WHY: shuffling prevents the model from learning the order of the data
            # and ensures each mini-batch is a representative sample of the dataset
            indices = rng.permutation(n_samples)

            # Accumulate loss across all mini-batches in this epoch for monitoring
            epoch_loss = 0.0

            # ── Mini-batch loop: process the dataset in chunks of batch_size ──
            for start in range(0, n_samples, batch_size):
                # Select the indices for the current mini-batch
                # WHY: slicing from shuffled indices gives us a random mini-batch without replacement
                idx = indices[start: start + batch_size]

                # Extract the mini-batch features and labels
                X_b, y_b = X[idx], y[idx]

                # Get the number of samples in this mini-batch (may be smaller for the last batch)
                m = len(y_b)

                # Compute the decision values: y_i * (w^T x_i + b)
                # WHY: this quantity determines whether each point satisfies the margin constraint;
                # if decision >= 1, the point is correctly classified with sufficient margin
                decision = y_b * (X_b @ self.w + self.b)

                # Create a boolean mask for points that violate the margin (decision < 1)
                # WHY: only margin-violating points contribute to the hinge loss gradient;
                # correctly classified points with sufficient margin have zero gradient
                mask = decision < 1  # Points inside or on wrong side of the margin

                # ── Compute gradients for weights and bias ──
                # Gradient of the full objective: dL/dw = w - (C/m) * sum_{violated} y_i * x_i
                # WHY: the first term (w) comes from the L2 regularization (1/2)||w||^2,
                # the second term comes from the hinge loss gradient on violated points;
                # X_b[mask].T @ y_b[mask] efficiently computes the sum of y_i * x_i for violated points
                dw = self.w - (self.C / m) * (X_b[mask].T @ y_b[mask])

                # Gradient of bias: dL/db = -(C/m) * sum_{violated} y_i
                # WHY: bias is not regularized (no w term), only updated by hinge loss gradients;
                # this is standard practice because regularizing the bias would shift the hyperplane
                db = -(self.C / m) * np.sum(y_b[mask])

                # ── SGD update step: move weights and bias in the negative gradient direction ──
                # WHY: gradient descent minimizes the objective by stepping opposite to the gradient
                self.w -= lr * dw
                self.b -= lr * db

                # ── Compute the loss for monitoring (not used for training, only for diagnostics) ──
                # Hinge loss: max(0, 1 - y * f(x)) for each sample in the mini-batch
                hinge = np.maximum(0, 1 - decision)

                # Total loss = regularization + C * mean hinge loss
                # WHY: this is the primal SVM objective we're minimizing;
                # tracking it helps verify that training is converging
                loss = 0.5 * np.dot(self.w, self.w) + self.C * np.mean(hinge)

                # Accumulate weighted loss for epoch-level averaging
                epoch_loss += loss * m

            # Store the average loss for this epoch
            # WHY: per-epoch loss allows plotting the convergence curve to diagnose training issues
            self.losses.append(epoch_loss / n_samples)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the signed distance from each sample to the separating hyperplane.
        f(x) = w^T x + b

        WHY: the decision function value indicates both the predicted class (sign)
        and the confidence (magnitude); points far from the hyperplane are more confidently classified
        """
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels {-1, +1} based on the sign of the decision function.

        WHY: the SVM decision boundary is the hyperplane where f(x) = 0;
        points on the positive side are class +1, negative side are class -1
        """
        # np.sign returns -1, 0, or +1; we cast to int for consistency
        # WHY: integer labels are easier to work with in downstream evaluation metrics
        return np.sign(self.decision_function(X)).astype(int)


class LinearSVMNumpy:
    """
    Multiclass SVM using One-vs-Rest (OvR) strategy.

    For K classes, we train K binary LinearSVMBinary classifiers.
    Each classifier separates one class from all others.
    Prediction uses the class with the highest decision function value.

    WHY OvR instead of OvO (One-vs-One):
        - OvR trains K classifiers (linear in K), while OvO trains K*(K-1)/2
        - OvR is simpler to implement and interpret
        - OvR works well when classes are well-separated
        - For binary classification, OvR reduces to a single classifier
    """

    def __init__(self, **kwargs: Any) -> None:
        # Store all hyperparameters to pass to each binary classifier
        # WHY: using **kwargs allows flexible hyperparameter forwarding
        # without duplicating the parameter list
        self.kwargs = kwargs

        # List of binary classifiers, one per class in OvR
        self.classifiers: List[LinearSVMBinary] = []

        # Unique class labels discovered during fitting
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMNumpy":
        """
        Fit the multiclass SVM using One-vs-Rest strategy.

        For binary problems, we train a single SVM with labels mapped to {-1, +1}.
        For multiclass, we train K separate binary SVMs, one for each class.
        """
        # Discover all unique class labels in the training data
        # WHY: we need to know how many classifiers to train and what labels to map
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            # ── Binary case: train a single SVM with labels {-1, +1} ──
            # Map the second class to +1 and the first to -1
            # WHY: SVM theory requires labels in {-1, +1} for the hinge loss formulation;
            # we map classes_[1] to +1 arbitrarily (consistent with sklearn convention)
            y_binary = np.where(y == self.classes_[1], 1, -1)

            # Create and train a single binary SVM classifier
            clf = LinearSVMBinary(**self.kwargs)
            clf.fit(X, y_binary)

            # Store as a single-element list for consistent access in predict/decision_function
            self.classifiers = [clf]
        else:
            # ── Multiclass case: One-vs-Rest (OvR) strategy ──
            self.classifiers = []
            for cls in self.classes_:
                # For each class, create binary labels: current class = +1, all others = -1
                # WHY: each classifier learns to distinguish one class from the rest,
                # building K independent decision boundaries
                y_binary = np.where(y == cls, 1, -1)

                # Train a fresh binary SVM for this class
                clf = LinearSVMBinary(**self.kwargs)
                clf.fit(X, y_binary)

                # Append to the list of classifiers (one per class)
                self.classifiers.append(clf)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for each sample.

        For binary: returns a 1D array of signed distances.
        For multiclass: returns a 2D array of shape (n_samples, n_classes).

        WHY: the decision function provides continuous scores for ranking and
        threshold-based decision making (e.g., adjusting precision-recall trade-off)
        """
        if len(self.classes_) == 2:
            # Binary: single decision function value per sample
            return self.classifiers[0].decision_function(X)

        # Multiclass: stack decision values from all K classifiers into columns
        # WHY: each column represents the "affinity" of samples to one class;
        # argmax across columns gives the predicted class
        return np.column_stack([clf.decision_function(X) for clf in self.classifiers])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each sample.

        Binary: uses sign of decision function.
        Multiclass: argmax over OvR decision values.

        WHY: for OvR, the class whose classifier gives the highest score wins;
        this is equivalent to choosing the class with the widest margin
        """
        if len(self.classes_) == 2:
            # Map sign of decision function back to original class labels
            decisions = self.classifiers[0].decision_function(X)
            return np.where(decisions >= 0, self.classes_[1], self.classes_[0])

        # For multiclass, pick the class with the highest decision score
        decisions = self.decision_function(X)
        return self.classes_[np.argmax(decisions, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Approximate probabilities using Platt-style sigmoid scaling.

        WHY: SVMs don't natively produce probabilities (they produce margin-based scores).
        Platt scaling applies a sigmoid function to convert decision values to pseudo-probabilities.
        For binary: P(y=1|x) = 1 / (1 + exp(-f(x)))
        For multiclass: we use softmax over OvR decision values.

        Note: these are APPROXIMATE probabilities, not true calibrated probabilities.
        For well-calibrated probabilities, use isotonic regression or Platt calibration on held-out data.
        """
        if len(self.classes_) == 2:
            # Apply sigmoid to the decision function value for binary case
            # WHY: sigmoid maps (-inf, +inf) to (0, 1), providing a probability estimate
            d = self.classifiers[0].decision_function(X)
            p1 = 1.0 / (1.0 + np.exp(-d))

            # Return two columns: P(class_0) and P(class_1)
            return np.column_stack([1 - p1, p1])
        else:
            # Apply softmax to multiclass decision values
            # WHY: softmax normalizes K raw scores into a probability distribution that sums to 1;
            # subtracting the max for numerical stability prevents exp() overflow
            decisions = self.decision_function(X)
            exp_d = np.exp(decisions - decisions.max(axis=1, keepdims=True))
            return exp_d / exp_d.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data and split into train/val/test sets.

    WHY we use a 60/20/20 split:
        - 60% training: sufficient data for the SVM to learn the margin
        - 20% validation: used for hyperparameter tuning (prevents data leakage into test)
        - 20% test: held out for final, unbiased performance estimate
    """
    # Generate a synthetic classification dataset with controlled complexity
    # WHY: make_classification creates a dataset with known informative and redundant features,
    # making it ideal for testing whether the SVM can identify the relevant features
    X, y = make_classification(
        n_samples=n_samples,           # Total number of synthetic samples to generate
        n_features=n_features,          # Total feature dimensionality
        n_informative=n_features // 2,  # Half the features carry real signal
        n_redundant=n_features // 4,    # Quarter are linear combinations of informative features
        n_classes=n_classes,            # Number of distinct target classes
        random_state=random_state,      # Seed for reproducibility
    )

    # First split: separate 60% train from 40% temp (which will become val + test)
    # WHY: stratify=y ensures each split preserves the class distribution,
    # preventing imbalanced splits that could bias evaluation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )

    # Second split: divide the 40% temp into 20% validation and 20% test
    # WHY: we split the remaining data equally so validation and test sets are the same size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    # Standardize features to zero mean and unit variance
    # WHY: SVMs are sensitive to feature scale because the margin depends on ||w||;
    # features with larger magnitudes would dominate the decision boundary
    # IMPORTANT: fit the scaler on training data only, then transform val/test
    # to prevent data leakage from the validation/test sets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Fit mean/std on train, then transform
    X_val = scaler.transform(X_val)           # Transform using train statistics only
    X_test = scaler.transform(X_test)         # Transform using train statistics only

    # Log the dataset dimensions for diagnostic purposes
    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> LinearSVMNumpy:
    """
    Train a from-scratch SVM model with configurable hyperparameters.

    WHY we use a defaults dict:
        This pattern allows calling code to override specific hyperparameters
        while keeping sensible defaults for the rest. This is especially useful
        for hyperparameter optimization where we sweep individual parameters.
    """
    # Define default hyperparameters for the SVM
    # WHY each default was chosen:
    #   C=1.0: balanced regularization (sklearn default)
    #   learning_rate=0.01: conservative step size that works for most datasets
    #   n_epochs=1000: enough iterations for convergence on medium datasets
    #   lr_decay=0.001: gentle decay that allows the LR to decrease ~50% over 1000 epochs
    #   batch_size=64: standard mini-batch size balancing speed and gradient quality
    #   random_state=42: reproducibility seed
    defaults = dict(
        C=1.0,
        learning_rate=0.01,
        n_epochs=1000,
        lr_decay=0.001,
        batch_size=64,
        random_state=42,
    )

    # Override defaults with any user-specified hyperparameters
    defaults.update(hyperparams)

    # Create the multiclass SVM wrapper with the merged hyperparameters
    # WHY: LinearSVMNumpy handles both binary and multiclass via OvR internally
    model = LinearSVMNumpy(**defaults)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Log the key hyperparameters for tracking which configuration was used
    logger.info("SVM (NumPy) trained: C=%.3f, lr=%.4f, epochs=%d",
                defaults["C"], defaults["learning_rate"], defaults["n_epochs"])
    return model


def _evaluate(model: LinearSVMNumpy, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    WHY we compute multiple metrics:
        - accuracy: overall correctness (can be misleading with imbalanced data)
        - precision: fraction of positive predictions that are correct (important when FP is costly)
        - recall: fraction of actual positives that are found (important when FN is costly)
        - f1: harmonic mean of precision and recall (balanced metric)
        - auc_roc: area under ROC curve (measures ranking quality, threshold-independent)
    """
    # Generate hard predictions (class labels)
    y_pred = model.predict(X)

    # Generate soft predictions (probability estimates) for AUC computation
    # WHY: AUC-ROC requires continuous scores, not just class labels;
    # our pseudo-probabilities from Platt scaling provide this
    y_proba = model.predict_proba(X)

    # Compute AUC-ROC differently for binary vs multiclass
    n_classes = len(np.unique(y))
    if n_classes == 2:
        # For binary, use the probability of the positive class
        auc = roc_auc_score(y, y_proba[:, 1])
    else:
        # For multiclass, use One-vs-Rest AUC with weighted averaging
        # WHY: weighted average accounts for class imbalance by weighting each class by its support
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: LinearSVMNumpy, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.

    WHY: validation metrics guide hyperparameter selection without
    touching the test set, preventing overfitting to the evaluation data.
    """
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: LinearSVMNumpy, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the model on the held-out test set and print detailed diagnostics.

    WHY: the test set provides an unbiased estimate of generalization performance.
    We also print the confusion matrix and classification report for deeper analysis
    of per-class performance and error patterns.
    """
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)

    # Log aggregate metrics
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    # Log the confusion matrix for visualizing error patterns
    # WHY: shows true positives, false positives, true negatives, false negatives;
    # reveals if the model is biased toward one class
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Log the full classification report with per-class precision, recall, F1
    # WHY: per-class metrics reveal if the model struggles with specific classes
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
    """
    Optuna objective function for hyperparameter optimization.

    WHY Optuna over grid search:
        - TPE (Tree-structured Parzen Estimator) focuses on promising regions
        - Handles log-uniform distributions naturally for scale-sensitive parameters
        - Pruning support stops unpromising trials early
        - Much more sample-efficient than grid search for large search spaces
    """
    # Define the hyperparameter search space
    params = {
        # C: log-uniform because the optimal value can span several orders of magnitude
        # WHY log scale: C=0.001 vs C=100 represents fundamentally different regularization regimes
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),

        # learning_rate: log-uniform because small changes at low LR matter more than at high LR
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),

        # n_epochs: step of 200 to avoid wasting trials on tiny epoch differences
        "n_epochs": trial.suggest_int("n_epochs", 200, 2000, step=200),

        # lr_decay: log-uniform because decay rate spans multiple orders of magnitude
        "lr_decay": trial.suggest_float("lr_decay", 1e-5, 0.01, log=True),

        # batch_size: categorical because batch sizes are typically powers of 2
        # WHY: powers of 2 align well with memory architectures and GPU warp sizes
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }

    # Train the model with the trial's hyperparameters
    model = train(X_train, y_train, **params)

    # Evaluate on validation set and return F1 score as the optimization target
    # WHY F1: it balances precision and recall, making it robust for imbalanced datasets
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Hyperparameter search using Ray Tune for distributed optimization.

    WHY Ray Tune in addition to Optuna:
        - Supports distributed training across multiple machines
        - Built-in schedulers (ASHA, PBT) for early stopping
        - Integrates with cloud providers for auto-scaling
    """
    # Lazy import: Ray is optional and may not be installed in all environments
    import ray
    from ray import tune as ray_tune

    # Initialize Ray cluster if not already running
    # WHY: ignore_reinit_error prevents crashes in notebooks where Ray may already be initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        """Inner function that Ray Tune calls for each trial."""
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        # Report metrics back to the Ray Tune scheduler
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    # Define the search space using Ray Tune's distribution primitives
    search_space = {
        "C": ray_tune.loguniform(1e-3, 100.0),
        "learning_rate": ray_tune.loguniform(1e-4, 0.1),
        "n_epochs": ray_tune.choice([500, 1000, 1500]),
        "lr_decay": ray_tune.loguniform(1e-5, 0.01),
        "batch_size": ray_tune.choice([32, 64, 128]),
    }

    # Create and run the tuner
    tuner = ray_tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=ray_tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"),
    )
    results = tuner.fit()

    # Extract the best configuration
    best = results.get_best_result(metric="f1", mode="max")
    logger.info("Ray Tune best config: %s", best.config)

    # Shut down Ray to free resources
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Compare Parameter Sets
# ---------------------------------------------------------------------------

def compare_parameter_sets() -> None:
    """
    Train the from-scratch SVM with 4 different hyperparameter configurations
    and display a formatted comparison table with reasoning.

    WHY this function exists:
        - Demonstrates how C, learning_rate, and epoch count affect SVM behavior
        - Shows the bias-variance trade-off in practice
        - Provides intuition for manual hyperparameter selection before running Optuna
        - Helps understand the sensitivity of from-scratch SVM to its hyperparameters
    """
    print("\n" + "=" * 85)
    print("COMPARE PARAMETER SETS - SVM NumPy From-Scratch")
    print("=" * 85)

    # Generate a fresh dataset for the comparison experiment
    # WHY: using a clean dataset prevents any data leakage from previous experiments
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # ── Define 4 hyperparameter configurations with different trade-offs ──
    configs = {
        # Config 1: Conservative / high regularization
        # WHY: low C (=0.1) means heavy regularization, allowing many margin violations.
        # This creates a wide margin that may underfit but generalizes well to noisy data.
        # Slow LR (0.005) with many epochs (1500) ensures careful convergence.
        "Low C (wide margin)": {
            "C": 0.1, "learning_rate": 0.005, "n_epochs": 1500,
            "lr_decay": 0.001, "batch_size": 64,
        },

        # Config 2: Balanced / default-like settings
        # WHY: C=1.0 is the standard default that balances margin width and classification accuracy.
        # Moderate LR and epochs provide a good baseline for most datasets.
        "Moderate C (balanced)": {
            "C": 1.0, "learning_rate": 0.01, "n_epochs": 1000,
            "lr_decay": 0.001, "batch_size": 64,
        },

        # Config 3: Aggressive / low regularization
        # WHY: high C (=10.0) penalizes margin violations heavily, forcing the SVM to classify
        # nearly all training points correctly. This can overfit on noisy data but works well
        # on clean, linearly separable datasets.
        "High C (tight margin)": {
            "C": 10.0, "learning_rate": 0.01, "n_epochs": 1000,
            "lr_decay": 0.001, "batch_size": 64,
        },

        # Config 4: Large batch with fast LR
        # WHY: larger batches (256) give smoother gradients, allowing a higher learning rate (0.05).
        # Fewer epochs (500) are sufficient because each update is more informative.
        # This configuration tests the speed-accuracy trade-off.
        "Large Batch + Fast LR": {
            "C": 1.0, "learning_rate": 0.05, "n_epochs": 500,
            "lr_decay": 0.005, "batch_size": 256,
        },
    }

    # ── Train each configuration and collect metrics ──
    results = {}
    for name, params in configs.items():
        print(f"\nTraining config: {name}...")
        model = train(X_train, y_train, **params)
        val_metrics = validate(model, X_val, y_val)
        test_metrics = _evaluate(model, X_test, y_test)

        # Get the final training loss for convergence analysis
        final_loss = model.classifiers[0].losses[-1] if model.classifiers[0].losses else float("nan")

        results[name] = {
            "val_f1": val_metrics["f1"],
            "val_acc": val_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_acc": test_metrics["accuracy"],
            "test_auc": test_metrics["auc_roc"],
            "final_loss": final_loss,
        }

    # ── Print the formatted comparison table ──
    print("\n" + "-" * 105)
    print(f"{'Config':<25} {'Val F1':>8} {'Val Acc':>8} {'Test F1':>8} "
          f"{'Test Acc':>8} {'Test AUC':>9} {'Final Loss':>11}")
    print("-" * 105)
    for name, m in results.items():
        print(f"{name:<25} {m['val_f1']:>8.4f} {m['val_acc']:>8.4f} {m['test_f1']:>8.4f} "
              f"{m['test_acc']:>8.4f} {m['test_auc']:>9.4f} {m['final_loss']:>11.4f}")
    print("-" * 105)

    # ── Interpretation guide ──
    print("\nInterpretation Guide:")
    print("  - Low C (wide margin): Heavy regularization creates a simpler model.")
    print("    Best for noisy data or when you want to prevent overfitting.")
    print("  - Moderate C (balanced): Standard trade-off between margin width and accuracy.")
    print("    Good starting point for most classification tasks.")
    print("  - High C (tight margin): Forces correct classification of most training points.")
    print("    Works well on clean, linearly separable data; may overfit on noisy data.")
    print("  - Large Batch + Fast LR: Smoother gradients allow faster learning rates.")
    print("    Trades precision for training speed; good for large datasets.")
    print("  - Compare Final Loss: lower loss does NOT always mean better generalization.")
    print("    The best model is the one with the highest validation F1 score.")


# ---------------------------------------------------------------------------
# Real-World Demo: Email Spam Detection
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """
    Demonstrate the from-scratch SVM on a realistic email spam detection task.

    WHY email spam detection:
        - Classic SVM use case (SVMs were state-of-the-art for spam filtering in the early 2000s)
        - Linear SVMs work exceptionally well on text/frequency features
        - Binary classification with clear business impact (blocking spam vs. passing legitimate email)
        - Features are naturally interpretable (word frequencies, character frequencies)

    Domain context:
        Email spam detection analyzes email content to classify messages as spam or legitimate (ham).
        Features typically include word frequencies, character frequencies, and formatting statistics.
        SVMs are well-suited because:
        1. Text features are high-dimensional but often linearly separable
        2. The margin concept aligns with the "confidence" of spam detection
        3. SVMs handle the bag-of-words feature space efficiently
    """
    print("\n" + "=" * 85)
    print("REAL-WORLD DEMO: Email Spam Detection (SVM NumPy From-Scratch)")
    print("=" * 85)

    # ── Generate synthetic but realistic email spam detection data ──
    np.random.seed(42)
    n_samples = 1200  # 1200 emails total (realistic for a labeled spam dataset)

    # Feature names that correspond to real email spam detection features
    # WHY these features: they mirror the features used in the famous Spambase dataset from UCI,
    # which analyzed email content to identify spam characteristics
    feature_names = [
        "word_freq_free",       # Frequency of the word "free" (strong spam indicator)
        "word_freq_money",      # Frequency of the word "money" (common in financial spam)
        "word_freq_offer",      # Frequency of the word "offer" (promotional spam)
        "word_freq_click",      # Frequency of the word "click" (phishing/clickbait)
        "capital_run_avg",      # Average length of consecutive capital letters
        "capital_run_max",      # Maximum length of consecutive capital letters
        "char_freq_exclaim",    # Frequency of '!' character (spam uses excessive punctuation)
        "char_freq_dollar",     # Frequency of '$' character (financial spam indicator)
    ]

    # ── Simulate spam emails (class 1) ──
    # WHY these distributions: spam emails tend to have higher frequencies of trigger words
    # and excessive use of capitals and special characters
    n_spam = 480  # ~40% of emails are spam (realistic spam prevalence)
    spam_features = np.column_stack([
        np.random.exponential(0.8, n_spam),    # word_freq_free: spam uses "free" frequently
        np.random.exponential(0.6, n_spam),    # word_freq_money: "money" appears in financial spam
        np.random.exponential(0.7, n_spam),    # word_freq_offer: "offer" is a spam trigger word
        np.random.exponential(0.5, n_spam),    # word_freq_click: "click here" is common in phishing
        np.random.exponential(3.0, n_spam),    # capital_run_avg: spam uses MORE CAPS ON AVERAGE
        np.random.exponential(15.0, n_spam),   # capital_run_max: spam has LONG CAPITAL RUNS
        np.random.exponential(0.4, n_spam),    # char_freq_exclaim: spam uses lots of !!!
        np.random.exponential(0.3, n_spam),    # char_freq_dollar: financial spam uses $ signs
    ])

    # ── Simulate legitimate (ham) emails (class 0) ──
    # WHY: legitimate emails have lower frequencies of spam trigger words
    # and more moderate formatting characteristics
    n_ham = n_samples - n_spam  # ~60% of emails are legitimate
    ham_features = np.column_stack([
        np.random.exponential(0.1, n_ham),     # word_freq_free: rarely used in legitimate email
        np.random.exponential(0.05, n_ham),    # word_freq_money: almost never in normal email
        np.random.exponential(0.08, n_ham),    # word_freq_offer: occasional in legitimate promotions
        np.random.exponential(0.1, n_ham),     # word_freq_click: rarely used without malicious intent
        np.random.exponential(1.0, n_ham),     # capital_run_avg: normal capitalization for sentences
        np.random.exponential(4.0, n_ham),     # capital_run_max: short capitals (names, acronyms)
        np.random.exponential(0.05, n_ham),    # char_freq_exclaim: minimal use of exclamation marks
        np.random.exponential(0.02, n_ham),    # char_freq_dollar: rarely used in normal email
    ])

    # Combine spam and ham into a single dataset with labels
    # WHY: 1 = spam, 0 = ham (standard convention in spam detection literature)
    X = np.vstack([spam_features, ham_features])
    y = np.array([1] * n_spam + [0] * n_ham)

    # Shuffle the data to mix spam and ham
    # WHY: without shuffling, the first 480 samples would all be spam,
    # which could bias the training process
    shuffle_idx = np.random.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Split into train/val/test with stratification
    # WHY: stratify ensures the spam/ham ratio is preserved in each split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp,
    )

    # Standardize features (fit on training data only)
    # WHY: SVMs are sensitive to feature scale; word frequencies and capital run lengths
    # have very different scales that must be normalized
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nDataset: {n_samples} emails ({n_spam} spam, {n_ham} ham)")
    print(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # ── Train the from-scratch SVM on the spam detection data ──
    print("\n--- Training From-Scratch Linear SVM ---")
    model = train(X_train, y_train, C=1.0, learning_rate=0.01, n_epochs=1000, batch_size=64)

    # ── Evaluate on validation and test sets ──
    val_metrics = validate(model, X_val, y_val)
    test_metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")

    # ── Display the confusion matrix with domain-specific labels ──
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Ham  Predicted Spam")
    print(f"  Actual Ham     {cm[0][0]:>12}  {cm[0][1]:>14}")
    print(f"  Actual Spam    {cm[1][0]:>12}  {cm[1][1]:>14}")

    # ── Analyze feature weights to understand what the SVM learned ──
    # For binary SVM, the weight vector directly indicates feature importance
    # WHY: in a linear SVM, w_j indicates how much feature j contributes to the decision;
    # positive weights push toward +1 (spam), negative weights push toward -1 (ham)
    if len(model.classifiers) == 1:
        weights = model.classifiers[0].w
        print(f"\nFeature Weights (positive = spam indicator, negative = ham indicator):")
        # Sort features by absolute weight for ranked importance
        weight_order = np.argsort(np.abs(weights))[::-1]
        for rank, idx in enumerate(weight_order, 1):
            direction = "SPAM" if weights[idx] > 0 else "HAM"
            print(f"  {rank}. {feature_names[idx]:<22} weight={weights[idx]:>+8.4f}  ({direction} indicator)")

    # ── Convergence analysis ──
    if model.classifiers and model.classifiers[0].losses:
        losses = model.classifiers[0].losses
        print(f"\nConvergence Analysis:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss:   {losses[-1]:.4f}")
        print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

    # ── Business insights ──
    print(f"\nBusiness Insights for Email Spam Detection:")
    print(f"  - False Positives (ham classified as spam): {cm[0][1]} legitimate emails blocked")
    print(f"    Impact: users miss important emails, customer complaints, lost business")
    print(f"  - False Negatives (spam classified as ham): {cm[1][0]} spam emails delivered")
    print(f"    Impact: user annoyance, potential security risks (phishing)")
    print(f"  - In production, tune the decision threshold to trade off FP vs FN")
    print(f"    based on business priorities (blocking spam vs. ensuring delivery)")
    print(f"  - Linear SVM is ideal for spam detection because:")
    print(f"    1. Word frequency features are naturally high-dimensional")
    print(f"    2. Linear boundary is sufficient for frequency-based features")
    print(f"    3. The model is fast enough for real-time email filtering")
    print(f"    4. Weight vector provides interpretable feature importance")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point: runs baseline training, hyperparameter optimization, testing,
    parameter comparison, and real-world demonstration.

    WHY this ordering:
        1. Baseline: establishes a performance reference point
        2. Optuna: searches for optimal hyperparameters
        3. Test: evaluates the best model on held-out data
        4. Compare: shows how different hyperparameters affect performance
        5. Real-world demo: demonstrates practical application to email spam detection
    """
    logger.info("=" * 70)
    logger.info("Support Vector Machine - NumPy From-Scratch Implementation")
    logger.info("=" * 70)

    # Generate the standard synthetic dataset for baseline and optimization
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # ── Baseline training with default hyperparameters ──
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # ── Hyperparameter optimization with Optuna ──
    logger.info("\n--- Optuna Hyperparameter Optimization ---")

    # Create a study that maximizes the F1 score
    # WHY: F1 balances precision and recall, making it a robust optimization target
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=20,              # 20 trials balances search quality with compute time
        show_progress_bar=True,   # Visual progress indicator for long-running searches
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Retrain the model with the best hyperparameters found by Optuna
    # WHY: the model trained during the Optuna trial may have been partially trained;
    # retraining ensures we get the full benefit of the optimal hyperparameters
    best_model = train(X_train, y_train, **study.best_params)

    # ── Final test evaluation ──
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # ── Run the parameter comparison experiment ──
    compare_parameter_sets()

    # ── Run the real-world email spam detection demo ──
    real_world_demo()


if __name__ == "__main__":
    main()
