"""
Logistic Regression - NumPy From-Scratch Implementation
========================================================

Theory & Mathematics:
    Logistic Regression models the probability of a binary outcome using the
    logistic (sigmoid) function applied to a linear combination of features.

    Forward Pass:
        z = X @ w + b                         (linear combination)
        a = sigma(z) = 1 / (1 + exp(-z))      (sigmoid activation)

    Loss Function (Binary Cross-Entropy):
        L(w, b) = -1/N * sum [y*log(a) + (1-y)*log(1-a)]

    With L2 regularization:
        L_reg = L + (lambda / 2N) * ||w||^2

    Gradient Computation:
        dL/dw = 1/N * X^T @ (a - y) + (lambda/N) * w
        dL/db = 1/N * sum(a - y)

    Parameter Update (Gradient Descent):
        w := w - lr * dL/dw
        b := b - lr * dL/db

    Decision boundary: classify as 1 if sigma(z) >= 0.5, else 0.

    For multiclass, this implementation uses One-vs-Rest (OvR):
        Train K binary classifiers, one per class; predict the class
        with the highest probability.

Business Use Cases:
    - Real-time fraud detection in financial transactions
    - A/B test analysis and conversion prediction
    - Patient risk stratification in healthcare
    - Sentiment classification (positive/negative)
    - Insurance claim prediction

Advantages:
    - Full control over training loop and convergence
    - Educational value: understanding gradient descent dynamics
    - No external ML library dependency (only NumPy)
    - Easy to extend with custom regularization or loss functions
    - Lightweight deployment footprint

Disadvantages:
    - Slower than optimized C/Fortran backends (scikit-learn, etc.)
    - Manual implementation of numerical stability tricks required
    - No built-in support for advanced solvers (L-BFGS, Newton-CG)
    - Must manually handle feature scaling
    - Harder to maintain and debug than library code

Hyperparameters:
    - learning_rate: Step size for gradient descent (typical: 0.001 - 0.1)
    - n_epochs: Number of full passes over the training data
    - lambda_reg: L2 regularization strength
    - batch_size: Mini-batch size for stochastic gradient descent
"""

# --- Standard library imports ---
# logging: structured output for tracking training progress and metrics.
# WHY: Provides severity levels (INFO, DEBUG, WARNING) and timestamps,
# which is superior to print() for ML experiment tracking.
import logging

# warnings: suppresses noisy warnings from sklearn metrics during evaluation.
# WHY: Some metric computations (e.g., zero_division in precision_score) generate
# warnings that are expected and would clutter the output.
import warnings

# typing: type hints for function signatures, improving code readability and IDE support.
# WHY: Optional is needed because some attributes start as None before fitting.
from typing import Any, Dict, Optional, Tuple

# --- Third-party imports ---
# numpy: the core numerical computing library; this is a "from-scratch" implementation
# meaning numpy is our ONLY computational dependency (no sklearn for the model itself).
# WHY: NumPy provides vectorized operations (matrix multiply, element-wise ops) that
# are much faster than pure Python loops, while still giving us full control.
import numpy as np

# optuna: Bayesian hyperparameter optimization framework.
# WHY: Even from-scratch implementations benefit from systematic hyperparameter search.
# Optuna's TPE algorithm explores the search space more efficiently than grid/random search.
import optuna

# sklearn imports are ONLY used for data generation, preprocessing, and evaluation metrics.
# WHY: The model itself is pure NumPy, but we use sklearn's utilities for fair comparison
# with the sklearn implementation (same data, same metrics, same splits).
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,           # Overall correctness: (TP+TN) / total.
    classification_report,    # Per-class precision/recall/F1 summary.
    confusion_matrix,         # TP/TN/FP/FN breakdown matrix.
    f1_score,                 # Harmonic mean of precision and recall.
    precision_score,          # Fraction of positive predictions that are correct.
    recall_score,             # Fraction of actual positives that we detected.
    roc_auc_score,            # Area under ROC curve; threshold-independent ranking metric.
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Logging configuration ---
# Configure logging with timestamps and severity levels for experiment tracking.
# WHY: level=INFO shows training progress without verbose DEBUG gradient values.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Create a module-level logger named after this file for consistent identification.
logger = logging.getLogger(__name__)
# Suppress all warnings to keep output focused on our metrics and logging.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Logistic Regression from scratch
# ---------------------------------------------------------------------------

class LogisticRegressionNumpy:
    """Binary / multiclass (OvR) logistic regression implemented in NumPy.

    WHY from scratch: Building logistic regression from scratch teaches the fundamental
    concepts of gradient descent, loss functions, and regularization that underpin ALL
    neural networks and many ML algorithms. Understanding these internals makes you
    a better practitioner when using high-level libraries.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        lambda_reg: float = 0.01,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        # learning_rate: controls how big each gradient descent step is.
        # WHY 0.01: A moderate default. Too large (>0.1) causes oscillation and divergence.
        # Too small (<0.001) makes training painfully slow with many epochs needed.
        self.learning_rate = learning_rate

        # n_epochs: number of full passes over the entire training dataset.
        # WHY 1000: Enough for convergence on most problems. The loss should plateau
        # well before 1000 epochs if the learning rate is appropriate.
        self.n_epochs = n_epochs

        # lambda_reg: L2 regularization strength, also called weight decay.
        # WHY 0.01: Mild regularization that prevents weights from growing too large
        # without overly constraining the model. Equivalent to C=100 in sklearn.
        self.lambda_reg = lambda_reg

        # batch_size: number of samples per gradient update (None = full batch).
        # WHY optional: Full-batch gradient descent is simpler but slower for large datasets.
        # Mini-batch SGD (e.g., 64) adds noise to gradients which can actually help
        # escape local minima, and is computationally efficient for large datasets.
        self.batch_size = batch_size

        # random_state: seed for reproducible random number generation.
        # WHY: Ensures weight initialization and data shuffling are deterministic.
        self.random_state = random_state

        # Model parameters (set during fitting).
        # WHY Optional: These are None until fit() is called, indicating an untrained model.
        self.weights_: Optional[np.ndarray] = None    # Weight vector w of shape (n_features,).
        self.bias_: Optional[float] = None            # Scalar bias term b.
        self.classes_: Optional[np.ndarray] = None    # Unique class labels in the data.
        self._models: Optional[list] = None           # List of (w, b, losses) for OvR multiclass.

    # --- Helper methods ---
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function: sigma(z) = 1 / (1 + exp(-z)).

        WHY sigmoid: It maps any real number to the (0, 1) interval, giving us
        a valid probability estimate. It's the canonical link function for
        binary logistic regression.

        WHY clip: Without clipping, very large negative z values cause exp(-z) to overflow
        to infinity (returning NaN), and very large positive z values cause exp(-z) to
        underflow to 0 (which is fine, but we clip symmetrically for safety).
        The range [-500, 500] is safely within float64 range.
        """
        z = np.clip(z, -500, 500)     # Prevent numerical overflow in exp().
        return 1.0 / (1.0 + np.exp(-z))  # The sigmoid formula.

    @staticmethod
    def _cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Compute binary cross-entropy loss: -mean[y*log(p) + (1-y)*log(1-p)].

        WHY cross-entropy: It measures how far our predicted probabilities are from
        the true labels. It penalizes confident wrong predictions heavily (log(0) -> -inf),
        making it a better training objective than MSE for classification.

        WHY eps clipping: log(0) = -infinity, which would crash the computation.
        Clipping y_hat to [eps, 1-eps] ensures we never take log of exactly 0 or 1.
        """
        eps = 1e-15                                # Small constant to prevent log(0).
        y_hat = np.clip(y_hat, eps, 1 - eps)       # Clip predictions to valid range.
        # Compute the mean binary cross-entropy loss over all samples.
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    # --- Binary training (core algorithm) ---
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, list]:
        """Train a binary logistic regression classifier using mini-batch SGD.

        WHY separate method: For multiclass (OvR), we call this method once per class.
        Isolating it makes the code modular and easier to understand.

        Returns:
            Tuple of (weights, bias, loss_history).
        """
        # Create a random number generator with a fixed seed for reproducibility.
        # WHY RandomState instead of np.random.seed: RandomState is instance-based,
        # so multiple calls don't interfere with each other's random sequences.
        rng = np.random.RandomState(self.random_state)

        # Get the dimensions of the training data.
        n_samples, n_features = X.shape

        # Initialize weights with small random values near zero.
        # WHY small random (0.01 * randn): Starting with large weights would put the
        # sigmoid in its saturated region where gradients are near zero (vanishing gradients).
        # Small random values keep the sigmoid in its linear region initially,
        # where gradients are largest and learning is fastest.
        # WHY not zeros: All-zero initialization works for logistic regression but would
        # cause symmetry breaking issues in neural networks. Using small random values
        # is a good habit.
        w = rng.randn(n_features) * 0.01

        # Initialize bias to zero.
        # WHY zero: The bias shifts the decision boundary. Starting at zero is conventional
        # and lets the model learn the appropriate shift during training.
        b = 0.0

        # Track loss over epochs for monitoring convergence.
        # WHY: Plotting the loss curve helps diagnose issues like divergence (loss increases),
        # oscillation (loss bounces), or premature convergence (loss plateaus too early).
        losses = []

        # Use the full dataset if batch_size is not specified.
        # WHY: Full-batch gradient descent gives the exact gradient (no noise) but is
        # slow for large datasets. Mini-batch SGD trades gradient precision for speed.
        batch_size = self.batch_size or n_samples

        # --- Main training loop ---
        for epoch in range(self.n_epochs):
            # Shuffle data indices at the start of each epoch.
            # WHY: Without shuffling, mini-batches would always contain the same samples
            # in the same order, introducing systematic bias in the gradient estimates.
            # Shuffling ensures each epoch sees the data in a different order.
            indices = rng.permutation(n_samples)

            # Process the data in mini-batches.
            for start in range(0, n_samples, batch_size):
                # Select the current batch using shuffled indices.
                idx = indices[start : start + batch_size]
                X_b, y_b = X[idx], y[idx]           # Batch features and labels.
                m = len(y_b)                         # Actual batch size (may be smaller at end).

                # --- Forward pass ---
                # Compute the linear combination z = X @ w + b.
                # WHY matrix multiply: Vectorized computation is O(m*n) but runs in optimized
                # BLAS routines, much faster than m separate dot products in a Python loop.
                z = X_b @ w + b

                # Apply sigmoid to get predicted probabilities.
                # WHY: The sigmoid maps the linear output z to [0,1], giving us P(y=1|x).
                a = self._sigmoid(z)

                # --- Gradient computation ---
                # Gradient of the loss w.r.t. weights: dL/dw = (1/m) * X^T @ (a - y) + (lambda/m) * w
                # WHY (a - y): This is the derivative of cross-entropy through the sigmoid.
                # The beauty of logistic regression is that the gradient has the same form as
                # linear regression: prediction error (a - y) projected onto the feature matrix.
                # The regularization term (lambda/m) * w penalizes large weights.
                dw = (1.0 / m) * (X_b.T @ (a - y_b)) + (self.lambda_reg / m) * w

                # Gradient of the loss w.r.t. bias: dL/db = mean(a - y)
                # WHY no regularization: The bias is not regularized because it only shifts
                # the decision boundary; regularizing it would force the boundary through the origin.
                db = np.mean(a - y_b)

                # --- Parameter update (gradient descent) ---
                # Move weights in the negative gradient direction to reduce the loss.
                # WHY subtract: We want to MINIMIZE the loss. The gradient points in the
                # direction of steepest INCREASE, so we go in the opposite direction.
                w -= self.learning_rate * dw
                b -= self.learning_rate * db

            # --- Epoch-level loss computation ---
            # Compute the full-dataset loss at the end of each epoch for monitoring.
            # WHY full dataset: Mini-batch loss is noisy. Computing loss on all samples
            # gives a smooth curve for monitoring convergence.
            a_full = self._sigmoid(X @ w + b)
            # Total loss = cross-entropy + L2 regularization penalty.
            # WHY: The regularization term prevents overfitting by penalizing large weights.
            loss = self._cross_entropy(y, a_full) + (self.lambda_reg / (2 * n_samples)) * np.sum(w ** 2)
            losses.append(loss)

        return w, b, losses

    # --- Public API ---
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionNumpy":
        """Fit the model to training data.

        WHY: This is the main entry point. For binary classification, we train one
        classifier. For multiclass, we use One-vs-Rest (OvR) strategy: train K separate
        binary classifiers, one for each class.
        """
        # Identify unique classes in the training data.
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            # Binary case: train a single classifier.
            # WHY: Binary is the native mode for logistic regression with sigmoid.
            self.weights_, self.bias_, self._losses = self._fit_binary(X, y)
        else:
            # Multiclass: One-vs-Rest strategy.
            # WHY OvR: Each binary classifier learns to separate one class from all others.
            # At prediction time, we pick the class with the highest predicted probability.
            # Alternative: softmax regression (multinomial) trains all classes jointly,
            # but OvR is simpler to implement from scratch.
            self._models = []
            for cls in self.classes_:
                # Create binary labels: 1 for the current class, 0 for all others.
                # WHY astype(float): The gradient computation expects float labels, not int.
                y_bin = (y == cls).astype(float)
                # Train a separate binary classifier for this class.
                w, b, losses = self._fit_binary(X, y_bin)
                # Store the trained parameters for this class.
                self._models.append((w, b, losses))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each sample.

        WHY: Probability estimates are essential for ranking, threshold selection,
        and computing metrics like AUC-ROC.
        """
        if len(self.classes_) == 2:
            # For binary: apply sigmoid to get P(y=1|x), stack with P(y=0|x) = 1 - P(y=1|x).
            p1 = self._sigmoid(X @ self.weights_ + self.bias_)
            # column_stack creates a (n_samples, 2) array: [P(y=0), P(y=1)] per sample.
            return np.column_stack([1 - p1, p1])
        else:
            # For multiclass: get each classifier's predicted probability.
            # WHY: Each OvR classifier outputs its own probability independently.
            scores = np.column_stack(
                [self._sigmoid(X @ w + b) for w, b, _ in self._models]
            )
            # Normalize to sum to 1 (they may not naturally sum to 1 with OvR).
            # WHY: OvR classifiers are trained independently, so their outputs can sum
            # to more or less than 1. Normalizing gives valid probability distributions.
            scores /= scores.sum(axis=1, keepdims=True) + 1e-15
            return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each sample.

        WHY: Returns the most likely class for each sample by taking argmax of probabilities.
        """
        proba = self.predict_proba(X)
        # Map probability indices back to original class labels.
        # WHY: If classes are [0, 1, 2], argmax gives indices 0/1/2 which map directly.
        # If classes were [3, 7, 11], we'd need this mapping to return actual labels.
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data and split into train/val/test.

    WHY identical to sklearn version: Using the exact same data generation ensures
    fair comparison between the from-scratch and library implementations.
    """
    # Generate synthetic data with controlled informative and redundant features.
    # WHY: See logistic_regression_sklearn.py for detailed explanations of each parameter.
    X, y = make_classification(
        n_samples=n_samples,            # Total number of samples.
        n_features=n_features,          # Total feature count.
        n_informative=n_features // 2,  # Half carry real signal.
        n_redundant=n_features // 4,    # Quarter are linear combos (multicollinearity).
        n_classes=n_classes,            # Number of target classes.
        random_state=random_state,      # Reproducibility seed.
    )

    # 60/20/20 stratified split into train/validation/test.
    # WHY stratify: Maintains class proportions across all splits.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    # Standardize features (critical for gradient descent convergence).
    # WHY: Our from-scratch gradient descent is even MORE sensitive to feature scaling
    # than sklearn's optimized solvers. Without scaling, features on different scales
    # would dominate the gradient and cause divergence or very slow convergence.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on train only (no data leakage).
    X_val = scaler.transform(X_val)          # Apply train statistics to val.
    X_test = scaler.transform(X_test)        # Apply train statistics to test.

    # Log dataset dimensions for verification.
    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> LogisticRegressionNumpy:
    """Train a from-scratch Logistic Regression model.

    WHY this wrapper: Provides the same interface as the sklearn version, making it
    easy to swap implementations without changing the pipeline code.
    """
    # Default hyperparameters tuned for our synthetic dataset.
    defaults = dict(
        learning_rate=0.01,    # Moderate step size for stable convergence.
        n_epochs=1000,         # Enough iterations for the loss to plateau.
        lambda_reg=0.01,       # Mild L2 regularization to prevent overfitting.
        batch_size=64,         # Mini-batch SGD for faster training than full-batch.
        # WHY 64: A common batch size that balances gradient noise with computation speed.
        random_state=42,       # Reproducibility.
    )
    # Override defaults with user-provided hyperparameters.
    defaults.update(hyperparams)
    # Create and fit the model.
    model = LogisticRegressionNumpy(**defaults)
    model.fit(X_train, y_train)
    logger.info("Model trained with params: %s", defaults)
    return model


def _evaluate(model: LogisticRegressionNumpy, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics for the from-scratch model.

    WHY same metrics as sklearn: Fair comparison requires identical evaluation methodology.
    """
    # Generate predictions and probabilities using our from-scratch model.
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Compute AUC-ROC (binary vs multiclass handling).
    n_classes = len(np.unique(y))
    if n_classes == 2:
        auc = roc_auc_score(y, y_proba[:, 1])    # Binary: use positive class probability.
    else:
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    # Return the same metric dictionary as the sklearn version.
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: LogisticRegressionNumpy, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Validate the model on the validation set."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: LogisticRegressionNumpy, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Final test evaluation -- call only once with the best model."""
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """Compare multiple hyperparameter configurations for the NumPy implementation.

    WHY: The from-scratch implementation has different hyperparameters than sklearn
    (learning_rate, n_epochs, batch_size instead of solver, max_iter). Understanding
    how these affect convergence and final performance is essential for tuning.
    """
    configs = {
        "Small LR, Many Epochs": {
            # Config 1: Slow but stable convergence.
            # WHY: Small learning rate (0.001) means tiny steps, reducing the risk of
            # overshooting the minimum. But it needs many epochs (2000) to converge.
            # This is the "safe" option when you're unsure about the loss landscape.
            # Best for: complex loss landscapes, sensitive to learning rate.
            "learning_rate": 0.001,
            "n_epochs": 2000,
            "lambda_reg": 0.01,
            "batch_size": 64,
        },
        "Moderate LR (default)": {
            # Config 2: Balanced defaults that work for most problems.
            # WHY: learning_rate=0.01 is the "Goldilocks" zone for many problems.
            # 1000 epochs is usually enough for convergence.
            # Best for: general-purpose starting point.
            "learning_rate": 0.01,
            "n_epochs": 1000,
            "lambda_reg": 0.01,
            "batch_size": 64,
        },
        "Large LR, Few Epochs": {
            # Config 3: Fast training but risk of instability.
            # WHY: Large learning rate (0.1) takes big steps. If the loss surface is smooth,
            # this converges very quickly. But for complex surfaces, it may oscillate
            # around the minimum or even diverge. Few epochs (300) keep training fast.
            # Best for: quick prototyping, linearly separable data.
            "learning_rate": 0.1,
            "n_epochs": 300,
            "lambda_reg": 0.01,
            "batch_size": 64,
        },
        "Strong Regularization": {
            # Config 4: Heavy regularization to test its effect.
            # WHY: lambda_reg=0.5 strongly penalizes large weights, producing a very
            # simple model. Combined with full-batch gradient descent (batch_size=None
            # would be full, but we use 256 which is close), this tests whether the
            # data has enough signal to overcome heavy regularization.
            # Best for: noisy data or when overfitting is a serious concern.
            "learning_rate": 0.01,
            "n_epochs": 1000,
            "lambda_reg": 0.5,
            "batch_size": 256,
        },
    }

    print("\n" + "=" * 90)
    print("LOGISTIC REGRESSION (NumPy) - HYPERPARAMETER COMPARISON")
    print("=" * 90)
    print(f"{'Config':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 90)

    for name, params in configs.items():
        model = train(X_train, y_train, **params)
        metrics = _evaluate(model, X_val, y_val)
        print(
            f"{name:<30} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f}"
        )

    print("-" * 90)
    print("INTERPRETATION GUIDE:")
    print("  - Small LR: Most stable, but slowest to train. Use when convergence issues arise.")
    print("  - Moderate LR: Best default. Good balance of speed and stability.")
    print("  - Large LR: Fastest training. Risk of divergence if loss surface is complex.")
    print("  - Strong Regularization: Simplest model. Good if you suspect many noisy features.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Real-World Demo: Customer Churn Prediction (from scratch)
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate the from-scratch logistic regression on customer churn prediction.

    WHY same scenario as sklearn version: Direct comparison shows how the from-scratch
    implementation performs vs the library version on the same problem. The from-scratch
    version may converge to a slightly different solution due to SGD vs LBFGS optimization.
    """
    print("\n" + "=" * 90)
    print("REAL-WORLD DEMO: Customer Churn Prediction (NumPy From-Scratch)")
    print("=" * 90)

    # Set random seed for reproducibility.
    np.random.seed(42)
    n_samples = 2000

    # Generate the same realistic features as the sklearn version.
    # WHY same features: Enables direct comparison of from-scratch vs library results.
    tenure_months = np.random.uniform(0, 72, n_samples)
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 120)
    total_charges = tenure_months * monthly_charges + np.random.normal(0, 200, n_samples)
    contract_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    num_support_tickets = np.random.poisson(2, n_samples)
    has_online_backup = np.random.binomial(1, 0.4, n_samples)

    # Generate churn labels using the same business logic.
    churn_score = (
        -0.05 * tenure_months
        + 0.03 * monthly_charges
        - 0.8 * contract_type
        + 0.3 * num_support_tickets
        - 0.5 * has_online_backup
        + np.random.normal(0, 1, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    y = (np.random.random(n_samples) < churn_prob).astype(int)

    # Build feature matrix.
    X = np.column_stack([
        tenure_months, monthly_charges, total_charges,
        contract_type, num_support_tickets, has_online_backup,
    ])
    feature_names = [
        "tenure_months", "monthly_charges", "total_charges",
        "contract_type", "num_support_tickets", "has_online_backup",
    ]

    print(f"\nDataset: {n_samples} customers, {len(feature_names)} features")
    print(f"Churn rate: {y.mean():.1%}")

    # Split and scale (same as sklearn version for fair comparison).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train from-scratch model with tuned hyperparameters.
    # WHY these values: learning_rate=0.05 and 500 epochs provide good convergence
    # for this 6-feature dataset. lambda_reg=0.01 gives mild regularization.
    model = LogisticRegressionNumpy(
        learning_rate=0.05, n_epochs=500, lambda_reg=0.01, batch_size=64, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate on test set.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n--- Model Performance (From Scratch) ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_test, y_proba[:, 1]):.4f}")

    # Show learned weights as feature importance.
    # WHY: In logistic regression, weight magnitude indicates feature importance
    # and sign indicates direction of influence.
    print("\n--- Learned Weights (Feature Importance) ---")
    weights = model.weights_
    sorted_idx = np.argsort(np.abs(weights))[::-1]
    for i in sorted_idx:
        direction = "increases churn" if weights[i] > 0 else "decreases churn"
        print(f"  {feature_names[i]:<25} weight={weights[i]:>8.4f}  ({direction})")

    # Show convergence info from the training loss history.
    print(f"\n--- Training Convergence ---")
    print(f"  Initial loss: {model._losses[0]:.4f}")
    print(f"  Final loss:   {model._losses[-1]:.4f}")
    print(f"  Loss reduction: {(1 - model._losses[-1]/model._losses[0])*100:.1f}%")
    print("=" * 90 + "\n")


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
    """Optuna objective for the from-scratch implementation.

    WHY different params than sklearn: Our implementation has learning_rate, n_epochs,
    and batch_size instead of solver and penalty type. These control SGD behavior.
    """
    # Learning rate: searched on log scale because optimal values span orders of magnitude.
    lr = trial.suggest_float("learning_rate", 1e-4, 0.5, log=True)
    # Number of epochs: more epochs = more training time but potentially better convergence.
    n_epochs = trial.suggest_int("n_epochs", 200, 2000, step=200)
    # Regularization strength: log scale because effect is multiplicative.
    lambda_reg = trial.suggest_float("lambda_reg", 1e-6, 1.0, log=True)
    # Batch size: common power-of-2 values for GPU memory alignment.
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    model = train(
        X_train, y_train,
        learning_rate=lr, n_epochs=n_epochs,
        lambda_reg=lambda_reg, batch_size=batch_size,
    )
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search for the from-scratch implementation."""
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "learning_rate": ray_tune.loguniform(1e-4, 0.5),
        "n_epochs": ray_tune.choice([500, 1000, 1500]),
        "lambda_reg": ray_tune.loguniform(1e-6, 1.0),
        "batch_size": ray_tune.choice([32, 64, 128]),
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
    """Full pipeline for Logistic Regression (NumPy from scratch)."""
    logger.info("=" * 70)
    logger.info("Logistic Regression - NumPy From-Scratch Implementation")
    logger.info("=" * 70)

    # Step 1: Generate data (same as sklearn version for fair comparison).
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Baseline training with default hyperparameters.
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # Step 3: Compare different hyperparameter configurations.
    logger.info("\n--- Hyperparameter Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 4: Real-world customer churn demo.
    real_world_demo()

    # Step 5: Optuna hyperparameter optimization.
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=20,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Step 6: Retrain with best params and final test evaluation.
    best_model = train(X_train, y_train, **study.best_params)
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
