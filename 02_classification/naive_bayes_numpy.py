"""
Naive Bayes Classifier - NumPy From-Scratch Implementation (Gaussian)
=====================================================================

Theory & Mathematics:
    This implements Gaussian Naive Bayes from scratch using only NumPy.
    Gaussian NB assumes that continuous features follow a normal (Gaussian)
    distribution within each class.

    Bayes' Theorem (Foundation):
        P(C_k | x) = P(x | C_k) * P(C_k) / P(x)

        For classification, we use the MAP (Maximum A Posteriori) rule:
        y_hat = argmax_k P(C_k | x) = argmax_k P(x | C_k) * P(C_k)
        (We can ignore P(x) because it's the same for all classes.)

    Step 1 - Compute Class Priors P(C_k):
        P(C_k) = count(y == k) / n_total
        WHY: The prior represents our belief about class frequency before
        seeing any features. If 60% of emails are spam, P(spam) = 0.6.

    Step 2 - Compute Class-Conditional Likelihoods P(x_i | C_k):
        Assuming Gaussian distribution for each feature within each class:
        P(x_i | C_k) = (1 / sqrt(2 * pi * sigma_{ik}^2)) *
                        exp(-(x_i - mu_{ik})^2 / (2 * sigma_{ik}^2))
        where:
            mu_{ik} = mean of feature i for class k
            sigma_{ik}^2 = variance of feature i for class k

        WHY Gaussian: It's the most natural assumption for continuous features.
        Only requires estimating two parameters per feature per class (mean, variance).

    Step 3 - Apply Naive Independence Assumption:
        P(x | C_k) = P(x_1 | C_k) * P(x_2 | C_k) * ... * P(x_n | C_k)
                    = prod_{i=1}^{n} P(x_i | C_k)

    Step 4 - Combine with Log Trick (numerical stability):
        log P(C_k | x) proportional to log P(C_k) + sum_{i=1}^{n} log P(x_i | C_k)

        In log space:
        log P(x_i | C_k) = -0.5 * log(2*pi*sigma_{ik}^2) - (x_i - mu_{ik})^2 / (2*sigma_{ik}^2)

        WHY log space: Multiplying many small probabilities causes numerical
        underflow (result becomes 0.0). Working in log space converts products
        to sums, which are numerically stable.

    Laplace Smoothing (Additive Smoothing):
        sigma_{ik}^2 = variance + epsilon
        WHY: If a feature has zero variance in a class (all values identical),
        the Gaussian PDF becomes a delta function (infinite at the mean, zero
        everywhere else). Adding epsilon prevents division by zero and infinite
        log-likelihood values. This is the continuous analog of Laplace smoothing.

Business Use Cases:
    - Spam filtering with continuous feature scores
    - Medical diagnosis with lab test results
    - Anomaly detection in sensor readings
    - Fast baseline classifier for any continuous-feature dataset
    - Real-time classification where speed is critical

Advantages:
    - O(n * d * k) training: count-based, no iterative optimization
    - O(d * k) prediction per sample: just evaluate Gaussians
    - Extremely simple implementation (few dozen lines of core code)
    - Naturally outputs calibrated probabilities
    - Works well with small training sets

Disadvantages:
    - Gaussian assumption: fails for multimodal or skewed distributions
    - Independence assumption: ignores feature correlations
    - Linear decision boundary (in log-probability space)
    - Cannot capture feature interactions
    - Sensitive to feature scaling (affects variance estimates)
"""

# --- Standard library imports ---
import logging  # Structured logging for training progress
import warnings  # Warning suppression during HPO
from typing import Any, Dict, Optional, Tuple  # Type annotations

# --- Third-party imports ---
import numpy as np  # Numerical computing for all array operations
import optuna  # Bayesian hyperparameter optimization

from sklearn.datasets import make_classification  # Synthetic data generator
from sklearn.metrics import (
    accuracy_score,           # Overall correctness
    classification_report,    # Per-class breakdown
    f1_score,                 # Harmonic mean of precision and recall
    precision_score,          # Positive predictive value
    recall_score,             # Sensitivity
    roc_auc_score,            # Area under ROC curve
)
from sklearn.model_selection import train_test_split  # Data splitting

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Gaussian Naive Bayes (From Scratch)
# ---------------------------------------------------------------------------

class GaussianNBFromScratch:
    """Gaussian Naive Bayes classifier implemented from scratch.

    This implements the complete GNB algorithm:
    1. Compute class priors from label frequencies
    2. Estimate per-class, per-feature Gaussian parameters (mean, variance)
    3. Predict using log-posterior computation with Bayes' theorem

    WHY from scratch: Understanding how NB works internally reveals why it's
    so fast, when the Gaussian assumption breaks down, and how smoothing
    prevents numerical issues.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """Initialize the Gaussian Naive Bayes classifier.

        Args:
            var_smoothing: Additive smoothing for variance estimates.
                WHY: Prevents zero variance (which causes division by zero in
                the Gaussian PDF). Analogous to Laplace smoothing for discrete NB.
                Value 1e-9 is sklearn's default; larger values add more regularization.
        """
        self.var_smoothing = var_smoothing  # Store smoothing parameter
        self.classes: Optional[np.ndarray] = None  # Unique class labels
        self.class_priors: Optional[np.ndarray] = None  # P(C_k) for each class
        self.means: Optional[np.ndarray] = None  # mu_{ik}: mean per feature per class
        self.variances: Optional[np.ndarray] = None  # sigma_{ik}^2: variance per feature per class

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBFromScratch":
        """Fit the Gaussian NB model by computing priors and class statistics.

        Training is simply computing:
        1. Class priors: P(C_k) = count(y==k) / n_total
        2. Class-conditional means: mu_{ik} = mean(X[y==k, i])
        3. Class-conditional variances: sigma_{ik}^2 = var(X[y==k, i]) + smoothing

        WHY this is so fast: No iterative optimization (gradient descent, etc.).
        Just counting and computing means/variances. O(n * d * k) time complexity.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self (for method chaining).
        """
        # Get unique class labels.
        self.classes = np.unique(y)  # Sorted unique classes
        n_classes = len(self.classes)  # Number of classes
        n_features = X.shape[1]       # Number of features

        # Initialize arrays for storing per-class statistics.
        self.class_priors = np.zeros(n_classes)           # Shape: (n_classes,)
        self.means = np.zeros((n_classes, n_features))    # Shape: (n_classes, n_features)
        self.variances = np.zeros((n_classes, n_features))  # Shape: (n_classes, n_features)

        # Compute statistics for each class.
        for idx, cls in enumerate(self.classes):  # Iterate over each class
            # Select samples belonging to this class.
            X_class = X[y == cls]  # Shape: (n_samples_in_class, n_features)

            # Compute class prior: fraction of samples in this class.
            # WHY: P(C_k) = n_k / n_total. This is the maximum likelihood estimate.
            self.class_priors[idx] = X_class.shape[0] / X.shape[0]  # P(C_k)

            # Compute mean of each feature for this class.
            # WHY: mu_{ik} is the MLE for the Gaussian mean parameter.
            self.means[idx] = X_class.mean(axis=0)  # Mean along sample axis

            # Compute variance of each feature for this class, plus smoothing.
            # WHY var + smoothing: Pure variance can be zero (if all values are
            # identical), causing division by zero in the Gaussian PDF.
            # Adding smoothing ensures finite, well-defined likelihoods.
            self.variances[idx] = X_class.var(axis=0) + self.var_smoothing  # Variance + epsilon

        # Log the fitted parameters summary.
        logger.info(f"Fitted GaussianNB: {n_classes} classes, {n_features} features")
        for idx, cls in enumerate(self.classes):
            logger.info(f"  Class {cls}: prior={self.class_priors[idx]:.4f}, "
                       f"n_samples={np.sum(y == cls)}")

        return self  # Return self for method chaining

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log-likelihood log P(x | C_k) for all classes.

        Uses the Gaussian PDF formula in log space:
        log P(x_i | C_k) = -0.5 * log(2*pi*sigma_{ik}^2) - (x_i - mu_{ik})^2 / (2*sigma_{ik}^2)

        Then sums across features (naive independence assumption):
        log P(x | C_k) = sum_{i=1}^{n} log P(x_i | C_k)

        WHY log space: Multiplying probabilities in [0,1] quickly underflows to 0.
        Adding log-probabilities is numerically stable.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Log-likelihood matrix of shape (n_samples, n_classes).
        """
        n_samples = X.shape[0]  # Number of samples
        n_classes = len(self.classes)  # Number of classes
        log_likelihoods = np.zeros((n_samples, n_classes))  # Initialize output

        for idx in range(n_classes):  # Compute for each class
            # Get class-specific parameters.
            mean = self.means[idx]      # mu_{ik}: shape (n_features,)
            var = self.variances[idx]   # sigma_{ik}^2: shape (n_features,)

            # Compute log of the normalization constant: -0.5 * log(2*pi*sigma^2).
            # WHY: This is the constant part of the Gaussian log-PDF. It depends on
            # the variance but not on the data point x, so it still matters for
            # comparing classes (different classes have different variances).
            log_norm = -0.5 * np.log(2 * np.pi * var)  # Shape: (n_features,)

            # Compute the exponent: -(x - mu)^2 / (2*sigma^2).
            # WHY: This is the data-dependent part of the Gaussian log-PDF.
            # Points closer to the class mean get higher (less negative) values.
            log_exp = -0.5 * ((X - mean) ** 2) / var  # Shape: (n_samples, n_features)

            # Sum across features (naive independence: product -> sum in log space).
            # log P(x|C_k) = sum_i [log_norm_i + log_exp_i]
            log_likelihoods[:, idx] = np.sum(log_norm + log_exp, axis=1)  # Sum features

        return log_likelihoods  # Shape: (n_samples, n_classes)

    def _log_posterior(self, X: np.ndarray) -> np.ndarray:
        """Compute log-posterior log P(C_k | x) for all classes (up to a constant).

        log P(C_k | x) = log P(C_k) + log P(x | C_k) + const
        (We drop the constant log P(x) because it's the same for all classes.)

        WHY: The posterior combines the prior belief about class frequency
        with the evidence from the features. The MAP prediction is simply
        the class with the highest log-posterior.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Log-posterior matrix of shape (n_samples, n_classes).
        """
        # Compute log class priors.
        # WHY log: Converts the prior multiplication to addition in log space.
        log_priors = np.log(self.class_priors)  # Shape: (n_classes,)

        # Compute log likelihoods for all classes.
        log_likelihoods = self._log_likelihood(X)  # Shape: (n_samples, n_classes)

        # Combine: log posterior = log prior + log likelihood.
        # WHY addition: In log space, P(C_k|x) proportional to P(C_k) * P(x|C_k)
        # becomes log P(C_k|x) = log P(C_k) + log P(x|C_k) + const.
        log_posteriors = log_priors + log_likelihoods  # Broadcasting: (n_samples, n_classes)

        return log_posteriors  # Shape: (n_samples, n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using MAP (Maximum A Posteriori) decision rule.

        y_hat = argmax_k log P(C_k | x) = argmax_k [log P(C_k) + log P(x|C_k)]

        WHY argmax: We want the most probable class given the observed features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        log_posteriors = self._log_posterior(X)  # Compute log posteriors
        # Argmax across classes to find the MAP prediction.
        class_indices = np.argmax(log_posteriors, axis=1)  # Index of highest posterior
        # Map indices back to actual class labels.
        return self.classes[class_indices]  # Return class labels (not indices)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the softmax of log-posteriors.

        WHY softmax: Converts log-posteriors to proper probabilities (non-negative,
        sum to 1). The softmax function is: P(k) = exp(log_p_k) / sum_j exp(log_p_j).
        We use the log-sum-exp trick for numerical stability.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        log_posteriors = self._log_posterior(X)  # Shape: (n_samples, n_classes)

        # Apply softmax with log-sum-exp trick for numerical stability.
        # WHY log-sum-exp: Without it, exp(large negative) underflows and
        # exp(large positive) overflows. Subtracting the max stabilizes both.
        log_max = np.max(log_posteriors, axis=1, keepdims=True)  # Max per sample
        # Subtract max before exp to prevent overflow.
        exp_posteriors = np.exp(log_posteriors - log_max)  # Shifted exponentials
        # Normalize to sum to 1.
        probabilities = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)

        return probabilities  # Shape: (n_samples, n_classes)


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

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_features // 2, n_redundant=n_features // 4,
        n_classes=n_classes, random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    logger.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    var_smoothing: float = 1e-9,
) -> GaussianNBFromScratch:
    """Train the from-scratch Gaussian NB on training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        var_smoothing: Variance smoothing parameter.

    Returns:
        Trained GaussianNBFromScratch model.
    """
    logger.info(f"Training GaussianNB (from scratch): var_smoothing={var_smoothing}")

    model = GaussianNBFromScratch(var_smoothing=var_smoothing)  # Create model
    model.fit(X_train, y_train)  # Fit model

    train_preds = model.predict(X_train)  # Training predictions
    train_acc = accuracy_score(y_train, train_preds)  # Training accuracy
    logger.info(f"Training accuracy: {train_acc:.4f}")

    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: GaussianNBFromScratch,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate on validation set.

    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_val)  # Hard predictions
    y_proba = model.predict_proba(X_val)  # Probability predictions

    n_classes = len(np.unique(y_val))
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_val, y_proba[:, 1]) if n_classes == 2 else 0.0,
    }

    logger.info("Validation Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test(
    model: GaussianNBFromScratch,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Final evaluation on test set.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    n_classes = len(np.unique(y_test))
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba[:, 1]) if n_classes == 2 else 0.0,
    }

    logger.info("=" * 50)
    logger.info("TEST SET RESULTS (Final Evaluation):")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    logger.info("=" * 50)
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial) -> float:
    """Optuna objective for Gaussian NB from scratch.

    The main hyperparameter is var_smoothing, which controls how much we
    regularize the variance estimates.

    Args:
        trial: Optuna Trial for parameter suggestions.

    Returns:
        Validation F1 score.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Suggest var_smoothing across a wide log range.
    # WHY log scale: var_smoothing spans many orders of magnitude (1e-12 to 1.0).
    var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1.0, log=True)

    model = train(X_train, y_train, var_smoothing=var_smoothing)
    metrics = validate(model, X_val, y_val)

    return metrics["f1"]


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune search space for Gaussian NB.

    Returns:
        Dictionary defining the search space.
    """
    search_space = {
        "var_smoothing": {"type": "loguniform", "lower": 1e-12, "upper": 1.0},
    }
    logger.info("Ray Tune search space:")
    for param, config in search_space.items():
        logger.info(f"  {param}: {config}")
    return search_space


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare different smoothing values for Gaussian NB.

    Configurations:
    1. No smoothing (0): Raw variance estimates, risk of division by zero.
    2. Minimal smoothing (1e-9): Default, tiny regularization.
    3. Moderate smoothing (1e-3): Noticeable regularization effect.
    4. Heavy smoothing (1.0): Strong regularization, flattens distributions.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    configs = {
        "no_smoothing": {
            "params": {"var_smoothing": 1e-15},
            "reasoning": (
                "Near-zero smoothing (1e-15). Uses raw variance estimates with minimal "
                "numerical protection. Risk: if any feature has zero variance in a class, "
                "the PDF becomes a delta function causing numerical issues. Best case: "
                "preserves true statistics for well-behaved data."
            ),
        },
        "minimal_smoothing_1e-9": {
            "params": {"var_smoothing": 1e-9},
            "reasoning": (
                "Default sklearn smoothing (1e-9). Adds a tiny amount to all variances "
                "to prevent division by zero while having negligible effect on predictions "
                "for features with reasonable variance. This is the sweet spot for most "
                "datasets: safe yet unintrusive."
            ),
        },
        "moderate_smoothing_1e-3": {
            "params": {"var_smoothing": 1e-3},
            "reasoning": (
                "Moderate smoothing (1e-3). Noticeably inflates all variances, making "
                "the Gaussian PDFs wider and flatter. Effect: reduces the influence of "
                "features with small variance, acts as implicit feature weighting. "
                "May help when features have unreliable variance estimates (small data)."
            ),
        },
        "heavy_smoothing_1.0": {
            "params": {"var_smoothing": 1.0},
            "reasoning": (
                "Heavy smoothing (1.0). Adds 1.0 to all variances, significantly "
                "flattening all Gaussian PDFs. The classifier becomes more 'prior-driven' "
                "because likelihoods are less discriminative. Expected: underfitting on "
                "clean data but potentially useful for very noisy or high-dimensional data."
            ),
        },
    }

    results = {}
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {name}")
        logger.info(f"Reasoning: {config['reasoning']}")

        model = train(X_train, y_train, **config["params"])
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY:")
    for name, metrics in results.items():
        logger.info(f"  {name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate from-scratch Gaussian NB on email spam detection.

    Domain: Email spam filtering.
    Features: word_freq_free, word_freq_money, word_freq_offer,
              capital_run_length_avg, capital_run_length_total
    Target: spam (0 = legitimate, 1 = spam)

    WHY this domain: Spam detection is the quintessential NB application.
    The independence assumption holds reasonably well for word frequencies.
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Email Spam Detection (From Scratch)")
    logger.info("=" * 60)

    np.random.seed(42)
    n_samples = 800

    # Generate spam-like features with realistic distributions.
    word_freq_free = np.random.exponential(0.5, n_samples)  # Freq of "free"
    word_freq_money = np.random.exponential(0.3, n_samples)  # Freq of "money"
    word_freq_offer = np.random.exponential(0.4, n_samples)  # Freq of "offer"
    capital_run_length = np.random.exponential(3.0, n_samples)  # Avg caps run
    capital_total = np.random.exponential(50.0, n_samples)  # Total caps

    X = np.column_stack([
        word_freq_free, word_freq_money, word_freq_offer,
        capital_run_length, capital_total
    ])
    feature_names = [
        "word_freq_free", "word_freq_money", "word_freq_offer",
        "capital_run_length_avg", "capital_run_length_total"
    ]

    # Generate spam labels.
    spam_score = (
        0.5 * word_freq_free + 0.7 * word_freq_money + 0.3 * word_freq_offer
        + 0.1 * capital_run_length + 0.01 * capital_total
    )
    probability = 1.0 / (1.0 + np.exp(-spam_score + 2.0))
    y = (probability + np.random.normal(0, 0.05, n_samples) > 0.5).astype(int)

    # Split data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info("\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        logger.info(f"  {name}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}")

    logger.info(f"Class dist: legit={np.sum(y_train==0)}, spam={np.sum(y_train==1)}")

    model = train(X_train, y_train, var_smoothing=1e-9)
    validate(model, X_val, y_val)
    test(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete Gaussian NB (from scratch) pipeline."""
    logger.info("=" * 60)
    logger.info("Naive Bayes (Gaussian) - NumPy From-Scratch Implementation")
    logger.info("=" * 60)

    # Step 1: Generate data.
    logger.info("\n--- Step 1: Generating Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Train baseline.
    logger.info("\n--- Step 2: Training Baseline ---")
    baseline = train(X_train, y_train)

    # Step 3: Validate.
    logger.info("\n--- Step 3: Validating Baseline ---")
    validate(baseline, X_val, y_val)

    # Step 4: Compare parameter sets.
    logger.info("\n--- Step 4: Comparing Parameter Sets ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 5: Optuna HPO.
    logger.info("\n--- Step 5: Optuna HPO ---")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(optuna_objective, n_trials=20)
    logger.info(f"Best F1: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    # Step 6: Ray Tune.
    logger.info("\n--- Step 6: Ray Tune Search Space ---")
    ray_tune_search()

    # Step 7: Train best.
    logger.info("\n--- Step 7: Training Best Model ---")
    best_model = train(X_train, y_train, **study.best_trial.params)

    # Step 8: Test.
    logger.info("\n--- Step 8: Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Step 9: Real-world demo.
    logger.info("\n--- Step 9: Real-World Demo ---")
    real_world_demo()

    logger.info("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    main()
