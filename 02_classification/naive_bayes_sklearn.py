"""
Naive Bayes Classifier - Scikit-Learn Implementation
=====================================================

Theory & Mathematics:
    Naive Bayes classifiers are a family of probabilistic classifiers based on
    applying Bayes' theorem with the "naive" assumption of conditional independence
    between every pair of features given the class.

    Bayes' Theorem:
        P(C_k | x) = P(x | C_k) * P(C_k) / P(x)

        where:
            P(C_k | x)  = posterior probability of class k given features x
            P(x | C_k)  = likelihood of features x given class k
            P(C_k)      = prior probability of class k
            P(x)        = evidence (normalizing constant, same for all classes)

    Naive Independence Assumption:
        P(x | C_k) = P(x_1 | C_k) * P(x_2 | C_k) * ... * P(x_n | C_k)
                    = prod_{i=1}^{n} P(x_i | C_k)

        WHY "naive": In reality, features are almost never independent. For example,
        in spam detection, "free" and "money" often co-occur. Despite this violation,
        Naive Bayes works surprisingly well because:
        1. Classification only needs the relative ordering of P(C_k|x), not exact values
        2. Errors in independence assumption tend to cancel out across features
        3. The model has very few parameters, reducing overfitting

    Classification Rule (MAP - Maximum A Posteriori):
        y_hat = argmax_k P(C_k) * prod_{i=1}^{n} P(x_i | C_k)

        In log space (to avoid numerical underflow from multiplying many small probs):
        y_hat = argmax_k [log P(C_k) + sum_{i=1}^{n} log P(x_i | C_k)]

    Variants (differ in how P(x_i | C_k) is modeled):

        1. GaussianNB: Assumes features are normally distributed within each class.
            P(x_i | C_k) = (1 / sqrt(2*pi*sigma_k^2)) * exp(-(x_i - mu_k)^2 / (2*sigma_k^2))
            WHERE: mu_k, sigma_k^2 are the mean and variance of feature i for class k.
            USE: Continuous features (measurements, sensor readings).

        2. MultinomialNB: Assumes features are counts following a multinomial distribution.
            P(x_i | C_k) = (N_ki + alpha) / (N_k + alpha * n_features)
            WHERE: N_ki = count of feature i in class k, alpha = smoothing parameter.
            USE: Text classification (word counts, TF-IDF), count data.

        3. BernoulliNB: Assumes features are binary (0/1).
            P(x_i | C_k) = p_ki^(x_i) * (1-p_ki)^(1-x_i)
            WHERE: p_ki = probability that feature i is 1 in class k.
            USE: Binary features (word presence/absence), boolean attributes.

Business Use Cases:
    - Email spam filtering: fast, accurate with word count features
    - Sentiment analysis: text classification with bag-of-words
    - Medical diagnosis: fast screening with independent test results
    - Document categorization: topic classification of articles
    - Real-time classification: extremely fast inference for streaming data

Advantages:
    - Extremely fast training and prediction (no iterative optimization)
    - Works well with small training sets (few parameters to estimate)
    - Naturally handles multi-class problems (no one-vs-rest needed)
    - Scales linearly with number of features and samples
    - Provides calibrated probabilities (with proper variant choice)
    - Robust to irrelevant features (they contribute equally to all classes)

Disadvantages:
    - Independence assumption is almost always violated in practice
    - Cannot model feature interactions (e.g., XOR patterns)
    - Zero-frequency problem: unseen feature values get probability 0
      (mitigated by Laplace smoothing / additive smoothing)
    - GaussianNB assumes bell-shaped distributions (violated by skewed data)
    - Decision boundary is always linear (in log-probability space)
    - Often outperformed by more complex models on large datasets
"""

# --- Standard library imports ---
import logging  # Structured logging for training progress and metrics
import warnings  # Warning suppression during HPO
from typing import Any, Dict, Tuple  # Type annotations for function signatures

# --- Third-party imports ---
import numpy as np  # Numerical computing for array operations
import optuna  # Bayesian hyperparameter optimization

# sklearn classifiers: three variants of Naive Bayes.
from sklearn.naive_bayes import (
    GaussianNB,       # For continuous features (assumes Gaussian distribution)
    MultinomialNB,    # For count data / term frequencies (multinomial distribution)
    BernoulliNB,      # For binary features (Bernoulli distribution)
)

from sklearn.datasets import make_classification  # Synthetic data generator
from sklearn.metrics import (
    accuracy_score,           # Overall correctness
    classification_report,    # Per-class precision, recall, F1
    f1_score,                 # Harmonic mean of precision and recall
    precision_score,          # Positive predictive value
    recall_score,             # Sensitivity
    roc_auc_score,            # Area under ROC curve
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # Feature scaling

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # Module-specific logger
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress warnings


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,      # Total samples
    n_features: int = 20,       # Total features
    n_classes: int = 2,         # Target classes
    random_state: int = 42,     # Random seed
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data and split into train/val/test.

    WHY: Synthetic data provides controlled conditions for benchmarking.
    The split ratio is 60/20/20 (train/val/test).

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Generate synthetic dataset with known signal-to-noise ratio.
    X, y = make_classification(
        n_samples=n_samples,            # Total data points
        n_features=n_features,          # Feature dimensionality
        n_informative=n_features // 2,  # Half carry real signal
        n_redundant=n_features // 4,    # Quarter are linear combos
        n_classes=n_classes,            # Number of classes
        random_state=random_state,      # Reproducibility
    )

    # First split: 60% train, 40% temp.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )

    # Second split: 50/50 of temp -> 20% val, 20% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    logger.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,                # Training features
    y_train: np.ndarray,                # Training labels
    model_type: str = "gaussian",       # NB variant: gaussian, multinomial, bernoulli
    var_smoothing: float = 1e-9,        # Variance smoothing (GaussianNB)
    alpha: float = 1.0,                 # Laplace smoothing (Multinomial/BernoulliNB)
) -> Any:
    """Train a Naive Bayes classifier using the specified variant.

    WHY separate variants: Different data types require different distributional
    assumptions. Using the wrong variant (e.g., MultinomialNB on negative values)
    will produce errors or poor results.

    Args:
        X_train: Training feature matrix.
        y_train: Training label array.
        model_type: Which NB variant to use.
            "gaussian" - for continuous features (assumes normal distribution)
            "multinomial" - for count/frequency data (non-negative)
            "bernoulli" - for binary features (0/1)
        var_smoothing: Smoothing for GaussianNB. Adds this fraction of the
            largest variance to all variances for numerical stability.
            WHY: Prevents division by zero when a feature has zero variance.
        alpha: Additive (Laplace) smoothing for Multinomial/BernoulliNB.
            WHY: Prevents zero probabilities for unseen feature values.
            alpha=1.0 is Laplace smoothing, alpha=0 is no smoothing.

    Returns:
        Trained sklearn Naive Bayes model.
    """
    logger.info(f"Training NB: type={model_type}, var_smoothing={var_smoothing}, alpha={alpha}")

    # Select and configure the appropriate NB variant.
    if model_type == "gaussian":  # Continuous features
        # GaussianNB estimates mean and variance of each feature per class.
        # WHY var_smoothing: Adds var_smoothing * max(sigma^2) to all variances
        # to prevent numerical issues with near-zero variance features.
        model = GaussianNB(var_smoothing=var_smoothing)  # Create Gaussian NB

    elif model_type == "multinomial":  # Count data
        # MultinomialNB requires non-negative features (counts or frequencies).
        # WHY: The multinomial distribution models counts; negative values are undefined.
        # We use MinMaxScaler to ensure non-negative features.
        model = MultinomialNB(alpha=alpha)  # Create Multinomial NB

    elif model_type == "bernoulli":  # Binary features
        # BernoulliNB expects binary features but can binarize continuous ones.
        # WHY binarize=0.0: Features > 0 become 1, <= 0 become 0. This is a
        # common default for converting continuous features to binary.
        model = BernoulliNB(alpha=alpha, binarize=0.0)  # Create Bernoulli NB

    else:
        raise ValueError(f"Unknown model_type: {model_type}")  # Invalid variant

    # Handle data preprocessing for MultinomialNB (requires non-negative features).
    if model_type == "multinomial":
        # Scale features to [0, 1] range for MultinomialNB compatibility.
        scaler = MinMaxScaler()  # Min-max scaler
        X_train_processed = scaler.fit_transform(X_train)  # Scale to [0, 1]
    else:
        X_train_processed = X_train  # No preprocessing needed

    # Fit the model.
    # WHY: NB training is O(n*d) - simply computes class priors and
    # feature statistics (mean/var for Gaussian, counts for Multinomial).
    model.fit(X_train_processed, y_train)  # Train the model

    # Compute and log training accuracy.
    train_preds = model.predict(X_train_processed)  # Predict on training data
    train_acc = accuracy_score(y_train, train_preds)  # Compute accuracy
    logger.info(f"Training accuracy: {train_acc:.4f}")  # Log result

    return model  # Return the trained model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: Any,             # Trained NB model
    X_val: np.ndarray,      # Validation features
    y_val: np.ndarray,      # Validation labels
    model_type: str = "gaussian",  # NB variant (for preprocessing)
    scaler: Any = None,     # Optional scaler for MultinomialNB
) -> Dict[str, float]:
    """Evaluate the Naive Bayes model on the validation set.

    WHY: Provides generalization estimate during hyperparameter tuning.

    Args:
        model: Trained sklearn NB model.
        X_val: Validation features.
        y_val: Validation labels.
        model_type: Which NB variant (affects preprocessing).
        scaler: Optional MinMaxScaler for MultinomialNB.

    Returns:
        Dictionary of metric names to values.
    """
    # Preprocess if needed for MultinomialNB.
    if model_type == "multinomial" and scaler is not None:
        X_val_processed = scaler.transform(X_val)  # Apply same scaling
    elif model_type == "multinomial":
        # If no scaler provided, clip to non-negative.
        X_val_processed = np.clip(X_val, 0, None)  # Ensure non-negative
    else:
        X_val_processed = X_val  # No preprocessing

    # Generate predictions.
    y_pred = model.predict(X_val_processed)  # Hard class predictions
    y_proba = model.predict_proba(X_val_processed)  # Class probabilities

    # Compute metrics.
    n_classes = len(np.unique(y_val))  # Number of classes
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),  # Overall correctness
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
    model: Any,             # Trained NB model
    X_test: np.ndarray,     # Test features
    y_test: np.ndarray,     # Test labels
    model_type: str = "gaussian",  # NB variant
    scaler: Any = None,     # Optional scaler
) -> Dict[str, float]:
    """Final evaluation on the held-out test set.

    WHY: Provides unbiased performance estimate after all tuning is done.

    Args:
        model: Trained sklearn NB model.
        X_test: Test features.
        y_test: Test labels.
        model_type: NB variant for preprocessing.
        scaler: Optional MinMaxScaler.

    Returns:
        Dictionary of metric names to values.
    """
    # Preprocess if needed.
    if model_type == "multinomial" and scaler is not None:
        X_test_processed = scaler.transform(X_test)
    elif model_type == "multinomial":
        X_test_processed = np.clip(X_test, 0, None)
    else:
        X_test_processed = X_test

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)

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
    """Optuna objective for Naive Bayes hyperparameter search.

    Searches over NB variant, smoothing parameters, and preprocessing.

    WHY: Even simple models benefit from tuning. The choice of NB variant
    and smoothing parameter significantly affects performance.

    Args:
        trial: Optuna Trial for hyperparameter suggestions.

    Returns:
        Validation F1 score to maximize.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Suggest model type.
    model_type = trial.suggest_categorical(
        "model_type", ["gaussian", "multinomial", "bernoulli"]
    )

    # Suggest variant-specific parameters.
    if model_type == "gaussian":
        var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True)
        alpha = 1.0  # Not used for GaussianNB
    else:
        var_smoothing = 1e-9  # Not used for non-Gaussian
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)

    model = train(X_train, y_train, model_type=model_type,
                  var_smoothing=var_smoothing, alpha=alpha)
    metrics = validate(model, X_val, y_val, model_type=model_type)

    return metrics["f1"]


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune hyperparameter search space for Naive Bayes.

    Returns:
        Dictionary defining the search space.
    """
    search_space = {
        "model_type": {"type": "choice", "values": ["gaussian", "multinomial", "bernoulli"]},
        "var_smoothing": {"type": "loguniform", "lower": 1e-12, "upper": 1e-1},
        "alpha": {"type": "loguniform", "lower": 0.01, "upper": 10.0},
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
    """Compare different Naive Bayes variants and smoothing values.

    Configurations:
    1. GaussianNB with default smoothing: Standard choice for continuous features.
    2. GaussianNB with high smoothing: More regularization via larger variance.
    3. MultinomialNB with Laplace smoothing: Standard for count data.
    4. BernoulliNB with Laplace smoothing: For binary feature interpretation.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    # Prepare MinMaxScaler for MultinomialNB.
    scaler = MinMaxScaler()  # Scale features to [0, 1]
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
    X_val_scaled = scaler.transform(X_val)  # Transform val

    configs = {
        "gaussian_default": {
            "params": {"model_type": "gaussian", "var_smoothing": 1e-9, "alpha": 1.0},
            "reasoning": (
                "GaussianNB with default var_smoothing=1e-9. The standard choice for "
                "continuous features. Assumes each feature follows a Gaussian distribution "
                "within each class. Minimal smoothing preserves learned statistics."
            ),
        },
        "gaussian_high_smoothing": {
            "params": {"model_type": "gaussian", "var_smoothing": 1e-2, "alpha": 1.0},
            "reasoning": (
                "GaussianNB with high var_smoothing=1e-2. Adding larger smoothing acts "
                "as regularization by inflating all variances, making the likelihood "
                "function flatter. Expected: slightly lower accuracy on clean data but "
                "potentially more robust to noisy or small datasets."
            ),
        },
        "multinomial_laplace": {
            "params": {"model_type": "multinomial", "var_smoothing": 1e-9, "alpha": 1.0},
            "reasoning": (
                "MultinomialNB with Laplace smoothing (alpha=1.0). Designed for count data "
                "but we force-scale features to [0,1]. Laplace smoothing ensures no feature "
                "gets zero probability. Expected: may underperform Gaussian on truly "
                "continuous data but excellent on count/frequency features."
            ),
        },
        "bernoulli_laplace": {
            "params": {"model_type": "bernoulli", "var_smoothing": 1e-9, "alpha": 1.0},
            "reasoning": (
                "BernoulliNB with Laplace smoothing. Binarizes features at threshold 0 "
                "(positive = 1, non-positive = 0). Discards magnitude information. "
                "Expected: lower accuracy on continuous data where magnitude matters, "
                "but effective when feature presence/absence is more informative than value."
            ),
        },
    }

    results = {}
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {name}")
        logger.info(f"Reasoning: {config['reasoning']}")

        model = train(X_train, y_train, **config["params"])

        # Use appropriate preprocessing for validation.
        model_type = config["params"]["model_type"]
        if model_type == "multinomial":
            metrics = validate(model, X_val_scaled, y_val, model_type=model_type)
        else:
            metrics = validate(model, X_val, y_val, model_type=model_type)

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
    """Demonstrate Naive Bayes on email spam detection.

    Domain: Email spam filtering.
    Features: word_freq_free, word_freq_money, word_freq_offer,
              capital_run_length_avg, capital_run_length_total
    Target: spam (0 = legitimate, 1 = spam)

    WHY this domain: Spam filtering is the classic Naive Bayes application because:
    1. Bag-of-words features are approximately independent (naive assumption holds better)
    2. Training is extremely fast (important for updating on new spam patterns)
    3. Probabilities are useful for spam scoring (not just binary classification)
    4. MultinomialNB on word counts is one of the most successful NB applications
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Email Spam Detection")
    logger.info("=" * 60)

    np.random.seed(42)  # Reproducibility
    n_samples = 800     # Synthetic emails

    # Generate realistic spam detection features.
    # WHY these features: Based on the UCI Spambase dataset feature set.

    # word_freq_free: frequency of the word "free" in the email.
    # WHY: "free" is a strong spam indicator (free offers, free trials).
    word_freq_free = np.random.exponential(0.5, n_samples)  # Exponential distribution

    # word_freq_money: frequency of the word "money".
    # WHY: Financial terms are common in spam (make money fast, send money).
    word_freq_money = np.random.exponential(0.3, n_samples)  # Lower mean for legitimate

    # word_freq_offer: frequency of the word "offer".
    # WHY: Promotional language is a spam signal.
    word_freq_offer = np.random.exponential(0.4, n_samples)  # Moderate frequency

    # capital_run_length_avg: average length of consecutive capital letters.
    # WHY: Spam often uses ALL CAPS for emphasis (BUY NOW, FREE).
    capital_run_length = np.random.exponential(3.0, n_samples)  # Avg caps run

    # capital_run_total: total number of capital letters.
    # WHY: Overall capitalization level correlates with spam.
    capital_total = np.random.exponential(50.0, n_samples)  # Total caps

    X = np.column_stack([
        word_freq_free, word_freq_money, word_freq_offer,
        capital_run_length, capital_total
    ])
    feature_names = [
        "word_freq_free", "word_freq_money", "word_freq_offer",
        "capital_run_length_avg", "capital_run_length_total"
    ]

    # Create spam labels based on feature values.
    # WHY this formula: Emails with more "free", "money", and caps are more likely spam.
    spam_score = (
        0.5 * word_freq_free        # "free" is a strong spam signal
        + 0.7 * word_freq_money     # "money" is an even stronger signal
        + 0.3 * word_freq_offer     # "offer" is a moderate signal
        + 0.1 * capital_run_length  # Caps runs are a mild signal
        + 0.01 * capital_total      # Total caps is a weak signal
    )
    probability = 1.0 / (1.0 + np.exp(-spam_score + 2.0))  # Sigmoid with offset
    y = (probability + np.random.normal(0, 0.05, n_samples) > 0.5).astype(int)

    # Split data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Log feature statistics.
    logger.info("\nFeature Statistics (Training Set):")
    for i, name in enumerate(feature_names):
        logger.info(
            f"  {name}: mean={X_train[:, i].mean():.2f}, "
            f"std={X_train[:, i].std():.2f}"
        )
    logger.info(f"\nClass distribution: legitimate={np.sum(y_train==0)}, spam={np.sum(y_train==1)}")

    # Train GaussianNB (best for continuous features).
    logger.info("\n--- GaussianNB ---")
    gauss_model = train(X_train, y_train, model_type="gaussian")
    validate(gauss_model, X_val, y_val)
    test(gauss_model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete Naive Bayes (sklearn) pipeline.

    Steps: data -> baseline -> validation -> comparison -> HPO -> test -> demo.
    """
    logger.info("=" * 60)
    logger.info("Naive Bayes Classifier - Scikit-Learn Implementation")
    logger.info("=" * 60)

    # Step 1: Generate data.
    logger.info("\n--- Step 1: Generating Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Train baseline (GaussianNB).
    logger.info("\n--- Step 2: Training Baseline (GaussianNB) ---")
    baseline = train(X_train, y_train, model_type="gaussian")

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

    # Step 6: Ray Tune search space.
    logger.info("\n--- Step 6: Ray Tune Search Space ---")
    ray_tune_search()

    # Step 7: Train best model.
    logger.info("\n--- Step 7: Training Best Model ---")
    best_params = study.best_trial.params
    best_model = train(X_train, y_train, **best_params)

    # Step 8: Test evaluation.
    logger.info("\n--- Step 8: Final Test Evaluation ---")
    model_type = best_params.get("model_type", "gaussian")
    test(best_model, X_test, y_test, model_type=model_type)

    # Step 9: Real-world demo.
    logger.info("\n--- Step 9: Real-World Demo ---")
    real_world_demo()

    logger.info("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    main()
