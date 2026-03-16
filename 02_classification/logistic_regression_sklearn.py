"""
Logistic Regression - Scikit-Learn Implementation
===================================================

Theory & Mathematics:
    Logistic Regression is a linear model for binary and multiclass classification.
    Despite its name, it is a classification algorithm, not a regression algorithm.

    The model applies the logistic (sigmoid) function to a linear combination of
    input features to produce a probability estimate:

        z = w^T * x + b
        P(y=1|x) = sigma(z) = 1 / (1 + exp(-z))

    For multiclass problems, the softmax function generalizes the sigmoid:

        P(y=k|x) = exp(z_k) / sum_j(exp(z_j))

    The model is trained by maximizing the log-likelihood (equivalently, minimizing
    the negative log-likelihood / cross-entropy loss):

        L(w) = -1/N * sum_{i=1}^{N} [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]

    Regularization can be applied to prevent overfitting:
        - L1 (Lasso): adds |w| penalty, encourages sparsity
        - L2 (Ridge): adds ||w||^2 penalty, discourages large weights
        - Elastic Net: combination of L1 and L2

    Optimization solvers available in scikit-learn:
        - 'lbfgs': Limited-memory BFGS, good for small-to-medium datasets
        - 'liblinear': Coordinate descent, good for small datasets and L1
        - 'newton-cg': Newton's method with conjugate gradient
        - 'sag'/'saga': Stochastic average gradient, good for large datasets

Business Use Cases:
    - Credit scoring and loan default prediction
    - Customer churn prediction
    - Email spam classification
    - Medical diagnosis (disease present/absent)
    - Click-through rate prediction in advertising

Advantages:
    - Simple, interpretable, and fast to train
    - Outputs well-calibrated probabilities
    - Works well with linearly separable data
    - Low variance, less prone to overfitting with regularization
    - Feature importance via coefficient magnitudes
    - Scales well to large datasets with SAG/SAGA solvers

Disadvantages:
    - Assumes linear decision boundary (may underfit complex patterns)
    - Sensitive to multicollinearity among features
    - Requires feature engineering for nonlinear relationships
    - Performance degrades with many irrelevant features without regularization
    - Not ideal for highly imbalanced datasets without adjustments

Hyperparameters:
    - C: Inverse of regularization strength (smaller = stronger regularization)
    - penalty: Type of regularization ('l1', 'l2', 'elasticnet', 'none')
    - solver: Optimization algorithm ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
    - max_iter: Maximum iterations for solver convergence
    - class_weight: Weights for classes to handle imbalance
"""

# --- Standard library imports ---
# logging: used for structured output messages throughout the pipeline.
# WHY: print() is fine for scripts, but logging provides severity levels (INFO, DEBUG, WARNING)
# and can be redirected to files or monitoring systems in production.
import logging

# warnings: used to suppress noisy convergence/deprecation warnings from sklearn.
# WHY: During hyperparameter search, many configurations will trigger convergence warnings.
# Suppressing them keeps the output clean and focused on metrics.
import warnings

# typing: provides type hints for function signatures to improve code readability.
# WHY: Type annotations serve as documentation and enable IDE autocompletion / static analysis.
from typing import Any, Dict, Tuple

# --- Third-party imports ---
# numpy: the foundational numerical computing library for array operations.
# WHY: All ML data flows through numpy arrays; sklearn uses numpy internally.
import numpy as np

# optuna: a Bayesian hyperparameter optimization framework.
# WHY: Optuna uses Tree-structured Parzen Estimator (TPE) to intelligently search
# the hyperparameter space, converging faster than random search or grid search.
import optuna

# make_classification: generates synthetic classification datasets.
# WHY: Synthetic data lets us control the number of informative/redundant features,
# class balance, and noise level, making it ideal for benchmarking algorithms.
from sklearn.datasets import make_classification

# LogisticRegression: sklearn's optimized logistic regression implementation.
# WHY: It wraps liblinear and LBFGS C/Fortran code for production-grade speed,
# and supports L1/L2/ElasticNet regularization out of the box.
from sklearn.linear_model import LogisticRegression

# Metrics imports: each metric captures a different aspect of classification performance.
from sklearn.metrics import (
    accuracy_score,           # Overall correctness: (TP+TN) / total
    classification_report,    # Per-class precision, recall, F1 summary
    confusion_matrix,         # Shows TP, TN, FP, FN breakdown
    f1_score,                 # Harmonic mean of precision and recall
    precision_score,          # Of all positive predictions, how many are correct
    recall_score,             # Of all actual positives, how many did we find
    roc_auc_score,            # Area under the ROC curve; threshold-independent
)

# train_test_split: splits data into training/validation/test sets.
# WHY: We need separate sets for training (learn), validation (tune), and test (final eval)
# to get an unbiased estimate of model performance on unseen data.
from sklearn.model_selection import train_test_split

# StandardScaler: standardizes features by removing the mean and scaling to unit variance.
# WHY: Logistic regression uses gradient-based optimization. Features on different scales
# (e.g., age 0-100 vs income 0-1,000,000) cause elongated loss surfaces and slow convergence.
# Standardization makes the loss surface more spherical, enabling faster and more stable training.
from sklearn.preprocessing import StandardScaler

# --- Logging configuration ---
# Set up logging to display timestamps, severity level, and message.
# WHY: level=INFO means we see training progress and metrics but not verbose DEBUG messages.
# In production, you might change this to WARNING to reduce noise.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Create a module-level logger so all functions in this file share the same logger.
# WHY: __name__ makes the logger name match the module name, useful in multi-module projects.
logger = logging.getLogger(__name__)

# Suppress sklearn warnings about solver convergence and label encoding.
# WHY: During Optuna hyperparameter search, some parameter combos may not converge
# within max_iter. These warnings are expected and would flood the output.
warnings.filterwarnings("ignore", category=UserWarning)


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
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Generate a synthetic classification dataset with controlled complexity.
    # WHY: make_classification gives us full control over the signal-to-noise ratio.
    # n_informative features carry real signal; n_redundant are linear combos of informative
    # features (simulating multicollinearity); remaining features are pure noise.
    X, y = make_classification(
        n_samples=n_samples,            # Total number of samples to generate.
        n_features=n_features,          # Total feature dimensionality.
        n_informative=n_features // 2,  # Half the features carry real signal.
        # WHY: This tests whether the model can identify relevant features.
        n_redundant=n_features // 4,    # Quarter are linear combos of informative ones.
        # WHY: Tests robustness to multicollinearity, which LR can struggle with.
        n_classes=n_classes,            # Number of target classes (2 for binary).
        random_state=random_state,      # Fix seed for reproducible experiments.
        # WHY: Reproducibility is critical for fair comparison across models/configs.
    )

    # Split dataset with stratification to maintain class balance across splits.
    # WHY: In classification, especially with imbalanced classes, random splits
    # could put all minority class samples in one split. Stratification ensures
    # each split has the same class proportion as the original dataset.
    # test_size=0.4 means 60% train, then we split the remaining 40% into
    # 20% validation + 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    # Split the temp set (40%) into equal validation (20%) and test (20%) sets.
    # WHY: Validation set is used for hyperparameter tuning (Optuna), test set
    # is used only once at the very end for unbiased final evaluation.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Standardize features: subtract mean, divide by standard deviation.
    # WHY: Logistic regression's gradient descent converges much faster when features
    # are on the same scale. Without scaling, features with large magnitudes dominate
    # the gradient updates, causing slow convergence or numerical instability.
    scaler = StandardScaler()
    # fit_transform on training data: learns mean and std, then transforms.
    # WHY: We only fit on training data to prevent data leakage. The validation/test
    # sets should be treated as "unseen" data that uses the training set's statistics.
    X_train = scaler.fit_transform(X_train)
    # transform only (no fit) on val/test: uses training set's mean and std.
    # WHY: In production, you would save this scaler and apply it to new data.
    # Using test-set statistics would be "peeking" at future data (data leakage).
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Log the dataset sizes so we can verify the split ratios are correct.
    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
        n_features, n_classes,
    )
    # Return all six arrays: features and labels for each of the three splits.
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        **hyperparams: Keyword arguments forwarded to LogisticRegression.

    Returns:
        Trained LogisticRegression model.
    """
    # Define sensible default hyperparameters that work well for most problems.
    # WHY: These defaults represent a well-regularized model with a robust solver.
    defaults = dict(
        C=1.0,              # Inverse regularization strength. C=1.0 is a balanced default.
        # WHY: Smaller C = stronger regularization (simpler model, less overfitting).
        # Larger C = weaker regularization (more complex model, risk of overfitting).
        penalty="l2",       # L2 regularization penalizes large weights with ||w||^2.
        # WHY: L2 shrinks all weights toward zero but rarely makes them exactly zero.
        # This works well when most features are somewhat informative.
        solver="lbfgs",     # Limited-memory BFGS quasi-Newton optimizer.
        # WHY: LBFGS is efficient for small-to-medium datasets, handles L2 penalty,
        # and converges faster than SGD for batch optimization. It approximates the
        # Hessian matrix without storing it fully (memory-efficient).
        max_iter=1000,      # Maximum iterations for the solver to converge.
        # WHY: 1000 is generous enough for most problems. If the solver doesn't converge,
        # it means the learning problem is too hard or features need better preprocessing.
        random_state=42,    # Fix seed for reproducibility.
        # WHY: Some solvers (like 'sag' and 'saga') use stochastic updates that
        # depend on random initialization. Fixing the seed ensures reproducible results.
    )
    # Override defaults with any user-provided hyperparameters.
    # WHY: This pattern lets callers customize specific params while keeping sensible defaults
    # for everything else. Optuna uses this to test different configurations.
    defaults.update(hyperparams)

    # Instantiate and train the logistic regression model.
    # WHY: sklearn's LogisticRegression wraps optimized C/Fortran code (liblinear, LBFGS)
    # that's orders of magnitude faster than a pure Python implementation.
    model = LogisticRegression(**defaults)

    # Fit the model to training data: finds optimal weights w and bias b.
    # WHY: .fit() runs the optimization algorithm (e.g., LBFGS) to minimize the
    # regularized cross-entropy loss and find the decision boundary.
    model.fit(X_train, y_train)

    # Log the parameters used for this training run.
    # WHY: This creates an audit trail so we can see exactly which configuration
    # produced each set of results, critical for experiment tracking.
    logger.info("Model trained with params: %s", defaults)

    return model


def _evaluate(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Shared evaluation logic for both validation and test phases.

    WHY this is a private function: Validation and test evaluation share the same
    metric computation. Extracting it avoids code duplication and ensures consistency.
    """
    # Generate class predictions using the learned decision boundary.
    # WHY: predict() applies the threshold P(y=1|x) >= 0.5 by default.
    # For imbalanced datasets, you might want to adjust this threshold.
    y_pred = model.predict(X)

    # Get probability estimates for each class.
    # WHY: Probabilities are needed for AUC-ROC calculation, which evaluates
    # the model's ranking ability across all possible thresholds, not just 0.5.
    y_proba = model.predict_proba(X)

    # Determine number of classes to choose the right AUC computation method.
    # WHY: Binary AUC uses the probability of the positive class directly.
    # Multiclass AUC requires a different approach (one-vs-rest averaging).
    n_classes = len(np.unique(y))
    if n_classes == 2:
        # For binary classification, AUC is computed on the positive class probability.
        # WHY: y_proba[:, 1] gives P(y=1|x), which is what the ROC curve plots against.
        auc = roc_auc_score(y, y_proba[:, 1])
    else:
        # For multiclass, use one-vs-rest (OvR) strategy with weighted average.
        # WHY: OvR computes a separate AUC for each class vs. all others, then
        # averages weighted by the number of true instances of each class.
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    # Compute all standard classification metrics.
    # WHY: Each metric captures a different aspect of performance:
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        # Accuracy: simple but misleading on imbalanced data (98% accuracy if 98% are class 0).
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        # Precision: "Of everything we predicted positive, how much was actually positive?"
        # WHY weighted: gives each class a weight proportional to its support.
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        # Recall: "Of everything that was actually positive, how much did we catch?"
        # WHY: In medical diagnosis, recall is often more important than precision.
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        # F1: harmonic mean of precision and recall. Balances both concerns.
        # WHY harmonic mean: it penalizes extreme imbalances (low precision OR low recall).
        "auc_roc": auc,
        # AUC-ROC: threshold-independent measure of the model's discrimination ability.
        # WHY: Unlike accuracy, AUC works well even with imbalanced classes.
    }
    return metrics


def validate(model: LogisticRegression, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Validate model on validation set.

    WHY separate from test: The validation set is used during development to tune
    hyperparameters. The test set is reserved for final, unbiased evaluation.

    Returns:
        Dictionary of metrics.
    """
    # Compute metrics on the validation set using the shared evaluation logic.
    metrics = _evaluate(model, X_val, y_val)
    # Log metrics formatted to 4 decimal places for readable output.
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Final test evaluation.

    WHY: This function should only be called ONCE with the best model.
    Calling it multiple times with different models and selecting the best
    would be "test set leakage" -- using the test set for model selection.

    Returns:
        Dictionary of metrics.
    """
    # Compute the same metrics used in validation for consistency.
    metrics = _evaluate(model, X_test, y_test)
    # Also get raw predictions for the confusion matrix and classification report.
    y_pred = model.predict(X_test)

    # Log all metrics for the final evaluation.
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    # Log the confusion matrix: shows the 2x2 (or KxK) table of actual vs predicted.
    # WHY: The confusion matrix reveals the TYPE of errors (false positives vs false negatives).
    # For example, in medical diagnosis, false negatives (missing a disease) are more costly
    # than false positives (unnecessary follow-up test).
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Log the full classification report with per-class precision, recall, F1.
    # WHY: This shows if the model performs well on ALL classes, not just the majority.
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
    """Compare multiple hyperparameter configurations and explain the trade-offs.

    WHY: Understanding how different hyperparameters affect model behavior is essential
    for practical ML. This function trains the model with 4 carefully chosen configurations
    and prints a comparison table so you can see the performance impact of each choice.

    The configurations span:
        - Regularization type (L1 vs L2) and strength (C values)
        - Solver choice (algorithm used for optimization)
        - Trade-off between model simplicity and expressiveness
    """
    # Define 4 configurations that illustrate key hyperparameter trade-offs.
    # WHY: Each config is designed to teach something different about logistic regression.
    configs = {
        "Strong L2 (conservative)": {
            # Config 1: Strong regularization with L2 penalty.
            # WHY: C=0.01 means very strong regularization. The model is forced to keep
            # weights small, producing a simpler decision boundary. This is the "safe" choice
            # when you're worried about overfitting or when you have many noisy features.
            # Best for: high-dimensional data with many irrelevant features.
            "C": 0.01,
            "penalty": "l2",
            "solver": "lbfgs",      # LBFGS is the default, efficient quasi-Newton method.
            "max_iter": 1000,
        },
        "Moderate L2 (balanced)": {
            # Config 2: Balanced regularization.
            # WHY: C=1.0 is the default and usually a good starting point. It provides
            # moderate regularization -- not too conservative, not too aggressive.
            # The model can learn meaningful patterns without overfitting.
            # Best for: general-purpose classification tasks.
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "L1 Sparse (feature selection)": {
            # Config 3: L1 regularization for automatic feature selection.
            # WHY: L1 (Lasso) penalty drives some weights to exactly zero, effectively
            # removing those features from the model. This is invaluable when you suspect
            # many features are irrelevant. The resulting model is more interpretable
            # because you can see which features were kept.
            # NOTE: L1 requires the 'saga' solver (supports L1 in sklearn).
            # Best for: interpretable models, datasets with many irrelevant features.
            "C": 0.5,
            "penalty": "l1",
            "solver": "saga",        # SAGA supports L1 penalty (LBFGS does not).
            # WHY saga: It's a stochastic solver that handles L1 regularization.
            # 'liblinear' also works for L1 but doesn't scale as well to large datasets.
            "max_iter": 2000,        # SAGA may need more iterations to converge.
        },
        "Weak Regularization (aggressive)": {
            # Config 4: Very weak regularization for maximum model complexity.
            # WHY: C=100 means almost no regularization. The model is free to fit the
            # training data as closely as possible. This can capture subtle patterns
            # but risks overfitting, especially with limited training data or noisy features.
            # Best for: clean data with few noise features, when underfitting is the concern.
            "C": 100.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 2000,
        },
    }

    # Print a header for the comparison table.
    print("\n" + "=" * 90)
    print("LOGISTIC REGRESSION - HYPERPARAMETER COMPARISON")
    print("=" * 90)
    # Print column headers aligned for readability.
    print(f"{'Config':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 90)

    # Train and evaluate each configuration.
    for name, params in configs.items():
        # Train the model with this specific configuration.
        # WHY: We use the train() function to ensure consistent handling of defaults
        # and the same random_state for fair comparison across configurations.
        model = train(X_train, y_train, **params)
        # Evaluate on the validation set (NOT test set -- that's reserved for final eval).
        # WHY: Using the validation set lets us compare configs without "using up" the test set.
        metrics = _evaluate(model, X_val, y_val)
        # Print formatted results for this configuration.
        print(
            f"{name:<32} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f}"
        )

    # Print explanatory notes about the configurations.
    print("-" * 90)
    print("INTERPRETATION GUIDE:")
    print("  - Strong L2: Safest against overfitting. May underfit if data has complex patterns.")
    print("  - Moderate L2: Good general-purpose starting point. Usually the best trade-off.")
    print("  - L1 Sparse: Use when you need feature selection or interpretability.")
    print("  - Weak Regularization: Best if your data is clean and you need maximum expressiveness.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Real-World Demo: Customer Churn Prediction
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate logistic regression on a realistic customer churn prediction problem.

    WHY this scenario: Customer churn is one of the most common and impactful classification
    problems in business. Telecom companies, SaaS platforms, and subscription services all
    need to predict which customers are likely to leave so they can take retention actions.

    Domain context:
        - A telecom company wants to predict which customers will cancel their subscription.
        - Features include customer tenure, monthly charges, contract type, etc.
        - The target variable is binary: 1 = churned, 0 = stayed.
        - Early identification of at-risk customers allows targeted retention campaigns
          (discounts, better plans) that are much cheaper than acquiring new customers.

    Features generated (synthetic but realistic):
        - tenure_months: How long the customer has been with the company (0-72 months).
          WHY: Longer-tenured customers are generally more loyal and less likely to churn.
        - monthly_charges: Monthly billing amount ($20-$120).
          WHY: Higher charges often correlate with churn, especially without perceived value.
        - total_charges: Cumulative amount paid (tenure * monthly + noise).
          WHY: This is correlated with tenure but adds information about overall spend.
        - contract_type: Encoded as 0=month-to-month, 1=one-year, 2=two-year.
          WHY: Month-to-month contracts have the lowest switching cost (highest churn risk).
        - num_support_tickets: Number of customer support interactions (0-10).
          WHY: More support tickets indicate dissatisfaction, a strong churn predictor.
        - has_online_backup: Whether the customer uses online backup (0 or 1).
          WHY: Value-added services increase switching cost, reducing churn probability.
    """
    print("\n" + "=" * 90)
    print("REAL-WORLD DEMO: Customer Churn Prediction (Telecom)")
    print("=" * 90)

    # Set random seed for reproducibility of this demo.
    np.random.seed(42)
    # Define the number of customers in our synthetic dataset.
    n_samples = 2000

    # --- Generate realistic synthetic features ---
    # Tenure: how long the customer has been subscribed, in months.
    # WHY uniform(0,72): typical telecom contract data spans 0-6 years.
    tenure_months = np.random.uniform(0, 72, n_samples)

    # Monthly charges: the customer's monthly bill amount.
    # WHY normal(65, 25): centers around $65/month with reasonable spread.
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 120)

    # Total charges: accumulated billing over the customer's lifetime.
    # WHY: This is computed from tenure and monthly charges with some noise,
    # mimicking real data where total_charges ~ tenure * monthly_charges.
    total_charges = tenure_months * monthly_charges + np.random.normal(0, 200, n_samples)

    # Contract type: 0=month-to-month, 1=one-year, 2=two-year.
    # WHY probabilities [0.5, 0.3, 0.2]: month-to-month is the most common contract type,
    # and also the one most strongly associated with churn.
    contract_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])

    # Number of support tickets: a proxy for customer dissatisfaction.
    # WHY poisson(2): most customers file 0-3 tickets, with a long tail of heavy users.
    num_support_tickets = np.random.poisson(2, n_samples)

    # Online backup service: a value-added feature that increases stickiness.
    # WHY binomial: each customer independently has a 40% chance of having this service.
    has_online_backup = np.random.binomial(1, 0.4, n_samples)

    # --- Generate the churn label based on realistic business logic ---
    # WHY this formula: We create a "churn score" that combines all features
    # with domain-appropriate signs and magnitudes:
    #   - Short tenure -> higher churn (negative coefficient)
    #   - High monthly charges -> higher churn (positive coefficient)
    #   - Long contract -> lower churn (negative coefficient, larger magnitude)
    #   - More support tickets -> higher churn (positive coefficient)
    #   - Online backup -> lower churn (negative coefficient)
    churn_score = (
        -0.05 * tenure_months        # Longer tenure reduces churn risk.
        + 0.03 * monthly_charges     # Higher charges increase churn risk.
        - 0.8 * contract_type        # Longer contracts reduce churn (bigger effect).
        + 0.3 * num_support_tickets  # More complaints increase churn risk.
        - 0.5 * has_online_backup    # Value-added services reduce churn.
        + np.random.normal(0, 1, n_samples)  # Random noise for realistic uncertainty.
    )
    # Convert the continuous score to a binary label using sigmoid + threshold.
    # WHY sigmoid: it maps any real number to [0,1], giving us a probability.
    churn_prob = 1 / (1 + np.exp(-churn_score))
    # WHY random threshold: using the probability as a Bernoulli parameter creates
    # realistic label noise (not all high-risk customers actually churn).
    y = (np.random.random(n_samples) < churn_prob).astype(int)

    # Stack all features into a feature matrix with named columns.
    # WHY column_stack: creates a 2D array where each column is a feature.
    X = np.column_stack([
        tenure_months,
        monthly_charges,
        total_charges,
        contract_type,
        num_support_tickets,
        has_online_backup,
    ])

    # Define feature names for interpretability.
    # WHY: When we examine model coefficients, we want to know which feature each weight corresponds to.
    feature_names = [
        "tenure_months", "monthly_charges", "total_charges",
        "contract_type", "num_support_tickets", "has_online_backup",
    ]

    # Print dataset summary statistics.
    print(f"\nDataset: {n_samples} customers, {len(feature_names)} features")
    print(f"Churn rate: {y.mean():.1%} ({y.sum()} churned / {n_samples - y.sum()} retained)")
    print(f"Features: {', '.join(feature_names)}")

    # Split into train/test with stratification to preserve churn rate in both sets.
    # WHY stratify=y: If 30% of customers churned overall, both train and test will have ~30%.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Scale features to zero mean and unit variance.
    # WHY: contract_type (0-2) and total_charges (0-10000) are on vastly different scales.
    # Without scaling, logistic regression's gradient descent would be dominated by total_charges.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model with L2 regularization (good default for most problems).
    # WHY C=1.0: moderate regularization prevents overfitting while capturing real patterns.
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Print classification performance metrics.
    print("\n--- Model Performance ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_test, y_proba[:, 1]):.4f}")

    # Display feature importance via model coefficients.
    # WHY: Logistic regression's coefficients directly tell us the direction and strength
    # of each feature's influence on the prediction. Positive = increases churn probability.
    print("\n--- Feature Importance (Model Coefficients) ---")
    # Get the raw coefficients from the trained model.
    coefficients = model.coef_[0]
    # Sort features by absolute coefficient magnitude (most important first).
    # WHY absolute value: the sign tells direction, but magnitude tells importance.
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    for i in sorted_idx:
        # Show the feature name, coefficient value, and what it means.
        direction = "increases churn" if coefficients[i] > 0 else "decreases churn"
        print(f"  {feature_names[i]:<25} coef={coefficients[i]:>8.4f}  ({direction})")

    # Business interpretation of the results.
    print("\n--- Business Insights ---")
    print("  1. Customers with short tenure and month-to-month contracts are highest risk.")
    print("  2. High monthly charges without value-added services drive churn.")
    print("  3. Offering online backup and long-term contracts reduces churn significantly.")
    print("  4. Support ticket volume is a leading indicator of dissatisfaction.")
    print("  5. Target retention campaigns at high-score customers BEFORE they churn.")
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
    """Optuna objective function for hyperparameter optimization.

    WHY Optuna: Unlike grid search (exhaustive but slow) or random search (fast but untargeted),
    Optuna uses a Tree-structured Parzen Estimator (TPE) that builds a probabilistic model of
    good vs. bad hyperparameter regions. It samples more from promising regions, converging
    to good configurations faster than random search.

    WHY these parameter ranges: Each range is chosen based on domain knowledge about
    logistic regression's behavior with different settings.
    """
    # C: Inverse regularization strength, searched on a log scale.
    # WHY log scale: C spans several orders of magnitude (0.0001 to 100).
    # Log-uniform sampling ensures equal attention to small and large values.
    C = trial.suggest_float("C", 1e-4, 100.0, log=True)

    # Penalty type: L1 encourages sparsity, L2 shrinks weights uniformly.
    # WHY these two: L1 and L2 are the most common and well-understood penalties.
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    # Solver: must be compatible with the chosen penalty.
    # WHY conditional: L1 penalty requires 'saga' or 'liblinear' solver.
    # L2 works with all solvers, so we let Optuna explore the options.
    solver = "saga" if penalty == "l1" else trial.suggest_categorical(
        "solver", ["lbfgs", "newton-cg", "saga"]
    )

    # Max iterations: how long the solver runs before giving up.
    # WHY step=200: finer granularity isn't needed; convergence is not that sensitive.
    max_iter = trial.suggest_int("max_iter", 200, 2000, step=200)

    # Train the model with the trial's hyperparameters.
    model = train(
        X_train, y_train,
        C=C, penalty=penalty, solver=solver, max_iter=max_iter,
    )
    # Evaluate on validation set and return F1 score for Optuna to maximize.
    # WHY F1: It balances precision and recall, making it more robust than accuracy
    # for imbalanced datasets.
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search.

    WHY Ray Tune: When you need distributed hyperparameter search across multiple machines
    or GPUs, Ray Tune provides a scalable framework. It also supports advanced scheduling
    algorithms like ASHA (Asynchronous Successive Halving) for early stopping of bad trials.

    Returns:
        Best hyperparameter configuration found.
    """
    # Import ray and tune lazily to avoid requiring ray as a dependency
    # for users who only want Optuna.
    # WHY lazy import: Ray is a heavy dependency (~200MB). Not everyone needs distributed HPO.
    import ray
    from ray import tune as ray_tune

    # Initialize Ray if not already running.
    # WHY ignore_reinit_error: Prevents crashes if Ray was already initialized in a notebook.
    # WHY log_to_driver=False: Reduces console noise from Ray's internal logging.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Define the trainable function that Ray will call for each trial.
    # WHY closure: The function captures X_train, y_train, etc. from the enclosing scope.
    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        # Report metrics back to Ray Tune for comparison across trials.
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    # Define the search space for Ray Tune.
    # WHY these ranges: Same rationale as Optuna, but using Ray's API for distributions.
    search_space = {
        "C": ray_tune.loguniform(1e-4, 100.0),           # Log-uniform for regularization.
        "penalty": ray_tune.choice(["l1", "l2"]),         # Categorical choice.
        "solver": "saga",                                  # SAGA works with both L1 and L2.
        # WHY fixed solver: Simplifies the search space. SAGA is the most versatile solver.
        "max_iter": ray_tune.choice([500, 1000, 1500, 2000]),  # Discrete choices.
    }

    # Create and run the Tuner.
    # WHY Tuner API: Ray Tune's Tuner provides a clean interface for configuring and
    # running hyperparameter searches with various scheduling/search algorithms.
    tuner = ray_tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=ray_tune.TuneConfig(
            num_samples=num_samples,   # Total number of trials to run.
            metric="f1",              # Metric to optimize.
            mode="max",               # Maximize F1 (not minimize).
        ),
    )
    # Run all trials and collect results.
    results = tuner.fit()
    # Extract the best configuration based on F1 score.
    best = results.get_best_result(metric="f1", mode="max")
    logger.info("Ray Tune best config: %s", best.config)
    # Clean up Ray resources.
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Logistic Regression (sklearn) pipeline.

    WHY this pipeline structure:
        1. Generate data -> establishes the learning problem.
        2. Baseline training -> gives us a reference point with default params.
        3. Parameter comparison -> helps understand the impact of each hyperparameter.
        4. Real-world demo -> shows the model applied to a business problem.
        5. Optuna optimization -> finds the best hyperparameters systematically.
        6. Final test evaluation -> unbiased performance estimate on held-out data.

    This order is deliberate: we understand the data and model behavior BEFORE
    running expensive hyperparameter optimization.
    """
    # Print a prominent header so this implementation is easy to identify in logs.
    logger.info("=" * 70)
    logger.info("Logistic Regression - Scikit-Learn Implementation")
    logger.info("=" * 70)

    # Step 1: Generate synthetic classification data.
    # WHY: Synthetic data gives us full control and reproducibility.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Train a baseline model with default hyperparameters.
    # WHY: The baseline serves as a reference point. If Optuna can't beat it,
    # either the defaults are already good or the search space needs adjustment.
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # Step 3: Compare different hyperparameter configurations.
    # WHY: Before running automated optimization, understanding how each parameter
    # affects performance gives us intuition for setting search ranges.
    logger.info("\n--- Hyperparameter Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 4: Run the real-world customer churn demo.
    # WHY: This demonstrates practical application of logistic regression
    # with domain-specific features and business interpretation.
    real_world_demo()

    # Step 5: Run Optuna hyperparameter optimization.
    # WHY: Optuna systematically searches the hyperparameter space using Bayesian
    # optimization, finding configurations that humans might not think to try.
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    # Create a study that maximizes F1 score.
    study = optuna.create_study(direction="maximize")
    # Run 30 trials of hyperparameter search.
    # WHY 30 trials: Balances search thoroughness with computation time.
    # For logistic regression (fast to train), 30 trials is usually sufficient.
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=30,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Step 6: Retrain with the best hyperparameters found by Optuna.
    # WHY retrain: During Optuna, models are trained and discarded. We need to
    # retrain with the best params to get the final model for test evaluation.
    best_params = study.best_params
    # Handle solver compatibility: L1 penalty requires SAGA solver.
    # WHY: Optuna might find that L1 penalty is best, but the solver might not match.
    if best_params.get("penalty") == "l1":
        best_params["solver"] = "saga"
    best_model = train(X_train, y_train, **best_params)

    # Step 7: Final test evaluation with the best model.
    # WHY: This is the ONLY time we touch the test set. The resulting metrics
    # are our unbiased estimate of how this model will perform on new data.
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Step 8: Ray Tune search (optional, commented out by default for speed).
    # WHY commented out: Ray requires a running Ray cluster and adds significant
    # startup overhead. Uncomment when you have distributed resources available.
    # logger.info("\n--- Ray Tune Search ---")
    # ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20)
    # ray_model = train(X_train, y_train, **ray_config)
    # test(ray_model, X_test, y_test)


# This guard ensures the main() function only runs when the script is executed directly,
# not when it's imported as a module by another script.
# WHY: Allows other scripts to import functions like train(), compare_parameter_sets(),
# or real_world_demo() without triggering the full pipeline.
if __name__ == "__main__":
    main()
