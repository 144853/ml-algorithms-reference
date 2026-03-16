"""
Decision Tree Classifier - Scikit-Learn Implementation
=======================================================

Theory & Mathematics:
    A Decision Tree is a non-parametric supervised learning method that recursively
    partitions the feature space into rectangular regions, each assigned to a class.

    The tree is built top-down using a greedy algorithm (CART - Classification and
    Regression Trees):

    1. At each node, the algorithm selects the feature and threshold that produces
       the "purest" child nodes, measured by an impurity criterion:

       Gini Impurity:
           Gini(S) = 1 - sum_{i=1}^{C} p_i^2
           where p_i is the proportion of class i in set S.
           Range: [0, 0.5] for binary (0 = pure, 0.5 = maximally impure).
           WHY: Computationally cheaper than entropy (no log), sklearn's default.

       Entropy (Information Gain):
           Entropy(S) = -sum_{i=1}^{C} p_i * log2(p_i)
           Information Gain = Entropy(parent) - weighted_avg(Entropy(children))
           Range: [0, log2(C)] (0 = pure, log2(C) = maximum disorder).
           WHY: Comes from information theory; tends to produce more balanced trees.

    2. The split that maximizes the impurity reduction (or equivalently, information
       gain) is chosen. The formula for impurity reduction is:
           Delta_I = I(parent) - (n_left/n_parent)*I(left) - (n_right/n_parent)*I(right)

    3. Recursion continues until a stopping condition is met:
       - max_depth reached
       - min_samples_split not met
       - min_samples_leaf not met
       - node is pure (all samples same class)

    Decision Boundary:
        The decision boundary is always axis-aligned (perpendicular to feature axes),
        which means each split divides the space with a threshold on a single feature:
            if feature_j <= threshold: go left
            else: go right

Business Use Cases:
    - Medical diagnosis: transparent rules for clinical decision-making
    - Credit approval: regulatory requirement for explainable decisions
    - Customer segmentation: easily interpretable marketing rules
    - Fraud detection: rule-based alert systems
    - Manufacturing quality control: root cause analysis

Advantages:
    - Highly interpretable: can be visualized as a flowchart of decisions
    - Handles both numerical and categorical features natively
    - Requires little data preprocessing (no scaling needed)
    - Non-parametric: no assumptions about data distribution
    - Feature importance computed naturally from impurity reduction
    - Can capture non-linear relationships and feature interactions

Disadvantages:
    - Prone to overfitting, especially deep trees (high variance)
    - Unstable: small changes in data can produce very different trees
    - Axis-aligned splits cannot capture diagonal decision boundaries efficiently
    - Biased toward features with more levels (more split points available)
    - Greedy splitting is locally optimal, not globally optimal
    - Single trees often underperform ensembles (Random Forest, XGBoost)

Hyperparameters:
    - max_depth: Maximum depth of the tree (None = unlimited)
    - criterion: Impurity measure ('gini' or 'entropy')
    - min_samples_split: Minimum samples needed to split a node
    - min_samples_leaf: Minimum samples required in a leaf node
    - max_features: Number of features considered for each split
    - class_weight: Weights for classes to handle imbalance
"""

# --- Standard library imports ---
# logging: provides structured output messages with severity levels (INFO, DEBUG, WARNING).
# WHY: print() is fine for scripts, but logging is better for production code because it
# supports log levels, can be redirected to files, and integrates with monitoring systems.
import logging  # Standard library module for structured logging output

# warnings: used to suppress noisy convergence/deprecation warnings from sklearn.
# WHY: During hyperparameter search, many configurations trigger warnings about
# min_samples_leaf violations or other edge cases. Suppressing keeps output clean.
import warnings  # Standard library module for warning control

# typing: provides type annotations for function signatures.
# WHY: Type hints serve as documentation, enable IDE autocompletion, and allow
# static analysis tools like mypy to catch type errors before runtime.
from typing import Any, Dict, Tuple  # Type annotation classes

# --- Third-party imports ---
# numpy: foundational numerical computing library for array operations.
# WHY: All ML data flows through numpy arrays; sklearn uses numpy internally.
import numpy as np  # Numerical computing library

# optuna: Bayesian hyperparameter optimization framework using TPE.
# WHY: Optuna uses Tree-structured Parzen Estimator to intelligently search
# the hyperparameter space, converging faster than grid or random search.
import optuna  # Hyperparameter optimization framework

# make_classification: generates synthetic classification datasets with controllable properties.
# WHY: Synthetic data gives us control over informative/redundant features, class balance,
# and noise level, making it ideal for benchmarking and understanding algorithm behavior.
from sklearn.datasets import make_classification  # Synthetic data generator

# DecisionTreeClassifier: sklearn's optimized CART implementation.
# WHY: It wraps optimized Cython code for production-grade speed, supports both
# Gini and Entropy criteria, and provides built-in feature importance computation.
from sklearn.tree import DecisionTreeClassifier  # CART decision tree implementation

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

# train_test_split: splits data into training/validation/test sets with stratification.
# WHY: We need separate sets for training (learn), validation (tune), and test (final eval)
# to get an unbiased estimate of model performance on unseen data.
from sklearn.model_selection import train_test_split  # Data splitting utility

# StandardScaler: standardizes features by removing mean and scaling to unit variance.
# WHY: While decision trees DON'T need scaling (they split on thresholds, not distances),
# we include it for consistency with other algorithms in this repository. In production
# decision tree pipelines, scaling is typically omitted.
from sklearn.preprocessing import StandardScaler  # Feature standardization

# --- Logging configuration ---
# Set up logging to display timestamps, severity level, and message.
# WHY: level=INFO shows training progress and metrics but hides verbose DEBUG output.
logging.basicConfig(
    level=logging.INFO,  # Only show INFO and above (not DEBUG)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + level + message
)
# Create a module-level logger named after this file for identification in multi-module projects.
# WHY: __name__ makes the logger name match the module name ("decision_tree_sklearn").
logger = logging.getLogger(__name__)  # Module-specific logger instance

# Suppress sklearn warnings about min_samples or other parameter edge cases.
# WHY: During Optuna search, some parameter combos produce valid but warned configurations.
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,      # Total number of samples to generate
    n_features: int = 20,       # Total number of features (informative + redundant + noise)
    n_classes: int = 2,         # Number of target classes (2 = binary classification)
    random_state: int = 42,     # Random seed for reproducibility
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data and split into train/val/test.

    WHY synthetic data: It gives us full control over the problem's difficulty,
    dimensionality, and class balance. This makes it easy to benchmark the algorithm
    under controlled conditions before applying it to real-world data.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays.
    """
    # Generate a synthetic dataset with make_classification.
    # WHY: make_classification creates linearly separable clusters with added noise,
    # allowing us to control the signal-to-noise ratio precisely.
    X, y = make_classification(
        n_samples=n_samples,            # Total number of data points to generate
        n_features=n_features,          # Total feature dimensionality
        n_informative=n_features // 2,  # Half carry real signal
        # WHY half informative: tests if the tree can identify relevant features
        # and ignore noise, which is one of decision trees' strengths.
        n_redundant=n_features // 4,    # Quarter are linear combos of informative features
        # WHY redundant: tests robustness to multicollinearity. Decision trees handle
        # this naturally because they pick the single best feature at each split.
        n_classes=n_classes,            # Number of distinct classes in the target
        random_state=random_state,      # Ensures identical data across runs
        # WHY fixed seed: reproducibility is essential for fair algorithm comparisons.
    )

    # First split: 60% train, 40% temp (to be split into val + test).
    # WHY stratify=y: ensures each split maintains the same class proportions as the
    # original dataset, critical for imbalanced classification problems.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,                           # Features and labels to split
        test_size=0.4,                  # 40% goes to temp (val + test)
        random_state=random_state,      # Reproducible split
        stratify=y,                     # Maintain class balance in each split
    )

    # Second split: 50% of temp = 20% val, 50% of temp = 20% test.
    # WHY separate val and test: validation set is for hyperparameter tuning (used
    # repeatedly), test set is for final evaluation (used exactly once).
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,                 # Split the temp set
        test_size=0.5,                  # Equal split into val and test
        random_state=random_state,      # Reproducible split
        stratify=y_temp,                # Maintain class balance
    )

    # NOTE: Decision trees do NOT require feature scaling because they make splits
    # based on thresholds (comparisons), not distances or gradients.
    # WHY we still include StandardScaler: for consistency with the repository's
    # other algorithm files and to demonstrate that scaling doesn't hurt trees.
    scaler = StandardScaler()  # Create scaler instance
    X_train = scaler.fit_transform(X_train)  # Fit on train, transform train
    # WHY fit only on train: prevents data leakage from val/test statistics.
    X_val = scaler.transform(X_val)    # Transform val using train statistics
    X_test = scaler.transform(X_test)  # Transform test using train statistics

    # Log dataset dimensions for verification.
    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0],  # Sample counts
        n_features, n_classes,  # Feature and class counts
    )
    # Return all six arrays: features and labels for train, val, and test splits.
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,    # Training feature matrix of shape (n_samples, n_features)
    y_train: np.ndarray,    # Training labels of shape (n_samples,)
    **hyperparams: Any,     # Arbitrary keyword arguments forwarded to DecisionTreeClassifier
) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier using sklearn's CART implementation.

    Args:
        X_train: Training features.
        y_train: Training labels.
        **hyperparams: Keyword arguments passed to DecisionTreeClassifier.

    Returns:
        Trained DecisionTreeClassifier model.
    """
    # Define sensible default hyperparameters for a decision tree.
    # WHY these defaults: they produce a moderately constrained tree that balances
    # learning capacity with overfitting prevention.
    defaults = dict(
        max_depth=10,            # Maximum tree depth. Limits tree complexity.
        # WHY 10: Deep enough to capture complex patterns, shallow enough to avoid
        # overfitting. A fully grown tree (max_depth=None) memorizes training data.
        criterion="gini",        # Impurity measure for splitting decisions.
        # WHY Gini: It's sklearn's default, slightly faster than entropy (no log),
        # and produces very similar trees in practice.
        min_samples_split=5,     # Minimum samples required to split a node.
        # WHY 5: Prevents splitting on tiny groups that likely represent noise.
        # A node with fewer than 5 samples shouldn't be split because the split
        # decision would be based on too little data.
        min_samples_leaf=2,      # Minimum samples required in each leaf node.
        # WHY 2: Prevents creating leaf nodes with a single sample, which is a
        # hallmark of overfitting (memorizing individual data points).
        random_state=42,         # Random seed for reproducibility.
        # WHY: When max_features < n_features, the tree randomly samples which
        # features to consider at each split. Fixed seed ensures reproducibility.
    )
    # Override defaults with any user-provided hyperparameters.
    # WHY: This pattern lets callers customize specific params while keeping defaults.
    # Optuna uses this to inject trial-specific configurations.
    defaults.update(hyperparams)

    # Instantiate the DecisionTreeClassifier with the merged parameters.
    # WHY sklearn: It wraps optimized Cython code that's orders of magnitude faster
    # than a pure Python implementation, especially for large datasets.
    model = DecisionTreeClassifier(**defaults)  # Create tree with specified params

    # Fit the model: build the decision tree by recursively finding optimal splits.
    # WHY .fit(): This triggers the CART algorithm, which greedily selects the best
    # feature and threshold at each node to maximize impurity reduction.
    model.fit(X_train, y_train)  # Build the tree from training data

    # Log which parameters were used for this training run.
    # WHY: Creates an audit trail for experiment tracking and debugging.
    logger.info("Model trained with params: %s", defaults)

    return model  # Return the trained decision tree


def _evaluate(
    model: DecisionTreeClassifier,  # Trained decision tree model
    X: np.ndarray,                  # Feature matrix to evaluate on
    y: np.ndarray,                  # True labels to compare predictions against
) -> Dict[str, float]:
    """Shared evaluation logic for both validation and test phases.

    WHY a private function: Validation and test evaluation use identical metric
    computation. Extracting it avoids code duplication and ensures consistency.
    The underscore prefix signals this is an internal helper, not part of the public API.

    Returns:
        Dictionary mapping metric names to their float values.
    """
    # Generate class predictions using the learned decision tree.
    # WHY predict(): Each sample traverses the tree from root to leaf, following
    # the learned split rules. The leaf's majority class becomes the prediction.
    y_pred = model.predict(X)  # Predicted class labels

    # Get probability estimates for each class.
    # WHY predict_proba(): Decision tree probabilities are computed as the fraction
    # of training samples of each class in the leaf node. For example, if a leaf
    # has 8 class-0 and 2 class-1 samples, P(class=1) = 0.2.
    y_proba = model.predict_proba(X)  # Class probability estimates

    # Determine number of classes to choose the right AUC computation method.
    # WHY: Binary and multiclass AUC use different computation strategies.
    n_classes = len(np.unique(y))  # Count distinct classes in true labels
    if n_classes == 2:
        # Binary AUC: use the positive class probability directly.
        # WHY y_proba[:, 1]: column 1 contains P(y=1|x), which the ROC curve uses.
        auc = roc_auc_score(y, y_proba[:, 1])  # Binary AUC-ROC
    else:
        # Multiclass AUC: one-vs-rest strategy with weighted averaging.
        # WHY OvR: computes a separate AUC for each class vs all others,
        # then averages weighted by class frequency.
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    # Compute all standard classification metrics.
    # WHY multiple metrics: each captures a different aspect of performance.
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        # Accuracy: fraction of correct predictions. Simple but misleading on imbalanced data.
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        # Precision: of everything predicted positive, how much was actually positive?
        # WHY weighted: gives each class weight proportional to its support (sample count).
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        # Recall: of everything actually positive, how much did we correctly identify?
        # WHY: In medical diagnosis, missing a disease (low recall) is worse than false alarm.
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        # F1: harmonic mean of precision and recall. Balances both concerns.
        # WHY harmonic mean: penalizes extreme imbalance between precision and recall.
        "auc_roc": auc,
        # AUC-ROC: threshold-independent measure of discrimination ability.
        # WHY: Unlike accuracy, AUC remains meaningful on imbalanced datasets.
    }
    return metrics  # Return dictionary of all computed metrics


def validate(
    model: DecisionTreeClassifier,  # Trained model to validate
    X_val: np.ndarray,              # Validation features
    y_val: np.ndarray,              # Validation labels
) -> Dict[str, float]:
    """Validate model on validation set during hyperparameter tuning.

    WHY separate from test: Validation is used repeatedly during development.
    Test is used exactly once at the end for unbiased evaluation.

    Returns:
        Dictionary of validation metrics.
    """
    # Compute metrics on validation set using the shared evaluation function.
    metrics = _evaluate(model, X_val, y_val)  # Evaluate on validation data
    # Log metrics formatted to 4 decimal places for readable output.
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics  # Return metrics for use by Optuna or comparison functions


def test(
    model: DecisionTreeClassifier,  # Best model selected after tuning
    X_test: np.ndarray,             # Test features (never seen during training/tuning)
    y_test: np.ndarray,             # Test labels
) -> Dict[str, float]:
    """Final test evaluation - should only be called ONCE with the best model.

    WHY only once: Evaluating multiple models on the test set and selecting the best
    constitutes "test set leakage" - you're effectively using the test set for model
    selection, which biases the final performance estimate upward.

    Returns:
        Dictionary of test metrics.
    """
    # Compute metrics using shared evaluation logic.
    metrics = _evaluate(model, X_test, y_test)  # Final evaluation
    # Get raw predictions for confusion matrix and classification report.
    y_pred = model.predict(X_test)  # Generate predictions for detailed analysis

    # Log all metrics for the final evaluation.
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    # Log confusion matrix: actual vs predicted class counts.
    # WHY: Reveals the TYPE of errors (false positives vs false negatives).
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Log per-class precision, recall, F1 for detailed class-level analysis.
    # WHY: Shows if the model performs well on ALL classes, not just the majority.
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))

    # Log tree-specific information: depth and number of leaves.
    # WHY: These indicate tree complexity. A deep tree with many leaves may overfit;
    # a shallow tree with few leaves may underfit.
    logger.info(
        "Tree depth: %d, Number of leaves: %d",
        model.get_depth(),          # Actual depth of the fitted tree
        model.get_n_leaves(),       # Number of leaf nodes (terminal nodes)
    )

    return metrics  # Return metrics dictionary


# ---------------------------------------------------------------------------
# Hyperparameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,   # Training features for fitting each configuration
    y_train: np.ndarray,   # Training labels
    X_val: np.ndarray,     # Validation features for evaluation
    y_val: np.ndarray,     # Validation labels
) -> None:
    """Compare multiple hyperparameter configurations and explain the trade-offs.

    WHY: Understanding how max_depth, criterion, and min_samples_leaf affect tree
    behavior is essential for practical ML. This function trains the model with 4
    carefully chosen configurations and shows a comparison table.

    Configurations span:
        - Shallow vs deep trees (underfitting vs overfitting trade-off)
        - Gini vs Entropy criterion (impurity measure comparison)
        - Conservative vs aggressive pruning (min_samples_leaf)
    """
    # Define 4 configurations that illustrate key decision tree trade-offs.
    # WHY these 4: Each teaches something different about tree behavior.
    configs = {
        "Shallow (depth=3, gini)": {
            # Config 1: Very shallow tree with Gini impurity.
            # WHY depth=3: Only 3 levels of decisions = at most 8 leaf nodes.
            # This creates a simple, highly interpretable model that captures only
            # the strongest patterns. Think of it as asking 3 yes/no questions.
            # Risk: may miss subtle but important patterns (underfitting).
            # Best for: initial exploration, when interpretability > accuracy.
            "max_depth": 3,            # Shallow tree: max 3 levels of splits
            "criterion": "gini",       # Gini impurity (faster, no log operation)
            "min_samples_leaf": 5,     # Conservative: leaves need at least 5 samples
        },
        "Medium (depth=10, entropy)": {
            # Config 2: Moderate depth tree with entropy (information gain).
            # WHY depth=10: Allows the tree to capture more complex patterns while
            # still preventing extreme overfitting. 10 levels = up to 1024 leaves.
            # WHY entropy: information gain from information theory sometimes produces
            # more balanced trees than Gini. Worth comparing on your data.
            # Best for: balanced accuracy vs complexity trade-off.
            "max_depth": 10,           # Medium depth: captures moderate complexity
            "criterion": "entropy",    # Entropy-based splitting (information gain)
            "min_samples_leaf": 2,     # Moderate constraint on leaf size
        },
        "Deep (depth=None, gini)": {
            # Config 3: Fully grown tree with no depth limit.
            # WHY depth=None: The tree grows until each leaf is pure or has fewer
            # samples than min_samples_split. This maximizes training accuracy but
            # often overfits because the tree memorizes training data noise.
            # Risk: high variance (different training data -> very different trees).
            # Best for: when you have very clean data or plan to prune later.
            "max_depth": None,         # No depth limit: tree grows fully
            "criterion": "gini",       # Gini impurity
            "min_samples_leaf": 1,     # Allow single-sample leaves (maximum fitting)
        },
        "Pruned (depth=None, min_leaf=10)": {
            # Config 4: Grown tree with aggressive leaf-size pruning.
            # WHY min_samples_leaf=10: Instead of limiting depth, we ensure every leaf
            # has at least 10 samples. This prevents the tree from memorizing outliers
            # while still allowing deep trees for complex patterns.
            # WHY entropy: combines information-theoretic splitting with leaf pruning.
            # Best for: when the optimal depth is unknown but you want regularization.
            "max_depth": None,         # No depth limit
            "criterion": "entropy",    # Entropy-based splitting
            "min_samples_leaf": 10,    # Each leaf must have at least 10 samples
        },
    }

    # Print a header for the comparison table.
    print("\n" + "=" * 100)  # Visual separator
    print("DECISION TREE (sklearn) - HYPERPARAMETER COMPARISON")
    print("=" * 100)
    # Print column headers aligned for readability.
    # WHY these columns: Accuracy, Precision, Recall, F1, AUC + tree structure info.
    print(
        f"{'Config':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'AUC':>10} {'Depth':>7} {'Leaves':>7}"
    )
    print("-" * 100)  # Column separator

    # Train and evaluate each configuration.
    for name, params in configs.items():
        # Train the model with this specific configuration.
        # WHY train(): ensures consistent default handling and random_state.
        model = train(X_train, y_train, **params)  # Build tree with this config
        # Evaluate on validation set (NOT test set).
        # WHY validation: test set is reserved for final, unbiased evaluation.
        metrics = _evaluate(model, X_val, y_val)  # Compute metrics

        # Print formatted results including tree structure information.
        print(
            f"{name:<35} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f} "
            f"{model.get_depth():>7d} {model.get_n_leaves():>7d}"
        )

    # Print interpretation guide explaining what the results mean.
    print("-" * 100)
    print("INTERPRETATION GUIDE:")
    print("  - Shallow (depth=3): Most interpretable, fewest leaves, may underfit complex data.")
    print("  - Medium (depth=10): Good balance between complexity and generalization.")
    print("  - Deep (no limit): Highest training accuracy, but likely overfits (watch val vs train gap).")
    print("  - Pruned (min_leaf=10): Controls overfitting through leaf size, not depth.")
    print("  - Gini vs Entropy: Usually similar results; entropy slightly favors balanced splits.")
    print("  - More leaves = more complex model = higher risk of overfitting.")
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Real-World Demo: Medical Diagnosis (Heart Disease)
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate Decision Tree on a realistic medical diagnosis problem.

    WHY this scenario: Medical diagnosis is one of the most compelling use cases for
    decision trees because:
    1. Doctors need EXPLAINABLE models - a tree can be printed as a set of rules.
    2. Medical guidelines are often already structured as decision trees.
    3. Regulatory compliance (FDA, HIPAA) often requires model interpretability.
    4. Lives are at stake, so understanding WHY a model makes a prediction is critical.

    Domain context:
        - Predict whether a patient has heart disease based on clinical measurements.
        - Features include age, blood pressure, cholesterol, max heart rate, etc.
        - The target is binary: 1 = heart disease present, 0 = no heart disease.
        - Early detection enables lifestyle changes and medical intervention.

    Features generated (synthetic but clinically realistic):
        - age: Patient age in years (30-80). Risk increases with age.
        - resting_bp: Resting blood pressure in mmHg (90-200). Hypertension = risk.
        - cholesterol: Serum cholesterol in mg/dl (120-400). High = risk.
        - max_heart_rate: Maximum heart rate during exercise (70-210). Low = risk.
        - exercise_angina: Exercise-induced chest pain (0 or 1). Angina = risk.
        - blood_sugar_high: Fasting blood sugar > 120 mg/dl (0 or 1). High = risk.
    """
    print("\n" + "=" * 90)  # Visual separator
    print("REAL-WORLD DEMO: Medical Diagnosis (Heart Disease Prediction)")
    print("=" * 90)

    # Set random seed for reproducible demo results.
    np.random.seed(42)  # Fixed seed for consistent synthetic data
    n_samples = 1500    # Number of synthetic patients

    # --- Generate clinically realistic synthetic features ---

    # Age: patient age in years. Heart disease risk increases with age.
    # WHY uniform(30, 80): typical clinical study population range.
    age = np.random.uniform(30, 80, n_samples)  # Age in years

    # Resting blood pressure: measured in mmHg.
    # WHY normal(130, 20): centers around 130 (pre-hypertension), with spread.
    # WHY clip(90, 200): physiologically plausible range.
    resting_bp = np.random.normal(130, 20, n_samples).clip(90, 200)  # Blood pressure

    # Cholesterol: serum cholesterol in mg/dl.
    # WHY normal(240, 50): centers around 240 (borderline high), realistic spread.
    # WHY clip(120, 400): physiologically plausible range.
    cholesterol = np.random.normal(240, 50, n_samples).clip(120, 400)  # Cholesterol level

    # Maximum heart rate achieved during exercise.
    # WHY formula: max HR decreases with age (rough formula: 220 - age + noise).
    # WHY clip(70, 210): prevents physiologically impossible values.
    max_heart_rate = (220 - age + np.random.normal(0, 15, n_samples)).clip(70, 210)  # Max HR

    # Exercise-induced angina: binary indicator of chest pain during exercise.
    # WHY binomial(1, 0.3): about 30% of clinical populations experience this.
    exercise_angina = np.random.binomial(1, 0.3, n_samples)  # Angina presence

    # High fasting blood sugar: indicator of potential diabetes (risk factor).
    # WHY binomial(1, 0.2): about 20% prevalence of elevated blood sugar.
    blood_sugar_high = np.random.binomial(1, 0.2, n_samples)  # High blood sugar

    # --- Generate heart disease label based on clinical risk factors ---
    # WHY this formula: combines risk factors with clinically appropriate signs.
    # Positive coefficients increase disease risk, negative decrease it.
    disease_score = (
        0.04 * age                    # Older age increases risk
        + 0.02 * resting_bp           # Higher BP increases risk
        + 0.01 * cholesterol          # Higher cholesterol increases risk
        - 0.03 * max_heart_rate       # Higher exercise capacity = lower risk
        + 1.5 * exercise_angina       # Chest pain during exercise is a strong indicator
        + 0.8 * blood_sugar_high      # Diabetes risk factor
        + np.random.normal(0, 1.5, n_samples)  # Random noise for realistic uncertainty
        - 5.0                         # Offset to center the probability distribution
    )
    # Convert continuous score to probability using the sigmoid function.
    # WHY sigmoid: maps any real number to [0, 1], giving us a valid probability.
    disease_prob = 1 / (1 + np.exp(-disease_score))  # Sigmoid transformation
    # Generate binary labels by sampling from Bernoulli distribution.
    # WHY Bernoulli: adds realistic label noise (not all high-risk patients get disease).
    y = (np.random.random(n_samples) < disease_prob).astype(int)  # Binary labels

    # Stack all features into a 2D feature matrix.
    # WHY column_stack: creates (n_samples, n_features) array from individual arrays.
    X = np.column_stack([
        age, resting_bp, cholesterol, max_heart_rate,  # Continuous features
        exercise_angina, blood_sugar_high,              # Binary features
    ])

    # Define feature names for interpretability when examining the tree.
    feature_names = [
        "age", "resting_bp", "cholesterol", "max_heart_rate",
        "exercise_angina", "blood_sugar_high",
    ]

    # Print dataset summary statistics.
    print(f"\nDataset: {n_samples} patients, {len(feature_names)} features")
    print(f"Disease prevalence: {y.mean():.1%} ({y.sum()} positive / {n_samples - y.sum()} negative)")
    print(f"Features: {', '.join(feature_names)}")

    # Split into train/test sets with stratification to preserve disease rate.
    # WHY 80/20 split: common choice; 80% for training, 20% for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # NOTE: We intentionally skip scaling here to demonstrate that decision trees
    # work perfectly well without feature normalization (unlike logistic regression or SVM).
    # WHY no scaling: Decision trees split on thresholds (e.g., "age > 55"),
    # so the absolute scale of features doesn't matter.

    # Train a moderately constrained tree for interpretability.
    # WHY max_depth=5: produces a tree that can be printed and understood by a doctor.
    model = DecisionTreeClassifier(
        max_depth=5,              # Limit depth for interpretability
        criterion="gini",         # Standard Gini impurity
        min_samples_leaf=10,      # Each leaf needs at least 10 patients
        random_state=42,          # Reproducibility
    )
    model.fit(X_train, y_train)  # Build the tree from training data

    # Evaluate on the test set.
    y_pred = model.predict(X_test)       # Class predictions
    y_proba = model.predict_proba(X_test)  # Probability estimates

    # Print classification performance metrics.
    print("\n--- Model Performance ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    # WHY recall matters here: missing a heart disease case (false negative) is far
    # more dangerous than a false positive (unnecessary follow-up test).
    print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_test, y_proba[:, 1]):.4f}")

    # Display feature importance computed by the tree.
    # WHY feature_importances_: computed as the total impurity reduction brought by
    # each feature across all splits. Higher = the feature is used more for splitting.
    print("\n--- Feature Importance (Impurity Reduction) ---")
    importances = model.feature_importances_  # Array of importance scores
    # Sort features by importance (most important first).
    sorted_idx = np.argsort(importances)[::-1]  # Descending order indices
    for i in sorted_idx:
        # Print each feature with its importance score and a visual bar.
        bar = "#" * int(importances[i] * 50)  # Visual importance bar
        print(f"  {feature_names[i]:<20} importance={importances[i]:.4f}  {bar}")

    # Display tree structure information.
    print(f"\n--- Tree Structure ---")
    print(f"  Tree depth: {model.get_depth()}")       # Actual depth achieved
    print(f"  Number of leaves: {model.get_n_leaves()}")  # Terminal nodes

    # Print medical interpretation and clinical recommendations.
    print("\n--- Clinical Insights ---")
    print("  1. Decision trees produce IF-THEN rules that doctors can validate.")
    print("  2. Max heart rate and exercise angina are typically strong predictors.")
    print("  3. The tree can be exported as a flowchart for clinical guidelines.")
    print("  4. Feature importance helps prioritize which tests to order.")
    print("  5. Shallow trees (depth=5) are preferred for regulatory compliance.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,         # Optuna trial object for suggesting hyperparameters
    X_train: np.ndarray,         # Training features
    y_train: np.ndarray,         # Training labels
    X_val: np.ndarray,           # Validation features
    y_val: np.ndarray,           # Validation labels
) -> float:
    """Optuna objective function for Decision Tree hyperparameter optimization.

    WHY Optuna: Uses Tree-structured Parzen Estimator (TPE) for intelligent search
    of the hyperparameter space. It models P(hyperparams | good_performance) and
    P(hyperparams | bad_performance) separately, then samples from regions with
    high probability of good performance.

    Returns:
        F1 score on validation set (Optuna will try to maximize this).
    """
    # max_depth: controls how deep the tree can grow.
    # WHY range 2-30: depth=2 is very simple (4 leaves max), depth=30 is very complex.
    # WHY int: depth must be a positive integer.
    max_depth = trial.suggest_int("max_depth", 2, 30)  # Tree depth limit

    # criterion: impurity measure used for splitting.
    # WHY categorical: only two valid options for classification trees.
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])  # Split criterion

    # min_samples_split: minimum samples required to consider splitting a node.
    # WHY range 2-50: 2 is the minimum (any node with 2+ samples can split),
    # 50 is conservative (only split if we have strong statistical support).
    min_samples_split = trial.suggest_int("min_samples_split", 2, 50)  # Min samples to split

    # min_samples_leaf: minimum samples required in each resulting leaf.
    # WHY range 1-30: 1 allows pure leaves (risk of overfitting),
    # 30 forces leaves to represent broader regions (stronger regularization).
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)  # Min leaf samples

    # max_features: number of features to consider at each split.
    # WHY these options: None=all features, "sqrt"=sqrt(n_features), "log2"=log2(n_features).
    # Using fewer features adds randomness and can reduce overfitting.
    max_features = trial.suggest_categorical(
        "max_features", [None, "sqrt", "log2"]  # Feature selection strategy
    )

    # Train the model with the trial's hyperparameters.
    model = train(
        X_train, y_train,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )
    # Evaluate on validation set and return F1 for Optuna to maximize.
    # WHY F1: balances precision and recall, more robust than accuracy for imbalanced data.
    metrics = validate(model, X_val, y_val)  # Compute validation metrics
    return metrics["f1"]  # Return F1 score as the optimization target


def ray_tune_search(
    X_train: np.ndarray,        # Training features
    y_train: np.ndarray,        # Training labels
    X_val: np.ndarray,          # Validation features
    y_val: np.ndarray,          # Validation labels
    num_samples: int = 20,      # Number of hyperparameter configurations to try
) -> Dict[str, Any]:
    """Ray Tune distributed hyperparameter search for Decision Tree.

    WHY Ray Tune: For distributed hyperparameter search across multiple machines
    or when you need advanced scheduling (ASHA for early stopping of bad trials).
    Overkill for decision trees (which train fast), but included for completeness.

    Returns:
        Best hyperparameter configuration found by Ray Tune.
    """
    # Lazy import: Ray is a heavy dependency (~200MB) not always needed.
    # WHY lazy: Allows using this file without Ray installed (only Optuna needed).
    import ray  # Distributed computing framework
    from ray import tune as ray_tune  # Hyperparameter tuning module

    # Initialize Ray if not already running.
    # WHY ignore_reinit_error: prevents crashes if Ray was already initialized.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Define the trainable function that Ray will call for each trial.
    # WHY closure: captures the data arrays from the enclosing scope.
    def _trainable(config: Dict[str, Any]) -> None:
        """Inner function that trains and evaluates one configuration."""
        model = train(X_train, y_train, **config)      # Train with this config
        metrics = validate(model, X_val, y_val)         # Evaluate
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])  # Report to Ray

    # Define the search space using Ray's API.
    # WHY these ranges: same rationale as Optuna, expressed with Ray's API.
    search_space = {
        "max_depth": ray_tune.randint(2, 31),            # Uniform integer [2, 30]
        "criterion": ray_tune.choice(["gini", "entropy"]),  # Categorical
        "min_samples_split": ray_tune.randint(2, 51),     # Uniform integer [2, 50]
        "min_samples_leaf": ray_tune.randint(1, 31),      # Uniform integer [1, 30]
    }

    # Create and run the Tuner.
    tuner = ray_tune.Tuner(
        _trainable,                     # Function to run for each trial
        param_space=search_space,       # Hyperparameter search space
        tune_config=ray_tune.TuneConfig(
            num_samples=num_samples,    # Total number of trials
            metric="f1",               # Metric to optimize
            mode="max",                # Maximize (not minimize)
        ),
    )
    results = tuner.fit()              # Run all trials
    # Extract the best configuration.
    best = results.get_best_result(metric="f1", mode="max")
    logger.info("Ray Tune best config: %s", best.config)
    ray.shutdown()                      # Clean up Ray resources
    return best.config                  # Return best hyperparameters


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Decision Tree (sklearn) pipeline.

    Pipeline structure (order is deliberate):
        1. Generate data -> establish the learning problem
        2. Baseline training -> reference point with default params
        3. Parameter comparison -> understand hyperparameter effects
        4. Real-world demo -> practical application with domain context
        5. Optuna optimization -> find best hyperparameters systematically
        6. Final test evaluation -> unbiased performance estimate
    """
    # Print a prominent header for this implementation.
    logger.info("=" * 70)
    logger.info("Decision Tree - Scikit-Learn Implementation")
    logger.info("=" * 70)

    # Step 1: Generate synthetic classification data.
    # WHY first: everything else depends on having data.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Train a baseline model with default hyperparameters.
    # WHY baseline first: gives us a reference to compare Optuna results against.
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)  # Train with defaults
    validate(model, X_val, y_val)    # Check baseline performance

    # Step 3: Compare different hyperparameter configurations.
    # WHY before Optuna: builds intuition about which parameters matter most.
    logger.info("\n--- Hyperparameter Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 4: Run the real-world medical diagnosis demo.
    # WHY: demonstrates practical application with domain-specific interpretation.
    real_world_demo()

    # Step 5: Run Optuna hyperparameter optimization.
    # WHY: systematically searches for the best configuration using Bayesian optimization.
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")  # Maximize F1 score
    # Run 30 trials: balances search thoroughness with computation time.
    # WHY 30: Decision trees train fast, so 30 trials complete quickly.
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=30,               # Number of hyperparameter configurations to try
        show_progress_bar=True,    # Visual progress indicator
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Step 6: Retrain with the best hyperparameters.
    # WHY retrain: Optuna trains and discards models; we need the final best model.
    best_model = train(X_train, y_train, **study.best_params)

    # Step 7: Final test evaluation (ONLY time we touch the test set).
    # WHY final: this gives us an unbiased estimate of real-world performance.
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Step 8: Ray Tune (optional, commented out for speed).
    # WHY commented: Ray adds significant startup overhead for a fast-training algorithm.
    # logger.info("\n--- Ray Tune Search ---")
    # ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20)
    # ray_model = train(X_train, y_train, **ray_config)
    # test(ray_model, X_test, y_test)


# This guard ensures main() only runs when the script is executed directly.
# WHY: Allows importing functions (train, compare_parameter_sets, etc.) without
# triggering the full pipeline.
if __name__ == "__main__":
    main()
