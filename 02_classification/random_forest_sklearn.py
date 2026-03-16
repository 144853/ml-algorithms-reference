"""
Random Forest Classifier - Scikit-Learn Implementation
=======================================================

Theory & Mathematics:
    Random Forest is an ensemble learning method that constructs multiple
    decision trees during training and outputs the class that is the mode
    (majority vote) of the individual trees' predictions.

    Key Concepts:

    1. Bootstrap Aggregating (Bagging):
       Each tree is trained on a bootstrap sample (random sample with
       replacement) of the training data. This reduces variance and helps
       prevent overfitting.

    2. Random Feature Subsets:
       At each split in a tree, only a random subset of features is
       considered. Typical choices:
           - sqrt(n_features) for classification
           - n_features / 3 for regression
       This decorrelates the trees, improving ensemble diversity.

    3. Decision Tree Splitting Criteria:
       - Gini Impurity: G(t) = 1 - sum_k p_k^2
       - Entropy / Information Gain: H(t) = -sum_k p_k * log2(p_k)

    4. Prediction:
       For classification: majority vote across all trees.
       Probability: average of predicted class probabilities.

    5. Out-of-Bag (OOB) Error:
       Each tree is trained on ~63% of data. The remaining ~37% (OOB samples)
       can be used for internal validation without a separate validation set.

    Bias-Variance Trade-off:
       Individual decision trees have low bias but high variance (overfitting).
       By averaging many trees trained on different bootstrap samples with
       random feature subsets, the forest reduces variance while maintaining
       low bias.

Business Use Cases:
    - Credit risk assessment and fraud detection
    - Medical diagnosis with feature importance for interpretability
    - Customer segmentation and churn prediction
    - Predictive maintenance in manufacturing
    - Environmental/ecological species classification

Advantages:
    - Handles high-dimensional data without feature selection
    - Robust to outliers and noisy features
    - Provides feature importance rankings
    - Parallelizable training (embarrassingly parallel)
    - Works well out-of-the-box with minimal tuning
    - Handles missing values and mixed feature types

Disadvantages:
    - Less interpretable than single decision trees
    - Can be slow for very large datasets (many trees)
    - Biased toward features with more levels in categorical splits
    - Memory-intensive (stores all trees)
    - May overfit on very noisy datasets with deep trees

Hyperparameters:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of each tree
    - min_samples_split: Minimum samples required to split an internal node
    - min_samples_leaf: Minimum samples required at a leaf node
    - max_features: Number of features to consider at each split
    - criterion: Splitting criterion ('gini' or 'entropy')
    - bootstrap: Whether to use bootstrap samples
    - class_weight: Weights for handling class imbalance
"""

# --- Standard library imports ---
import logging    # Structured logging with severity levels for experiment tracking.
import warnings   # Suppress sklearn's convergence/deprecation warnings.
from typing import Any, Dict, Tuple   # Type annotations for code clarity.

# --- Third-party imports ---
import numpy as np   # Foundation for all numerical operations and array handling.
import optuna         # Bayesian hyperparameter optimization (TPE sampler).
# WHY optuna: Intelligent search converges faster than grid or random search.

from sklearn.datasets import make_classification   # Synthetic data generation.
# WHY: Controlled complexity for benchmarking. Real datasets have unknown ground truth.

from sklearn.ensemble import RandomForestClassifier  # sklearn's optimized RF implementation.
# WHY sklearn: Wraps C-optimized code with parallel execution via joblib.
# It's typically 100x faster than a pure Python implementation.

from sklearn.metrics import (
    accuracy_score,           # (TP+TN) / total - simple but can be misleading.
    classification_report,    # Per-class precision/recall/F1 breakdown.
    confusion_matrix,         # Shows error types (FP vs FN).
    f1_score,                 # Harmonic mean of precision and recall.
    precision_score,          # Of positive predictions, how many are correct.
    recall_score,             # Of actual positives, how many did we catch.
    roc_auc_score,            # Area under ROC curve - threshold independent.
)
from sklearn.model_selection import train_test_split  # Stratified data splitting.
from sklearn.preprocessing import StandardScaler      # Feature standardization.

# --- Logging setup ---
# WHY: Logging with timestamps helps track experiment progress and debug issues.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# Suppress warnings from sklearn during hyperparameter search.
warnings.filterwarnings("ignore")


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

    WHY synthetic data: Full control over informative/redundant/noise features
    lets us benchmark the algorithm's ability to identify true signal.
    """
    # Generate data with controlled signal-to-noise ratio.
    X, y = make_classification(
        n_samples=n_samples,            # Total samples in the dataset.
        n_features=n_features,          # Total number of features.
        n_informative=n_features // 2,  # Half carry real predictive signal.
        n_redundant=n_features // 4,    # Quarter are linear combos of informative features.
        # WHY: Tests RF's ability to handle multicollinearity (which trees handle well).
        n_classes=n_classes,            # Binary classification by default.
        random_state=random_state,      # Reproducibility.
    )

    # 60/20/20 stratified split to maintain class balance.
    # WHY stratify: Ensures each split has the same class distribution as the full dataset.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    # Random forests are tree-based and don't strictly require scaling,
    # but we include it for consistency with other implementations.
    # WHY: Tree-based models split on thresholds, so the absolute scale of features
    # doesn't affect the splits. However, scaling ensures fair comparison with LR/SVM.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    WHY Random Forest: It's one of the most reliable "off-the-shelf" classifiers.
    It rarely overfits with enough trees, handles nonlinear relationships naturally,
    and provides feature importance out of the box.
    """
    # Default hyperparameters chosen for a good balance of performance and speed.
    defaults = dict(
        n_estimators=100,       # Number of trees in the forest.
        # WHY 100: More trees = lower variance (better generalization).
        # Diminishing returns after ~100-200 trees. Training time scales linearly.
        max_depth=None,         # No maximum depth limit (trees grow until pure leaves).
        # WHY None: Letting trees grow deep gives low bias. The ensemble averaging
        # handles the high variance of individual deep trees.
        min_samples_split=2,    # Minimum samples to attempt a split (smallest possible).
        # WHY 2: Allows maximum tree complexity. Regularization comes from the ensemble.
        min_samples_leaf=1,     # Minimum samples at a leaf node.
        # WHY 1: Each leaf can represent a single sample. Again, the ensemble provides regularization.
        max_features="sqrt",    # Consider sqrt(n_features) features at each split.
        # WHY sqrt: This is the classic Random Forest choice for classification.
        # It forces trees to use different feature subsets, decorrelating them.
        # Lower max_features = more diverse trees = lower correlation = better ensemble.
        criterion="gini",       # Splitting criterion: Gini impurity.
        # WHY gini: Gini and entropy produce very similar trees in practice.
        # Gini is slightly faster to compute (no logarithm needed).
        bootstrap=True,         # Use bootstrap sampling (sampling with replacement).
        # WHY True: Bootstrap is essential for Random Forests. Without it,
        # all trees see the same data and become correlated, reducing ensemble benefit.
        n_jobs=-1,              # Use all CPU cores for parallel training.
        # WHY -1: Each tree is independent, so training is embarrassingly parallel.
        # Using all cores gives a near-linear speedup.
        random_state=42,        # Reproducibility.
    )
    # Override defaults with user-provided hyperparameters.
    defaults.update(hyperparams)

    # Instantiate and train the Random Forest.
    model = RandomForestClassifier(**defaults)
    # .fit() builds all n_estimators trees in parallel (using n_jobs cores).
    # WHY: Each tree is fitted on a bootstrap sample with random feature subsets.
    model.fit(X_train, y_train)

    # Log training info (exclude n_jobs from the output as it's not a model hyperparameter).
    logger.info("Random Forest trained with params: %s", {k: v for k, v in defaults.items() if k != "n_jobs"})
    return model


def _evaluate(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics for the Random Forest model.

    WHY separate from validate/test: Shared evaluation logic avoids code duplication
    and ensures consistent metric computation across validation and test phases.
    """
    # Get class predictions (majority vote across all trees).
    y_pred = model.predict(X)
    # Get probability estimates (average of per-tree class distributions).
    # WHY probabilities: Needed for AUC-ROC and for ranking predictions.
    y_proba = model.predict_proba(X)

    # Handle binary vs multiclass AUC computation.
    n_classes = len(np.unique(y))
    if n_classes == 2:
        auc = roc_auc_score(y, y_proba[:, 1])  # Use P(class=1) for binary.
    else:
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: RandomForestClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Validate the model on the validation set for hyperparameter tuning."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Final test evaluation -- call only once with the best model."""
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))

    # Feature importance: measures how much each feature contributed to reducing impurity.
    # WHY: This is a key advantage of Random Forests -- built-in feature importance.
    # It's computed as the mean decrease in Gini impurity across all splits using that feature.
    importances = model.feature_importances_
    # Sort by importance (most important first).
    indices = np.argsort(importances)[::-1]
    logger.info("\nTop 10 Feature Importances:")
    for rank, idx in enumerate(indices[:10], 1):
        logger.info("  %d. Feature %d: %.4f", rank, idx, importances[idx])

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
    """Compare different Random Forest configurations to illustrate hyperparameter effects.

    WHY: Random Forests have many hyperparameters that control the bias-variance trade-off
    of individual trees and the diversity of the ensemble. Understanding these trade-offs
    is essential for tuning RF effectively.
    """
    configs = {
        "Few Shallow Trees": {
            # Config 1: Small forest with shallow trees.
            # WHY: Shallow trees (max_depth=5) have high bias but low variance.
            # Fewer trees (50) means less averaging, so the ensemble is weaker.
            # This configuration underfits complex datasets but trains very fast.
            # Best for: quick prototyping, when speed matters more than accuracy.
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "criterion": "gini",
        },
        "Many Deep Trees (default)": {
            # Config 2: Large forest with unrestricted trees (the default).
            # WHY: Deep trees (max_depth=None) have very low bias. Many trees (200)
            # provide strong averaging that controls the high variance of deep trees.
            # This is the "standard" RF configuration that works well out of the box.
            # Best for: general-purpose classification; rarely needs modification.
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "criterion": "gini",
        },
        "Entropy + More Features": {
            # Config 3: Using entropy criterion with larger feature subsets.
            # WHY entropy: Information gain (entropy) is slightly more theoretically
            # principled than Gini, but in practice produces very similar results.
            # WHY log2: log2(n_features) considers fewer features per split than sqrt,
            # making trees more diverse (more decorrelated) at the cost of individual accuracy.
            # Best for: when you want maximum ensemble diversity.
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_leaf": 2,
            "max_features": "log2",
            "criterion": "entropy",
        },
        "Regularized Forest": {
            # Config 4: Heavily regularized trees with large minimum leaf size.
            # WHY: min_samples_leaf=10 prevents leaves from overfitting to tiny groups.
            # max_depth=8 limits tree complexity. This produces a simpler model that
            # generalizes better on noisy data but may underfit clean data.
            # Best for: noisy datasets, when overfitting is a major concern.
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "criterion": "gini",
        },
    }

    print("\n" + "=" * 90)
    print("RANDOM FOREST - HYPERPARAMETER COMPARISON")
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
    print("  - Few Shallow Trees: Fastest training, may underfit. Good for quick baselines.")
    print("  - Many Deep Trees: Best default. Strong performance with minimal tuning.")
    print("  - Entropy + More Features: Tests alternative splitting criterion and feature sampling.")
    print("  - Regularized Forest: Best for noisy data. Prevents overfitting via tree constraints.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Real-World Demo: Loan Approval Prediction
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate Random Forest on a realistic loan approval prediction problem.

    WHY this scenario: Loan approval is a classic RF use case because:
    1. Features have complex nonlinear interactions (income AND credit score matter).
    2. Feature importance helps explain decisions to regulators and applicants.
    3. The model must be robust to noisy and missing data (RF handles this well).
    4. The problem has clear business value: bad loans cost money, good loans generate revenue.

    Domain context:
        - A bank needs to predict whether a loan application should be approved or denied.
        - Features include applicant's income, credit score, debt-to-income ratio, etc.
        - The target is binary: 1 = approved, 0 = denied.
        - Feature importance helps explain WHY a decision was made (regulatory requirement).
    """
    print("\n" + "=" * 90)
    print("REAL-WORLD DEMO: Loan Approval Prediction (Banking)")
    print("=" * 90)

    np.random.seed(42)
    n_samples = 2000

    # --- Generate realistic loan application features ---
    # Annual income: right-skewed distribution (most earn moderate, few earn very high).
    # WHY lognormal: Income distributions are typically right-skewed in real populations.
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).clip(20000, 300000)

    # Credit score: normally distributed around 680 (average US credit score).
    # WHY 680 center: This is the median FICO score in the US population.
    credit_score = np.random.normal(680, 60, n_samples).clip(300, 850).astype(int)

    # Debt-to-income ratio: fraction of income going to debt payments.
    # WHY uniform(0.05, 0.6): DTI ranges from very low debt to heavily indebted.
    debt_to_income = np.random.uniform(0.05, 0.6, n_samples)

    # Employment length: years at current employer.
    # WHY exponential: Most people have short tenure, few have very long tenure.
    employment_years = np.random.exponential(5, n_samples).clip(0, 40)

    # Loan amount requested.
    # WHY proportional to income: People tend to request loans proportional to their income.
    loan_amount = annual_income * np.random.uniform(0.1, 0.5, n_samples)

    # Number of existing credit lines (credit cards, mortgages, etc.).
    # WHY poisson: Count data is well-modeled by Poisson distribution.
    num_credit_lines = np.random.poisson(5, n_samples)

    # --- Generate approval labels based on realistic business rules ---
    # WHY this formula: Combines all features with domain-appropriate weights.
    # Higher income, better credit score, lower DTI, longer employment = more likely approved.
    approval_score = (
        0.00003 * annual_income       # Higher income increases approval odds.
        + 0.01 * credit_score          # Better credit score strongly increases odds.
        - 3.0 * debt_to_income         # High DTI is a strong negative signal.
        + 0.05 * employment_years      # Stable employment is a positive signal.
        - 0.000005 * loan_amount       # Larger loan amounts are riskier.
        + 0.05 * num_credit_lines      # More credit lines show credit experience.
        - 8.0                          # Offset to balance approval rates.
        + np.random.normal(0, 1.0, n_samples)  # Noise for realistic uncertainty.
    )
    approval_prob = 1 / (1 + np.exp(-approval_score))
    y = (np.random.random(n_samples) < approval_prob).astype(int)

    X = np.column_stack([
        annual_income, credit_score, debt_to_income,
        employment_years, loan_amount, num_credit_lines,
    ])
    feature_names = [
        "annual_income", "credit_score", "debt_to_income",
        "employment_years", "loan_amount", "num_credit_lines",
    ]

    print(f"\nDataset: {n_samples} applications, {len(feature_names)} features")
    print(f"Approval rate: {y.mean():.1%} ({y.sum()} approved / {n_samples - y.sum()} denied)")
    print(f"Features: {', '.join(feature_names)}")

    # Split and train.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Train Random Forest (no scaling needed for trees, but we do it for consistency).
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n--- Model Performance ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_test, y_proba[:, 1]):.4f}")

    # Feature importance (key advantage of Random Forests).
    # WHY: In banking, regulators require model explainability.
    # RF feature importance shows which factors drive lending decisions.
    print("\n--- Feature Importance (Mean Decrease in Impurity) ---")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        bar = "#" * int(importances[i] * 50)  # Visual bar chart.
        print(f"  {feature_names[i]:<20} {importances[i]:.4f}  {bar}")

    print("\n--- Business Insights ---")
    print("  1. Credit score and income are the top predictors (as expected in lending).")
    print("  2. Debt-to-income ratio is a strong negative predictor of approval.")
    print("  3. Random Forest captures nonlinear interactions (e.g., high income + low DTI).")
    print("  4. Feature importance helps satisfy regulatory explainability requirements.")
    print("  5. The model can handle missing values and outliers robustly.")
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
    """Optuna objective for Random Forest hyperparameter optimization.

    WHY these ranges: Each range reflects practical knowledge about RF behavior.
    """
    # Number of trees: more trees rarely hurt but increase training time.
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    # Tree depth: controls complexity of individual trees.
    max_depth = trial.suggest_int("max_depth", 3, 30)
    # Minimum samples to split: higher values prevent splits on tiny groups.
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    # Minimum samples in a leaf: regularization via leaf size constraints.
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    # Feature subset strategy at each split.
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    # Splitting criterion: gini vs entropy (usually makes little difference).
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    model = train(
        X_train, y_train,
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, criterion=criterion,
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
    """Ray Tune distributed hyperparameter search for Random Forest."""
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "n_estimators": ray_tune.choice([50, 100, 200, 300, 500]),
        "max_depth": ray_tune.randint(3, 30),
        "min_samples_split": ray_tune.randint(2, 20),
        "min_samples_leaf": ray_tune.randint(1, 10),
        "max_features": ray_tune.choice(["sqrt", "log2"]),
        "criterion": ray_tune.choice(["gini", "entropy"]),
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
    """Run the full Random Forest (sklearn) pipeline."""
    logger.info("=" * 70)
    logger.info("Random Forest Classifier - Scikit-Learn Implementation")
    logger.info("=" * 70)

    # Step 1: Generate synthetic data.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Baseline training with default hyperparameters.
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # Step 3: Compare hyperparameter configurations.
    logger.info("\n--- Hyperparameter Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 4: Real-world loan approval demo.
    real_world_demo()

    # Step 5: Optuna optimization.
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=30,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Step 6: Retrain with best params and evaluate on test set.
    best_model = train(X_train, y_train, **study.best_params)
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
