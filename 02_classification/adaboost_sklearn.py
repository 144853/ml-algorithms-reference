"""
AdaBoost Classifier - Scikit-Learn Implementation
===================================================

Theory & Mathematics:
    AdaBoost (Adaptive Boosting), introduced by Freund & Schapire (1997), is an
    ensemble method that combines multiple weak classifiers into a strong classifier
    by iteratively re-weighting training samples to focus on misclassified examples.

    Core Idea:
        Instead of training one powerful model, train a sequence of simple models
        (weak learners) where each subsequent model focuses on the mistakes of
        previous models. The final prediction is a weighted vote of all models.

    Algorithm (AdaBoost.M1 / SAMME for multi-class):
        Input: Training set {(x_i, y_i)}, i = 1..N, number of rounds T

        1. Initialize sample weights: w_i = 1/N for all i
           WHY uniform: We start with no knowledge of which samples are hard.

        2. For t = 1 to T:
            a. Train weak learner h_t on weighted dataset (X, y, w).
               WHY weak learner: A model slightly better than random (e.g., a
               decision stump - tree with depth 1). Using simple models prevents
               the individual learners from overfitting.

            b. Compute weighted error rate:
               epsilon_t = sum_{i: h_t(x_i) != y_i} w_i / sum_i w_i
               WHY weighted: Samples we've struggled with have higher weights,
               so errors on hard samples contribute more to epsilon.

            c. Compute learner weight (alpha):
               alpha_t = 0.5 * ln((1 - epsilon_t) / epsilon_t)
               WHY this formula: Derived from the exponential loss minimization.
               Better learners (lower epsilon) get higher alpha (more influence).
               A learner with epsilon=0 (perfect) gets alpha=inf.
               A learner with epsilon=0.5 (random) gets alpha=0 (no influence).

            d. Update sample weights:
               w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))
               For correct predictions: w decreases (sample becomes "easier")
               For incorrect predictions: w increases (sample becomes "harder")
               WHY: Forces the next learner to focus on previously misclassified
               samples. This is the "adaptive" in AdaBoost.

            e. Normalize weights: w_i <- w_i / sum_j w_j
               WHY: Keep weights as a valid probability distribution.

        3. Final prediction:
           H(x) = sign(sum_{t=1}^{T} alpha_t * h_t(x))
           WHY weighted sum: Better learners (higher alpha) contribute more
           to the final decision. The ensemble is a weighted majority vote.

    SAMME vs SAMME.R:
        SAMME (Stagewise Additive Modeling using Multi-class Exponential loss):
            Uses discrete predictions from each weak learner.
            alpha_t = ln((1-epsilon_t)/epsilon_t) + ln(K-1)  (K = num classes)
            WHY: Generalization of AdaBoost.M1 to K > 2 classes.

        SAMME.R (R = "Real"):
            Uses probability predictions (real-valued) from each weak learner.
            Weighted class probabilities instead of hard votes.
            WHY: Generally faster convergence because it uses more information
            from each weak learner (probabilities vs hard labels).

    Exponential Loss Function:
        AdaBoost can be viewed as forward stagewise additive modeling that
        minimizes the exponential loss: L(y, f(x)) = exp(-y * f(x))
        WHY exponential: It gives a smooth, differentiable surrogate for the
        0-1 loss, and leads to the elegant alpha_t formula.

    Learning Rate (shrinkage):
        Multiplies alpha_t by a factor nu in (0, 1]:
        H(x) = sum_t (nu * alpha_t) * h_t(x)
        WHY: Slower learning (smaller nu) requires more estimators but often
        produces better generalization. It's a form of regularization that
        makes each step smaller, allowing better fine-grained optimization.

Business Use Cases:
    - Customer churn prediction: detecting at-risk customers early
    - Fraud detection: combining many simple rules into strong detector
    - Face detection: Viola-Jones algorithm uses cascade of AdaBoost classifiers
    - Medical diagnosis: ensemble of simple tests for screening
    - Credit scoring: interpretable weak learners with strong ensemble

Advantages:
    - Theoretically motivated: provably reduces training error exponentially
    - Automatically focuses on hard examples (adaptive weighting)
    - Weak learners can be very simple (decision stumps), making it interpretable
    - Less prone to overfitting than bagging (on many datasets)
    - No hyperparameter for regularization needed (learning rate is optional)
    - Feature importance from weighted vote of stump features

Disadvantages:
    - Sensitive to noisy data and outliers (they get high weights forever)
    - Sequential training: cannot be parallelized (each learner depends on previous)
    - Can overfit if number of estimators is too large with noisy data
    - Performance degrades with very weak learners (epsilon near 0.5)
    - Exponential loss is sensitive to mislabeled data
    - Slower than bagging methods because of sequential nature
"""

# --- Standard library imports ---
import logging  # Structured logging for training progress
import warnings  # Warning suppression during HPO
from typing import Any, Dict, Tuple  # Type annotations

# --- Third-party imports ---
import numpy as np  # Numerical computing
import optuna  # Bayesian hyperparameter optimization

# AdaBoostClassifier: sklearn's implementation of the AdaBoost ensemble method.
# WHY sklearn: Production-grade implementation with SAMME/SAMME.R support,
# built-in sample weighting, and integration with any sklearn base estimator.
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost ensemble classifier

# DecisionTreeClassifier: default weak learner for AdaBoost.
# WHY decision stump: A tree with max_depth=1 is the simplest possible split
# and is the canonical weak learner for boosting.
from sklearn.tree import DecisionTreeClassifier  # Weak learner

from sklearn.datasets import make_classification  # Synthetic data
from sklearn.metrics import (
    accuracy_score,           # Overall correctness
    classification_report,    # Per-class breakdown
    f1_score,                 # Harmonic mean of precision and recall
    precision_score,          # Positive predictive value
    recall_score,             # Sensitivity
    roc_auc_score,            # Area under ROC curve
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardization

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,      # Total number of samples
    n_features: int = 20,       # Total features
    n_classes: int = 2,         # Number of classes
    random_state: int = 42,     # Random seed
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data and split into train/val/test.

    WHY synthetic: Controlled benchmarking conditions with known properties.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Generate synthetic dataset with controllable signal-to-noise.
    X, y = make_classification(
        n_samples=n_samples,            # Total data points
        n_features=n_features,          # Feature dimensionality
        n_informative=n_features // 2,  # Half carry real signal
        n_redundant=n_features // 4,    # Quarter are linear combos
        n_classes=n_classes,            # Number of classes
        random_state=random_state,      # Reproducibility seed
    )

    # First split: 60% train, 40% temp.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )

    # Second split: 50/50 temp -> 20% val, 20% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Standardize features.
    # WHY for AdaBoost: Decision stumps don't need scaling, but standardizing
    # keeps consistency across the repository and doesn't hurt performance.
    scaler = StandardScaler()  # Create scaler
    X_train = scaler.fit_transform(X_train)  # Fit on train, transform
    X_val = scaler.transform(X_val)          # Transform val with train stats
    X_test = scaler.transform(X_test)        # Transform test with train stats

    logger.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,                # Training features
    y_train: np.ndarray,                # Training labels
    n_estimators: int = 50,             # Number of weak learners (boosting rounds)
    learning_rate: float = 1.0,         # Learning rate (shrinkage)
    algorithm: str = "SAMME",           # Boosting algorithm variant
    max_depth: int = 1,                 # Base estimator depth (1 = stump)
) -> AdaBoostClassifier:
    """Train an AdaBoost classifier with decision tree weak learners.

    Args:
        X_train: Training feature matrix.
        y_train: Training label array.
        n_estimators: Number of boosting rounds (weak learners to train).
            WHY: More estimators = more capacity but risk of overfitting.
            With learning_rate < 1, more estimators are needed.
        learning_rate: Shrinkage factor applied to each learner's contribution.
            WHY: Smaller values (e.g., 0.1) produce better generalization but
            require more estimators. Trade-off between speed and quality.
        algorithm: "SAMME" (discrete) or "SAMME.R" (real/probabilistic).
            WHY SAMME.R: Uses probability estimates for more granular updates.
            Usually converges faster than SAMME. Requires base estimator to
            support predict_proba (decision trees do).
        max_depth: Maximum depth of the base decision tree estimator.
            WHY 1 (stump): A single split is the canonical weak learner.
            Deeper base learners can capture more complex patterns but risk
            making individual learners too strong (reducing ensemble benefit).

    Returns:
        Trained AdaBoostClassifier model.
    """
    logger.info(
        f"Training AdaBoost: n_estimators={n_estimators}, lr={learning_rate}, "
        f"algorithm={algorithm}, base_depth={max_depth}"
    )

    # Create the base estimator (weak learner).
    # WHY DecisionTreeClassifier: Trees naturally handle weighted samples
    # (which AdaBoost requires), and with max_depth=1, they're simple enough
    # to be "weak" learners that are just slightly better than random.
    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth,     # Depth limit (1 = stump)
        random_state=42,         # Reproducibility
    )

    # Create the AdaBoost ensemble.
    model = AdaBoostClassifier(
        estimator=base_estimator,     # Weak learner template
        n_estimators=n_estimators,    # Number of boosting rounds
        learning_rate=learning_rate,  # Shrinkage factor
        algorithm=algorithm,          # SAMME or SAMME.R
        random_state=42,              # Reproducibility
    )

    # Fit the ensemble on training data.
    # WHY: This runs T rounds of boosting, each time training a new weak learner
    # on the re-weighted dataset and computing the learner's weight alpha_t.
    model.fit(X_train, y_train)  # Train the AdaBoost ensemble

    # Log training accuracy.
    train_preds = model.predict(X_train)  # Training predictions
    train_acc = accuracy_score(y_train, train_preds)  # Accuracy
    logger.info(f"Training accuracy: {train_acc:.4f}")  # Log result

    # Log number of estimators actually used (may be fewer if early stopped).
    logger.info(f"Estimators trained: {len(model.estimators_)}")  # Actual count

    return model  # Return the trained ensemble


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: AdaBoostClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate the AdaBoost model on the validation set.

    WHY: Validation set provides a generalization estimate during HPO.

    Args:
        model: Trained AdaBoost classifier.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Dictionary of metric names to values.
    """
    # Generate predictions.
    y_pred = model.predict(X_val)  # Hard class predictions
    y_proba = model.predict_proba(X_val)  # Probability predictions

    # Compute comprehensive metrics.
    n_classes = len(np.unique(y_val))  # Number of classes
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),  # Overall correctness
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_val, y_proba[:, 1]) if n_classes == 2 else 0.0,
    }

    logger.info("Validation Metrics:")  # Header
    for name, value in metrics.items():  # Log each metric
        logger.info(f"  {name}: {value:.4f}")

    return metrics  # Return metrics dictionary


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test(
    model: AdaBoostClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Final evaluation on the held-out test set.

    WHY: Provides unbiased estimate after all tuning is complete.

    Args:
        model: Trained AdaBoost classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of metric names to values.
    """
    y_pred = model.predict(X_test)  # Hard predictions
    y_proba = model.predict_proba(X_test)  # Probability predictions

    n_classes = len(np.unique(y_test))
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba[:, 1]) if n_classes == 2 else 0.0,
    }

    logger.info("=" * 50)  # Visual separator
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
    """Optuna objective function for AdaBoost hyperparameter optimization.

    Searches over n_estimators, learning_rate, algorithm, and base depth.

    WHY these ranges:
    - n_estimators 10-500: covers quick ensembles to large ones.
    - learning_rate 0.01-2.0: covers cautious to aggressive boosting.
    - algorithm SAMME vs SAMME: tests discrete vs probabilistic.
    - max_depth 1-3: stump vs slightly deeper weak learner.

    Args:
        trial: Optuna Trial for parameter suggestions.

    Returns:
        Validation F1 score to maximize.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()  # Get data

    # Suggest hyperparameters.
    n_estimators = trial.suggest_int("n_estimators", 10, 500)  # Number of rounds
    learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True)  # LR
    algorithm = trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"])  # Variant
    max_depth = trial.suggest_int("max_depth", 1, 3)  # Base estimator depth

    # Train and evaluate.
    model = train(
        X_train, y_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        max_depth=max_depth,
    )
    metrics = validate(model, X_val, y_val)  # Evaluate

    return metrics["f1"]  # Return F1 as objective


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune hyperparameter search space for AdaBoost.

    Returns:
        Dictionary defining the search space.
    """
    search_space = {
        "n_estimators": {"type": "randint", "lower": 10, "upper": 501},
        "learning_rate": {"type": "loguniform", "lower": 0.01, "upper": 2.0},
        "algorithm": {"type": "choice", "values": ["SAMME", "SAMME.R"]},
        "max_depth": {"type": "randint", "lower": 1, "upper": 4},
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
    """Compare different AdaBoost configurations with reasoning.

    Configurations:
    1. Few estimators + high LR (n=50, lr=1.0, SAMME): Quick baseline.
    2. Many estimators + low LR (n=200, lr=0.1, SAMME.R): Slow careful boosting.
    3. Medium + SAMME.R (n=100, lr=0.5): Balanced probabilistic.
    4. Many estimators + high LR (n=200, lr=1.0, SAMME): Test overfitting risk.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    configs = {
        "few_highLR_SAMME": {
            "params": {
                "n_estimators": 50, "learning_rate": 1.0,
                "algorithm": "SAMME", "max_depth": 1,
            },
            "reasoning": (
                "50 estimators with full learning rate and SAMME (discrete). "
                "Quick to train, each stump contributes fully to the ensemble. "
                "Expected: reasonable baseline but may underfit if 50 rounds "
                "aren't enough to capture complex patterns."
            ),
        },
        "many_lowLR_SAMMER": {
            "params": {
                "n_estimators": 200, "learning_rate": 0.1,
                "algorithm": "SAMME.R", "max_depth": 1,
            },
            "reasoning": (
                "200 estimators with low learning rate (0.1) and SAMME.R. "
                "Each round contributes only 10% of its full weight. This is "
                "'slow and careful' boosting: more robust against overfitting. "
                "SAMME.R uses probability predictions for smoother updates. "
                "Expected: best generalization if enough estimators."
            ),
        },
        "medium_SAMMER_balanced": {
            "params": {
                "n_estimators": 100, "learning_rate": 0.5,
                "algorithm": "SAMME.R", "max_depth": 1,
            },
            "reasoning": (
                "100 estimators with moderate learning rate (0.5) and SAMME.R. "
                "Balanced approach: not too aggressive, not too slow. "
                "SAMME.R typically converges faster than SAMME due to using "
                "real-valued confidence instead of discrete predictions."
            ),
        },
        "many_highLR_SAMME_overfit": {
            "params": {
                "n_estimators": 200, "learning_rate": 1.0,
                "algorithm": "SAMME", "max_depth": 1,
            },
            "reasoning": (
                "200 estimators with full learning rate. Tests potential overfitting: "
                "many rounds of full-strength boosting may memorize training noise. "
                "With noisy data, sample weights can become extremely skewed toward "
                "outliers, degrading generalization. Expected: high train accuracy, "
                "potentially lower validation accuracy than low-LR variant."
            ),
        },
    }

    results = {}
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {name}")
        logger.info(f"Reasoning: {config['reasoning']}")
        logger.info(f"Parameters: {config['params']}")

        model = train(X_train, y_train, **config["params"])
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    # Log comparison summary.
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY:")
    for name, metrics in results.items():
        logger.info(f"  {name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate AdaBoost on customer churn prediction.

    Domain: Telecommunications customer churn.
    Features: tenure_months, monthly_charges, contract_type_encoded, payment_method_encoded
    Target: churned (0 = stayed, 1 = churned)

    WHY this domain: Customer churn prediction is a classic business problem where:
    1. Simple rules (short tenure + high charges -> churn risk) are intuitive
    2. Decision stumps can capture these individual rules effectively
    3. Boosting combines many simple rules into a powerful predictor
    4. Feature importance reveals which factors drive churn
    5. Both precision (avoid annoying loyal customers) and recall (catch churners) matter
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Customer Churn Prediction")
    logger.info("=" * 60)

    np.random.seed(42)  # Reproducibility
    n_samples = 1000    # Synthetic customers

    # Generate realistic telecom customer features.

    # tenure_months: how long the customer has been with the company.
    # WHY: New customers churn more (haven't built loyalty yet).
    tenure_months = np.random.exponential(24, n_samples).clip(1, 72)  # 1-72 months

    # monthly_charges: how much the customer pays per month.
    # WHY: Higher charges increase churn risk (price sensitivity).
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 120)  # $20-$120

    # contract_type: encoded as 0=month-to-month, 1=one-year, 2=two-year.
    # WHY: Month-to-month contracts have highest churn (no commitment).
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])

    # payment_method: encoded as 0=electronic_check, 1=mailed_check, 2=bank_transfer, 3=credit_card.
    # WHY: Electronic check users churn more (correlated with less engagement).
    payment_method = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.25, 0.2, 0.2])

    X = np.column_stack([tenure_months, monthly_charges, contract_type, payment_method])
    feature_names = ["tenure_months", "monthly_charges", "contract_type", "payment_method"]

    # Create churn labels based on realistic business logic.
    # WHY: Short tenure, high charges, month-to-month contract, electronic check
    # are all known churn risk factors.
    churn_score = (
        -0.05 * (tenure_months - 12)      # Longer tenure = lower churn risk
        + 0.02 * (monthly_charges - 60)    # Higher charges = higher risk
        - 0.8 * (contract_type)            # Longer contracts = lower risk
        + 0.3 * (payment_method == 0)      # Electronic check = higher risk
    )
    probability = 1.0 / (1.0 + np.exp(-churn_score))  # Sigmoid to probability
    noise = np.random.normal(0, 0.1, n_samples)  # Add noise for realism
    y = (probability + noise > 0.5).astype(int)  # Binary churn label

    # Split data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Standardize.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Log feature statistics.
    logger.info("\nFeature Statistics (Training Set):")
    for i, name in enumerate(feature_names):
        logger.info(
            f"  {name}: mean={X_train[:, i].mean():.2f}, "
            f"std={X_train[:, i].std():.2f}"
        )
    logger.info(f"\nClass distribution: stayed={np.sum(y_train==0)}, churned={np.sum(y_train==1)}")

    # Train AdaBoost.
    logger.info("\n--- Training AdaBoost for Churn Prediction ---")
    model = train(
        X_train, y_train,
        n_estimators=100,     # 100 boosting rounds
        learning_rate=0.5,    # Moderate shrinkage
        algorithm="SAMME.R",  # Probabilistic variant
        max_depth=1,          # Decision stumps as weak learners
    )

    # Log feature importances.
    # WHY: AdaBoost's feature importance shows which features the ensemble
    # relies on most, aggregated across all weak learners.
    logger.info("\nFeature Importances:")
    for name, importance in zip(feature_names, model.feature_importances_):
        logger.info(f"  {name}: {importance:.4f}")

    # Evaluate.
    logger.info("\nValidation Results:")
    validate(model, X_val, y_val)

    logger.info("\nTest Results:")
    test(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete AdaBoost pipeline.

    Steps:
    1. Generate synthetic data
    2. Train baseline model
    3. Validate on validation set
    4. Compare parameter configurations
    5. Run Optuna HPO
    6. Define Ray Tune search space
    7. Train best model from HPO
    8. Final test evaluation
    9. Real-world customer churn demo
    """
    logger.info("=" * 60)
    logger.info("AdaBoost Classifier - Scikit-Learn Implementation")
    logger.info("=" * 60)

    # Step 1: Generate data.
    logger.info("\n--- Step 1: Generating Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Train baseline (50 estimators, LR=1.0).
    logger.info("\n--- Step 2: Training Baseline ---")
    baseline = train(X_train, y_train, n_estimators=50, learning_rate=1.0)

    # Step 3: Validate baseline.
    logger.info("\n--- Step 3: Validating Baseline ---")
    validate(baseline, X_val, y_val)

    # Step 4: Compare parameter sets.
    logger.info("\n--- Step 4: Comparing Parameter Sets ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 5: Optuna HPO.
    logger.info("\n--- Step 5: Optuna HPO ---")
    study = optuna.create_study(direction="maximize")  # Maximize F1
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Quiet Optuna
    study.optimize(optuna_objective, n_trials=25)  # Run 25 trials
    logger.info(f"Best trial F1: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    # Step 6: Ray Tune search space.
    logger.info("\n--- Step 6: Ray Tune Search Space ---")
    ray_tune_search()

    # Step 7: Train best model from HPO.
    logger.info("\n--- Step 7: Training Best Model ---")
    best_params = study.best_trial.params  # Extract best parameters
    best_model = train(X_train, y_train, **best_params)  # Train with best params

    # Step 8: Final test evaluation.
    logger.info("\n--- Step 8: Final Test Evaluation ---")
    test(best_model, X_test, y_test)  # Final unbiased evaluation

    # Step 9: Real-world demo.
    logger.info("\n--- Step 9: Real-World Demo ---")
    real_world_demo()

    logger.info("\n--- Pipeline Complete ---")


# Entry point.
if __name__ == "__main__":
    main()
