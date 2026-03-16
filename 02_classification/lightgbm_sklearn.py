"""
LightGBM Classifier - Scikit-Learn API Implementation
======================================================

Theory & Mathematics:
    LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework
    developed by Microsoft that uses histogram-based algorithms and leaf-wise
    (best-first) tree growth instead of the traditional level-wise approach.

    Core Innovations over XGBoost:
        1. Histogram-Based Splitting:
            - Bins continuous features into discrete buckets (e.g., 255 bins)
            - Reduces split-finding from O(n * features) to O(bins * features)
            - Memory-efficient: stores bin indices (uint8) instead of float64

        2. Leaf-Wise (Best-First) Tree Growth:
            - Instead of growing level-by-level, LightGBM picks the leaf with
              the largest gain reduction and splits that leaf
            - Achieves lower loss with fewer splits than level-wise growth
            - Risk: can overfit on small data if num_leaves is too large

        3. Gradient-Based One-Side Sampling (GOSS):
            - Keeps all instances with large gradients (top a%)
            - Randomly samples from instances with small gradients (b%)
            - Amplifies sampled small-gradient instances by (1-a)/b
            - Preserves gradient distribution while reducing data size

        4. Exclusive Feature Bundling (EFB):
            - Bundles mutually exclusive features into a single feature
            - Reduces feature dimensionality without losing information
            - Particularly effective for sparse, high-dimensional data

    Additive Ensemble Model:
        F(x) = F_0 + sum_{m=1}^{M} eta * f_m(x)
        where eta is the learning rate (shrinkage), f_m is tree m.

    Split Gain (same as XGBoost):
        Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                      - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma

    Key Hyperparameters:
        - num_leaves: Maximum number of leaves per tree (controls complexity)
        - max_depth: Maximum depth (-1 for no limit, used with leaf-wise)
        - learning_rate: Shrinkage factor to prevent overfitting
        - n_estimators: Number of boosting rounds
        - min_child_samples: Minimum data in a leaf (regularization)
        - subsample (bagging_fraction): Row subsampling ratio
        - colsample_bytree (feature_fraction): Column subsampling ratio
        - reg_alpha: L1 regularization on leaf weights
        - reg_lambda: L2 regularization on leaf weights

Business Use Cases:
    - Click-through rate (CTR) prediction in online advertising
    - Large-scale ranking systems (search, recommendation)
    - Real-time fraud detection with high-throughput requirements
    - IoT sensor anomaly detection at scale
    - Financial risk scoring with massive feature sets

Advantages:
    - Faster training than XGBoost (histogram + GOSS + EFB)
    - Lower memory consumption due to histogram representation
    - Native categorical feature support (no one-hot encoding needed)
    - Excellent performance on large, high-dimensional datasets
    - Built-in early stopping and feature importance

Disadvantages:
    - Leaf-wise growth can overfit on small datasets
    - num_leaves is harder to tune than max_depth
    - Less stable than XGBoost on very noisy data
    - Sensitive to num_leaves / learning_rate interaction
    - Categorical feature handling differs from CatBoost's approach
"""

# ============================================================================
# IMPORTS - Each import serves a specific purpose in the pipeline
# ============================================================================

import logging  # Structured logging for tracking experiment progress
import warnings  # Suppress convergence and deprecation warnings for clean output
from typing import Any, Dict, Tuple  # Type hints for function signatures

import numpy as np  # Numerical arrays for data manipulation and metric computation
import optuna  # Bayesian hyperparameter optimization framework
import lightgbm as lgb  # LightGBM library providing the gradient boosting implementation
from sklearn.datasets import make_classification  # Synthetic dataset generator for testing
from sklearn.metrics import (  # Evaluation metrics for classification performance
    accuracy_score,  # Fraction of correct predictions
    classification_report,  # Per-class precision, recall, F1 summary
    confusion_matrix,  # True/false positive/negative counts
    f1_score,  # Harmonic mean of precision and recall
    precision_score,  # Fraction of positive predictions that are correct
    recall_score,  # Fraction of actual positives that are detected
    roc_auc_score,  # Area under the ROC curve for ranking quality
)
from sklearn.model_selection import train_test_split  # Stratified data splitting
from sklearn.preprocessing import StandardScaler  # Z-score normalization for features

# ============================================================================
# LOGGING CONFIGURATION - Set up structured logging for experiment tracking
# ============================================================================

logging.basicConfig(  # Configure root logger with timestamp, level, and message format
    level=logging.INFO,  # Show INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + severity + content
)
logger = logging.getLogger(__name__)  # Create module-specific logger to avoid conflicts
warnings.filterwarnings("ignore")  # Suppress sklearn/lgb deprecation warnings for clean output


# ============================================================================
# DATA GENERATION - Create synthetic classification data with train/val/test splits
# ============================================================================

def generate_data(
    n_samples: int = 1000,  # Total number of data points to generate
    n_features: int = 20,  # Dimensionality of the feature space
    n_classes: int = 2,  # Number of target classes (binary or multiclass)
    random_state: int = 42,  # Seed for reproducibility across experiments
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data split into train/validation/test sets.

    Returns six arrays: X_train, X_val, X_test, y_train, y_val, y_test.
    Uses stratified splitting to maintain class balance across all sets.
    """
    # Create synthetic data with informative and redundant features
    # n_informative: features that actually carry signal for classification
    # n_redundant: linear combinations of informative features (adds correlation)
    X, y = make_classification(  # Generate feature matrix X and label vector y
        n_samples=n_samples,  # Total number of samples to generate
        n_features=n_features,  # Total number of features per sample
        n_informative=n_features // 2,  # Half the features carry real signal
        n_redundant=n_features // 4,  # Quarter are linear combos (correlated noise)
        n_classes=n_classes,  # Number of distinct target classes
        random_state=random_state,  # Fix randomness for reproducible datasets
    )

    # Split into 60% train, 20% validation, 20% test with stratification
    # Stratification ensures each split has the same class proportions as the full data
    X_train, X_temp, y_train, y_temp = train_test_split(  # First split: 60/40
        X, y,  # Full feature matrix and labels
        test_size=0.4,  # Hold out 40% for validation + test
        random_state=random_state,  # Same seed ensures reproducible splits
        stratify=y,  # Maintain class balance in both partitions
    )
    X_val, X_test, y_val, y_test = train_test_split(  # Second split: 50/50 of the 40%
        X_temp, y_temp,  # The held-out 40% from the first split
        test_size=0.5,  # Split evenly into validation and test (20% each)
        random_state=random_state,  # Reproducible split
        stratify=y_temp,  # Maintain class balance in val and test
    )

    # Standardize features to zero mean and unit variance
    # LightGBM is tree-based and technically invariant to scaling, but scaling
    # helps when comparing with other algorithms and stabilizes numeric precision
    scaler = StandardScaler()  # Create a scaler that learns mean/std from training data
    X_train = scaler.fit_transform(X_train)  # Fit on train data and transform it
    X_val = scaler.transform(X_val)  # Transform validation using train statistics only
    X_test = scaler.transform(X_test)  # Transform test using train statistics only

    # Log dataset dimensions for experiment tracking and verification
    logger.info(  # Report sizes so we can verify data generation worked correctly
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test  # Return all six arrays


# ============================================================================
# TRAINING - Fit a LightGBM classifier with configurable hyperparameters
# ============================================================================

def train(
    X_train: np.ndarray,  # Training feature matrix (n_samples x n_features)
    y_train: np.ndarray,  # Training labels (n_samples,)
    **hyperparams: Any,  # Override any default hyperparameters via kwargs
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier using the scikit-learn compatible API.

    Default hyperparameters provide a solid baseline. Pass keyword arguments
    to override any parameter for experimentation or tuning.
    """
    # Define sensible default hyperparameters for LightGBM
    # These values represent a balanced starting point for most datasets
    defaults = dict(
        n_estimators=200,  # Number of boosting rounds (trees to build sequentially)
        num_leaves=31,  # Max leaves per tree; controls model complexity (2^5 - 1)
        max_depth=-1,  # No depth limit; leaf-wise growth controlled by num_leaves
        learning_rate=0.1,  # Shrinkage factor: smaller = more regularization, slower training
        subsample=0.8,  # Use 80% of rows per tree to reduce overfitting (bagging)
        colsample_bytree=0.8,  # Use 80% of features per tree to reduce correlation
        reg_alpha=0.0,  # L1 regularization: set to 0 by default (no sparsity penalty)
        reg_lambda=1.0,  # L2 regularization: mild penalty on large leaf weights
        min_child_samples=20,  # Minimum samples per leaf to prevent overfitting on noise
        objective="binary",  # Binary cross-entropy loss for two-class problems
        metric="binary_logloss",  # Evaluation metric for early stopping decisions
        random_state=42,  # Seed for reproducibility in row/column sampling
        n_jobs=-1,  # Use all CPU cores for parallel histogram construction
        verbose=-1,  # Suppress LightGBM's internal logging output
    )

    # Detect multiclass and switch objective accordingly
    # LightGBM requires explicit objective specification for multiclass
    n_classes = len(np.unique(y_train))  # Count distinct class labels in training data
    if n_classes > 2:  # If more than 2 classes, switch to multiclass objective
        defaults["objective"] = "multiclass"  # Softmax-based multiclass loss
        defaults["num_class"] = n_classes  # Tell LightGBM how many classes to model
        defaults["metric"] = "multi_logloss"  # Multiclass log-loss evaluation metric

    # Override defaults with any user-provided hyperparameters
    # This allows the tuning functions to pass specific configurations
    defaults.update(hyperparams)  # Merge user params into defaults (user wins on conflict)

    # Create and fit the LightGBM classifier
    model = lgb.LGBMClassifier(**defaults)  # Instantiate with merged hyperparameters
    model.fit(X_train, y_train)  # Train the model on training data

    # Log the key hyperparameters used for this training run
    logger.info(  # Record configuration so we can reproduce this experiment
        "LightGBM trained: n_estimators=%d, num_leaves=%d, lr=%.3f",
        defaults["n_estimators"], defaults["num_leaves"], defaults["learning_rate"],
    )
    return model  # Return the trained LightGBM classifier


# ============================================================================
# EVALUATION - Compute classification metrics for any dataset split
# ============================================================================

def _evaluate(
    model: lgb.LGBMClassifier,  # Trained LightGBM model to evaluate
    X: np.ndarray,  # Feature matrix for the split being evaluated
    y: np.ndarray,  # True labels for the split being evaluated
) -> Dict[str, float]:
    """Compute comprehensive classification metrics for a trained model.

    Returns a dictionary with accuracy, precision, recall, F1, and AUC-ROC.
    Handles both binary and multiclass classification automatically.
    """
    y_pred = model.predict(X)  # Get hard class predictions (argmax of probabilities)
    y_proba = model.predict_proba(X)  # Get probability estimates for each class

    # Compute AUC-ROC: binary uses positive class proba, multiclass uses OVR
    n_classes = len(np.unique(y))  # Determine number of classes from true labels
    if n_classes == 2:  # Binary classification: use probability of positive class
        auc = roc_auc_score(y, y_proba[:, 1])  # AUC from P(y=1) scores
    else:  # Multiclass: use One-vs-Rest strategy with weighted averaging
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    # Return all metrics in a dictionary for easy comparison across experiments
    return {
        "accuracy": accuracy_score(y, y_pred),  # Overall fraction correct
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),  # Weighted precision
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),  # Weighted recall
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),  # Weighted F1 score
        "auc_roc": auc,  # Area under ROC curve
    }


def validate(
    model: lgb.LGBMClassifier,  # Trained model to validate
    X_val: np.ndarray,  # Validation feature matrix
    y_val: np.ndarray,  # Validation true labels
) -> Dict[str, float]:
    """Evaluate model on validation set and log the results."""
    metrics = _evaluate(model, X_val, y_val)  # Compute all metrics on validation set
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})  # Log formatted metrics
    return metrics  # Return metric dictionary for downstream use


def test(
    model: lgb.LGBMClassifier,  # Trained model to test
    X_test: np.ndarray,  # Test feature matrix
    y_test: np.ndarray,  # Test true labels
) -> Dict[str, float]:
    """Evaluate model on held-out test set with detailed reporting."""
    metrics = _evaluate(model, X_test, y_test)  # Compute all metrics on test set
    y_pred = model.predict(X_test)  # Get predictions for confusion matrix and report

    # Log all test metrics for final evaluation
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))  # Show TP/FP/TN/FN
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))  # Per-class detail

    # Log feature importances to understand which features drive predictions
    # LightGBM computes importance as the number of times a feature is used in splits
    importances = model.feature_importances_  # Get feature importance array (split-based)
    indices = np.argsort(importances)[::-1]  # Sort features by importance (descending)
    logger.info("\nTop 10 Feature Importances (split count):")  # Header for importance list
    for rank, idx in enumerate(indices[:10], 1):  # Iterate over top 10 most important features
        logger.info("  %d. Feature %d: %.4f", rank, idx, importances[idx])  # Log rank and value

    return metrics  # Return metric dictionary for programmatic comparison


# ============================================================================
# HYPERPARAMETER OPTIMIZATION - Optuna Bayesian Search
# ============================================================================

def optuna_objective(
    trial: optuna.Trial,  # Optuna trial object that suggests hyperparameter values
    X_train: np.ndarray,  # Training features for fitting
    y_train: np.ndarray,  # Training labels for fitting
    X_val: np.ndarray,  # Validation features for evaluation
    y_val: np.ndarray,  # Validation labels for evaluation
) -> float:
    """Optuna objective function for LightGBM hyperparameter optimization.

    Uses Tree-structured Parzen Estimator (TPE) to suggest hyperparameters.
    Returns the F1 score on the validation set as the optimization target.
    """
    # Define the search space for each hyperparameter
    # Each suggest_* call registers the parameter with Optuna's sampler
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),  # Boosting rounds
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),  # Leaf count controls tree complexity
        "max_depth": trial.suggest_int("max_depth", 3, 15),  # Depth limit for additional regularization
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  # Log-uniform for rates
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # Row sampling fraction
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Column sampling fraction
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),  # L1 regularization strength
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),  # L2 regularization strength
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),  # Min samples per leaf
    }

    model = train(X_train, y_train, **params)  # Train with suggested hyperparameters
    metrics = validate(model, X_val, y_val)  # Evaluate on validation set
    return metrics["f1"]  # Return F1 as the objective to maximize


# ============================================================================
# HYPERPARAMETER OPTIMIZATION - Ray Tune Distributed Search
# ============================================================================

def ray_tune_search(
    X_train: np.ndarray,  # Training features for fitting
    y_train: np.ndarray,  # Training labels for fitting
    X_val: np.ndarray,  # Validation features for evaluation
    y_val: np.ndarray,  # Validation labels for evaluation
    num_samples: int = 20,  # Number of hyperparameter configurations to try
) -> Dict[str, Any]:
    """Run distributed hyperparameter search using Ray Tune.

    Ray Tune enables parallel evaluation of multiple configurations across
    CPUs/GPUs, significantly faster than sequential Optuna for large searches.
    """
    import ray  # Lazy import: Ray is heavy and only needed when this function is called
    from ray import tune as ray_tune  # Ray's hyperparameter tuning module

    if not ray.is_initialized():  # Only initialize Ray once to avoid overhead
        ray.init(ignore_reinit_error=True, log_to_driver=False)  # Quiet initialization

    def _trainable(config: Dict[str, Any]) -> None:
        """Inner training function that Ray Tune calls with each config."""
        model = train(X_train, y_train, **config)  # Train with this trial's config
        metrics = validate(model, X_val, y_val)  # Evaluate on validation set
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])  # Report back to Ray

    # Define the search space using Ray Tune's distribution primitives
    search_space = {
        "n_estimators": ray_tune.choice([100, 200, 300, 500]),  # Discrete choices for tree count
        "num_leaves": ray_tune.randint(15, 255),  # Uniform integer for leaf count
        "max_depth": ray_tune.randint(3, 15),  # Uniform integer for tree depth
        "learning_rate": ray_tune.loguniform(0.01, 0.3),  # Log-uniform for learning rate
        "subsample": ray_tune.uniform(0.5, 1.0),  # Uniform for row sampling
        "colsample_bytree": ray_tune.uniform(0.5, 1.0),  # Uniform for column sampling
        "reg_alpha": ray_tune.loguniform(1e-8, 10.0),  # Log-uniform for L1 regularization
        "reg_lambda": ray_tune.loguniform(1e-8, 10.0),  # Log-uniform for L2 regularization
        "min_child_samples": ray_tune.randint(5, 100),  # Uniform integer for min leaf samples
    }

    # Create and run the tuner with the specified search space
    tuner = ray_tune.Tuner(  # Configure the distributed tuning job
        _trainable,  # The training function to evaluate each config
        param_space=search_space,  # The hyperparameter search space
        tune_config=ray_tune.TuneConfig(  # Tuning behavior settings
            num_samples=num_samples,  # How many random configurations to try
            metric="f1",  # Which metric to optimize
            mode="max",  # Maximize F1 (higher is better)
        ),
    )
    results = tuner.fit()  # Execute all trials (potentially in parallel)
    best = results.get_best_result(metric="f1", mode="max")  # Get the top-performing config
    logger.info("Ray Tune best config: %s", best.config)  # Log the winning hyperparameters
    ray.shutdown()  # Clean up Ray resources after tuning is complete
    return best.config  # Return the best hyperparameter configuration


# ============================================================================
# COMPARE PARAMETER SETS - Systematic comparison of key hyperparameter values
# ============================================================================

def compare_parameter_sets(
    X_train: np.ndarray,  # Training features for fitting each configuration
    y_train: np.ndarray,  # Training labels for fitting each configuration
    X_val: np.ndarray,  # Validation features for evaluating each configuration
    y_val: np.ndarray,  # Validation labels for evaluating each configuration
) -> Dict[str, Dict[str, float]]:
    """Compare predefined LightGBM hyperparameter configurations head-to-head.

    Tests three key hyperparameter axes:
        1. num_leaves=31 vs 127 (tree complexity)
        2. learning_rate=0.01 vs 0.1 (shrinkage aggressiveness)
        3. n_estimators=100 vs 500 (ensemble size)

    Each comparison isolates one parameter while holding others at defaults,
    allowing us to understand the marginal effect of each hyperparameter.
    """
    logger.info("=" * 70)  # Visual separator for the comparison section
    logger.info("PARAMETER SET COMPARISON")  # Section header
    logger.info("=" * 70)  # Visual separator

    # Define all parameter configurations to compare
    # Each key is a descriptive name, each value is a dict of hyperparameters
    configs = {
        # --- num_leaves comparison ---
        # num_leaves=31: Default LightGBM value, moderate tree complexity
        # Fewer leaves mean simpler trees that generalize better on small data
        "num_leaves=31 (default)": {"num_leaves": 31, "n_estimators": 200, "learning_rate": 0.1},
        # num_leaves=127: High complexity trees with many splits
        # More leaves capture finer patterns but risk overfitting on noise
        "num_leaves=127 (complex)": {"num_leaves": 127, "n_estimators": 200, "learning_rate": 0.1},

        # --- learning_rate comparison ---
        # learning_rate=0.01: Very conservative shrinkage, needs many trees
        # Small learning rate means each tree contributes less, better generalization
        "learning_rate=0.01 (slow)": {"num_leaves": 31, "n_estimators": 200, "learning_rate": 0.01},
        # learning_rate=0.1: Standard shrinkage, balanced speed and accuracy
        # Larger learning rate means faster convergence but higher variance
        "learning_rate=0.1 (standard)": {"num_leaves": 31, "n_estimators": 200, "learning_rate": 0.1},

        # --- n_estimators comparison ---
        # n_estimators=100: Fewer boosting rounds, faster training
        # May underfit if learning rate is small or data is complex
        "n_estimators=100 (fewer)": {"num_leaves": 31, "n_estimators": 100, "learning_rate": 0.1},
        # n_estimators=500: Many boosting rounds, thorough optimization
        # More rounds allow finer corrections but risk overfitting without early stopping
        "n_estimators=500 (many)": {"num_leaves": 31, "n_estimators": 500, "learning_rate": 0.1},
    }

    results = {}  # Dictionary to store metrics for each configuration
    for name, params in configs.items():  # Iterate over each named configuration
        logger.info("\n--- Config: %s ---", name)  # Log which configuration is being trained
        model = train(X_train, y_train, **params)  # Train with this specific configuration
        metrics = validate(model, X_val, y_val)  # Evaluate on validation set
        results[name] = metrics  # Store metrics for comparison

    # Print a summary table comparing all configurations
    logger.info("\n" + "=" * 70)  # Visual separator before summary
    logger.info("COMPARISON SUMMARY")  # Summary header
    logger.info("%-35s | Accuracy | F1     | AUC-ROC", "Configuration")  # Table header
    logger.info("-" * 70)  # Table separator line
    for name, metrics in results.items():  # Iterate over all results
        logger.info(  # Print each configuration's key metrics in aligned columns
            "%-35s | %.4f   | %.4f | %.4f",
            name, metrics["accuracy"], metrics["f1"], metrics["auc_roc"],
        )

    return results  # Return all results for programmatic access


# ============================================================================
# REAL-WORLD DEMO - Click-Through Rate (CTR) Prediction
# ============================================================================

def real_world_demo() -> None:
    """Simulate a real-world click-through rate (CTR) prediction task.

    Scenario: An online advertising platform needs to predict whether a user
    will click on an ad based on user demographics and ad placement features.

    Features:
        - user_age: Age of the user viewing the ad (continuous, 18-65)
        - device_type: Type of device (categorical: mobile=0, desktop=1, tablet=2)
        - ad_position: Position of the ad on the page (categorical: top=0, middle=1, bottom=2)
        - time_of_day: Hour of the day the ad is shown (continuous, 0-23)

    Target: clicked (1) or not clicked (0)

    Why LightGBM for CTR?
        - Extremely fast training on billions of ad impressions
        - Handles categorical features natively without one-hot encoding
        - GOSS reduces training data while preserving gradient information
        - Histogram-based splits are ideal for binned features like time buckets
    """
    logger.info("=" * 70)  # Visual separator for the demo section
    logger.info("REAL-WORLD DEMO: Click-Through Rate (CTR) Prediction")  # Demo header
    logger.info("=" * 70)  # Visual separator

    # Set random seed for reproducible synthetic data generation
    rng = np.random.RandomState(42)  # Fixed seed ensures same data every run
    n_samples = 2000  # Number of synthetic ad impressions to generate

    # Generate synthetic features that mimic real CTR prediction data
    user_age = rng.randint(18, 66, size=n_samples).astype(float)  # User age: 18 to 65 years
    device_type = rng.choice([0, 1, 2], size=n_samples).astype(float)  # 0=mobile, 1=desktop, 2=tablet
    ad_position = rng.choice([0, 1, 2], size=n_samples).astype(float)  # 0=top, 1=middle, 2=bottom
    time_of_day = rng.randint(0, 24, size=n_samples).astype(float)  # Hour of day: 0 to 23

    # Stack all features into a single feature matrix
    # Each row is one ad impression, each column is one feature
    X = np.column_stack([user_age, device_type, ad_position, time_of_day])  # Shape: (2000, 4)

    # Generate synthetic click labels with realistic patterns
    # Real CTR is typically 1-5%, so we create a low click-rate scenario
    # Click probability depends on: younger users, mobile devices, top position, evening hours
    click_prob = (  # Combine feature effects into a click probability
        0.02  # Base click rate of 2% (most ads are not clicked)
        + 0.01 * (user_age < 35).astype(float)  # Younger users click more (+1%)
        + 0.02 * (device_type == 0).astype(float)  # Mobile users click more (+2%)
        + 0.03 * (ad_position == 0).astype(float)  # Top position gets more clicks (+3%)
        + 0.01 * ((time_of_day >= 18) & (time_of_day <= 22)).astype(float)  # Evening boost (+1%)
    )
    y = (rng.random(n_samples) < click_prob).astype(int)  # Sample clicks from probabilities

    # Log class distribution to verify realistic imbalance
    click_rate = y.mean()  # Compute the overall click-through rate
    logger.info("Click-through rate: %.2f%% (%d clicks / %d impressions)",
                click_rate * 100, y.sum(), n_samples)  # Report CTR statistics

    # Split data with stratification to maintain CTR ratio in each split
    X_train, X_temp, y_train, y_temp = train_test_split(  # 60% train, 40% temp
        X, y, test_size=0.4, random_state=42, stratify=y,  # Stratify on clicks
    )
    X_val, X_test, y_val, y_test = train_test_split(  # Split temp into 50/50
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp,  # Stratify again
    )

    # Train LightGBM with settings tuned for imbalanced CTR data
    # is_unbalance=True tells LightGBM to weight the minority class more heavily
    model = train(  # Train the classifier
        X_train, y_train,  # Use training data
        n_estimators=300,  # More trees to capture subtle click patterns
        num_leaves=31,  # Moderate complexity to prevent overfitting on sparse clicks
        learning_rate=0.05,  # Conservative learning rate for stable convergence
        is_unbalance=True,  # Handle class imbalance (few clicks vs many non-clicks)
    )

    # Evaluate the CTR model
    logger.info("\n--- CTR Model Validation ---")  # Section header
    validate(model, X_val, y_val)  # Check performance on validation set
    logger.info("\n--- CTR Model Test ---")  # Section header
    test(model, X_test, y_test)  # Final evaluation on held-out test set

    # Log feature names for interpretability in the CTR context
    feature_names = ["user_age", "device_type", "ad_position", "time_of_day"]  # Meaningful names
    importances = model.feature_importances_  # Get importance scores from the trained model
    logger.info("\nCTR Feature Importances:")  # Header for feature importance listing
    for name, imp in sorted(  # Sort features by importance descending
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    ):
        logger.info("  %s: %.4f", name, imp)  # Log each feature's importance score


# ============================================================================
# MAIN - Full pipeline execution: baseline, tuning, comparison, and demo
# ============================================================================

def main() -> None:
    """Run the complete LightGBM classification pipeline.

    Steps:
        1. Generate synthetic classification data
        2. Train baseline model with default hyperparameters
        3. Run Optuna hyperparameter optimization
        4. Train best model and evaluate on test set
        5. Compare predefined parameter configurations
        6. Run real-world CTR prediction demo
    """
    logger.info("=" * 70)  # Visual separator for the main pipeline
    logger.info("LightGBM Classifier - Scikit-Learn API Implementation")  # Pipeline header
    logger.info("=" * 70)  # Visual separator

    # Step 1: Generate synthetic classification data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()  # Create train/val/test splits

    # Step 2: Train baseline model with default hyperparameters
    logger.info("\n--- Baseline Training ---")  # Section header
    model = train(X_train, y_train)  # Train with all defaults
    validate(model, X_val, y_val)  # Check baseline performance

    # Step 3: Optuna hyperparameter optimization
    logger.info("\n--- Optuna Hyperparameter Optimization ---")  # Section header
    study = optuna.create_study(direction="maximize")  # Create study to maximize F1
    study.optimize(  # Run the optimization loop
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),  # Objective closure
        n_trials=30,  # Try 30 different hyperparameter configurations
        show_progress_bar=True,  # Display progress during optimization
    )
    logger.info("Optuna best params: %s", study.best_params)  # Log the winning parameters
    logger.info("Optuna best F1: %.4f", study.best_value)  # Log the best F1 achieved

    # Step 4: Retrain with best parameters and evaluate on test set
    best_model = train(X_train, y_train, **study.best_params)  # Train with optimal config
    logger.info("\n--- Final Test Evaluation ---")  # Section header
    test(best_model, X_test, y_test)  # Final test evaluation

    # Step 5: Compare predefined parameter configurations
    logger.info("\n--- Parameter Set Comparison ---")  # Section header
    compare_parameter_sets(X_train, y_train, X_val, y_val)  # Run systematic comparison

    # Step 6: Real-world CTR prediction demo
    real_world_demo()  # Simulate click-through rate prediction


if __name__ == "__main__":  # Only run main() when this file is executed directly
    main()  # Execute the full pipeline
