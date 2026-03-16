"""
CatBoost Classifier - CatBoost Library Implementation
========================================================

Theory & Mathematics:
    CatBoost (Categorical Boosting) is a gradient boosting algorithm developed
    by Yandex that introduces two key innovations: ordered boosting and native
    categorical feature handling.

    1. Ordered Boosting (Fights Prediction Shift):
        Standard gradient boosting suffers from "prediction shift" - the
        gradients used to train tree t are computed using the model F_{t-1}
        that was trained on the same data. This creates a subtle form of
        data leakage. CatBoost addresses this by:

        a. Randomly permuting the training data
        b. For each sample i, computing its residual using a model trained
           only on samples that appear before i in the permutation
        c. This ensures that the residual for sample i is computed from
           a model that never saw sample i during training

        Formally, for sample sigma(i) at position i in permutation sigma:
            r_i = y_i - F_{t-1}^{(i)}(x_i)
        where F_{t-1}^{(i)} is trained on {sigma(1), ..., sigma(i-1)} only.

    2. Ordered Target Encoding for Categorical Features:
        Instead of one-hot encoding (which creates high-dimensional sparse
        features) or naive target encoding (which leaks target information),
        CatBoost uses ordered target statistics:

        For sample i with category c at position sigma(i) in the permutation:
            TS(x_i) = (sum_{j < i, x_j = c} y_j + prior) / (count_{j < i, x_j = c} + 1)

        This is a leave-one-out encoding that respects the ordering, preventing
        target leakage while preserving categorical information.

    3. Oblivious Decision Trees (Symmetric Trees):
        CatBoost uses oblivious trees where the same splitting condition is
        used at every node of a given depth level. This means:
        - All nodes at depth d use the same (feature, threshold) pair
        - The tree has exactly 2^depth leaf nodes
        - Faster inference (can be evaluated as a lookup table)
        - Acts as an implicit regularizer

    4. Gradient Boosting Core:
        L(y, F) = -[y * log(sigma(F)) + (1-y) * log(1 - sigma(F))]
        g_i = sigma(F(x_i)) - y_i  (gradient)
        h_i = sigma(F(x_i)) * (1 - sigma(F(x_i)))  (hessian)
        Update: F_m(x) = F_{m-1}(x) + eta * h_m(x)

Business Use Cases:
    - Hotel booking cancellation prediction
    - E-commerce product recommendation
    - Insurance claim prediction with many categorical features
    - Customer segmentation with mixed data types
    - Fraud detection with categorical transaction metadata

Advantages:
    - Native categorical feature handling (no need for manual encoding)
    - Ordered boosting reduces overfitting
    - Oblivious trees provide fast inference
    - Built-in GPU support
    - Handles missing values natively
    - Often achieves top performance with minimal tuning
    - Built-in cross-validation and overfitting detection

Disadvantages:
    - Can be slower to train than LightGBM
    - Large model files (symmetric trees store more parameters)
    - Less flexible tree structure than XGBoost/LightGBM
    - Memory-intensive for very large datasets
    - External dependency (catboost package)

Hyperparameters:
    - iterations: Number of boosting rounds
    - learning_rate: Shrinkage factor
    - depth: Depth of oblivious trees (same split at each level)
    - l2_leaf_reg: L2 regularization on leaf values
    - border_count: Number of splits per feature (like max_bins)
    - cat_features: Indices of categorical feature columns
    - random_seed: Reproducibility seed
"""

import logging  # Standard logging for progress tracking
import warnings  # Suppress non-critical warnings
from typing import Any, Dict, List, Optional, Tuple  # Type annotations

import numpy as np  # Numerical computing
import optuna  # Bayesian hyperparameter optimization
from sklearn.datasets import make_classification  # Synthetic data generation
from sklearn.metrics import (  # Classification evaluation metrics
    accuracy_score,  # Overall prediction correctness
    classification_report,  # Detailed per-class metrics
    confusion_matrix,  # TP/FP/TN/FN matrix
    f1_score,  # Harmonic mean of precision and recall
    precision_score,  # Positive predictive value
    recall_score,  # Sensitivity / true positive rate
    roc_auc_score,  # Area under the ROC curve
)
from sklearn.model_selection import train_test_split  # Stratified data splitting
from sklearn.preprocessing import LabelEncoder  # Encode categorical labels as integers

# Attempt to import CatBoost; set a flag if unavailable
try:
    from catboost import CatBoostClassifier, Pool  # CatBoost classifier and data pool
    CATBOOST_AVAILABLE = True  # Flag indicating catboost is installed
except ImportError:
    CATBOOST_AVAILABLE = False  # CatBoost not installed
    logger_temp = logging.getLogger(__name__)  # Temporary logger for warning
    logger_temp.warning(
        "CatBoost is not installed. Install via: pip install catboost. "
        "This module will use a sklearn GradientBoostingClassifier fallback."
    )

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fallback: sklearn GradientBoostingClassifier if catboost unavailable
# ---------------------------------------------------------------------------

if not CATBOOST_AVAILABLE:
    from sklearn.ensemble import GradientBoostingClassifier as _FallbackGBC  # Fallback model
    from sklearn.preprocessing import OrdinalEncoder  # For encoding categoricals


# ---------------------------------------------------------------------------
# CatBoost Wrapper (handles both installed and fallback cases)
# ---------------------------------------------------------------------------

class CatBoostWrapper:
    """
    Wrapper around CatBoost that gracefully falls back to sklearn's
    GradientBoostingClassifier when catboost is not installed.
    """

    def __init__(
        self,
        iterations: int = 500,  # Number of boosting iterations
        learning_rate: float = 0.1,  # Step size shrinkage
        depth: int = 6,  # Oblivious tree depth
        l2_leaf_reg: float = 3.0,  # L2 regularization on leaf values
        border_count: int = 128,  # Number of split candidates per feature
        cat_features: Optional[List[int]] = None,  # Indices of categorical columns
        random_seed: int = 42,  # Reproducibility seed
        verbose: int = 0,  # CatBoost verbosity level
    ) -> None:
        """Initialize the CatBoost wrapper with hyperparameters."""
        self.iterations = iterations  # Boosting rounds
        self.learning_rate = learning_rate  # Shrinkage
        self.depth = depth  # Tree depth
        self.l2_leaf_reg = l2_leaf_reg  # Regularization
        self.border_count = border_count  # Bin count
        self.cat_features = cat_features if cat_features is not None else []  # Cat columns
        self.random_seed = random_seed  # Seed
        self.verbose = verbose  # Logging verbosity
        self.model = None  # Will hold the fitted model
        self._using_fallback = not CATBOOST_AVAILABLE  # Track which backend we use
        self._ordinal_encoder = None  # For fallback: encode categoricals

    def fit(
        self,
        X: np.ndarray,  # Training features (n_samples, n_features)
        y: np.ndarray,  # Training labels
    ) -> "CatBoostWrapper":
        """
        Fit the CatBoost model (or fallback) on training data.
        If catboost is installed, uses native categorical handling.
        Otherwise, ordinal-encodes categoricals and uses sklearn GBC.
        """
        if CATBOOST_AVAILABLE:  # Use native CatBoost
            # Create a CatBoost Pool with explicit categorical feature specification
            train_pool = Pool(
                data=X,  # Feature matrix
                label=y,  # Target labels
                cat_features=self.cat_features,  # Which columns are categorical
            )
            self.model = CatBoostClassifier(
                iterations=self.iterations,  # Number of trees
                learning_rate=self.learning_rate,  # Shrinkage rate
                depth=self.depth,  # Symmetric tree depth
                l2_leaf_reg=self.l2_leaf_reg,  # L2 regularization
                border_count=self.border_count,  # Number of splits per feature
                random_seed=self.random_seed,  # Reproducibility
                verbose=self.verbose,  # Logging level
                loss_function="Logloss",  # Binary cross-entropy loss
                eval_metric="AUC",  # Monitor AUC during training
                allow_writing_files=False,  # Don't write model files
            )
            self.model.fit(train_pool)  # Train on the data pool
            logger.info(
                "CatBoost trained: %d iterations, depth=%d, lr=%.3f",
                self.iterations, self.depth, self.learning_rate,
            )
        else:  # Use sklearn fallback
            # Ordinal-encode categorical features for sklearn compatibility
            X_encoded = X.copy()  # Don't modify the original data
            if len(self.cat_features) > 0:  # Has categorical features
                self._ordinal_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                X_encoded[:, self.cat_features] = self._ordinal_encoder.fit_transform(
                    X[:, self.cat_features],
                )
            X_encoded = X_encoded.astype(np.float64)  # Ensure float dtype

            self.model = _FallbackGBC(
                n_estimators=self.iterations,  # Number of trees
                learning_rate=self.learning_rate,  # Shrinkage
                max_depth=self.depth,  # Max depth (not symmetric like CatBoost)
                random_state=self.random_seed,  # Seed
            )
            self.model.fit(X_encoded, y)  # Train the fallback model
            logger.info(
                "Fallback GBC trained: %d estimators, depth=%d, lr=%.3f",
                self.iterations, self.depth, self.learning_rate,
            )

        return self  # Return self for chaining

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Prepare features for prediction (encode categoricals if using fallback)."""
        if self._using_fallback and self._ordinal_encoder is not None:
            X_encoded = X.copy()  # Don't modify original
            X_encoded[:, self.cat_features] = self._ordinal_encoder.transform(
                X[:, self.cat_features],
            )
            return X_encoded.astype(np.float64)  # Convert to float
        return X  # No transformation needed for native CatBoost

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(y=0), P(y=1)]."""
        X_prep = self._prepare_X(X)  # Prepare features
        if CATBOOST_AVAILABLE:  # Native CatBoost
            pool = Pool(data=X_prep, cat_features=self.cat_features)
            return self.model.predict_proba(pool)  # CatBoost returns (n, 2)
        else:  # Fallback
            return self.model.predict_proba(X_prep)  # sklearn returns (n, 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {0, 1}."""
        proba = self.predict_proba(X)  # Get probabilities
        return (proba[:, 1] >= 0.5).astype(int)  # Threshold at 0.5


# ---------------------------------------------------------------------------
# Data Generation (with categorical features)
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,  # Total samples
    random_state: int = 42,  # Seed
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Generate a dataset with a mix of continuous and categorical features,
    simulating a hotel booking cancellation scenario.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, cat_feature_indices).
    """
    rng = np.random.RandomState(random_state)  # Reproducible RNG

    # Continuous features
    lead_time = rng.exponential(100, n_samples)  # Days between booking and arrival
    adr = rng.normal(100, 40, n_samples).clip(20, 500)  # Average daily rate ($)
    stays_weekend = rng.poisson(1, n_samples)  # Weekend night stays
    stays_week = rng.poisson(3, n_samples)  # Weekday night stays
    previous_cancellations = rng.poisson(0.3, n_samples)  # Past cancellation count
    special_requests = rng.poisson(1.5, n_samples)  # Number of special requests

    # Categorical features (encoded as strings for CatBoost)
    country_codes = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "USA", "BRA", "CHN", "OTHER"]
    country = rng.choice(country_codes, n_samples, p=[0.3, 0.12, 0.1, 0.08, 0.08, 0.07, 0.07, 0.05, 0.05, 0.08])
    market_segment = rng.choice(
        ["Online_TA", "Offline_TA", "Direct", "Corporate", "Groups"],
        n_samples, p=[0.45, 0.15, 0.15, 0.15, 0.10],
    )
    deposit_type = rng.choice(["No_Deposit", "Non_Refund", "Refundable"], n_samples, p=[0.7, 0.2, 0.1])
    meal_type = rng.choice(["BB", "HB", "FB", "SC"], n_samples, p=[0.5, 0.25, 0.1, 0.15])

    # Stack features: continuous first, then categorical
    X = np.column_stack([
        lead_time, adr, stays_weekend, stays_week,  # Continuous features [0:4]
        previous_cancellations, special_requests,  # Continuous features [4:6]
        country, market_segment, deposit_type, meal_type,  # Categorical features [6:10]
    ])

    # Categorical feature column indices
    cat_features = [6, 7, 8, 9]  # Columns containing categorical data

    # Generate cancellation labels using a logistic model
    cancel_logit = (
        0.005 * lead_time  # Longer lead time -> higher cancellation
        - 0.005 * adr  # Higher price -> slightly lower cancellation
        + 0.5 * previous_cancellations  # History of cancellation -> higher risk
        - 0.3 * special_requests  # Special requests -> lower cancellation (invested)
        + 1.0 * (deposit_type == "Non_Refund").astype(float)  # Non-refundable -> higher cancel
        - 0.5 * (market_segment == "Corporate").astype(float)  # Corporate -> lower cancel
        + 0.3 * (country == "PRT").astype(float)  # Domestic bookings -> higher cancel
        + rng.normal(0, 0.5, n_samples)  # Noise
    )
    cancel_prob = 1.0 / (1.0 + np.exp(-cancel_logit))  # Sigmoid
    y = (rng.random(n_samples) < cancel_prob).astype(int)  # Binary label

    # Split (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, cat_features=%s",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], cat_features,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, cat_features


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,  # Training features
    y_train: np.ndarray,  # Training labels
    cat_features: Optional[List[int]] = None,  # Categorical column indices
    **hyperparams: Any,  # Additional hyperparameters
) -> CatBoostWrapper:
    """Train a CatBoost classifier with given hyperparameters."""
    defaults = dict(
        iterations=500,  # 500 boosting rounds
        learning_rate=0.1,  # Moderate learning rate
        depth=6,  # Balanced tree depth
        l2_leaf_reg=3.0,  # Default L2 regularization
        border_count=128,  # Default bin count
        cat_features=cat_features if cat_features is not None else [],
        random_seed=42,
        verbose=0,
    )
    defaults.update(hyperparams)  # Override with user values
    model = CatBoostWrapper(**defaults)  # Create the wrapper
    model.fit(X_train, y_train)  # Train the model
    return model


def _evaluate(model: CatBoostWrapper, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = model.predict(X)  # Hard predictions
    y_proba = model.predict_proba(X)  # Probability estimates
    auc = roc_auc_score(y, y_proba[:, 1])  # AUC using P(y=1)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: CatBoostWrapper, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Evaluate on validation data."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: CatBoostWrapper, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate on test data with full reporting."""
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
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
    cat_features: Optional[List[int]] = None,
) -> float:
    """Optuna objective: suggest params, train CatBoost, return val F1."""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 256]),
    }
    model = train(X_train, y_train, cat_features=cat_features, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_features: Optional[List[int]] = None,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search for CatBoost."""
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, cat_features=cat_features, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "iterations": ray_tune.choice([200, 500, 800]),
        "learning_rate": ray_tune.loguniform(0.01, 0.3),
        "depth": ray_tune.randint(3, 10),
        "l2_leaf_reg": ray_tune.loguniform(0.1, 30.0),
        "border_count": ray_tune.choice([64, 128, 256]),
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
# Compare Parameter Sets
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_features: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare 4 CatBoost configurations exploring depth and regularization.

    Configurations:
        1. shallow_lowreg (depth=4, l2=1, lr=0.1):
           Shallow trees with light regularization. Fast training.
           Good baseline for small datasets.

        2. deep_lowreg (depth=8, l2=1, lr=0.1):
           Deeper trees capture more complex interactions but risk overfitting.
           l2=1 provides minimal regularization.

        3. shallow_highreg (depth=4, l2=10, lr=0.03):
           Conservative approach: shallow trees + strong regularization +
           low learning rate. Best for noisy data or small samples.

        4. deep_highreg (depth=8, l2=10, lr=0.1):
           Deep trees with strong regularization. Tries to get complex
           patterns while preventing overfitting through L2 penalty.
    """
    configs = {
        "shallow_lowreg (d=4, l2=1, lr=0.1)": {
            # Reasoning: Shallow oblivious trees (4 levels = 16 leaves) with
            # minimal regularization. Fast and simple. Baseline configuration.
            "depth": 4,
            "l2_leaf_reg": 1.0,
            "learning_rate": 0.1,
            "iterations": 300,
        },
        "deep_lowreg (d=8, l2=1, lr=0.1)": {
            # Reasoning: Deep oblivious trees (8 levels = 256 leaves) capture
            # higher-order feature interactions. Risk of overfitting on small data.
            "depth": 8,
            "l2_leaf_reg": 1.0,
            "learning_rate": 0.1,
            "iterations": 300,
        },
        "shallow_highreg (d=4, l2=10, lr=0.03)": {
            # Reasoning: Conservative config. Shallow trees + strong L2 + low LR.
            # Requires more iterations but generalizes better on noisy data.
            "depth": 4,
            "l2_leaf_reg": 10.0,
            "learning_rate": 0.03,
            "iterations": 500,
        },
        "deep_highreg (d=8, l2=10, lr=0.1)": {
            # Reasoning: Deep trees for capacity, heavy L2 for regularization.
            # Tries to balance model complexity with overfitting prevention.
            "depth": 8,
            "l2_leaf_reg": 10.0,
            "learning_rate": 0.1,
            "iterations": 300,
        },
    }

    results = {}
    logger.info("=" * 70)
    logger.info("Comparing %d CatBoost configurations", len(configs))
    logger.info("=" * 70)

    for name, params in configs.items():
        logger.info("\n--- Config: %s ---", name)
        model = train(X_train, y_train, cat_features=cat_features, **params)
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("%-45s | Accuracy | F1     | AUC-ROC", "Configuration")
    logger.info("-" * 80)
    for name, metrics in results.items():
        logger.info(
            "%-45s | %.4f   | %.4f | %.4f",
            name, metrics["accuracy"], metrics["f1"], metrics["auc_roc"],
        )

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Hotel Booking Cancellation
# ---------------------------------------------------------------------------

def real_world_demo() -> Dict[str, float]:
    """
    Demonstrate CatBoost on a simulated hotel booking cancellation problem.

    Domain: Hospitality / hotel revenue management
    Goal: Predict whether a hotel booking will be cancelled.

    Features:
        - lead_time: Days between booking date and arrival date
        - country: Country of origin (categorical)
        - market_segment: How the booking was made (categorical)
        - deposit_type: Deposit type (categorical)
        - adr: Average daily rate in dollars
        - stays_weekend: Number of weekend night stays
        - stays_week: Number of weekday night stays
        - previous_cancellations: Number of prior cancellations
        - special_requests: Number of special requests
        - meal_type: Meal plan type (categorical)
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Hotel Booking Cancellation (CatBoost)")
    logger.info("=" * 70)

    # Generate the hotel booking dataset with categorical features
    X_train, X_val, X_test, y_train, y_val, y_test, cat_features = generate_data(
        n_samples=1000, random_state=42,
    )

    feature_names = [
        "lead_time", "adr", "stays_weekend", "stays_week",
        "previous_cancellations", "special_requests",
        "country", "market_segment", "deposit_type", "meal_type",
    ]
    logger.info("Features: %s", feature_names)
    logger.info("Categorical features: %s", [feature_names[i] for i in cat_features])
    logger.info("Cancellation rate: %.1f%%", 100 * np.mean(y_train))

    # Train CatBoost with native categorical handling
    model = train(
        X_train, y_train,
        cat_features=cat_features,
        iterations=500,
        depth=6,
        learning_rate=0.1,
    )
    validate(model, X_val, y_val)
    metrics = test(model, X_test, y_test)

    return metrics


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full CatBoost pipeline."""
    logger.info("=" * 70)
    logger.info("CatBoost Classifier - %s",
                "CatBoost Library" if CATBOOST_AVAILABLE else "sklearn Fallback")
    logger.info("=" * 70)

    # Generate data with categorical features
    X_train, X_val, X_test, y_train, y_val, y_test, cat_features = generate_data(
        n_samples=1000,
    )

    # Baseline
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, cat_features=cat_features, iterations=300, depth=6)
    validate(model, X_val, y_val)

    # Optuna HPO
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train, y_train, X_val, y_val, cat_features,
        ),
        n_trials=10,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    best_model = train(X_train, y_train, cat_features=cat_features, **study.best_params)

    # Test
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Compare
    logger.info("\n--- Parameter Set Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val, cat_features)

    # Demo
    logger.info("\n--- Real-World Demo: Hotel Booking Cancellation ---")
    real_world_demo()


if __name__ == "__main__":
    main()
