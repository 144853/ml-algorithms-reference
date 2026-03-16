"""
Isolation Forest - Scikit-Learn Implementation
===============================================

Theory & Mathematics:
    Isolation Forest is an unsupervised anomaly detection algorithm based on the
    principle that anomalies are "few and different," making them easier to isolate
    than normal points. The algorithm builds an ensemble of Isolation Trees (iTrees)
    by recursively partitioning data through random feature selection and random
    split values.

    Key insight: anomalies require fewer random partitions to be isolated, resulting
    in shorter average path lengths across the ensemble.

    Path Length & Anomaly Score:
        For a data point x, the anomaly score is defined as:

            s(x, n) = 2^(-E(h(x)) / c(n))

        where:
            - E(h(x)) is the average path length of x across all iTrees
            - c(n) is the average path length of unsuccessful search in a BST:
              c(n) = 2 * H(n-1) - 2*(n-1)/n
            - H(i) = ln(i) + 0.5772156649 (Euler-Mascheroni constant)

        Score interpretation:
            - s(x, n) close to 1   => anomaly
            - s(x, n) close to 0.5 => normal
            - s(x, n) close to 0   => very normal (dense region)

    Tree Construction:
        Each iTree is built by:
        1. Randomly selecting a feature q
        2. Randomly selecting a split value p between [min(q), max(q)]
        3. Recursively partitioning until:
           - The node has only one sample (isolated)
           - Maximum tree height is reached (ceil(log2(sub_sample_size)))
           - All samples at a node have identical values

    Sub-sampling:
        A key design choice is sub-sampling without replacement (default 256).
        This makes the algorithm:
        - Linear time complexity O(t * psi * log(psi)) where t = n_trees, psi = sub_sample_size
        - Memory efficient
        - Effective even on high-dimensional data

Business Use Cases:
    - Fraud detection in financial transactions (credit card, insurance claims)
    - Network intrusion detection (identifying malicious traffic patterns)
    - Manufacturing defect detection (sensor readings outside normal ranges)
    - Medical anomaly detection (unusual patient vitals or lab results)
    - IoT sensor monitoring (detecting equipment failures)
    - Log analysis (identifying unusual system behavior)

Advantages:
    - Linear time complexity with low memory requirement
    - No need to define a distance metric or density estimation
    - Handles high-dimensional data effectively
    - Works well with large datasets via sub-sampling
    - Robust to irrelevant features
    - No assumptions about data distribution

Disadvantages:
    - Axis-parallel splits may miss anomalies detectable only by combinations of features
    - Performance degrades with many irrelevant features (masking effect)
    - Contamination parameter must be set or estimated
    - Not ideal for local anomaly detection (vs. LOF)
    - Cannot capture temporal dependencies in sequential data

Key Hyperparameters:
    - n_estimators: Number of isolation trees (default=100)
    - max_samples: Sub-sampling size (default=256 or 'auto')
    - contamination: Expected proportion of anomalies (default='auto')
    - max_features: Number of features to draw per tree (default=1.0)
    - bootstrap: Whether to use bootstrap sampling (default=False)
    - random_state: Reproducibility seed

References:
    - Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008. Isolation forest.
      In 2008 eighth ieee international conference on data mining (pp. 413-422).
    - Liu, F.T., Ting, K.M. and Zhou, Z.H., 2012. Isolation-based anomaly
      detection. ACM Transactions on Knowledge Discovery from Data (TKDD), 6(1), 3.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

# warnings lets us suppress convergence and deprecation warnings from
# sklearn so that log output focuses on our algorithmic messages.
import warnings

# Type hints for self-documenting function signatures.
from typing import Any, Dict, Optional, Tuple

# NumPy provides the numerical foundation for array operations,
# random number generation, and metric computation.
import numpy as np

# Optuna performs Bayesian hyperparameter optimisation using TPE,
# which is more sample-efficient than grid or random search.
import optuna

# IsolationForest is sklearn's production-grade implementation that
# uses optimised C/Cython internals for tree construction and scoring.
# WHY: sklearn's implementation is the industry standard -- it handles
# sub-sampling, parallel tree building, and anomaly scoring efficiently.
from sklearn.ensemble import IsolationForest

# Anomaly detection metrics:
# - average_precision_score: area under the precision-recall curve
# - classification_report: text summary of precision/recall/f1
# - confusion_matrix: TP/FP/TN/FN counts
# - f1_score: harmonic mean of precision and recall
# - precision_recall_fscore_support: precision, recall, f1 in one call
# - roc_auc_score: area under the ROC curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# train_test_split creates reproducible, stratified data partitions.
from sklearn.model_selection import train_test_split

# StandardScaler z-normalises features to mean=0, std=1.
# WHY: while Isolation Forest is not distance-based, standardisation
# ensures that random splits in each feature have comparable granularity.
from sklearn.preprocessing import StandardScaler

# Suppress all warnings to keep output clean.
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 2000,
    n_features: int = 10,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic anomaly-detection data.

    Normal samples are drawn from a multivariate Gaussian distribution centered
    at the origin. Anomalies are drawn from a uniform distribution spanning a
    wider range so they appear in low-density regions.

    The data is split 60/20/20 into train/validation/test sets. Training data
    contains *mostly* normal samples (semi-supervised setup).

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Feature matrices for each split.
    y_train, y_val, y_test : np.ndarray
        Labels (0 = normal, 1 = anomaly).
    """
    # Create a reproducible random number generator.
    # WHY: RandomState ensures identical data across runs for fair comparison.
    rng = np.random.RandomState(random_state)

    # Calculate the number of normal and anomalous samples.
    # WHY: contamination controls the anomaly ratio; 5% is typical for
    # fraud detection where anomalies are rare.
    n_normal = int(n_samples * (1 - contamination))
    n_anomaly = n_samples - n_normal

    # Normal data: multivariate Gaussian centred at origin with unit variance.
    # WHY: Gaussian data forms a dense cluster that the Isolation Forest
    # will learn as "normal" -- points near the centre have long path lengths.
    X_normal = rng.randn(n_normal, n_features)
    y_normal = np.zeros(n_normal, dtype=int)

    # Anomalies: uniform over [-6, 6]^d, a much wider range than the Gaussian.
    # WHY: uniform points in a wide range land in low-density regions,
    # making them easy to isolate (short path lengths in the trees).
    X_anomaly = rng.uniform(low=-6, high=6, size=(n_anomaly, n_features))
    y_anomaly = np.ones(n_anomaly, dtype=int)

    # Combine normal and anomaly data into a single dataset.
    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])

    # Shuffle to remove ordering bias.
    # WHY: without shuffling, all anomalies would be at the end, which
    # could introduce subtle biases in train/test splits.
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split: 60% train, 20% val, 20% test with stratification.
    # WHY: stratify=y ensures each split has the same anomaly ratio,
    # which is critical for consistent evaluation across splits.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # Standardize using training statistics only.
    # WHY: fitting the scaler on training data prevents data leakage from
    # validation/test sets, which would give an optimistic performance estimate.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Data generated: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Contamination: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: np.ndarray, **hyperparams) -> IsolationForest:
    """
    Train an Isolation Forest model (unsupervised).

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (mostly normal samples).
    **hyperparams
        Keyword arguments forwarded to ``sklearn.ensemble.IsolationForest``.

    Returns
    -------
    IsolationForest
        Fitted model.
    """
    # Define sensible defaults for all hyperparameters.
    # WHY: these defaults are the sklearn recommended values; callers
    # (Optuna, Ray Tune) can override any of them.
    defaults = dict(
        n_estimators=100,     # 100 trees provides a good bias-variance trade-off
        max_samples="auto",   # 'auto' = min(256, n_samples) -- the original paper default
        contamination=0.05,   # expected 5% anomaly rate
        max_features=1.0,     # use all features for each tree
        bootstrap=False,      # sampling without replacement as per the original paper
        random_state=42,      # reproducibility
        n_jobs=-1,            # use all CPU cores for parallel tree construction
    )
    defaults.update(hyperparams)

    # Create and fit the IsolationForest model.
    # WHY: fit() builds the ensemble of isolation trees on the training data;
    # sklearn handles sub-sampling, tree construction, and path-length caching.
    model = IsolationForest(**defaults)
    model.fit(X_train)
    return model


def _predict_labels(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Convert sklearn predictions (-1=anomaly, 1=normal) to (1=anomaly, 0=normal)."""
    # sklearn uses -1 for anomalies and +1 for normal points.
    # WHY: we convert to the standard binary classification convention
    # (0=normal, 1=anomaly) so that sklearn metrics work correctly.
    preds = model.predict(X)
    return (preds == -1).astype(int)


def _anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores (higher = more anomalous). Negate decision_function."""
    # sklearn's decision_function returns higher values for normal points.
    # WHY: negating makes the score intuitive -- higher = more anomalous,
    # which aligns with roc_auc_score and average_precision_score conventions.
    return -model.decision_function(X)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive anomaly detection metrics."""
    # Compute precision, recall, and F1 for the anomaly class (binary).
    # WHY: F1 is the primary metric because it balances precision (avoiding
    # false alarms) and recall (catching actual anomalies).
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # ROC AUC: probability that a random anomaly scores higher than a random normal.
    # WHY: AUC is threshold-independent and gives a holistic view of model
    # discrimination ability across all possible thresholds.
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0

    # Average precision: area under the precision-recall curve.
    # WHY: more informative than AUC for imbalanced datasets because it
    # focuses on the minority (anomaly) class.
    avg_precision = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0

    # Confusion matrix: TP, FP, TN, FN counts for detailed analysis.
    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_precision),
        "confusion_matrix": cm,
    }


def validate(
    model: IsolationForest,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate the model on the validation set.

    Returns
    -------
    dict
        Metrics: precision, recall, f1, roc_auc, average_precision, confusion_matrix.
    """
    # Generate binary predictions and continuous anomaly scores.
    # WHY: we need both for threshold-dependent (F1) and threshold-independent
    # (AUC) evaluation.
    y_pred = _predict_labels(model, X_val)
    scores = _anomaly_scores(model, X_val)
    metrics = _compute_metrics(y_val, y_pred, scores)
    return metrics


def test(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate the model on the held-out test set.

    Returns
    -------
    dict
        Metrics: precision, recall, f1, roc_auc, average_precision, confusion_matrix.
    """
    y_pred = _predict_labels(model, X_test)
    scores = _anomaly_scores(model, X_test)
    metrics = _compute_metrics(y_test, y_pred, scores)
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Optuna objective function maximising validation F1-score.
    """
    # Number of isolation trees in the ensemble.
    # WHY: more trees reduce variance but increase training time linearly.
    n_estimators = trial.suggest_int("n_estimators", 50, 500)

    # Fraction of training data to sub-sample for each tree.
    # WHY: sub-sampling is key to Isolation Forest's effectiveness;
    # smaller sub-samples make anomalies easier to isolate.
    max_samples_frac = trial.suggest_float("max_samples_frac", 0.1, 1.0)
    max_samples = max(2, int(max_samples_frac * len(X_train)))

    # Expected contamination rate (controls the decision threshold).
    # WHY: if contamination is set too low, the model misses anomalies;
    # too high and normal points are falsely flagged.
    contamination = trial.suggest_float("contamination", 0.01, 0.15)

    # Fraction of features to use per tree.
    # WHY: using fewer features increases diversity among trees,
    # similar to the random subspace method in Random Forest.
    max_features = trial.suggest_float("max_features", 0.3, 1.0)

    # Whether to use bootstrap (with replacement) or sub-sampling (without).
    # WHY: the original paper uses without replacement; bootstrap can
    # increase diversity but changes the theoretical properties.
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    model = train(
        X_train,
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
    )

    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Hyperparameter search using Ray Tune with Optuna integration.

    Returns
    -------
    dict
        Best hyperparameters and validation metrics.
    """
    # Import Ray components inside the function to avoid import errors
    # when Ray is not installed.
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    # Initialise Ray for distributed trial execution.
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    def trainable(config: Dict[str, Any]) -> None:
        max_samples = max(2, int(config["max_samples_frac"] * len(X_train)))
        model = train(
            X_train,
            n_estimators=config["n_estimators"],
            max_samples=max_samples,
            contamination=config["contamination"],
            max_features=config["max_features"],
            bootstrap=config["bootstrap"],
        )
        metrics = validate(model, X_val, y_val)
        tune.report(
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
        )

    search_space = {
        "n_estimators": tune.randint(50, 500),
        "max_samples_frac": tune.uniform(0.1, 1.0),
        "contamination": tune.uniform(0.01, 0.15),
        "max_features": tune.uniform(0.3, 1.0),
        "bootstrap": tune.choice([True, False]),
    }

    optuna_search = OptunaSearch(metric="f1", mode="max")

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=num_samples,
            metric="f1",
            mode="max",
        ),
        run_config=ray.train.RunConfig(verbose=0),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="f1", mode="max")
    best_config = best_result.config
    best_metrics = best_result.metrics

    ray.shutdown()

    return {
        "best_config": best_config,
        "best_f1": best_metrics.get("f1"),
        "best_roc_auc": best_metrics.get("roc_auc"),
    }


# ---------------------------------------------------------------------------
# Compare parameter sets
# ---------------------------------------------------------------------------


def compare_parameter_sets() -> None:
    """Compare different Isolation Forest hyperparameter configurations.

    Tests n_estimators = 50 vs 100 vs 300 and contamination = 0.01 vs 0.05
    vs 0.1 to show how each setting affects precision, recall, and F1.
    Each configuration is annotated with BEST-FOR, RISK, and WHY.
    """
    print("=" * 70)
    print("COMPARE PARAMETER SETS (Isolation Forest sklearn)")
    print("=" * 70)

    # Generate data for comparison.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=2000, n_features=10, contamination=0.05, random_state=42
    )

    configs = [
        {
            "n_estimators": 50,
            "contamination": 0.05,
            "label": "FEW TREES (n_estimators=50, contamination=0.05)",
            "best_for": "Quick prototyping or when training speed is critical",
            "risk": "Higher variance: anomaly scores may be noisy with few trees, "
                    "leading to inconsistent predictions across runs",
            "why": "50 trees is the minimum for reasonable averaging. Each tree "
                   "provides one path-length sample; fewer trees = noisier estimate.",
        },
        {
            "n_estimators": 100,
            "contamination": 0.05,
            "label": "DEFAULT (n_estimators=100, contamination=0.05)",
            "best_for": "General-purpose anomaly detection with known ~5% anomaly rate",
            "risk": "If true contamination differs significantly from 0.05, the "
                    "threshold will be miscalibrated",
            "why": "100 trees is the original paper recommendation. Good balance "
                   "between computation cost and score stability.",
        },
        {
            "n_estimators": 300,
            "contamination": 0.05,
            "label": "MANY TREES (n_estimators=300, contamination=0.05)",
            "best_for": "Production systems where score stability is paramount",
            "risk": "3x slower training with diminishing returns beyond ~200 trees",
            "why": "More trees reduce variance in path-length estimates, producing "
                   "more stable anomaly scores at the cost of linear time increase.",
        },
        {
            "n_estimators": 100,
            "contamination": 0.01,
            "label": "LOW CONTAMINATION (n_estimators=100, contamination=0.01)",
            "best_for": "High-security applications where false positives are costly "
                        "(e.g., blocking legitimate transactions)",
            "risk": "Low recall: will miss many true anomalies because the threshold "
                    "is set very high",
            "why": "Contamination=0.01 means only the top 1% most anomalous points "
                   "are flagged. Prioritises precision over recall.",
        },
        {
            "n_estimators": 100,
            "contamination": 0.1,
            "label": "HIGH CONTAMINATION (n_estimators=100, contamination=0.1)",
            "best_for": "Scenarios with high anomaly rates or when recall is critical "
                        "(e.g., medical screening)",
            "risk": "High false positive rate: 10% of data is flagged, which may "
                    "overwhelm human reviewers",
            "why": "Contamination=0.1 lowers the anomaly threshold so more points "
                   "are flagged. Catches more anomalies but at the cost of precision.",
        },
    ]

    for cfg in configs:
        print("-" * 60)
        print(f"Config: {cfg['label']}")
        print(f"  BEST FOR : {cfg['best_for']}")
        print(f"  RISK     : {cfg['risk']}")
        print(f"  WHY      : {cfg['why']}")

        model = train(
            X_train,
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
        )
        metrics = validate(model, X_val, y_val)

        print(f"  Precision       : {metrics['precision']:.4f}")
        print(f"  Recall          : {metrics['recall']:.4f}")
        print(f"  F1              : {metrics['f1']:.4f}")
        print(f"  ROC AUC         : {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision   : {metrics['average_precision']:.4f}")

    print("=" * 70)
    print("Parameter comparison complete.")


# ---------------------------------------------------------------------------
# Real-world demo: credit card fraud detection
# ---------------------------------------------------------------------------


def real_world_demo() -> None:
    """Demonstrate Isolation Forest on a realistic credit card fraud scenario.

    Scenario: A bank wants to detect fraudulent credit card transactions.
    Each transaction has three features:
      - transaction_amount: dollar value of the purchase
      - time_of_day: hour of the transaction (0-24)
      - distance_from_home: miles from the cardholder's home address

    Normal transactions: moderate amounts, during business hours, near home.
    Fraudulent transactions: unusual amounts, odd hours, far from home.

    Isolation Forest is ideal here because:
      - Fraud is rare (~2-5% of transactions)
      - Fraudsters exhibit different spending patterns (outliers)
      - No labelled fraud data is needed for training (semi-supervised)
      - Real-time scoring is fast (linear in number of trees)
    """
    print("=" * 70)
    print("REAL-WORLD DEMO: Credit Card Fraud Detection")
    print("=" * 70)

    rng = np.random.RandomState(42)
    feature_names = ["transaction_amount", "time_of_day", "distance_from_home"]

    # --- Generate synthetic transaction data ---

    n_normal = 1900   # 95% normal transactions
    n_fraud = 100     # 5% fraudulent transactions

    # Normal transactions: moderate amounts ($20-200), business hours (8-20),
    # and close to home (0-15 miles).
    # WHY: most legitimate purchases happen during the day, for moderate
    # amounts, at local stores.
    normal_amount = rng.exponential(scale=50, size=(n_normal, 1)) + 10
    normal_amount = np.clip(normal_amount, 10, 300)  # clip extreme values
    normal_time = rng.normal(loc=14, scale=3, size=(n_normal, 1))
    normal_time = np.clip(normal_time, 0, 24)
    normal_distance = rng.exponential(scale=5, size=(n_normal, 1))
    normal_distance = np.clip(normal_distance, 0, 30)
    X_normal = np.hstack([normal_amount, normal_time, normal_distance])
    y_normal = np.zeros(n_normal, dtype=int)

    # Fraudulent transactions: large amounts ($500-5000), late night (0-5 AM),
    # and far from home (50-200 miles).
    # WHY: fraudsters tend to make large purchases quickly, often in different
    # cities, during hours when the cardholder is unlikely to notice.
    fraud_amount = rng.uniform(500, 5000, size=(n_fraud, 1))
    fraud_time = rng.uniform(0, 5, size=(n_fraud, 1))
    fraud_distance = rng.uniform(50, 200, size=(n_fraud, 1))
    X_fraud = np.hstack([fraud_amount, fraud_time, fraud_distance])
    y_fraud = np.ones(n_fraud, dtype=int)

    # Combine and shuffle.
    X_raw = np.vstack([X_normal, X_fraud])
    y_raw = np.concatenate([y_normal, y_fraud])
    idx = rng.permutation(len(X_raw))
    X_raw, y_raw = X_raw[idx], y_raw[idx]

    print(f"Generated {len(X_raw)} transactions ({n_normal} normal, {n_fraud} fraud)")
    print(f"Features: {feature_names}")
    print(f"Normal transaction profile: amount ~$50 (median), time ~2PM, distance ~5 mi")
    print(f"Fraud transaction profile:  amount ~$2500 (median), time ~2AM, distance ~125 mi")

    # --- Split and standardise ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=0.4, random_state=42, stratify=y_raw
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # --- Train Isolation Forest ---
    print("\nTraining Isolation Forest for fraud detection...")
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    # --- Evaluate ---
    y_pred = _predict_labels(model, X_test_s)
    scores = _anomaly_scores(model, X_test_s)
    metrics = _compute_metrics(y_test, y_pred, scores)

    print("\nTest Set Results:")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  F1              : {metrics['f1']:.4f}")
    print(f"  ROC AUC         : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision   : {metrics['average_precision']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    # --- Analyse flagged transactions ---
    flagged_mask = y_pred == 1
    flagged_raw = X_test[flagged_mask]  # use original (unscaled) values

    if len(flagged_raw) > 0:
        print(f"\nFlagged {flagged_mask.sum()} transactions as potential fraud:")
        print(f"  Avg amount       : ${flagged_raw[:, 0].mean():.2f}")
        print(f"  Avg time of day  : {flagged_raw[:, 1].mean():.1f}h")
        print(f"  Avg distance     : {flagged_raw[:, 2].mean():.1f} miles")

        # Show how flagged transactions differ from normal ones.
        normal_mask = y_pred == 0
        normal_raw = X_test[normal_mask]
        print(f"\n  Normal transactions for comparison:")
        print(f"    Avg amount     : ${normal_raw[:, 0].mean():.2f}")
        print(f"    Avg time of day: {normal_raw[:, 1].mean():.1f}h")
        print(f"    Avg distance   : {normal_raw[:, 2].mean():.1f} miles")

    print("=" * 70)
    print("Real-world demo complete.")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete Isolation Forest anomaly detection pipeline."""
    print("=" * 70)
    print("Isolation Forest - Scikit-Learn Implementation")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/5] Generating synthetic anomaly detection data...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=2000, n_features=10, contamination=0.05, random_state=42
    )

    # 2. Train with defaults
    print("\n[2/5] Training Isolation Forest with default hyperparameters...")
    model = train(X_train, contamination=0.05)

    # 3. Validate
    print("\n[3/5] Validation results:")
    val_metrics = validate(model, X_val, y_val)
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")

    # 4. Optuna hyperparameter optimization
    print("\n[4/5] Running Optuna hyperparameter optimization (20 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, y_val),
        n_trials=20,
        show_progress_bar=False,
    )
    print(f"  Best trial F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Retrain with best params
    best_params = study.best_params.copy()
    best_max_samples = max(2, int(best_params.pop("max_samples_frac") * len(X_train)))
    best_model = train(X_train, max_samples=best_max_samples, **best_params)

    # 5. Test
    print("\n[5/5] Test results (best model):")
    test_metrics = test(best_model, X_test, y_test)
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # 6. Parameter comparison and real-world demo
    compare_parameter_sets()
    real_world_demo()

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
