"""
Isolation Forest - NumPy From-Scratch Implementation
=====================================================

Theory & Mathematics:
    This module implements the Isolation Forest algorithm from scratch using
    only NumPy. The Isolation Forest detects anomalies by exploiting the fact
    that anomalous points are "few and different" -- they are isolated by random
    partitions in far fewer steps than normal points.

    iTree Construction (Algorithm 1):
        Given a subset of data X of size psi (sub-sample size):
        1. If |X| <= 1 or depth >= depth_limit: return ExternalNode(size=|X|)
        2. Randomly select feature q from available features
        3. Randomly select split value p in [min(X[:, q]), max(X[:, q])]
        4. X_left  = { x in X : x[q] < p }
           X_right = { x in X : x[q] >= p }
        5. return InternalNode(left=iTree(X_left), right=iTree(X_right), q, p)

    Path Length h(x) (Algorithm 2):
        For point x traversing an iTree:
        1. If node is ExternalNode: return node.depth + c(node.size)
        2. If x[node.q] < node.p: traverse left
        3. Else: traverse right
        where c(n) adjusts for unbuilt subtrees when the tree was truncated.

    Normalization Factor c(n):
        c(n) = 2 * H(n-1) - 2*(n-1)/n,    for n >= 2
        c(1) = 0
        c(0) = 0
        where H(i) = ln(i) + gamma  (Euler-Mascheroni constant, gamma ~ 0.5772)

    Anomaly Score s(x, n):
        s(x, n) = 2^{ -E[h(x)] / c(n) }
        - E[h(x)] = average path length over all trees in the forest
        - n = total number of training samples (or sub-sample size psi)

        Interpretation:
            s -> 1   : definite anomaly
            s -> 0.5 : normal (average path length ~ c(n))
            s -> 0   : very normal (long path lengths)

    Sub-Sampling:
        Each tree is trained on a random sub-sample of size psi (default 256)
        drawn without replacement. This:
        - Reduces swamping (normal points masking anomalies)
        - Reduces masking (anomalies masking each other)
        - Makes the algorithm O(t * psi * log(psi))

Business Use Cases:
    - Fraud detection in financial transactions
    - Network intrusion detection
    - Manufacturing defect detection
    - Medical anomaly detection
    - Sensor data monitoring for predictive maintenance

Advantages:
    - Linear time complexity O(t * psi * log(psi))
    - Low memory footprint
    - No distance or density estimation required
    - Handles high-dimensional data
    - Sub-sampling naturally addresses swamping/masking

Disadvantages:
    - Axis-parallel splits miss rotated or correlated anomaly clusters
    - Requires tuning contamination for threshold selection
    - Not effective for local anomalies (use LOF instead)
    - Performance can degrade with many irrelevant features

Key Hyperparameters:
    - n_estimators : number of iTrees (default=100)
    - max_samples  : sub-sample size psi (default=256)
    - contamination: proportion of anomalies for threshold (default=0.05)
    - random_state : reproducibility seed

References:
    - Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008. Isolation forest.
      IEEE ICDM, pp. 413-422.
    - Liu, F.T., Ting, K.M. and Zhou, Z.H., 2012. Isolation-based anomaly
      detection. ACM TKDD, 6(1), 3.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

# warnings suppresses convergence/deprecation noise from sklearn metrics.
import warnings

# Type hints for self-documenting function signatures.
from typing import Any, Dict, List, Optional, Tuple, Union

# NumPy is the sole computation library for this from-scratch implementation.
# WHY: building from scratch with NumPy ensures full understanding of every
# algorithmic step without relying on opaque library internals.
import numpy as np

# Optuna performs Bayesian hyperparameter optimisation via TPE.
import optuna

# sklearn metrics for evaluating anomaly detection performance.
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# train_test_split for reproducible data partitioning.
from sklearn.model_selection import train_test_split

# StandardScaler for feature normalisation.
from sklearn.preprocessing import StandardScaler

# Suppress all warnings for clean output.
warnings.filterwarnings("ignore")

# Euler-Mascheroni constant used in the c(n) normalisation factor.
# WHY: this mathematical constant appears in the expected path length
# of an unsuccessful search in a Binary Search Tree, which is used to
# normalise raw path lengths into the [0, 1] anomaly score range.
EULER_MASCHERONI = 0.5772156649


# ---------------------------------------------------------------------------
# iTree Node Classes
# ---------------------------------------------------------------------------


class ExternalNode:
    """Leaf node of an Isolation Tree.

    Stores the number of samples that reached this node (size) and the
    depth at which the node was created.  The size is used to estimate
    the path length for the unbuilt subtree via c(size).
    """

    # Use __slots__ to save memory since trees can have thousands of nodes.
    # WHY: each node is a lightweight object; __slots__ avoids the overhead
    # of a full __dict__ per instance.
    __slots__ = ("size", "depth")

    def __init__(self, size: int, depth: int) -> None:
        # size = number of samples that reached this leaf.
        # WHY: needed to compute c(size), the average path length
        # of the unbuilt subtree below this node.
        self.size = size

        # depth = how many splits were made to reach this node.
        # WHY: the total path length = depth + c(size), combining
        # the actual depth traversed with the estimated remainder.
        self.depth = depth


class InternalNode:
    """Internal (split) node of an Isolation Tree.

    Stores the split feature, split value, and pointers to left/right children.
    """

    __slots__ = ("left", "right", "split_feature", "split_value", "depth")

    def __init__(
        self,
        left: Union["InternalNode", ExternalNode],
        right: Union["InternalNode", ExternalNode],
        split_feature: int,
        split_value: float,
        depth: int,
    ) -> None:
        # left child: samples where X[:, split_feature] < split_value.
        self.left = left
        # right child: samples where X[:, split_feature] >= split_value.
        self.right = right
        # split_feature: randomly selected feature index for this split.
        self.split_feature = split_feature
        # split_value: randomly selected threshold within [min, max] of the feature.
        self.split_value = split_value
        # depth: number of splits from the root to this node.
        self.depth = depth


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _c(n: int) -> float:
    """
    Average path length of unsuccessful search in a Binary Search Tree.

    c(n) = 2*H(n-1) - 2*(n-1)/n  for n >= 2
    c(1) = 0;  c(0) = 0

    H(i) = ln(i) + Euler-Mascheroni constant

    WHY: this normalisation factor converts raw path lengths into the
    [0, 1] anomaly score range.  Without it, the score would depend on
    the sub-sample size, making comparisons across datasets impossible.
    """
    # Base cases: zero or one sample cannot form a meaningful BST.
    if n <= 1:
        return 0.0

    # Harmonic number approximation using the Euler-Mascheroni constant.
    # WHY: H(n) = sum(1/k for k=1..n) ~ ln(n) + gamma, and this
    # approximation is accurate enough for all practical n.
    h = np.log(n - 1) + EULER_MASCHERONI
    return 2.0 * h - 2.0 * (n - 1) / n


def _build_itree(
    X: np.ndarray,
    depth: int,
    depth_limit: int,
    rng: np.random.RandomState,
) -> Union[InternalNode, ExternalNode]:
    """
    Recursively build an Isolation Tree.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Data subset for this node.
    depth : int
        Current tree depth.
    depth_limit : int
        Maximum allowed depth (ceil(log2(psi))).
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    InternalNode or ExternalNode
    """
    n, d = X.shape

    # Base case 1: single sample or depth limit reached.
    # WHY: a single sample is already isolated (path length = depth).
    # Reaching the depth limit triggers early stopping to keep trees shallow.
    if n <= 1 or depth >= depth_limit:
        return ExternalNode(size=n, depth=depth)

    # Randomly select a feature to split on.
    # WHY: random feature selection (not best-feature) is fundamental to
    # Isolation Forest -- it ensures anomalies are isolated by chance
    # rather than by information gain.
    q = rng.randint(0, d)
    col = X[:, q]
    col_min, col_max = col.min(), col.max()

    # If all values are identical, cannot split further.
    # WHY: a uniform feature provides no separation; creating a split
    # would produce an empty child, which is degenerate.
    if col_min == col_max:
        return ExternalNode(size=n, depth=depth)

    # Random split value uniformly sampled within [col_min, col_max).
    # WHY: uniform random splits are the second key ingredient -- they
    # make anomalies (extreme values) easier to isolate because random
    # thresholds are more likely to separate outliers from the bulk.
    p = rng.uniform(col_min, col_max)

    # Partition data into left (< p) and right (>= p) subsets.
    mask = col < p
    X_left = X[mask]
    X_right = X[~mask]

    # Edge case: if partition produces an empty side, stop splitting.
    # WHY: this can happen due to floating-point edge cases; an empty
    # child would have no samples to recurse on.
    if len(X_left) == 0 or len(X_right) == 0:
        return ExternalNode(size=n, depth=depth)

    # Recursively build left and right subtrees.
    left_child = _build_itree(X_left, depth + 1, depth_limit, rng)
    right_child = _build_itree(X_right, depth + 1, depth_limit, rng)

    return InternalNode(
        left=left_child,
        right=right_child,
        split_feature=q,
        split_value=p,
        depth=depth,
    )


def _path_length(x: np.ndarray, node: Union[InternalNode, ExternalNode]) -> float:
    """
    Compute the path length of a single sample x through an iTree.

    For external nodes, adds c(node.size) to account for the average depth
    of the unbuilt subtree.

    WHY: the path length is the fundamental quantity in Isolation Forest --
    anomalies have short path lengths (easy to isolate) while normal points
    have long path lengths (hard to isolate).
    """
    # Leaf node: return depth + estimated remaining path length.
    # WHY: c(node.size) estimates how much deeper the tree would have gone
    # if we had not stopped at the depth limit.
    if isinstance(node, ExternalNode):
        return float(node.depth) + _c(node.size)

    # Internal node: follow the split to the appropriate child.
    # WHY: this mirrors the tree traversal that would occur during the
    # construction phase -- each split adds 1 to the path length.
    if x[node.split_feature] < node.split_value:
        return _path_length(x, node.left)
    else:
        return _path_length(x, node.right)


# ---------------------------------------------------------------------------
# Isolation Forest Model
# ---------------------------------------------------------------------------


class IsolationForestNumpy:
    """
    Isolation Forest implemented from scratch using NumPy.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    max_samples : int
        Sub-sample size for each tree.
    contamination : float
        Expected proportion of anomalies (used for threshold).
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        # Number of isolation trees in the ensemble.
        # WHY: more trees = more stable anomaly scores, at linear cost.
        self.n_estimators = n_estimators

        # Sub-sample size (psi) for each tree.
        # WHY: sub-sampling is key to Isolation Forest's effectiveness;
        # smaller samples make anomalies easier to isolate.
        self.max_samples = max_samples

        # Expected anomaly proportion for threshold-based classification.
        # WHY: the threshold is set at the (1-contamination) percentile
        # of training scores.
        self.contamination = contamination

        # Random seed for reproducibility.
        self.random_state = random_state

        # List of built isolation trees (populated during fit).
        self.trees_: List[Union[InternalNode, ExternalNode]] = []

        # Decision threshold computed from training scores.
        self.threshold_: float = 0.0

        # Actual sub-sample size used (may be less than max_samples if
        # the dataset is smaller).
        self._psi: int = 0

    def fit(self, X: np.ndarray) -> "IsolationForestNumpy":
        """
        Build the isolation forest from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Training data (mostly normal).

        Returns
        -------
        self
        """
        n, d = X.shape

        # Use min(max_samples, n) as the actual sub-sample size.
        # WHY: if the dataset is smaller than max_samples, we use all of it.
        self._psi = min(self.max_samples, n)

        # Depth limit = ceil(log2(psi)).
        # WHY: the average path length of a BST with psi nodes is O(log(psi)),
        # so growing trees deeper is wasteful.
        depth_limit = int(np.ceil(np.log2(max(self._psi, 2))))

        # Create a reproducible random number generator.
        rng = np.random.RandomState(self.random_state)
        self.trees_ = []

        # Build n_estimators isolation trees, each on a different sub-sample.
        for _ in range(self.n_estimators):
            # Sub-sample without replacement.
            # WHY: sampling without replacement is the original paper's
            # recommendation; it reduces both swamping and masking effects.
            idx = rng.choice(n, size=self._psi, replace=False)
            X_sub = X[idx]
            tree = _build_itree(X_sub, depth=0, depth_limit=depth_limit, rng=rng)
            self.trees_.append(tree)

        # Compute anomaly scores on training data to set the threshold.
        # WHY: the threshold converts continuous scores into binary
        # predictions (normal vs anomaly).
        scores = self.anomaly_score(X)
        self.threshold_ = float(np.percentile(scores, 100 * (1 - self.contamination)))

        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for each sample.

        s(x, psi) = 2^(-E[h(x)] / c(psi))

        Returns
        -------
        np.ndarray of shape (n,)
            Anomaly scores in [0, 1]. Higher = more anomalous.
        """
        n = X.shape[0]

        # Accumulate path lengths across all trees for each sample.
        avg_path_lengths = np.zeros(n)

        for tree in self.trees_:
            for i in range(n):
                # Add this tree's path length for sample i.
                avg_path_lengths[i] += _path_length(X[i], tree)

        # Average over all trees.
        # WHY: averaging reduces the variance of individual tree estimates.
        avg_path_lengths /= self.n_estimators

        # Normalisation factor c(psi).
        # WHY: dividing by c(psi) puts the path length in a scale-independent
        # range that can be converted to a [0, 1] score.
        c_psi = _c(self._psi)

        # Edge case: if c_psi is 0 (psi <= 1), return neutral score.
        if c_psi == 0:
            return np.ones(n) * 0.5

        # Compute anomaly scores: s = 2^(-E[h(x)] / c(psi)).
        # WHY: this formula maps short path lengths to scores near 1 (anomaly)
        # and long path lengths to scores near 0 (normal).
        scores = np.power(2.0, -avg_path_lengths / c_psi)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Returns
        -------
        np.ndarray of shape (n,)
            1 = anomaly, 0 = normal.
        """
        scores = self.anomaly_score(X)
        # Points with scores above the threshold are classified as anomalies.
        return (scores >= self.threshold_).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores (higher = more anomalous)."""
        return self.anomaly_score(X)


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

    Normal samples: multivariate Gaussian centered at origin.
    Anomalies: uniform distribution in [-6, 6]^d.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    rng = np.random.RandomState(random_state)

    n_normal = int(n_samples * (1 - contamination))
    n_anomaly = n_samples - n_normal

    # Normal: Gaussian centred at origin.
    X_normal = rng.randn(n_normal, n_features)
    y_normal = np.zeros(n_normal, dtype=int)

    # Anomalies: uniform in a wide range, landing in low-density regions.
    X_anomaly = rng.uniform(low=-6, high=6, size=(n_anomaly, n_features))
    y_anomaly = np.ones(n_anomaly, dtype=int)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])

    # Shuffle to remove ordering bias.
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Stratified split: 60/20/20.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # Standardise using training statistics only.
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


def train(X_train: np.ndarray, **hyperparams) -> IsolationForestNumpy:
    """Train an Isolation Forest model from scratch."""
    model = IsolationForestNumpy(**hyperparams)
    model.fit(X_train)
    return model


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> Dict[str, Any]:
    """Compute anomaly detection metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    avg_precision = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
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
    model: IsolationForestNumpy,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate model on validation set."""
    y_pred = model.predict(X_val)
    scores = model.anomaly_score(X_val)
    return _compute_metrics(y_val, y_pred, scores)


def test(
    model: IsolationForestNumpy,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate model on held-out test set."""
    y_pred = model.predict(X_test)
    scores = model.anomaly_score(X_test)
    return _compute_metrics(y_test, y_pred, scores)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for Isolation Forest hyperparameter search."""
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_samples = trial.suggest_int("max_samples", 64, min(512, len(X_train)))
    contamination = trial.suggest_float("contamination", 0.01, 0.15)

    model = train(
        X_train,
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42,
    )

    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Hyperparameter search using Ray Tune with Optuna."""
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    def trainable(config: Dict[str, Any]) -> None:
        model = train(
            X_train,
            n_estimators=config["n_estimators"],
            max_samples=config["max_samples"],
            contamination=config["contamination"],
            random_state=42,
        )
        metrics = validate(model, X_val, y_val)
        tune.report(
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
        )

    search_space = {
        "n_estimators": tune.randint(50, 300),
        "max_samples": tune.randint(64, min(512, len(X_train))),
        "contamination": tune.uniform(0.01, 0.15),
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
    vs 0.1.  Each configuration includes BEST-FOR, RISK, and WHY annotations.
    """
    print("=" * 70)
    print("COMPARE PARAMETER SETS (Isolation Forest NumPy from scratch)")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=1500, n_features=10, contamination=0.05, random_state=42
    )

    configs = [
        {
            "n_estimators": 50,
            "contamination": 0.05,
            "label": "FEW TREES (n_estimators=50, contamination=0.05)",
            "best_for": "Quick prototyping when speed matters more than accuracy",
            "risk": "High variance: anomaly scores are noisy with few trees",
            "why": "50 trees provide minimal averaging. From-scratch implementation "
                   "is slower per tree than sklearn, so fewer trees save significant time.",
        },
        {
            "n_estimators": 100,
            "contamination": 0.05,
            "label": "DEFAULT (n_estimators=100, contamination=0.05)",
            "best_for": "General-purpose anomaly detection with known ~5% anomaly rate",
            "risk": "Miscalibrated threshold if true contamination differs from 0.05",
            "why": "100 trees is the original paper recommendation. Balances score "
                   "stability with computation time for our from-scratch implementation.",
        },
        {
            "n_estimators": 300,
            "contamination": 0.05,
            "label": "MANY TREES (n_estimators=300, contamination=0.05)",
            "best_for": "Production systems where score stability is essential",
            "risk": "3x slower than 100 trees; diminishing returns beyond ~200",
            "why": "More trees reduce variance in path-length estimates. Our from-scratch "
                   "implementation is O(n * n_estimators) for scoring, so 300 is notable.",
        },
        {
            "n_estimators": 100,
            "contamination": 0.01,
            "label": "LOW CONTAMINATION (n_estimators=100, contamination=0.01)",
            "best_for": "High-precision applications (blocking transactions is costly)",
            "risk": "Low recall: misses many true anomalies",
            "why": "Only the top 1% most anomalous points are flagged. The threshold is "
                   "set at the 99th percentile of training scores.",
        },
        {
            "n_estimators": 100,
            "contamination": 0.1,
            "label": "HIGH CONTAMINATION (n_estimators=100, contamination=0.1)",
            "best_for": "High-recall screening (medical, safety-critical)",
            "risk": "High false positive rate overwhelms human reviewers",
            "why": "Threshold is lowered so 10% of data is flagged. Catches more "
                   "anomalies but generates more false alarms.",
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
            random_state=42,
        )
        metrics = validate(model, X_val, y_val)

        print(f"  Precision       : {metrics['precision']:.4f}")
        print(f"  Recall          : {metrics['recall']:.4f}")
        print(f"  F1              : {metrics['f1']:.4f}")
        print(f"  ROC AUC         : {metrics['roc_auc']:.4f}")

    print("=" * 70)
    print("Parameter comparison complete.")


# ---------------------------------------------------------------------------
# Real-world demo: credit card fraud detection
# ---------------------------------------------------------------------------


def real_world_demo() -> None:
    """Demonstrate from-scratch Isolation Forest on credit card fraud detection.

    Scenario: detect fraudulent credit card transactions using three features:
      - transaction_amount ($)
      - time_of_day (0-24h)
      - distance_from_home (miles)

    Normal: moderate amounts, daytime, near home.
    Fraud: large amounts, late night, far from home.
    """
    print("=" * 70)
    print("REAL-WORLD DEMO: Credit Card Fraud Detection (from scratch)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    feature_names = ["transaction_amount", "time_of_day", "distance_from_home"]

    # --- Generate synthetic transaction data ---
    n_normal = 1900
    n_fraud = 100

    # Normal transactions.
    normal_amount = np.clip(rng.exponential(50, (n_normal, 1)) + 10, 10, 300)
    normal_time = np.clip(rng.normal(14, 3, (n_normal, 1)), 0, 24)
    normal_distance = np.clip(rng.exponential(5, (n_normal, 1)), 0, 30)
    X_normal = np.hstack([normal_amount, normal_time, normal_distance])
    y_normal = np.zeros(n_normal, dtype=int)

    # Fraudulent transactions.
    fraud_amount = rng.uniform(500, 5000, (n_fraud, 1))
    fraud_time = rng.uniform(0, 5, (n_fraud, 1))
    fraud_distance = rng.uniform(50, 200, (n_fraud, 1))
    X_fraud = np.hstack([fraud_amount, fraud_time, fraud_distance])
    y_fraud = np.ones(n_fraud, dtype=int)

    X_raw = np.vstack([X_normal, X_fraud])
    y_raw = np.concatenate([y_normal, y_fraud])
    idx = rng.permutation(len(X_raw))
    X_raw, y_raw = X_raw[idx], y_raw[idx]

    print(f"Generated {len(X_raw)} transactions ({n_normal} normal, {n_fraud} fraud)")
    print(f"Features: {feature_names}")

    # Split and standardise.
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

    # Train from-scratch Isolation Forest.
    print("\nTraining from-scratch Isolation Forest...")
    model = IsolationForestNumpy(
        n_estimators=150,
        max_samples=256,
        contamination=0.05,
        random_state=42,
    )
    model.fit(X_train_s)

    # Evaluate.
    y_pred = model.predict(X_test_s)
    scores = model.anomaly_score(X_test_s)
    metrics = _compute_metrics(y_test, y_pred, scores)

    print("\nTest Set Results:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    # Analyse flagged transactions.
    flagged_mask = y_pred == 1
    flagged_raw = X_test[flagged_mask]

    if len(flagged_raw) > 0:
        print(f"\nFlagged {flagged_mask.sum()} transactions as potential fraud:")
        print(f"  Avg amount       : ${flagged_raw[:, 0].mean():.2f}")
        print(f"  Avg time of day  : {flagged_raw[:, 1].mean():.1f}h")
        print(f"  Avg distance     : {flagged_raw[:, 2].mean():.1f} miles")

    print("=" * 70)
    print("Real-world demo complete.")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete from-scratch Isolation Forest pipeline."""
    print("=" * 70)
    print("Isolation Forest - NumPy From-Scratch Implementation")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/5] Generating synthetic anomaly detection data...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=1500, n_features=10, contamination=0.05, random_state=42
    )

    # 2. Train with defaults
    print("\n[2/5] Training Isolation Forest from scratch...")
    model = train(
        X_train,
        n_estimators=100,
        max_samples=256,
        contamination=0.05,
        random_state=42,
    )

    # 3. Validate
    print("\n[3/5] Validation results:")
    val_metrics = validate(model, X_val, y_val)
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>20s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")

    # 4. Optuna
    print("\n[4/5] Running Optuna hyperparameter optimization (15 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, y_val),
        n_trials=15,
        show_progress_bar=False,
    )
    print(f"  Best trial F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Retrain with best params
    best_model = train(X_train, **study.best_params, random_state=42)

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
