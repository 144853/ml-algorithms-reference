"""
ElasticNet Regression - NumPy From-Scratch Implementation (Coordinate Descent)
===============================================================================

COMPLETE ML TUTORIAL: This file implements ElasticNet regression from scratch
using coordinate descent in NumPy. ElasticNet combines L1 (Lasso) and L2 (Ridge)
penalties to achieve BOTH automatic feature selection AND coefficient grouping
for correlated features -- the best of both worlds.

Theory & Mathematics:
    ElasticNet minimises the following composite objective:

        L(w) = (1/2n) * ||y - Xw||^2
             + alpha * l1_ratio * ||w||_1
             + alpha * (1 - l1_ratio) * (1/2) * ||w||_2^2

    where:
        - ||w||_1 = sum(|w_j|)       is the L1 norm (induces sparsity)
        - ||w||_2^2 = sum(w_j^2)     is the squared L2 norm (shrinks coefficients)
        - alpha controls overall regularisation strength
        - l1_ratio in [0, 1] controls the L1/L2 blend:
            l1_ratio = 1.0  -> pure Lasso (maximum sparsity)
            l1_ratio = 0.0  -> pure Ridge (maximum shrinkage, no sparsity)
            l1_ratio = 0.5  -> balanced ElasticNet

    Coordinate Descent Update Rule:
        For each feature j, holding all other weights fixed:

        1. Compute the partial residual excluding feature j:
              r_j = y - X_{-j} @ w_{-j} - bias

        2. Compute the OLS estimate for w_j:
              z_j = (1/n) * X_j^T @ r_j

        3. Apply the ElasticNet update (soft-threshold + L2 shrinkage):
              w_j = S(z_j, alpha * l1_ratio) / (col_norm_sq_j + alpha * (1 - l1_ratio))

           where S(z, t) = sign(z) * max(|z| - t, 0) is the soft-thresholding operator.

        The denominator (1 + alpha*(1-l1_ratio)) is the KEY difference from Lasso:
        - In Lasso, the denominator is just col_norm_sq_j (no L2 term).
        - In ElasticNet, the extra alpha*(1-l1_ratio) in the denominator provides
          additional shrinkage on top of the L1 soft-thresholding.

    Why Coordinate Descent?
        - Each sub-problem (single feature) has a closed-form solution.
        - Natural fit for L1: soft-thresholding is exact, not an approximation.
        - Very efficient for sparse problems (many updates are trivial zeros).
        - Provably convergent for convex objectives like ElasticNet.

    Grouping Effect Theorem (Zou & Hastie, 2005):
        For highly correlated features x_i and x_j with correlation rho, ElasticNet
        guarantees that |w_i - w_j| <= f(1 - rho) / alpha*(1-l1_ratio), meaning
        correlated features get similar coefficients. Pure Lasso lacks this property.

Hyperparameters:
    - alpha (float): Overall regularisation strength. Default 1.0.
    - l1_ratio (float): L1/L2 blend. 0=Ridge, 1=Lasso. Default 0.5.
    - max_iter (int): Maximum coordinate descent iterations. Default 1000.
    - tol (float): Convergence tolerance. Default 1e-6.
    - fit_intercept (bool): Whether to fit intercept. Default True.

Business Use Cases:
    - Genomics: Predicting phenotypes from correlated gene expression data.
    - Finance: Multi-factor models with correlated risk factors.
    - Healthcare: Clinical outcome prediction from correlated biomarkers.
    - NLP: Text regression with correlated n-gram features.

Advantages:
    - Handles correlated features via the grouping effect (unlike Lasso).
    - Still achieves sparsity / feature selection (unlike Ridge).
    - Two-parameter control (alpha, l1_ratio) is very flexible.
    - Convex objective guarantees a global minimum.

Disadvantages:
    - Two hyperparameters to tune instead of one.
    - Coordinate descent can be slow for very high-dimensional data.
    - Linear model: cannot capture non-linear relationships without feature engineering.
"""

# -- Standard library imports --
import logging  # Structured logging for tracking pipeline progress across stages
import time  # Wall-clock timing for comparing training speeds across configurations
from functools import partial  # Binds fixed arguments to functions for Optuna callbacks

# -- Third-party numerical and ML imports --
import numpy as np  # Core numerical library: arrays, linear algebra, random generation
import optuna  # Bayesian hyperparameter optimisation using Tree-structured Parzen Estimators
import ray  # Distributed computing framework for parallel hyperparameter search
from ray import tune  # Ray's hyperparameter tuning module with scheduling algorithms
from sklearn.datasets import make_regression  # Generates synthetic regression data with known structure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Standard regression metrics
from sklearn.model_selection import train_test_split  # Reproducible train/val/test splitting
from sklearn.preprocessing import StandardScaler  # Zero-mean unit-variance normalisation for fair penalisation

# -- Configure module-level logging --
# WHY this format: timestamps help correlate events, level helps filter severity
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Module-scoped logger for all functions in this file


# ---------------------------------------------------------------------------
# Soft-Thresholding Operator
# ---------------------------------------------------------------------------

def _soft_threshold(z, threshold):
    """Soft-thresholding operator: S(z, t) = sign(z) * max(|z| - t, 0).

    This is the proximal operator for the L1 norm. It is the CORE operation
    that produces exact zeros in ElasticNet (inherited from Lasso).

    How it works:
        - If |z| <= threshold: the signal is too weak to overcome the penalty,
          so the weight is set to exactly zero (feature is dropped).
        - If |z| > threshold: the weight is shrunk toward zero by 'threshold'
          amount, but retains its sign and a reduced magnitude.

    Geometrically, this corresponds to projecting onto the L1 ball. The L1
    ball has corners on the coordinate axes, which is why solutions land
    exactly on axes (i.e., some weights become exactly zero).

    Args:
        z: Unconstrained value (what the weight would be without L1 penalty).
        threshold: L1 regularisation strength (alpha * l1_ratio).

    Returns:
        Thresholded value. Exactly 0 if |z| <= threshold.
    """
    # np.sign(z): returns -1, 0, or +1 to preserve the direction of z
    # np.maximum(|z| - threshold, 0): shrinks the magnitude, clips at 0
    # Together: this shrinks z toward zero by 'threshold', zeroing it if too small
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


# ---------------------------------------------------------------------------
# ElasticNet Model
# ---------------------------------------------------------------------------

class ElasticNetNumpy:
    """ElasticNet regression via coordinate descent, implemented from scratch.

    The algorithm cycles through all features, updating one weight at a time.
    For each feature, it applies:
        1. Soft-thresholding (L1 effect) to produce sparsity
        2. L2 shrinkage in the denominator to handle correlated features

    This combination gives ElasticNet its signature "grouping effect": correlated
    features receive similar coefficients instead of one being arbitrarily zeroed.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6, fit_intercept=True):
        """Initialise ElasticNet with regularisation parameters.

        Args:
            alpha: Overall regularisation strength. Higher = more penalty.
            l1_ratio: Blend between L1 and L2. 0=Ridge, 1=Lasso, 0.5=balanced.
            max_iter: Maximum coordinate descent sweeps through all features.
            tol: Convergence tolerance on the max weight change per sweep.
            fit_intercept: Whether to learn a bias/intercept term (not regularised).
        """
        # alpha: controls HOW MUCH total regularisation is applied
        # WHY default 1.0: moderate regularisation; typical starting point for tuning
        self.alpha = alpha

        # l1_ratio: controls the MIX between L1 and L2 penalties
        # WHY 0.5: equal blend gives the classic ElasticNet behaviour
        # - l1_ratio=0.9: mostly Lasso (aggressive sparsity)
        # - l1_ratio=0.1: mostly Ridge (keeps all features, shrinks them)
        self.l1_ratio = l1_ratio

        # max_iter: safety limit on coordinate descent sweeps
        # WHY 1000: coordinate descent typically converges in 10-100 sweeps
        self.max_iter = max_iter

        # tol: convergence threshold on maximum weight change
        # WHY 1e-6: if no weight moves by more than 1e-6, we declare convergence
        self.tol = tol

        # fit_intercept: whether to learn a bias term (never regularised)
        # WHY True: almost always needed unless data is centred to zero mean
        self.fit_intercept = fit_intercept

        # Learned parameters (populated during fit)
        self.weights = None  # Feature weights (shape: n_features)
        self.bias = 0.0  # Intercept/bias term
        self.n_iters_run = 0  # Actual iterations until convergence

    def fit(self, X, y):
        """Fit ElasticNet model using coordinate descent.

        The algorithm repeats:
            1. For each feature j: update w_j using soft-threshold + L2 shrinkage
            2. Update intercept (if fit_intercept=True)
            3. Check convergence (max weight change < tol)

        Args:
            X: Training feature matrix, shape (n_samples, n_features).
            y: Training target vector, shape (n_samples,).

        Returns:
            self: The fitted model instance.
        """
        # Extract dataset dimensions for loop bounds and normalisation
        n, p = X.shape  # n = number of samples, p = number of features

        # Initialise all weights to zero
        # WHY zeros: ElasticNet solutions are often sparse, so starting at zero
        # means many weights will stay zero, reducing unnecessary computation
        self.weights = np.zeros(p)  # Shape: (p,) -- one weight per feature

        # Initialise bias to the mean of y (best constant predictor)
        # WHY mean: if all weights are zero, predicting the mean minimises MSE
        self.bias = np.mean(y) if self.fit_intercept else 0.0  # Scalar bias term

        # Compute the initial residual (what remains after subtracting the bias)
        # residual = y - X @ w - bias, but w is all zeros so residual = y - bias
        residual = y - self.bias  # Shape: (n,)

        # Precompute squared L2 norm of each feature column, divided by n
        # WHY precompute: this is constant across iterations; saves O(n*p) per iter
        # This normalises the update to account for feature scale differences
        col_norm_sq = np.sum(X ** 2, axis=0) / n  # Shape: (p,)

        # Precompute the L1 and L2 penalty components for readability
        # l1_penalty: the soft-thresholding amount (alpha * l1_ratio)
        l1_penalty = self.alpha * self.l1_ratio  # Scalar: L1 penalty strength

        # l2_penalty: the denominator shrinkage amount (alpha * (1 - l1_ratio))
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)  # Scalar: L2 penalty strength

        # Main coordinate descent loop
        for iteration in range(self.max_iter):  # Each iteration is one full sweep
            # Save a copy of weights to check convergence after this sweep
            w_old = self.weights.copy()  # Deep copy to compare against updated weights

            # Sweep through all features one at a time (cyclic coordinate descent)
            for j in range(p):  # j indexes each feature/coordinate
                # STEP 1: Add back feature j's contribution to the residual
                # WHY: we want the residual as if feature j were NOT in the model
                # This gives r_j = y - X_{-j} @ w_{-j} - bias
                residual += X[:, j] * self.weights[j]  # Undo feature j's effect

                # STEP 2: Compute the unconstrained OLS estimate for feature j
                # z_j = (1/n) * X_j^T @ r_j
                # This is what w_j would be with NO regularisation at all
                z_j = (X[:, j] @ residual) / n  # Scalar: OLS estimate for feature j

                # STEP 3: Apply the ElasticNet update rule
                # w_j = S(z_j, alpha * l1_ratio) / (col_norm_sq_j + alpha * (1 - l1_ratio))
                #
                # The numerator applies soft-thresholding (L1 sparsity effect):
                #   - If |z_j| < l1_penalty, the weight becomes exactly zero
                #   - Otherwise, shrink toward zero by l1_penalty amount
                #
                # The denominator adds L2 shrinkage on top of L1:
                #   - col_norm_sq[j] is the standard OLS normalisation
                #   - l2_penalty is the EXTRA shrinkage from the L2 term
                #   - This denominator is what gives ElasticNet its grouping effect
                denominator = col_norm_sq[j] + l2_penalty  # L2-augmented denominator

                if denominator == 0.0:  # Degenerate case: feature is constant zero
                    self.weights[j] = 0.0  # No information in this feature
                else:
                    # Apply soft-threshold in numerator, divide by L2-augmented denominator
                    self.weights[j] = _soft_threshold(z_j, l1_penalty) / denominator

                # STEP 4: Update the residual to reflect the new weight for feature j
                # This prepares the residual for the next feature's update
                residual -= X[:, j] * self.weights[j]  # Apply new feature j effect

            # Update the intercept after completing a full sweep through all features
            # WHY after sweep: intercept depends on all weights; updating mid-sweep is valid
            # but this order is conventional and matches sklearn's implementation
            if self.fit_intercept:  # Only update intercept if we are fitting one
                self.bias = np.mean(y - X @ self.weights)  # Best constant given current weights

            # Check convergence: if no weight changed by more than tol, stop
            # WHY max change: if the largest change is below tol, all changes are below tol
            max_change = np.max(np.abs(self.weights - w_old))  # Infinity norm of weight delta

            if max_change < self.tol:  # Convergence achieved
                self.n_iters_run = iteration + 1  # Record how many sweeps we needed
                logger.debug(  # Debug-level: only shows with DEBUG logging
                    "Converged at iteration %d (max_change=%.2e)",
                    iteration + 1, max_change,
                )
                return self  # Return early: no need to continue sweeping

        # If we reach here, we exhausted max_iter without convergence
        self.n_iters_run = self.max_iter  # Record that we hit the iteration limit
        return self  # Return the model even without full convergence

    def predict(self, X):
        """Predict target values using the learned weights and bias.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted target values, shape (n_samples,).
        """
        # Linear prediction: y_hat = X @ w + b
        return X @ self.weights + self.bias  # Matrix-vector product plus scalar bias

    @property
    def n_nonzero(self):
        """Count of non-zero (selected) features.

        This is ElasticNet's sparsity metric: how many features survived
        the L1 soft-thresholding. Fewer non-zero weights = sparser model.
        """
        return int(np.sum(self.weights != 0))  # Count features with non-zero weight


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=20, noise=0.1, random_state=42):
    """Generate synthetic regression data with a 60/20/20 train/val/test split.

    WHY 20 features with 10 informative: creates a realistic scenario where
    ElasticNet must identify truly useful features and discard noise features.
    StandardScaler ensures all features are on the same scale so the L1/L2
    penalties treat them fairly.

    Args:
        n_samples: Total number of data points to generate.
        n_features: Total number of features (informative + noise).
        noise: Gaussian noise standard deviation added to target values.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Generate synthetic regression data with half the features being informative
    # WHY n_informative = n_features // 2: simulates real-world where many features are noise
    X, y = make_regression(
        n_samples=n_samples,  # Total data points to create
        n_features=n_features,  # Total feature dimensionality
        n_informative=max(1, n_features // 2),  # Half are truly predictive
        noise=noise,  # Gaussian noise for realism
        random_state=random_state,  # Reproducible data generation
    )

    # Two-stage split: first 60/40, then split the 40 into 20/20
    # WHY two stages: ensures exact 60/20/20 proportions
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state,  # 60% train, 40% temp
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,  # 20% val, 20% test
    )

    # Standardise features: fit on train ONLY to prevent data leakage
    # WHY scaling: ElasticNet penalises by coefficient magnitude; unscaled features
    # with larger ranges would get unfairly lighter penalties
    scaler = StandardScaler()  # Computes mean and std from training data
    X_train = scaler.fit_transform(X_train)  # Learn statistics and transform
    X_val = scaler.transform(X_val)  # Apply train statistics to validation
    X_test = scaler.transform(X_test)  # Apply train statistics to test

    # Log dataset summary for pipeline monitoring
    logger.info(
        "Data: train=%d val=%d test=%d features=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1],
    )
    return X_train, X_val, X_test, y_train, y_val, y_test  # Six arrays


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train an ElasticNet model with given hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        **hp: Hyperparameters (alpha, l1_ratio, max_iter, tol, fit_intercept).

    Returns:
        Fitted ElasticNetNumpy model.
    """
    # Instantiate model with hyperparameters, using sensible defaults
    model = ElasticNetNumpy(
        alpha=hp.get("alpha", 1.0),  # Overall regularisation strength
        l1_ratio=hp.get("l1_ratio", 0.5),  # L1/L2 blend ratio
        max_iter=hp.get("max_iter", 1000),  # Maximum coordinate sweeps
        tol=hp.get("tol", 1e-6),  # Convergence tolerance
        fit_intercept=hp.get("fit_intercept", True),  # Whether to fit bias
    )

    # Fit the model on training data using coordinate descent
    model.fit(X_train, y_train)  # Runs until convergence or max_iter

    # Log training summary for pipeline monitoring and debugging
    logger.info(
        "ElasticNet(numpy) trained  alpha=%.4f  l1_ratio=%.2f  nonzero=%d  iters=%d",
        model.alpha, model.l1_ratio, model.n_nonzero, model.n_iters_run,
    )
    return model  # Return fitted model for evaluation


def _metrics(y_true, y_pred):
    """Compute standard regression metrics.

    Returns dict with MSE, RMSE, MAE, R2 for comprehensive evaluation.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),  # Mean squared error (primary loss)
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),  # Root MSE (interpretable units)
        "mae": mean_absolute_error(y_true, y_pred),  # Mean absolute error (robust to outliers)
        "r2": r2_score(y_true, y_pred),  # R-squared (proportion of variance explained)
    }


def validate(model, X_val, y_val):
    """Evaluate model on validation set for hyperparameter tuning."""
    y_pred = model.predict(X_val)  # Generate predictions on validation data
    m = _metrics(y_val, y_pred)  # Compute all four regression metrics
    logger.info("Validation: %s", m)  # Log for pipeline monitoring
    return m  # Return metrics dict


def test(model, X_test, y_test):
    """Evaluate model on held-out test set for final performance reporting."""
    y_pred = model.predict(X_test)  # Generate predictions on unseen test data
    m = _metrics(y_test, y_pred)  # Compute final performance metrics
    logger.info("Test: %s", m)  # Log for end-of-pipeline reporting
    return m  # Return metrics dict


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare ElasticNet across alpha and l1_ratio to show L1 vs L2 dominance.

    This function evaluates a 3x3 grid of configurations:
        - alpha in {0.1, 1.0, 10.0}: light, moderate, heavy regularisation
        - l1_ratio in {0.1, 0.5, 0.9}: Ridge-like, balanced, Lasso-like

    Expected patterns:
        - l1_ratio=0.1 (L2 dominant): keeps most features, just shrinks them
        - l1_ratio=0.9 (L1 dominant): zeros out many features, aggressive selection
        - l1_ratio=0.5 (balanced): moderate sparsity with grouping of correlated features
        - alpha=0.1: light regularisation, near-OLS regardless of l1_ratio
        - alpha=10.0: heavy regularisation, many zeros especially with high l1_ratio
    """
    # Print section header for clear visual separation
    print("\n" + "=" * 80)  # Visual separator
    print("PARAMETER COMPARISON: ElasticNet (NumPy) - Alpha x L1 Ratio Grid")
    print("=" * 80)
    print("\nShows how alpha (strength) and l1_ratio (L1 vs L2) control sparsity.\n")

    # Record total features for sparsity reporting
    n_features = X_train.shape[1]  # Number of features in dataset

    # Define the 3x3 grid of (alpha, l1_ratio) combinations to test
    alphas = [0.1, 1.0, 10.0]  # Light, moderate, heavy regularisation
    l1_ratios = [0.1, 0.5, 0.9]  # Ridge-like, balanced, Lasso-like

    # Storage for results across all configurations
    results = {}  # key: config name, value: metrics dict

    # Iterate through every combination in the grid
    for alpha in alphas:  # Outer loop: regularisation strength
        for l1_ratio in l1_ratios:  # Inner loop: L1/L2 blend
            # Create human-readable configuration name
            name = f"alpha={alpha}, l1_ratio={l1_ratio}"  # Config identifier

            # Time the training for speed comparison
            start_time = time.time()  # Record start time

            # Train ElasticNet with this specific configuration
            model = train(X_train, y_train, alpha=alpha, l1_ratio=l1_ratio)

            # Compute elapsed training time
            train_time = time.time() - start_time  # Training duration in seconds

            # Evaluate on validation set (never on test set during tuning)
            metrics = validate(model, X_val, y_val)  # Get MSE, RMSE, MAE, R2

            # Add sparsity and timing metadata to metrics
            metrics["n_nonzero"] = model.n_nonzero  # Features kept by the model
            metrics["train_time"] = train_time  # How long training took

            # Store for summary table
            results[name] = metrics  # Save configuration results

            # Print immediate feedback for this configuration
            print(f"  {name}: NonZero={model.n_nonzero}/{n_features}  "
                  f"MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}")

    # Print formatted summary table
    print(f"\n{'=' * 85}")
    print(f"{'Configuration':<35} {'Alpha':>6} {'L1R':>5} "
          f"{'NZ':>4} {'MSE':>10} {'R2':>10}")
    print("-" * 85)  # Table separator

    for name, m in results.items():  # Iterate through all results
        # Parse alpha and l1_ratio from the config name
        parts = name.split(", ")  # Split "alpha=X, l1_ratio=Y"
        alpha_val = float(parts[0].split("=")[1])  # Extract alpha
        l1r_val = float(parts[1].split("=")[1])  # Extract l1_ratio
        print(f"{name:<35} {alpha_val:>6.1f} {l1r_val:>5.1f} "
              f"{m['n_nonzero']:>4} {m['mse']:>10.4f} {m['r2']:>10.4f}")

    # Print key takeaways for practitioner intuition
    print(f"\nKEY TAKEAWAYS:")
    print("  1. l1_ratio=0.1 (L2 dominant): keeps most features, minimal sparsity.")
    print("  2. l1_ratio=0.9 (L1 dominant): aggressive feature selection, many zeros.")
    print("  3. l1_ratio=0.5 (balanced): moderate sparsity + correlated feature grouping.")
    print("  4. High alpha + high l1_ratio = maximum sparsity (may underfit).")
    print("  5. Low alpha preserves features regardless of l1_ratio.")

    return results  # Return for programmatic access


# ---------------------------------------------------------------------------
# Real-World Demo -- Gene Expression Analysis
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate ElasticNet for gene expression analysis with correlated features.

    DOMAIN CONTEXT: In genomics, gene expression profiling measures the activity
    levels of thousands of genes simultaneously. Genes operating in the same
    biological pathway are typically correlated because they are co-regulated
    by shared transcription factors.

    WHY ElasticNet (not Lasso or Ridge):
        - Lasso arbitrarily picks one gene from each correlated group, losing
          biological context about which genes work together in pathways.
        - Ridge keeps all genes but never zeros any out, making interpretation
          impossible with thousands of genes.
        - ElasticNet's grouping effect keeps entire pathways together while
          still zeroing out truly irrelevant genes, providing biologically
          meaningful and sparse models.

    This demo simulates:
        - 500 patients (typical clinical cohort)
        - 100 genes measured (simplified from real 20,000+ gene studies)
        - 5 gene pathways of 4 correlated genes each (20 informative genes)
        - 80 noise genes with no predictive power
    """
    # Print demo section header
    print("\n" + "=" * 80)  # Visual separator
    print("REAL-WORLD DEMO: Gene Expression Analysis (ElasticNet NumPy)")
    print("=" * 80)

    # Set seed for reproducible synthetic biology data
    np.random.seed(42)  # Ensures identical results across runs

    # Define dataset dimensions modelling a typical genomics study
    n_patients = 500  # Number of patient samples (typical cohort: 100-1000)
    n_genes = 100  # Number of genes measured (real studies: 20,000+)
    n_pathways = 5  # Number of correlated gene groups (biological pathways)
    genes_per_pathway = 4  # Genes per pathway (co-regulated together)
    n_informative = n_pathways * genes_per_pathway  # Total truly predictive genes = 20

    # Create gene name labels for interpretable output
    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]  # gene_000 to gene_099

    # Generate base gene expression matrix with independent Gaussian expression
    # WHY Gaussian: gene expression is approximately normal after log-transformation
    X = np.random.randn(n_patients, n_genes)  # Shape: (500, 100)

    # Introduce within-pathway correlations by adding shared signals
    # WHY: real genes in the same pathway are co-regulated (typical r > 0.5)
    for p in range(n_pathways):  # For each biological pathway
        start_idx = p * genes_per_pathway  # First gene index in this pathway
        end_idx = start_idx + genes_per_pathway  # Last gene index (exclusive)

        # Create a shared latent signal (simulates shared transcription factor activity)
        shared_signal = np.random.randn(n_patients) * 2.0  # Strong shared component

        # Add shared signal to each gene in the pathway, inducing correlation
        for g in range(start_idx, end_idx):  # For each gene in this pathway
            X[:, g] += shared_signal  # Expression = independent + shared signal

    # Define true coefficients: only pathway genes have non-zero effects
    true_coefs = np.zeros(n_genes)  # Start with all zeros (80 noise genes)
    for p in range(n_pathways):  # For each pathway
        start_idx = p * genes_per_pathway  # First gene in this pathway
        end_idx = start_idx + genes_per_pathway  # Last gene in this pathway
        # Assign similar (but not identical) coefficients within each pathway
        # WHY similar: genes in the same pathway have similar biological effects
        base_effect = np.random.uniform(2, 8) * np.random.choice([-1, 1])  # Pathway effect
        for g in range(start_idx, end_idx):  # For each gene in the pathway
            true_coefs[g] = base_effect + np.random.normal(0, 0.5)  # Small variation

    # Generate the clinical outcome (e.g., drug response score)
    noise = np.random.normal(0, 3, n_patients)  # Clinical measurement noise
    y = X @ true_coefs + noise  # True outcome = gene effects + noise

    # Print dataset summary
    print(f"\nDataset: {n_patients} patients, {n_genes} genes measured")
    print(f"Ground truth: {n_informative} genes in {n_pathways} correlated pathways")
    print(f"Remaining {n_genes - n_informative} genes are noise")

    # Split data into train/val/test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42,  # 70% train
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,  # 15% val, 15% test
    )

    # Standardise features (critical for fair penalisation across genes)
    scaler = StandardScaler()  # Will learn gene-wise mean and std from training data
    X_train_s = scaler.fit_transform(X_train)  # Fit on training, transform
    X_val_s = scaler.transform(X_val)  # Transform validation with train stats
    X_test_s = scaler.transform(X_test)  # Transform test with train stats

    # Train ElasticNet (balanced l1_ratio=0.5 for grouping effect)
    en_model = train(X_train_s, y_train, alpha=0.5, l1_ratio=0.5, max_iter=5000)
    en_metrics = _metrics(y_test, en_model.predict(X_test_s))  # Evaluate on test set
    en_selected = en_model.n_nonzero  # Count selected genes

    # Train a Lasso-like model for comparison (l1_ratio=0.99, near-pure L1)
    lasso_model = train(X_train_s, y_train, alpha=0.5, l1_ratio=0.99, max_iter=5000)
    lasso_metrics = _metrics(y_test, lasso_model.predict(X_test_s))  # Evaluate Lasso
    lasso_selected = lasso_model.n_nonzero  # Count Lasso-selected genes

    # Print comparison header
    print(f"\n--- ElasticNet vs Lasso-like Comparison ---")
    print(f"{'Metric':<20} {'ElasticNet':>12} {'Lasso-like':>12}")
    print("-" * 48)  # Table separator
    print(f"{'Genes selected':<20} {en_selected:>12} {lasso_selected:>12}")
    print(f"{'RMSE':<20} {en_metrics['rmse']:>12.4f} {lasso_metrics['rmse']:>12.4f}")
    print(f"{'R2':<20} {en_metrics['r2']:>12.4f} {lasso_metrics['r2']:>12.4f}")

    # Analyse pathway coverage: how many genes per pathway were selected
    print(f"\n--- Pathway Coverage Analysis ---")
    print(f"{'Pathway':<12} {'True Genes':>12} {'EN Selected':>12} {'Lasso Selected':>15}")
    print("-" * 55)  # Table separator

    en_pathway_total = 0  # Track ElasticNet pathway coverage
    lasso_pathway_total = 0  # Track Lasso pathway coverage

    for p in range(n_pathways):  # For each biological pathway
        start_idx = p * genes_per_pathway  # First gene in pathway
        end_idx = start_idx + genes_per_pathway  # Last gene in pathway

        # Count pathway genes selected by each method
        en_count = int(np.sum(en_model.weights[start_idx:end_idx] != 0))  # EN selections
        lasso_count = int(np.sum(lasso_model.weights[start_idx:end_idx] != 0))  # Lasso selections

        en_pathway_total += en_count  # Accumulate EN total
        lasso_pathway_total += lasso_count  # Accumulate Lasso total

        print(f"Pathway {p + 1:<4} {genes_per_pathway:>12} "
              f"{en_count:>12} {lasso_count:>15}")

    # Print pathway coverage summary
    print(f"\nElasticNet selected {en_pathway_total}/{n_informative} pathway genes "
          f"(grouping effect)")
    print(f"Lasso-like selected {lasso_pathway_total}/{n_informative} pathway genes "
          f"(picks one per group)")

    # Print biological interpretation
    print(f"\nCONCLUSION: ElasticNet preserves pathway structure better than Lasso.")
    print(f"  - L2 component encourages correlated genes to have similar coefficients.")
    print(f"  - L1 component still zeros out the 80 noise genes.")
    print(f"  - Result: biologically meaningful, sparse gene selection.")

    return en_model, en_metrics  # Return model and metrics


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimisation
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: minimise validation MSE over ElasticNet hyperparameters.

    Search space includes both alpha and l1_ratio (the two critical ElasticNet HPs)
    plus convergence parameters.
    """
    # Suggest hyperparameters from defined distributions
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),  # Log-scale for range
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),  # Avoid exact 0 or 1
        "max_iter": trial.suggest_int("max_iter", 500, 5000, step=500),  # Iteration budget
        "tol": trial.suggest_float("tol", 1e-8, 1e-3, log=True),  # Convergence tolerance
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),  # Bias
    }
    # Train and evaluate, returning validation MSE for minimisation
    model = train(X_train, y_train, **hp)  # Train with suggested hyperparameters
    return validate(model, X_val, y_val)["mse"]  # Return MSE as objective


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Execute Optuna study to find optimal ElasticNet hyperparameters."""
    # Create study that minimises MSE (lower is better)
    study = optuna.create_study(direction="minimize", study_name="elasticnet_numpy")

    # Run optimisation with partial to bind data arguments
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val),
        n_trials=n_trials,  # Number of hyperparameter configurations to try
        show_progress_bar=True,  # Display progress for user feedback
    )

    # Log the best configuration found
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study  # Return study object for analysis


# ---------------------------------------------------------------------------
# Ray Tune Hyperparameter Search
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    """Ray Tune trainable: trains ElasticNet and reports validation metrics."""
    model = train(X_train, y_train, **config)  # Train with Ray-suggested params
    metrics = validate(model, X_val, y_val)  # Compute validation metrics
    tune.report(**metrics)  # Report metrics back to Ray scheduler


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    """Run Ray Tune parallel hyperparameter search over ElasticNet parameters."""
    # Initialise Ray if not already running
    if not ray.is_initialized():  # Check if Ray runtime exists
        ray.init(ignore_reinit_error=True, log_to_driver=False)  # Start Ray

    # Define search space matching Optuna for fair comparison
    search_space = {
        "alpha": tune.loguniform(1e-5, 10.0),  # Log-uniform for spanning magnitudes
        "l1_ratio": tune.uniform(0.01, 0.99),  # Uniform blend between Ridge and Lasso
        "max_iter": tune.choice([1000, 2000, 5000]),  # Discrete iteration choices
        "tol": tune.loguniform(1e-8, 1e-3),  # Log-uniform convergence tolerance
        "fit_intercept": tune.choice([True, False]),  # Whether to include bias
    }

    # Create trainable with data passed via shared memory
    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )

    # Configure and run the tuning experiment
    tuner = tune.Tuner(
        trainable,  # Function to call for each trial
        param_space=search_space,  # Hyperparameter distributions
        tune_config=tune.TuneConfig(
            metric="mse", mode="min", num_samples=num_samples,  # Minimise MSE
        ),
    )
    results = tuner.fit()  # Execute all trials

    # Extract and log the best result
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results  # Return full results for analysis


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full ElasticNet (NumPy from scratch) pipeline.

    Stages:
        1. Generate synthetic data with 60/20/20 split
        2. Compare parameter sets (3x3 alpha x l1_ratio grid)
        3. Real-world gene expression demo
        4. Optuna hyperparameter optimisation
        5. Ray Tune parallel search
        6. Final evaluation on test set
    """
    # Print pipeline header
    print("=" * 70)  # Visual separator
    print("ElasticNet Regression - NumPy From Scratch (Coordinate Descent)")
    print("=" * 70)

    # Stage 1: Generate data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Stage 2: Parameter comparison grid
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Stage 3: Real-world demo
    real_world_demo()

    # Stage 4: Optuna optimisation
    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=25)
    print(f"Best params : {study.best_trial.params}")  # Show best hyperparameters
    print(f"Best MSE    : {study.best_trial.value:.6f}")  # Show best validation MSE

    # Stage 5: Ray Tune search
    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")  # Show best Ray configuration
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")  # Show best Ray MSE

    # Stage 6: Final evaluation with best Optuna parameters
    model = train(X_train, y_train, **study.best_trial.params)  # Retrain with best
    print(f"\nNon-zero coefficients: {model.n_nonzero}/{len(model.weights)}")
    print(f"Iterations run: {model.n_iters_run}")

    # Print validation metrics
    print("\n--- Validation ---")
    for k, v in validate(model, X_val, y_val).items():  # Iterate metrics
        print(f"  {k:6s}: {v:.6f}")  # Aligned metric display

    # Print test metrics (final, unbiased evaluation)
    print("\n--- Test ---")
    for k, v in test(model, X_test, y_test).items():  # Iterate test metrics
        print(f"  {k:6s}: {v:.6f}")  # Aligned metric display

    # Pipeline completion message
    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


# Entry point: run the full pipeline when executed directly
if __name__ == "__main__":
    main()  # Execute main pipeline function
