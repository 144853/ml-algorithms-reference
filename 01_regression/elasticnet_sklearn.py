"""
ElasticNet Regression - scikit-learn Implementation
====================================================

COMPLETE ML TUTORIAL: This file teaches ElasticNet regularisation using
scikit-learn. ElasticNet combines L1 (Lasso) and L2 (Ridge) penalties,
giving you BOTH feature selection AND coefficient shrinkage in one model.

Theory & Mathematics:
    ElasticNet minimises the following objective:

        L(w) = (1/2n) * ||y - Xw||^2
             + alpha * l1_ratio * ||w||_1
             + alpha * (1 - l1_ratio) * (1/2) * ||w||_2^2

    where:
        - ||w||_1 = sum(|w_j|)       is the L1 norm (sparsity inducer)
        - ||w||_2^2 = sum(w_j^2)     is the squared L2 norm (shrinkage)
        - alpha controls overall regularisation strength
        - l1_ratio in [0, 1] blends between Ridge (0) and Lasso (1)

    Why combine L1 and L2?
        - Pure Lasso (L1) struggles with groups of correlated features:
          it arbitrarily picks one and zeros out the rest.
        - Pure Ridge (L2) keeps all features, never zeroing any out.
        - ElasticNet gets the best of both: it selects groups of correlated
          features together (L2 grouping effect) while still achieving
          sparsity (L1 zeroing effect).

    The "grouping effect" theorem guarantees that for highly correlated
    features x_i and x_j, their ElasticNet coefficients will be similar
    in magnitude, unlike Lasso which picks one arbitrarily.

    scikit-learn solves this via cyclic coordinate descent, where each
    coordinate update applies both soft-thresholding (L1) and shrinkage (L2).

Hyperparameters:
    - alpha (float): Overall regularisation strength. Default 1.0.
    - l1_ratio (float): Mix between L1 and L2. 0=Ridge, 1=Lasso. Default 0.5.
    - max_iter (int): Maximum coordinate descent iterations. Default 1000.
    - tol (float): Convergence tolerance for the dual gap. Default 1e-4.
    - fit_intercept (bool): Whether to fit intercept term. Default True.
    - selection (str): "cyclic" or "random" coordinate update order. Default "cyclic".

Business Use Cases:
    - Genomics: Predicting phenotypes from gene expression where genes form
      correlated groups (pathways). ElasticNet selects entire pathways.
    - Finance: Multi-factor asset pricing with correlated market factors.
    - NLP: Text regression with correlated n-gram features.
    - Healthcare: Predicting patient outcomes from correlated clinical features.

Advantages:
    - Handles correlated features better than Lasso alone.
    - Still produces sparse models (unlike pure Ridge).
    - Two-parameter control (alpha, l1_ratio) is very flexible.
    - Convex optimisation guarantees a global minimum.

Disadvantages:
    - Two hyperparameters to tune instead of one (alpha AND l1_ratio).
    - Coordinate descent can be slow for very large feature spaces.
    - Not suitable for non-linear relationships without feature engineering.
"""

# -- Standard library imports for logging, timing, and partial application --
import logging  # Structured logging for tracking pipeline progress across stages
import time  # Wall-clock timing for comparing training speeds across configurations
from functools import partial  # Binds fixed arguments to functions for Optuna callbacks

# -- Third-party numerical and ML imports --
import numpy as np  # Core array operations for data manipulation and metric computation
import optuna  # Bayesian hyperparameter optimisation with Tree-structured Parzen Estimators
import ray  # Distributed computing framework for parallelising hyperparameter search
from ray import tune  # Ray's hyperparameter tuning module with scheduling algorithms
from sklearn.datasets import make_regression  # Generates synthetic regression data with known structure
from sklearn.linear_model import ElasticNet  # scikit-learn's optimised ElasticNet implementation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Standard regression metrics
from sklearn.model_selection import train_test_split  # Splits data into train/val/test with reproducibility
from sklearn.preprocessing import StandardScaler  # Zero-mean unit-variance normalisation for fair penalisation

# -- Configure module-level logging for consistent pipeline output --
# WHY this format: timestamps help correlate events, level helps filter severity
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Module-scoped logger for all functions in this file


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=20, noise=0.1, random_state=42):
    """Generate synthetic regression data with a 60/20/20 train/val/test split.

    WHY 20 features: ElasticNet shines when there are many features, some correlated.
    make_regression creates n_informative truly useful features and the rest are noise,
    which lets us evaluate the model's ability to identify the useful ones.

    WHY StandardScaler: ElasticNet penalises coefficients by magnitude. Without
    scaling, features with larger numeric ranges would be penalised less, creating
    an unfair bias toward high-magnitude features.
    """
    # Generate synthetic data where only half the features are truly informative
    # WHY n_informative = n_features // 2: creates a realistic scenario where
    # some features are noise and the model must identify the useful ones
    X, y = make_regression(
        n_samples=n_samples,  # Total number of data points to generate
        n_features=n_features,  # Total feature dimensionality (informative + noise)
        n_informative=max(1, n_features // 2),  # Only half of features carry signal
        noise=noise,  # Gaussian noise added to target values for realism
        random_state=random_state,  # Seed for reproducible data generation
    )

    # Split into 60% train, 20% validation, 20% test
    # WHY two-stage split: first extract 40% for temp, then split temp 50/50
    # This ensures exact 60/20/20 proportions for fair evaluation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,  # Full dataset to be partitioned
        test_size=0.4,  # Reserve 40% for validation + test combined
        random_state=random_state,  # Same seed ensures reproducible splits
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,  # The 40% temporary set
        test_size=0.5,  # Split evenly into 20% val and 20% test
        random_state=random_state,  # Consistent split across runs
    )

    # Standardise features: fit on train only, transform val/test with train statistics
    # WHY fit only on train: prevents data leakage from validation/test distributions
    scaler = StandardScaler()  # Will compute mean and std from training data only
    X_train = scaler.fit_transform(X_train)  # Learn mean/std and transform in one step
    X_val = scaler.transform(X_val)  # Apply training mean/std to validation data
    X_test = scaler.transform(X_test)  # Apply training mean/std to test data

    # Log dataset summary for pipeline visibility and debugging
    logger.info(
        "Data: train=%d val=%d test=%d features=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1],
    )
    return X_train, X_val, X_test, y_train, y_val, y_test  # Return all six arrays


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train an sklearn ElasticNet model with the given hyperparameters.

    sklearn's ElasticNet uses optimised Cython coordinate descent that cycles
    through features, applying both soft-thresholding (L1) and shrinkage (L2)
    at each coordinate update. This is much faster than our NumPy implementation.

    Key hyperparameters:
        alpha: Overall regularisation strength. Higher values push more
               coefficients toward zero. THE most important hyperparameter.
        l1_ratio: Blend between L1 and L2.
                  l1_ratio=1.0 is pure Lasso (maximum sparsity).
                  l1_ratio=0.0 is pure Ridge (no sparsity).
                  l1_ratio=0.5 is equal blend (ElasticNet sweet spot).
    """
    # Extract hyperparameters with sensible defaults for ElasticNet
    # WHY alpha=1.0: sklearn's default; moderate regularisation strength
    alpha = hp.get("alpha", 1.0)  # Controls overall penalty magnitude

    # WHY l1_ratio=0.5: balanced blend of L1 and L2, the core ElasticNet advantage
    l1_ratio = hp.get("l1_ratio", 0.5)  # 0=Ridge, 1=Lasso, 0.5=balanced ElasticNet

    # WHY max_iter=1000: sufficient for most problems; increase for large alpha
    max_iter = hp.get("max_iter", 1000)  # Maximum coordinate descent sweeps

    # WHY tol=1e-4: convergence threshold for the dual gap optimality condition
    tol = hp.get("tol", 1e-4)  # Stops when optimality gap is below this

    # WHY fit_intercept=True: intercept captures the mean of y, preventing
    # bias in predictions; should almost always be True
    fit_intercept = hp.get("fit_intercept", True)  # Whether to learn bias term

    # WHY selection="cyclic": deterministic update order; "random" can help
    # with highly correlated features by breaking update-order correlations
    selection = hp.get("selection", "cyclic")  # Feature update order strategy

    # Instantiate the ElasticNet estimator with all specified hyperparameters
    model = ElasticNet(
        alpha=alpha,  # Overall regularisation strength (lambda in some notations)
        l1_ratio=l1_ratio,  # Blend parameter between L1 and L2 penalties
        max_iter=max_iter,  # Maximum iterations before forced stop
        tol=tol,  # Convergence tolerance for dual gap
        fit_intercept=fit_intercept,  # Whether to include bias/intercept term
        selection=selection,  # Coordinate update order: cyclic or random
    )

    # Fit the model using coordinate descent on the training data
    # WHY fit on train only: prevents data leakage from val/test sets
    model.fit(X_train, y_train)  # Runs coordinate descent until convergence or max_iter

    # Count non-zero coefficients to measure sparsity (L1 feature selection effect)
    # WHY track this: sparsity is ElasticNet's key advantage over Ridge
    n_nonzero = np.sum(model.coef_ != 0)  # Number of features kept by the model

    # Log training summary for debugging and pipeline monitoring
    logger.info(
        "ElasticNet trained  alpha=%.4f  l1_ratio=%.2f  nonzero=%d/%d",
        alpha, l1_ratio, n_nonzero, len(model.coef_),
    )
    return model  # Return the fitted estimator for validation and testing


def _metrics(y_true, y_pred):
    """Compute standard regression metrics for model evaluation.

    WHY these four metrics:
        - MSE: primary loss being optimised (mean squared error)
        - RMSE: same units as target variable, more interpretable than MSE
        - MAE: robust to outliers, gives median-like error estimate
        - R2: proportion of variance explained, comparable across datasets
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),  # Average squared prediction error
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),  # Root of MSE for interpretability
        "mae": mean_absolute_error(y_true, y_pred),  # Average absolute prediction error
        "r2": r2_score(y_true, y_pred),  # 1.0 = perfect, 0.0 = predicts mean, <0 = worse than mean
    }


def validate(model, X_val, y_val):
    """Evaluate the trained model on the validation set.

    WHY separate validation: validation metrics guide hyperparameter tuning
    without touching the test set, preventing optimistic bias in final results.
    """
    # Generate predictions on validation data using the trained model
    y_pred = model.predict(X_val)  # Uses learned weights to predict target values
    m = _metrics(y_val, y_pred)  # Compute all four regression metrics
    logger.info("Validation: %s", m)  # Log for pipeline monitoring
    return m  # Return metrics dict for Optuna/Ray objective functions


def test(model, X_test, y_test):
    """Evaluate the trained model on the held-out test set.

    WHY separate test: the test set is only used ONCE at the end for final
    reporting, ensuring an unbiased estimate of generalisation performance.
    """
    # Generate predictions on test data (never seen during training or tuning)
    y_pred = model.predict(X_test)  # Final predictions on held-out data
    m = _metrics(y_test, y_pred)  # Compute final performance metrics
    logger.info("Test: %s", m)  # Log for end-of-pipeline reporting
    return m  # Return metrics for the main function to display


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare ElasticNet across alpha and l1_ratio configurations.

    This function systematically evaluates how alpha (regularisation strength)
    and l1_ratio (L1/L2 blend) affect model performance and sparsity.

    Configurations tested:
        - alpha in {0.1, 1.0, 10.0}: from light to heavy regularisation
        - l1_ratio in {0.1, 0.5, 0.9}: from Ridge-like to Lasso-like
        Total: 9 combinations (3 x 3 grid)

    Expected patterns:
        - Low alpha + any l1_ratio: near-OLS performance, little sparsity
        - High alpha + high l1_ratio: maximum sparsity, potential underfitting
        - Moderate alpha + moderate l1_ratio: balanced (ElasticNet sweet spot)
    """
    # Print header for clear visual separation in terminal output
    print("\n" + "=" * 80)  # Visual separator for readability
    print("PARAMETER COMPARISON: sklearn ElasticNet - Alpha x L1 Ratio Grid")
    print("=" * 80)

    # Store the total feature count for sparsity reporting
    n_features = X_train.shape[1]  # Number of features in the dataset

    # Define the 3x3 grid of alpha and l1_ratio values to test
    # WHY these specific values: they span the full range of ElasticNet behaviour
    alphas = [0.1, 1.0, 10.0]  # Light, moderate, and heavy regularisation
    l1_ratios = [0.1, 0.5, 0.9]  # Ridge-like, balanced, and Lasso-like blends

    # Dictionary to accumulate results for all configurations
    results = {}  # key: config name, value: metrics dict

    # Iterate through every combination of alpha and l1_ratio
    for alpha in alphas:  # Outer loop: regularisation strength
        for l1_ratio in l1_ratios:  # Inner loop: L1/L2 blend ratio
            # Create a descriptive name for this configuration
            name = f"alpha={alpha}, l1_ratio={l1_ratio}"  # Human-readable config identifier

            # Time the training to compare computational costs
            start_time = time.time()  # Record wall-clock start time

            # Train ElasticNet with this specific alpha and l1_ratio combination
            model = train(X_train, y_train, alpha=alpha, l1_ratio=l1_ratio)

            # Compute training duration in seconds
            train_time = time.time() - start_time  # Elapsed wall-clock time

            # Evaluate on validation set (not test set, to avoid data snooping)
            metrics = validate(model, X_val, y_val)  # Get MSE, RMSE, MAE, R2

            # Count non-zero coefficients to measure sparsity
            n_nz = int(np.sum(model.coef_ != 0))  # Features selected by the model

            # Add sparsity and timing info to the metrics dictionary
            metrics["n_nonzero"] = n_nz  # Track how many features were kept
            metrics["train_time"] = train_time  # Track training speed

            # Store results for the summary table
            results[name] = metrics  # Save for later comparison

            # Print individual result for immediate feedback
            print(f"\n  {name}: NonZero={n_nz}/{n_features}  "
                  f"MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}")

    # Print formatted summary table for easy comparison across all 9 configs
    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<35} {'Alpha':>6} {'L1R':>5} "
          f"{'NZ':>4} {'MSE':>10} {'R2':>10}")
    print("-" * 75)  # Table separator line

    # Iterate through results in insertion order to print the summary
    for name, m in results.items():
        # Parse alpha and l1_ratio from the config name for display
        parts = name.split(", ")  # Split "alpha=X, l1_ratio=Y"
        alpha_val = float(parts[0].split("=")[1])  # Extract alpha value
        l1r_val = float(parts[1].split("=")[1])  # Extract l1_ratio value
        print(f"{name:<35} {alpha_val:>6.1f} {l1r_val:>5.1f} "
              f"{m['n_nonzero']:>4} {m['mse']:>10.4f} {m['r2']:>10.4f}")

    # Print key takeaways to help practitioners build intuition
    print(f"\nKEY TAKEAWAYS:")
    print("  1. Low alpha (0.1) preserves most features regardless of l1_ratio.")
    print("  2. High alpha (10.0) with high l1_ratio (0.9) produces maximum sparsity.")
    print("  3. l1_ratio=0.5 gives the classic ElasticNet balance of selection + shrinkage.")
    print("  4. l1_ratio=0.1 behaves like Ridge (keeps all features, just shrinks them).")
    print("  5. The best config maximises R2 while keeping the model sparse.")

    return results  # Return for programmatic access


# ---------------------------------------------------------------------------
# Real-World Demo -- Gene Expression Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate ElasticNet for gene expression prediction.

    DOMAIN CONTEXT: In genomics, researchers measure expression levels of
    thousands of genes simultaneously using microarrays or RNA-seq. The goal
    is to predict a clinical outcome (e.g., drug response, disease severity)
    from gene expression profiles.

    WHY ElasticNet (not Lasso):
        Genes operate in PATHWAYS -- groups of correlated genes that work
        together. Pure Lasso arbitrarily picks one gene from each correlated
        group and zeros out the rest, losing biological context. ElasticNet's
        grouping effect keeps correlated genes together, which:
        1. Improves prediction stability across different patient cohorts
        2. Preserves pathway-level interpretability for biologists
        3. Handles the "p >> n" problem (more genes than patients)

    This demo simulates:
        - 500 patients (typical clinical cohort size)
        - 100 genes measured (simplified; real studies measure 20,000+)
        - 5 gene pathways of 4 correlated genes each (20 informative genes)
        - 80 noise genes with no predictive power
    """
    # Print header for the real-world demo section
    print("\n" + "=" * 80)  # Visual separator for demo section
    print("REAL-WORLD DEMO: Gene Expression Prediction (ElasticNet vs Lasso)")
    print("=" * 80)

    # Set random seed for reproducible synthetic biology data
    np.random.seed(42)  # Ensures identical results across runs

    # Define dataset dimensions mimicking a real genomics study
    n_patients = 500  # Typical clinical cohort size (often 100-1000)
    n_genes = 100  # Simplified from real genomics (20,000+ genes in practice)
    n_pathways = 5  # Number of correlated gene groups (biological pathways)
    genes_per_pathway = 4  # Genes in each pathway that are correlated with each other
    n_informative = n_pathways * genes_per_pathway  # Total truly predictive genes = 20

    # Create gene names for interpretable output
    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]  # gene_000 through gene_099

    # Generate base expression matrix: each gene has independent Gaussian expression
    # WHY Gaussian: gene expression data is approximately normal after log-transformation
    X = np.random.randn(n_patients, n_genes)  # Shape: (500, 100)

    # Introduce within-pathway correlations by adding shared pathway signals
    # WHY correlations: real genes in the same pathway co-express (r > 0.5 is common)
    for p in range(n_pathways):  # For each of the 5 biological pathways
        # Determine which genes belong to this pathway
        start_idx = p * genes_per_pathway  # First gene index in this pathway
        end_idx = start_idx + genes_per_pathway  # Last gene index (exclusive)

        # Create a shared latent signal for this pathway (simulates co-regulation)
        # WHY shared signal: genes in the same pathway are regulated by common
        # transcription factors, causing their expression levels to be correlated
        shared_signal = np.random.randn(n_patients) * 2.0  # Strong shared component

        # Add the shared signal to each gene in the pathway, creating correlation
        for g in range(start_idx, end_idx):  # For each gene in this pathway
            X[:, g] += shared_signal  # Gene expression = independent + shared component

    # Create true coefficients: only pathway genes have non-zero effects
    true_coefs = np.zeros(n_genes)  # Start with all zeros (no effect)
    for p in range(n_pathways):  # For each pathway
        start_idx = p * genes_per_pathway  # First gene in this pathway
        end_idx = start_idx + genes_per_pathway  # Last gene in this pathway
        # Assign similar but not identical coefficients within each pathway
        # WHY similar: genes in the same pathway have similar biological effects
        base_effect = np.random.uniform(2, 8) * np.random.choice([-1, 1])  # Pathway-level effect
        for g in range(start_idx, end_idx):  # For each gene in the pathway
            true_coefs[g] = base_effect + np.random.normal(0, 0.5)  # Add small within-pathway variation

    # Generate the clinical outcome (e.g., drug response score)
    noise = np.random.normal(0, 3, n_patients)  # Measurement noise in clinical outcome
    y = X @ true_coefs + noise  # True response = gene effects + noise

    # Print dataset summary
    print(f"\nDataset: {n_patients} patients, {n_genes} genes measured")
    print(f"Ground truth: {n_informative} genes in {n_pathways} correlated "
          f"pathways affect outcome")
    print(f"Remaining {n_genes - n_informative} genes are noise")

    # Split data into train/val/test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42,  # 70% train, 30% temp
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,  # 15% val, 15% test
    )

    # Standardise features (critical for fair regularisation penalisation)
    scaler = StandardScaler()  # Will learn mean/std from training data only
    X_train_s = scaler.fit_transform(X_train)  # Fit and transform training data
    X_val_s = scaler.transform(X_val)  # Transform validation with train statistics
    X_test_s = scaler.transform(X_test)  # Transform test with train statistics

    # --- Compare ElasticNet vs Lasso on correlated gene data ---
    # WHY this comparison: demonstrates ElasticNet's grouping advantage
    from sklearn.linear_model import Lasso  # Import Lasso for direct comparison

    # Train ElasticNet (l1_ratio=0.5 for balanced L1+L2)
    en_model = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, fit_intercept=True)
    en_model.fit(X_train_s, y_train)  # Fit on standardised training data
    en_metrics = _metrics(y_test, en_model.predict(X_test_s))  # Evaluate on test set
    en_selected = int(np.sum(en_model.coef_ != 0))  # Count selected genes

    # Train Lasso for comparison (l1_ratio=1.0 effectively)
    lasso_model = Lasso(alpha=0.5, max_iter=5000, fit_intercept=True)
    lasso_model.fit(X_train_s, y_train)  # Fit on same standardised training data
    lasso_metrics = _metrics(y_test, lasso_model.predict(X_test_s))  # Evaluate on test set
    lasso_selected = int(np.sum(lasso_model.coef_ != 0))  # Count selected genes

    # Print comparison header
    print(f"\n--- ElasticNet vs Lasso Comparison ---")
    print(f"{'Metric':<20} {'ElasticNet':>12} {'Lasso':>12}")
    print("-" * 48)  # Table separator
    print(f"{'Genes selected':<20} {en_selected:>12} {lasso_selected:>12}")
    print(f"{'RMSE':<20} {en_metrics['rmse']:>12.4f} {lasso_metrics['rmse']:>12.4f}")
    print(f"{'R2':<20} {en_metrics['r2']:>12.4f} {lasso_metrics['r2']:>12.4f}")

    # Analyse pathway coverage: how many genes per pathway were selected?
    # WHY: This demonstrates ElasticNet's grouping effect vs Lasso's arbitrary selection
    print(f"\n--- Pathway Coverage Analysis ---")
    print(f"{'Pathway':<12} {'True Genes':>12} {'EN Selected':>12} {'Lasso Selected':>15}")
    print("-" * 55)  # Table separator

    # Track total pathway genes selected by each method
    en_pathway_total = 0  # Running count of pathway genes selected by ElasticNet
    lasso_pathway_total = 0  # Running count of pathway genes selected by Lasso

    for p in range(n_pathways):  # For each biological pathway
        start_idx = p * genes_per_pathway  # First gene in this pathway
        end_idx = start_idx + genes_per_pathway  # Last gene (exclusive)

        # Count how many genes in this pathway were selected by each method
        en_count = int(np.sum(en_model.coef_[start_idx:end_idx] != 0))  # ElasticNet selections
        lasso_count = int(np.sum(lasso_model.coef_[start_idx:end_idx] != 0))  # Lasso selections

        # Accumulate totals for summary statistics
        en_pathway_total += en_count  # Add to ElasticNet running total
        lasso_pathway_total += lasso_count  # Add to Lasso running total

        # Print per-pathway breakdown
        print(f"Pathway {p+1:<4} {genes_per_pathway:>12} "
              f"{en_count:>12} {lasso_count:>15}")

    # Print summary of pathway coverage
    print(f"\nElasticNet selected {en_pathway_total}/{n_informative} pathway genes "
          f"(grouping effect preserves correlated genes)")
    print(f"Lasso selected {lasso_pathway_total}/{n_informative} pathway genes "
          f"(tends to pick one gene per group)")

    # Print interpretability insight
    print(f"\nCONCLUSION: ElasticNet preserves pathway structure better than Lasso.")
    print(f"  - ElasticNet's L2 component encourages correlated genes to have similar coefficients.")
    print(f"  - Lasso's L1-only penalty arbitrarily picks one gene and zeros the rest.")
    print(f"  - For genomics, ElasticNet gives biologically meaningful gene selections.")

    return en_model, en_metrics  # Return model and metrics for further analysis


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimisation
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function that minimises validation MSE.

    Search space covers both alpha (regularisation strength) and l1_ratio
    (L1/L2 blend), which are the two critical ElasticNet hyperparameters.

    WHY log scale for alpha: regularisation strength varies across orders of
    magnitude (1e-5 to 10), so uniform sampling would waste trials in [1, 10].
    """
    # Suggest alpha on log scale: spans 5 orders of magnitude for thorough search
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),  # Overall penalty strength
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),  # L1/L2 blend (avoid exact 0 or 1)
        "max_iter": trial.suggest_int("max_iter", 500, 5000, step=500),  # Iteration budget
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),  # Convergence tolerance
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),  # Bias term
        "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),  # Update order
    }
    # Train with suggested hyperparameters and return validation MSE for minimisation
    model = train(X_train, y_train, **hp)  # Train model with Optuna-suggested params
    return validate(model, X_val, y_val)["mse"]  # Return MSE as the objective to minimise


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Create and execute an Optuna study to find optimal ElasticNet hyperparameters.

    WHY Optuna: uses TPE (Tree-structured Parzen Estimator) which models the
    objective function and focuses trials on promising regions of the search space.
    """
    # Create study that minimises MSE (lower is better for regression)
    study = optuna.create_study(direction="minimize", study_name="elasticnet_sklearn")

    # Run optimisation with partial to bind data arguments to the objective function
    study.optimize(
        partial(  # Bind data arguments so Optuna only needs to pass the trial object
            optuna_objective,
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        ),
        n_trials=n_trials,  # Total number of hyperparameter configurations to evaluate
        show_progress_bar=True,  # Display progress for user feedback
    )

    # Log the best configuration found across all trials
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study  # Return the study object for analysis


# ---------------------------------------------------------------------------
# Ray Tune Hyperparameter Search
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    """Ray Tune trainable function that trains ElasticNet and reports metrics.

    WHY separate function: Ray Tune requires a callable with a specific signature.
    tune.with_parameters handles passing the data arrays efficiently via shared memory.
    """
    # Train model with the hyperparameter configuration suggested by Ray Tune
    model = train(X_train, y_train, **config)  # Train with Ray-suggested params
    # Evaluate and report metrics back to Ray Tune for tracking and scheduling
    metrics = validate(model, X_val, y_val)  # Compute validation metrics
    tune.report(**metrics)  # Report all metrics to Ray's trial scheduler


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    """Run a Ray Tune hyperparameter search over ElasticNet parameters.

    WHY Ray Tune: can parallelise trials across multiple CPU cores, making
    it faster than sequential Optuna for large search spaces.
    """
    # Initialise Ray runtime if not already running
    # WHY ignore_reinit_error: prevents crash if Ray was already started
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Define search space matching the Optuna search space for fair comparison
    search_space = {
        "alpha": tune.loguniform(1e-5, 10.0),  # Log-uniform for spanning orders of magnitude
        "l1_ratio": tune.uniform(0.01, 0.99),  # Uniform between near-Ridge and near-Lasso
        "max_iter": tune.choice([1000, 2000, 5000]),  # Discrete set of iteration budgets
        "tol": tune.loguniform(1e-6, 1e-2),  # Log-uniform convergence tolerance
        "fit_intercept": tune.choice([True, False]),  # Whether to include bias term
        "selection": tune.choice(["cyclic", "random"]),  # Coordinate update order
    }

    # Create trainable with data passed via Ray's shared memory (tune.with_parameters)
    # WHY with_parameters: avoids serialising large arrays for each trial
    trainable = tune.with_parameters(
        _ray_trainable,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    )

    # Configure and run the tuning experiment
    tuner = tune.Tuner(
        trainable,  # Function to call for each trial
        param_space=search_space,  # Hyperparameter distributions to sample from
        tune_config=tune.TuneConfig(
            metric="mse",  # Optimise for minimum MSE
            mode="min",  # Lower MSE is better
            num_samples=num_samples,  # Total number of trials to run
        ),
    )
    results = tuner.fit()  # Execute all trials

    # Extract and log the best result
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results  # Return full results grid for analysis


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full ElasticNet (sklearn) pipeline.

    Pipeline stages:
        1. Generate synthetic data with train/val/test split
        2. Compare parameter sets (3x3 grid of alpha x l1_ratio)
        3. Run real-world gene expression demo
        4. Optuna hyperparameter optimisation (Bayesian search)
        5. Ray Tune hyperparameter search (parallel random search)
        6. Final evaluation with best hyperparameters on test set
    """
    # Print pipeline header for clear terminal output
    print("=" * 70)  # Visual separator
    print("ElasticNet Regression - scikit-learn")
    print("=" * 70)

    # Stage 1: Generate synthetic regression data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Stage 2: Systematic parameter comparison (alpha x l1_ratio grid)
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Stage 3: Real-world gene expression demo
    real_world_demo()

    # Stage 4: Optuna Bayesian hyperparameter optimisation
    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=25)
    print(f"Best params : {study.best_trial.params}")  # Display best hyperparameters
    print(f"Best MSE    : {study.best_trial.value:.6f}")  # Display best validation MSE

    # Stage 5: Ray Tune parallel hyperparameter search
    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")  # Display best Ray Tune configuration
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")  # Display best Ray Tune MSE

    # Stage 6: Final evaluation with best Optuna parameters on test set
    model = train(X_train, y_train, **study.best_trial.params)  # Retrain with best params
    n_nonzero = np.sum(model.coef_ != 0)  # Count selected features in final model
    print(f"\nNon-zero coefficients: {n_nonzero}/{len(model.coef_)}")

    # Print validation metrics for the final model
    print("\n--- Validation ---")
    for k, v in validate(model, X_val, y_val).items():  # Iterate through metric dict
        print(f"  {k:6s}: {v:.6f}")  # Format each metric with consistent alignment

    # Print test metrics for the final model (reported only once)
    print("\n--- Test ---")
    for k, v in test(model, X_test, y_test).items():  # Iterate through test metrics
        print(f"  {k:6s}: {v:.6f}")  # Format with same alignment as validation

    # Print pipeline completion message
    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


# Entry point: run the full pipeline when this script is executed directly
if __name__ == "__main__":
    main()  # Execute the main pipeline function
