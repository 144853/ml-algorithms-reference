"""
Lasso Regression - NumPy From-Scratch Implementation (Coordinate Descent)
=========================================================================

COMPLETE ML TUTORIAL: This file implements Lasso Regression from scratch using
coordinate descent in NumPy. Lasso is the go-to algorithm when you need
AUTOMATIC FEATURE SELECTION -- it drives irrelevant feature weights to exactly
zero, producing sparse, interpretable models.

Theory & Mathematics:
    Lasso adds an L1 penalty to linear regression:

        L(w) = (1/2n) * ||y - Xw||^2  +  alpha * ||w||_1

    The L1 norm is not differentiable at zero, so standard gradient descent
    cannot be used directly. Instead, we implement **coordinate descent**,
    which optimises one weight at a time while holding the rest fixed.

    Coordinate Descent Update for feature j:
        1. Compute the partial residual excluding feature j:
              r_j = y - X_{-j} w_{-j}
        2. Compute the unconstrained OLS estimate for w_j:
              rho_j = (1/n) * X_j^T r_j
        3. Apply soft-thresholding:
              w_j = S(rho_j, alpha) / ((1/n) * ||X_j||^2)
           where S(z, t) = sign(z) * max(|z| - t, 0)

    Why Coordinate Descent?
        - Natural fit for L1: each sub-problem has a closed-form solution.
        - Very efficient for sparse problems; many updates are trivial.
        - Provably convergent for convex objectives.

Hyperparameters:
    - alpha (float): L1 regularisation strength. Default 0.1.
    - max_iter (int): Maximum coordinate sweeps. Default 1000.
    - tol (float): Convergence tolerance. Default 1e-6.
    - fit_intercept (bool): Whether to fit intercept. Default True.
"""

import logging
import time
from functools import partial

import numpy as np
import optuna
import ray
from ray import tune
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-Thresholding Operator
# ---------------------------------------------------------------------------

def _soft_threshold(z, threshold):
    """Soft-thresholding operator: S(z, t) = sign(z) * max(|z| - t, 0).

    This is the KEY operation that makes Lasso produce sparse solutions.

    WHY it works: If the unconstrained OLS estimate |z| is smaller than the
    threshold (alpha), the weight is set to exactly zero. Only features whose
    signal strength exceeds the regularisation threshold survive.

    Geometrically: The L1 ball has corners on the coordinate axes. The
    soft-threshold operator is the proximal operator that projects onto
    the L1 constraint set.

    Args:
        z: Unconstrained value (what the weight would be without regularisation).
        threshold: Regularisation strength alpha.

    Returns:
        Thresholded value (exactly 0 if |z| <= threshold).
    """
    # np.sign(z): Returns -1, 0, or +1 (preserves the sign).
    # np.maximum(|z| - threshold, 0): Shrinks magnitude by threshold, clips at 0.
    # Together: shrinks z toward zero by threshold amount, setting it to 0 if |z| < threshold.
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LassoRegressionNumpy:
    """Lasso via coordinate descent, implemented from scratch in NumPy.

    The coordinate descent algorithm optimises one feature weight at a time,
    cycling through all features. For each feature, it computes the optimal
    weight given all other weights are fixed, using soft-thresholding.
    """

    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-6, fit_intercept=True):
        # alpha: L1 regularisation strength.
        # WHY alpha matters: Controls the sparsity of the solution.
        # - alpha=0: No regularisation -> OLS (all weights non-zero).
        # - alpha large: Strong regularisation -> most/all weights become 0.
        # - Optimal alpha: Keeps informative features, zeros out noise features.
        self.alpha = alpha

        # max_iter: Maximum number of full sweeps through all features.
        # WHY 1000: Coordinate descent typically converges in 10-100 sweeps
        # for well-conditioned data. 1000 provides a safety margin.
        self.max_iter = max_iter

        # tol: Convergence tolerance on the maximum weight change per sweep.
        # WHY 1e-6: If no weight changes by more than 1e-6, we have converged.
        # Making this smaller gives more precision but takes longer.
        self.tol = tol

        # fit_intercept: Whether to learn a bias term (not regularised).
        self.fit_intercept = fit_intercept

        # Learned parameters.
        self.weights = None
        self.bias = 0.0
        self.n_iters_run = 0  # Actual iterations until convergence.

    def fit(self, X, y):
        """Fit Lasso model using coordinate descent.

        The algorithm alternates between:
        1. Cycling through features and updating each weight via soft-thresholding.
        2. Updating the intercept (if fit_intercept=True).
        3. Checking convergence.
        """
        n, p = X.shape

        # Initialise all weights to zero.
        # WHY zeros: For Lasso, starting at zero is efficient because many
        # weights will STAY at zero (the solution is sparse). This avoids
        # unnecessary work of first inflating then shrinking weights.
        self.weights = np.zeros(p)

        # Initialise bias to the mean of y.
        # WHY mean: If all weights are zero, the best constant predictor is
        # the mean of y. Starting here gives the intercept a head start.
        self.bias = np.mean(y) if self.fit_intercept else 0.0

        # Compute the initial residual: what is left after subtracting the bias.
        residual = y - self.bias

        # Precompute the squared L2 norm of each feature column (divided by n).
        # WHY precompute: This is constant across all iterations. Computing
        # it once saves O(n*p) operations per iteration.
        # This quantity normalises the soft-threshold output to account for
        # feature scaling.
        col_norm_sq = np.sum(X ** 2, axis=0) / n

        # Main coordinate descent loop.
        for iteration in range(self.max_iter):
            # Save a copy of current weights to check convergence.
            w_old = self.weights.copy()

            # Sweep through all features one at a time.
            for j in range(p):
                # STEP 1: Add back the current feature's contribution to the residual.
                # WHY: We want the residual as if feature j were not in the model.
                # This gives us r_j = y - X_{-j} @ w_{-j} - bias.
                residual += X[:, j] * self.weights[j]

                # STEP 2: Compute the unconstrained OLS estimate for feature j.
                # rho_j = (1/n) * X_j^T @ r_j
                # This is what w_j would be without any regularisation.
                rho_j = (X[:, j] @ residual) / n

                # STEP 3: Apply soft-thresholding to get the regularised estimate.
                # If |rho_j| < alpha, the weight becomes exactly 0 (feature is dropped).
                # If |rho_j| > alpha, the weight is shrunk by alpha toward 0.
                if col_norm_sq[j] == 0.0:
                    # Degenerate case: feature column is all zeros.
                    self.weights[j] = 0.0
                else:
                    self.weights[j] = _soft_threshold(rho_j, self.alpha) / col_norm_sq[j]

                # STEP 4: Update the residual to reflect the new weight for feature j.
                # This prepares the residual for the next feature's update.
                residual -= X[:, j] * self.weights[j]

            # Update the intercept after each full sweep.
            # WHY after sweep: The intercept depends on all weights being updated.
            # Updating it mid-sweep would be correct but this order is conventional.
            if self.fit_intercept:
                self.bias = np.mean(y - X @ self.weights)

            # Check convergence: if no weight changed by more than tol, stop.
            # WHY max instead of sum: max is more interpretable (no single weight
            # moved by more than tol) and is the standard convergence criterion.
            max_change = np.max(np.abs(self.weights - w_old))
            if max_change < self.tol:
                self.n_iters_run = iteration + 1
                logger.debug("Converged at iteration %d (max_change=%.2e)", iteration + 1, max_change)
                return self

        self.n_iters_run = self.max_iter
        return self

    def predict(self, X):
        """Predict target values."""
        return X @ self.weights + self.bias

    @property
    def n_nonzero(self):
        """Number of non-zero (selected) features."""
        return int(np.sum(self.weights != 0))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic data with 60/20/20 split and standard scaling."""
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(1, n_features // 2), noise=noise,
        random_state=random_state,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train Lasso model with given hyperparameters."""
    model = LassoRegressionNumpy(
        alpha=hp.get("alpha", 0.1),
        max_iter=hp.get("max_iter", 1000),
        tol=hp.get("tol", 1e-6),
        fit_intercept=hp.get("fit_intercept", True),
    )
    model.fit(X_train, y_train)
    logger.info("Lasso trained  alpha=%.4f  nonzero=%d  iters=%d",
                model.alpha, model.n_nonzero, model.n_iters_run)
    return model


def _metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def validate(model, X_val, y_val):
    m = _metrics(y_val, model.predict(X_val))
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    m = _metrics(y_test, model.predict(X_test))
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare different alpha values to show Lasso's feature selection behaviour.

    This is the most important demonstration: as alpha increases, more features
    are eliminated (set to exactly zero). The "regularisation path" shows
    which features are truly informative.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Lasso Sparsity vs Alpha (Feature Selection)")
    print("=" * 80)
    print("\nAs alpha increases, Lasso zeros out more features (sparse solutions).")
    print("This is Lasso's killer feature: AUTOMATIC FEATURE SELECTION.\n")

    n_features = X_train.shape[1]

    configs = {
        "Very Low alpha (0.001) - Nearly OLS": {
            "alpha": 0.001,
            # WHY: Almost no regularisation. All features likely non-zero.
            # USE WHEN: You want to keep all features and just need slight shrinkage.
        },
        "Low alpha (0.01) - Light Selection": {
            "alpha": 0.01,
            # WHY: Mild regularisation. Only the most irrelevant features are zeroed.
            # USE WHEN: Most features are informative, but a few are noise.
        },
        "Moderate alpha (0.1) - Balanced": {
            "alpha": 0.1,
            # WHY: Moderate regularisation. A good balance between sparsity and fit.
            # USE WHEN: You want a compact model but do not know how many features matter.
        },
        "High alpha (1.0) - Aggressive Selection": {
            "alpha": 1.0,
            # WHY: Strong regularisation. Only the strongest features survive.
            # USE WHEN: You want maximum interpretability or suspect many features are noise.
            # TRADE-OFF: May zero out some useful features, increasing bias.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, alpha=params["alpha"])
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        metrics["train_time"] = train_time
        metrics["n_nonzero"] = model.n_nonzero
        metrics["n_iters"] = model.n_iters_run
        results[name] = metrics

        # Show which features were selected.
        selected = [i for i, w in enumerate(model.weights) if w != 0]
        print(f"  {name}:")
        print(f"    Non-zero: {model.n_nonzero}/{n_features}  Selected features: {selected}")
        print(f"    MSE: {metrics['mse']:.6f}  R2: {metrics['r2']:.6f}  Iters: {model.n_iters_run}")

    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<40} {'alpha':>6} {'NonZero':>8} {'MSE':>10} {'R2':>10}")
    print("-" * 80)
    for name, m in results.items():
        alpha = configs[name]["alpha"]
        print(f"{name:<40} {alpha:>6.3f} {m['n_nonzero']:>8} {m['mse']:>10.4f} {m['r2']:>10.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Lasso produces EXACTLY zero weights -- true feature selection.")
    print("  2. Ridge (L2) only shrinks weights; Lasso (L1) eliminates them.")
    print("  3. The optimal alpha keeps informative features and drops noise.")
    print("  4. Too high alpha drops useful features (underfitting).")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Marketing Attribution
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate Lasso for marketing channel attribution.

    DOMAIN CONTEXT: Marketing teams spend budgets across many channels (TV,
    social media, search, email, etc.). Lasso regression identifies which
    channels actually drive sales, automatically ignoring channels with no
    impact. This is called "marketing mix modelling" (MMM).

    WHY Lasso: In marketing, many channels may be tracked but only a few
    actually drive sales. Lasso automatically identifies the effective
    channels by zeroing out the rest, giving marketers a clear action plan.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Marketing Channel Attribution (Lasso)")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 1500

    # 10 marketing channels, but only 4 actually drive sales.
    channel_names = [
        "tv_spend", "social_media", "search_ads", "email_marketing",
        "billboard", "podcast_ads", "influencer", "print_media",
        "radio_ads", "direct_mail"
    ]

    # Generate spend data for each channel (in thousands of dollars).
    X = np.column_stack([
        np.random.exponential(50, n_samples),    # TV
        np.random.exponential(20, n_samples),    # Social media
        np.random.exponential(30, n_samples),    # Search
        np.random.exponential(10, n_samples),    # Email
        np.random.exponential(15, n_samples),    # Billboard (no effect)
        np.random.exponential(5, n_samples),     # Podcast (no effect)
        np.random.exponential(8, n_samples),     # Influencer (no effect)
        np.random.exponential(12, n_samples),    # Print (no effect)
        np.random.exponential(7, n_samples),     # Radio (no effect)
        np.random.exponential(6, n_samples),     # Direct mail (no effect)
    ])

    # True coefficients: only first 4 channels actually drive sales.
    true_coefs = np.array([
        5.0,    # TV: $5K sales per $1K spent (5x ROAS)
        8.0,    # Social: $8K sales per $1K spent
        10.0,   # Search: $10K per $1K (highest ROAS)
        3.0,    # Email: $3K per $1K
        0.0,    # Billboard: no effect
        0.0,    # Podcast: no effect
        0.0,    # Influencer: no effect
        0.0,    # Print: no effect
        0.0,    # Radio: no effect
        0.0,    # Direct mail: no effect
    ])

    base_sales = 100  # Baseline sales without marketing ($100K)
    noise = np.random.normal(0, 50, n_samples)
    y = base_sales + X @ true_coefs + noise

    print(f"\nDataset: {n_samples} weekly observations, {len(channel_names)} marketing channels")
    print(f"Ground truth: Only {np.sum(true_coefs != 0)} channels actually drive sales")
    print(f"Challenge: Can Lasso identify which channels matter?")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train Lasso with moderate alpha for feature selection.
    model = train(X_train_s, y_train, alpha=0.5)

    metrics = _metrics(y_test, model.predict(X_test_s))
    print(f"\n--- Model Performance ---")
    print(f"  RMSE: ${metrics['rmse']:,.2f}K  R2: {metrics['r2']:.4f}")

    print(f"\n--- Channel Attribution Results ---")
    print(f"{'Channel':<20} {'Lasso Weight':>12} {'True Coef':>10} {'Selected':>10}")
    print("-" * 55)
    for i, name in enumerate(channel_names):
        w = model.weights[i]
        selected = "YES" if w != 0 else "no"
        print(f"{name:<20} {w:>12.4f} {true_coefs[i]:>10.1f} {selected:>10}")

    print(f"\n  Features selected: {model.n_nonzero}/{len(channel_names)}")
    print(f"  Ground truth active: {np.sum(true_coefs != 0)}/{len(channel_names)}")

    correct_zeros = sum(1 for i in range(len(true_coefs))
                       if true_coefs[i] == 0 and model.weights[i] == 0)
    correct_nonzeros = sum(1 for i in range(len(true_coefs))
                          if true_coefs[i] != 0 and model.weights[i] != 0)
    print(f"\n  Correctly identified active channels: {correct_nonzeros}/{int(np.sum(true_coefs != 0))}")
    print(f"  Correctly identified inactive channels: {correct_zeros}/{int(np.sum(true_coefs == 0))}")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 500, 5000, step=500),
        "tol": trial.suggest_float("tol", 1e-8, 1e-3, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    study = optuna.create_study(direction="minimize", study_name="lasso_numpy")
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        n_trials=n_trials, show_progress_bar=True,
    )
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    model = train(X_train, y_train, **config)
    metrics = validate(model, X_val, y_val)
    tune.report(**metrics)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "alpha": tune.loguniform(1e-5, 10.0),
        "max_iter": tune.choice([500, 1000, 2000, 5000]),
        "tol": tune.loguniform(1e-8, 1e-3),
        "fit_intercept": tune.choice([True, False]),
    }
    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    )
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="mse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Lasso Regression - NumPy From Scratch (Coordinate Descent)")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=25)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    model = train(X_train, y_train, **study.best_trial.params)
    print(f"\nNon-zero coefficients: {model.n_nonzero}/{len(model.weights)}")
    print(f"Iterations run: {model.n_iters_run}")

    print("\n--- Validation ---")
    for k, v in validate(model, X_val, y_val).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n--- Test ---")
    for k, v in test(model, X_test, y_test).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
