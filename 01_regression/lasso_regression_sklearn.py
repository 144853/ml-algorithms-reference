"""
Lasso Regression - scikit-learn Implementation
===============================================

COMPLETE ML TUTORIAL: This file teaches Lasso (L1 regularisation) using
scikit-learn. Lasso's superpower is AUTOMATIC FEATURE SELECTION -- it drives
irrelevant feature weights to exactly zero, creating sparse, interpretable models.

Theory & Mathematics:
    Lasso (Least Absolute Shrinkage and Selection Operator) adds an L1
    penalty to the OLS objective:

        L(w) = (1/2n) * ||y - Xw||^2  +  alpha * ||w||_1

    where ||w||_1 = sum(|w_j|) is the L1 norm.

    Unlike Ridge (L2), the L1 penalty produces SPARSE solutions: many
    coefficients are driven exactly to zero, effectively performing
    automatic feature selection.

    scikit-learn's Lasso uses coordinate descent, cycling through each
    feature and applying the soft-thresholding operator.

Hyperparameters:
    - alpha (float): L1 regularisation strength. Default 1.0.
    - max_iter (int): Maximum coordinate descent iterations. Default 1000.
    - tol (float): Convergence tolerance. Default 1e-4.
    - fit_intercept (bool): Whether to fit bias. Default True.
    - selection (str): "cyclic" or "random" feature update order. Default "cyclic".
"""

import logging
import time
from functools import partial

import numpy as np
import optuna
import ray
from ray import tune
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    """Train sklearn Lasso model.

    sklearn.linear_model.Lasso uses optimised coordinate descent written in
    Cython, making it much faster than our NumPy from-scratch version.
    """
    # alpha: L1 penalty strength. THE most important hyperparameter.
    alpha = hp.get("alpha", 1.0)

    # max_iter: Maximum coordinate descent iterations before stopping.
    # WHY might need increase: Large alpha with many features can require
    # more iterations to converge (many weights slowly approaching zero).
    max_iter = hp.get("max_iter", 1000)

    # tol: Convergence tolerance. Stops when the optimality condition
    # (dual gap) is below this threshold.
    tol = hp.get("tol", 1e-4)

    fit_intercept = hp.get("fit_intercept", True)

    # selection: Order of feature updates in coordinate descent.
    # "cyclic": Updates features in order 0, 1, 2, ..., p, 0, 1, ...
    # "random": Updates features in random order each sweep.
    # WHY random can help: Breaks correlations between consecutive updates,
    # sometimes converging faster on highly correlated features.
    selection = hp.get("selection", "cyclic")

    model = Lasso(
        alpha=alpha, max_iter=max_iter, tol=tol,
        fit_intercept=fit_intercept, selection=selection,
    )
    model.fit(X_train, y_train)

    # Count non-zero coefficients (selected features).
    n_nonzero = np.sum(model.coef_ != 0)
    logger.info("Lasso trained  alpha=%.4f  nonzero=%d/%d",
                alpha, n_nonzero, len(model.coef_))
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
    """Compare alpha values and selection strategies for sklearn Lasso.

    Demonstrates Lasso's regularisation path: as alpha increases,
    more features are eliminated until the model becomes trivial.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: sklearn Lasso - Alpha & Selection Strategy")
    print("=" * 80)

    n_features = X_train.shape[1]

    configs = {
        "Very Light (alpha=0.001, cyclic)": {
            "alpha": 0.001, "selection": "cyclic",
            # WHY: Nearly unregularised. Should keep all features.
        },
        "Moderate (alpha=0.1, cyclic)": {
            "alpha": 0.1, "selection": "cyclic",
            # WHY: Balanced. Drops clearly irrelevant features.
        },
        "Strong (alpha=1.0, cyclic)": {
            "alpha": 1.0, "selection": "cyclic",
            # WHY: Aggressive selection. Only strongest features survive.
        },
        "Moderate (alpha=0.1, random)": {
            "alpha": 0.1, "selection": "random",
            # WHY: Compare random vs cyclic selection at same alpha.
            # Random selection can converge faster with correlated features.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        n_nz = int(np.sum(model.coef_ != 0))
        metrics["n_nonzero"] = n_nz
        metrics["train_time"] = train_time
        results[name] = metrics

        print(f"\n  {name}: NonZero={n_nz}/{n_features}  MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}")

    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<38} {'alpha':>6} {'NZ':>4} {'MSE':>10} {'R2':>10}")
    print("-" * 72)
    for name, m in results.items():
        alpha = configs[name]["alpha"]
        print(f"{name:<38} {alpha:>6.3f} {m['n_nonzero']:>4} {m['mse']:>10.4f} {m['r2']:>10.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Lasso with small alpha keeps most features (like OLS).")
    print("  2. Increasing alpha aggressively prunes features to zero.")
    print("  3. Random selection can sometimes converge faster than cyclic.")
    print("  4. The best alpha maximises R2 while minimising the number of features.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Genomics Feature Selection
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate Lasso for gene expression analysis.

    DOMAIN CONTEXT: In genomics, researchers measure expression levels of
    thousands of genes but only a handful are related to a disease outcome.
    Lasso regression is the standard approach for identifying which genes
    matter, because it naturally handles the "p >> n" problem (more features
    than samples) through L1-driven sparsity.

    WHY Lasso: With 50 gene features but only 20 truly relevant, Lasso
    can automatically identify the informative genes.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Gene Expression Analysis (Lasso Feature Selection)")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 500   # Typical genomics: limited patient samples
    n_genes = 50      # Simplified (real genomics has thousands)
    n_informative = 8 # Only 8 genes actually affect the outcome

    # Generate gene expression levels (normalised, mean=0, std=1).
    gene_names = [f"gene_{i:02d}" for i in range(n_genes)]
    X = np.random.randn(n_samples, n_genes)

    # True model: only first n_informative genes matter.
    true_coefs = np.zeros(n_genes)
    true_coefs[:n_informative] = np.random.uniform(2, 10, n_informative) * np.random.choice([-1, 1], n_informative)

    noise = np.random.normal(0, 3, n_samples)
    y = X @ true_coefs + noise  # Disease severity score

    print(f"\nDataset: {n_samples} patients, {n_genes} genes measured")
    print(f"Ground truth: Only {n_informative} genes affect disease severity")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Use Lasso for feature selection.
    model = Lasso(alpha=0.5, max_iter=5000, fit_intercept=True)
    model.fit(X_train_s, y_train)

    metrics = _metrics(y_test, model.predict(X_test_s))
    n_selected = int(np.sum(model.coef_ != 0))

    print(f"\n--- Lasso Results ---")
    print(f"  Genes selected: {n_selected}/{n_genes}")
    print(f"  RMSE: {metrics['rmse']:.4f}  R2: {metrics['r2']:.4f}")

    print(f"\n--- Selected Genes ---")
    print(f"{'Gene':<12} {'Lasso Coef':>12} {'True Coef':>10} {'Correct?':>10}")
    print("-" * 48)
    for i in range(n_genes):
        if model.coef_[i] != 0 or true_coefs[i] != 0:
            correct = "YES" if (model.coef_[i] != 0) == (true_coefs[i] != 0) else "MISS"
            print(f"{gene_names[i]:<12} {model.coef_[i]:>12.4f} {true_coefs[i]:>10.4f} {correct:>10}")

    # Compute feature selection accuracy.
    true_pos = sum(1 for i in range(n_genes) if model.coef_[i] != 0 and true_coefs[i] != 0)
    true_neg = sum(1 for i in range(n_genes) if model.coef_[i] == 0 and true_coefs[i] == 0)
    print(f"\n  True positives (correctly identified active genes): {true_pos}/{n_informative}")
    print(f"  True negatives (correctly ignored inactive genes): {true_neg}/{n_genes - n_informative}")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 500, 5000, step=500),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    study = optuna.create_study(direction="minimize", study_name="lasso_sklearn")
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
        "max_iter": tune.choice([1000, 2000, 5000]),
        "tol": tune.loguniform(1e-6, 1e-2),
        "fit_intercept": tune.choice([True, False]),
        "selection": tune.choice(["cyclic", "random"]),
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
    print("Lasso Regression - scikit-learn")
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
    n_nonzero = np.sum(model.coef_ != 0)
    print(f"\nNon-zero coefficients: {n_nonzero}/{len(model.coef_)}")

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
