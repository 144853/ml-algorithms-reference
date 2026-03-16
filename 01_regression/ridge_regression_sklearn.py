"""
Ridge Regression - scikit-learn Implementation
===============================================

COMPLETE ML TUTORIAL: This file teaches Ridge Regression (L2 regularisation)
using scikit-learn. It explains the bias-variance trade-off through alpha
tuning and demonstrates when Ridge is preferred over plain OLS.

Theory & Mathematics:
    Ridge Regression (Tikhonov regularisation) extends OLS by adding an L2
    penalty on the weight magnitudes:

        L(w) = (1/2n) * ||y - Xw||^2  +  alpha * ||w||^2

    The closed-form solution becomes:

        w* = (X^T X + alpha * I)^{-1} X^T y

    The L2 penalty shrinks coefficients toward zero but never sets them
    exactly to zero. This makes Ridge especially useful when features are
    correlated (multicollinearity), because it stabilises the matrix inverse.

    Bias-Variance Trade-off:
        - Increasing alpha reduces variance (simpler model) at the cost of
          higher bias.
        - alpha = 0 recovers OLS.

Hyperparameters:
    - alpha (float): Regularisation strength. Larger = more shrinkage.
    - fit_intercept (bool): Whether to compute the intercept.
    - solver (str): "auto", "svd", "cholesky", "lsqr", "sag", "saga".
"""

import logging
import time
from functools import partial

import numpy as np
import optuna
import ray
from ray import tune
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data with 60/20/20 split and scaling."""
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(1, n_features // 2), noise=noise,
        random_state=random_state,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # StandardScaler: fit ONLY on training data to prevent data leakage.
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
    """Train sklearn Ridge regression model.

    Args:
        **hp: Hyperparameters including alpha, fit_intercept, solver.
    """
    # alpha: The regularisation strength. This is THE key hyperparameter.
    alpha = hp.get("alpha", 1.0)
    fit_intercept = hp.get("fit_intercept", True)
    # solver: Algorithm used internally by sklearn.
    # "auto": sklearn picks the best solver based on data characteristics.
    # "svd": Singular Value Decomposition -- most stable, handles singular matrices.
    # "cholesky": Cholesky decomposition -- fastest for dense data, requires X^T X + alpha*I > 0.
    # "lsqr": Iterative least-squares -- good for very large sparse data.
    solver = hp.get("solver", "auto")

    # Create and fit the Ridge model.
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
    model.fit(X_train, y_train)
    logger.info("Ridge trained  alpha=%.4f  intercept=%.6f", alpha, model.intercept_)
    return model


def _metrics(y_true, y_pred):
    """Compute MSE, RMSE, MAE, R2."""
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
    """Compare alpha values and solver types to understand their impact.

    This function sweeps across alpha values and solver choices, showing
    how regularisation strength affects both prediction quality and
    coefficient magnitudes.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Alpha & Solver Trade-offs (sklearn Ridge)")
    print("=" * 80)

    configs = {
        "Very Low alpha (alpha=0.001)": {
            "alpha": 0.001,
            "solver": "auto",
            "fit_intercept": True,
            # WHY: Nearly OLS. Baseline for comparison.
            # USE WHEN: Data is clean, low multicollinearity, many samples.
        },
        "Moderate alpha (alpha=1.0)": {
            "alpha": 1.0,
            "solver": "auto",
            "fit_intercept": True,
            # WHY: Standard default. Good general-purpose setting.
            # USE WHEN: Moderate multicollinearity, reasonable sample size.
        },
        "High alpha (alpha=10.0)": {
            "alpha": 10.0,
            "solver": "auto",
            "fit_intercept": True,
            # WHY: Strong shrinkage.
            # USE WHEN: Many correlated features, small sample size, noisy data.
        },
        "SVD Solver (stable, alpha=1.0)": {
            "alpha": 1.0,
            "solver": "svd",
            "fit_intercept": True,
            # WHY: SVD is the most numerically stable solver.
            # USE WHEN: Data might be ill-conditioned or near-singular.
            # TRADE-OFF: Slightly slower than Cholesky for well-conditioned data.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        metrics["train_time"] = train_time
        metrics["coef_l2_norm"] = np.sqrt(np.sum(model.coef_ ** 2))
        results[name] = metrics

        print(f"\n  {name}:")
        print(f"    MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}  ||w||={metrics['coef_l2_norm']:.4f}")

    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<35} {'MSE':>10} {'R2':>10} {'||w||':>10} {'Time(s)':>10}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<35} {m['mse']:>10.4f} {m['r2']:>10.4f} {m['coef_l2_norm']:>10.4f} {m['train_time']:>10.6f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Higher alpha -> smaller coefficient norm (||w||) -> simpler model.")
    print("  2. SVD solver is most stable but may be slightly slower.")
    print("  3. The optimal alpha balances underfitting (too high) and overfitting (too low).")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Student Performance Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate Ridge regression on predicting student exam scores.

    DOMAIN CONTEXT: Educational institutions use predictive models to identify
    students at risk of poor performance. Features like study hours, attendance,
    and prior grades are often correlated (good students tend to study more AND
    attend class regularly), making Ridge regression appropriate.

    WHY Ridge: Study hours, assignment scores, and attendance are correlated.
    OLS would give unstable coefficients; Ridge stabilises them.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Student Exam Score Prediction")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 1500

    # Study hours per week (correlated with engagement).
    study_hours = np.random.normal(15, 6, n_samples).clip(0, 40)

    # Class attendance percentage (correlated with study hours).
    attendance_pct = 50 + 2 * study_hours + np.random.normal(0, 10, n_samples)
    attendance_pct = attendance_pct.clip(0, 100)

    # Previous semester GPA (0-4 scale).
    prev_gpa = np.random.normal(3.0, 0.5, n_samples).clip(0, 4.0)

    # Assignment completion rate (%).
    assignment_rate = 40 + 1.5 * study_hours + np.random.normal(0, 12, n_samples)
    assignment_rate = assignment_rate.clip(0, 100)

    # Hours of sleep per night.
    sleep_hours = np.random.normal(7, 1.5, n_samples).clip(3, 12)

    # Part-time work hours (negative effect on studying).
    work_hours = np.random.exponential(5, n_samples).clip(0, 30)

    feature_names = ["study_hours", "attendance_pct", "prev_gpa",
                     "assignment_rate", "sleep_hours", "work_hours"]
    X = np.column_stack([study_hours, attendance_pct, prev_gpa,
                         assignment_rate, sleep_hours, work_hours])

    # True exam score model (0-100 scale).
    true_coefs = np.array([1.2, 0.3, 10.0, 0.2, 2.0, -0.8])
    base_score = 20
    noise = np.random.normal(0, 5, n_samples)
    y = base_score + X @ true_coefs + noise
    y = y.clip(0, 100)

    print(f"\nDataset: {n_samples} students, {len(feature_names)} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Note: study_hours, attendance_pct, and assignment_rate are CORRELATED")
    print(f"Score range: {y.min():.1f} - {y.max():.1f}")
    print(f"Mean score: {y.mean():.1f}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train with moderate regularisation.
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_train_s, y_train)

    metrics = _metrics(y_test, model.predict(X_test_s))

    print(f"\n--- Model Performance ---")
    print(f"  RMSE: {metrics['rmse']:.2f} points (average prediction error)")
    print(f"  R2:   {metrics['r2']:.4f}")

    print(f"\n--- Feature Importances (Standardised) ---")
    print(f"{'Feature':<20} {'Weight':>10} {'Interpretation':>25}")
    print("-" * 58)
    for i, name in enumerate(feature_names):
        w = model.coef_[i]
        if w > 0:
            interp = f"+{w:.2f} points/std"
        else:
            interp = f"{w:.2f} points/std"
        print(f"{name:<20} {w:>10.4f} {interp:>25}")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    study = optuna.create_study(direction="minimize", study_name="ridge_sklearn")
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
        "alpha": tune.loguniform(1e-4, 100.0),
        "fit_intercept": tune.choice([True, False]),
        "solver": tune.choice(["auto", "svd", "cholesky", "lsqr"]),
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
    print("Ridge Regression - scikit-learn")
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
