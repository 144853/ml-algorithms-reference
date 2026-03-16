"""
Ridge Regression - NumPy From-Scratch Implementation
=====================================================

COMPLETE ML TUTORIAL: This file implements Ridge Regression (L2 regularisation)
entirely from scratch using NumPy. It teaches you WHY regularisation matters,
HOW the L2 penalty changes the solution, and WHEN to choose Ridge over OLS.

Theory & Mathematics:
    Ridge Regression adds an L2 regularisation term to the OLS objective:

        L(w) = (1/2n) * ||y - Xw||^2  +  alpha * ||w||^2

    This implementation provides two solvers written entirely in NumPy:

    1. **Closed-Form (Normal Equation with regularisation)**

           w* = (X^T X + alpha * n * I)^{-1} X^T y

       The addition of alpha * I ensures the matrix is always invertible,
       even when X^T X is singular or ill-conditioned.

    2. **Gradient Descent**

           dL/dw = -(1/n) X^T (y - Xw)  +  2 * alpha * w
           w := w - lr * dL/dw

       The L2 term adds a weight-decay component (2 * alpha * w) to the
       gradient. This continuously pulls weights toward zero during
       training, which is equivalent to weight decay in neural networks.

    Relationship to Bayesian Regression:
        Ridge regression is equivalent to MAP estimation with a Gaussian
        prior on the weights: p(w) = N(0, (1/alpha) * I).

Business Use Cases:
    - Scientific modelling with many correlated measurements
    - Embedded environments where only NumPy is available
    - Understanding the internal mechanics of regularised regression
    - Custom pipelines requiring from-scratch implementations

Advantages:
    - Complete control over the algorithm's internals
    - Closed-form solution still available with regularisation
    - Gradient descent enables online / streaming updates
    - No external ML library required beyond NumPy

Disadvantages:
    - Manual code lacks production-level optimisations
    - GD requires careful tuning of learning rate and iterations
    - Closed-form O(p^3) is slow for very high dimensions
    - Does not perform feature selection (all weights remain non-zero)

Hyperparameters:
    - alpha (float): L2 regularisation strength. Default 1.0.
    - solver (str): "closed_form" or "gd". Default "closed_form".
    - lr (float): Learning rate for GD. Default 0.01.
    - n_iters (int): Number of GD iterations. Default 1000.
    - fit_intercept (bool): Whether to fit bias. Default True.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

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
# Model
# ---------------------------------------------------------------------------

class RidgeRegressionNumpy:
    """Ridge Regression from scratch using NumPy.

    The key insight: adding alpha * ||w||^2 to the loss shrinks all weights
    toward zero but never makes them exactly zero. This reduces model
    complexity and prevents overfitting, especially when features are
    correlated (multicollinearity).
    """

    def __init__(self, alpha=1.0, solver="closed_form", lr=0.01, n_iters=1000, fit_intercept=True):
        # alpha: The regularisation strength (lambda in some textbooks).
        # WHY this matters: alpha controls the bias-variance trade-off.
        # - alpha=0: Equivalent to ordinary least squares (no regularisation).
        # - alpha->infinity: All weights shrink to 0, model predicts the mean.
        # - alpha=1.0: A reasonable default for standardised features.
        # INTUITION: Think of alpha as a "leash" on the weights. Higher alpha
        # keeps weights closer to zero (shorter leash = simpler model).
        self.alpha = alpha

        # solver: Algorithm to use for finding optimal weights.
        # "closed_form": One-shot exact solution using matrix algebra.
        # "gd": Iterative gradient descent (useful for very large datasets).
        self.solver = solver

        # lr: Learning rate for gradient descent (only used when solver="gd").
        self.lr = lr

        # n_iters: Maximum gradient descent iterations.
        self.n_iters = n_iters

        # fit_intercept: Whether to learn a bias term.
        # WHY True: The intercept should NOT be regularised (we do not want
        # to penalise the model for predicting a non-zero mean).
        self.fit_intercept = fit_intercept

        # weights and bias: Learned parameters, set during fit().
        self.weights = None
        self.bias = 0.0

        # loss_history: MSE + L2 penalty at each GD iteration.
        self.loss_history = []

    def fit(self, X, y):
        """Fit the Ridge model to training data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).

        Returns:
            self (fitted model).
        """
        # Dispatch to the appropriate solver.
        if self.solver == "closed_form":
            self._fit_closed(X, y)
        elif self.solver == "gd":
            self._fit_gd(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        return self

    def _fit_closed(self, X, y):
        """Solve using the regularised normal equation.

        The closed-form solution for Ridge is:
            w* = (X^T X + alpha * n * I)^{-1} X^T y

        WHY alpha * n * I: The factor of n makes alpha comparable across
        different dataset sizes. Without it, doubling the dataset would
        halve the effective regularisation.

        WHY np.linalg.solve instead of np.linalg.inv:
            solve(A, b) computes A^{-1} b WITHOUT explicitly inverting A.
            This is numerically more stable and ~2x faster.
        """
        n, p = X.shape

        if self.fit_intercept:
            # Augment X with a column of ones to absorb the bias.
            X_aug = np.column_stack([X, np.ones(n)])

            # Create the regularisation matrix.
            # WHY p+1 x p+1: The augmented feature matrix has p+1 columns.
            reg = self.alpha * n * np.eye(p + 1)

            # CRITICAL: Set the bias regularisation to 0.
            # WHY: We should NOT penalise the intercept term. The intercept
            # captures the mean of y, and penalising it would bias predictions
            # toward zero regardless of the data distribution.
            reg[-1, -1] = 0

            # Solve the linear system: (X^T X + reg) @ theta = X^T @ y
            theta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)

            # Extract weights and bias from the solution vector.
            self.weights = theta[:-1]
            self.bias = theta[-1]
        else:
            # Without intercept, simpler formulation.
            reg = self.alpha * n * np.eye(p)
            self.weights = np.linalg.solve(X.T @ X + reg, X.T @ y)
            self.bias = 0.0

    def _fit_gd(self, X, y):
        """Solve using gradient descent with L2 regularisation.

        The gradient of the Ridge loss is:
            dL/dw = (1/n) X^T (Xw - y) + 2 * alpha * w

        The term 2 * alpha * w is the L2 gradient (weight decay).
        It pulls weights toward zero at each step, proportional to
        their current magnitude.
        """
        n, p = X.shape

        # Initialise weights to zeros.
        self.weights = np.zeros(p)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iters):
            # Forward pass: compute predictions.
            y_pred = X @ self.weights + self.bias

            # Compute residuals (prediction errors).
            residual = y_pred - y

            # Compute gradient WITH L2 regularisation term.
            # The regularisation gradient 2*alpha*w pushes weights toward zero.
            # WHY this helps: Without regularisation, noisy features can get
            # large weights that fit the noise in training data. The L2 term
            # penalises large weights, forcing the model to use all features
            # moderately rather than relying heavily on a few.
            dw = (1.0 / n) * (X.T @ residual) + 2 * self.alpha * self.weights

            # Bias gradient: NOT regularised.
            db = (1.0 / n) * np.sum(residual)

            # Update weights.
            self.weights -= self.lr * dw
            if self.fit_intercept:
                self.bias -= self.lr * db

            # Track the FULL Ridge loss (MSE + L2 penalty).
            mse = np.mean(residual ** 2) + self.alpha * np.sum(self.weights ** 2)
            self.loss_history.append(mse)

    def predict(self, X):
        """Generate predictions: y_hat = X @ w + b."""
        return X @ self.weights + self.bias


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data with the standard 60/20/20 split."""
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
    """Train Ridge regression model with given hyperparameters."""
    model = RidgeRegressionNumpy(
        alpha=hp.get("alpha", 1.0),
        solver=hp.get("solver", "closed_form"),
        lr=hp.get("lr", 0.01),
        n_iters=hp.get("n_iters", 1000),
        fit_intercept=hp.get("fit_intercept", True),
    )
    model.fit(X_train, y_train)
    logger.info("Trained (solver=%s, alpha=%.4f)  bias=%.6f", model.solver, model.alpha, model.bias)
    return model


def _metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def validate(model, X_val, y_val):
    """Evaluate on validation set."""
    m = _metrics(y_val, model.predict(X_val))
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    """Evaluate on test set."""
    m = _metrics(y_test, model.predict(X_test))
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare different alpha (regularisation strength) configurations.

    This is THE key hyperparameter for Ridge regression. This function
    demonstrates the bias-variance trade-off by sweeping alpha from
    very small (nearly OLS) to very large (heavy shrinkage).
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Ridge Regularisation Strength (alpha)")
    print("=" * 80)
    print("\nThe bias-variance trade-off in action:")
    print("  Low alpha  -> Low bias, high variance  (close to OLS, may overfit)")
    print("  High alpha -> High bias, low variance  (heavy shrinkage, may underfit)")

    configs = {
        "No Regularisation (alpha=0.001)": {
            "alpha": 0.001,
            "solver": "closed_form",
            "fit_intercept": True,
            # WHY: Nearly equivalent to OLS. Useful as a baseline to see
            # how much regularisation helps (or hurts) on this data.
            # TRADE-OFF: Best fit on training data, but may overfit.
        },
        "Light Regularisation (alpha=0.1)": {
            "alpha": 0.1,
            "solver": "closed_form",
            "fit_intercept": True,
            # WHY: Gentle constraint on weights. Helps when features are
            # slightly correlated but the signal is strong.
            # TRADE-OFF: Slight bias increase for meaningful variance reduction.
        },
        "Moderate Regularisation (alpha=1.0)": {
            "alpha": 1.0,
            "solver": "closed_form",
            "fit_intercept": True,
            # WHY: The default. A good starting point for most problems.
            # TRADE-OFF: Balanced bias-variance. Works well in most scenarios.
        },
        "Heavy Regularisation (alpha=100.0)": {
            "alpha": 100.0,
            "solver": "closed_form",
            "fit_intercept": True,
            # WHY: Very strong shrinkage. Appropriate when you have many noisy
            # features and few samples, or severe multicollinearity.
            # TRADE-OFF: May underfit by shrinking informative weights too much.
        },
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        metrics["train_time"] = train_time
        metrics["weight_norm"] = np.sqrt(np.sum(model.weights ** 2))
        results[name] = metrics

        print(f"\n  {name}:")
        print(f"    MSE: {metrics['mse']:.6f}  R2: {metrics['r2']:.6f}")
        print(f"    Weight L2 norm: {metrics['weight_norm']:.4f}")
        print(f"    (Larger alpha -> smaller weight norm -> simpler model)")

    # Summary
    print(f"\n{'=' * 85}")
    print(f"{'Configuration':<40} {'alpha':>8} {'MSE':>10} {'R2':>10} {'||w||':>10}")
    print("-" * 85)
    for name, metrics in results.items():
        alpha = configs[name]["alpha"]
        print(f"{name:<40} {alpha:>8.3f} {metrics['mse']:>10.4f} {metrics['r2']:>10.4f} {metrics['weight_norm']:>10.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. As alpha increases, ||w|| decreases (weights shrink toward zero).")
    print("  2. There is an optimal alpha that minimises validation MSE.")
    print("  3. Too little regularisation: overfitting (high variance).")
    print("  4. Too much regularisation: underfitting (high bias).")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Medical Cost Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate Ridge regression on medical insurance cost prediction.

    DOMAIN CONTEXT: Health insurance companies estimate expected medical costs
    for policyholders based on demographic and health factors. Ridge regression
    is appropriate here because many health factors are correlated (e.g., BMI
    and blood pressure, age and number of conditions), making OLS unstable.

    WHY Ridge for this problem: Medical features are often correlated.
    Ridge regression stabilises coefficient estimates when multicollinearity
    is present, giving more reliable predictions.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Medical Insurance Cost Prediction")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 2000

    # Age: 18-65 years, uniformly distributed.
    age = np.random.uniform(18, 65, n_samples)

    # BMI: Body Mass Index, normally distributed around 27 (slightly overweight average).
    bmi = np.random.normal(27, 5, n_samples).clip(15, 50)

    # Blood pressure (systolic): Correlated with age and BMI.
    # WHY correlated: In real medical data, older and heavier patients tend
    # to have higher blood pressure. This creates multicollinearity.
    blood_pressure = 90 + 0.3 * age + 0.5 * bmi + np.random.normal(0, 10, n_samples)

    # Number of chronic conditions: increases with age.
    n_conditions = np.random.poisson(age / 20, n_samples).clip(0, 10)

    # Smoker status: binary (0 or 1). Smokers have higher medical costs.
    smoker = np.random.binomial(1, 0.2, n_samples)

    # Exercise frequency (hours per week): inversely related to medical costs.
    exercise_hours = np.random.exponential(3, n_samples).clip(0, 20)

    # Number of dependents: affects insurance plan type.
    n_dependents = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.25, 0.15, 0.05])

    feature_names = ["age", "bmi", "blood_pressure", "n_conditions",
                     "smoker", "exercise_hours", "n_dependents"]
    X = np.column_stack([age, bmi, blood_pressure, n_conditions,
                         smoker, exercise_hours, n_dependents])

    # True cost model with realistic coefficients.
    true_coefs = np.array([
        250.0,      # $250 per year of age
        300.0,      # $300 per BMI point
        100.0,      # $100 per mmHg of blood pressure
        3000.0,     # $3000 per chronic condition
        15000.0,    # $15000 smoker surcharge
        -500.0,     # -$500 per hour of weekly exercise
        1200.0,     # $1200 per dependent
    ])

    base_cost = 2000
    noise = np.random.normal(0, 5000, n_samples)
    y = base_cost + X @ true_coefs + noise
    y = y.clip(500, None)

    print(f"\nDataset: {n_samples} insurance policyholders")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Note: blood_pressure is correlated with age and BMI (multicollinearity!)")
    print(f"Cost range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Mean cost: ${y.mean():,.0f}")

    # Pipeline.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Compare OLS vs Ridge on this multicollinear data.
    print(f"\n--- Comparing OLS vs Ridge on Correlated Medical Data ---")

    # OLS (very low alpha).
    ols_model = train(X_train_s, y_train, alpha=0.0001, solver="closed_form")
    ols_metrics = _metrics(y_test, ols_model.predict(X_test_s))

    # Ridge.
    ridge_model = train(X_train_s, y_train, alpha=1.0, solver="closed_form")
    ridge_metrics = _metrics(y_test, ridge_model.predict(X_test_s))

    print(f"\n  OLS (alpha~0):  RMSE=${ols_metrics['rmse']:,.2f}  R2={ols_metrics['r2']:.4f}")
    print(f"  Ridge (alpha=1): RMSE=${ridge_metrics['rmse']:,.2f}  R2={ridge_metrics['r2']:.4f}")
    print(f"\n  Ridge stabilises coefficients when features are correlated.")

    # Show coefficient comparison.
    print(f"\n--- Feature Importance (Ridge) ---")
    print(f"{'Feature':<20} {'Weight':>10}")
    print("-" * 32)
    for i, name in enumerate(feature_names):
        print(f"{name:<20} {ridge_model.weights[i]:>10.2f}")

    return ridge_model, ridge_metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for Ridge regression."""
    solver = trial.suggest_categorical("solver", ["closed_form", "gd"])
    hp = {
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        "solver": solver,
        "lr": trial.suggest_float("lr", 1e-4, 0.5, log=True) if solver == "gd" else 0.01,
        "n_iters": trial.suggest_int("n_iters", 200, 3000, step=200) if solver == "gd" else 1000,
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }
    model = train(X_train, y_train, **hp)
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Run Optuna study for Ridge regression."""
    study = optuna.create_study(direction="minimize", study_name="ridge_numpy")
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
        "solver": tune.choice(["closed_form", "gd"]),
        "lr": tune.loguniform(1e-4, 0.5),
        "n_iters": tune.choice([500, 1000, 2000]),
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
    print("Ridge Regression - NumPy From Scratch")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Educational comparisons.
    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    # Hyperparameter search.
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
