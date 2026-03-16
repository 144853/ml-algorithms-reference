"""
Linear Regression - scikit-learn Implementation
================================================

COMPLETE ML TUTORIAL: This file teaches linear regression using scikit-learn,
the industry standard ML library in Python. It covers not just HOW to use
the API, but WHY each step matters and what trade-offs you are making.

Theory & Mathematics:
    Linear Regression models the relationship between a dependent variable y
    and one or more independent variables X by fitting a linear equation:

        y = X @ w + b

    where w is the weight vector and b is the bias (intercept) term.

    The Ordinary Least Squares (OLS) objective minimizes the residual sum of
    squares (RSS):

        L(w) = (1/2n) * ||y - Xw||^2

    The closed-form solution (Normal Equation) is:

        w* = (X^T X)^{-1} X^T y

    scikit-learn's LinearRegression uses a numerically stable SVD-based solver
    by default, which is equivalent to OLS but avoids explicit matrix inversion.

    Key Metrics:
        - MSE  = (1/n) * sum((y_i - y_hat_i)^2)
        - RMSE = sqrt(MSE)
        - MAE  = (1/n) * sum(|y_i - y_hat_i|)
        - R^2  = 1 - SS_res / SS_tot

Business Use Cases:
    - Predicting house prices based on property features (area, bedrooms, etc.)
    - Forecasting sales revenue from advertising spend across channels
    - Estimating customer lifetime value from engagement metrics
    - Modeling dose-response relationships in pharmaceutical trials

Advantages:
    - Simple, interpretable, and fast to train
    - Closed-form solution guarantees a global optimum
    - No hyperparameters to tune (for basic OLS)
    - Coefficients directly indicate feature importance and direction

Disadvantages:
    - Assumes a linear relationship between features and target
    - Sensitive to outliers (squared loss amplifies large residuals)
    - Susceptible to multicollinearity (correlated features inflate variance)
    - Cannot capture non-linear patterns without manual feature engineering

Hyperparameters (scikit-learn LinearRegression):
    - fit_intercept (bool): Whether to calculate the intercept term. Default True.
    - copy_X (bool): Whether to copy X or overwrite. Default True.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# logging: Industry-standard way to output diagnostic information.
# WHY over print: Configurable levels (DEBUG/INFO/WARNING), can redirect to files,
# and can be toggled off in production without changing code.
import logging

# time: For measuring wall-clock training and inference time.
# WHY: Comparing configurations by both accuracy AND speed reveals
# whether complex setups are worth the computational cost.
import time

# functools.partial: Creates a new function with pre-filled arguments.
# WHY: Optuna's objective needs signature f(trial) -> float, but we also
# need to pass data. partial() "bakes in" the data arrays.
from functools import partial

# numpy: Numerical computing foundation. Used for array operations and metrics.
import numpy as np

# optuna: Bayesian hyperparameter optimisation using TPE.
# WHY over grid search: For linear regression with few hyperparameters,
# it does not matter much. But learning the Optuna pattern is valuable
# because it scales to complex models where grid search is infeasible.
import optuna

# ray and ray.tune: Distributed hyperparameter search.
# WHY: Parallelises trials across CPU cores for faster search.
import ray
from ray import tune

# sklearn.datasets.make_regression: Generates synthetic linear regression data.
# WHY synthetic: Controlled experiments where we know the true relationship.
from sklearn.datasets import make_regression

# sklearn.linear_model.LinearRegression: Production-quality OLS implementation.
# WHY sklearn: Highly optimised (uses LAPACK), well-tested, familiar API.
from sklearn.linear_model import LinearRegression

# sklearn.metrics: Standard evaluation metrics for regression.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# sklearn.model_selection.train_test_split: Shuffled stratified data splitting.
# WHY: Ensures random but reproducible train/val/test splits.
from sklearn.model_selection import train_test_split

# sklearn.preprocessing.StandardScaler: Z-score normalisation.
# WHY: While sklearn's LinearRegression works without scaling (it uses SVD
# internally), scaling is still good practice and is critical for gradient-based
# methods used in hyperparameter search evaluation.
from sklearn.preprocessing import StandardScaler

# Configure logging to show timestamps and severity.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data and split into train/val/test.

    WHY three-way split:
    - Train: fit model parameters (weights w, bias b)
    - Validation: tune hyperparameters WITHOUT touching test data
    - Test: final unbiased performance estimate (used exactly once)

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) after
        standard-scaling features.
    """
    # make_regression creates y = Xw* + noise where w* is a random true weight vector.
    # n_informative = n_features // 2 means half the features are noise columns.
    # WHY: Tests if the model handles irrelevant features (it does -- OLS just
    # assigns them near-zero coefficients).
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(1, n_features // 2),
        noise=noise,
        random_state=random_state,
    )

    # First split: 60% train, 40% temp (will become val + test).
    # WHY 60/20/20: Standard split ratio. More training data generally helps,
    # but we need enough validation/test data for reliable metric estimates.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    # Second split: 50/50 of the 40% temp = 20% val, 20% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    # StandardScaler: Transforms each feature to have mean=0 and std=1.
    # Formula: x_scaled = (x - mean) / std
    # WHY: Makes features comparable. A feature in dollars (10000s) would
    # otherwise dominate a feature in percentages (0-1).
    scaler = StandardScaler()

    # fit_transform on TRAINING data only -- learn mean/std from train set.
    # WHY not fit on all data: Prevents data leakage. In production, you
    # would not have access to test/validation statistics before deployment.
    X_train = scaler.fit_transform(X_train)

    # transform (NOT fit_transform) on val/test: use training statistics.
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        "Data generated  train=%d  val=%d  test=%d  features=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1],
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hyperparams):
    """Train a scikit-learn LinearRegression model.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training target vector of shape (n_samples,).
        **hyperparams: Keyword arguments forwarded to LinearRegression().
            fit_intercept (bool): Whether to learn a bias term. Default True.

    Returns:
        Trained LinearRegression model.
    """
    # Extract hyperparameters with sensible defaults.
    # WHY .get() with default: Makes this function robust when called with
    # partial hyperparameter specifications (common in hyperparameter search).
    fit_intercept = hyperparams.get("fit_intercept", True)

    # Create the sklearn LinearRegression model.
    # Under the hood, sklearn uses scipy.linalg.lstsq which calls LAPACK's
    # dgelsd routine (SVD-based least squares). This is numerically stable
    # and much faster than our NumPy from-scratch implementation for large data.
    model = LinearRegression(fit_intercept=fit_intercept)

    # model.fit() computes the closed-form OLS solution.
    # After this call, model.coef_ contains the weights and model.intercept_
    # contains the bias. There is no iteration or convergence -- it is exact.
    model.fit(X_train, y_train)

    # Log the intercept for debugging. In a well-scaled dataset, the intercept
    # should be close to the mean of y (when features are centered).
    logger.info("Model trained  intercept=%.6f", model.intercept_ if fit_intercept else 0.0)
    return model


def validate(model, X_val, y_val):
    """Evaluate model on validation data.

    WHY separate validate and test functions: Although they compute the same
    metrics, separating them makes the code self-documenting. The names convey
    when each evaluation is appropriate in the ML workflow.

    Returns:
        Dictionary with mse, rmse, mae, and r2.
    """
    # model.predict() computes X_val @ w + b using the learned weights.
    y_pred = model.predict(X_val)

    # Compute all four standard regression metrics.
    metrics = {
        "mse": mean_squared_error(y_val, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
        "mae": mean_absolute_error(y_val, y_pred),
        "r2": r2_score(y_val, y_pred),
    }
    logger.info("Validation metrics: %s", metrics)
    return metrics


def test(model, X_test, y_test):
    """Evaluate model on held-out test data.

    WHY: Final unbiased performance estimate. This simulates deployment
    performance because the model has never seen this data during training
    or hyperparameter selection.

    Returns:
        Dictionary with mse, rmse, mae, and r2.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    logger.info("Test metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Parameter Comparison -- understand the effect of different configurations
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare different hyperparameter configurations for sklearn LinearRegression.

    For vanilla LinearRegression, the main "hyperparameter" is feature selection
    (which features to include). This function demonstrates how the number of
    features affects model performance and training time.

    WHY feature selection matters: In real-world data, including irrelevant
    features can add noise without improving predictions. With OLS (no
    regularisation), irrelevant features increase variance of coefficient
    estimates (multicollinearity effect).
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Feature Selection & Intercept Trade-offs")
    print("=" * 80)

    n_features = X_train.shape[1]

    configs = {
        "All Features + Intercept": {
            "fit_intercept": True,
            "n_selected": n_features,
            # WHY: The standard approach. Uses all available features.
            # TRADE-OFF: Maximum expressiveness but may overfit if features
            # are noisy or correlated. Coefficient estimates may have high variance.
        },
        "All Features, No Intercept": {
            "fit_intercept": False,
            "n_selected": n_features,
            # WHY: Forces the regression line through the origin.
            # TRADE-OFF: Only appropriate when y=0 when all x=0 (rare in practice).
            # Usually hurts performance because it constrains the model unnecessarily.
            # Included to demonstrate why intercept is almost always needed.
        },
        "Top 50% Features": {
            "fit_intercept": True,
            "n_selected": max(1, n_features // 2),
            # WHY: Simulates feature selection. In our synthetic data, exactly
            # half the features are informative, so this should perform well.
            # TRADE-OFF: Reduces model complexity but may lose information.
            # In practice, you would use statistical tests or LASSO to select features.
        },
        "Top 25% Features (Aggressive Reduction)": {
            "fit_intercept": True,
            "n_selected": max(1, n_features // 4),
            # WHY: Very aggressive feature selection. Tests model robustness
            # with minimal inputs.
            # TRADE-OFF: Likely drops some informative features, increasing bias.
            # But dramatically reduces overfitting risk with very few training samples.
        },
    }

    results = {}
    for name, params in configs.items():
        n_sel = params["n_selected"]
        fit_intercept = params["fit_intercept"]

        # Select the first n_sel features (in practice, you would use
        # importance ranking or correlation analysis to choose which features).
        idx = list(range(n_sel))

        start_time = time.time()
        model = train(X_train[:, idx], y_train, fit_intercept=fit_intercept)
        train_time = time.time() - start_time

        metrics = validate(model, X_val[:, idx], y_val)
        metrics["train_time"] = train_time
        metrics["n_features_used"] = n_sel
        results[name] = metrics

        print(f"\n{'=' * 60}")
        print(f"Config: {name}")
        print(f"{'=' * 60}")
        print(f"  Features used:  {n_sel}/{n_features}")
        print(f"  Fit intercept:  {fit_intercept}")
        print(f"  MSE:            {metrics['mse']:.6f}")
        print(f"  R2:             {metrics['r2']:.6f}")
        print(f"  Train Time:     {train_time:.6f}s")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<40} {'Features':>8} {'MSE':>10} {'R2':>10} {'Time(s)':>10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<40} {metrics['n_features_used']:>8} {metrics['mse']:>10.4f} {metrics['r2']:>10.4f} {metrics['train_time']:>10.6f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Dropping irrelevant features can IMPROVE performance (less noise).")
    print("  2. Removing the intercept almost always hurts performance.")
    print("  3. sklearn LinearRegression is extremely fast -- even with all features.")
    print("  4. Feature selection is the main 'tuning knob' for vanilla OLS.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Salary Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate linear regression on a salary prediction problem.

    DOMAIN CONTEXT: Predicting employee salaries is a common HR analytics task.
    Companies use models like this for salary benchmarking, pay equity analysis,
    and budgeting for new hires. The features represent attributes that
    typically influence compensation.

    WHY salary prediction: It is a relatable problem where the linear
    assumption is reasonable (more experience, education, and skills
    generally correlate linearly with higher pay, at least approximately).
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Employee Salary Prediction")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 2000

    # Generate realistic feature distributions for employee attributes.

    # Years of experience: Most employees have 0-30 years, exponentially distributed.
    # WHY exponential: Workforce is younger-skewed; fewer people have 30+ years.
    years_experience = np.random.exponential(8, n_samples).clip(0, 40)

    # Education level: 1=High School, 2=Bachelor's, 3=Master's, 4=PhD
    # WHY ordinal encoding: Education has a natural ordering where more education
    # generally means higher salary. This is a simplification; real models might
    # use one-hot encoding for non-linear effects.
    education_level = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.45, 0.30, 0.10])

    # Number of technical skills (programming languages, frameworks, etc.)
    num_skills = np.random.poisson(5, n_samples).clip(1, 15)

    # Company size (number of employees): larger companies tend to pay more.
    company_size = np.random.lognormal(6, 1.5, n_samples).clip(10, 100000)

    # Industry sector encoded as numeric (simplified)
    # 1=Nonprofit, 2=Government, 3=Retail, 4=Manufacturing, 5=Finance, 6=Tech
    industry_code = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.10, 0.15, 0.20, 0.25, 0.25])

    # City cost of living index (100 = national average)
    cost_of_living = np.random.normal(100, 25, n_samples).clip(60, 200)

    feature_names = [
        "years_experience", "education_level", "num_skills",
        "company_size", "industry_code", "cost_of_living"
    ]

    X = np.column_stack([
        years_experience, education_level, num_skills,
        company_size, industry_code, cost_of_living
    ])

    # True salary model: base salary + contributions from each feature.
    # These coefficients represent the "true" value of each attribute in the job market.
    true_coefs = np.array([
        2500.0,     # $2,500 per year of experience
        12000.0,    # $12K per education level step
        1500.0,     # $1,500 per technical skill
        0.05,       # $0.05 per company employee (larger company premium)
        5000.0,     # $5K per industry tier
        300.0,      # $300 per cost-of-living index point
    ])

    base_salary = 30000  # Minimum base salary
    noise = np.random.normal(0, 8000, n_samples)  # $8K noise
    y = base_salary + X @ true_coefs + noise

    print(f"\nDataset: {n_samples} employees with {len(feature_names)} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Target: Annual salary (USD)")
    print(f"Salary range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Mean salary: ${y.mean():,.0f}")

    # Standard ML pipeline: split, scale, train, evaluate.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train with sklearn's LinearRegression.
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_s, y_train)

    # Evaluate on test set.
    y_pred = model.predict(X_test_s)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    print(f"\n--- Model Performance ---")
    print(f"  RMSE: ${metrics['rmse']:,.2f}  (average prediction error)")
    print(f"  MAE:  ${metrics['mae']:,.2f}")
    print(f"  R2:   {metrics['r2']:.4f}  (explains {metrics['r2']*100:.1f}% of salary variance)")

    # Interpret learned coefficients.
    print(f"\n--- Learned Feature Weights (Standardised) ---")
    print(f"{'Feature':<20} {'Weight':>12} {'Impact':>10}")
    print("-" * 45)
    for i, name in enumerate(feature_names):
        w = model.coef_[i]
        impact = "Strong +" if w > 5000 else "Moderate +" if w > 1000 else "Weak +" if w > 0 else "Negative"
        print(f"{name:<20} {w:>12.2f} {impact:>10}")

    print(f"\n  Base salary estimate: ${model.intercept_:,.2f}")

    # Sample predictions
    print(f"\n--- Sample Predictions ---")
    print(f"{'Actual Salary':>15} {'Predicted':>12} {'Error':>10}")
    print("-" * 40)
    for i in range(min(8, len(y_test))):
        print(f"${y_test[i]:>13,.0f} ${y_pred[i]:>10,.0f} ${y_pred[i]-y_test[i]:>8,.0f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective that minimises validation MSE.

    For LinearRegression, the main tunable aspect is feature selection
    (how many features to include). This is wrapped in an Optuna objective
    to demonstrate the Optuna pattern even for simple models.
    """
    # Whether to fit an intercept term.
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    # Feature selection: how many of the available features to use.
    # WHY this is an Optuna parameter: In practice, feature selection is
    # one of the most impactful "hyperparameters" for linear models.
    n_features = X_train.shape[1]
    selected_features = trial.suggest_int("n_selected_features", 1, n_features)
    feature_indices = list(range(selected_features))

    # Train and evaluate with the selected features.
    model = train(X_train[:, feature_indices], y_train, fit_intercept=fit_intercept)
    metrics = validate(model, X_val[:, feature_indices], y_val)
    return metrics["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Run an Optuna study and return the best trial.

    WHY Optuna for LinearRegression: Even though OLS has no "traditional"
    hyperparameters, feature selection and intercept fitting ARE choices
    that affect performance. Optuna automates this search.
    """
    study = optuna.create_study(direction="minimize", study_name="linear_regression_sklearn")
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    logger.info("Optuna best trial: %s  value=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune Hyperparameter Search
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    """Ray Tune trainable function.

    WHY separate from Optuna: Ray Tune parallelises across CPU cores,
    making it faster for expensive training runs. For LinearRegression
    (which trains in milliseconds), the overhead of Ray is not worth it,
    but the pattern is useful to learn for more complex models.
    """
    fit_intercept = config["fit_intercept"]
    n_selected = config["n_selected_features"]
    idx = list(range(n_selected))

    model = train(X_train[:, idx], y_train, fit_intercept=fit_intercept)
    metrics = validate(model, X_val[:, idx], y_val)
    tune.report(mse=metrics["mse"], rmse=metrics["rmse"], mae=metrics["mae"], r2=metrics["r2"])


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    """Launch a Ray Tune hyperparameter search."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    n_features = X_train.shape[1]
    search_space = {
        "fit_intercept": tune.choice([True, False]),
        "n_selected_features": tune.randint(1, n_features + 1),
    }

    trainable = tune.with_parameters(
        _ray_trainable,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="mse",
            mode="min",
            num_samples=num_samples,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best config: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the complete sklearn LinearRegression tutorial pipeline."""
    print("=" * 70)
    print("Linear Regression - scikit-learn")
    print("=" * 70)

    # 1. Generate synthetic data.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(
        n_samples=1000, n_features=10, noise=0.1,
    )

    # 2. Educational parameter comparison.
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # 3. Real-world salary prediction demo.
    real_world_demo()

    # 4. Optuna optimisation.
    print("\n--- Optuna Hyperparameter Optimisation ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=20)
    print(f"Best Optuna params : {study.best_trial.params}")
    print(f"Best Optuna MSE    : {study.best_trial.value:.6f}")

    # 5. Ray Tune optimisation.
    print("\n--- Ray Tune Hyperparameter Search ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best Ray config    : {ray_best.config}")
    print(f"Best Ray MSE       : {ray_best.metrics['mse']:.6f}")

    # 6. Train with best params from Optuna.
    best_params = study.best_trial.params
    n_sel = best_params.pop("n_selected_features", X_train.shape[1])
    idx = list(range(n_sel))
    model = train(X_train[:, idx], y_train, **best_params)

    # 7. Validate.
    print("\n--- Validation Results ---")
    val_metrics = validate(model, X_val[:, idx], y_val)
    for k, v in val_metrics.items():
        print(f"  {k:6s}: {v:.6f}")

    # 8. Test.
    print("\n--- Test Results ---")
    test_metrics = test(model, X_test[:, idx], y_test)
    for k, v in test_metrics.items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
