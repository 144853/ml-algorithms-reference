"""
Linear Regression - NumPy From-Scratch Implementation
=====================================================

COMPLETE ML TUTORIAL: This file teaches you linear regression from the ground up,
implementing everything with raw NumPy so you can see exactly what happens inside
the algorithm. No black boxes.

Theory & Mathematics:
    Linear Regression models the target y as a linear combination of input
    features:

        y = X @ w + b

    This implementation provides TWO solvers built entirely with NumPy:

    1. **Normal Equation (closed-form)**
       The optimal weights minimise the sum of squared residuals:

           L(w) = (1/2n) * ||y - Xw||^2

       Setting dL/dw = 0 gives the analytic solution:

           w* = (X^T X)^{-1} X^T y

       We use np.linalg.lstsq for numerical stability (SVD-based) rather
       than explicit inversion.

    2. **Batch Gradient Descent**
       Iteratively updates weights by moving in the direction of steepest
       descent:

           w := w - lr * (1/n) * X^T (X @ w - y)
           b := b - lr * (1/n) * sum(X @ w - y)

       Convergence depends on learning rate and number of iterations.

    Key Metrics:
        MSE  = (1/n) sum (y_i - y_hat_i)^2
        RMSE = sqrt(MSE)
        MAE  = (1/n) sum |y_i - y_hat_i|
        R^2  = 1 - SS_res / SS_tot

Business Use Cases:
    - Educational tool for understanding regression internals
    - Lightweight inference on edge devices without heavy ML libraries
    - Baseline model for quick comparison in custom pipelines
    - Embedded systems where only NumPy is available

Advantages:
    - Full transparency: every computation is visible and debuggable
    - No external ML library dependency beyond NumPy
    - Gradient descent variant enables online / mini-batch learning extensions
    - Easy to extend to weighted least squares or custom loss functions

Disadvantages:
    - Manual implementation lacks production optimisations (parallelism, caching)
    - Gradient descent requires careful learning-rate tuning
    - Normal equation is O(n_features^3) and infeasible for very high-dimensional data
    - No built-in regularisation (see Ridge / Lasso implementations)

Hyperparameters:
    - solver: "normal" | "gd" -- which solver to use
    - lr (float): Learning rate for gradient descent. Default 0.01.
    - n_iters (int): Number of gradient descent iterations. Default 1000.
    - fit_intercept (bool): Whether to fit an intercept term. Default True.
"""

# ---------------------------------------------------------------------------
# Imports -- each one is here for a specific reason
# ---------------------------------------------------------------------------

# logging: We use Python's built-in logging module instead of print statements.
# WHY: logging is configurable (DEBUG/INFO/WARNING), can be redirected to files,
# and is the industry standard for production code. print() is not filterable.
import logging

# time: Used to measure wall-clock training time for benchmarking.
# WHY: Understanding training time helps us decide between normal equation
# (fast for small data) vs gradient descent (scalable to large data).
import time

# functools.partial: Creates a new function with some arguments pre-filled.
# WHY: Optuna's objective function signature is fixed (trial only), but we
# need to pass data arrays. partial() lets us "bake in" X_train, y_train etc.
from functools import partial

# numpy: The foundation of numerical computing in Python.
# WHY: NumPy provides vectorised array operations that are 10-100x faster
# than pure Python loops. It also provides linear algebra routines (lstsq)
# that we need for the normal equation solver.
import numpy as np

# optuna: A hyperparameter optimisation framework that uses Bayesian methods.
# WHY: Unlike grid search (which tries every combination) or random search,
# Optuna uses Tree-structured Parzen Estimators (TPE) to intelligently explore
# the hyperparameter space, finding good configs in fewer trials.
import optuna

# ray and ray.tune: Distributed hyperparameter tuning framework.
# WHY: Ray Tune can parallelise trials across multiple CPU cores or machines.
# For production ML pipelines, this dramatically reduces wall-clock search time.
import ray
from ray import tune

# sklearn utilities -- we use these for data generation and evaluation only.
# WHY: We implement the MODEL from scratch, but reuse sklearn's data generation
# (make_regression) and metrics (MSE, R2) because those are not the learning focus.
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure the root logger to show timestamps and severity levels.
# WHY: format includes time, level, and message so we can trace execution
# order and distinguish INFO from DEBUG messages in long training runs.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create a module-level logger with __name__ so log messages show which file they came from.
# WHY: In multi-file projects, __name__ = "linear_regression_numpy" helps us
# filter logs. Using logger.info() instead of logging.info() attaches the module name.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model class (pure NumPy) -- the heart of this tutorial
# ---------------------------------------------------------------------------

class LinearRegressionNumpy:
    """Linear Regression implemented from scratch with NumPy.

    This class provides two solvers:
    - "normal": Closed-form solution via the normal equation (fast, exact)
    - "gd": Batch gradient descent (iterative, scalable to streaming data)

    Both solve the same Ordinary Least Squares (OLS) problem:
        minimize (1/2n) * ||y - Xw||^2
    """

    def __init__(self, solver="normal", lr=0.01, n_iters=1000, fit_intercept=True):
        # solver: Determines which algorithm to use for finding optimal weights.
        # WHY we offer both: Normal equation is O(p^3) where p = number of features.
        # For p > 10,000 features, this becomes slow. Gradient descent is O(n*p) per
        # iteration, making it more practical for high-dimensional data.
        self.solver = solver

        # lr (learning rate): Controls the step size in gradient descent.
        # WHY 0.01 default: Too large (e.g., 1.0) causes oscillation/divergence.
        # Too small (e.g., 1e-6) converges extremely slowly. 0.01 is a safe default
        # for standardised features (zero mean, unit variance).
        self.lr = lr

        # n_iters: Maximum number of gradient descent iterations.
        # WHY 1000 default: For well-scaled data, GD typically converges within
        # 100-500 iterations. 1000 gives headroom for harder problems.
        self.n_iters = n_iters

        # fit_intercept: Whether to learn a bias term b in y = Xw + b.
        # WHY True default: Almost all real-world data has a non-zero mean target.
        # Setting this False forces the regression line through the origin,
        # which is rarely appropriate unless you have centered data.
        self.fit_intercept = fit_intercept

        # weights: Will hold the learned coefficient vector w after training.
        # Initialised to None; set during fit().
        self.weights = None

        # bias: The intercept term b. Initialised to 0.0.
        # WHY 0.0: For the normal equation, it gets computed exactly.
        # For GD, starting at 0 is fine since gradients will adjust it.
        self.bias = 0.0

        # loss_history: Stores MSE at each GD iteration for convergence analysis.
        # WHY: Plotting this curve reveals if learning rate is too high (oscillation),
        # too low (slow descent), or just right (smooth exponential decay).
        self.loss_history = []

    def fit(self, X, y):
        """Fit the model to training data using the configured solver.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Each row is one data point, each column is one feature.
            y: Target vector of shape (n_samples,).

        Returns:
            self: The fitted model (allows method chaining like sklearn).
        """
        # Extract dimensions from the input matrix.
        # n_samples: number of training examples (rows)
        # n_features: number of input features (columns)
        # WHY: We need n_features to initialise weight vectors of correct size,
        # and n_samples to normalise gradients (divide by n for mean).
        n_samples, n_features = X.shape

        # Dispatch to the appropriate solver method.
        # WHY separate methods: Keeps each algorithm's logic clean and testable.
        # The fit() method acts as a router / strategy pattern.
        if self.solver == "normal":
            self._fit_normal(X, y)
        elif self.solver == "gd":
            self._fit_gd(X, y)
        else:
            # Fail fast with a clear error message if solver name is misspelled.
            raise ValueError(f"Unknown solver: {self.solver}")

        # Return self to allow method chaining: model.fit(X, y).predict(X_test)
        return self

    def _fit_normal(self, X, y):
        """Solve using the Normal Equation (closed-form OLS solution).

        The normal equation finds w* that minimises ||y - Xw||^2:
            w* = (X^T X)^{-1} X^T y

        We use np.linalg.lstsq which is numerically more stable than
        explicit inversion because it uses SVD decomposition internally.
        """
        if self.fit_intercept:
            # Augment X with a column of ones to absorb the bias term.
            # WHY: Instead of solving for w and b separately, we can treat
            # the bias as an extra weight by adding a constant feature = 1.
            # This turns y = Xw + b into y = X_aug @ theta where
            # theta = [w1, w2, ..., wp, b]. Mathematically equivalent but simpler.
            X_aug = np.column_stack([X, np.ones(X.shape[0])])
        else:
            # No augmentation needed -- the model passes through the origin.
            X_aug = X

        # np.linalg.lstsq: Solves the least-squares problem min ||X_aug @ theta - y||^2
        # WHY lstsq instead of np.linalg.inv(X.T @ X) @ X.T @ y:
        #   1. lstsq uses SVD, which is numerically stable even when X^T X is
        #      nearly singular (condition number ~ 10^15).
        #   2. Explicit inversion can amplify floating-point errors catastrophically.
        #   3. lstsq also handles underdetermined systems (more features than samples).
        # rcond=None: Suppresses a deprecation warning; uses machine precision.
        # Returns: (solution, residuals, rank, singular_values) -- we only need solution.
        theta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        if self.fit_intercept:
            # The last element of theta is the bias (coefficient of the ones column).
            # All other elements are the feature weights.
            self.weights = theta[:-1]
            self.bias = theta[-1]
        else:
            self.weights = theta
            self.bias = 0.0

    def _fit_gd(self, X, y):
        """Solve using Batch Gradient Descent.

        At each iteration, we compute the gradient of MSE loss with respect
        to all weights simultaneously, then take a step in the negative
        gradient direction (steepest descent).

        Gradient derivation:
            L = (1/2n) ||y - Xw||^2
            dL/dw = -(1/n) X^T (y - Xw) = (1/n) X^T (Xw - y)
            dL/db = (1/n) sum(Xw - y)
        """
        # Get dataset dimensions for gradient normalisation.
        n_samples, n_features = X.shape

        # Initialise weights to zeros. This is a common choice for linear regression.
        # WHY zeros: For linear models, the loss surface is convex (bowl-shaped),
        # so the starting point does not matter -- GD will always find the global
        # minimum. Starting at zero is simple and symmetric.
        # NOTE: For neural networks, zero initialisation would be catastrophic
        # (all neurons would learn the same thing), but for linear regression it is fine.
        self.weights = np.zeros(n_features)

        # Initialise bias to zero. Will be updated each iteration if fit_intercept=True.
        self.bias = 0.0

        # Clear any previous loss history from prior fits.
        self.loss_history = []

        # Main gradient descent loop: repeat for n_iters iterations.
        # WHY fixed iterations (not convergence-based): Simpler to implement.
        # In practice, you could add early stopping when ||dw|| < tolerance.
        for i in range(self.n_iters):
            # STEP 1: Forward pass -- compute predictions with current weights.
            # X @ self.weights computes the dot product of each sample with the
            # weight vector, then we add the bias.
            # Shape: (n_samples,) = (n_samples, n_features) @ (n_features,) + scalar
            y_pred = X @ self.weights + self.bias

            # STEP 2: Compute residuals (prediction errors).
            # residual[i] = y_pred[i] - y[i]
            # Positive residual means we over-predicted, negative means under-predicted.
            residual = y_pred - y

            # STEP 3: Compute gradient of MSE w.r.t. weights.
            # dL/dw = (1/n) * X^T @ residual
            # WHY X.T @ residual: Each column of X^T dot-producted with the residual
            # gives how much that feature "contributed" to the error. This is the
            # direction of steepest ascent; we negate it by subtracting.
            # Division by n_samples makes the gradient a MEAN, not a SUM, so the
            # learning rate does not need to change with dataset size.
            dw = (1.0 / n_samples) * (X.T @ residual)

            # STEP 4: Compute gradient of MSE w.r.t. bias.
            # dL/db = (1/n) * sum(residual)
            # WHY sum: The bias affects every prediction equally (it is added to
            # every output), so its gradient is the mean of all residuals.
            db = (1.0 / n_samples) * np.sum(residual)

            # STEP 5: Update weights by stepping in the negative gradient direction.
            # w_new = w_old - lr * gradient
            # WHY subtract: The gradient points UPHILL (direction of increasing loss).
            # We want to go DOWNHILL, so we subtract.
            self.weights -= self.lr * dw

            # STEP 6: Update bias only if we are fitting an intercept.
            # WHY conditional: If fit_intercept=False, the bias stays at 0.
            if self.fit_intercept:
                self.bias -= self.lr * db

            # STEP 7: Record the MSE loss for this iteration.
            # WHY track loss: Monitoring loss across iterations lets us detect:
            #   - Convergence (loss plateaus)
            #   - Divergence (loss increases -- learning rate too high)
            #   - Slow progress (loss decreases very gradually -- lr too low)
            mse = np.mean(residual ** 2)
            self.loss_history.append(mse)

            # Log progress every 20% of total iterations to avoid console spam.
            # WHY max(1, ...): Prevents division by zero when n_iters < 5.
            if (i + 1) % max(1, self.n_iters // 5) == 0:
                logger.debug("GD iter %d/%d  MSE=%.6f", i + 1, self.n_iters, mse)

    def predict(self, X):
        """Generate predictions for new data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        # Simple linear prediction: y_hat = X @ w + b
        # WHY this is all we need: The model IS this equation. All the complexity
        # is in finding the right w and b during training.
        return X @ self.weights + self.bias


# ---------------------------------------------------------------------------
# Data Generation -- creating synthetic data for controlled experiments
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data and split into train/val/test sets.

    WHY synthetic data: Allows us to control the true relationship, noise level,
    and number of informative features. This makes it easy to verify our
    implementation is correct (we know the ground truth).

    Args:
        n_samples: Total number of data points to generate.
        n_features: Total number of features (some will be informative, rest noise).
        noise: Standard deviation of Gaussian noise added to the target.
        random_state: Seed for reproducibility. Same seed = same data every time.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # make_regression generates data from a true linear model: y = Xw* + noise
    # n_informative: Only half the features actually contribute to y.
    # WHY n_informative < n_features: Tests if the model can handle irrelevant
    # features. In real data, not every column matters.
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        # max(1, ...) ensures at least 1 informative feature even with n_features=1
        n_informative=max(1, n_features // 2),
        noise=noise,
        random_state=random_state,
    )

    # Split into 60% train, 20% validation, 20% test.
    # WHY three splits (not just train/test):
    #   - Train: used to fit model parameters (weights)
    #   - Validation: used to tune hyperparameters (lr, n_iters) without touching test
    #   - Test: final unbiased performance estimate, used ONCE at the end
    # Using validation for hyperparameter tuning prevents "information leakage"
    # from the test set into our model selection process.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # Create a StandardScaler to normalise features to zero mean and unit variance.
    # WHY: Gradient descent converges MUCH faster when features are on the same scale.
    # Without scaling, a feature with range [0, 1000000] would dominate gradients
    # over a feature with range [0, 1], causing zigzagging in the loss landscape.
    # The normal equation also benefits: condition number of X^T X decreases.
    scaler = StandardScaler()

    # fit_transform on training data: computes mean/std FROM training set,
    # then applies the transformation. We ONLY fit on training data.
    # WHY: If we fit on all data, the scaler would "see" validation/test statistics,
    # leaking future information into training. This is called DATA LEAKAGE and
    # would give overly optimistic evaluation results.
    X_train = scaler.fit_transform(X_train)

    # transform (not fit_transform) on val/test: uses the SAME mean/std from training.
    # WHY: In production, we will not have access to future data statistics.
    # The model must use only training-time statistics for normalisation.
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Log dataset sizes for verification.
    logger.info("Data: train=%d  val=%d  test=%d  features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test -- the standard ML workflow functions
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hyperparams):
    """Train the from-scratch linear regression model.

    Args:
        X_train: Training feature matrix (n_samples, n_features).
        y_train: Training target vector (n_samples,).
        **hyperparams: Keyword arguments for model configuration.
            solver (str): "normal" or "gd". Default "normal".
            lr (float): Learning rate for GD. Default 0.01.
            n_iters (int): GD iterations. Default 1000.
            fit_intercept (bool): Learn bias term. Default True.

    Returns:
        Trained LinearRegressionNumpy model.
    """
    # Create model instance with hyperparameters from the search space.
    # .get() provides defaults so the function works even with empty hyperparams.
    # WHY .get() with defaults: Makes the function robust to partial configs,
    # which is common during hyperparameter search where not all params are specified.
    model = LinearRegressionNumpy(
        solver=hyperparams.get("solver", "normal"),
        lr=hyperparams.get("lr", 0.01),
        n_iters=hyperparams.get("n_iters", 1000),
        fit_intercept=hyperparams.get("fit_intercept", True),
    )

    # Fit the model to training data. This is where the actual learning happens.
    model.fit(X_train, y_train)

    # Log what we trained for debugging and experiment tracking.
    logger.info("Trained (solver=%s)  bias=%.6f", model.solver, model.bias)
    return model


def _compute_metrics(y_true, y_pred):
    """Compute standard regression evaluation metrics.

    WHY these four metrics:
    - MSE: Primary optimisation target. Penalises large errors quadratically.
    - RMSE: Same units as target variable. More interpretable than MSE.
    - MAE: Robust to outliers. Gives average absolute error in target units.
    - R^2: Scale-free measure of how much variance the model explains.
           R^2 = 1.0 is perfect, R^2 = 0.0 means no better than predicting the mean.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def validate(model, X_val, y_val):
    """Evaluate model on validation data.

    WHY separate from test: Validation is used for hyperparameter selection.
    We can evaluate on validation data many times without biasing our final
    test score. The test set is used exactly ONCE.

    Returns:
        Dictionary with mse, rmse, mae, and r2.
    """
    metrics = _compute_metrics(y_val, model.predict(X_val))
    logger.info("Validation: %s", metrics)
    return metrics


def test(model, X_test, y_test):
    """Evaluate model on held-out test data.

    WHY: This gives the final, unbiased estimate of model performance.
    It simulates how the model will perform on truly unseen real-world data.

    Returns:
        Dictionary with mse, rmse, mae, and r2.
    """
    metrics = _compute_metrics(y_test, model.predict(X_test))
    logger.info("Test: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Parameter Comparison -- understand the effect of different configurations
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare different hyperparameter configurations to understand trade-offs.

    This function runs the model with several carefully chosen configurations
    and prints a comparison table. Each configuration represents a different
    strategy for a different scenario.

    WHY this function exists: New ML practitioners often do not understand WHY
    certain hyperparameter values work. This function makes the trade-offs
    visible and concrete with real numbers.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Understanding Hyperparameter Trade-offs")
    print("=" * 80)

    # Define configurations that represent distinct strategies.
    # Each config is chosen to illustrate a specific point about the algorithm.
    configs = {
        "Normal Equation (Exact Solution)": {
            "solver": "normal",
            "fit_intercept": True,
            # WHY this config: The normal equation gives the EXACT OLS solution
            # in one shot. No learning rate or iterations needed. Best when
            # n_features < ~10,000 and the data fits in memory.
            # TRADE-OFF: O(n_features^3) complexity. Infeasible for >10K features.
        },
        "GD Conservative (Low LR, Many Iters)": {
            "solver": "gd",
            "lr": 0.001,        # Small steps -- very cautious
            "n_iters": 3000,    # Many iterations to compensate for small steps
            "fit_intercept": True,
            # WHY this config: Low learning rate with many iterations is the
            # "safe" approach. Almost guaranteed to converge, but slow.
            # TRADE-OFF: Takes 3x longer than balanced, but handles noisy/
            # ill-conditioned data better. Like driving slowly on a windy road.
        },
        "GD Balanced (Moderate Settings)": {
            "solver": "gd",
            "lr": 0.01,         # Standard learning rate
            "n_iters": 1000,    # Reasonable iteration count
            "fit_intercept": True,
            # WHY this config: Good default for standardised data. Balances
            # convergence speed with stability.
            # TRADE-OFF: May not fully converge on difficult problems, but
            # usually gets very close. Good starting point for most tasks.
        },
        "GD Aggressive (High LR, Few Iters)": {
            "solver": "gd",
            "lr": 0.1,          # Large steps -- aggressive
            "n_iters": 200,     # Few iterations -- hopes to converge fast
            "fit_intercept": True,
            # WHY this config: When data is clean and well-scaled, a high
            # learning rate can converge in very few iterations, saving time.
            # TRADE-OFF: Risk of oscillation or divergence on noisy data.
            # The loss curve may bounce around instead of smoothly decreasing.
            # Like sprinting -- fast if the path is clear, dangerous if not.
        },
    }

    # Store results for comparison table.
    results = {}

    for name, params in configs.items():
        # Time each configuration to understand computational trade-offs.
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        # Evaluate on validation set (NOT test set -- that is reserved for final eval).
        metrics = validate(model, X_val, y_val)
        metrics["train_time"] = train_time
        results[name] = metrics

        # Print detailed results for this configuration.
        print(f"\n{'=' * 60}")
        print(f"Config: {name}")
        print(f"{'=' * 60}")
        print(f"  Parameters: {params}")
        print(f"  MSE:        {metrics['mse']:.6f}")
        print(f"  RMSE:       {metrics['rmse']:.6f}")
        print(f"  MAE:        {metrics['mae']:.6f}")
        print(f"  R2:         {metrics['r2']:.6f}")
        print(f"  Train Time: {train_time:.4f}s")

    # Print a summary comparison table for quick reference.
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<45} {'MSE':>10} {'R2':>10} {'Time(s)':>10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<45} {metrics['mse']:>10.4f} {metrics['r2']:>10.4f} {metrics['train_time']:>10.4f}")

    # Print educational takeaways.
    print(f"\n{'=' * 80}")
    print("KEY TAKEAWAYS:")
    print("  1. Normal Equation gives exact results instantly for small datasets.")
    print("  2. Conservative GD is safest but slowest -- use for noisy data.")
    print("  3. Balanced GD is the best default starting point.")
    print("  4. Aggressive GD is fastest but risks divergence.")
    print("  5. For standard linear regression, Normal Equation is almost always best.")
    print(f"{'=' * 80}")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- House Price Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate linear regression on a realistic house price prediction problem.

    DOMAIN CONTEXT: Predicting residential house prices is one of the most
    classic regression problems. Real estate appraisers use similar models
    to estimate fair market value based on property characteristics.

    We simulate a dataset with named features that mimic real property data.
    The true relationship is linear (by construction), which is appropriate
    for linear regression. In practice, some relationships (e.g., location)
    are non-linear, requiring feature engineering or more complex models.
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: House Price Prediction")
    print("=" * 80)

    # Set random seed for reproducibility.
    np.random.seed(42)

    # Number of houses in our simulated dataset.
    n_samples = 1500

    # Generate realistic feature distributions for each property attribute.
    # Each feature is generated from a distribution that matches real-world data.

    # Square footage: Most homes are 800-3500 sq ft, normally distributed around 1800.
    # WHY this matters: Larger homes generally cost more. This is typically the
    # single strongest predictor of price.
    sqft = np.random.normal(1800, 500, n_samples).clip(500, 5000)

    # Number of bedrooms: Discrete values 1-6, with 3 being most common.
    # WHY: More bedrooms usually means higher price, but the effect per bedroom
    # diminishes (going from 1->2 bedrooms matters more than 5->6).
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.35, 0.25, 0.15, 0.05])

    # Number of bathrooms: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], n_samples, p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.1])

    # Age of home in years: Newer homes command a premium.
    # WHY log-normal: Most homes are relatively new, with a long tail of older homes.
    age_years = np.random.exponential(20, n_samples).clip(0, 100)

    # Lot size in acres: Affects price, especially in suburban areas.
    lot_acres = np.random.exponential(0.3, n_samples).clip(0.05, 5.0)

    # Distance to city center in miles: Closer = more expensive (usually).
    dist_to_city = np.random.exponential(10, n_samples).clip(0.5, 50)

    # Garage capacity (0-3 cars).
    garage_cars = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.4, 0.2])

    # School rating (1-10 scale): Higher rated school districts increase home values.
    school_rating = np.random.uniform(1, 10, n_samples)

    # Stack all features into a matrix.
    feature_names = [
        "sqft", "bedrooms", "bathrooms", "age_years",
        "lot_acres", "dist_to_city", "garage_cars", "school_rating"
    ]

    X = np.column_stack([sqft, bedrooms, bathrooms, age_years,
                         lot_acres, dist_to_city, garage_cars, school_rating])

    # Generate target (house price) from a TRUE linear model.
    # These coefficients represent the true "value" of each feature.
    # In reality, these would be unknown -- we are learning them.
    true_coefficients = np.array([
        150.0,      # $150 per square foot
        15000.0,    # $15K per bedroom
        20000.0,    # $20K per bathroom
        -1500.0,    # -$1500 per year of age (older = cheaper)
        50000.0,    # $50K per acre of lot
        -3000.0,    # -$3K per mile from city center
        12000.0,    # $12K per garage space
        8000.0,     # $8K per school rating point
    ])

    # Base price + linear combination + random noise
    base_price = 50000  # Minimum base price for any property
    noise = np.random.normal(0, 20000, n_samples)  # $20K standard deviation noise
    y = base_price + X @ true_coefficients + noise

    # Print dataset summary.
    print(f"\nDataset: {n_samples} houses with {len(feature_names)} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Target: House price (USD)")
    print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Mean price: ${y.mean():,.0f}")

    # Split into train/val/test with the same methodology as our main pipeline.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale features. Critical for gradient descent to work well.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train with normal equation (best for this small dataset).
    model = train(X_train_scaled, y_train, solver="normal", fit_intercept=True)

    # Evaluate on test set.
    test_metrics = _compute_metrics(y_test, model.predict(X_test_scaled))

    print(f"\n--- Model Performance ---")
    print(f"  MSE:  {test_metrics['mse']:,.2f}")
    print(f"  RMSE: {test_metrics['rmse']:,.2f}  (average prediction error in dollars)")
    print(f"  MAE:  {test_metrics['mae']:,.2f}")
    print(f"  R2:   {test_metrics['r2']:.4f}  (explains {test_metrics['r2']*100:.1f}% of price variance)")

    # Show learned coefficients vs true coefficients.
    # WHY: This validates that our implementation correctly recovers the true
    # relationship. The learned coefficients should be close to the true ones
    # (scaled by the StandardScaler).
    print(f"\n--- Learned Feature Importances ---")
    print(f"{'Feature':<20} {'Learned Weight':>15} {'Direction':>10}")
    print("-" * 50)
    for i, name in enumerate(feature_names):
        w = model.weights[i]
        direction = "+" if w > 0 else "-"
        print(f"{name:<20} {w:>15.2f} {direction:>10}")

    print(f"\n  Intercept (base price estimate): ${model.bias:,.2f}")

    # Show example predictions.
    print(f"\n--- Sample Predictions vs Actual ---")
    print(f"{'Actual':>12} {'Predicted':>12} {'Error':>12}")
    print("-" * 40)
    preds = model.predict(X_test_scaled)
    for i in range(min(8, len(y_test))):
        error = preds[i] - y_test[i]
        print(f"${y_test[i]:>10,.0f} ${preds[i]:>10,.0f} ${error:>10,.0f}")

    return model, test_metrics


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimisation
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function that returns validation MSE to minimise.

    WHY Optuna: It uses TPE (Tree-structured Parzen Estimators) which builds
    a probabilistic model of the hyperparameter-to-loss mapping and suggests
    promising configurations. This is much more efficient than random search.

    Args:
        trial: Optuna trial object that suggests hyperparameter values.
        X_train, y_train: Training data (pre-bound via functools.partial).
        X_val, y_val: Validation data for evaluation.

    Returns:
        Validation MSE (lower is better; Optuna will minimise this).
    """
    # suggest_categorical: Picks from a finite set of options.
    # WHY both solvers: We want Optuna to discover whether normal equation
    # or gradient descent works better for this particular dataset.
    solver = trial.suggest_categorical("solver", ["normal", "gd"])

    # suggest_float with log=True: Samples from a log-uniform distribution.
    # WHY log scale: Learning rates span orders of magnitude (0.0001 to 1.0).
    # Uniform sampling would waste most trials on the [0.5, 1.0] range.
    # Log-uniform gives equal probability to each order of magnitude.
    lr = trial.suggest_float("lr", 1e-4, 1.0, log=True) if solver == "gd" else 0.01

    # suggest_int with step: Only considers multiples of 100.
    # WHY step=100: Avoids wasting trials on tiny differences (999 vs 1000 iters).
    n_iters = trial.suggest_int("n_iters", 100, 3000, step=100) if solver == "gd" else 1000

    # Whether to fit an intercept term.
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    # Train and evaluate with these hyperparameters.
    model = train(X_train, y_train, solver=solver, lr=lr, n_iters=n_iters, fit_intercept=fit_intercept)
    metrics = validate(model, X_val, y_val)

    # Return MSE as the objective value. Optuna will try to minimise this.
    return metrics["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Run an Optuna hyperparameter search study.

    Args:
        n_trials: Number of hyperparameter configurations to try.
            WHY 30: Usually enough for TPE to find a good region. More trials
            give diminishing returns for simple models like linear regression.

    Returns:
        The completed Optuna study object.
    """
    # Create a study that MINIMISES the objective (MSE).
    # study_name: Useful for persistent storage and resuming studies.
    study = optuna.create_study(direction="minimize", study_name="linreg_numpy")

    # optimize() runs the objective function n_trials times.
    # partial() pre-fills the data arguments so the objective signature matches
    # what Optuna expects: objective(trial) -> float.
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Log the best result found.
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune -- Distributed Hyperparameter Search
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    """Ray Tune trainable function.

    WHY a separate function: Ray Tune serialises this function and ships it
    to worker processes. It must be a standalone function (not a method or lambda).
    tune.report() sends the metrics back to the Ray Tune controller.
    """
    model = train(X_train, y_train, **config)
    metrics = validate(model, X_val, y_val)
    # Report metrics back to Ray Tune for comparison across trials.
    tune.report(**metrics)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    """Launch a Ray Tune hyperparameter search.

    WHY Ray Tune when we already have Optuna: Ray Tune excels at DISTRIBUTED
    parallel search across multiple machines. Optuna is sequential by default.
    In production, you might use both: Optuna for local prototyping, Ray Tune
    for large-scale distributed search.

    Args:
        num_samples: Number of random configurations to try.
    """
    # Initialise Ray if not already running.
    # ignore_reinit_error=True: Prevents crash if Ray was already initialised.
    # log_to_driver=False: Reduces console noise from Ray's internal logging.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Define the hyperparameter search space.
    # tune.choice: Picks uniformly from a list.
    # tune.loguniform: Samples from log-uniform distribution (good for learning rates).
    search_space = {
        "solver": tune.choice(["normal", "gd"]),
        "lr": tune.loguniform(1e-4, 1.0),
        "n_iters": tune.choice([200, 500, 1000, 2000]),
        "fit_intercept": tune.choice([True, False]),
    }

    # tune.with_parameters: Efficiently passes large data arrays to workers.
    # WHY: Without this, Ray would serialise X_train etc. for every trial,
    # which is wasteful. with_parameters shares the data in shared memory.
    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    )

    # Create and run the tuner.
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="mse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()

    # Get the best result from all trials.
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results


# ---------------------------------------------------------------------------
# Main Pipeline -- orchestrates everything
# ---------------------------------------------------------------------------

def main():
    """Main entry point that runs the complete ML pipeline.

    Pipeline steps:
    1. Generate synthetic data
    2. Compare parameter configurations (educational)
    3. Run real-world demo (house prices)
    4. Run Optuna hyperparameter optimisation
    5. Run Ray Tune hyperparameter search
    6. Train final model with best params
    7. Evaluate on validation and test sets
    """
    print("=" * 70)
    print("Linear Regression - NumPy From Scratch")
    print("=" * 70)

    # Step 1: Generate synthetic data for the main pipeline.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Educational parameter comparison.
    # This runs BEFORE hyperparameter search to build intuition about
    # what the parameters do and how they affect performance.
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 3: Real-world demo with named features and domain context.
    real_world_demo()

    # Step 4: Optuna hyperparameter optimisation.
    print("\n--- Optuna Hyperparameter Optimisation ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=20)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    # Step 5: Ray Tune hyperparameter search.
    print("\n--- Ray Tune Hyperparameter Search ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    # Step 6: Train final model with the best parameters found by Optuna.
    # WHY Optuna over Ray: For this simple model, both find similar results.
    # We use Optuna's result here, but you could compare both.
    best_p = study.best_trial.params
    model = train(X_train, y_train, **best_p)

    # Step 7: Final evaluation.
    print("\n--- Validation Results ---")
    for k, v in validate(model, X_val, y_val).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n--- Test Results ---")
    for k, v in test(model, X_test, y_test).items():
        print(f"  {k:6s}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
