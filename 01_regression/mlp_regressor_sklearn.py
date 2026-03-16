"""
MLP Regressor - scikit-learn Implementation
============================================

COMPLETE ML TUTORIAL: This file teaches Multi-Layer Perceptron (MLP) regression
using scikit-learn's MLPRegressor.  An MLP is a fully-connected feed-forward
neural network that can learn NON-LINEAR mappings -- unlike linear / Ridge /
Lasso which are restricted to linear decision boundaries.

Theory & Mathematics:
    A Multi-Layer Perceptron (MLP) Regressor is a fully-connected
    feed-forward neural network for regression tasks.  It consists of:

        Input layer  -->  Hidden layer(s)  -->  Output layer

    Each hidden unit computes:

        h = activation(W_h @ x + b_h)

    and the output layer produces a scalar prediction:

        y_hat = W_out @ h_last + b_out

    The network is trained by minimising a loss function (typically MSE)
    via back-propagation and a first-order optimiser (SGD, Adam, L-BFGS).

    Universal Approximation Theorem:
        A single hidden layer with sufficiently many units can approximate
        any continuous function to arbitrary precision.  In practice,
        multiple narrower layers often generalise better because deeper
        networks learn hierarchical feature representations.

    Activation Functions:
        - ReLU:    f(z) = max(0, z)           -- most common, mitigates vanishing gradients
        - tanh:    f(z) = (e^z - e^{-z}) / (e^z + e^{-z})  -- zero-centred
        - logistic: f(z) = 1 / (1 + e^{-z})  -- bounded, can cause vanishing gradients

    scikit-learn's ``MLPRegressor`` wraps this in a convenient API with
    built-in regularisation, early stopping, and multiple solvers.

Business Use Cases:
    - Non-linear regression when linear models underfit
    - Tabular data modelling (customer churn prediction, pricing)
    - Quick neural-network baseline without PyTorch boilerplate
    - Ensemble component alongside tree-based models

Advantages:
    - Can capture non-linear relationships automatically
    - Minimal code with scikit-learn API (fit/predict)
    - Built-in L2 regularisation and early stopping
    - Multiple solver options (adam, sgd, lbfgs)

Disadvantages:
    - No GPU support in scikit-learn (CPU only)
    - Limited architecture flexibility (no custom layers, skip connections)
    - Sensitive to feature scaling (requires standardisation)
    - Harder to interpret than linear models
    - Convergence can be slow for large datasets

Hyperparameters:
    - hidden_layer_sizes (tuple): Architecture. Default (100,).
    - activation (str): "relu", "tanh", "logistic". Default "relu".
    - solver (str): "adam", "sgd", "lbfgs". Default "adam".
    - alpha (float): L2 regularisation. Default 0.0001.
    - learning_rate_init (float): Initial LR for SGD/Adam. Default 0.001.
    - max_iter (int): Maximum epochs. Default 200.
    - batch_size (int): Mini-batch size. Default "auto".
    - early_stopping (bool): Use validation-based stopping. Default False.
"""

import logging  # WHY: Structured logging beats print() for production code.
import time  # WHY: Measure wall-clock training time to compare architectures.
from functools import partial  # WHY: Freeze data args into Optuna/Ray callables.

import numpy as np  # WHY: Numerical operations for metrics, data manipulation.
import optuna  # WHY: Bayesian hyperparameter optimisation via TPE sampler.
import ray  # WHY: Distributed computing framework for parallel HP search.
from ray import tune  # WHY: Ray's hyperparameter tuning module.
from sklearn.datasets import make_regression  # WHY: Generate synthetic regression data.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # WHY: Standard regression metrics.
from sklearn.model_selection import train_test_split  # WHY: Stratified data splitting.
from sklearn.neural_network import MLPRegressor  # WHY: scikit-learn's built-in neural network.
from sklearn.preprocessing import StandardScaler  # WHY: MLPs require scaled features (mean=0, std=1).

# WHY: Configure logging at module level so all functions share the same format.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # WHY: Module-level logger avoids hard-coded logger names.


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic data with 60/20/20 split and standard scaling.

    WHY 60/20/20: We need three splits -- train for fitting, validation for
    hyperparameter tuning (Optuna/Ray), and test for final unbiased evaluation.
    60/20/20 gives enough training data while keeping meaningful val/test sets.
    """
    # WHY make_regression: Generates linear ground truth. MLPs should recover
    # this perfectly, but the point is demonstrating the training pipeline.
    # n_informative=n_features//2 means half the features are noise.
    X, y = make_regression(
        n_samples=n_samples,  # WHY: 1000 is enough for tabular MLP demos.
        n_features=n_features,  # WHY: 10 features -- small enough to iterate fast.
        n_informative=max(1, n_features // 2),  # WHY: Half are informative, half are noise.
        noise=noise,  # WHY: Irreducible error. Tests whether MLP can handle noise.
        random_state=random_state,  # WHY: Reproducibility across runs.
    )

    # WHY test_size=0.4: First split off 40% (which becomes 20% val + 20% test).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state,
    )

    # WHY test_size=0.5: Split the remaining 40% equally into val (20%) and test (20%).
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,
    )

    # WHY StandardScaler: MLPs are EXTREMELY sensitive to feature scale.
    # Without scaling, features with large magnitudes dominate gradients,
    # causing slow convergence or divergence. StandardScaler transforms
    # each feature to mean=0, std=1.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # WHY fit_transform: Learn mean/std from TRAINING data only.
    X_val = scaler.transform(X_val)  # WHY transform only: Prevent data leakage from val into training stats.
    X_test = scaler.transform(X_test)  # WHY transform only: Same -- test must be truly unseen.

    # WHY log shapes: Verify data split sizes are correct before training.
    logger.info("Data: train=%d val=%d test=%d features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Train sklearn MLPRegressor model.

    scikit-learn's MLPRegressor handles forward pass, backpropagation, and
    weight updates internally. It supports three solvers:
      - adam: Adaptive moment estimation. Best general-purpose solver.
      - sgd:  Stochastic gradient descent. More control (momentum, LR schedules).
      - lbfgs: Quasi-Newton method. Best for small datasets (< ~10K samples).
    """
    # hidden_layer_sizes: Tuple defining the architecture.
    # (100,) = one hidden layer with 100 units.
    # (64, 32) = two hidden layers with 64 and 32 units respectively.
    # WHY tuple: scikit-learn convention -- each element is a layer width.
    hidden_sizes = hp.get("hidden_layer_sizes", (100,))
    if isinstance(hidden_sizes, list):
        hidden_sizes = tuple(hidden_sizes)  # WHY: sklearn expects tuple, but Optuna/Ray may pass list.

    # activation: Non-linear function applied after each hidden layer.
    # WHY relu default: ReLU avoids vanishing gradients (gradient is 1 for z>0)
    # and is computationally cheap (just a max operation).
    activation = hp.get("activation", "relu")

    # solver: Optimisation algorithm for weight updates.
    # WHY adam default: Adam adapts learning rates per-parameter using first
    # and second moment estimates, making it robust to hyperparameter choices.
    solver = hp.get("solver", "adam")

    # alpha: L2 regularisation strength (weight decay).
    # WHY 0.0001: Light regularisation prevents overfitting without
    # under-fitting. L2 penalises large weights: Loss += alpha * ||W||^2.
    alpha = hp.get("alpha", 0.0001)

    # learning_rate_init: Starting learning rate for Adam/SGD.
    # WHY 0.001: Standard starting point for Adam. Too high = divergence,
    # too low = slow convergence and getting stuck in local minima.
    lr_init = hp.get("learning_rate_init", 0.001)

    # max_iter: Maximum number of training epochs.
    # WHY 200: Usually sufficient for convergence on small-medium datasets.
    # Increase for large/complex datasets, decrease with early_stopping.
    max_iter = hp.get("max_iter", 200)

    # batch_size: Number of samples per gradient update.
    # WHY "auto": sklearn picks min(200, n_samples), a reasonable default.
    # Smaller batches add noise (regularisation effect) but are slower.
    batch_size = hp.get("batch_size", "auto")

    # early_stopping: Hold out a fraction of training data for validation.
    # Stop when validation score stops improving for n_iter_no_change epochs.
    # WHY useful: Prevents overfitting without needing to set max_iter exactly.
    early_stopping = hp.get("early_stopping", False)

    # WHY validation_fraction=0.15: When early_stopping=True, sklearn uses
    # this fraction of training data as internal validation. 15% is a good
    # balance -- enough to detect overfitting, not so much that training suffers.
    model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,  # Architecture definition.
        activation=activation,  # Non-linearity between layers.
        solver=solver,  # Optimisation algorithm.
        alpha=alpha,  # L2 penalty on weights.
        learning_rate_init=lr_init,  # Initial learning rate.
        max_iter=max_iter,  # Maximum epochs.
        batch_size=batch_size,  # Mini-batch size.
        early_stopping=early_stopping,  # Validation-based stopping.
        validation_fraction=0.15 if early_stopping else 0.1,  # Internal val split for early stopping.
        random_state=42,  # WHY: Reproducibility of weight initialisation and shuffling.
    )

    # WHY model.fit: scikit-learn handles the entire training loop internally:
    # weight initialisation, forward pass, loss computation, backprop, updates.
    model.fit(X_train, y_train)

    # WHY log n_iter_: Shows how many epochs were actually needed. If equal
    # to max_iter, the model may not have converged -- consider increasing max_iter.
    logger.info("MLP trained  hidden=%s  activation=%s  solver=%s  iters=%d",
                hidden_sizes, activation, solver, model.n_iter_)
    return model


def _metrics(y_true, y_pred):
    """Compute regression metrics.

    WHY four metrics: MSE is the training objective, RMSE is in original units,
    MAE is robust to outliers, R2 measures explained variance (1.0 = perfect).
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),  # WHY: Primary loss, differentiable.
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),  # WHY: Same units as target.
        "mae": mean_absolute_error(y_true, y_pred),  # WHY: Robust to outliers.
        "r2": r2_score(y_true, y_pred),  # WHY: 1.0 = perfect, 0 = baseline mean predictor.
    }


def validate(model, X_val, y_val):
    """Evaluate on validation set (used for hyperparameter tuning)."""
    m = _metrics(y_val, model.predict(X_val))  # WHY: predict() runs forward pass on val data.
    logger.info("Validation: %s", m)
    return m


def test(model, X_test, y_test):
    """Evaluate on test set (final unbiased evaluation)."""
    m = _metrics(y_test, model.predict(X_test))  # WHY: Test set is NEVER used for tuning.
    logger.info("Test: %s", m)
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison -- Shallow vs. Deep & Solver/Activation Strategies
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare MLP architectures, solvers, and activations.

    This function demonstrates the KEY decisions when configuring an MLP:
    1. DEPTH: How many hidden layers? (shallow vs. deep)
    2. WIDTH: How many neurons per layer?
    3. SOLVER: Which optimiser? (adam vs. sgd vs. lbfgs)
    4. ACTIVATION: Which non-linearity? (relu vs. tanh)

    WHY these comparisons matter:
    - Shallow networks (1 layer) can approximate any function (Universal
      Approximation Theorem) but may need exponentially many neurons.
    - Deep networks (2-3 layers) learn hierarchical features more efficiently
      but are harder to train (vanishing gradients, more hyperparameters).
    - The solver determines HOW weights are updated -- lbfgs is excellent
      for small datasets, adam is the robust general-purpose choice.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: sklearn MLP - Architecture & Solver Strategy")
    print("=" * 80)

    # WHY these four configs: They cover the key architectural trade-offs.
    configs = {
        # CONFIG 1: Shallow Wide -- single hidden layer with many neurons.
        # WHY: Tests the Universal Approximation Theorem directly. One wide
        # layer can fit any function, but may need many parameters and can
        # memorise noise (overfit) rather than learn generalisable patterns.
        "Shallow Wide (100,)": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
        },
        # CONFIG 2: Standard Deep -- two layers, typical architecture.
        # WHY: Two hidden layers can learn hierarchical features. The first
        # layer learns low-level combinations, the second combines them.
        # 64->32 is a "funnel" shape that progressively compresses information.
        "Standard Deep (64,32)": {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
        },
        # CONFIG 3: Deep Narrow with tanh -- three layers, different activation.
        # WHY tanh: Zero-centred output means next layer receives balanced inputs.
        # WHY three layers: Tests if additional depth helps on this data.
        # Three layers can learn more abstract features but risk vanishing
        # gradients (especially with tanh, whose gradients saturate near +/-1).
        "Deep Narrow tanh (128,64,32)": {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.001,  # WHY stronger alpha: Deeper nets have more params, need more regularisation.
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
        },
        # CONFIG 4: L-BFGS solver -- quasi-Newton method.
        # WHY lbfgs: For small datasets (< ~10K samples), L-BFGS uses second-
        # order curvature information to find better optima faster than Adam.
        # It processes the ENTIRE dataset per iteration (no mini-batches),
        # so it's memory-intensive but converges in fewer iterations.
        "L-BFGS (64,32)": {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "lbfgs",
            "alpha": 0.001,
            "max_iter": 300,
            # NOTE: early_stopping is not supported with lbfgs solver.
            "early_stopping": False,
        },
    }

    results = {}
    for name, params in configs.items():
        # WHY time each config: Training time varies dramatically across
        # architectures (deeper = slower) and solvers (lbfgs = different scaling).
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        metrics = validate(model, X_val, y_val)
        # WHY track n_iter_: Shows convergence speed. If n_iter_ == max_iter,
        # the model may need more epochs (did not converge).
        metrics["n_iters"] = model.n_iter_
        metrics["train_time"] = train_time
        # WHY count params: Total learnable parameters = sum of all weight matrices
        # and bias vectors. More params = more capacity but more overfitting risk.
        n_params = sum(c.size for c in model.coefs_) + sum(i.size for i in model.intercepts_)
        metrics["n_params"] = n_params
        results[name] = metrics

        arch_str = params["hidden_layer_sizes"]
        print(f"\n  {name}:")
        print(f"    Params={n_params}  Iters={model.n_iter_}  Time={train_time:.2f}s")
        print(f"    MSE={metrics['mse']:.6f}  R2={metrics['r2']:.6f}")

    # WHY comparison table: Side-by-side view makes it easy to see which
    # architecture/solver combination gives the best accuracy-efficiency trade-off.
    print(f"\n{'=' * 90}")
    print(f"{'Configuration':<30} {'Solver':>6} {'Params':>7} {'Iters':>6} "
          f"{'Time':>6} {'MSE':>10} {'R2':>8}")
    print("-" * 85)
    for name, m in results.items():
        solver = configs[name]["solver"]
        print(f"{name:<30} {solver:>6} {m['n_params']:>7} {m['n_iters']:>6} "
              f"{m['train_time']:>5.2f}s {m['mse']:>10.4f} {m['r2']:>8.4f}")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Shallow wide networks have many params but may not learn hierarchical features.")
    print("  2. Deep funnel architectures (e.g., 64->32) often generalise better with fewer params.")
    print("  3. L-BFGS converges in fewer iterations but each iteration is expensive (full-batch).")
    print("  4. tanh activation gives zero-centred outputs but risks vanishing gradients in deep nets.")
    print("  5. early_stopping prevents overfitting by monitoring an internal validation score.")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Power Plant Electrical Output Prediction
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate MLP regression for predicting power plant output.

    DOMAIN CONTEXT: Combined Cycle Power Plants (CCPPs) generate electricity
    using both gas and steam turbines. The electrical output depends on ambient
    conditions (temperature, pressure, humidity) and turbine settings in a
    NON-LINEAR way -- hot days reduce gas turbine efficiency, while humid air
    affects cooling tower performance.

    WHY MLP: Linear regression CANNOT capture the non-linear interactions
    between temperature, pressure, and humidity that determine power output.
    An MLP can learn these complex interactions automatically through its
    hidden layers.

    FEATURES:
        - ambient_temp (C):  Outdoor temperature affects gas turbine inlet air density.
        - exhaust_vacuum (cmHg): Steam turbine backpressure. Higher vacuum = more efficiency.
        - ambient_pressure (mbar): Atmospheric pressure affects air mass flow.
        - relative_humidity (%): Humidity affects cooling tower and condenser performance.
        - wind_speed (m/s): Wind over cooling towers improves heat rejection.
        - fuel_flow_rate (kg/s): Natural gas flow rate to combustion chamber.

    TARGET: net_output_mw -- Net electrical output in megawatts.

    NON-LINEARITIES:
        - Temperature and pressure interact multiplicatively (air density = P / (R*T)).
        - Humidity has a threshold effect (condensation changes heat transfer).
        - Exhaust vacuum has diminishing returns (below a certain vacuum, gains plateau).
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Power Plant Electrical Output Prediction")
    print("=" * 80)

    # WHY seed: Ensures this demo produces identical results every run.
    np.random.seed(42)

    n_samples = 800  # WHY: Typical size for industrial sensor datasets.
    feature_names = [
        "ambient_temp",       # Celsius, affects air density and turbine efficiency.
        "exhaust_vacuum",     # cmHg, steam turbine backpressure.
        "ambient_pressure",   # mbar, atmospheric pressure at plant location.
        "relative_humidity",  # %, humidity level affecting cooling.
        "wind_speed",         # m/s, affects cooling tower performance.
        "fuel_flow_rate",     # kg/s, natural gas input rate.
    ]
    n_features = len(feature_names)

    # WHY randn: Standardised features (mean=0, std=1) simulate pre-processed
    # sensor data. In reality these would come from a SCADA/DCS system.
    X = np.random.randn(n_samples, n_features)

    # WHY complex formula: This mimics real thermodynamic relationships.
    # Power output is NOT a linear combination of inputs.
    y = (
        50.0  # Base load in MW.
        + 5.0 * X[:, 0]  # Linear effect of temperature.
        - 3.0 * X[:, 0] ** 2  # Quadratic: extreme temps reduce output (non-linear!).
        - 4.0 * X[:, 1]  # Higher exhaust vacuum (lower pressure) = more output.
        + 2.0 * X[:, 2]  # Higher ambient pressure = denser air = more output.
        - 1.5 * X[:, 3]  # High humidity reduces cooling efficiency.
        + 1.0 * X[:, 4]  # Wind helps cooling towers.
        + 8.0 * X[:, 5]  # Fuel flow has strong positive effect.
        + 2.0 * X[:, 0] * X[:, 2]  # INTERACTION: temp * pressure (air density).
        - 1.5 * np.abs(X[:, 3]) * X[:, 1]  # INTERACTION: humidity * vacuum.
        + np.random.normal(0, 2.0, n_samples)  # Measurement noise (2 MW std).
    )

    print(f"\nDataset: {n_samples} hourly readings from a combined cycle power plant")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Target: net electrical output (MW)")
    print(f"Non-linearities: quadratic temp effect, temp*pressure interaction, humidity*vacuum interaction")

    # WHY 70/15/15 split: Slightly more training data for this demo since
    # the non-linear relationships need more samples to learn.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
    )

    # WHY scale: Even though we generated standardised data, in production
    # sensor readings have different scales (temp in C, pressure in mbar, etc.).
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # --- Compare Linear vs. MLP ---
    # WHY this comparison: Demonstrates that MLP captures non-linearities
    # that a linear model completely misses.
    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    lr_model.fit(X_train_s, y_train)
    lr_pred = lr_model.predict(X_test_s)
    lr_metrics = _metrics(y_test, lr_pred)

    print(f"\n--- Linear Regression Baseline ---")
    print(f"  RMSE: {lr_metrics['rmse']:.4f} MW   R2: {lr_metrics['r2']:.4f}")

    # WHY (64, 32): Two-layer funnel architecture. First layer (64 units)
    # learns feature interactions, second layer (32 units) combines them.
    # This is enough capacity for 6 features with quadratic interactions.
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  # WHY: Two layers to capture interactions.
        activation="relu",  # WHY: ReLU can approximate any piecewise-linear function.
        solver="adam",  # WHY: Robust default for medium-sized datasets.
        alpha=0.001,  # WHY: Moderate L2 regularisation to prevent overfitting.
        learning_rate_init=0.001,  # WHY: Standard Adam learning rate.
        max_iter=500,  # WHY: More epochs since non-linear targets need longer training.
        early_stopping=True,  # WHY: Automatically stops when val loss plateaus.
        validation_fraction=0.15,  # WHY: 15% of training data for internal validation.
        random_state=42,
    )
    mlp_model.fit(X_train_s, y_train)
    mlp_pred = mlp_model.predict(X_test_s)
    mlp_metrics = _metrics(y_test, mlp_pred)

    print(f"\n--- MLP Regressor (64, 32) ---")
    print(f"  RMSE: {mlp_metrics['rmse']:.4f} MW   R2: {mlp_metrics['r2']:.4f}")
    print(f"  Epochs used: {mlp_model.n_iter_}")

    # WHY compute improvement: Quantifies the benefit of using a non-linear model.
    improvement = (lr_metrics['rmse'] - mlp_metrics['rmse']) / lr_metrics['rmse'] * 100
    print(f"\n  MLP improvement over Linear: {improvement:.1f}% lower RMSE")

    # --- Feature importance via permutation (approximate) ---
    # WHY permutation importance: MLPs don't have coefficients like linear models.
    # We estimate importance by shuffling each feature and measuring the
    # increase in error. Larger increase = more important feature.
    print(f"\n--- Approximate Feature Importance (Permutation) ---")
    base_mse = mlp_metrics["mse"]
    importances = []
    for i, fname in enumerate(feature_names):
        # WHY: Shuffle one feature at a time to break its relationship with y.
        X_test_perm = X_test_s.copy()
        np.random.shuffle(X_test_perm[:, i])  # WHY: Random permutation destroys signal.
        perm_mse = mean_squared_error(y_test, mlp_model.predict(X_test_perm))
        imp = perm_mse - base_mse  # WHY: Increase in error = importance of that feature.
        importances.append((fname, imp))

    # WHY sort: Most important features first for easy reading.
    importances.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Feature':<20} {'MSE Increase':>12} {'Importance':>10}")
    print("-" * 45)
    for fname, imp in importances:
        bar = "#" * int(min(imp / max(i[1] for i in importances) * 20, 20))
        print(f"{fname:<20} {imp:>12.4f} {bar}")

    print(f"\nNOTE: fuel_flow_rate and ambient_temp should rank highest because they")
    print(f"have the largest true coefficients (8.0 and 5.0) plus non-linear effects.")

    return mlp_model, mlp_metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for MLP hyperparameter optimisation.

    WHY Optuna: Bayesian optimisation (TPE sampler) is much more efficient
    than grid search or random search. It builds a probabilistic model of
    the objective function and focuses trials on promising regions.
    """
    # WHY suggest n_layers first: Architecture search is hierarchical --
    # we first decide depth, then width of each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)  # WHY 1-3: Covers shallow to moderately deep.
    layers = []
    for i in range(n_layers):
        # WHY step=16: Keeps layer sizes as multiples of 16, which aligns
        # well with CPU SIMD and GPU warp sizes for efficient computation.
        layers.append(trial.suggest_int(f"n_units_l{i}", 16, 256, step=16))

    hp = {
        "hidden_layer_sizes": tuple(layers),  # WHY tuple: sklearn requirement.
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        # WHY not logistic: Logistic sigmoid causes severe vanishing gradients
        # in multi-layer networks, making it a poor choice for modern MLPs.
        "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
        # WHY log scale: L2 penalty spans orders of magnitude (1e-6 to 1.0).
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 0.1, log=True),
        # WHY log scale: LR also spans orders of magnitude.
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
    }
    model = train(X_train, y_train, **hp)
    # WHY return MSE: Optuna minimises this objective. MSE is smooth and
    # differentiable, which helps the TPE sampler model the objective surface.
    return validate(model, X_val, y_val)["mse"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """Run Optuna hyperparameter search.

    WHY create_study with minimize: Lower MSE = better regression performance.
    """
    study = optuna.create_study(
        direction="minimize",  # WHY: We want to minimise MSE (lower = better).
        study_name="mlp_sklearn",  # WHY: Named study for logging/dashboard.
    )
    study.optimize(
        # WHY partial: Freeze the data arguments so Optuna only passes the trial.
        partial(optuna_objective, X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val),
        n_trials=n_trials,  # WHY 30: Good balance between search quality and time.
        show_progress_bar=True,  # WHY: Visual feedback during long searches.
    )
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    """Ray Tune trainable function.

    WHY separate function: Ray serialises this and runs it in separate processes.
    tune.report() sends metrics back to the Ray scheduler for comparison.
    """
    model = train(X_train, y_train, **config)
    metrics = validate(model, X_val, y_val)
    tune.report(**metrics)  # WHY: Report ALL metrics so Ray can log them.


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10):
    """Run Ray Tune hyperparameter search.

    WHY Ray Tune: Distributes trials across multiple processes/machines.
    Supports advanced scheduling (ASHA, PBT) and search algorithms.
    """
    if not ray.is_initialized():
        # WHY ignore_reinit_error: Prevents crash if Ray was already initialised.
        # WHY log_to_driver=False: Reduces console noise from Ray workers.
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # WHY these ranges: Cover the most common MLP configurations.
    search_space = {
        "hidden_layer_sizes": tune.choice([
            (64,),  # Shallow: single layer baseline.
            (128,),  # Shallow: wider single layer.
            (64, 32),  # Standard: two-layer funnel.
            (128, 64),  # Standard: wider two-layer funnel.
            (128, 64, 32),  # Deep: three-layer progressive funnel.
        ]),
        "activation": tune.choice(["relu", "tanh"]),
        "solver": tune.choice(["adam", "lbfgs"]),
        # WHY not sgd in Ray: SGD requires more careful LR tuning, which
        # adds complexity without clear benefit in a broad search.
        "alpha": tune.loguniform(1e-5, 0.1),
        "learning_rate_init": tune.loguniform(1e-4, 0.05),
        "max_iter": tune.choice([200, 300, 500]),
        "early_stopping": tune.choice([True, False]),
    }

    # WHY with_parameters: Passes large numpy arrays by reference, not serialisation.
    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="mse",  # WHY: Optimise for MSE (same as Optuna).
            mode="min",  # WHY: Lower MSE is better.
            num_samples=num_samples,  # WHY: Number of random configs to try.
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="mse", mode="min")
    logger.info("Ray best: %s  mse=%.6f", best.config, best.metrics["mse"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full MLP sklearn pipeline.

    Pipeline order:
    1. Generate and split data
    2. Compare parameter sets (architecture/solver/activation study)
    3. Real-world demo (power plant output prediction)
    4. Optuna hyperparameter search (Bayesian optimisation)
    5. Ray Tune hyperparameter search (distributed random search)
    6. Train final model with best hyperparameters
    7. Evaluate on validation and test sets
    """
    print("=" * 70)
    print("MLP Regressor - scikit-learn")
    print("=" * 70)

    # WHY: Generate synthetic data first for the HP tuning pipeline.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # WHY compare first: Understanding architecture trade-offs BEFORE running
    # automated HP search helps interpret the search results.
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # WHY real_world_demo: Shows practical application with named features
    # and domain context, making the tutorial more concrete.
    real_world_demo()

    # WHY Optuna: Bayesian optimisation to find best hyperparameters.
    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=20)
    print(f"Best params : {study.best_trial.params}")
    print(f"Best MSE    : {study.best_trial.value:.6f}")

    # WHY Ray Tune: Alternative distributed search for comparison.
    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=8)
    ray_best = ray_results.get_best_result(metric="mse", mode="min")
    print(f"Best config : {ray_best.config}")
    print(f"Best MSE    : {ray_best.metrics['mse']:.6f}")

    # WHY reconstruct: Optuna stores layer sizes as separate parameters
    # (n_layers, n_units_l0, n_units_l1, ...) but our train() function
    # expects a single hidden_layer_sizes tuple. We must rebuild it.
    best_p = dict(study.best_trial.params)
    n_layers = best_p.pop("n_layers", 1)  # WHY pop: Remove from dict so it's not passed to train().
    layers = []
    for i in range(n_layers):
        key = f"n_units_l{i}"
        layers.append(best_p.pop(key, 64))  # WHY default 64: Fallback if key missing.
    best_p["hidden_layer_sizes"] = tuple(layers)

    # WHY retrain: The Optuna study found the best hyperparameters, but we
    # need to retrain a fresh model with those params on the full training set.
    model = train(X_train, y_train, **best_p)

    # WHY print validation AND test: Validation confirms the HP search results,
    # test gives the final unbiased performance estimate.
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
