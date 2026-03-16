"""
ARIMA-like Model - PyTorch Implementation
==========================================

Theory & Mathematics:
    This module implements an ARIMA-inspired model in PyTorch where the AR and MA
    coefficients are learnable parameters optimised via gradient descent.

    Model Architecture:
        y_hat_t = bias
                  + sum_{i=1..p} phi_i * y_{t-i}           (AR component)
                  + sum_{j=1..q} theta_j * eps_{t-j}       (MA component)

    Unlike classical ARIMA which uses MLE or Yule-Walker equations, this
    implementation:
        1. Treats AR coefficients (phi) and MA coefficients (theta) as
           torch.nn.Parameter objects
        2. Optimises them via backpropagation with MSE loss
        3. Handles differencing as a preprocessing step
        4. Computes residuals (innovations) in the forward pass

    Training Process:
        - Input: windowed sequences of length max(p, q) + 1
        - The model predicts the next value given past values and past residuals
        - Loss: MSE between predicted and actual next value
        - Optimiser: Adam with configurable learning rate

    Differencing:
        - Applied as preprocessing (not learnable)
        - Reversed during inference to produce forecasts on original scale

Business Use Cases:
    - Demand forecasting with automatic feature learning
    - Financial time series prediction
    - Anomaly detection in time series via residual analysis
    - Transfer learning: pre-train AR/MA on one series, fine-tune on another

Advantages:
    - Gradient-based optimisation allows flexible extensions
    - Can be combined with neural network layers (hybrid models)
    - GPU acceleration for large datasets
    - Easy to add regularisation (L1/L2 on coefficients)
    - Batched training for efficiency

Disadvantages:
    - May overfit without proper regularisation
    - No closed-form solution (iterative optimisation)
    - MA component requires sequential residual computation
    - Less interpretable than classical ARIMA estimation
    - Sensitive to learning rate and initialisation

Key Hyperparameters:
    - p (int): AR order
    - d (int): Differencing order (preprocessing)
    - q (int): MA order
    - lr (float): Learning rate for Adam optimiser
    - epochs (int): Number of training epochs
    - weight_decay (float): L2 regularisation

References:
    - Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis.
    - Salinas, D. et al. (2020). DeepAR: Probabilistic forecasting with autoregressive
      recurrent networks. International Journal of Forecasting.
"""

import numpy as np  # NumPy for data generation, metrics computation, and array manipulation
import pandas as pd  # Pandas for date handling in real-world demo
import warnings  # Suppress PyTorch and convergence warnings during HPO
from typing import Dict, Tuple, Any, Optional  # Type hints for function signatures

import torch  # PyTorch core: tensors, autograd engine
import torch.nn as nn  # Neural network module: nn.Parameter, nn.Module, loss functions
import torch.optim as optim  # Optimizers: Adam with weight decay for L2 regularisation

import optuna  # Bayesian hyperparameter optimization (TPE sampler)
import ray  # Ray distributed computing for parallel HPO
from ray import tune  # Ray Tune search spaces and trial management

# Suppress all warnings during HPO sweeps where many configs may produce warnings
# WHY: PyTorch may warn about gradient issues for degenerate AR/MA orders,
# and these warnings are not actionable during automated search
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_points: int = 1000,
    freq: str = "D",
    seasonal_period: int = 7,
    trend_slope: float = 0.02,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic time series with trend, seasonality, and noise.

    Returns:
        Tuple of (train, val, test) as 1-D numpy arrays.
        Split is 70/15/15 chronological.
    """
    # Set random seed for reproducibility
    # WHY: Ensures identical data across runs for fair hyperparameter comparisons
    np.random.seed(seed)

    # Create float64 time index for numerical precision
    # WHY: float64 prevents overflow for large n_points and ensures smooth
    # multiplication with trend_slope
    t = np.arange(n_points, dtype=np.float64)

    # Linear trend component: simulates growth over time
    # WHY: ARIMA handles this via differencing (d parameter); the PyTorch version
    # applies differencing as a preprocessing step before gradient-based fitting
    trend = trend_slope * t

    # Sinusoidal seasonality: periodic oscillation with amplitude 2.0
    # WHY: Creates a repeating pattern every seasonal_period steps; this tests
    # whether the learned AR coefficients can capture periodic behavior
    seasonality = 2.0 * np.sin(2 * np.pi * t / seasonal_period)

    # Gaussian white noise with specified standard deviation
    # WHY: Adds realistic randomness; the MA component should learn to smooth
    # the noise by incorporating past forecast errors
    noise = np.random.normal(0, noise_std, n_points)

    # Combine all components with a positive baseline
    # WHY: baseline=10.0 keeps values positive, mimicking real-world business metrics
    values = 10.0 + trend + seasonality + noise

    # Chronological split: 70/15/15
    # WHY: Time series must be split in order (not randomly) to prevent future leakage
    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)

    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and SMAPE forecast evaluation metrics.

    WHY four metrics: Each captures a different aspect of forecast quality -
    MAE for interpretability, RMSE for optimization, MAPE for scale-free
    comparison, SMAPE for symmetric percentage error.
    """
    # Convert to float64 for numerical precision in divisions
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Compute element-wise forecast errors
    errors = actual - predicted

    # MAE: robust to outliers, in original units
    mae = np.mean(np.abs(errors))

    # RMSE: penalises large errors, differentiable (good as loss function)
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE: scale-independent percentage error; mask zeros to avoid division by zero
    mask = actual != 0
    mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100 if mask.any() else np.inf

    # SMAPE: symmetric percentage error bounded [0, 200%]
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    denom = np.where(denom == 0, 1e-10, denom)  # Prevent division by zero
    smape = np.mean(np.abs(errors) / denom) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


# ---------------------------------------------------------------------------
# PyTorch ARIMA Model
# ---------------------------------------------------------------------------

class ARIMANet(nn.Module):
    """ARIMA-like model with learnable AR and MA coefficients.

    Architecture:
        prediction = bias + AR_component + MA_component
        AR: dot product of p past values with learned phi coefficients
        MA: dot product of q past residuals with learned theta coefficients

    WHY PyTorch for ARIMA:
        Traditional ARIMA uses closed-form solutions (Yule-Walker, MLE).
        The PyTorch version uses gradient descent, which:
        1. Can be extended with nonlinear layers (hybrid models)
        2. Supports GPU acceleration for large-scale forecasting
        3. Enables transfer learning (pretrain on one series, fine-tune on another)
        4. Can incorporate regularisation (L1/L2) on coefficients easily
    """

    def __init__(self, p: int = 2, q: int = 2):
        """Initialize ARIMANet with learnable AR and MA coefficients.

        Args:
            p: AR order - number of past values to use as predictors.
            q: MA order - number of past residuals (forecast errors) to use.
        """
        super().__init__()  # Initialize nn.Module parent class
        self.p = p  # Store AR order for use in forward pass
        self.q = q  # Store MA order for use in forward pass

        # AR coefficients as learnable parameters, initialized with small random values
        # WHY small initialization (0.01 scale): Large initial AR coefficients can cause
        # numerical instability in the first few training steps because predictions
        # depend multiplicatively on these coefficients
        self.ar_coeffs = nn.Parameter(torch.randn(p) * 0.01) if p > 0 else None

        # MA coefficients as learnable parameters, similarly initialized
        # WHY separate from AR: The MA component models residual autocorrelation,
        # which is structurally different from the AR component that models
        # value autocorrelation
        self.ma_coeffs = nn.Parameter(torch.randn(q) * 0.01) if q > 0 else None

        # Bias term (constant/intercept in the ARIMA equation)
        # WHY initialized to zero: The mean of differenced series is typically near
        # zero, so a zero initial bias is a reasonable starting point
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, y_history: torch.Tensor,
                residual_history: torch.Tensor) -> torch.Tensor:
        """Predict next value given past values and past residuals.

        Args:
            y_history: (batch, p) tensor of past differenced values.
                y_history[:, 0] is the oldest, y_history[:, -1] is the most recent.
            residual_history: (batch, q) tensor of past forecast errors.
                residual_history[:, 0] is the oldest, [:, -1] is most recent.

        Returns:
            (batch,) predicted values on the differenced scale.

        Mathematical operation:
            pred = bias + sum(phi_i * y_{t-i}) + sum(theta_j * eps_{t-j})
        """
        # Start with the bias term, expanded to match batch size
        # WHY expand: self.bias is a scalar parameter; we need one copy per
        # batch element for the addition to broadcast correctly
        pred = self.bias.expand(y_history.shape[0])

        if self.ar_coeffs is not None and self.p > 0:
            # AR component: dot product of past values with AR coefficients
            # WHY flip: ar_coeffs[0] should multiply the most recent value y_{t-1},
            # but y_history stores oldest first. Flipping ar_coeffs aligns them
            # so the matmul produces sum(phi_i * y_{t-i}) correctly
            ar_part = torch.matmul(y_history, self.ar_coeffs.flip(0))
            pred = pred + ar_part  # Add AR contribution to prediction

        if self.ma_coeffs is not None and self.q > 0:
            # MA component: dot product of past residuals with MA coefficients
            # WHY same flip logic: ma_coeffs[0] multiplies the most recent residual,
            # but residual_history stores oldest first
            ma_part = torch.matmul(residual_history, self.ma_coeffs.flip(0))
            pred = pred + ma_part  # Add MA contribution to prediction

        return pred


def _difference_series(series: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply differencing d times as a preprocessing step.

    Returns:
        Tuple of (differenced_series, original_series_copy).

    WHY preprocessing rather than learnable: Differencing is a fixed transformation
    (not learned), so it is applied before training. The original series is preserved
    for undifferencing forecasts back to the original scale.
    """
    result = series.copy()  # Copy to avoid modifying original data

    for _ in range(d):
        # First-order differencing: z_t = y_t - y_{t-1}
        # WHY: Each application removes one order of polynomial trend from the series
        result = np.diff(result)

    return result, series  # Return both differenced and original


def _undifference(forecasts: np.ndarray, last_value: float, d: int) -> np.ndarray:
    """Reverse d orders of differencing to recover original scale.

    Args:
        forecasts: Predictions on the differenced scale.
        last_value: Last observed value on the original scale (anchor for cumsum).
        d: Number of differencing operations to reverse.

    WHY cumulative sum: Differencing = subtraction, so inverse = cumulative addition.
    Starting from the last known original value, we cumulatively add the
    differenced forecasts to reconstruct predictions on the original scale.
    """
    result = forecasts.copy()  # Avoid modifying input

    for _ in range(d):
        # Prepend the anchor value, compute cumulative sum, remove anchor from output
        # WHY: cumsum([anchor, z1, z2, ...]) = [anchor, anchor+z1, anchor+z1+z2, ...]
        # Taking [1:] removes the anchor, leaving the undifferenced forecasts
        result = np.cumsum(np.concatenate([[last_value], result]))[1:]

    return result


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    p: int = 3,
    d: int = 1,
    q: int = 2,
    lr: float = 0.01,
    epochs: int = 100,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train ARIMA-like PyTorch model via gradient descent.

    Training procedure:
        1. Apply differencing as preprocessing
        2. For each timestep t (sequentially):
           a. Build y_history from the p values before t
           b. Build residual_history from the q residuals before t
           c. Forward pass: predict z_hat_t
           d. Compute MSE loss vs actual z_t
           e. Backward pass: compute gradients of loss w.r.t. AR/MA coefficients
           f. Update coefficients via Adam optimizer
           g. Record residual (z_t - z_hat_t) for subsequent MA computation

    WHY sequential training: The MA component requires residuals from previous
    timesteps, creating a sequential dependency. Unlike pure AR models which
    can be batched, the MA residual computation forces step-by-step processing.

    Returns:
        Dict with trained model, metadata, and training diagnostics.
    """
    # Select compute device (GPU if available, else CPU)
    # WHY: GPU acceleration provides 10-100x speedup for matrix operations,
    # though for small ARIMA models the benefit is modest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Apply differencing to remove trend
    # WHY: The AR/MA model assumes stationarity; differencing removes polynomial
    # trends so the model only needs to learn the stationary dynamics
    z, original = _difference_series(train_data, d)

    # Initialize the ARIMANet model and move to the selected device
    model = ARIMANet(p=p, q=q).to(device)

    # Adam optimizer with weight decay for L2 regularization on coefficients
    # WHY Adam: Adaptive learning rates per-parameter handle the different scales
    # of AR vs MA coefficients; weight_decay adds L2 penalty to prevent
    # coefficients from growing too large (overfitting)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # MSE loss function for regression
    # WHY MSE: Standard choice for continuous prediction; it is the negative
    # log-likelihood under Gaussian error assumptions, connecting to MLE theory
    criterion = nn.MSELoss()

    # Determine the minimum window size needed for history
    # WHY max(p, q): We need at least p past values for AR and q past residuals
    # for MA; the window must satisfy both requirements
    window = max(p, q)
    if window == 0:
        window = 1  # Minimum window of 1 for the bias-only model

    # Convert differenced series to a PyTorch tensor on the target device
    # WHY float32: Standard precision for PyTorch training; float64 is rarely
    # needed and doubles memory usage and computation time
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device)

    best_loss = float("inf")  # Track best training loss for reporting

    for epoch in range(epochs):
        model.train()  # Set model to training mode (enables dropout if any)
        total_loss = 0.0  # Accumulate loss across timesteps
        count = 0          # Count timesteps processed

        # Initialize residual buffer for this epoch
        # WHY re-initialize each epoch: Residuals depend on current model parameters,
        # which change during training. Fresh computation ensures consistency.
        residuals = torch.zeros(len(z), device=device)

        for t in range(window, len(z)):
            # Build AR input: p most recent differenced values before time t
            if p > 0:
                # Slice the p values immediately before t and reshape for batch dim
                # WHY unsqueeze(0): Adds a batch dimension of 1 since we process
                # one timestep at a time (batch_size=1)
                y_hist = z_tensor[t - p : t].unsqueeze(0)  # Shape: (1, p)
            else:
                # No AR component: provide a dummy input
                y_hist = torch.zeros(1, 1, device=device)

            # Build MA input: q most recent residuals before time t
            if q > 0:
                # Detach residuals from the computation graph
                # WHY detach: Residuals were computed in previous timesteps using
                # the model's parameters. If we don't detach, backprop would try
                # to differentiate through ALL previous timesteps (full BPTT),
                # which is extremely expensive and unstable. Detaching treats
                # past residuals as fixed inputs, similar to teacher forcing.
                r_hist = residuals[t - q : t].unsqueeze(0).detach()  # Shape: (1, q)
            else:
                r_hist = torch.zeros(1, 1, device=device)

            # Forward pass: predict the differenced value at time t
            pred = model(y_hist, r_hist).squeeze()  # Shape: scalar

            # Target: actual differenced value at time t
            target = z_tensor[t]

            # Compute MSE loss between prediction and target
            loss = criterion(pred, target)
            total_loss += loss.item()  # Accumulate for epoch-level reporting
            count += 1

            # Record the residual for use in subsequent MA computations
            # WHY detach: We want to use this residual as a fixed input for future
            # timesteps, not as part of the gradient computation graph
            residuals[t] = (target - pred).detach()

            # Backpropagation and parameter update
            optimizer.zero_grad()  # Clear gradients from previous timestep
            loss.backward()        # Compute gradients of loss w.r.t. parameters
            optimizer.step()       # Update AR/MA coefficients and bias via Adam

        # Compute average loss for this epoch
        avg_loss = total_loss / max(count, 1)

        # Track best loss achieved during training
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Print progress every 20 epochs if verbose
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # Return trained model and all metadata needed for forecasting
    return {
        "model": model,                    # Trained ARIMANet with learned coefficients
        "d": d,                            # Differencing order for undifferencing
        "p": p,                            # AR order for forecast function
        "q": q,                            # MA order for forecast function
        "original_series": train_data,     # Original series for undifferencing anchor
        "differenced_series": z,           # Differenced series for forecast history
        "residuals": residuals.detach().cpu().numpy(),  # Residuals for MA forecast
        "train_loss": best_loss,           # Best training loss achieved
    }


def _forecast(result: Dict, n_steps: int) -> np.ndarray:
    """Generate multi-step forecasts using the trained ARIMANet.

    Forecasting is recursive: each prediction becomes input for the next step.
    Future residuals are set to zero (expected value of unknown errors).

    WHY recursive: Unlike batched training, forecasting requires each prediction
    to be computed before the next because it becomes part of the AR history.
    """
    model = result["model"]
    device = next(model.parameters()).device  # Get device from model parameters
    model.eval()  # Set to evaluation mode (disables dropout)

    d = result["d"]       # Differencing order
    p = result["p"]       # AR order
    q = result["q"]       # MA order
    z = list(result["differenced_series"])  # Copy as list for efficient append
    res = list(result["residuals"])          # Copy residual history

    # No gradient computation needed during forecasting
    # WHY: We are not training, so disabling autograd saves memory and compute
    with torch.no_grad():
        for _ in range(n_steps):
            # Build AR input from the most recent p differenced values
            if p > 0:
                y_hist = torch.tensor(z[-p:], dtype=torch.float32, device=device).unsqueeze(0)
            else:
                y_hist = torch.zeros(1, 1, device=device)

            # Build MA input from the most recent q residuals
            if q > 0:
                r_hist = torch.tensor(res[-q:], dtype=torch.float32, device=device).unsqueeze(0)
            else:
                r_hist = torch.zeros(1, 1, device=device)

            # Generate one-step prediction
            pred = model(y_hist, r_hist).item()  # Convert to Python scalar

            # Append prediction to history for next step's AR input
            z.append(pred)

            # Assume zero future residual (best guess for unknown errors)
            # WHY: The expected value of future errors is zero; this causes the
            # MA component's influence to decay as we forecast further ahead
            res.append(0.0)

    # Extract the n_steps forecasts (the last n_steps values appended to z)
    diff_forecasts = np.array(z[-n_steps:])

    # Undifference to recover forecasts on the original scale
    if d > 0:
        # Use the last value of the original training series as the anchor
        return _undifference(diff_forecasts, result["original_series"][-1], d)
    return diff_forecasts


def validate(result: Dict, val_data: np.ndarray) -> Dict[str, float]:
    """Validate model by forecasting the validation period and computing metrics.

    WHY separate validation: Validation metrics guide hyperparameter selection
    without contaminating the test set reserved for final evaluation.
    """
    forecasts = _forecast(result, len(val_data))
    return _compute_metrics(val_data, forecasts)


def test(result: Dict, test_data: np.ndarray,
         val_len: int = 0) -> Dict[str, float]:
    """Test model by forecasting through validation + test periods.

    WHY forecast through validation: Since the model only knows training data,
    we must forecast through the entire validation period before reaching
    test timestamps. Recursive forecasts accumulate error, so test metrics
    reflect realistic long-horizon forecast quality.
    """
    total = val_len + len(test_data)  # Total forecast horizon
    all_fc = _forecast(result, total)  # Generate all forecasts
    return _compute_metrics(test_data, all_fc[val_len:])  # Evaluate only test portion


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, train_data: np.ndarray,
                     val_data: np.ndarray) -> float:
    """Optuna objective: find the best ARIMA order and learning rate.

    WHY optimize lr alongside orders: Unlike classical ARIMA, the PyTorch version's
    performance is sensitive to learning rate. Too high causes oscillation,
    too low causes slow convergence and suboptimal coefficients.
    """
    # Suggest ARIMA structure parameters
    p = trial.suggest_int("p", 1, 6)    # AR order (at least 1 for meaningful model)
    d = trial.suggest_int("d", 0, 2)    # Differencing order
    q = trial.suggest_int("q", 0, 5)    # MA order

    # Suggest training hyperparameters (unique to the PyTorch version)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  # Learning rate (log-uniform)
    epochs = trial.suggest_int("epochs", 50, 200, step=50)  # Training epochs

    try:
        result = train(train_data, p=p, d=d, q=q, lr=lr, epochs=epochs, verbose=False)
        metrics = validate(result, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10  # Penalty for failed configurations


def run_optuna(train_data, val_data, n_trials=30):
    """Run Optuna Bayesian hyperparameter search.

    WHY Optuna: TPE sampler efficiently explores the joint space of discrete
    (p,d,q) and continuous (lr) hyperparameters, outperforming random search.
    """
    study = optuna.create_study(direction="minimize", study_name="arima_pytorch")
    study.optimize(
        lambda t: optuna_objective(t, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data, val_data, num_samples=20):
    """Run Ray Tune distributed hyperparameter search.

    WHY Ray Tune: Parallel trial execution across CPUs reduces wall-clock time.
    Each ARIMA configuration trains independently, enabling near-linear speedup.
    """
    def _trainable(config):
        """Inner function called by Ray for each trial."""
        try:
            result = train(train_data, p=config["p"], d=config["d"],
                           q=config["q"], lr=config["lr"],
                           epochs=config["epochs"], verbose=False)
            metrics = validate(result, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    # Search space matching Optuna ranges
    search_space = {
        "p": tune.randint(1, 6),
        "d": tune.randint(0, 3),
        "q": tune.randint(0, 6),
        "lr": tune.loguniform(1e-4, 1e-1),       # Log-uniform for learning rate
        "epochs": tune.choice([50, 100, 150]),     # Discrete epoch choices
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    print(f"Ray Tune Best config: {best.config}")
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(train_data: np.ndarray, val_data: np.ndarray) -> None:
    """Compare different ARIMA-PyTorch configurations and explain tradeoffs.

    This function shows how AR order, MA order, learning rate, and training
    duration affect the quality of the learned coefficients and forecast accuracy.

    Parameter Selection Reasoning:
        - (2,1,1) lr=0.01: Small model, moderate learning rate. Few parameters
          to learn means faster convergence but limited capacity.
          Best for: Simple series where overfitting is the main risk.

        - (5,1,2) lr=0.005: Large model, slower learning rate. More AR coefficients
          capture longer temporal dependencies, but need careful optimization.
          Best for: Complex series with multi-step autoregressive structure.

        - (3,1,0) lr=0.01: Pure AR (no MA). Tests whether the MA component adds
          value or just adds noise to the optimization.
          Best for: Series where residuals are white noise (no MA structure).

        - (1,1,3) lr=0.01: MA-dominant model. Only one AR coefficient but three
          MA coefficients. Tests whether forecast errors have richer structure
          than the values themselves.
          Best for: Shock-driven series (financial returns, error-correction models).

        - (3,1,2) lr=0.001: Same model as balanced but with very low learning rate.
          Tests whether slower optimization finds better or worse solutions.
          WHY: Low lr avoids overshooting but may not converge within epoch budget.
    """
    configs = [
        {"label": "ARIMA(2,1,1) lr=0.01 - Small fast",
         "p": 2, "d": 1, "q": 1, "lr": 0.01, "epochs": 100},
        {"label": "ARIMA(5,1,2) lr=0.005 - Large slow",
         "p": 5, "d": 1, "q": 2, "lr": 0.005, "epochs": 150},
        {"label": "ARIMA(3,1,0) lr=0.01 - Pure AR",
         "p": 3, "d": 1, "q": 0, "lr": 0.01, "epochs": 100},
        {"label": "ARIMA(1,1,3) lr=0.01 - MA-dominant",
         "p": 1, "d": 1, "q": 3, "lr": 0.01, "epochs": 100},
        {"label": "ARIMA(3,1,2) lr=0.001 - Low lr",
         "p": 3, "d": 1, "q": 2, "lr": 0.001, "epochs": 150},
    ]

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: PyTorch ARIMA - order and learning rate effects")
    print("=" * 70)

    results_summary = []

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            result = train(train_data, p=cfg["p"], d=cfg["d"], q=cfg["q"],
                           lr=cfg["lr"], epochs=cfg["epochs"], verbose=False)
            metrics = validate(result, val_data)

            # Display learned coefficients
            m = result["model"]
            if m.ar_coeffs is not None:
                print(f"  Learned AR coeffs: {m.ar_coeffs.data.cpu().numpy()}")
            if m.ma_coeffs is not None:
                print(f"  Learned MA coeffs: {m.ma_coeffs.data.cpu().numpy()}")
            print(f"  Training loss: {result['train_loss']:.6f}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            results_summary.append({"config": cfg["label"], "RMSE": metrics["RMSE"],
                                    "MAE": metrics["MAE"], "loss": result["train_loss"]})
        except Exception as e:
            print(f"  FAILED: {e}")
            results_summary.append({"config": cfg["label"], "RMSE": float("inf"),
                                    "MAE": float("inf"), "loss": float("inf")})

    # Rank by validation RMSE
    print("\n" + "-" * 70)
    print("RANKING (by validation RMSE):")
    results_summary.sort(key=lambda x: x["RMSE"])
    for i, r in enumerate(results_summary, 1):
        print(f"  {i}. {r['config']}: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}")

    print("\nKey Takeaways:")
    print("  - Learning rate is critical: too high causes instability, too low fails to converge")
    print("  - More parameters (higher p,q) need more epochs and lower lr to optimize well")
    print("  - Pure AR models are faster to train but miss MA structure if present")
    print("  - PyTorch ARIMA can capture similar patterns as classical ARIMA with proper tuning")


# ---------------------------------------------------------------------------
# Real-World Demo: Financial Time Series Prediction
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate PyTorch ARIMA on a financial time series prediction scenario.

    Domain: Finance / Trading
    Task: Predict daily stock returns for risk management

    This demo generates synthetic data mimicking daily stock price dynamics:
        - Mean-reverting returns with slight positive drift
        - Volatility clustering (GARCH-like effects)
        - Day-of-week effects
        - Occasional regime shifts

    Business Context:
        Financial institutions need return predictions for:
        - Value at Risk (VaR) calculations
        - Portfolio optimization and hedging
        - Algorithmic trading signal generation
        - Risk budgeting and exposure management
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Daily Stock Return Prediction")
    print("=" * 70)

    # ---- Generate synthetic financial data ----
    np.random.seed(77)

    # 3 years of trading days (~252 trading days per year)
    n_days = 756
    t = np.arange(n_days, dtype=np.float64)

    # Simulate stock price with geometric Brownian motion properties
    # WHY GBM: Standard model for stock price dynamics in quantitative finance;
    # produces realistic-looking price paths with drift and volatility
    daily_drift = 0.0003  # ~7.5% annual return
    daily_vol = 0.015     # ~24% annualized volatility

    # Generate returns with mild autocorrelation (mean-reversion)
    # WHY mean-reverting: Many financial series show short-term mean reversion
    # (negative autocorrelation at lag 1), which AR models can capture
    returns = np.zeros(n_days)
    returns[0] = np.random.normal(daily_drift, daily_vol)

    for i in range(1, n_days):
        # Mean-reverting return with lag-1 autocorrelation of -0.05
        # WHY: Negative autocorrelation means yesterday's positive return
        # slightly predicts today's negative return (classic mean reversion)
        mean_reversion = -0.05 * returns[i - 1]
        returns[i] = daily_drift + mean_reversion + np.random.normal(0, daily_vol)

    # Convert returns to price levels starting at $100
    # WHY cumulative product: Price_t = Price_0 * prod(1 + r_i) for i=1..t
    prices = 100.0 * np.cumprod(1 + returns)

    print(f"\nDataset: {n_days} trading days of synthetic stock data")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Starting price: ${prices[0]:.2f}")
    print(f"Ending price: ${prices[-1]:.2f}")
    print(f"Total return: {(prices[-1] / prices[0] - 1) * 100:.1f}%")
    print(f"Annualized vol: {np.std(returns) * np.sqrt(252) * 100:.1f}%")

    # Split: 70% train, 15% val, 15% test
    n_train = int(0.70 * n_days)
    n_val = int(0.15 * n_days)

    train_prices = prices[:n_train]
    val_prices = prices[n_train:n_train + n_val]
    test_prices = prices[n_train + n_val:]

    print(f"\nTrain: {len(train_prices)} days | Val: {len(val_prices)} days | "
          f"Test: {len(test_prices)} days")

    # ---- Train and evaluate ----
    print("\n--- Model Comparison ---")
    configs = [
        {"label": "ARIMA(1,1,0) lr=0.01", "p": 1, "d": 1, "q": 0, "lr": 0.01},
        {"label": "ARIMA(2,1,1) lr=0.005", "p": 2, "d": 1, "q": 1, "lr": 0.005},
        {"label": "ARIMA(3,1,2) lr=0.005", "p": 3, "d": 1, "q": 2, "lr": 0.005},
    ]

    best_result = None
    best_rmse = float("inf")
    best_label = ""

    for cfg in configs:
        try:
            result = train(train_prices, p=cfg["p"], d=cfg["d"], q=cfg["q"],
                           lr=cfg["lr"], epochs=80, verbose=False)
            metrics = validate(result, val_prices)
            print(f"\n  {cfg['label']}:")
            print(f"    Val RMSE: ${metrics['RMSE']:.2f}")
            print(f"    Val MAPE: {metrics['MAPE']:.2f}%")

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_result = result
                best_label = cfg["label"]
        except Exception as e:
            print(f"\n  {cfg['label']}: FAILED - {e}")

    # ---- Final evaluation ----
    if best_result is not None:
        print(f"\n--- Best Model: {best_label} ---")
        test_metrics = test(best_result, test_prices, val_len=len(val_prices))
        print(f"  Test RMSE: ${test_metrics['RMSE']:.2f}")
        print(f"  Test MAE:  ${test_metrics['MAE']:.2f}")
        print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")

        print("\n  Business Interpretation:")
        avg_price = np.mean(test_prices)
        print(f"    Average test-period price: ${avg_price:.2f}")
        print(f"    Average absolute forecast error: ${test_metrics['MAE']:.2f}")
        print(f"    This represents {test_metrics['MAPE']:.1f}% of the asset price")
        print(f"    Note: ARIMA on price levels has limitations for financial data;")
        print(f"    practitioners typically model returns or log-returns instead.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Full pipeline: data -> HPO -> train -> validate -> test -> compare -> demo."""
    print("=" * 70)
    print("ARIMA-like Model - PyTorch Implementation")
    print("=" * 70)

    # Generate synthetic data
    train_data, val_data, test_data = generate_data(n_points=1000)
    print(f"\nData splits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    # Optuna HPO
    print("\n--- Optuna Hyperparameter Search ---")
    study = run_optuna(train_data, val_data, n_trials=20)
    bp = study.best_params

    # Train with best hyperparameters
    print("\n--- Training with Best Parameters ---")
    result = train(train_data, p=bp["p"], d=bp["d"], q=bp["q"],
                   lr=bp["lr"], epochs=bp["epochs"])
    m = result["model"]
    if m.ar_coeffs is not None:
        print(f"AR coefficients: {m.ar_coeffs.data.cpu().numpy()}")
    if m.ma_coeffs is not None:
        print(f"MA coefficients: {m.ma_coeffs.data.cpu().numpy()}")

    # Validate
    print("\n--- Validation Metrics ---")
    val_metrics = validate(result, val_data)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test
    print("\n--- Test Metrics ---")
    test_metrics = test(result, test_data, val_len=len(val_data))
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Ray Tune
    print("\n--- Ray Tune Search ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=10)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    # Parameter comparison
    compare_parameter_sets(train_data, val_data)

    # Real-world demo
    real_world_demo()

    print("\nDone.")


if __name__ == "__main__":
    main()
