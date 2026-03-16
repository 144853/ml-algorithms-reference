"""
LSTM Time Series Forecasting - NumPy From-Scratch Implementation
=================================================================

Theory & Mathematics:
    This module implements a Long Short-Term Memory (LSTM) network from scratch
    using only NumPy, including both forward and backward passes through all gates.

    LSTM Cell Forward Pass:
    ----------------------
    Given input x_t and previous hidden state h_{t-1}, cell state c_{t-1}:

    1. Concatenate: z_t = [h_{t-1}, x_t]

    2. Forget Gate: Decides what to discard from cell state
       f_t = sigma(W_f @ z_t + b_f)

    3. Input Gate: Decides what new information to store
       i_t = sigma(W_i @ z_t + b_i)

    4. Candidate Cell State: New candidate values
       c_tilde_t = tanh(W_c @ z_t + b_c)

    5. Cell State Update:
       c_t = f_t * c_{t-1} + i_t * c_tilde_t

    6. Output Gate: Decides what to output
       o_t = sigma(W_o @ z_t + b_o)

    7. Hidden State:
       h_t = o_t * tanh(c_t)

    LSTM Cell Backward Pass (BPTT):
    --------------------------------
    Given upstream gradients dh_next, dc_next:

    1. d_tanh_c = dh_next * o_t * (1 - tanh(c_t)^2)
    2. dc = d_tanh_c + dc_next
    3. d_f = dc * c_{t-1} * f_t * (1 - f_t)          (sigmoid derivative)
    4. d_i = dc * c_tilde * i_t * (1 - i_t)
    5. d_c_tilde = dc * i_t * (1 - c_tilde^2)         (tanh derivative)
    6. d_o = dh_next * tanh(c_t) * o_t * (1 - o_t)
    7. Compute weight gradients: dW = d_gate @ z_t^T
    8. Compute bias gradients: db = d_gate
    9. Backpropagate: d_z = W^T @ d_gate -> split into dh_{t-1} and dx_t
    10. dc_{t-1} = dc * f_t

    Architecture for Time Series:
        - Input: (lookback, 1) sequences
        - Single LSTM layer with hidden_size units
        - Dense output layer: h_T -> y_hat

    Training:
        - Truncated BPTT over the lookback window
        - Gradient clipping to prevent exploding gradients
        - Adam optimiser implemented from scratch

Business Use Cases:
    - Understanding LSTM internals for research
    - Custom deployment without framework dependencies
    - Edge devices with limited compute
    - Educational: teach LSTM mechanics step by step

Advantages:
    - Complete transparency into forward/backward computations
    - No external ML framework needed
    - Useful for debugging and understanding gradients
    - Can be modified for custom gate architectures

Disadvantages:
    - Orders of magnitude slower than GPU-accelerated frameworks
    - No CUDA/GPU support
    - Numerically less stable than optimised implementations
    - Limited to simple architectures

Key Hyperparameters:
    - lookback (int): Sequence length
    - hidden_size (int): LSTM hidden state dimension
    - lr (float): Learning rate
    - epochs (int): Training epochs
    - grad_clip (float): Gradient clipping threshold

References:
    - Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
    - Olah, C. (2015). Understanding LSTM Networks. Blog post.
    - Karpathy, A. (2015). The Unreasonable Effectiveness of RNNs.
"""

import numpy as np  # NumPy is the sole computation backend - all matrix ops implemented here
import warnings  # Suppress warnings during automated HPO sweeps
from typing import Dict, Tuple, Any, Optional  # Type annotations for function signatures
from dataclasses import dataclass, field  # Dataclass for clean parameter/state containers

import optuna  # Bayesian hyperparameter optimization (TPE sampler)
import ray  # Ray for distributed parallel HPO execution
from ray import tune  # Ray Tune search space definitions

# Suppress numerical warnings during HPO (expected with small hidden sizes or high lr)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_points: int = 1000, freq: str = "D", seasonal_period: int = 7,
    trend_slope: float = 0.02, noise_std: float = 0.5, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic time series with trend, seasonality, and noise."""
    np.random.seed(seed)  # Reproducibility
    t = np.arange(n_points, dtype=np.float64)  # Float64 time index
    # Linear trend: simulates growth over time (LSTM should learn to extrapolate)
    trend = trend_slope * t
    # Sinusoidal seasonality: periodic pattern for LSTM to capture via its memory
    seasonality = 2.0 * np.sin(2 * np.pi * t / seasonal_period)
    # Gaussian noise: random perturbations
    noise = np.random.normal(0, noise_std, n_points)
    # Combine with positive baseline
    values = 10.0 + trend + seasonality + noise
    # Chronological 70/15/15 split
    n_train = int(0.70 * n_points)
    n_val = int(0.15 * n_points)
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


def _compute_metrics(actual, predicted):
    """Compute MAE, RMSE, MAPE, SMAPE forecast evaluation metrics."""
    actual = np.asarray(actual, dtype=np.float64).flatten()
    predicted = np.asarray(predicted, dtype=np.float64).flatten()
    min_len = min(len(actual), len(predicted))
    actual, predicted = actual[:min_len], predicted[:min_len]
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mask = actual != 0
    mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100 if mask.any() else np.inf
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    denom = np.where(denom == 0, 1e-10, denom)
    smape = np.mean(np.abs(errors) / denom) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Sigmoid activation: sigma(x) = 1 / (1 + exp(-x)).
    WHY clip: Prevents overflow in exp(-x) for very large/small x values,
    which would produce NaN or Inf and corrupt all subsequent computations.
    """
    x = np.clip(x, -500, 500)  # Clip to prevent overflow in exp
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(s):
    """Derivative of sigmoid given the sigmoid output s.
    d_sigma/dx = sigma(x) * (1 - sigma(x)) = s * (1 - s)
    WHY use output form: Computing from the output avoids recomputing sigmoid,
    which is both faster and numerically more stable.
    """
    return s * (1.0 - s)


def tanh_derivative(t):
    """Derivative of tanh given the tanh output t.
    d_tanh/dx = 1 - tanh(x)^2 = 1 - t^2
    WHY: Same benefit as sigmoid_derivative - avoids redundant computation.
    """
    return 1.0 - t ** 2


# ---------------------------------------------------------------------------
# LSTM Cell - From Scratch
# ---------------------------------------------------------------------------

@dataclass
class LSTMParams:
    """LSTM parameters for a single-layer LSTM cell.

    Weight matrices W_* have shape (hidden_size, hidden_size + input_size)
    because the input to each gate is the concatenation [h_{t-1}, x_t].
    Biases b_* have shape (hidden_size,).
    """
    # Gate weight matrices: map [h, x] concatenation to gate activations
    W_f: np.ndarray = None  # Forget gate: controls what to discard from cell state
    W_i: np.ndarray = None  # Input gate: controls what new info to store
    W_c: np.ndarray = None  # Candidate cell: generates new candidate values
    W_o: np.ndarray = None  # Output gate: controls what to output

    # Gate biases
    b_f: np.ndarray = None  # Forget gate bias (initialized to 1 for better gradient flow)
    b_i: np.ndarray = None  # Input gate bias
    b_c: np.ndarray = None  # Candidate cell bias
    b_o: np.ndarray = None  # Output gate bias

    # Output layer: maps final hidden state h_T to scalar prediction y_hat
    W_out: np.ndarray = None  # (1, hidden_size)
    b_out: np.ndarray = None  # (1,)


@dataclass
class LSTMGradients:
    """Accumulated gradients for all LSTM parameters.
    Mirrors the structure of LSTMParams with 'd' prefix for derivatives.
    """
    dW_f: np.ndarray = None   # Gradient of loss w.r.t. forget gate weights
    dW_i: np.ndarray = None   # Gradient of loss w.r.t. input gate weights
    dW_c: np.ndarray = None   # Gradient of loss w.r.t. candidate cell weights
    dW_o: np.ndarray = None   # Gradient of loss w.r.t. output gate weights
    db_f: np.ndarray = None   # Gradient of loss w.r.t. forget gate bias
    db_i: np.ndarray = None   # Gradient of loss w.r.t. input gate bias
    db_c: np.ndarray = None   # Gradient of loss w.r.t. candidate cell bias
    db_o: np.ndarray = None   # Gradient of loss w.r.t. output gate bias
    dW_out: np.ndarray = None  # Gradient of loss w.r.t. output layer weights
    db_out: np.ndarray = None  # Gradient of loss w.r.t. output layer bias


@dataclass
class LSTMCache:
    """Cache for forward pass intermediate values needed during backward pass.
    WHY cache: BPTT requires access to all gate activations, cell states, and
    hidden states from the forward pass to compute gradients efficiently.
    """
    z: list = field(default_factory=list)       # Concatenated [h, x] at each timestep
    f: list = field(default_factory=list)       # Forget gate activations f_t
    i: list = field(default_factory=list)       # Input gate activations i_t
    c_tilde: list = field(default_factory=list)  # Candidate cell states c_tilde_t
    c: list = field(default_factory=list)       # Cell states c_t (including c_0)
    o: list = field(default_factory=list)       # Output gate activations o_t
    h: list = field(default_factory=list)       # Hidden states h_t (including h_0)
    tanh_c: list = field(default_factory=list)  # tanh(c_t) - cached for backward pass


def init_params(input_size: int, hidden_size: int, seed: int = 42) -> LSTMParams:
    """Initialise LSTM parameters with Xavier/Glorot initialisation.

    WHY Xavier: Scale factor sqrt(2/(fan_in + fan_out)) keeps the variance
    of activations roughly constant across layers, preventing vanishing/exploding
    gradients at initialization time.

    WHY forget bias = 1: Initializing the forget gate bias to 1.0 means the forget
    gate starts "open" (f_t ~ 1), allowing gradients to flow through the cell state
    from the beginning of training. This was shown by Gers et al. (2000) to
    significantly improve LSTM training convergence.
    """
    rng = np.random.RandomState(seed)  # Local RNG for reproducible initialization
    concat_size = hidden_size + input_size  # Size of [h, x] concatenation

    # Xavier scale factor for weight initialization
    scale = np.sqrt(2.0 / (concat_size + hidden_size))

    params = LSTMParams()
    # Initialize all gate weights with Xavier-scaled random normals
    params.W_f = rng.randn(hidden_size, concat_size) * scale
    params.W_i = rng.randn(hidden_size, concat_size) * scale
    params.W_c = rng.randn(hidden_size, concat_size) * scale
    params.W_o = rng.randn(hidden_size, concat_size) * scale

    # Bias initialization
    params.b_f = np.ones(hidden_size)   # Forget gate: start open (bias=1)
    params.b_i = np.zeros(hidden_size)  # Input gate: start neutral (bias=0)
    params.b_c = np.zeros(hidden_size)  # Candidate: start neutral
    params.b_o = np.zeros(hidden_size)  # Output gate: start neutral

    # Output layer: maps hidden state to scalar prediction
    scale_out = np.sqrt(2.0 / hidden_size)
    params.W_out = rng.randn(1, hidden_size) * scale_out
    params.b_out = np.zeros(1)

    return params


def lstm_forward(x_seq: np.ndarray, params: LSTMParams,
                 h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LSTMCache]:
    """Forward pass through LSTM for an input sequence.

    Processes the sequence one timestep at a time, updating the cell state
    and hidden state at each step according to the LSTM gate equations.

    Args:
        x_seq: (seq_len, input_size) input sequence of observations.
        params: LSTM parameters (weights and biases for all gates).
        h_prev: (hidden_size,) initial hidden state (typically zeros).
        c_prev: (hidden_size,) initial cell state (typically zeros).

    Returns:
        y_hat: scalar prediction from the output layer applied to h_T.
        h_final: (hidden_size,) final hidden state after processing the sequence.
        c_final: (hidden_size,) final cell state.
        cache: LSTMCache with all intermediate values for backward pass.
    """
    seq_len = x_seq.shape[0]      # Number of timesteps in the sequence
    hidden_size = h_prev.shape[0]  # LSTM hidden dimension
    cache = LSTMCache()

    h = h_prev.copy()  # Working copy of hidden state
    c = c_prev.copy()  # Working copy of cell state

    # Store initial states (needed for backward pass at t=0)
    cache.h.append(h.copy())
    cache.c.append(c.copy())

    for t in range(seq_len):
        x_t = x_seq[t]  # Current input: (input_size,)

        # Step 1: Concatenate previous hidden state and current input
        # WHY concatenate: All four gates receive the same input [h_{t-1}, x_t];
        # this lets each gate learn different linear combinations of the same features
        z = np.concatenate([h, x_t])
        cache.z.append(z.copy())

        # Step 2: Forget gate - decides what to discard from cell state
        # f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
        # WHY sigmoid: Output in [0,1] represents "keep probability" for each cell dimension
        f_t = sigmoid(params.W_f @ z + params.b_f)
        cache.f.append(f_t.copy())

        # Step 3: Input gate - decides what new information to store
        # i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
        i_t = sigmoid(params.W_i @ z + params.b_i)
        cache.i.append(i_t.copy())

        # Step 4: Candidate cell state - generates new candidate values
        # c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)
        # WHY tanh: Output in [-1,1] represents potential new cell content
        c_tilde_t = np.tanh(params.W_c @ z + params.b_c)
        cache.c_tilde.append(c_tilde_t.copy())

        # Step 5: Cell state update - combine forget (old) and input (new)
        # c_t = f_t * c_{t-1} + i_t * c_tilde_t
        # WHY additive: This is the key LSTM innovation - the additive cell update
        # creates a "gradient highway" where gradients flow through multiplication
        # by f_t rather than through matrix multiplication, preventing vanishing gradients
        c = f_t * c + i_t * c_tilde_t
        cache.c.append(c.copy())

        # Step 6: Output gate - controls what to output from cell state
        # o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
        o_t = sigmoid(params.W_o @ z + params.b_o)
        cache.o.append(o_t.copy())

        # Step 7: Hidden state - filtered version of cell state
        # h_t = o_t * tanh(c_t)
        # WHY tanh(c_t): Squashes cell state to [-1,1] before output gate filtering
        tanh_c = np.tanh(c)
        cache.tanh_c.append(tanh_c.copy())
        h = o_t * tanh_c
        cache.h.append(h.copy())

    # Output prediction: linear layer from final hidden state to scalar
    # y_hat = W_out @ h_T + b_out
    y_hat = params.W_out @ h + params.b_out

    return y_hat.item(), h, c, cache


def lstm_backward(
    y_hat: float, y_true: float,
    params: LSTMParams, cache: LSTMCache,
    grad_clip: float = 5.0,
) -> LSTMGradients:
    """Backward pass through LSTM (Backpropagation Through Time).

    Computes gradients of MSE loss w.r.t. all LSTM parameters by backpropagating
    through time from the output layer through every gate at every timestep.

    WHY BPTT: Unlike feedforward networks where each layer is visited once,
    recurrent networks reuse the same parameters at every timestep. BPTT
    accumulates gradients from all timesteps into a single gradient per parameter.

    Args:
        y_hat: Model prediction (scalar).
        y_true: Ground truth target (scalar).
        params: LSTM parameters for computing gradient directions.
        cache: Forward pass cache with all intermediate values.
        grad_clip: Maximum gradient norm (prevents exploding gradients).

    Returns:
        LSTMGradients with gradients for all parameters.
    """
    hidden_size = params.b_f.shape[0]
    seq_len = len(cache.z)  # Number of timesteps processed

    grads = LSTMGradients()
    concat_size = params.W_f.shape[1]

    # Initialize all gradients to zero (will be accumulated across timesteps)
    grads.dW_f = np.zeros_like(params.W_f)
    grads.dW_i = np.zeros_like(params.W_i)
    grads.dW_c = np.zeros_like(params.W_c)
    grads.dW_o = np.zeros_like(params.W_o)
    grads.db_f = np.zeros_like(params.b_f)
    grads.db_i = np.zeros_like(params.b_i)
    grads.db_c = np.zeros_like(params.b_c)
    grads.db_o = np.zeros_like(params.b_o)

    # Output layer gradient: dL/dy_hat = (y_hat - y_true) for MSE loss
    # WHY: MSE = (y_hat - y_true)^2, derivative = 2*(y_hat - y_true)/2 = (y_hat - y_true)
    dy = y_hat - y_true

    # dL/dW_out = dy * h_T^T (outer product)
    grads.dW_out = dy * cache.h[-1].reshape(1, -1)
    grads.db_out = np.array([dy])

    # Gradient flowing into the final hidden state from the output layer
    # dL/dh_T = dy * W_out^T
    dh_next = dy * params.W_out.flatten()
    dc_next = np.zeros(hidden_size)  # No gradient flows into c from the output layer

    # Backpropagate through time (reverse order)
    for t in reversed(range(seq_len)):
        # Retrieve cached values for timestep t
        f_t = cache.f[t]           # Forget gate activation
        i_t = cache.i[t]           # Input gate activation
        c_tilde_t = cache.c_tilde[t]  # Candidate cell state
        o_t = cache.o[t]           # Output gate activation
        tanh_c_t = cache.tanh_c[t]  # tanh(c_t)
        z_t = cache.z[t]           # Concatenated input [h_{t-1}, x_t]
        c_prev = cache.c[t]        # c_{t-1} (cell state from previous step)

        # Output gate gradient
        # h_t = o_t * tanh(c_t), so dL/do_t = dL/dh_t * tanh(c_t) * sigmoid'(o_t)
        d_o = dh_next * tanh_c_t * sigmoid_derivative(o_t)

        # Cell state gradient (two sources: hidden state path + carry-forward)
        # From h_t: dL/dc_t = dL/dh_t * o_t * tanh'(c_t)
        # From next timestep: dc_next (gradient flowing back through c_{t+1} = f_{t+1} * c_t)
        dc = dh_next * o_t * tanh_derivative(tanh_c_t) + dc_next

        # Forget gate gradient
        # c_t = f_t * c_{t-1} + i_t * c_tilde_t
        # dL/df_t = dL/dc_t * c_{t-1} * sigmoid'(f_t)
        d_f = dc * c_prev * sigmoid_derivative(f_t)

        # Input gate gradient
        # dL/di_t = dL/dc_t * c_tilde_t * sigmoid'(i_t)
        d_i = dc * c_tilde_t * sigmoid_derivative(i_t)

        # Candidate cell state gradient
        # dL/dc_tilde_t = dL/dc_t * i_t * tanh'(c_tilde_t)
        d_c_tilde = dc * i_t * tanh_derivative(c_tilde_t)

        # Accumulate weight gradients: dW = d_gate @ z^T (outer product)
        # WHY accumulate: The same weights are used at every timestep, so the
        # total gradient is the sum of per-timestep gradients
        z_row = z_t.reshape(1, -1)
        grads.dW_f += d_f.reshape(-1, 1) @ z_row
        grads.dW_i += d_i.reshape(-1, 1) @ z_row
        grads.dW_c += d_c_tilde.reshape(-1, 1) @ z_row
        grads.dW_o += d_o.reshape(-1, 1) @ z_row

        # Accumulate bias gradients (bias gradient = gate gradient directly)
        grads.db_f += d_f
        grads.db_i += d_i
        grads.db_c += d_c_tilde
        grads.db_o += d_o

        # Gradient flowing backward to the input z = [h_{t-1}, x_t]
        # dL/dz = sum_gates(W_gate^T @ d_gate)
        dz = (params.W_f.T @ d_f + params.W_i.T @ d_i +
              params.W_c.T @ d_c_tilde + params.W_o.T @ d_o)

        # Split dz into gradient for h_{t-1} and x_t
        # WHY split: z = [h_{t-1}, x_t], so the first hidden_size elements
        # correspond to dh_{t-1} and the rest to dx_t (which we don't need)
        dh_next = dz[:hidden_size]

        # Cell state gradient to previous timestep
        # dc_{t-1} = dc * f_t (gradient flows back through the forget gate)
        # WHY f_t: This is the "gradient highway" - when f_t is close to 1,
        # gradients flow through unchanged, preventing vanishing gradients
        dc_next = dc * f_t

    # Gradient clipping: scale gradients if their norm exceeds the threshold
    # WHY: BPTT can produce very large gradients (exploding gradient problem),
    # especially for long sequences. Clipping prevents parameter updates from
    # being too large and destabilizing training
    for attr in ['dW_f', 'dW_i', 'dW_c', 'dW_o', 'db_f', 'db_i', 'db_c', 'db_o',
                 'dW_out', 'db_out']:
        grad = getattr(grads, attr)
        if grad is not None:
            norm = np.linalg.norm(grad)
            if norm > grad_clip:
                # Scale gradient to have norm = grad_clip
                setattr(grads, attr, grad * grad_clip / norm)

    return grads


# ---------------------------------------------------------------------------
# Adam Optimiser (from scratch)
# ---------------------------------------------------------------------------

@dataclass
class AdamState:
    """Adam optimiser state tracking first and second moment estimates.

    Adam combines momentum (first moment m) with RMSProp (second moment v)
    to provide adaptive per-parameter learning rates.

    Update rule: param -= lr * m_hat / (sqrt(v_hat) + eps)
    where m_hat, v_hat are bias-corrected moment estimates.
    """
    m: dict = field(default_factory=dict)  # First moment estimates (momentum)
    v: dict = field(default_factory=dict)  # Second moment estimates (RMSProp)
    t: int = 0               # Timestep counter for bias correction
    lr: float = 0.001        # Learning rate
    beta1: float = 0.9       # First moment decay rate (momentum)
    beta2: float = 0.999     # Second moment decay rate (RMSProp)
    eps: float = 1e-8        # Small constant to prevent division by zero


def adam_init(params: LSTMParams, lr: float = 0.001) -> AdamState:
    """Initialize Adam optimizer state with zero moment estimates for all parameters."""
    state = AdamState(lr=lr)
    for name in ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o',
                 'W_out', 'b_out']:
        p = getattr(params, name)
        state.m[name] = np.zeros_like(p)  # First moment = 0
        state.v[name] = np.zeros_like(p)  # Second moment = 0
    return state


def adam_update(params: LSTMParams, grads: LSTMGradients,
               state: AdamState) -> LSTMParams:
    """Apply one step of Adam optimization to all LSTM parameters.

    WHY Adam over SGD: Adam adapts the learning rate per-parameter based on
    the history of gradients. Parameters with consistently large gradients
    get smaller updates, and vice versa. This is especially important for
    LSTM where different gates may have very different gradient magnitudes.
    """
    state.t += 1  # Increment timestep for bias correction

    for name in ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o',
                 'W_out', 'b_out']:
        g = getattr(grads, 'd' + name)  # Get gradient (e.g., dW_f)
        if g is None:
            continue

        # Update first moment estimate (exponential moving average of gradients)
        # m = beta1 * m + (1 - beta1) * g
        state.m[name] = state.beta1 * state.m[name] + (1 - state.beta1) * g

        # Update second moment estimate (exponential moving average of squared gradients)
        # v = beta2 * v + (1 - beta2) * g^2
        state.v[name] = state.beta2 * state.v[name] + (1 - state.beta2) * g ** 2

        # Bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
        # WHY: Early in training, m and v are biased toward zero because they
        # were initialized to zero. Bias correction compensates for this.
        m_hat = state.m[name] / (1 - state.beta1 ** state.t)
        v_hat = state.v[name] / (1 - state.beta2 ** state.t)

        # Parameter update: param -= lr * m_hat / (sqrt(v_hat) + eps)
        p = getattr(params, name)
        setattr(params, name, p - state.lr * m_hat / (np.sqrt(v_hat) + state.eps))

    return params


# ---------------------------------------------------------------------------
# LSTM Model Container
# ---------------------------------------------------------------------------

@dataclass
class LSTMModel:
    """Container for the trained LSTM model and associated metadata."""
    params: LSTMParams = None        # Trained LSTM parameters
    lookback: int = 30               # Input sequence length
    hidden_size: int = 32            # LSTM hidden state dimension
    scaler_mean: float = 0.0         # Training data mean (for inverse scaling)
    scaler_std: float = 1.0          # Training data std (for inverse scaling)
    train_tail: np.ndarray = field(default_factory=lambda: np.array([]))  # Last lookback values


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    lookback: int = 30,
    hidden_size: int = 32,
    lr: float = 0.001,
    epochs: int = 50,
    grad_clip: float = 5.0,
    verbose: bool = True,
    seed: int = 42,
) -> LSTMModel:
    """Train LSTM from scratch using manual forward/backward passes.

    Training procedure:
        1. Standardize data (zero mean, unit variance)
        2. Create sliding window sequences of length lookback
        3. For each epoch, shuffle sequences and iterate:
           a. Forward pass through LSTM to get prediction
           b. Compute MSE loss against true next value
           c. Backward pass (BPTT) to compute gradients
           d. Update parameters using Adam optimizer
    """
    # Step 1: Standardize data for stable gradients
    # WHY: Neural networks train much better when inputs have mean~0, std~1;
    # without scaling, gradients can be very large or very small
    mean = np.mean(train_data)
    std = np.std(train_data) + 1e-8  # Add epsilon to prevent division by zero
    data = (train_data - mean) / std

    # Step 2: Create sliding window sequences
    # WHY: LSTM processes fixed-length sequences; each (X, y) pair consists of
    # lookback past values (X) and the next value to predict (y)
    X_seqs, y_targets = [], []
    for idx in range(lookback, len(data)):
        X_seqs.append(data[idx - lookback:idx].reshape(-1, 1))  # (lookback, 1)
        y_targets.append(data[idx])  # scalar

    n_samples = len(X_seqs)
    input_size = 1  # Univariate time series

    # Step 3: Initialize LSTM parameters with Xavier initialization
    params = init_params(input_size, hidden_size, seed)
    adam_state = adam_init(params, lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Shuffle training order each epoch to reduce variance
        # WHY: Without shuffling, the model sees sequences in chronological order
        # every epoch, which can bias gradient updates toward the end of the series
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for idx in indices:
            x_seq = X_seqs[idx]    # (lookback, 1) input sequence
            y_true = y_targets[idx]  # scalar target

            # Forward pass: process entire sequence, get prediction from last hidden state
            h0 = np.zeros(hidden_size)  # Initial hidden state (zeros)
            c0 = np.zeros(hidden_size)  # Initial cell state (zeros)
            y_hat, h_final, c_final, cache = lstm_forward(x_seq, params, h0, c0)

            # MSE loss: L = (y_hat - y_true)^2
            loss = (y_hat - y_true) ** 2
            epoch_loss += loss

            # Backward pass: compute gradients via BPTT
            grads = lstm_backward(y_hat, y_true, params, cache, grad_clip)

            # Adam update: adjust all parameters
            params = adam_update(params, grads, adam_state)

        avg_loss = epoch_loss / n_samples
        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, MSE: {avg_loss:.6f}")

    # Package trained model with all metadata needed for prediction
    model = LSTMModel(
        params=params, lookback=lookback, hidden_size=hidden_size,
        scaler_mean=mean, scaler_std=std,
        train_tail=data[-lookback:],  # Scaled tail for recursive forecasting
    )
    return model


def _predict_n_steps(model: LSTMModel, n_steps: int) -> np.ndarray:
    """Generate n-step ahead forecasts recursively.

    Each prediction becomes input for the next step (autoregressive prediction).
    WHY recursive: We do not have future observations, so each forecast extends
    the input history for the next forecast.
    """
    history = list(model.train_tail)  # Start with scaled training tail
    predictions = []
    h = np.zeros(model.hidden_size)  # Fresh hidden state
    c = np.zeros(model.hidden_size)  # Fresh cell state

    for _ in range(n_steps):
        # Use last lookback values as input sequence
        seq = np.array(history[-model.lookback:]).reshape(-1, 1)
        # Forward pass to get prediction
        y_hat, h, c, _ = lstm_forward(seq, model.params, h, c)
        predictions.append(y_hat)
        history.append(y_hat)  # Add prediction to history for next step
        # Reset states for stateless prediction
        h = np.zeros(model.hidden_size)
        c = np.zeros(model.hidden_size)

    # Inverse scale predictions back to original units
    preds = np.array(predictions) * model.scaler_std + model.scaler_mean
    return preds


def validate(model: LSTMModel, val_data: np.ndarray) -> Dict[str, float]:
    """Validate: forecast validation period and compute metrics."""
    predictions = _predict_n_steps(model, len(val_data))
    return _compute_metrics(val_data, predictions)


def test(model: LSTMModel, test_data: np.ndarray,
         val_data: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Test: forecast through validation + test, evaluate test portion."""
    val_len = len(val_data) if val_data is not None else 0
    total = val_len + len(test_data)
    all_preds = _predict_n_steps(model, total)
    return _compute_metrics(test_data, all_preds[val_len:])


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, train_data, val_data):
    """Optuna objective: minimize validation RMSE for LSTM hyperparameters."""
    lookback = trial.suggest_int("lookback", 7, 50)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 20, 60, step=10)

    try:
        model = train(train_data, lookback=lookback, hidden_size=hidden_size,
                      lr=lr, epochs=epochs, verbose=False)
        metrics = validate(model, val_data)
        return metrics["RMSE"] if np.isfinite(metrics["RMSE"]) else 1e10
    except Exception:
        return 1e10


def run_optuna(train_data, val_data, n_trials=15):
    """Run Optuna Bayesian HPO search."""
    study = optuna.create_study(direction="minimize", study_name="lstm_numpy")
    study.optimize(lambda t: optuna_objective(t, train_data, val_data),
                   n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(train_data, val_data, num_samples=20):
    """Ray Tune distributed HPO search."""
    def _trainable(config):
        try:
            model = train(train_data, lookback=config["lookback"],
                          hidden_size=config["hidden_size"],
                          lr=config["lr"], epochs=config["epochs"], verbose=False)
            metrics = validate(model, val_data)
            ray.train.report({"rmse": metrics["RMSE"], "mae": metrics["MAE"]})
        except Exception:
            ray.train.report({"rmse": 1e10, "mae": 1e10})

    search_space = {
        "lookback": tune.randint(7, 50),
        "hidden_size": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([20, 30, 40, 50]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    tuner = tune.Tuner(
        _trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    print(f"\nRay Tune Best RMSE: {best.metrics['rmse']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(train_data: np.ndarray, val_data: np.ndarray) -> None:
    """Compare different LSTM hyperparameter configurations and explain tradeoffs.

    Parameter Selection Reasoning:
        - hidden_size=16, lookback=10: Minimal model. Very few parameters means
          fast training and low overfitting risk, but limited capacity to learn
          complex temporal patterns.
          Best for: Short, simple series with clear patterns.

        - hidden_size=32, lookback=30: Moderate model. 32 hidden units provide
          enough capacity for medium-complexity patterns; 30-step lookback covers
          approximately one month of daily data.
          Best for: General-purpose forecasting with moderate complexity.

        - hidden_size=64, lookback=50: Large model. 64 hidden units can represent
          complex nonlinear dynamics; 50-step lookback captures long-range dependencies.
          But more parameters = more data needed and higher overfitting risk.
          Best for: Complex series with long-range dependencies and abundant data.

        - hidden_size=32, lookback=7: Short lookback. Only one week of history
          for weekly-seasonal data. Tests whether the LSTM can learn with minimal
          context.
          Best for: Series where only recent values matter (low autocorrelation lag).

        - hidden_size=32, lookback=30, lr=0.01: High learning rate. Faster
          convergence but risk of overshooting the optimal solution.
          WHY compare: Shows how learning rate interacts with model capacity.
    """
    configs = [
        {"label": "Small (h=16, lb=10)", "hidden_size": 16, "lookback": 10,
         "lr": 0.001, "epochs": 30},
        {"label": "Medium (h=32, lb=30)", "hidden_size": 32, "lookback": 30,
         "lr": 0.001, "epochs": 40},
        {"label": "Large (h=64, lb=50)", "hidden_size": 64, "lookback": 50,
         "lr": 0.001, "epochs": 40},
        {"label": "Short lookback (h=32, lb=7)", "hidden_size": 32, "lookback": 7,
         "lr": 0.001, "epochs": 30},
        {"label": "High lr (h=32, lb=30, lr=0.01)", "hidden_size": 32, "lookback": 30,
         "lr": 0.01, "epochs": 30},
    ]

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: LSTM hidden size, lookback, and learning rate")
    print("=" * 70)

    results_summary = []
    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            model = train(train_data, lookback=cfg["lookback"],
                          hidden_size=cfg["hidden_size"],
                          lr=cfg["lr"], epochs=cfg["epochs"], verbose=False)
            metrics = validate(model, val_data)
            n_params = (cfg["hidden_size"] * (cfg["hidden_size"] + 1) * 4 +
                        cfg["hidden_size"] * 4 + cfg["hidden_size"] + 1)
            print(f"  Parameters: ~{n_params}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            results_summary.append({"config": cfg["label"], "RMSE": metrics["RMSE"]})
        except Exception as e:
            print(f"  FAILED: {e}")
            results_summary.append({"config": cfg["label"], "RMSE": float("inf")})

    print("\n" + "-" * 70)
    print("RANKING (by validation RMSE):")
    results_summary.sort(key=lambda x: x["RMSE"])
    for i, r in enumerate(results_summary, 1):
        print(f"  {i}. {r['config']}: RMSE={r['RMSE']:.4f}")

    print("\nKey Takeaways:")
    print("  - Larger hidden_size captures more complex patterns but needs more data")
    print("  - Lookback should match the relevant temporal context (e.g., seasonal period)")
    print("  - Learning rate must balance convergence speed vs optimization stability")
    print("  - From-scratch LSTM is much slower than PyTorch but gives full transparency")


# ---------------------------------------------------------------------------
# Real-World Demo: Energy Consumption Forecasting
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate from-scratch LSTM on energy consumption forecasting.

    Domain: Smart Grid / Energy Management
    Task: Predict hourly household energy consumption

    Synthetic data characteristics:
        - Daily pattern (high during day, low at night)
        - Weekly pattern (weekdays vs weekends)
        - Gradual increase trend (more devices, EV charging)
        - Temperature-correlated noise

    Business Context:
        Smart grid operators need energy forecasts for:
        - Demand response program activation
        - Battery storage charge/discharge scheduling
        - Dynamic pricing decisions
        - Grid stability maintenance
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Household Energy Consumption Forecasting")
    print("=" * 70)

    np.random.seed(88)

    # 6 months of daily data (reduced from hourly for tractability)
    n_days = 180
    t = np.arange(n_days, dtype=np.float64)

    # Base consumption: 30 kWh/day (typical US household)
    base = 30.0

    # Weekly pattern: weekends ~15% higher (people at home)
    weekly = 0.08 * base * np.sin(2 * np.pi * t / 7)

    # Growth: 1% monthly increase (new appliances, EV adoption)
    trend = base * 0.01 * t / 30

    # Noise: 10% random variation
    noise = np.random.normal(0, 0.05 * base, n_days)

    energy = base + weekly + trend + noise
    energy = np.maximum(energy, 5.0)  # Minimum 5 kWh/day

    print(f"\nDataset: {n_days} days of daily energy consumption")
    print(f"Range: {energy.min():.1f} - {energy.max():.1f} kWh/day")
    print(f"Mean: {energy.mean():.1f} kWh/day")

    # Split
    n_train = int(0.75 * n_days)
    n_val = int(0.125 * n_days)
    train_e = energy[:n_train]
    val_e = energy[n_train:n_train + n_val]
    test_e = energy[n_train + n_val:]

    print(f"Train: {len(train_e)} | Val: {len(val_e)} | Test: {len(test_e)}")

    # Train with modest settings (from-scratch is slow)
    print("\n--- Training LSTM (h=24, lb=14, epochs=30) ---")
    try:
        model = train(train_e, lookback=14, hidden_size=24,
                      lr=0.002, epochs=30, verbose=True)

        val_metrics = validate(model, val_e)
        print(f"\nVal RMSE: {val_metrics['RMSE']:.2f} kWh/day")
        print(f"Val MAPE: {val_metrics['MAPE']:.2f}%")

        test_metrics = test(model, test_e, val_e)
        print(f"\nTest RMSE: {test_metrics['RMSE']:.2f} kWh/day")
        print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")

        print("\nBusiness Interpretation:")
        print(f"  Average daily error: {test_metrics['MAE']:.1f} kWh")
        print(f"  At ~$0.15/kWh, this is ~${test_metrics['MAE'] * 0.15:.2f}/day error")
        print(f"  For demand response, {test_metrics['MAPE']:.1f}% MAPE is "
              f"{'adequate' if test_metrics['MAPE'] < 15 else 'needs improvement'}")
    except Exception as e:
        print(f"Training failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Full pipeline: data -> HPO -> train -> validate -> test -> compare -> demo."""
    print("=" * 70)
    print("LSTM - NumPy From-Scratch (Forward + Backward Pass)")
    print("=" * 70)

    # Use smaller dataset for from-scratch implementation (it's slow)
    train_data, val_data, test_data = generate_data(n_points=500)
    print(f"\nSplits - Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    print("\n--- Optuna HPO ---")
    study = run_optuna(train_data, val_data, n_trials=10)
    bp = study.best_params

    print("\n--- Training Best ---")
    best_model = train(train_data, lookback=bp["lookback"],
                       hidden_size=bp["hidden_size"],
                       lr=bp["lr"], epochs=bp["epochs"])
    print(f"Hidden size: {best_model.hidden_size}")
    print(f"Lookback: {best_model.lookback}")

    print("\n--- Validation ---")
    for k, v in validate(best_model, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Test ---")
    for k, v in test(best_model, test_data, val_data).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Ray Tune ---")
    try:
        ray_tune_search(train_data, val_data, num_samples=8)
    except Exception as e:
        print(f"Ray Tune skipped: {e}")

    compare_parameter_sets(train_data, val_data)
    real_world_demo()

    print("\nDone.")


if __name__ == "__main__":
    main()
