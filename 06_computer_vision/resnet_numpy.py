"""
Simplified Residual Network (ResNet) - NumPy From-Scratch Implementation
========================================================================

Architecture:
    This implementation builds a simplified ResNet from scratch using only
    NumPy, demonstrating the key innovation of skip (residual) connections.

    Network Architecture:
    Input (1, 16, 16)
    -> Conv2D(1->8, 3x3, pad=1) -> BN -> ReLU
    -> ResidualBlock(8->8, 3x3)   [identity shortcut: y = F(x) + x]
    -> AvgPool(2x2)
    -> ResidualBlock(8->16, 3x3)  [projection shortcut: y = F(x) + W_s * x]
    -> AvgPool(2x2)
    -> GlobalAvgPool -> FC(n_classes) -> Softmax

    Residual Block Structure:
    x --> Conv -> BN -> ReLU -> Conv -> BN --> (+) --> ReLU --> y
    |                                          |
    +--------- identity or projection ---------+

Theory & Mathematics:
    Residual Learning Framework:
        Instead of learning H(x) directly, learn the residual F(x) = H(x) - x
        Then the output is: H(x) = F(x) + x

        Why this works:
        1. If the optimal mapping is close to identity, F(x) -> 0 is easier
           to learn than H(x) -> x
        2. Gradients flow directly through skip connections:
           dL/dx = dL/dH * (dF/dx + 1)
           The "+1" term ensures gradients never vanish completely

    Skip Connection Types:
        Identity: y = F(x) + x
            Used when input and output dimensions match.

        Projection: y = F(x) + W_s * x
            Used when dimensions change (e.g., channel increase).
            W_s is a 1x1 convolution for channel matching.

    Batch Normalization (simplified):
        For each channel: normalize to zero mean, unit variance.
        mu = mean(x), var = variance(x)
        x_hat = (x - mu) / sqrt(var + eps)
        y = gamma * x_hat + beta  (learnable parameters)

    Global Average Pooling:
        For each channel, average all spatial positions.
        Reduces (N, C, H, W) -> (N, C)
        Replaces large fully connected layers, reduces overfitting.

Business Use Cases:
    - Deep understanding of residual learning for custom architectures
    - Educational: visualize gradient flow through skip connections
    - Research: prototype novel residual block designs
    - Embedded systems: implement on custom hardware
    - Debugging framework implementations

Advantages:
    - Full transparency: every gradient computation visible
    - Demonstrates why skip connections solve vanishing gradients
    - No framework dependencies
    - Teaches core deep learning concepts
    - Easy to extend with custom block designs

Disadvantages:
    - Extremely slow compared to GPU frameworks
    - Limited to very small models (memory and speed)
    - No GPU acceleration
    - Manual gradient computation through complex paths
    - Not practical for production use

Key Hyperparameters:
    - learning_rate: SGD step size (0.001 - 0.05)
    - n_epochs: Training iterations
    - batch_size: Mini-batch size
    - n_filters: Number of filters in conv layers
    - init_scale: Weight initialization scale

References:
    - He, K. et al. (2016). "Deep Residual Learning for Image Recognition."
      CVPR. arXiv:1512.03385.
    - He, K. et al. (2016). "Identity Mappings in Deep Residual Networks."
      ECCV. arXiv:1603.05027.
"""

# numpy is the ONLY numerical dependency for this from-scratch ResNet implementation,
# providing all array operations needed for convolution, batch normalization, residual
# connections, and gradient computation through the entire network.
# Every matrix multiply, element-wise op, and reshape is done via numpy arrays.
import numpy as np

# warnings module suppresses convergence/deprecation warnings during training
# so that console output stays focused on training metrics rather than library noise
import warnings

# Type hints for self-documenting function signatures and static analysis support.
# Dict stores results, Tuple for multi-return, Any for flexible model types,
# List for layer configs, Optional for nullable parameters.
from typing import Dict, Tuple, Any, List, Optional

# Optuna is an optional dependency for Bayesian hyperparameter optimization.
# It uses a Tree-structured Parzen Estimator (TPE) to efficiently search the
# hyperparameter space (learning_rate, n_filters, init_scale, batch_size).
# We wrap the import in try/except so the script still works without it.
try:
    import optuna  # Bayesian hyperparameter optimizer using TPE algorithm

    OPTUNA_AVAILABLE = True  # Flag to guard Optuna-specific code sections
except ImportError:
    OPTUNA_AVAILABLE = False  # Gracefully degrade if Optuna not installed

# Ray Tune is an optional dependency for distributed hyperparameter tuning.
# It supports parallel trial execution across multiple CPUs/GPUs, making
# large-scale hyperparameter searches feasible on cluster environments.
try:
    import ray  # Distributed computing framework for parallel trial execution
    from ray import tune  # Ray's hyperparameter tuning library

    RAY_AVAILABLE = True  # Flag to guard Ray-specific code sections
except ImportError:
    RAY_AVAILABLE = False  # Gracefully degrade if Ray not installed

# Suppress all warnings globally to keep output clean during training.
# Common warnings include numpy casting warnings and sklearn convergence warnings.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 1500,  # Total number of synthetic images to create
    img_size: int = 16,  # Spatial dimension (H=W) of each square image
    n_classes: int = 5,  # Number of distinct pattern classes to generate
    random_state: int = 42,  # Seed for reproducibility across runs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic grayscale images with class-specific patterns.

    Each class gets a distinctive spatial pattern that a ResNet should learn
    to distinguish. The patterns are designed to require spatial feature
    extraction (not just pixel statistics), making convolution essential.

    Returns X: (N, 1, H, W), y: (N,).
    """
    # Create a seeded random number generator for reproducible data generation.
    # Using RandomState instead of global seed ensures isolation from other code.
    rng = np.random.RandomState(random_state)

    # Calculate samples per class for balanced dataset creation.
    # Integer division ensures equal representation; remainders are discarded.
    spc = n_samples // n_classes

    # Accumulator lists for images and labels before array conversion.
    # Lists are used because appending is O(1), unlike numpy concatenation.
    X_list, y_list = [], []

    # Iterate through each class to generate class-specific patterns.
    # Each class has a unique spatial signature that requires different
    # convolutional features to detect.
    for cls in range(n_classes):
        for _ in range(spc):
            # Start with low-amplitude random noise as the base image.
            # The 0.1 multiplier ensures patterns dominate over background noise,
            # giving the network a clear signal to learn from.
            img = rng.rand(img_size, img_size).astype(np.float32) * 0.1

            if cls == 0:  # HORIZONTAL LINES pattern
                # Add bright horizontal stripes every 3 rows.
                # This pattern tests the network's ability to detect horizontal edges,
                # which requires filters with horizontal orientation sensitivity.
                for r in range(0, img_size, 3):
                    img[r, :] += 0.7  # Add 0.7 intensity to entire row
            elif cls == 1:  # VERTICAL LINES pattern
                # Add bright vertical stripes every 3 columns.
                # Complementary to horizontal lines, tests vertical edge detection.
                # A well-trained ResNet should easily distinguish these two orientations.
                for c in range(0, img_size, 3):
                    img[:, c] += 0.7  # Add 0.7 intensity to entire column
            elif cls == 2:  # CENTER BLOB pattern
                # Create a Gaussian blob centered in the image.
                # This tests the network's ability to detect localized features
                # vs. distributed patterns like lines.
                cx, cy = img_size // 2, img_size // 2  # Center coordinates
                yy, xx = np.ogrid[:img_size, :img_size]  # Open meshgrid for distance calc
                # Gaussian function: exp(-(x^2+y^2)/sigma^2) creates smooth circular blob.
                # sigma^2=8 controls the spread; smaller values create tighter blobs.
                img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8.0).astype(np.float32) * 0.8
            elif cls == 3:  # DIAGONAL pattern
                # Draw a diagonal line from top-left toward bottom-right.
                # Tests the network's ability to detect oriented features at 45 degrees,
                # which requires combinations of horizontal and vertical filters.
                for i in range(img_size):
                    j = i % img_size  # Wrap around to stay within image bounds
                    img[i, j] += 0.7  # Brighten pixel on the diagonal
            elif cls == 4:  # CHECKERBOARD pattern
                # Create a checkerboard of alternating bright/dark blocks.
                # This is the most complex pattern, requiring the network to detect
                # periodic spatial frequency -- a task that benefits from multiple
                # convolutional layers and residual connections.
                block = max(2, img_size // 4)  # Block size, at least 2x2 pixels
                for r in range(0, img_size, block):
                    for c in range(0, img_size, block):
                        # Alternating pattern: bright when (row_block + col_block) is even
                        if ((r // block) + (c // block)) % 2 == 0:
                            img[r : r + block, c : c + block] += 0.6

            # Add small Gaussian noise to simulate real-world sensor noise.
            # The 0.05 std dev is small enough to not destroy patterns but large
            # enough to prevent the network from memorizing exact pixel values.
            img += rng.randn(img_size, img_size).astype(np.float32) * 0.05

            # Clip pixel values to valid [0, 1] range to prevent artifacts.
            # Without clipping, pattern additions could push values above 1.0.
            img = np.clip(img, 0, 1)

            X_list.append(img)  # Accumulate the generated image
            y_list.append(cls)  # Record the corresponding class label

    # Convert lists to numpy arrays with proper dtypes and shapes.
    # Reshape adds a channel dimension: (N, H, W) -> (N, 1, H, W) for NCHW format,
    # which is the standard format for convolutional neural networks.
    X = np.array(X_list, dtype=np.float32).reshape(-1, 1, img_size, img_size)

    # Labels as int64 for indexing into softmax probability arrays.
    y = np.array(y_list, dtype=np.int64)

    # Randomly shuffle the dataset to break class ordering.
    # Without shuffling, all samples of class 0 come first, then class 1, etc.,
    # which would cause the network to learn class boundaries instead of patterns.
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Layer Implementations
# ---------------------------------------------------------------------------


class Conv2D:
    """
    2D Convolution with optional same-padding.

    This is a from-scratch implementation using explicit nested loops over
    spatial positions. While extremely slow compared to optimized implementations,
    it makes every step of the convolution operation transparent for learning.

    The forward pass computes: out[n, co, i, j] = sum(X_pad[n, :, i:i+ks, j:j+ks] * W[co]) + b[co]
    The backward pass computes gradients for W, b, and X using the chain rule.
    """

    def __init__(
        self,
        in_ch: int,  # Number of input channels (e.g., 1 for grayscale, 8 for feature maps)
        out_ch: int,  # Number of output channels (number of learned filters)
        kernel_size: int,  # Spatial size of each filter (e.g., 3 for 3x3)
        padding: int = 0,  # Zero-padding added to input borders for spatial preservation
        init_scale: float = 0.1,  # Standard deviation for random weight initialization
    ):
        # Store layer configuration for use in forward and backward passes.
        self.in_ch = in_ch  # Input channels, needed for weight shape verification
        self.out_ch = out_ch  # Output channels, determines number of feature maps produced
        self.ks = kernel_size  # Kernel spatial size, typically 3 for 3x3 or 1 for 1x1
        self.padding = padding  # Padding amount, pad=1 with ks=3 preserves spatial dimensions

        # Fan-in calculation for potential He/Xavier initialization.
        # fan_in = in_ch * ks * ks is the number of input connections per output neuron.
        # Currently using simple Gaussian init scaled by init_scale instead.
        fan_in = in_ch * kernel_size * kernel_size

        # Initialize filter weights with small random Gaussian values.
        # Shape: (out_ch, in_ch, ks, ks) -- each of out_ch filters has in_ch channels.
        # The init_scale controls initial weight magnitude; too large causes divergence,
        # too small causes vanishing gradients. 0.1 is a reasonable default for small networks.
        self.W = np.random.randn(out_ch, in_ch, kernel_size, kernel_size).astype(np.float32) * init_scale

        # Bias terms initialized to zero, one per output channel.
        # Biases shift the activation function's operating point; zero init is standard.
        self.b = np.zeros(out_ch, dtype=np.float32)

        # Cache for storing forward pass intermediates needed in backward pass.
        # Stores (original_X, padded_X) to compute gradients w.r.t. weights and input.
        self.cache = None

    def _pad(self, X: np.ndarray) -> np.ndarray:
        """
        Apply zero-padding to the spatial dimensions of the input tensor.

        Zero-padding adds rows/columns of zeros around the image borders,
        allowing the convolution to produce output of the same spatial size
        as the input (when padding = kernel_size // 2).

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
            Input tensor in NCHW format.

        Returns
        -------
        np.ndarray : Padded tensor with shape (N, C, H+2p, W+2p).
        """
        # If no padding requested, return input unchanged to avoid unnecessary copy.
        if self.padding == 0:
            return X
        # np.pad with mode="constant" adds zeros around spatial dims (axes 2 and 3).
        # The (0, 0) tuples for axes 0 and 1 mean no padding on batch and channel dims.
        return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of 2D convolution.

        For each output position (i, j) and each output channel co:
        out[n, co, i, j] = sum over (in_ch, kh, kw) of X_pad[n, :, i:i+ks, j:j+ks] * W[co] + b[co]

        This is the correlation operation (not true convolution, which would flip the kernel).
        In deep learning, this distinction doesn't matter because the network learns
        the appropriate filter orientations during training.

        Parameters
        ----------
        X : np.ndarray of shape (N, C_in, H, W)
            Input feature maps.

        Returns
        -------
        np.ndarray of shape (N, C_out, H_out, W_out)
            Output feature maps after convolution.
        """
        # Apply zero-padding to preserve spatial dimensions (or not, if padding=0).
        X_pad = self._pad(X)

        # Extract dimensions of the padded input for output size calculation.
        N, C, H, W = X_pad.shape

        # Compute output spatial dimensions using the convolution formula:
        # H_out = (H_in + 2*padding - kernel_size) / stride + 1
        # Since stride=1 and padding is already applied, simplifies to H - ks + 1.
        H_out = H - self.ks + 1
        W_out = W - self.ks + 1

        # Pre-allocate output tensor filled with zeros.
        # Shape: (batch, output_channels, output_height, output_width)
        out = np.zeros((N, self.out_ch, H_out, W_out), dtype=np.float32)

        # Explicit 4-nested loop over batch, output channels, and spatial positions.
        # This is O(N * out_ch * H_out * W_out * in_ch * ks * ks) -- extremely slow
        # but makes every step of the convolution transparent for educational purposes.
        for n in range(N):  # Loop over each sample in the batch
            for co in range(self.out_ch):  # Loop over each output filter
                for i in range(H_out):  # Loop over output height positions
                    for j in range(W_out):  # Loop over output width positions
                        # Extract the receptive field patch from the padded input:
                        # X_pad[n, :, i:i+ks, j:j+ks] has shape (in_ch, ks, ks).
                        # Multiply element-wise with filter W[co] (same shape) and sum.
                        # This computes the dot product between the patch and the filter,
                        # producing a single scalar activation value.
                        out[n, co, i, j] = (
                            np.sum(X_pad[n, :, i : i + self.ks, j : j + self.ks] * self.W[co]) + self.b[co]
                        )

        # Cache original and padded input for the backward pass.
        # The backward pass needs X_pad to compute weight gradients (dW)
        # and the original X shape to un-pad the input gradient (dX).
        self.cache = (X, X_pad)
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """
        Compute gradients and update weights via backpropagation.

        The chain rule gives us three gradients:
        1. dW[co] += sum over (n, i, j) of d_out[n, co, i, j] * X_pad[n, :, i:i+ks, j:j+ks]
        2. db[co] += sum over (n, i, j) of d_out[n, co, i, j]
        3. dX_pad[n, :, i:i+ks, j:j+ks] += sum over co of d_out[n, co, i, j] * W[co]

        Parameters
        ----------
        d_out : np.ndarray of shape (N, C_out, H_out, W_out)
            Gradient flowing back from the next layer.
        lr : float
            Learning rate for SGD weight update.

        Returns
        -------
        np.ndarray : Gradient with respect to the input X (after removing padding).
        """
        # Retrieve cached forward pass data for gradient computation.
        X, X_pad = self.cache
        N, C, H, W = X_pad.shape  # Padded input dimensions
        _, _, H_out, W_out = d_out.shape  # Output gradient dimensions

        # Initialize gradient accumulators to zeros with same shapes as parameters.
        dW = np.zeros_like(self.W)  # Weight gradient accumulator: (out_ch, in_ch, ks, ks)
        db = np.zeros_like(self.b)  # Bias gradient accumulator: (out_ch,)
        dX_pad = np.zeros_like(X_pad)  # Input gradient accumulator: (N, C, H, W)

        # Compute gradients using the same nested loop structure as forward pass.
        # This mirrors the forward computation but propagates gradients backward.
        for n in range(N):  # Loop over each sample in the batch
            for co in range(self.out_ch):  # Loop over each output filter
                # Bias gradient: sum of all upstream gradients for this filter.
                # db[co] = sum of d_out[n, co, :, :] across all samples and positions.
                db[co] += np.sum(d_out[n, co])
                for i in range(H_out):  # Loop over output height positions
                    for j in range(W_out):  # Loop over output width positions
                        # Weight gradient: accumulate outer product of upstream gradient
                        # and the corresponding input patch. This is because:
                        # d(loss)/dW[co] = sum d_out[n,co,i,j] * X_pad[n,:,i:i+ks,j:j+ks]
                        dW[co] += d_out[n, co, i, j] * X_pad[n, :, i : i + self.ks, j : j + self.ks]

                        # Input gradient: scatter upstream gradient back through the filter.
                        # Each position in the receptive field contributed to the output,
                        # so it receives gradient proportional to the filter weight.
                        # d(loss)/dX_pad[n,:,i:i+ks,j:j+ks] += d_out[n,co,i,j] * W[co]
                        dX_pad[n, :, i : i + self.ks, j : j + self.ks] += d_out[n, co, i, j] * self.W[co]

        # SGD weight update: W = W - lr * (dW / N)
        # Dividing by N gives the mean gradient across the batch, preventing
        # the effective learning rate from scaling with batch size.
        self.W -= lr * dW / N

        # SGD bias update: b = b - lr * (db / N)
        self.b -= lr * db / N

        # Remove padding from the input gradient to match the original input shape.
        # The padded positions don't correspond to real input values, so their
        # gradients should not be propagated further.
        if self.padding > 0:
            dX = dX_pad[:, :, self.padding : -self.padding, self.padding : -self.padding]
        else:
            dX = dX_pad  # No padding to remove
        return dX


class BatchNorm:
    """
    Simplified Batch Normalization for convolutional layers.

    Batch normalization normalizes activations across the batch dimension
    for each channel independently. This stabilizes training by:
    1. Reducing internal covariate shift (distribution changes between layers)
    2. Allowing higher learning rates without divergence
    3. Providing slight regularization effect from batch statistics noise

    During training: uses batch statistics (mean, variance).
    During inference: uses exponentially smoothed running statistics.

    The learnable parameters gamma and beta allow the network to undo
    the normalization if that's optimal, making BN a flexible identity
    mapping when needed.
    """

    def __init__(
        self,
        n_channels: int,  # Number of feature channels to normalize independently
        momentum: float = 0.1,  # Exponential moving average decay for running stats
        eps: float = 1e-5,  # Small constant for numerical stability in division
    ):
        # Learnable scale parameter gamma, initialized to 1.0 (identity scaling).
        # After normalization, gamma * x_norm rescales the normalized values.
        # The network can learn gamma=1 to keep normalization, or other values
        # to adjust the distribution as needed for downstream layers.
        self.gamma = np.ones(n_channels, dtype=np.float32)

        # Learnable shift parameter beta, initialized to 0.0 (no shift).
        # After scaling by gamma, beta shifts the distribution center.
        # Together, gamma and beta give the network full control over the
        # output distribution: y = gamma * x_norm + beta.
        self.beta = np.zeros(n_channels, dtype=np.float32)

        # Epsilon prevents division by zero when variance is very small.
        # Added inside the square root: 1/sqrt(var + eps).
        self.eps = eps

        # Momentum controls how fast running statistics update.
        # running_stat = (1 - momentum) * running_stat + momentum * batch_stat.
        # momentum=0.1 means 10% of each new batch influences the running estimate.
        self.momentum = momentum

        # Running mean tracks the exponential moving average of batch means.
        # Used at inference time (when training=False) instead of batch statistics,
        # because at inference we might process single samples (batch statistics meaningless).
        self.running_mean = np.zeros(n_channels, dtype=np.float32)

        # Running variance tracks the exponential moving average of batch variances.
        # Initialized to 1.0 (unit variance) so initial inference predictions are reasonable.
        self.running_var = np.ones(n_channels, dtype=np.float32)

        # Cache stores forward pass intermediates needed for backward pass gradient computation.
        self.cache = None

        # Training flag determines whether to use batch stats (True) or running stats (False).
        self.training = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of batch normalization.

        During training: normalize using batch statistics, update running statistics.
        During inference: normalize using accumulated running statistics.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
            Input feature maps in NCHW format.

        Returns
        -------
        np.ndarray of shape (N, C, H, W)
            Normalized, scaled, and shifted feature maps.
        """
        N, C, H, W = X.shape  # Unpack: batch size, channels, height, width

        if self.training:
            # Compute per-channel mean across batch and spatial dimensions.
            # axis=(0, 2, 3) averages over N samples and H*W spatial positions,
            # leaving one mean value per channel C. Shape: (C,)
            mean = X.mean(axis=(0, 2, 3))

            # Compute per-channel variance across the same dimensions.
            # Variance measures the spread of activations for each channel.
            var = X.var(axis=(0, 2, 3))

            # Update running statistics using exponential moving average (EMA).
            # running_mean slowly tracks the long-term average of batch means.
            # At test time, this provides a stable estimate of the data distribution.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # At inference time, use the accumulated running statistics.
            # These are more stable than single-batch statistics and allow
            # consistent predictions regardless of batch composition.
            mean = self.running_mean
            var = self.running_var

        # Reshape statistics from (C,) to (1, C, 1, 1) for broadcasting.
        # This allows element-wise operations with the (N, C, H, W) tensor.
        # Broadcasting automatically expands dimensions of size 1 to match.
        mean_r = mean.reshape(1, C, 1, 1)  # Reshape mean for broadcasting with 4D tensor
        var_r = var.reshape(1, C, 1, 1)  # Reshape variance for broadcasting
        gamma_r = self.gamma.reshape(1, C, 1, 1)  # Reshape gamma scale parameter
        beta_r = self.beta.reshape(1, C, 1, 1)  # Reshape beta shift parameter

        # Normalize: subtract mean and divide by standard deviation.
        # X_norm = (X - mu) / sqrt(var + eps)
        # After this, each channel has approximately zero mean and unit variance.
        # The eps prevents division by zero when variance is near zero.
        X_norm = (X - mean_r) / np.sqrt(var_r + self.eps)

        # Scale and shift: out = gamma * X_norm + beta.
        # This affine transformation lets the network learn to undo normalization
        # if that improves the loss. With gamma=1, beta=0, output equals X_norm.
        out = gamma_r * X_norm + beta_r

        # Cache all intermediates needed for the backward pass.
        # X is needed for dvar and dmean, X_norm for dgamma, statistics for dX.
        self.cache = (X, X_norm, mean_r, var_r, gamma_r)
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """
        Backward pass through batch normalization.

        Computes gradients for gamma, beta, and the input X.
        The gradient through BatchNorm is complex because the normalization
        involves the batch mean and variance, which depend on ALL inputs.

        The derivation follows the chain rule through:
        1. d(loss)/d(gamma) = sum(d_out * X_norm)
        2. d(loss)/d(beta) = sum(d_out)
        3. d(loss)/d(X) requires going through the mean and variance paths

        Parameters
        ----------
        d_out : np.ndarray of shape (N, C, H, W)
            Upstream gradient from the next layer.
        lr : float
            Learning rate for parameter updates.

        Returns
        -------
        np.ndarray of shape (N, C, H, W)
            Gradient with respect to the input X.
        """
        # Retrieve cached forward pass intermediates.
        X, X_norm, mean_r, var_r, gamma_r = self.cache
        N, C, H, W = X.shape
        M = N * H * W  # Total number of elements per channel (for averaging)

        # Gradient for gamma: dgamma[c] = sum over (n, h, w) of d_out[n,c,h,w] * X_norm[n,c,h,w]
        # This measures how much each channel's scale should change.
        dgamma = np.sum(d_out * X_norm, axis=(0, 2, 3))

        # Gradient for beta: dbeta[c] = sum over (n, h, w) of d_out[n,c,h,w]
        # This measures how much each channel's shift should change.
        dbeta = np.sum(d_out, axis=(0, 2, 3))

        # Gradient through the gamma scaling: dX_norm = d_out * gamma
        dX_norm = d_out * gamma_r

        # Inverse standard deviation, reused multiple times in gradient computation.
        std_inv = 1.0 / np.sqrt(var_r + self.eps)

        # Gradient for variance: dvar = sum(dX_norm * (X - mean) * (-0.5) * (var + eps)^(-3/2))
        # This comes from differentiating 1/sqrt(var + eps) with respect to var.
        dvar = np.sum(dX_norm * (X - mean_r) * (-0.5) * (var_r + self.eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)

        # Gradient for mean: dmean = sum(dX_norm * (-std_inv)) + dvar * sum(-2*(X-mean)) / M
        # The first term comes from the (X - mean) subtraction in normalization.
        # The second term comes from mean's influence on variance computation.
        dmean = np.sum(dX_norm * (-std_inv), axis=(0, 2, 3), keepdims=True) + dvar * np.sum(
            -2.0 * (X - mean_r), axis=(0, 2, 3), keepdims=True
        ) / M

        # Final input gradient: combine all three paths.
        # dX = dX_norm * std_inv (direct path through normalization)
        #    + dvar * 2*(X-mean)/M (path through variance computation)
        #    + dmean / M (path through mean computation)
        dX = dX_norm * std_inv + dvar * 2.0 * (X - mean_r) / M + dmean / M

        # Update learnable parameters using SGD.
        # gamma controls the scale of normalized activations.
        self.gamma -= lr * dgamma
        # beta controls the shift of normalized activations.
        self.beta -= lr * dbeta

        return dX  # Return gradient for the previous layer


class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.

    ReLU(x) = max(0, x)

    The simplest and most widely used activation function in deep learning.
    It introduces non-linearity while being computationally cheap (just a threshold).

    Properties:
    - Forward: f(x) = max(0, x) -- zero out negative values, pass positive unchanged
    - Backward: f'(x) = 1 if x > 0, else 0 -- gradient is 1 for positive, 0 for negative
    - No vanishing gradient for positive activations (gradient = 1)
    - Can cause "dying ReLU" problem: neurons stuck at 0 if all inputs become negative
    - No learnable parameters: purely a fixed non-linear transformation
    """

    def __init__(self):
        # Cache stores the input tensor for backward pass gradient computation.
        # We need to know which elements were positive (gradient = 1) vs negative (gradient = 0).
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply ReLU activation: output = max(0, input).

        All negative values become zero (killed), positive values pass through unchanged.
        This creates sparse activations (many zeros), which helps with computational
        efficiency and prevents feature interference.

        Parameters
        ----------
        X : np.ndarray
            Input tensor of any shape.

        Returns
        -------
        np.ndarray : Same shape as input, with negatives zeroed out.
        """
        self.cache = X  # Store input to determine gradient mask in backward pass
        return np.maximum(0, X)  # Element-wise max with zero

    def backward(self, d_out: np.ndarray, lr: float = 0.0) -> np.ndarray:
        """
        Backward pass through ReLU.

        The gradient of ReLU is a binary mask: 1 where input was positive, 0 elsewhere.
        d(loss)/dX = d_out * (X > 0)

        Negative inputs had zero output, so they get zero gradient (no learning signal).
        This is the "dying ReLU" problem: if a neuron's inputs are always negative,
        it permanently stops learning because gradients are always zero.

        Parameters
        ----------
        d_out : np.ndarray
            Upstream gradient from the next layer.
        lr : float
            Not used (ReLU has no learnable parameters).

        Returns
        -------
        np.ndarray : Gradient with respect to input, masked by positive input positions.
        """
        # Create binary mask: 1.0 where cached input > 0, 0.0 elsewhere.
        # Multiply upstream gradient by this mask to zero out gradients for
        # positions that were killed (set to zero) in the forward pass.
        return d_out * (self.cache > 0).astype(np.float32)


class AvgPool2D:
    """
    Average Pooling layer with configurable pool size.

    Downsamples spatial dimensions by averaging non-overlapping patches.
    Unlike max pooling (which takes the maximum), average pooling computes
    the mean of each patch, preserving more information about the overall
    activation magnitude at the cost of losing sharp feature localization.

    In ResNets, average pooling is preferred over max pooling for downsampling
    because it preserves gradient flow more uniformly (every input contributes
    to the output, unlike max pooling where only the max contributes).
    """

    def __init__(self, pool_size: int = 2):
        # Pool size determines both the spatial window and the downsampling factor.
        # pool_size=2 means 2x2 windows, reducing spatial dims by half.
        self.ps = pool_size

        # Cache stores the input shape for reconstructing the gradient tensor
        # during backward pass (we need to know the original spatial dimensions).
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: average each non-overlapping pool_size x pool_size patch.

        For a 2x2 pool on a 16x16 image, this produces an 8x8 output where
        each output pixel is the mean of a 2x2 patch of input pixels.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
            Input feature maps.

        Returns
        -------
        np.ndarray of shape (N, C, H//ps, W//ps)
            Downsampled feature maps.
        """
        N, C, H, W = X.shape
        Ho = H // self.ps  # Output height: integer division by pool size
        Wo = W // self.ps  # Output width: integer division by pool size

        # Pre-allocate output tensor for the downsampled feature maps.
        out = np.zeros((N, C, Ho, Wo), dtype=np.float32)

        # Explicit loop over all dimensions to compute the average of each patch.
        # This is O(N * C * Ho * Wo * ps * ps) but makes the operation transparent.
        for n in range(N):  # Loop over batch samples
            for c in range(C):  # Loop over channels
                for i in range(Ho):  # Loop over output height positions
                    for j in range(Wo):  # Loop over output width positions
                        # Extract the pool_size x pool_size patch and compute its mean.
                        # The patch spans from (i*ps, j*ps) to ((i+1)*ps, (j+1)*ps).
                        out[n, c, i, j] = np.mean(
                            X[n, c, i * self.ps : (i + 1) * self.ps, j * self.ps : (j + 1) * self.ps]
                        )

        # Cache the original input shape for the backward pass.
        # We need to know (N, C, H, W) to create the properly-sized gradient tensor.
        self.cache = X.shape
        return out

    def backward(self, d_out: np.ndarray, lr: float = 0.0) -> np.ndarray:
        """
        Backward pass through average pooling.

        Each output pixel was the average of ps*ps input pixels, so the gradient
        is distributed equally: each input pixel in the patch receives d_out / (ps*ps).

        This uniform distribution of gradients is a key advantage of average pooling
        over max pooling, where only the max-valued input receives gradient.

        Parameters
        ----------
        d_out : np.ndarray of shape (N, C, Ho, Wo)
            Upstream gradient from the next layer.
        lr : float
            Not used (no learnable parameters in pooling).

        Returns
        -------
        np.ndarray of shape (N, C, H, W)
            Gradient with respect to the full-resolution input.
        """
        orig_shape = self.cache  # Retrieve original input shape
        N, C, Ho, Wo = d_out.shape

        # Initialize input gradient to zeros at the original spatial resolution.
        dX = np.zeros(orig_shape, dtype=np.float32)

        # Scale factor: each input pixel contributed 1/(ps*ps) to the output average,
        # so it receives that fraction of the upstream gradient.
        scale = 1.0 / (self.ps * self.ps)

        # Distribute gradient equally to all pixels in each pooling patch.
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        # Each pixel in the ps x ps patch receives the same scaled gradient.
                        # This is because d(mean(x1,...,xk))/dxi = 1/k for all i.
                        dX[n, c, i * self.ps : (i + 1) * self.ps, j * self.ps : (j + 1) * self.ps] += (
                            d_out[n, c, i, j] * scale
                        )
        return dX


class GlobalAvgPool:
    """
    Global Average Pooling: (N, C, H, W) -> (N, C).

    Computes the mean of each channel's entire spatial feature map,
    producing a single value per channel. This replaces the need for
    large fully connected layers at the end of the network.

    Benefits:
    1. Drastically reduces parameters (no large FC weight matrix)
    2. Reduces overfitting by removing spatial position dependence
    3. Makes the network more robust to spatial translations
    4. Output size is independent of input spatial dimensions

    In the original ResNet paper, Global Average Pooling feeds directly
    into the final classification FC layer, eliminating the need for
    multiple FC layers that earlier architectures (VGG, AlexNet) used.
    """

    def __init__(self):
        # Cache stores input shape for backward pass gradient reconstruction.
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute global average over spatial dimensions.

        For each sample and channel, average all H*W spatial positions into
        a single scalar value. This aggregates spatial information into a
        channel descriptor.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
            Input feature maps.

        Returns
        -------
        np.ndarray of shape (N, C)
            Spatially-averaged channel descriptors.
        """
        self.cache = X.shape  # Store shape for backward pass gradient expansion
        # Mean over axes 2 and 3 (H and W), collapsing spatial dims.
        # Result shape: (N, C) -- one value per sample per channel.
        return X.mean(axis=(2, 3))

    def backward(self, d_out: np.ndarray, lr: float = 0.0) -> np.ndarray:
        """
        Backward pass through Global Average Pooling.

        Each of the H*W spatial positions contributed equally to the mean,
        so each receives gradient = d_out / (H * W).

        Parameters
        ----------
        d_out : np.ndarray of shape (N, C)
            Upstream gradient from the fully connected layer.
        lr : float
            Not used (no learnable parameters).

        Returns
        -------
        np.ndarray of shape (N, C, H, W)
            Gradient distributed uniformly across all spatial positions.
        """
        N, C, H, W = self.cache
        scale = 1.0 / (H * W)  # Each spatial position contributes 1/(H*W) to the mean
        # Reshape d_out from (N, C) to (N, C, 1, 1) then broadcast to (N, C, H, W).
        # Every spatial position receives the same gradient value, scaled by 1/(H*W).
        return (d_out.reshape(N, C, 1, 1) * np.ones((N, C, H, W), dtype=np.float32)) * scale


class FullyConnected:
    """
    Dense (fully connected) layer: y = X @ W + b.

    Each output neuron is connected to every input feature.
    In this ResNet, the FC layer maps from the global average pooled
    features (n_filters * 2 channels) to class logits (n_classes).

    This is the simplest neural network layer: a linear transformation
    followed by (optionally) an activation function (applied externally).
    """

    def __init__(
        self,
        in_f: int,  # Number of input features (e.g., number of channels after GAP)
        out_f: int,  # Number of output features (e.g., number of classes)
        init_scale: float = 0.1,  # Weight initialization standard deviation
    ):
        # Weight matrix W: shape (in_features, out_features).
        # Each column is a weight vector for one output neuron.
        # Initialized with small random Gaussian values to break symmetry.
        self.W = np.random.randn(in_f, out_f).astype(np.float32) * init_scale

        # Bias vector b: shape (out_features,).
        # One bias per output neuron, initialized to zero.
        self.b = np.zeros(out_f, dtype=np.float32)

        # Cache stores the input for backward pass weight gradient computation.
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute y = X @ W + b.

        Matrix multiplication maps each input vector to the output space.
        The bias shifts the decision boundary, allowing classification of
        data that doesn't pass through the origin.

        Parameters
        ----------
        X : np.ndarray of shape (N, in_features)
            Input feature vectors.

        Returns
        -------
        np.ndarray of shape (N, out_features)
            Output logits (unnormalized class scores).
        """
        self.cache = X  # Store input for backward pass dW computation
        # Matrix multiply X (N, in_f) @ W (in_f, out_f) -> (N, out_f), then add bias.
        return X @ self.W + self.b

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """
        Backward pass through the fully connected layer.

        Gradients:
        - dX = d_out @ W^T (gradient for previous layer)
        - dW = X^T @ d_out (gradient for weights)
        - db = mean(d_out, axis=0) (gradient for biases)

        Parameters
        ----------
        d_out : np.ndarray of shape (N, out_features)
            Upstream gradient from the loss function.
        lr : float
            Learning rate for SGD parameter updates.

        Returns
        -------
        np.ndarray of shape (N, in_features)
            Gradient with respect to the input.
        """
        X = self.cache  # Retrieve cached input
        N = X.shape[0]  # Batch size for gradient averaging

        # Input gradient: propagate error back through the weight matrix.
        # dX = d_out @ W^T maps the output gradient back to input space.
        dX = d_out @ self.W.T

        # Weight update: dW = X^T @ d_out computes how each weight contributed
        # to the error. Divide by N for mean gradient across the batch.
        self.W -= lr * (X.T @ d_out) / N

        # Bias update: average gradient across the batch.
        # Each bias shifts all samples equally, so we average their gradients.
        self.b -= lr * np.mean(d_out, axis=0)

        return dX  # Return gradient for the previous layer


class SoftmaxCrossEntropy:
    """
    Combined Softmax + Cross-Entropy loss function.

    Combining these two operations is numerically more stable than computing
    softmax probabilities first and then cross-entropy loss separately.

    Softmax: P(class=k) = exp(z_k) / sum(exp(z_j))
    Cross-Entropy: L = -log(P(class=y_true))

    The combined gradient has a beautifully simple form:
    dL/dz_k = P(k) - 1{k == y_true}
    (predicted probability minus the one-hot true label)

    This simplicity is one reason softmax + cross-entropy is the standard
    loss function for multi-class classification.
    """

    def __init__(self):
        # Cache stores (probs, labels) for the backward pass.
        self.cache = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute softmax probabilities and cross-entropy loss.

        The logit-shifting trick (subtracting max) prevents numerical overflow
        in the exponential: exp(z - max(z)) produces the same probabilities
        but avoids computing exp(very_large_number) = inf.

        Parameters
        ----------
        logits : np.ndarray of shape (N, n_classes)
            Raw (unnormalized) class scores from the final FC layer.
        y : np.ndarray of shape (N,)
            True class labels (integer indices).

        Returns
        -------
        probs : np.ndarray of shape (N, n_classes)
            Softmax probability distribution over classes.
        loss : float
            Mean cross-entropy loss across the batch.
        """
        # Numerical stability: subtract the max logit per sample.
        # This prevents exp(very_large) = inf while maintaining the same probabilities.
        # Mathematically: softmax(z - c) = softmax(z) for any constant c.
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        # Compute exponentials of the shifted logits.
        exp_s = np.exp(shifted)

        # Normalize to get probabilities that sum to 1 per sample.
        # probs[n, k] = exp(shifted[n, k]) / sum_j(exp(shifted[n, j]))
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

        N = logits.shape[0]  # Batch size

        # Cross-entropy loss: L = -mean(log(P(y_true)))
        # We index probs with the true labels to get predicted probability of correct class.
        # The 1e-12 prevents log(0) which would produce -inf.
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

        # Cache probabilities and labels for the backward pass.
        self.cache = (probs, y)

        return probs, loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss with respect to logits.

        The gradient has the elegant form: dL/dz_k = P(k) - 1{k == y_true}
        For the correct class: gradient = predicted_prob - 1 (pushes prob toward 1)
        For wrong classes: gradient = predicted_prob (pushes prob toward 0)

        Returns
        -------
        np.ndarray of shape (N, n_classes)
            Gradient of loss with respect to input logits.
        """
        probs, y = self.cache
        N = probs.shape[0]
        d = probs.copy()  # Start with predicted probabilities
        # Subtract 1 from the true class probability: P(y_true) - 1.
        # This creates the gradient dL/dz that pushes the correct class probability
        # toward 1.0 and all other class probabilities toward 0.0.
        d[np.arange(N), y] -= 1.0
        return d / N  # Average over the batch


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------


class ResidualBlock:
    """
    A residual block implementing the core innovation of ResNet: y = F(x) + shortcut(x)

    The residual block computes:
        F(x) = BN(Conv(ReLU(BN(Conv(x)))))  -- the "residual function"
        shortcut(x) = x  (identity, if dimensions match)
                     OR BN(Conv_1x1(x))  (projection, if dimensions change)
        output = ReLU(F(x) + shortcut(x))

    The key insight is that learning F(x) = H(x) - x (the residual) is easier
    than learning H(x) directly. If the optimal mapping is close to identity
    (which is common in deep networks), F(x) just needs to learn values near zero,
    which is much easier than learning a full identity mapping.

    Skip Connection Types:
    - Identity shortcut (in_ch == out_ch): shortcut = x
      No parameters needed. Gradient flows directly through: dL/dx += dL/dout
    - Projection shortcut (in_ch != out_ch): shortcut = BN(Conv1x1(x))
      Uses a 1x1 convolution to match channel dimensions.

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    init_scale : float
        Weight initialization scale for conv layers.
    """

    def __init__(self, in_ch: int, out_ch: int, init_scale: float = 0.1):
        # MAIN PATH: Conv -> BN -> ReLU -> Conv -> BN
        # First convolution: transforms input features while preserving spatial dims (padding=1).
        # 3x3 kernel captures local spatial patterns (edges, textures, corners).
        self.conv1 = Conv2D(in_ch, out_ch, 3, padding=1, init_scale=init_scale)

        # First batch normalization: normalizes activations after conv1.
        # Stabilizes training by reducing internal covariate shift between layers.
        self.bn1 = BatchNorm(out_ch)

        # First ReLU: introduces non-linearity after BN.
        # Without non-linearity, stacking linear layers would collapse to a single linear layer.
        self.relu1 = ReLU()

        # Second convolution: further transforms features while preserving spatial dims.
        # out_ch -> out_ch maintains channel count within the residual function F(x).
        self.conv2 = Conv2D(out_ch, out_ch, 3, padding=1, init_scale=init_scale)

        # Second batch normalization: normalizes before the skip connection addition.
        # BN is placed before the addition so both paths are on similar scales.
        self.bn2 = BatchNorm(out_ch)

        # Final ReLU: applied AFTER the skip connection addition.
        # This is the "post-addition activation" from the original ResNet paper.
        self.relu_out = ReLU()

        # SHORTCUT PATH: determine if we need a projection (dimension change).
        # Identity shortcut (no parameters) when input and output channels match.
        # Projection shortcut (1x1 conv + BN) when channels differ.
        self.use_projection = in_ch != out_ch

        if self.use_projection:
            # 1x1 convolution for channel dimension matching.
            # This is the "projection shortcut" from the ResNet paper.
            # 1x1 conv changes channel count without affecting spatial dimensions.
            self.proj_conv = Conv2D(in_ch, out_ch, 1, padding=0, init_scale=init_scale)

            # BN on the projection path normalizes to match the main path's scale.
            self.proj_bn = BatchNorm(out_ch)

    def set_training(self, training: bool):
        """
        Set training mode for all batch normalization layers in this block.

        During training, BN uses batch statistics. During inference, BN uses
        accumulated running statistics for consistent predictions.

        Parameters
        ----------
        training : bool
            True for training mode, False for inference mode.
        """
        self.bn1.training = training  # Main path first BN
        self.bn2.training = training  # Main path second BN
        if self.use_projection:
            self.proj_bn.training = training  # Projection path BN (if exists)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass with skip connection: y = ReLU(F(x) + shortcut(x)).

        The skip connection is the KEY innovation of ResNet.
        Without it, gradients must flow through every layer sequentially,
        causing vanishing gradients in deep networks.
        With it, gradients have a direct path: dL/dx = dL/dy * (dF/dx + 1).
        The "+1" from the identity shortcut ensures gradients never vanish.

        Parameters
        ----------
        X : np.ndarray of shape (N, in_ch, H, W)
            Input feature maps.

        Returns
        -------
        np.ndarray of shape (N, out_ch, H, W)
            Output feature maps after residual connection.
        """
        # MAIN PATH: F(x) = BN(Conv(ReLU(BN(Conv(x)))))
        h = self.conv1.forward(X)  # Conv: (N, in_ch, H, W) -> (N, out_ch, H, W)
        h = self.bn1.forward(h)  # BN: normalize activations per channel
        h = self.relu1.forward(h)  # ReLU: introduce non-linearity
        h = self.conv2.forward(h)  # Conv: (N, out_ch, H, W) -> (N, out_ch, H, W)
        h = self.bn2.forward(h)  # BN: normalize before addition

        # SHORTCUT PATH: identity or projection
        if self.use_projection:
            # Projection shortcut: 1x1 conv changes channel count to match main path.
            # Without this, we can't add F(x) and x when they have different channel counts.
            shortcut = self.proj_conv.forward(X)  # 1x1 conv: (N, in_ch, H, W) -> (N, out_ch, H, W)
            shortcut = self.proj_bn.forward(shortcut)  # BN: normalize projection output
        else:
            # Identity shortcut: x passes through unchanged.
            # This is the most common case and requires zero additional parameters.
            shortcut = X

        # RESIDUAL CONNECTION: y = F(x) + shortcut(x)
        # This element-wise addition is the core of residual learning.
        # If F(x) learns to be zero, the block becomes an identity mapping.
        out = h + shortcut

        # Post-addition ReLU activation.
        # Applied after the skip connection to introduce non-linearity.
        out = self.relu_out.forward(out)
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """
        Backward pass through the residual block.

        The gradient splits at the addition point: both the main path and
        the shortcut path receive the FULL upstream gradient (by the chain rule).
        At the input X, gradients from both paths are summed: dX = dX_main + dX_shortcut.

        This gradient splitting/merging is what makes ResNet training stable:
        even if the main path gradients vanish (dF/dx -> 0), the shortcut path
        still provides gradient signal (didentity/dx = 1).

        Parameters
        ----------
        d_out : np.ndarray
            Upstream gradient from the next layer.
        lr : float
            Learning rate for parameter updates.

        Returns
        -------
        np.ndarray : Gradient with respect to the block's input X.
        """
        # Backward through the post-addition ReLU.
        # This masks gradients where the output of F(x) + shortcut(x) was negative.
        d = self.relu_out.backward(d_out, lr)

        # GRADIENT SPLITS at the addition: both paths get the full gradient.
        # This is because d(a + b)/da = 1 and d(a + b)/db = 1.
        # Both the main path and shortcut path contributed to the sum.
        d_main = d.copy()  # Gradient flowing back through the main path F(x)
        d_shortcut = d.copy()  # Gradient flowing back through the shortcut path

        # BACKWARD THROUGH MAIN PATH: reverse the forward operations.
        # Order is reversed: BN2 -> Conv2 -> ReLU1 -> BN1 -> Conv1
        d_main = self.bn2.backward(d_main, lr)  # Backward through second BN
        d_main = self.conv2.backward(d_main, lr)  # Backward through second conv
        d_main = self.relu1.backward(d_main, lr)  # Backward through first ReLU
        d_main = self.bn1.backward(d_main, lr)  # Backward through first BN
        d_main = self.conv1.backward(d_main, lr)  # Backward through first conv

        # BACKWARD THROUGH SHORTCUT PATH
        if self.use_projection:
            # If projection was used, gradient flows through projection BN and conv.
            d_shortcut = self.proj_bn.backward(d_shortcut, lr)
            d_shortcut = self.proj_conv.backward(d_shortcut, lr)
        # If identity shortcut: d_shortcut = d (gradient passes through unchanged)

        # SUM gradients from both paths.
        # This is where the "gradient highway" effect manifests:
        # dX = dX_main + dX_shortcut
        # Even if dX_main is small (vanishing gradients in main path),
        # dX_shortcut provides a direct gradient signal.
        dX = d_main + d_shortcut
        return dX


# ---------------------------------------------------------------------------
# ResNet Model
# ---------------------------------------------------------------------------


class ResNet:
    """
    Simplified ResNet with residual blocks.

    Full architecture:
    Input (1, 16, 16)
    -> Conv2D(1->n_filters, 3x3, pad=1) -> BN -> ReLU         [Initial stem]
    -> ResidualBlock(n_filters->n_filters)                       [Identity shortcut]
    -> AvgPool(2x2)                                              [8x8]
    -> ResidualBlock(n_filters->n_filters*2)                     [Projection shortcut]
    -> AvgPool(2x2)                                              [4x4]
    -> GlobalAvgPool                                             [n_filters*2]
    -> FC(n_filters*2 -> n_classes)                              [Class logits]
    -> SoftmaxCrossEntropy                                       [Loss]

    The two residual blocks demonstrate both types of skip connections:
    1. Block 1: identity shortcut (same channels in and out)
    2. Block 2: projection shortcut (channel count doubles)
    """

    def __init__(
        self,
        img_channels: int = 1,  # Input image channels (1 for grayscale)
        img_size: int = 16,  # Input spatial dimension (H = W = 16)
        n_classes: int = 5,  # Number of classification categories
        n_filters: int = 8,  # Base number of filters (doubles in block 2)
        init_scale: float = 0.1,  # Weight initialization standard deviation
    ):
        # INITIAL STEM: Conv -> BN -> ReLU
        # The stem converts raw pixels into initial feature maps.
        # 3x3 conv with padding=1 preserves spatial dimensions: 16x16 -> 16x16.
        # n_filters output channels create the initial feature representation.
        self.conv_init = Conv2D(img_channels, n_filters, 3, padding=1, init_scale=init_scale)

        # Batch normalization on the stem stabilizes the very first feature maps.
        self.bn_init = BatchNorm(n_filters)

        # ReLU activation introduces non-linearity after the first conv+BN.
        self.relu_init = ReLU()

        # RESIDUAL BLOCK 1: n_filters -> n_filters (IDENTITY shortcut)
        # Input and output have the same channel count, so the skip connection
        # is a simple identity: y = F(x) + x. No extra parameters needed.
        # This block refines the initial features while maintaining dimensions.
        self.res_block1 = ResidualBlock(n_filters, n_filters, init_scale)

        # AVERAGE POOL 1: reduce spatial dimensions by half (16x16 -> 8x8)
        # Downsampling increases the effective receptive field of subsequent layers,
        # allowing them to capture larger-scale patterns.
        self.pool1 = AvgPool2D(2)

        # RESIDUAL BLOCK 2: n_filters -> n_filters*2 (PROJECTION shortcut)
        # Channel count doubles (e.g., 8 -> 16), requiring a projection shortcut
        # with a 1x1 conv to match dimensions for the addition.
        # More channels = more capacity to represent complex features.
        self.res_block2 = ResidualBlock(n_filters, n_filters * 2, init_scale)

        # AVERAGE POOL 2: further reduce spatial dims (8x8 -> 4x4)
        self.pool2 = AvgPool2D(2)

        # CLASSIFICATION HEAD: GlobalAvgPool -> FC -> SoftmaxCrossEntropy
        # Global average pooling collapses 4x4 spatial dims to a single value per channel.
        self.gap = GlobalAvgPool()

        # Fully connected layer maps channel features to class logits.
        # Input: n_filters*2 (from res_block2), Output: n_classes.
        self.fc = FullyConnected(n_filters * 2, n_classes, init_scale)

        # Loss function: combined softmax + cross-entropy for stable training.
        self.loss_fn = SoftmaxCrossEntropy()

    def set_training(self, training: bool):
        """
        Set training mode for all BatchNorm layers in the network.

        This affects whether BN uses batch statistics (training) or
        running statistics (inference). Must be called before forward pass.

        Parameters
        ----------
        training : bool
            True for training mode, False for inference mode.
        """
        self.bn_init.training = training  # Stem BN layer
        self.res_block1.set_training(training)  # Block 1 BN layers
        self.res_block2.set_training(training)  # Block 2 BN layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire ResNet.

        Data flow:
        (N,1,16,16) -> Conv+BN+ReLU -> (N,8,16,16)
        -> ResBlock1 -> (N,8,16,16) -> AvgPool -> (N,8,8,8)
        -> ResBlock2 -> (N,16,8,8) -> AvgPool -> (N,16,4,4)
        -> GAP -> (N,16) -> FC -> (N,n_classes)

        Parameters
        ----------
        X : np.ndarray of shape (N, 1, 16, 16)
            Input images in NCHW format.

        Returns
        -------
        np.ndarray of shape (N, n_classes)
            Raw logits (unnormalized class scores).
        """
        h = self.conv_init.forward(X)  # Initial conv: (N,1,16,16) -> (N,8,16,16)
        h = self.bn_init.forward(h)  # Normalize initial features
        h = self.relu_init.forward(h)  # Non-linearity
        h = self.res_block1.forward(h)  # ResBlock1 with identity shortcut
        h = self.pool1.forward(h)  # Downsample: (N,8,16,16) -> (N,8,8,8)
        h = self.res_block2.forward(h)  # ResBlock2 with projection shortcut
        h = self.pool2.forward(h)  # Downsample: (N,16,8,8) -> (N,16,4,4)
        h = self.gap.forward(h)  # Global avg pool: (N,16,4,4) -> (N,16)
        logits = self.fc.forward(h)  # FC: (N,16) -> (N,n_classes)
        return logits

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute softmax probabilities and cross-entropy loss.

        Parameters
        ----------
        logits : np.ndarray of shape (N, n_classes)
            Raw class scores from forward pass.
        y : np.ndarray of shape (N,)
            True class labels.

        Returns
        -------
        probs : np.ndarray of shape (N, n_classes)
            Softmax probability distribution.
        loss : float
            Mean cross-entropy loss.
        """
        return self.loss_fn.forward(logits, y)

    def backward(self, lr: float):
        """
        Backward pass through the entire ResNet, updating all parameters.

        Propagates gradients in reverse order through every layer,
        from the loss function back to the initial convolution.

        The gradient flow through residual blocks demonstrates the key benefit:
        gradients flow through BOTH the main path and the skip connection,
        preventing vanishing gradients even in this simple 2-block network.

        Parameters
        ----------
        lr : float
            Learning rate for SGD parameter updates in every layer.
        """
        d = self.loss_fn.backward()  # Start: gradient from loss function
        d = self.fc.backward(d, lr)  # FC layer: update W, b; propagate gradient
        d = self.gap.backward(d, lr)  # GAP: distribute gradient to spatial positions
        d = self.pool2.backward(d, lr)  # AvgPool2: expand gradient to 8x8
        d = self.res_block2.backward(d, lr)  # ResBlock2: gradient splits at skip connection
        d = self.pool1.backward(d, lr)  # AvgPool1: expand gradient to 16x16
        d = self.res_block1.backward(d, lr)  # ResBlock1: gradient splits at skip connection
        d = self.relu_init.backward(d, lr)  # ReLU: mask negative gradients
        d = self.bn_init.backward(d, lr)  # BN: complex gradient through normalization
        d = self.conv_init.backward(d, lr)  # Initial conv: update first filters

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using inference mode (running BN statistics).

        Temporarily switches to eval mode for prediction, then restores
        training mode. This ensures BN uses stable running statistics
        instead of noisy single-batch statistics.

        Parameters
        ----------
        X : np.ndarray of shape (N, 1, H, W)
            Input images.

        Returns
        -------
        np.ndarray of shape (N,)
            Predicted class indices.
        """
        self.set_training(False)  # Switch to inference mode (use running BN stats)
        logits = self.forward(X)  # Forward pass
        self.set_training(True)  # Restore training mode for subsequent training
        return np.argmax(logits, axis=1)  # Return class with highest logit


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,  # Training images: (N, 1, H, W) float32
    y_train: np.ndarray,  # Training labels: (N,) int64
    img_channels: int = 1,  # Number of image channels (1 for grayscale)
    img_size: int = 16,  # Spatial dimension of images
    n_classes: int = 5,  # Number of classification categories
    n_filters: int = 8,  # Base filter count for the ResNet
    learning_rate: float = 0.01,  # SGD step size for weight updates
    n_epochs: int = 10,  # Number of full passes through the training data
    batch_size: int = 32,  # Number of samples per mini-batch gradient update
    init_scale: float = 0.1,  # Weight initialization standard deviation
    verbose: bool = True,  # Whether to print training progress
) -> Dict[str, Any]:
    """
    Train the NumPy ResNet from scratch using mini-batch SGD.

    This function creates a new ResNet model, trains it on the provided data
    using stochastic gradient descent with mini-batches, and returns the
    trained model along with training history.

    Mini-batch SGD provides a balance between:
    - Full-batch GD: stable gradients but slow (uses all data per step)
    - Single-sample SGD: fast but noisy (uses one sample per step)
    - Mini-batch SGD: moderate noise (regularization) with efficient computation

    Returns dict with 'model', 'train_losses', 'train_accs'.
    """
    # Create a fresh ResNet model with the specified hyperparameters.
    # The model architecture is: Conv->BN->ReLU->ResBlock->Pool->ResBlock->Pool->GAP->FC
    model = ResNet(
        img_channels=img_channels,
        img_size=img_size,
        n_classes=n_classes,
        n_filters=n_filters,
        init_scale=init_scale,
    )

    # Set model to training mode: BN uses batch statistics.
    model.set_training(True)

    N = len(X_train)  # Total number of training samples

    # History tracking: record loss and accuracy per epoch for monitoring convergence.
    train_losses, train_accs = [], []

    # TRAINING LOOP: iterate over the dataset n_epochs times.
    for epoch in range(n_epochs):
        # Shuffle training data at each epoch to break sequential patterns.
        # Without shuffling, the model would always see samples in the same order,
        # which can cause poor generalization and convergence to local minima.
        perm = np.random.permutation(N)

        epoch_loss = 0.0  # Accumulate loss across batches
        epoch_correct = 0  # Count correct predictions across batches
        n_batches = 0  # Count batches for averaging

        # MINI-BATCH LOOP: process the training data in chunks of batch_size.
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)  # Handle last batch (may be smaller)
            idx = perm[start:end]  # Get shuffled indices for this batch
            Xb = X_train[idx]  # Extract batch images
            yb = y_train[idx]  # Extract batch labels

            # FORWARD PASS: compute predictions and loss.
            logits = model.forward(Xb)  # (batch, n_classes) raw scores
            probs, loss = model.compute_loss(logits, yb)  # Softmax + CE loss

            # BACKWARD PASS: compute gradients and update all parameters.
            # This propagates the loss gradient through every layer in reverse order,
            # updating weights via SGD at each layer.
            model.backward(learning_rate)

            # Accumulate metrics for epoch-level reporting.
            epoch_loss += loss  # Sum batch losses
            epoch_correct += np.sum(np.argmax(probs, axis=1) == yb)  # Count correct
            n_batches += 1  # Count batches

        # Compute epoch-level metrics.
        avg_loss = epoch_loss / n_batches  # Mean loss across batches
        acc = epoch_correct / N  # Accuracy: fraction of correct predictions

        # Record training history for later analysis or plotting.
        train_losses.append(avg_loss)
        train_accs.append(acc)

        # Print progress at regular intervals (every 20% of total epochs).
        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    # Return trained model and training history in a dictionary.
    return {"model": model, "train_losses": train_losses, "train_accs": train_accs}


def validate(
    model_dict: Dict[str, Any],  # Dictionary containing the trained model
    X_val: np.ndarray,  # Validation images: (N, 1, H, W) float32
    y_val: np.ndarray,  # Validation labels: (N,) int64
) -> Dict[str, Any]:
    """
    Validate the model on a held-out validation set.

    Uses inference mode (running BN statistics) for evaluation.
    Computes overall accuracy, loss, and per-class accuracy to detect
    class imbalance issues or classes the model struggles with.

    Returns dict with 'accuracy', 'loss', 'per_class_accuracy'.
    """
    model = model_dict["model"]  # Extract the ResNet model from the dict

    # Switch to evaluation mode: BN uses running statistics instead of batch stats.
    # This provides consistent predictions regardless of batch composition.
    model.set_training(False)

    # Forward pass on the entire validation set.
    logits = model.forward(X_val)  # (N_val, n_classes)
    probs, loss = model.compute_loss(logits, y_val)  # Softmax + CE loss

    # Get predicted classes: argmax of probability distribution per sample.
    preds = np.argmax(probs, axis=1)

    # Compute overall accuracy: fraction of correctly predicted samples.
    accuracy = np.mean(preds == y_val)

    # Restore training mode for subsequent training iterations.
    model.set_training(True)

    # Compute per-class accuracy to identify classes the model struggles with.
    # This is important for detecting bias: overall accuracy can be high even if
    # the model completely fails on minority classes.
    classes = np.unique(y_val)
    per_class = {}
    for c in classes:
        mask = y_val == c  # Boolean mask for samples of this class
        # Accuracy for this class: fraction of correct predictions among its samples.
        per_class[int(c)] = float(np.mean(preds[mask] == y_val[mask])) if mask.sum() > 0 else 0.0

    # Print validation metrics.
    print(f"Validation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    for c, acc in per_class.items():
        print(f"  Class {c}: {acc:.4f}")

    return {"accuracy": accuracy, "loss": loss, "per_class_accuracy": per_class}


def test(
    model_dict: Dict[str, Any],  # Dictionary containing the trained model
    X_test: np.ndarray,  # Test images: (N, 1, H, W) float32
    y_test: np.ndarray,  # Test labels: (N,) int64
) -> Dict[str, Any]:
    """
    Final test evaluation on the held-out test set.

    This should only be called ONCE after all hyperparameter tuning is complete.
    Using the test set for model selection (e.g., choosing hyperparameters)
    would leak test information into the training process, giving overly
    optimistic performance estimates.

    Returns dict with accuracy, loss, and per-class metrics.
    """
    print("\n--- Test Set Evaluation ---")
    # Delegate to validate() which handles the actual evaluation.
    return validate(model_dict, X_test, y_test)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: "optuna.Trial",  # Optuna trial object for suggesting hyperparameters
    X_train: np.ndarray,  # Training images
    y_train: np.ndarray,  # Training labels
    X_val: np.ndarray,  # Validation images
    y_val: np.ndarray,  # Validation labels
    n_classes: int = 5,  # Number of classification categories
    img_size: int = 16,  # Image spatial dimension
) -> float:
    """
    Optuna objective function for ResNet hyperparameter optimization.

    Optuna uses TPE (Tree-structured Parzen Estimator) to intelligently
    search the hyperparameter space. Each trial suggests a set of hyperparameters,
    trains a model, and returns the validation accuracy as the objective to maximize.

    Returns
    -------
    float : Validation accuracy (higher is better).
    """
    # Suggest hyperparameters from defined ranges.
    # log=True means we search in log space, appropriate for learning rates
    # because the difference between 0.001 and 0.002 matters more than 0.049 and 0.050.
    lr = trial.suggest_float("learning_rate", 1e-3, 0.05, log=True)

    # Weight initialization scale: affects how quickly the network starts learning.
    # Too small: slow initial learning. Too large: unstable initial gradients.
    init_scale = trial.suggest_float("init_scale", 0.01, 0.2)

    # Number of filters: controls model capacity (4 = small, 8 = larger).
    n_filters = trial.suggest_categorical("n_filters", [4, 8])

    # Batch size: affects gradient noise and training dynamics.
    # Smaller batches = more noise (regularization) but slower convergence.
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # Train a model with the suggested hyperparameters (reduced epochs for speed).
    md = train(
        X_train, y_train,
        n_classes=n_classes, img_size=img_size,
        n_filters=n_filters, learning_rate=lr,
        n_epochs=5, batch_size=batch_size,
        init_scale=init_scale, verbose=False,
    )

    # Evaluate on validation set and return accuracy for Optuna to maximize.
    metrics = validate(md, X_val, y_val)
    return metrics["accuracy"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 5,
    img_size: int = 16,
    num_samples: int = 6,  # Number of hyperparameter configurations to try
) -> Dict[str, Any]:
    """
    Ray Tune distributed hyperparameter search.

    Ray Tune enables parallel trial execution across multiple CPUs,
    making large-scale hyperparameter searches much faster.
    Each trial runs independently with a different configuration.

    Returns
    -------
    dict : Best hyperparameter configuration found.
    """
    # Guard: skip if Ray is not installed.
    if not RAY_AVAILABLE:
        print("Ray not installed. Skipping.")
        return {}

    def trainable(config):
        """Inner function that Ray Tune calls with each trial configuration."""
        md = train(
            X_train, y_train,
            n_classes=n_classes, img_size=img_size,
            n_filters=config["n_filters"], learning_rate=config["lr"],
            n_epochs=5, batch_size=config["batch_size"],
            init_scale=config["init_scale"], verbose=False,
        )
        metrics = validate(md, X_val, y_val)
        # Report accuracy to Ray Tune for tracking and early stopping decisions.
        tune.report({"accuracy": metrics["accuracy"]})

    # Define the hyperparameter search space.
    # loguniform: sample uniformly in log space (good for learning rates).
    # uniform: sample uniformly in linear space (good for initialization scale).
    # choice: sample from discrete options (good for architecture choices).
    search_space = {
        "lr": tune.loguniform(1e-3, 0.05),
        "init_scale": tune.uniform(0.01, 0.2),
        "n_filters": tune.choice([4, 8]),
        "batch_size": tune.choice([16, 32]),
    }

    # Initialize Ray runtime with 2 CPUs for parallel trial execution.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    # Create and run the tuning job.
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, metric="accuracy", mode="max"),
    )
    results = tuner.fit()

    # Extract the best configuration and print results.
    best = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config: {best.config}")
    print(f"Best accuracy: {best.metrics['accuracy']:.4f}")

    # Clean up Ray resources to free memory.
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------


def compare_parameter_sets(
    n_samples: int = 800,  # Number of synthetic images for comparison
    img_size: int = 16,  # Image spatial dimension
) -> Dict[str, Any]:
    """
    Compare different ResNet configurations to demonstrate the impact of:

    1. Number of filters (n_filters=4 vs 8): affects model capacity
       - More filters = more feature maps = can detect more distinct patterns
       - But also more parameters = risk of overfitting on small datasets

    2. Learning rate sensitivity: 0.005 vs 0.01 vs 0.03
       - Too low: slow convergence, may get stuck in poor local minima
       - Too high: unstable training, loss may diverge or oscillate
       - Optimal: depends on model size, batch size, and data complexity

    3. Weight initialization scale: 0.05 vs 0.1 vs 0.2
       - Too small: vanishing gradients, slow initial learning
       - Too large: exploding gradients, unstable early training
       - The init_scale interacts with the learning rate: larger init may need smaller lr

    Parameters
    ----------
    n_samples : int
        Number of synthetic images.
    img_size : int
        Image spatial dimension.

    Returns
    -------
    results : dict mapping configuration names to accuracy scores.
    """
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: NumPy ResNet Configurations")
    print("=" * 70)

    n_classes = 5  # Number of pattern classes in synthetic data

    # Generate shared dataset for fair comparison across all configurations.
    # Using the same data ensures that accuracy differences reflect
    # hyperparameter effects, not data variation.
    X, y = generate_data(n_samples=n_samples, img_size=img_size, n_classes=n_classes, random_state=42)

    # Split into train and validation sets (70/15/15 split).
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]

    # Define configurations to compare.
    # Each config is a dict of hyperparameters that differ from the baseline.
    configs = {
        # --- Filter count comparison ---
        # Fewer filters: less capacity but faster training and less overfitting risk
        "n_filters=4": {
            "n_filters": 4, "learning_rate": 0.01, "n_epochs": 6,
            "batch_size": 32, "init_scale": 0.1,
        },
        # More filters: more capacity to capture diverse patterns but more parameters
        "n_filters=8": {
            "n_filters": 8, "learning_rate": 0.01, "n_epochs": 6,
            "batch_size": 32, "init_scale": 0.1,
        },
        # --- Learning rate comparison ---
        # Low learning rate: conservative updates, stable but potentially slow
        "lr=0.005": {
            "n_filters": 8, "learning_rate": 0.005, "n_epochs": 6,
            "batch_size": 32, "init_scale": 0.1,
        },
        # Medium learning rate: balanced convergence speed and stability
        "lr=0.01": {
            "n_filters": 8, "learning_rate": 0.01, "n_epochs": 6,
            "batch_size": 32, "init_scale": 0.1,
        },
        # High learning rate: faster convergence but risk of overshooting minima
        "lr=0.03": {
            "n_filters": 8, "learning_rate": 0.03, "n_epochs": 6,
            "batch_size": 32, "init_scale": 0.1,
        },
        # --- Initialization scale comparison ---
        # Small init: weights start near zero, gradients may be very small initially
        "init_scale=0.05": {
            "n_filters": 8, "learning_rate": 0.01, "n_epochs": 6,
            "batch_size": 32, "init_scale": 0.05,
        },
        # Large init: weights start further from zero, risk of initial instability
        "init_scale=0.2": {
            "n_filters": 8, "learning_rate": 0.01, "n_epochs": 6,
            "batch_size": 32, "init_scale": 0.2,
        },
    }

    # Train and evaluate each configuration.
    results = {}
    for name, params in configs.items():
        print(f"\n--- Configuration: {name} ---")
        model_dict = train(
            X_train, y_train,
            img_size=img_size, n_classes=n_classes,
            verbose=False, **params,
        )
        metrics = validate(model_dict, X_val, y_val)
        results[name] = metrics["accuracy"]

    # Print ranked summary of all configurations.
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Val Accuracy':>12}")
    print("-" * 50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<35} {acc:>12.4f}")
    print("-" * 50)

    # Interpretation of results.
    print("\nKey Takeaways:")
    print("  - Skip connections are the key innovation (compare with/without)")
    print("  - More filters increase capacity but add computation cost")
    print("  - Learning rate is critical: too high causes instability")
    print("  - Weight initialization affects early training dynamics")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Manufacturing Defect Detection
# ---------------------------------------------------------------------------


def real_world_demo() -> Dict[str, Any]:
    """
    Demonstrate the from-scratch ResNet on a manufacturing defect detection
    scenario: classifying product surface images as Clean, Scratch, or Dent.

    Manufacturing Quality Context:
    - Clean surfaces: uniform texture with only random micro-variations
    - Scratches: linear marks caused by friction or sharp contact
    - Dents: circular depressions caused by impact damage

    This demo shows that even a small from-scratch ResNet can learn to distinguish
    subtle surface texture differences between defect types, demonstrating the
    practical relevance of residual learning for industrial quality inspection.

    In production, manufacturing defect detection uses:
    - Pretrained ResNet/EfficientNet with transfer learning
    - High-resolution cameras (512x512 or larger)
    - Real defect databases (MVTec AD, Severstal Steel Defect)
    - Data augmentation for rare defect types

    Returns
    -------
    results : dict with model performance metrics.
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Manufacturing Defect Detection (Scratch vs Dent vs Clean)")
    print("=" * 70)

    # Seed random number generator for reproducible demo results.
    rng = np.random.RandomState(42)

    # Dataset parameters: small due to slow NumPy training.
    n_samples = 450  # 150 per class (small enough for CPU training)
    img_size = 16  # Spatial resolution (kept small for speed)
    n_classes = 3  # Clean, Scratch, Dent
    class_names = ["Clean", "Scratch", "Dent"]  # Human-readable class labels
    samples_per_class = n_samples // n_classes  # Balanced classes

    X_list, y_list = [], []  # Accumulator lists for images and labels

    # Generate synthetic manufacturing surface images for each defect class.
    for cls_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Base surface: uniform mid-gray (0.5) with subtle random texture.
            # Real metal/plastic surfaces have a baseline reflectance with
            # micro-variations from surface roughness.
            img = np.ones((img_size, img_size), dtype=np.float32) * 0.5
            img += rng.rand(img_size, img_size).astype(np.float32) * 0.05

            if cls_idx == 0:
                # CLEAN surface: only subtle noise on top of the uniform base.
                # Real clean surfaces have minimal variation, just sensor noise
                # and micro-roughness that doesn't constitute a defect.
                img += rng.randn(img_size, img_size).astype(np.float32) * 0.01

            elif cls_idx == 1:
                # SCRATCH defect: linear marks across the surface.
                # Scratches are caused by friction or sharp object contact,
                # producing elongated bright/dark lines along the scratch direction.
                n_scratches = rng.randint(1, 3)  # 1-2 scratches per image
                for _ in range(n_scratches):
                    # Random starting point and angle for each scratch.
                    y_s = rng.randint(0, img_size)  # Starting y-coordinate
                    x_s = rng.randint(0, img_size)  # Starting x-coordinate
                    angle = rng.uniform(0, np.pi)  # Scratch direction (radians)
                    length = rng.randint(5, 12)  # Scratch length in pixels
                    # Draw the scratch as a line of altered pixels.
                    for t in range(length):
                        sx = int(x_s + t * np.cos(angle))  # x position along scratch
                        sy = int(y_s + t * np.sin(angle))  # y position along scratch
                        if 0 <= sx < img_size and 0 <= sy < img_size:
                            # Scratches can be bright (specular reflection from groove edges)
                            # or dark (shadow inside the groove).
                            img[sy, sx] += rng.choice([-0.12, 0.12])

            elif cls_idx == 2:
                # DENT defect: circular depressions in the surface.
                # Dents are caused by impact damage, creating circular or elliptical
                # depressions. They appear as dark regions because the concave surface
                # reflects light away from the camera.
                n_dents = rng.randint(1, 3)  # 1-2 dents per image
                for _ in range(n_dents):
                    # Random center and radius for each dent.
                    dx = rng.randint(3, img_size - 3)  # Dent center x (avoiding edges)
                    dy = rng.randint(3, img_size - 3)  # Dent center y (avoiding edges)
                    radius = rng.randint(2, 5)  # Dent radius in pixels
                    # Create a 2D Gaussian mask for the dent shape.
                    yy, xx = np.ogrid[:img_size, :img_size]  # Coordinate grids
                    dist = np.sqrt((xx - dx) ** 2 + (yy - dy) ** 2)  # Distance from center
                    # Gaussian profile: smooth circular depression.
                    dent = np.exp(-dist**2 / (2 * radius**2)).astype(np.float32)
                    img -= dent * 0.10  # Subtract to create a dark depression

            # Add sensor noise to all images (realistic imaging noise).
            img += rng.randn(img_size, img_size).astype(np.float32) * 0.02

            # Clip to valid pixel range [0, 1].
            img = np.clip(img, 0, 1)

            X_list.append(img)
            y_list.append(cls_idx)

    # Convert to numpy arrays with NCHW format (1 channel for grayscale).
    X = np.array(X_list, dtype=np.float32).reshape(-1, 1, img_size, img_size)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle dataset to break class ordering.
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Generated {len(X)} synthetic product images ({img_size}x{img_size})")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

    # Split into train/val/test (70/15/15).
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train ResNet for defect detection.
    print("\n--- Training ResNet for defect detection ---")
    model_dict = train(
        X_train, y_train,
        img_size=img_size, n_classes=n_classes,
        n_filters=8, learning_rate=0.01,
        n_epochs=10, batch_size=16,
    )

    # Evaluate on validation and test sets.
    print("\n--- Validation ---")
    val_metrics = validate(model_dict, X_val, y_val)

    print("\n--- Test ---")
    test_metrics = test(model_dict, X_test, y_test)

    # Manufacturing-specific interpretation of results.
    print("\n--- Manufacturing Quality Metrics ---")
    print(f"Overall Accuracy: {test_metrics['accuracy']:.4f}")
    for i, name in enumerate(class_names):
        if i in test_metrics.get("per_class_accuracy", {}):
            print(f"  {name}: {test_metrics['per_class_accuracy'][i]:.4f}")

    print("\nNote: This from-scratch ResNet is for educational purposes.")
    print("Production defect detection uses PyTorch/TensorFlow with pretrained models.")

    return {
        "test_accuracy": test_metrics["accuracy"],
        "per_class_accuracy": test_metrics.get("per_class_accuracy", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Run the full NumPy ResNet pipeline demonstrating all capabilities:

    1. Generate synthetic data with class-specific spatial patterns
    2. Split into train/validation/test sets
    3. Train the ResNet with residual (skip) connections
    4. Validate on held-out data
    5. Optuna hyperparameter optimization (if available)
    6. Final test evaluation
    7. Compare different hyperparameter configurations
    8. Manufacturing defect detection real-world demo
    """
    print("=" * 70)
    print("Simplified Residual Network - NumPy From-Scratch Implementation")
    print("=" * 70)

    IMG_SIZE = 16  # Image spatial dimension (small for CPU-based NumPy training)
    N_CLASSES = 5  # Number of distinct pattern classes

    # Step 1: Generate synthetic data with class-specific patterns.
    print("\n[1/8] Generating synthetic data...")
    X, y = generate_data(n_samples=1000, img_size=IMG_SIZE, n_classes=N_CLASSES)
    print(f"    Dataset: {X.shape}, Classes: {np.unique(y)}")

    # Step 2: Split data into train/validation/test sets (70/15/15).
    print("\n[2/8] Splitting data...")
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 3: Train the ResNet from scratch using mini-batch SGD.
    print("\n[3/8] Training ResNet (this may take a while)...")
    model_dict = train(
        X_train, y_train,
        img_size=IMG_SIZE, n_classes=N_CLASSES,
        n_filters=8, learning_rate=0.01,
        n_epochs=8, batch_size=32,
    )

    # Step 4: Validate on held-out validation set.
    print("\n[4/8] Validation...")
    val_metrics = validate(model_dict, X_val, y_val)

    # Step 5: Optuna hyperparameter optimization (if available).
    print("\n[5/8] Optuna hyperparameter search...")
    if OPTUNA_AVAILABLE:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, N_CLASSES, IMG_SIZE),
            n_trials=6,
        )
        print(f"    Best accuracy: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        # Retrain with best hyperparameters found by Optuna.
        bp = study.best_params
        model_dict = train(
            X_train, y_train,
            img_size=IMG_SIZE, n_classes=N_CLASSES,
            n_filters=bp["n_filters"], learning_rate=bp["learning_rate"],
            n_epochs=8, batch_size=bp["batch_size"],
            init_scale=bp["init_scale"],
        )
    else:
        print("    Optuna not installed, skipping.")

    # Step 6: Final test evaluation.
    print("\n[6/8] Test evaluation...")
    test_metrics = test(model_dict, X_test, y_test)

    # Step 7: Compare different hyperparameter configurations.
    print("\n[7/8] Running parameter comparison...")
    compare_parameter_sets(n_samples=600, img_size=IMG_SIZE)

    # Step 8: Manufacturing defect detection real-world demo.
    print("\n[8/8] Running manufacturing defect detection demo...")
    real_world_demo()

    # Final summary.
    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
