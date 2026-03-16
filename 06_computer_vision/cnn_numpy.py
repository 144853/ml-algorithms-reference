"""
Convolutional Neural Network (CNN) - NumPy From-Scratch Implementation
======================================================================

Architecture:
    This implementation builds a complete CNN from scratch using only NumPy,
    including forward and backward passes through all layers.

    Network Architecture:
    Input (1, 16, 16) -> Conv2D(1->4, 3x3) -> ReLU -> MaxPool(2x2)
                      -> Conv2D(4->8, 3x3) -> ReLU -> MaxPool(2x2)
                      -> Flatten -> FC(128) -> ReLU -> FC(n_classes) -> Softmax

    Layer Dimensions (for 16x16 input):
    - Input:    (1, 16, 16)
    - Conv1:    (4, 14, 14)  [3x3 kernel, no padding]
    - Pool1:    (4, 7, 7)    [2x2 max pooling]
    - Conv2:    (8, 5, 5)    [3x3 kernel, no padding]
    - Pool2:    (8, 2, 2)    [2x2 max pooling]
    - Flatten:  (32,)
    - FC1:      (128,)
    - FC2:      (n_classes,)

Theory & Mathematics:
    Convolution (Cross-Correlation):
        For input X of shape (C_in, H, W) and kernel K of shape (C_out, C_in, kH, kW):
        Y[c_out, i, j] = sum_{c_in} sum_{m} sum_{n} X[c_in, i+m, j+n] * K[c_out, c_in, m, n] + b[c_out]

        Output dimensions: H_out = H - kH + 1, W_out = W - kW + 1 (no padding, stride=1)

    Backpropagation through Convolution:
        dL/dK[c_out, c_in, m, n] = sum_{i,j} dL/dY[c_out, i, j] * X[c_in, i+m, j+n]
        dL/dX[c_in, i, j] = sum_{c_out} sum_{m,n} dL/dY[c_out, i-m, j-n] * K[c_out, c_in, m, n]
        (which is a full convolution of dL/dY with the 180-degree rotated kernel)

    Max Pooling:
        Y[c, i, j] = max(X[c, i*s:i*s+p, j*s:j*s+p])
        Gradient: Only flows back to the position of the maximum value.

    ReLU:
        f(x) = max(0, x)
        Gradient: 1 if x > 0, else 0

    Softmax + Cross-Entropy Loss:
        softmax(z_i) = exp(z_i) / sum_j(exp(z_j))
        L = -sum_i(y_i * log(p_i))  (where y is one-hot, p is softmax output)
        Combined gradient: dL/dz = p - y (elegant simplification)

Business Use Cases:
    - Understanding CNN internals for educational purposes
    - Debugging and validating deep learning frameworks
    - Deploying lightweight models in resource-constrained environments
    - Custom hardware implementations where frameworks are unavailable
    - Research prototyping of novel layer types

Advantages:
    - Complete transparency: every computation is visible
    - No framework dependencies beyond NumPy
    - Educational value: understand backprop through conv layers
    - Customizable: easy to experiment with novel architectures
    - Small memory footprint (no framework overhead)

Disadvantages:
    - Very slow compared to GPU-accelerated frameworks
    - No automatic differentiation
    - Limited to small models and datasets
    - No GPU support, no CUDA acceleration
    - Manual gradient computation is error-prone

Key Hyperparameters:
    - learning_rate: Step size for gradient descent (typical: 0.001-0.01)
    - n_epochs: Number of training passes over the dataset
    - batch_size: Mini-batch size for stochastic gradient descent
    - weight_init_scale: Standard deviation for weight initialization
    - n_filters_conv1, n_filters_conv2: Number of convolutional filters per layer
    - fc_hidden_size: Number of neurons in the hidden fully connected layer

References:
    - LeCun, Y. et al. (1998). "Gradient-based learning applied to document
      recognition." Proceedings of the IEEE, 86(11), 2278-2324.
    - Goodfellow, I. et al. (2016). "Deep Learning." MIT Press, Chapter 9.
"""

# numpy provides all the numerical operations we need: matrix multiplication, convolution,
# activation functions, and gradient computation -- it is the ONLY dependency for this
# from-scratch CNN implementation, proving that deep learning is just linear algebra + calculus
import numpy as np

# warnings module suppresses convergence and deprecation warnings that would clutter
# the training output during hyperparameter search experiments
import warnings

# Type hints make function signatures self-documenting and enable static analysis tools
# like mypy to catch shape mismatches and type errors before runtime
from typing import Dict, Tuple, Any, List, Optional

# Optuna is a Bayesian hyperparameter optimization framework -- it uses Tree-structured
# Parzen Estimators (TPE) to intelligently explore hyperparameter space, focusing on
# promising regions rather than exhaustive grid search
try:
    import optuna  # Optional dependency for automated hyperparameter tuning

    OPTUNA_AVAILABLE = True  # Flag checked before using Optuna features
except ImportError:
    OPTUNA_AVAILABLE = False  # Gracefully skip Optuna if not installed

# Ray Tune enables distributed hyperparameter tuning across multiple CPUs/machines --
# useful when each trial is expensive (like training a CNN) and you want parallelism
try:
    import ray  # Distributed computing framework
    from ray import tune  # Ray's hyperparameter tuning module

    RAY_AVAILABLE = True  # Flag checked before using Ray features
except ImportError:
    RAY_AVAILABLE = False  # Gracefully skip Ray if not installed

# Suppress all warnings to keep training output clean -- in production, you'd use
# targeted warning filters, but for educational demos this prevents clutter
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 2000,  # Total dataset size -- 2000 is enough for small CNN to learn patterns
    img_size: int = 16,  # 16x16 pixels keeps computation manageable for pure NumPy CNN
    n_classes: int = 5,  # 5 classes provides enough variety without excessive training time
    random_state: int = 42,  # Fixed seed ensures reproducible experiments across runs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic grayscale image data with class-specific patterns.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    img_size : int
        Image height and width.
    n_classes : int
        Number of classes (up to 10).
    random_state : int
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 1, img_size, img_size)
        Images in NCHW format, values in [0, 1].
    y : np.ndarray of shape (n_samples,)
        Integer labels.
    """
    # Create a seeded random generator for reproducibility -- RandomState is preferred
    # over np.random.seed() because it creates an isolated generator that won't interfere
    # with other code's random state
    rng = np.random.RandomState(random_state)

    # Equal samples per class ensures balanced dataset, preventing classifier bias
    # toward majority classes during training
    samples_per_class = n_samples // n_classes

    # Lists for accumulating images/labels -- appending to lists is O(1) amortized,
    # much faster than incrementally growing numpy arrays with np.concatenate
    X_list, y_list = [], []

    # Generate class-specific visual patterns -- each class has a distinct spatial structure
    # that the CNN's convolutional filters will learn to detect
    for cls in range(n_classes):
        for _ in range(samples_per_class):
            # Initialize with low-amplitude uniform noise as background texture --
            # float32 saves memory (4 bytes vs 8 for float64) while being sufficient precision
            img = rng.rand(img_size, img_size).astype(np.float32) * 0.1

            if cls == 0:  # Horizontal stripes -- CNN should learn horizontal edge detectors
                for r in range(0, img_size, 3):  # Stripes every 3 pixels
                    img[r, :] += 0.7  # Bright horizontal line spanning full width
            elif cls == 1:  # Vertical stripes -- CNN should learn vertical edge detectors
                for c in range(0, img_size, 3):  # Stripes every 3 pixels
                    img[:, c] += 0.7  # Bright vertical line spanning full height
            elif cls == 2:  # Center blob -- tests radially symmetric feature detection
                cx, cy = img_size // 2, img_size // 2  # Center of the image
                # ogrid creates open meshgrids for memory-efficient distance computation
                yy, xx = np.ogrid[:img_size, :img_size]
                # Euclidean distance from center for each pixel position
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                # Gaussian blob: exp(-d^2/2*sigma^2) with sigma^2=4 creates smooth circular pattern
                img += np.exp(-dist**2 / 8.0).astype(np.float32) * 0.8
            elif cls == 3:  # Diagonal pattern -- tests 45-degree edge detection
                for i in range(img_size):  # Draw diagonal line pixel by pixel
                    for d in range(-1, 2):  # 3-pixel thickness for visibility
                        j = i + d  # Column offset creates line thickness
                        if 0 <= j < img_size:  # Boundary check prevents index errors
                            img[i, j] += 0.7  # Bright pixel along the diagonal
            elif cls == 4:  # Checkerboard -- tests detection of regular 2D spatial frequency
                block = max(2, img_size // 4)  # Block size; max(2, ...) prevents degenerate case
                for r in range(0, img_size, block):
                    for c in range(0, img_size, block):
                        # XOR pattern: alternate bright/dark blocks
                        if ((r // block) + (c // block)) % 2 == 0:
                            img[r : r + block, c : c + block] += 0.6
            elif cls == 5:  # Top-left corner -- tests spatially localized features
                q = img_size // 3  # Corner region covers 1/9 of image area
                img[:q, :q] += 0.8  # Bright patch in top-left corner only
            elif cls == 6:  # Bottom-right corner -- asymmetric to class 5
                q = img_size // 3
                img[-q:, -q:] += 0.8  # Bright patch in bottom-right corner only
            elif cls == 7:  # Cross pattern -- combines horizontal and vertical edges
                mid = img_size // 2  # Center of the cross
                img[mid - 1 : mid + 1, :] += 0.7  # Horizontal arm
                img[:, mid - 1 : mid + 1] += 0.7  # Vertical arm (overlap gets brighter)
            elif cls == 8:  # Border pattern -- tests edge/boundary detection
                img[0:2, :] += 0.7  # Top edge
                img[-2:, :] += 0.7  # Bottom edge
                img[:, 0:2] += 0.7  # Left edge
                img[:, -2:] += 0.7  # Right edge
            elif cls == 9:  # Gradient pattern -- tests response to smooth intensity transitions
                # Create horizontal gradient from 0 to 1
                grad = np.linspace(0, 1, img_size).reshape(1, -1).astype(np.float32)
                img += grad * 0.7  # Broadcasting adds gradient to every row

            # Add Gaussian noise to simulate sensor noise -- std=0.05 provides subtle
            # variation without overwhelming the pattern, forcing the CNN to learn
            # robust features rather than memorizing exact pixel values
            img += rng.randn(img_size, img_size).astype(np.float32) * 0.05

            # Clip to valid [0, 1] range -- additions may exceed 1.0 and noise may go negative
            img = np.clip(img, 0, 1)

            X_list.append(img)
            y_list.append(cls)

    # Reshape to NCHW format: (N, Channels=1, Height, Width) -- this is the standard
    # format for convolutional layers where channel dimension comes before spatial dims,
    # matching PyTorch convention (unlike TensorFlow's NHWC format)
    X = np.array(X_list, dtype=np.float32).reshape(-1, 1, img_size, img_size)

    # Integer labels for cross-entropy loss computation
    y = np.array(y_list, dtype=np.int64)

    # Random shuffle breaks the sequential class ordering -- important because training
    # on ordered data can cause the model to forget early classes (catastrophic forgetting)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Layer Implementations
# ---------------------------------------------------------------------------


class Conv2D:
    """
    2D Convolution layer (technically cross-correlation) with full backpropagation.

    Cross-correlation differs from true convolution in that the kernel is NOT flipped
    before sliding. In deep learning, this distinction doesn't matter because the
    learned kernel weights absorb any flipping during training. We use cross-correlation
    because it's simpler and matches what all major frameworks actually implement.

    Forward: Y[n, co, i, j] = sum_{ci, m, n} X[n, ci, i+m, j+n] * W[co, ci, m, n] + b[co]
    Backward (dW): dW[co, ci, m, n] = sum_{n, i, j} dout[n, co, i, j] * X[n, ci, i+m, j+n]
    Backward (dX): dX[n, ci, i, j] = sum_{co, m, n} dout[n, co, i-m, j-n] * W[co, ci, m, n]
    """

    def __init__(
        self,
        in_channels: int,  # Number of input feature maps (1 for grayscale, 3 for RGB)
        out_channels: int,  # Number of convolutional filters to learn (= output feature maps)
        kernel_size: int,  # Spatial size of each filter (3 means 3x3, the most common choice)
        init_scale: float = 0.1,  # Standard deviation for random weight initialization
    ):
        # Store layer configuration for use in forward/backward passes
        self.in_channels = in_channels  # Input depth dimension
        self.out_channels = out_channels  # Output depth dimension (number of learned filters)
        self.kernel_size = kernel_size  # Spatial extent of each filter

        # Initialize weights with small random values drawn from a normal distribution.
        # init_scale controls the standard deviation -- too large causes exploding gradients,
        # too small causes vanishing gradients. The value 0.1 is a reasonable starting point
        # for small networks, though He initialization (sqrt(2/fan_in)) is theoretically optimal.
        # Shape: (out_channels, in_channels, kernel_h, kernel_w)
        # Each of the out_channels filters has a 3D volume matching the input channel depth
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * init_scale

        # Bias: one scalar per output channel, initialized to zero.
        # Zero initialization is safe for biases because the random weights already break symmetry.
        # Each output feature map gets its own bias that shifts all activations uniformly.
        self.b = np.zeros(out_channels, dtype=np.float32)

        # Cache stores the input tensor during forward pass for use in backward pass.
        # This is necessary because the gradient computation needs the original input values.
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute cross-correlation between input and learned filters.

        The operation slides each filter across the input spatially, computing a dot product
        at each position. This produces one output feature map per filter, where each value
        represents how strongly that local patch matches the filter's pattern.

        Parameters
        ----------
        X : np.ndarray of shape (N, C_in, H, W)
            Batch of input feature maps.

        Returns
        -------
        out : np.ndarray of shape (N, C_out, H_out, W_out)
            Output feature maps after convolution.
        """
        # Unpack input dimensions for computing output size and loop bounds
        N, C, H, W = X.shape  # N=batch size, C=channels, H=height, W=width

        # Kernel spatial dimensions (both equal to self.kernel_size for square kernels)
        kH, kW = self.kernel_size, self.kernel_size

        # Output spatial dimensions: no padding, stride=1 means output shrinks by (kernel-1)
        # This formula comes from: out_dim = (in_dim - kernel_dim) / stride + 1
        H_out = H - kH + 1  # Output height after valid convolution
        W_out = W - kW + 1  # Output width after valid convolution

        # Pre-allocate output tensor with zeros -- will be filled by the convolution loops
        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)

        # Four nested loops implement the naive convolution algorithm.
        # This is O(N * C_out * H_out * W_out * C_in * kH * kW), which is very slow
        # compared to im2col + GEMM or FFT-based convolution used by frameworks like PyTorch.
        # However, it makes the operation completely transparent for educational purposes.
        for n in range(N):  # Loop over each image in the batch
            for co in range(self.out_channels):  # Loop over each output filter
                for i in range(H_out):  # Loop over each output row position
                    for j in range(W_out):  # Loop over each output column position
                        # Extract the input patch that this output position corresponds to.
                        # The patch has shape (C_in, kH, kW) -- it spans all input channels
                        # and covers a kH x kW spatial region starting at position (i, j)
                        patch = X[n, :, i : i + kH, j : j + kW]

                        # Compute the dot product between the patch and this filter's weights,
                        # then add the bias. np.sum(patch * self.W[co]) performs element-wise
                        # multiplication and summation across all three dimensions (channels, h, w),
                        # which is equivalent to a flattened dot product
                        out[n, co, i, j] = np.sum(patch * self.W[co]) + self.b[co]

        # Cache the input for backward pass -- we need X to compute weight gradients
        self.cache = X
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """
        Backward pass through convolution: compute gradients and update weights.

        The gradient computation involves three parts:
        1. dW: gradient of loss w.r.t. weights -- used to update filters
        2. db: gradient of loss w.r.t. biases -- used to update biases
        3. dX: gradient of loss w.r.t. input -- passed to the previous layer

        Parameters
        ----------
        d_out : np.ndarray of shape (N, C_out, H_out, W_out)
            Gradient of loss with respect to this layer's output.
        lr : float
            Learning rate for weight updates.

        Returns
        -------
        dX : np.ndarray of shape (N, C_in, H, W)
            Gradient of loss with respect to this layer's input.
        """
        # Retrieve cached input from forward pass
        X = self.cache
        N, C, H, W = X.shape
        kH, kW = self.kernel_size, self.kernel_size
        _, _, H_out, W_out = d_out.shape

        # Initialize gradient accumulators with zeros
        dW = np.zeros_like(self.W)  # Will accumulate weight gradients over the batch
        db = np.zeros_like(self.b)  # Will accumulate bias gradients over the batch
        dX = np.zeros_like(X)  # Will accumulate input gradients for backprop to previous layer

        # Compute gradients using the same loop structure as forward pass.
        # This mirrors the forward computation because the chain rule links each
        # output position back to the same input patch and filter weights.
        for n in range(N):  # Loop over each image in the batch
            for co in range(self.out_channels):  # Loop over each output filter
                # Bias gradient: sum of all spatial gradients for this filter.
                # Since bias is added uniformly to all spatial positions,
                # its gradient is the sum of upstream gradients across all positions.
                db[co] += np.sum(d_out[n, co])

                for i in range(H_out):  # Loop over each output spatial position
                    for j in range(W_out):
                        # Weight gradient: upstream gradient * input patch.
                        # dL/dW = dL/dY * X -- the gradient of loss w.r.t. each weight element
                        # is the product of the upstream gradient and the input that was multiplied
                        # by that weight during the forward pass
                        dW[co] += d_out[n, co, i, j] * X[n, :, i : i + kH, j : j + kW]

                        # Input gradient: upstream gradient * corresponding filter weights.
                        # dL/dX = dL/dY * W -- each input pixel contributed to multiple outputs,
                        # so we accumulate the gradient from all output positions that used this pixel.
                        # This is mathematically equivalent to a "full" convolution of d_out with
                        # the 180-degree rotated kernel (flipped along both spatial axes)
                        dX[n, :, i : i + kH, j : j + kW] += d_out[n, co, i, j] * self.W[co]

        # Update weights using vanilla SGD (stochastic gradient descent).
        # We divide by N to average the gradients over the batch -- this makes the
        # learning rate independent of batch size, so the same lr works for different batch sizes.
        self.W -= lr * dW / N  # Update convolutional filter weights
        self.b -= lr * db / N  # Update bias terms
        return dX  # Return input gradients for the previous layer's backward pass


class MaxPool2D:
    """
    2x2 Max Pooling layer with stride 2.

    Max pooling reduces spatial dimensions by selecting the maximum value within
    each non-overlapping window. This achieves:
    1. Translation invariance: small shifts in input don't change the max value
    2. Dimensionality reduction: halves both spatial dimensions (4x reduction in area)
    3. Feature selection: keeps only the strongest activation in each region

    During backpropagation, the gradient flows only to the position that held the
    maximum value (the "winner"), while all other positions receive zero gradient.
    This is known as the "max routing" or "switch" mechanism.
    """

    def __init__(self, pool_size: int = 2):
        # pool_size determines the window size and stride (both equal for standard max pooling)
        # A pool_size of 2 means 2x2 windows with stride 2, reducing each spatial dimension by half
        self.pool_size = pool_size

        # Cache stores the input shape and the mask of maximum positions for backward pass
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: select the maximum value in each pooling window.

        For each non-overlapping pool_size x pool_size window in each feature map,
        output the maximum value. This creates a downsampled feature map that
        retains the strongest activations.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
            Input feature maps.

        Returns
        -------
        out : np.ndarray of shape (N, C, H//pool, W//pool)
            Max-pooled feature maps.
        """
        N, C, H, W = X.shape  # Unpack input dimensions
        p = self.pool_size  # Pool window size (typically 2)

        # Calculate output spatial dimensions (integer division halves each dimension)
        H_out = H // p
        W_out = W // p

        # Pre-allocate output and mask tensors
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        # Mask records which positions held the maximum -- needed for backward pass
        # to route gradients only to the "winning" positions
        mask = np.zeros_like(X)

        # Iterate over every pooling window in every sample and channel
        for n in range(N):  # Batch dimension
            for c in range(C):  # Channel dimension (each channel pooled independently)
                for i in range(H_out):  # Output row
                    for j in range(W_out):  # Output column
                        # Extract the pool_size x pool_size window from the input
                        patch = X[n, c, i * p : (i + 1) * p, j * p : (j + 1) * p]

                        # Find the maximum value in this window
                        max_val = np.max(patch)
                        out[n, c, i, j] = max_val

                        # Create a binary mask marking the position(s) of the maximum.
                        # If multiple positions share the same max value, all are marked.
                        # This mask is used during backpropagation to route gradients
                        # only to the position(s) that produced the maximum value.
                        local_mask = patch == max_val
                        mask[n, c, i * p : (i + 1) * p, j * p : (j + 1) * p] = local_mask

        # Cache input shape and mask for backward pass
        self.cache = (X.shape, mask)
        return out

    def backward(self, d_out: np.ndarray, lr: float = 0.0) -> np.ndarray:
        """
        Backward pass: route upstream gradients to the positions that held the maximum.

        The gradient of max(x1, x2, ...) with respect to the max element is 1,
        and 0 for all other elements. This means only the "winning" position in each
        pool window receives the upstream gradient; all others get zero.

        Parameters
        ----------
        d_out : np.ndarray of shape (N, C, H_out, W_out)
            Gradient from the upstream layer.
        lr : float
            Unused -- max pooling has no learnable parameters.

        Returns
        -------
        dX : np.ndarray of shape (N, C, H, W)
            Gradient with respect to the input.
        """
        # Retrieve cached information from forward pass
        orig_shape, mask = self.cache
        N, C, H_out, W_out = d_out.shape
        p = self.pool_size

        # Initialize input gradient with zeros
        dX = np.zeros(orig_shape, dtype=np.float32)

        # Distribute upstream gradients to max positions using the cached mask
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # Multiply the upstream gradient by the mask to route it only
                        # to the position(s) that produced the maximum during forward pass.
                        # Non-max positions have mask=0, so they receive zero gradient.
                        dX[n, c, i * p : (i + 1) * p, j * p : (j + 1) * p] += (
                            d_out[n, c, i, j] * mask[n, c, i * p : (i + 1) * p, j * p : (j + 1) * p]
                        )
        return dX


class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function: f(x) = max(0, x).

    ReLU is the most widely used activation function in deep learning because:
    1. It's computationally efficient (just a threshold comparison)
    2. It doesn't saturate for positive values (unlike sigmoid/tanh), preventing
       vanishing gradients for positive activations
    3. It produces sparse activations (negative inputs become zero), which acts
       as a form of regularization

    The gradient is simply 1 for positive inputs and 0 for negative inputs.
    The non-differentiability at x=0 is handled by convention (usually set to 0).
    """

    def __init__(self):
        # Cache stores the input for computing gradients in backward pass
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: zero out all negative values, pass positive values through unchanged."""
        self.cache = X  # Save input for backward pass gradient computation
        # np.maximum(0, X) applies element-wise max, zeroing negative values
        return np.maximum(0, X)

    def backward(self, d_out: np.ndarray, lr: float = 0.0) -> np.ndarray:
        """
        Backward pass: pass gradient through where input was positive, zero elsewhere.
        The gradient of ReLU is a binary mask: 1 where x > 0, 0 where x <= 0.
        """
        X = self.cache  # Retrieve cached input
        # (X > 0) creates a boolean mask, cast to float32 for multiplication.
        # Multiplying upstream gradient by this mask kills gradients for neurons
        # that had negative input (were "off" during forward pass)
        return d_out * (X > 0).astype(np.float32)


class Flatten:
    """
    Flatten layer: reshapes multi-dimensional feature maps into a 1D vector.

    This layer bridges the gap between convolutional layers (which produce 3D outputs:
    channels x height x width) and fully connected layers (which expect 1D input vectors).
    The flattening preserves the batch dimension and collapses all spatial and channel
    dimensions into a single feature vector dimension.

    For input of shape (N, C, H, W), output is (N, C*H*W).
    """

    def __init__(self):
        # Cache stores the original shape for reshaping back during backward pass
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: flatten all dimensions except batch into a single vector."""
        self.cache = X.shape  # Save original shape (N, C, H, W) for backward
        # reshape(N, -1) keeps batch dimension and collapses everything else
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out: np.ndarray, lr: float = 0.0) -> np.ndarray:
        """Backward pass: reshape gradient back to the original multi-dimensional shape."""
        # Simply reshape the flat gradient back to (N, C, H, W) for the conv layer upstream
        return d_out.reshape(self.cache)


class FullyConnected:
    """
    Fully Connected (Dense) layer: Y = X @ W + b.

    Each neuron in this layer is connected to every element of the input vector.
    This layer performs a linear transformation followed by an optional activation.

    For input X of shape (N, D_in) and weights W of shape (D_in, D_out):
    - Forward: Y = X @ W + b, output shape (N, D_out)
    - Backward (dW): dW = X^T @ d_out / N
    - Backward (db): db = mean(d_out, axis=0)
    - Backward (dX): dX = d_out @ W^T

    The /N in gradient computation averages over the batch, making the effective
    learning rate independent of batch size.
    """

    def __init__(
        self,
        in_features: int,  # Input dimension (number of features from previous layer)
        out_features: int,  # Output dimension (number of neurons in this layer)
        init_scale: float = 0.1,  # Standard deviation for random weight initialization
    ):
        # Initialize weights with small random values -- shape (in_features, out_features)
        # so that X @ W produces the correct output dimension.
        # Small values prevent saturation of activation functions in the first forward pass.
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * init_scale

        # Bias initialized to zero -- safe because random weights already break symmetry
        # between neurons. Shape (out_features,) broadcasts across the batch dimension.
        self.b = np.zeros(out_features, dtype=np.float32)

        # Cache stores the input for computing weight gradients during backward pass
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: linear transformation Y = X @ W + b.
        X has shape (N, D_in), output has shape (N, D_out).
        """
        self.cache = X  # Save input for backward gradient computation
        # Matrix multiplication X @ W: each row of X is transformed by W.
        # Adding self.b broadcasts the bias across all N samples in the batch.
        return X @ self.W + self.b

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """
        Backward pass: compute gradients and update parameters.

        Uses the chain rule:
        - dX = d_out @ W^T (gradient flows backward through the linear transformation)
        - dW = X^T @ d_out (input transpose times upstream gradient gives weight gradient)
        - db = mean(d_out) (bias gradient is the average upstream gradient)
        """
        X = self.cache  # Retrieve cached input
        N = X.shape[0]  # Batch size for averaging gradients

        # Input gradient: propagate upstream gradient backward through the weight matrix.
        # d_out has shape (N, D_out), W^T has shape (D_out, D_in), so dX has shape (N, D_in)
        dX = d_out @ self.W.T

        # Weight gradient: outer product of input and upstream gradient, averaged over batch.
        # X^T has shape (D_in, N), d_out has shape (N, D_out), so dW has shape (D_in, D_out)
        dW = X.T @ d_out / N

        # Bias gradient: average upstream gradient across all samples in the batch
        db = np.mean(d_out, axis=0)

        # Update weights and biases using vanilla SGD (gradient descent step)
        self.W -= lr * dW
        self.b -= lr * db
        return dX  # Return input gradient for the previous layer


class SoftmaxCrossEntropy:
    """
    Combined Softmax activation and Cross-Entropy loss function.

    We combine these two operations because their combined gradient has an
    elegant closed-form solution: dL/dz = softmax(z) - y_onehot = p - y.

    This is both numerically stable and computationally efficient compared to
    computing softmax gradient and cross-entropy gradient separately.

    Softmax: p_i = exp(z_i) / sum_j(exp(z_j))  -- converts logits to probabilities
    Cross-Entropy: L = -sum_i(y_i * log(p_i))    -- measures divergence from true distribution

    For classification with one-hot labels, this simplifies to: L = -log(p_correct_class)
    """

    def __init__(self):
        # Cache stores probabilities and labels for backward pass
        self.cache = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute softmax probabilities and cross-entropy loss.

        Parameters
        ----------
        logits : np.ndarray of shape (N, C)
            Raw class scores (unnormalized log-probabilities).
        y : np.ndarray of shape (N,)
            Integer class labels (not one-hot encoded).

        Returns
        -------
        probs : np.ndarray of shape (N, C)
            Softmax probabilities (each row sums to 1).
        loss : float
            Average cross-entropy loss across the batch.
        """
        # Numerical stability trick: subtract the maximum logit from each sample's scores.
        # This prevents overflow in exp() without changing the softmax result because
        # exp(z_i - max) / sum(exp(z_j - max)) = exp(z_i) / sum(exp(z_j))
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        # Compute exponentials of the shifted logits
        exp_scores = np.exp(shifted)

        # Normalize to get probabilities: each row sums to 1
        # This is the softmax function: p_i = exp(z_i) / sum_j(exp(z_j))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        N = logits.shape[0]  # Batch size

        # Cross-entropy loss: -log(probability of correct class), averaged over batch.
        # np.arange(N) indexes each sample, y indexes the correct class for each sample.
        # The 1e-12 epsilon prevents log(0) which would give -infinity.
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

        # Cache for backward pass
        self.cache = (probs, y)
        return probs, loss

    def backward(self) -> np.ndarray:
        """
        Gradient of combined softmax + cross-entropy: dL/dz = p - y_onehot.

        This elegant result comes from the calculus of softmax + cross-entropy:
        The Jacobian of softmax combined with the gradient of cross-entropy simplifies to
        just subtracting 1 from the probability of the correct class.

        Returns
        -------
        d_logits : np.ndarray of shape (N, C)
            Gradient of loss with respect to the input logits.
        """
        probs, y = self.cache  # Retrieve cached probabilities and labels
        N = probs.shape[0]  # Batch size

        # Start with softmax probabilities as the gradient
        d_logits = probs.copy()

        # Subtract 1 from the correct class probability: this is the p - y_onehot formula.
        # For the correct class c: gradient = p_c - 1 (push probability toward 1)
        # For wrong classes i != c: gradient = p_i - 0 = p_i (push probability toward 0)
        d_logits[np.arange(N), y] -= 1.0

        # Average over batch to match the loss computation (which uses mean, not sum)
        return d_logits / N


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------


class CNN:
    """
    Simple CNN assembled from the layer primitives above.

    Architecture: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> FC

    This design follows the classic CNN pattern established by LeNet-5:
    1. Alternating convolution and pooling layers extract increasingly abstract features
    2. Convolution layers detect local patterns (edges, textures, parts)
    3. Pooling layers provide translation invariance and reduce spatial dimensions
    4. Fully connected layers combine spatial features for final classification
    """

    def __init__(
        self,
        img_channels: int = 1,  # Input channels: 1 for grayscale, 3 for RGB
        img_size: int = 16,  # Input spatial size (assumed square)
        n_classes: int = 5,  # Number of output classes for classification
        n_filters1: int = 4,  # Number of filters in first conv layer (detects low-level features)
        n_filters2: int = 8,  # Number of filters in second conv layer (detects higher-level features)
        fc_hidden: int = 64,  # Hidden units in the fully connected layer
        init_scale: float = 0.1,  # Weight initialization scale (std of random normal)
    ):
        # Build the layer sequence -- order matters because each layer's output
        # becomes the next layer's input during forward pass
        self.layers = []

        # First convolutional block: extracts low-level features (edges, corners, gradients).
        # 3x3 kernels are the standard choice because they capture local spatial patterns
        # with minimum parameters while stacking multiple layers achieves larger receptive fields.
        self.layers.append(Conv2D(img_channels, n_filters1, 3, init_scale))
        # ReLU introduces non-linearity -- without it, stacking linear layers (conv + FC)
        # would collapse to a single linear transformation, unable to learn complex patterns
        self.layers.append(ReLU())
        # Max pooling halves spatial dimensions: provides translation invariance and reduces
        # computation for subsequent layers
        self.layers.append(MaxPool2D(2))

        # Second convolutional block: builds higher-level features by combining the edge
        # detectors from the first block into more complex patterns (textures, parts).
        # Doubling the filter count (4->8) is common practice because higher layers
        # need more filters to represent the growing variety of feature combinations.
        self.layers.append(Conv2D(n_filters1, n_filters2, 3, init_scale))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(2))

        # Flatten bridge: converts 3D feature maps to 1D vector for fully connected layers
        self.layers.append(Flatten())

        # Compute the flattened feature dimension by tracing the spatial size through
        # two conv(3x3, no padding) + pool(2x2, stride 2) operations:
        # After conv1 (3x3, valid): img_size -> img_size - 2
        s1 = img_size - 2
        # After pool1 (2x2, stride 2): (img_size - 2) -> (img_size - 2) // 2
        s2 = s1 // 2
        # After conv2 (3x3, valid): -> ((img_size - 2) // 2) - 2
        s3 = s2 - 2
        # After pool2 (2x2, stride 2): -> (((img_size - 2) // 2) - 2) // 2
        s4 = s3 // 2
        # Total flattened size = n_filters2 * spatial_h * spatial_w
        flat_size = n_filters2 * s4 * s4

        # Hidden fully connected layer: learns non-linear combinations of spatial features.
        # 64 hidden units is a reasonable default for small images -- larger images or
        # more complex tasks would benefit from more hidden units.
        self.layers.append(FullyConnected(flat_size, fc_hidden, init_scale))
        self.layers.append(ReLU())

        # Output layer: produces one logit per class, fed into softmax for probabilities
        self.layers.append(FullyConnected(fc_hidden, n_classes, init_scale))

        # Loss function: combined softmax + cross-entropy for numerical stability
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers sequentially.
        Each layer transforms its input and passes the result to the next layer.
        """
        out = X  # Start with the raw input image batch
        for layer in self.layers:
            out = layer.forward(out)  # Each layer transforms the data in sequence
        return out  # Returns the final logits (unnormalized class scores)

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute softmax probabilities and cross-entropy loss from raw logits."""
        return self.loss_fn.forward(logits, y)

    def backward(self, lr: float):
        """
        Backward pass through all layers in reverse order (backpropagation).

        The gradient flows from the loss function backward through every layer,
        with each layer: (1) computing input gradients to pass upstream, and
        (2) updating its own learnable parameters using the gradient.
        """
        # Start with the gradient from the loss function (dL/dz = p - y_onehot)
        d = self.loss_fn.backward()

        # Propagate gradient backward through layers in reverse order.
        # Each layer receives the gradient from the layer above and returns
        # the gradient to pass to the layer below.
        for layer in reversed(self.layers):
            d = layer.backward(d, lr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices by running forward pass and taking argmax."""
        logits = self.forward(X)  # Get raw class scores
        return np.argmax(logits, axis=1)  # Select class with highest score


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,  # Training images in NCHW format: (N, 1, H, W)
    y_train: np.ndarray,  # Training labels: (N,) with integer class indices
    img_channels: int = 1,  # Number of input channels (1 for grayscale)
    img_size: int = 16,  # Spatial dimension of input images (assumed square)
    n_classes: int = 5,  # Number of output classes
    n_filters1: int = 4,  # Filters in first conv layer (low-level feature detectors)
    n_filters2: int = 8,  # Filters in second conv layer (higher-level feature detectors)
    fc_hidden: int = 64,  # Hidden units in fully connected classifier head
    learning_rate: float = 0.01,  # SGD learning rate -- controls step size in weight space
    n_epochs: int = 10,  # Number of complete passes through the training data
    batch_size: int = 32,  # Mini-batch size for stochastic gradient descent
    init_scale: float = 0.1,  # Weight initialization standard deviation
    verbose: bool = True,  # Whether to print progress during training
) -> Dict[str, Any]:
    """
    Train the NumPy CNN using mini-batch SGD.

    Parameters
    ----------
    X_train : np.ndarray of shape (N, 1, H, W)
    y_train : np.ndarray of shape (N,)
    Various hyperparameters.

    Returns
    -------
    result : dict with 'model', 'train_losses', 'train_accs'.
    """
    # Construct the CNN model with specified architecture parameters
    model = CNN(
        img_channels=img_channels,
        img_size=img_size,
        n_classes=n_classes,
        n_filters1=n_filters1,
        n_filters2=n_filters2,
        fc_hidden=fc_hidden,
        init_scale=init_scale,
    )

    N = len(X_train)  # Total number of training samples
    train_losses, train_accs = [], []  # Track training metrics per epoch

    # Training loop: iterate over the dataset n_epochs times
    for epoch in range(n_epochs):
        # Shuffle training data each epoch -- this ensures each mini-batch is different
        # across epochs, providing better gradient estimates and preventing the model
        # from memorizing the order of samples
        perm = np.random.permutation(N)

        # Epoch-level metric accumulators
        epoch_loss = 0.0  # Accumulates loss across all batches
        epoch_correct = 0  # Counts correct predictions across all batches
        n_batches = 0  # Counts batches for averaging

        # Mini-batch training loop: process data in chunks of batch_size
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)  # Handle last batch which may be smaller
            idx = perm[start:end]  # Get shuffled indices for this batch
            X_batch = X_train[idx]  # Extract batch images
            y_batch = y_train[idx]  # Extract batch labels

            # Forward pass: compute predictions through all layers
            logits = model.forward(X_batch)

            # Compute loss and softmax probabilities
            probs, loss = model.compute_loss(logits, y_batch)

            # Backward pass: compute gradients and update all layer weights
            model.backward(learning_rate)

            # Accumulate epoch metrics
            epoch_loss += loss  # Accumulate batch loss
            preds = np.argmax(probs, axis=1)  # Get predicted class indices
            epoch_correct += np.sum(preds == y_batch)  # Count correct predictions
            n_batches += 1

        # Compute epoch-averaged metrics
        avg_loss = epoch_loss / n_batches  # Average loss per batch
        accuracy = epoch_correct / N  # Overall training accuracy for this epoch
        train_losses.append(avg_loss)
        train_accs.append(accuracy)

        # Print progress at regular intervals (every 20% of epochs)
        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    return {"model": model, "train_losses": train_losses, "train_accs": train_accs}


def validate(
    model_dict: Dict[str, Any],  # Output from train() containing the trained model
    X_val: np.ndarray,  # Validation images: (N, 1, H, W)
    y_val: np.ndarray,  # Validation labels: (N,)
) -> Dict[str, Any]:
    """
    Validate on held-out data.

    Parameters
    ----------
    model_dict : dict from train()
    X_val : np.ndarray of shape (N, 1, H, W)
    y_val : np.ndarray of shape (N,)

    Returns
    -------
    metrics : dict with 'accuracy', 'loss', 'per_class_accuracy'.
    """
    model = model_dict["model"]  # Extract the trained CNN model

    # Forward pass on validation data -- no gradient computation needed
    logits = model.forward(X_val)

    # Compute loss for monitoring (helps detect overfitting: training loss decreasing
    # while validation loss increasing indicates the model is memorizing training data)
    probs, loss = model.compute_loss(logits, y_val)

    # Get predicted class labels
    preds = np.argmax(probs, axis=1)

    # Overall accuracy: fraction of correctly classified samples
    accuracy = np.mean(preds == y_val)

    # Per-class accuracy reveals if certain classes are harder to classify
    # than others, which aggregate accuracy would mask
    classes = np.unique(y_val)
    per_class_acc = {}
    for c in classes:
        mask = y_val == c  # Boolean mask for samples of class c
        if np.sum(mask) > 0:  # Guard against empty classes
            per_class_acc[int(c)] = float(np.mean(preds[mask] == y_val[mask]))

    # Print detailed results
    print(f"Validation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    for c, acc in per_class_acc.items():
        print(f"  Class {c}: {acc:.4f}")

    return {"accuracy": accuracy, "loss": loss, "per_class_accuracy": per_class_acc}


def test(
    model_dict: Dict[str, Any],  # Output from train()
    X_test: np.ndarray,  # Test images -- used only ONCE for final unbiased evaluation
    y_test: np.ndarray,  # Test labels
) -> Dict[str, Any]:
    """
    Final test evaluation.

    Parameters
    ----------
    model_dict : dict from train()
    X_test, y_test : test data

    Returns
    -------
    metrics : dict
    """
    print("\n--- Test Set Evaluation ---")
    # Reuse validate() -- semantically different (test = final eval) but same computation
    return validate(model_dict, X_test, y_test)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: "optuna.Trial",  # Optuna trial object for suggesting hyperparameters
    X_train: np.ndarray,  # Training data
    y_train: np.ndarray,  # Training labels
    X_val: np.ndarray,  # Validation data for evaluating each configuration
    y_val: np.ndarray,  # Validation labels
    n_classes: int = 5,  # Number of classes (passed through to train())
    img_size: int = 16,  # Image size (passed through to train())
) -> float:
    """
    Optuna objective for CNN hyperparameter tuning.

    Searches over learning rate, init_scale, fc_hidden, batch_size, and filter counts.
    Returns validation accuracy to maximize.
    """
    # Suggest learning rate on log scale because its effect is multiplicative --
    # the difference between 0.001 and 0.01 is as significant as between 0.01 and 0.1
    lr = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)

    # Weight initialization scale affects gradient flow in early training
    init_scale = trial.suggest_float("init_scale", 0.01, 0.3)

    # Hidden layer size controls model capacity
    fc_hidden = trial.suggest_categorical("fc_hidden", [32, 64, 128])

    # Batch size affects gradient noise: smaller = more noise = more exploration
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Number of convolutional filters determines feature detection capacity
    n_filters1 = trial.suggest_categorical("n_filters1", [2, 4])
    n_filters2 = trial.suggest_categorical("n_filters2", [4, 8])

    # Train with suggested hyperparameters (fewer epochs for faster search)
    model_dict = train(
        X_train,
        y_train,
        n_classes=n_classes,
        img_size=img_size,
        n_filters1=n_filters1,
        n_filters2=n_filters2,
        fc_hidden=fc_hidden,
        learning_rate=lr,
        n_epochs=5,  # Reduced epochs for faster hyperparameter search
        batch_size=batch_size,
        init_scale=init_scale,
        verbose=False,  # Suppress per-epoch output during search
    )
    # Evaluate on validation set
    metrics = validate(model_dict, X_val, y_val)
    return metrics["accuracy"]  # Return accuracy as the optimization objective


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 5,
    img_size: int = 16,
    num_samples: int = 6,  # Number of random configurations to try
) -> Dict[str, Any]:
    """
    Ray Tune hyperparameter search for CNN.

    Returns best config found.
    """
    if not RAY_AVAILABLE:
        print("Ray not installed. Skipping.")
        return {}

    # Define the trainable function that Ray will call for each configuration
    def trainable(config):
        md = train(
            X_train, y_train,
            n_classes=n_classes, img_size=img_size,
            n_filters1=config["n_filters1"], n_filters2=config["n_filters2"],
            fc_hidden=config["fc_hidden"], learning_rate=config["learning_rate"],
            n_epochs=5, batch_size=config["batch_size"],
            init_scale=config["init_scale"], verbose=False,
        )
        metrics = validate(md, X_val, y_val)
        tune.report({"accuracy": metrics["accuracy"]})

    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-3, 0.1),
        "init_scale": tune.uniform(0.01, 0.3),
        "fc_hidden": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([16, 32]),
        "n_filters1": tune.choice([2, 4]),
        "n_filters2": tune.choice([4, 8]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, metric="accuracy", mode="max"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config: {best.config}")
    print(f"Best accuracy: {best.metrics['accuracy']:.4f}")
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------


def compare_parameter_sets(
    n_samples: int = 1000,  # Moderate dataset for meaningful comparison without excessive runtime
    img_size: int = 16,  # Standard image size matching default
) -> Dict[str, Any]:
    """
    Compare different CNN architecture and training configurations to demonstrate
    how key hyperparameters affect model performance.

    Comparisons include:
    1. Number of filters (num_filters): 2 vs 4 vs 8 in first conv layer
       - More filters = more feature detectors = higher capacity but more computation
    2. Filter counts in second layer: 4 vs 8 vs 16
       - Second layer combines first-layer features into higher-level patterns
    3. Fully connected hidden size: 32 vs 64 vs 128
       - Larger FC layer has more capacity for combining spatial features
    4. Learning rate: 0.001 vs 0.01 vs 0.05
       - Too low = slow convergence, too high = instability/divergence

    Parameters
    ----------
    n_samples : int
        Number of synthetic images for comparison.
    img_size : int
        Image spatial dimension.

    Returns
    -------
    results : dict
        Maps configuration names to validation accuracy.
    """
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: CNN Architecture Configurations")
    print("=" * 70)

    n_classes = 5  # Number of classes for the comparison experiments

    # Generate shared dataset for fair comparison
    X, y = generate_data(n_samples=n_samples, img_size=img_size, n_classes=n_classes, random_state=42)

    # Split data
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]

    # Define configurations to compare
    configs = {
        # --- First layer filter count comparison ---
        # Fewer filters (2) detect only the most prominent edges/patterns.
        # More filters (8) can detect a wider variety of low-level features.
        "filters1=2_filters2=8": {
            "n_filters1": 2, "n_filters2": 8, "fc_hidden": 64,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        "filters1=4_filters2=8": {
            "n_filters1": 4, "n_filters2": 8, "fc_hidden": 64,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        "filters1=8_filters2=8": {
            "n_filters1": 8, "n_filters2": 8, "fc_hidden": 64,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        # --- Second layer filter count comparison ---
        "filters1=4_filters2=4": {
            "n_filters1": 4, "n_filters2": 4, "fc_hidden": 64,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        "filters1=4_filters2=16": {
            "n_filters1": 4, "n_filters2": 16, "fc_hidden": 64,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        # --- FC hidden size comparison ---
        "fc_hidden=32": {
            "n_filters1": 4, "n_filters2": 8, "fc_hidden": 32,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        "fc_hidden=128": {
            "n_filters1": 4, "n_filters2": 8, "fc_hidden": 128,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        # --- Learning rate comparison ---
        "lr=0.001": {
            "n_filters1": 4, "n_filters2": 8, "fc_hidden": 64,
            "learning_rate": 0.001, "n_epochs": 8, "batch_size": 32,
        },
        "lr=0.01": {
            "n_filters1": 4, "n_filters2": 8, "fc_hidden": 64,
            "learning_rate": 0.01, "n_epochs": 8, "batch_size": 32,
        },
        "lr=0.05": {
            "n_filters1": 4, "n_filters2": 8, "fc_hidden": 64,
            "learning_rate": 0.05, "n_epochs": 8, "batch_size": 32,
        },
    }

    results = {}

    for name, params in configs.items():
        print(f"\n--- Configuration: {name} ---")
        model_dict = train(
            X_train, y_train,
            img_size=img_size, n_classes=n_classes,
            init_scale=0.1, verbose=False,
            **params,
        )
        metrics = validate(model_dict, X_val, y_val)
        results[name] = metrics["accuracy"]

    # Print summary
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Val Accuracy':>12}")
    print("-" * 50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<35} {acc:>12.4f}")
    print("-" * 50)

    print("\nKey Takeaways:")
    print("  - More filters increase capacity but add computation cost")
    print("  - Learning rate is the most critical hyperparameter (too high/low = poor results)")
    print("  - FC hidden size matters less than conv architecture for image classification")
    print("  - Second layer filter count affects high-level feature variety")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Medical X-Ray Classification
# ---------------------------------------------------------------------------


def real_world_demo() -> Dict[str, Any]:
    """
    Demonstrate the from-scratch CNN on a realistic medical imaging scenario:
    classifying chest X-ray images as Normal vs Pneumonia.

    This demonstrates that even a simple CNN built entirely in NumPy can learn
    to distinguish medical imaging patterns, though performance will be limited
    by the small model size and lack of GPU acceleration.

    Returns
    -------
    results : dict
        Dictionary containing model performance metrics.
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Medical X-Ray Classification (Normal vs Pneumonia)")
    print("=" * 70)

    # Configuration for the medical imaging demo
    rng = np.random.RandomState(42)  # Fixed seed for reproducible results
    n_samples = 600  # Smaller dataset due to slow NumPy training
    img_size = 16  # Small images to keep training time manageable
    n_classes = 2  # Binary classification: Normal vs Pneumonia
    class_names = ["Normal", "Pneumonia"]

    samples_per_class = n_samples // n_classes
    X_list, y_list = [], []

    # Generate synthetic chest X-ray patterns
    for cls_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Base X-ray appearance: mid-gray with subtle noise
            img = np.ones((img_size, img_size), dtype=np.float32) * 0.4
            img += rng.rand(img_size, img_size).astype(np.float32) * 0.08

            # Add rib-like horizontal patterns (present in both normal and pneumonia)
            for row in range(2, img_size - 2, 3):
                rib_intensity = rng.uniform(0.04, 0.08)
                img[row, 2:-2] += rib_intensity

            if cls_idx == 0:
                # NORMAL: clear lung fields with subtle vascular markings
                # Lung regions are slightly darker (air absorbs fewer X-rays)
                quarter = img_size // 4
                img[quarter:-quarter, quarter:-quarter] -= 0.06

                # Subtle radial vascular markings from center
                cx, cy = img_size // 2, img_size // 2
                for _ in range(4):
                    angle = rng.uniform(0, 2 * np.pi)
                    for t in range(3, 7):
                        vx = int(cx + t * np.cos(angle))
                        vy = int(cy + t * np.sin(angle))
                        if 0 <= vx < img_size and 0 <= vy < img_size:
                            img[vy, vx] += 0.02

            elif cls_idx == 1:
                # PNEUMONIA: localized bright opacities in lung fields
                n_opacities = rng.randint(1, 3)
                for _ in range(n_opacities):
                    ox = rng.randint(3, img_size - 3)
                    oy = rng.randint(3, img_size - 3)
                    radius = rng.randint(2, 5)
                    yy, xx = np.ogrid[:img_size, :img_size]
                    dist = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2)
                    opacity = np.exp(-dist**2 / (2 * radius**2)).astype(np.float32)
                    img += opacity * rng.uniform(0.12, 0.25)

            # Add sensor noise
            img += rng.randn(img_size, img_size).astype(np.float32) * 0.02
            img = np.clip(img, 0, 1)
            X_list.append(img)
            y_list.append(cls_idx)

    # Convert to NCHW format
    X = np.array(X_list, dtype=np.float32).reshape(-1, 1, img_size, img_size)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Generated {len(X)} synthetic X-ray images ({img_size}x{img_size})")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

    # Split data
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train CNN
    print("\n--- Training CNN for X-ray classification ---")
    model_dict = train(
        X_train, y_train,
        img_size=img_size, n_classes=n_classes,
        n_filters1=4, n_filters2=8, fc_hidden=32,
        learning_rate=0.01, n_epochs=15, batch_size=16,
    )

    # Evaluate
    print("\n--- Validation Results ---")
    val_metrics = validate(model_dict, X_val, y_val)

    print("\n--- Test Results ---")
    test_metrics = test(model_dict, X_test, y_test)

    # Clinical interpretation
    print("\n--- Clinical Interpretation ---")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    if 1 in test_metrics.get("per_class_accuracy", {}):
        pneumonia_acc = test_metrics["per_class_accuracy"][1]
        print(f"Pneumonia Detection Rate: {pneumonia_acc:.4f}")
        if pneumonia_acc >= 0.8:
            print("  -> Reasonable detection rate for a from-scratch NumPy CNN")
        else:
            print("  -> Limited by small model size; GPU-accelerated frameworks would do better")

    print("\nNote: This from-scratch CNN is for educational purposes.")
    print("Production medical AI would use PyTorch/TensorFlow with pretrained models.")

    return {
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "per_class_accuracy": test_metrics.get("per_class_accuracy", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the full NumPy CNN pipeline."""
    print("=" * 70)
    print("Convolutional Neural Network - NumPy From-Scratch Implementation")
    print("=" * 70)

    # Constants for the pipeline
    IMG_SIZE = 16  # Small images for manageable computation in pure NumPy
    N_CLASSES = 5  # Number of visual pattern classes to distinguish

    # ---- Step 1: Generate synthetic data ----
    print("\n[1/8] Generating synthetic data...")
    X, y = generate_data(n_samples=1500, img_size=IMG_SIZE, n_classes=N_CLASSES)
    print(f"    Dataset: {X.shape}, Classes: {np.unique(y)}")

    # ---- Step 2: Split data ----
    # Manual split (70/15/15) without sklearn's train_test_split to demonstrate
    # that the only dependency is NumPy
    print("\n[2/8] Splitting data...")
    n = len(X)
    n_train = int(0.7 * n)  # 70% for training
    n_val = int(0.15 * n)  # 15% for validation
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- Step 3: Train the CNN ----
    print("\n[3/8] Training CNN (this may take a moment)...")
    model_dict = train(
        X_train, y_train,
        img_size=IMG_SIZE, n_classes=N_CLASSES,
        n_filters1=4, n_filters2=8, fc_hidden=64,
        learning_rate=0.01, n_epochs=10, batch_size=32,
    )

    # ---- Step 4: Validate ----
    print("\n[4/8] Validation...")
    val_metrics = validate(model_dict, X_val, y_val)

    # ---- Step 5: Optuna hyperparameter search ----
    print("\n[5/8] Optuna hyperparameter search...")
    if OPTUNA_AVAILABLE:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, N_CLASSES, IMG_SIZE),
            n_trials=8,
        )
        print(f"    Best accuracy: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        bp = study.best_params
        print("\n    Retraining with best params...")
        model_dict = train(
            X_train, y_train,
            img_size=IMG_SIZE, n_classes=N_CLASSES,
            n_filters1=bp["n_filters1"], n_filters2=bp["n_filters2"],
            fc_hidden=bp["fc_hidden"], learning_rate=bp["learning_rate"],
            n_epochs=10, batch_size=bp["batch_size"], init_scale=bp["init_scale"],
        )
    else:
        print("    Optuna not installed, skipping.")

    # ---- Step 6: Test ----
    print("\n[6/8] Test evaluation...")
    test_metrics = test(model_dict, X_test, y_test)

    # ---- Step 7: Parameter comparison ----
    print("\n[7/8] Running parameter comparison...")
    compare_parameter_sets(n_samples=800, img_size=IMG_SIZE)

    # ---- Step 8: Real-world demo ----
    print("\n[8/8] Running real-world medical X-ray classification demo...")
    real_world_demo()

    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
