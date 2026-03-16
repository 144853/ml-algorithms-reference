"""
Simplified EfficientNet Concepts - NumPy From-Scratch Implementation
=====================================================================

Architecture:
    This implementation builds the key innovations of EfficientNet from
    scratch using only NumPy, including depthwise separable convolutions,
    squeeze-and-excitation blocks, and the MBConv (Mobile inverted Bottleneck
    Convolution) block structure. All forward and backward passes are
    implemented manually.

    Network Architecture:
    Input (1, 16, 16)
    -> Conv2D(1->8, 3x3, pad=1) -> BN -> Swish
    -> MBConv Block (8->8, expand_ratio=1, SE)
    -> AvgPool(2x2)
    -> MBConv Block (8->16, expand_ratio=2, SE)
    -> AvgPool(2x2)
    -> GlobalAvgPool -> FC(n_classes)

    MBConv Block Structure (Inverted Residual + SE):
    x --> Expand(1x1 conv, channels * expand_ratio) -> BN -> Swish
      --> Depthwise Conv(3x3) -> BN -> Swish
      --> Squeeze-Excitation
      --> Project(1x1 conv, out_channels) -> BN
      --> (+x if skip) --> output

Theory & Mathematics:
    Depthwise Separable Convolution:
        Standard conv: O(C_in * C_out * k^2 * H * W)
        Depthwise separable = Depthwise + Pointwise:
            Depthwise: O(C_in * k^2 * H * W)     [one filter per channel]
            Pointwise: O(C_in * C_out * H * W)    [1x1 conv to mix channels]
            Total:     O(C_in * (k^2 + C_out) * H * W)
        Speedup: k^2 * C_out / (k^2 + C_out) = 8-9x for 3x3, C_out=32

    Squeeze-and-Excitation (SE) Block:
        Purpose: Learn channel-wise attention (which channels are important).
        1. Squeeze:    z_c = GlobalAvgPool(X_c) = (1/HW) * sum_{h,w} X[c,h,w]
        2. Excitation: s = sigmoid(W_2 * ReLU(W_1 * z))
                       where W_1: R^C -> R^{C/r}, W_2: R^{C/r} -> R^C
        3. Scale:      Y_c = s_c * X_c  (element-wise channel scaling)
        Gradient:      dL/dX = s * dL/dY + X * dL/ds * ds/dz * dz/dX

    MBConv (Mobile inverted Bottleneck Convolution):
        Key insight: Invert the bottleneck.
        Traditional: Wide -> Narrow -> Wide (bottleneck in middle)
        Inverted:    Narrow -> Wide -> Narrow (expansion in middle)
        This works because the expanded representation captures richer
        features, while the narrow input/output keeps memory low.

    Swish Activation:
        f(x) = x * sigmoid(x) = x / (1 + exp(-x))
        Gradient: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
        Swish is smoother than ReLU and can produce negative values,
        which helps with gradient flow. EfficientNet uses SiLU (=Swish-1).

    Compound Scaling (conceptual):
        EfficientNet balances three dimensions simultaneously:
        - Depth (d): Number of layers, d = alpha^phi
        - Width (w): Number of channels, w = beta^phi
        - Resolution (r): Input image size, r = gamma^phi
        Constraint: alpha * beta^2 * gamma^2 ~= 2

Business Use Cases:
    - Understanding mobile-optimized neural network design
    - Educational: learn depthwise separable convolutions and SE blocks
    - Prototyping novel efficient architectures
    - Research on attention mechanisms in CNNs
    - Custom hardware implementations for edge devices

Advantages:
    - Complete transparency of all computations
    - Implements all key EfficientNet innovations from scratch
    - No framework dependencies
    - Educational: understand WHY these operations work
    - Demonstrates attention mechanisms in vision models

Disadvantages:
    - Very slow: nested loops instead of optimized BLAS operations
    - Limited to tiny models and datasets
    - No GPU acceleration
    - Complex backward pass through SE blocks
    - Not suitable for production use

Key Hyperparameters:
    - learning_rate: Step size (0.001 - 0.05)
    - expand_ratio: MBConv expansion ratio (1, 2, 4, 6)
    - se_ratio: SE reduction ratio (4, 8, 16)
    - n_epochs: Training iterations
    - batch_size: Mini-batch size
    - init_scale: Weight initialization scale

References:
    - Tan, M. and Le, Q.V. (2019). "EfficientNet: Rethinking Model Scaling
      for Convolutional Neural Networks." ICML. arXiv:1905.11946.
    - Sandler, M. et al. (2018). "MobileNetV2: Inverted Residuals and Linear
      Bottlenecks." CVPR. arXiv:1801.04381.
    - Hu, J. et al. (2018). "Squeeze-and-Excitation Networks." CVPR.
    - Ramachandran, P. et al. (2017). "Searching for Activation Functions."
      arXiv:1710.05941.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any, Optional

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    from ray import tune

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 1200,
    img_size: int = 16,
    n_classes: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic grayscale images with class-specific patterns."""
    rng = np.random.RandomState(random_state)
    spc = n_samples // n_classes
    X_list, y_list = [], []

    for cls in range(n_classes):
        for _ in range(spc):
            img = rng.rand(img_size, img_size).astype(np.float32) * 0.1

            if cls == 0:
                for r in range(0, img_size, 3):
                    img[r, :] += 0.7
            elif cls == 1:
                for c in range(0, img_size, 3):
                    img[:, c] += 0.7
            elif cls == 2:
                cx, cy = img_size // 2, img_size // 2
                yy, xx = np.ogrid[:img_size, :img_size]
                img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8.0).astype(np.float32) * 0.8
            elif cls == 3:
                for i in range(img_size):
                    img[i, i % img_size] += 0.7
            elif cls == 4:
                block = max(2, img_size // 4)
                for r in range(0, img_size, block):
                    for c in range(0, img_size, block):
                        if ((r // block) + (c // block)) % 2 == 0:
                            img[r : r + block, c : c + block] += 0.6

            img += rng.randn(img_size, img_size).astype(np.float32) * 0.05
            img = np.clip(img, 0, 1)
            X_list.append(img)
            y_list.append(cls)

    X = np.array(X_list, dtype=np.float32).reshape(-1, 1, img_size, img_size)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Layer Implementations
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class Swish:
    """Swish activation: f(x) = x * sigmoid(x)."""

    def __init__(self):
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        sig = _sigmoid(X)
        out = X * sig
        self.cache = (X, sig, out)
        return out

    def backward(self, d_out: np.ndarray, lr: float = 0.0) -> np.ndarray:
        X, sig, out = self.cache
        # f'(x) = f(x) + sigmoid(x) * (1 - f(x))
        grad = out + sig * (1.0 - out)
        return d_out * grad


class Conv2D:
    """2D Convolution with optional padding."""

    def __init__(self, in_ch: int, out_ch: int, ks: int, padding: int = 0, init_scale: float = 0.1):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ks = ks
        self.padding = padding
        self.W = np.random.randn(out_ch, in_ch, ks, ks).astype(np.float32) * init_scale
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.cache = None

    def _pad(self, X):
        if self.padding == 0:
            return X
        return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")

    def forward(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pad(X)
        N, C, H, W = Xp.shape
        Ho = H - self.ks + 1
        Wo = W - self.ks + 1
        out = np.zeros((N, self.out_ch, Ho, Wo), dtype=np.float32)
        for n in range(N):
            for co in range(self.out_ch):
                for i in range(Ho):
                    for j in range(Wo):
                        out[n, co, i, j] = np.sum(Xp[n, :, i : i + self.ks, j : j + self.ks] * self.W[co]) + self.b[co]
        self.cache = (X, Xp)
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        X, Xp = self.cache
        N, C, H, W = Xp.shape
        _, _, Ho, Wo = d_out.shape
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dXp = np.zeros_like(Xp)
        for n in range(N):
            for co in range(self.out_ch):
                db[co] += np.sum(d_out[n, co])
                for i in range(Ho):
                    for j in range(Wo):
                        dW[co] += d_out[n, co, i, j] * Xp[n, :, i : i + self.ks, j : j + self.ks]
                        dXp[n, :, i : i + self.ks, j : j + self.ks] += d_out[n, co, i, j] * self.W[co]
        self.W -= lr * dW / N
        self.b -= lr * db / N
        if self.padding > 0:
            return dXp[:, :, self.padding : -self.padding, self.padding : -self.padding]
        return dXp


class DepthwiseConv2D:
    """
    Depthwise Convolution: applies one filter per input channel.

    Unlike standard convolution which mixes channels, depthwise conv
    processes each channel independently, greatly reducing computation.
    """

    def __init__(self, channels: int, ks: int, padding: int = 0, init_scale: float = 0.1):
        self.channels = channels
        self.ks = ks
        self.padding = padding
        # One filter per channel: (C, kH, kW)
        self.W = np.random.randn(channels, ks, ks).astype(np.float32) * init_scale
        self.b = np.zeros(channels, dtype=np.float32)
        self.cache = None

    def _pad(self, X):
        if self.padding == 0:
            return X
        return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply depthwise convolution.

        Parameters
        ----------
        X : (N, C, H, W)

        Returns
        -------
        out : (N, C, Ho, Wo) - same number of channels
        """
        Xp = self._pad(X)
        N, C, H, W = Xp.shape
        Ho = H - self.ks + 1
        Wo = W - self.ks + 1
        out = np.zeros((N, C, Ho, Wo), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        out[n, c, i, j] = (
                            np.sum(Xp[n, c, i : i + self.ks, j : j + self.ks] * self.W[c]) + self.b[c]
                        )
        self.cache = (X, Xp)
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass through depthwise convolution."""
        X, Xp = self.cache
        N, C, H, W = Xp.shape
        _, _, Ho, Wo = d_out.shape
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dXp = np.zeros_like(Xp)

        for n in range(N):
            for c in range(C):
                db[c] += np.sum(d_out[n, c])
                for i in range(Ho):
                    for j in range(Wo):
                        dW[c] += d_out[n, c, i, j] * Xp[n, c, i : i + self.ks, j : j + self.ks]
                        dXp[n, c, i : i + self.ks, j : j + self.ks] += d_out[n, c, i, j] * self.W[c]

        self.W -= lr * dW / N
        self.b -= lr * db / N
        if self.padding > 0:
            return dXp[:, :, self.padding : -self.padding, self.padding : -self.padding]
        return dXp


class SqueezeExcitation:
    """
    Squeeze-and-Excitation block: learns channel-wise attention.

    Squeeze:    Global Average Pooling -> (N, C)
    Excitation: FC(C -> C//r) -> ReLU -> FC(C//r -> C) -> Sigmoid
    Scale:      Channel-wise multiplication
    """

    def __init__(self, channels: int, reduction: int = 4, init_scale: float = 0.1):
        self.channels = channels
        self.reduced = max(1, channels // reduction)
        self.fc1_W = np.random.randn(channels, self.reduced).astype(np.float32) * init_scale
        self.fc1_b = np.zeros(self.reduced, dtype=np.float32)
        self.fc2_W = np.random.randn(self.reduced, channels).astype(np.float32) * init_scale
        self.fc2_b = np.zeros(channels, dtype=np.float32)
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass for SE block.

        Parameters
        ----------
        X : (N, C, H, W)

        Returns
        -------
        out : (N, C, H, W) - scaled by channel attention
        """
        N, C, H, W = X.shape

        # Squeeze: Global Average Pooling
        squeezed = X.mean(axis=(2, 3))  # (N, C)

        # Excitation
        fc1_out = squeezed @ self.fc1_W + self.fc1_b  # (N, reduced)
        relu_out = np.maximum(0, fc1_out)  # ReLU
        fc2_out = relu_out @ self.fc2_W + self.fc2_b  # (N, C)
        scale = _sigmoid(fc2_out)  # (N, C)

        # Scale
        out = X * scale.reshape(N, C, 1, 1)

        self.cache = (X, squeezed, fc1_out, relu_out, fc2_out, scale)
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass through SE block."""
        X, squeezed, fc1_out, relu_out, fc2_out, scale = self.cache
        N, C, H, W = X.shape

        # d_out = dL/d(X * scale)
        # dL/dX_direct = d_out * scale (broadcast)
        dX_direct = d_out * scale.reshape(N, C, 1, 1)

        # dL/dscale = sum over H,W of (d_out * X)
        d_scale = (d_out * X).sum(axis=(2, 3))  # (N, C)

        # Backward through sigmoid
        d_fc2_out = d_scale * scale * (1.0 - scale)  # (N, C)

        # Backward through FC2
        dfc2_W = relu_out.T @ d_fc2_out / N  # (reduced, C)
        dfc2_b = d_fc2_out.mean(axis=0)  # (C,)
        d_relu = d_fc2_out @ self.fc2_W.T  # (N, reduced)

        # Backward through ReLU
        d_fc1_out = d_relu * (fc1_out > 0).astype(np.float32)  # (N, reduced)

        # Backward through FC1
        dfc1_W = squeezed.T @ d_fc1_out / N  # (C, reduced)
        dfc1_b = d_fc1_out.mean(axis=0)  # (reduced,)
        d_squeezed = d_fc1_out @ self.fc1_W.T  # (N, C)

        # Backward through Global Average Pooling
        dX_gap = d_squeezed.reshape(N, C, 1, 1) / (H * W) * np.ones((N, C, H, W), dtype=np.float32)

        # Total gradient
        dX = dX_direct + dX_gap

        # Update weights
        self.fc1_W -= lr * dfc1_W
        self.fc1_b -= lr * dfc1_b
        self.fc2_W -= lr * dfc2_W
        self.fc2_b -= lr * dfc2_b

        return dX


class BatchNorm:
    """Simplified batch normalization."""

    def __init__(self, n_ch: int, eps: float = 1e-5):
        self.gamma = np.ones(n_ch, dtype=np.float32)
        self.beta = np.zeros(n_ch, dtype=np.float32)
        self.eps = eps
        self.running_mean = np.zeros(n_ch, dtype=np.float32)
        self.running_var = np.ones(n_ch, dtype=np.float32)
        self.training = True
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        N, C, H, W = X.shape
        if self.training:
            mean = X.mean(axis=(0, 2, 3))
            var = X.var(axis=(0, 2, 3))
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean, var = self.running_mean, self.running_var

        mean_r = mean.reshape(1, C, 1, 1)
        var_r = var.reshape(1, C, 1, 1)
        X_norm = (X - mean_r) / np.sqrt(var_r + self.eps)
        out = self.gamma.reshape(1, C, 1, 1) * X_norm + self.beta.reshape(1, C, 1, 1)
        self.cache = (X, X_norm, mean_r, var_r)
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        X, X_norm, mean_r, var_r = self.cache
        N, C, H, W = X.shape
        M = N * H * W
        gamma_r = self.gamma.reshape(1, C, 1, 1)

        dgamma = np.sum(d_out * X_norm, axis=(0, 2, 3))
        dbeta = np.sum(d_out, axis=(0, 2, 3))

        dX_norm = d_out * gamma_r
        std_inv = 1.0 / np.sqrt(var_r + self.eps)
        dvar = np.sum(dX_norm * (X - mean_r) * (-0.5) * (var_r + self.eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dX_norm * (-std_inv), axis=(0, 2, 3), keepdims=True)
        dX = dX_norm * std_inv + dvar * 2.0 * (X - mean_r) / M + dmean / M

        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta
        return dX


class AvgPool2D:
    """Average pooling."""

    def __init__(self, ps: int = 2):
        self.ps = ps
        self.cache = None

    def forward(self, X):
        N, C, H, W = X.shape
        Ho, Wo = H // self.ps, W // self.ps
        out = np.zeros((N, C, Ho, Wo), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        out[n, c, i, j] = np.mean(X[n, c, i*self.ps:(i+1)*self.ps, j*self.ps:(j+1)*self.ps])
        self.cache = X.shape
        return out

    def backward(self, d_out, lr=0.0):
        N, C, Ho, Wo = d_out.shape
        dX = np.zeros(self.cache, dtype=np.float32)
        scale = 1.0 / (self.ps * self.ps)
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        dX[n, c, i*self.ps:(i+1)*self.ps, j*self.ps:(j+1)*self.ps] += d_out[n, c, i, j] * scale
        return dX


class GlobalAvgPool:
    """Global average pooling: (N, C, H, W) -> (N, C)."""

    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X.shape
        return X.mean(axis=(2, 3))

    def backward(self, d_out, lr=0.0):
        N, C, H, W = self.cache
        return (d_out.reshape(N, C, 1, 1) * np.ones((N, C, H, W), dtype=np.float32)) / (H * W)


class FullyConnected:
    """Dense layer."""

    def __init__(self, in_f, out_f, init_scale=0.1):
        self.W = np.random.randn(in_f, out_f).astype(np.float32) * init_scale
        self.b = np.zeros(out_f, dtype=np.float32)
        self.cache = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, d_out, lr):
        X = self.cache
        N = X.shape[0]
        dX = d_out @ self.W.T
        self.W -= lr * (X.T @ d_out) / N
        self.b -= lr * np.mean(d_out, axis=0)
        return dX


class SoftmaxCrossEntropy:
    """Softmax + Cross-Entropy."""

    def __init__(self):
        self.cache = None

    def forward(self, logits, y):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        N = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        self.cache = (probs, y)
        return probs, loss

    def backward(self):
        probs, y = self.cache
        N = probs.shape[0]
        d = probs.copy()
        d[np.arange(N), y] -= 1.0
        return d / N


# ---------------------------------------------------------------------------
# MBConv Block (Mobile Inverted Bottleneck Convolution)
# ---------------------------------------------------------------------------


class MBConvBlock:
    """
    Mobile Inverted Bottleneck Convolution block.

    Structure:
    x -> Expand (1x1, channels*expand_ratio) -> BN -> Swish
      -> Depthwise Conv (3x3) -> BN -> Swish
      -> Squeeze-Excitation
      -> Project (1x1, out_channels) -> BN
      -> (+x if skip connection applicable) -> output

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    expand_ratio : int
        Expansion ratio for the inverted bottleneck.
    se_ratio : int
        Squeeze-Excitation reduction ratio.
    init_scale : float
        Weight initialization scale.
    """

    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 1, se_ratio: int = 4, init_scale: float = 0.1):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.expand_ratio = expand_ratio
        self.use_skip = (in_ch == out_ch)
        mid_ch = in_ch * expand_ratio

        # Expansion (only if expand_ratio > 1)
        self.use_expand = expand_ratio > 1
        if self.use_expand:
            self.expand_conv = Conv2D(in_ch, mid_ch, 1, padding=0, init_scale=init_scale)
            self.expand_bn = BatchNorm(mid_ch)
            self.expand_act = Swish()

        # Depthwise convolution
        self.dw_conv = DepthwiseConv2D(mid_ch, 3, padding=1, init_scale=init_scale)
        self.dw_bn = BatchNorm(mid_ch)
        self.dw_act = Swish()

        # Squeeze-Excitation
        self.se = SqueezeExcitation(mid_ch, reduction=se_ratio, init_scale=init_scale)

        # Projection (pointwise)
        self.proj_conv = Conv2D(mid_ch, out_ch, 1, padding=0, init_scale=init_scale)
        self.proj_bn = BatchNorm(out_ch)

    def set_training(self, training: bool):
        if self.use_expand:
            self.expand_bn.training = training
        self.dw_bn.training = training
        self.proj_bn.training = training

    def forward(self, X: np.ndarray) -> np.ndarray:
        residual = X

        # Expansion
        if self.use_expand:
            h = self.expand_conv.forward(X)
            h = self.expand_bn.forward(h)
            h = self.expand_act.forward(h)
        else:
            h = X

        # Depthwise
        h = self.dw_conv.forward(h)
        h = self.dw_bn.forward(h)
        h = self.dw_act.forward(h)

        # SE
        h = self.se.forward(h)

        # Projection
        h = self.proj_conv.forward(h)
        h = self.proj_bn.forward(h)

        # Skip connection
        if self.use_skip:
            h = h + residual

        return h

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        if self.use_skip:
            d_skip = d_out.copy()
            d_main = d_out.copy()
        else:
            d_main = d_out
            d_skip = None

        # Backward through projection
        d_main = self.proj_bn.backward(d_main, lr)
        d_main = self.proj_conv.backward(d_main, lr)

        # Backward through SE
        d_main = self.se.backward(d_main, lr)

        # Backward through depthwise
        d_main = self.dw_act.backward(d_main, lr)
        d_main = self.dw_bn.backward(d_main, lr)
        d_main = self.dw_conv.backward(d_main, lr)

        # Backward through expansion
        if self.use_expand:
            d_main = self.expand_act.backward(d_main, lr)
            d_main = self.expand_bn.backward(d_main, lr)
            d_main = self.expand_conv.backward(d_main, lr)

        # Add skip gradient
        if self.use_skip and d_skip is not None:
            d_main = d_main + d_skip

        return d_main


# ---------------------------------------------------------------------------
# EfficientNet-like Model
# ---------------------------------------------------------------------------


class EfficientNet:
    """Simplified EfficientNet built from MBConv blocks."""

    def __init__(
        self,
        img_ch: int = 1,
        img_size: int = 16,
        n_classes: int = 5,
        base_filters: int = 8,
        expand_ratio: int = 2,
        se_ratio: int = 4,
        init_scale: float = 0.1,
    ):
        # Stem
        self.stem_conv = Conv2D(img_ch, base_filters, 3, padding=1, init_scale=init_scale)
        self.stem_bn = BatchNorm(base_filters)
        self.stem_act = Swish()

        # MBConv blocks
        self.block1 = MBConvBlock(base_filters, base_filters, expand_ratio=1, se_ratio=se_ratio, init_scale=init_scale)
        self.pool1 = AvgPool2D(2)
        self.block2 = MBConvBlock(base_filters, base_filters * 2, expand_ratio=expand_ratio, se_ratio=se_ratio, init_scale=init_scale)
        self.pool2 = AvgPool2D(2)

        # Head
        self.gap = GlobalAvgPool()
        self.fc = FullyConnected(base_filters * 2, n_classes, init_scale)
        self.loss_fn = SoftmaxCrossEntropy()

    def set_training(self, training: bool):
        self.stem_bn.training = training
        self.block1.set_training(training)
        self.block2.set_training(training)

    def forward(self, X):
        h = self.stem_conv.forward(X)
        h = self.stem_bn.forward(h)
        h = self.stem_act.forward(h)
        h = self.block1.forward(h)
        h = self.pool1.forward(h)
        h = self.block2.forward(h)
        h = self.pool2.forward(h)
        h = self.gap.forward(h)
        return self.fc.forward(h)

    def compute_loss(self, logits, y):
        return self.loss_fn.forward(logits, y)

    def backward(self, lr):
        d = self.loss_fn.backward()
        d = self.fc.backward(d, lr)
        d = self.gap.backward(d, lr)
        d = self.pool2.backward(d, lr)
        d = self.block2.backward(d, lr)
        d = self.pool1.backward(d, lr)
        d = self.block1.backward(d, lr)
        d = self.stem_act.backward(d, lr)
        d = self.stem_bn.backward(d, lr)
        d = self.stem_conv.backward(d, lr)

    def predict(self, X):
        self.set_training(False)
        logits = self.forward(X)
        self.set_training(True)
        return np.argmax(logits, axis=1)


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    img_ch: int = 1,
    img_size: int = 16,
    n_classes: int = 5,
    base_filters: int = 8,
    expand_ratio: int = 2,
    se_ratio: int = 4,
    learning_rate: float = 0.01,
    n_epochs: int = 10,
    batch_size: int = 32,
    init_scale: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train the NumPy EfficientNet. Returns dict with model and metrics."""
    model = EfficientNet(img_ch, img_size, n_classes, base_filters, expand_ratio, se_ratio, init_scale)
    model.set_training(True)

    N = len(X_train)
    train_losses, train_accs = [], []

    for epoch in range(n_epochs):
        perm = np.random.permutation(N)
        ep_loss, ep_correct, n_b = 0.0, 0, 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            Xb, yb = X_train[idx], y_train[idx]

            logits = model.forward(Xb)
            probs, loss = model.compute_loss(logits, yb)
            model.backward(learning_rate)

            ep_loss += loss
            ep_correct += np.sum(np.argmax(probs, axis=1) == yb)
            n_b += 1

        avg_loss = ep_loss / n_b
        acc = ep_correct / N
        train_losses.append(avg_loss)
        train_accs.append(acc)

        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    return {"model": model, "train_losses": train_losses, "train_accs": train_accs}


def validate(model_dict: Dict, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """Validate the model. Returns metrics dict."""
    model = model_dict["model"]
    model.set_training(False)
    logits = model.forward(X_val)
    probs, loss = model.compute_loss(logits, y_val)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_val)
    model.set_training(True)

    classes = np.unique(y_val)
    per_class = {}
    for c in classes:
        mask = y_val == c
        per_class[int(c)] = float(np.mean(preds[mask] == y_val[mask])) if mask.sum() > 0 else 0.0

    print(f"Validation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    for c, a in per_class.items():
        print(f"  Class {c}: {a:.4f}")

    return {"accuracy": accuracy, "loss": loss, "per_class_accuracy": per_class}


def test(model_dict: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Final test evaluation."""
    print("\n--- Test Set Evaluation ---")
    return validate(model_dict, X_test, y_test)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial, X_train, y_train, X_val, y_val, n_classes=5, img_size=16,
) -> float:
    """Optuna objective for EfficientNet tuning."""
    lr = trial.suggest_float("learning_rate", 1e-3, 0.05, log=True)
    init_scale = trial.suggest_float("init_scale", 0.01, 0.2)
    base_filters = trial.suggest_categorical("base_filters", [4, 8])
    expand_ratio = trial.suggest_categorical("expand_ratio", [1, 2])
    se_ratio = trial.suggest_categorical("se_ratio", [2, 4])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    md = train(
        X_train, y_train, n_classes=n_classes, img_size=img_size,
        base_filters=base_filters, expand_ratio=expand_ratio,
        se_ratio=se_ratio, learning_rate=lr, n_epochs=5,
        batch_size=batch_size, init_scale=init_scale, verbose=False,
    )
    return validate(md, X_val, y_val)["accuracy"]


def ray_tune_search(
    X_train, y_train, X_val, y_val, n_classes=5, img_size=16, num_samples=6,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search."""
    if not RAY_AVAILABLE:
        print("Ray not installed. Skipping.")
        return {}

    def trainable(config):
        md = train(
            X_train, y_train, n_classes=n_classes, img_size=img_size,
            base_filters=config["base_filters"], expand_ratio=config["expand_ratio"],
            se_ratio=config["se_ratio"], learning_rate=config["lr"],
            n_epochs=5, batch_size=config["batch_size"],
            init_scale=config["init_scale"], verbose=False,
        )
        metrics = validate(md, X_val, y_val)
        tune.report({"accuracy": metrics["accuracy"]})

    search_space = {
        "lr": tune.loguniform(1e-3, 0.05),
        "init_scale": tune.uniform(0.01, 0.2),
        "base_filters": tune.choice([4, 8]),
        "expand_ratio": tune.choice([1, 2]),
        "se_ratio": tune.choice([2, 4]),
        "batch_size": tune.choice([16, 32]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    tuner = tune.Tuner(
        trainable, param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, metric="accuracy", mode="max"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config: {best.config}")
    print(f"Best accuracy: {best.metrics['accuracy']:.4f}")
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the full NumPy EfficientNet pipeline."""
    print("=" * 70)
    print("Simplified EfficientNet - NumPy From-Scratch Implementation")
    print("=" * 70)

    IMG_SIZE = 16
    N_CLASSES = 5

    # 1. Generate
    print("\n[1/6] Generating synthetic data...")
    X, y = generate_data(n_samples=1000, img_size=IMG_SIZE, n_classes=N_CLASSES)
    print(f"    Dataset: {X.shape}")

    # 2. Split
    print("\n[2/6] Splitting data...")
    n = len(X)
    nt = int(0.7 * n)
    nv = int(0.15 * n)
    X_train, y_train = X[:nt], y[:nt]
    X_val, y_val = X[nt : nt + nv], y[nt : nt + nv]
    X_test, y_test = X[nt + nv :], y[nt + nv :]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Train
    print("\n[3/6] Training EfficientNet (this will take a while)...")
    model_dict = train(
        X_train, y_train, img_size=IMG_SIZE, n_classes=N_CLASSES,
        base_filters=8, expand_ratio=2, se_ratio=4,
        learning_rate=0.01, n_epochs=8, batch_size=32,
    )

    # 4. Validate
    print("\n[4/6] Validation...")
    validate(model_dict, X_val, y_val)

    # 5. Optuna
    print("\n[5/6] Optuna hyperparameter search...")
    if OPTUNA_AVAILABLE:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, N_CLASSES, IMG_SIZE),
            n_trials=6,
        )
        print(f"    Best accuracy: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        bp = study.best_params
        model_dict = train(
            X_train, y_train, img_size=IMG_SIZE, n_classes=N_CLASSES,
            base_filters=bp["base_filters"], expand_ratio=bp["expand_ratio"],
            se_ratio=bp["se_ratio"], learning_rate=bp["learning_rate"],
            n_epochs=8, batch_size=bp["batch_size"], init_scale=bp["init_scale"],
        )
    else:
        print("    Optuna not installed, skipping.")

    # 6. Test
    print("\n[6/6] Test evaluation...")
    test_metrics = test(model_dict, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
