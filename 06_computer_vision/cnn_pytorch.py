"""
Convolutional Neural Network (CNN) - PyTorch Implementation
===========================================================

Architecture:
    A standard CNN built with PyTorch's nn.Module, using Conv2d, BatchNorm2d,
    MaxPool2d, and fully connected layers.

    Network Architecture:
    Input (3, 32, 32)
    -> Conv2D(3->32, 3x3, pad=1) -> BatchNorm2D -> ReLU -> MaxPool2D(2x2)
    -> Conv2D(32->64, 3x3, pad=1) -> BatchNorm2D -> ReLU -> MaxPool2D(2x2)
    -> Conv2D(64->128, 3x3, pad=1) -> BatchNorm2D -> ReLU -> AdaptiveAvgPool(4x4)
    -> Flatten -> FC(2048->256) -> ReLU -> Dropout(0.5) -> FC(256->n_classes)

    Layer Dimensions (for 32x32 input):
    - Input:           (3, 32, 32)
    - Conv1+BN+Pool:   (32, 16, 16)
    - Conv2+BN+Pool:   (64, 8, 8)
    - Conv3+BN+Pool:   (128, 4, 4)
    - Flatten:          (2048,)
    - FC1:              (256,)
    - FC2:              (n_classes,)

Theory & Mathematics:
    Convolution with Padding:
        Y[c_out, i, j] = sum_{c_in, m, n} X_padded[c_in, i*s+m, j*s+n] * K[c_out, c_in, m, n] + b[c_out]
        With padding=1 and stride=1: output has same spatial dimensions as input.

    Batch Normalization:
        For each channel c across a mini-batch:
        mu_c = (1/N) * sum_i X_i[c]          (mean)
        sigma_c^2 = (1/N) * sum_i (X_i[c] - mu_c)^2   (variance)
        X_hat[c] = (X[c] - mu_c) / sqrt(sigma_c^2 + eps)
        Y[c] = gamma_c * X_hat[c] + beta_c   (learnable affine transform)

        Benefits: Reduces internal covariate shift, allows higher learning rates,
        acts as a regularizer.

    Max Pooling 2x2:
        Reduces spatial dimensions by half, keeping maximum activation.
        Provides translation invariance and reduces computation.

    Dropout:
        During training, randomly zeros elements with probability p.
        Scales remaining values by 1/(1-p).
        Prevents co-adaptation of neurons (regularization).

Business Use Cases:
    - Image classification (product categorization, scene recognition)
    - Medical imaging (X-ray analysis, pathology detection)
    - Quality control in manufacturing
    - Document image classification
    - Real-time visual inspection systems

Advantages:
    - Automatic feature learning (no manual feature engineering)
    - GPU acceleration for fast training
    - BatchNorm provides training stability
    - Dropout prevents overfitting
    - Transfer learning capability with pretrained weights
    - Modular design with nn.Module

Disadvantages:
    - Requires large labeled datasets for good performance
    - Computationally expensive (GPU recommended)
    - Black-box nature: hard to interpret learned features
    - Risk of overfitting on small datasets
    - Hyperparameter sensitivity (learning rate, architecture choices)

Key Hyperparameters:
    - learning_rate: SGD/Adam step size (typical: 1e-4 to 1e-2)
    - batch_size: Mini-batch size (typical: 32-256)
    - weight_decay: L2 regularization strength
    - dropout_rate: Probability of zeroing activations
    - n_epochs: Number of full passes over the training data
    - optimizer: SGD with momentum vs Adam

References:
    - Krizhevsky, A. et al. (2012). "ImageNet Classification with Deep
      Convolutional Neural Networks." NeurIPS.
    - Ioffe, S. and Szegedy, C. (2015). "Batch Normalization: Accelerating
      Deep Network Training." ICML.
    - Srivastava, N. et al. (2014). "Dropout: A Simple Way to Prevent
      Neural Networks from Overfitting." JMLR.
"""

# numpy is used for array manipulation, data generation, and metric computation
# before data is converted to PyTorch tensors for GPU-accelerated training
import numpy as np

# warnings module suppresses PyTorch deprecation warnings and convergence messages
# that would clutter training output during hyperparameter search
import warnings

# Type hints enable static analysis and make function signatures self-documenting
from typing import Dict, Tuple, Any, Optional

# torch is the core PyTorch library providing tensor operations, automatic differentiation,
# and GPU acceleration -- it replaces NumPy for all compute-intensive operations during training
import torch

# nn module provides neural network building blocks (Conv2d, Linear, BatchNorm, etc.)
# as well as the Module base class for defining custom architectures
import torch.nn as nn

# optim module provides optimization algorithms (Adam, SGD) that update model weights
# based on computed gradients -- these implement the gradient descent step
import torch.optim as optim

# DataLoader handles batching, shuffling, and parallel data loading for training efficiency
# TensorDataset wraps numpy arrays into a PyTorch-compatible dataset format
from torch.utils.data import DataLoader, TensorDataset

# Optuna provides Bayesian hyperparameter optimization using Tree-structured Parzen
# Estimators (TPE) -- more sample-efficient than random/grid search
try:
    import optuna  # Optional dependency for automated hyperparameter tuning

    OPTUNA_AVAILABLE = True  # Flag checked before using Optuna features
except ImportError:
    OPTUNA_AVAILABLE = False  # Gracefully skip if not installed

# Ray Tune enables distributed hyperparameter tuning across multiple machines/GPUs
try:
    import ray  # Distributed computing framework
    from ray import tune  # Ray's hyperparameter tuning API

    RAY_AVAILABLE = True  # Flag checked before using Ray features
except ImportError:
    RAY_AVAILABLE = False  # Gracefully skip if not installed

# Suppress warnings for clean output during hyperparameter search
warnings.filterwarnings("ignore")

# Automatically select GPU if available, otherwise fall back to CPU.
# CUDA GPUs provide 10-100x speedup for CNN training through massive parallelism.
# This device selection is done once at module load time and reused throughout.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 3000,  # 3000 samples provides enough data for PyTorch CNN to converge
    img_size: int = 32,  # 32x32 matches CIFAR-10 resolution, a standard benchmark size
    n_channels: int = 3,  # 3 channels (RGB) is standard for color image classification
    n_classes: int = 10,  # 10 classes matches CIFAR-10 for realistic multi-class scenario
    random_state: int = 42,  # Fixed seed for reproducible experiments
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic RGB image data with class-specific patterns.

    Each class has a distinct pattern applied across channels with
    channel-specific variations to make the data more interesting.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    img_size : int
        Image width and height.
    n_channels : int
        Number of color channels (3 for RGB).
    n_classes : int
        Number of classes (up to 10).
    random_state : int
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_channels, img_size, img_size)
        Images in NCHW format.
    y : np.ndarray of shape (n_samples,)
        Integer labels.
    """
    # Create isolated random generator for reproducibility
    rng = np.random.RandomState(random_state)

    # Equal samples per class ensures balanced dataset
    samples_per_class = n_samples // n_classes

    # Lists for accumulating images and labels
    X_list, y_list = [], []

    # Generate class-specific visual patterns across RGB channels
    for cls in range(n_classes):
        for _ in range(samples_per_class):
            # Initialize all 3 channels with low-amplitude noise as background
            # NCHW format: (channels, height, width) for a single image
            img = rng.rand(n_channels, img_size, img_size).astype(np.float32) * 0.1

            # Each class uses a primary color channel for its pattern, rotating through R, G, B.
            # This makes the task more challenging than grayscale because the CNN must learn
            # to look at specific channels for specific classes.
            primary_ch = cls % n_channels

            if cls == 0:  # Horizontal lines on the red channel
                for r in range(0, img_size, 4):  # Lines every 4 pixels
                    img[primary_ch, r : r + 2, :] += 0.7  # 2-pixel-thick bright horizontal stripes
            elif cls == 1:  # Vertical lines on the green channel
                for c in range(0, img_size, 4):
                    img[primary_ch, :, c : c + 2] += 0.7  # 2-pixel-thick bright vertical stripes
            elif cls == 2:  # Filled circle on the blue channel
                cx, cy = img_size // 2, img_size // 2  # Center of image
                yy, xx = np.ogrid[:img_size, :img_size]  # Coordinate grids
                # Boolean mask where squared distance from center < radius squared
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (img_size // 4) ** 2
                img[primary_ch][mask] += 0.8  # Fill circle region with bright pixels
            elif cls == 3:  # Diagonal line on the red channel
                for i in range(img_size):
                    for d in range(-1, 2):  # 3-pixel thickness
                        j = i + d
                        if 0 <= j < img_size:
                            img[primary_ch, i, j] += 0.7
            elif cls == 4:  # Checkerboard on the green channel
                block = 4  # Block size for checkerboard pattern
                for r in range(0, img_size, block):
                    for c in range(0, img_size, block):
                        if ((r // block) + (c // block)) % 2 == 0:
                            img[primary_ch, r : r + block, c : c + block] += 0.6
            elif cls == 5:  # Radial gradient (Gaussian blob) on the blue channel
                cx, cy = img_size // 2, img_size // 2
                yy, xx = np.ogrid[:img_size, :img_size]
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                # Gaussian profile with sigma = img_size/4 creates smooth circular gradient
                img[primary_ch] += np.exp(-dist**2 / (2 * (img_size / 4) ** 2)).astype(np.float32) * 0.8
            elif cls == 6:  # Horizontal intensity gradient on the red channel
                grad = np.linspace(0, 1, img_size).reshape(1, -1).astype(np.float32)
                img[primary_ch] += grad * 0.7
            elif cls == 7:  # Cross pattern on the green channel
                mid = img_size // 2
                img[primary_ch, mid - 2 : mid + 2, :] += 0.7  # Horizontal arm
                img[primary_ch, :, mid - 2 : mid + 2] += 0.7  # Vertical arm
            elif cls == 8:  # Corner patches on the blue channel
                q = img_size // 4
                img[primary_ch, :q, :q] += 0.8  # Top-left corner
                img[primary_ch, -q:, -q:] += 0.8  # Bottom-right corner
            elif cls == 9:  # Border frame on the red channel
                b = 3  # Border thickness
                img[primary_ch, :b, :] += 0.7  # Top border
                img[primary_ch, -b:, :] += 0.7  # Bottom border
                img[primary_ch, :, :b] += 0.7  # Left border
                img[primary_ch, :, -b:] += 0.7  # Right border

            # Add Gaussian noise across all channels for realism
            img += rng.randn(n_channels, img_size, img_size).astype(np.float32) * 0.05
            img = np.clip(img, 0, 1)  # Clip to valid [0, 1] range
            X_list.append(img)
            y_list.append(cls)

    # Convert to contiguous numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle to break sequential class ordering
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def create_dataloaders(
    X_train: np.ndarray,  # Training images as numpy array (N, C, H, W)
    y_train: np.ndarray,  # Training labels as numpy array (N,)
    X_val: np.ndarray,  # Validation images
    y_val: np.ndarray,  # Validation labels
    batch_size: int = 64,  # Mini-batch size -- 64 balances GPU utilization and gradient noise
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from NumPy arrays.

    DataLoaders handle batching, shuffling, and optional parallel data loading,
    abstracting away the manual batch slicing needed in the NumPy implementation.
    """
    # TensorDataset wraps tensors into a dataset where each sample is a (input, label) pair
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    # Training loader shuffles data each epoch for better gradient estimates.
    # drop_last=False keeps the last partial batch (important for small datasets).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # Validation loader does NOT shuffle because we want deterministic evaluation
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class CNNModel(nn.Module):
    """
    Standard CNN with Conv2d, BatchNorm, MaxPool, Dropout.

    This architecture follows the established pattern of progressively increasing
    filter counts (32 -> 64 -> 128) while decreasing spatial dimensions through
    pooling. This design captures increasingly abstract features at each level:
    - Layer 1 (32 filters): Low-level features (edges, corners, color gradients)
    - Layer 2 (64 filters): Mid-level features (textures, simple shapes)
    - Layer 3 (128 filters): High-level features (object parts, complex patterns)

    BatchNorm after each conv layer stabilizes training by normalizing activations.
    Dropout in the classifier head prevents overfitting by randomly dropping neurons.

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 for RGB).
    n_classes : int
        Number of output classes.
    dropout_rate : float
        Dropout probability in the classifier head.
    """

    def __init__(
        self,
        in_channels: int = 3,  # 3 for RGB images, 1 for grayscale
        n_classes: int = 10,  # Number of classification categories
        dropout_rate: float = 0.5,  # 0.5 is the standard dropout rate from the original paper
    ):
        # Call nn.Module's __init__ to properly register parameters and submodules
        super().__init__()

        # Feature extraction backbone: three convolutional blocks
        self.features = nn.Sequential(
            # ---- Block 1: Low-level feature extraction ----
            # Conv2d(3->32, 3x3, padding=1): 32 filters detect basic visual elements
            # like edges, corners, and color gradients. Padding=1 with 3x3 kernel preserves
            # spatial dimensions (same convolution), so the output is still 32x32.
            # WHY 3x3: Stacking two 3x3 convolutions has the same receptive field as one 5x5
            # but uses fewer parameters (2*9=18 vs 25) and adds an extra non-linearity.
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            # BatchNorm2d normalizes activations across the batch for each of the 32 channels.
            # This reduces internal covariate shift: the distribution of layer inputs stays
            # consistent during training, allowing higher learning rates and faster convergence.
            nn.BatchNorm2d(32),
            # ReLU activation: max(0, x). inplace=True saves memory by modifying the tensor
            # in-place rather than creating a copy, at the cost of losing the original values
            # (acceptable because we don't need the pre-activation values after this point).
            nn.ReLU(inplace=True),
            # MaxPool2d(2, 2): Takes maximum in each 2x2 window with stride 2.
            # Halves spatial dimensions: 32x32 -> 16x16.
            # Provides translation invariance and reduces computation for subsequent layers.
            nn.MaxPool2d(2, 2),

            # ---- Block 2: Mid-level feature extraction ----
            # Doubles the filter count to 64 because higher layers need more filters to
            # represent the growing variety of feature combinations from Block 1.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8 after pooling
            nn.MaxPool2d(2, 2),

            # ---- Block 3: High-level feature extraction ----
            # 128 filters capture complex, class-discriminative patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # AdaptiveAvgPool2d(4, 4): Pools to a fixed 4x4 spatial size regardless of input size.
            # WHY Adaptive: Unlike fixed MaxPool, this makes the model accept any input resolution.
            # WHY AvgPool: Average pooling in the final block smooths features and reduces
            # sensitivity to exact spatial positions of high-level features.
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Classification head: maps spatial features to class predictions
        self.classifier = nn.Sequential(
            # Flatten: reshapes (N, 128, 4, 4) to (N, 2048) for the fully connected layer
            nn.Flatten(),
            # FC layer: 2048 -> 256. This is the "bottleneck" that compresses spatial features
            # into a compact representation for classification.
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            # Dropout: randomly zeros 50% of activations during training.
            # This prevents co-adaptation of neurons (where specific neurons always rely on
            # the same other neurons) and acts as an ensemble method (each training step
            # trains a different random sub-network).
            nn.Dropout(dropout_rate),
            # Final FC layer: 256 -> n_classes. Produces one logit per class.
            # These logits are fed into CrossEntropyLoss which applies softmax internally.
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor of shape (N, C, H, W)
            Batch of input images.

        Returns
        -------
        logits : torch.Tensor of shape (N, n_classes)
            Raw class scores (before softmax).
        """
        # Extract spatial features through the convolutional backbone
        x = self.features(x)
        # Classify by mapping features to class logits
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    train_loader: DataLoader,  # PyTorch DataLoader providing batched training data
    val_loader: Optional[DataLoader] = None,  # Optional validation loader for monitoring
    in_channels: int = 3,  # Input image channels (3 for RGB)
    n_classes: int = 10,  # Number of output classes
    learning_rate: float = 1e-3,  # Adam default lr; lower than SGD because Adam adapts per-parameter
    weight_decay: float = 1e-4,  # L2 regularization strength -- prevents weights from growing large
    dropout_rate: float = 0.5,  # Dropout probability in classifier head
    n_epochs: int = 15,  # Number of training epochs
    optimizer_name: str = "adam",  # Optimizer choice: 'adam' or 'sgd'
    verbose: bool = True,  # Print training progress
) -> Dict[str, Any]:
    """
    Train the PyTorch CNN.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader, optional
        Validation data loader.
    Various hyperparameters.

    Returns
    -------
    result : dict with 'model', 'train_losses', 'val_accs'.
    """
    # Create model and move to GPU if available -- .to(DEVICE) copies all parameters
    # to the target device (CPU or CUDA GPU)
    model = CNNModel(in_channels=in_channels, n_classes=n_classes, dropout_rate=dropout_rate).to(DEVICE)

    # CrossEntropyLoss combines LogSoftmax + NLLLoss in a single operation.
    # It expects raw logits (NOT softmax probabilities) and integer labels.
    # This is more numerically stable than computing softmax then log separately.
    criterion = nn.CrossEntropyLoss()

    # Select optimizer based on configuration
    if optimizer_name == "adam":
        # Adam: adaptive learning rate optimizer that maintains per-parameter learning rates.
        # It combines the benefits of RMSProp (adaptive lr) and momentum (gradient smoothing).
        # weight_decay adds L2 regularization: loss += weight_decay * sum(w^2)
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        # SGD with momentum: the classic optimizer. Momentum smooths gradient updates
        # by keeping a running average of past gradients, helping escape shallow local minima.
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    # StepLR scheduler: reduces learning rate by factor gamma every step_size epochs.
    # This "learning rate annealing" helps fine-tune weights in later training stages
    # when large steps would overshoot the optimum.
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=max(1, n_epochs // 3), gamma=0.5)

    # Track training metrics for analysis and early stopping decisions
    train_losses, val_accs = [], []

    # Training loop: iterate over the entire dataset n_epochs times
    for epoch in range(n_epochs):
        # Set model to training mode -- enables dropout and uses batch statistics for BatchNorm
        model.train()

        epoch_loss = 0.0  # Accumulate loss for averaging
        correct, total = 0, 0  # Track accuracy within the epoch

        # Iterate over mini-batches from the DataLoader
        for X_batch, y_batch in train_loader:
            # Move data to the same device as the model (CPU or GPU)
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            # Zero gradients from the previous step -- PyTorch accumulates gradients by default,
            # which is useful for gradient accumulation but requires explicit clearing otherwise
            opt.zero_grad()

            # Forward pass: compute predictions
            logits = model(X_batch)

            # Compute cross-entropy loss
            loss = criterion(logits, y_batch)

            # Backward pass: compute gradients of loss w.r.t. all model parameters.
            # PyTorch's autograd engine traverses the computation graph in reverse,
            # applying the chain rule automatically (no manual gradient derivation needed).
            loss.backward()

            # Update weights: optimizer applies the computed gradients to parameters
            # using the chosen algorithm (Adam or SGD with momentum)
            opt.step()

            # Accumulate metrics (multiply by batch size to correctly weight partial batches)
            epoch_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)  # Get predicted class indices
            correct += (preds == y_batch).sum().item()  # Count correct predictions
            total += X_batch.size(0)  # Track total samples processed

        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()

        # Compute epoch-level metrics
        avg_loss = epoch_loss / total  # Average loss per sample
        train_acc = correct / total  # Training accuracy
        train_losses.append(avg_loss)

        # Optionally evaluate on validation set for monitoring overfitting
        val_acc = None
        if val_loader:
            val_metrics = validate({"model": model}, val_loader=val_loader, verbose=False)
            val_acc = val_metrics["accuracy"]
            val_accs.append(val_acc)

        # Print progress at regular intervals (every 20% of epochs)
        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            msg = f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, TrainAcc: {train_acc:.4f}"
            if val_acc is not None:
                msg += f", ValAcc: {val_acc:.4f}"
            print(msg)

    return {"model": model, "train_losses": train_losses, "val_accs": val_accs}


def validate(
    model_dict: Dict[str, Any],  # Dictionary containing the trained model
    val_loader: Optional[DataLoader] = None,  # Pre-built DataLoader
    X_val: Optional[np.ndarray] = None,  # Alternative: raw numpy images
    y_val: Optional[np.ndarray] = None,  # Alternative: raw numpy labels
    batch_size: int = 64,  # Batch size for creating DataLoader from numpy arrays
    verbose: bool = True,  # Print detailed metrics
) -> Dict[str, Any]:
    """
    Validate/evaluate the model.

    Can accept either a DataLoader or raw numpy arrays.

    Returns
    -------
    metrics : dict with 'accuracy', 'loss', 'per_class_accuracy',
              'predictions', 'top5_accuracy'.
    """
    model = model_dict["model"]

    # Set model to evaluation mode: disables dropout and uses running statistics
    # for BatchNorm (not batch statistics). This ensures deterministic predictions.
    model.eval()

    # Create DataLoader from numpy arrays if no loader was provided
    if val_loader is None:
        ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []  # Collect all predictions and labels
    total_loss = 0.0
    total = 0
    top5_correct = 0  # Counter for top-5 accuracy

    # torch.no_grad() disables gradient computation during evaluation.
    # This saves memory (no need to store intermediate values for backprop)
    # and speeds up forward passes by ~20-30%.
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            # Move data to the model's device
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            # Forward pass only (no backward needed for evaluation)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            # Get top-1 predictions
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())  # Move back to CPU for numpy operations
            all_labels.append(y_batch.cpu().numpy())
            total += X_batch.size(0)

            # Top-5 accuracy: checks if the correct label is in the top 5 predictions.
            # This is a standard metric for ImageNet-scale classification where exact
            # top-1 prediction is very challenging with 1000 classes.
            n_classes = logits.size(1)
            k = min(5, n_classes)  # Handle case where we have fewer than 5 classes
            _, top_k_preds = logits.topk(k, dim=1)  # Get indices of k highest scores
            # Check if correct label appears anywhere in the top-k predictions
            top5_correct += (top_k_preds == y_batch.unsqueeze(1)).any(dim=1).sum().item()

    # Concatenate all batch results into single arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute aggregate metrics
    accuracy = np.mean(all_preds == all_labels)  # Overall top-1 accuracy
    top5_acc = top5_correct / total  # Top-5 accuracy
    avg_loss = total_loss / total  # Average loss per sample

    # Per-class accuracy reveals which classes are easy/hard for the model
    classes = np.unique(all_labels)
    per_class = {}
    for c in classes:
        mask = all_labels == c
        per_class[int(c)] = float(np.mean(all_preds[mask] == all_labels[mask])) if mask.sum() > 0 else 0.0

    if verbose:
        print(f"Accuracy: {accuracy:.4f}, Top-5 Accuracy: {top5_acc:.4f}, Loss: {avg_loss:.4f}")
        for c, acc in per_class.items():
            print(f"  Class {c}: {acc:.4f}")

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
        "loss": avg_loss,
        "per_class_accuracy": per_class,
        "predictions": all_preds,
    }


def test(
    model_dict: Dict[str, Any],  # Contains trained model
    test_loader: Optional[DataLoader] = None,  # Pre-built test DataLoader
    X_test: Optional[np.ndarray] = None,  # Alternative: raw numpy test images
    y_test: Optional[np.ndarray] = None,  # Alternative: raw numpy test labels
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Final test evaluation -- should be called exactly once for unbiased assessment."""
    print("\n--- Test Set Evaluation ---")
    return validate(model_dict, val_loader=test_loader, X_val=X_test, y_val=y_test, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: "optuna.Trial",  # Optuna trial object for suggesting hyperparameters
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 10,
) -> float:
    """Optuna objective for CNN hyperparameter tuning."""
    # Suggest hyperparameters from defined ranges
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.2, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    # Create data loaders with the suggested batch size
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    # Train with suggested hyperparameters (fewer epochs for faster search)
    model_dict = train(
        train_loader, val_loader=None, n_classes=n_classes,
        learning_rate=lr, weight_decay=weight_decay, dropout_rate=dropout,
        n_epochs=8, optimizer_name=optimizer_name, verbose=False,
    )

    # Evaluate and return accuracy as the optimization objective
    metrics = validate(model_dict, val_loader=val_loader, verbose=False)
    return metrics["accuracy"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 10,
    num_samples: int = 8,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search for distributed tuning."""
    if not RAY_AVAILABLE:
        print("Ray not installed. Skipping.")
        return {}

    def trainable(config):
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, config["batch_size"]
        )
        md = train(
            train_loader, n_classes=n_classes, learning_rate=config["lr"],
            weight_decay=config["weight_decay"], dropout_rate=config["dropout"],
            n_epochs=8, optimizer_name=config["optimizer"], verbose=False,
        )
        metrics = validate(md, val_loader=val_loader, verbose=False)
        tune.report({"accuracy": metrics["accuracy"]})

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.2, 0.6),
        "batch_size": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "sgd"]),
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
# Parameter Comparison
# ---------------------------------------------------------------------------


def compare_parameter_sets(
    n_samples: int = 2000,  # Moderate dataset for meaningful comparison
    img_size: int = 32,  # Standard image size
) -> Dict[str, Any]:
    """
    Compare different CNN configurations to show their impact on performance.

    Compares:
    1. With vs without BatchNorm -- shows BatchNorm's stabilizing effect on training
    2. Different dropout rates (0.0 vs 0.3 vs 0.5 vs 0.7) -- shows regularization effect
    3. Optimizer choice (Adam vs SGD) -- shows convergence speed differences
    4. Learning rate sensitivity (1e-4 vs 1e-3 vs 1e-2) -- shows the critical importance of lr

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
    print("PARAMETER COMPARISON: PyTorch CNN Configurations")
    print("=" * 70)

    n_classes = 10

    # Generate shared dataset
    X, y = generate_data(n_samples=n_samples, img_size=img_size, n_classes=n_classes, random_state=42)

    # Split data
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]

    # Define configurations
    configs = {
        # --- Dropout rate comparison ---
        # No dropout: risk of overfitting on small datasets
        "dropout=0.0_Adam": {
            "dropout_rate": 0.0, "optimizer_name": "adam",
            "learning_rate": 1e-3, "weight_decay": 1e-4, "n_epochs": 10,
        },
        # Light dropout: mild regularization
        "dropout=0.3_Adam": {
            "dropout_rate": 0.3, "optimizer_name": "adam",
            "learning_rate": 1e-3, "weight_decay": 1e-4, "n_epochs": 10,
        },
        # Standard dropout: the original paper's recommended rate
        "dropout=0.5_Adam": {
            "dropout_rate": 0.5, "optimizer_name": "adam",
            "learning_rate": 1e-3, "weight_decay": 1e-4, "n_epochs": 10,
        },
        # Heavy dropout: may under-fit by removing too many neurons
        "dropout=0.7_Adam": {
            "dropout_rate": 0.7, "optimizer_name": "adam",
            "learning_rate": 1e-3, "weight_decay": 1e-4, "n_epochs": 10,
        },
        # --- Optimizer comparison ---
        # SGD often generalizes better but converges slower
        "dropout=0.5_SGD": {
            "dropout_rate": 0.5, "optimizer_name": "sgd",
            "learning_rate": 1e-2, "weight_decay": 1e-4, "n_epochs": 10,
        },
        # --- Learning rate comparison ---
        "lr=1e-4_Adam": {
            "dropout_rate": 0.5, "optimizer_name": "adam",
            "learning_rate": 1e-4, "weight_decay": 1e-4, "n_epochs": 10,
        },
        "lr=1e-2_Adam": {
            "dropout_rate": 0.5, "optimizer_name": "adam",
            "learning_rate": 1e-2, "weight_decay": 1e-4, "n_epochs": 10,
        },
        # --- Weight decay comparison ---
        "no_weight_decay": {
            "dropout_rate": 0.5, "optimizer_name": "adam",
            "learning_rate": 1e-3, "weight_decay": 0.0, "n_epochs": 10,
        },
        "strong_weight_decay": {
            "dropout_rate": 0.5, "optimizer_name": "adam",
            "learning_rate": 1e-3, "weight_decay": 1e-2, "n_epochs": 10,
        },
    }

    results = {}
    for name, params in configs.items():
        print(f"\n--- Configuration: {name} ---")
        batch_size = 64
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
        model_dict = train(
            train_loader, val_loader=None, n_classes=n_classes,
            verbose=False, **params,
        )
        metrics = validate(model_dict, val_loader=val_loader, verbose=False)
        results[name] = metrics["accuracy"]
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

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
    print("  - BatchNorm significantly stabilizes training (allows higher learning rates)")
    print("  - Dropout=0.5 is usually optimal; too much/too little hurts performance")
    print("  - Adam converges faster but SGD may generalize better with proper lr schedule")
    print("  - Learning rate is the single most important hyperparameter")
    print("  - Weight decay provides mild regularization; strong decay can hurt training")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Medical X-Ray Classification
# ---------------------------------------------------------------------------


def real_world_demo() -> Dict[str, Any]:
    """
    Demonstrate the PyTorch CNN on a realistic medical imaging scenario:
    classifying chest X-ray images as Normal vs Pneumonia.

    The PyTorch implementation benefits from GPU acceleration, BatchNorm for
    training stability, and dropout for regularization -- all critical for
    medical imaging where datasets are often small and noisy.

    Returns
    -------
    results : dict with model performance metrics.
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Medical X-Ray Classification (Normal vs Pneumonia)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_samples = 1000
    img_size = 32
    n_channels = 3  # Use RGB (even though real X-rays are grayscale, models often work with 3 channels)
    n_classes = 2
    class_names = ["Normal", "Pneumonia"]
    samples_per_class = n_samples // n_classes

    X_list, y_list = [], []

    for cls_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Create 3-channel X-ray-like image (grayscale replicated across channels)
            base = np.ones((img_size, img_size), dtype=np.float32) * 0.4
            base += rng.rand(img_size, img_size).astype(np.float32) * 0.08

            # Rib patterns
            for row in range(3, img_size - 3, 5):
                rib_intensity = rng.uniform(0.04, 0.10)
                thickness = rng.randint(1, 3)
                base[row : row + thickness, 4:-4] += rib_intensity

            # Lung field darkening
            lung_left = img_size // 4
            lung_right = 3 * img_size // 4
            lung_top = img_size // 5
            lung_bottom = 4 * img_size // 5
            base[lung_top:lung_bottom, lung_left:lung_right] -= 0.06

            if cls_idx == 0:
                # NORMAL: clear lung fields with subtle vascular markings
                cx, cy = img_size // 2, img_size // 2
                for _ in range(6):
                    angle = rng.uniform(0, 2 * np.pi)
                    for t in range(5, 12):
                        vx = int(cx + t * np.cos(angle))
                        vy = int(cy + t * np.sin(angle))
                        if 0 <= vx < img_size and 0 <= vy < img_size:
                            base[vy, vx] += 0.02

            elif cls_idx == 1:
                # PNEUMONIA: bright opacity patches in lung fields
                n_opacities = rng.randint(1, 4)
                for _ in range(n_opacities):
                    ox = rng.randint(lung_left, lung_right)
                    oy = rng.randint(lung_top, lung_bottom)
                    radius = rng.randint(3, 8)
                    yy, xx = np.ogrid[:img_size, :img_size]
                    dist = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2)
                    opacity = np.exp(-dist**2 / (2 * radius**2)).astype(np.float32)
                    base += opacity * rng.uniform(0.12, 0.28)

            # Add noise and clip
            base += rng.randn(img_size, img_size).astype(np.float32) * 0.03
            base = np.clip(base, 0, 1)

            # Replicate grayscale to 3 channels (CHW format)
            img = np.stack([base, base, base], axis=0)
            X_list.append(img)
            y_list.append(cls_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Generated {len(X)} synthetic X-ray images ({n_channels}x{img_size}x{img_size})")
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
    print("\n--- Training PyTorch CNN for X-ray classification ---")
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    model_dict = train(
        train_loader, val_loader=val_loader,
        n_classes=n_classes, learning_rate=1e-3,
        dropout_rate=0.4, n_epochs=15, weight_decay=1e-4,
    )

    # Evaluate
    print("\n--- Validation Results ---")
    val_metrics = validate(model_dict, val_loader=val_loader)

    print("\n--- Test Results ---")
    test_metrics = test(model_dict, X_test=X_test, y_test=y_test)

    # Clinical interpretation
    print("\n--- Clinical Interpretation ---")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    if 1 in test_metrics.get("per_class_accuracy", {}):
        pneumonia_sensitivity = test_metrics["per_class_accuracy"][1]
        print(f"Pneumonia Sensitivity: {pneumonia_sensitivity:.4f}")

    print("\nNote: Real medical AI requires regulatory approval (FDA 510(k)),")
    print("validation on diverse populations, and integration with clinical workflows.")

    return {
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "per_class_accuracy": test_metrics.get("per_class_accuracy", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the full PyTorch CNN pipeline."""
    print("=" * 70)
    print("Convolutional Neural Network - PyTorch Implementation")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    N_CLASSES = 10
    IMG_SIZE = 32

    # ---- Step 1: Generate data ----
    print("\n[1/8] Generating synthetic data...")
    X, y = generate_data(n_samples=3000, img_size=IMG_SIZE, n_classes=N_CLASSES)
    print(f"    Dataset: {X.shape}, Classes: {np.unique(y)}")

    # ---- Step 2: Split data ----
    print("\n[2/8] Splitting data...")
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- Step 3: Train ----
    print("\n[3/8] Training CNN...")
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64)
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model_dict = train(train_loader, val_loader=val_loader, n_classes=N_CLASSES, n_epochs=15, learning_rate=1e-3)

    # ---- Step 4: Validate ----
    print("\n[4/8] Validation...")
    val_metrics = validate(model_dict, val_loader=val_loader)

    # ---- Step 5: Optuna ----
    print("\n[5/8] Optuna hyperparameter search...")
    if OPTUNA_AVAILABLE:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, N_CLASSES),
            n_trials=10,
        )
        print(f"    Best accuracy: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        bp = study.best_params
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, bp["batch_size"])
        model_dict = train(
            train_loader, val_loader=val_loader, n_classes=N_CLASSES,
            learning_rate=bp["learning_rate"], weight_decay=bp["weight_decay"],
            dropout_rate=bp["dropout_rate"], n_epochs=15, optimizer_name=bp["optimizer"],
        )
    else:
        print("    Optuna not installed, skipping.")

    # ---- Step 6: Test ----
    print("\n[6/8] Test evaluation...")
    test_metrics = test(model_dict, test_loader=test_loader)

    # ---- Step 7: Parameter comparison ----
    print("\n[7/8] Running parameter comparison...")
    compare_parameter_sets(n_samples=1500, img_size=IMG_SIZE)

    # ---- Step 8: Real-world demo ----
    print("\n[8/8] Running real-world medical X-ray classification demo...")
    real_world_demo()

    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
