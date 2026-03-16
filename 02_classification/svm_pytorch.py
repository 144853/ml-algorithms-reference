"""
Support Vector Machine - PyTorch Implementation
=================================================

Theory & Mathematics:
    This module implements an SVM-style classifier in PyTorch using hinge
    loss optimization. The model learns a linear decision boundary by
    minimizing the hinge loss combined with L2 regularization.

    Linear SVM Formulation:
        Minimize: (1/2) * ||W||^2 + C * (1/N) * sum max(0, 1 - y_i * (W^T x_i + b))

    In PyTorch, this is implemented as:
        - A nn.Linear layer: f(x) = W^T x + b
        - Hinge loss: torch.clamp(1 - y * f(x), min=0).mean()
        - L2 regularization via optimizer weight_decay

    Multi-class Hinge Loss (Weston-Watkins formulation):
        L_i = sum_{j != y_i} max(0, f_j(x_i) - f_{y_i}(x_i) + 1)
        This penalizes when the correct class score is not at least 1
        greater than any other class score.

    PyTorch's MultiMarginLoss implements this:
        loss(x, y) = sum_{j != y} max(0, margin - x[y] + x[j])^p / x.size(0)

    Comparison with traditional SVM:
        - Traditional SVM solves the dual QP (quadratic programming)
        - PyTorch SVM uses SGD/Adam on the primal objective
        - Both converge to similar solutions for linear SVM
        - PyTorch version scales better to large datasets

    Extended Architecture:
        This implementation also supports a "kernel-approximation" mode
        using a shallow neural network that maps features to a higher-
        dimensional space before applying the linear SVM layer:
            phi(x) = ReLU(W_1 x + b_1)  (feature mapping)
            f(x) = W_2 phi(x) + b_2     (SVM layer)

Business Use Cases:
    - Large-scale linear classification with GPU acceleration
    - Online learning with streaming data
    - Integration in deep learning pipelines as a loss function
    - Differentiable SVM for end-to-end training
    - Real-time classification in production systems

Advantages:
    - GPU-accelerated training and inference
    - Scales to large datasets with mini-batch SGD
    - Compatible with PyTorch ecosystem (ONNX, TorchServe)
    - Supports nonlinear feature mapping via neural layers
    - Autograd handles gradient computation

Disadvantages:
    - Does not directly support kernel trick (RBF, polynomial)
    - SGD convergence may be slower than dual solvers for small datasets
    - Hinge loss is not smooth (subgradient issues at kink)
    - No built-in support vector identification
    - Hyperparameter sensitivity (learning rate, weight decay)

Hyperparameters:
    - C: Hinge loss weight (regularization trade-off)
    - learning_rate: Optimizer step size
    - n_epochs: Training epochs
    - batch_size: Mini-batch size
    - weight_decay: L2 regularization (acts as 1/(2*C*N) in SVM theory)
    - optimizer: Choice of optimizer (SGD, Adam, AdamW)
    - use_feature_map: Whether to use a neural feature mapping layer
    - feature_map_dim: Dimension of the feature mapping
"""

# ── Standard library imports ──
# logging: provides structured diagnostic output for tracking training progress
# WHY: essential for monitoring convergence and debugging in production ML pipelines
import logging

# warnings: controls display of non-critical Python warnings
# WHY: we suppress sklearn/pytorch warnings to keep console output focused on our metrics
import warnings

# typing: provides type annotations for function signatures
# WHY: type hints make code self-documenting and catch bugs early via mypy
from typing import Any, Dict, Tuple

# ── Third-party imports ──
# numpy: numerical computing for data manipulation and metric computation
# WHY: used for data generation, label mapping, and post-processing predictions
import numpy as np

# optuna: Bayesian hyperparameter optimization framework
# WHY: TPE sampler efficiently explores complex hyperparameter spaces
import optuna

# torch: PyTorch deep learning framework
# WHY: provides GPU acceleration, autograd for automatic gradient computation,
# and the nn module for building SVM-style models
import torch
import torch.nn as nn

# DataLoader/TensorDataset: PyTorch data utilities for mini-batch training
# WHY: DataLoader handles shuffling, batching, and parallel data loading efficiently
from torch.utils.data import DataLoader, TensorDataset

# sklearn utilities: ONLY used for data generation, splitting, scaling, and evaluation
# WHY: building these utilities from scratch would distract from the PyTorch SVM implementation
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Logging configuration ──
# Set up structured logging with timestamps for tracking training progress
# WHY: timestamps help correlate training events with wall-clock time
logging.basicConfig(
    level=logging.INFO,  # INFO level shows training milestones without flooding the console
    format="%(asctime)s - %(levelname)s - %(message)s",  # Include timestamp and severity level
)
logger = logging.getLogger(__name__)  # Module-level logger for consistent naming

# Suppress non-critical warnings to keep output clean
warnings.filterwarnings("ignore")

# ── Device selection: use GPU if available, otherwise CPU ──
# WHY: GPU acceleration dramatically speeds up matrix multiplications in training;
# automatic device selection makes the code portable across machines
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class LinearSVMModel(nn.Module):
    """
    Linear SVM: single linear layer with hinge loss.

    Architecture: f(x) = W^T x + b
    This is mathematically equivalent to the primal SVM formulation.

    WHY nn.Module:
        - Integrates with PyTorch autograd for automatic gradient computation
        - Supports GPU acceleration via .to(device)
        - Compatible with PyTorch optimizers (SGD, Adam, etc.)
        - Can be exported to ONNX for production deployment
    """

    def __init__(self, n_features: int, n_classes: int = 2) -> None:
        """
        Initialize the linear SVM model.

        Args:
            n_features: number of input features (dimensionality of feature space)
            n_classes: number of target classes
        """
        super().__init__()

        # For binary classification, we only need a single output neuron
        # WHY: binary SVM uses a single decision function f(x) with sign(f(x)) as the prediction;
        # for multiclass, we need K output neurons (one per class) for the Weston-Watkins formulation
        out_dim = 1 if n_classes == 2 else n_classes

        # Single linear layer: f(x) = W^T x + b
        # WHY: a linear SVM IS a linear model; the decision boundary is a hyperplane
        # defined by the weight vector W and bias b
        self.linear = nn.Linear(n_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute the decision function value for each input.

        WHY: in a linear SVM, the forward pass is just a matrix multiplication;
        the hinge loss and regularization are handled by the loss function and optimizer
        """
        return self.linear(x)


class KernelApproxSVMModel(nn.Module):
    """
    SVM with neural feature mapping (kernel approximation).

    Architecture:
        phi(x) = ReLU(W_2 * ReLU(W_1 * x + b_1) + b_2)   (feature mapping)
        f(x) = W_3 * phi(x) + b_3                          (SVM classification layer)

    WHY this architecture:
        - Linear SVMs cannot learn nonlinear decision boundaries
        - Traditional kernel SVMs (RBF, polynomial) don't scale to large datasets
        - A neural feature mapping learns a nonlinear transformation of the input space,
          after which a linear SVM layer can find a separating hyperplane
        - This is related to Random Kitchen Sinks (Rahimi & Recht, 2007) and
          kernel approximation via random features, but uses learned features instead
        - The feature map is trained end-to-end with the SVM objective
    """

    def __init__(self, n_features: int, n_classes: int = 2, feature_map_dim: int = 64) -> None:
        """
        Initialize the kernel approximation SVM model.

        Args:
            n_features: number of input features
            n_classes: number of target classes
            feature_map_dim: dimensionality of the learned feature space
        """
        super().__init__()

        # For binary classification, we need a single output neuron
        out_dim = 1 if n_classes == 2 else n_classes

        # Feature mapping: two-layer MLP that projects inputs to a higher-dimensional space
        # WHY: the two-layer architecture provides enough capacity to learn nonlinear transformations
        # while keeping the model small enough to serve as a feature extractor (not a full classifier)
        self.feature_map = nn.Sequential(
            # First layer: project from input space to feature map space
            nn.Linear(n_features, feature_map_dim),
            # ReLU activation: introduces nonlinearity (the key advantage over linear SVM)
            # WHY: without nonlinearity, the feature map would just be a linear transformation,
            # and the overall model would still be equivalent to a linear SVM
            nn.ReLU(),
            # Second layer: refine the feature representation
            nn.Linear(feature_map_dim, feature_map_dim),
            # Second ReLU: adds another nonlinear transformation for richer feature mapping
            nn.ReLU(),
        )

        # SVM classification layer: linear layer on top of the learned features
        # WHY: this is where the actual SVM decision boundary is learned;
        # the feature map above transforms the input so that a linear boundary suffices
        self.svm_layer = nn.Linear(feature_map_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: feature mapping followed by linear SVM layer.

        WHY: the feature map transforms the data into a space where linear separation works,
        then the SVM layer finds the optimal separating hyperplane in that space
        """
        # Apply the nonlinear feature mapping
        phi = self.feature_map(x)

        # Apply the linear SVM classification layer
        return self.svm_layer(phi)


# ---------------------------------------------------------------------------
# Hinge loss
# ---------------------------------------------------------------------------

class HingeLoss(nn.Module):
    """
    Binary hinge loss: L = max(0, 1 - y * f(x)).

    WHY a custom loss class:
        - PyTorch does not have a built-in binary hinge loss function
        - MultiMarginLoss handles multiclass but not binary with {-1, +1} labels
        - This implementation follows the SVM mathematical formulation exactly
        - The hinge loss penalizes predictions that are correct but too close to the boundary
          (margin < 1), encouraging a wide separation between classes

    Mathematical properties:
        - Convex but non-smooth (has a "kink" at y * f(x) = 1)
        - Zero gradient for correctly classified points outside the margin
        - Linear penalty for margin violations (unlike squared hinge loss)
    """

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean hinge loss over the batch.

        Args:
            output: model predictions of shape (batch_size, 1) or (batch_size,)
            target: true labels in {-1, +1} of shape (batch_size,)

        Returns:
            Scalar mean hinge loss value
        """
        # target * output.squeeze() computes y_i * f(x_i) for each sample
        # torch.clamp(..., min=0) implements the max(0, ...) operation
        # .mean() averages the loss over the batch
        # WHY mean instead of sum: mean makes the loss scale-invariant to batch size,
        # so the learning rate doesn't need to change when batch size changes
        return torch.clamp(1.0 - target * output.squeeze(), min=0).mean()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data and split into train/val/test sets.

    WHY we use a 60/20/20 split:
        - 60% training: sufficient data for the SVM to learn the margin
        - 20% validation: used for hyperparameter tuning without test set leakage
        - 20% test: held out for final, unbiased performance estimate
    """
    # Generate synthetic classification data with controlled complexity
    # WHY: make_classification creates a reproducible dataset with known properties
    X, y = make_classification(
        n_samples=n_samples,           # Total number of samples
        n_features=n_features,          # Dimensionality of feature space
        n_informative=n_features // 2,  # Half the features carry real signal
        n_redundant=n_features // 4,    # Quarter are linear combinations of informative features
        n_classes=n_classes,            # Number of target classes
        random_state=random_state,      # Reproducibility seed
    )

    # First split: 60% train, 40% temp
    # WHY: stratified split preserves class proportions in each subset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )

    # Second split: divide temp into 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    # Standardize features to zero mean and unit variance
    # WHY: SVM decision boundaries depend on feature magnitudes;
    # standardization ensures all features contribute equally to the margin
    # IMPORTANT: fit on training data ONLY to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Fit and transform on training data
    X_val = scaler.transform(X_val)           # Transform val using training statistics
    X_test = scaler.transform(X_test)         # Transform test using training statistics

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, **hyperparams: Any) -> nn.Module:
    """
    Train a PyTorch SVM model (linear or kernel-approximation).

    WHY PyTorch for SVM:
        - Autograd handles gradient computation automatically (no manual gradient derivation)
        - GPU acceleration for large datasets
        - Mini-batch SGD scales to millions of samples
        - Compatible with the broader PyTorch ecosystem for deployment
    """
    # ── Extract hyperparameters with sensible defaults ──
    # learning_rate: step size for the optimizer
    # WHY 0.01: conservative default that works for most SVM problems
    lr = hyperparams.get("learning_rate", 0.01)

    # n_epochs: number of complete passes over the training data
    # WHY 200: sufficient for convergence on medium datasets with SGD/Adam
    n_epochs = hyperparams.get("n_epochs", 200)

    # batch_size: samples per mini-batch
    # WHY 64: standard mini-batch size balancing gradient noise and compute efficiency
    batch_size = hyperparams.get("batch_size", 64)

    # weight_decay: L2 regularization coefficient in the optimizer
    # WHY: in SVM terms, weight_decay ~ 1/(2*C*N), providing the (1/2)||w||^2 regularization
    weight_decay = hyperparams.get("weight_decay", 1e-3)

    # C: hinge loss weight (how much to penalize margin violations)
    # WHY 1.0: standard default that balances regularization and classification accuracy
    C = hyperparams.get("C", 1.0)

    # optimizer: which PyTorch optimizer to use
    # WHY Adam: adaptive learning rate works well without manual LR tuning
    optimizer_name = hyperparams.get("optimizer", "Adam")

    # use_feature_map: whether to use the neural kernel approximation model
    # WHY False default: start with the simpler linear model; use feature map for nonlinear problems
    use_feature_map = hyperparams.get("use_feature_map", False)

    # feature_map_dim: dimensionality of the learned feature mapping
    # WHY 64: provides sufficient capacity for moderate nonlinearity without overfitting
    feature_map_dim = hyperparams.get("feature_map_dim", 64)

    # ── Determine input/output dimensions ──
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # ── Select and initialize the model architecture ──
    if use_feature_map:
        # Use the kernel approximation model with neural feature mapping
        # WHY: for datasets that are not linearly separable, the feature map transforms
        # the data into a space where a linear boundary works
        model = KernelApproxSVMModel(n_features, n_classes, feature_map_dim).to(DEVICE)
    else:
        # Use the simple linear SVM model
        # WHY: for linearly separable data, this is more efficient and interpretable
        model = LinearSVMModel(n_features, n_classes).to(DEVICE)

    # ── Create the optimizer dynamically ──
    # getattr(torch.optim, optimizer_name) fetches the optimizer class by name string
    # WHY: this allows switching between SGD, Adam, AdamW via a hyperparameter
    # without writing separate code paths for each optimizer
    opt_cls = getattr(torch.optim, optimizer_name)

    # Instantiate the optimizer with weight_decay for L2 regularization
    # WHY weight_decay in optimizer: this is equivalent to adding (weight_decay/2)||w||^2
    # to the loss, which is the regularization term in the SVM objective
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Prepare training data as PyTorch tensors ──
    # Convert numpy arrays to float32 tensors (standard for neural network training)
    X_t = torch.tensor(X_train, dtype=torch.float32)

    if n_classes == 2:
        # ── Binary classification: map labels {0, 1} to {-1, +1} for hinge loss ──
        # WHY: the hinge loss formulation requires labels in {-1, +1};
        # the mapping 2*y - 1 converts 0 -> -1 and 1 -> +1
        y_svm = 2.0 * y_train - 1.0
        y_t = torch.tensor(y_svm, dtype=torch.float32)

        # Use our custom HingeLoss for binary classification
        # WHY: PyTorch's MultiMarginLoss expects class indices, not {-1, +1} labels
        criterion = HingeLoss()
    else:
        # ── Multiclass: use class indices with MultiMarginLoss ──
        # WHY: MultiMarginLoss implements the Weston-Watkins multiclass hinge loss,
        # which penalizes when the correct class score is not margin-separated from other classes
        y_t = torch.tensor(y_train, dtype=torch.long)
        criterion = nn.MultiMarginLoss(margin=1.0)

    # Create a DataLoader for efficient mini-batch iteration
    # WHY: DataLoader handles shuffling, batching, and (optionally) parallel data loading
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    # ── Training loop ──
    model.train()  # Set model to training mode (enables dropout, batch norm, etc.)
    for epoch in range(n_epochs):
        epoch_loss = 0.0  # Accumulate loss for logging

        for X_b, y_b in loader:
            # Move batch tensors to the compute device (CPU or GPU)
            # WHY: data must be on the same device as the model for computation
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

            # Forward pass: compute decision function values
            logits = model(X_b)

            # Compute the hinge loss, scaled by C
            # WHY: C controls the trade-off between margin maximization (weight_decay)
            # and loss minimization (hinge loss); the weight_decay handles regularization
            if n_classes == 2:
                loss = C * criterion(logits, y_b)
            else:
                loss = C * criterion(logits, y_b)

            # ── Backpropagation and weight update ──
            optimizer.zero_grad()  # Reset gradients from previous iteration
            loss.backward()        # Compute gradients via autograd
            optimizer.step()       # Update weights using the chosen optimizer

            # Accumulate weighted loss for epoch-level monitoring
            epoch_loss += loss.item() * X_b.size(0)

        # Log training progress at regular intervals (every 20% of epochs)
        # WHY: too-frequent logging slows training; too-infrequent misses convergence issues
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            avg_loss = epoch_loss / len(loader.dataset)
            logger.debug("Epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, avg_loss)

    logger.info(
        "SVM (PyTorch) trained: %d epochs, C=%.3f, lr=%.4f, feature_map=%s",
        n_epochs, C, lr, use_feature_map,
    )
    return model


@torch.no_grad()
def _evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics on a dataset.

    @torch.no_grad(): disables gradient tracking during evaluation
    WHY: saves memory and computation since we don't need gradients for inference

    WHY we compute pseudo-probabilities:
        - AUC-ROC requires continuous scores, not just class labels
        - Sigmoid/softmax converts decision values to [0, 1] pseudo-probabilities
        - These are NOT calibrated probabilities; for calibrated predictions,
          use Platt scaling or temperature scaling on a calibration set
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    # Convert input to PyTorch tensor and move to device
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # Compute model outputs (decision function values)
    logits = model(X_t).cpu().numpy()

    # Determine if binary or multiclass
    n_classes = len(np.unique(y))

    if n_classes == 2:
        # ── Binary: threshold decision function at 0 ──
        decisions = logits.squeeze()

        # Predict class 1 if decision >= 0, class 0 otherwise
        # WHY: the SVM decision boundary is at f(x) = 0
        y_pred = (decisions >= 0).astype(int)

        # Approximate probabilities via sigmoid of decision values
        # WHY: sigmoid maps decision values to (0, 1), providing a pseudo-probability;
        # this is Platt scaling without the calibration step
        p1 = 1.0 / (1.0 + np.exp(-decisions))
        y_proba = np.column_stack([1 - p1, p1])

        # AUC-ROC using the positive class probability
        auc = roc_auc_score(y, p1)
    else:
        # ── Multiclass: argmax over class scores ──
        y_pred = np.argmax(logits, axis=1)

        # Convert logits to probabilities via softmax
        # WHY: softmax normalizes K scores into a probability distribution;
        # subtracting max prevents numerical overflow in exp()
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Multiclass AUC-ROC using One-vs-Rest
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.

    WHY: validation metrics guide hyperparameter selection without
    touching the test set, preventing information leakage.
    """
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate on the held-out test set and print detailed diagnostics.

    WHY: the test set provides an unbiased estimate of how the model
    will perform on unseen data in production.
    """
    metrics = _evaluate(model, X_test, y_test)

    # Generate predictions for confusion matrix and classification report
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(X_t).cpu().numpy()

    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        y_pred = (logits.squeeze() >= 0).astype(int)
    else:
        y_pred = np.argmax(logits, axis=1)

    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    # Confusion matrix shows error patterns between predicted and actual classes
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Classification report shows per-class precision, recall, F1
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    WHY Optuna:
        - TPE sampler efficiently navigates the joint hyperparameter space
        - Handles mixed types (continuous, categorical, conditional)
        - Conditional parameters (feature_map_dim only when use_feature_map=True)
    """
    params = {
        # C: log-uniform because optimal C can range from 0.01 to 100
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),

        # learning_rate: log-uniform because LR is scale-sensitive
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),

        # n_epochs: step of 50 to avoid wasting trials on tiny differences
        "n_epochs": trial.suggest_int("n_epochs", 50, 500, step=50),

        # batch_size: standard power-of-2 choices
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),

        # weight_decay: L2 regularization strength (log-uniform scale)
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),

        # optimizer: choice of gradient-based optimizer
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"]),

        # use_feature_map: whether to use neural feature mapping
        "use_feature_map": trial.suggest_categorical("use_feature_map", [True, False]),
    }

    # Conditional parameter: feature_map_dim only relevant when using feature mapping
    # WHY: this avoids wasting search budget on irrelevant parameters
    if params["use_feature_map"]:
        params["feature_map_dim"] = trial.suggest_categorical("feature_map_dim", [32, 64, 128])

    model = train(X_train, y_train, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Distributed hyperparameter search using Ray Tune.

    WHY Ray Tune:
        - Parallelizes trials across multiple GPUs/machines
        - ASHA scheduler enables early stopping of poor trials
        - Integrates with PyTorch for GPU-aware resource management
    """
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "C": ray_tune.loguniform(0.01, 100.0),
        "learning_rate": ray_tune.loguniform(1e-4, 0.1),
        "n_epochs": ray_tune.choice([100, 200, 300]),
        "batch_size": ray_tune.choice([32, 64, 128]),
        "weight_decay": ray_tune.loguniform(1e-6, 0.1),
        "optimizer": ray_tune.choice(["SGD", "Adam", "AdamW"]),
        "use_feature_map": ray_tune.choice([True, False]),
        "feature_map_dim": ray_tune.choice([32, 64, 128]),
    }

    tuner = ray_tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=ray_tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="f1", mode="max")
    logger.info("Ray Tune best config: %s", best.config)
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Compare Parameter Sets
# ---------------------------------------------------------------------------

def compare_parameter_sets() -> None:
    """
    Train the PyTorch SVM with 4 different configurations and compare results.

    WHY this function exists:
        - Demonstrates the impact of model architecture (linear vs. feature map)
        - Shows how optimizer choice affects SVM training
        - Compares training speed vs. accuracy trade-offs
        - Provides intuition for selecting the right configuration
    """
    print("\n" + "=" * 85)
    print("COMPARE PARAMETER SETS - SVM PyTorch Implementation")
    print("=" * 85)

    # Generate a dataset for the comparison
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # ── Define 4 hyperparameter configurations ──
    configs = {
        # Config 1: Simple linear SVM with SGD
        # WHY: pure linear SVM with SGD is the closest to the traditional SVM formulation;
        # SGD provides direct control over the learning process but requires LR tuning
        "Linear + SGD": {
            "C": 1.0, "learning_rate": 0.01, "n_epochs": 200,
            "batch_size": 64, "weight_decay": 1e-3, "optimizer": "SGD",
            "use_feature_map": False,
        },

        # Config 2: Linear SVM with Adam optimizer
        # WHY: Adam's adaptive learning rate often converges faster than SGD;
        # it adjusts per-parameter learning rates based on gradient history,
        # which helps when features have different scales
        "Linear + Adam": {
            "C": 1.0, "learning_rate": 0.001, "n_epochs": 200,
            "batch_size": 64, "weight_decay": 1e-3, "optimizer": "Adam",
            "use_feature_map": False,
        },

        # Config 3: Kernel approximation with small feature map
        # WHY: the neural feature map adds nonlinear capacity to handle data
        # that isn't linearly separable; 32 dimensions keeps it lightweight
        "Feature Map (dim=32)": {
            "C": 1.0, "learning_rate": 0.001, "n_epochs": 200,
            "batch_size": 64, "weight_decay": 1e-3, "optimizer": "Adam",
            "use_feature_map": True, "feature_map_dim": 32,
        },

        # Config 4: Kernel approximation with large feature map
        # WHY: larger feature map (128 dims) provides more expressive power
        # but risks overfitting on small datasets; good for complex nonlinear boundaries
        "Feature Map (dim=128)": {
            "C": 1.0, "learning_rate": 0.001, "n_epochs": 200,
            "batch_size": 64, "weight_decay": 1e-3, "optimizer": "Adam",
            "use_feature_map": True, "feature_map_dim": 128,
        },
    }

    # ── Train each configuration and collect metrics ──
    results = {}
    for name, params in configs.items():
        print(f"\nTraining config: {name}...")
        model = train(X_train, y_train, **params)
        val_metrics = validate(model, X_val, y_val)
        test_metrics = _evaluate(model, X_test, y_test)

        # Count model parameters to compare complexity
        # WHY: model complexity correlates with overfitting risk and inference speed
        n_params = sum(p.numel() for p in model.parameters())

        results[name] = {
            "val_f1": val_metrics["f1"],
            "val_acc": val_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_acc": test_metrics["accuracy"],
            "test_auc": test_metrics["auc_roc"],
            "n_params": n_params,
        }

    # ── Print the formatted comparison table ──
    print("\n" + "-" * 110)
    print(f"{'Config':<25} {'Val F1':>8} {'Val Acc':>8} {'Test F1':>8} "
          f"{'Test Acc':>8} {'Test AUC':>9} {'Params':>8}")
    print("-" * 110)
    for name, m in results.items():
        print(f"{name:<25} {m['val_f1']:>8.4f} {m['val_acc']:>8.4f} {m['test_f1']:>8.4f} "
              f"{m['test_acc']:>8.4f} {m['test_auc']:>9.4f} {m['n_params']:>8}")
    print("-" * 110)

    # ── Interpretation guide ──
    print("\nInterpretation Guide:")
    print("  - Linear + SGD: Closest to traditional SVM. Simple but requires careful LR tuning.")
    print("  - Linear + Adam: Adaptive LR often converges faster. Good default for linear SVM.")
    print("  - Feature Map (dim=32): Adds nonlinear capacity with minimal overhead.")
    print("    Best when data is not linearly separable but the boundary is simple.")
    print("  - Feature Map (dim=128): Maximum nonlinear capacity for complex boundaries.")
    print("    Watch for overfitting on small datasets (compare val vs test metrics).")
    print("  - Compare Params column: more parameters = more capacity but slower inference.")
    print("  - If Linear and Feature Map models perform similarly, prefer Linear (simpler).")


# ---------------------------------------------------------------------------
# Real-World Demo: Email Spam Detection
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """
    Demonstrate the PyTorch SVM on a realistic email spam detection task.

    WHY email spam detection for SVM:
        - SVMs were the dominant approach for spam filtering from ~2000-2010
        - Linear SVMs excel on high-dimensional text features (word frequencies)
        - PyTorch SVM enables GPU acceleration for processing millions of emails
        - The hinge loss margin provides natural confidence scores for spam scoring
        - Production deployment via ONNX export or TorchServe integration

    Domain context:
        Email providers classify incoming messages as spam or ham (legitimate).
        Features include word frequencies, character statistics, and formatting patterns.
        False positives (blocking legitimate email) are typically worse than false negatives
        (letting spam through), so precision is often prioritized in production systems.
    """
    print("\n" + "=" * 85)
    print("REAL-WORLD DEMO: Email Spam Detection (SVM PyTorch)")
    print("=" * 85)

    # ── Generate synthetic email spam detection data ──
    np.random.seed(42)
    n_samples = 1200

    # Feature names mirror the UCI Spambase dataset features
    feature_names = [
        "word_freq_free",       # Frequency of "free" (strong spam signal)
        "word_freq_money",      # Frequency of "money" (financial spam)
        "word_freq_offer",      # Frequency of "offer" (promotional spam)
        "word_freq_click",      # Frequency of "click" (phishing/clickbait)
        "capital_run_avg",      # Average length of capital letter runs
        "capital_run_max",      # Maximum length of capital letter runs
        "char_freq_exclaim",    # Frequency of '!' (spam uses lots of !!!)
        "char_freq_dollar",     # Frequency of '$' (financial spam indicator)
    ]

    # ── Simulate spam emails (class 1) ──
    n_spam = 480
    spam_features = np.column_stack([
        np.random.exponential(0.8, n_spam),    # word_freq_free: high in spam
        np.random.exponential(0.6, n_spam),    # word_freq_money: high in spam
        np.random.exponential(0.7, n_spam),    # word_freq_offer: high in spam
        np.random.exponential(0.5, n_spam),    # word_freq_click: high in spam
        np.random.exponential(3.0, n_spam),    # capital_run_avg: LOTS OF CAPS IN SPAM
        np.random.exponential(15.0, n_spam),   # capital_run_max: LOOOONG CAPITAL RUNS
        np.random.exponential(0.4, n_spam),    # char_freq_exclaim: excessive !!!
        np.random.exponential(0.3, n_spam),    # char_freq_dollar: $$$ in financial spam
    ])

    # ── Simulate legitimate (ham) emails (class 0) ──
    n_ham = n_samples - n_spam
    ham_features = np.column_stack([
        np.random.exponential(0.1, n_ham),     # word_freq_free: rare in normal email
        np.random.exponential(0.05, n_ham),    # word_freq_money: very rare
        np.random.exponential(0.08, n_ham),    # word_freq_offer: occasional
        np.random.exponential(0.1, n_ham),     # word_freq_click: rare
        np.random.exponential(1.0, n_ham),     # capital_run_avg: normal sentence caps
        np.random.exponential(4.0, n_ham),     # capital_run_max: short caps (names)
        np.random.exponential(0.05, n_ham),    # char_freq_exclaim: minimal use
        np.random.exponential(0.02, n_ham),    # char_freq_dollar: very rare
    ])

    # Combine and shuffle
    X = np.vstack([spam_features, ham_features])
    y = np.array([1] * n_spam + [0] * n_ham)
    shuffle_idx = np.random.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Split with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp,
    )

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nDataset: {n_samples} emails ({n_spam} spam, {n_ham} ham)")
    print(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Device: {DEVICE}")

    # ── Train both linear and feature-map models ──
    print("\n--- Training Linear SVM (PyTorch) ---")
    linear_model = train(X_train, y_train, C=1.0, learning_rate=0.001,
                         n_epochs=200, optimizer="Adam", use_feature_map=False)
    linear_test = _evaluate(linear_model, X_test, y_test)

    print("\n--- Training Feature-Map SVM (PyTorch) ---")
    fm_model = train(X_train, y_train, C=1.0, learning_rate=0.001,
                     n_epochs=200, optimizer="Adam", use_feature_map=True, feature_map_dim=32)
    fm_test = _evaluate(fm_model, X_test, y_test)

    # ── Show comparison of both models ──
    print(f"\nModel Comparison:")
    print(f"  {'Metric':<15} {'Linear SVM':>12} {'Feature Map SVM':>16}")
    print(f"  {'-'*45}")
    for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        print(f"  {metric:<15} {linear_test[metric]:>12.4f} {fm_test[metric]:>16.4f}")

    # ── Pick the best model and show detailed results ──
    best_model = linear_model if linear_test["f1"] >= fm_test["f1"] else fm_model
    best_name = "Linear" if linear_test["f1"] >= fm_test["f1"] else "Feature Map"
    best_metrics = linear_test if linear_test["f1"] >= fm_test["f1"] else fm_test

    # Generate predictions from the best model
    best_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = best_model(X_t).cpu().numpy()
    y_pred = (logits.squeeze() >= 0).astype(int)

    # Display confusion matrix with domain labels
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nBest Model: {best_name} SVM")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Ham  Predicted Spam")
    print(f"  Actual Ham     {cm[0][0]:>12}  {cm[0][1]:>14}")
    print(f"  Actual Spam    {cm[1][0]:>12}  {cm[1][1]:>14}")

    # ── Analyze linear model weights if applicable ──
    if isinstance(best_model, LinearSVMModel):
        weights = best_model.linear.weight.data.cpu().numpy().squeeze()
        print(f"\nLinear SVM Feature Weights (positive = spam indicator):")
        weight_order = np.argsort(np.abs(weights))[::-1]
        for rank, idx in enumerate(weight_order, 1):
            direction = "SPAM" if weights[idx] > 0 else "HAM"
            print(f"  {rank}. {feature_names[idx]:<22} weight={weights[idx]:>+8.4f}  ({direction} indicator)")

    # ── PyTorch deployment options ──
    print(f"\nPyTorch Deployment Options:")
    print(f"  1. TorchScript: torch.jit.script(model) for C++ inference")
    print(f"  2. ONNX Export: torch.onnx.export(model, ...) for cross-framework deployment")
    print(f"  3. TorchServe: package model for REST API serving")
    print(f"  4. Mobile: torch.utils.mobile_optimizer for on-device spam filtering")

    # ── Business insights ──
    print(f"\nBusiness Insights for Email Spam Detection:")
    print(f"  - False Positives (ham as spam): {cm[0][1]} legitimate emails blocked")
    print(f"    Impact: missed business emails, customer complaints")
    print(f"  - False Negatives (spam as ham): {cm[1][0]} spam delivered")
    print(f"    Impact: user annoyance, phishing risk, inbox clutter")
    print(f"  - PyTorch SVM enables GPU-accelerated batch processing of incoming emails")
    print(f"  - The model processes email features in microseconds for real-time filtering")
    print(f"  - Retrain periodically as spam techniques evolve (concept drift)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point: runs baseline training, hyperparameter optimization, testing,
    parameter comparison, and real-world demonstration.
    """
    logger.info("=" * 70)
    logger.info("Support Vector Machine - PyTorch Implementation")
    logger.info("=" * 70)

    # Generate the standard synthetic dataset
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # ── Baseline training: Linear SVM ──
    logger.info("\n--- Baseline Training (Linear SVM) ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # ── Baseline training: Kernel Approximation SVM ──
    logger.info("\n--- Baseline Training (Kernel Approx SVM) ---")
    model_fm = train(X_train, y_train, use_feature_map=True, feature_map_dim=64)
    validate(model_fm, X_val, y_val)

    # ── Hyperparameter optimization with Optuna ──
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=20,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Retrain with the best hyperparameters
    best_model = train(X_train, y_train, **study.best_params)

    # ── Final test evaluation ──
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # ── Compare parameter configurations ──
    compare_parameter_sets()

    # ── Run the real-world email spam detection demo ──
    real_world_demo()


if __name__ == "__main__":
    main()
