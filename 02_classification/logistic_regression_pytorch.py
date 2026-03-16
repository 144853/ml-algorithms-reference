"""
Logistic Regression - PyTorch Implementation
==============================================

Theory & Mathematics:
    This module implements logistic regression as a single-layer neural network
    in PyTorch. The model learns a linear mapping followed by a sigmoid activation:

        z = X @ W + b
        P(y=1|x) = sigmoid(z)

    Training uses BCEWithLogitsLoss, which combines the sigmoid and binary
    cross-entropy into a single numerically stable operation:

        loss = -1/N sum [y*log(sigma(z)) + (1-y)*log(1-sigma(z))]

    For multiclass classification, we use CrossEntropyLoss (softmax + NLL):

        loss = -1/N sum_i sum_k [y_{ik} * log(softmax(z_i)_k)]

    PyTorch advantages:
        - Automatic differentiation (autograd) for gradient computation
        - GPU acceleration for larger datasets
        - Seamless integration with deep learning pipelines
        - Built-in optimizers (SGD, Adam, AdamW, etc.)

    Weight decay in the optimizer serves as L2 regularization:
        L_reg = L + (lambda/2) * ||W||^2

Business Use Cases:
    - Real-time scoring APIs requiring GPU acceleration
    - Transfer learning pipelines where logistic regression is the final layer
    - Rapid prototyping before scaling to deeper models
    - Online learning with streaming data
    - Edge deployment via TorchScript / ONNX export

Advantages:
    - GPU-accelerated training for large-scale datasets
    - Native integration with PyTorch ecosystem (TorchServe, ONNX)
    - Autograd removes need for manual gradient derivation
    - Easy to extend to deeper architectures
    - Built-in learning rate schedulers

Disadvantages:
    - Heavier dependency than scikit-learn for a simple model
    - More boilerplate code (DataLoader, training loop)
    - Overkill for small datasets with few features
    - Debugging can be harder than NumPy

Hyperparameters:
    - learning_rate: Optimizer step size
    - n_epochs: Number of training epochs
    - batch_size: Mini-batch size for DataLoader
    - weight_decay: L2 regularization (via optimizer)
    - optimizer: Choice of optimizer (SGD, Adam, AdamW)
"""

# --- Standard library imports ---
# logging: structured output with severity levels for experiment tracking.
import logging
# warnings: suppress PyTorch/sklearn convergence warnings during hyperparameter search.
import warnings
# typing: type annotations for function signatures.
from typing import Any, Dict, Tuple

# --- Third-party imports ---
# numpy: used for data manipulation and metric computation (sklearn metrics expect numpy).
import numpy as np
# optuna: Bayesian hyperparameter optimization framework.
# WHY: Optuna's TPE sampler intelligently navigates the hyperparameter space.
import optuna

# torch: PyTorch deep learning framework.
# WHY PyTorch for logistic regression: While sklearn is simpler, PyTorch offers:
# 1. GPU acceleration for large datasets (millions of samples)
# 2. Automatic differentiation (no manual gradient derivation)
# 3. Easy extension to deeper architectures (just add layers)
# 4. Production deployment via TorchScript/ONNX
import torch
# torch.nn: neural network building blocks (layers, loss functions).
import torch.nn as nn
# DataLoader/TensorDataset: efficient batched data loading with shuffling.
# WHY: DataLoader handles batching, shuffling, and optional multi-process loading,
# which is essential for efficient GPU utilization.
from torch.utils.data import DataLoader, TensorDataset

# sklearn imports for data generation, preprocessing, and evaluation only.
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,           # Overall fraction of correct predictions.
    classification_report,    # Per-class precision/recall/F1 summary.
    confusion_matrix,         # Matrix showing TP/TN/FP/FN counts.
    f1_score,                 # Harmonic mean of precision and recall.
    precision_score,          # Fraction of positive predictions that are correct.
    recall_score,             # Fraction of actual positives that we found.
    roc_auc_score,            # Area under the ROC curve.
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# --- Device selection ---
# Automatically use GPU if available, otherwise fall back to CPU.
# WHY: GPU can train models 10-100x faster than CPU for large datasets.
# For logistic regression with small data, CPU is fast enough.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class LogisticRegressionModel(nn.Module):
    """Single linear layer logistic regression in PyTorch.

    WHY nn.Module: This is the base class for all PyTorch models. Subclassing it
    gives us automatic parameter tracking, GPU movement, and integration with
    optimizers and serialization (save/load).

    Architecture: Input (n_features) -> Linear -> Output (1 for binary, K for multiclass)
    The sigmoid/softmax is applied in the loss function, not in the forward pass.
    WHY: BCEWithLogitsLoss and CrossEntropyLoss apply sigmoid/softmax internally
    for better numerical stability (log-sum-exp trick).
    """

    def __init__(self, n_features: int, n_classes: int = 2) -> None:
        """Initialize the model with a single linear layer.

        Args:
            n_features: Number of input features.
            n_classes: Number of target classes. 2 for binary classification.
        """
        # Call the parent nn.Module constructor (required for PyTorch).
        super().__init__()
        # For binary classification, we only need 1 output neuron (sigmoid applied externally).
        # WHY: With BCEWithLogitsLoss, a single output represents log-odds.
        # For multiclass, we need K outputs (one per class, softmax applied externally).
        out_dim = 1 if n_classes == 2 else n_classes
        # nn.Linear implements y = x @ W^T + b, learning both weight matrix W and bias b.
        # WHY Linear: Logistic regression IS a linear model. The nonlinearity comes from
        # the sigmoid/softmax applied in the loss function.
        self.linear = nn.Linear(n_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute raw logits (no activation).

        WHY raw logits: The loss function (BCEWithLogitsLoss or CrossEntropyLoss)
        applies the sigmoid/softmax internally. This is more numerically stable
        than applying sigmoid first and then computing log, because it avoids
        computing log(sigmoid(x)) which can be numerically unstable for large |x|.
        """
        return self.linear(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data with train/val/test splits.

    WHY identical data generation: Same data as sklearn/numpy versions for fair comparison.
    """
    # Generate synthetic classification data with controlled signal-to-noise ratio.
    X, y = make_classification(
        n_samples=n_samples,            # Total samples.
        n_features=n_features,          # Feature dimensionality.
        n_informative=n_features // 2,  # Features that carry real signal.
        n_redundant=n_features // 4,    # Linear combinations of informative features.
        n_classes=n_classes,            # Number of target classes.
        random_state=random_state,      # Reproducibility.
    )

    # 60/20/20 stratified split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    # Standardize features (critical for gradient-based optimization in PyTorch).
    # WHY: PyTorch optimizers assume features are roughly on the same scale.
    # Adam is more robust to scale differences than SGD, but scaling still helps.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Convert numpy arrays to a PyTorch DataLoader for batched training.

    WHY DataLoader: It handles mini-batch creation, shuffling, and can optionally
    use multiple worker processes for data loading. This is essential for efficient
    GPU utilization -- while the GPU processes one batch, the CPU can prepare the next.
    """
    # Convert numpy arrays to PyTorch tensors.
    # WHY float32: PyTorch's default precision. float64 is slower on GPU with no benefit
    # for ML (the noise in the data is much larger than float32 precision).
    X_t = torch.tensor(X, dtype=torch.float32)
    # Label dtype depends on the loss function:
    # - BCEWithLogitsLoss expects float32 labels (for binary cross-entropy).
    # - CrossEntropyLoss expects long (int64) labels (class indices).
    y_t = torch.tensor(y, dtype=torch.float32 if len(np.unique(y)) == 2 else torch.long)
    # Create a DataLoader that yields (X_batch, y_batch) tuples.
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **hyperparams: Any,
) -> LogisticRegressionModel:
    """Train PyTorch logistic regression model.

    WHY this function signature: Same interface as sklearn/numpy versions,
    accepting numpy arrays and returning a trained model. The PyTorch-specific
    details (tensors, DataLoader, optimizer) are handled internally.
    """
    # Extract hyperparameters with sensible defaults.
    lr = hyperparams.get("learning_rate", 0.01)
    # WHY 0.01: A moderate learning rate that works well with Adam optimizer.
    # Adam adapts per-parameter learning rates, so the initial lr is less critical.
    n_epochs = hyperparams.get("n_epochs", 200)
    # WHY 200: PyTorch with Adam converges faster than NumPy SGD, needing fewer epochs.
    batch_size = hyperparams.get("batch_size", 64)
    # WHY 64: Balances GPU utilization with gradient noise. Smaller batches add
    # regularizing noise; larger batches give smoother gradients.
    weight_decay = hyperparams.get("weight_decay", 1e-4)
    # WHY 1e-4: weight_decay in PyTorch optimizers implements L2 regularization.
    # This is equivalent to lambda_reg in the NumPy version. 1e-4 is mild.
    optimizer_name = hyperparams.get("optimizer", "Adam")
    # WHY Adam: Adaptive learning rate per parameter. More robust than SGD to
    # learning rate choice and feature scaling. Good default for most problems.

    # Determine dataset characteristics.
    n_features = X_train.shape[1]                  # Input dimensionality.
    n_classes = len(np.unique(y_train))            # Number of target classes.

    # Instantiate the model and move it to the appropriate device (CPU or GPU).
    # WHY .to(DEVICE): This moves all model parameters to GPU memory if available.
    model = LogisticRegressionModel(n_features, n_classes).to(DEVICE)

    # Select the appropriate loss function based on number of classes.
    if n_classes == 2:
        # BCEWithLogitsLoss: combines sigmoid + binary cross-entropy.
        # WHY combined: Numerically stable. Computing sigmoid(x) then log can underflow
        # for large negative x. The combined function uses the log-sum-exp trick.
        criterion = nn.BCEWithLogitsLoss()
    else:
        # CrossEntropyLoss: combines softmax + negative log-likelihood.
        # WHY: Same numerical stability benefits as BCEWithLogitsLoss, but for multiclass.
        criterion = nn.CrossEntropyLoss()

    # Create the optimizer by dynamically looking up the class from torch.optim.
    # WHY getattr: Allows the optimizer to be specified as a string ("Adam", "SGD", "AdamW")
    # without a large if/elif chain. This makes it easy to add new optimizers.
    opt_cls = getattr(torch.optim, optimizer_name)
    # weight_decay parameter implements L2 regularization directly in the optimizer update.
    # WHY in optimizer: This is equivalent to adding (weight_decay/2)*||W||^2 to the loss,
    # but more efficient because it's applied during the parameter update step.
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create a DataLoader for batched training with shuffling.
    loader = _make_loader(X_train, y_train, batch_size)

    # Set the model to training mode.
    # WHY: model.train() enables dropout and batch normalization training behavior.
    # For logistic regression (no dropout/BN), this is a no-op but good practice.
    model.train()

    # --- Training loop ---
    for epoch in range(n_epochs):
        epoch_loss = 0.0  # Accumulate loss over all batches for epoch-level reporting.

        for X_b, y_b in loader:
            # Move batch data to the same device as the model (CPU or GPU).
            # WHY: Data and model must be on the same device for computation.
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

            # Forward pass: compute raw logits (no sigmoid/softmax applied).
            logits = model(X_b)

            # Compute the loss based on the number of classes.
            if n_classes == 2:
                # Binary: squeeze logits from (batch, 1) to (batch,) to match label shape.
                # WHY squeeze: BCEWithLogitsLoss expects matching shapes for output and target.
                loss = criterion(logits.squeeze(), y_b)
            else:
                # Multiclass: CrossEntropyLoss expects (batch, n_classes) logits and (batch,) int labels.
                loss = criterion(logits, y_b.long())

            # --- Backward pass and parameter update ---
            # Zero out gradients from the previous step.
            # WHY: PyTorch accumulates gradients by default (useful for gradient accumulation
            # across multiple mini-batches). For standard training, we clear them each step.
            optimizer.zero_grad()

            # Compute gradients via backpropagation (automatic differentiation).
            # WHY: This is the core advantage of PyTorch -- autograd computes all gradients
            # automatically, no matter how complex the model architecture.
            loss.backward()

            # Update model parameters using the computed gradients.
            # WHY: The optimizer applies the update rule (e.g., Adam's adaptive moment estimation)
            # to move parameters in the direction that reduces the loss.
            optimizer.step()

            # Accumulate batch loss for epoch-level reporting.
            # WHY multiply by batch size: loss.item() is the MEAN loss for this batch.
            # We want the total loss to compute a proper average over the entire epoch.
            epoch_loss += loss.item() * X_b.size(0)

        # Log progress at regular intervals (every 20% of epochs).
        # WHY not every epoch: Logging every epoch creates too much output for 200+ epochs.
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            avg_loss = epoch_loss / len(loader.dataset)
            logger.debug("Epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, avg_loss)

    logger.info(
        "Model trained (%d epochs, lr=%.4f, wd=%.5f, opt=%s)",
        n_epochs, lr, weight_decay, optimizer_name,
    )
    return model


@torch.no_grad()
def _evaluate(model: LogisticRegressionModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate the model and compute classification metrics.

    WHY @torch.no_grad(): Disables gradient computation during evaluation.
    This saves memory (no gradient tensors stored) and speeds up computation.
    Gradients are only needed during training for backpropagation.
    """
    # Switch to evaluation mode (disables dropout, uses running stats for BN).
    model.eval()

    # Convert numpy data to tensor and move to the model's device.
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    # Get raw logits from the model, move back to CPU for numpy conversion.
    # WHY cpu().numpy(): sklearn metrics expect numpy arrays, not PyTorch tensors.
    logits = model(X_t).cpu().numpy()

    n_classes = len(np.unique(y))
    if n_classes == 2:
        # Binary: apply sigmoid to logits to get probabilities.
        # WHY manual sigmoid: We need probabilities for AUC-ROC and predictions.
        # The model outputs raw logits (not probabilities) for numerical stability.
        proba_1 = 1.0 / (1.0 + np.exp(-logits.squeeze()))
        # Threshold at 0.5 to get class predictions.
        y_pred = (proba_1 >= 0.5).astype(int)
        # Stack probabilities for both classes: [P(y=0), P(y=1)].
        y_proba = np.column_stack([1 - proba_1, proba_1])
        # AUC-ROC using positive class probability.
        auc = roc_auc_score(y, proba_1)
    else:
        # Multiclass: apply softmax to logits to get probabilities.
        # WHY subtract max: Numerical stability trick for softmax.
        # exp(x - max(x)) prevents overflow while giving the same result.
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Predict the class with highest probability.
        y_pred = np.argmax(y_proba, axis=1)
        # Multiclass AUC using one-vs-rest with weighted averaging.
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(model: LogisticRegressionModel, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Validate on the validation set for hyperparameter tuning."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(model: LogisticRegressionModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Final test evaluation -- call only once with the best model."""
    metrics = _evaluate(model, X_test, y_test)

    # Get predictions for confusion matrix and classification report.
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(X_t).cpu().numpy()
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        y_pred = (1.0 / (1.0 + np.exp(-logits.squeeze())) >= 0.5).astype(int)
    else:
        y_pred = np.argmax(logits, axis=1)

    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """Compare multiple optimizer and hyperparameter configurations in PyTorch.

    WHY: PyTorch offers multiple optimizers (SGD, Adam, AdamW), each with different
    convergence properties. Understanding these differences is critical for efficient
    deep learning practice, even with a simple logistic regression model.
    """
    configs = {
        "SGD (basic)": {
            # Config 1: Vanilla Stochastic Gradient Descent.
            # WHY: SGD is the simplest optimizer. It uses a fixed learning rate for all
            # parameters. It requires careful learning rate tuning but can generalize
            # better than adaptive methods on some problems.
            # Best for: well-scaled features, when you want tight control over convergence.
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "n_epochs": 200,
            "weight_decay": 1e-4,
            "batch_size": 64,
        },
        "Adam (adaptive)": {
            # Config 2: Adam optimizer with adaptive per-parameter learning rates.
            # WHY: Adam maintains running averages of gradients (momentum) and squared
            # gradients (RMSprop), adapting the learning rate for each parameter.
            # This makes it robust to learning rate choice and feature scaling.
            # Best for: general-purpose default, especially with poorly scaled features.
            "learning_rate": 0.01,
            "optimizer": "Adam",
            "n_epochs": 200,
            "weight_decay": 1e-4,
            "batch_size": 64,
        },
        "AdamW (decoupled WD)": {
            # Config 3: AdamW with decoupled weight decay.
            # WHY: Standard Adam applies weight decay to the adaptive moment, which is
            # theoretically incorrect. AdamW decouples weight decay from the gradient update,
            # implementing true L2 regularization. This was shown to improve generalization
            # in the paper "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019).
            # Best for: when regularization is important and you want "correct" L2 behavior.
            "learning_rate": 0.01,
            "optimizer": "AdamW",
            "n_epochs": 200,
            "weight_decay": 0.01,       # Stronger WD because AdamW handles it better.
            "batch_size": 64,
        },
        "Adam (high WD)": {
            # Config 4: Adam with strong weight decay (heavy regularization).
            # WHY: Tests the effect of strong L2 regularization through the optimizer.
            # With weight_decay=0.1, the model is strongly penalized for large weights,
            # producing a simpler decision boundary that may underfit complex patterns.
            # Best for: preventing overfitting on small or noisy datasets.
            "learning_rate": 0.01,
            "optimizer": "Adam",
            "n_epochs": 200,
            "weight_decay": 0.1,
            "batch_size": 64,
        },
    }

    print("\n" + "=" * 90)
    print("LOGISTIC REGRESSION (PyTorch) - HYPERPARAMETER COMPARISON")
    print("=" * 90)
    print(f"{'Config':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 90)

    for name, params in configs.items():
        model = train(X_train, y_train, **params)
        metrics = _evaluate(model, X_val, y_val)
        print(
            f"{name:<30} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f}"
        )

    print("-" * 90)
    print("INTERPRETATION GUIDE:")
    print("  - SGD: Simplest optimizer. Needs more tuning but can generalize better.")
    print("  - Adam: Best default. Adaptive LR makes it robust to hyperparameter choices.")
    print("  - AdamW: Theoretically correct weight decay. Preferred for regularized models.")
    print("  - High WD: Tests the effect of strong regularization on model simplicity.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Real-World Demo: Customer Churn Prediction (PyTorch)
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate PyTorch logistic regression on customer churn prediction.

    WHY PyTorch for this scenario: In production, you might serve this model as a
    TorchScript module via TorchServe, or export it to ONNX for cross-platform
    deployment. The PyTorch ecosystem enables seamless transition from prototyping
    to production serving.
    """
    print("\n" + "=" * 90)
    print("REAL-WORLD DEMO: Customer Churn Prediction (PyTorch)")
    print("=" * 90)

    # Set seeds for reproducibility across both numpy and PyTorch.
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 2000

    # Generate the same realistic features as sklearn/numpy versions.
    tenure_months = np.random.uniform(0, 72, n_samples)
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 120)
    total_charges = tenure_months * monthly_charges + np.random.normal(0, 200, n_samples)
    contract_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    num_support_tickets = np.random.poisson(2, n_samples)
    has_online_backup = np.random.binomial(1, 0.4, n_samples)

    # Generate churn labels with the same business logic.
    churn_score = (
        -0.05 * tenure_months + 0.03 * monthly_charges
        - 0.8 * contract_type + 0.3 * num_support_tickets
        - 0.5 * has_online_backup + np.random.normal(0, 1, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    y = (np.random.random(n_samples) < churn_prob).astype(int)

    X = np.column_stack([
        tenure_months, monthly_charges, total_charges,
        contract_type, num_support_tickets, has_online_backup,
    ])
    feature_names = [
        "tenure_months", "monthly_charges", "total_charges",
        "contract_type", "num_support_tickets", "has_online_backup",
    ]

    print(f"\nDataset: {n_samples} customers, {len(feature_names)} features")
    print(f"Churn rate: {y.mean():.1%}")
    print(f"Device: {DEVICE}")

    # Split and scale.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the PyTorch model.
    # WHY these hyperparameters: Adam optimizer with moderate weight decay converges
    # quickly for this small dataset. 150 epochs is more than enough.
    model = LogisticRegressionModel(n_features=6, n_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Convert to tensors.
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    # Training loop.
    model.train()
    for epoch in range(150):
        logits = model(X_train_t)
        loss = criterion(logits.squeeze(), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on test set.
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(X_test_t).cpu().numpy().squeeze()
    proba_1 = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (proba_1 >= 0.5).astype(int)

    print("\n--- Model Performance (PyTorch) ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_test, proba_1):.4f}")

    # Feature importance from learned weights.
    weights = model.linear.weight.data.cpu().numpy().flatten()
    print("\n--- Learned Weights (Feature Importance) ---")
    sorted_idx = np.argsort(np.abs(weights))[::-1]
    for i in sorted_idx:
        direction = "increases churn" if weights[i] > 0 else "decreases churn"
        print(f"  {feature_names[i]:<25} weight={weights[i]:>8.4f}  ({direction})")

    print("\n--- PyTorch Deployment Options ---")
    print("  1. TorchScript: model_scripted = torch.jit.script(model)")
    print("  2. ONNX Export: torch.onnx.export(model, dummy_input, 'churn_model.onnx')")
    print("  3. TorchServe: package as .mar file for REST API serving")
    print("=" * 90 + "\n")


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
    """Optuna objective for PyTorch logistic regression.

    WHY these search ranges: Each range is based on practical experience with
    PyTorch training dynamics.
    """
    lr = trial.suggest_float("learning_rate", 1e-4, 0.5, log=True)
    n_epochs = trial.suggest_int("n_epochs", 50, 500, step=50)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

    model = train(
        X_train, y_train,
        learning_rate=lr, n_epochs=n_epochs, batch_size=batch_size,
        weight_decay=weight_decay, optimizer=optimizer,
    )
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Ray Tune distributed hyperparameter search."""
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "learning_rate": ray_tune.loguniform(1e-4, 0.5),
        "n_epochs": ray_tune.choice([100, 200, 300]),
        "batch_size": ray_tune.choice([32, 64, 128]),
        "weight_decay": ray_tune.loguniform(1e-6, 0.1),
        "optimizer": ray_tune.choice(["SGD", "Adam", "AdamW"]),
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Logistic Regression (PyTorch) pipeline."""
    logger.info("=" * 70)
    logger.info("Logistic Regression - PyTorch Implementation")
    logger.info("=" * 70)

    # Step 1: Generate data.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Baseline training.
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train)
    validate(model, X_val, y_val)

    # Step 3: Compare optimizer/hyperparameter configurations.
    logger.info("\n--- Hyperparameter Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 4: Real-world demo.
    real_world_demo()

    # Step 5: Optuna optimization.
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=20,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Step 6: Retrain with best params.
    best_model = train(X_train, y_train, **study.best_params)

    # Step 7: Final test evaluation.
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
