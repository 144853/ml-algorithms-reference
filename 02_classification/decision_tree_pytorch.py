"""
Soft Decision Tree Classifier - PyTorch Implementation
=======================================================

Theory & Mathematics:
    A Soft Decision Tree replaces the hard (binary) routing decisions of a
    traditional decision tree with soft (probabilistic) routing using sigmoid
    functions. This makes the entire tree differentiable, enabling end-to-end
    training with gradient descent (backpropagation).

    Traditional vs Soft Decision Tree:
        Traditional: At node i, if x_j <= threshold -> go LEFT, else go RIGHT.
            This is a hard, non-differentiable step function.
        Soft: At node i, probability of going LEFT = sigma(w_i^T x + b_i)
            where sigma is the sigmoid function. The sample goes to BOTH
            children with complementary probabilities.

    Mathematical Formulation:
        For a tree of depth D with 2^D - 1 internal nodes and 2^D leaves:

        1. At each internal node i, compute the routing probability:
            p_i(x) = sigma((w_i^T x + b_i) / tau)
            where:
                w_i = learnable weight vector (same dim as input features)
                b_i = learnable bias scalar
                tau = temperature parameter (controls softness of routing)

        2. The probability of reaching leaf l is the product of routing
           probabilities along the path from root to leaf l:
            P(leaf_l | x) = prod_{i in path(l)} [p_i(x) if left, (1-p_i(x)) if right]

        3. Each leaf l has a learnable class distribution pi_l (softmax output).

        4. The final prediction is a mixture of all leaf distributions:
            P(y | x) = sum_{l=1}^{2^D} P(leaf_l | x) * pi_l(y)

        5. Training minimizes cross-entropy loss via gradient descent.

    Temperature Parameter (tau):
        - tau -> 0: soft tree approaches hard tree (sharp sigmoid)
        - tau -> inf: all routing probabilities approach 0.5 (uniform routing)
        - tau = 1.0: standard sigmoid behavior
        WHY: Temperature annealing can be used during training to gradually
        transition from soft to hard decisions.

    Key Innovation:
        By making routing differentiable, we can train the entire tree
        end-to-end with standard neural network optimizers (SGD, Adam).
        This eliminates the need for greedy splitting and allows the tree
        to find globally better split parameters.

Business Use Cases:
    - Medical diagnosis: combines tree interpretability with neural network power
    - Financial modeling: hierarchical decision structures with learned features
    - Autonomous systems: differentiable decision-making for gradient-based planning
    - Ensemble learning: soft trees as base learners in differentiable ensembles

Advantages:
    - End-to-end differentiable: can be trained with backpropagation
    - Oblique splits: w^T x allows non-axis-aligned decision boundaries
    - Soft routing: smoother decision boundaries, better generalization
    - Compatible with deep learning pipelines and GPU acceleration
    - No greedy splitting: can learn globally better tree structures

Disadvantages:
    - Less interpretable than hard decision trees
    - Requires more compute (forward pass through all paths)
    - Sensitive to temperature parameter and initialization
    - Can be harder to train than standard neural networks
    - Memory grows exponentially with tree depth (2^D leaves)

References:
    - Frosst & Hinton (2017): "Distilling a Neural Network Into a Soft Decision Tree"
    - Kontschieder et al. (2015): "Deep Neural Decision Forests"
    - Tanno et al. (2019): "Adaptive Neural Trees"
"""

# --- Standard library imports ---
# logging: provides structured output messages with severity levels.
# WHY: Structured logging is superior to print() for tracking training progress.
import logging  # Standard library module for structured logging

# warnings: used to suppress noisy convergence/deprecation warnings.
# WHY: PyTorch and sklearn can produce warnings during HPO that clutter output.
import warnings  # Standard library module for warning control

# typing: provides type annotations for function signatures.
# WHY: Type hints improve code readability and enable static analysis.
from typing import Any, Dict, Tuple  # Type annotation classes

# --- Third-party imports ---
# numpy: foundational numerical computing library.
# WHY: Data generation and metric computation use numpy arrays.
import numpy as np  # Numerical computing library

# torch: PyTorch deep learning framework.
# WHY: Provides automatic differentiation, GPU support, and nn.Module for
# building the differentiable soft decision tree.
import torch  # PyTorch core library
import torch.nn as nn  # Neural network modules (layers, loss functions)
import torch.optim as optim  # Optimization algorithms (SGD, Adam)

# optuna: Bayesian hyperparameter optimization.
# WHY: Intelligently searches the hyperparameter space using TPE.
import optuna  # Hyperparameter optimization framework

# sklearn utilities for data generation, splitting, and metrics.
from sklearn.datasets import make_classification  # Synthetic data generator
from sklearn.metrics import (
    accuracy_score,           # Overall correctness
    classification_report,    # Per-class precision, recall, F1
    f1_score,                 # Harmonic mean of precision and recall
    precision_score,          # Positive predictive value
    recall_score,             # Sensitivity
    roc_auc_score,            # Area under ROC curve
)
from sklearn.model_selection import train_test_split  # Data splitting utility
from sklearn.preprocessing import StandardScaler  # Feature standardization

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + level + message
)
logger = logging.getLogger(__name__)  # Module-specific logger
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings


# ---------------------------------------------------------------------------
# Soft Decision Tree Model (PyTorch nn.Module)
# ---------------------------------------------------------------------------

class SoftDecisionTree(nn.Module):
    """Differentiable Soft Decision Tree for classification.

    Architecture:
        - Internal nodes: Each has a linear layer (w^T x + b) followed by sigmoid
          to compute the probability of routing left vs right.
        - Leaf nodes: Each has a learnable class distribution (logits -> softmax).
        - Forward pass: Computes path probabilities for all leaves, then
          outputs a weighted mixture of leaf distributions.

    The tree has:
        - 2^depth - 1 internal nodes (decision nodes)
        - 2^depth leaf nodes (prediction nodes)

    WHY nn.Module: Integrates seamlessly with PyTorch's autograd for
    automatic differentiation, optimizer support, and GPU acceleration.
    """

    def __init__(
        self,
        input_dim: int,       # Number of input features
        n_classes: int,       # Number of output classes
        depth: int = 5,       # Depth of the tree
        temperature: float = 1.0,  # Sigmoid temperature (softness control)
    ):
        """Initialize the soft decision tree.

        Args:
            input_dim: Dimensionality of input features.
            n_classes: Number of classification classes.
            depth: Tree depth. Determines capacity: 2^depth leaves.
                WHY: Deeper trees have more capacity but require more memory
                (exponential in depth) and can overfit.
            temperature: Controls sigmoid sharpness.
                WHY: Low temperature -> hard decisions (like traditional tree).
                High temperature -> soft decisions (more like averaging).
        """
        super(SoftDecisionTree, self).__init__()  # Initialize nn.Module parent

        self.depth = depth              # Store tree depth
        self.n_classes = n_classes      # Store number of classes
        self.temperature = temperature  # Store temperature parameter
        self.n_internal = 2 ** depth - 1  # Number of internal (decision) nodes
        self.n_leaves = 2 ** depth        # Number of leaf (prediction) nodes

        # Internal node decision layers: each maps input to a single routing logit.
        # WHY nn.ModuleList: PyTorch needs to track these as submodules for
        # parameter registration, gradient computation, and device placement.
        # WHY nn.Linear(input_dim, 1): Each node makes a single binary decision
        # based on a linear combination of all input features (oblique split).
        self.decision_layers = nn.ModuleList([
            nn.Linear(input_dim, 1)  # w^T x + b -> single routing logit
            for _ in range(self.n_internal)  # One linear layer per internal node
        ])

        # Leaf node class distributions: learnable logits for each class.
        # WHY nn.Parameter: These are directly optimized parameters (no layer needed).
        # Shape: (n_leaves, n_classes) - each leaf has a distribution over classes.
        self.leaf_distributions = nn.Parameter(
            torch.randn(self.n_leaves, n_classes) * 0.01  # Small random init
        )
        # WHY small init (0.01): Large initial logits would create confident
        # but random predictions, making early training unstable.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute class probabilities for input batch.

        Algorithm:
            1. Start with probability 1.0 for reaching the root.
            2. At each internal node, compute sigmoid routing probability.
            3. Split the path probability: left gets p, right gets (1-p).
            4. Multiply each leaf's path probability by its class distribution.
            5. Sum across all leaves for final prediction.

        WHY this approach: By processing all paths simultaneously, we avoid
        the sequential nature of traditional tree traversal and enable
        efficient GPU computation.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Class probabilities of shape (batch_size, n_classes).
        """
        batch_size = x.shape[0]  # Number of samples in the batch

        # Initialize path probabilities: all samples start at root with probability 1.
        # WHY ones: Every sample reaches the root node with certainty.
        # Shape: (batch_size, 1) because there's one root node.
        path_probs = torch.ones(batch_size, 1, device=x.device)  # Start at root

        # Process tree level by level (breadth-first).
        # WHY level-by-level: At each level, we double the number of path probabilities
        # (each node splits into left and right children).
        for level in range(self.depth):  # Iterate through each tree level
            # Compute the index range for nodes at this level.
            # WHY: In a binary tree stored as an array, level l has nodes
            # at indices [2^l - 1, 2^(l+1) - 2].
            start_idx = 2 ** level - 1      # First node index at this level
            end_idx = 2 ** (level + 1) - 1  # One past last node at this level
            n_nodes = end_idx - start_idx    # Number of nodes at this level

            # Compute routing probabilities for all nodes at this level.
            # WHY batch computation: Process all nodes at the same level simultaneously
            # for GPU efficiency.
            routing_probs = []  # List to collect routing probabilities
            for i in range(start_idx, end_idx):  # For each node at this level
                # Compute the routing logit: w^T x + b.
                logit = self.decision_layers[i](x)  # Shape: (batch_size, 1)
                # Apply sigmoid with temperature to get routing probability.
                # WHY temperature: Dividing logit by temperature controls sharpness.
                # Low tau -> sharp sigmoid (near 0 or 1), high tau -> flat (near 0.5).
                prob = torch.sigmoid(logit / self.temperature)  # Shape: (batch_size, 1)
                routing_probs.append(prob)  # Collect routing probability

            # Stack routing probabilities into a single tensor.
            # Shape: (batch_size, n_nodes_at_level)
            routing_probs = torch.cat(routing_probs, dim=1)  # Concatenate along node dim

            # Split path probabilities: left child gets p, right child gets (1-p).
            # WHY interleave: Each parent's probability splits into two children.
            # Parent i -> left child at 2i+1, right child at 2i+2 in array representation.
            left_probs = path_probs * routing_probs      # P(left) = P(parent) * p
            right_probs = path_probs * (1 - routing_probs)  # P(right) = P(parent) * (1-p)

            # Interleave left and right probabilities for the next level.
            # WHY interleave: Maintains the correct parent-child mapping in the array.
            # [left_0, right_0, left_1, right_1, ...] becomes the path_probs for next level.
            path_probs = torch.stack([left_probs, right_probs], dim=2)  # Stack along new dim
            path_probs = path_probs.view(batch_size, -1)  # Flatten to (batch, 2*n_nodes)

        # path_probs now has shape (batch_size, n_leaves): probability of reaching each leaf.

        # Compute leaf class distributions using softmax.
        # WHY softmax: Converts raw logits to valid probability distributions
        # (non-negative, sum to 1) for each leaf node.
        leaf_dists = torch.softmax(self.leaf_distributions, dim=1)  # (n_leaves, n_classes)

        # Compute final prediction as weighted mixture of leaf distributions.
        # WHY mixture: The prediction is the expected class distribution over all
        # possible paths through the tree, weighted by path probabilities.
        # path_probs: (batch_size, n_leaves), leaf_dists: (n_leaves, n_classes)
        # Result: (batch_size, n_classes) via matrix multiplication.
        output = torch.matmul(path_probs, leaf_dists)  # Weighted mixture of leaf distributions

        return output  # Return class probabilities


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,      # Total number of samples
    n_features: int = 20,       # Total number of features
    n_classes: int = 2,         # Number of target classes
    random_state: int = 42,     # Random seed for reproducibility
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data and split into train/val/test.

    WHY synthetic data: Provides controlled benchmarking conditions with known
    signal-to-noise ratio, allowing fair algorithm comparison.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Generate synthetic classification dataset.
    X, y = make_classification(
        n_samples=n_samples,            # Total data points
        n_features=n_features,          # Feature dimensionality
        n_informative=n_features // 2,  # Half carry real signal
        n_redundant=n_features // 4,    # Quarter are linear combinations
        n_classes=n_classes,            # Number of classes
        random_state=random_state,      # Reproducibility seed
    )

    # First split: 60% train, 40% temp (val + test).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )

    # Second split: 50% of temp for val, 50% for test (20% each of total).
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Standardize features for neural network training.
    # WHY: Neural networks train faster and more stably when features have
    # zero mean and unit variance, because gradient magnitudes are similar.
    scaler = StandardScaler()  # Create scaler instance
    X_train = scaler.fit_transform(X_train)  # Fit on train, transform train
    X_val = scaler.transform(X_val)          # Transform val (using train statistics)
    X_test = scaler.transform(X_test)        # Transform test (using train statistics)

    logger.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,            # Training features
    y_train: np.ndarray,            # Training labels
    depth: int = 5,                 # Soft tree depth
    temperature: float = 1.0,       # Sigmoid temperature
    learning_rate: float = 0.01,    # Optimizer learning rate
    n_epochs: int = 100,            # Number of training epochs
    batch_size: int = 64,           # Mini-batch size
) -> SoftDecisionTree:
    """Train a Soft Decision Tree using gradient descent.

    WHY gradient descent: The soft decision tree is fully differentiable,
    so we can use standard neural network training with backpropagation.
    This replaces the greedy splitting of traditional CART.

    Args:
        X_train: Training features as numpy array.
        y_train: Training labels as numpy array.
        depth: Tree depth (number of levels).
        temperature: Sigmoid temperature parameter.
        learning_rate: Step size for optimizer.
        n_epochs: Number of passes through the training data.
        batch_size: Number of samples per gradient update.

    Returns:
        Trained SoftDecisionTree model.
    """
    logger.info(
        f"Training Soft Decision Tree: depth={depth}, temp={temperature}, "
        f"lr={learning_rate}, epochs={n_epochs}, batch={batch_size}"
    )

    # Determine device: use GPU if available, else CPU.
    # WHY: GPU parallelism dramatically accelerates matrix operations.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto device

    # Get input dimensions from data.
    n_features = X_train.shape[1]  # Number of features
    n_classes = len(np.unique(y_train))  # Number of unique classes

    # Create model and move to device.
    model = SoftDecisionTree(
        input_dim=n_features,     # Input feature count
        n_classes=n_classes,      # Output class count
        depth=depth,              # Tree depth
        temperature=temperature,  # Sigmoid temperature
    ).to(device)  # Move model parameters to GPU/CPU

    # Convert numpy arrays to PyTorch tensors.
    # WHY float32: Standard precision for neural network training.
    # WHY long for labels: CrossEntropyLoss expects integer class indices.
    X_tensor = torch.FloatTensor(X_train).to(device)  # Features to GPU
    y_tensor = torch.LongTensor(y_train).to(device)    # Labels to GPU

    # Define loss function and optimizer.
    # WHY CrossEntropyLoss: Standard loss for multi-class classification.
    # It combines log-softmax and NLL loss, numerically stable.
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss function

    # WHY Adam: Adaptive learning rate optimizer that works well with default
    # settings. Maintains per-parameter learning rates using momentum estimates.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Training loop.
    model.train()  # Set model to training mode (enables dropout, batch norm, etc.)
    n_samples = len(X_tensor)  # Total training samples

    for epoch in range(n_epochs):  # Iterate over epochs
        # Shuffle data at each epoch.
        # WHY: Prevents the model from learning the order of samples.
        perm = torch.randperm(n_samples)  # Random permutation of indices
        epoch_loss = 0.0  # Accumulate loss for this epoch

        # Process mini-batches.
        # WHY mini-batches: Full-batch gradient descent is too slow for large
        # datasets, and single-sample SGD is too noisy. Mini-batches balance
        # computational efficiency with gradient estimation quality.
        for i in range(0, n_samples, batch_size):  # Iterate over batches
            # Extract mini-batch.
            batch_idx = perm[i:i + batch_size]  # Indices for this batch
            X_batch = X_tensor[batch_idx]        # Batch features
            y_batch = y_tensor[batch_idx]        # Batch labels

            # Forward pass: compute predictions.
            outputs = model(X_batch)  # Shape: (batch_size, n_classes)

            # Compute loss between predictions and true labels.
            loss = criterion(outputs, y_batch)  # Cross-entropy loss

            # Backward pass: compute gradients.
            optimizer.zero_grad()  # Clear previous gradients (PyTorch accumulates)
            loss.backward()        # Backpropagate gradients through the tree

            # Update parameters using optimizer.
            optimizer.step()  # Apply gradient update (Adam step)

            epoch_loss += loss.item()  # Accumulate batch loss

        # Log progress every 20 epochs.
        if (epoch + 1) % 20 == 0:  # Periodic logging
            avg_loss = epoch_loss / (n_samples / batch_size)  # Average loss per batch
            logger.info(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model  # Return the trained model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: SoftDecisionTree,  # Trained model
    X_val: np.ndarray,        # Validation features
    y_val: np.ndarray,        # Validation labels
) -> Dict[str, float]:
    """Evaluate the soft decision tree on the validation set.

    WHY: Validation provides an estimate of generalization performance
    during hyperparameter tuning without touching the test set.

    Args:
        model: Trained SoftDecisionTree model.
        X_val: Validation feature matrix.
        y_val: Validation label array.

    Returns:
        Dictionary of metric names to values.
    """
    device = next(model.parameters()).device  # Get model's device

    model.eval()  # Set model to evaluation mode (disables dropout etc.)

    # Convert validation data to tensors.
    X_tensor = torch.FloatTensor(X_val).to(device)  # Features to device

    # Generate predictions without computing gradients.
    # WHY torch.no_grad: Saves memory and computation during inference.
    with torch.no_grad():  # Disable gradient tracking
        outputs = model(X_tensor)  # Forward pass: (n_samples, n_classes)
        y_proba = outputs.cpu().numpy()  # Move to CPU and convert to numpy
        y_pred = np.argmax(y_proba, axis=1)  # Hard predictions (argmax of probs)

    # Compute classification metrics.
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),  # Overall correctness
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_val, y_proba[:, 1]) if y_proba.shape[1] == 2 else 0.0,
    }

    logger.info("Validation Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test(
    model: SoftDecisionTree,  # Trained model
    X_test: np.ndarray,       # Test features
    y_test: np.ndarray,       # Test labels
) -> Dict[str, float]:
    """Final evaluation on the held-out test set.

    WHY: The test set provides an unbiased estimate of real-world performance.
    It is used exactly once after all tuning is complete.

    Args:
        model: Trained SoftDecisionTree model.
        X_test: Test feature matrix.
        y_test: Test label array.

    Returns:
        Dictionary of metric names to values.
    """
    device = next(model.parameters()).device  # Get model's device

    model.eval()  # Evaluation mode

    X_tensor = torch.FloatTensor(X_test).to(device)  # Convert to tensor

    with torch.no_grad():  # No gradient computation needed
        outputs = model(X_tensor)  # Forward pass
        y_proba = outputs.cpu().numpy()  # Probabilities
        y_pred = np.argmax(y_proba, axis=1)  # Hard predictions

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba[:, 1]) if y_proba.shape[1] == 2 else 0.0,
    }

    logger.info("=" * 50)
    logger.info("TEST SET RESULTS (Final Evaluation):")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    logger.info("=" * 50)
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization - Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial) -> float:
    """Optuna objective for soft decision tree hyperparameter optimization.

    Searches over tree depth, temperature, learning rate, and batch size.

    WHY these ranges: Depth 2-6 covers simple to moderately complex trees
    (memory grows exponentially). Temperature 0.1-10.0 covers hard to soft.
    Learning rate 1e-4 to 1e-1 covers fine to coarse gradient steps.

    Args:
        trial: Optuna Trial object for hyperparameter suggestions.

    Returns:
        Validation F1 score to maximize.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Suggest hyperparameters.
    depth = trial.suggest_int("depth", 2, 6)  # Tree depth (exponential memory)
    temperature = trial.suggest_float("temperature", 0.1, 10.0, log=True)  # Sigmoid temp
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)  # LR
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Batch size

    model = train(
        X_train, y_train,
        depth=depth, temperature=temperature,
        learning_rate=learning_rate, batch_size=batch_size,
        n_epochs=50,  # Fewer epochs for faster HPO
    )

    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune hyperparameter search space for the soft decision tree.

    Returns:
        Dictionary defining the Ray Tune search space.
    """
    search_space = {
        "depth": {"type": "randint", "lower": 2, "upper": 7},
        "temperature": {"type": "loguniform", "lower": 0.1, "upper": 10.0},
        "learning_rate": {"type": "loguniform", "lower": 1e-4, "upper": 1e-1},
        "batch_size": {"type": "choice", "values": [32, 64, 128]},
    }
    logger.info("Ray Tune search space defined:")
    for param, config in search_space.items():
        logger.info(f"  {param}: {config}")
    return search_space


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple soft decision tree configurations with reasoning.

    Configurations:
    1. Shallow + cold (depth=3, temp=0.1): Near-hard tree, limited capacity.
    2. Medium + standard (depth=5, temp=1.0): Balanced soft tree.
    3. Deep + standard (depth=8, temp=1.0): High capacity, risk of overfitting.
    4. Medium + hot (depth=5, temp=10.0): Very soft routing, near-uniform mixing.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    configs = {
        "shallow_cold_d3_t0.1": {
            "params": {"depth": 3, "temperature": 0.1, "learning_rate": 0.01, "n_epochs": 80},
            "reasoning": (
                "Shallow tree (depth=3) with low temperature (0.1). The low temperature "
                "makes sigmoid nearly binary (hard decisions). Only 8 leaves limit capacity. "
                "Expected: clean, interpretable but potentially underfitting."
            ),
        },
        "medium_standard_d5_t1.0": {
            "params": {"depth": 5, "temperature": 1.0, "learning_rate": 0.01, "n_epochs": 80},
            "reasoning": (
                "Medium tree (depth=5, 32 leaves) with standard temperature. "
                "Soft routing allows smooth decision boundaries. "
                "Expected: good balance between capacity and generalization."
            ),
        },
        "deep_standard_d8_t1.0": {
            "params": {"depth": 8, "temperature": 1.0, "learning_rate": 0.005, "n_epochs": 80},
            "reasoning": (
                "Deep tree (depth=8, 256 leaves) with standard temperature. "
                "Very high capacity but may overfit with soft routing spreading "
                "probability mass across too many leaves. Lower LR for stability."
            ),
        },
        "medium_hot_d5_t10.0": {
            "params": {"depth": 5, "temperature": 10.0, "learning_rate": 0.01, "n_epochs": 80},
            "reasoning": (
                "Medium tree with high temperature (10.0). Very soft sigmoid makes "
                "all routing probabilities near 0.5, effectively averaging all leaf "
                "distributions equally. Expected: underfitting due to lack of specialization."
            ),
        },
    }

    results = {}
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {name}")
        logger.info(f"Reasoning: {config['reasoning']}")
        logger.info(f"Parameters: {config['params']}")

        model = train(X_train, y_train, **config["params"])
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY:")
    for name, metrics in results.items():
        logger.info(f"  {name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo
# ---------------------------------------------------------------------------

def real_world_demo() -> None:
    """Demonstrate the soft decision tree on a medical diagnosis task.

    Domain: Cardiovascular disease risk assessment.
    Features: age, systolic_blood_pressure, cholesterol_level, bmi
    Target: heart_disease (0 = no, 1 = yes)

    WHY this domain: Medical diagnosis benefits from the soft decision tree's
    ability to provide calibrated probabilities. Unlike hard trees, soft trees
    output continuous confidence scores, useful for clinical risk stratification.
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Medical Diagnosis (Heart Disease) - Soft Tree")
    logger.info("=" * 60)

    np.random.seed(42)  # Reproducibility
    n_samples = 500  # Synthetic patients

    # Generate realistic medical features.
    age = np.random.normal(55, 15, n_samples).clip(20, 90)
    systolic_bp = np.random.normal(130, 20, n_samples).clip(90, 200)
    cholesterol = np.random.normal(220, 40, n_samples).clip(120, 350)
    bmi = np.random.normal(27, 5, n_samples).clip(15, 45)

    X = np.column_stack([age, systolic_bp, cholesterol, bmi])
    feature_names = ["age", "systolic_blood_pressure", "cholesterol_level", "bmi"]

    # Create target with realistic risk factors.
    risk_score = (
        0.03 * (age - 50)
        + 0.02 * (systolic_bp - 120)
        + 0.01 * (cholesterol - 200)
        + 0.05 * (bmi - 25)
    )
    probability = 1.0 / (1.0 + np.exp(-risk_score))
    noise = np.random.normal(0, 0.1, n_samples)
    y = (probability + noise > 0.5).astype(int)

    # Split data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Standardize features for neural network training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Log feature statistics.
    logger.info("\nFeature Statistics (Training Set):")
    for i, name in enumerate(feature_names):
        logger.info(
            f"  {name}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}"
        )

    # Train soft decision tree.
    model = train(
        X_train, y_train,
        depth=4, temperature=1.0, learning_rate=0.01, n_epochs=100
    )

    # Evaluate.
    logger.info("\nValidation Results:")
    validate(model, X_val, y_val)
    logger.info("\nTest Results:")
    test(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete Soft Decision Tree pipeline.

    Steps: data generation -> baseline training -> validation -> parameter
    comparison -> Optuna HPO -> best model training -> test evaluation ->
    real-world demo.
    """
    logger.info("=" * 60)
    logger.info("Soft Decision Tree Classifier - PyTorch Implementation")
    logger.info("=" * 60)

    # Step 1: Generate data.
    logger.info("\n--- Step 1: Generating Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    # Step 2: Train baseline.
    logger.info("\n--- Step 2: Training Baseline ---")
    baseline = train(X_train, y_train, depth=5, temperature=1.0)

    # Step 3: Validate baseline.
    logger.info("\n--- Step 3: Validating Baseline ---")
    validate(baseline, X_val, y_val)

    # Step 4: Compare parameter sets.
    logger.info("\n--- Step 4: Comparing Parameter Sets ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Step 5: Optuna HPO.
    logger.info("\n--- Step 5: Optuna HPO ---")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(optuna_objective, n_trials=15)
    logger.info(f"Best trial F1: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    # Step 6: Ray Tune search space.
    logger.info("\n--- Step 6: Ray Tune Search Space ---")
    ray_tune_search()

    # Step 7: Train best model.
    logger.info("\n--- Step 7: Training Best Model ---")
    best_params = study.best_trial.params
    best_model = train(
        X_train, y_train,
        depth=best_params["depth"],
        temperature=best_params["temperature"],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        n_epochs=100,
    )

    # Step 8: Final test evaluation.
    logger.info("\n--- Step 8: Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Step 9: Real-world demo.
    logger.info("\n--- Step 9: Real-World Demo ---")
    real_world_demo()

    logger.info("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    main()
