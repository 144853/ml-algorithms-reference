"""
Naive Bayes-Inspired Neural Classifier - PyTorch Implementation
================================================================

Theory & Mathematics:
    This module implements a neural network classifier inspired by Naive Bayes
    principles. Instead of computing exact Bayes posteriors with fixed Gaussian
    parameters, we build a probabilistic neural network with learnable class
    priors and feature distribution parameters, trained end-to-end with
    gradient descent.

    Architecture Design Philosophy:
        Traditional NB:
            1. Estimates P(C_k) from label frequencies (fixed after training)
            2. Estimates P(x_i|C_k) from per-class feature statistics (fixed)
            3. Combines via Bayes' theorem (no optimization)

        Neural NB-Inspired Model:
            1. Learnable class prior logits: log P(C_k) are nn.Parameters
            2. Learnable per-class feature distribution parameters (mean, log-variance)
               embedded in neural network layers
            3. End-to-end optimization with cross-entropy loss via backpropagation

    Mathematical Formulation:
        The model computes a "pseudo log-posterior" for each class:

        score(C_k | x) = log_prior_k + f_theta(x, k)

        where:
            log_prior_k = learnable parameter (initialized from data frequencies)
            f_theta(x, k) = neural network output for class k, designed to
                            approximate log P(x | C_k)

        The neural network architecture uses:
        1. Shared feature extraction layers (optional, for feature learning)
        2. Per-class scoring heads that compute class-conditional log-likelihoods
        3. Final softmax over scores for classification

    Why Neural NB-Inspired (not Pure NB):
        - Pure NB with Gaussian assumption is limited to linear decision boundaries
        - Neural version can learn non-linear feature transformations
        - Learnable priors can adapt during training (vs fixed frequency counts)
        - Can be extended with hidden layers for feature interaction modeling
        - End-to-end training optimizes the actual classification objective

Business Use Cases:
    - Spam detection with adaptive feature learning
    - Probabilistic document classification with learned representations
    - Medical screening with calibrated risk scores
    - Fraud detection with evolving patterns (fine-tunable priors)

Advantages:
    - Combines interpretability of probabilistic models with neural network power
    - Learnable priors adapt to domain shift (vs fixed NB priors)
    - GPU-accelerated training and inference
    - Can model feature interactions via hidden layers
    - Outputs calibrated probabilities (trained with cross-entropy)

Disadvantages:
    - More complex than traditional NB (more parameters, training required)
    - Requires hyperparameter tuning (learning rate, architecture, epochs)
    - Less interpretable than pure NB (hidden layers obscure reasoning)
    - May overfit on small datasets (more parameters than pure NB)
    - Training requires gradient descent (not one-pass like pure NB)
"""

# --- Standard library imports ---
import logging  # Structured logging for training progress
import warnings  # Warning suppression during HPO
from typing import Any, Dict, Tuple  # Type annotations

# --- Third-party imports ---
import numpy as np  # Numerical computing
import torch  # PyTorch core
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import optuna  # Bayesian hyperparameter optimization

from sklearn.datasets import make_classification  # Synthetic data
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardization

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Naive Bayes-Inspired Neural Classifier
# ---------------------------------------------------------------------------

class NaiveBayesNet(nn.Module):
    """Neural network classifier inspired by Naive Bayes probabilistic framework.

    Architecture:
        Input -> Shared Hidden Layers -> Per-Class Scoring -> Class Priors -> Softmax

    The model has:
        1. Shared feature extraction: transforms raw features into learned representations
        2. Class-conditional scoring: per-class linear heads approximate log-likelihoods
        3. Learnable priors: additive bias terms representing log P(C_k)

    WHY this architecture: It mirrors the NB computation
    (log P(C_k|x) = log P(C_k) + log P(x|C_k)) while allowing the neural network
    to learn better feature representations and non-linear decision boundaries.
    """

    def __init__(
        self,
        input_dim: int,           # Number of input features
        n_classes: int,           # Number of output classes
        hidden_dim: int = 64,     # Hidden layer size
        n_hidden_layers: int = 2,  # Number of hidden layers
        dropout_rate: float = 0.2,  # Dropout probability for regularization
    ):
        """Initialize the NB-inspired neural classifier.

        Args:
            input_dim: Number of input features.
            n_classes: Number of classification classes.
            hidden_dim: Width of hidden layers.
                WHY: Controls model capacity. Larger = more expressive but
                more prone to overfitting. 64 is a reasonable default for
                tabular data.
            n_hidden_layers: Depth of the shared feature extractor.
                WHY: More layers can learn more abstract features but risk
                overfitting on small datasets.
            dropout_rate: Fraction of neurons dropped during training.
                WHY: Regularization technique that prevents co-adaptation
                of neurons and reduces overfitting.
        """
        super(NaiveBayesNet, self).__init__()  # Initialize nn.Module parent

        self.n_classes = n_classes  # Store class count

        # Build shared feature extraction layers.
        # WHY shared: Feature learning is common across classes; we don't need
        # separate feature extractors for each class (would be wasteful).
        layers = []  # List to build sequential model
        prev_dim = input_dim  # Track previous layer output dimension

        for i in range(n_hidden_layers):  # Build each hidden layer
            # Linear transformation: learns feature combinations.
            layers.append(nn.Linear(prev_dim, hidden_dim))  # Affine transform
            # Batch normalization: stabilizes training by normalizing activations.
            # WHY: Reduces internal covariate shift, allows higher learning rates.
            layers.append(nn.BatchNorm1d(hidden_dim))  # Normalize activations
            # ReLU activation: introduces non-linearity.
            # WHY: Without non-linearity, stacking linear layers is equivalent
            # to a single linear layer. ReLU is simple, fast, and effective.
            layers.append(nn.ReLU())  # Non-linear activation
            # Dropout: randomly zeros neurons during training.
            # WHY: Prevents overfitting by forcing the network to learn
            # redundant representations.
            layers.append(nn.Dropout(dropout_rate))  # Regularization
            prev_dim = hidden_dim  # Update dimension for next layer

        # Package layers into a sequential module.
        self.feature_extractor = nn.Sequential(*layers)  # Shared feature network

        # Per-class scoring head: maps features to class scores.
        # WHY separate from feature extractor: This layer directly outputs
        # class logits, analogous to log P(x|C_k) in Naive Bayes.
        self.class_scorer = nn.Linear(hidden_dim, n_classes)  # Class scoring layer

        # Learnable class priors (log-scale).
        # WHY nn.Parameter: These are directly optimized during training.
        # Initialized to uniform (equal priors), but training will adjust them.
        # WHY log scale: Priors are multiplied in probability space, which
        # translates to addition in log space. Log priors are additive biases.
        self.log_priors = nn.Parameter(
            torch.zeros(n_classes)  # Initialize to uniform log-priors
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute class scores combining learned features and priors.

        Computation:
            1. Extract features: h = feature_extractor(x)
            2. Compute class scores: s = class_scorer(h)
            3. Add learnable priors: output = s + log_priors

        The output represents unnormalized log-posteriors, which are passed
        to CrossEntropyLoss (which applies log-softmax internally).

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Class logits of shape (batch_size, n_classes).
        """
        # Step 1: Extract learned features through shared layers.
        # WHY: Transform raw features into a representation that makes
        # classification easier (learned non-linear feature combinations).
        features = self.feature_extractor(x)  # Shape: (batch_size, hidden_dim)

        # Step 2: Compute per-class scores (approximate log-likelihoods).
        # WHY: Each class gets a score based on the learned features.
        class_scores = self.class_scorer(features)  # Shape: (batch_size, n_classes)

        # Step 3: Add learnable priors (log P(C_k)).
        # WHY: Mirrors Bayes' theorem: log P(C_k|x) = log P(x|C_k) + log P(C_k) + const.
        # Broadcasting adds the same priors to all samples in the batch.
        output = class_scores + self.log_priors  # Shape: (batch_size, n_classes)

        return output  # Return class logits


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data and split into train/val/test with standardization.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_features // 2, n_redundant=n_features // 4,
        n_classes=n_classes, random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Standardize for neural network training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
    learning_rate: float = 0.001,
    n_epochs: int = 100,
    batch_size: int = 64,
    dropout_rate: float = 0.2,
) -> NaiveBayesNet:
    """Train the NB-inspired neural classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        hidden_dim: Hidden layer width.
        n_hidden_layers: Number of hidden layers.
        learning_rate: Optimizer learning rate.
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        dropout_rate: Dropout probability.

    Returns:
        Trained NaiveBayesNet model.
    """
    logger.info(
        f"Training NB-Net: hidden={hidden_dim}, layers={n_hidden_layers}, "
        f"lr={learning_rate}, epochs={n_epochs}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Create model.
    model = NaiveBayesNet(
        input_dim=n_features, n_classes=n_classes,
        hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
        dropout_rate=dropout_rate,
    ).to(device)

    # Convert data to tensors.
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)

    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss()  # Standard classification loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Training loop.
    model.train()
    n_samples = len(X_tensor)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)  # Shuffle indices
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            X_batch = X_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]

            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate
            optimizer.step()  # Update parameters

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / max(1, n_samples / batch_size)
            logger.info(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: NaiveBayesNet,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate on validation set.

    Args:
        model: Trained NaiveBayesNet.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Dictionary of metrics.
    """
    device = next(model.parameters()).device
    model.eval()

    X_tensor = torch.FloatTensor(X_val).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = np.argmax(y_proba, axis=1)

    n_classes = len(np.unique(y_val))
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_val, y_proba[:, 1]) if n_classes == 2 else 0.0,
    }

    logger.info("Validation Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test(
    model: NaiveBayesNet,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Final evaluation on test set.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of metrics.
    """
    device = next(model.parameters()).device
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = np.argmax(y_proba, axis=1)

    n_classes = len(np.unique(y_test))
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba[:, 1]) if n_classes == 2 else 0.0,
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
    """Optuna objective for NB-Net hyperparameter search.

    Args:
        trial: Optuna Trial.

    Returns:
        Validation F1 score.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = train(
        X_train, y_train,
        hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
        learning_rate=learning_rate, dropout_rate=dropout_rate,
        batch_size=batch_size, n_epochs=50,
    )

    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search() -> Dict[str, Any]:
    """Define Ray Tune search space.

    Returns:
        Dictionary defining the search space.
    """
    search_space = {
        "hidden_dim": {"type": "choice", "values": [32, 64, 128, 256]},
        "n_hidden_layers": {"type": "randint", "lower": 1, "upper": 4},
        "learning_rate": {"type": "loguniform", "lower": 1e-4, "upper": 1e-1},
        "dropout_rate": {"type": "uniform", "lower": 0.0, "upper": 0.5},
    }
    logger.info("Ray Tune search space:")
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
    """Compare different NB-Net configurations.

    Configurations:
    1. Small network (hidden=32, lr=0.001): Lightweight, fast, NB-like capacity.
    2. Medium network (hidden=128, lr=0.001): More capacity for feature learning.
    3. Large + fast (hidden=128, lr=0.01): Aggressive learning, risk of instability.
    4. Deep + regularized (hidden=64, 3 layers, dropout=0.3): Deep with regularization.

    Returns:
        Dictionary mapping config names to validation metrics.
    """
    configs = {
        "small_h32_lr001": {
            "params": {
                "hidden_dim": 32, "n_hidden_layers": 1,
                "learning_rate": 0.001, "dropout_rate": 0.1, "n_epochs": 80,
            },
            "reasoning": (
                "Small network (32 hidden units, 1 layer). Closest to traditional NB "
                "in capacity. Few parameters means less overfitting risk but may "
                "underfit complex patterns. Good baseline for comparison."
            ),
        },
        "medium_h128_lr001": {
            "params": {
                "hidden_dim": 128, "n_hidden_layers": 2,
                "learning_rate": 0.001, "dropout_rate": 0.2, "n_epochs": 80,
            },
            "reasoning": (
                "Medium network (128 hidden, 2 layers). Significantly more capacity "
                "for learning feature interactions. Moderate dropout prevents overfitting. "
                "Expected: better than small on complex data, may overfit on simple data."
            ),
        },
        "medium_h128_lr01": {
            "params": {
                "hidden_dim": 128, "n_hidden_layers": 2,
                "learning_rate": 0.01, "dropout_rate": 0.2, "n_epochs": 80,
            },
            "reasoning": (
                "Same architecture as medium but with 10x higher learning rate. "
                "Faster convergence but risk of overshooting optima. May oscillate "
                "or diverge if the loss landscape is rugged. Tests LR sensitivity."
            ),
        },
        "deep_h64_3layers": {
            "params": {
                "hidden_dim": 64, "n_hidden_layers": 3,
                "learning_rate": 0.001, "dropout_rate": 0.3, "n_epochs": 80,
            },
            "reasoning": (
                "Deep network (3 layers, 64 hidden) with higher dropout (0.3). "
                "More depth for hierarchical feature learning. Higher dropout "
                "compensates for increased parameter count. Expected: competitive "
                "if data has hierarchical structure, otherwise similar to medium."
            ),
        },
    }

    results = {}
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {name}")
        logger.info(f"Reasoning: {config['reasoning']}")

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
    """Demonstrate NB-Net on email spam detection.

    Domain: Email spam filtering.
    Features: word_freq_free, word_freq_money, word_freq_offer,
              capital_run_length_avg, capital_run_length_total
    Target: spam (0 = legitimate, 1 = spam)
    """
    logger.info("\n" + "=" * 60)
    logger.info("REAL-WORLD DEMO: Email Spam Detection (NB-Net)")
    logger.info("=" * 60)

    np.random.seed(42)
    n_samples = 800

    # Generate spam detection features.
    word_freq_free = np.random.exponential(0.5, n_samples)
    word_freq_money = np.random.exponential(0.3, n_samples)
    word_freq_offer = np.random.exponential(0.4, n_samples)
    capital_run_length = np.random.exponential(3.0, n_samples)
    capital_total = np.random.exponential(50.0, n_samples)

    X = np.column_stack([
        word_freq_free, word_freq_money, word_freq_offer,
        capital_run_length, capital_total
    ])
    feature_names = [
        "word_freq_free", "word_freq_money", "word_freq_offer",
        "capital_run_length_avg", "capital_run_length_total"
    ]

    # Generate labels.
    spam_score = (
        0.5 * word_freq_free + 0.7 * word_freq_money + 0.3 * word_freq_offer
        + 0.1 * capital_run_length + 0.01 * capital_total
    )
    probability = 1.0 / (1.0 + np.exp(-spam_score + 2.0))
    y = (probability + np.random.normal(0, 0.05, n_samples) > 0.5).astype(int)

    # Split and standardize.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        logger.info(f"  {name}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}")

    model = train(X_train, y_train, hidden_dim=64, n_epochs=100)
    validate(model, X_val, y_val)
    test(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete NB-Net pipeline."""
    logger.info("=" * 60)
    logger.info("Naive Bayes-Inspired Neural Classifier - PyTorch")
    logger.info("=" * 60)

    logger.info("\n--- Step 1: Generating Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    logger.info("\n--- Step 2: Training Baseline ---")
    baseline = train(X_train, y_train)

    logger.info("\n--- Step 3: Validating Baseline ---")
    validate(baseline, X_val, y_val)

    logger.info("\n--- Step 4: Comparing Parameter Sets ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    logger.info("\n--- Step 5: Optuna HPO ---")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(optuna_objective, n_trials=15)
    logger.info(f"Best F1: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    logger.info("\n--- Step 6: Ray Tune Search Space ---")
    ray_tune_search()

    logger.info("\n--- Step 7: Training Best Model ---")
    best_p = study.best_trial.params
    best_model = train(
        X_train, y_train,
        hidden_dim=best_p["hidden_dim"],
        n_hidden_layers=best_p["n_hidden_layers"],
        learning_rate=best_p["learning_rate"],
        dropout_rate=best_p["dropout_rate"],
        batch_size=best_p["batch_size"],
        n_epochs=100,
    )

    logger.info("\n--- Step 8: Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    logger.info("\n--- Step 9: Real-World Demo ---")
    real_world_demo()

    logger.info("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    main()
