"""
AdaBoost-Inspired Attention-Weighted Ensemble - PyTorch Implementation
=======================================================================

Theory & Mathematics:
    This module implements an AdaBoost-inspired ensemble of small neural
    networks using PyTorch. Instead of decision stumps, each weak learner
    is a compact multi-layer perceptron (MLP). An attention mechanism
    learns to weight each learner's contribution, replacing AdaBoost's
    analytical alpha computation with a learned, data-dependent weighting.

    Key Concepts:

    1. Boosting-Style Sequential Training:
        Like AdaBoost, weak learners are trained sequentially. Each new
        network is trained on a reweighted version of the data that
        emphasizes the samples misclassified by the current ensemble.

        Weight update (AdaBoost-style):
            w_i = w_i * exp(-alpha * y_i * h_t(x_i))
            w_i = w_i / sum(w_j)

    2. Attention-Based Ensemble Weighting:
        Instead of the fixed alpha_t = 0.5 * ln((1-eps)/eps) formula,
        we learn a parameterized attention mechanism that computes
        context-dependent weights for each learner:
            a_t(x) = softmax(W_attn * [h_1(x), ..., h_T(x)] + b_attn)
            H(x) = sum(a_t(x) * h_t(x))

        This allows different learners to specialize on different regions
        of the input space, unlike classic AdaBoost where weights are global.

    3. Neural Network Weak Learner:
        Each weak learner is a small MLP:
            h_t(x) = sigma(W_2 * ReLU(W_1 * x + b_1) + b_2)
        Deliberately small to maintain the weak learner property.

    4. Training Loss:
        Binary Cross-Entropy with Logits:
            L = -[y * log(sigma(z)) + (1-y) * log(1-sigma(z))]
        where z is the combined ensemble output.

    Connections to AdaBoost:
        - Sequential training with reweighting -> same as SAMME
        - Attention mechanism -> generalization of alpha weights
        - Small networks -> equivalent to weak learners
        - Final combination -> weighted vote like AdaBoost

Business Use Cases:
    - Customer churn prediction with complex feature interactions
    - Fraud detection requiring adaptive ensemble weighting
    - Medical diagnosis combining multiple diagnostic models
    - Sentiment analysis with ensemble of text classifiers
    - Click-through rate prediction with nonlinear patterns

Advantages:
    - Neural networks capture nonlinear patterns within each learner
    - Attention mechanism provides data-dependent ensemble weighting
    - GPU-accelerated training for large datasets
    - Differentiable end-to-end for potential fine-tuning
    - More flexible than fixed alpha weights in classic AdaBoost

Disadvantages:
    - More hyperparameters than classic AdaBoost (learning rate, hidden dim, etc.)
    - Sequential training is inherently slow (like all boosting)
    - Risk of overfitting if weak learners are too large
    - Less interpretable than decision stump ensembles
    - No theoretical training error bound like classic AdaBoost

Hyperparameters:
    - n_learners: Number of weak neural network learners
    - hidden_dim: Hidden layer size in each weak learner MLP
    - lr: Learning rate for the neural network optimizer
    - n_epochs_per_learner: Training epochs per weak learner
    - weight_decay: L2 regularization strength
    - batch_size: Mini-batch size for training
    - attention_dim: Dimensionality of the attention layer
"""

import logging  # Standard logging for progress tracking and debugging
import warnings  # Suppress non-critical warnings for cleaner terminal output
from typing import Any, Dict, List, Tuple  # Type annotations for code clarity

import numpy as np  # Numerical operations for data manipulation
import optuna  # Bayesian hyperparameter optimization framework
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules (layers, activations, losses)
from torch.utils.data import DataLoader, TensorDataset  # Efficient data loading
from sklearn.datasets import make_classification  # Synthetic dataset generation
from sklearn.metrics import (  # Standard classification evaluation metrics
    accuracy_score,  # Fraction of correct predictions
    classification_report,  # Per-class precision, recall, F1 report
    confusion_matrix,  # Matrix of true/false positive/negative counts
    f1_score,  # Harmonic mean of precision and recall
    precision_score,  # True positives / predicted positives
    recall_score,  # True positives / actual positives
    roc_auc_score,  # Area under the ROC curve
)
from sklearn.model_selection import train_test_split  # Stratified data splitting
from sklearn.preprocessing import StandardScaler  # Feature standardization

# Configure module-level logging with timestamps
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + level + message
)
logger = logging.getLogger(__name__)  # Create logger specific to this module
warnings.filterwarnings("ignore")  # Suppress sklearn/torch convergence warnings

# Select GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Weak Learner: Small MLP
# ---------------------------------------------------------------------------

class WeakLearnerMLP(nn.Module):
    """
    A deliberately small multi-layer perceptron serving as a weak learner.
    Kept small (few parameters) so individual learners don't overfit,
    mimicking the role of decision stumps in classical AdaBoost.
    """

    def __init__(
        self,
        input_dim: int,  # Number of input features
        hidden_dim: int = 16,  # Hidden layer width (small for weak learner)
    ) -> None:
        """Build a 2-layer MLP: input -> hidden -> 1 output (logit)."""
        super().__init__()  # Initialize nn.Module parent class
        self.net = nn.Sequential(  # Stack layers sequentially
            nn.Linear(input_dim, hidden_dim),  # First linear: input_dim -> hidden_dim
            nn.ReLU(),  # ReLU activation for nonlinearity
            nn.Linear(hidden_dim, 1),  # Second linear: hidden_dim -> 1 (single logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute the raw logit output for each sample."""
        return self.net(x).squeeze(-1)  # Remove trailing dim: (batch, 1) -> (batch,)


# ---------------------------------------------------------------------------
# Attention-Weighted Ensemble
# ---------------------------------------------------------------------------

class AttentionEnsemble(nn.Module):
    """
    Learns to weight the outputs of multiple weak learners using a
    simple linear attention mechanism. The attention weights are
    computed from the concatenated learner outputs, allowing the
    ensemble to assign context-dependent importance to each learner.
    """

    def __init__(self, n_learners: int) -> None:
        """Initialize attention parameters for n_learners weak learners."""
        super().__init__()  # Initialize nn.Module parent class
        # Linear layer that maps n_learners logits to n_learners attention scores
        self.attention = nn.Linear(n_learners, n_learners)  # Learner-to-weight mapping

    def forward(self, learner_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted combination of learner outputs.
        Input: (batch, n_learners) tensor of raw logits from each learner.
        Output: (batch,) tensor of weighted ensemble logits.
        """
        # Compute attention weights via softmax over learned scores
        attn_scores = self.attention(learner_outputs)  # (batch, n_learners) raw scores
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Normalize to sum to 1
        # Weighted combination: sum(w_t * h_t(x)) for each sample
        combined = torch.sum(attn_weights * learner_outputs, dim=-1)  # (batch,)
        return combined  # Return the ensemble's combined prediction


# ---------------------------------------------------------------------------
# AdaBoost-Inspired Attention Ensemble Classifier
# ---------------------------------------------------------------------------

class AdaBoostAttentionClassifier:
    """
    Full AdaBoost-inspired classifier with neural network weak learners
    and a learned attention-based ensemble weighting mechanism.
    """

    def __init__(
        self,
        input_dim: int,  # Number of input features (set during training)
        n_learners: int = 10,  # Number of weak MLP learners in the ensemble
        hidden_dim: int = 16,  # Hidden layer size for each weak learner
        lr: float = 0.01,  # Learning rate for the Adam optimizer
        n_epochs_per_learner: int = 20,  # Epochs to train each learner
        weight_decay: float = 1e-4,  # L2 regularization strength
        batch_size: int = 64,  # Mini-batch size for DataLoader
        random_state: int = 42,  # Seed for reproducibility
    ) -> None:
        """Store hyperparameters and prepare for sequential training."""
        self.input_dim = input_dim  # Feature dimensionality
        self.n_learners = n_learners  # How many weak learners to train
        self.hidden_dim = hidden_dim  # Width of each weak learner's hidden layer
        self.lr = lr  # Optimizer learning rate
        self.n_epochs_per_learner = n_epochs_per_learner  # Training epochs per learner
        self.weight_decay = weight_decay  # L2 regularization (weight decay)
        self.batch_size = batch_size  # Samples per mini-batch
        self.random_state = random_state  # Random seed
        self.learners: List[WeakLearnerMLP] = []  # Container for trained weak learners
        self.attention: Optional[AttentionEnsemble] = None  # Attention module (trained last)

    def fit(
        self,
        X: np.ndarray,  # Training feature matrix (n_samples, n_features)
        y: np.ndarray,  # Training labels in {0, 1}
    ) -> "AdaBoostAttentionClassifier":
        """
        Train the ensemble sequentially with boosting-style reweighting,
        then train the attention mechanism on top of the frozen learners.
        """
        torch.manual_seed(self.random_state)  # Set PyTorch random seed
        np.random.seed(self.random_state)  # Set NumPy random seed for consistency

        n_samples = X.shape[0]  # Total number of training examples
        self.input_dim = X.shape[1]  # Update input dimension from data

        # Convert numpy arrays to PyTorch tensors on the target device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)  # Features
        y_tensor = torch.tensor(y, dtype=torch.float32, device=DEVICE)  # Labels

        # Initialize sample weights uniformly (1/N), same as AdaBoost
        sample_weights = torch.ones(n_samples, device=DEVICE) / n_samples  # Uniform init

        self.learners = []  # Reset the learner list

        # Phase 1: Sequentially train weak learners with boosting-style reweighting
        for t in range(self.n_learners):  # Train each weak learner one by one
            # Create a new small MLP as the current weak learner
            learner = WeakLearnerMLP(self.input_dim, self.hidden_dim).to(DEVICE)
            optimizer = torch.optim.Adam(  # Adam optimizer for each learner
                learner.parameters(),
                lr=self.lr,  # Learning rate
                weight_decay=self.weight_decay,  # L2 regularization
            )
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")  # Per-sample BCE loss

            # Train this learner for n_epochs_per_learner epochs
            learner.train()  # Set to training mode (enables dropout if any)
            for epoch in range(self.n_epochs_per_learner):
                optimizer.zero_grad()  # Clear gradients from previous step
                logits = learner(X_tensor)  # Forward pass: get raw predictions
                # Weighted loss: multiply per-sample loss by the sample weight
                per_sample_loss = loss_fn(logits, y_tensor)  # (n_samples,) losses
                weighted_loss = torch.sum(sample_weights * per_sample_loss)  # Scalar
                weighted_loss.backward()  # Compute gradients via backpropagation
                optimizer.step()  # Update learner parameters

            # After training, compute predictions and update sample weights
            learner.eval()  # Switch to evaluation mode
            with torch.no_grad():  # No gradients needed for weight update
                logits = learner(X_tensor)  # Get final predictions from this learner
                preds = (torch.sigmoid(logits) > 0.5).float()  # Hard predictions {0, 1}
                # Encode predictions as {-1, +1} for AdaBoost-style weight update
                y_coded = 2 * y_tensor - 1  # Convert {0,1} -> {-1,+1}
                pred_coded = 2 * preds - 1  # Convert {0,1} -> {-1,+1}

                # Compute weighted error rate (fraction of weighted misclassifications)
                misclassified = (preds != y_tensor).float()  # 1 if wrong, 0 if correct
                epsilon = torch.sum(sample_weights * misclassified)  # Weighted error
                epsilon = torch.clamp(epsilon, min=1e-10, max=1.0 - 1e-10)  # Numerical stability

                # Compute learner importance alpha (same as classic AdaBoost)
                alpha = 0.5 * torch.log((1.0 - epsilon) / epsilon)  # Log odds ratio

                # Update sample weights: increase for misclassified, decrease for correct
                sample_weights = sample_weights * torch.exp(-alpha * y_coded * pred_coded)
                sample_weights = sample_weights / torch.sum(sample_weights)  # Normalize

            self.learners.append(learner)  # Store the trained learner
            logger.info("Learner %d/%d trained, weighted error=%.4f, alpha=%.4f",
                        t + 1, self.n_learners, epsilon.item(), alpha.item())

        # Phase 2: Train attention mechanism on the frozen learner outputs
        self.attention = AttentionEnsemble(self.n_learners).to(DEVICE)  # Create attention
        attn_optimizer = torch.optim.Adam(  # Separate optimizer for attention
            self.attention.parameters(), lr=self.lr,
        )
        attn_loss_fn = nn.BCEWithLogitsLoss()  # Standard BCE for attention training

        # Collect all learner outputs (frozen) to train attention
        with torch.no_grad():  # No gradients for the learners during attention training
            all_outputs = torch.stack(  # Stack outputs: (n_samples, n_learners)
                [learner(X_tensor) for learner in self.learners], dim=1
            )

        # Train attention for a fixed number of epochs
        self.attention.train()  # Set attention module to training mode
        for epoch in range(50):  # 50 epochs for attention convergence
            attn_optimizer.zero_grad()  # Clear attention gradients
            combined = self.attention(all_outputs)  # Attention-weighted combination
            loss = attn_loss_fn(combined, y_tensor)  # BCE loss on combined output
            loss.backward()  # Backpropagate through attention only
            attn_optimizer.step()  # Update attention parameters

        logger.info("Attention mechanism trained over %d learners", self.n_learners)
        return self  # Return self for method chaining

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the attention-weighted ensemble.
        Returns shape (n_samples, 2) with [P(y=0), P(y=1)].
        """
        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)  # To tensor

        # Collect predictions from all learners
        with torch.no_grad():  # No gradient computation needed for inference
            all_outputs = torch.stack(  # (n_samples, n_learners)
                [learner(X_tensor) for learner in self.learners], dim=1
            )
            combined = self.attention(all_outputs)  # Attention-weighted combination
            prob_pos = torch.sigmoid(combined).cpu().numpy()  # Sigmoid -> probabilities

        prob_pos = np.clip(prob_pos, 1e-7, 1.0 - 1e-7)  # Numerical stability
        return np.column_stack([1.0 - prob_pos, prob_pos])  # [P(y=0), P(y=1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hard class labels {0, 1} by thresholding probabilities."""
        proba = self.predict_proba(X)  # Get probability estimates
        return (proba[:, 1] >= 0.5).astype(int)  # Threshold at 0.5


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,  # Total samples to generate
    n_features: int = 20,  # Number of features per sample
    n_classes: int = 2,  # Number of classes (binary)
    random_state: int = 42,  # Seed for reproducibility
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data split into train/val/test.
    Features are standardized to zero mean and unit variance.
    """
    # Generate synthetic dataset with informative and noisy features
    X, y = make_classification(
        n_samples=n_samples,  # Total number of generated samples
        n_features=n_features,  # Dimensionality of feature space
        n_informative=n_features // 2,  # Half of features carry useful signal
        n_redundant=n_features // 4,  # Quarter are linear combos of informative ones
        n_classes=n_classes,  # Binary classification
        random_state=random_state,  # Fixed seed for reproducibility
    )

    # Split: 60% train, 20% validation, 20% test (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    # Standardize features (fit on train only to prevent data leakage)
    scaler = StandardScaler()  # Zero mean, unit variance scaler
    X_train = scaler.fit_transform(X_train)  # Fit and transform training data
    X_val = scaler.transform(X_val)  # Transform validation with train stats
    X_test = scaler.transform(X_test)  # Transform test with train stats

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, features=%d, classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], n_features, n_classes,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,  # Training feature matrix
    y_train: np.ndarray,  # Training labels
    **hyperparams: Any,  # Keyword arguments for hyperparameters
) -> AdaBoostAttentionClassifier:
    """
    Train an AdaBoost-inspired attention ensemble with given hyperparameters.
    Merges user hyperparams with sensible defaults.
    """
    defaults = dict(
        input_dim=X_train.shape[1],  # Infer input dimension from data
        n_learners=10,  # 10 weak learners as default
        hidden_dim=16,  # Small hidden layer for weak learner property
        lr=0.01,  # Moderate learning rate
        n_epochs_per_learner=20,  # 20 epochs per weak learner
        weight_decay=1e-4,  # Light L2 regularization
        batch_size=64,  # Standard mini-batch size
        random_state=42,  # Reproducibility
    )
    defaults.update(hyperparams)  # Override with user-provided values
    model = AdaBoostAttentionClassifier(**defaults)  # Create the classifier
    model.fit(X_train, y_train)  # Train the ensemble
    logger.info(
        "AdaBoost Attention (PyTorch) trained: %d learners, hidden_dim=%d",
        defaults["n_learners"], defaults["hidden_dim"],
    )
    return model  # Return trained model


def _evaluate(
    model: AdaBoostAttentionClassifier,  # Trained ensemble model
    X: np.ndarray,  # Feature matrix
    y: np.ndarray,  # True labels
) -> Dict[str, float]:
    """Compute all standard classification metrics."""
    y_pred = model.predict(X)  # Hard class predictions
    y_proba = model.predict_proba(X)  # Probability estimates for AUC
    auc = roc_auc_score(y, y_proba[:, 1])  # AUC using P(y=1)

    return {
        "accuracy": accuracy_score(y, y_pred),  # Overall correctness
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,  # Discrimination ability
    }


def validate(
    model: AdaBoostAttentionClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate model on validation data and log metrics."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(
    model: AdaBoostAttentionClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate model on test data with full reporting."""
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,  # Optuna trial for hyperparameter suggestions
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Optuna objective function: suggest hyperparameters, train, evaluate.
    Returns validation F1 score for Optuna to maximize.
    """
    params = {
        "n_learners": trial.suggest_int("n_learners", 3, 25),  # 3 to 25 learners
        "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64]),  # Hidden size
        "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),  # Log-scale LR search
        "n_epochs_per_learner": trial.suggest_int("n_epochs_per_learner", 10, 50, step=10),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }

    model = train(X_train, y_train, **params)  # Train with suggested params
    metrics = validate(model, X_val, y_val)  # Evaluate on val set
    return metrics["f1"]  # Return F1 for optimization


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Run Ray Tune hyperparameter search for distributed optimization."""
    import ray  # Distributed computing framework
    from ray import tune as ray_tune  # Ray's tuning module

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "n_learners": ray_tune.choice([5, 10, 15, 20]),
        "hidden_dim": ray_tune.choice([8, 16, 32, 64]),
        "lr": ray_tune.loguniform(1e-4, 0.1),
        "n_epochs_per_learner": ray_tune.choice([10, 20, 30]),
        "weight_decay": ray_tune.loguniform(1e-6, 1e-2),
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

def compare_parameter_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare 4 configurations to understand the interplay between
    ensemble size and weak learner complexity.

    Configurations:
        1. few_small: n_learners=5, hidden_dim=16
           - Small ensemble with tiny learners, fast but may underfit.
        2. many_small: n_learners=20, hidden_dim=16
           - Large ensemble of small learners, classic boosting approach.
        3. few_large: n_learners=5, hidden_dim=64
           - Few but powerful learners, risk of overfitting per learner.
        4. many_large: n_learners=20, hidden_dim=64
           - Large ensemble of strong learners, most expensive but most flexible.
    """
    configs = {
        "few_small (T=5, h=16)": {
            # Reasoning: Minimal ensemble with weak learners. Very fast to train.
            # Should show whether even a few neural weak learners can compete.
            "n_learners": 5,
            "hidden_dim": 16,  # Small hidden layer
        },
        "many_small (T=20, h=16)": {
            # Reasoning: Classic boosting philosophy - many simple learners.
            # Each learner handles a small part of the residual error.
            "n_learners": 20,
            "hidden_dim": 16,  # Small hidden layer
        },
        "few_large (T=5, h=64)": {
            # Reasoning: Fewer but more powerful learners. Each MLP can
            # model complex patterns. Risk: may violate weak learner assumption.
            "n_learners": 5,
            "hidden_dim": 64,  # Larger hidden layer
        },
        "many_large (T=20, h=64)": {
            # Reasoning: Best of both worlds in theory. More parameters,
            # higher capacity, but slowest to train and most prone to overfitting.
            "n_learners": 20,
            "hidden_dim": 64,  # Larger hidden layer
        },
    }

    results = {}
    logger.info("=" * 70)
    logger.info("Comparing %d AdaBoost Attention configurations", len(configs))
    logger.info("=" * 70)

    for name, params in configs.items():
        logger.info("\n--- Config: %s ---", name)
        model = train(X_train, y_train, **params)
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("%-35s | Accuracy | F1     | AUC-ROC", "Configuration")
    logger.info("-" * 70)
    for name, metrics in results.items():
        logger.info(
            "%-35s | %.4f   | %.4f | %.4f",
            name, metrics["accuracy"], metrics["f1"], metrics["auc_roc"],
        )

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Customer Churn Prediction
# ---------------------------------------------------------------------------

def real_world_demo() -> Dict[str, float]:
    """
    Demonstrate the attention-weighted AdaBoost ensemble on a simulated
    customer churn prediction problem.

    Domain: Telecom / SaaS customer retention
    Goal: Predict which customers are likely to churn.

    Features:
        - tenure_months: Duration of customer relationship
        - monthly_charges: Monthly subscription fee
        - total_charges: Cumulative revenue from customer
        - num_support_tickets: Number of support interactions
        - contract_length_months: Contract commitment duration
        - num_products: Number of subscribed products
        - payment_delay_days: Average payment lateness
        - usage_hours_per_week: Weekly product engagement
    """
    rng = np.random.RandomState(42)  # Reproducible random state
    n_samples = 800  # Realistic small-to-medium dataset

    # Generate realistic features with domain-appropriate distributions
    tenure_months = rng.exponential(24, n_samples)  # Right-skewed tenure
    monthly_charges = np.clip(rng.normal(70, 25, n_samples), 10, 200)  # Charges ~$70
    total_charges = tenure_months * monthly_charges  # Cumulative billing
    num_support_tickets = rng.poisson(2, n_samples)  # Poisson-distributed tickets
    contract_length = rng.choice([1, 6, 12, 24], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    num_products = rng.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    payment_delay = rng.exponential(5, n_samples)  # Delay in days
    usage_hours = rng.gamma(3, 5, n_samples)  # Engagement hours

    X = np.column_stack([  # Combine all features into a matrix
        tenure_months, monthly_charges, total_charges,
        num_support_tickets, contract_length, num_products,
        payment_delay, usage_hours,
    ])

    # Create churn labels based on business logic
    churn_logit = (
        -0.03 * tenure_months + 0.01 * monthly_charges
        + 0.15 * num_support_tickets - 0.05 * contract_length
        - 0.3 * num_products + 0.05 * payment_delay
        - 0.04 * usage_hours + rng.normal(0, 0.5, n_samples)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))  # Sigmoid transform
    y = (rng.random(n_samples) < churn_prob).astype(int)  # Binary labels

    # Split and standardize
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Customer Churn with AdaBoost Attention Ensemble")
    logger.info("=" * 70)
    logger.info("Churn rate: %.1f%%", 100 * np.mean(y))

    model = train(X_train, y_train, n_learners=10, hidden_dim=32)
    validate(model, X_val, y_val)
    metrics = test(model, X_test, y_test)
    return metrics


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full pipeline: data generation, baseline training, HPO, comparison, demo.
    """
    logger.info("=" * 70)
    logger.info("AdaBoost Attention-Weighted Ensemble - PyTorch Implementation")
    logger.info("=" * 70)

    # Generate synthetic data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data(n_samples=800, n_features=15)

    # Baseline training
    logger.info("\n--- Baseline Training ---")
    model = train(X_train, y_train, n_learners=10, hidden_dim=16)
    validate(model, X_val, y_val)

    # Optuna HPO
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=10,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    best_model = train(X_train, y_train, **study.best_params)

    # Final test
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Compare parameter sets
    logger.info("\n--- Parameter Set Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val)

    # Real-world demo
    logger.info("\n--- Real-World Demo: Customer Churn ---")
    real_world_demo()


if __name__ == "__main__":
    main()
