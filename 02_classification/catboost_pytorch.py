"""
CatBoost-Inspired Model with Categorical Embeddings - PyTorch Implementation
==============================================================================

Theory & Mathematics:
    This module implements a CatBoost-inspired neural network that natively
    handles categorical features using learned embeddings, combined with
    continuous features through an MLP, and optionally uses gradient boosting
    over neural network predictions for improved performance.

    1. Categorical Feature Embeddings:
        Instead of one-hot encoding (sparse, high-dimensional) or target
        encoding (requires careful handling to avoid leakage), we use
        learned embedding vectors for each categorical feature:

        For categorical feature j with V_j unique values:
            e_j(x_j) = Embedding(x_j) in R^{d_j}

        where d_j is the embedding dimension (hyperparameter).
        The embedding table is a learnable matrix E_j in R^{V_j x d_j}.

        Benefits over one-hot:
        - Dense representation captures semantic similarity
        - Reduces dimensionality: V_j -> d_j where d_j << V_j
        - Learns feature interactions through shared latent space
        - Handles high-cardinality categories efficiently

    2. Feature Concatenation:
        Continuous features and embedded categorical features are
        concatenated to form the full input:
            h = [x_cont; e_1(x_1); e_2(x_2); ...; e_K(x_K)]
        where x_cont are continuous features and e_k are embeddings.

        Total input dimension = n_continuous + sum(d_k for k = 1..K)

    3. MLP Classification Head:
        The concatenated features are fed through a multi-layer perceptron:
            z = MLP(h) = W_L * ReLU(... W_2 * ReLU(W_1 * h + b_1) + b_2) + b_L

        With batch normalization and dropout for regularization.

    4. Gradient Boosting over Neural Network Predictions:
        Optionally, we can use gradient boosting over the neural network:
        - Train base MLP M_0 on the full training data
        - For m = 1, ..., M:
            - Compute residuals: r_i = y_i - sigma(M_{m-1}(x_i))
            - Train a new MLP M_m on (X, r_i)
            - Update: F_m(x) = F_{m-1}(x) + eta * M_m(x)

        This combines the representation power of embeddings with
        the iterative refinement of gradient boosting.

    5. Loss Function:
        Binary Cross-Entropy with Logits (BCEWithLogitsLoss):
            L = -[y * log(sigma(z)) + (1-y) * log(1 - sigma(z))]
        Numerically stable implementation using the log-sum-exp trick.

Business Use Cases:
    - Hotel booking cancellation with categorical metadata
    - E-commerce recommendation with user/item features
    - Insurance risk prediction with categorical policy features
    - Ad targeting with categorical user demographics
    - Medical diagnosis with categorical symptom/diagnosis codes

Advantages:
    - Learned embeddings capture semantic relationships between categories
    - End-to-end training optimizes embeddings for the task
    - Handles high-cardinality categorical features gracefully
    - GPU-accelerated training and inference
    - Embedding tables can be pre-trained and transferred
    - Combines continuous and categorical features naturally

Disadvantages:
    - Requires specifying embedding dimensions per feature
    - More parameters to tune than tree-based CatBoost
    - Needs more data to learn good embeddings
    - Less interpretable than tree-based methods
    - Embedding lookup can be slow for very high cardinality
    - No built-in ordered boosting (target leakage must be handled separately)

Hyperparameters:
    - embedding_dim: Dimension of categorical embeddings
    - hidden_layers: Tuple of hidden layer sizes for the MLP
    - lr: Learning rate for the optimizer
    - n_epochs: Total training epochs
    - weight_decay: L2 regularization
    - batch_size: Mini-batch size
    - dropout: Dropout rate in the MLP
    - n_boosting_rounds: Number of gradient boosting rounds (0 = no boosting)
    - boosting_lr: Learning rate for the boosting ensemble
"""

import logging  # Standard logging for training progress
import warnings  # Suppress non-critical warnings
from typing import Any, Dict, List, Optional, Tuple  # Type annotations

import numpy as np  # Numerical computing
import optuna  # Bayesian hyperparameter optimization
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network building blocks
from torch.utils.data import DataLoader, TensorDataset  # Data loading utilities
from sklearn.metrics import (  # Classification evaluation metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Preprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Categorical Embedding Network
# ---------------------------------------------------------------------------

class CategoricalEmbeddingNet(nn.Module):
    """
    Neural network with learned embeddings for categorical features and
    an MLP head for classification. Categorical features are embedded
    into dense vectors and concatenated with continuous features.
    """

    def __init__(
        self,
        n_continuous: int,  # Number of continuous features
        cat_cardinalities: List[int],  # Number of unique values per categorical feature
        embedding_dim: int = 8,  # Embedding vector dimension for each category
        hidden_layers: Tuple[int, ...] = (64, 32),  # MLP hidden layer sizes
        dropout: float = 0.2,  # Dropout rate for regularization
    ) -> None:
        """Build the embedding layers and MLP classification head."""
        super().__init__()  # Initialize nn.Module base class

        # Create one nn.Embedding per categorical feature
        self.embeddings = nn.ModuleList()  # List of embedding layers
        total_emb_dim = 0  # Track total embedding dimension for MLP input

        for cardinality in cat_cardinalities:  # One embedding per categorical feature
            # Embedding table: cardinality -> embedding_dim
            # +1 for unknown/unseen categories (mapped to index=cardinality)
            emb = nn.Embedding(
                num_embeddings=cardinality + 1,  # +1 for unknown category index
                embedding_dim=embedding_dim,  # Dense vector dimension
            )
            nn.init.xavier_uniform_(emb.weight)  # Xavier initialization
            self.embeddings.append(emb)  # Add to the module list
            total_emb_dim += embedding_dim  # Accumulate total embedding size

        # Build the MLP classification head
        input_dim = n_continuous + total_emb_dim  # Continuous + all embeddings
        layers = []  # Accumulate MLP layers

        prev_dim = input_dim  # Track previous layer's output dimension
        for hidden_dim in hidden_layers:  # Add each hidden layer
            layers.append(nn.Linear(prev_dim, hidden_dim))  # Linear transform
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch normalization
            layers.append(nn.ReLU())  # ReLU activation
            layers.append(nn.Dropout(dropout))  # Dropout regularization
            prev_dim = hidden_dim  # Update previous dimension

        layers.append(nn.Linear(prev_dim, 1))  # Final output: single logit

        self.mlp = nn.Sequential(*layers)  # Wrap in Sequential

        self.n_continuous = n_continuous  # Store for forward pass
        self.cat_cardinalities = cat_cardinalities  # Store cardinalities

    def forward(
        self,
        x_continuous: torch.Tensor,  # Continuous features: (batch, n_continuous)
        x_categorical: torch.Tensor,  # Categorical indices: (batch, n_cat_features)
    ) -> torch.Tensor:
        """
        Forward pass: embed categoricals, concatenate with continuous, feed to MLP.
        """
        # Embed each categorical feature and collect the embeddings
        cat_embeddings = []  # List of embedding vectors
        for i, emb_layer in enumerate(self.embeddings):  # Process each cat feature
            cat_idx = x_categorical[:, i].long()  # Get integer indices for this feature
            cat_emb = emb_layer(cat_idx)  # Look up embeddings: (batch, embedding_dim)
            cat_embeddings.append(cat_emb)  # Collect

        if len(cat_embeddings) > 0:  # Has categorical features
            # Concatenate all embeddings: (batch, total_emb_dim)
            cat_combined = torch.cat(cat_embeddings, dim=1)
            # Concatenate continuous features with embedded categoricals
            combined = torch.cat([x_continuous, cat_combined], dim=1)  # (batch, total_dim)
        else:  # No categorical features, only continuous
            combined = x_continuous  # Just use continuous features

        return self.mlp(combined).squeeze(-1)  # MLP output: (batch,)


# ---------------------------------------------------------------------------
# CatBoost-Inspired Classifier with Embeddings and Optional Boosting
# ---------------------------------------------------------------------------

class CatBoostEmbeddingClassifier:
    """
    Classifier combining categorical embeddings with an MLP, optionally
    enhanced by gradient boosting over multiple neural networks.
    """

    def __init__(
        self,
        n_continuous: int = 0,  # Number of continuous features
        cat_cardinalities: Optional[List[int]] = None,  # Unique values per cat feature
        embedding_dim: int = 8,  # Embedding dimension
        hidden_layers: Tuple[int, ...] = (64, 32),  # MLP architecture
        lr: float = 0.001,  # Optimizer learning rate
        n_epochs: int = 50,  # Training epochs
        weight_decay: float = 1e-4,  # L2 regularization
        batch_size: int = 64,  # Mini-batch size
        dropout: float = 0.2,  # Dropout rate
        n_boosting_rounds: int = 0,  # 0 = no boosting, >0 = boosted ensemble
        boosting_lr: float = 0.1,  # Shrinkage for boosting
        random_state: int = 42,  # Random seed
    ) -> None:
        """Store all hyperparameters."""
        self.n_continuous = n_continuous
        self.cat_cardinalities = cat_cardinalities if cat_cardinalities else []
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout = dropout
        self.n_boosting_rounds = n_boosting_rounds
        self.boosting_lr = boosting_lr
        self.random_state = random_state
        self.models: List[CategoricalEmbeddingNet] = []  # Trained networks
        self.cat_encoders: List[LabelEncoder] = []  # Categorical label encoders
        self.scaler: Optional[StandardScaler] = None  # Continuous feature scaler
        self.cat_feature_indices: List[int] = []  # Indices of categorical columns
        self.cont_feature_indices: List[int] = []  # Indices of continuous columns

    def _prepare_data(
        self,
        X: np.ndarray,  # Raw feature matrix
        y: Optional[np.ndarray] = None,  # Labels (only for training)
        fit: bool = False,  # Whether to fit the encoders
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare data: encode categoricals to integer indices, scale continuous
        features, convert everything to tensors.
        """
        # Extract continuous features
        if len(self.cont_feature_indices) > 0:
            X_cont = X[:, self.cont_feature_indices].astype(np.float64)
            if fit:  # Fit scaler on training data
                self.scaler = StandardScaler()
                X_cont = self.scaler.fit_transform(X_cont)
            else:  # Transform with pre-fit scaler
                X_cont = self.scaler.transform(X_cont)
        else:
            X_cont = np.zeros((X.shape[0], 0))  # Empty continuous array

        # Encode categorical features to integer indices
        X_cat = np.zeros((X.shape[0], len(self.cat_feature_indices)), dtype=np.int64)
        for i, col_idx in enumerate(self.cat_feature_indices):
            if fit:  # Fit the label encoder
                le = LabelEncoder()
                encoded = le.fit_transform(X[:, col_idx].astype(str))
                self.cat_encoders.append(le)
            else:  # Transform with pre-fit encoder
                le = self.cat_encoders[i]
                # Handle unseen categories by mapping to max_index + 1
                raw_vals = X[:, col_idx].astype(str)
                encoded = np.zeros(len(raw_vals), dtype=np.int64)
                for j, val in enumerate(raw_vals):
                    if val in le.classes_:  # Known category
                        encoded[j] = le.transform([val])[0]
                    else:  # Unknown category -> use last index (reserved)
                        encoded[j] = len(le.classes_)  # Maps to unknown embedding
            X_cat[:, i] = encoded

        # Convert to tensors
        x_cont_t = torch.tensor(X_cont, dtype=torch.float32, device=DEVICE)
        x_cat_t = torch.tensor(X_cat, dtype=torch.long, device=DEVICE)
        y_t = None
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        return x_cont_t, x_cat_t, y_t

    def _train_single_model(
        self,
        x_cont: torch.Tensor,  # Continuous features tensor
        x_cat: torch.Tensor,  # Categorical indices tensor
        targets: torch.Tensor,  # Target values (labels or residuals)
        is_residual: bool = False,  # True if training on residuals
    ) -> CategoricalEmbeddingNet:
        """Train a single CategoricalEmbeddingNet on the given targets."""
        model = CategoricalEmbeddingNet(
            n_continuous=x_cont.shape[1],
            cat_cardinalities=[c + 1 for c in (self.cat_cardinalities or [])],
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        # Use MSE for residuals, BCE for direct classification
        if is_residual:
            loss_fn = nn.MSELoss()  # Mean squared error for residual learning
        else:
            loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy for classification

        # Create DataLoader for mini-batch training
        dataset = TensorDataset(x_cont, x_cat, targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0  # Accumulate loss for logging
            for batch_cont, batch_cat, batch_y in loader:
                optimizer.zero_grad()  # Clear gradients
                logits = model(batch_cont, batch_cat)  # Forward pass
                loss = loss_fn(logits, batch_y)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update parameters
                total_loss += loss.item() * batch_cont.shape[0]

            if (epoch + 1) % 20 == 0:  # Log every 20 epochs
                avg_loss = total_loss / x_cont.shape[0]
                logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, self.n_epochs, avg_loss)

        return model

    def fit(
        self,
        X: np.ndarray,  # Training features (mixed continuous + categorical)
        y: np.ndarray,  # Training labels {0, 1}
        cat_feature_indices: Optional[List[int]] = None,  # Which columns are categorical
    ) -> "CatBoostEmbeddingClassifier":
        """
        Train the classifier with categorical embeddings and optional boosting.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Determine which columns are categorical vs continuous
        if cat_feature_indices is not None:
            self.cat_feature_indices = cat_feature_indices
        self.cont_feature_indices = [
            i for i in range(X.shape[1]) if i not in self.cat_feature_indices
        ]
        self.n_continuous = len(self.cont_feature_indices)

        # Compute cardinalities for each categorical feature
        self.cat_cardinalities = []
        for col_idx in self.cat_feature_indices:
            n_unique = len(np.unique(X[:, col_idx].astype(str)))
            self.cat_cardinalities.append(n_unique)

        # Reset encoders for fresh fit
        self.cat_encoders = []
        self.models = []

        # Prepare data (fit encoders and scaler)
        x_cont, x_cat, y_t = self._prepare_data(X, y, fit=True)

        if self.n_boosting_rounds <= 0:
            # Single model training (no boosting)
            logger.info("Training single embedding model (no boosting)")
            model = self._train_single_model(x_cont, x_cat, y_t, is_residual=False)
            self.models.append(model)
        else:
            # Gradient boosting over neural network predictions
            logger.info("Training boosted ensemble: %d rounds", self.n_boosting_rounds)
            n_samples = X.shape[0]
            raw_predictions = torch.zeros(n_samples, device=DEVICE)

            for m in range(self.n_boosting_rounds):
                if m == 0:
                    # First model: train directly on labels
                    logger.info("  Boosting round %d/%d (base model)", m + 1, self.n_boosting_rounds)
                    model = self._train_single_model(x_cont, x_cat, y_t, is_residual=False)
                    model.eval()
                    with torch.no_grad():
                        raw_predictions = model(x_cont, x_cat)
                else:
                    # Subsequent models: train on residuals
                    logger.info("  Boosting round %d/%d (residual model)", m + 1, self.n_boosting_rounds)
                    probs = torch.sigmoid(raw_predictions)
                    residuals = y_t - probs  # Pseudo-residuals

                    model = self._train_single_model(x_cont, x_cat, residuals, is_residual=True)
                    model.eval()
                    with torch.no_grad():
                        update = model(x_cont, x_cat)
                    raw_predictions = raw_predictions + self.boosting_lr * update

                self.models.append(model)

        logger.info(
            "CatBoostEmbedding trained: %d models, emb_dim=%d, hidden=%s",
            len(self.models), self.embedding_dim, self.hidden_layers,
        )
        return self

    def _raw_predict(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw logit predictions from the ensemble."""
        if len(self.models) == 1:
            self.models[0].eval()
            with torch.no_grad():
                return self.models[0](x_cont, x_cat)
        else:
            # Boosted ensemble: base model + sum of residual corrections
            self.models[0].eval()
            with torch.no_grad():
                raw = self.models[0](x_cont, x_cat)
            for model in self.models[1:]:
                model.eval()
                with torch.no_grad():
                    raw = raw + self.boosting_lr * model(x_cont, x_cat)
            return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        x_cont, x_cat, _ = self._prepare_data(X, fit=False)
        raw = self._raw_predict(x_cont, x_cat)
        p1 = torch.sigmoid(raw).cpu().numpy()
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Generate a hotel booking dataset with categorical and continuous features.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, cat_feature_indices).
    """
    rng = np.random.RandomState(random_state)

    # Continuous features
    lead_time = rng.exponential(100, n_samples)
    adr = rng.normal(100, 40, n_samples).clip(20, 500)
    stays_weekend = rng.poisson(1, n_samples)
    stays_week = rng.poisson(3, n_samples)
    previous_cancellations = rng.poisson(0.3, n_samples)
    special_requests = rng.poisson(1.5, n_samples)

    # Categorical features
    country = rng.choice(
        ["PRT", "GBR", "FRA", "ESP", "DEU", "USA", "BRA", "OTHER"],
        n_samples, p=[0.3, 0.12, 0.1, 0.1, 0.08, 0.1, 0.1, 0.1],
    )
    market_segment = rng.choice(
        ["Online_TA", "Offline_TA", "Direct", "Corporate", "Groups"],
        n_samples, p=[0.45, 0.15, 0.15, 0.15, 0.10],
    )
    deposit_type = rng.choice(
        ["No_Deposit", "Non_Refund", "Refundable"],
        n_samples, p=[0.7, 0.2, 0.1],
    )
    meal_type = rng.choice(
        ["BB", "HB", "FB", "SC"],
        n_samples, p=[0.5, 0.25, 0.1, 0.15],
    )

    X = np.column_stack([
        lead_time, adr, stays_weekend, stays_week,
        previous_cancellations, special_requests,
        country, market_segment, deposit_type, meal_type,
    ])
    cat_features = [6, 7, 8, 9]  # Categorical column indices

    # Generate cancellation labels
    cancel_logit = (
        0.005 * lead_time - 0.005 * adr
        + 0.5 * previous_cancellations - 0.3 * special_requests
        + 1.0 * (deposit_type == "Non_Refund").astype(float)
        - 0.5 * (market_segment == "Corporate").astype(float)
        + 0.3 * (country == "PRT").astype(float)
        + rng.normal(0, 0.5, n_samples)
    )
    cancel_prob = 1.0 / (1.0 + np.exp(-cancel_logit))
    y = (rng.random(n_samples) < cancel_prob).astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp,
    )

    logger.info(
        "Data generated: train=%d, val=%d, test=%d, cat_features=%s",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], cat_features,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, cat_features


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cat_feature_indices: Optional[List[int]] = None,
    **hyperparams: Any,
) -> CatBoostEmbeddingClassifier:
    """Train a CatBoost-style embedding classifier."""
    defaults = dict(
        embedding_dim=8,
        hidden_layers=(64, 32),
        lr=0.001,
        n_epochs=50,
        weight_decay=1e-4,
        batch_size=64,
        dropout=0.2,
        n_boosting_rounds=0,
        boosting_lr=0.1,
        random_state=42,
    )
    defaults.update(hyperparams)

    # Handle hidden_layers if passed as list from Optuna
    if isinstance(defaults.get("hidden_layers"), list):
        defaults["hidden_layers"] = tuple(defaults["hidden_layers"])

    model = CatBoostEmbeddingClassifier(**defaults)
    model.fit(X_train, y_train, cat_feature_indices=cat_feature_indices)
    return model


def _evaluate(
    model: CatBoostEmbeddingClassifier, X: np.ndarray, y: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    auc = roc_auc_score(y, y_proba[:, 1])
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc,
    }


def validate(
    model: CatBoostEmbeddingClassifier, X_val: np.ndarray, y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluate on validation data."""
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def test(
    model: CatBoostEmbeddingClassifier, X_test: np.ndarray, y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate on test data with full reporting."""
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
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_feature_indices: Optional[List[int]] = None,
) -> float:
    """Optuna objective: suggest params, train, return val F1."""
    params = {
        "embedding_dim": trial.suggest_categorical("embedding_dim", [4, 8, 16]),
        "hidden_layers": trial.suggest_categorical(
            "hidden_layers", ["(64,)", "(128, 64)", "(64, 32)", "(128, 64, 32)"]
        ),
        "lr": trial.suggest_float("lr", 1e-4, 0.01, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 30, 100, step=10),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "n_boosting_rounds": trial.suggest_int("n_boosting_rounds", 0, 5),
    }
    # Convert hidden_layers string to tuple
    params["hidden_layers"] = eval(params["hidden_layers"])  # Safe: predefined choices

    model = train(X_train, y_train, cat_feature_indices=cat_feature_indices, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_feature_indices: Optional[List[int]] = None,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search."""
    import ray
    from ray import tune as ray_tune

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _trainable(config: Dict[str, Any]) -> None:
        model = train(X_train, y_train, cat_feature_indices=cat_feature_indices, **config)
        metrics = validate(model, X_val, y_val)
        ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])

    search_space = {
        "embedding_dim": ray_tune.choice([4, 8, 16]),
        "hidden_layers": ray_tune.choice([(64,), (128, 64), (64, 32)]),
        "lr": ray_tune.loguniform(1e-4, 0.01),
        "n_epochs": ray_tune.choice([30, 50, 80]),
        "weight_decay": ray_tune.loguniform(1e-6, 1e-2),
        "dropout": ray_tune.uniform(0.1, 0.4),
        "n_boosting_rounds": ray_tune.choice([0, 2, 5]),
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
    cat_feature_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare 4 configurations exploring embedding dimension and MLP architecture.

    Configurations:
        1. small_emb_small_mlp (emb=4, hidden=(64,)):
           Compact model with small embeddings. Fast training, low memory.
           Categories are represented in low-dimensional space.

        2. large_emb_small_mlp (emb=16, hidden=(64,)):
           Rich categorical embeddings but simple MLP. Tests whether
           more expressive embeddings improve performance with simple head.

        3. small_emb_large_mlp (emb=4, hidden=(128, 64)):
           Compact embeddings with a deeper MLP. Tests whether a more
           complex classification head compensates for simpler embeddings.

        4. large_emb_large_mlp (emb=16, hidden=(128, 64)):
           Full capacity: rich embeddings + deep MLP. Most parameters.
           Best potential accuracy but highest overfitting risk.
    """
    configs = {
        "small_emb_small_mlp (e=4, h=(64,))": {
            # Reasoning: Compact representation. 4-dim embeddings capture basic
            # category relationships. Single hidden layer is fast to train.
            # Good when categories have simple relationships with the target.
            "embedding_dim": 4,
            "hidden_layers": (64,),
            "n_epochs": 50,
        },
        "large_emb_small_mlp (e=16, h=(64,))": {
            # Reasoning: 16-dim embeddings can capture nuanced category semantics.
            # Simple MLP keeps total parameters manageable. Tests if richer
            # embeddings alone can improve predictions significantly.
            "embedding_dim": 16,
            "hidden_layers": (64,),
            "n_epochs": 50,
        },
        "small_emb_large_mlp (e=4, h=(128,64))": {
            # Reasoning: Simple embeddings with a deeper classification head.
            # The MLP can learn complex decision boundaries from compact
            # representations. Tests MLP depth vs embedding richness.
            "embedding_dim": 4,
            "hidden_layers": (128, 64),
            "n_epochs": 50,
        },
        "large_emb_large_mlp (e=16, h=(128,64))": {
            # Reasoning: Maximum model capacity. Rich embeddings + deep MLP.
            # Best chance of capturing complex patterns but needs more data
            # and regularization to avoid overfitting.
            "embedding_dim": 16,
            "hidden_layers": (128, 64),
            "n_epochs": 50,
        },
    }

    results = {}
    logger.info("=" * 70)
    logger.info("Comparing %d CatBoost Embedding configurations", len(configs))
    logger.info("=" * 70)

    for name, params in configs.items():
        logger.info("\n--- Config: %s ---", name)
        model = train(X_train, y_train, cat_feature_indices=cat_feature_indices, **params)
        metrics = validate(model, X_val, y_val)
        results[name] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("%-45s | Accuracy | F1     | AUC-ROC", "Configuration")
    logger.info("-" * 80)
    for name, metrics in results.items():
        logger.info(
            "%-45s | %.4f   | %.4f | %.4f",
            name, metrics["accuracy"], metrics["f1"], metrics["auc_roc"],
        )

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Hotel Booking Cancellation
# ---------------------------------------------------------------------------

def real_world_demo() -> Dict[str, float]:
    """
    Demonstrate the embedding-based classifier on hotel booking cancellation.

    Domain: Hospitality / revenue management
    Goal: Predict booking cancellations using categorical embeddings.

    Features:
        - lead_time: Days between booking and arrival (continuous)
        - adr: Average daily rate (continuous)
        - stays_weekend: Weekend nights (continuous)
        - stays_week: Weekday nights (continuous)
        - previous_cancellations: Prior cancellations (continuous)
        - special_requests: Number of special requests (continuous)
        - country: Country of origin (categorical)
        - market_segment: Booking channel (categorical)
        - deposit_type: Deposit type (categorical)
        - meal_type: Meal plan (categorical)
    """
    logger.info("=" * 70)
    logger.info("REAL-WORLD DEMO: Hotel Booking Cancellation (Embedding Model)")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test, cat_features = generate_data(
        n_samples=1000, random_state=42,
    )

    feature_names = [
        "lead_time", "adr", "stays_weekend", "stays_week",
        "previous_cancellations", "special_requests",
        "country", "market_segment", "deposit_type", "meal_type",
    ]
    logger.info("Features: %s", feature_names)
    logger.info("Categorical: %s", [feature_names[i] for i in cat_features])
    logger.info("Cancellation rate: %.1f%%", 100 * np.mean(y_train))

    model = train(
        X_train, y_train,
        cat_feature_indices=cat_features,
        embedding_dim=8,
        hidden_layers=(128, 64),
        n_epochs=50,
        n_boosting_rounds=3,
        boosting_lr=0.1,
    )
    validate(model, X_val, y_val)
    metrics = test(model, X_test, y_test)
    return metrics


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full CatBoost embedding pipeline."""
    logger.info("=" * 70)
    logger.info("CatBoost-Inspired Model with Categorical Embeddings - PyTorch")
    logger.info("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test, cat_features = generate_data(n_samples=1000)

    # Baseline (no boosting)
    logger.info("\n--- Baseline Training (no boosting) ---")
    model = train(
        X_train, y_train,
        cat_feature_indices=cat_features,
        embedding_dim=8,
        hidden_layers=(64, 32),
    )
    validate(model, X_val, y_val)

    # Optuna HPO
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train, y_train, X_val, y_val, cat_features,
        ),
        n_trials=10,
        show_progress_bar=True,
    )
    logger.info("Optuna best params: %s", study.best_params)
    logger.info("Optuna best F1: %.4f", study.best_value)

    # Retrain with best params
    best_params = study.best_params.copy()
    if "hidden_layers" in best_params and isinstance(best_params["hidden_layers"], str):
        best_params["hidden_layers"] = eval(best_params["hidden_layers"])
    best_model = train(X_train, y_train, cat_feature_indices=cat_features, **best_params)

    # Test
    logger.info("\n--- Final Test Evaluation ---")
    test(best_model, X_test, y_test)

    # Compare
    logger.info("\n--- Parameter Set Comparison ---")
    compare_parameter_sets(X_train, y_train, X_val, y_val, cat_features)

    # Demo
    logger.info("\n--- Real-World Demo: Hotel Booking Cancellation ---")
    real_world_demo()


if __name__ == "__main__":
    main()
