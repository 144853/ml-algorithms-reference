"""
Neural Collaborative Filtering (NCF) - PyTorch Implementation
===============================================================

Theory & Mathematics:
---------------------
Neural Collaborative Filtering (NCF) replaces the inner product used in
traditional matrix factorization with a neural network that can learn
an arbitrary function from user and item latent features.

Architecture (He et al., 2017):

    Input Layer:
        - User one-hot vector  -->  User Embedding (p_u in R^d)
        - Item one-hot vector  -->  Item Embedding (q_i in R^d)

    Two parallel pathways:

    1. Generalized Matrix Factorization (GMF):
       phi_GMF(p_u, q_i) = p_u * q_i   (element-wise product)

    2. Multi-Layer Perceptron (MLP):
       z_1 = ReLU(W_1 * [p_u; q_i] + b_1)
       z_2 = ReLU(W_2 * z_1 + b_2)
       ...
       phi_MLP = z_L

    Fusion (NeuMF):
       r_hat(u, i) = sigma(h^T * [phi_GMF; phi_MLP])

    where sigma is sigmoid for implicit feedback or identity for explicit.

    Loss function (explicit ratings):
       L = (1/N) * sum_{(u,i) in Omega} (r_{u,i} - r_hat(u,i))^2 + lambda * ||theta||^2

    For implicit feedback (binary):
       L = -sum [y * log(r_hat) + (1-y) * log(1 - r_hat)]

This implementation handles explicit ratings (1-5 scale) with MSE loss.

Business Use Cases:
-------------------
- E-commerce product recommendations with rich interaction patterns
- Movie/music recommendations (Netflix Prize, Spotify)
- Content personalization on media platforms
- Ad click-through rate prediction
- App store recommendation
- Cross-selling and upselling

Advantages:
-----------
- Learns non-linear user-item interaction patterns
- More expressive than linear matrix factorization
- Can incorporate side features (user demographics, item metadata)
- GPU-accelerated training scales to large datasets
- Flexible architecture: easy to add new components
- Handles both explicit and implicit feedback

Disadvantages:
--------------
- Requires more data to train effectively than simple CF
- Hyperparameter sensitive (embedding dim, layers, learning rate)
- Black-box: harder to explain recommendations
- Longer training time compared to linear methods
- Cold-start problem remains without side features
- Risk of overfitting with small datasets

Key Hyperparameters:
--------------------
- embedding_dim: Size of user/item embedding vectors
- mlp_layers: List of hidden layer sizes for MLP tower
- learning_rate: Optimizer step size
- weight_decay: L2 regularization strength
- batch_size: Mini-batch size for SGD
- dropout: Dropout rate for regularization
- n_epochs: Number of training epochs
- negative_ratio: Ratio of negative samples (for implicit feedback)

References:
-----------
- He et al., "Neural Collaborative Filtering" (WWW 2017)
- He et al., "Fast Matrix Factorization for Online Recommendation with Implicit Feedback" (SIGIR 2016)
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
"""

import numpy as np
import warnings
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import ray
from ray import tune

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(
    n_users: int = 500,
    n_items: int = 200,
    density: float = 0.10,
    rating_range: Tuple[int, int] = (1, 5),
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic user-item rating data and split into train/val/test.

    Returns matrices of shape (n_users, n_items) with 0 = unobserved.
    """
    rng = np.random.RandomState(random_state)
    n_factors = 10
    P = rng.randn(n_users, n_factors) * 0.5
    Q = rng.randn(n_items, n_factors) * 0.5
    raw = P @ Q.T + 3.0
    low, high = rating_range
    raw = np.clip(np.round(raw + rng.randn(n_users, n_items) * 0.3), low, high)
    raw = raw.astype(np.float32)

    mask = rng.rand(n_users, n_items) < density
    observed = raw * mask

    indices = np.argwhere(mask)
    rng.shuffle(indices)
    n_obs = len(indices)
    n_val = int(n_obs * val_frac)
    n_test = int(n_obs * test_frac)

    train_mat = np.zeros_like(observed)
    val_mat = np.zeros_like(observed)
    test_mat = np.zeros_like(observed)

    for u, i in indices[n_val + n_test :]:
        train_mat[u, i] = observed[u, i]
    for u, i in indices[:n_val]:
        val_mat[u, i] = observed[u, i]
    for u, i in indices[n_val : n_val + n_test]:
        test_mat[u, i] = observed[u, i]

    print(f"Data: {n_users} users x {n_items} items | "
          f"Train={n_obs - n_val - n_test}, Val={n_val}, Test={n_test}")
    return train_mat, val_mat, test_mat


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RatingDataset(Dataset):
    """PyTorch Dataset for (user, item, rating) triplets."""

    def __init__(self, rating_matrix: np.ndarray):
        mask = rating_matrix > 0
        self.users, self.items = np.where(mask)
        self.ratings = rating_matrix[mask].astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Neural Collaborative Filtering Model
# ---------------------------------------------------------------------------

class NeuralCollaborativeFiltering(nn.Module):
    """
    NeuMF = GMF + MLP combined model.

    GMF pathway: element-wise product of user/item embeddings.
    MLP pathway: concatenated embeddings through fully connected layers.
    Output: concatenation of GMF and MLP outputs through a final linear layer.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        self.n_users = n_users
        self.n_items = n_items

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings (separate from GMF)
        self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP tower
        mlp_input_dim = 2 * embedding_dim
        layers = []
        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            mlp_input_dim = hidden_dim
        self.mlp_tower = nn.Sequential(*layers)

        # Final prediction layer
        # GMF output: embedding_dim, MLP output: last hidden dim
        self.output_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        user_ids : torch.Tensor of shape (batch_size,)
        item_ids : torch.Tensor of shape (batch_size,)

        Returns
        -------
        predictions : torch.Tensor of shape (batch_size,)
        """
        # GMF pathway
        gmf_user = self.gmf_user_embedding(user_ids)  # (B, d)
        gmf_item = self.gmf_item_embedding(item_ids)  # (B, d)
        gmf_out = gmf_user * gmf_item  # element-wise product (B, d)

        # MLP pathway
        mlp_user = self.mlp_user_embedding(user_ids)  # (B, d)
        mlp_item = self.mlp_item_embedding(item_ids)  # (B, d)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)  # (B, 2d)
        mlp_out = self.mlp_tower(mlp_input)  # (B, last_hidden)

        # Combine GMF + MLP
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        prediction = self.output_layer(combined).squeeze(-1)

        return prediction

    def predict_all(self) -> np.ndarray:
        """Predict ratings for ALL (user, item) pairs. Returns (n_users, n_items)."""
        self.eval()
        with torch.no_grad():
            all_users = torch.arange(self.n_users, device=next(self.parameters()).device)
            all_items = torch.arange(self.n_items, device=next(self.parameters()).device)

            # Predict in chunks to avoid memory issues
            predictions = np.zeros((self.n_users, self.n_items), dtype=np.float32)
            batch_size = 256
            for u_start in range(0, self.n_users, batch_size):
                u_end = min(u_start + batch_size, self.n_users)
                user_batch = all_users[u_start:u_end]

                for i_start in range(0, self.n_items, batch_size):
                    i_end = min(i_start + batch_size, self.n_items)
                    item_batch = all_items[i_start:i_end]

                    # Create meshgrid of user-item pairs
                    u_mesh = user_batch.unsqueeze(1).expand(-1, len(item_batch)).reshape(-1)
                    i_mesh = item_batch.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)

                    preds = self.forward(u_mesh, i_mesh)
                    preds = preds.cpu().numpy().reshape(len(user_batch), len(item_batch))
                    predictions[u_start:u_end, i_start:i_end] = preds

        return np.clip(predictions, 1.0, 5.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _dcg(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    return float(np.sum(rel / np.log2(np.arange(1, len(rel) + 1) + 1)))


def _ndcg_at_k(actual: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    dcg = _dcg(actual[top_k], k)
    idcg = _dcg(np.sort(actual)[::-1], k)
    return dcg / idcg if idcg > 0 else 0.0


def _precision_recall_at_k(
    actual: np.ndarray, scores: np.ndarray, k: int = 10, thresh: float = 3.5
) -> Tuple[float, float]:
    top_k = set(np.argsort(scores)[::-1][:k].tolist())
    relevant = set(np.where(actual >= thresh)[0].tolist())
    if not relevant:
        return 0.0, 0.0
    hits = len(top_k & relevant)
    return hits / k, hits / len(relevant)


def _hit_rate_at_k(
    actual: np.ndarray, scores: np.ndarray, k: int = 10, thresh: float = 3.5
) -> float:
    top_k = set(np.argsort(scores)[::-1][:k].tolist())
    relevant = set(np.where(actual >= thresh)[0].tolist())
    return 1.0 if top_k & relevant else 0.0


def compute_metrics(
    true_mat: np.ndarray, pred_mat: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Compute RMSE, MAE, Precision@K, Recall@K, NDCG@K, HitRate@K."""
    mask = true_mat > 0
    if mask.sum() == 0:
        return {m: 0.0 for m in
                ["rmse", "mae", "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"]}

    rmse = _rmse(true_mat[mask], pred_mat[mask])
    mae = _mae(true_mat[mask], pred_mat[mask])

    precs, recs, ndcgs, hrs = [], [], [], []
    for u in range(true_mat.shape[0]):
        if (true_mat[u] > 0).sum() == 0:
            continue
        p, r = _precision_recall_at_k(true_mat[u], pred_mat[u], k)
        precs.append(p)
        recs.append(r)
        ndcgs.append(_ndcg_at_k(true_mat[u], pred_mat[u], k))
        hrs.append(_hit_rate_at_k(true_mat[u], pred_mat[u], k))

    return {
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": float(np.mean(precs)) if precs else 0.0,
        "recall_at_k": float(np.mean(recs)) if recs else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate_at_k": float(np.mean(hrs)) if hrs else 0.0,
    }


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    **hyperparams,
) -> NeuralCollaborativeFiltering:
    """
    Train the NCF model.

    Parameters
    ----------
    train_data : np.ndarray
        Training rating matrix.
    val_data : np.ndarray, optional
        Validation rating matrix for early stopping.
    **hyperparams
        embedding_dim, mlp_layers, lr, weight_decay, batch_size, dropout, n_epochs.

    Returns
    -------
    NeuralCollaborativeFiltering
    """
    n_users, n_items = train_data.shape
    embedding_dim = hyperparams.get("embedding_dim", 32)
    mlp_layers = hyperparams.get("mlp_layers", [64, 32, 16])
    lr = hyperparams.get("lr", 1e-3)
    weight_decay = hyperparams.get("weight_decay", 1e-5)
    batch_size = hyperparams.get("batch_size", 256)
    dropout = hyperparams.get("dropout", 0.2)
    n_epochs = hyperparams.get("n_epochs", 30)
    patience = hyperparams.get("patience", 5)

    model = NeuralCollaborativeFiltering(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_dataset = RatingDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(DEVICE)
            item_ids = item_ids.to(DEVICE)
            ratings = ratings.to(DEVICE)

            optimizer.zero_grad()
            preds = model(user_ids, item_ids)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation for early stopping
        if val_data is not None:
            pred_mat = model.predict_all()
            val_mask = val_data > 0
            if val_mask.sum() > 0:
                val_rmse = _rmse(val_data[val_mask], pred_mat[val_mask])
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch % 5 == 0 or epoch == n_epochs - 1:
                    print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                          f"Loss={avg_loss:.4f} | ValRMSE={val_rmse:.4f}")

                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss={avg_loss:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model


def validate(
    model: NeuralCollaborativeFiltering,
    val_data: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Validate the model on held-out data."""
    pred_mat = model.predict_all()
    return compute_metrics(val_data, pred_mat, k=k)


def test(
    model: NeuralCollaborativeFiltering,
    test_data: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Test evaluation."""
    pred_mat = model.predict_all()
    return compute_metrics(test_data, pred_mat, k=k)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> float:
    """Optuna objective: minimize validation RMSE."""
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    n_mlp_layers = trial.suggest_int("n_mlp_layers", 2, 4)

    # Build MLP layer sizes (decreasing)
    first_layer = trial.suggest_categorical("first_layer_size", [64, 128, 256])
    mlp_layers = []
    current = first_layer
    for _ in range(n_mlp_layers):
        mlp_layers.append(current)
        current = max(current // 2, 8)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    model = train(
        train_data,
        val_data=val_data,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        dropout=dropout,
        n_epochs=30,
        patience=5,
    )
    metrics = validate(model, val_data)
    return metrics["rmse"]


def run_optuna(
    train_data: np.ndarray, val_data: np.ndarray, n_trials: int = 15
) -> Dict:
    """Run Optuna hyperparameter optimization."""
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: optuna_objective(t, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"Optuna best RMSE: {study.best_value:.4f}")
    print(f"Optuna best params: {study.best_params}")
    return {"best_params": study.best_params, "best_rmse": study.best_value}


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def ray_tune_search(
    train_data: np.ndarray, val_data: np.ndarray, num_samples: int = 10
) -> Dict:
    """Run Ray Tune hyperparameter search."""
    ray.init(ignore_reinit_error=True, num_cpus=4)

    def trainable(config):
        # Build MLP layers
        first = config["first_layer_size"]
        n_layers = config["n_mlp_layers"]
        mlp_layers = []
        current = first
        for _ in range(n_layers):
            mlp_layers.append(current)
            current = max(current // 2, 8)

        model = train(
            train_data,
            val_data=val_data,
            embedding_dim=config["embedding_dim"],
            mlp_layers=mlp_layers,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            dropout=config["dropout"],
            batch_size=config["batch_size"],
            n_epochs=30,
            patience=5,
        )
        metrics = validate(model, val_data)
        tune.report(rmse=metrics["rmse"], mae=metrics["mae"])

    search_space = {
        "embedding_dim": tune.choice([16, 32, 64]),
        "n_mlp_layers": tune.randint(2, 5),
        "first_layer_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.0, 0.5),
        "batch_size": tune.choice([128, 256, 512]),
    }

    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=num_samples,
        metric="rmse",
        mode="min",
        verbose=1,
    )

    best_config = analysis.best_config
    best_rmse = analysis.best_result["rmse"]
    print(f"Ray Tune best RMSE: {best_rmse:.4f}")
    print(f"Ray Tune best config: {best_config}")
    ray.shutdown()
    return {"best_config": best_config, "best_rmse": best_rmse}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full Neural Collaborative Filtering pipeline."""
    print("=" * 70)
    print("Neural Collaborative Filtering (NCF) - PyTorch Implementation")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/5] Generating synthetic data...")
    train_data, val_data, test_data = generate_data(
        n_users=400, n_items=150, density=0.12, random_state=42
    )

    # 2. Train default
    print("\n[2/5] Training NCF with default hyperparameters...")
    model = train(
        train_data,
        val_data=val_data,
        embedding_dim=32,
        mlp_layers=[64, 32, 16],
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        dropout=0.2,
        n_epochs=40,
        patience=7,
    )

    # 3. Validate
    print("\n[3/5] Validation metrics:")
    val_metrics = validate(model, val_data)
    for name, value in val_metrics.items():
        print(f"  {name:20s}: {value:.4f}")

    # 4. Optuna
    print("\n[4/5] Running Optuna hyperparameter search...")
    optuna_result = run_optuna(train_data, val_data, n_trials=10)

    # 5. Test with best model from default (Optuna params need reconstruction)
    print("\n[5/5] Test set metrics:")
    test_metrics = test(model, test_data)
    for name, value in test_metrics.items():
        print(f"  {name:20s}: {value:.4f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
