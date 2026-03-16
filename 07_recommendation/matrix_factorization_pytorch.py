"""
Matrix Factorization - PyTorch Implementation (Learnable Embeddings)
=====================================================================

Theory & Mathematics:
---------------------
This implementation uses PyTorch's nn.Embedding layers to learn user and item
latent factor vectors, effectively implementing matrix factorization as a
neural network model. This approach brings GPU acceleration, automatic
differentiation, and modern optimization techniques to classical MF.

Model Architecture:
    User u  -->  nn.Embedding  -->  p_u in R^k
    Item i  -->  nn.Embedding  -->  q_i in R^k

    Prediction:
        r_hat(u, i) = mu + b_u + b_i + p_u . q_i

    where:
        mu   = global mean rating (scalar)
        b_u  = user bias (learnable scalar per user)
        b_i  = item bias (learnable scalar per item)
        p_u  = user latent vector (learnable, k-dimensional)
        q_i  = item latent vector (learnable, k-dimensional)

Loss Function:
    L = (1/|Omega|) * sum_{(u,i) in Omega} (r_{u,i} - r_hat(u,i))^2
        + lambda * (||P||_F^2 + ||Q||_F^2 + ||b_u||^2 + ||b_i||^2)

    This is implemented via MSE loss + weight_decay in the optimizer.

Optimization:
    - Adam optimizer with learning rate scheduling
    - Mini-batch SGD over observed ratings
    - Early stopping based on validation RMSE
    - Optional learning rate warmup and cosine annealing

Equivalence to Classical MF:
    This PyTorch model is mathematically equivalent to the SGD-based matrix
    factorization (Funk SVD), but gains:
    - Automatic gradient computation via autograd
    - GPU acceleration for large-scale training
    - Modern optimizers (Adam, AdamW) with momentum
    - Easy extension to add side features, deeper models, etc.
    - Integration with PyTorch ecosystem (DataLoaders, schedulers, etc.)

Business Use Cases:
-------------------
- Large-scale recommendation systems (Netflix, Amazon, YouTube)
- Real-time recommendation serving with pre-computed embeddings
- Transfer learning: user/item embeddings as features for downstream tasks
- Multi-task learning: jointly optimize ratings + click-through
- A/B testing of recommendation models
- Cold-start mitigation via embedding initialization from side features

Advantages:
-----------
- GPU-accelerated training for millions of users/items
- Automatic differentiation (no manual gradient derivation)
- Modern optimizers with adaptive learning rates
- Easy to extend architecture (add side features, layers, attention)
- Embedding vectors can be used for downstream tasks
- Mini-batch training with DataLoader for efficient I/O
- Learning rate schedulers for better convergence
- Seamless integration with neural network components

Disadvantages:
--------------
- More overhead than pure NumPy for small datasets
- Requires understanding of PyTorch framework
- Embedding tables can be memory-intensive for millions of users
- Still a linear interaction model (dot product)
- Harder to interpret than explicit similarity methods
- Training instability possible without careful hyperparameter tuning

Key Hyperparameters:
--------------------
- n_factors (k): Embedding dimension / number of latent factors
- learning_rate: Optimizer step size
- weight_decay: L2 regularization (equivalent to lambda in classical MF)
- batch_size: Mini-batch size
- n_epochs: Number of training passes
- lr_scheduler: Learning rate schedule (step, cosine, plateau)
- optimizer: Adam, AdamW, SGD

References:
-----------
- Koren, "Factorization Meets the Neighborhood" (KDD 2008)
- Rendle, "Factorization Machines" (ICDM 2010)
- He & McAuley, "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation" (2016)
- Covington et al., "Deep Neural Networks for YouTube Recommendations" (RecSys 2016)
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
    Generate synthetic sparse user-item rating matrix and split into
    train / validation / test by randomly masking entries.

    Returns
    -------
    train_matrix, val_matrix, test_matrix : np.ndarray
        Each of shape (n_users, n_items), 0 = unobserved.
    """
    rng = np.random.RandomState(random_state)

    n_factors = 10
    P_true = rng.randn(n_users, n_factors) * 0.5
    Q_true = rng.randn(n_items, n_factors) * 0.5
    raw = P_true @ Q_true.T + 3.0

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

class RatingTripletDataset(Dataset):
    """PyTorch Dataset for (user_id, item_id, rating) triplets."""

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
# Model
# ---------------------------------------------------------------------------

class MatrixFactorizationPyTorch(nn.Module):
    """
    Matrix Factorization with learnable embedding layers.

    Prediction: r_hat(u, i) = global_bias + user_bias[u] + item_bias[i] + P[u] . Q[i]

    All parameters (P, Q, biases) are learned via backpropagation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 32,
        use_biases: bool = True,
    ):
        """
        Parameters
        ----------
        n_users : int
            Total number of users.
        n_items : int
            Total number of items.
        n_factors : int
            Embedding dimension (k).
        use_biases : bool
            Whether to include user/item biases.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.use_biases = use_biases

        # Embedding layers (latent factor matrices)
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)

        if use_biases:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = nn.Parameter(torch.tensor(0.0))

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        if self.use_biases:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute predicted ratings.

        Parameters
        ----------
        user_ids : torch.Tensor of shape (batch_size,)
        item_ids : torch.Tensor of shape (batch_size,)

        Returns
        -------
        predictions : torch.Tensor of shape (batch_size,)
        """
        # Look up embeddings
        p_u = self.user_embedding(user_ids)  # (B, k)
        q_i = self.item_embedding(item_ids)  # (B, k)

        # Dot product: sum over latent dimensions
        dot = (p_u * q_i).sum(dim=-1)  # (B,)

        if self.use_biases:
            b_u = self.user_bias(user_ids).squeeze(-1)  # (B,)
            b_i = self.item_bias(item_ids).squeeze(-1)  # (B,)
            prediction = self.global_bias + b_u + b_i + dot
        else:
            prediction = dot

        return prediction

    def predict_all(self) -> np.ndarray:
        """
        Predict ratings for ALL (user, item) pairs.

        Returns
        -------
        predictions : np.ndarray of shape (n_users, n_items)
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            all_users = torch.arange(self.n_users, device=device)
            all_items = torch.arange(self.n_items, device=device)

            # Get all embeddings
            P = self.user_embedding(all_users)  # (m, k)
            Q = self.item_embedding(all_items)  # (n, k)

            # Matrix multiplication: P @ Q^T gives (m, n)
            predictions = P @ Q.T

            if self.use_biases:
                b_u = self.user_bias(all_users).squeeze(-1)  # (m,)
                b_i = self.item_bias(all_items).squeeze(-1)  # (n,)
                predictions = (
                    self.global_bias
                    + b_u.unsqueeze(1)
                    + b_i.unsqueeze(0)
                    + predictions
                )

            predictions = predictions.cpu().numpy()
        return np.clip(predictions, 1.0, 5.0).astype(np.float32)


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
) -> MatrixFactorizationPyTorch:
    """
    Train the PyTorch matrix factorization model.

    Parameters
    ----------
    train_data : np.ndarray
        Training rating matrix (n_users, n_items), 0 = unobserved.
    val_data : np.ndarray, optional
        Validation matrix for early stopping.
    **hyperparams
        n_factors, lr, weight_decay, batch_size, n_epochs, patience, use_biases,
        optimizer_type, scheduler_type.

    Returns
    -------
    MatrixFactorizationPyTorch (on DEVICE)
    """
    n_users, n_items = train_data.shape
    n_factors = hyperparams.get("n_factors", 32)
    lr = hyperparams.get("lr", 1e-3)
    weight_decay = hyperparams.get("weight_decay", 1e-4)
    batch_size = hyperparams.get("batch_size", 256)
    n_epochs = hyperparams.get("n_epochs", 50)
    patience = hyperparams.get("patience", 7)
    use_biases = hyperparams.get("use_biases", True)
    optimizer_type = hyperparams.get("optimizer_type", "adam")
    scheduler_type = hyperparams.get("scheduler_type", "plateau")

    # Create model
    model = MatrixFactorizationPyTorch(
        n_users=n_users,
        n_items=n_items,
        n_factors=n_factors,
        use_biases=use_biases,
    ).to(DEVICE)

    # Initialize global bias to global mean
    if use_biases:
        train_mask = train_data > 0
        global_mean = train_data[train_mask].mean() if train_mask.sum() > 0 else 3.0
        model.global_bias.data = torch.tensor(global_mean, device=DEVICE)

    # Optimizer
    if optimizer_type == "adam":
        optimizer_obj = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer_obj = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer_obj = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        optimizer_obj = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_obj, mode="min", factor=0.5, patience=3, verbose=False
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_obj, T_max=n_epochs, eta_min=lr * 0.01
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer_obj, step_size=15, gamma=0.5
        )
    else:
        scheduler = None

    criterion = nn.MSELoss()

    # DataLoader
    train_dataset = RatingTripletDataset(train_data)
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

            optimizer_obj.zero_grad()
            preds = model(user_ids, item_ids)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer_obj.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        if val_data is not None:
            pred_mat = model.predict_all()
            val_mask = val_data > 0
            if val_mask.sum() > 0:
                val_rmse = _rmse(val_data[val_mask], pred_mat[val_mask])

                # Learning rate scheduling
                if scheduler is not None:
                    if scheduler_type == "plateau":
                        scheduler.step(val_rmse)
                    else:
                        scheduler.step()

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    current_lr = optimizer_obj.param_groups[0]["lr"]
                    print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                          f"Loss={avg_loss:.4f} | ValRMSE={val_rmse:.4f} | "
                          f"LR={current_lr:.6f}")

                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            if scheduler is not None and scheduler_type != "plateau":
                scheduler.step()
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss={avg_loss:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model


def validate(
    model: MatrixFactorizationPyTorch,
    val_data: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Validate the model."""
    pred_mat = model.predict_all()
    return compute_metrics(val_data, pred_mat, k=k)


def test(
    model: MatrixFactorizationPyTorch,
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
    params = {
        "n_factors": trial.suggest_categorical("n_factors", [8, 16, 32, 64, 128]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "n_epochs": 50,
        "patience": 7,
        "use_biases": trial.suggest_categorical("use_biases", [True, False]),
        "optimizer_type": trial.suggest_categorical("optimizer_type", ["adam", "adamw"]),
        "scheduler_type": trial.suggest_categorical(
            "scheduler_type", ["plateau", "cosine", "step"]
        ),
    }

    model = train(train_data, val_data=val_data, **params)
    metrics = validate(model, val_data)
    return metrics["rmse"]


def run_optuna(
    train_data: np.ndarray, val_data: np.ndarray, n_trials: int = 20
) -> Dict:
    """Run Optuna hyperparameter search."""
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
        model = train(
            train_data,
            val_data=val_data,
            n_factors=config["n_factors"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            batch_size=config["batch_size"],
            n_epochs=50,
            patience=7,
            use_biases=config["use_biases"],
            optimizer_type=config["optimizer_type"],
            scheduler_type=config["scheduler_type"],
        )
        metrics = validate(model, val_data)
        tune.report(rmse=metrics["rmse"], mae=metrics["mae"])

    search_space = {
        "n_factors": tune.choice([8, 16, 32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "use_biases": tune.choice([True, False]),
        "optimizer_type": tune.choice(["adam", "adamw"]),
        "scheduler_type": tune.choice(["plateau", "cosine", "step"]),
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
    """Run the full PyTorch Matrix Factorization pipeline."""
    print("=" * 70)
    print("Matrix Factorization - PyTorch Implementation")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/6] Generating synthetic data...")
    train_data, val_data, test_data = generate_data(
        n_users=500, n_items=200, density=0.12, random_state=42
    )

    # 2. Train with biases
    print("\n[2/6] Training MF with biases...")
    model_biased = train(
        train_data,
        val_data=val_data,
        n_factors=32,
        lr=5e-3,
        weight_decay=1e-4,
        batch_size=256,
        n_epochs=60,
        patience=8,
        use_biases=True,
        optimizer_type="adam",
        scheduler_type="plateau",
    )

    print("\n  Biased MF Validation:")
    biased_val = validate(model_biased, val_data)
    for name, value in biased_val.items():
        print(f"    {name:20s}: {value:.4f}")

    # 3. Train without biases
    print("\n[3/6] Training MF without biases...")
    model_unbiased = train(
        train_data,
        val_data=val_data,
        n_factors=32,
        lr=5e-3,
        weight_decay=1e-4,
        batch_size=256,
        n_epochs=60,
        patience=8,
        use_biases=False,
        optimizer_type="adam",
        scheduler_type="plateau",
    )

    print("\n  Unbiased MF Validation:")
    unbiased_val = validate(model_unbiased, val_data)
    for name, value in unbiased_val.items():
        print(f"    {name:20s}: {value:.4f}")

    # 4. Optuna
    print("\n[4/6] Running Optuna hyperparameter search...")
    optuna_result = run_optuna(train_data, val_data, n_trials=10)

    # 5. Embedding analysis
    print("\n[5/6] Embedding analysis:")
    with torch.no_grad():
        user_emb = model_biased.user_embedding.weight.cpu().numpy()
        item_emb = model_biased.item_embedding.weight.cpu().numpy()
        print(f"  User embedding shape : {user_emb.shape}")
        print(f"  Item embedding shape : {item_emb.shape}")
        print(f"  User embedding norm  : {np.linalg.norm(user_emb, axis=1).mean():.4f}")
        print(f"  Item embedding norm  : {np.linalg.norm(item_emb, axis=1).mean():.4f}")

        if model_biased.use_biases:
            user_bias = model_biased.user_bias.weight.cpu().numpy().flatten()
            item_bias = model_biased.item_bias.weight.cpu().numpy().flatten()
            print(f"  Global bias          : {model_biased.global_bias.item():.4f}")
            print(f"  User bias range      : [{user_bias.min():.3f}, {user_bias.max():.3f}]")
            print(f"  Item bias range      : [{item_bias.min():.3f}, {item_bias.max():.3f}]")

    # 6. Test
    print("\n[6/6] Test set metrics:")
    for name, model_obj in [("Biased", model_biased), ("Unbiased", model_unbiased)]:
        metrics = test(model_obj, test_data)
        print(f"  {name:10s} | RMSE={metrics['rmse']:.4f} | "
              f"MAE={metrics['mae']:.4f} | NDCG@10={metrics['ndcg_at_k']:.4f} | "
              f"P@10={metrics['precision_at_k']:.4f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
