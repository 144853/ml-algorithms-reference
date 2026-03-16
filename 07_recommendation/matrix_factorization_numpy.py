"""
Matrix Factorization - NumPy From-Scratch Implementation (ALS & SGD)
=====================================================================

Theory & Mathematics:
---------------------
Matrix Factorization decomposes the sparse user-item rating matrix R (m x n)
into two low-rank matrices:

    R  ~  P * Q^T

where:
    P is (m x k): each row p_u is the latent factor vector for user u
    Q is (n x k): each row q_i is the latent factor vector for item i
    k: number of latent factors (dimensionality of the latent space)

The predicted rating for user u on item i is:
    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

where mu is the global mean, b_u is user bias, b_i is item bias.

This implementation provides two optimization algorithms from scratch:

1. Alternating Least Squares (ALS):
   ------------------------------------
   Alternates between fixing P and solving for Q, then fixing Q and solving for P.

   When Q is fixed, each row of P is computed via a regularized least squares:
     p_u = (Q_Iu^T Q_Iu + lambda * I)^{-1} * Q_Iu^T * (r_u - biases)

   where Q_Iu is the submatrix of Q for items rated by user u, and r_u is
   the vector of ratings by user u.

   Similarly when P is fixed:
     q_i = (P_Ui^T P_Ui + lambda * I)^{-1} * P_Ui^T * (r_i - biases)

   Advantages of ALS:
   - Each sub-problem is convex (closed-form solution)
   - Parallelizable: each user/item update is independent
   - Handles implicit feedback well
   - Guaranteed convergence (monotonic decrease in objective)

2. Stochastic Gradient Descent (SGD):
   ------------------------------------
   For each observed rating r_{u,i}, compute the prediction error:
     e_{u,i} = r_{u,i} - r_hat(u, i)

   Update rules:
     p_u <- p_u + lr * (e_{u,i} * q_i - lambda * p_u)
     q_i <- q_i + lr * (e_{u,i} * p_u - lambda * q_i)
     b_u <- b_u + lr * (e_{u,i} - lambda * b_u)
     b_i <- b_i + lr * (e_{u,i} - lambda * b_i)

   Advantages of SGD:
   - Simple implementation
   - Fast convergence on sparse data
   - Memory efficient (processes one rating at a time)
   - Easy to extend with additional features

Objective function (both methods minimize):
    L = sum_{(u,i) in Omega} (r_{u,i} - mu - b_u - b_i - p_u . q_i)^2
        + lambda * (||P||_F^2 + ||Q||_F^2 + sum b_u^2 + sum b_i^2)

where Omega is the set of observed ratings.

Business Use Cases:
-------------------
- Movie/music recommendation (Netflix Prize, Spotify Discover Weekly)
- E-commerce product recommendation
- Content personalization
- Latent feature discovery (automatically learns genres, styles, etc.)
- User profiling and segmentation
- Demand forecasting in retail

Advantages:
-----------
- Handles sparsity: only factorizes observed entries
- Compact model: O(k * (m + n)) parameters vs O(m * n) ratings
- Interpretable latent factors
- Fast prediction: simple dot product O(k)
- ALS is embarrassingly parallelizable
- Can incorporate biases for better accuracy

Disadvantages:
--------------
- Linear model: cannot capture non-linear interactions
- Cold-start problem for new users/items
- Latent factors may not be individually interpretable
- SGD: sensitive to learning rate, may oscillate
- ALS: requires matrix inversion per user/item (O(k^3))
- Assumes ratings are missing at random (MAR)

Key Hyperparameters:
--------------------
- n_factors (k): Number of latent dimensions
- regularization (lambda): L2 penalty strength
- learning_rate (SGD only): Step size for gradient updates
- n_epochs: Number of training iterations
- optimizer: 'als' or 'sgd'
- lr_decay: Learning rate decay factor (SGD only)
- init_std: Standard deviation for random initialization

References:
-----------
- Koren, Bell, Volinsky, "Matrix Factorization Techniques for Recommender Systems" (IEEE Computer, 2009)
- Hu, Koren, Volinsky, "Collaborative Filtering for Implicit Feedback Datasets" (ICDM 2008)
- Funk, "Netflix Update: Try This at Home" (2006, Funk SVD blog post)
- Zhou et al., "Large-Scale Parallel Collaborative Filtering for the Netflix Prize" (2008)
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional
import optuna
import ray
from ray import tune

warnings.filterwarnings("ignore")


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
    Generate synthetic user-item rating matrix with latent structure and
    split into train / val / test by randomly masking entries.

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
# Matrix Factorization From Scratch
# ---------------------------------------------------------------------------

class MatrixFactorizationNumPy:
    """
    Matrix Factorization from scratch using ALS or SGD.

    Learns user factor matrix P (m x k), item factor matrix Q (n x k),
    user biases b_u, item biases b_i, and global mean mu.

    Prediction: r_hat(u,i) = mu + b_u + b_i + p_u . q_i
    """

    def __init__(
        self,
        n_factors: int = 20,
        regularization: float = 0.02,
        learning_rate: float = 0.005,
        n_epochs: int = 50,
        optimizer: str = "sgd",
        lr_decay: float = 0.95,
        init_std: float = 0.1,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors (k).
        regularization : float
            L2 regularization coefficient (lambda).
        learning_rate : float
            SGD learning rate (ignored for ALS).
        n_epochs : int
            Number of training epochs.
        optimizer : str
            'sgd' or 'als'.
        lr_decay : float
            Learning rate decay per epoch (SGD only).
        init_std : float
            Standard deviation for random initialization.
        random_state : int
            Random seed.
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.init_std = init_std
        self.random_state = random_state

        self.P = None
        self.Q = None
        self.b_u = None
        self.b_i = None
        self.mu = 0.0
        self.n_users = 0
        self.n_items = 0
        self.train_loss_history = []

    def fit(
        self,
        train_matrix: np.ndarray,
        val_matrix: Optional[np.ndarray] = None,
        patience: int = 5,
        verbose: bool = True,
    ):
        """
        Train the model using ALS or SGD.

        Parameters
        ----------
        train_matrix : np.ndarray of shape (n_users, n_items)
            Training rating matrix (0 = unobserved).
        val_matrix : np.ndarray, optional
            Validation matrix for early stopping.
        patience : int
            Early stopping patience (epochs without improvement).
        verbose : bool
            Print training progress.
        """
        rng = np.random.RandomState(self.random_state)
        self.n_users, self.n_items = train_matrix.shape

        # Initialize
        self.P = rng.normal(0, self.init_std, (self.n_users, self.n_factors))
        self.Q = rng.normal(0, self.init_std, (self.n_items, self.n_factors))
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)

        # Global mean
        mask = train_matrix > 0
        self.mu = train_matrix[mask].mean() if mask.sum() > 0 else 3.0

        # Get observed indices
        observed_users, observed_items = np.where(mask)
        observed_ratings = train_matrix[mask]

        best_val_rmse = float("inf")
        best_state = None
        no_improve_count = 0
        self.train_loss_history = []

        for epoch in range(self.n_epochs):
            if self.optimizer == "sgd":
                train_loss = self._sgd_epoch(
                    observed_users, observed_items, observed_ratings, rng, epoch
                )
            elif self.optimizer == "als":
                train_loss = self._als_epoch(train_matrix, mask)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

            self.train_loss_history.append(train_loss)

            # Validation
            if val_matrix is not None:
                pred = self.predict()
                val_mask = val_matrix > 0
                if val_mask.sum() > 0:
                    val_rmse = _rmse(val_matrix[val_mask], pred[val_mask])
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_state = (
                            self.P.copy(), self.Q.copy(),
                            self.b_u.copy(), self.b_i.copy(),
                        )
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                        print(f"  Epoch {epoch+1:3d}/{self.n_epochs} | "
                              f"Loss={train_loss:.4f} | ValRMSE={val_rmse:.4f}")

                    if no_improve_count >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                    print(f"  Epoch {epoch+1:3d}/{self.n_epochs} | Loss={train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.P, self.Q, self.b_u, self.b_i = best_state

        return self

    def _sgd_epoch(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        rng: np.random.RandomState,
        epoch: int,
    ) -> float:
        """
        One epoch of SGD updates.

        For each observed rating (u, i, r):
            e = r - (mu + b_u + b_i + p_u . q_i)
            p_u += lr * (e * q_i - lambda * p_u)
            q_i += lr * (e * p_u - lambda * q_i)
            b_u += lr * (e - lambda * b_u)
            b_i += lr * (e - lambda * b_i)
        """
        lr = self.learning_rate * (self.lr_decay ** epoch)
        reg = self.regularization

        # Shuffle training data
        indices = rng.permutation(len(ratings))
        total_loss = 0.0

        for idx in indices:
            u = users[idx]
            i = items[idx]
            r = ratings[idx]

            # Prediction
            pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
            error = r - pred
            total_loss += error ** 2

            # Update biases
            self.b_u[u] += lr * (error - reg * self.b_u[u])
            self.b_i[i] += lr * (error - reg * self.b_i[i])

            # Update latent factors
            p_u_old = self.P[u].copy()
            self.P[u] += lr * (error * self.Q[i] - reg * self.P[u])
            self.Q[i] += lr * (error * p_u_old - reg * self.Q[i])

        # Add regularization to loss
        total_loss += reg * (
            np.sum(self.P ** 2)
            + np.sum(self.Q ** 2)
            + np.sum(self.b_u ** 2)
            + np.sum(self.b_i ** 2)
        )
        return total_loss / len(ratings)

    def _als_epoch(
        self, train_matrix: np.ndarray, mask: np.ndarray
    ) -> float:
        """
        One epoch of Alternating Least Squares.

        Step 1: Fix Q, solve for each p_u:
            p_u = (Q_Iu^T Q_Iu + lambda*I)^{-1} Q_Iu^T (r_u - mu - b_u - b_Iu)

        Step 2: Fix P, solve for each q_i:
            q_i = (P_Ui^T P_Ui + lambda*I)^{-1} P_Ui^T (r_i - mu - b_Ui - b_i)

        Step 3: Update biases.
        """
        reg = self.regularization
        identity = np.eye(self.n_factors)

        # Step 1: Update P (fix Q)
        for u in range(self.n_users):
            rated_items = np.where(mask[u])[0]
            if len(rated_items) == 0:
                continue
            Q_Iu = self.Q[rated_items]  # (|I_u|, k)
            r_u = train_matrix[u, rated_items] - self.mu - self.b_u[u] - self.b_i[rated_items]
            # Solve: (Q^T Q + lambda*I) p = Q^T r
            A = Q_Iu.T @ Q_Iu + reg * len(rated_items) * identity
            b = Q_Iu.T @ r_u
            self.P[u] = np.linalg.solve(A, b)

        # Step 2: Update Q (fix P)
        for i in range(self.n_items):
            rated_users = np.where(mask[:, i])[0]
            if len(rated_users) == 0:
                continue
            P_Ui = self.P[rated_users]  # (|U_i|, k)
            r_i = train_matrix[rated_users, i] - self.mu - self.b_u[rated_users] - self.b_i[i]
            A = P_Ui.T @ P_Ui + reg * len(rated_users) * identity
            b = P_Ui.T @ r_i
            self.Q[i] = np.linalg.solve(A, b)

        # Step 3: Update biases
        for u in range(self.n_users):
            rated_items = np.where(mask[u])[0]
            if len(rated_items) == 0:
                continue
            residuals = (
                train_matrix[u, rated_items]
                - self.mu
                - self.b_i[rated_items]
                - self.P[u] @ self.Q[rated_items].T
            )
            self.b_u[u] = residuals.sum() / (len(rated_items) + reg)

        for i in range(self.n_items):
            rated_users = np.where(mask[:, i])[0]
            if len(rated_users) == 0:
                continue
            residuals = (
                train_matrix[rated_users, i]
                - self.mu
                - self.b_u[rated_users]
                - self.P[rated_users] @ self.Q[i]
            )
            self.b_i[i] = residuals.sum() / (len(rated_users) + reg)

        # Compute training loss
        pred = self.predict()
        errors = train_matrix[mask] - pred[mask]
        loss = np.mean(errors ** 2) + reg * (
            np.sum(self.P ** 2) + np.sum(self.Q ** 2)
            + np.sum(self.b_u ** 2) + np.sum(self.b_i ** 2)
        ) / mask.sum()
        return float(loss)

    def predict(self) -> np.ndarray:
        """
        Predict all ratings: r_hat(u,i) = mu + b_u + b_i + p_u . q_i

        Returns
        -------
        predicted : np.ndarray of shape (n_users, n_items)
        """
        predicted = (
            self.mu
            + self.b_u[:, np.newaxis]
            + self.b_i[np.newaxis, :]
            + self.P @ self.Q.T
        )
        return np.clip(predicted, 1.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(
    train_data: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    **hyperparams,
) -> MatrixFactorizationNumPy:
    """
    Train a matrix factorization model from scratch.

    Parameters
    ----------
    train_data : np.ndarray
        Training rating matrix.
    val_data : np.ndarray, optional
        For early stopping.
    **hyperparams
        n_factors, regularization, learning_rate, n_epochs, optimizer, etc.

    Returns
    -------
    MatrixFactorizationNumPy
    """
    model = MatrixFactorizationNumPy(
        n_factors=hyperparams.get("n_factors", 20),
        regularization=hyperparams.get("regularization", 0.02),
        learning_rate=hyperparams.get("learning_rate", 0.005),
        n_epochs=hyperparams.get("n_epochs", 50),
        optimizer=hyperparams.get("optimizer", "sgd"),
        lr_decay=hyperparams.get("lr_decay", 0.95),
        init_std=hyperparams.get("init_std", 0.1),
        random_state=hyperparams.get("random_state", 42),
    )
    model.fit(
        train_data,
        val_matrix=val_data,
        patience=hyperparams.get("patience", 7),
        verbose=hyperparams.get("verbose", True),
    )
    return model


def validate(
    model: MatrixFactorizationNumPy, val_data: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Validate the model. Returns metrics dict."""
    predicted = model.predict()
    return compute_metrics(val_data, predicted, k=k)


def test(
    model: MatrixFactorizationNumPy, test_data: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Test evaluation. Returns metrics dict."""
    predicted = model.predict()
    return compute_metrics(test_data, predicted, k=k)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> float:
    """Optuna objective: minimize validation RMSE."""
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "als"])

    params = {
        "n_factors": trial.suggest_int("n_factors", 5, 50),
        "regularization": trial.suggest_float("regularization", 0.001, 0.5, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 20, 100),
        "optimizer": optimizer,
        "init_std": trial.suggest_float("init_std", 0.01, 0.5, log=True),
        "verbose": False,
        "patience": 7,
    }

    if optimizer == "sgd":
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 0.05, log=True)
        params["lr_decay"] = trial.suggest_float("lr_decay", 0.9, 1.0)

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
    train_data: np.ndarray, val_data: np.ndarray, num_samples: int = 20
) -> Dict:
    """Run Ray Tune hyperparameter search."""
    ray.init(ignore_reinit_error=True, num_cpus=4)

    def trainable(config):
        model = train(train_data, val_data=val_data, verbose=False, **config)
        metrics = validate(model, val_data)
        tune.report(rmse=metrics["rmse"], mae=metrics["mae"])

    search_space = {
        "n_factors": tune.randint(5, 51),
        "regularization": tune.loguniform(0.001, 0.5),
        "n_epochs": tune.randint(20, 101),
        "optimizer": tune.choice(["sgd", "als"]),
        "init_std": tune.loguniform(0.01, 0.5),
        "learning_rate": tune.loguniform(1e-4, 0.05),
        "lr_decay": tune.uniform(0.9, 1.0),
        "patience": 7,
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
    """Run the full matrix factorization from-scratch pipeline."""
    print("=" * 70)
    print("Matrix Factorization - NumPy From-Scratch (ALS & SGD)")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/7] Generating synthetic data...")
    train_data, val_data, test_data = generate_data(
        n_users=400, n_items=150, density=0.12, random_state=42
    )

    # 2. Train SGD model
    print("\n[2/7] Training SGD model...")
    sgd_model = train(
        train_data,
        val_data=val_data,
        n_factors=20,
        regularization=0.02,
        learning_rate=0.005,
        n_epochs=60,
        optimizer="sgd",
        lr_decay=0.95,
    )

    print("\n  SGD Validation metrics:")
    sgd_val = validate(sgd_model, val_data)
    for name, value in sgd_val.items():
        print(f"    {name:20s}: {value:.4f}")

    # 3. Train ALS model
    print("\n[3/7] Training ALS model...")
    als_model = train(
        train_data,
        val_data=val_data,
        n_factors=20,
        regularization=0.05,
        n_epochs=30,
        optimizer="als",
    )

    print("\n  ALS Validation metrics:")
    als_val = validate(als_model, val_data)
    for name, value in als_val.items():
        print(f"    {name:20s}: {value:.4f}")

    # 4. Compare on validation
    print("\n[4/7] SGD vs ALS comparison (validation):")
    print(f"  SGD RMSE: {sgd_val['rmse']:.4f} | ALS RMSE: {als_val['rmse']:.4f}")

    # 5. Optuna
    print("\n[5/7] Running Optuna hyperparameter search...")
    optuna_result = run_optuna(train_data, val_data, n_trials=15)

    # 6. Train best
    print("\n[6/7] Training best model from Optuna...")
    best_params = optuna_result["best_params"].copy()
    best_params["verbose"] = True
    best_params["patience"] = 7
    if best_params["optimizer"] == "als":
        best_params.pop("learning_rate", None)
        best_params.pop("lr_decay", None)
    best_model = train(train_data, val_data=val_data, **best_params)

    # 7. Test
    print("\n[7/7] Test set metrics:")
    for name, model_obj in [("SGD", sgd_model), ("ALS", als_model), ("Best", best_model)]:
        metrics = test(model_obj, test_data)
        print(f"  {name:6s} | RMSE={metrics['rmse']:.4f} | "
              f"MAE={metrics['mae']:.4f} | NDCG@10={metrics['ndcg_at_k']:.4f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
