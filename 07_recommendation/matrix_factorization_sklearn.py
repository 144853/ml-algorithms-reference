"""
Matrix Factorization - scikit-learn Implementation (NMF / TruncatedSVD)
========================================================================

Theory & Mathematics:
---------------------
Matrix Factorization (MF) decomposes the user-item rating matrix R (m x n) into
the product of two lower-rank matrices:

    R  ~  P * Q^T

where P is (m x k) and Q is (n x k), with k << min(m, n) being the number
of latent factors. Each row of P represents a user in the latent space and
each row of Q represents an item in the latent space.

1. Non-negative Matrix Factorization (NMF):
   - Constraint: P >= 0, Q >= 0 (all entries non-negative)
   - Objective: min_{P,Q >= 0} ||R - P * Q^T||_F^2 + alpha * (||P||_F^2 + ||Q||_F^2)
   - Solved via coordinate descent or multiplicative update rules
   - Parts-based representation: latent factors have clear positive interpretations
   - Requires non-negative input matrix (ratings 1-5 are naturally non-negative)

   Update rules (multiplicative):
     P <- P * (R * Q) / (P * Q^T * Q + epsilon)
     Q <- Q * (R^T * P) / (Q * P^T * P + epsilon)

2. Truncated SVD (Singular Value Decomposition):
   - R = U * Sigma * V^T  (exact decomposition)
   - Keep top-k singular values: R_k = U_k * Sigma_k * V_k^T
   - This gives the best rank-k approximation in Frobenius norm (Eckart-Young theorem)
   - Works on centered/normalized data; can handle negative values
   - Handles sparse matrices efficiently via Lanczos/randomized algorithms

   The predicted rating:
     r_hat(u, i) = sum_{f=1}^{k} sigma_f * u_{u,f} * v_{i,f}

   Or equivalently with P = U_k * Sigma_k, Q = V_k:
     r_hat(u, i) = P[u, :] . Q[i, :]

Business Use Cases:
-------------------
- Movie/music recommendation (Netflix Prize winning approach)
- E-commerce product recommendations
- News article personalization
- Latent feature discovery (genre, style, mood)
- Dimensionality reduction of sparse interaction data
- User segmentation based on latent factors

Advantages:
-----------
- Handles sparsity well (only factorizes observed entries in variants)
- Compact model: only store P and Q matrices
- Interpretable latent factors (especially NMF)
- Fast prediction: simple dot product
- SVD gives provably optimal rank-k approximation
- NMF ensures non-negative factors with clear interpretation
- Well-studied with theoretical guarantees

Disadvantages:
--------------
- Assumes linear latent factor interaction
- Cold-start problem for new users/items
- Standard SVD/NMF factorize the full matrix (including zeros)
  -- Need careful handling of missing values
- NMF is sensitive to initialization and may converge to local minima
- Cannot easily incorporate side information
- Offline training: not naturally online/incremental

Key Hyperparameters:
--------------------
- n_components (k): Number of latent factors
- alpha / regularization: L2 penalty strength
- init: Initialization method (random, nndsvd, nndsvda)
- max_iter: Maximum number of iterations
- solver: Algorithm variant (cd for NMF, randomized for SVD)
- tol: Convergence tolerance
- l1_ratio: Mix of L1/L2 regularization (NMF with 'cd' solver)

References:
-----------
- Lee & Seung, "Learning the parts of objects by non-negative matrix factorization" (Nature, 1999)
- Koren, Bell, Volinsky, "Matrix Factorization Techniques for Recommender Systems" (Computer, 2009)
- Halko, Martinsson, Tropp, "Finding Structure with Randomness: Probabilistic Algorithms
  for Constructing Approximate Matrix Decompositions" (SIAM Review, 2011)
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse import csr_matrix
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
    Generate synthetic sparse user-item rating matrix and split into
    train / validation / test by randomly masking entries.

    Returns
    -------
    train_matrix, val_matrix, test_matrix : np.ndarray
        Each of shape (n_users, n_items), 0 = unobserved.
    """
    rng = np.random.RandomState(random_state)

    n_factors = 10
    P = np.abs(rng.randn(n_users, n_factors)) * 0.5
    Q = np.abs(rng.randn(n_items, n_factors)) * 0.5
    raw = P @ Q.T + 1.0  # shift up to ensure positivity

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
# Model Wrapper
# ---------------------------------------------------------------------------

class MatrixFactorizationSKLearn:
    """
    Matrix Factorization using sklearn NMF or TruncatedSVD.

    Wraps sklearn decomposition models to provide a consistent interface
    for recommendation tasks.
    """

    def __init__(
        self,
        n_components: int = 20,
        method: str = "nmf",
        alpha: float = 0.1,
        l1_ratio: float = 0.0,
        max_iter: int = 200,
        init: str = "nndsvda",
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of latent factors.
        method : str
            'nmf' or 'svd'.
        alpha : float
            Regularization strength (NMF only).
        l1_ratio : float
            Mix of L1/L2 regularization (NMF with cd solver).
        max_iter : int
            Maximum iterations.
        init : str
            Initialization for NMF ('random', 'nndsvd', 'nndsvda', 'nndsvdar').
        random_state : int
            Random seed.
        """
        self.n_components = n_components
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.model = None
        self.P = None  # User factors (n_users, k)
        self.Q = None  # Item factors (n_items, k)
        self.global_mean = 0.0

    def fit(self, train_matrix: np.ndarray):
        """
        Fit the matrix factorization model.

        Parameters
        ----------
        train_matrix : np.ndarray of shape (n_users, n_items)
            Training matrix (0 = unobserved).
        """
        n_users, n_items = train_matrix.shape

        if self.method == "nmf":
            # NMF requires non-negative input
            # Fill unobserved with small positive value to avoid zeros affecting factorization
            fill_matrix = train_matrix.copy()
            mask = train_matrix > 0
            self.global_mean = train_matrix[mask].mean() if mask.sum() > 0 else 3.0
            fill_matrix[~mask] = max(self.global_mean * 0.5, 0.1)

            self.model = NMF(
                n_components=self.n_components,
                init=self.init,
                max_iter=self.max_iter,
                alpha_W=self.alpha,
                alpha_H=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
            )
            self.P = self.model.fit_transform(fill_matrix)  # (n_users, k)
            self.Q = self.model.components_.T  # (n_items, k)

        elif self.method == "svd":
            # TruncatedSVD works with sparse matrices
            mask = train_matrix > 0
            self.global_mean = train_matrix[mask].mean() if mask.sum() > 0 else 3.0

            # Mean-center the observed entries
            centered = train_matrix.copy()
            centered[mask] -= self.global_mean

            sparse_mat = csr_matrix(centered)

            self.model = TruncatedSVD(
                n_components=self.n_components,
                algorithm="randomized",
                n_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.P = self.model.fit_transform(sparse_mat)  # (n_users, k)
            self.Q = self.model.components_.T  # (n_items, k)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'nmf' or 'svd'.")

        return self

    def predict(self) -> np.ndarray:
        """
        Predict all user-item ratings.

        Returns
        -------
        predicted : np.ndarray of shape (n_users, n_items)
        """
        predicted = self.P @ self.Q.T

        if self.method == "svd":
            # Add back global mean for SVD (was centered)
            predicted += self.global_mean

        return np.clip(predicted, 1.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(train_data: np.ndarray, **hyperparams) -> MatrixFactorizationSKLearn:
    """
    Train a matrix factorization model.

    Parameters
    ----------
    train_data : np.ndarray
        Training rating matrix.
    **hyperparams
        n_components, method, alpha, l1_ratio, max_iter, init.

    Returns
    -------
    MatrixFactorizationSKLearn
    """
    model = MatrixFactorizationSKLearn(
        n_components=hyperparams.get("n_components", 20),
        method=hyperparams.get("method", "nmf"),
        alpha=hyperparams.get("alpha", 0.1),
        l1_ratio=hyperparams.get("l1_ratio", 0.0),
        max_iter=hyperparams.get("max_iter", 200),
        init=hyperparams.get("init", "nndsvda"),
        random_state=hyperparams.get("random_state", 42),
    )
    model.fit(train_data)
    return model


def validate(
    model: MatrixFactorizationSKLearn, val_data: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Validate the model. Returns metrics dict."""
    predicted = model.predict()
    return compute_metrics(val_data, predicted, k=k)


def test(
    model: MatrixFactorizationSKLearn, test_data: np.ndarray, k: int = 10
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
    method = trial.suggest_categorical("method", ["nmf", "svd"])

    params = {
        "method": method,
        "n_components": trial.suggest_int("n_components", 5, 50),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
    }

    if method == "nmf":
        params["alpha"] = trial.suggest_float("alpha", 0.001, 1.0, log=True)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        params["init"] = trial.suggest_categorical(
            "init", ["nndsvd", "nndsvda", "nndsvdar", "random"]
        )

    model = train(train_data, **params)
    metrics = validate(model, val_data)
    return metrics["rmse"]


def run_optuna(
    train_data: np.ndarray, val_data: np.ndarray, n_trials: int = 30
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
        model = train(train_data, **config)
        metrics = validate(model, val_data)
        tune.report(rmse=metrics["rmse"], mae=metrics["mae"])

    search_space = {
        "method": tune.choice(["nmf", "svd"]),
        "n_components": tune.randint(5, 51),
        "max_iter": tune.randint(100, 501),
        "alpha": tune.loguniform(0.001, 1.0),
        "l1_ratio": tune.uniform(0.0, 1.0),
        "init": tune.choice(["nndsvda", "nndsvdar", "random"]),
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
    """Run the full matrix factorization pipeline."""
    print("=" * 70)
    print("Matrix Factorization - scikit-learn Implementation (NMF / SVD)")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/6] Generating synthetic data...")
    train_data, val_data, test_data = generate_data(
        n_users=400, n_items=150, density=0.12, random_state=42
    )

    # 2. Train NMF
    print("\n[2/6] Training NMF model...")
    nmf_model = train(train_data, method="nmf", n_components=20, alpha=0.1)

    print("  NMF Validation metrics:")
    nmf_val = validate(nmf_model, val_data)
    for name, value in nmf_val.items():
        print(f"    {name:20s}: {value:.4f}")

    # 3. Train TruncatedSVD
    print("\n[3/6] Training TruncatedSVD model...")
    svd_model = train(train_data, method="svd", n_components=20)

    print("  SVD Validation metrics:")
    svd_val = validate(svd_model, val_data)
    for name, value in svd_val.items():
        print(f"    {name:20s}: {value:.4f}")

    # 4. Optuna hyperparameter search
    print("\n[4/6] Running Optuna hyperparameter search...")
    optuna_result = run_optuna(train_data, val_data, n_trials=20)

    # 5. Train best model
    print("\n[5/6] Training best model from Optuna...")
    best_model = train(train_data, **optuna_result["best_params"])

    # 6. Test
    print("\n[6/6] Test set metrics (best model):")
    test_metrics = test(best_model, test_data)
    for name, value in test_metrics.items():
        print(f"  {name:20s}: {value:.4f}")

    # Compare methods
    print("\n--- Method Comparison (Test Set) ---")
    for name, model_obj in [("NMF", nmf_model), ("SVD", svd_model), ("Best", best_model)]:
        metrics = test(model_obj, test_data)
        print(f"  {name:6s} | RMSE={metrics['rmse']:.4f} | "
              f"MAE={metrics['mae']:.4f} | NDCG@10={metrics['ndcg_at_k']:.4f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
