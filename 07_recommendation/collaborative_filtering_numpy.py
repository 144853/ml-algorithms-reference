"""
Collaborative Filtering - NumPy From-Scratch Implementation
============================================================

Theory & Mathematics:
---------------------
Collaborative Filtering predicts a user's preference for an item based on the
preferences of similar users (user-based) or similar items (item-based).
This implementation builds both variants entirely from scratch using NumPy.

User-Based CF:
    1. Compute pairwise user similarity using cosine similarity:
       sim(u, v) = (r_u . r_v) / (||r_u|| * ||r_v||)
       where r_u and r_v are the rating vectors of users u and v,
       computed only over co-rated items.

    2. Select top-K most similar users (neighbors) for each target user.

    3. Predict rating using weighted mean:
       r_hat(u, i) = mu_u + [sum_{v in N(u)} sim(u,v) * (r_{v,i} - mu_v)]
                           / [sum_{v in N(u)} |sim(u,v)|]

       where mu_u is user u's mean rating and N(u) is the set of K nearest
       neighbors of u who have rated item i.

Item-Based CF:
    1. Compute pairwise item similarity using cosine similarity:
       sim(i, j) = (r_i . r_j) / (||r_i|| * ||r_j||)
       where r_i and r_j are the column vectors of items i and j.

    2. Select top-K most similar items for each target item.

    3. Predict rating:
       r_hat(u, i) = [sum_{j in N(i)} sim(i,j) * r_{u,j}]
                   / [sum_{j in N(i)} |sim(i,j)|]

       where N(i) is the set of K nearest items to i that user u has rated.

Cosine Similarity (from scratch):
    cos(a, b) = sum(a_k * b_k) / (sqrt(sum(a_k^2)) * sqrt(sum(b_k^2)))

Business Use Cases:
-------------------
- E-commerce product recommendations
- Movie/music recommendations (Netflix, Spotify)
- Content personalization on news/media sites
- Social network friend suggestions
- Restaurant/hotel recommendations

Advantages:
-----------
- Transparent: easy to explain why an item was recommended
- No model training for KNN (lazy learner)
- Captures subtle user preferences
- Item-based CF gives stable recommendations (item similarities change slowly)

Disadvantages:
--------------
- Cold-start problem for new users and new items
- Sparsity: most users rate few items leading to unreliable similarities
- Does not scale to millions of users (user-based) without approximations
- No side information (demographics, item metadata) utilized
- Memory-based: stores full similarity matrix in RAM

Key Hyperparameters:
--------------------
- n_neighbors (K): number of nearest neighbors to consider
- mode: 'user' or 'item' based collaborative filtering
- min_common: minimum number of co-rated items/users for valid similarity
- mean_centering: whether to subtract user/item means before computing similarity

References:
-----------
- Sarwar et al., "Item-Based Collaborative Filtering" (WWW 2001)
- Herlocker et al., "Empirical Analysis of Design Choices in Neighborhood-Based CF" (2002)
- Aggarwal, "Recommender Systems: The Textbook" (Springer, 2016), Chapter 2
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
    Generate a synthetic sparse user-item rating matrix with latent structure
    and split into train / validation / test by randomly masking entries.

    Parameters
    ----------
    n_users : int
        Number of users.
    n_items : int
        Number of items.
    density : float
        Fraction of user-item pairs that have ratings.
    rating_range : tuple
        (min_rating, max_rating) inclusive.
    val_frac : float
        Fraction of observed ratings for validation.
    test_frac : float
        Fraction of observed ratings for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    train_matrix, val_matrix, test_matrix : np.ndarray
        Each of shape (n_users, n_items), 0 = unobserved.
    """
    rng = np.random.RandomState(random_state)

    # Latent factor model for realistic ratings
    n_factors = 10
    P = rng.randn(n_users, n_factors) * 0.5
    Q = rng.randn(n_items, n_factors) * 0.5
    raw = P @ Q.T + 3.0  # center around 3

    low, high = rating_range
    raw = np.clip(np.round(raw + rng.randn(n_users, n_items) * 0.3), low, high)
    raw = raw.astype(np.float32)

    # Sparsity mask
    mask = rng.rand(n_users, n_items) < density
    observed = raw * mask

    # Split
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


def _dcg(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k]
    pos = np.arange(1, len(rel) + 1)
    return float(np.sum(rel / np.log2(pos + 1)))


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
    true_matrix: np.ndarray, pred_matrix: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Compute RMSE, MAE, Precision@K, Recall@K, NDCG@K, HitRate@K."""
    mask = true_matrix > 0
    if mask.sum() == 0:
        return {m: 0.0 for m in
                ["rmse", "mae", "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"]}

    rmse = _rmse(true_matrix[mask], pred_matrix[mask])
    mae = _mae(true_matrix[mask], pred_matrix[mask])

    precs, recs, ndcgs, hrs = [], [], [], []
    for u in range(true_matrix.shape[0]):
        if (true_matrix[u] > 0).sum() == 0:
            continue
        p, r = _precision_recall_at_k(true_matrix[u], pred_matrix[u], k)
        precs.append(p)
        recs.append(r)
        ndcgs.append(_ndcg_at_k(true_matrix[u], pred_matrix[u], k))
        hrs.append(_hit_rate_at_k(true_matrix[u], pred_matrix[u], k))

    return {
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": float(np.mean(precs)) if precs else 0.0,
        "recall_at_k": float(np.mean(recs)) if recs else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate_at_k": float(np.mean(hrs)) if hrs else 0.0,
    }


# ---------------------------------------------------------------------------
# From-Scratch CF Model
# ---------------------------------------------------------------------------

class CollaborativeFilteringNumPy:
    """
    User-Based and Item-Based Collaborative Filtering from scratch.

    All similarity computation and prediction logic is implemented
    using pure NumPy operations.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        mode: str = "item",
        min_common: int = 2,
        mean_centering: bool = True,
    ):
        """
        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors (K).
        mode : str
            'user' or 'item'.
        min_common : int
            Minimum co-rated items/users required for valid similarity.
        mean_centering : bool
            Subtract user means before computing similarity.
        """
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.min_common = min_common
        self.mean_centering = mean_centering
        self.train_matrix = None
        self.user_means = None
        self.sim_matrix = None

    @staticmethod
    def _cosine_similarity_from_scratch(
        A: np.ndarray, mask_A: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix from scratch.

        Parameters
        ----------
        A : np.ndarray of shape (n, d)
            Rating vectors (already mean-centered if desired).
        mask_A : np.ndarray of shape (n, d), bool
            True where rating is observed.

        Returns
        -------
        sim : np.ndarray of shape (n, n)
            Cosine similarity matrix.
        """
        n = A.shape[0]
        sim = np.zeros((n, n), dtype=np.float64)

        # Precompute norms
        norms = np.sqrt(np.sum(A ** 2, axis=1))

        for i in range(n):
            for j in range(i + 1, n):
                # Find co-rated dimensions
                common = mask_A[i] & mask_A[j]
                if common.sum() < 2:
                    continue
                a_i = A[i, common]
                a_j = A[j, common]
                dot_product = np.dot(a_i, a_j)
                norm_i = np.sqrt(np.sum(a_i ** 2))
                norm_j = np.sqrt(np.sum(a_j ** 2))
                if norm_i > 0 and norm_j > 0:
                    sim[i, j] = dot_product / (norm_i * norm_j)
                    sim[j, i] = sim[i, j]
        return sim

    def fit(self, train_matrix: np.ndarray):
        """
        Compute the similarity matrix.

        Parameters
        ----------
        train_matrix : np.ndarray of shape (n_users, n_items)
        """
        self.train_matrix = train_matrix.copy()
        n_users, n_items = train_matrix.shape

        # Compute user means
        self.user_means = np.zeros(n_users, dtype=np.float64)
        for u in range(n_users):
            rated = train_matrix[u] > 0
            if rated.sum() > 0:
                self.user_means[u] = train_matrix[u, rated].mean()

        if self.mode == "item":
            # Item-based: compute item-item similarity
            # Item vectors: columns of the rating matrix
            item_mat = train_matrix.T.copy()  # (n_items, n_users)
            item_mask = (train_matrix.T > 0)

            if self.mean_centering:
                # Subtract user means from each rating (per column = per user)
                for u in range(n_users):
                    rated = item_mask[:, u]
                    item_mat[rated, u] -= self.user_means[u]

            self.sim_matrix = self._cosine_similarity_from_scratch(
                item_mat, item_mask
            )
        else:
            # User-based: compute user-user similarity
            user_mat = train_matrix.copy()  # (n_users, n_items)
            user_mask = (train_matrix > 0)

            if self.mean_centering:
                for u in range(n_users):
                    rated = user_mask[u]
                    user_mat[u, rated] -= self.user_means[u]

            self.sim_matrix = self._cosine_similarity_from_scratch(
                user_mat, user_mask
            )
        return self

    def predict(self) -> np.ndarray:
        """
        Predict all ratings using KNN from the similarity matrix.

        Returns
        -------
        predicted : np.ndarray of shape (n_users, n_items)
        """
        n_users, n_items = self.train_matrix.shape
        predicted = np.full((n_users, n_items), fill_value=0.0, dtype=np.float64)

        if self.mode == "item":
            # For each item, find K nearest items
            for i in range(n_items):
                sims = self.sim_matrix[i].copy()
                sims[i] = -np.inf  # exclude self

                # Top K neighbors
                if self.n_neighbors < n_items - 1:
                    top_k = np.argsort(sims)[::-1][: self.n_neighbors]
                else:
                    top_k = np.argsort(sims)[::-1]

                for u in range(n_users):
                    # Among neighbors, keep only those rated by user u
                    neighbor_ratings = self.train_matrix[u, top_k]
                    rated_mask = neighbor_ratings > 0
                    if rated_mask.sum() == 0:
                        predicted[u, i] = self.user_means[u]
                        continue
                    k_sims = sims[top_k[rated_mask]]
                    k_rats = neighbor_ratings[rated_mask]
                    denom = np.abs(k_sims).sum()
                    if denom > 1e-10:
                        predicted[u, i] = (k_sims * k_rats).sum() / denom
                    else:
                        predicted[u, i] = self.user_means[u]
        else:
            # User-based: for each user, find K nearest users
            for u in range(n_users):
                sims = self.sim_matrix[u].copy()
                sims[u] = -np.inf

                if self.n_neighbors < n_users - 1:
                    top_k = np.argsort(sims)[::-1][: self.n_neighbors]
                else:
                    top_k = np.argsort(sims)[::-1]

                for i in range(n_items):
                    neighbor_ratings = self.train_matrix[top_k, i]
                    rated_mask = neighbor_ratings > 0
                    if rated_mask.sum() == 0:
                        predicted[u, i] = self.user_means[u]
                        continue
                    k_sims = sims[top_k[rated_mask]]
                    k_rats = neighbor_ratings[rated_mask]
                    k_means = self.user_means[top_k[rated_mask]]
                    denom = np.abs(k_sims).sum()
                    if denom > 1e-10:
                        predicted[u, i] = self.user_means[u] + (
                            k_sims * (k_rats - k_means)
                        ).sum() / denom
                    else:
                        predicted[u, i] = self.user_means[u]

        predicted = np.clip(predicted, 1.0, 5.0)
        return predicted


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(train_data: np.ndarray, **hyperparams) -> CollaborativeFilteringNumPy:
    """Train a CF model from scratch."""
    model = CollaborativeFilteringNumPy(
        n_neighbors=hyperparams.get("n_neighbors", 20),
        mode=hyperparams.get("mode", "item"),
        min_common=hyperparams.get("min_common", 2),
        mean_centering=hyperparams.get("mean_centering", True),
    )
    model.fit(train_data)
    return model


def validate(
    model: CollaborativeFilteringNumPy, val_data: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Validate the model. Returns metrics dict."""
    predicted = model.predict()
    return compute_metrics(val_data, predicted, k=k)


def test(
    model: CollaborativeFilteringNumPy, test_data: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Test evaluation. Returns metrics dict."""
    predicted = model.predict()
    return compute_metrics(test_data, predicted, k=k)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial, train_data: np.ndarray, val_data: np.ndarray
) -> float:
    """Optuna objective: minimize validation RMSE."""
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 40),
        "mode": trial.suggest_categorical("mode", ["user", "item"]),
        "mean_centering": trial.suggest_categorical("mean_centering", [True, False]),
        "min_common": trial.suggest_int("min_common", 1, 5),
    }
    model = train(train_data, **params)
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
        model = train(train_data, **config)
        metrics = validate(model, val_data)
        tune.report(rmse=metrics["rmse"], mae=metrics["mae"])

    search_space = {
        "n_neighbors": tune.randint(5, 41),
        "mode": tune.choice(["user", "item"]),
        "mean_centering": tune.choice([True, False]),
        "min_common": tune.randint(1, 6),
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
    """Run the full CF from-scratch pipeline."""
    print("=" * 70)
    print("Collaborative Filtering - NumPy From-Scratch Implementation")
    print("=" * 70)

    # 1. Generate data (smaller for from-scratch computation)
    print("\n[1/5] Generating synthetic data...")
    train_data, val_data, test_data = generate_data(
        n_users=150, n_items=80, density=0.15, random_state=42
    )

    # 2. Train default
    print("\n[2/5] Training item-based CF (K=15)...")
    model = train(train_data, n_neighbors=15, mode="item", mean_centering=True)

    # 3. Validate
    print("\n[3/5] Validation metrics:")
    val_metrics = validate(model, val_data)
    for name, value in val_metrics.items():
        print(f"  {name:20s}: {value:.4f}")

    # 4. Optuna
    print("\n[4/5] Running Optuna hyperparameter search...")
    optuna_result = run_optuna(train_data, val_data, n_trials=10)
    best_model = train(train_data, **optuna_result["best_params"])

    # 5. Test
    print("\n[5/5] Test set metrics (best model):")
    test_metrics = test(best_model, test_data)
    for name, value in test_metrics.items():
        print(f"  {name:20s}: {value:.4f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
