"""
Collaborative Filtering - scikit-learn Implementation
======================================================

Theory & Mathematics:
---------------------
Collaborative Filtering (CF) is a recommendation technique that makes predictions
about a user's interests by collecting preference information from many users.
The core assumption is that users who agreed in the past will agree in the future.

There are two main approaches:

1. User-Based CF:
   - Find users similar to the target user
   - Predict ratings based on what similar users rated
   - Similarity: sim(u, v) = cos(r_u, r_v) = (r_u . r_v) / (||r_u|| * ||r_v||)
   - Prediction: r_hat(u, i) = r_bar_u + [sum_v sim(u,v) * (r_vi - r_bar_v)] / [sum_v |sim(u,v)|]

2. Item-Based CF:
   - Find items similar to items the user has rated
   - Predict ratings based on the user's ratings of similar items
   - Similarity computed between item rating vectors
   - Prediction: r_hat(u, i) = [sum_j sim(i,j) * r_uj] / [sum_j |sim(i,j)|]

This implementation uses sklearn's NearestNeighbors to find similar users/items
efficiently using approximate or exact nearest neighbor search with cosine distance.

Business Use Cases:
-------------------
- E-commerce product recommendations (Amazon-style "customers who bought...")
- Movie/music recommendations (Netflix, Spotify)
- Content personalization on news/media sites
- Social network friend suggestions
- Ad targeting and click-through prediction

Advantages:
-----------
- Intuitive and easy to explain
- No need for item/user feature engineering
- Can capture subtle preferences
- Item-based CF is highly scalable (item similarities are stable)
- Works well when user-item interaction data is abundant

Disadvantages:
--------------
- Cold-start problem for new users/items
- Scalability issues for user-based CF with millions of users
- Sparsity: most users rate very few items
- Popularity bias: tends to recommend popular items
- Cannot incorporate side information (user demographics, item features)

Key Hyperparameters:
--------------------
- n_neighbors (K): Number of similar users/items to consider
- metric: Distance metric (cosine, euclidean, manhattan)
- algorithm: NearestNeighbors algorithm (brute, ball_tree, kd_tree)
- mode: user-based vs item-based
- min_common: Minimum common ratings required for valid similarity

References:
-----------
- Sarwar et al., "Item-Based Collaborative Filtering Recommendation Algorithms" (2001)
- Herlocker et al., "An Algorithmic Framework for Performing Collaborative Filtering" (1999)
- Koren et al., "Matrix Factorization Techniques for Recommender Systems" (2009)
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    Generate a synthetic sparse user-item rating matrix and split into
    train / validation / test sets by randomly masking entries.

    Parameters
    ----------
    n_users : int
        Number of users.
    n_items : int
        Number of items.
    density : float
        Fraction of (user, item) pairs that have ratings (0 < density <= 1).
    rating_range : tuple
        (min_rating, max_rating) inclusive.
    val_frac : float
        Fraction of observed ratings held out for validation.
    test_frac : float
        Fraction of observed ratings held out for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    train_matrix : np.ndarray of shape (n_users, n_items)
        Training ratings (0 = unobserved).
    val_matrix : np.ndarray of shape (n_users, n_items)
        Validation ratings (0 = unobserved).
    test_matrix : np.ndarray of shape (n_users, n_items)
        Test ratings (0 = unobserved).
    """
    rng = np.random.RandomState(random_state)

    # Create a full rating matrix with latent structure for realism
    n_factors = 10
    user_factors = rng.randn(n_users, n_factors)
    item_factors = rng.randn(n_items, n_factors)
    full_ratings = user_factors @ item_factors.T

    # Scale to rating range
    low, high = rating_range
    full_ratings = (full_ratings - full_ratings.min()) / (
        full_ratings.max() - full_ratings.min()
    )
    full_ratings = np.round(full_ratings * (high - low) + low).astype(np.float32)
    full_ratings = np.clip(full_ratings, low, high)

    # Add noise
    noise = rng.randn(n_users, n_items) * 0.5
    full_ratings = np.clip(np.round(full_ratings + noise), low, high).astype(
        np.float32
    )

    # Apply sparsity mask
    mask = rng.rand(n_users, n_items) < density
    observed_ratings = full_ratings * mask

    # Split observed entries into train / val / test
    observed_indices = np.argwhere(mask)
    rng.shuffle(observed_indices)
    n_observed = len(observed_indices)
    n_val = int(n_observed * val_frac)
    n_test = int(n_observed * test_frac)

    val_indices = observed_indices[:n_val]
    test_indices = observed_indices[n_val : n_val + n_test]
    train_indices = observed_indices[n_val + n_test :]

    train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    val_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    test_matrix = np.zeros((n_users, n_items), dtype=np.float32)

    for u, i in train_indices:
        train_matrix[u, i] = observed_ratings[u, i]
    for u, i in val_indices:
        val_matrix[u, i] = observed_ratings[u, i]
    for u, i in test_indices:
        test_matrix[u, i] = observed_ratings[u, i]

    print(f"Data generated: {n_users} users x {n_items} items")
    print(f"  Observed ratings : {n_observed}")
    print(f"  Train : {len(train_indices)}")
    print(f"  Val   : {n_val}")
    print(f"  Test  : {n_test}")
    return train_matrix, val_matrix, test_matrix


# ---------------------------------------------------------------------------
# Recommendation Metrics
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _dcg(relevances: np.ndarray, k: int) -> float:
    relevances = relevances[:k]
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def _ndcg_at_k(actual: np.ndarray, predicted_scores: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@K for a single user."""
    top_k_idx = np.argsort(predicted_scores)[::-1][:k]
    relevances = actual[top_k_idx]
    dcg = _dcg(relevances, k)
    ideal_relevances = np.sort(actual)[::-1]
    idcg = _dcg(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _precision_recall_at_k(
    actual: np.ndarray,
    predicted_scores: np.ndarray,
    k: int = 10,
    threshold: float = 3.5,
) -> Tuple[float, float]:
    """Compute Precision@K and Recall@K for a single user."""
    top_k_idx = np.argsort(predicted_scores)[::-1][:k]
    relevant_items = set(np.where(actual >= threshold)[0])
    if len(relevant_items) == 0:
        return 0.0, 0.0
    recommended_relevant = len(set(top_k_idx) & relevant_items)
    precision = recommended_relevant / k
    recall = recommended_relevant / len(relevant_items)
    return precision, recall


def _hit_rate_at_k(
    actual: np.ndarray,
    predicted_scores: np.ndarray,
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    """Compute Hit Rate@K for a single user."""
    top_k_idx = np.argsort(predicted_scores)[::-1][:k]
    relevant_items = set(np.where(actual >= threshold)[0])
    return 1.0 if len(set(top_k_idx) & relevant_items) > 0 else 0.0


def compute_metrics(
    rating_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    k: int = 10,
    threshold: float = 3.5,
) -> Dict[str, float]:
    """
    Compute recommendation metrics on observed entries.

    Parameters
    ----------
    rating_matrix : np.ndarray
        Ground truth rating matrix (0 = unobserved).
    predicted_matrix : np.ndarray
        Predicted rating matrix (dense).
    k : int
        Cut-off for ranking metrics.
    threshold : float
        Rating threshold to consider an item as relevant.

    Returns
    -------
    dict with RMSE, MAE, Precision@K, Recall@K, NDCG@K, HitRate@K.
    """
    mask = rating_matrix > 0
    if mask.sum() == 0:
        return {
            "rmse": float("inf"),
            "mae": float("inf"),
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate_at_k": 0.0,
        }

    y_true = rating_matrix[mask]
    y_pred = predicted_matrix[mask]

    rmse = _rmse(y_true, y_pred)
    mae = _mae(y_true, y_pred)

    # Ranking metrics per user
    precisions, recalls, ndcgs, hits = [], [], [], []
    n_users = rating_matrix.shape[0]
    for u in range(n_users):
        user_mask = rating_matrix[u] > 0
        if user_mask.sum() == 0:
            continue
        actual = rating_matrix[u].copy()
        preds = predicted_matrix[u].copy()
        p, r = _precision_recall_at_k(actual, preds, k, threshold)
        n = _ndcg_at_k(actual, preds, k)
        h = _hit_rate_at_k(actual, preds, k, threshold)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)
        hits.append(h)

    return {
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate_at_k": float(np.mean(hits)) if hits else 0.0,
    }


# ---------------------------------------------------------------------------
# Model: Collaborative Filtering with sklearn NearestNeighbors
# ---------------------------------------------------------------------------

class CollaborativeFilteringSKLearn:
    """
    Collaborative Filtering using sklearn NearestNeighbors.

    Supports both user-based and item-based modes.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "cosine",
        algorithm: str = "brute",
        mode: str = "item",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.mode = mode  # "user" or "item"
        self.nn_model = None
        self.train_matrix = None
        self.user_means = None

    def fit(self, train_matrix: np.ndarray):
        """
        Fit the NearestNeighbors model on the training rating matrix.

        Parameters
        ----------
        train_matrix : np.ndarray of shape (n_users, n_items)
            Training rating matrix (0 = unobserved).
        """
        self.train_matrix = train_matrix.copy()
        self.user_means = np.zeros(train_matrix.shape[0])
        for u in range(train_matrix.shape[0]):
            rated = train_matrix[u] > 0
            if rated.sum() > 0:
                self.user_means[u] = train_matrix[u, rated].mean()

        if self.mode == "item":
            # Item-based: find neighbors among items
            # Use item x user matrix so each row is an item
            data = csr_matrix(train_matrix.T)
        else:
            # User-based: find neighbors among users
            data = csr_matrix(train_matrix)

        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, data.shape[0]),
            metric=self.metric,
            algorithm=self.algorithm,
        )
        self.nn_model.fit(data)
        return self

    def predict(self, train_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict ratings for all user-item pairs.

        Returns
        -------
        predicted : np.ndarray of shape (n_users, n_items)
        """
        if train_matrix is None:
            train_matrix = self.train_matrix

        n_users, n_items = train_matrix.shape
        predicted = np.full((n_users, n_items), fill_value=0.0, dtype=np.float32)

        if self.mode == "item":
            # Item-based prediction
            item_data = csr_matrix(train_matrix.T)
            distances, indices = self.nn_model.kneighbors(item_data)

            for i in range(n_items):
                neighbor_items = indices[i]
                neighbor_dists = distances[i]

                # Convert cosine distance to similarity
                neighbor_sims = 1.0 - neighbor_dists
                # Exclude self (first neighbor is usually self)
                valid = neighbor_items != i
                neighbor_items = neighbor_items[valid]
                neighbor_sims = neighbor_sims[valid]

                if len(neighbor_items) == 0:
                    continue

                for u in range(n_users):
                    # Weighted average of user's ratings on neighbor items
                    neighbor_ratings = train_matrix[u, neighbor_items]
                    rated_mask = neighbor_ratings > 0
                    if rated_mask.sum() == 0:
                        predicted[u, i] = self.user_means[u]
                        continue
                    sims = neighbor_sims[rated_mask]
                    rats = neighbor_ratings[rated_mask]
                    denom = np.abs(sims).sum()
                    if denom > 0:
                        predicted[u, i] = (sims * rats).sum() / denom
                    else:
                        predicted[u, i] = self.user_means[u]
        else:
            # User-based prediction
            user_data = csr_matrix(train_matrix)
            distances, indices = self.nn_model.kneighbors(user_data)

            for u in range(n_users):
                neighbor_users = indices[u]
                neighbor_dists = distances[u]
                neighbor_sims = 1.0 - neighbor_dists

                valid = neighbor_users != u
                neighbor_users = neighbor_users[valid]
                neighbor_sims = neighbor_sims[valid]

                if len(neighbor_users) == 0:
                    continue

                for i in range(n_items):
                    neighbor_ratings = train_matrix[neighbor_users, i]
                    rated_mask = neighbor_ratings > 0
                    if rated_mask.sum() == 0:
                        predicted[u, i] = self.user_means[u]
                        continue
                    sims = neighbor_sims[rated_mask]
                    rats = neighbor_ratings[rated_mask]
                    # Mean-centered prediction
                    n_means = np.array(
                        [self.user_means[v] for v in neighbor_users[rated_mask]]
                    )
                    denom = np.abs(sims).sum()
                    if denom > 0:
                        predicted[u, i] = self.user_means[u] + (
                            sims * (rats - n_means)
                        ).sum() / denom
                    else:
                        predicted[u, i] = self.user_means[u]

        # Clip to valid rating range
        predicted = np.clip(predicted, 1.0, 5.0)
        return predicted


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(train_data: np.ndarray, **hyperparams) -> CollaborativeFilteringSKLearn:
    """
    Train a collaborative filtering model.

    Parameters
    ----------
    train_data : np.ndarray
        Training rating matrix.
    **hyperparams
        n_neighbors, metric, algorithm, mode.

    Returns
    -------
    CollaborativeFilteringSKLearn
    """
    model = CollaborativeFilteringSKLearn(
        n_neighbors=hyperparams.get("n_neighbors", 20),
        metric=hyperparams.get("metric", "cosine"),
        algorithm=hyperparams.get("algorithm", "brute"),
        mode=hyperparams.get("mode", "item"),
    )
    model.fit(train_data)
    return model


def validate(
    model: CollaborativeFilteringSKLearn,
    val_data: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """
    Validate the model on held-out ratings.

    Parameters
    ----------
    model : CollaborativeFilteringSKLearn
    val_data : np.ndarray
        Validation rating matrix.
    k : int
        Cut-off for ranking metrics.

    Returns
    -------
    dict of metrics.
    """
    predicted = model.predict()
    metrics = compute_metrics(val_data, predicted, k=k)
    return metrics


def test(
    model: CollaborativeFilteringSKLearn,
    test_data: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate the model on the test set.

    Parameters
    ----------
    model : CollaborativeFilteringSKLearn
    test_data : np.ndarray
        Test rating matrix.
    k : int
        Cut-off for ranking metrics.

    Returns
    -------
    dict of metrics.
    """
    predicted = model.predict()
    metrics = compute_metrics(test_data, predicted, k=k)
    return metrics


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimization
# ---------------------------------------------------------------------------

def optuna_objective(
    trial: optuna.Trial,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> float:
    """
    Optuna objective function for hyperparameter tuning.

    Returns RMSE on the validation set (to minimize).
    """
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    mode = trial.suggest_categorical("mode", ["user", "item"])
    metric = trial.suggest_categorical("metric", ["cosine", "euclidean"])

    model = train(
        train_data,
        n_neighbors=n_neighbors,
        mode=mode,
        metric=metric,
    )
    metrics = validate(model, val_data)
    return metrics["rmse"]


def run_optuna(
    train_data: np.ndarray,
    val_data: np.ndarray,
    n_trials: int = 30,
) -> Dict:
    """
    Run Optuna hyperparameter search.

    Returns
    -------
    dict with best_params, best_rmse.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optuna_objective(trial, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nOptuna best RMSE: {study.best_value:.4f}")
    print(f"Optuna best params: {study.best_params}")
    return {"best_params": study.best_params, "best_rmse": study.best_value}


# ---------------------------------------------------------------------------
# Ray Tune Hyperparameter Search
# ---------------------------------------------------------------------------

def ray_tune_search(
    train_data: np.ndarray,
    val_data: np.ndarray,
    num_samples: int = 20,
) -> Dict:
    """
    Run Ray Tune hyperparameter search.

    Parameters
    ----------
    train_data : np.ndarray
    val_data : np.ndarray
    num_samples : int
        Number of random configurations to try.

    Returns
    -------
    dict with best_config and best_rmse.
    """
    ray.init(ignore_reinit_error=True, num_cpus=4)

    def trainable(config):
        model = train(
            train_data,
            n_neighbors=config["n_neighbors"],
            mode=config["mode"],
            metric=config["metric"],
        )
        metrics = validate(model, val_data)
        tune.report(rmse=metrics["rmse"], mae=metrics["mae"])

    search_space = {
        "n_neighbors": tune.randint(5, 51),
        "mode": tune.choice(["user", "item"]),
        "metric": tune.choice(["cosine", "euclidean"]),
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
    print(f"\nRay Tune best RMSE: {best_rmse:.4f}")
    print(f"Ray Tune best config: {best_config}")
    ray.shutdown()
    return {"best_config": best_config, "best_rmse": best_rmse}


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full collaborative filtering pipeline."""
    print("=" * 70)
    print("Collaborative Filtering - scikit-learn Implementation")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/5] Generating synthetic data...")
    train_data, val_data, test_data = generate_data(
        n_users=300, n_items=100, density=0.12, random_state=42
    )

    # 2. Train with default hyperparameters
    print("\n[2/5] Training with default hyperparameters...")
    model = train(train_data, n_neighbors=20, mode="item", metric="cosine")

    # 3. Validate
    print("\n[3/5] Validation metrics:")
    val_metrics = validate(model, val_data)
    for name, value in val_metrics.items():
        print(f"  {name:20s}: {value:.4f}")

    # 4. Hyperparameter optimization with Optuna
    print("\n[4/5] Running Optuna hyperparameter search...")
    optuna_result = run_optuna(train_data, val_data, n_trials=15)

    # Train best model
    best_model = train(train_data, **optuna_result["best_params"])

    # 5. Test evaluation
    print("\n[5/5] Test set metrics (best model):")
    test_metrics = test(best_model, test_data)
    for name, value in test_metrics.items():
        print(f"  {name:20s}: {value:.4f}")

    # Optional: Ray Tune (uncomment to run)
    # print("\n[Bonus] Running Ray Tune hyperparameter search...")
    # ray_result = ray_tune_search(train_data, val_data, num_samples=10)

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
