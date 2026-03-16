"""
t-SNE (t-distributed Stochastic Neighbour Embedding) - NumPy From-Scratch
==========================================================================

COMPLETE ML TUTORIAL: This file implements t-SNE from scratch in NumPy,
following the original algorithm by van der Maaten & Hinton (2008).
t-SNE is a non-linear dimensionality reduction technique that maps
high-dimensional data to 2D/3D while preserving local neighbourhood structure.

Theory & Mathematics:

    t-SNE operates in three stages:

    STAGE 1 -- Compute High-Dimensional Affinities P:
        For each point x_i, compute conditional probabilities using a Gaussian kernel:

            p_{j|i} = exp(-||x_i - x_j||^2 / (2 * sigma_i^2))
                      / sum_{k != i} exp(-||x_i - x_k||^2 / (2 * sigma_i^2))

        The bandwidth sigma_i is found via BINARY SEARCH such that the
        perplexity of the conditional distribution matches the user's target:

            Perp(P_i) = 2^{H(P_i)} = 2^{-sum_j p_{j|i} * log2(p_{j|i})}

        Perplexity is the effective number of neighbours: perplexity=30 means
        each point's probability mass is spread over ~30 neighbours.

        Symmetrise: p_{ij} = (p_{j|i} + p_{i|j}) / (2*N)

    STAGE 2 -- Compute Low-Dimensional Affinities Q:
        In the embedding space (2D), use a Student-t kernel with 1 DOF (Cauchy):

            q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Z
            Z = sum_{k != l} (1 + ||y_k - y_l||^2)^{-1}

        WHY Student-t (not Gaussian)?
            The "crowding problem": in high dimensions, a point has many
            equidistant neighbours, but in 2D there is not enough space
            to accommodate them all at the same distance. The heavy tails
            of Student-t push moderately distant points EVEN FURTHER apart,
            creating space for close neighbours to remain close.

    STAGE 3 -- Minimise KL Divergence via Gradient Descent:
        Cost function:
            C = KL(P || Q) = sum_{i != j} p_{ij} * log(p_{ij} / q_{ij})

        Gradient with respect to embedding point y_i:
            dC/dy_i = 4 * sum_j (p_{ij} - q_{ij}) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^{-1}

        Optimisation tricks:
            - Early exaggeration: multiply P by 4 (or 12) for the first 250 iterations
              to encourage tight cluster formation before fine-tuning.
            - Momentum: accelerates convergence by accumulating velocity.
              momentum = 0.5 for first 250 iters, then 0.8.
            - Learning rate: typically 100-500.

    Complexity:
        Time: O(n^2) per iteration for exact t-SNE (pairwise distances).
        Space: O(n^2) for the distance/affinity matrices.

Hyperparameters:
    - perplexity (float): Effective number of neighbours (5-50). Default 30.
    - learning_rate (float): Gradient descent step size. Default 200.
    - n_iter (int): Number of gradient descent iterations. Default 1000.
    - early_exaggeration (float): P multiplier in early iterations. Default 4.
    - n_components (int): Embedding dimensions (2 or 3). Default 2.

Advantages:
    - Reveals non-linear cluster structure invisible to PCA.
    - Excellent for visualisation of high-dimensional data.
    - Student-t kernel solves the crowding problem elegantly.

Disadvantages:
    - O(n^2) complexity limits scalability (Barnes-Hut helps).
    - Stochastic: different runs give different embeddings.
    - Hyperparameter-sensitive (especially perplexity).
    - No out-of-sample transform (must re-fit for new data).
    - Distances in the embedding are NOT meaningful.
"""

# -- Standard library imports --
import logging  # Structured logging for pipeline monitoring
import time  # Wall-clock timing for comparisons
from functools import partial  # Binds arguments for callbacks

# -- Third-party imports --
import numpy as np  # Core numerical library for all operations
import optuna  # Bayesian hyperparameter optimisation
import ray  # Distributed computing for parallel HPO
from ray import tune  # Ray's tuning module
from sklearn.datasets import make_classification  # Synthetic data generation
from sklearn.metrics import silhouette_score  # Embedding quality metric
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.neighbors import KNeighborsClassifier  # k-NN for quality evaluation
from sklearn.preprocessing import StandardScaler  # Feature standardisation

# -- Configure logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# t-SNE From Scratch
# ---------------------------------------------------------------------------

def _compute_pairwise_distances(X):
    """Compute the squared Euclidean distance matrix between all point pairs.

    ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T * x_j

    This vectorised formula avoids explicit loops and is O(n^2 * d).

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        Squared distance matrix, shape (n, n). D[i,j] = ||x_i - x_j||^2.
    """
    # Compute squared norms for each point: ||x_i||^2
    sum_sq = np.sum(X ** 2, axis=1)  # Shape: (n,)

    # Use the expansion: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    # sum_sq[:, None] broadcasts to (n, 1), sum_sq[None, :] to (1, n)
    # X @ X.T gives the dot product matrix (n, n)
    D = sum_sq[:, None] + sum_sq[None, :] - 2.0 * (X @ X.T)  # Shape: (n, n)

    # Clip negative values (can occur due to floating-point imprecision)
    D = np.maximum(D, 0.0)  # Ensure all distances are non-negative

    return D  # Squared Euclidean distance matrix


def _binary_search_sigma(distances_i, target_perplexity, tol=1e-5, max_iter=50):
    """Find the Gaussian bandwidth sigma_i via binary search.

    For point i, we want to find sigma_i such that:
        Perp(P_i) = 2^{H(P_i)} = target_perplexity

    where H(P_i) = -sum_j p_{j|i} * log2(p_{j|i}) is Shannon entropy.

    The binary search operates on beta = 1 / (2 * sigma^2), which is the
    precision of the Gaussian. Higher beta = smaller sigma = fewer neighbours.

    Args:
        distances_i: Squared distances from point i to all other points.
        target_perplexity: Desired perplexity (effective number of neighbours).
        tol: Tolerance on the difference between achieved and target perplexity.
        max_iter: Maximum binary search iterations.

    Returns:
        Conditional probability vector p_{j|i} for point i.
    """
    # Target entropy: H = log2(perplexity) <=> perplexity = 2^H
    target_entropy = np.log(target_perplexity)  # Using natural log

    # Initialise beta = 1 / (2 * sigma^2) (precision)
    beta = 1.0  # Starting precision (sigma = 1/sqrt(2))

    # Binary search bounds
    beta_min = -np.inf  # Lower bound for beta
    beta_max = np.inf  # Upper bound for beta

    for _ in range(max_iter):  # Binary search iterations
        # Compute unnormalised conditional probabilities
        # p_{j|i} proportional to exp(-beta * ||x_i - x_j||^2)
        exp_neg = np.exp(-beta * distances_i)  # Gaussian kernel values

        # Normalise to get proper conditional probabilities
        sum_exp = np.sum(exp_neg)  # Normalisation constant

        if sum_exp == 0.0:  # Prevent division by zero
            sum_exp = 1e-12  # Small epsilon fallback

        p_i = exp_neg / sum_exp  # Normalised conditional probabilities

        # Compute Shannon entropy H(P_i) = -sum p * log(p)
        # Add small epsilon to prevent log(0)
        p_nonzero = np.maximum(p_i, 1e-12)  # Avoid log(0)
        entropy = -np.sum(p_i * np.log(p_nonzero))  # Shannon entropy

        # Compute the difference between achieved and target entropy
        entropy_diff = entropy - target_entropy  # Positive = too many neighbours

        # Check convergence
        if np.abs(entropy_diff) < tol:  # Close enough to target perplexity
            break  # Converged

        # Adjust beta using binary search
        if entropy_diff > 0:  # Too many neighbours -> increase beta (smaller sigma)
            beta_min = beta  # Update lower bound
            if beta_max == np.inf:  # No upper bound yet
                beta *= 2  # Double beta (halve sigma)
            else:
                beta = (beta + beta_max) / 2  # Bisect
        else:  # Too few neighbours -> decrease beta (larger sigma)
            beta_max = beta  # Update upper bound
            if beta_min == -np.inf:  # No lower bound yet
                beta /= 2  # Halve beta (double sigma)
            else:
                beta = (beta + beta_min) / 2  # Bisect

    return p_i  # Conditional probability vector for point i


def _compute_joint_probabilities(X, perplexity):
    """Compute the symmetric joint probability matrix P from high-dim data.

    For each point i:
        1. Compute conditional probabilities p_{j|i} using binary search for sigma_i
        2. Symmetrise: p_{ij} = (p_{j|i} + p_{i|j}) / (2*N)

    Args:
        X: Data matrix, shape (n, d).
        perplexity: Target perplexity (effective number of neighbours).

    Returns:
        Symmetric joint probability matrix P, shape (n, n).
    """
    n = X.shape[0]  # Number of data points

    # Compute full pairwise distance matrix (n x n)
    D = _compute_pairwise_distances(X)  # Squared Euclidean distances

    # Compute conditional probabilities for each point
    P = np.zeros((n, n))  # Will hold conditional probabilities

    for i in range(n):  # For each point
        # Get distances from point i to all OTHER points
        distances_i = D[i].copy()  # Copy to avoid modifying D
        distances_i[i] = np.inf  # Set self-distance to infinity (exclude self)

        # Binary search for sigma_i to match target perplexity
        P[i] = _binary_search_sigma(distances_i, perplexity)

        # Set self-probability to zero (a point is not its own neighbour)
        P[i, i] = 0.0

    # Symmetrise: p_{ij} = (p_{j|i} + p_{i|j}) / (2*N)
    # WHY symmetrise: makes the gradient simpler and ensures p_{ij} = p_{ji}
    P = (P + P.T) / (2.0 * n)  # Symmetric joint probabilities

    # Ensure minimum probability to prevent log(0) in KL divergence
    P = np.maximum(P, 1e-12)  # Floor at epsilon

    return P  # Symmetric joint probability matrix


def _compute_q_and_gradient(Y, P):
    """Compute low-dim affinities Q and the gradient of KL divergence.

    Student-t kernel with 1 DOF (Cauchy distribution):
        q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Z

    Gradient:
        dC/dy_i = 4 * sum_j (p_{ij} - q_{ij}) * (y_i - y_j) * (1 + ||y_i-y_j||^2)^{-1}

    Args:
        Y: Embedding matrix, shape (n, n_components).
        P: High-dim joint probability matrix, shape (n, n).

    Returns:
        Tuple of (Q, gradient, kl_divergence):
            Q: Low-dim affinity matrix, shape (n, n).
            gradient: Gradient of KL w.r.t. Y, shape (n, n_components).
            kl_divergence: KL(P || Q) scalar value.
    """
    n = Y.shape[0]  # Number of points

    # Compute squared pairwise distances in embedding space
    D_low = _compute_pairwise_distances(Y)  # Shape: (n, n)

    # Compute Student-t kernel: (1 + ||y_i - y_j||^2)^{-1}
    inv_dist = 1.0 / (1.0 + D_low)  # Shape: (n, n)

    # Set diagonal to zero (a point is not its own neighbour)
    np.fill_diagonal(inv_dist, 0.0)  # Exclude self-affinities

    # Normalise to get Q probabilities
    Z = np.sum(inv_dist)  # Normalisation constant (sum of all off-diagonal entries)
    Q = inv_dist / Z  # Normalised low-dim affinities

    # Floor Q at epsilon to prevent division by zero in KL
    Q = np.maximum(Q, 1e-12)  # Minimum probability

    # Compute KL divergence: KL(P||Q) = sum p_{ij} * log(p_{ij} / q_{ij})
    kl_divergence = np.sum(P * np.log(P / Q))  # Scalar KL divergence

    # Compute the gradient of KL w.r.t. each embedding point y_i
    # dC/dy_i = 4 * sum_j (p_{ij} - q_{ij}) * (y_i - y_j) * (1 + ||y_i-y_j||^2)^{-1}
    PQ_diff = P - Q  # Shape: (n, n) -- the "attractive/repulsive" force matrix
    gradient = np.zeros_like(Y)  # Shape: (n, n_components)

    for i in range(n):  # For each point
        # (y_i - y_j) for all j: shape (n, n_components)
        diff = Y[i] - Y  # Broadcasting: (n_components,) - (n, n_components) = (n, n_components)

        # Multiply by (p_{ij} - q_{ij}) * (1 + ||y_i - y_j||^2)^{-1}
        # PQ_diff[i] has shape (n,), inv_dist[i] has shape (n,)
        weights = PQ_diff[i] * inv_dist[i]  # Shape: (n,) -- force magnitude per neighbour

        # Sum weighted differences to get the gradient for point i
        gradient[i] = 4.0 * np.sum(weights[:, None] * diff, axis=0)  # Shape: (n_components,)

    return Q, gradient, kl_divergence


class TSNENumpy:
    """t-SNE implemented from scratch in NumPy.

    Full implementation with:
        - Binary search for sigma (perplexity matching)
        - Student-t kernel for low-dim affinities
        - KL divergence minimisation via momentum gradient descent
        - Early exaggeration for initial cluster formation
    """

    def __init__(self, n_components=2, perplexity=30, learning_rate=200,
                 n_iter=1000, early_exaggeration=4.0, random_state=42):
        """Initialise t-SNE.

        Args:
            n_components: Embedding dimensionality (2 for visualisation).
            perplexity: Effective number of neighbours (5-50).
            learning_rate: Gradient descent step size.
            n_iter: Total optimisation iterations.
            early_exaggeration: Factor to multiply P by in early iterations.
            random_state: Random seed for reproducible embeddings.
        """
        self.n_components = n_components  # Target embedding dimensions
        self.perplexity = perplexity  # Controls neighbourhood size
        self.learning_rate = learning_rate  # Gradient descent step size
        self.n_iter = n_iter  # Total iterations
        self.early_exaggeration = early_exaggeration  # P exaggeration factor
        self.random_state = random_state  # Random seed

        # Results (populated after fit)
        self.embedding_ = None  # Final 2D embedding
        self.kl_divergence_ = None  # Final KL divergence value

    def fit_transform(self, X):
        """Compute the t-SNE embedding for the given data.

        Algorithm:
            1. Compute joint probabilities P from high-dim data
            2. Initialise random embedding Y
            3. Apply early exaggeration (multiply P by factor)
            4. Optimise KL(P||Q) via momentum gradient descent
            5. Remove early exaggeration after 250 iterations

        Args:
            X: Data matrix, shape (n, d).

        Returns:
            Embedding matrix, shape (n, n_components).
        """
        np.random.seed(self.random_state)  # Reproducible initialisation

        n = X.shape[0]  # Number of data points

        # Stage 1: Compute joint probabilities P
        logger.info("Computing joint probabilities (perplexity=%.1f)...", self.perplexity)
        P = _compute_joint_probabilities(X, self.perplexity)  # Shape: (n, n)

        # Stage 2: Initialise embedding randomly (small values for stability)
        Y = np.random.randn(n, self.n_components) * 0.01  # Shape: (n, 2)

        # Initialise momentum and velocity for gradient descent
        velocity = np.zeros_like(Y)  # Accumulated velocity for momentum

        # Stage 3: Apply early exaggeration
        # WHY: exaggerating P forces clusters to form quickly in early iterations
        P_exaggerated = P * self.early_exaggeration  # Amplified probabilities

        # Early exaggeration phase duration
        exaggeration_iters = min(250, self.n_iter)  # Exaggerate for first 250 iters

        # Stage 4: Gradient descent loop
        logger.info("Running gradient descent (%d iterations)...", self.n_iter)
        for t in range(self.n_iter):  # Each iteration updates the embedding

            # Choose P matrix: exaggerated for early iterations, normal after
            if t < exaggeration_iters:  # Early phase
                P_current = P_exaggerated  # Use exaggerated probabilities
                momentum = 0.5  # Lower momentum during early phase
            else:  # Late phase
                P_current = P  # Use normal probabilities
                momentum = 0.8  # Higher momentum for faster convergence

            # Compute Q affinities and gradient
            Q, gradient, kl_div = _compute_q_and_gradient(Y, P_current)

            # Update velocity with momentum
            # v_t = momentum * v_{t-1} - lr * gradient
            velocity = momentum * velocity - self.learning_rate * gradient

            # Update embedding positions
            # Y_t = Y_{t-1} + v_t
            Y = Y + velocity  # Move points according to gradient + momentum

            # Log progress periodically
            if (t + 1) % 100 == 0 or t == 0:  # Every 100 iterations
                logger.debug("Iteration %d/%d  KL=%.4f", t + 1, self.n_iter, kl_div)

        # Store final results
        self.embedding_ = Y  # Final embedding positions
        self.kl_divergence_ = kl_div  # Final KL divergence
        logger.info("t-SNE complete  KL=%.4f", kl_div)

        return Y  # Return the 2D embedding


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_data(n_samples=300, n_features=30, n_classes=5, random_state=42):
    """Generate synthetic data for t-SNE evaluation.

    WHY n_samples=300: exact t-SNE is O(n^2), so we use fewer samples
    for the from-scratch implementation to keep runtime reasonable.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=15, n_redundant=5,
        n_classes=n_classes, n_clusters_per_class=1,
        class_sep=2.0, random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Data: train=%d val=%d test=%d features=%d classes=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0],
                X_train.shape[1], len(np.unique(y_train)))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(X_train, y_train, **hp):
    """Fit t-SNE on training data."""
    model = TSNENumpy(
        n_components=hp.get("n_components", 2),
        perplexity=min(hp.get("perplexity", 30), X_train.shape[0] - 1),
        learning_rate=hp.get("learning_rate", 200),
        n_iter=hp.get("n_iter", 1000),
        early_exaggeration=hp.get("early_exaggeration", 4.0),
        random_state=42,
    )
    model.fit_transform(X_train)
    return model


def _embedding_quality(embedding, labels):
    """Evaluate embedding quality."""
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["silhouette"] = silhouette_score(embedding, labels)
    else:
        metrics["silhouette"] = 0.0

    knn = KNeighborsClassifier(n_neighbors=min(5, len(labels) - 1))
    knn.fit(embedding, labels)
    metrics["knn_accuracy"] = knn.score(embedding, labels)
    return metrics


def validate(model, X_val, y_val):
    """Evaluate t-SNE quality (uses training embedding)."""
    m = {"kl_divergence": model.kl_divergence_}
    logger.info("Validation: kl_div=%.4f", m["kl_divergence"])
    return m


def test(model, X_test, y_test):
    """Report final metrics."""
    m = {"kl_divergence": model.kl_divergence_}
    logger.info("Test: kl_div=%.4f", m["kl_divergence"])
    return m


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare t-SNE with different perplexity, learning_rate, and exaggeration.

    Tests a representative set of configurations to show the effect of each
    hyperparameter on embedding quality.
    """
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: t-SNE (NumPy from scratch)")
    print("=" * 80)
    print("\nPerplexity, learning rate, and early exaggeration effects.\n")

    configs = {
        "perp=5, lr=200, exag=4": {"perplexity": 5, "learning_rate": 200, "early_exaggeration": 4, "n_iter": 500},
        "perp=30, lr=200, exag=4": {"perplexity": 30, "learning_rate": 200, "early_exaggeration": 4, "n_iter": 500},
        "perp=50, lr=200, exag=4": {"perplexity": 50, "learning_rate": 200, "early_exaggeration": 4, "n_iter": 500},
        "perp=30, lr=10, exag=4": {"perplexity": 30, "learning_rate": 10, "early_exaggeration": 4, "n_iter": 500},
        "perp=30, lr=200, exag=12": {"perplexity": 30, "learning_rate": 200, "early_exaggeration": 12, "n_iter": 500},
    }

    results = {}
    for name, params in configs.items():
        start_time = time.time()
        model = train(X_train, y_train, **params)
        train_time = time.time() - start_time

        quality = _embedding_quality(model.embedding_, y_train)
        quality["kl_divergence"] = model.kl_divergence_
        quality["train_time"] = train_time
        results[name] = quality

        print(f"  {name:<35} KL={quality['kl_divergence']:.4f}  "
              f"Sil={quality['silhouette']:.4f}  "
              f"kNN={quality['knn_accuracy']:.4f}  "
              f"Time={train_time:.1f}s")

    print(f"\nKEY TAKEAWAYS:")
    print("  1. Perplexity controls local vs global neighbourhood focus.")
    print("  2. Higher early_exaggeration forces tighter initial clusters.")
    print("  3. Low learning rate slows convergence (may need more iterations).")
    print("  4. From-scratch is slower than sklearn (no Barnes-Hut optimisation).")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo -- Customer Segments
# ---------------------------------------------------------------------------

def real_world_demo():
    """t-SNE for visualising customer segments (from scratch)."""
    print("\n" + "=" * 80)
    print("REAL-WORLD DEMO: Customer Segments (t-SNE NumPy)")
    print("=" * 80)

    np.random.seed(42)

    n_customers = 300  # Smaller for O(n^2) from-scratch implementation
    n_features = 25
    n_segments = 5
    segment_names = ["Budget", "Mid-Range", "Premium", "VIP", "Churned"]

    X = np.random.randn(n_customers, n_features)
    segment_labels = np.random.randint(0, n_segments, n_customers)

    for seg in range(n_segments):
        mask = segment_labels == seg
        segment_shift = np.random.randn(n_features) * 3
        X[mask] += segment_shift

    X[:, 1] += 0.6 * X[:, 0]
    X[:, 2] += 0.4 * X[:, 1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nDataset: {n_customers} customers, {n_features} features, {n_segments} segments")

    for perp in [5, 30]:
        model = TSNENumpy(n_components=2, perplexity=perp, learning_rate=200,
                          n_iter=500, early_exaggeration=4.0)
        embedding = model.fit_transform(X_scaled)
        quality = _embedding_quality(embedding, segment_labels)

        print(f"\n  Perplexity={perp}:")
        print(f"    KL divergence: {model.kl_divergence_:.4f}")
        print(f"    Silhouette: {quality['silhouette']:.4f}")
        print(f"    k-NN accuracy: {quality['knn_accuracy']:.4f}")

        for seg in range(n_segments):
            mask = segment_labels == seg
            centroid = np.mean(embedding[mask], axis=0)
            print(f"    {segment_names[seg]:>10}: ({centroid[0]:>7.2f}, {centroid[1]:>7.2f})")

    print(f"\nCONCLUSION: t-SNE reveals cluster structure even from scratch.")

    return model, quality


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    hp = {
        "perplexity": trial.suggest_float("perplexity", 5, min(50, X_train.shape[0] - 1)),
        "learning_rate": trial.suggest_float("learning_rate", 10, 500, log=True),
        "n_iter": trial.suggest_int("n_iter", 300, 800, step=100),
        "early_exaggeration": trial.suggest_float("early_exaggeration", 2, 15),
    }
    model = train(X_train, y_train, **hp)
    quality = _embedding_quality(model.embedding_, y_train)
    return quality["silhouette"]


def run_optuna(X_train, y_train, X_val, y_val, n_trials=10):
    study = optuna.create_study(direction="maximize", study_name="tsne_numpy")
    study.optimize(
        partial(optuna_objective, X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val),
        n_trials=n_trials, show_progress_bar=True,
    )
    logger.info("Optuna best: %s  val=%.6f", study.best_trial.params, study.best_trial.value)
    return study


# ---------------------------------------------------------------------------
# Ray Tune
# ---------------------------------------------------------------------------

def _ray_trainable(config, X_train, y_train, X_val, y_val):
    model = train(X_train, y_train, **config)
    quality = _embedding_quality(model.embedding_, y_train)
    quality["kl_divergence"] = model.kl_divergence_
    tune.report(**quality)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=5):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    search_space = {
        "perplexity": tune.uniform(5, min(50, X_train.shape[0] - 1)),
        "learning_rate": tune.loguniform(10, 500),
        "n_iter": tune.choice([300, 500, 800]),
        "early_exaggeration": tune.uniform(2, 15),
    }

    trainable = tune.with_parameters(
        _ray_trainable, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    tuner = tune.Tuner(
        trainable, param_space=search_space,
        tune_config=tune.TuneConfig(metric="silhouette", mode="max", num_samples=num_samples),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="silhouette", mode="max")
    logger.info("Ray best: %s  sil=%.6f", best.config, best.metrics["silhouette"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Execute the full t-SNE (NumPy from scratch) pipeline."""
    print("=" * 70)
    print("t-SNE - NumPy From Scratch")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()

    print("\n--- Optuna ---")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=8)
    print(f"Best params     : {study.best_trial.params}")
    print(f"Best silhouette : {study.best_trial.value:.6f}")

    print("\n--- Ray Tune ---")
    ray_results = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=5)
    ray_best = ray_results.get_best_result(metric="silhouette", mode="max")
    print(f"Best config     : {ray_best.config}")
    print(f"Best silhouette : {ray_best.metrics['silhouette']:.6f}")

    model = train(X_train, y_train, **study.best_trial.params)
    quality = _embedding_quality(model.embedding_, y_train)
    print(f"\n--- Final ---")
    print(f"  KL divergence : {model.kl_divergence_:.6f}")
    print(f"  Silhouette    : {quality['silhouette']:.6f}")
    print(f"  k-NN accuracy : {quality['knn_accuracy']:.6f}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
