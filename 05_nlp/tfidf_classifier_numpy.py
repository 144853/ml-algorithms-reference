"""
TF-IDF + Naive Bayes Classifier - NumPy From-Scratch Implementation
====================================================================

Theory & Mathematics:
---------------------
This module implements TF-IDF vectorization and Multinomial Naive Bayes
classification entirely from scratch using NumPy, without any sklearn
text processing utilities.

TF-IDF Computation (from scratch):
    1. Tokenization: Split text into lowercase tokens, remove punctuation.
    2. Vocabulary Building: Map each unique token to an index.
    3. Term Frequency (TF):
        TF(t, d) = count(t, d) / sum(count(t', d) for t' in d)
       Sublinear TF variant: TF(t, d) = 1 + log(count(t, d)) if count > 0
    4. Inverse Document Frequency (IDF):
        IDF(t) = log((1 + N) / (1 + df(t))) + 1
       where N = total documents, df(t) = documents containing term t.
    5. TF-IDF(t, d) = TF(t, d) * IDF(t)
    6. L2 Normalization: Each document vector is normalized to unit length.

Multinomial Naive Bayes (from scratch):
    Based on Bayes' theorem: P(C|X) = P(X|C) * P(C) / P(X)

    For text classification:
        P(C_k | d) proportional to P(C_k) * prod(P(w_i | C_k)^x_i)

    In log space:
        log P(C_k | d) = log P(C_k) + sum(x_i * log P(w_i | C_k))

    Where:
        P(C_k) = N_k / N  (class prior)
        P(w_i | C_k) = (count(w_i, C_k) + alpha) /
                        (sum(count(w, C_k) for all w) + alpha * |V|)
        alpha = Laplace smoothing parameter

    Decision: argmax_k log P(C_k | d)

Business Use Cases:
-------------------
- Educational: Understanding TF-IDF internals
- Lightweight text classification without heavy dependencies
- Embedded systems or environments without sklearn
- Spam filtering, sentiment analysis, topic detection

Advantages:
-----------
- Full transparency: every computation is visible
- No external ML library dependencies (only NumPy)
- Fast training and prediction (closed-form solution)
- Naive Bayes handles high-dimensional sparse data well
- Interpretable: can inspect word probabilities per class

Disadvantages:
--------------
- "Naive" conditional independence assumption is often violated
- From-scratch TF-IDF may be slower than optimized C implementations
- No n-gram support without additional code
- L2 normalization on sparse data wastes some discriminative power for NB
- Limited to the Multinomial NB variant here

Key Hyperparameters:
--------------------
- alpha: Laplace smoothing (0 = no smoothing, 1 = standard Laplace)
- max_vocab_size: Maximum vocabulary size (frequency-based cutoff)
- min_df: Minimum document frequency for a term to be included
- use_sublinear_tf: Whether to apply sublinear TF scaling

References:
-----------
- Manning, Raghavan & Schutze (2008). Introduction to Information Retrieval.
- McCallum & Nigam (1998). A Comparison of Event Models for Naive Bayes.
"""

import warnings
import logging
import re
import time
from collections import Counter
from typing import Any

import numpy as np
import optuna
import ray
from ray import tune
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

POSITIVE_TEMPLATES = [
    "this product is {adv} {adj} and i {verb} it",
    "i am {adv} {adj} with this purchase",
    "{adj} quality and {adj} service overall",
    "would {adv} recommend this to everyone",
    "the best {noun} i have ever {verb_past}",
    "absolutely {adj} experience from start to finish",
    "i {verb} how {adj} this {noun} is",
    "five stars for this {adv} {adj} {noun}",
    "this {noun} exceeded my {adj} expectations",
    "great {noun} with {adj} features and {adj} design",
    "really {adj} product that works {adv} well",
    "i am {adv} impressed by the {adj} quality",
]

NEGATIVE_TEMPLATES = [
    "this product is {adv} {adj} and i {verb} it",
    "i am {adv} {adj} with this purchase",
    "{adj} quality and {adj} service overall",
    "would {adv} not recommend this to anyone",
    "the worst {noun} i have ever {verb_past}",
    "absolutely {adj} experience from start to finish",
    "i {verb} how {adj} this {noun} is",
    "one star for this {adv} {adj} {noun}",
    "this {noun} failed my basic expectations",
    "terrible {noun} with {adj} features and {adj} design",
    "really {adj} product that works {adv} poorly",
    "i am {adv} disappointed by the {adj} quality",
]

POS_ADJ = ["amazing", "excellent", "fantastic", "wonderful", "great", "superb", "outstanding"]
POS_ADV = ["really", "truly", "absolutely", "incredibly", "very", "extremely"]
POS_VERB = ["love", "adore", "appreciate", "enjoy"]
POS_VERB_PAST = ["used", "bought", "tried", "owned"]

NEG_ADJ = ["terrible", "awful", "horrible", "dreadful", "poor", "disappointing", "broken"]
NEG_ADV = ["really", "truly", "absolutely", "incredibly", "very", "extremely"]
NEG_VERB = ["hate", "dislike", "regret", "despise"]
NEG_VERB_PAST = ["used", "bought", "tried", "owned"]

NOUNS = ["product", "item", "device", "gadget", "tool", "appliance", "purchase"]


def generate_data(n_samples: int = 1000, random_state: int = 42):
    """
    Generate synthetic sentiment analysis data.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    random_state : int
        Random seed.

    Returns
    -------
    texts : list[str]
        Generated review texts.
    labels : np.ndarray
        Binary labels (1=positive, 0=negative).
    """
    rng = np.random.RandomState(random_state)
    texts, labels = [], []
    n_per_class = n_samples // 2

    for _ in range(n_per_class):
        tpl = rng.choice(POSITIVE_TEMPLATES)
        text = tpl.format(
            adj=rng.choice(POS_ADJ), adv=rng.choice(POS_ADV),
            verb=rng.choice(POS_VERB), verb_past=rng.choice(POS_VERB_PAST),
            noun=rng.choice(NOUNS),
        )
        texts.append(text)
        labels.append(1)

    for _ in range(n_per_class):
        tpl = rng.choice(NEGATIVE_TEMPLATES)
        text = tpl.format(
            adj=rng.choice(NEG_ADJ), adv=rng.choice(NEG_ADV),
            verb=rng.choice(NEG_VERB), verb_past=rng.choice(NEG_VERB_PAST),
            noun=rng.choice(NOUNS),
        )
        texts.append(text)
        labels.append(0)

    idx = rng.permutation(len(texts))
    texts = [texts[i] for i in idx]
    labels = np.array([labels[i] for i in idx])
    return texts, labels


# ---------------------------------------------------------------------------
# From-scratch TF-IDF Vectorizer
# ---------------------------------------------------------------------------


class TfidfVectorizerFromScratch:
    """
    TF-IDF Vectorizer implemented from scratch with NumPy.

    Parameters
    ----------
    max_vocab_size : int
        Maximum vocabulary size (most frequent terms kept).
    min_df : int
        Minimum document frequency for a term.
    use_sublinear_tf : bool
        If True, apply sublinear TF scaling: 1 + log(tf).
    """

    def __init__(self, max_vocab_size: int = 5000, min_df: int = 1,
                 use_sublinear_tf: bool = True):
        self.max_vocab_size = max_vocab_size
        self.min_df = min_df
        self.use_sublinear_tf = use_sublinear_tf
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, remove punctuation, split into tokens."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def fit(self, documents: list[str]) -> "TfidfVectorizerFromScratch":
        """
        Build vocabulary and compute IDF values.

        Parameters
        ----------
        documents : list[str]
            Training documents.

        Returns
        -------
        self
        """
        N = len(documents)
        # Count document frequency for each token
        df_counter: Counter = Counter()
        token_freq: Counter = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            token_freq.update(tokens)
            unique_tokens = set(tokens)
            df_counter.update(unique_tokens)

        # Filter by min_df
        valid_tokens = {t for t, df in df_counter.items() if df >= self.min_df}

        # Select top-k by frequency
        sorted_tokens = sorted(
            [(t, token_freq[t]) for t in valid_tokens],
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_tokens = sorted_tokens[: self.max_vocab_size]

        self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted_tokens)}

        # Compute IDF: log((1 + N) / (1 + df)) + 1
        V = len(self.vocabulary_)
        self.idf_ = np.zeros(V)
        for token, idx in self.vocabulary_.items():
            df = df_counter[token]
            self.idf_[idx] = np.log((1 + N) / (1 + df)) + 1

        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF matrix.

        Parameters
        ----------
        documents : list[str]
            Documents to vectorize.

        Returns
        -------
        tfidf_matrix : np.ndarray
            Shape (n_documents, vocab_size).
        """
        if self.idf_ is None:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")

        V = len(self.vocabulary_)
        n = len(documents)
        tfidf = np.zeros((n, V))

        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)
            total = len(tokens) if len(tokens) > 0 else 1

            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    tf = count / total
                    if self.use_sublinear_tf and count > 0:
                        tf = 1 + np.log(count)
                    tfidf[i, idx] = tf * self.idf_[idx]

            # L2 normalization
            norm = np.linalg.norm(tfidf[i])
            if norm > 0:
                tfidf[i] /= norm

        return tfidf

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


# ---------------------------------------------------------------------------
# From-scratch Multinomial Naive Bayes
# ---------------------------------------------------------------------------


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier implemented from scratch.

    Parameters
    ----------
    alpha : float
        Laplace smoothing parameter.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        """
        Fit Multinomial NB on TF-IDF features.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            TF-IDF matrix (non-negative values).
        y : np.ndarray, shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Class priors: P(C_k) = N_k / N
        self.class_log_prior_ = np.zeros(n_classes)
        # Feature log probabilities: log P(w_i | C_k)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for k, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.class_log_prior_[k] = np.log(X_c.shape[0] / X.shape[0])

            # Sum of feature values for this class (acts as "counts")
            feature_sum = X_c.sum(axis=0) + self.alpha
            total_sum = feature_sum.sum()

            self.feature_log_prob_[k] = np.log(feature_sum / total_sum)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log probabilities for each class.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        log_proba : np.ndarray, shape (n_samples, n_classes)
        """
        # log P(C_k | d) = log P(C_k) + sum(x_i * log P(w_i | C_k))
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


# ---------------------------------------------------------------------------
# Combined model wrapper
# ---------------------------------------------------------------------------


class TfidfNaiveBayesModel:
    """Wrapper combining TF-IDF vectorizer and Naive Bayes classifier."""

    def __init__(self, vectorizer: TfidfVectorizerFromScratch,
                 classifier: MultinomialNaiveBayes):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: list[str], y_train: np.ndarray, **hyperparams) -> TfidfNaiveBayesModel:
    """
    Train TF-IDF + Naive Bayes model from scratch.

    Parameters
    ----------
    X_train : list[str]
        Training texts.
    y_train : np.ndarray
        Training labels.
    **hyperparams
        alpha, max_vocab_size, min_df, use_sublinear_tf

    Returns
    -------
    model : TfidfNaiveBayesModel
    """
    alpha = hyperparams.get("alpha", 1.0)
    max_vocab_size = hyperparams.get("max_vocab_size", 5000)
    min_df = hyperparams.get("min_df", 1)
    use_sublinear_tf = hyperparams.get("use_sublinear_tf", True)

    vectorizer = TfidfVectorizerFromScratch(
        max_vocab_size=max_vocab_size,
        min_df=min_df,
        use_sublinear_tf=use_sublinear_tf,
    )
    X_tfidf = vectorizer.fit_transform(X_train)

    classifier = MultinomialNaiveBayes(alpha=alpha)
    # Ensure non-negative values for Multinomial NB
    X_tfidf_nn = np.maximum(X_tfidf, 0)
    classifier.fit(X_tfidf_nn, y_train)

    return TfidfNaiveBayesModel(vectorizer, classifier)


def validate(model: TfidfNaiveBayesModel, X_val: list[str],
             y_val: np.ndarray) -> dict[str, float]:
    """
    Validate and return metrics.

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }


def test(model: TfidfNaiveBayesModel, X_test: list[str],
         y_test: np.ndarray) -> dict[str, Any]:
    """
    Final test evaluation.

    Returns
    -------
    results : dict with metrics and classification report.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report": classification_report(
            y_test, y_pred, target_names=["negative", "positive"], zero_division=0
        ),
    }


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------


def optuna_objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective: maximize weighted F1."""
    params = {
        "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
        "max_vocab_size": trial.suggest_int("max_vocab_size", 500, 10000, step=500),
        "min_df": trial.suggest_int("min_df", 1, 5),
        "use_sublinear_tf": trial.suggest_categorical("use_sublinear_tf", [True, False]),
    }
    model = train(X_train, y_train, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20) -> dict:
    """Ray Tune hyperparameter search."""

    def ray_objective(config):
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        tune.report({"f1": metrics["f1"], "accuracy": metrics["accuracy"]})

    search_space = {
        "alpha": tune.loguniform(0.001, 10.0),
        "max_vocab_size": tune.choice([500, 1000, 2000, 5000, 8000]),
        "min_df": tune.choice([1, 2, 3, 5]),
        "use_sublinear_tf": tune.choice([True, False]),
    }

    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level=logging.WARNING)
    try:
        tuner = tune.Tuner(
            ray_objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(metric="f1", mode="max", num_samples=num_samples),
            run_config=ray.train.RunConfig(verbose=0),
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric="f1", mode="max")
        logger.info(f"Ray Tune best F1: {best_result.metrics['f1']:.4f}")
        return best_result.config
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the full from-scratch TF-IDF + Naive Bayes pipeline."""
    logger.info("=" * 70)
    logger.info("TF-IDF + Naive Bayes (NumPy from scratch) Pipeline")
    logger.info("=" * 70)

    # 1. Generate data
    logger.info("Generating synthetic data...")
    texts, labels = generate_data(n_samples=2000, random_state=42)
    logger.info(f"  Total samples: {len(texts)}")

    # 2. Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Baseline
    logger.info("\n--- Baseline Training ---")
    start = time.time()
    model = train(X_train, y_train, alpha=1.0, max_vocab_size=5000)
    logger.info(f"  Training time: {time.time() - start:.3f}s")

    val_metrics = validate(model, X_val, y_val)
    logger.info(f"  Validation: {val_metrics}")

    # 4. Demonstrate from-scratch components
    logger.info("\n--- From-Scratch Component Details ---")
    logger.info(f"  Vocabulary size: {len(model.vectorizer.vocabulary_)}")
    logger.info(f"  IDF shape: {model.vectorizer.idf_.shape}")
    logger.info(f"  Feature log prob shape: {model.classifier.feature_log_prob_.shape}")
    sample_tokens = list(model.vectorizer.vocabulary_.keys())[:10]
    logger.info(f"  Sample vocabulary: {sample_tokens}")

    # 5. Optuna
    logger.info("\n--- Optuna Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=25,
        show_progress_bar=True,
    )
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 6. Ray Tune
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=15)
    logger.info(f"  Best Ray config: {best_ray_config}")

    # 7. Final model
    logger.info("\n--- Final Model ---")
    final_model = train(X_train, y_train, **study.best_params)
    test_results = test(final_model, X_test, y_test)
    logger.info(f"  Test Accuracy:  {test_results['accuracy']:.4f}")
    logger.info(f"  Test Precision: {test_results['precision']:.4f}")
    logger.info(f"  Test Recall:    {test_results['recall']:.4f}")
    logger.info(f"  Test F1:        {test_results['f1']:.4f}")
    logger.info(f"\n{test_results['report']}")

    logger.info("=" * 70)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
