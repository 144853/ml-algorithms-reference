"""
TF-IDF + Logistic Regression / SVM Text Classifier - scikit-learn Implementation
=================================================================================

Theory & Mathematics:
---------------------
TF-IDF (Term Frequency - Inverse Document Frequency) is a numerical statistic
that reflects how important a word is to a document in a collection (corpus).

Term Frequency (TF):
    TF(t, d) = (Number of times term t appears in document d) /
               (Total number of terms in document d)

Inverse Document Frequency (IDF):
    IDF(t, D) = log(N / (1 + |{d in D : t in d}|))
    where N = total number of documents, and the denominator counts documents
    containing term t. The +1 prevents division by zero.

TF-IDF Weight:
    TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)

The resulting TF-IDF matrix has shape (n_documents, n_vocabulary), where each
entry represents the importance of a word in a document relative to the corpus.

Classifiers:
    - Logistic Regression: Models P(y=1|x) = sigma(w^T x + b) using the
      sigmoid function. Trained via maximum likelihood with L1/L2 regularization.
    - Support Vector Machine (SVM): Finds the optimal hyperplane that maximizes
      the margin between classes. Uses hinge loss: max(0, 1 - y * f(x)).
      Linear SVM is particularly effective for high-dimensional sparse data
      like TF-IDF vectors.

Pipeline Architecture:
    Raw Text -> TF-IDF Vectorizer -> Classifier (LR or SVM) -> Prediction

Business Use Cases:
-------------------
- Sentiment analysis (product reviews, social media)
- Spam / ham email classification
- Topic categorization of news articles
- Customer support ticket routing
- Legal document classification
- Medical text categorization

Advantages:
-----------
- Simple, fast, and interpretable
- Works well with small to medium datasets
- No GPU required; trains in seconds
- TF-IDF captures word importance effectively
- Sparse representation is memory-efficient
- Strong baseline that often rivals deep learning on small data

Disadvantages:
--------------
- Bag-of-words assumption loses word order and context
- Cannot capture semantic meaning (synonyms, polysemy)
- Vocabulary grows with corpus size (curse of dimensionality)
- Struggles with out-of-vocabulary words at inference time
- Not suitable for tasks requiring deep language understanding

Key Hyperparameters:
--------------------
- max_features: Maximum vocabulary size
- ngram_range: (min_n, max_n) for n-gram extraction
- max_df / min_df: Document frequency thresholds for filtering
- C (regularization): Inverse of regularization strength
- penalty: 'l1', 'l2', or 'elasticnet'
- kernel (SVM): 'linear', 'rbf', 'poly'

References:
-----------
- Salton & Buckley (1988). Term-weighting approaches in automatic text retrieval.
- Joachims (1998). Text categorization with SVMs.
- scikit-learn documentation: https://scikit-learn.org/stable/
"""

import warnings
import logging
import time
from typing import Any

import numpy as np
import optuna
import ray
from ray import tune
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
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
    "This product is {adv} {adj} and I {verb} it",
    "I am {adv} {adj} with this purchase",
    "{adj} quality and {adj} service overall",
    "Would {adv} recommend this to everyone",
    "The best {noun} I have ever {verb_past}",
    "Absolutely {adj} experience from start to finish",
    "I {verb} how {adj} this {noun} is",
    "Five stars for this {adv} {adj} {noun}",
    "This {noun} exceeded my {adj} expectations",
    "Great {noun} with {adj} features and {adj} design",
    "Really {adj} product that works {adv} well",
    "I am {adv} impressed by the {adj} quality",
    "Everything about this {noun} is {adv} {adj}",
    "The {noun} arrived quickly and is {adv} {adj}",
    "This is exactly what I needed and it is {adj}",
]

NEGATIVE_TEMPLATES = [
    "This product is {adv} {adj} and I {verb} it",
    "I am {adv} {adj} with this purchase",
    "{adj} quality and {adj} service overall",
    "Would {adv} not recommend this to anyone",
    "The worst {noun} I have ever {verb_past}",
    "Absolutely {adj} experience from start to finish",
    "I {verb} how {adj} this {noun} is",
    "One star for this {adv} {adj} {noun}",
    "This {noun} failed my {adj} expectations",
    "Terrible {noun} with {adj} features and {adj} design",
    "Really {adj} product that works {adv} poorly",
    "I am {adv} disappointed by the {adj} quality",
    "Everything about this {noun} is {adv} {adj}",
    "The {noun} broke after one day and is {adv} {adj}",
    "Total waste of money and the {noun} is {adj}",
]

POS_ADJ = ["amazing", "excellent", "fantastic", "wonderful", "great", "superb", "outstanding", "brilliant"]
POS_ADV = ["really", "truly", "absolutely", "incredibly", "very", "extremely", "remarkably"]
POS_VERB = ["love", "adore", "appreciate", "enjoy", "cherish"]
POS_VERB_PAST = ["used", "bought", "tried", "owned", "experienced"]

NEG_ADJ = ["terrible", "awful", "horrible", "dreadful", "poor", "disappointing", "mediocre", "broken"]
NEG_ADV = ["really", "truly", "absolutely", "incredibly", "very", "extremely", "utterly"]
NEG_VERB = ["hate", "dislike", "regret", "despise", "loathe"]
NEG_VERB_PAST = ["used", "bought", "tried", "owned", "experienced"]

NOUNS = ["product", "item", "device", "gadget", "tool", "appliance", "purchase", "thing"]


def generate_data(n_samples: int = 1000, random_state: int = 42):
    """
    Generate synthetic text classification data for sentiment analysis.

    Creates positive and negative review-like sentences using templates
    filled with random adjectives, adverbs, verbs, and nouns.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate (split evenly between classes).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    texts : list[str]
        Generated review texts.
    labels : np.ndarray
        Binary labels (1 = positive, 0 = negative).
    """
    rng = np.random.RandomState(random_state)
    texts, labels = [], []
    n_per_class = n_samples // 2

    for _ in range(n_per_class):
        template = rng.choice(POSITIVE_TEMPLATES)
        text = template.format(
            adj=rng.choice(POS_ADJ),
            adv=rng.choice(POS_ADV),
            verb=rng.choice(POS_VERB),
            verb_past=rng.choice(POS_VERB_PAST),
            noun=rng.choice(NOUNS),
        )
        texts.append(text)
        labels.append(1)

    for _ in range(n_per_class):
        template = rng.choice(NEGATIVE_TEMPLATES)
        text = template.format(
            adj=rng.choice(NEG_ADJ),
            adv=rng.choice(NEG_ADV),
            verb=rng.choice(NEG_VERB),
            verb_past=rng.choice(NEG_VERB_PAST),
            noun=rng.choice(NOUNS),
        )
        texts.append(text)
        labels.append(0)

    # Shuffle
    indices = rng.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = np.array([labels[i] for i in indices])

    return texts, labels


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: list[str], y_train: np.ndarray, **hyperparams) -> Pipeline:
    """
    Train a TF-IDF + Classifier pipeline.

    Parameters
    ----------
    X_train : list[str]
        Training texts.
    y_train : np.ndarray
        Training labels.
    **hyperparams
        classifier : str ('logistic_regression' or 'svm')
        max_features : int
        ngram_range : tuple[int, int]
        max_df : float
        min_df : int
        C : float (regularization)

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline.
    """
    classifier_type = hyperparams.get("classifier", "logistic_regression")
    max_features = hyperparams.get("max_features", 5000)
    ngram_range = hyperparams.get("ngram_range", (1, 2))
    max_df = hyperparams.get("max_df", 0.95)
    min_df = hyperparams.get("min_df", 2)
    C = hyperparams.get("C", 1.0)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )

    if classifier_type == "svm":
        clf = LinearSVC(C=C, max_iter=10000, class_weight="balanced")
    else:
        clf = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs", class_weight="balanced"
        )

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipeline.fit(X_train, y_train)
    return pipeline


def validate(model: Pipeline, X_val: list[str], y_val: np.ndarray) -> dict[str, float]:
    """
    Validate the model and return classification metrics.

    Parameters
    ----------
    model : Pipeline
        Trained sklearn pipeline.
    X_val : list[str]
        Validation texts.
    y_val : np.ndarray
        Validation labels.

    Returns
    -------
    metrics : dict
        Dictionary with accuracy, precision, recall, f1.
    """
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }
    return metrics


def test(model: Pipeline, X_test: list[str], y_test: np.ndarray) -> dict[str, Any]:
    """
    Final test evaluation with detailed classification report.

    Parameters
    ----------
    model : Pipeline
        Trained sklearn pipeline.
    X_test : list[str]
        Test texts.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    results : dict
        Dictionary with metrics and classification report string.
    """
    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report": classification_report(
            y_test, y_pred, target_names=["negative", "positive"], zero_division=0
        ),
    }
    return results


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: optuna.Trial,
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    X_train, y_train : training data
    X_val, y_val : validation data

    Returns
    -------
    f1 : float
        Weighted F1 score on validation set.
    """
    params = {
        "classifier": trial.suggest_categorical("classifier", ["logistic_regression", "svm"]),
        "max_features": trial.suggest_int("max_features", 1000, 10000, step=1000),
        "ngram_range": (1, trial.suggest_int("max_ngram", 1, 3)),
        "max_df": trial.suggest_float("max_df", 0.8, 1.0),
        "min_df": trial.suggest_int("min_df", 1, 5),
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
    }

    model = train(X_train, y_train, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
    num_samples: int = 20,
) -> dict:
    """
    Ray Tune hyperparameter search.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    num_samples : int
        Number of hyperparameter configurations to try.

    Returns
    -------
    best_config : dict
        Best hyperparameter configuration found.
    """

    def ray_objective(config):
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        tune.report({"f1": metrics["f1"], "accuracy": metrics["accuracy"]})

    search_space = {
        "classifier": tune.choice(["logistic_regression", "svm"]),
        "max_features": tune.choice([1000, 3000, 5000, 8000, 10000]),
        "ngram_range": tune.choice([(1, 1), (1, 2), (1, 3)]),
        "max_df": tune.uniform(0.8, 1.0),
        "min_df": tune.choice([1, 2, 3, 5]),
        "C": tune.loguniform(0.01, 10.0),
    }

    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level=logging.WARNING)
    try:
        tuner = tune.Tuner(
            ray_objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="f1",
                mode="max",
                num_samples=num_samples,
            ),
            run_config=ray.train.RunConfig(verbose=0),
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric="f1", mode="max")
        best_config = best_result.config
        logger.info(f"Ray Tune best F1: {best_result.metrics['f1']:.4f}")
        return best_config
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    """Run the full TF-IDF + sklearn classifier pipeline."""
    logger.info("=" * 70)
    logger.info("TF-IDF + sklearn Classifier Pipeline")
    logger.info("=" * 70)

    # 1. Generate data
    logger.info("Generating synthetic text data...")
    texts, labels = generate_data(n_samples=2000, random_state=42)
    logger.info(f"  Total samples: {len(texts)}")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    logger.info(f"  Sample positive: '{texts[np.where(labels == 1)[0][0]]}'")
    logger.info(f"  Sample negative: '{texts[np.where(labels == 0)[0][0]]}'")

    # 2. Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Train baseline model
    logger.info("\n--- Baseline Training ---")
    start = time.time()
    baseline_model = train(X_train, y_train, classifier="logistic_regression", C=1.0)
    train_time = time.time() - start
    logger.info(f"  Training time: {train_time:.3f}s")

    val_metrics = validate(baseline_model, X_val, y_val)
    logger.info(f"  Validation metrics: {val_metrics}")

    # 4. Optuna hyperparameter optimization
    logger.info("\n--- Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=30,
        show_progress_bar=True,
    )
    logger.info(f"  Best Optuna F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 5. Ray Tune hyperparameter search
    logger.info("\n--- Ray Tune Hyperparameter Search ---")
    best_ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=15)
    logger.info(f"  Best Ray config: {best_ray_config}")

    # 6. Train final model with best Optuna params
    logger.info("\n--- Final Model Training ---")
    best_params = study.best_params.copy()
    best_params["ngram_range"] = (1, best_params.pop("max_ngram"))
    final_model = train(X_train, y_train, **best_params)

    # 7. Test evaluation
    logger.info("\n--- Test Evaluation ---")
    test_results = test(final_model, X_test, y_test)
    logger.info(f"  Test Accuracy:  {test_results['accuracy']:.4f}")
    logger.info(f"  Test Precision: {test_results['precision']:.4f}")
    logger.info(f"  Test Recall:    {test_results['recall']:.4f}")
    logger.info(f"  Test F1:        {test_results['f1']:.4f}")
    logger.info(f"\n{test_results['report']}")

    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
