"""
BERT Embeddings + scikit-learn Classifier
==========================================

Theory & Mathematics:
---------------------
This module uses a pre-trained BERT (or DistilBERT) model to extract dense
contextual embeddings from text, then feeds these embeddings into a
scikit-learn classifier (Logistic Regression or SVM) for text classification.

BERT Architecture (Bidirectional Encoder Representations from Transformers):
    BERT is a transformer-based model pre-trained on large text corpora using:
    1. Masked Language Modeling (MLM): Predict randomly masked tokens.
    2. Next Sentence Prediction (NSP): Predict if two sentences are consecutive.

    Key components:
    - Token Embeddings: Map each token to a dense vector.
    - Positional Embeddings: Encode token position in the sequence.
    - Multi-Head Self-Attention:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        where Q, K, V are linear projections of the input.
    - Feed-Forward Networks: Two linear layers with GELU activation.
    - Layer Normalization: Applied after each sub-layer.

    The [CLS] token embedding from the final layer serves as a sentence-level
    representation, encoding the meaning of the entire input sequence.

Pipeline:
    Raw Text -> BERT Tokenizer -> BERT Model -> [CLS] Embedding (768-dim)
    -> Logistic Regression / SVM -> Class Prediction

Feature Extraction Strategy:
    - Use the [CLS] token's hidden state as the document embedding.
    - Alternatively, mean-pool all token hidden states.
    - These dense 768-dimensional vectors replace sparse TF-IDF vectors.

Business Use Cases:
-------------------
- High-accuracy sentiment analysis
- Intent classification with contextual understanding
- Semantic document classification
- Transfer learning from large pre-trained models to domain-specific tasks
- Rapid prototyping: BERT features + simple classifier

Advantages:
-----------
- Captures deep contextual and semantic meaning
- Transfer learning: leverages knowledge from pre-training
- Fixed-size dense embeddings regardless of input length
- Works well even with small labeled datasets
- Simple sklearn classifier on top = fast training after embedding

Disadvantages:
--------------
- Embedding extraction is slow without GPU
- BERT model is large (DistilBERT: ~250MB)
- Fixed context window (512 tokens)
- Requires transformers library and model download
- [CLS] embedding may not capture all nuances for complex tasks

Key Hyperparameters:
--------------------
- model_name: Pre-trained model ('distilbert-base-uncased', etc.)
- pooling_strategy: 'cls' or 'mean' pooling
- classifier: 'logistic_regression' or 'svm'
- C: Regularization strength
- max_length: Maximum token sequence length

References:
-----------
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
- Sanh et al. (2019). DistilBERT, a distilled version of BERT.
"""

import warnings
import logging
import time
from typing import Any

import numpy as np
import torch
import optuna
import ray
from ray import tune
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "great {noun} with {adj} features and value",
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
    "terrible {noun} with {adj} build and design",
]

POS_ADJ = ["amazing", "excellent", "fantastic", "wonderful", "great", "superb"]
POS_ADV = ["really", "truly", "absolutely", "incredibly", "very"]
POS_VERB = ["love", "adore", "appreciate", "enjoy"]
POS_VERB_PAST = ["used", "bought", "tried", "owned"]
NEG_ADJ = ["terrible", "awful", "horrible", "dreadful", "poor", "disappointing"]
NEG_ADV = ["really", "truly", "absolutely", "incredibly", "very"]
NEG_VERB = ["hate", "dislike", "regret", "despise"]
NEG_VERB_PAST = ["used", "bought", "tried", "owned"]
NOUNS = ["product", "item", "device", "gadget", "tool", "appliance"]


def generate_data(n_samples: int = 500, random_state: int = 42):
    """Generate synthetic sentiment data (smaller default for BERT speed)."""
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
# BERT embedding extraction
# ---------------------------------------------------------------------------


def extract_bert_embeddings(
    texts: list[str],
    model_name: str = "distilbert-base-uncased",
    pooling: str = "cls",
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract BERT embeddings from a list of texts.

    Parameters
    ----------
    texts : list[str]
        Input texts.
    model_name : str
        HuggingFace model name.
    pooling : str
        'cls' for [CLS] token, 'mean' for mean pooling.
    max_length : int
        Maximum token sequence length.
    batch_size : int
        Batch size for embedding extraction.

    Returns
    -------
    embeddings : np.ndarray
        Shape (n_texts, hidden_size).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if pooling == "cls":
            embeddings = hidden_states[:, 0, :]  # [CLS] token
        else:
            # Mean pooling with attention mask
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            sum_embeddings = (hidden_states * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1)
            embeddings = sum_embeddings / counts

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


class BertSklearnModel:
    """Wrapper for BERT embeddings + sklearn classifier."""

    def __init__(self, classifier, scaler: StandardScaler,
                 model_name: str, pooling: str, max_length: int):
        self.classifier = classifier
        self.scaler = scaler
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length

    def predict(self, texts: list[str]) -> np.ndarray:
        embeddings = extract_bert_embeddings(
            texts, self.model_name, self.pooling, self.max_length
        )
        embeddings_scaled = self.scaler.transform(embeddings)
        return self.classifier.predict(embeddings_scaled)


def train(X_train: list[str], y_train: np.ndarray, **hyperparams) -> BertSklearnModel:
    """
    Train BERT embeddings + sklearn classifier.

    Parameters
    ----------
    X_train : list[str]
        Training texts.
    y_train : np.ndarray
        Training labels.
    **hyperparams
        model_name, pooling, max_length, classifier_type, C

    Returns
    -------
    model : BertSklearnModel
    """
    model_name = hyperparams.get("model_name", "distilbert-base-uncased")
    pooling = hyperparams.get("pooling", "cls")
    max_length = hyperparams.get("max_length", 128)
    classifier_type = hyperparams.get("classifier_type", "logistic_regression")
    C = hyperparams.get("C", 1.0)

    # Extract embeddings
    logger.info(f"  Extracting embeddings with {model_name} ({pooling} pooling)...")
    embeddings = extract_bert_embeddings(
        X_train, model_name=model_name, pooling=pooling, max_length=max_length
    )

    # Scale features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Train classifier
    if classifier_type == "svm":
        clf = LinearSVC(C=C, max_iter=10000, class_weight="balanced")
    else:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", class_weight="balanced")

    clf.fit(embeddings_scaled, y_train)

    return BertSklearnModel(clf, scaler, model_name, pooling, max_length)


def validate(model: BertSklearnModel, X_val: list[str],
             y_val: np.ndarray) -> dict[str, float]:
    """Validate and return metrics."""
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }


def test(model: BertSklearnModel, X_test: list[str],
         y_test: np.ndarray) -> dict[str, Any]:
    """Final test evaluation."""
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

# Cache embeddings globally to avoid recomputation during HPO
_embedding_cache: dict[str, np.ndarray] = {}


def _get_cached_embeddings(texts: list[str], key: str,
                           model_name: str, pooling: str, max_length: int) -> np.ndarray:
    """Cache embeddings to speed up hyperparameter search."""
    cache_key = f"{key}_{model_name}_{pooling}_{max_length}"
    if cache_key not in _embedding_cache:
        _embedding_cache[cache_key] = extract_bert_embeddings(
            texts, model_name, pooling, max_length
        )
    return _embedding_cache[cache_key]


def optuna_objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective with embedding caching."""
    model_name = "distilbert-base-uncased"
    pooling = trial.suggest_categorical("pooling", ["cls", "mean"])
    max_length = 128
    classifier_type = trial.suggest_categorical(
        "classifier_type", ["logistic_regression", "svm"]
    )
    C = trial.suggest_float("C", 0.01, 100.0, log=True)

    # Get cached embeddings
    train_emb = _get_cached_embeddings(X_train, "train", model_name, pooling, max_length)
    val_emb = _get_cached_embeddings(X_val, "val", model_name, pooling, max_length)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_emb)
    val_scaled = scaler.transform(val_emb)

    if classifier_type == "svm":
        clf = LinearSVC(C=C, max_iter=10000, class_weight="balanced")
    else:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", class_weight="balanced")

    clf.fit(train_scaled, y_train)
    y_pred = clf.predict(val_scaled)
    return f1_score(y_val, y_pred, average="weighted", zero_division=0)


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20) -> dict:
    """Ray Tune search (uses pre-computed embeddings)."""
    model_name = "distilbert-base-uncased"

    # Pre-compute embeddings for both pooling strategies
    emb_cls_train = extract_bert_embeddings(X_train, model_name, "cls", 128)
    emb_cls_val = extract_bert_embeddings(X_val, model_name, "cls", 128)
    emb_mean_train = extract_bert_embeddings(X_train, model_name, "mean", 128)
    emb_mean_val = extract_bert_embeddings(X_val, model_name, "mean", 128)

    def ray_objective(config):
        if config["pooling"] == "cls":
            tr_emb, vl_emb = emb_cls_train, emb_cls_val
        else:
            tr_emb, vl_emb = emb_mean_train, emb_mean_val

        scaler = StandardScaler()
        tr_scaled = scaler.fit_transform(tr_emb)
        vl_scaled = scaler.transform(vl_emb)

        if config["classifier_type"] == "svm":
            clf = LinearSVC(C=config["C"], max_iter=10000, class_weight="balanced")
        else:
            clf = LogisticRegression(
                C=config["C"], max_iter=1000, solver="lbfgs", class_weight="balanced"
            )
        clf.fit(tr_scaled, y_train)
        y_pred = clf.predict(vl_scaled)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        tune.report({"f1": f1})

    search_space = {
        "pooling": tune.choice(["cls", "mean"]),
        "classifier_type": tune.choice(["logistic_regression", "svm"]),
        "C": tune.loguniform(0.01, 100.0),
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
    """Run the full BERT + sklearn pipeline."""
    logger.info("=" * 70)
    logger.info("BERT Embeddings + sklearn Classifier Pipeline")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 70)

    # 1. Generate data (smaller for BERT)
    logger.info("Generating synthetic data...")
    texts, labels = generate_data(n_samples=500, random_state=42)
    logger.info(f"  Total samples: {len(texts)}")

    # 2. Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Train baseline
    logger.info("\n--- Baseline Training ---")
    start = time.time()
    baseline_model = train(
        X_train, y_train,
        model_name="distilbert-base-uncased",
        classifier_type="logistic_regression",
        C=1.0,
    )
    logger.info(f"  Training time: {time.time() - start:.3f}s")

    val_metrics = validate(baseline_model, X_val, y_val)
    logger.info(f"  Validation: {val_metrics}")

    # 4. Optuna
    logger.info("\n--- Optuna Optimization ---")
    _embedding_cache.clear()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=15,
        show_progress_bar=True,
    )
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 5. Ray Tune
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    logger.info(f"  Best Ray config: {best_ray_config}")

    # 6. Final model
    logger.info("\n--- Final Model ---")
    best_params = study.best_params.copy()
    best_params["model_name"] = "distilbert-base-uncased"
    best_params["max_length"] = 128
    final_model = train(X_train, y_train, **best_params)

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
