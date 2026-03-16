"""
BERT Fine-Tuning for Text Classification - PyTorch + HuggingFace Implementation
=================================================================================

Theory & Mathematics:
---------------------
This module fine-tunes a pre-trained BERT (DistilBERT) model for text
classification using PyTorch and the HuggingFace Transformers library.

BERT Fine-Tuning Strategy:
    1. Start with pre-trained weights from a large-scale language model
       (DistilBERT: 6 layers, 768 hidden, 12 heads, 66M params).
    2. Add a classification head on top of the [CLS] token representation.
    3. Fine-tune ALL parameters end-to-end on the downstream task with a
       small learning rate.

Architecture:
    Input Text -> WordPiece Tokenizer -> Token IDs + Attention Mask
    -> DistilBERT Encoder (6 transformer layers)
    -> [CLS] Token Hidden State (768-dim)
    -> Dropout -> Linear (768 -> n_classes) -> Softmax

Fine-Tuning Details:
    - Learning rate: Much smaller than training from scratch (2e-5 to 5e-5)
    - Epochs: Typically 2-5 (pre-trained model converges quickly)
    - Warmup: Linear warmup for the first ~10% of training steps
    - Weight decay: Applied to all parameters except bias and LayerNorm

    Loss: Cross-Entropy
        L = -sum(y_i * log(softmax(z_i)))

    The HuggingFace BertForSequenceClassification model handles the
    classification head internally.

Transfer Learning Theory:
    Pre-trained transformers encode rich linguistic knowledge:
    - Syntax (lower layers)
    - Semantics (middle layers)
    - Task-specific features (upper layers)

    Fine-tuning allows adapting this knowledge to a specific task with
    minimal labeled data. The small learning rate preserves pre-trained
    knowledge while adapting to the new task.

Business Use Cases:
-------------------
- State-of-the-art sentiment analysis
- Intent classification for chatbots
- Named entity recognition (with token-level head)
- Question answering (with span extraction head)
- Toxicity and content moderation
- Multi-label document classification

Advantages:
-----------
- Leverages billions of tokens of pre-training
- State-of-the-art accuracy on most NLP benchmarks
- Works well with as few as 100 labeled examples
- End-to-end fine-tuning optimizes all layers jointly
- Rich ecosystem (HuggingFace Hub, thousands of pre-trained models)

Disadvantages:
--------------
- Requires GPU for practical training times
- Large model size (DistilBERT: ~250MB)
- Risk of catastrophic forgetting with aggressive fine-tuning
- Fixed context window (512 tokens)
- High inference latency compared to TF-IDF + linear models

Key Hyperparameters:
--------------------
- model_name: Pre-trained model identifier
- learning_rate: Fine-tuning LR (2e-5 to 5e-5 typical)
- n_epochs: Number of fine-tuning epochs (2-5)
- batch_size: Training batch size
- warmup_ratio: Fraction of steps for LR warmup
- weight_decay: L2 regularization (0.01 typical)
- max_length: Maximum token sequence length
- dropout_rate: Dropout in classification head

References:
-----------
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
- Sanh et al. (2019). DistilBERT, a distilled version of BERT.
- Howard & Ruder (2018). Universal Language Model Fine-tuning (ULMFiT).
- Wolf et al. (2020). HuggingFace Transformers library.
"""

import warnings
import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

POSITIVE_TEMPLATES = [
    "this product is {adv} {adj} and i {verb} it so much",
    "i am {adv} {adj} with this purchase and service",
    "{adj} quality and {adj} customer service overall",
    "would {adv} recommend this to all my friends",
    "the best {noun} i have ever {verb_past} in my life",
    "absolutely {adj} experience from start to finish",
    "i {verb} how {adj} and well made this {noun} is",
    "five stars for this {adv} {adj} {noun} purchase",
    "this {noun} exceeded my {adj} expectations completely",
    "great {noun} with {adj} features and {adj} design quality",
]

NEGATIVE_TEMPLATES = [
    "this product is {adv} {adj} and i {verb} it completely",
    "i am {adv} {adj} with this terrible purchase",
    "{adj} quality and {adj} customer service overall",
    "would {adv} not recommend this to anyone at all",
    "the worst {noun} i have ever {verb_past} in my life",
    "absolutely {adj} experience from start to finish",
    "i {verb} how {adj} and cheaply made this {noun} is",
    "one star for this {adv} {adj} {noun} purchase",
    "this {noun} failed all my expectations completely",
    "terrible {noun} with {adj} features and {adj} build quality",
]

POS_ADJ = ["amazing", "excellent", "fantastic", "wonderful", "great", "superb", "outstanding"]
POS_ADV = ["really", "truly", "absolutely", "incredibly", "very", "extremely"]
POS_VERB = ["love", "adore", "appreciate", "enjoy"]
POS_VERB_PAST = ["used", "bought", "tried", "owned"]
NEG_ADJ = ["terrible", "awful", "horrible", "dreadful", "poor", "disappointing", "broken"]
NEG_ADV = ["really", "truly", "absolutely", "incredibly", "very", "utterly"]
NEG_VERB = ["hate", "dislike", "regret", "despise"]
NEG_VERB_PAST = ["used", "bought", "tried", "wasted"]
NOUNS = ["product", "item", "device", "gadget", "tool", "appliance", "purchase"]


def generate_data(n_samples: int = 500, random_state: int = 42):
    """Generate synthetic sentiment data (smaller default for BERT)."""
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
# PyTorch Dataset
# ---------------------------------------------------------------------------


class TextClassificationDataset(Dataset):
    """PyTorch dataset for text classification with BERT tokenization."""

    def __init__(self, texts: list[str], labels: np.ndarray,
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class BertClassifierModel:
    """Wrapper for BERT fine-tuned classifier."""

    def __init__(self, model, tokenizer, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.append(preds)

        return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: list[str], y_train: np.ndarray, **hyperparams) -> BertClassifierModel:
    """
    Fine-tune BERT for text classification.

    Parameters
    ----------
    X_train : list[str]
        Training texts.
    y_train : np.ndarray
        Training labels.
    **hyperparams
        model_name, learning_rate, n_epochs, batch_size,
        warmup_ratio, weight_decay, max_length, dropout_rate

    Returns
    -------
    model : BertClassifierModel
    """
    model_name = hyperparams.get("model_name", "distilbert-base-uncased")
    learning_rate = hyperparams.get("learning_rate", 2e-5)
    n_epochs = hyperparams.get("n_epochs", 3)
    batch_size = hyperparams.get("batch_size", 16)
    warmup_ratio = hyperparams.get("warmup_ratio", 0.1)
    weight_decay = hyperparams.get("weight_decay", 0.01)
    max_length = hyperparams.get("max_length", 128)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(DEVICE)

    # Dataset and DataLoader
    dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler with warmup
    total_steps = len(dataloader) * n_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"    Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return BertClassifierModel(model, tokenizer, max_length)


def validate(model: BertClassifierModel, X_val: list[str],
             y_val: np.ndarray) -> dict[str, float]:
    """Validate and return metrics."""
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }


def test(model: BertClassifierModel, X_test: list[str],
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


def optuna_objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective for BERT fine-tuning hyperparameters."""
    params = {
        "model_name": "distilbert-base-uncased",
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 2, 5),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "max_length": 128,
    }

    model = train(X_train, y_train, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20) -> dict:
    """Ray Tune hyperparameter search for BERT fine-tuning."""

    def ray_objective(config):
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        tune.report({"f1": metrics["f1"], "accuracy": metrics["accuracy"]})

    search_space = {
        "model_name": "distilbert-base-uncased",
        "learning_rate": tune.loguniform(1e-5, 5e-5),
        "n_epochs": tune.choice([2, 3, 4]),
        "batch_size": tune.choice([8, 16]),
        "warmup_ratio": tune.uniform(0.0, 0.2),
        "weight_decay": tune.uniform(0.0, 0.1),
        "max_length": 128,
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
    """Run the full BERT fine-tuning pipeline."""
    logger.info("=" * 70)
    logger.info("BERT Fine-Tuning for Text Classification Pipeline")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 70)

    # 1. Generate data
    logger.info("Generating synthetic data...")
    texts, labels = generate_data(n_samples=400, random_state=42)
    logger.info(f"  Total samples: {len(texts)}")

    # 2. Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Baseline fine-tuning
    logger.info("\n--- Baseline Fine-Tuning ---")
    start = time.time()
    baseline_model = train(
        X_train, y_train,
        model_name="distilbert-base-uncased",
        learning_rate=2e-5,
        n_epochs=3,
        batch_size=16,
    )
    logger.info(f"  Training time: {time.time() - start:.3f}s")

    val_metrics = validate(baseline_model, X_val, y_val)
    logger.info(f"  Validation: {val_metrics}")

    # 4. Optuna (fewer trials for BERT)
    logger.info("\n--- Optuna Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=5,
        show_progress_bar=True,
    )
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 5. Ray Tune (fewer samples for BERT)
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=3)
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

    # 7. Example predictions
    logger.info("\n--- Example Predictions ---")
    sample_texts = [
        "this product is truly amazing and i love it",
        "terrible quality and i regret buying this device",
        "absolutely fantastic experience with this gadget",
        "the worst purchase i have ever made in my life",
    ]
    preds = final_model.predict(sample_texts)
    for text, pred in zip(sample_texts, preds):
        sentiment = "positive" if pred == 1 else "negative"
        logger.info(f"  '{text}' -> {sentiment}")

    logger.info("=" * 70)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
