"""
TF-IDF + PyTorch Neural Network Text Classifier
=================================================

Theory & Mathematics:
---------------------
This module combines TF-IDF feature extraction with a PyTorch neural network
classifier. The TF-IDF vectors serve as input features to a multi-layer
feedforward neural network (MLP) for text classification.

Architecture:
    Raw Text -> TF-IDF Vectorizer -> Dense MLP -> Softmax -> Class Prediction

Neural Network Details:
    Input layer:  TF-IDF vector (n_features dimensions)
    Hidden layers: Fully connected with ReLU activation + Dropout + BatchNorm
    Output layer:  2 neurons (binary classification) with softmax

    Forward pass:
        h_1 = ReLU(BatchNorm(W_1 * x + b_1))
        h_1 = Dropout(h_1, p)
        h_2 = ReLU(BatchNorm(W_2 * h_1 + b_2))
        h_2 = Dropout(h_2, p)
        output = Softmax(W_out * h_2 + b_out)

    Loss function: Cross-Entropy Loss
        L = -sum(y_i * log(p_i))

    Optimizer: Adam with learning rate scheduling
        theta_{t+1} = theta_t - lr * m_t / (sqrt(v_t) + epsilon)
        where m_t and v_t are first and second moment estimates.

Why TF-IDF + Neural Network?
    TF-IDF provides a fixed-size, interpretable representation of text.
    The neural network can learn non-linear decision boundaries that
    linear classifiers (LR, SVM) cannot capture, potentially improving
    performance on complex classification tasks.

Business Use Cases:
-------------------
- Sentiment analysis with complex decision boundaries
- Multi-class document categorization
- Intent classification in conversational AI
- Content moderation and toxicity detection

Advantages:
-----------
- Non-linear decision boundaries (vs linear TF-IDF + LR)
- GPU acceleration for large datasets
- Flexible architecture (depth, width, regularization)
- Can be extended with additional features beyond TF-IDF
- Batch training enables large-scale processing

Disadvantages:
--------------
- More hyperparameters to tune than linear models
- Requires more data than simpler models to generalize well
- Longer training time than sklearn linear models
- Still limited by bag-of-words TF-IDF representation
- Risk of overfitting on small datasets

Key Hyperparameters:
--------------------
- hidden_dims: List of hidden layer sizes
- dropout_rate: Dropout probability (0.0 to 0.5)
- learning_rate: Adam optimizer learning rate
- batch_size: Mini-batch size for training
- n_epochs: Number of training epochs
- weight_decay: L2 regularization strength
- max_features: TF-IDF vocabulary size

References:
-----------
- Goodfellow et al. (2016). Deep Learning, Chapter 6: Deep Feedforward Networks.
- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.
"""

import warnings
import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import ray
from ray import tune
from sklearn.feature_extraction.text import TfidfVectorizer
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
    "i am {adv} {adj} with this purchase overall",
    "{adj} quality and {adj} service from this company",
    "would {adv} recommend this to all my friends",
    "the best {noun} i have ever {verb_past} in my life",
    "absolutely {adj} experience from start to finish",
    "i {verb} how {adj} this {noun} is every day",
    "five stars for this {adv} {adj} {noun}",
    "this {noun} exceeded my expectations and is {adj}",
    "great {noun} with {adj} features and {adj} design",
]

NEGATIVE_TEMPLATES = [
    "this product is {adv} {adj} and i {verb} it completely",
    "i am {adv} {adj} with this terrible purchase",
    "{adj} quality and {adj} service from this company",
    "would {adv} not recommend this to anyone ever",
    "the worst {noun} i have ever {verb_past} in my life",
    "absolutely {adj} experience from start to finish",
    "i {verb} how {adj} this {noun} turned out",
    "one star for this {adv} {adj} {noun}",
    "this {noun} failed my expectations and is {adj}",
    "terrible {noun} with {adj} features and {adj} build",
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


def generate_data(n_samples: int = 1000, random_state: int = 42):
    """Generate synthetic sentiment analysis data."""
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
# PyTorch model
# ---------------------------------------------------------------------------


class TextMLP(nn.Module):
    """
    Multi-Layer Perceptron for text classification on TF-IDF features.

    Parameters
    ----------
    input_dim : int
        Dimension of TF-IDF input vectors.
    hidden_dims : list[int]
        List of hidden layer sizes.
    n_classes : int
        Number of output classes.
    dropout_rate : float
        Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = None,
                 n_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int,
                       shuffle: bool = True) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TfidfPytorchModel:
    """Wrapper combining TF-IDF vectorizer and PyTorch MLP."""

    def __init__(self, vectorizer: TfidfVectorizer, model: TextMLP):
        self.vectorizer = vectorizer
        self.model = model

    def predict(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: list[str], y_train: np.ndarray, **hyperparams) -> TfidfPytorchModel:
    """
    Train TF-IDF + PyTorch MLP model.

    Parameters
    ----------
    X_train : list[str]
        Training texts.
    y_train : np.ndarray
        Training labels.
    **hyperparams
        max_features, hidden_dims, dropout_rate, learning_rate,
        batch_size, n_epochs, weight_decay

    Returns
    -------
    model : TfidfPytorchModel
    """
    max_features = hyperparams.get("max_features", 5000)
    hidden_dims = hyperparams.get("hidden_dims", [256, 128])
    dropout_rate = hyperparams.get("dropout_rate", 0.3)
    learning_rate = hyperparams.get("learning_rate", 1e-3)
    batch_size = hyperparams.get("batch_size", 64)
    n_epochs = hyperparams.get("n_epochs", 20)
    weight_decay = hyperparams.get("weight_decay", 1e-4)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2), sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(X_train).toarray().astype(np.float32)
    input_dim = X_tfidf.shape[1]

    # Model
    model = TextMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_classes=2,
        dropout_rate=dropout_rate,
    ).to(DEVICE)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    dataloader = _create_dataloader(X_tfidf, y_train, batch_size=batch_size)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

    return TfidfPytorchModel(vectorizer, model)


def validate(model: TfidfPytorchModel, X_val: list[str],
             y_val: np.ndarray) -> dict[str, float]:
    """Validate and return metrics."""
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }


def test(model: TfidfPytorchModel, X_test: list[str],
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
    """Optuna objective: maximize weighted F1."""
    params = {
        "max_features": trial.suggest_int("max_features", 1000, 10000, step=1000),
        "hidden_dims": [
            trial.suggest_int("hidden_dim_1", 64, 512, step=64),
            trial.suggest_int("hidden_dim_2", 32, 256, step=32),
        ],
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_epochs": trial.suggest_int("n_epochs", 10, 30),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
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
        "max_features": tune.choice([3000, 5000, 8000]),
        "hidden_dims": tune.choice([[128, 64], [256, 128], [512, 256]]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "n_epochs": tune.choice([10, 15, 20]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
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
    """Run the full TF-IDF + PyTorch MLP pipeline."""
    logger.info("=" * 70)
    logger.info("TF-IDF + PyTorch Neural Network Pipeline")
    logger.info(f"Device: {DEVICE}")
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
    baseline_model = train(X_train, y_train, n_epochs=15, learning_rate=1e-3)
    logger.info(f"  Training time: {time.time() - start:.3f}s")

    val_metrics = validate(baseline_model, X_val, y_val)
    logger.info(f"  Validation: {val_metrics}")

    # 4. Optuna
    logger.info("\n--- Optuna Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=20,
        show_progress_bar=True,
    )
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 5. Ray Tune
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    logger.info(f"  Best Ray config: {best_ray_config}")

    # 6. Final model with best params
    logger.info("\n--- Final Model ---")
    best_params = study.best_params.copy()
    best_params["hidden_dims"] = [
        best_params.pop("hidden_dim_1"),
        best_params.pop("hidden_dim_2"),
    ]
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
