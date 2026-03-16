"""
Transformer Attention Mechanism - NumPy From-Scratch Implementation (Educational)
==================================================================================

Theory & Mathematics:
---------------------
This module implements the core components of the Transformer architecture
from scratch using NumPy for educational purposes. It includes:

1. Token Embeddings + Positional Encoding
2. Scaled Dot-Product Attention
3. Multi-Head Attention
4. Feedforward Network
5. Transformer Encoder Block
6. Full Transformer Encoder for Classification

Scaled Dot-Product Attention:
    Given queries Q, keys K, and values V:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Where d_k is the dimension of the key vectors. The scaling by sqrt(d_k)
    prevents the dot products from growing too large, which would push the
    softmax into regions with vanishingly small gradients.

Multi-Head Attention:
    head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O

    Multiple attention heads allow the model to jointly attend to information
    from different representation subspaces at different positions.

Positional Encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Sinusoidal positional encoding injects position information since the
    transformer has no inherent notion of token order.

Layer Normalization:
    LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
    Applied after each sub-layer for training stability.

Transformer Encoder Block:
    1. x = x + MultiHeadAttention(LayerNorm(x))
    2. x = x + FeedForward(LayerNorm(x))
    (Pre-norm variant for more stable training)

Classification Head:
    Take the mean of all token representations, then apply a linear layer
    followed by softmax for classification.

Training:
    Gradient descent with cross-entropy loss, computed via backpropagation.
    This implementation uses a simplified training loop with manual gradient
    computation for the classification head only (embeddings are randomly
    initialized and fixed for simplicity in this educational version).

Business Use Cases:
-------------------
- Educational: Understanding transformer internals
- Research: Experimenting with attention mechanisms
- Prototyping: Testing architecture ideas before PyTorch implementation

Advantages:
-----------
- Complete transparency: every matrix multiplication is visible
- No deep learning framework dependency
- Excellent for learning and teaching transformers
- Step-by-step debuggable attention computations

Disadvantages:
--------------
- Extremely slow compared to GPU-accelerated frameworks
- No automatic differentiation (manual gradients)
- Not suitable for production use
- Limited to very small models and datasets
- No pre-trained weights available

Key Hyperparameters:
--------------------
- d_model: Embedding dimension (e.g., 64)
- n_heads: Number of attention heads (e.g., 4)
- d_ff: Feedforward hidden dimension (e.g., 128)
- n_layers: Number of encoder blocks (e.g., 2)
- max_seq_len: Maximum sequence length
- learning_rate: For gradient descent
- vocab_size: Size of token vocabulary

References:
-----------
- Vaswani et al. (2017). Attention Is All You Need.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
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
    "this is {adv} {adj} product",
    "i {verb} this {adj} {noun}",
    "{adj} quality and {adj} value",
    "would recommend this {adj} {noun}",
    "best {noun} ever {adj}",
    "{adv} {adj} experience overall",
    "five stars {adj} {noun}",
    "this {noun} is {adv} {adj}",
]

NEGATIVE_TEMPLATES = [
    "this is {adv} {adj} product",
    "i {verb} this {adj} {noun}",
    "{adj} quality and {adj} value",
    "would not recommend this {adj} {noun}",
    "worst {noun} ever {adj}",
    "{adv} {adj} experience overall",
    "one star {adj} {noun}",
    "this {noun} is {adv} {adj}",
]

POS_ADJ = ["amazing", "excellent", "fantastic", "wonderful", "great", "superb"]
POS_ADV = ["really", "truly", "absolutely", "very"]
POS_VERB = ["love", "adore", "enjoy"]
NEG_ADJ = ["terrible", "awful", "horrible", "poor", "disappointing", "broken"]
NEG_ADV = ["really", "truly", "absolutely", "very"]
NEG_VERB = ["hate", "dislike", "regret"]
NOUNS = ["product", "item", "device", "gadget", "tool"]


def generate_data(n_samples: int = 600, random_state: int = 42):
    """Generate synthetic sentiment data."""
    rng = np.random.RandomState(random_state)
    texts, labels = [], []
    n_per_class = n_samples // 2

    for _ in range(n_per_class):
        tpl = rng.choice(POSITIVE_TEMPLATES)
        text = tpl.format(
            adj=rng.choice(POS_ADJ), adv=rng.choice(POS_ADV),
            verb=rng.choice(POS_VERB), noun=rng.choice(NOUNS),
        )
        texts.append(text)
        labels.append(1)

    for _ in range(n_per_class):
        tpl = rng.choice(NEGATIVE_TEMPLATES)
        text = tpl.format(
            adj=rng.choice(NEG_ADJ), adv=rng.choice(NEG_ADV),
            verb=rng.choice(NEG_VERB), noun=rng.choice(NOUNS),
        )
        texts.append(text)
        labels.append(0)

    idx = rng.permutation(len(texts))
    texts = [texts[i] for i in idx]
    labels = np.array([labels[i] for i in idx])
    return texts, labels


# ---------------------------------------------------------------------------
# Tokenizer (from scratch)
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """Simple word-level tokenizer with special tokens."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    CLS_TOKEN = "<CLS>"

    def __init__(self, max_vocab_size: int = 500):
        self.max_vocab_size = max_vocab_size
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def fit(self, texts: list[str]) -> "SimpleTokenizer":
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Special tokens
        self.token2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
        }

        for token, _ in counter.most_common(self.max_vocab_size - 3):
            self.token2idx[token] = len(self.token2idx)

        self.idx2token = {v: k for k, v in self.token2idx.items()}
        return self

    def encode(self, text: str, max_length: int = 32) -> np.ndarray:
        """Encode text to token IDs with CLS token and padding."""
        tokens = self._tokenize(text)
        ids = [self.token2idx.get(self.CLS_TOKEN, 2)]
        for t in tokens[: max_length - 1]:
            ids.append(self.token2idx.get(t, self.token2idx[self.UNK_TOKEN]))
        # Pad
        while len(ids) < max_length:
            ids.append(self.token2idx[self.PAD_TOKEN])
        return np.array(ids[:max_length])

    def encode_batch(self, texts: list[str], max_length: int = 32) -> np.ndarray:
        return np.array([self.encode(t, max_length) for t in texts])

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# Transformer components (from scratch)
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-6) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Parameters
    ----------
    Q : np.ndarray, shape (..., seq_len_q, d_k)
    K : np.ndarray, shape (..., seq_len_k, d_k)
    V : np.ndarray, shape (..., seq_len_k, d_v)
    mask : np.ndarray, optional, shape (..., seq_len_q, seq_len_k)

    Returns
    -------
    output : np.ndarray, shape (..., seq_len_q, d_v)
    attention_weights : np.ndarray, shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention from scratch.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int, rng: np.random.RandomState):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model) * scale
        self.W_k = rng.randn(d_model, d_model) * scale
        self.W_v = rng.randn(d_model, d_model) * scale
        self.W_o = rng.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Forward pass for self-attention.

        Parameters
        ----------
        x : np.ndarray, shape (batch, seq_len, d_model)
        mask : optional, shape (batch, seq_len)

        Returns
        -------
        output : np.ndarray, shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        Q = np.matmul(x, self.W_q)  # (batch, seq_len, d_model)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # Reshape for multi-head: (batch, n_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention mask: (batch, 1, 1, seq_len) for broadcasting
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, np.newaxis, np.newaxis, :]

        # Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, attn_mask)

        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = np.matmul(attn_output, self.W_o)
        return output


class FeedForward:
    """
    Position-wise feedforward network.

    FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2
    """

    def __init__(self, d_model: int, d_ff: int, rng: np.random.RandomState):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W_1 = rng.randn(d_model, d_ff) * scale1
        self.b_1 = np.zeros(d_ff)
        self.W_2 = rng.randn(d_ff, d_model) * scale2
        self.b_2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = gelu(np.matmul(x, self.W_1) + self.b_1)
        output = np.matmul(hidden, self.W_2) + self.b_2
        return output


class TransformerEncoderBlock:
    """
    Single Transformer Encoder block (pre-norm variant).

    Architecture:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 rng: np.random.RandomState):
        self.mha = MultiHeadAttention(d_model, n_heads, rng)
        self.ff = FeedForward(d_model, d_ff, rng)

        # Layer norm parameters
        self.gamma_1 = np.ones(d_model)
        self.beta_1 = np.zeros(d_model)
        self.gamma_2 = np.ones(d_model)
        self.beta_2 = np.zeros(d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        # Pre-norm self-attention with residual
        norm_x = layer_norm(x, self.gamma_1, self.beta_1)
        attn_out = self.mha.forward(norm_x, mask)
        x = x + attn_out

        # Pre-norm feedforward with residual
        norm_x = layer_norm(x, self.gamma_2, self.beta_2)
        ff_out = self.ff.forward(norm_x)
        x = x + ff_out

        return x


class TransformerClassifier:
    """
    Transformer Encoder for text classification (from scratch).

    Architecture:
        Token Embedding + Positional Encoding
        -> N x TransformerEncoderBlock
        -> Mean Pooling
        -> Linear Classification Head
        -> Softmax
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 128, n_layers: int = 2, max_seq_len: int = 32,
                 n_classes: int = 2, random_state: int = 42):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes
        rng = np.random.RandomState(random_state)

        # Token embedding
        self.embedding = rng.randn(vocab_size, d_model) * 0.02
        # Positional encoding
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

        # Encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, n_heads, d_ff, rng)
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.final_gamma = np.ones(d_model)
        self.final_beta = np.zeros(d_model)

        # Classification head
        scale = np.sqrt(2.0 / d_model)
        self.W_cls = rng.randn(d_model, n_classes) * scale
        self.b_cls = np.zeros(n_classes)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        token_ids : np.ndarray, shape (batch, seq_len)

        Returns
        -------
        logits : np.ndarray, shape (batch, n_classes)
        """
        batch_size, seq_len = token_ids.shape

        # Create padding mask (1 for real tokens, 0 for PAD)
        mask = (token_ids != 0).astype(np.float64)

        # Token embedding + positional encoding
        x = self.embedding[token_ids]  # (batch, seq_len, d_model)
        x = x + self.pos_encoding[:seq_len]

        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block.forward(x, mask)

        # Final layer norm
        x = layer_norm(x, self.final_gamma, self.final_beta)

        # Mean pooling (only over non-padded tokens)
        mask_expanded = mask[:, :, np.newaxis]
        x_masked = x * mask_expanded
        pooled = x_masked.sum(axis=1) / mask_expanded.sum(axis=1).clip(min=1)

        # Classification
        logits = np.matmul(pooled, self.W_cls) + self.b_cls
        return logits

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        logits = self.forward(token_ids)
        return np.argmax(logits, axis=1)

    def train_step(self, token_ids: np.ndarray, labels: np.ndarray,
                   lr: float = 0.01) -> float:
        """
        Simple training step with gradient descent on classification head.

        For educational purposes, we only update the classification head weights.
        A full implementation would backpropagate through all layers.

        Parameters
        ----------
        token_ids : np.ndarray, shape (batch, seq_len)
        labels : np.ndarray, shape (batch,)
        lr : float
            Learning rate.

        Returns
        -------
        loss : float
            Cross-entropy loss.
        """
        batch_size = token_ids.shape[0]

        # Forward pass
        logits = self.forward(token_ids)
        probs = softmax(logits, axis=-1)

        # Cross-entropy loss
        eps = 1e-10
        log_probs = np.log(probs + eps)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), labels] = 1
        loss = -np.mean(np.sum(one_hot * log_probs, axis=-1))

        # Gradient of loss w.r.t. logits (softmax + cross-entropy)
        grad_logits = (probs - one_hot) / batch_size  # (batch, n_classes)

        # Get pooled representation (recompute)
        mask = (token_ids != 0).astype(np.float64)
        x = self.embedding[token_ids] + self.pos_encoding[: token_ids.shape[1]]
        for block in self.encoder_blocks:
            x = block.forward(x, mask)
        x = layer_norm(x, self.final_gamma, self.final_beta)
        mask_expanded = mask[:, :, np.newaxis]
        x_masked = x * mask_expanded
        pooled = x_masked.sum(axis=1) / mask_expanded.sum(axis=1).clip(min=1)

        # Gradient for W_cls and b_cls
        grad_W_cls = np.matmul(pooled.T, grad_logits)
        grad_b_cls = np.sum(grad_logits, axis=0)

        # Update classification head
        self.W_cls -= lr * grad_W_cls
        self.b_cls -= lr * grad_b_cls

        return loss


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class TransformerModel:
    """Wrapper combining tokenizer and transformer classifier."""

    def __init__(self, tokenizer: SimpleTokenizer,
                 classifier: TransformerClassifier, max_seq_len: int):
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.max_seq_len = max_seq_len

    def predict(self, texts: list[str]) -> np.ndarray:
        token_ids = self.tokenizer.encode_batch(texts, self.max_seq_len)
        return self.classifier.predict(token_ids)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train: list[str], y_train: np.ndarray, **hyperparams) -> TransformerModel:
    """
    Train the from-scratch transformer classifier.

    Parameters
    ----------
    X_train : list[str]
        Training texts.
    y_train : np.ndarray
        Training labels.
    **hyperparams
        d_model, n_heads, d_ff, n_layers, max_seq_len, max_vocab_size,
        n_epochs, learning_rate, batch_size

    Returns
    -------
    model : TransformerModel
    """
    d_model = hyperparams.get("d_model", 64)
    n_heads = hyperparams.get("n_heads", 4)
    d_ff = hyperparams.get("d_ff", 128)
    n_layers = hyperparams.get("n_layers", 2)
    max_seq_len = hyperparams.get("max_seq_len", 32)
    max_vocab_size = hyperparams.get("max_vocab_size", 300)
    n_epochs = hyperparams.get("n_epochs", 30)
    learning_rate = hyperparams.get("learning_rate", 0.01)
    batch_size = hyperparams.get("batch_size", 32)

    # Tokenizer
    tokenizer = SimpleTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.fit(X_train)

    # Encode training data
    token_ids = tokenizer.encode_batch(X_train, max_seq_len)

    # Build model
    classifier = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        n_classes=2,
    )

    # Training loop
    n_samples = len(X_train)
    for epoch in range(n_epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_ids = token_ids[batch_idx]
            batch_labels = y_train[batch_idx]

            loss = classifier.train_step(batch_ids, batch_labels, lr=learning_rate)
            epoch_loss += loss
            n_batches += 1

    return TransformerModel(tokenizer, classifier, max_seq_len)


def validate(model: TransformerModel, X_val: list[str],
             y_val: np.ndarray) -> dict[str, float]:
    """Validate and return metrics."""
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }


def test(model: TransformerModel, X_test: list[str],
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
    """Optuna objective: maximize F1."""
    params = {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "d_ff": trial.suggest_categorical("d_ff", [64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "n_epochs": trial.suggest_int("n_epochs", 15, 40),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_vocab_size": trial.suggest_int("max_vocab_size", 100, 500, step=50),
        "max_seq_len": 32,
    }
    # Ensure d_model is divisible by n_heads
    if params["d_model"] % params["n_heads"] != 0:
        params["n_heads"] = 2

    model = train(X_train, y_train, **params)
    metrics = validate(model, X_val, y_val)
    return metrics["f1"]


def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20) -> dict:
    """Ray Tune hyperparameter search."""

    def ray_objective(config):
        if config["d_model"] % config["n_heads"] != 0:
            config["n_heads"] = 2
        model = train(X_train, y_train, **config)
        metrics = validate(model, X_val, y_val)
        tune.report({"f1": metrics["f1"], "accuracy": metrics["accuracy"]})

    search_space = {
        "d_model": tune.choice([32, 64]),
        "n_heads": tune.choice([2, 4]),
        "d_ff": tune.choice([64, 128]),
        "n_layers": tune.choice([1, 2]),
        "n_epochs": tune.choice([20, 30]),
        "learning_rate": tune.loguniform(0.001, 0.1),
        "batch_size": tune.choice([16, 32]),
        "max_vocab_size": tune.choice([200, 300, 500]),
        "max_seq_len": 32,
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
    """Run the from-scratch transformer pipeline."""
    logger.info("=" * 70)
    logger.info("Transformer Encoder (NumPy from scratch) Pipeline")
    logger.info("=" * 70)

    # 1. Generate data
    logger.info("Generating synthetic data...")
    texts, labels = generate_data(n_samples=600, random_state=42)
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
    model = train(X_train, y_train, d_model=64, n_heads=4, n_epochs=30)
    logger.info(f"  Training time: {time.time() - start:.3f}s")

    val_metrics = validate(model, X_val, y_val)
    logger.info(f"  Validation: {val_metrics}")

    # 4. Demonstrate attention internals
    logger.info("\n--- Attention Internals ---")
    sample_text = X_train[0]
    sample_ids = model.tokenizer.encode(sample_text, 32)[np.newaxis, :]
    x = model.classifier.embedding[sample_ids] + model.classifier.pos_encoding[:32]

    Q = np.matmul(x, model.classifier.encoder_blocks[0].mha.W_q)
    K = np.matmul(x, model.classifier.encoder_blocks[0].mha.W_k)
    d_k = Q.shape[-1]
    attn_scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
    attn_weights = softmax(attn_scores, axis=-1)
    logger.info(f"  Sample text: '{sample_text}'")
    logger.info(f"  Attention weight matrix shape: {attn_weights.shape}")
    logger.info(f"  Attention weights (first row): {attn_weights[0, 0, :8]}")

    # 5. Optuna
    logger.info("\n--- Optuna Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=15,
        show_progress_bar=True,
    )
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 6. Ray Tune
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(X_train, y_train, X_val, y_val, num_samples=10)
    logger.info(f"  Best Ray config: {best_ray_config}")

    # 7. Final model
    logger.info("\n--- Final Model ---")
    final_model = train(X_train, y_train, **study.best_params, max_seq_len=32)

    test_results = test(final_model, X_test, y_test)
    logger.info(f"  Test Accuracy:  {test_results['accuracy']:.4f}")
    logger.info(f"  Test Precision: {test_results['precision']:.4f}")
    logger.info(f"  Test Recall:    {test_results['recall']:.4f}")
    logger.info(f"  Test F1:        {test_results['f1']:.4f}")
    logger.info(f"\n{test_results['report']}")

    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info("NOTE: This is an educational implementation. For production use,")
    logger.info("prefer PyTorch/TensorFlow with pre-trained BERT weights.")


if __name__ == "__main__":
    main()
