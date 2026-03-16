"""
RAG From Scratch: Cosine Similarity Retrieval + Template Generation - NumPy
============================================================================

Theory & Mathematics:
---------------------
This module implements a complete Retrieval-Augmented Generation (RAG) pipeline
from scratch using only NumPy. Every component - tokenization, TF-IDF
vectorization, cosine similarity retrieval, and template-based generation -
is built without any ML library dependencies (only NumPy + sklearn metrics).

RAG Pipeline (from scratch):
    1. DOCUMENT PROCESSING
       - Chunking: Split documents into overlapping segments
       - Tokenization: Lowercase, remove punctuation, split on whitespace

    2. INDEXING (TF-IDF from scratch)
       - Build vocabulary from all chunks
       - Compute Term Frequency: TF(t,d) = count(t,d) / |d|
       - Compute Inverse Document Frequency: IDF(t) = log((1+N)/(1+df(t))) + 1
       - TF-IDF(t,d) = TF(t,d) * IDF(t)
       - L2-normalize each document vector

    3. RETRIEVAL (Cosine Similarity from scratch)
       - Vectorize the query using the same vocabulary and IDF values
       - Compute cosine similarity: cos(q,d) = (q . d) / (||q|| * ||d||)
         Since vectors are L2-normalized: cos(q,d) = q . d
       - Return top-k chunks by similarity

    4. RE-RANKING (BM25 from scratch)
       BM25(q, d) = sum over q_i:
           IDF(q_i) * (f(q_i,d) * (k1+1)) / (f(q_i,d) + k1*(1-b+b*|d|/avgdl))
       Default: k1=1.5, b=0.75

    5. GENERATION (Template-based)
       - Extract most relevant sentences from retrieved chunks
       - Score sentences by query term overlap
       - Construct answer from best matching sentences
       - Apply answer templates for formatting

Cosine Similarity Derivation:
    For vectors a, b:
        cos(a, b) = sum(a_i * b_i) / (sqrt(sum(a_i^2)) * sqrt(sum(b_i^2)))

    Range: [-1, 1] for general vectors, [0, 1] for TF-IDF (non-negative).
    Cosine similarity measures directional alignment, ignoring magnitude.
    This makes it effective for document comparison where length varies.

Business Use Cases:
-------------------
- Educational RAG system implementation
- Lightweight Q&A without external dependencies
- Embedded or restricted environments
- Understanding RAG internals for debugging production systems
- Prototyping retrieval strategies

Advantages:
-----------
- Complete transparency: every computation visible
- No ML framework dependencies
- Lightweight and portable
- Excellent for learning and teaching
- Fast for small knowledge bases

Disadvantages:
--------------
- No semantic understanding (lexical matching only)
- Template-based generation is rigid
- Manual implementation is slower than optimized libraries
- Cannot handle large-scale knowledge bases efficiently
- No deep learning-based re-ranking or generation

Key Hyperparameters:
--------------------
- max_vocab_size: Maximum vocabulary size for TF-IDF
- top_k: Number of chunks to retrieve
- chunk_size: Words per chunk
- chunk_overlap: Overlap between chunks
- bm25_k1: BM25 term frequency saturation parameter
- bm25_b: BM25 document length normalization parameter

References:
-----------
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP.
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25.
- Manning, Raghavan & Schutze (2008). Introduction to Information Retrieval.
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
# Synthetic knowledge base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    "Machine learning is a branch of artificial intelligence focused on building systems that learn from data. "
    "Supervised learning uses labeled examples. Unsupervised learning finds patterns in unlabeled data. "
    "Reinforcement learning trains agents through rewards and penalties in an environment.",

    "Deep learning uses neural networks with many layers to model complex patterns. "
    "Convolutional neural networks process grid-like data such as images. "
    "Recurrent neural networks handle sequences like text and time series data.",

    "Natural language processing enables computers to understand human language. "
    "Tokenization breaks text into words or subwords. Embeddings represent words as dense vectors. "
    "Transformers use attention mechanisms to process text in parallel.",

    "Computer vision enables machines to interpret visual information from the world. "
    "Object detection identifies and localizes objects in images. "
    "Image segmentation assigns a class label to each pixel in an image.",

    "Databases store and manage structured data efficiently. "
    "Relational databases use SQL for querying. NoSQL databases handle unstructured data. "
    "Graph databases excel at representing relationships between entities.",

    "Cloud computing delivers computing services over the internet. "
    "Infrastructure as a Service provides virtual computing resources. "
    "Platform as a Service offers development and deployment environments. "
    "Software as a Service delivers applications over the internet.",

    "Cybersecurity protects digital systems and data from unauthorized access. "
    "Encryption converts data into coded form that only authorized parties can read. "
    "Firewalls monitor and filter network traffic based on security rules.",

    "Renewable energy sources include solar, wind, and hydroelectric power. "
    "Solar panels convert sunlight directly into electricity using photovoltaic cells. "
    "Wind turbines capture kinetic energy from moving air to generate power.",

    "Genetics is the study of genes and heredity in living organisms. "
    "DNA carries the genetic instructions for development and functioning. "
    "CRISPR technology allows precise editing of genetic material in cells.",

    "Quantum computing uses quantum mechanical phenomena for computation. "
    "Qubits can exist in superposition of states unlike classical bits. "
    "Quantum entanglement allows correlated measurements across distances.",
]

QA_PAIRS = [
    {"question": "What is machine learning?", "answer": "Machine learning is a branch of artificial intelligence focused on building systems that learn from data.", "relevant_doc": 0},
    {"question": "What are neural networks used for in deep learning?", "answer": "Deep learning uses neural networks with many layers to model complex patterns in data.", "relevant_doc": 1},
    {"question": "How do transformers process text?", "answer": "Transformers use attention mechanisms to process text in parallel.", "relevant_doc": 2},
    {"question": "What is object detection?", "answer": "Object detection identifies and localizes objects in images.", "relevant_doc": 3},
    {"question": "What types of databases exist?", "answer": "There are relational databases using SQL, NoSQL databases for unstructured data, and graph databases for relationships.", "relevant_doc": 4},
    {"question": "What is cloud computing?", "answer": "Cloud computing delivers computing services over the internet including IaaS, PaaS, and SaaS.", "relevant_doc": 5},
    {"question": "How does encryption protect data?", "answer": "Encryption converts data into coded form that only authorized parties can read.", "relevant_doc": 6},
    {"question": "How do solar panels work?", "answer": "Solar panels convert sunlight directly into electricity using photovoltaic cells.", "relevant_doc": 7},
    {"question": "What is CRISPR?", "answer": "CRISPR technology allows precise editing of genetic material in cells.", "relevant_doc": 8},
    {"question": "What makes quantum computing different?", "answer": "Quantum computing uses qubits that can exist in superposition, unlike classical bits.", "relevant_doc": 9},
]


def generate_data(n_samples: int = 200, random_state: int = 42):
    """
    Generate synthetic RAG evaluation data with relevance labels.

    Returns query-document pairs with labels (1=relevant, 0=irrelevant).
    """
    rng = np.random.RandomState(random_state)
    queries, documents, labels = [], [], []

    for _ in range(n_samples // 2):
        qa = rng.choice(QA_PAIRS)
        # Positive pair
        queries.append(qa["question"])
        documents.append(KNOWLEDGE_BASE[qa["relevant_doc"]])
        labels.append(1)
        # Negative pair
        neg_idx = rng.choice([i for i in range(len(KNOWLEDGE_BASE)) if i != qa["relevant_doc"]])
        queries.append(qa["question"])
        documents.append(KNOWLEDGE_BASE[neg_idx])
        labels.append(0)

    idx = rng.permutation(len(queries))
    queries = [queries[i] for i in idx]
    documents = [documents[i] for i in idx]
    labels = np.array([labels[i] for i in idx])
    return queries, documents, labels, QA_PAIRS


# ---------------------------------------------------------------------------
# From-scratch components
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def chunk_documents(documents: list[str], chunk_size: int = 40,
                    chunk_overlap: int = 10) -> list[dict]:
    """Split documents into overlapping word-level chunks."""
    chunks = []
    for doc_id, doc in enumerate(documents):
        words = doc.split()
        stride = max(chunk_size - chunk_overlap, 1)
        for i in range(0, len(words), stride):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 5:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "doc_id": doc_id,
                "chunk_id": len(chunks),
            })
    return chunks


class TfidfIndexFromScratch:
    """
    TF-IDF index built entirely from scratch with NumPy.

    Supports:
    - Vocabulary building with frequency cutoffs
    - TF-IDF vectorization with sublinear TF
    - L2-normalized document vectors
    - Cosine similarity retrieval
    - BM25 scoring
    """

    def __init__(self, max_vocab_size: int = 2000, min_df: int = 1,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.max_vocab_size = max_vocab_size
        self.min_df = min_df
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.doc_vectors: np.ndarray | None = None
        self.doc_lengths: np.ndarray | None = None
        self.avg_doc_length: float = 0.0
        self.doc_term_freqs: list[dict[int, int]] = []

    def build_index(self, texts: list[str]):
        """Build vocabulary, IDF, and document vectors."""
        N = len(texts)
        tokenized_docs = [tokenize(t) for t in texts]
        self.doc_lengths = np.array([len(doc) for doc in tokenized_docs], dtype=np.float64)
        self.avg_doc_length = np.mean(self.doc_lengths)

        # Build vocabulary
        df_counter = Counter()
        freq_counter = Counter()
        for tokens in tokenized_docs:
            freq_counter.update(tokens)
            df_counter.update(set(tokens))

        valid = {t for t, df in df_counter.items() if df >= self.min_df}
        sorted_tokens = sorted(
            [(t, freq_counter[t]) for t in valid], key=lambda x: x[1], reverse=True
        )[:self.max_vocab_size]
        self.vocabulary = {t: i for i, (t, _) in enumerate(sorted_tokens)}
        V = len(self.vocabulary)

        # IDF
        self.idf = np.zeros(V)
        for token, idx in self.vocabulary.items():
            df = df_counter[token]
            self.idf[idx] = np.log((1 + N) / (1 + df)) + 1

        # Document vectors (TF-IDF with sublinear TF, L2-normalized)
        self.doc_vectors = np.zeros((N, V))
        self.doc_term_freqs = []
        for i, tokens in enumerate(tokenized_docs):
            token_counts = Counter(tokens)
            tf_dict = {}
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    tf = 1 + np.log(count) if count > 0 else 0
                    self.doc_vectors[i, idx] = tf * self.idf[idx]
                    tf_dict[idx] = count
            self.doc_term_freqs.append(tf_dict)
            # L2 normalize
            norm = np.linalg.norm(self.doc_vectors[i])
            if norm > 0:
                self.doc_vectors[i] /= norm

    def vectorize_query(self, query: str) -> np.ndarray:
        """Vectorize a query using the existing vocabulary and IDF."""
        V = len(self.vocabulary)
        vec = np.zeros(V)
        tokens = tokenize(query)
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = 1 + np.log(count) if count > 0 else 0
                vec[idx] = tf * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def cosine_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Retrieve top-k documents by cosine similarity."""
        q_vec = self.vectorize_query(query)
        # Since vectors are L2-normalized, cosine sim = dot product
        similarities = self.doc_vectors @ q_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def bm25_score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query-document pair."""
        tokens = tokenize(query)
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.doc_term_freqs[doc_idx]

        for token in set(tokens):
            if token not in self.vocabulary:
                continue
            idx = self.vocabulary[token]
            tf = tf_dict.get(idx, 0)
            idf_val = self.idf[idx]

            numerator = tf * (self.bm25_k1 + 1)
            denominator = tf + self.bm25_k1 * (
                1 - self.bm25_b + self.bm25_b * doc_len / self.avg_doc_length
            )
            score += idf_val * numerator / denominator

        return score

    def hybrid_search(self, query: str, top_k: int = 5,
                      alpha: float = 0.7) -> list[tuple[int, float]]:
        """
        Hybrid search: weighted combination of cosine similarity and BM25.

        score = alpha * cosine_sim + (1-alpha) * normalized_bm25
        """
        q_vec = self.vectorize_query(query)
        cos_sims = self.doc_vectors @ q_vec

        bm25_scores = np.array([self.bm25_score(query, i) for i in range(len(self.doc_vectors))])
        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_normalized = bm25_scores / bm25_max
        else:
            bm25_normalized = bm25_scores

        combined = alpha * cos_sims + (1 - alpha) * bm25_normalized
        top_indices = np.argsort(combined)[::-1][:top_k]
        return [(int(idx), float(combined[idx])) for idx in top_indices]


# ---------------------------------------------------------------------------
# Template-based generation
# ---------------------------------------------------------------------------


def extract_best_sentences(query: str, context: str, max_sentences: int = 2) -> str:
    """
    Extract the most relevant sentences from context for a query.

    Scores each sentence by query term overlap and returns the best ones.
    """
    sentences = re.split(r"[.!?]+", context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return context[:200]

    query_terms = set(tokenize(query))
    # Remove stopwords
    stopwords = {"what", "is", "how", "does", "do", "the", "a", "an", "in", "of",
                 "for", "to", "and", "or", "are", "was", "were", "be", "been", "being"}
    query_terms -= stopwords

    scored = []
    for sent in sentences:
        sent_terms = set(tokenize(sent))
        overlap = len(query_terms & sent_terms)
        coverage = overlap / max(len(query_terms), 1)
        scored.append((sent, coverage))

    scored.sort(key=lambda x: x[1], reverse=True)
    best = [s for s, _ in scored[:max_sentences]]
    return ". ".join(best) + "."


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Generate an answer from retrieved chunks using templates.

    Parameters
    ----------
    query : str
    retrieved_chunks : list[dict]
        Each dict has 'text' and 'score'.

    Returns
    -------
    answer : str
    """
    if not retrieved_chunks:
        return "I could not find relevant information to answer your question."

    # Combine text from top chunks
    combined_context = " ".join([c["text"] for c in retrieved_chunks[:2]])
    best_extract = extract_best_sentences(query, combined_context)

    return best_extract


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipelineFromScratch:
    """Complete RAG pipeline built from scratch with NumPy."""

    def __init__(self, max_vocab_size: int = 2000, top_k: int = 3,
                 chunk_size: int = 40, chunk_overlap: int = 10,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75,
                 search_alpha: float = 0.7):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_alpha = search_alpha
        self.index = TfidfIndexFromScratch(
            max_vocab_size=max_vocab_size, bm25_k1=bm25_k1, bm25_b=bm25_b
        )
        self.chunks: list[dict] = []

    def build(self, documents: list[str]):
        """Index documents."""
        self.chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        chunk_texts = [c["text"] for c in self.chunks]
        self.index.build_index(chunk_texts)
        logger.info(f"  Indexed {len(self.chunks)} chunks (vocab: {len(self.index.vocabulary)})")

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve top-k chunks for a query using hybrid search."""
        results = self.index.hybrid_search(query, self.top_k, self.search_alpha)
        retrieved = []
        for idx, score in results:
            chunk = self.chunks[idx].copy()
            chunk["score"] = score
            retrieved.append(chunk)
        return retrieved

    def query(self, question: str) -> dict:
        """Full RAG: retrieve + generate."""
        retrieved = self.retrieve(question)
        answer = generate_answer(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
        }

    def predict_relevance(self, query: str, document: str) -> float:
        """Predict relevance score for a query-document pair."""
        q_vec = self.index.vectorize_query(query)
        d_tokens = tokenize(document)
        d_counter = Counter(d_tokens)
        V = len(self.index.vocabulary)
        d_vec = np.zeros(V)
        for token, count in d_counter.items():
            if token in self.index.vocabulary:
                idx = self.index.vocabulary[token]
                tf = 1 + np.log(count) if count > 0 else 0
                d_vec[idx] = tf * self.index.idf[idx]
        norm = np.linalg.norm(d_vec)
        if norm > 0:
            d_vec /= norm
        return float(np.dot(q_vec, d_vec))


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train_queries: list[str], y_train: np.ndarray,
          X_train_docs: list[str] = None, **hyperparams) -> RAGPipelineFromScratch:
    """
    Build and configure the RAG pipeline.

    Parameters
    ----------
    X_train_queries, y_train : training data (used for threshold calibration)
    X_train_docs : paired documents
    **hyperparams : pipeline configuration

    Returns
    -------
    pipeline : RAGPipelineFromScratch
    """
    pipeline = RAGPipelineFromScratch(
        max_vocab_size=hyperparams.get("max_vocab_size", 2000),
        top_k=hyperparams.get("top_k", 3),
        chunk_size=hyperparams.get("chunk_size", 40),
        chunk_overlap=hyperparams.get("chunk_overlap", 10),
        bm25_k1=hyperparams.get("bm25_k1", 1.5),
        bm25_b=hyperparams.get("bm25_b", 0.75),
        search_alpha=hyperparams.get("search_alpha", 0.7),
    )
    pipeline.build(KNOWLEDGE_BASE)
    return pipeline


def validate(model: RAGPipelineFromScratch, X_val_queries: list[str],
             y_val: np.ndarray, X_val_docs: list[str] = None) -> dict[str, float]:
    """
    Validate retrieval quality using relevance predictions.

    Uses cosine similarity threshold to predict relevance.
    """
    if X_val_docs is None:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    scores = []
    for q, d in zip(X_val_queries, X_val_docs):
        score = model.predict_relevance(q, d)
        scores.append(score)

    scores = np.array(scores)
    # Find best threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred = (scores >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (scores >= best_thresh).astype(int)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }


def test(model: RAGPipelineFromScratch, X_test_queries: list[str],
         y_test: np.ndarray, X_test_docs: list[str] = None) -> dict[str, Any]:
    """Final test evaluation."""
    metrics = validate(model, X_test_queries, y_test, X_test_docs)

    if X_test_docs is not None:
        scores = np.array([model.predict_relevance(q, d)
                           for q, d in zip(X_test_queries, X_test_docs)])
        # Use median as threshold
        thresh = np.median(scores)
        y_pred = (scores >= thresh).astype(int)
        metrics["report"] = classification_report(
            y_test, y_pred, target_names=["irrelevant", "relevant"], zero_division=0
        )
    else:
        metrics["report"] = "No documents provided for evaluation."

    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------


def optuna_objective(trial, X_train_q, y_train, X_train_d,
                     X_val_q, y_val, X_val_d) -> float:
    """Optuna objective for RAG pipeline."""
    params = {
        "max_vocab_size": trial.suggest_int("max_vocab_size", 500, 5000, step=500),
        "top_k": trial.suggest_int("top_k", 1, 5),
        "chunk_size": trial.suggest_int("chunk_size", 20, 60, step=10),
        "chunk_overlap": trial.suggest_int("chunk_overlap", 5, 20, step=5),
        "bm25_k1": trial.suggest_float("bm25_k1", 0.5, 3.0),
        "bm25_b": trial.suggest_float("bm25_b", 0.3, 0.9),
        "search_alpha": trial.suggest_float("search_alpha", 0.3, 0.9),
    }
    model = train(X_train_q, y_train, X_train_d, **params)
    metrics = validate(model, X_val_q, y_val, X_val_d)
    return metrics["f1"]


def ray_tune_search(X_train_q, y_train, X_train_d,
                    X_val_q, y_val, X_val_d, num_samples=20) -> dict:
    """Ray Tune hyperparameter search."""

    def ray_objective(config):
        model = train(X_train_q, y_train, X_train_d, **config)
        metrics = validate(model, X_val_q, y_val, X_val_d)
        tune.report({"f1": metrics["f1"], "accuracy": metrics["accuracy"]})

    search_space = {
        "max_vocab_size": tune.choice([1000, 2000, 3000]),
        "top_k": tune.choice([1, 2, 3, 5]),
        "chunk_size": tune.choice([20, 30, 40, 50]),
        "chunk_overlap": tune.choice([5, 10, 15]),
        "bm25_k1": tune.uniform(0.5, 3.0),
        "bm25_b": tune.uniform(0.3, 0.9),
        "search_alpha": tune.uniform(0.3, 0.9),
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
    """Run the from-scratch RAG pipeline."""
    logger.info("=" * 70)
    logger.info("RAG Pipeline From Scratch (NumPy)")
    logger.info("=" * 70)

    # 1. Generate data
    logger.info("Generating synthetic RAG data...")
    queries, documents, labels, qa_pairs = generate_data(n_samples=200, random_state=42)
    logger.info(f"  Total pairs: {len(queries)}")
    logger.info(f"  Knowledge base docs: {len(KNOWLEDGE_BASE)}")

    # 2. Split
    indices = np.arange(len(queries))
    idx_temp, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.2, random_state=42, stratify=labels[idx_temp])

    X_train_q = [queries[i] for i in idx_train]
    X_train_d = [documents[i] for i in idx_train]
    y_train = labels[idx_train]
    X_val_q = [queries[i] for i in idx_val]
    X_val_d = [documents[i] for i in idx_val]
    y_val = labels[idx_val]
    X_test_q = [queries[i] for i in idx_test]
    X_test_d = [documents[i] for i in idx_test]
    y_test = labels[idx_test]

    logger.info(f"  Train: {len(X_train_q)}, Val: {len(X_val_q)}, Test: {len(X_test_q)}")

    # 3. Build baseline
    logger.info("\n--- Baseline RAG Pipeline ---")
    start = time.time()
    pipeline = train(X_train_q, y_train, X_train_d)
    logger.info(f"  Build time: {time.time() - start:.3f}s")

    val_metrics = validate(pipeline, X_val_q, y_val, X_val_d)
    logger.info(f"  Validation: {val_metrics}")

    # 4. Show internals
    logger.info("\n--- From-Scratch Component Details ---")
    logger.info(f"  Vocabulary size: {len(pipeline.index.vocabulary)}")
    logger.info(f"  Number of chunks: {len(pipeline.chunks)}")
    logger.info(f"  Doc vectors shape: {pipeline.index.doc_vectors.shape}")
    logger.info(f"  Average doc length: {pipeline.index.avg_doc_length:.1f}")
    sample_vocab = list(pipeline.index.vocabulary.keys())[:10]
    logger.info(f"  Sample vocabulary: {sample_vocab}")

    # 5. RAG query examples
    logger.info("\n--- RAG Query Examples ---")
    for qa in qa_pairs[:4]:
        result = pipeline.query(qa["question"])
        logger.info(f"  Q: {result['question']}")
        logger.info(f"  A: {result['answer']}")
        logger.info(f"  Top chunk score: {result['retrieved_chunks'][0]['score']:.4f}")
        logger.info(f"  Expected: {qa['answer'][:80]}...")
        logger.info("")

    # 6. Optuna
    logger.info("\n--- Optuna Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train_q, y_train, X_train_d, X_val_q, y_val, X_val_d
        ),
        n_trials=20,
        show_progress_bar=True,
    )
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 7. Ray Tune
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(
        X_train_q, y_train, X_train_d, X_val_q, y_val, X_val_d, num_samples=10
    )
    logger.info(f"  Best Ray config: {best_ray_config}")

    # 8. Final model
    logger.info("\n--- Final Model ---")
    final_pipeline = train(X_train_q, y_train, X_train_d, **study.best_params)

    test_results = test(final_pipeline, X_test_q, y_test, X_test_d)
    logger.info(f"  Test Accuracy:  {test_results['accuracy']:.4f}")
    logger.info(f"  Test Precision: {test_results['precision']:.4f}")
    logger.info(f"  Test Recall:    {test_results['recall']:.4f}")
    logger.info(f"  Test F1:        {test_results['f1']:.4f}")
    logger.info(f"\n{test_results['report']}")

    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info("NOTE: This is an educational from-scratch implementation.")


if __name__ == "__main__":
    main()
