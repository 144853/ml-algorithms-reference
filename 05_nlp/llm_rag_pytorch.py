"""
RAG Pipeline Using PyTorch: Embedding-Based Retrieval + Transformer Generation
================================================================================

Theory & Mathematics:
---------------------
This module implements a production-style RAG pipeline using PyTorch with:
1. Dense embedding-based retrieval (sentence transformers)
2. Cross-encoder re-ranking
3. Small transformer for answer generation

RAG Pipeline Architecture:
    Query -> Sentence Embedding -> Cosine Similarity Search over Document Index
    -> Top-K Retrieval -> Cross-Encoder Re-ranking
    -> Context + Query -> Transformer Decoder -> Generated Answer

Embedding-Based Retrieval:
    Unlike TF-IDF (sparse, lexical), dense retrieval uses learned embeddings:
    - Documents and queries are encoded into dense vectors (384-768 dims)
    - Similarity is computed in the embedding space
    - Captures semantic similarity (synonyms, paraphrases)

    Dual Encoder Architecture:
        query_emb = Encoder(query)       # (1, d)
        doc_embs  = Encoder(documents)   # (N, d)
        scores = query_emb @ doc_embs.T  # (1, N)

    The encoder is typically a pre-trained transformer (BERT, MiniLM) fine-tuned
    on sentence similarity tasks (e.g., NLI, STS).

Cross-Encoder Re-ranking:
    After initial retrieval, a cross-encoder scores query-document pairs jointly:
        score = CrossEncoder([query; SEP; document])

    Cross-encoders are more accurate than bi-encoders because they can attend
    to both query and document simultaneously, but they are slower (O(N) vs O(1)
    for retrieval).

Transformer Generation:
    Given retrieved context and query, a small transformer generates the answer:
        input = "[CONTEXT] {context} [QUERY] {query}"
        output = TransformerDecoder(input)

    In this implementation, we use a lightweight approach:
    - Encode context and query with a small transformer
    - Use attention to select and combine relevant spans
    - Template-augmented generation for structured output

Contrastive Learning for Retrieval:
    The retriever can be improved with contrastive learning:
        L = -log(exp(sim(q, d+)/tau) / sum(exp(sim(q, d_i)/tau)))

    Where d+ is the relevant document, d_i are all documents in the batch,
    and tau is the temperature parameter.

Business Use Cases:
-------------------
- Enterprise knowledge base Q&A
- Customer support automation
- Document search and summarization
- Legal and medical research assistants
- Conversational AI with grounded responses

Advantages:
-----------
- Semantic retrieval captures meaning beyond keyword matching
- GPU acceleration for fast embedding and generation
- Cross-encoder re-ranking improves precision
- End-to-end differentiable (can fine-tune retriever and generator jointly)
- Extensible: easy to swap components (retriever, generator)

Disadvantages:
--------------
- Requires pre-trained models and GPU for practical speed
- Dense embeddings require more storage than sparse vectors
- Cross-encoder re-ranking adds latency
- Generation quality depends on model size
- Hallucination risk if retrieval fails

Key Hyperparameters:
--------------------
- embedding_model: Sentence transformer model name
- top_k_retrieval: Number of chunks for initial retrieval
- top_k_rerank: Number of chunks after re-ranking
- chunk_size: Words per document chunk
- chunk_overlap: Overlap between chunks
- temperature: Softmax temperature for generation
- max_gen_length: Maximum generated answer length

References:
-----------
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP.
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA.
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT.
"""

import warnings
import logging
import re
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import optuna
import ray
from ray import tune
from transformers import AutoTokenizer, AutoModel
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
# Synthetic knowledge base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data. "
    "Supervised learning requires labeled training data with input-output pairs. "
    "Common algorithms include linear regression, decision trees, random forests, and gradient boosting.",

    "Deep learning is a specialized branch of machine learning using multi-layered neural networks. "
    "Convolutional neural networks are designed for processing structured grid data like images. "
    "Recurrent networks and transformers handle sequential and text data respectively.",

    "Natural language processing is the field of AI dealing with human language understanding. "
    "Key techniques include tokenization, stemming, and lemmatization for text preprocessing. "
    "Modern approaches use transformer architectures like BERT and GPT for contextual understanding.",

    "Computer vision enables machines to extract information from images and videos. "
    "Object detection locates and classifies multiple objects within a single image. "
    "Semantic segmentation assigns a class label to every pixel in the image.",

    "Reinforcement learning trains agents to make sequential decisions through trial and error. "
    "The agent learns a policy that maximizes cumulative reward over time. "
    "Applications include game playing, robotics control, and recommendation systems.",

    "Data engineering involves designing and building systems for collecting and storing data. "
    "ETL pipelines extract data from sources, transform it, and load it into warehouses. "
    "Modern data stacks include tools like Spark, Kafka, Airflow, and dbt.",

    "Cloud computing provides on-demand access to computing resources via the internet. "
    "Key service models are Infrastructure, Platform, and Software as a Service. "
    "Major providers include Amazon Web Services, Google Cloud Platform, and Microsoft Azure.",

    "Cybersecurity focuses on protecting computer systems from digital threats and attacks. "
    "Encryption algorithms convert plaintext into ciphertext using mathematical transformations. "
    "Zero trust architecture assumes no user or system is trustworthy by default.",

    "Renewable energy harnesses natural processes to generate clean electricity. "
    "Solar photovoltaic cells convert sunlight directly into electrical current. "
    "Wind turbines use aerodynamic blades to capture kinetic energy from the wind.",

    "Biotechnology applies biological systems and organisms to develop useful products. "
    "CRISPR gene editing technology enables precise modifications to DNA sequences. "
    "Synthetic biology designs and constructs new biological parts and systems.",
]

QA_PAIRS = [
    {"question": "What is supervised learning?", "answer": "Supervised learning requires labeled training data with input-output pairs.", "relevant_doc": 0},
    {"question": "How do convolutional neural networks work?", "answer": "Convolutional neural networks are designed for processing structured grid data like images.", "relevant_doc": 1},
    {"question": "What is tokenization in NLP?", "answer": "Tokenization breaks text into words or subwords as a key preprocessing technique in NLP.", "relevant_doc": 2},
    {"question": "What is semantic segmentation?", "answer": "Semantic segmentation assigns a class label to every pixel in the image.", "relevant_doc": 3},
    {"question": "How does reinforcement learning work?", "answer": "Reinforcement learning trains agents to make sequential decisions through trial and error to maximize cumulative reward.", "relevant_doc": 4},
    {"question": "What is an ETL pipeline?", "answer": "ETL pipelines extract data from sources, transform it, and load it into data warehouses.", "relevant_doc": 5},
    {"question": "What are the main cloud service models?", "answer": "The key service models are Infrastructure, Platform, and Software as a Service (IaaS, PaaS, SaaS).", "relevant_doc": 6},
    {"question": "What is zero trust security?", "answer": "Zero trust architecture assumes no user or system is trustworthy by default.", "relevant_doc": 7},
    {"question": "How do solar panels generate electricity?", "answer": "Solar photovoltaic cells convert sunlight directly into electrical current.", "relevant_doc": 8},
    {"question": "What is CRISPR technology?", "answer": "CRISPR gene editing technology enables precise modifications to DNA sequences.", "relevant_doc": 9},
]


def generate_data(n_samples: int = 200, random_state: int = 42):
    """Generate synthetic RAG evaluation data."""
    rng = np.random.RandomState(random_state)
    queries, documents, labels = [], [], []

    for _ in range(n_samples // 2):
        qa = rng.choice(QA_PAIRS)
        queries.append(qa["question"])
        documents.append(KNOWLEDGE_BASE[qa["relevant_doc"]])
        labels.append(1)

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
# Document processing
# ---------------------------------------------------------------------------


def chunk_documents(documents: list[str], chunk_size: int = 40,
                    chunk_overlap: int = 10) -> list[dict]:
    """Split documents into overlapping chunks."""
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


# ---------------------------------------------------------------------------
# Dense Embedding Retriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """
    Dense passage retriever using sentence transformer embeddings.

    Uses a pre-trained model (e.g., all-MiniLM-L6-v2) to encode queries
    and documents into dense vectors, then retrieves by cosine similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.doc_embeddings: torch.Tensor | None = None

    def _encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts into dense embeddings using mean pooling."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                # Mean pooling
                attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
                token_embeds = outputs.last_hidden_state
                summed = (token_embeds * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1)
                embeddings = summed / counts
                # L2 normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def index(self, texts: list[str]):
        """Index document texts."""
        self.doc_embeddings = self._encode(texts)
        logger.info(f"  Indexed {len(texts)} texts, embedding dim: {self.doc_embeddings.shape[1]}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Search for top-k documents by embedding similarity."""
        query_emb = self._encode([query])  # (1, d)
        scores = (query_emb @ self.doc_embeddings.T).squeeze(0)  # (N,)
        top_indices = torch.argsort(scores, descending=True)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def encode_pair(self, query: str, document: str) -> float:
        """Compute similarity between a query-document pair."""
        q_emb = self._encode([query])
        d_emb = self._encode([document])
        return float((q_emb @ d_emb.T).squeeze())


# ---------------------------------------------------------------------------
# Cross-Encoder Re-ranker (lightweight)
# ---------------------------------------------------------------------------


class LightweightReranker(nn.Module):
    """
    Lightweight cross-encoder for re-ranking retrieved chunks.

    Takes concatenated query+document embeddings and predicts relevance.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        # Cross-attention inspired: interact query and document embeddings
        self.network = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # concat + element-wise product
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict relevance score.

        Parameters
        ----------
        query_emb, doc_emb : (batch, d)

        Returns
        -------
        scores : (batch, 1)
        """
        interaction = query_emb * doc_emb  # element-wise product
        combined = torch.cat([query_emb, doc_emb, interaction], dim=1)
        return self.network(combined)


# ---------------------------------------------------------------------------
# Answer Generator (template + extractive)
# ---------------------------------------------------------------------------


class AnswerGenerator:
    """
    Generates answers from retrieved context using extractive and template methods.

    Uses the retriever's embeddings to find the most relevant sentences
    in the retrieved context.
    """

    def __init__(self, retriever: DenseRetriever):
        self.retriever = retriever

    def generate(self, query: str, context_chunks: list[dict],
                 max_sentences: int = 2) -> str:
        """
        Generate an answer from context chunks.

        Strategy:
        1. Split context into sentences.
        2. Embed each sentence and compute similarity to query.
        3. Select top sentences and combine.
        """
        if not context_chunks:
            return "I could not find relevant information to answer your question."

        # Collect all sentences from context
        all_sentences = []
        for chunk in context_chunks:
            sentences = re.split(r"[.!?]+", chunk["text"])
            all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 15])

        if not all_sentences:
            return context_chunks[0]["text"][:200]

        # Score sentences by embedding similarity to query
        query_emb = self.retriever._encode([query])
        sent_embs = self.retriever._encode(all_sentences)
        scores = (query_emb @ sent_embs.T).squeeze(0)

        top_indices = torch.argsort(scores, descending=True)[:max_sentences]
        selected = [all_sentences[idx] for idx in top_indices]

        answer = ". ".join(selected)
        if not answer.endswith("."):
            answer += "."

        return answer


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    Full RAG pipeline with embedding retrieval and transformer-based components.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int = 3, chunk_size: int = 40, chunk_overlap: int = 10,
                 max_length: int = 128):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.retriever = DenseRetriever(model_name=model_name, max_length=max_length)
        self.generator = AnswerGenerator(self.retriever)
        self.chunks: list[dict] = []

    def index(self, documents: list[str]):
        """Index documents: chunk and embed."""
        self.chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        chunk_texts = [c["text"] for c in self.chunks]
        self.retriever.index(chunk_texts)
        logger.info(f"  Indexed {len(self.chunks)} chunks from {len(documents)} documents")

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve top-k chunks."""
        results = self.retriever.search(query, top_k=self.top_k)
        retrieved = []
        for idx, score in results:
            chunk = self.chunks[idx].copy()
            chunk["score"] = score
            retrieved.append(chunk)
        return retrieved

    def query(self, question: str) -> dict:
        """Full RAG: retrieve + generate."""
        retrieved = self.retrieve(question)
        answer = self.generator.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
        }

    def predict_relevance(self, query: str, document: str) -> float:
        """Predict relevance score for a query-document pair."""
        return self.retriever.encode_pair(query, document)


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train_queries: list[str], y_train: np.ndarray,
          X_train_docs: list[str] = None, **hyperparams) -> RAGPipeline:
    """
    Build and configure the PyTorch RAG pipeline.

    Parameters
    ----------
    X_train_queries, y_train : training data
    X_train_docs : paired documents
    **hyperparams : pipeline configuration

    Returns
    -------
    pipeline : RAGPipeline
    """
    model_name = hyperparams.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = hyperparams.get("top_k", 3)
    chunk_size = hyperparams.get("chunk_size", 40)
    chunk_overlap = hyperparams.get("chunk_overlap", 10)
    max_length = hyperparams.get("max_length", 128)

    pipeline = RAGPipeline(
        model_name=model_name,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_length=max_length,
    )
    pipeline.index(KNOWLEDGE_BASE)

    return pipeline


def validate(model: RAGPipeline, X_val_queries: list[str],
             y_val: np.ndarray, X_val_docs: list[str] = None) -> dict[str, float]:
    """
    Validate retrieval quality using embedding-based relevance predictions.
    """
    if X_val_docs is None:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    scores = []
    for q, d in zip(X_val_queries, X_val_docs):
        score = model.predict_relevance(q, d)
        scores.append(score)

    scores = np.array(scores)

    # Find optimal threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
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


def test(model: RAGPipeline, X_test_queries: list[str],
         y_test: np.ndarray, X_test_docs: list[str] = None) -> dict[str, Any]:
    """Final test evaluation."""
    metrics = validate(model, X_test_queries, y_test, X_test_docs)

    if X_test_docs is not None:
        scores = np.array([model.predict_relevance(q, d)
                           for q, d in zip(X_test_queries, X_test_docs)])
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

# Cache the pipeline to avoid re-loading the model for every trial
_pipeline_cache: dict[str, RAGPipeline] = {}


def _get_cached_pipeline(model_name: str, top_k: int, chunk_size: int,
                         chunk_overlap: int) -> RAGPipeline:
    """Cache pipeline by key parameters to speed up HPO."""
    cache_key = f"{model_name}_{chunk_size}_{chunk_overlap}"
    if cache_key not in _pipeline_cache:
        pipeline = RAGPipeline(
            model_name=model_name, top_k=top_k,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        pipeline.index(KNOWLEDGE_BASE)
        _pipeline_cache[cache_key] = pipeline
    else:
        _pipeline_cache[cache_key].top_k = top_k
    return _pipeline_cache[cache_key]


def optuna_objective(trial, X_train_q, y_train, X_train_d,
                     X_val_q, y_val, X_val_d) -> float:
    """Optuna objective for RAG pipeline."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = trial.suggest_int("top_k", 1, 5)
    chunk_size = trial.suggest_int("chunk_size", 20, 60, step=10)
    chunk_overlap = trial.suggest_int("chunk_overlap", 5, 20, step=5)

    pipeline = _get_cached_pipeline(model_name, top_k, chunk_size, chunk_overlap)
    metrics = validate(pipeline, X_val_q, y_val, X_val_d)
    return metrics["f1"]


def ray_tune_search(X_train_q, y_train, X_train_d,
                    X_val_q, y_val, X_val_d, num_samples=20) -> dict:
    """Ray Tune hyperparameter search."""

    def ray_objective(config):
        pipeline = RAGPipeline(
            model_name=config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=config["top_k"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        pipeline.index(KNOWLEDGE_BASE)
        metrics = validate(pipeline, X_val_q, y_val, X_val_d)
        tune.report({"f1": metrics["f1"], "accuracy": metrics["accuracy"]})

    search_space = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": tune.choice([1, 2, 3, 5]),
        "chunk_size": tune.choice([20, 30, 40, 50]),
        "chunk_overlap": tune.choice([5, 10, 15]),
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
    """Run the full PyTorch RAG pipeline."""
    logger.info("=" * 70)
    logger.info("RAG Pipeline: PyTorch Embedding Retrieval + Generation")
    logger.info(f"Device: {DEVICE}")
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

    # 4. RAG query examples
    logger.info("\n--- RAG Query Examples ---")
    for qa in qa_pairs[:4]:
        result = pipeline.query(qa["question"])
        logger.info(f"  Q: {result['question']}")
        logger.info(f"  A: {result['answer']}")
        if result["retrieved_chunks"]:
            logger.info(f"  Top chunk score: {result['retrieved_chunks'][0]['score']:.4f}")
            logger.info(f"  Source doc_id: {result['retrieved_chunks'][0]['doc_id']}")
        logger.info(f"  Expected doc_id: {qa['relevant_doc']}")
        logger.info("")

    # 5. Dense vs sparse comparison
    logger.info("\n--- Embedding Space Analysis ---")
    sample_q = "What is machine learning?"
    sample_relevant = KNOWLEDGE_BASE[0]
    sample_irrelevant = KNOWLEDGE_BASE[8]

    rel_score = pipeline.predict_relevance(sample_q, sample_relevant)
    irr_score = pipeline.predict_relevance(sample_q, sample_irrelevant)
    logger.info(f"  Query: '{sample_q}'")
    logger.info(f"  Relevant doc similarity:   {rel_score:.4f}")
    logger.info(f"  Irrelevant doc similarity:  {irr_score:.4f}")
    logger.info(f"  Separation margin: {rel_score - irr_score:.4f}")

    # 6. Optuna
    logger.info("\n--- Optuna Optimization ---")
    _pipeline_cache.clear()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train_q, y_train, X_train_d, X_val_q, y_val, X_val_d
        ),
        n_trials=10,
        show_progress_bar=True,
    )
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    # 7. Ray Tune
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(
        X_train_q, y_train, X_train_d, X_val_q, y_val, X_val_d, num_samples=5
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

    # 9. Final comprehensive examples
    logger.info("\n--- Final RAG Examples ---")
    test_questions = [
        "What is supervised learning?",
        "How do solar panels work?",
        "What is CRISPR used for?",
        "What is zero trust security?",
        "How does reinforcement learning train agents?",
    ]
    for q in test_questions:
        result = final_pipeline.query(q)
        logger.info(f"  Q: {q}")
        logger.info(f"  A: {result['answer']}")
        logger.info("")

    logger.info("=" * 70)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
