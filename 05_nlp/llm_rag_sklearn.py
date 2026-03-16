"""
RAG Pipeline: TF-IDF Retrieval + sklearn Answer Scoring
=========================================================

Theory & Mathematics:
---------------------
Retrieval-Augmented Generation (RAG) is a paradigm that combines information
retrieval with text generation to produce grounded, factual answers. This
module implements a RAG pipeline using TF-IDF for retrieval and sklearn
classifiers for answer relevance scoring.

RAG Pipeline Architecture:
    1. INDEXING: Documents are chunked and vectorized using TF-IDF.
    2. RETRIEVAL: Given a query, find top-k most relevant document chunks
       using cosine similarity on TF-IDF vectors.
    3. ANSWER SCORING: An sklearn classifier scores and ranks candidate
       answers based on query-context relevance features.
    4. GENERATION: Template-based answer construction using retrieved context.

TF-IDF Retrieval:
    - Each document chunk is represented as a TF-IDF vector.
    - Query is also vectorized using the same TF-IDF vocabulary.
    - Cosine similarity measures relevance:
        cos(q, d) = (q . d) / (||q|| * ||d||)
    - Top-k chunks with highest cosine similarity are retrieved.

Answer Relevance Scoring:
    Features for the sklearn classifier:
    - Cosine similarity between query and chunk
    - Term overlap ratio
    - Chunk length (normalized)
    - Query term coverage in chunk
    - BM25-like score

    The classifier predicts a relevance score [0, 1] for each
    query-context pair, used to re-rank retrieved results.

BM25 Scoring (used as a feature):
    BM25(q, d) = sum over q_i in q:
        IDF(q_i) * (f(q_i, d) * (k1 + 1)) / (f(q_i, d) + k1 * (1 - b + b * |d|/avgdl))

    Where f(q_i, d) is term frequency of q_i in d, k1 and b are parameters,
    |d| is document length, avgdl is average document length.

Business Use Cases:
-------------------
- Customer support: Retrieve relevant knowledge base articles
- Enterprise search: Find and summarize relevant documents
- Legal research: Search through case law and regulations
- Medical Q&A: Retrieve relevant clinical guidelines
- FAQ systems: Match questions to pre-written answers

Advantages:
-----------
- No GPU required; runs entirely on CPU
- Interpretable: can inspect which chunks were retrieved and why
- Fast indexing and retrieval with sparse TF-IDF vectors
- Sklearn classifier provides calibrated relevance scores
- Easy to add new documents without retraining the full system

Disadvantages:
--------------
- TF-IDF retrieval misses semantic similarity (synonym problem)
- Template-based generation is rigid and non-creative
- Cannot generate novel text beyond templates
- Performance degrades with out-of-vocabulary query terms
- No deep understanding of context or question semantics

Key Hyperparameters:
--------------------
- max_features: TF-IDF vocabulary size
- top_k: Number of chunks to retrieve
- chunk_size: Number of words per chunk
- chunk_overlap: Overlap between consecutive chunks
- C: Regularization for the relevance classifier

References:
-----------
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP.
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25.
"""

import warnings
import logging
import re
import time
from typing import Any

import numpy as np
import optuna
import ray
from ray import tune
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
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
# Synthetic knowledge base and Q&A data
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    # Technology
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
    "It uses statistical techniques to give computers the ability to learn without being explicitly programmed. "
    "Common algorithms include decision trees, random forests, neural networks, and support vector machines.",

    "Deep learning is a branch of machine learning based on artificial neural networks with multiple layers. "
    "Convolutional neural networks excel at image recognition tasks. Recurrent neural networks handle sequential data. "
    "Transformers have revolutionized natural language processing with self-attention mechanisms.",

    "Natural language processing enables computers to understand and generate human language. "
    "Key tasks include sentiment analysis, named entity recognition, and machine translation. "
    "Modern NLP relies heavily on transformer models like BERT and GPT for contextual understanding.",

    "Cloud computing provides on-demand computing resources over the internet. "
    "Major providers include AWS, Google Cloud, and Microsoft Azure. "
    "Services range from virtual machines to managed databases and serverless computing.",

    "Cybersecurity protects systems and data from digital attacks. "
    "Common threats include phishing, malware, ransomware, and social engineering. "
    "Defense strategies include firewalls, encryption, multi-factor authentication, and regular audits.",

    # Science
    "Climate change refers to long-term shifts in global temperatures and weather patterns. "
    "Human activities, particularly burning fossil fuels, have been the main driver since the 1800s. "
    "Effects include rising sea levels, extreme weather events, and loss of biodiversity.",

    "Renewable energy comes from sources that naturally replenish. "
    "Solar panels convert sunlight into electricity using photovoltaic cells. "
    "Wind turbines generate power from moving air. Hydroelectric dams use flowing water.",

    "The human genome contains approximately 20,000 to 25,000 protein-coding genes. "
    "DNA sequencing technology has advanced rapidly with next-generation methods. "
    "Gene editing tools like CRISPR allow precise modifications to genetic material.",

    # Business
    "Agile methodology is an iterative approach to project management and software development. "
    "Key practices include sprints, daily standups, retrospectives, and user stories. "
    "Scrum and Kanban are popular frameworks within the agile philosophy.",

    "Data analytics transforms raw data into actionable insights for business decisions. "
    "Descriptive analytics summarizes past data. Predictive analytics forecasts future trends. "
    "Prescriptive analytics recommends optimal actions based on data analysis.",
]

QA_PAIRS = [
    {"question": "What is machine learning?", "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data using statistical techniques.", "relevant_doc": 0},
    {"question": "How do neural networks work in deep learning?", "answer": "Deep learning uses artificial neural networks with multiple layers. CNNs handle images and RNNs handle sequential data.", "relevant_doc": 1},
    {"question": "What are the main NLP tasks?", "answer": "Key NLP tasks include sentiment analysis, named entity recognition, and machine translation using transformer models.", "relevant_doc": 2},
    {"question": "What services does cloud computing provide?", "answer": "Cloud computing provides on-demand computing resources including virtual machines, managed databases, and serverless computing.", "relevant_doc": 3},
    {"question": "What are common cybersecurity threats?", "answer": "Common threats include phishing, malware, ransomware, and social engineering attacks.", "relevant_doc": 4},
    {"question": "What causes climate change?", "answer": "Human activities, particularly burning fossil fuels, have been the main driver of climate change since the 1800s.", "relevant_doc": 5},
    {"question": "How does solar energy work?", "answer": "Solar panels convert sunlight into electricity using photovoltaic cells as a renewable energy source.", "relevant_doc": 6},
    {"question": "What is CRISPR used for?", "answer": "CRISPR is a gene editing tool that allows precise modifications to genetic material in the human genome.", "relevant_doc": 7},
    {"question": "What is agile methodology?", "answer": "Agile is an iterative approach to project management using sprints, standups, and retrospectives.", "relevant_doc": 8},
    {"question": "What types of data analytics exist?", "answer": "There are descriptive, predictive, and prescriptive analytics for transforming data into business insights.", "relevant_doc": 9},
]


def generate_data(n_samples: int = 200, random_state: int = 42):
    """
    Generate synthetic RAG evaluation data.

    Creates query-document pairs with relevance labels:
    - Positive: query paired with its correct knowledge base document
    - Negative: query paired with a random irrelevant document

    Returns
    -------
    queries : list[str]
    documents : list[str]
    labels : np.ndarray
        1 = relevant, 0 = irrelevant
    qa_data : list[dict]
        Original Q&A pairs for generation evaluation.
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
# Document chunking
# ---------------------------------------------------------------------------


def chunk_documents(documents: list[str], chunk_size: int = 50,
                    chunk_overlap: int = 10) -> list[dict]:
    """
    Split documents into overlapping chunks.

    Parameters
    ----------
    documents : list[str]
    chunk_size : int
        Number of words per chunk.
    chunk_overlap : int
        Overlap in words between consecutive chunks.

    Returns
    -------
    chunks : list[dict]
        Each dict has 'text', 'doc_id', 'chunk_id'.
    """
    chunks = []
    for doc_id, doc in enumerate(documents):
        words = doc.split()
        stride = max(chunk_size - chunk_overlap, 1)
        for i in range(0, len(words), stride):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 10:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": len(chunks),
            })
    return chunks


# ---------------------------------------------------------------------------
# Feature extraction for relevance scoring
# ---------------------------------------------------------------------------


def extract_relevance_features(query: str, chunk: str,
                               cosine_sim: float) -> np.ndarray:
    """
    Extract features for query-chunk relevance scoring.

    Features:
    1. Cosine similarity (TF-IDF)
    2. Term overlap ratio
    3. Chunk length (normalized)
    4. Query term coverage
    5. Exact match ratio
    """
    q_tokens = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split())
    c_tokens = set(re.sub(r"[^a-z0-9\s]", "", chunk.lower()).split())

    # Term overlap
    overlap = len(q_tokens & c_tokens)
    overlap_ratio = overlap / max(len(q_tokens), 1)

    # Chunk length (normalized by max expected)
    chunk_len = len(chunk.split()) / 100.0

    # Query term coverage
    coverage = overlap / max(len(q_tokens), 1)

    # Exact match ratio
    exact_ratio = 1.0 if query.lower() in chunk.lower() else 0.0

    return np.array([cosine_sim, overlap_ratio, chunk_len, coverage, exact_ratio])


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    RAG Pipeline using TF-IDF retrieval and sklearn relevance scoring.

    Parameters
    ----------
    max_features : int
        TF-IDF vocabulary size.
    top_k : int
        Number of chunks to retrieve.
    chunk_size : int
        Words per chunk.
    chunk_overlap : int
        Overlap words between chunks.
    C : float
        Regularization for relevance classifier.
    """

    def __init__(self, max_features: int = 5000, top_k: int = 3,
                 chunk_size: int = 50, chunk_overlap: int = 10,
                 C: float = 1.0):
        self.max_features = max_features
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.C = C

        self.vectorizer: TfidfVectorizer | None = None
        self.chunk_vectors: np.ndarray | None = None
        self.chunks: list[dict] = []
        self.relevance_classifier: LogisticRegression | None = None

    def index(self, documents: list[str]):
        """Index documents: chunk and vectorize."""
        self.chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        chunk_texts = [c["text"] for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=(1, 2), sublinear_tf=True
        )
        self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
        logger.info(f"  Indexed {len(self.chunks)} chunks from {len(documents)} documents")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve top-k relevant chunks for a query.

        Returns list of dicts with 'text', 'doc_id', 'chunk_id', 'score'.
        """
        if top_k is None:
            top_k = self.top_k

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.chunk_vectors).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)

        return results

    def train_relevance_scorer(self, queries: list[str], documents: list[str],
                               labels: np.ndarray):
        """
        Train the relevance scoring classifier.

        Parameters
        ----------
        queries : list[str]
            Query texts.
        documents : list[str]
            Document/chunk texts (paired with queries).
        labels : np.ndarray
            Relevance labels (1=relevant, 0=irrelevant).
        """
        all_texts = list(set(queries + documents))
        feature_vectorizer = TfidfVectorizer(max_features=self.max_features, sublinear_tf=True)
        feature_vectorizer.fit(all_texts)

        features = []
        for q, d in zip(queries, documents):
            q_vec = feature_vectorizer.transform([q])
            d_vec = feature_vectorizer.transform([d])
            cos_sim = cosine_similarity(q_vec, d_vec)[0, 0]
            feat = extract_relevance_features(q, d, cos_sim)
            features.append(feat)

        X = np.array(features)
        self.relevance_classifier = LogisticRegression(C=self.C, max_iter=1000)
        self.relevance_classifier.fit(X, labels)

    def score_relevance(self, query: str, chunks: list[dict]) -> list[dict]:
        """Re-rank chunks using the relevance classifier."""
        if self.relevance_classifier is None:
            return chunks

        scored_chunks = []
        for chunk in chunks:
            feat = extract_relevance_features(query, chunk["text"], chunk["score"])
            relevance_prob = self.relevance_classifier.predict_proba(
                feat.reshape(1, -1)
            )[0, 1]
            chunk_copy = chunk.copy()
            chunk_copy["relevance_score"] = float(relevance_prob)
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_chunks

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """
        Generate an answer using template-based approach.

        Parameters
        ----------
        query : str
            User question.
        context_chunks : list[dict]
            Retrieved and scored chunks.

        Returns
        -------
        answer : str
        """
        if not context_chunks:
            return "I could not find relevant information to answer your question."

        # Use the most relevant chunk
        best_chunk = context_chunks[0]
        context = best_chunk["text"]

        # Simple template-based generation
        answer = f"Based on the available information: {context}"

        # Try to extract a more specific sentence
        sentences = re.split(r"[.!?]+", context)
        query_words = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split())
        query_words -= {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an"}

        best_sentence = ""
        best_overlap = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sent

        if best_sentence:
            answer = f"{best_sentence}."

        return answer

    def query(self, question: str) -> dict:
        """
        Full RAG pipeline: retrieve, score, generate.

        Parameters
        ----------
        question : str

        Returns
        -------
        result : dict with 'answer', 'retrieved_chunks', 'scores'
        """
        # Retrieve
        retrieved = self.retrieve(question, top_k=self.top_k)

        # Re-rank with relevance scorer
        scored = self.score_relevance(question, retrieved)

        # Generate
        answer = self.generate_answer(question, scored)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": scored,
            "num_chunks_retrieved": len(scored),
        }


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------


def train(X_train_queries: list[str], y_train: np.ndarray,
          X_train_docs: list[str] = None, **hyperparams) -> RAGPipeline:
    """
    Train the RAG pipeline.

    Parameters
    ----------
    X_train_queries : list[str]
        Training queries.
    y_train : np.ndarray
        Relevance labels.
    X_train_docs : list[str]
        Training documents (paired with queries).
    **hyperparams
        max_features, top_k, chunk_size, chunk_overlap, C

    Returns
    -------
    pipeline : RAGPipeline
    """
    max_features = hyperparams.get("max_features", 5000)
    top_k = hyperparams.get("top_k", 3)
    chunk_size = hyperparams.get("chunk_size", 50)
    chunk_overlap = hyperparams.get("chunk_overlap", 10)
    C = hyperparams.get("C", 1.0)

    pipeline = RAGPipeline(
        max_features=max_features,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        C=C,
    )

    # Index knowledge base
    pipeline.index(KNOWLEDGE_BASE)

    # Train relevance scorer
    if X_train_docs is not None:
        pipeline.train_relevance_scorer(X_train_queries, X_train_docs, y_train)

    return pipeline


def validate(model: RAGPipeline, X_val_queries: list[str],
             y_val: np.ndarray, X_val_docs: list[str] = None) -> dict[str, float]:
    """
    Validate retrieval quality.

    For relevance scoring evaluation: predict relevance of query-document pairs.
    For retrieval evaluation: check if correct document is in top-k.
    """
    if X_val_docs is not None and model.relevance_classifier is not None:
        # Evaluate relevance scoring
        feature_vectorizer = TfidfVectorizer(max_features=model.max_features, sublinear_tf=True)
        all_texts = list(set(X_val_queries + X_val_docs))
        feature_vectorizer.fit(all_texts)

        features = []
        for q, d in zip(X_val_queries, X_val_docs):
            q_vec = feature_vectorizer.transform([q])
            d_vec = feature_vectorizer.transform([d])
            cos_sim = cosine_similarity(q_vec, d_vec)[0, 0]
            feat = extract_relevance_features(q, d, cos_sim)
            features.append(feat)

        X = np.array(features)
        y_pred = model.relevance_classifier.predict(X)

        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        }

    return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test(model: RAGPipeline, X_test_queries: list[str],
         y_test: np.ndarray, X_test_docs: list[str] = None) -> dict[str, Any]:
    """Final test evaluation with classification report."""
    metrics = validate(model, X_test_queries, y_test, X_test_docs)

    if X_test_docs is not None and model.relevance_classifier is not None:
        feature_vectorizer = TfidfVectorizer(max_features=model.max_features, sublinear_tf=True)
        all_texts = list(set(X_test_queries + X_test_docs))
        feature_vectorizer.fit(all_texts)

        features = []
        for q, d in zip(X_test_queries, X_test_docs):
            q_vec = feature_vectorizer.transform([q])
            d_vec = feature_vectorizer.transform([d])
            cos_sim = cosine_similarity(q_vec, d_vec)[0, 0]
            feat = extract_relevance_features(q, d, cos_sim)
            features.append(feat)

        X = np.array(features)
        y_pred = model.relevance_classifier.predict(X)

        metrics["report"] = classification_report(
            y_test, y_pred, target_names=["irrelevant", "relevant"], zero_division=0
        )
    else:
        metrics["report"] = "No relevance classifier trained."

    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------


def optuna_objective(trial, X_train_q, y_train, X_train_d,
                     X_val_q, y_val, X_val_d) -> float:
    """Optuna objective for RAG pipeline."""
    params = {
        "max_features": trial.suggest_int("max_features", 1000, 10000, step=1000),
        "top_k": trial.suggest_int("top_k", 1, 5),
        "chunk_size": trial.suggest_int("chunk_size", 20, 80, step=10),
        "chunk_overlap": trial.suggest_int("chunk_overlap", 5, 20, step=5),
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
    }
    model = train(X_train_q, y_train, X_train_d, **params)
    metrics = validate(model, X_val_q, y_val, X_val_d)
    return metrics["f1"]


def ray_tune_search(X_train_q, y_train, X_train_d,
                    X_val_q, y_val, X_val_d, num_samples=20) -> dict:
    """Ray Tune search for RAG pipeline."""

    def ray_objective(config):
        model = train(X_train_q, y_train, X_train_d, **config)
        metrics = validate(model, X_val_q, y_val, X_val_d)
        tune.report({"f1": metrics["f1"], "accuracy": metrics["accuracy"]})

    search_space = {
        "max_features": tune.choice([2000, 5000, 8000]),
        "top_k": tune.choice([1, 2, 3, 5]),
        "chunk_size": tune.choice([30, 50, 70]),
        "chunk_overlap": tune.choice([5, 10, 15]),
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
    """Run the full RAG pipeline with TF-IDF retrieval and sklearn scoring."""
    logger.info("=" * 70)
    logger.info("RAG Pipeline: TF-IDF Retrieval + sklearn Scoring")
    logger.info("=" * 70)

    # 1. Generate data
    logger.info("Generating synthetic RAG data...")
    queries, documents, labels, qa_pairs = generate_data(n_samples=200, random_state=42)
    logger.info(f"  Total query-document pairs: {len(queries)}")
    logger.info(f"  Knowledge base documents: {len(KNOWLEDGE_BASE)}")

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

    # 3. Train baseline
    logger.info("\n--- Baseline Training ---")
    start = time.time()
    pipeline = train(X_train_q, y_train, X_train_d, top_k=3, chunk_size=50)
    logger.info(f"  Training time: {time.time() - start:.3f}s")

    val_metrics = validate(pipeline, X_val_q, y_val, X_val_d)
    logger.info(f"  Validation: {val_metrics}")

    # 4. Demonstrate RAG query
    logger.info("\n--- RAG Query Examples ---")
    for qa in qa_pairs[:3]:
        result = pipeline.query(qa["question"])
        logger.info(f"  Q: {result['question']}")
        logger.info(f"  A: {result['answer']}")
        logger.info(f"  Chunks retrieved: {result['num_chunks_retrieved']}")
        if result["retrieved_chunks"]:
            logger.info(f"  Top chunk score: {result['retrieved_chunks'][0].get('relevance_score', 'N/A')}")
        logger.info("")

    # 5. Optuna
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

    # 6. Ray Tune
    logger.info("\n--- Ray Tune Search ---")
    best_ray_config = ray_tune_search(
        X_train_q, y_train, X_train_d, X_val_q, y_val, X_val_d, num_samples=10
    )
    logger.info(f"  Best Ray config: {best_ray_config}")

    # 7. Final model
    logger.info("\n--- Final Model ---")
    final_pipeline = train(X_train_q, y_train, X_train_d, **study.best_params)

    test_results = test(final_pipeline, X_test_q, y_test, X_test_d)
    logger.info(f"  Test Accuracy:  {test_results['accuracy']:.4f}")
    logger.info(f"  Test Precision: {test_results['precision']:.4f}")
    logger.info(f"  Test Recall:    {test_results['recall']:.4f}")
    logger.info(f"  Test F1:        {test_results['f1']:.4f}")
    logger.info(f"\n{test_results['report']}")

    # 8. Final RAG examples
    logger.info("\n--- Final RAG Examples ---")
    test_questions = [
        "What is machine learning?",
        "How does solar energy work?",
        "What is agile methodology?",
        "What are cybersecurity threats?",
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
