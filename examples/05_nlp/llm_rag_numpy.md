# LLM + RAG (Retrieval-Augmented Generation) - Complete Guide with Stock Market Applications

## Overview

Retrieval-Augmented Generation (RAG) combines the power of Large Language Models (LLMs) with a retrieval system that fetches relevant documents from an external knowledge base before generating a response. Instead of relying solely on the LLM's parametric memory (which can be outdated or hallucinate), RAG grounds the model's answers in actual source documents. In the stock market domain, this is transformative for querying SEC filings, annual reports, and financial documents where accuracy and traceability are paramount.

The RAG pipeline consists of two main phases: retrieval and generation. During retrieval, the user's query is converted into a vector embedding and compared against a pre-indexed database of document chunks using similarity search. The most relevant chunks are then passed as context to the LLM, which generates a coherent answer grounded in the retrieved information. This architecture allows traders, analysts, and compliance officers to ask natural language questions about thousands of financial documents and receive accurate, cited answers.

For stock market applications, RAG solves a critical problem: financial professionals need to quickly find specific information across vast collections of 10-K filings, 10-Q reports, proxy statements, and earnings transcripts. Traditional keyword search often fails because the same concept can be expressed many different ways in financial documents. RAG understands semantic meaning, so a query like "What are the company's biggest risks?" will find relevant passages even if they use terms like "risk factors," "material uncertainties," or "potential headwinds."

## How It Works - The Math Behind It

### Document Embedding

Each document chunk is converted to a dense vector using an embedding model:

```
e_doc = EmbeddingModel(document_chunk)  # e.g., shape: (768,) for BERT-based embeddings
```

### Query Embedding

The user's question is embedded using the same model:

```
e_query = EmbeddingModel(query_text)
```

### Similarity Search

Retrieve the top-k most similar documents using cosine similarity:

```
cosine_similarity(e_query, e_doc) = (e_query . e_doc) / (||e_query|| * ||e_doc||)
```

Or using dot product for normalized vectors:

```
score = e_query^T * e_doc
```

### Context Construction

Retrieved documents are concatenated into a context string:

```
context = concat(doc_1, doc_2, ..., doc_k)
```

### Prompt Template

```
prompt = f"""
Based on the following SEC filing excerpts, answer the question.
Only use information from the provided documents.

Context:
{context}

Question: {query}

Answer:
"""
```

### LLM Generation (Transformer Decoder)

The LLM generates tokens autoregressively:

```
P(token_t | token_1, ..., token_{t-1}, context) = softmax(W * h_t)
```

Where h_t is the hidden state at position t from the transformer decoder.

### Attention Over Context

The key innovation is that the LLM attends to the retrieved document tokens:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Where K and V include the embedded context documents, allowing the model to "look at" the retrieved information while generating.

### Vector Index (Approximate Nearest Neighbor)

For efficient retrieval at scale, vectors are indexed using structures like:

```
HNSW (Hierarchical Navigable Small World) graph:
- Build time: O(n * log(n))
- Query time: O(log(n))
- Space: O(n * d)
```

### Chunking Strategy

Documents are split into overlapping chunks:

```
chunk_size = 512 tokens
overlap = 50 tokens
n_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
```

## Stock Market Use Case: Querying SEC Filings and Financial Documents

### The Problem

A portfolio manager covers 50 stocks and needs to quickly understand risk factors, revenue breakdowns, management discussions, and competitive dynamics across all covered companies. Each company files multiple documents per year (10-K, 10-Q, 8-K, proxy statements), totaling thousands of pages. Manually reading through this volume is impossible. The manager needs a system that can answer specific questions like "Which of my portfolio companies mentioned supply chain risks?" or "What was Company X's revenue growth guidance for next year?" with cited sources from the actual filings.

### Stock Market Features (Input Data)

| Feature | Description | Example |
|---------|-------------|---------|
| filing_type | Type of SEC document | 10-K, 10-Q, 8-K, DEF 14A |
| ticker | Company stock symbol | AAPL, MSFT, GOOGL |
| filing_date | Date of filing | 2024-02-02 |
| section | Document section | Risk Factors, MD&A, Financial Statements |
| text_chunk | Chunk of document text | "The Company faces risks from..." |
| chunk_id | Unique chunk identifier | AAPL_10K_2024_RF_003 |
| embedding | Vector representation | [0.023, -0.156, 0.089, ...] |
| metadata | Additional context | {page: 45, paragraph: 3} |

### Example Data Structure

```python
import numpy as np
from collections import defaultdict

# Simulated SEC filing document chunks
sec_documents = [
    {
        "chunk_id": "AAPL_10K_2024_RF_001",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "text": "The Company is subject to risks associated with global supply chain "
                "disruptions. Component shortages, logistics constraints, and geopolitical "
                "tensions could materially impact the Company's ability to manufacture and "
                "deliver products in sufficient quantities to meet demand.",
    },
    {
        "chunk_id": "AAPL_10K_2024_MDA_001",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section": "MD&A",
        "text": "Total net revenue increased 8 percent year over year to 383.3 billion "
                "driven primarily by growth in Services revenue which increased 16 percent "
                "to 85.2 billion. Products revenue increased 5 percent to 298.1 billion "
                "with iPhone revenue growing 6 percent.",
    },
    {
        "chunk_id": "MSFT_10K_2024_RF_001",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "text": "Cybersecurity threats and data breaches represent significant risks. The "
                "Company processes and stores vast amounts of customer data and any security "
                "incident could result in regulatory penalties, reputational damage, and loss "
                "of customer trust affecting our cloud business.",
    },
    {
        "chunk_id": "MSFT_10K_2024_MDA_001",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "section": "MD&A",
        "text": "Intelligent Cloud revenue grew 22 percent to 96.8 billion with Azure and "
                "other cloud services revenue growing 29 percent. Enterprise demand for AI "
                "services contributed approximately 6 percentage points to Azure growth as "
                "customers accelerated AI workload adoption.",
    },
    {
        "chunk_id": "NVDA_10K_2024_RF_001",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "text": "The Company faces risks from concentration of revenue in a limited number "
                "of customers. A significant portion of data center revenue comes from large "
                "cloud service providers and any reduction in orders from these customers "
                "could materially affect financial results.",
    },
    {
        "chunk_id": "NVDA_10K_2024_MDA_001",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "section": "MD&A",
        "text": "Data Center revenue increased 217 percent year over year to 47.5 billion "
                "driven by unprecedented demand for AI training and inference accelerators. "
                "The Company raised full year guidance to 60 billion reflecting strong "
                "visibility into the AI infrastructure buildout cycle.",
    },
    {
        "chunk_id": "TSLA_10K_2024_RF_001",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "text": "The Company faces intense competition in the electric vehicle market from "
                "both traditional automakers and new entrants. Price reductions to maintain "
                "market share have compressed margins and future profitability depends on "
                "achieving manufacturing cost efficiencies at scale.",
    },
    {
        "chunk_id": "JPM_10K_2024_MDA_001",
        "ticker": "JPM",
        "filing_type": "10-K",
        "section": "MD&A",
        "text": "Net interest income increased 23 percent to 89.3 billion driven by higher "
                "interest rates and strong loan growth. Investment banking fees were 8.1 "
                "billion an increase of 37 percent reflecting improved capital markets "
                "activity and market share gains.",
    },
]

# Simple embedding function (in production, use sentence-transformers or OpenAI embeddings)
class SimpleEmbedder:
    def __init__(self, dim=128):
        self.dim = dim
        np.random.seed(42)
        self.word_vectors = {}

    def _get_word_vector(self, word):
        if word not in self.word_vectors:
            np.random.seed(hash(word) % 2**31)
            self.word_vectors[word] = np.random.randn(self.dim)
        return self.word_vectors[word]

    def embed(self, text):
        words = text.lower().split()
        if not words:
            return np.zeros(self.dim)
        vectors = [self._get_word_vector(w) for w in words]
        embedding = np.mean(vectors, axis=0)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

embedder = SimpleEmbedder(dim=128)

# Pre-compute document embeddings
for doc in sec_documents:
    doc["embedding"] = embedder.embed(doc["text"])

print(f"Indexed {len(sec_documents)} document chunks")
print(f"Embedding dimension: {sec_documents[0]['embedding'].shape}")
```

### The Model in Action

```python
class FinancialRAGSystem:
    def __init__(self, embedder, documents, top_k=3):
        self.embedder = embedder
        self.documents = documents
        self.top_k = top_k
        self._build_index()

    def _build_index(self):
        self.doc_embeddings = np.array([d["embedding"] for d in self.documents])
        print(f"Built index with {len(self.documents)} chunks")
        print(f"Index shape: {self.doc_embeddings.shape}")

    def retrieve(self, query, top_k=None, filters=None):
        k = top_k or self.top_k
        query_embedding = self.embedder.embed(query)
        # Cosine similarity (embeddings are already normalized)
        similarities = self.doc_embeddings @ query_embedding
        # Apply filters
        if filters:
            for i, doc in enumerate(self.documents):
                for key, value in filters.items():
                    if doc.get(key) != value:
                        similarities[i] = -1
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] > -1:
                results.append({
                    "document": self.documents[idx],
                    "score": float(similarities[idx]),
                    "rank": len(results) + 1
                })
        return results

    def generate_answer(self, query, retrieved_docs):
        """
        Simulated LLM generation. In production, this calls
        OpenAI GPT-4, Anthropic Claude, or similar LLM API.
        """
        context_parts = []
        citations = []
        for i, r in enumerate(retrieved_docs):
            doc = r["document"]
            context_parts.append(f"[Source {i+1}: {doc['ticker']} {doc['filing_type']} "
                                f"- {doc['section']}]\n{doc['text']}")
            citations.append(f"[{i+1}] {doc['chunk_id']} ({doc['ticker']} {doc['filing_type']} "
                           f"- {doc['section']}, Score: {r['score']:.3f})")

        context = "\n\n".join(context_parts)

        # In production, this would be an LLM API call:
        # response = llm.generate(prompt_template.format(context=context, query=query))
        # For demonstration, we return the structured context
        return {
            "query": query,
            "answer": f"Based on {len(retrieved_docs)} relevant SEC filing excerpts, "
                     f"here is the information found for: '{query}'",
            "context_used": context,
            "citations": citations,
            "n_sources": len(retrieved_docs)
        }

    def query(self, question, filters=None):
        retrieved = self.retrieve(question, filters=filters)
        answer = self.generate_answer(question, retrieved)
        return answer

# Build RAG system
rag = FinancialRAGSystem(embedder, sec_documents, top_k=3)

# Example queries
queries = [
    "What are the main supply chain risks in the portfolio?",
    "Which companies reported the strongest revenue growth?",
    "What AI-related growth are companies seeing?",
    "What competitive risks do companies face?",
]

for query in queries:
    result = rag.query(query)
    print(f"\n{'='*60}")
    print(f"QUERY: {result['query']}")
    print(f"{'='*60}")
    print(f"\nSources Used ({result['n_sources']}):")
    for citation in result['citations']:
        print(f"  {citation}")
    print(f"\n{result['answer']}")
```

## Advantages

1. **Grounds Answers in Actual Documents**: Unlike pure LLMs that might hallucinate financial figures, RAG retrieves and cites actual SEC filings. When a portfolio manager asks about a company's risk factors, the system provides verbatim passages from the 10-K filing with traceable citations, ensuring accuracy for investment decisions.

2. **Always Up-to-Date Without Retraining**: When new filings are submitted to the SEC, they can be immediately indexed and made queryable without retraining the LLM. This is critical in finance where a quarterly filing can change the entire investment thesis for a stock.

3. **Scales to Massive Document Collections**: A single RAG system can index and search across thousands of SEC filings, earnings transcripts, and analyst reports. Vector similarity search operates in logarithmic time, enabling sub-second retrieval across millions of document chunks.

4. **Natural Language Interface**: Analysts can ask questions in plain English rather than constructing complex search queries or XBRL lookups. Questions like "Did management express concern about margins?" work naturally, lowering the barrier to information access.

5. **Supports Multi-Document Analysis**: RAG can retrieve relevant chunks from multiple companies' filings simultaneously, enabling cross-company comparisons like "Which of my portfolio companies mentioned tariff risks?" This multi-document reasoning is extremely difficult with traditional document search.

6. **Reduces Analyst Research Time**: What previously took hours of manual document review can be accomplished in seconds. Analysts can quickly verify facts, compare disclosures, and identify material changes between filing periods.

7. **Audit Trail and Compliance**: Every answer includes citations to specific document chunks, filing types, and sections. This traceability is essential for compliance and audit requirements in regulated financial institutions.

## Disadvantages

1. **Retrieval Quality Bottleneck**: The system is only as good as its retrieval. If the relevant document chunk is not retrieved (due to poor embedding quality or query-document mismatch), the LLM will either hallucinate or provide an incomplete answer. Financial documents with highly technical jargon can be particularly challenging for general-purpose embedding models.

2. **Chunking Challenges with Financial Tables**: SEC filings contain complex tables (financial statements, footnotes) that are difficult to chunk properly. Breaking a table across chunks can lose critical context, and embedding models struggle with tabular data, potentially missing important financial figures.

3. **High Infrastructure Cost**: Running a production RAG system requires vector databases (Pinecone, Weaviate), embedding model inference, and LLM API calls. For a large financial institution with millions of documents, infrastructure costs can be substantial, often exceeding $10,000/month.

4. **Context Window Limitations**: Even with retrieval, the LLM's context window limits how much information can be processed at once. Complex financial questions that require synthesizing information from 20+ document sections may exceed the context window, forcing lossy summarization.

5. **Latency for Real-Time Applications**: The full RAG pipeline (embed query, search index, retrieve documents, generate answer) typically takes 2-10 seconds. This latency is acceptable for research but too slow for real-time trading decisions during market hours.

6. **Difficulty with Quantitative Reasoning**: While RAG excels at finding qualitative information (risk factors, strategy discussions), it struggles with quantitative questions that require computation, like "Calculate the 3-year CAGR of services revenue." The retrieval may find the right numbers, but the LLM may compute incorrectly.

7. **Document Parsing and Cleaning**: SEC filings come in HTML, XBRL, and PDF formats with inconsistent formatting. Extracting clean text while preserving structure (tables, lists, headers) requires significant preprocessing effort. Poor parsing leads to poor retrieval quality.

## When to Use in Stock Market

- **Fundamental research**: When analysts need to quickly find specific information across multiple company filings
- **Due diligence**: For M&A analysis requiring comprehensive review of target company disclosures
- **Risk monitoring**: To identify emerging risk factors across a portfolio of covered stocks
- **Compliance review**: When compliance teams need to verify disclosures and representations in filings
- **Cross-company analysis**: For comparing how different companies discuss similar topics (competition, regulation, ESG)
- **Earnings season preparation**: To quickly review prior filings before new earnings releases
- **Thematic research**: To identify which companies in a universe are exposed to specific themes (AI, tariffs, ESG)

## When NOT to Use in Stock Market

- **Real-time trading signals**: When sub-second latency is required for trade execution
- **Precise quantitative analysis**: When exact financial calculations are needed (use structured data instead)
- **Small document sets**: When you only need to search a few documents; manual review may be faster
- **Highly structured queries**: When XBRL or SQL queries against structured financial databases would be more appropriate
- **Confidential information**: When documents contain material non-public information (MNPI) that must be strictly access-controlled
- **Legal document interpretation**: When the stakes require actual legal expertise, not AI-generated analysis

## Hyperparameters Guide

| Hyperparameter | Typical Range | Stock Market Recommendation | Effect |
|---------------|---------------|---------------------------|--------|
| chunk_size | 256 - 1024 tokens | 512 tokens | Larger chunks preserve context; smaller chunks improve precision |
| chunk_overlap | 0 - 200 tokens | 50 - 100 tokens | Overlap prevents splitting concepts across chunks |
| top_k | 3 - 10 | 5 | More chunks provide broader context but may dilute relevance |
| embedding_model | various | finance-specific | Domain-specific embeddings improve retrieval for financial text |
| similarity_threshold | 0.5 - 0.9 | 0.7 | Filter out low-relevance results to reduce noise |
| temperature (LLM) | 0.0 - 1.0 | 0.0 - 0.2 | Low temperature for factual financial responses |
| max_tokens (LLM) | 256 - 4096 | 1024 | Adequate for detailed financial analysis responses |
| reranking | True/False | True | Cross-encoder reranking significantly improves retrieval quality |

## Stock Market Performance Tips

1. **Use financial-specific embedding models**: Models like FinBERT embeddings or instructor-xl with financial prompts significantly outperform general-purpose embeddings for SEC filing retrieval. The vocabulary and semantic relationships in financial text are distinct from general text.

2. **Implement hybrid search**: Combine dense vector search with sparse keyword search (BM25). Financial documents contain specific terms (ticker symbols, GAAP metrics, regulatory references) that benefit from exact keyword matching alongside semantic similarity.

3. **Preserve document metadata**: Index filing type, section headers, ticker, filing date, and page numbers as metadata filters. This allows queries like "Show me risk factors from AAPL's latest 10-K" that filter before vector search, dramatically improving precision.

4. **Handle tables separately**: Extract tables from SEC filings and store them as structured data. Use specialized table-understanding approaches or convert tables to natural language descriptions before embedding.

5. **Implement multi-stage retrieval**: First retrieve broadly (top-20), then rerank with a cross-encoder model, then pass top-5 to the LLM. This two-stage approach significantly improves the quality of retrieved context.

6. **Version control your index**: When companies file amendments (10-K/A) or restatements, update the index accordingly. Maintain historical versions for audit trails but ensure current queries return the latest filings.

## Comparison with Other Algorithms

| Feature | LLM + RAG | Pure LLM | TF-IDF Search | BERT Classifier | SQL/XBRL Query |
|---------|-----------|----------|---------------|-----------------|----------------|
| Answer Quality | High (grounded) | Variable (may hallucinate) | Keyword matches only | Classification only | Structured data only |
| Handles Natural Language | Excellent | Excellent | Poor | N/A | No |
| Source Citations | Yes (built-in) | No | Yes (document links) | N/A | Yes (table/row) |
| Scalability | Millions of docs | Limited by context | Millions of docs | N/A | Structured only |
| Latency | 2-10 seconds | 1-5 seconds | <100ms | <50ms | <100ms |
| Cost per Query | $0.01 - $0.10 | $0.01 - $0.05 | <$0.001 | <$0.001 | <$0.001 |
| Quantitative Accuracy | Moderate | Poor | N/A | N/A | Excellent |
| Setup Complexity | High | Low | Medium | Medium | High |
| Handles Tables | Poor | Poor | Poor | N/A | Excellent |

## Real-World Stock Market Example

```python
import numpy as np
from datetime import datetime

class SECFilingRAGSystem:
    """
    Complete RAG system for querying SEC filings and financial documents.
    """

    def __init__(self, embedding_dim=128, chunk_size=200, top_k=5):
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.documents = []
        self.embeddings = None
        self.word_vectors = {}
        np.random.seed(42)

    def _get_word_vector(self, word):
        if word not in self.word_vectors:
            np.random.seed(hash(word) % 2**31)
            self.word_vectors[word] = np.random.randn(self.embedding_dim)
        return self.word_vectors[word]

    def _embed_text(self, text):
        words = text.lower().split()
        if not words:
            return np.zeros(self.embedding_dim)
        vectors = np.array([self._get_word_vector(w) for w in words])
        # Weighted mean: give more weight to financial terms
        financial_terms = {'revenue', 'profit', 'growth', 'risk', 'loss', 'margin',
                          'guidance', 'earnings', 'dividend', 'debt', 'competition',
                          'regulatory', 'demand', 'supply', 'market', 'share'}
        weights = np.array([2.0 if w in financial_terms else 1.0 for w in words])
        weighted = vectors * weights[:, np.newaxis]
        embedding = np.sum(weighted, axis=0) / np.sum(weights)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def index_documents(self, documents):
        self.documents = documents
        embeddings_list = []
        for doc in documents:
            emb = self._embed_text(doc["text"])
            embeddings_list.append(emb)
        self.embeddings = np.array(embeddings_list)
        print(f"Indexed {len(documents)} document chunks")
        print(f"Embedding matrix shape: {self.embeddings.shape}")

    def retrieve(self, query, top_k=None, ticker_filter=None, section_filter=None):
        k = top_k or self.top_k
        query_emb = self._embed_text(query)
        scores = self.embeddings @ query_emb

        # Apply metadata filters
        mask = np.ones(len(self.documents), dtype=bool)
        if ticker_filter:
            mask &= np.array([d["ticker"] == ticker_filter for d in self.documents])
        if section_filter:
            mask &= np.array([d["section"] == section_filter for d in self.documents])

        filtered_scores = np.where(mask, scores, -np.inf)
        top_indices = np.argsort(filtered_scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            if filtered_scores[idx] > -np.inf:
                results.append({
                    "chunk_id": self.documents[idx]["chunk_id"],
                    "ticker": self.documents[idx]["ticker"],
                    "section": self.documents[idx]["section"],
                    "filing_type": self.documents[idx]["filing_type"],
                    "text": self.documents[idx]["text"],
                    "relevance_score": float(filtered_scores[idx])
                })
        return results

    def generate_response(self, query, retrieved_docs):
        """
        Simulates LLM response generation.
        In production: calls GPT-4, Claude, or similar API.
        """
        if not retrieved_docs:
            return {"answer": "No relevant documents found.", "sources": []}

        # Build context
        context_lines = []
        sources = []
        for i, doc in enumerate(retrieved_docs):
            context_lines.append(
                f"[{i+1}] {doc['ticker']} {doc['filing_type']} - {doc['section']}:\n"
                f"   {doc['text'][:200]}..."
            )
            sources.append({
                "citation": f"[{i+1}]",
                "chunk_id": doc["chunk_id"],
                "ticker": doc["ticker"],
                "filing": doc["filing_type"],
                "section": doc["section"],
                "relevance": f"{doc['relevance_score']:.3f}"
            })

        return {
            "query": query,
            "answer": f"Found {len(retrieved_docs)} relevant passages. "
                     f"Top sources: {', '.join(d['ticker'] for d in retrieved_docs[:3])}",
            "context": "\n\n".join(context_lines),
            "sources": sources
        }

    def ask(self, question, ticker=None, section=None):
        retrieved = self.retrieve(question, ticker_filter=ticker, section_filter=section)
        response = self.generate_response(question, retrieved)
        return response


# Build and populate the RAG system
rag_system = SECFilingRAGSystem(embedding_dim=128, top_k=3)

# Financial document corpus
filings = [
    {"chunk_id": "AAPL_10K_RF_01", "ticker": "AAPL", "filing_type": "10-K",
     "section": "Risk Factors",
     "text": "Global supply chain disruptions including component shortages semiconductor "
             "constraints and logistics bottlenecks could adversely impact production capacity "
             "and product availability across all hardware product lines."},
    {"chunk_id": "AAPL_10K_MDA_01", "ticker": "AAPL", "filing_type": "10-K",
     "section": "MD&A",
     "text": "Total net revenue for fiscal year 2024 was 383 billion an increase of 8 percent "
             "from the prior year. Services revenue grew 16 percent to 85 billion while "
             "Products revenue increased 5 percent to 298 billion."},
    {"chunk_id": "MSFT_10K_RF_01", "ticker": "MSFT", "filing_type": "10-K",
     "section": "Risk Factors",
     "text": "Cybersecurity threats pose material risks to our operations. Data breaches "
             "could result in regulatory penalties loss of customer trust and material "
             "financial impact particularly to our Azure cloud platform."},
    {"chunk_id": "MSFT_10K_MDA_01", "ticker": "MSFT", "filing_type": "10-K",
     "section": "MD&A",
     "text": "Azure and cloud services revenue grew 29 percent driven by enterprise AI "
             "adoption contributing 6 percentage points of growth. Intelligent Cloud segment "
             "revenue reached 96.8 billion up 22 percent year over year."},
    {"chunk_id": "NVDA_10K_RF_01", "ticker": "NVDA", "filing_type": "10-K",
     "section": "Risk Factors",
     "text": "Revenue concentration in a small number of large cloud customers creates "
             "dependency risk. Loss of any major customer could materially affect data "
             "center revenue which grew 217 percent this fiscal year."},
    {"chunk_id": "NVDA_10K_MDA_01", "ticker": "NVDA", "filing_type": "10-K",
     "section": "MD&A",
     "text": "Data center revenue surged 217 percent to 47.5 billion driven by unprecedented "
             "demand for AI training and inference accelerators. Full year guidance raised to "
             "60 billion reflecting strong AI infrastructure buildout visibility."},
    {"chunk_id": "TSLA_10K_RF_01", "ticker": "TSLA", "filing_type": "10-K",
     "section": "Risk Factors",
     "text": "Intense competition in electric vehicle market from established automakers and "
             "new entrants. Price reductions to maintain market share have compressed gross "
             "margins and profitability depends on manufacturing cost efficiencies."},
    {"chunk_id": "JPM_10K_MDA_01", "ticker": "JPM", "filing_type": "10-K",
     "section": "MD&A",
     "text": "Net interest income increased 23 percent to 89.3 billion. Investment banking "
             "fees surged 37 percent to 8.1 billion reflecting improved capital markets "
             "activity. Consumer banking net revenue grew 18 percent."},
]

rag_system.index_documents(filings)

# Run example queries
print("\n" + "=" * 70)
print("SEC FILING RAG SYSTEM - QUERY RESULTS")
print("=" * 70)

queries = [
    {"question": "What are the biggest supply chain and competition risks?",
     "ticker": None, "section": "Risk Factors"},
    {"question": "Which companies showed the strongest revenue growth?",
     "ticker": None, "section": "MD&A"},
    {"question": "How fast is Azure cloud growing?",
     "ticker": "MSFT", "section": None},
    {"question": "What is driving NVIDIA data center demand?",
     "ticker": "NVDA", "section": None},
]

for q in queries:
    result = rag_system.ask(q["question"], ticker=q.get("ticker"), section=q.get("section"))
    filters = []
    if q.get("ticker"):
        filters.append(f"ticker={q['ticker']}")
    if q.get("section"):
        filters.append(f"section={q['section']}")
    filter_str = f" [{', '.join(filters)}]" if filters else ""

    print(f"\n{'─'*70}")
    print(f"Q: {result['query']}{filter_str}")
    print(f"{'─'*70}")
    print(f"\n{result['answer']}\n")
    print("Sources:")
    for src in result["sources"]:
        print(f"  {src['citation']} {src['ticker']} {src['filing']} - "
              f"{src['section']} (relevance: {src['relevance']})")
    print(f"\nRetrieved Context:\n{result['context']}")
```

## Key Takeaways

1. **RAG eliminates hallucination risk** by grounding LLM answers in actual SEC filings. Every answer can be traced back to specific document sections, which is essential for investment decisions where accuracy is non-negotiable.

2. **The retrieval step is the most critical component**. Invest heavily in embedding quality, chunking strategy, and hybrid search. A mediocre LLM with excellent retrieval outperforms a great LLM with poor retrieval every time.

3. **Metadata filtering is essential** for financial RAG. Users need to filter by ticker, filing type, section, and date. Pre-filtering by metadata before vector search dramatically improves both relevance and speed.

4. **Financial tables require special handling**. Standard chunking destroys tabular structure. Consider extracting tables as structured data and creating natural language descriptions for embedding, or use table-specific retrieval methods.

5. **RAG is complementary to structured data systems**, not a replacement. Use RAG for qualitative analysis (risk factors, management discussion) and SQL/XBRL for quantitative queries (exact revenue figures, ratios). The best systems combine both.

6. **Monitor retrieval quality continuously**. Track metrics like mean reciprocal rank (MRR) and precision@k on a validation set of known query-document pairs. Retrieval quality degrades as the document corpus grows without index maintenance.

7. **Cost management matters at scale**. Each RAG query involves embedding computation, vector search, and LLM API calls. Implement caching, batch processing, and query deduplication to manage costs when serving many analysts simultaneously.
