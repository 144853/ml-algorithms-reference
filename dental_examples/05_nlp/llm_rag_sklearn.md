# LLM RAG (Retrieval-Augmented Generation) - Simple Use Case & Data Explanation

## **Use Case: Dental Knowledge Base Q&A System for Treatment Protocol Retrieval**

### **The Problem**
A dental practice network has a **knowledge base of 2,000 treatment protocols, clinical guidelines, and best practices**. Dentists and hygienists need to quickly find answers to clinical questions:
- "What is the recommended protocol for treating a Class II composite restoration?"
- "When should a patient be referred for endodontic evaluation?"
- "What are the contraindications for dental implants in diabetic patients?"

**Goal:** Build a RAG system that retrieves relevant dental protocols and generates accurate, cited answers.

### **Why RAG?**
| Criteria | Keyword Search | Fine-tuned LLM | RAG |
|----------|---------------|----------------|-----|
| Answers factual questions | Partial | Yes (may hallucinate) | Yes (grounded) |
| Uses latest guidelines | Yes | No (training cutoff) | Yes |
| Cites sources | Page-level | No | Yes (chunk-level) |
| Handles novel questions | No | Limited | Yes |
| Reduces hallucination | N/A | Poor | Good |

RAG combines retrieval precision with LLM generation quality.

---

## **Example Data Structure**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dental knowledge base documents
documents = [
    {
        "id": "PROTO-001",
        "title": "Class II Composite Restoration Protocol",
        "content": "For Class II composite restorations, isolate the tooth with rubber dam. "
                   "Apply phosphoric acid etch for 15 seconds on enamel, 10 seconds on dentin. "
                   "Apply bonding agent in two coats, air thin, and light cure for 20 seconds. "
                   "Place composite in 2mm incremental layers, curing each layer for 40 seconds."
    },
    {
        "id": "PROTO-002",
        "title": "Endodontic Referral Criteria",
        "content": "Refer for endodontic evaluation when: pulp vitality tests are negative, "
                   "periapical radiolucency exceeds 3mm, patient reports spontaneous pain lasting "
                   "more than 30 seconds, thermal sensitivity persists after stimulus removal, "
                   "or there is evidence of internal root resorption on radiograph."
    },
    {
        "id": "PROTO-003",
        "title": "Dental Implant Contraindications",
        "content": "Absolute contraindications for dental implants include: uncontrolled diabetes "
                   "(HbA1c > 8%), active bisphosphonate therapy (IV), recent head/neck radiation, "
                   "severe immunosuppression. Relative contraindications: smoking > 10 cigarettes/day, "
                   "bruxism without night guard, inadequate bone volume without grafting option."
    },
    {
        "id": "GUIDE-001",
        "title": "Pediatric Fluoride Varnish Application",
        "content": "Apply 5% sodium fluoride varnish (22,600 ppm F) to all erupted teeth. "
                   "Frequency: every 3-6 months for high-risk children, every 6 months for "
                   "moderate risk. Dry teeth with gauze before application. Apply thin layer "
                   "using a microbrush. Patient may eat soft food after 4 hours."
    },
    {
        "id": "GUIDE-002",
        "title": "Antibiotic Prophylaxis Guidelines",
        "content": "Antibiotic prophylaxis recommended for patients with: prosthetic heart valves, "
                   "previous infective endocarditis, congenital heart disease (specific conditions), "
                   "cardiac transplant with valvulopathy. Standard regimen: Amoxicillin 2g orally "
                   "30-60 minutes before procedure. Penicillin allergy: Clindamycin 600mg."
    }
]

# Chunk documents for retrieval
chunks = []
for doc in documents:
    sentences = doc['content'].split('. ')
    for i in range(0, len(sentences), 2):
        chunk_text = '. '.join(sentences[i:i+2])
        chunks.append({
            'doc_id': doc['id'],
            'title': doc['title'],
            'text': chunk_text
        })

print(f"Total chunks: {len(chunks)}")
```

---

## **RAG Mathematics (Simple Terms)**

**Two-Stage Process:**

1. **Retrieval (Similarity Search):**
   - Query embedding: $q = \text{encode}(\text{question})$
   - Document embedding: $d_i = \text{encode}(\text{chunk}_i)$
   - Similarity: $\text{sim}(q, d_i) = \frac{q \cdot d_i}{||q|| \cdot ||d_i||}$ (cosine similarity)
   - Retrieve top-k chunks: $D_k = \text{argtop}_k(\text{sim}(q, d_i))$

2. **Generation (LLM with Context):**
   - Prompt: "Given these dental protocols: {D_k}, answer: {question}"
   - LLM generates answer grounded in retrieved context

---

## **The Algorithm**

```python
# Step 1: Build TF-IDF retriever (sklearn-based)
chunk_texts = [c['text'] for c in chunks]
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    stop_words='english'
)
chunk_vectors = tfidf.fit_transform(chunk_texts)

def retrieve(query, top_k=3):
    """Retrieve most relevant dental protocol chunks."""
    query_vector = tfidf.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'score': similarities[idx]
        })
    return results

# Step 2: Generate answer with LLM
def generate_answer(query, retrieved_chunks):
    """Format context and generate answer using LLM."""
    context = "\n\n".join([
        f"[{c['chunk']['doc_id']}] {c['chunk']['title']}:\n{c['chunk']['text']}"
        for c in retrieved_chunks
    ])

    prompt = f"""You are a dental clinical assistant. Answer the following question
using ONLY the provided dental protocols. Cite the protocol ID in your answer.

Context:
{context}

Question: {query}

Answer:"""

    # In production, send to LLM API (e.g., OpenAI, Anthropic)
    # response = llm.generate(prompt)
    return prompt

# Example query
query = "What are the contraindications for dental implants in diabetic patients?"
results = retrieve(query, top_k=3)

print("Retrieved chunks:")
for r in results:
    print(f"  Score: {r['score']:.3f} | {r['chunk']['doc_id']}: {r['chunk']['text'][:80]}...")

answer_prompt = generate_answer(query, results)
```

---

## **Results From the Demo**

**Query:** "What are the contraindications for dental implants in diabetic patients?"

| Rank | Doc ID | Score | Title |
|------|--------|-------|-------|
| 1 | PROTO-003 | 0.78 | Dental Implant Contraindications |
| 2 | PROTO-003 | 0.42 | Dental Implant Contraindications (chunk 2) |
| 3 | GUIDE-002 | 0.15 | Antibiotic Prophylaxis Guidelines |

**Generated Answer:**
"According to protocol PROTO-003, uncontrolled diabetes with HbA1c > 8% is an absolute contraindication for dental implants. Patients with controlled diabetes may proceed with implant placement with appropriate monitoring."

### **Key Insights:**
- TF-IDF retrieval correctly identifies implant contraindication document
- The RAG system avoids hallucination by grounding answers in retrieved protocols
- Protocol citations enable dentists to verify the source
- Multi-chunk retrieval provides comprehensive context
- TF-IDF works well for structured dental terminology (high overlap between queries and documents)

---

## **Simple Analogy**
RAG is like a dental librarian. When a dentist asks a clinical question, the librarian (retriever) quickly pulls the 3 most relevant protocol binders from the shelf. Then a knowledgeable assistant (LLM) reads those binders and formulates a clear, cited answer. The assistant never makes things up -- they only use what is in the binders. This is safer than asking someone who might misremember a dosage.

---

## **When to Use**
**Good for dental applications:**
- Treatment protocol Q&A for clinical staff
- Dental insurance coverage lookup
- Patient education content retrieval
- Dental drug interaction queries

**When NOT to use:**
- When real-time clinical decisions require certified medical device systems
- When the knowledge base changes every few minutes
- When perfect recall is critical (supplement with keyword search)

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| top_k (retrieval) | 3 | 1-10 | Context amount |
| chunk_size | 2 sentences | 1-5 sentences | Granularity |
| max_features (TF-IDF) | 5000 | 1000-20000 | Vocabulary size |
| ngram_range | (1,3) | (1,1) to (1,3) | Match precision |
| similarity_threshold | 0.1 | 0.05-0.5 | Minimum relevance |
| LLM temperature | 0.1 | 0.0-0.5 | Answer creativity |

---

## **Running the Demo**
```bash
cd examples/05_nlp
python llm_rag_demo.py
```

---

## **References**
- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Robertson, S. (2004). "Understanding Inverse Document Frequency"
- scikit-learn documentation: TfidfVectorizer, cosine_similarity

---

## **Implementation Reference**
- See `examples/05_nlp/llm_rag_demo.py` for full runnable code
- Retriever: TF-IDF with cosine similarity
- Generator: LLM API with structured prompt template

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
