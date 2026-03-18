# TF-IDF Text Classifier - Complete Guide with Stock Market Applications

## Overview

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a word in a document relative to a collection (corpus) of documents. When combined with a classifier such as Logistic Regression, Naive Bayes, or SVM, it becomes a powerful tool for text classification tasks. In the stock market domain, TF-IDF classifiers are widely used to analyze financial news, press releases, and analyst reports to determine whether the sentiment is positive or negative for a given stock.

The TF-IDF approach works by converting raw text into a weighted feature vector. Words that appear frequently in a single document but rarely across the entire corpus receive higher weights, capturing their discriminative power. This is particularly useful in financial news classification because domain-specific terms like "earnings beat," "guidance raised," or "downgrade" carry strong predictive signals for stock price movement.

Unlike deep learning approaches, TF-IDF classifiers are lightweight, interpretable, and require minimal computational resources. They serve as an excellent baseline for financial NLP tasks and often perform surprisingly well when combined with careful feature engineering and domain-specific preprocessing. Many quantitative trading firms use TF-IDF-based systems as a first pass for news filtering before applying more sophisticated models.

## How It Works - The Math Behind It

### Term Frequency (TF)

Term Frequency measures how often a word appears in a document. It is computed as:

```
TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```

For example, if the word "earnings" appears 5 times in a 100-word news article, TF("earnings", d) = 5/100 = 0.05.

### Inverse Document Frequency (IDF)

IDF measures how important a term is across the entire corpus. Common words like "the" or "is" appear in many documents and receive low IDF scores, while rare, informative terms receive high scores:

```
IDF(t, D) = log(Total number of documents / Number of documents containing term t)
```

If "earnings" appears in 50 out of 10,000 documents, IDF("earnings") = log(10000/50) = log(200) = 5.30.

### TF-IDF Score

The final TF-IDF weight is the product of TF and IDF:

```
TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
```

### Classification Pipeline

1. **Preprocessing**: Tokenize text, remove stop words, apply stemming/lemmatization
2. **Vocabulary Construction**: Build vocabulary from training corpus
3. **TF-IDF Matrix**: Compute TF-IDF vectors for all documents (shape: n_documents x n_vocabulary)
4. **Classifier Training**: Train a classifier (e.g., Logistic Regression) on TF-IDF features
5. **Prediction**: Transform new documents to TF-IDF vectors and classify

### Logistic Regression Decision Function

```
P(positive | x) = sigmoid(w^T * x + b) = 1 / (1 + exp(-(w^T * x + b)))
```

Where `x` is the TF-IDF vector, `w` is the weight vector, and `b` is the bias term.

### Loss Function (Binary Cross-Entropy)

```
L = -1/N * SUM[y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

### Gradient Update

```
w = w - learning_rate * (1/N) * X^T * (predictions - y)
b = b - learning_rate * (1/N) * SUM(predictions - y)
```

## Stock Market Use Case: Classifying Financial News Articles for Stock Impact

### The Problem

A quantitative trading firm receives thousands of financial news articles daily from sources like Reuters, Bloomberg, and SEC filings. They need to automatically classify each article as having a **positive** or **negative** impact on the mentioned stock's price within the next trading session. Manual analysis is impossible at scale, so they need an automated system that can process articles in milliseconds to generate trading signals before the market fully prices in the information.

### Stock Market Features (Input Data)

| Feature | Description | Example |
|---------|-------------|---------|
| headline | News article headline | "AAPL beats Q3 earnings expectations" |
| body | Full article text | "Apple Inc. reported quarterly..." |
| source | News provider | Reuters, Bloomberg, CNBC |
| ticker | Stock symbol mentioned | AAPL, MSFT, GOOGL |
| published_time | Publication timestamp | 2024-01-15 09:30:00 |
| article_type | Category of news | earnings, merger, analyst_rating |
| word_count | Length of article | 450 |
| has_numbers | Contains financial figures | True/False |

### Example Data Structure

```python
import numpy as np
from collections import Counter
import re

# Sample financial news dataset
news_articles = [
    {"text": "Apple beats quarterly earnings expectations revenue surges 15 percent strong iPhone sales drive growth",
     "label": 1},  # Positive
    {"text": "Tesla misses delivery targets production delays plague factory output falls short of estimates",
     "label": 0},  # Negative
    {"text": "Microsoft announces record cloud revenue Azure growth accelerates enterprise adoption increases",
     "label": 1},  # Positive
    {"text": "Amazon faces antitrust investigation regulators scrutinize market dominance potential fines loom",
     "label": 0},  # Negative
    {"text": "NVIDIA guidance raised strong AI chip demand data center revenue exceeds analyst forecasts",
     "label": 1},  # Positive
    {"text": "Boeing reports wider than expected loss supply chain disruptions continue delivery delays mount",
     "label": 0},  # Negative
    {"text": "Google parent Alphabet stock buyback program expanded dividend initiated shareholder value focus",
     "label": 1},  # Positive
    {"text": "Meta faces declining user engagement advertising revenue drops privacy concerns impact growth",
     "label": 0},  # Negative
    {"text": "JPMorgan reports record trading revenue investment banking fees surge market share gains",
     "label": 1},  # Positive
    {"text": "Wells Fargo hit with regulatory penalty compliance failures lead to massive fine stock drops",
     "label": 0},  # Negative
]

# Financial stop words (common words to remove)
financial_stop_words = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'can', 'shall',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'and',
    'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
    'than', 'that', 'this', 'these', 'those', 'it', 'its'
}

def preprocess(text):
    """Clean and tokenize financial text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in financial_stop_words and len(t) > 2]
    return tokens

# Build vocabulary
all_tokens = []
for article in news_articles:
    all_tokens.extend(preprocess(article["text"]))

vocab = sorted(set(all_tokens))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocab: {vocab[:10]}")
```

### The Model in Action

```python
class TFIDFClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.vocab = None
        self.idf = None
        self.weights = None
        self.bias = None

    def _build_vocab(self, documents):
        all_tokens = set()
        for doc in documents:
            all_tokens.update(preprocess(doc))
        self.vocab = sorted(all_tokens)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

    def _compute_tf(self, tokens):
        tf = np.zeros(len(self.vocab))
        counter = Counter(tokens)
        for word, count in counter.items():
            if word in self.word_to_idx:
                tf[self.word_to_idx[word]] = count / len(tokens)
        return tf

    def _compute_idf(self, documents):
        n_docs = len(documents)
        self.idf = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            doc_count = sum(1 for doc in documents if word in preprocess(doc))
            self.idf[i] = np.log((n_docs + 1) / (doc_count + 1)) + 1

    def _compute_tfidf(self, documents):
        tfidf_matrix = np.zeros((len(documents), len(self.vocab)))
        for i, doc in enumerate(documents):
            tokens = preprocess(doc)
            tf = self._compute_tf(tokens)
            tfidf_matrix[i] = tf * self.idf
        # L2 normalize
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf_matrix = tfidf_matrix / norms
        return tfidf_matrix

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, documents, labels):
        self._build_vocab(documents)
        self._compute_idf(documents)
        X = self._compute_tfidf(documents)
        y = np.array(labels)

        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iteration in range(self.n_iter):
            z = X.dot(self.weights) + self.bias
            predictions = self._sigmoid(z)
            error = predictions - y
            self.weights -= self.lr * X.T.dot(error) / len(y)
            self.bias -= self.lr * np.mean(error)

            if iteration % 200 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-8) +
                               (1 - y) * np.log(1 - predictions + 1e-8))
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

    def predict(self, documents):
        X = self._compute_tfidf(documents)
        z = X.dot(self.weights) + self.bias
        probabilities = self._sigmoid(z)
        return (probabilities >= 0.5).astype(int), probabilities

# Train the model
texts = [a["text"] for a in news_articles]
labels = [a["label"] for a in news_articles]
clf = TFIDFClassifier(learning_rate=0.1, n_iterations=1000)
clf.fit(texts, labels)

# Test on new articles
test_articles = [
    "Company reports strong quarterly profit revenue growth exceeds guidance raised outlook",
    "Stock plunges after earnings miss revenue decline warns of challenging outlook ahead",
    "Firm announces major acquisition premium deal expands market presence strategic growth",
]

predictions, probs = clf.predict(test_articles)
for article, pred, prob in zip(test_articles, predictions, probs):
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"Article: {article[:60]}...")
    print(f"  Prediction: {sentiment} (confidence: {prob:.2%})\n")
```

## Advantages

1. **Extreme Speed for Real-Time Trading**: TF-IDF vectorization and classification can process thousands of news articles per second, making it ideal for high-frequency trading systems that need to react to news before competitors. A typical article can be classified in under 1 millisecond.

2. **High Interpretability for Compliance**: Financial regulators require trading firms to explain their automated decisions. TF-IDF classifiers provide clear feature weights, so you can show exactly which words (e.g., "beat," "miss," "downgrade") drove each classification decision. This transparency is critical for SEC compliance.

3. **Minimal Training Data Requirements**: Unlike deep learning models that need millions of labeled examples, TF-IDF classifiers can perform well with just a few hundred labeled financial articles. This is valuable when building classifiers for niche market sectors where labeled data is scarce.

4. **Low Computational Cost**: No GPU required. A TF-IDF classifier can run on a basic cloud instance, reducing infrastructure costs for smaller trading firms. The entire model can fit in memory on a standard laptop, making it accessible for independent traders and researchers.

5. **Robust to Overfitting with Proper Regularization**: With L1 or L2 regularization, TF-IDF classifiers generalize well even on small financial datasets. The sparse nature of TF-IDF features naturally prevents overfitting to noise in the training data.

6. **Easy to Update and Retrain**: As market language evolves (e.g., new terms like "AI-driven" or "ESG compliance"), the model can be quickly retrained on new data without the lengthy training cycles required by deep learning models. Incremental vocabulary updates are straightforward.

7. **Domain Adaptation Through Feature Engineering**: Financial-specific preprocessing such as preserving ticker symbols, recognizing financial entities, and handling numerical expressions can significantly boost performance without changing the core algorithm.

## Disadvantages

1. **No Understanding of Word Order or Context**: TF-IDF treats documents as bags of words, losing all sequential information. "Stock rises despite earnings miss" and "earnings miss causes stock decline" would have similar TF-IDF vectors despite opposite meanings. This is particularly dangerous in financial contexts where word order conveys critical sentiment shifts.

2. **Struggles with Sarcasm and Nuanced Language**: Financial commentary often contains subtle language, hedging, and conditional statements. TF-IDF cannot capture phrases like "hardly a disaster" or "not as bad as feared," potentially misclassifying nuanced analyst commentary.

3. **Vocabulary Explosion with Financial Jargon**: The financial domain has an enormous vocabulary including ticker symbols, financial ratios, regulatory terms, and industry-specific jargon. This leads to very high-dimensional sparse vectors that can be computationally expensive and may contain noise.

4. **Cannot Handle Numerical Context**: Stock market articles are filled with numbers (price targets, earnings per share, revenue figures). TF-IDF treats "revenue of $10 billion" the same as "revenue of $1 billion" unless extensive feature engineering is applied. Missing numerical context can lead to incorrect sentiment classification.

5. **Poor Generalization Across Market Conditions**: A model trained during a bull market may not perform well during bear markets because the language and sentiment distribution shifts dramatically. The static nature of TF-IDF features makes domain adaptation challenging without retraining.

6. **Limited Handling of Multi-Stock Articles**: Many financial news articles mention multiple stocks with different sentiment directions. TF-IDF classifiers struggle to attribute sentiment to specific stocks within the same article, potentially generating incorrect trading signals.

7. **Sensitivity to Text Preprocessing Choices**: The performance of TF-IDF classifiers is highly sensitive to preprocessing decisions like stemming, stop word removal, and n-gram selection. Poor preprocessing choices specific to financial text can dramatically reduce accuracy.

## When to Use in Stock Market

- **Real-time news filtering**: When you need to process thousands of articles per second to generate trading signals
- **Baseline sentiment analysis**: As a first-pass classifier before applying more complex models
- **Low-latency trading systems**: Where millisecond classification speed is critical for alpha generation
- **Small labeled datasets**: When you have limited labeled financial news data (100-1000 articles)
- **Interpretable trading signals**: When regulatory compliance requires explainable model decisions
- **Resource-constrained environments**: When GPU resources are not available or cost-prohibitive
- **Earnings season screening**: Rapidly classifying earnings reports as beat/miss for portfolio screening

## When NOT to Use in Stock Market

- **Complex sentiment analysis**: When articles contain sarcasm, hedging, or nuanced conditional language
- **Multi-stock articles**: When a single article discusses multiple stocks with different sentiments
- **Long-form documents**: For analyzing lengthy 10-K filings or earnings call transcripts where context matters
- **Multilingual markets**: When processing financial news in multiple languages simultaneously
- **Price prediction**: TF-IDF sentiment alone is insufficient for direct price prediction models
- **High-stakes automated trading**: When classification errors could lead to significant financial losses without human oversight

## Hyperparameters Guide

| Hyperparameter | Typical Range | Stock Market Recommendation | Effect |
|---------------|---------------|---------------------------|--------|
| max_features | 5,000 - 50,000 | 10,000 - 20,000 | Controls vocabulary size; too large adds noise |
| ngram_range | (1,1) to (1,3) | (1,2) | Bigrams capture "earnings beat" vs single words |
| min_df | 1 - 10 | 3 - 5 | Removes very rare terms; reduces noise |
| max_df | 0.7 - 1.0 | 0.85 | Removes overly common terms |
| sublinear_tf | True/False | True | Applies log scaling to TF; reduces impact of repeated terms |
| norm | 'l1', 'l2' | 'l2' | Normalization method; L2 works best for cosine similarity |
| learning_rate | 0.001 - 1.0 | 0.01 - 0.1 | Classifier learning rate; lower for stable convergence |
| regularization (C) | 0.01 - 100 | 0.1 - 10 | Controls overfitting; lower C = more regularization |

## Stock Market Performance Tips

1. **Create a financial-specific stop word list**: Remove common financial boilerplate words like "shares," "company," "Inc.," "Corp." that appear in nearly every article but carry no sentiment signal.

2. **Preserve important bigrams and trigrams**: Phrases like "earnings beat," "guidance raised," "short squeeze," and "hostile takeover" carry much stronger signals than their individual words. Use ngram_range=(1,2) at minimum.

3. **Time-aware training**: Train separate models for different market regimes (bull/bear) or use recent data with higher weight. Financial language sentiment shifts over time as market conditions change.

4. **Sector-specific models**: Train separate TF-IDF classifiers for different sectors (tech, finance, healthcare) since each sector has its own vocabulary and sentiment patterns.

5. **Feature engineering with financial entities**: Extract and normalize ticker symbols, financial figures, analyst names, and institutional investors as separate features alongside TF-IDF vectors.

6. **Combine with market data**: Use TF-IDF sentiment scores as one feature among many in a larger trading model that includes price, volume, and technical indicators.

## Comparison with Other Algorithms

| Feature | TF-IDF Classifier | BERT Classifier | LLM (GPT) | Word2Vec + CNN |
|---------|-------------------|-----------------|------------|----------------|
| Training Speed | Very Fast (seconds) | Slow (hours) | N/A (pre-trained) | Medium (minutes) |
| Inference Speed | <1ms | 10-50ms | 100-500ms | 5-20ms |
| Accuracy (financial) | 75-85% | 88-93% | 85-92% | 82-88% |
| Context Understanding | None | Excellent | Excellent | Limited |
| Data Requirements | 100+ articles | 5,000+ articles | Few-shot | 1,000+ articles |
| Interpretability | High | Low | Medium | Low |
| GPU Required | No | Yes | Yes | Optional |
| Model Size | <10 MB | 400+ MB | 10+ GB | 50-200 MB |
| Handles Negation | Poor | Excellent | Excellent | Moderate |

## Real-World Stock Market Example

```python
import numpy as np
from collections import Counter
import re
from datetime import datetime

# Complete TF-IDF Financial News Classifier
class FinancialNewsTFIDF:
    def __init__(self, max_features=5000, ngram_range=(1, 2), learning_rate=0.05, n_iterations=500):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.vocab = None
        self.idf = None
        self.weights = None
        self.bias = None

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
                      'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'it',
                      'this', 'that', 'has', 'have', 'had', 'be', 'been', 'being'}
        words = [w for w in words if w not in stop_words and len(w) > 1]
        # Generate n-grams
        tokens = list(words)
        if self.ngram_range[1] >= 2:
            for i in range(len(words) - 1):
                tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _build_vocab(self, documents):
        doc_freq = Counter()
        all_freq = Counter()
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_freq[t] += 1
            all_freq.update(self._tokenize(doc))
        # Select top features by frequency, excluding very common/rare terms
        n_docs = len(documents)
        candidates = {t: f for t, f in all_freq.items()
                      if 2 <= doc_freq[t] <= 0.9 * n_docs}
        top_features = sorted(candidates, key=candidates.get, reverse=True)[:self.max_features]
        self.vocab = top_features
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

    def _compute_idf(self, documents):
        n_docs = len(documents)
        self.idf = np.ones(len(self.vocab))
        tokenized_docs = [set(self._tokenize(doc)) for doc in documents]
        for i, word in enumerate(self.vocab):
            doc_count = sum(1 for tokens in tokenized_docs if word in tokens)
            self.idf[i] = np.log((n_docs + 1) / (doc_count + 1)) + 1

    def _transform(self, documents):
        matrix = np.zeros((len(documents), len(self.vocab)))
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            counter = Counter(tokens)
            n_tokens = len(tokens) if tokens else 1
            for word, count in counter.items():
                if word in self.word_to_idx:
                    tf = np.log(1 + count)  # sublinear TF
                    matrix[i, self.word_to_idx[word]] = tf * self.idf[self.word_to_idx[word]]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, documents, labels):
        self._build_vocab(documents)
        self._compute_idf(documents)
        X = self._transform(documents)
        y = np.array(labels, dtype=float)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for epoch in range(self.n_iter):
            preds = self._sigmoid(X.dot(self.weights) + self.bias)
            error = preds - y
            self.weights -= self.lr * (X.T.dot(error) / len(y) + 0.01 * self.weights)
            self.bias -= self.lr * np.mean(error)
        return self

    def predict(self, documents):
        X = self._transform(documents)
        probs = self._sigmoid(X.dot(self.weights) + self.bias)
        return (probs >= 0.5).astype(int), probs

    def top_features(self, n=10):
        pos_idx = np.argsort(self.weights)[-n:][::-1]
        neg_idx = np.argsort(self.weights)[:n]
        print("Top POSITIVE signal words:")
        for i in pos_idx:
            print(f"  {self.vocab[i]:30s} weight: {self.weights[i]:.4f}")
        print("\nTop NEGATIVE signal words:")
        for i in neg_idx:
            print(f"  {self.vocab[i]:30s} weight: {self.weights[i]:.4f}")

# Training data - financial news headlines and snippets
train_texts = [
    "Apple beats earnings expectations iPhone revenue surges record quarter strong demand",
    "Tesla stock drops after missing delivery estimates production challenges continue",
    "Microsoft cloud revenue grows 30 percent Azure adoption accelerates enterprise wins",
    "Boeing faces new safety concerns FAA investigation potential grounding order",
    "NVIDIA raises guidance AI chip demand exceeds forecasts data center growth",
    "Goldman Sachs trading revenue declines layoffs announced cost cutting measures",
    "Amazon Web Services posts record revenue cloud market share expands significantly",
    "Wells Fargo fined by regulators compliance failures customer account scandal",
    "Google advertising revenue beats estimates search dominance drives strong quarter",
    "Intel warns of revenue decline chip market share loss manufacturing delays",
    "JPMorgan reports record profit investment banking fees surge deal pipeline strong",
    "Pfizer stock falls after vaccine demand drops revenue guidance lowered significantly",
    "Netflix subscriber growth exceeds expectations international expansion successful content wins",
    "Meta advertising revenue misses reality labs losses widen spending concerns mount",
    "Salesforce cloud bookings accelerate enterprise AI adoption drives growth momentum",
    "Uber reports first operating loss in quarters driver shortage impacts margins",
]
train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Train the classifier
model = FinancialNewsTFIDF(max_features=500, ngram_range=(1, 2), learning_rate=0.1, n_iterations=500)
model.fit(train_texts, train_labels)

# Show most important features
model.top_features(n=8)

# Classify new articles
test_texts = [
    "Company reports strong quarterly profit revenue growth beats analyst expectations guidance raised",
    "Stock plunges after earnings miss declining revenue warns of challenging market conditions ahead",
    "Firm announces strategic acquisition expanding market presence premium deal shareholders approve",
    "Regulatory investigation launched potential antitrust violations significant fines expected stock falls",
]

predictions, probabilities = model.predict(test_texts)
print("\n--- Classification Results ---")
for text, pred, prob in zip(test_texts, predictions, probabilities):
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = prob if pred == 1 else 1 - prob
    signal = "BUY SIGNAL" if pred == 1 else "SELL SIGNAL"
    print(f"\nHeadline: {text[:70]}...")
    print(f"  Sentiment: {label} | Confidence: {confidence:.1%} | Signal: {signal}")
```

## Key Takeaways

1. **TF-IDF classifiers are the fastest option** for real-time financial news classification, processing articles in under 1ms, making them ideal for latency-sensitive trading systems.

2. **Bigrams are essential** for financial text since phrases like "earnings beat" and "guidance raised" carry far more signal than individual words.

3. **Interpretability is a major advantage** in regulated financial environments where you must explain why a model generated a particular trading signal.

4. **TF-IDF works best as part of a pipeline** rather than a standalone solution. Combine with market data, technical indicators, and more advanced NLP models for robust trading strategies.

5. **Domain-specific preprocessing matters enormously**. Custom stop words, financial entity recognition, and sector-specific vocabulary significantly boost classification accuracy.

6. **Regularly retrain on recent data** since financial language evolves with market conditions, new regulations, and emerging sectors. A model trained on 2020 data may miss AI-related sentiment terms prevalent in 2024.

7. **Always validate with out-of-time testing** rather than random train/test splits. Financial markets are non-stationary, so temporal validation better reflects real-world performance.
