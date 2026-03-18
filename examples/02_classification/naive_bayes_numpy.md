# Naive Bayes - Complete Guide with Stock Market Applications

## Overview

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with the "naive" assumption of conditional independence between features. Despite this simplifying assumption being almost always violated in practice, Naive Bayes classifiers perform remarkably well in many real-world applications, particularly text classification. The algorithm computes the posterior probability of each class given the observed features and assigns the class with the highest posterior probability.

In stock market applications, Naive Bayes is particularly well-suited for news sentiment classification -- determining whether financial news articles, earnings call transcripts, or social media posts convey positive, negative, or neutral sentiment about a stock. The text classification strength of Naive Bayes maps naturally to this problem: given a bag of words from a financial news headline, what is the probability that this news is bullish or bearish for the stock? The algorithm's speed and simplicity make it ideal for processing thousands of news articles in real-time and generating sentiment scores that feed into trading signals.

The Naive Bayes approach is especially valuable in financial NLP because it provides calibrated probability estimates rather than just binary classifications. A headline classified as "positive" with 95% confidence should be treated differently from one classified as "positive" with 55% confidence. These probability outputs enable nuanced position sizing -- larger positions for high-confidence sentiment signals and smaller or no positions for ambiguous sentiment. Additionally, Naive Bayes requires minimal training data compared to deep learning alternatives, making it practical for niche financial topics where labeled data is scarce.

## How It Works - The Math Behind It

### Bayes' Theorem

The foundation of the algorithm:

```
P(C_k | x) = P(x | C_k) * P(C_k) / P(x)
```

Where:
- `P(C_k | x)` = posterior probability of class k given features x
- `P(x | C_k)` = likelihood of features x given class k
- `P(C_k)` = prior probability of class k
- `P(x)` = evidence (normalizing constant)

### The Naive Independence Assumption

The "naive" part assumes all features are conditionally independent given the class:

```
P(x_1, x_2, ..., x_n | C_k) = P(x_1 | C_k) * P(x_2 | C_k) * ... * P(x_n | C_k)
                              = product(P(x_i | C_k))  for i = 1 to n
```

### Classification Decision

The predicted class is:

```
y_hat = argmax_k P(C_k) * product(P(x_i | C_k))
```

Since P(x) is constant across classes, it can be ignored for classification.

### Log-Space Computation (Numerical Stability)

To avoid floating-point underflow from multiplying many small probabilities:

```
y_hat = argmax_k [log P(C_k) + sum(log P(x_i | C_k))]
```

### Multinomial Naive Bayes (for Text/Word Counts)

For text classification of financial news, Multinomial NB is the standard choice:

```
P(x_i | C_k) = (N_ki + alpha) / (N_k + alpha * |V|)
```

Where:
- `N_ki` = number of times word i appears in class k documents
- `N_k` = total word count in class k
- `alpha` = Laplace smoothing parameter (prevents zero probabilities)
- `|V|` = vocabulary size

### Gaussian Naive Bayes (for Continuous Features)

For numerical stock market features:

```
P(x_i | C_k) = (1 / sqrt(2 * pi * sigma_k_i^2)) * exp(-(x_i - mu_k_i)^2 / (2 * sigma_k_i^2))
```

Where `mu_k_i` and `sigma_k_i` are the mean and standard deviation of feature i for class k.

### Bernoulli Naive Bayes (for Binary Features)

For binary presence/absence of keywords:

```
P(x_i | C_k) = p_ki^(x_i) * (1 - p_ki)^(1 - x_i)
```

Where `p_ki` is the probability that word i appears in class k.

### Prior Probabilities

Estimated from training data:

```
P(C_k) = count(C_k) / N_total
```

For financial news, the prior reflects the base rate of positive vs. negative vs. neutral news in the training corpus.

## Stock Market Use Case: News Sentiment Classification for Trading

### The Problem

A quantitative hedge fund wants to incorporate news sentiment into its trading models. The system must classify incoming financial news headlines and article snippets in real-time as Bullish, Bearish, or Neutral. These sentiment scores are then used to generate trading signals -- buying stocks with positive sentiment and shorting stocks with negative sentiment, with position sizes proportional to sentiment confidence.

### Stock Market Features (Input Data)

| Feature | Description | Type | Example Value |
|---------|-------------|------|---------------|
| word_frequencies | TF-IDF or raw count of financial terms | Continuous (sparse) | {"earnings": 2, "beat": 1, "guidance": 1} |
| has_numbers | Contains specific financial figures | Binary | 1 |
| exclamation_count | Number of exclamation marks (urgency indicator) | Count | 0 |
| headline_length | Number of words in headline | Count | 12 |
| source_tier | News source credibility (1=top, 3=blog) | Ordinal | 1 |
| market_hours | Published during market hours | Binary | 1 |
| ticker_mentioned | Specific stock ticker referenced | Binary | 1 |
| sector_category | Sector of the referenced stock | Categorical | "Technology" |
| has_analyst_name | Mentions a specific analyst | Binary | 0 |
| day_of_week | Day of publication | Categorical | "Monday" |
| contains_upgrade | Contains "upgrade" or "downgrade" | Binary | 1 |
| earnings_related | Related to earnings announcement | Binary | 1 |
| management_quote | Contains direct management quote | Binary | 0 |
| word_sentiment_score | Average word-level sentiment | Continuous | 0.35 |

**Target Variable:** `sentiment` -- 0 (Bearish), 1 (Neutral), 2 (Bullish)

### Example Data Structure

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# Simulated financial news vocabulary
bullish_words = ['beat', 'exceeded', 'growth', 'upgrade', 'strong', 'record',
                 'outperform', 'raised', 'acceleration', 'bullish', 'breakout',
                 'momentum', 'guidance_raised', 'buyback', 'dividend_increase',
                 'expansion', 'innovation', 'partnership', 'upside', 'rally']

bearish_words = ['missed', 'decline', 'downgrade', 'weak', 'loss', 'warning',
                 'layoffs', 'investigation', 'recall', 'bearish', 'crash',
                 'default', 'guidance_cut', 'restructuring', 'debt_concern',
                 'slowdown', 'competition', 'lawsuit', 'overvalued', 'selloff']

neutral_words = ['quarterly', 'report', 'results', 'announced', 'shares',
                 'market', 'trading', 'volume', 'price', 'stock', 'company',
                 'sector', 'industry', 'analysts', 'expectations', 'revenue',
                 'earnings', 'fiscal', 'statement', 'conference']

all_words = bullish_words + bearish_words + neutral_words
vocab_size = len(all_words)

n_articles = 2000

# Generate word count features (bag of words)
X_text = np.zeros((n_articles, vocab_size))
y_sentiment = np.zeros(n_articles, dtype=int)

for i in range(n_articles):
    # Assign sentiment label
    sentiment = np.random.choice([0, 1, 2], p=[0.25, 0.35, 0.40])
    y_sentiment[i] = sentiment

    # Generate word counts based on sentiment
    n_words = np.random.randint(5, 25)
    for _ in range(n_words):
        if sentiment == 2:  # Bullish
            word_idx = np.random.choice(range(vocab_size),
                       p=np.array([0.04]*20 + [0.005]*20 + [0.02]*20) /
                         sum([0.04]*20 + [0.005]*20 + [0.02]*20))
        elif sentiment == 0:  # Bearish
            word_idx = np.random.choice(range(vocab_size),
                       p=np.array([0.005]*20 + [0.04]*20 + [0.02]*20) /
                         sum([0.005]*20 + [0.04]*20 + [0.02]*20))
        else:  # Neutral
            word_idx = np.random.choice(range(vocab_size),
                       p=np.array([0.015]*20 + [0.015]*20 + [0.035]*20) /
                         sum([0.015]*20 + [0.015]*20 + [0.035]*20))
        X_text[i, word_idx] += 1

print(f"Dataset: {n_articles} articles, {vocab_size} word features")
print(f"Sentiment distribution: Bearish={np.sum(y_sentiment==0)}, "
      f"Neutral={np.sum(y_sentiment==1)}, Bullish={np.sum(y_sentiment==2)}")
print(f"Average words per article: {X_text.sum(axis=1).mean():.1f}")
```

### The Model in Action

```python
# Naive Bayes classification of a financial news headline:
# "Apple beats earnings estimates, raises guidance on strong iPhone demand"

# Step 1: Tokenize and count words
# word_counts = {beats: 1, earnings: 1, estimates: 1, raises: 1,
#                guidance: 1, strong: 1, demand: 1}

# Step 2: Compute log-likelihoods for each class

# log P(Bullish | words):
#   log P(Bullish) = log(0.40) = -0.916
#   log P("beats" | Bullish)    = log(0.045) = -3.101
#   log P("raises" | Bullish)   = log(0.038) = -3.270
#   log P("strong" | Bullish)   = log(0.042) = -3.170
#   log P("guidance" | Bullish) = log(0.030) = -3.507
#   Total log score = -0.916 + (-3.101) + (-3.270) + (-3.170) + (-3.507) + ...
#                   = -17.43

# log P(Bearish | words):
#   log P(Bearish) = log(0.25) = -1.386
#   log P("beats" | Bearish)    = log(0.005) = -5.298  (rare in bearish news)
#   log P("raises" | Bearish)   = log(0.004) = -5.521
#   log P("strong" | Bearish)   = log(0.003) = -5.809
#   Total log score = -1.386 + (-5.298) + (-5.521) + (-5.809) + ...
#                   = -28.15

# log P(Neutral | words):
#   Total log score = -22.84

# Step 3: Classification
#   argmax = Bullish (-17.43 > -22.84 > -28.15)
#   Confidence: P(Bullish) = exp(-17.43) / (exp(-17.43) + exp(-22.84) + exp(-28.15))
#             = 0.9955 (very high confidence)

# Trading Action: BUY AAPL with full position size (confidence > 0.90 threshold)
```

## Advantages

1. **Extremely fast training and prediction for real-time news processing.** Naive Bayes training is O(N*d) where N is documents and d is vocabulary size -- essentially a single pass through the data counting word occurrences per class. Prediction is O(d) per document. This means a model can be trained on millions of historical articles in seconds and classify incoming news in microseconds, critical for news-driven trading where the first few seconds after a headline determine price impact.

2. **Works well with small labeled datasets.** Financial sentiment labeling is expensive -- it requires domain experts who understand market context. Naive Bayes can achieve good accuracy with just a few hundred labeled examples due to its low parameter count. This is a major advantage over deep learning approaches like BERT fine-tuning, which require thousands of labeled examples to perform well on financial text.

3. **Naturally produces calibrated probability estimates.** The posterior probabilities from Naive Bayes are directly interpretable as confidence levels, unlike many other classifiers that require additional calibration. A prediction of P(Bullish)=0.85 vs. P(Bullish)=0.55 provides a natural basis for position sizing: allocate more capital when sentiment is clear and less when it is ambiguous.

4. **Handles high-dimensional sparse features naturally.** Financial text represented as bag-of-words creates very high-dimensional (10,000+ words) but sparse feature vectors. Naive Bayes handles this naturally because it estimates parameters independently per feature. The conditional independence assumption is actually an advantage here, preventing the curse of dimensionality that affects other classifiers.

5. **Robust to irrelevant features.** Financial news contains many words that are irrelevant to sentiment (company names, dates, boilerplate language). Naive Bayes effectively ignores these by learning near-uniform probability distributions for them across classes. There is no need for extensive feature selection, though TF-IDF weighting can further improve focus on discriminative terms.

6. **Transparent and interpretable model.** The word-class probability table is directly inspectable -- you can see exactly which words the model associates with bullish vs. bearish sentiment. Compliance officers and portfolio managers can audit the model by examining these probabilities, and domain experts can spot obvious errors (e.g., if "short" is associated with bearish when it should be context-dependent).

7. **Easy to update incrementally.** When new labeled articles arrive, the word counts and class priors can be updated without retraining from scratch. This online learning capability is valuable for adapting to evolving financial language -- new jargon, ticker symbols, and market terms can be incorporated continuously without maintaining the full historical dataset.

## Disadvantages

1. **The conditional independence assumption is strongly violated in financial text.** Words in financial news are highly correlated: "raised guidance" is far more bullish than "raised" and "guidance" independently, but Naive Bayes treats them as independent features. Bigrams like "not good" are misclassified because "not" and "good" are evaluated separately. This limits accuracy on nuanced financial language where word order and context matter.

2. **Cannot capture word order or contextual meaning.** Financial language is highly context-dependent. "Short interest increased" is bearish, but "short-term growth" is bullish. Naive Bayes with bag-of-words representation loses all word order information, missing critical semantic distinctions. N-gram features can partially mitigate this but increase dimensionality dramatically.

3. **Probability estimates can be overconfident.** Despite being "calibrated," Naive Bayes posterior probabilities tend to be pushed toward 0 and 1 due to the independence assumption multiplying many conditionally dependent terms. A headline might receive P(Bullish)=0.99 when the true confidence should be 0.75. Overconfident probabilities lead to oversized positions and poor risk management.

4. **Struggles with negation and sarcasm.** Financial commentary, especially on social media, frequently uses negation ("not expected to beat estimates") and sarcasm ("great, another revenue miss"). Naive Bayes cannot handle negation without explicit negation tagging preprocessing, and sarcasm detection is beyond its capabilities entirely, leading to misclassification of ironic or sarcastic financial commentary.

5. **Sensitive to vocabulary distribution shifts.** Financial language evolves -- terms like "meme stock," "SPAC," "NFT," and "AI-driven" emerge and change meaning over time. A model trained on pre-2020 financial news may not understand pandemic-era terminology. The Laplace smoothing handles unseen words but assigns them uniform probability, which may not reflect their actual sentiment.

6. **Assumes all features contribute equally (without TF-IDF).** Without feature weighting, common words like "stock," "market," and "company" that appear in all sentiment classes receive equal treatment as discriminative words like "downgrade" or "breakout." TF-IDF weighting is essential but introduces a preprocessing dependency that must be consistently applied between training and prediction.

7. **Poor at multi-document reasoning.** A trading decision often requires synthesizing sentiment across multiple articles about the same stock. Naive Bayes classifies each article independently and cannot reason about contradictory reports (one bullish, one bearish). Aggregating multiple Naive Bayes outputs requires an additional meta-model layer.

## When to Use in Stock Market

- For real-time news headline sentiment classification where speed is critical (sub-millisecond prediction)
- When labeled training data is limited (fewer than 1000 labeled articles)
- For processing high volumes of text data (thousands of articles per day across many stocks)
- As a baseline sentiment model before evaluating more complex alternatives (BERT, GPT)
- When model interpretability is required -- showing which words drive the sentiment classification
- For email/alert filtering to flag relevant financial news for human traders
- As a first-stage filter in a multi-stage NLP pipeline (Naive Bayes filters, then deep model refines)
- When the trading strategy is not extremely sensitive to sentiment accuracy (e.g., portfolio tilting rather than single-stock bets)

## When NOT to Use in Stock Market

- When nuanced context understanding is critical (e.g., distinguishing between types of "growth")
- For processing long-form documents like 10-K filings or earnings call transcripts where document structure matters
- When sarcasm, irony, or figurative language is common in the text source (e.g., financial Twitter)
- For multi-class sentiment with fine granularity (strongly bullish/bullish/neutral/bearish/strongly bearish)
- When word order and phrase-level meaning are essential for accurate classification
- For cross-lingual sentiment analysis where direct word translation is unreliable
- When the highest possible accuracy is required and computational resources are not constrained

## Hyperparameters Guide

| Hyperparameter | Description | Typical Range | Stock Market Recommendation |
|----------------|-------------|---------------|----------------------------|
| alpha | Laplace smoothing parameter | 0.01 - 10.0 | Start with 1.0. Decrease to 0.1 for large vocabularies where smoothing dilutes discriminative words. Increase for small training sets to prevent overfitting to rare words. |
| fit_prior | Whether to learn class prior probabilities | True / False | True for imbalanced sentiment data (more neutral than bullish/bearish). Set False if you want equal prior assumption regardless of training distribution. |
| class_prior | Explicit prior probabilities per class | Array summing to 1 | Set manually if the training data's sentiment distribution differs from expected real-world distribution. E.g., [0.25, 0.50, 0.25] for bearish/neutral/bullish. |
| Feature representation | Bag-of-words, TF-IDF, binary | - | TF-IDF with sublinear TF is best for financial text. Downweights common financial terms and highlights discriminative sentiment words. |
| Vocabulary size | Maximum number of features | 5000 - 50000 | 10,000-20,000 words captures financial terminology well. Include bigrams for phrases like "raised guidance" and "missed estimates." |
| Min document frequency | Minimum documents a word must appear in | 2 - 10 | Set to 3-5 to remove typos and extremely rare terms. Financial tickers that appear rarely should be handled separately, not as vocabulary features. |
| Max document frequency | Maximum fraction of documents a word can appear in | 0.7 - 0.95 | Set to 0.85 to remove ubiquitous words like "stock" and "market" that carry no sentiment information. |

## Stock Market Performance Tips

1. **Build a financial-specific vocabulary.** General NLP vocabularies miss critical financial terms. Include analyst-specific language ("outperform," "accumulate," "hold"), earnings terminology ("beat," "miss," "in-line"), and market structure terms ("short squeeze," "dead cat bounce"). A domain-specific vocabulary of 15,000-20,000 terms typically outperforms a general 100,000-word vocabulary.

2. **Apply financial negation handling.** Before tokenization, apply negation tagging: when a negation word ("not," "no," "never," "failed to") is detected, prefix subsequent words with "NOT_" until the next punctuation mark. This transforms "not expected to beat" into "not NOT_expected NOT_to NOT_beat," allowing Naive Bayes to learn that "NOT_beat" is bearish while "beat" is bullish.

3. **Use TF-IDF with sublinear term frequency.** Apply `tf = 1 + log(tf)` to prevent a single word mentioned many times from dominating the feature vector. Financial press releases sometimes repeat key terms ("growth growth growth"), and sublinear TF ensures diminishing returns from repetition.

4. **Weight recent training data more heavily.** Financial language and its sentiment connotations evolve over time. Use exponential decay weighting on training samples: more recent articles receive higher weight in the word-class probability estimates. This keeps the model current with evolving financial terminology without discarding historical patterns.

5. **Combine with entity recognition.** Pre-process articles to replace stock tickers, company names, and analyst names with generic tokens (TICKER, COMPANY, ANALYST). This prevents the model from learning stock-specific biases (e.g., a model that associates "TSLA" with "bullish" because Tesla had more bullish coverage in training data).

6. **Aggregate sentiment across multiple sources.** When multiple news articles reference the same stock within a short window, aggregate their individual Naive Bayes sentiment probabilities using logarithmic opinion pooling: `P_combined = normalize(product(P_i^w_i))` where `w_i` reflects source credibility. This produces a more reliable composite sentiment signal.

7. **Monitor and recalibrate regularly.** Track the model's predicted probability distribution and actual outcomes weekly. If the model is systematically overconfident (predicting P(Bullish)=0.90 but only 70% of such predictions are correct), apply Platt scaling recalibration to align probabilities with observed frequencies.

## Comparison with Other Algorithms

| Criterion | Multinomial Naive Bayes | Logistic Regression | Random Forest | SVM (Linear) | BERT / Transformers |
|-----------|------------------------|---------------------|---------------|--------------|---------------------|
| Training speed | Very fast (seconds) | Fast (minutes) | Moderate | Moderate | Very slow (hours/GPU) |
| Prediction speed | Very fast (microseconds) | Very fast | Moderate | Fast | Slow (milliseconds) |
| Small data performance | Good | Moderate | Poor | Moderate | Poor (needs fine-tuning data) |
| Text classification accuracy | Good | Good | Moderate | Good | Excellent |
| Context understanding | None (bag of words) | None (bag of words) | None | None | Excellent (contextual) |
| Probability calibration | Moderate (overconfident) | Good | Good | Poor | Moderate |
| Interpretability | Excellent (word probs) | Good (coefficients) | Moderate | Moderate | Very poor |
| Negation handling | Poor (without preprocessing) | Poor | Poor | Poor | Good |
| Incremental learning | Easy (update counts) | SGD variant possible | Must retrain | Must retrain | Must fine-tune |
| Memory footprint | Very small | Small | Large | Moderate | Very large (GPU) |
| Best financial NLP use | Fast headline screening | Balanced accuracy/speed | Structured data | Long documents | Nuanced analysis |

## Real-World Stock Market Example

```python
import numpy as np
from collections import Counter, defaultdict

# =============================================================================
# Naive Bayes from Scratch for Financial News Sentiment Classification
# =============================================================================

class MultinomialNaiveBayes:
    """Multinomial Naive Bayes for financial text classification."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_log_prior = None
        self.feature_log_prob = None
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.n_features = X.shape[1]

        # Compute class priors
        class_counts = np.array([np.sum(y == c) for c in self.classes])
        self.class_log_prior = np.log(class_counts / class_counts.sum())

        # Compute feature likelihoods with Laplace smoothing
        self.feature_log_prob = np.zeros((n_classes, self.n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # Total word counts per feature for this class
            feature_counts = X_c.sum(axis=0) + self.alpha
            total_count = feature_counts.sum()
            self.feature_log_prob[idx] = np.log(feature_counts / total_count)

        return self

    def predict_log_proba(self, X):
        """Compute log posterior probabilities."""
        # log P(C) + sum(x_i * log P(w_i | C))
        log_proba = X @ self.feature_log_prob.T + self.class_log_prior
        return log_proba

    def predict_proba(self, X):
        """Compute posterior probabilities using log-sum-exp trick."""
        log_proba = self.predict_log_proba(X)
        # Log-sum-exp for numerical stability
        log_proba_max = log_proba.max(axis=1, keepdims=True)
        log_proba_shifted = log_proba - log_proba_max
        proba = np.exp(log_proba_shifted)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """Predict class labels."""
        log_proba = self.predict_log_proba(X)
        return self.classes[np.argmax(log_proba, axis=1)]

    def top_features_per_class(self, feature_names, n_top=10):
        """Get the most indicative words for each class."""
        results = {}
        for idx, c in enumerate(self.classes):
            # Words with highest log probability in this class
            # relative to average across classes
            avg_log_prob = self.feature_log_prob.mean(axis=0)
            relative_importance = self.feature_log_prob[idx] - avg_log_prob
            top_indices = np.argsort(relative_importance)[-n_top:][::-1]
            results[c] = [(feature_names[i], relative_importance[i])
                         for i in top_indices]
        return results


class GaussianNaiveBayes:
    """Gaussian Naive Bayes for continuous stock market features."""

    def __init__(self):
        self.class_log_prior = None
        self.means = None
        self.variances = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        class_counts = np.array([np.sum(y == c) for c in self.classes])
        self.class_log_prior = np.log(class_counts / class_counts.sum())

        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx] = X_c.mean(axis=0)
            self.variances[idx] = X_c.var(axis=0) + 1e-9  # stability

        return self

    def _log_likelihood(self, X):
        """Compute log P(X | class) for each class."""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihood = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            diff = X - self.means[idx]
            log_likelihood[:, idx] = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances[idx]) +
                diff**2 / self.variances[idx],
                axis=1
            )

        return log_likelihood

    def predict_proba(self, X):
        log_proba = self._log_likelihood(X) + self.class_log_prior
        log_proba_max = log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba - log_proba_max)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        log_proba = self._log_likelihood(X) + self.class_log_prior
        return self.classes[np.argmax(log_proba, axis=1)]


# =============================================================================
# Financial News Sentiment Data Generation
# =============================================================================

def create_financial_vocabulary():
    """Create a financial-specific vocabulary for sentiment analysis."""
    vocab = {
        # Bullish words (indices 0-24)
        'beat': 0, 'exceeded': 1, 'growth': 2, 'upgrade': 3, 'strong': 4,
        'record': 5, 'outperform': 6, 'raised': 7, 'acceleration': 8,
        'bullish': 9, 'breakout': 10, 'momentum': 11, 'guidance_raised': 12,
        'buyback': 13, 'dividend_increase': 14, 'expansion': 15,
        'innovation': 16, 'partnership': 17, 'upside': 18, 'rally': 19,
        'optimistic': 20, 'surged': 21, 'profit': 22, 'gained': 23,
        'positive': 24,

        # Bearish words (indices 25-49)
        'missed': 25, 'decline': 26, 'downgrade': 27, 'weak': 28, 'loss': 29,
        'warning': 30, 'layoffs': 31, 'investigation': 32, 'recall': 33,
        'bearish': 34, 'crash': 35, 'default': 36, 'guidance_cut': 37,
        'restructuring': 38, 'debt': 39, 'slowdown': 40, 'competition': 41,
        'lawsuit': 42, 'overvalued': 43, 'selloff': 44, 'pessimistic': 45,
        'plunged': 46, 'deficit': 47, 'dropped': 48, 'negative': 49,

        # Neutral words (indices 50-74)
        'quarterly': 50, 'report': 51, 'results': 52, 'announced': 53,
        'shares': 54, 'market': 55, 'trading': 56, 'volume': 57,
        'price': 58, 'stock': 59, 'company': 60, 'sector': 61,
        'industry': 62, 'analysts': 63, 'expectations': 64, 'revenue': 65,
        'earnings': 66, 'fiscal': 67, 'statement': 68, 'conference': 69,
        'management': 70, 'investors': 71, 'holdings': 72, 'portfolio': 73,
        'exchange': 74,
    }
    return vocab


def generate_financial_news_data(n_articles=3000, seed=42):
    """Generate synthetic financial news sentiment data."""
    np.random.seed(seed)
    vocab = create_financial_vocabulary()
    vocab_size = len(vocab)
    word_names = list(vocab.keys())

    X = np.zeros((n_articles, vocab_size))
    y = np.zeros(n_articles, dtype=int)

    # Sentiment-specific word probability distributions
    bullish_probs = np.zeros(vocab_size)
    bullish_probs[0:25] = 0.035    # bullish words: high probability
    bullish_probs[25:50] = 0.003   # bearish words: low probability
    bullish_probs[50:75] = 0.015   # neutral words: moderate probability
    bullish_probs /= bullish_probs.sum()

    bearish_probs = np.zeros(vocab_size)
    bearish_probs[0:25] = 0.003    # bullish words: low probability
    bearish_probs[25:50] = 0.035   # bearish words: high probability
    bearish_probs[50:75] = 0.015   # neutral words: moderate probability
    bearish_probs /= bearish_probs.sum()

    neutral_probs = np.zeros(vocab_size)
    neutral_probs[0:25] = 0.012    # bullish words: moderate probability
    neutral_probs[25:50] = 0.012   # bearish words: moderate probability
    neutral_probs[50:75] = 0.030   # neutral words: high probability
    neutral_probs /= neutral_probs.sum()

    prob_maps = {0: bearish_probs, 1: neutral_probs, 2: bullish_probs}

    for i in range(n_articles):
        sentiment = np.random.choice([0, 1, 2], p=[0.25, 0.35, 0.40])
        y[i] = sentiment

        n_words = np.random.randint(8, 30)
        word_indices = np.random.choice(vocab_size, size=n_words,
                                        p=prob_maps[sentiment])
        for idx in word_indices:
            X[i, idx] += 1

    return X, y, word_names


# =============================================================================
# TF-IDF Transformation
# =============================================================================

class TFIDF:
    """TF-IDF transformer for financial text."""

    def __init__(self, sublinear_tf=True):
        self.sublinear_tf = sublinear_tf
        self.idf = None

    def fit(self, X):
        n_docs = X.shape[0]
        df = np.sum(X > 0, axis=0)  # document frequency
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1  # smooth IDF
        return self

    def transform(self, X):
        if self.sublinear_tf:
            tf = np.where(X > 0, 1 + np.log(X + 1e-10), 0)
        else:
            tf = X
        tfidf = tf * self.idf

        # L2 normalize each document
        norms = np.sqrt(np.sum(tfidf**2, axis=1, keepdims=True))
        norms = np.where(norms == 0, 1, norms)
        return tfidf / norms

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# =============================================================================
# Training and Evaluation Pipeline
# =============================================================================

# Generate data
X_raw, y, word_names = generate_financial_news_data(n_articles=3000)
sentiment_names = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}

print("=" * 65)
print("Naive Bayes Financial News Sentiment Classifier")
print("=" * 65)
print(f"\nDataset: {X_raw.shape[0]} articles, {X_raw.shape[1]} word features")
print(f"Sentiment distribution: {dict(Counter(y))}")

# Time-based split (articles arrive chronologically)
split = int(0.7 * len(X_raw))
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train, y_test = y[:split], y[split:]

# Apply TF-IDF
tfidf = TFIDF(sublinear_tf=True)
X_train = tfidf.fit_transform(X_train_raw)
X_test = tfidf.transform(X_test_raw)

print(f"\nTrain: {len(X_train)} articles | Test: {len(X_test)} articles")

# --- Multinomial NB on raw counts ---
print("\n--- Multinomial Naive Bayes (raw counts) ---")
mnb_raw = MultinomialNaiveBayes(alpha=1.0)
mnb_raw.fit(X_train_raw, y_train)
train_preds_raw = mnb_raw.predict(X_train_raw)
test_preds_raw = mnb_raw.predict(X_test_raw)
print(f"Train accuracy: {np.mean(train_preds_raw == y_train):.4f}")
print(f"Test accuracy:  {np.mean(test_preds_raw == y_test):.4f}")

# --- Multinomial NB on TF-IDF ---
print("\n--- Multinomial Naive Bayes (TF-IDF) ---")
# For MNB on TF-IDF, ensure non-negative values
X_train_pos = np.maximum(X_train, 0)
X_test_pos = np.maximum(X_test, 0)
mnb_tfidf = MultinomialNaiveBayes(alpha=0.5)
mnb_tfidf.fit(X_train_pos, y_train)
train_preds_tfidf = mnb_tfidf.predict(X_train_pos)
test_preds_tfidf = mnb_tfidf.predict(X_test_pos)
print(f"Train accuracy: {np.mean(train_preds_tfidf == y_train):.4f}")
print(f"Test accuracy:  {np.mean(test_preds_tfidf == y_test):.4f}")

# Per-class accuracy
print("\nPer-class accuracy (test, TF-IDF):")
for cls in [0, 1, 2]:
    mask = y_test == cls
    if mask.sum() > 0:
        acc = np.mean(test_preds_tfidf[mask] == cls)
        print(f"  {sentiment_names[cls]:>10}: {acc:.4f} ({mask.sum()} samples)")

# Confusion matrix
print("\nConfusion Matrix (Test, TF-IDF):")
print(f"{'':>14} {'Pred Bear':>10} {'Pred Neut':>10} {'Pred Bull':>10}")
for true_cls in [0, 1, 2]:
    row = []
    for pred_cls in [0, 1, 2]:
        count = np.sum((y_test == true_cls) & (test_preds_tfidf == pred_cls))
        row.append(count)
    name = sentiment_names[true_cls]
    print(f"{'True '+name:>14} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

# Top discriminative words per sentiment
print("\n--- Top Discriminative Words ---")
top_words = mnb_tfidf.top_features_per_class(word_names, n_top=8)
for cls, words in top_words.items():
    print(f"\n{sentiment_names[cls]} indicators:")
    for word, score in words:
        bar = '#' * int(max(0, score) * 20)
        print(f"  {word:>20s}: {score:+.4f} {bar}")

# Probability calibration analysis
print("\n--- Probability Calibration ---")
probas = mnb_tfidf.predict_proba(X_test_pos)
max_probas = probas.max(axis=1)

# Bin by confidence and check accuracy
confidence_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
print(f"{'Confidence':>12} {'Accuracy':>10} {'Count':>8}")
for low, high in confidence_bins:
    mask = (max_probas >= low) & (max_probas < high)
    if mask.sum() > 0:
        acc = np.mean(test_preds_tfidf[mask] == y_test[mask])
        print(f"  [{low:.1f}, {high:.1f}): {acc:>10.4f} {mask.sum():>8d}")

# Simulated sentiment-based trading
print("\n--- Sentiment-Based Trading Simulation ---")
simulated_stock_returns = np.random.normal(0.0005, 0.018, len(y_test))

# Position based on sentiment prediction and confidence
positions = np.zeros(len(y_test))
for i in range(len(y_test)):
    pred = test_preds_tfidf[i]
    conf = max_probas[i]

    if pred == 2 and conf > 0.6:      # Bullish with confidence
        positions[i] = conf            # Long, sized by confidence
    elif pred == 0 and conf > 0.6:     # Bearish with confidence
        positions[i] = -conf           # Short, sized by confidence
    else:
        positions[i] = 0               # No position

strategy_returns = positions * simulated_stock_returns
active_days = np.sum(positions != 0)

cum_strategy = np.cumprod(1 + strategy_returns)
cum_buyhold = np.cumprod(1 + simulated_stock_returns)

print(f"Active trading days: {active_days} / {len(y_test)} "
      f"({active_days/len(y_test)*100:.1f}%)")
print(f"Strategy cumulative return: {(cum_strategy[-1]-1)*100:.2f}%")
print(f"Buy & Hold cumulative return: {(cum_buyhold[-1]-1)*100:.2f}%")

if np.std(strategy_returns[positions != 0]) > 0:
    active_sharpe = (np.sqrt(252) * np.mean(strategy_returns[positions != 0]) /
                     np.std(strategy_returns[positions != 0]))
    print(f"Strategy Sharpe (active days): {active_sharpe:.2f}")

# Example single prediction
print("\n--- Example Prediction ---")
example_idx = 0
example_probs = probas[example_idx]
print(f"Article word counts: {dict(zip([word_names[j] for j in np.where(X_test_raw[example_idx] > 0)[0]], X_test_raw[example_idx][X_test_raw[example_idx] > 0].astype(int)))}")
print(f"Predicted probabilities:")
for cls in [0, 1, 2]:
    print(f"  {sentiment_names[cls]:>10}: {example_probs[cls]:.4f}")
print(f"Prediction: {sentiment_names[test_preds_tfidf[example_idx]]}")
print(f"True label: {sentiment_names[y_test[example_idx]]}")
```

## Key Takeaways

1. **Naive Bayes is the optimal choice for real-time financial news sentiment classification** when speed and simplicity are paramount. Its microsecond prediction time and minimal training data requirements make it ideal for processing high-volume news feeds.

2. **The conditional independence assumption is a known weakness** but is partially compensated by its benefits: low variance, resistance to overfitting, and natural probability outputs. For financial text where many words are independently informative ("beat," "raised," "downgrade"), the assumption is less damaging than in other domains.

3. **TF-IDF preprocessing is essential.** Raw word counts allow common financial terms to overwhelm sentiment-bearing words. TF-IDF with sublinear term frequency ensures that discriminative words like "upgrade" or "selloff" receive appropriate weight relative to ubiquitous terms like "stock" and "market."

4. **Probability calibration requires monitoring.** While Naive Bayes provides natural probability outputs, they tend to be overconfident due to the independence assumption. Regular calibration checks and correction (via Platt scaling) are necessary for reliable position sizing.

5. **Financial vocabulary engineering is as important as algorithm selection.** Including domain-specific bigrams ("raised guidance," "missed estimates"), handling negation explicitly, and removing non-informative terms can improve accuracy more than switching to a more complex algorithm.

6. **Naive Bayes excels as a fast first-stage filter** in multi-stage NLP pipelines. Use it to quickly classify obvious bullish/bearish headlines, then pass ambiguous cases (low confidence) to a slower but more accurate model (BERT, GPT) for refined classification.

7. **Incremental learning is a unique advantage** for adapting to evolving financial language. The model can be updated with new labeled articles by simply updating word counts, without storing or reprocessing the full historical corpus.
