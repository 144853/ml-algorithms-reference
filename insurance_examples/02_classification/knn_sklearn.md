# K-Nearest Neighbors - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of KNN for **insurance claim benchmark pricing**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[K-Nearest Neighbors - Full Documentation](knn_numpy.md)**

---

## 🛡️ **Insurance Use Case: Similar Claim Benchmark Pricing**

Classify claims into **pricing benchmark tiers** by finding similar historical claims based on:
- Injury type code (1-12: soft tissue, fracture, TBI, spinal, burn, etc.)
- Treatment duration in months (1 - 36)
- Geographic region code (1-9: Northeast, Southeast, Midwest, etc.)
- Attorney involvement (0 = no, 1 = yes)

### **Why Scikit-learn for Claim Benchmarking?**
- **Instance-based**: Finds actual similar claims, not abstract decision boundaries
- **KD-tree indexing**: Fast nearest-neighbor search on large claim databases
- **Distance metrics**: Customize similarity for insurance domain features
- **No training phase**: Instant updates as new claims are settled

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Efficient search**: Ball tree and KD-tree for fast neighbor lookup

---

## 💻 **Quick Start**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Insurance claim benchmark data
# Features: injury_type, treatment_months, region_code, attorney_involved
X, y = load_claim_benchmark_data()  # y: pricing tier (low/medium/high/severe)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale features (critical - treatment months vs binary attorney)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    metric='euclidean',
    algorithm='kd_tree',
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
    target_names=['Low ($1K-$10K)', 'Medium ($10K-$50K)', 'High ($50K-$150K)', 'Severe ($150K+)']))
```

---

## 🔧 **Key Parameters for Claim Benchmarking**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `n_neighbors` | 5 | 7-15 for stable claim benchmark estimates |
| `weights` | `'uniform'` | `'distance'` to weight closer claims more heavily |
| `metric` | `'minkowski'` | `'euclidean'` or custom metric for injury similarity |
| `algorithm` | `'auto'` | `'kd_tree'` for claim databases under 100K |
| `leaf_size` | 30 | Tune for query speed vs memory tradeoff |

### **Insurance-Specific Workflow**

```python
# Find similar claims for a new case
new_claim = scaler.transform([[5, 8, 3, 1]])  # Fracture, 8 months treatment, Midwest, attorney
distances, indices = model.kneighbors(new_claim, n_neighbors=5)

print("--- 5 Most Similar Historical Claims ---")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    claim = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
    settled_amount = settled_amounts[idx]
    print(f"  Claim {i+1}: distance={dist:.3f}, settled=${settled_amount:,.0f}")

# Benchmark: weighted average of similar claim settlements
weights = 1.0 / (distances[0] + 1e-6)
benchmark = np.average(similar_settlements, weights=weights)
print(f"\nBenchmark reserve: ${benchmark:,.0f}")

# Cross-validation for optimal K
from sklearn.model_selection import cross_val_score
for k in [3, 5, 7, 11, 15]:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"K={k}: accuracy={scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Claim Benchmark Classification ---
Accuracy:    79.3%

                    precision  recall  f1-score  support
  Low ($1K-$10K)      0.84    0.86     0.85      420
 Med ($10K-$50K)      0.77    0.75     0.76      380
High ($50K-$150K)     0.76    0.78     0.77      290
    Severe ($150K+)   0.80    0.76     0.78      210

--- Similar Claim Lookup Example ---
New claim: Fracture, 8 months, Midwest, Attorney
  Similar 1: distance=0.21, settled=$67,500
  Similar 2: distance=0.34, settled=$72,100
  Similar 3: distance=0.41, settled=$58,900
  Benchmark reserve: $66,800
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/knn_sklearn.py`](../../02_classification/knn_sklearn.py)

Related:
- [K-Nearest Neighbors - NumPy (from scratch)](knn_numpy.md)
- [K-Nearest Neighbors - PyTorch](knn_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
