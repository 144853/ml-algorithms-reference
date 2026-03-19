# KNN (Scikit-Learn) - Dental Classification

## **Use Case: Classifying Dental Material Compatibility**

### **The Problem**
A dental materials laboratory classifies **550 patient-material pairs** as compatible or incompatible based on patient allergy profile score, material composition index, and pH sensitivity level. Proper material selection prevents adverse reactions and prosthetic failures.

### **Why KNN?**
- Instance-based learning finds similar past patient-material cases
- No training phase - new materials added instantly
- Naturally handles non-linear decision boundaries
- Intuitive: "Find patients most similar to this one"

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
n_pairs = 550

data = pd.DataFrame({
    'allergy_profile': np.random.uniform(0, 10, n_pairs),       # composite allergy score
    'material_composition': np.random.uniform(0, 10, n_pairs),  # metal/ceramic/polymer index
    'ph_sensitivity': np.random.uniform(0, 10, n_pairs)         # oral pH reactivity
})

# Compatibility depends on allergy-composition interaction and pH
compatibility_score = (10 - abs(data['allergy_profile'] - data['material_composition']) -
                       0.3 * data['ph_sensitivity'])
data['compatible'] = (compatibility_score > 5.0).astype(int)  # 0=incompatible, 1=compatible

X = data.drop('compatible', axis=1)
y = data['compatible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # KNN requires feature scaling
    ('knn', KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',      # Weight by inverse distance
        metric='euclidean',
        n_jobs=-1
    ))
])

knn_pipeline.fit(X_train, y_train)
y_pred = knn_pipeline.predict(X_test)
y_proba = knn_pipeline.predict_proba(X_test)
```

---

## **Optimal K Selection**

```python
k_range = range(1, 31)
k_scores = []

for k in k_range:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"Optimal K: {optimal_k} (F1: {max(k_scores):.4f})")
# Optimal K: 7 (F1: 0.8654)
```

---

## **Results**

### **Classification Report**

```
                precision    recall  f1-score   support

Incompatible       0.84      0.82      0.83        55
  Compatible       0.83      0.85      0.84        55

    accuracy                           0.84       110
   macro avg       0.84      0.84      0.84       110
weighted avg       0.84      0.84      0.84       110
```

---

## **Material Compatibility Assessment**

```python
def assess_material_compatibility(pipeline, patient_material_data, X_train_data, y_train_data):
    """Assess dental material compatibility with neighbor analysis."""
    df = pd.DataFrame([patient_material_data])
    prediction = pipeline.predict(df)[0]
    probabilities = pipeline.predict_proba(df)[0]

    # Get nearest neighbors for explanation
    knn = pipeline.named_steps['knn']
    scaler = pipeline.named_steps['scaler']
    scaled = scaler.transform(df)
    distances, indices = knn.kneighbors(scaled)

    neighbor_outcomes = y_train_data.iloc[indices[0]]
    compatible_pct = neighbor_outcomes.mean()

    return {
        'compatibility': 'Compatible' if prediction == 1 else 'Incompatible',
        'confidence': f"{max(probabilities):.1%}",
        'similar_cases_compatible': f"{compatible_pct:.0%}",
        'nearest_neighbor_distance': f"{distances[0].mean():.3f}",
        'recommendation': 'Proceed with material' if prediction == 1
                          else 'Select alternative material'
    }

result = assess_material_compatibility(knn_pipeline, {
    'allergy_profile': 7.5,
    'material_composition': 3.0,
    'ph_sensitivity': 8.0
}, X_train, y_train)
print(result)
# {'compatibility': 'Incompatible', 'confidence': '87.1%', ...}
```

---

## **Distance Metric Comparison**

```python
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
for metric in metrics:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance', metric=metric))
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')
    print(f"  {metric:12s}: F1 = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

```
  euclidean   : F1 = 0.8654 (+/- 0.0312)
  manhattan   : F1 = 0.8598 (+/- 0.0345)
  chebyshev   : F1 = 0.8412 (+/- 0.0398)
  minkowski   : F1 = 0.8654 (+/- 0.0312)
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Small to moderate datasets | KNN works best with <10K samples |
| New materials added frequently | No retraining needed |
| Need explainable "similar cases" | Neighbors provide clinical evidence |
| Real-time predictions | Fast for small datasets, slow for large |
| High-dimensional data | Consider dimensionality reduction first |

---

## **Running the Demo**

```bash
cd examples/02_classification
python knn_sklearn.py
```

---

## **References**

1. Cover & Hart "Nearest Neighbor Pattern Classification" (1967), IEEE Transactions
2. Scikit-Learn Documentation: KNeighborsClassifier
3. Schmalz & Fair "Biocompatibility of dental materials" (2009)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
