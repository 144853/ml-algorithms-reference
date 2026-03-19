# SVM (Scikit-Learn) - Dental Classification

## **Use Case: Classifying TMJ Disorder Type**

### **The Problem**
A maxillofacial clinic classifies **700 patients** with temporomandibular joint (TMJ) disorders into three categories: muscular, articular, or combined. Features include jaw opening range, clicking frequency, pain score, and stress level.

### **Why SVM?**
- Excellent for multi-class classification with clear margins
- Effective in high-dimensional spaces
- Kernel trick captures non-linear relationships between symptoms
- Works well with moderate-sized clinical datasets

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
n_patients = 700

data = pd.DataFrame({
    'jaw_opening_mm': np.random.uniform(15, 55, n_patients),          # mm (normal ~40-50)
    'clicking_frequency': np.random.uniform(0, 10, n_patients),        # clicks per minute
    'pain_score': np.random.uniform(0, 10, n_patients),                # VAS 0-10
    'stress_level': np.random.uniform(0, 10, n_patients)               # PSS scale 0-10
})

# TMJ type classification logic
muscular_score = 0.6 * data['stress_level'] + 0.4 * data['pain_score'] - 0.2 * data['clicking_frequency']
articular_score = 0.7 * data['clicking_frequency'] - 0.3 * data['jaw_opening_mm'] / 10 + 0.2 * data['pain_score']
combined_score = 0.4 * data['stress_level'] + 0.4 * data['clicking_frequency'] + 0.3 * data['pain_score']

scores = np.column_stack([muscular_score, articular_score, combined_score])
data['tmj_type'] = np.array(['muscular', 'articular', 'combined'])[scores.argmax(axis=1)]

X = data.drop('tmj_type', axis=1)
y = data['tmj_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training with Pipeline**

```python
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',           # Radial basis function kernel
        C=10.0,                 # Regularization parameter
        gamma='scale',          # Kernel coefficient
        class_weight='balanced',
        probability=True,       # Enable probability estimates
        random_state=42
    ))
])

svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)
y_proba = svm_pipeline.predict_proba(X_test)
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

   articular       0.84      0.82      0.83        45
    combined       0.80      0.83      0.81        47
    muscular       0.86      0.84      0.85        48

    accuracy                           0.83       140
   macro avg       0.83      0.83      0.83       140
weighted avg       0.83      0.83      0.83       140
```

### **Confusion Matrix**

```
              Predicted Muscular  Predicted Articular  Predicted Combined
Actual Muscular         40               3                5
Actual Articular         3              37                5
Actual Combined          4              4               39
```

---

## **Hyperparameter Tuning**

```python
param_grid = {
    'svm__C': [0.1, 1.0, 10.0, 100.0],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1],
    'svm__kernel': ['rbf', 'poly', 'linear']
}

grid_search = GridSearchCV(
    svm_pipeline, param_grid, cv=5,
    scoring='f1_macro', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1 (macro): {grid_search.best_score_:.4f}")
```

---

## **Clinical TMJ Assessment**

```python
def classify_tmj_disorder(pipeline, patient_data):
    """Classify TMJ disorder type for a new patient."""
    df = pd.DataFrame([patient_data])
    prediction = pipeline.predict(df)[0]
    probabilities = pipeline.predict_proba(df)[0]
    classes = pipeline.classes_

    treatment_plans = {
        'muscular': ['Muscle relaxants', 'Stress management therapy',
                     'Physical therapy', 'Night guard (soft)'],
        'articular': ['Hard stabilization splint', 'Joint mobilization',
                      'Arthrocentesis if severe', 'Anti-inflammatory medication'],
        'combined': ['Combined splint therapy', 'Physical therapy + medication',
                     'Behavioral therapy', 'Consider MRI for further evaluation']
    }

    return {
        'tmj_type': prediction.upper(),
        'probabilities': {c: f"{p:.1%}" for c, p in zip(classes, probabilities)},
        'treatment_plan': treatment_plans[prediction],
        'confidence': f"{max(probabilities):.1%}"
    }

result = classify_tmj_disorder(svm_pipeline, {
    'jaw_opening_mm': 28.0,
    'clicking_frequency': 7.5,
    'pain_score': 6.0,
    'stress_level': 3.0
})
print(result)
# {'tmj_type': 'ARTICULAR', 'confidence': '82.3%', ...}
```

---

## **Kernel Comparison**

```python
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, probability=True, random_state=42))
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='f1_macro')
    print(f"  {kernel:8s}: F1 = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

```
  linear  : F1 = 0.7812 (+/- 0.0432)
  rbf     : F1 = 0.8345 (+/- 0.0321)
  poly    : F1 = 0.8087 (+/- 0.0387)
```

---

## **Support Vectors Analysis**

```python
svm_model = svm_pipeline.named_steps['svm']
print(f"Total support vectors: {svm_model.n_support_.sum()}")
print(f"Per class: {dict(zip(svm_model.classes_, svm_model.n_support_))}")
# Total support vectors: 245
# Per class: {'articular': 82, 'combined': 85, 'muscular': 78}
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Clear separation between TMJ types | SVM finds optimal decision boundaries |
| Moderate dataset size | SVM works well with 100-10K samples |
| Non-linear symptom relationships | RBF kernel captures complex patterns |
| Very large datasets (>50K) | Consider tree-based methods (faster) |
| Need probabilistic output | Set `probability=True` (slower training) |

---

## **Running the Demo**

```bash
cd examples/02_classification
python svm_sklearn.py
```

---

## **References**

1. Cortes & Vapnik "Support-Vector Networks" (1995), Machine Learning
2. Scikit-Learn Documentation: SVC
3. De Leeuw & Klasser "Orofacial Pain: Guidelines for Assessment, Diagnosis, and Management" (2018)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
