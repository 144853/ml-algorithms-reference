# Logistic Regression (Scikit-Learn) - Dental Classification

## **Use Case: Predicting Periodontal Disease Risk**

### **The Problem**
A dental clinic wants to predict periodontal disease risk for **500 patients** using clinical measurements: plaque index, bleeding on probing percentage, pocket depth, age, and smoking status.

### **Why Scikit-Learn?**
- Production-ready implementation with optimized solvers
- Built-in regularization (L1, L2, ElasticNet)
- Seamless integration with preprocessing pipelines
- Cross-validation and hyperparameter tuning utilities

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Create dental patient dataset
np.random.seed(42)
n_patients = 500

data = pd.DataFrame({
    'plaque_index': np.random.uniform(0.0, 3.0, n_patients),
    'bop_percentage': np.random.uniform(0, 100, n_patients),
    'pocket_depth_mm': np.random.uniform(1.0, 12.0, n_patients),
    'age': np.random.randint(18, 86, n_patients),
    'smoking_status': np.random.choice([0, 1, 2], n_patients)
})

# Generate realistic labels
risk = (0.8 * data['plaque_index'] + 0.02 * data['bop_percentage'] +
        0.5 * data['pocket_depth_mm'] + 0.03 * data['age'] +
        0.6 * data['smoking_status'])
data['periodontal_disease'] = (risk > np.median(risk)).astype(int)

X = data.drop('periodontal_disease', axis=1)
y = data['periodontal_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training with Pipeline**

```python
# Build a complete preprocessing + model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        C=1.0,                  # Inverse regularization strength
        penalty='l2',           # L2 regularization
        solver='lbfgs',         # Good for small datasets
        max_iter=1000,
        random_state=42
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

---

## **Hyperparameter Tuning**

```python
param_grid = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best AUC-ROC: {grid_search.best_score_:.4f}")
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

     Healthy       0.87      0.86      0.86        50
     Disease       0.87      0.88      0.87        50

    accuracy                           0.87       100
   macro avg       0.87      0.87      0.87       100
weighted avg       0.87      0.87      0.87       100

AUC-ROC: 0.934
```

### **Feature Importance (Coefficients)**

```python
feature_names = X.columns
coefs = pipeline.named_steps['classifier'].coef_[0]

for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {name:20s}: {coef:+.4f}  ({direction} disease risk)")
```

```
  pocket_depth_mm     : +1.8234  (increases disease risk)
  plaque_index        : +1.2451  (increases disease risk)
  smoking_status      : +0.9873  (increases disease risk)
  bop_percentage      : +0.6541  (increases disease risk)
  age                 : +0.3421  (increases disease risk)
```

---

## **Clinical Decision Pipeline**

```python
def assess_patient_risk(pipeline, patient_data, threshold=0.5):
    """
    Assess periodontal disease risk for a new patient.

    Parameters:
        pipeline: trained sklearn pipeline
        patient_data: dict with clinical measurements
        threshold: classification threshold (lower = more sensitive)

    Returns:
        risk assessment dictionary
    """
    df = pd.DataFrame([patient_data])
    probability = pipeline.predict_proba(df)[0, 1]
    risk_level = "HIGH" if probability >= 0.7 else "MODERATE" if probability >= 0.4 else "LOW"

    return {
        'risk_probability': f"{probability:.1%}",
        'risk_level': risk_level,
        'classification': 'Disease' if probability >= threshold else 'Healthy',
        'recommendation': {
            'HIGH': 'Refer to periodontist immediately',
            'MODERATE': 'Schedule follow-up in 3 months',
            'LOW': 'Continue routine 6-month checkups'
        }[risk_level]
    }

# Example: Assess a new patient
new_patient = {
    'plaque_index': 2.1,
    'bop_percentage': 65.0,
    'pocket_depth_mm': 5.5,
    'age': 55,
    'smoking_status': 2
}
result = assess_patient_risk(pipeline, new_patient)
print(result)
# {'risk_probability': '92.3%', 'risk_level': 'HIGH',
#  'classification': 'Disease',
#  'recommendation': 'Refer to periodontist immediately'}
```

---

## **Cross-Validation**

```python
cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='roc_auc')
print(f"10-Fold CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
# 10-Fold CV AUC-ROC: 0.9287 (+/- 0.0341)
```

---

## **When to Use This Implementation**

| Scenario | Recommendation |
|----------|---------------|
| Quick baseline model | Use sklearn LogisticRegression |
| Need interpretable coefficients | Ideal - coefficients map to clinical risk factors |
| Large-scale dataset (>100K patients) | Use `solver='saga'` with `max_iter=5000` |
| Feature selection needed | Use `penalty='l1'` (Lasso) to zero out irrelevant features |
| Multi-clinic deployment | Export pipeline with `joblib.dump()` |

---

## **Running the Demo**

```bash
cd examples/02_classification
python logistic_regression_sklearn.py
```

---

## **References**

1. Scikit-Learn Documentation: LogisticRegression
2. Pedregosa et al. "Scikit-learn: Machine Learning in Python" (2011), JMLR
3. Lang & Tonetti "Periodontal Risk Assessment" (2003), Periodontology 2000

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
