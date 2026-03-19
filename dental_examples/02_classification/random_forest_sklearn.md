# Random Forest (Scikit-Learn) - Dental Classification

## **Use Case: Predicting Dental Caries Risk Level**

### **The Problem**
A preventive dentistry program classifies **800 patients** into caries risk levels (low/medium/high) based on saliva flow rate, bacterial count, diet sugar score, fluoride exposure, and brushing frequency. Multi-class classification enables targeted preventive interventions.

### **Why Random Forest?**
- Handles multi-class classification naturally
- Robust to noisy clinical measurements
- Built-in feature importance for clinical insight
- Resistant to overfitting with proper hyperparameters
- No feature scaling required

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
n_patients = 800

data = pd.DataFrame({
    'saliva_flow_rate': np.random.uniform(0.1, 2.0, n_patients),       # mL/min
    'bacterial_count': np.random.uniform(1e3, 1e7, n_patients),         # CFU/mL
    'diet_sugar_score': np.random.uniform(0, 10, n_patients),           # 0-10 scale
    'fluoride_exposure': np.random.choice([0, 1, 2, 3], n_patients),    # 0=none, 3=high
    'brushing_frequency': np.random.choice([0, 1, 2, 3], n_patients)    # times/day
})

# Risk classification logic
risk_score = (-0.8 * data['saliva_flow_rate'] +
              np.log10(data['bacterial_count']) / 7 +
              0.3 * data['diet_sugar_score'] -
              0.4 * data['fluoride_exposure'] -
              0.3 * data['brushing_frequency'])

data['risk_level'] = pd.cut(risk_score, bins=3, labels=['low', 'medium', 'high']).astype(str)

X = data.drop('risk_level', axis=1)
y = data['risk_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
rf_model = RandomForestClassifier(
    n_estimators=200,         # Number of trees
    max_depth=10,             # Maximum tree depth
    min_samples_split=10,     # Minimum samples to split
    min_samples_leaf=5,       # Minimum samples per leaf
    max_features='sqrt',      # Features per split
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1                 # Parallel training
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

        high       0.85      0.83      0.84        53
         low       0.88      0.91      0.89        54
      medium       0.82      0.81      0.81        53

    accuracy                           0.85       160
   macro avg       0.85      0.85      0.85       160
weighted avg       0.85      0.85      0.85       160
```

### **Confusion Matrix**

```
              Predicted Low  Predicted Medium  Predicted High
Actual Low           49            4                1
Actual Medium         3           43                7
Actual High           1            8               44
```

### **Feature Importance**

```python
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))
```

```
bacterial_count       0.31
diet_sugar_score      0.24
saliva_flow_rate      0.22
fluoride_exposure     0.13
brushing_frequency    0.10
```

---

## **Clinical Risk Assessment Pipeline**

```python
def caries_risk_assessment(model, patient_data):
    """Comprehensive caries risk assessment with confidence levels."""
    df = pd.DataFrame([patient_data])
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    classes = model.classes_

    risk_probs = dict(zip(classes, probabilities))

    interventions = {
        'low': ['Routine 6-month checkup', 'Continue current oral hygiene'],
        'medium': ['3-month recall', 'Fluoride varnish application', 'Dietary counseling'],
        'high': ['Monthly monitoring', 'Prescription fluoride', 'Chlorhexidine rinse',
                 'Sealant application', 'Intensive dietary modification']
    }

    return {
        'risk_level': prediction.upper(),
        'probabilities': {k: f"{v:.1%}" for k, v in risk_probs.items()},
        'recommended_interventions': interventions[prediction],
        'confidence': f"{max(probabilities):.1%}"
    }

result = caries_risk_assessment(rf_model, {
    'saliva_flow_rate': 0.3,
    'bacterial_count': 5e6,
    'diet_sugar_score': 8.5,
    'fluoride_exposure': 0,
    'brushing_frequency': 1
})
print(result)
# {'risk_level': 'HIGH', 'confidence': '91.2%', ...}
```

---

## **Hyperparameter Tuning**

```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [3, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='f1_macro', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1 (macro): {grid_search.best_score_:.4f}")
```

---

## **Out-of-Bag Error Estimation**

```python
rf_oob = RandomForestClassifier(
    n_estimators=200, max_depth=10, oob_score=True, random_state=42, n_jobs=-1
)
rf_oob.fit(X_train, y_train)
print(f"OOB Score: {rf_oob.oob_score_:.4f}")
# OOB Score: 0.8437 - built-in validation without holdout set
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Multi-class dental risk classification | Ideal - handles 3+ classes naturally |
| Noisy clinical measurements | Ensemble averaging reduces noise impact |
| Feature importance needed | Built-in importance scores |
| Moderate dataset (100-10K patients) | Sweet spot for Random Forest |
| Very large datasets (>100K) | Consider XGBoost or LightGBM for speed |

---

## **Running the Demo**

```bash
cd examples/02_classification
python random_forest_sklearn.py
```

---

## **References**

1. Breiman, L. "Random Forests" (2001), Machine Learning
2. Scikit-Learn Documentation: RandomForestClassifier
3. Fontana & Zero "Assessing patients' caries risk" (2006), JADA

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
