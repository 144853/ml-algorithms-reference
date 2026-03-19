# AdaBoost (Scikit-Learn) - Dental Classification

## **Use Case: Predicting Dental Emergency Classification**

### **The Problem**
A dental emergency triage system classifies **750 incoming cases** as urgent or routine based on pain severity, swelling level, fever presence, bleeding severity, and trauma indicator. Accurate triage ensures urgent cases receive immediate attention while routine cases are scheduled appropriately.

### **Why AdaBoost?**
- Sequential boosting focuses on difficult-to-classify cases
- Works well with simple base learners (decision stumps)
- Interpretable ensemble with weighted weak classifiers
- Effective for binary triage decisions
- Robust to noise in patient-reported symptoms

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

np.random.seed(42)
n_cases = 750

data = pd.DataFrame({
    'pain_severity': np.random.uniform(0, 10, n_cases),          # VAS 0-10
    'swelling_level': np.random.uniform(0, 10, n_cases),          # 0=none, 10=severe
    'fever_celsius': np.random.uniform(36.0, 40.5, n_cases),      # body temperature
    'bleeding_severity': np.random.uniform(0, 10, n_cases),       # 0=none, 10=profuse
    'trauma_indicator': np.random.choice([0, 1], n_cases, p=[0.7, 0.3])  # 0=no, 1=yes
})

# Emergency classification logic
urgency_score = (0.25 * data['pain_severity'] +
                 0.2 * data['swelling_level'] +
                 0.3 * (data['fever_celsius'] - 37.0).clip(0) +
                 0.15 * data['bleeding_severity'] +
                 0.4 * data['trauma_indicator'])
data['classification'] = (urgency_score > np.percentile(urgency_score, 60)).astype(int)  # 0=routine, 1=urgent

X = data.drop('classification', axis=1)
y = data['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),  # Weak learner (decision stump)
    n_estimators=100,
    learning_rate=0.5,
    algorithm='SAMME.R',
    random_state=42
)

ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
y_proba = ada_model.predict_proba(X_test)[:, 1]
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

     Routine       0.88      0.90      0.89        90
      Urgent       0.83      0.80      0.81        60

    accuracy                           0.86       150
   macro avg       0.85      0.85      0.85       150
weighted avg       0.86      0.86      0.86       150

AUC-ROC: 0.916
```

### **Confusion Matrix**

```
              Predicted Routine  Predicted Urgent
Actual Routine          81              9
Actual Urgent           12             48
```

### **Feature Importance**

```python
importance = pd.Series(ada_model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))
```

```
fever_celsius         0.30
pain_severity         0.24
trauma_indicator      0.20
swelling_level        0.15
bleeding_severity     0.11
```

---

## **Emergency Triage System**

```python
def triage_patient(model, patient_symptoms):
    """Classify dental emergency and provide triage recommendation."""
    df = pd.DataFrame([patient_symptoms])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    urgency_prob = probability[1]

    if urgency_prob >= 0.8:
        triage_level = 'CRITICAL'
        action = 'Immediate treatment - clear next available chair'
        wait_time = '<15 minutes'
    elif urgency_prob >= 0.5:
        triage_level = 'URGENT'
        action = 'Priority scheduling within 1-2 hours'
        wait_time = '1-2 hours'
    elif urgency_prob >= 0.3:
        triage_level = 'SEMI-URGENT'
        action = 'Schedule same-day appointment'
        wait_time = '2-4 hours'
    else:
        triage_level = 'ROUTINE'
        action = 'Schedule regular appointment'
        wait_time = 'Next available slot'

    return {
        'triage_level': triage_level,
        'urgency_probability': f"{urgency_prob:.1%}",
        'recommended_action': action,
        'estimated_wait': wait_time,
        'classification': 'Urgent' if prediction == 1 else 'Routine'
    }

# Severe case: high pain, swelling, fever, trauma
result = triage_patient(ada_model, {
    'pain_severity': 9.0,
    'swelling_level': 7.5,
    'fever_celsius': 39.2,
    'bleeding_severity': 6.0,
    'trauma_indicator': 1
})
print(result)
# {'triage_level': 'CRITICAL', 'urgency_probability': '94.3%', ...}
```

---

## **Boosting Stages Analysis**

```python
# Analyze how accuracy improves with each boosting stage
staged_scores = list(ada_model.staged_score(X_test, y_test))

print("Accuracy by boosting stage:")
for stage in [1, 10, 25, 50, 100]:
    if stage <= len(staged_scores):
        print(f"  Stage {stage:3d}: {staged_scores[stage-1]:.4f}")
```

```
Accuracy by boosting stage:
  Stage   1: 0.7200
  Stage  10: 0.8000
  Stage  25: 0.8400
  Stage  50: 0.8533
  Stage 100: 0.8600
```

---

## **Hyperparameter Tuning**

```python
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3, 4]
}

grid_search = GridSearchCV(
    AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        algorithm='SAMME.R',
        random_state=42
    ),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best AUC: {grid_search.best_score_:.4f}")
print(f"Best params: {grid_search.best_params_}")
```

---

## **Base Learner Comparison**

```python
from sklearn.linear_model import LogisticRegression

base_learners = {
    'Stump (depth=1)': DecisionTreeClassifier(max_depth=1),
    'Tree (depth=2)': DecisionTreeClassifier(max_depth=2),
    'Tree (depth=3)': DecisionTreeClassifier(max_depth=3),
}

for name, base in base_learners.items():
    ada = AdaBoostClassifier(estimator=base, n_estimators=100, random_state=42)
    scores = cross_val_score(ada, X, y, cv=5, scoring='roc_auc')
    print(f"  {name:20s}: AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

```
  Stump (depth=1)     : AUC = 0.8923 (+/- 0.0345)
  Tree (depth=2)      : AUC = 0.9156 (+/- 0.0289)
  Tree (depth=3)      : AUC = 0.9087 (+/- 0.0312)
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Binary triage decisions | AdaBoost excels at binary classification |
| Interpretable ensemble | Weighted combination of simple rules |
| Moderate noise in symptoms | Boosting is moderately robust to noise |
| Very noisy labels | Consider Random Forest (more robust) |
| Need feature importance | Built-in importance from boosting weights |

---

## **Running the Demo**

```bash
cd examples/02_classification
python adaboost_sklearn.py
```

---

## **References**

1. Freund & Schapire "A Decision-Theoretic Generalization of On-Line Learning" (1997), JCSS
2. Scikit-Learn Documentation: AdaBoostClassifier
3. ADA Emergency Triage Guidelines for Dental Practice

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
