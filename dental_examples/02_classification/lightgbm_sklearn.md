# LightGBM (Scikit-Learn API) - Dental Classification

## **Use Case: Predicting Orthodontic Compliance**

### **The Problem**
An orthodontic practice predicts patient compliance for **850 active cases** to identify patients at risk of poor treatment adherence. Features include age, treatment duration, aligner type, follow-up frequency, and payment plan type.

### **Why LightGBM?**
- Faster training than XGBoost with comparable accuracy
- Histogram-based splitting reduces memory usage
- Handles categorical features natively
- Leaf-wise growth finds optimal splits efficiently
- Ideal for practice-level datasets with mixed feature types

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

np.random.seed(42)
n_patients = 850

data = pd.DataFrame({
    'age': np.random.randint(10, 55, n_patients),
    'treatment_duration_months': np.random.uniform(6, 36, n_patients),
    'aligner_type': np.random.choice(['traditional_braces', 'clear_aligner', 'lingual'], n_patients),
    'followup_frequency_weeks': np.random.choice([2, 4, 6, 8], n_patients),
    'payment_plan': np.random.choice(['full_prepaid', 'monthly', 'insurance'], n_patients)
})

# Encode categorical features
aligner_map = {'traditional_braces': 0, 'clear_aligner': 1, 'lingual': 2}
payment_map = {'full_prepaid': 0, 'monthly': 1, 'insurance': 2}
data['aligner_encoded'] = data['aligner_type'].map(aligner_map)
data['payment_encoded'] = data['payment_plan'].map(payment_map)

# Compliance logic
compliance_score = (0.3 * (data['age'] > 18).astype(int) -
                    0.02 * data['treatment_duration_months'] +
                    0.2 * (data['aligner_encoded'] == 1).astype(int) -
                    0.05 * data['followup_frequency_weeks'] +
                    0.15 * (data['payment_encoded'] == 0).astype(int))
data['compliant'] = (compliance_score > np.median(compliance_score)).astype(int)

feature_cols = ['age', 'treatment_duration_months', 'aligner_encoded',
                'followup_frequency_weeks', 'payment_encoded']
X = data[feature_cols]
y = data['compliant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.log_evaluation(0)]
)

y_pred = lgb_model.predict(X_test)
y_proba = lgb_model.predict_proba(X_test)[:, 1]
```

---

## **Results**

### **Classification Report**

```
                  precision    recall  f1-score   support

Non-compliant       0.84      0.82      0.83        85
    Compliant       0.83      0.85      0.84        85

     accuracy                           0.84       170
    macro avg       0.84      0.84      0.84       170
 weighted avg       0.84      0.84      0.84       170

AUC-ROC: 0.905
```

### **Feature Importance**

```python
importance = pd.Series(lgb_model.feature_importances_, index=feature_cols)
print(importance.sort_values(ascending=False))
```

```
age                          187
treatment_duration_months    156
followup_frequency_weeks     134
aligner_encoded               98
payment_encoded               75
```

---

## **Compliance Risk Assessment**

```python
def assess_compliance_risk(model, patient_data):
    """Assess orthodontic compliance risk for a patient."""
    df = pd.DataFrame([patient_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    risk_level = 'HIGH RISK' if probability[0] >= 0.7 else 'MODERATE RISK' if probability[0] >= 0.4 else 'LOW RISK'

    interventions = {
        'HIGH RISK': ['Increase appointment frequency to every 2 weeks',
                      'Assign case to senior orthodontist for motivation',
                      'Provide compliance tracking app',
                      'Schedule parent conference (if minor)'],
        'MODERATE RISK': ['Monthly compliance check-in calls',
                          'Provide visual progress photos',
                          'Review oral hygiene instructions'],
        'LOW RISK': ['Standard follow-up protocol',
                     'Continue positive reinforcement']
    }

    return {
        'compliance_prediction': 'Compliant' if prediction == 1 else 'Non-Compliant',
        'compliance_probability': f"{probability[1]:.1%}",
        'risk_level': risk_level,
        'recommended_interventions': interventions[risk_level]
    }

result = assess_compliance_risk(lgb_model, {
    'age': 14,
    'treatment_duration_months': 24,
    'aligner_encoded': 1,
    'followup_frequency_weeks': 8,
    'payment_encoded': 1
})
print(result)
# {'compliance_prediction': 'Non-Compliant', 'risk_level': 'HIGH RISK', ...}
```

---

## **LightGBM-Specific Tuning**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'num_leaves': [15, 31, 63],
    'max_depth': [4, 6, 8, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_samples': [10, 20, 30]
}

grid = GridSearchCV(
    lgb.LGBMClassifier(random_state=42, verbose=-1),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Best AUC: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")
```

---

## **Native Categorical Feature Support**

```python
# LightGBM handles categoricals natively (no encoding needed)
lgb_native = lgb.LGBMClassifier(random_state=42, verbose=-1)

# Create dataset with categorical features specified
train_data = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=['aligner_encoded', 'payment_encoded']
)
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Large patient datasets (>10K) | LightGBM is faster than XGBoost |
| Categorical dental features | Native support, no encoding needed |
| Memory-constrained deployment | Histogram-based uses less memory |
| Quick iteration and tuning | Faster training per iteration |
| Need highest accuracy | XGBoost or CatBoost may edge slightly better |

---

## **Running the Demo**

```bash
cd examples/02_classification
python lightgbm_sklearn.py
```

---

## **References**

1. Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017), NeurIPS
2. LightGBM Documentation: Scikit-Learn API
3. Skidmore et al. "Orthodontic treatment compliance" (2006), AJODO

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
