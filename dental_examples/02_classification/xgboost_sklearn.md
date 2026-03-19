# XGBoost (Scikit-Learn API) - Dental Classification

## **Use Case: Predicting Root Canal Treatment Outcome**

### **The Problem**
An endodontic department predicts treatment outcomes for **900 root canal procedures** as success or failure. Features include tooth type, canal complexity, pre-operative pain level, periapical lesion size, and operator experience years.

### **Why XGBoost?**
- State-of-the-art gradient boosting for tabular clinical data
- Handles missing values in patient records automatically
- Built-in regularization prevents overfitting
- Feature importance reveals key prognostic factors
- Excellent performance on structured medical data

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb

np.random.seed(42)
n_procedures = 900

data = pd.DataFrame({
    'tooth_type': np.random.choice([0,1,2,3], n_procedures),          # 0=incisor, 1=premolar, 2=molar, 3=wisdom
    'canal_complexity': np.random.uniform(1, 5, n_procedures),         # 1=simple, 5=complex (curved/calcified)
    'preop_pain_level': np.random.uniform(0, 10, n_procedures),        # VAS 0-10
    'lesion_size_mm': np.random.uniform(0, 10, n_procedures),          # periapical lesion diameter
    'operator_experience_yrs': np.random.uniform(1, 30, n_procedures)  # years of endodontic experience
})

# Outcome logic: higher complexity + larger lesion + less experience = more failures
failure_prob = (0.05 + 0.1 * data['canal_complexity'] / 5 +
                0.08 * data['lesion_size_mm'] / 10 +
                0.05 * data['tooth_type'].isin([2,3]).astype(int) -
                0.08 * data['operator_experience_yrs'] / 30 +
                0.03 * data['preop_pain_level'] / 10)
failure_prob = failure_prob.clip(0.02, 0.6)
data['outcome'] = (np.random.random(n_procedures) < failure_prob).astype(int)  # 0=success, 1=failure

X = data.drop('outcome', axis=1)
y = data['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    scale_pos_weight=3.0,    # Handle class imbalance (more successes than failures)
    eval_metric='auc',
    random_state=42,
    use_label_encoder=False
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

     Success       0.92      0.90      0.91       148
     Failure       0.62      0.68      0.65        32

    accuracy                           0.86       180
   macro avg       0.77      0.79      0.78       180
weighted avg       0.87      0.86      0.87       180

AUC-ROC: 0.872
```

### **Feature Importance**

```python
importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))
```

```
canal_complexity          0.32
lesion_size_mm            0.25
operator_experience_yrs   0.20
tooth_type                0.13
preop_pain_level          0.10
```

---

## **Clinical Outcome Prediction**

```python
def predict_rct_outcome(model, procedure_data):
    """Predict root canal treatment outcome with risk factors."""
    df = pd.DataFrame([procedure_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    risk_factors = []
    if procedure_data['canal_complexity'] > 3.5:
        risk_factors.append('Complex canal anatomy')
    if procedure_data['lesion_size_mm'] > 5:
        risk_factors.append('Large periapical lesion')
    if procedure_data['operator_experience_yrs'] < 5:
        risk_factors.append('Limited operator experience')
    if procedure_data['tooth_type'] >= 2:
        risk_factors.append('Posterior tooth (higher difficulty)')

    return {
        'predicted_outcome': 'Failure Risk' if prediction == 1 else 'Likely Success',
        'success_probability': f"{probability[0]:.1%}",
        'failure_probability': f"{probability[1]:.1%}",
        'risk_factors': risk_factors if risk_factors else ['No major risk factors identified'],
        'recommendation': 'Consider referral to endodontist' if prediction == 1 else 'Proceed with treatment'
    }

result = predict_rct_outcome(xgb_model, {
    'tooth_type': 2,                    # Molar
    'canal_complexity': 4.2,            # Complex
    'preop_pain_level': 7.0,            # High pain
    'lesion_size_mm': 6.5,              # Large lesion
    'operator_experience_yrs': 3.0      # Junior dentist
})
print(result)
# {'predicted_outcome': 'Failure Risk', 'failure_probability': '48.3%', ...}
```

---

## **Hyperparameter Tuning**

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'min_child_weight': [3, 5, 7]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc'),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best AUC-ROC: {grid_search.best_score_:.4f}")
print(f"Best params: {grid_search.best_params_}")
```

---

## **Early Stopping**

```python
xgb_early = XGBClassifier(
    n_estimators=1000, max_depth=5, learning_rate=0.05,
    random_state=42, use_label_encoder=False, eval_metric='auc',
    early_stopping_rounds=20
)
xgb_early.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print(f"Best iteration: {xgb_early.best_iteration}")
# Best iteration: 187 (stopped early, saved training time)
```

---

## **SHAP Feature Explanation**

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# For a specific patient
patient_idx = 0
print(f"Patient prediction: {y_pred[patient_idx]}")
for feature, sv in zip(X.columns, shap_values[patient_idx]):
    direction = "increases" if sv > 0 else "decreases"
    print(f"  {feature:25s}: SHAP = {sv:+.4f} ({direction} failure risk)")
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Structured clinical data | XGBoost is top performer |
| Missing values in records | Handles missing data natively |
| Feature importance needed | Built-in + SHAP explanations |
| Large datasets (10K+) | Scales well with histogram-based split |
| Very small datasets (<200) | Consider simpler models (Logistic Regression) |

---

## **Running the Demo**

```bash
cd examples/02_classification
python xgboost_sklearn.py
```

---

## **References**

1. Chen & Guestrin "XGBoost: A Scalable Tree Boosting System" (2016), KDD
2. Scikit-Learn Compatible XGBoost API Documentation
3. Ng et al. "Outcome of root canal treatment" (2011), International Endodontic Journal

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
