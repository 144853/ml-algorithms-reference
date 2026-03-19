# CatBoost (Scikit-Learn API) - Dental Classification

## **Use Case: Classifying Dental Insurance Claim Approval**

### **The Problem**
A dental insurance company classifies **1200 claims** as approved or denied based on procedure code, policy tier, pre-authorization status, claim amount, and provider network status. Automated classification speeds up claims processing and reduces human review burden.

### **Why CatBoost?**
- Handles categorical features natively without encoding
- Ordered boosting prevents target leakage
- Robust to overfitting with built-in regularization
- Excellent out-of-the-box performance with minimal tuning
- Handles mixed categorical/numerical dental claim data

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier, Pool

np.random.seed(42)
n_claims = 1200

data = pd.DataFrame({
    'procedure_code': np.random.choice(['D0120', 'D0274', 'D1110', 'D2750',
                                         'D3310', 'D4341', 'D7210', 'D8080'], n_claims),
    'policy_tier': np.random.choice(['basic', 'standard', 'premium', 'comprehensive'], n_claims),
    'pre_authorization': np.random.choice([0, 1], n_claims, p=[0.4, 0.6]),
    'claim_amount': np.random.uniform(50, 5000, n_claims),
    'provider_network': np.random.choice(['in_network', 'out_of_network', 'preferred'], n_claims)
})

# Approval logic
approval_score = (
    0.3 * data['pre_authorization'] +
    0.2 * data['policy_tier'].map({'basic': 0, 'standard': 1, 'premium': 2, 'comprehensive': 3}) / 3 -
    0.15 * data['claim_amount'] / 5000 +
    0.25 * data['provider_network'].map({'out_of_network': 0, 'in_network': 1, 'preferred': 2}) / 2 +
    0.1 * data['procedure_code'].isin(['D0120', 'D0274', 'D1110']).astype(int)
)
data['approved'] = (approval_score > np.median(approval_score)).astype(int)

X = data.drop('approved', axis=1)
y = data['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
cat_features = ['procedure_code', 'policy_tier', 'provider_network']

catboost_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    border_count=128,
    auto_class_weights='Balanced',
    cat_features=cat_features,
    eval_metric='AUC',
    random_seed=42,
    verbose=0
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

catboost_model.fit(train_pool, eval_set=test_pool, verbose=0)

y_pred = catboost_model.predict(X_test)
y_proba = catboost_model.predict_proba(X_test)[:, 1]
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

      Denied       0.87      0.85      0.86       120
    Approved       0.86      0.88      0.87       120

    accuracy                           0.86       240
   macro avg       0.86      0.86      0.86       240
weighted avg       0.86      0.86      0.86       240

AUC-ROC: 0.928
```

### **Feature Importance**

```python
importance = pd.Series(catboost_model.feature_importances_,
                       index=catboost_model.feature_names_)
print(importance.sort_values(ascending=False))
```

```
claim_amount          28.4
provider_network      22.1
policy_tier           19.8
pre_authorization     17.3
procedure_code        12.4
```

---

## **Claims Processing Pipeline**

```python
def process_claim(model, claim_data):
    """Automated dental insurance claim classification."""
    df = pd.DataFrame([claim_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    review_needed = 0.35 <= probability[1] <= 0.65

    return {
        'decision': 'APPROVED' if prediction == 1 else 'DENIED',
        'approval_probability': f"{probability[1]:.1%}",
        'denial_probability': f"{probability[0]:.1%}",
        'confidence': f"{max(probability):.1%}",
        'requires_human_review': review_needed,
        'processing_note': 'Auto-processed' if not review_needed else 'Flagged for manual review'
    }

result = process_claim(catboost_model, {
    'procedure_code': 'D3310',          # Root canal
    'policy_tier': 'basic',
    'pre_authorization': 0,
    'claim_amount': 1500.00,
    'provider_network': 'out_of_network'
})
print(result)
# {'decision': 'DENIED', 'approval_probability': '18.7%', ...}
```

---

## **CatBoost-Specific Features**

### **Feature Interaction Analysis**

```python
# Get feature interaction strengths
interactions = catboost_model.get_feature_importance(type='Interaction')
print("Top feature interactions:")
for pair in interactions[:5]:
    print(f"  Features {int(pair[0])} x {int(pair[1])}: {pair[2]:.4f}")
```

### **SHAP Values**

```python
shap_values = catboost_model.get_feature_importance(
    data=test_pool, type='ShapValues'
)
# Each row: SHAP values per feature + expected value
print(f"SHAP matrix shape: {shap_values.shape}")
```

---

## **Hyperparameter Tuning**

```python
from catboost import cv

params = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0
}

cv_results = cv(
    Pool(X_train, y_train, cat_features=cat_features),
    params,
    fold_count=5,
    stratified=True,
    verbose=False
)
print(f"Best AUC: {cv_results['test-AUC-mean'].max():.4f}")
print(f"Best iteration: {cv_results['test-AUC-mean'].argmax()}")
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Heavy categorical data | CatBoost excels (no encoding needed) |
| Minimal hyperparameter tuning | Best out-of-the-box defaults |
| Target leakage concerns | Ordered boosting prevents leakage |
| Claims processing automation | Fast inference, handles mixed types |
| Need GPU training | CatBoost supports GPU natively |

---

## **Running the Demo**

```bash
cd examples/02_classification
python catboost_sklearn.py
```

---

## **References**

1. Prokhorenkova et al. "CatBoost: unbiased boosting with categorical features" (2018), NeurIPS
2. CatBoost Documentation
3. ADA Procedure Code Reference Guide

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
