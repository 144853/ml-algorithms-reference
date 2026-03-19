# LightGBM - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of LightGBM for **insurance customer churn prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[LightGBM - Full Documentation](lightgbm_numpy.md)**

---

## 🛡️ **Insurance Use Case: Customer Churn Prediction**

Predict customer churn -- **churn** or **retain** -- based on:
- Premium change percentage at renewal (-30% to +50%)
- Claim experience (number of claims in past 3 years: 0 - 8)
- Net Promoter Score (NPS: -100 to 100)
- Policy bundle count (number of policies held: 1 - 6)
- Competitor quote differential (% cheaper than current: -20% to +40%)

### **Why Scikit-learn for Churn Prediction?**
- **LightGBM speed**: Histogram-based splitting for fast training on large portfolios
- **Categorical support**: Native handling of policy type categories
- **Feature importance**: SHAP-compatible for explaining churn drivers
- **Leaf-wise growth**: Better accuracy with fewer trees than XGBoost

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Fast training**: LightGBM is often 5-10x faster than XGBoost

---

## 💻 **Quick Start**

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Insurance customer churn data
# Features: premium_change_pct, claim_count, nps_score, bundle_count, competitor_diff
X, y = load_churn_data()  # y: 0=retain, 1=churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Create and train model
model = LGBMClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,  # Handle churn class imbalance
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(20)])

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Retain', 'Churn']))
```

---

## 🔧 **Key Parameters for Churn Prediction**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `n_estimators` | 100 | 200-500 with early stopping |
| `num_leaves` | 31 | 20-50 for customer behavior patterns |
| `max_depth` | -1 | Set to 6-8 to prevent overfitting |
| `learning_rate` | 0.1 | 0.03-0.08 for better generalization |
| `is_unbalance` | False | True if churn rate < 20% |
| `min_child_samples` | 20 | 30-50 for stable churn segments |

### **Insurance-Specific Workflow**

```python
import numpy as np

# Feature importance for retention strategy
feature_names = ['premium_change_pct', 'claim_count', 'nps_score', 'bundle_count', 'competitor_diff']
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp}")
# competitor_diff: 312       <- Competitor pricing is #1 churn driver
# premium_change_pct: 278    <- Rate increases drive churn
# nps_score: 198             <- Customer satisfaction matters
# bundle_count: 145          <- Multi-policy = stickier
# claim_count: 67            <- Claims experience less predictive

# Retention campaign targeting
churn_probs = model.predict_proba(X_all_renewals)[:, 1]

# Segment by churn risk and premium value
high_risk_high_value = (churn_probs > 0.60) & (premium_amounts > 3000)
high_risk_standard = (churn_probs > 0.60) & (premium_amounts <= 3000)

print(f"\n--- Retention Targeting ---")
print(f"VIP retention (high risk, high value): {high_risk_high_value.sum()} customers")
print(f"  -> Offer: Personal agent call + 10% loyalty discount")
print(f"Standard retention (high risk): {high_risk_standard.sum()} customers")
print(f"  -> Offer: Auto-email with bundle discount")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Customer Churn Prediction ---
Accuracy:    85.9%
AUC-ROC:     0.921

              precision  recall  f1-score  support
      Retain     0.89    0.92     0.90      1380
       Churn     0.79    0.73     0.76       520

--- Retention Campaign ROI ---
Customers targeted: 412
Churn prevented (est.): 185
Retained annual premium: $832K
Campaign cost: $45K
Net retention value: $787K
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/lightgbm_sklearn.py`](../../02_classification/lightgbm_sklearn.py)

Related:
- [LightGBM - NumPy (from scratch)](lightgbm_numpy.md)
- [LightGBM - PyTorch](lightgbm_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
