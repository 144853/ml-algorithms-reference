# XGBoost - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of XGBoost for **subrogation recovery prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[XGBoost - Full Documentation](xgboost_numpy.md)**

---

## 🛡️ **Insurance Use Case: Subrogation Recovery Prediction**

Predict subrogation recovery potential -- **recoverable** or **non-recoverable** -- based on:
- Fault determination percentage (0 - 100)
- Third-party insurer rating (1 = top-tier, 2 = mid-tier, 3 = small/unknown)
- Claim type code (1 = auto collision, 2 = property damage, 3 = liability, 4 = medical)
- Jurisdiction code (1-50 state codes)
- Evidence quality score (1 = poor, 2 = fair, 3 = good, 4 = excellent)

### **Why Scikit-learn for Subrogation?**
- **XGBoost API**: Compatible with sklearn pipelines and cross-validation
- **Feature importance**: Identify which factors predict successful recovery
- **Early stopping**: Prevent overfitting on imbalanced recovery outcomes
- **Missing value handling**: Native support for incomplete claim records

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **GPU support**: XGBoost offers `tree_method='gpu_hist'` for acceleration

---

## 💻 **Quick Start**

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Insurance subrogation data
# Features: fault_pct, third_party_rating, claim_type, jurisdiction, evidence_quality
X, y = load_subrogation_data()  # y: 0=non-recoverable, 1=recoverable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Create and train model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2.5,  # Recoverable claims are rarer
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Non-Recoverable', 'Recoverable']))
```

---

## 🔧 **Key Parameters for Subrogation**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `n_estimators` | 100 | 200-500 with early stopping |
| `max_depth` | 6 | 4-8 to capture fault/evidence interactions |
| `learning_rate` | 0.3 | 0.05-0.1 for better generalization |
| `scale_pos_weight` | 1 | Set to ratio of non-recoverable/recoverable |
| `subsample` | 1.0 | 0.7-0.9 for regularization |
| `colsample_bytree` | 1.0 | 0.7-0.9 to reduce feature correlation |

### **Insurance-Specific Workflow**

```python
import numpy as np

# Feature importance for subrogation strategy
feature_names = ['fault_pct', 'third_party_rating', 'claim_type', 'jurisdiction', 'evidence_quality']
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.3f}")
# fault_pct: 0.342         <- Clear fault -> better recovery
# evidence_quality: 0.268  <- Strong evidence crucial
# third_party_rating: 0.178
# jurisdiction: 0.128      <- Some states favor subrogation
# claim_type: 0.084

# Prioritize subrogation efforts by recovery probability
recovery_probs = model.predict_proba(X_all_open)[:, 1]
high_recovery = recovery_probs > 0.75
medium_recovery = (recovery_probs > 0.45) & (recovery_probs <= 0.75)

print(f"\nSubrogation Priority Queue:")
print(f"  High priority (>75%): {high_recovery.sum()} claims, est. recovery: ${high_recovery_amount:,.0f}")
print(f"  Medium priority: {medium_recovery.sum()} claims")
print(f"  Low priority (skip): {(~high_recovery & ~medium_recovery).sum()} claims")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Subrogation Recovery Prediction ---
Accuracy:    87.6%
AUC-ROC:     0.934

                    precision  recall  f1-score  support
 Non-Recoverable      0.90    0.92     0.91      680
     Recoverable      0.83    0.79     0.81      320

--- Subrogation ROI ---
Claims pursued (>75% recovery prob): 248
Successful recoveries: 196 (79% hit rate)
Total recovered: $4.2M
Pursuit costs: $310K
Net benefit: $3.89M
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/xgboost_sklearn.py`](../../02_classification/xgboost_sklearn.py)

Related:
- [XGBoost - NumPy (from scratch)](xgboost_numpy.md)
- [XGBoost - PyTorch](xgboost_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
