# Random Forest - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of Random Forest for **insurance policy lapse prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Random Forest - Full Documentation](random_forest_numpy.md)**

---

## 🛡️ **Insurance Use Case: Policy Lapse Prediction**

Predict whether a policy will **lapse** or be **retained** based on:
- Premium-to-income ratio (0.02 - 0.35)
- Payment frequency (monthly=1, quarterly=2, semi-annual=3, annual=4)
- Policy age in years (0.5 - 25)
- Customer complaints in last 12 months (0 - 8)
- Agent engagement score (0 - 100)

### **Why Scikit-learn for Lapse Prediction?**
- **Feature importance**: Identify key drivers of lapse for retention campaigns
- **Parallel training**: `n_jobs=-1` for fast training on large policy portfolios
- **OOB estimation**: Out-of-bag error for model validation without hold-out set
- **Probability outputs**: Rank policies by lapse risk for targeted outreach

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Parallel processing**: Train multiple trees simultaneously

---

## 💻 **Quick Start**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Insurance policy lapse data
# Features: premium_income_ratio, payment_freq, policy_age, complaints, agent_score
X, y = load_policy_lapse_data()  # y: 0=retain, 1=lapse
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Create and train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=15,
    class_weight='balanced',
    n_jobs=-1,
    oob_score=True,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Lapse probability

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"OOB Score: {model.oob_score_:.4f}")
print(classification_report(y_test, y_pred, target_names=['Retain', 'Lapse']))
```

---

## 🔧 **Key Parameters for Lapse Prediction**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `n_estimators` | 100 | 200-500 for stable lapse probability estimates |
| `max_depth` | None | Set to 8-12 to prevent overfitting on policy features |
| `min_samples_leaf` | 1 | Set to 15-30 for statistically credible lapse segments |
| `class_weight` | None | `'balanced'` if lapse rate is below 15% |
| `max_features` | `'sqrt'` | Default works well for decorrelated trees |
| `oob_score` | False | Set to True for free validation score |

### **Insurance-Specific Workflow**

```python
import numpy as np

# Feature importance for retention strategy
feature_names = ['premium_income_ratio', 'payment_freq', 'policy_age', 'complaints', 'agent_score']
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.3f}")
# premium_income_ratio: 0.298  <- Top driver: affordability
# agent_score: 0.241           <- Agent relationship matters
# complaints: 0.198            <- Service quality
# policy_age: 0.156            <- New policies lapse more
# payment_freq: 0.107          <- Monthly payers lapse more

# Segment policies by lapse risk for targeted retention
lapse_probs = model.predict_proba(X_all)[:, 1]
high_risk = lapse_probs > 0.70    # Intensive retention campaign
medium_risk = (lapse_probs > 0.40) & (lapse_probs <= 0.70)  # Proactive outreach
low_risk = lapse_probs <= 0.40    # Standard service

print(f"High risk policies: {high_risk.sum()} ({high_risk.mean():.1%})")
print(f"Medium risk: {medium_risk.sum()} ({medium_risk.mean():.1%})")
print(f"Low risk: {low_risk.sum()} ({low_risk.mean():.1%})")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Lapse Prediction Performance ---
Accuracy:    86.4%
AUC-ROC:     0.913
OOB Score:   0.854

              precision  recall  f1-score  support
      Retain     0.90    0.91     0.90      1420
       Lapse     0.78    0.76     0.77       580

--- Retention Campaign Impact Estimate ---
High risk policies:    412 (identified 76% of actual lapsers)
Estimated saved:       198 policies ($594K annual premium)
Campaign cost:         $82K
Net benefit:           $512K
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/random_forest_sklearn.py`](../../02_classification/random_forest_sklearn.py)

Related:
- [Random Forest - NumPy (from scratch)](random_forest_numpy.md)
- [Random Forest - PyTorch](random_forest_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
