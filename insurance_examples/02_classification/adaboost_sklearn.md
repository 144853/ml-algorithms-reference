# AdaBoost - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of AdaBoost for **insurance application approval prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[AdaBoost - Full Documentation](adaboost_numpy.md)**

---

## 🛡️ **Insurance Use Case: Application Approval Prediction**

Predict insurance application outcome -- **approved** or **rejected** -- based on:
- Credit score (300 - 850)
- Claims history count (previous claims in past 5 years: 0 - 12)
- Coverage gap in months (time without insurance: 0 - 36)
- Income verification status (0 = unverified, 1 = partially verified, 2 = fully verified)
- Property inspection score (1 = poor, 2 = fair, 3 = good, 4 = excellent)

### **Why Scikit-learn for Application Approval?**
- **Weak learner boosting**: Combines simple decision stumps into strong approval model
- **Sample weighting**: Focuses on hard-to-classify borderline applications
- **Interpretable base learners**: Each stump is a simple underwriting rule
- **Resistant to overfitting**: With proper number of estimators and learning rate

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Flexible base estimators**: Decision stumps, shallow trees, or other classifiers

---

## 💻 **Quick Start**

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Insurance application data
# Features: credit_score, claims_history, coverage_gap, income_verified, property_score
X, y = load_application_data()  # y: 0=rejected, 1=approved
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Create and train model
base_estimator = DecisionTreeClassifier(max_depth=2)  # Weak learner: shallow tree
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=200,
    learning_rate=0.1,
    algorithm='SAMME.R',
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
```

---

## 🔧 **Key Parameters for Application Approval**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `n_estimators` | 50 | 100-300 for stable approval decisions |
| `learning_rate` | 1.0 | 0.05-0.2 for gradual boosting |
| `estimator` | DecisionTree(depth=1) | Depth 2-3 for feature interaction capture |
| `algorithm` | `'SAMME.R'` | SAMME.R for probability-based boosting |

### **Insurance-Specific Workflow**

```python
import numpy as np

# Feature importance for underwriting guidelines
feature_names = ['credit_score', 'claims_history', 'coverage_gap', 'income_verified', 'property_score']
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.3f}")
# credit_score: 0.312       <- Top approval factor
# claims_history: 0.248     <- Claims frequency matters
# coverage_gap: 0.195       <- Gaps indicate higher risk
# property_score: 0.138     <- Property condition
# income_verified: 0.107    <- Verification status

# Staged decision making with boosting rounds
approval_probs = model.predict_proba(X_test)[:, 1]

# Decision zones
auto_approve = approval_probs > 0.85
auto_reject = approval_probs < 0.20
manual_review = ~auto_approve & ~auto_reject

print(f"\n--- Application Routing ---")
print(f"Auto-approve: {auto_approve.sum()} ({auto_approve.mean():.1%})")
print(f"Auto-reject: {auto_reject.sum()} ({auto_reject.mean():.1%})")
print(f"Manual review: {manual_review.sum()} ({manual_review.mean():.1%})")

# Analyze individual estimator contributions
print(f"\n--- Boosting Analysis ---")
print(f"Number of estimators used: {len(model.estimators_)}")
print(f"Estimator weights range: [{min(model.estimator_weights_):.3f}, {max(model.estimator_weights_):.3f}]")
for i in range(3):
    tree = model.estimators_[i]
    feat_idx = tree.tree_.feature[0]
    threshold = tree.tree_.threshold[0]
    weight = model.estimator_weights_[i]
    print(f"  Stump {i}: if {feature_names[feat_idx]} <= {threshold:.1f} (weight: {weight:.3f})")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Application Approval Prediction ---
Accuracy:    84.5%
AUC-ROC:     0.908

              precision  recall  f1-score  support
    Rejected     0.81    0.78     0.79      420
    Approved     0.87    0.89     0.88      680

--- Application Processing Impact ---
Auto-approved (>85% confidence): 48% of applications
Auto-rejected (<20% confidence): 12% of applications
Manual review needed: 40% of applications
Processing time reduction: 55%
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/adaboost_sklearn.py`](../../02_classification/adaboost_sklearn.py)

Related:
- [AdaBoost - NumPy (from scratch)](adaboost_numpy.md)
- [AdaBoost - PyTorch](adaboost_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
