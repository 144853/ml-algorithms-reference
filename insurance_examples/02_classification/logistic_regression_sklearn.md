# Logistic Regression - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of Logistic Regression for **insurance claim fraud detection**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Logistic Regression - Full Documentation](logistic_regression_numpy.md)**

---

## 🛡️ **Insurance Use Case: Claim Fraud Detection**

Predict whether an insurance claim is **fraudulent** or **legitimate** based on:
- Claim amount, time to report, witness count, police report filed, policy age

### **Why Scikit-learn for Insurance Fraud?**
- **Regulatory compliance**: Easy to export coefficients for audit trails
- **Pipeline integration**: Combine with StandardScaler, SMOTE for imbalanced fraud data
- **Cross-validation**: Built-in stratified CV preserves fraud/legitimate ratio
- **Probability calibration**: `predict_proba()` provides SIU risk scores

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Multiple solvers**: Auto-selects best solver based on data characteristics

---

## 💻 **Quick Start**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Insurance claim fraud data
# Features: claim_amount, time_to_report, witness_count, police_report, policy_age
X, y = load_claim_fraud_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale features (important for claim amounts vs binary features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model with class weight balancing for imbalanced fraud data
model = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Fraud probability for SIU

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
```

---

## 🔧 **Key Parameters for Insurance Fraud**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `C` | 1.0 | Regularization strength (inverse). Lower C = stronger regularization |
| `class_weight` | None | Set to `'balanced'` for 90/10 legitimate/fraud split |
| `solver` | `'lbfgs'` | Use `'saga'` for large claim datasets (>100K claims) |
| `max_iter` | 100 | Increase to 1000 for convergence on scaled insurance data |
| `penalty` | `'l2'` | Use `'l1'` for feature selection (identify top fraud indicators) |

### **Insurance-Specific Workflow**

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Production pipeline for fraud scoring
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight='balanced'))
])

# Tune regularization strength
param_grid = {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

# Extract fraud indicators (odds ratios)
best_model = grid_search.best_estimator_.named_steps['classifier']
feature_names = ['claim_amount', 'time_to_report', 'witness_count', 'police_report', 'policy_age']
for name, coef in zip(feature_names, best_model.coef_[0]):
    print(f"{name}: coefficient={coef:.3f}, odds_ratio={np.exp(coef):.3f}")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Fraud Detection Performance ---
Accuracy:    88.1%
AUC-ROC:     0.926
Precision:   74.6%  (of flagged claims, 74.6% were actual fraud)
Recall:      71.2%  (of actual fraud, caught 71.2%)

--- Top Fraud Indicators ---
time_to_report:  odds_ratio = 1.053  (each extra day -> 5.3% higher fraud odds)
claim_amount:    odds_ratio = 1.039  (each $1K -> 3.9% higher fraud odds)
police_report:   odds_ratio = 0.410  (filing report -> 59% lower fraud odds)
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/logistic_regression_sklearn.py`](../../02_classification/logistic_regression_sklearn.py)

Related:
- [Logistic Regression - NumPy (from scratch)](logistic_regression_numpy.md)
- [Logistic Regression - PyTorch](logistic_regression_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
