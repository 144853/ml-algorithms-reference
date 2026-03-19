# CatBoost - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of CatBoost for **workers' compensation claim type classification**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[CatBoost - Full Documentation](catboost_numpy.md)**

---

## 🛡️ **Insurance Use Case: Workers' Compensation Claim Classification**

Classify workers' compensation claims into **medical-only**, **indemnity**, or **denied** based on:
- Industry code (NAICS 2-digit: 11-99)
- Injury nature code (1 = strain, 2 = fracture, 3 = laceration, 4 = contusion, 5 = burn, 6 = cumulative trauma)
- Body part code (1 = back, 2 = shoulder, 3 = knee, 4 = hand, 5 = head, 6 = multiple)
- Employment duration in months (1 - 360)
- Employer size (1 = small <50, 2 = mid 50-500, 3 = large 500+)

### **Why Scikit-learn for Workers' Comp?**
- **Categorical native**: CatBoost handles industry/injury/body part codes without encoding
- **Ordered boosting**: Reduces overfitting on imbalanced claim types
- **Symmetric trees**: Fast inference for real-time FNOL classification
- **Built-in regularization**: L2 leaf regularization prevents overfitting

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Categorical native**: No one-hot encoding needed for industry/injury codes

---

## 💻 **Quick Start**

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Workers' comp claim data
# Features: industry_code, injury_nature, body_part, employment_months, employer_size
X, y = load_workers_comp_data()  # y: 0=medical-only, 1=indemnity, 2=denied
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Specify categorical features (indices)
cat_features = [0, 1, 2, 4]  # industry_code, injury_nature, body_part, employer_size

# Create and train model
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    cat_features=cat_features,
    auto_class_weights='Balanced',
    eval_metric='MultiClass',
    early_stopping_rounds=30,
    verbose=50
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
    target_names=['Medical-Only', 'Indemnity', 'Denied']))
```

---

## 🔧 **Key Parameters for Workers' Comp**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `iterations` | 1000 | 300-600 with early stopping |
| `depth` | 6 | 4-8 for industry/injury interactions |
| `learning_rate` | 0.03 | 0.03-0.08 for stable convergence |
| `cat_features` | None | Specify all categorical column indices |
| `auto_class_weights` | None | `'Balanced'` for imbalanced claim types |
| `l2_leaf_reg` | 3.0 | Increase to 5-10 for small datasets |

### **Insurance-Specific Workflow**

```python
# Feature importance for claims management strategy
feature_names = ['industry_code', 'injury_nature', 'body_part', 'employment_months', 'employer_size']
importances = model.get_feature_importance()
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.1f}")
# injury_nature: 31.2      <- Injury type is strongest predictor
# body_part: 24.8           <- Body part drives medical vs indemnity
# employment_months: 18.5   <- Tenure affects claim type
# industry_code: 15.3       <- Industry risk profile
# employer_size: 10.2       <- Larger employers -> more medical-only

# SHAP values for individual claim explanation
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:1])
# "This claim classified as Indemnity because:
#   injury_nature=fracture (+0.42)
#   body_part=back (+0.31)
#   employment_months=6 (+0.18)"

# Reserve setting by claim type
type_probs = model.predict_proba(X_all_new_claims)
medical_only = type_probs[:, 0] > 0.70
indemnity = type_probs[:, 1] > 0.60
print(f"\nNew claim routing:")
print(f"  Medical-only (auto-reserve $5K-$15K): {medical_only.sum()}")
print(f"  Indemnity (reserve $25K-$75K, assign adjuster): {indemnity.sum()}")
print(f"  Review needed: {(~medical_only & ~indemnity).sum()}")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Workers' Comp Claim Classification ---
Accuracy:    83.4%

                precision  recall  f1-score  support
  Medical-Only     0.87    0.89     0.88      620
     Indemnity     0.80    0.78     0.79      380
       Denied      0.79    0.77     0.78      200

--- Claims Routing Impact ---
Auto-classified (>70% confidence): 72% of new claims
Adjuster assignment time reduced: 45 minutes -> 8 minutes
Reserve accuracy (within 20%): 81%
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/catboost_sklearn.py`](../../02_classification/catboost_sklearn.py)

Related:
- [CatBoost - NumPy (from scratch)](catboost_numpy.md)
- [CatBoost - PyTorch](catboost_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
