# Decision Tree - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of Decision Tree for **insurance risk tier classification**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Decision Tree - Full Documentation](decision_tree_numpy.md)**

---

## 🛡️ **Insurance Use Case: Risk Tier Classification**

Classify insurance applicants into **standard**, **preferred**, or **substandard** risk tiers based on:
- Health metrics (BMI, blood pressure, cholesterol)
- Occupation hazard class (1-5)
- Driving record (violations in past 5 years)
- Credit score (300-850)

### **Why Scikit-learn for Risk Tiering?**
- **Visual explainability**: Export tree as diagram for underwriter review
- **Rule extraction**: Convert tree paths to underwriting guidelines
- **No scaling needed**: Decision trees handle raw feature values natively
- **Multi-class native**: Handles standard/preferred/substandard directly

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Tree visualization**: Export to graphviz for underwriter presentations

---

## 💻 **Quick Start**

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Insurance risk tier data
# Features: bmi, systolic_bp, cholesterol, occupation_class, driving_violations, credit_score
X, y = load_risk_tier_data()  # y: 0=standard, 1=preferred, 2=substandard
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Create and train model
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=20,
    class_weight='balanced',
    criterion='gini'
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Standard', 'Preferred', 'Substandard']))

# Extract underwriting rules
feature_names = ['bmi', 'systolic_bp', 'cholesterol', 'occupation_class', 'driving_violations', 'credit_score']
print(export_text(model, feature_names=feature_names, max_depth=3))
```

---

## 🔧 **Key Parameters for Insurance Risk Tiering**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `max_depth` | None | Set to 5-8 for interpretable underwriting rules |
| `min_samples_leaf` | 1 | Set to 20-50 for statistically credible tier assignments |
| `criterion` | `'gini'` | Use `'entropy'` for information-theoretic splits |
| `class_weight` | None | `'balanced'` if preferred tier is underrepresented |
| `min_impurity_decrease` | 0.0 | Set to 0.01 to prune insignificant splits |

### **Insurance-Specific Workflow**

```python
from sklearn.tree import export_graphviz
import graphviz

# Visualize tree for underwriter review
dot_data = export_graphviz(model, feature_names=feature_names,
                           class_names=['Standard', 'Preferred', 'Substandard'],
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("risk_tier_tree")  # Export to PDF for underwriting committee

# Feature importance for risk factor analysis
for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
    print(f"{name}: {imp:.3f}")
# credit_score: 0.312
# bmi: 0.228
# cholesterol: 0.195
# driving_violations: 0.142
# systolic_bp: 0.078
# occupation_class: 0.045
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Risk Tier Classification Performance ---
Accuracy:    82.7%

              precision  recall  f1-score  support
   Standard      0.85    0.88     0.86      540
   Preferred     0.81    0.76     0.78      320
Substandard      0.80    0.82     0.81      340

--- Top Underwriting Rule Paths ---
IF credit_score > 720 AND bmi <= 27.5 AND driving_violations == 0
  THEN -> Preferred (confidence: 89%)

IF credit_score <= 620 OR (bmi > 35 AND cholesterol > 260)
  THEN -> Substandard (confidence: 84%)
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/decision_tree_sklearn.py`](../../02_classification/decision_tree_sklearn.py)

Related:
- [Decision Tree - NumPy (from scratch)](decision_tree_numpy.md)
- [Decision Tree - PyTorch](decision_tree_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
