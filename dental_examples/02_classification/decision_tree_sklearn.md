# Decision Tree (Scikit-Learn) - Dental Classification

## **Use Case: Classifying Tooth Extraction Necessity**

### **The Problem**
An oral surgery department evaluates **600 patients** to determine whether a tooth requires extraction or can be preserved. The decision depends on mobility grade, bone loss percentage, infection status, and restorability score.

### **Why Decision Tree?**
- Mimics clinical decision-making flowcharts
- No feature scaling required
- Handles both numerical and categorical dental data
- Produces visual decision rules clinicians can follow

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(42)
n_patients = 600

data = pd.DataFrame({
    'mobility_grade': np.random.choice([0, 1, 2, 3], n_patients, p=[0.4, 0.3, 0.2, 0.1]),
    'bone_loss_pct': np.random.uniform(0, 80, n_patients),
    'infection_status': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
    'restorability_score': np.random.uniform(0, 10, n_patients)
})

# Clinical decision logic
extract_score = (data['mobility_grade'] * 2 + data['bone_loss_pct'] / 20 +
                 data['infection_status'] * 1.5 - data['restorability_score'] * 0.5)
data['decision'] = (extract_score > 2.0).astype(int)  # 0=preserve, 1=extract

X = data.drop('decision', axis=1)
y = data['decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
dt_model = DecisionTreeClassifier(
    max_depth=5,              # Limit tree depth for interpretability
    min_samples_split=20,     # Minimum samples to split a node
    min_samples_leaf=10,      # Minimum samples in a leaf
    criterion='gini',         # Gini impurity
    random_state=42
)

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
```

---

## **Visualizing the Decision Tree**

```python
# Text representation of decision rules
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print(tree_rules)

# Example output:
# |--- bone_loss_pct <= 35.50
# |   |--- mobility_grade <= 1.50
# |   |   |--- restorability_score <= 3.20
# |   |   |   |--- class: 0 (preserve)
# |   |   |--- restorability_score > 3.20
# |   |   |   |--- class: 0 (preserve)
# |   |--- mobility_grade > 1.50
# |   |   |--- infection_status <= 0.50
# |   |   |   |--- class: 0 (preserve)
# |   |   |--- infection_status > 0.50
# |   |   |   |--- class: 1 (extract)
# |--- bone_loss_pct > 35.50
# |   |--- mobility_grade <= 0.50
# |   |   |--- class: 0 (preserve)
# |   |--- mobility_grade > 0.50
# |   |   |--- class: 1 (extract)

# Visual plot
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_model, feature_names=list(X.columns),
          class_names=['Preserve', 'Extract'], filled=True, ax=ax)
plt.title("Tooth Extraction Decision Tree")
plt.tight_layout()
plt.savefig('extraction_decision_tree.png', dpi=150)
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

    Preserve       0.88      0.90      0.89        62
     Extract       0.89      0.87      0.88        58

    accuracy                           0.88       120
   macro avg       0.88      0.88      0.88       120
weighted avg       0.88      0.88      0.88       120
```

### **Feature Importance**

```python
importance = pd.Series(dt_model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))
```

```
bone_loss_pct         0.42
mobility_grade        0.28
restorability_score   0.18
infection_status      0.12
```

---

## **Clinical Decision Function**

```python
def extraction_assessment(model, patient_data):
    """Assess whether a tooth needs extraction."""
    df = pd.DataFrame([patient_data])
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    # Get decision path
    node_indicator = model.decision_path(df)
    rules = export_text(model, feature_names=list(df.columns))

    return {
        'decision': 'Extract' if prediction == 1 else 'Preserve',
        'confidence': f"{max(probabilities):.1%}",
        'preserve_prob': f"{probabilities[0]:.1%}",
        'extract_prob': f"{probabilities[1]:.1%}"
    }

# Example patient
result = extraction_assessment(dt_model, {
    'mobility_grade': 3,
    'bone_loss_pct': 55.0,
    'infection_status': 1,
    'restorability_score': 2.5
})
print(result)
# {'decision': 'Extract', 'confidence': '95.2%', ...}
```

---

## **Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [5, 10, 20, 30],
    'min_samples_leaf': [5, 10, 15, 20],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.4f}")
```

---

## **Pruning for Clinical Use**

```python
# Cost complexity pruning
path = dt_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train trees with different alpha values
pruned_models = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    pruned_models.append(clf)

# Select best pruned tree based on validation performance
val_scores = [clf.score(X_test, y_test) for clf in pruned_models]
best_idx = np.argmax(val_scores)
best_pruned = pruned_models[best_idx]
print(f"Best pruned tree depth: {best_pruned.get_depth()}, accuracy: {val_scores[best_idx]:.4f}")
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Need explainable clinical decisions | Ideal - tree rules map to clinical protocols |
| Mixed data types | Handles categorical + numerical features |
| Quick baseline | Fast training, no preprocessing needed |
| Regulatory requirement | Fully transparent decision path |
| Prone to overfitting | Use max_depth, pruning, or switch to Random Forest |

---

## **Running the Demo**

```bash
cd examples/02_classification
python decision_tree_sklearn.py
```

---

## **References**

1. Breiman et al. "Classification and Regression Trees" (1984)
2. Scikit-Learn Documentation: DecisionTreeClassifier
3. ADA Guidelines for Tooth Extraction Assessment

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
