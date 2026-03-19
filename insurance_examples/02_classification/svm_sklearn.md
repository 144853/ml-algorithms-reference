# Support Vector Machine - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of SVM for **auto accident severity classification**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Support Vector Machine - Full Documentation](svm_numpy.md)**

---

## 🛡️ **Insurance Use Case: Auto Accident Severity Classification**

Classify auto accident claims into **minor**, **moderate**, or **severe** categories based on:
- Vehicle speed at impact (mph)
- Airbag deployment (0 = no, 1 = yes)
- Number of vehicles involved (1 - 5)
- Road condition score (1 = dry, 2 = wet, 3 = icy, 4 = construction)
- Weather condition (1 = clear, 2 = rain, 3 = snow, 4 = fog)

### **Why Scikit-learn for Severity Classification?**
- **Kernel trick**: RBF kernel captures non-linear speed-severity relationships
- **Margin optimization**: Maximizes separation between severity classes
- **Robust to outliers**: Soft margin handles unusual accident scenarios
- **Probability calibration**: Platt scaling provides severity probabilities

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Multiple kernels**: Linear, RBF, polynomial for different data patterns

---

## 💻 **Quick Start**

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Auto accident severity data
# Features: speed, airbag_deployed, num_vehicles, road_condition, weather
X, y = load_accident_severity_data()  # y: 0=minor, 1=moderate, 2=severe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale features (critical for SVM - speed in mph vs binary airbag)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model with RBF kernel
model = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    class_weight='balanced',
    probability=True  # Enable Platt scaling for severity probabilities
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Minor', 'Moderate', 'Severe']))
```

---

## 🔧 **Key Parameters for Severity Classification**

| Parameter | Default | Insurance Recommendation |
|-----------|---------|--------------------------|
| `kernel` | `'rbf'` | RBF for non-linear speed/severity relationship |
| `C` | 1.0 | Higher C (10-100) for strict severity boundaries |
| `gamma` | `'scale'` | `'scale'` auto-adjusts; tune if underfitting |
| `class_weight` | None | `'balanced'` if severe accidents are rare |
| `probability` | False | Set True for severity confidence scores |

### **Insurance-Specific Workflow**

```python
from sklearn.model_selection import GridSearchCV

# Tune SVM for accident severity
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly']
}
grid = GridSearchCV(SVC(class_weight='balanced', probability=True),
                    param_grid, scoring='f1_weighted', cv=5)
grid.fit(X_train_scaled, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best F1: {grid.best_score_:.4f}")

# Reserve setting: Route severe claims to senior adjuster
severity_probs = grid.predict_proba(X_test_scaled)
severe_prob = severity_probs[:, 2]  # Probability of 'severe'
fast_track_mask = severe_prob > 0.75
print(f"Fast-track to senior adjuster: {fast_track_mask.sum()} claims")
```

---

## 📊 **Sample Results on Insurance Data**

```
--- Accident Severity Classification ---
Accuracy:    84.2%

              precision  recall  f1-score  support
       Minor     0.88    0.90     0.89      480
    Moderate     0.80    0.78     0.79      350
      Severe     0.83    0.84     0.83      270

--- Reserve Routing ---
Severe probability > 75%:  218 claims -> Senior adjuster
Moderate probability > 60%: 312 claims -> Standard adjuster
Minor probability > 70%:    470 claims -> Auto-adjudication
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/svm_sklearn.py`](../../02_classification/svm_sklearn.py)

Related:
- [Support Vector Machine - NumPy (from scratch)](svm_numpy.md)
- [Support Vector Machine - PyTorch](svm_pytorch.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
