# Linear Regression - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of Linear Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Linear Regression - Full Documentation](linear_regression_numpy.md)**

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized Cython implementation
- **Numerical stability**: SVD-based solver (more stable than direct matrix inversion)
- **Minimal code**: Fit in 2 lines
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Multiple solvers**: Auto-selects best solver based on data characteristics

---

## 💻 **Quick Start**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X, y = load_your_data()  # (n_samples, n_features), (n_samples,)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")

# Inspect learned parameters
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

---

## 🔧 **Key Parameters**

```python
LinearRegression(
    fit_intercept=True,    # Whether to calculate intercept (almost always True)
    copy_X=True,           # Copy X to avoid overwriting original data
    n_jobs=None,           # Number of CPU cores (-1 for all cores)
    positive=False         # Force coefficients to be positive
)
```

---

## 📊 **Scikit-learn Specific Features**

### **1. Integration with Pipelines**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),        # Standardize features
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### **2. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    LinearRegression(), 
    X, y, 
    cv=5,                    # 5-fold CV
    scoring='neg_mean_squared_error'
)
rmse_scores = (-scores) ** 0.5
print(f"CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
```

### **3. Feature Importance (via coefficients)**
```python
import pandas as pd

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(feature_importance)
```

---

## ⚖️ **When to Use Scikit-learn vs NumPy**

| Scenario | Use This |
|----------|----------|
| Production deployment | ✅ Scikit-learn |
| Learning internals | NumPy from-scratch |
| Integration with sklearn ecosystem | ✅ Scikit-learn |
| Custom modifications | NumPy from-scratch |
| Maximum performance | ✅ Scikit-learn (Cython) |

---

## 📝 **Code Reference**

Full implementation: [`01_regression/linear_regression_sklearn.py`](../../01_regression/linear_regression_sklearn.py)

Related:
- [Linear Regression - NumPy (from scratch)](linear_regression_numpy.md)
- [Linear Regression - PyTorch](linear_regression_pytorch.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
