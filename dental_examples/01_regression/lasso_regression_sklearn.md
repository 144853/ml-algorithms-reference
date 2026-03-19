# Lasso Regression (Scikit-Learn) - Dental Implant Success Prediction

## 🦷 **Use Case: Predicting Dental Implant Success Score**

Predict implant success scores (0-100) from bone density, gum health, smoking status, diabetes indicator, and implant material using scikit-learn's Lasso with automatic feature selection.

---

## 📦 **Quick Start**

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Dental implant dataset: 550 patients, 5 features
X, y = load_implant_data()  # y = success score (0-100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1, max_iter=10000))
])

pipeline.fit(X_train, y_train)
print(f"R^2 Score: {pipeline.score(X_test, y_test):.4f}")
```

---

## 🔧 **Scikit-Learn API Details**

### **Key Parameters**
```python
Lasso(
    alpha=1.0,            # L1 regularization strength (higher = more sparsity)
    fit_intercept=True,   # Include bias term
    max_iter=10000,       # Increase for convergence with dental data
    tol=1e-4,             # Convergence tolerance
    warm_start=False,     # Reuse previous solution for faster fitting
    selection='cyclic',   # 'cyclic' or 'random' coordinate descent
    random_state=42       # For 'random' selection
)
```

### **LassoCV: Automatic Alpha Selection**
```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(
    alphas=None,          # Auto-generate alpha path
    cv=5,                 # 5-fold cross-validation
    max_iter=10000,
    n_alphas=100,         # Number of alphas along the path
    random_state=42
)

pipeline_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso_cv', lasso_cv)
])

pipeline_cv.fit(X_train, y_train)
best_alpha = pipeline_cv.named_steps['lasso_cv'].alpha_
print(f"Best alpha: {best_alpha:.4f}")
# Best alpha: 0.0842
```

---

## 🏥 **Feature Selection Analysis**

### **Identifying Important Predictors**
```python
import numpy as np

model = pipeline_cv.named_steps['lasso_cv']
feature_names = ['bone_density', 'gum_health', 'smoking_status',
                 'diabetes_indicator', 'implant_material']

# Get non-zero coefficients (selected features)
coefs = model.coef_
for name, coef in zip(feature_names, coefs):
    status = "SELECTED" if abs(coef) > 0 else "ELIMINATED"
    print(f"  {name:25s}: {coef:8.4f}  [{status}]")

# bone_density             :  18.5200  [SELECTED]
# gum_health               :   4.2100  [SELECTED]
# smoking_status           : -12.7800  [SELECTED]
# diabetes_indicator       :   0.0000  [ELIMINATED]
# implant_material         :   0.0000  [ELIMINATED]

n_selected = np.sum(coefs != 0)
print(f"\nFeatures selected: {n_selected}/{len(feature_names)}")
```

### **Lasso Path Visualization**
```python
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

alphas, coef_path, _ = lasso_path(X_train_scaled, y_train, alphas=np.logspace(-4, 1, 100))

plt.figure(figsize=(10, 6))
for i, name in enumerate(feature_names):
    plt.plot(alphas, coef_path[i], label=name)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient value')
plt.title('Lasso Path: Dental Implant Feature Selection')
plt.legend()
plt.axvline(x=best_alpha, color='k', linestyle='--', label='Best alpha')
plt.show()
```

---

## 📊 **Comparing with Ridge for Feature Selection**

```python
from sklearn.linear_model import Ridge

ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
lasso = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=0.1))])

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

print("Feature      | Ridge Coef | Lasso Coef")
print("-" * 45)
for name, rc, lc in zip(feature_names,
                         ridge.named_steps['ridge'].coef_,
                         lasso.named_steps['lasso'].coef_):
    print(f"{name:20s} | {rc:10.4f} | {lc:10.4f}")
```

---

## 📈 **Model Evaluation**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = pipeline_cv.predict(X_test)

print(f"R^2 Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:       {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} points")
print(f"MAE:        {mean_absolute_error(y_test, y_pred):.2f} points")

# R^2 Score:  0.8712
# RMSE:       6.95 points
# MAE:        5.42 points
```

---

## 💾 **Model Persistence**

```python
import joblib

joblib.dump(pipeline_cv, 'implant_success_predictor.pkl')
loaded_model = joblib.load('implant_success_predictor.pkl')
```

---

## 🎯 **Production Tips for Implant Clinics**

1. **Use LassoCV** -- it automatically finds the best alpha through cross-validation
2. **Check stability** -- run Lasso on multiple bootstrap samples to see if selected features are consistent
3. **Increase max_iter** -- dental datasets with correlated features may need 10K+ iterations
4. **Combine with domain knowledge** -- if Lasso drops a feature you know is important, consider ElasticNet
5. **Build a clinical score** -- use selected features to create a simple implant risk calculator

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/lasso_regression_sklearn.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
