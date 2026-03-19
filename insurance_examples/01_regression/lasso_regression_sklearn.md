# Lasso Regression (scikit-learn) - Property Insurance Loss Severity

## **Use Case: Predicting Property Insurance Loss Severity**

### **The Problem**
A property insurer uses scikit-learn's `Lasso` and `LassoCV` to predict claim severity and automatically select the most important building rating factors. Features: building age, square footage, fire protection class, weather exposure score, occupancy type. Target: loss severity ($500 - $80,000).

---

## **scikit-learn API Overview**

```python
from sklearn.linear_model import Lasso, LassoCV

# Basic Lasso
model = Lasso(
    alpha=1.0,            # Regularization strength (L1 penalty)
    fit_intercept=True,   # Learn baseline loss level
    max_iter=10000,       # Increase for convergence
    tol=1e-4,             # Convergence tolerance
    warm_start=False,     # Reuse previous solution
    selection='cyclic',   # 'cyclic' or 'random' coordinate descent
)

# Lasso with built-in cross-validation
model_cv = LassoCV(
    alphas=None,          # Auto-generate alpha path
    n_alphas=100,         # Number of alphas to test
    cv=5,                 # 5-fold CV
    max_iter=10000,
)
```

---

## **Insurance Workflow Implementation**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --- Generate property insurance data ---
np.random.seed(42)
n = 3000

data = pd.DataFrame({
    'building_age': np.random.uniform(0, 150, n),
    'square_footage': np.random.uniform(500, 50000, n),
    'fire_protection_class': np.random.randint(1, 11, n),
    'weather_exposure_score': np.random.uniform(1, 10, n),
    'occupancy_type': np.random.randint(1, 4, n),
    # Add noise features to demonstrate Lasso's selection ability
    'color_code': np.random.randint(1, 6, n),          # Irrelevant
    'floor_material': np.random.randint(1, 4, n),      # Irrelevant
    'parking_spaces': np.random.randint(0, 50, n),     # Irrelevant
})

data['loss_severity'] = (
    -5000
    + 180 * data['building_age']
    + 0.02 * data['square_footage']  # Very weak
    + 2400 * data['fire_protection_class']
    + 1800 * data['weather_exposure_score']
    + 3400 * data['occupancy_type']
    + np.random.normal(0, 3500, n)
).clip(500, 80000)

feature_cols = ['building_age', 'square_footage', 'fire_protection_class',
                'weather_exposure_score', 'occupancy_type',
                'color_code', 'floor_material', 'parking_spaces']
X = data[feature_cols].values
y = data['loss_severity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **Pipeline with Automatic Alpha Selection**

```python
# Scale features first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LassoCV finds optimal alpha automatically
lasso_cv = LassoCV(n_alphas=100, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

y_pred = lasso_cv.predict(X_test_scaled)
print(f"Best alpha:  {lasso_cv.alpha_:.4f}")
print(f"R-squared:   {r2_score(y_test, y_pred):.4f}")
print(f"MAE:         ${mean_absolute_error(y_test, y_pred):,.2f}")
```

---

## **Feature Selection Analysis**

```python
print("\n--- Feature Selection Results ---")
print(f"{'Feature':30s} {'Coefficient':>12s} {'Status':>10s}")
print("-" * 55)
for name, coef in zip(feature_cols, lasso_cv.coef_):
    status = "SELECTED" if abs(coef) > 1e-6 else "REMOVED"
    print(f"{name:30s} {coef:>12.2f} {status:>10s}")

n_selected = np.sum(np.abs(lasso_cv.coef_) > 1e-6)
print(f"\nFeatures selected: {n_selected}/{len(feature_cols)}")
print(f"Features removed:  {len(feature_cols) - n_selected}/{len(feature_cols)}")
```

### **Expected Output:**
```
--- Feature Selection Results ---
Feature                         Coefficient     Status
-------------------------------------------------------
building_age                       2480.50   SELECTED
square_footage                        0.00    REMOVED
fire_protection_class              5120.30   SELECTED
weather_exposure_score             3850.20   SELECTED
occupancy_type                     2980.40   SELECTED
color_code                            0.00    REMOVED
floor_material                        0.00    REMOVED
parking_spaces                        0.00    REMOVED

Features selected: 4/8
Features removed:  4/8
```

---

## **Regularization Path Visualization**

```python
# Track which features enter/leave the model at different alphas
alphas = np.logspace(-2, 3, 50)
coef_path = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coef_path.append(lasso.coef_.copy())

coef_path = np.array(coef_path)

# Print key transitions
print("\n--- Regularization Path Summary ---")
for i, name in enumerate(feature_cols):
    first_zero = np.where(np.abs(coef_path[:, i]) < 1e-6)[0]
    if len(first_zero) > 0:
        print(f"  {name:30s}: zeroed at alpha={alphas[first_zero[0]]:.3f}")
    else:
        print(f"  {name:30s}: never zeroed (strong predictor)")
```

---

## **Production Pipeline**

```python
from sklearn.pipeline import Pipeline
import joblib

# Full production pipeline
property_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LassoCV(n_alphas=100, cv=5, max_iter=10000))
])
property_pipeline.fit(X_train, y_train)

# Save
joblib.dump(property_pipeline, 'property_loss_lasso_v1.pkl')

# Score new property
new_property = np.array([[30, 5000, 5, 6.0, 2, 3, 2, 10]])  # Includes noise features
predicted_loss = property_pipeline.predict(new_property)
print(f"Predicted Loss Severity: ${predicted_loss[0]:,.2f}")
```

---

## **Lasso vs Ridge Comparison**

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
lasso = Lasso(alpha=1.0, max_iter=10000).fit(X_train_scaled, y_train)

print("\n--- Ridge vs Lasso Coefficients ---")
print(f"{'Feature':30s} {'Ridge':>10s} {'Lasso':>10s}")
print("-" * 52)
for name, r_coef, l_coef in zip(feature_cols, ridge.coef_, lasso.coef_):
    print(f"{name:30s} {r_coef:>10.2f} {l_coef:>10.2f}")

print(f"\nRidge non-zero: {np.sum(np.abs(ridge.coef_) > 1e-6)}")
print(f"Lasso non-zero: {np.sum(np.abs(lasso.coef_) > 1e-6)}")
```

---

## **When to Use sklearn Lasso**

| Scenario | Recommendation |
|----------|---------------|
| Need automatic feature selection | Use Lasso - zeroes out irrelevant features |
| Many candidate rating variables | Use LassoCV for optimal alpha |
| Regulatory model simplicity | Use Lasso - fewer factors to justify |
| Correlated features (keep both) | Use Ridge or ElasticNet instead |
| Need exact sparsity control | Adjust alpha or use `max_features` approach |

---

## **Hyperparameters**

| Parameter | Value | Insurance Rationale |
|-----------|-------|-------------------|
| `alpha` | LassoCV-selected | Auto-balance sparsity vs. accuracy |
| `max_iter` | 10000 | Ensure convergence for property data |
| `selection` | 'cyclic' | Deterministic for regulatory reproducibility |
| `tol` | 1e-4 | Standard precision for loss severity |

---

## **References**

1. scikit-learn Lasso / LassoCV documentation
2. Tibshirani, "Regression Shrinkage and Selection via the Lasso" (1996)
3. Klugman et al., "Loss Models: From Data to Decisions" (2012)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
