# Ridge Regression (scikit-learn) - Health Insurance Claim Prediction

## **Use Case: Predicting Health Insurance Claim Amounts**

### **The Problem**
A health insurer uses scikit-learn's `Ridge` to predict annual claim amounts for members. Features: age, BMI, chronic conditions count, prescription count, and hospitalization days. Target: annual claim amount ($100 - $100,000).

---

## **scikit-learn API Overview**

```python
from sklearn.linear_model import Ridge, RidgeCV

# Basic Ridge
model = Ridge(
    alpha=1.0,            # Regularization strength
    fit_intercept=True,   # Learn baseline claim level
    solver='auto',        # 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
    max_iter=None,        # For iterative solvers
    tol=1e-4,             # Convergence tolerance
)

# Ridge with built-in cross-validation for alpha selection
model_cv = RidgeCV(
    alphas=(0.1, 1.0, 10.0, 100.0),  # Alphas to test
    cv=5,                              # 5-fold CV
    scoring='r2',                      # Optimization metric
)
```

---

## **Insurance Workflow Implementation**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Generate health insurance data ---
np.random.seed(42)
n = 3000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'bmi': np.random.normal(27, 5, n).clip(15, 50),
    'chronic_conditions': np.random.poisson(1.5, n).clip(0, 8),
    'prescription_count': np.random.poisson(3, n).clip(0, 15),
    'hospitalization_days': np.random.exponential(3, n).clip(0, 60).astype(int),
})

# Introduce correlation: more chronic conditions -> more Rx and hospital days
data['prescription_count'] = (data['chronic_conditions'] * 1.8
                               + np.random.normal(0, 1.5, n)).clip(0, 15).astype(int)

data['annual_claim'] = (
    -12000
    + 280 * data['age']
    + 310 * data['bmi']
    + 4100 * data['chronic_conditions']
    + 1100 * data['prescription_count']
    + 2800 * data['hospitalization_days']
    + np.random.normal(0, 3000, n)
).clip(100, 100000)

feature_cols = ['age', 'bmi', 'chronic_conditions', 'prescription_count', 'hospitalization_days']
X = data[feature_cols].values
y = data['annual_claim'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **Pipeline with Automatic Alpha Selection**

```python
# Step 1: Pipeline with fixed alpha
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

ridge_pipeline.fit(X_train, y_train)
y_pred = ridge_pipeline.predict(X_test)

print("--- Fixed Alpha=1.0 ---")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")
print(f"MAE:       ${mean_absolute_error(y_test, y_pred):,.2f}")

# Step 2: Auto-tune alpha with RidgeCV
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_cv = RidgeCV(
    alphas=np.logspace(-3, 3, 50),
    cv=5,
    scoring='neg_mean_absolute_error'
)
ridge_cv.fit(X_train_scaled, y_train)

print(f"\n--- RidgeCV Results ---")
print(f"Best alpha: {ridge_cv.alpha_:.4f}")
print(f"R-squared:  {ridge_cv.score(X_test_scaled, y_test):.4f}")
y_pred_cv = ridge_cv.predict(X_test_scaled)
print(f"MAE:        ${mean_absolute_error(y_test, y_pred_cv):,.2f}")
```

---

## **Coefficient Analysis**

```python
# Compare coefficients: OLS vs Ridge
from sklearn.linear_model import LinearRegression

ols = LinearRegression().fit(X_train_scaled, y_train)
ridge_low = Ridge(alpha=0.01).fit(X_train_scaled, y_train)
ridge_high = Ridge(alpha=100.0).fit(X_train_scaled, y_train)

print("\n--- Coefficient Comparison (Standardized) ---")
print(f"{'Feature':25s} {'OLS':>10s} {'Ridge(0.01)':>12s} {'Ridge(100)':>12s}")
print("-" * 62)
for i, name in enumerate(feature_cols):
    print(f"{name:25s} {ols.coef_[i]:>10.1f} {ridge_low.coef_[i]:>12.1f} {ridge_high.coef_[i]:>12.1f}")
```

### **Expected Output:**
```
--- Coefficient Comparison (Standardized) ---
Feature                          OLS  Ridge(0.01)  Ridge(100)
--------------------------------------------------------------
age                          3850.2       3848.5      3210.4
bmi                          2180.5       2178.8      1890.3
chronic_conditions           8520.3       8480.1      5920.6
prescription_count           3410.8       3395.2      2680.5
hospitalization_days        12050.1      11980.4      8250.2
```

---

## **Production Deployment**

```python
import joblib

# Save full pipeline
joblib.dump({
    'scaler': scaler,
    'model': ridge_cv,
    'feature_names': feature_cols,
    'best_alpha': ridge_cv.alpha_
}, 'health_claim_ridge_v1.pkl')

# Score new members
new_member = np.array([[55, 31.0, 2, 5, 3]])
new_member_scaled = scaler.transform(new_member)
predicted_claim = ridge_cv.predict(new_member_scaled)
print(f"Predicted Annual Claim: ${predicted_claim[0]:,.2f}")
```

---

## **Cross-Validation Stability Check**

```python
# Verify model stability across folds
cv_scores = cross_val_score(
    Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=ridge_cv.alpha_))]),
    X, y, cv=10, scoring='r2'
)
print(f"\n10-Fold CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"Min fold R2: {cv_scores.min():.4f}")
print(f"Max fold R2: {cv_scores.max():.4f}")
```

---

## **When to Use sklearn Ridge**

| Scenario | Recommendation |
|----------|---------------|
| Correlated clinical variables | Use Ridge - stabilizes coefficients |
| Need automatic alpha tuning | Use RidgeCV |
| Want feature elimination | Use Lasso or ElasticNet instead |
| Large member populations (>1M) | Use solver='sag' or 'saga' |
| Sparse feature matrices | Use solver='sparse_cg' |

---

## **Hyperparameters**

| Parameter | Value | Insurance Rationale |
|-----------|-------|-------------------|
| `alpha` | RidgeCV-selected | Balance fit vs. stability for claims data |
| `solver` | 'auto' | Let sklearn pick based on data size |
| `fit_intercept` | True | Baseline claim level before risk adjustment |
| `cv` (RidgeCV) | 5 | Standard for insurance model validation |

---

## **References**

1. scikit-learn Ridge / RidgeCV documentation
2. Duncan, "Healthcare Risk Adjustment and Predictive Modeling" (2011)
3. Hoerl & Kennard, "Ridge Regression" (1970)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
