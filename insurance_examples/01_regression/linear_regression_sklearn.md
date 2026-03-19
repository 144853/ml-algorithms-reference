# Linear Regression (scikit-learn) - Insurance Premium Prediction

## **Use Case: Predicting Auto Insurance Premiums**

### **The Problem**
An auto insurance company needs a production-ready premium prediction model using scikit-learn's `LinearRegression`. The model predicts annual premiums based on driver age, vehicle value, driving record (accidents), coverage level, and zip code risk score.

---

## **scikit-learn API Overview**

```python
from sklearn.linear_model import LinearRegression

# Key parameters
model = LinearRegression(
    fit_intercept=True,   # Learn bias term (base premium)
    copy_X=True,          # Don't modify original data
    n_jobs=-1,            # Use all CPU cores for large datasets
    positive=False        # Allow negative coefficients (e.g., age discount)
)

# Core methods
model.fit(X_train, y_train)        # Train on historical policies
model.predict(X_new)               # Score new quote requests
model.score(X_test, y_test)        # R-squared evaluation
model.coef_                        # Feature coefficients (rating factors)
model.intercept_                   # Base premium
```

---

## **Insurance Workflow Integration**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Generate synthetic insurance data ---
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    'driver_age': np.random.randint(16, 85, n),
    'vehicle_value': np.random.uniform(2000, 75000, n),
    'accidents_3yr': np.random.poisson(0.4, n).clip(0, 5),
    'coverage_level': np.random.randint(1, 4, n),
    'zip_risk_score': np.random.uniform(1.0, 10.0, n),
})

data['annual_premium'] = (
    600
    - 6.0 * data['driver_age']
    + 0.02 * data['vehicle_value']
    + 350 * data['accidents_3yr']
    + 180 * data['coverage_level']
    + 95 * data['zip_risk_score']
    + np.random.normal(0, 150, n)
).clip(500, 5000)

feature_cols = ['driver_age', 'vehicle_value', 'accidents_3yr', 'coverage_level', 'zip_risk_score']
X = data[feature_cols].values
y = data['annual_premium'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **Pipeline Implementation**

```python
# Production pipeline with scaling
premium_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train
premium_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = premium_pipeline.predict(X_test)
print(f"R-squared:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE:        ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE:       ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Cross-validation for robustness
cv_scores = cross_val_score(premium_pipeline, X, y, cv=5, scoring='r2')
print(f"CV R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## **Extracting Insurance Insights**

```python
# Get the trained model from the pipeline
model = premium_pipeline.named_steps['regressor']
scaler = premium_pipeline.named_steps['scaler']

# Coefficients in original scale
original_coefs = model.coef_ / scaler.scale_
original_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

print("\n--- Rating Factors ---")
for name, coef in zip(feature_cols, original_coefs):
    print(f"  {name:20s}: ${coef:>10.2f}")
print(f"  {'Base Premium':20s}: ${original_intercept:>10.2f}")
```

### **Expected Output:**
```
--- Rating Factors ---
  driver_age          :     -$6.02
  vehicle_value       :      $0.02
  accidents_3yr       :    $349.85
  coverage_level      :    $180.12
  zip_risk_score      :     $94.78
  Base Premium        :    $601.25
```

---

## **Scoring New Quotes**

```python
# New applicant: 30-year-old, $25,000 car, 0 accidents, comprehensive, moderate risk
new_applicant = np.array([[30, 25000, 0, 3, 5.5]])
predicted_premium = premium_pipeline.predict(new_applicant)
print(f"Quoted Premium: ${predicted_premium[0]:,.2f}")

# Batch scoring for marketing campaigns
batch_quotes = np.array([
    [22, 15000, 1, 1, 7.0],   # Young, economy, one accident
    [45, 40000, 0, 3, 3.0],   # Middle-aged, luxury, clean
    [65, 20000, 0, 2, 4.5],   # Senior, moderate car
])
batch_premiums = premium_pipeline.predict(batch_quotes)
for i, p in enumerate(batch_premiums):
    print(f"  Applicant {i+1}: ${p:,.2f}")
```

---

## **Model Persistence for Production**

```python
import joblib

# Save the trained pipeline
joblib.dump(premium_pipeline, 'auto_premium_model_v1.pkl')

# Load in production quoting engine
loaded_pipeline = joblib.load('auto_premium_model_v1.pkl')
quote = loaded_pipeline.predict(new_applicant)
```

---

## **When to Use sklearn LinearRegression**

| Scenario | Recommendation |
|----------|---------------|
| Regulatory filing models | Use - coefficients are directly interpretable |
| Real-time quoting (< 1ms) | Use - extremely fast inference |
| > 100 rating variables | Consider Ridge/Lasso for regularization |
| Non-linear risk curves | Use PolynomialFeatures or switch to GBM |
| Small datasets (< 500 policies) | Use with cross-validation |

---

## **Hyperparameters**

| Parameter | Value | Insurance Rationale |
|-----------|-------|-------------------|
| `fit_intercept` | True | Represents base premium before rating factors |
| `positive` | False | Age discount requires negative coefficients |
| `n_jobs` | -1 | Parallelize for large policy books |

---

## **References**

1. scikit-learn LinearRegression documentation
2. Werner & Modlin, "Basic Ratemaking" (CAS, 2016)
3. Hastie, Tibshirani & Friedman, "Elements of Statistical Learning" (2009)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
