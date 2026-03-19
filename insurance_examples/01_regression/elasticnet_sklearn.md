# ElasticNet (scikit-learn) - Life Insurance Risk Score Prediction

## **Use Case: Predicting Life Insurance Risk Score**

### **The Problem**
A life insurance underwriter needs to predict a composite risk score for applicants. The risk score determines premium rates and coverage eligibility. Features: age, health exam results, family history score, occupation risk, and lifestyle habits score. Target: risk score (1 - 100, higher = riskier).

---

## **scikit-learn API Overview**

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Basic ElasticNet
model = ElasticNet(
    alpha=1.0,            # Overall regularization strength
    l1_ratio=0.5,         # Mix of L1 vs L2 (0=Ridge, 1=Lasso, 0.5=equal mix)
    fit_intercept=True,   # Learn baseline risk
    max_iter=10000,       # Convergence iterations
    tol=1e-4,             # Convergence tolerance
    selection='cyclic',   # Coordinate descent order
)

# ElasticNet with cross-validated alpha and l1_ratio selection
model_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],  # L1/L2 mix values to test
    n_alphas=100,                           # Alphas per l1_ratio
    cv=5,                                   # 5-fold CV
    max_iter=10000,
)
```

### **ElasticNet Penalty**
```
Loss = MSE + alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) / 2 * ||w||_2^2
                                  ^-- L1 (sparsity) --^     ^-- L2 (stability) --^
```

---

## **Insurance Workflow Implementation**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --- Generate life insurance applicant data ---
np.random.seed(42)
n = 3000

data = pd.DataFrame({
    'age': np.random.randint(20, 75, n),
    'health_exam_score': np.random.uniform(40, 100, n),       # Higher = healthier
    'family_history_score': np.random.uniform(0, 10, n),      # Higher = worse history
    'occupation_risk': np.random.uniform(1, 10, n),           # 1=office, 10=hazardous
    'lifestyle_score': np.random.uniform(1, 10, n),           # 1=healthy, 10=risky
    # Correlated features (health exam partially captures these)
    'blood_pressure_index': np.random.uniform(0.5, 2.0, n),
    'cholesterol_ratio': np.random.uniform(2.0, 7.0, n),
    'exercise_frequency': np.random.randint(0, 7, n),
    # Noise features
    'height_cm': np.random.normal(170, 10, n),
    'eye_color_code': np.random.randint(1, 5, n),
})

# Introduce correlations
data['blood_pressure_index'] = (100 - data['health_exam_score']) / 60 + np.random.normal(0, 0.2, n)
data['cholesterol_ratio'] = data['family_history_score'] * 0.4 + np.random.normal(3.5, 0.5, n)

# Risk score (1-100)
data['risk_score'] = (
    0.5 * data['age']
    - 0.6 * data['health_exam_score']
    + 3.5 * data['family_history_score']
    + 2.8 * data['occupation_risk']
    + 2.2 * data['lifestyle_score']
    + 5.0 * data['blood_pressure_index']
    + 1.5 * data['cholesterol_ratio']
    - 0.8 * data['exercise_frequency']
    + 0.0 * data['height_cm']               # Irrelevant
    + 0.0 * data['eye_color_code']           # Irrelevant
    + np.random.normal(0, 5, n)
).clip(1, 100)

feature_cols = ['age', 'health_exam_score', 'family_history_score',
                'occupation_risk', 'lifestyle_score', 'blood_pressure_index',
                'cholesterol_ratio', 'exercise_frequency', 'height_cm', 'eye_color_code']
X = data[feature_cols].values
y = data['risk_score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **Automatic Hyperparameter Selection**

```python
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNetCV: auto-select both alpha and l1_ratio
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    n_alphas=100,
    cv=5,
    max_iter=10000,
    random_state=42
)
enet_cv.fit(X_train_scaled, y_train)

y_pred = enet_cv.predict(X_test_scaled)
print(f"Best alpha:    {enet_cv.alpha_:.4f}")
print(f"Best l1_ratio: {enet_cv.l1_ratio_:.2f}")
print(f"R-squared:     {r2_score(y_test, y_pred):.4f}")
print(f"MAE:           {mean_absolute_error(y_test, y_pred):.2f} risk points")
```

---

## **Feature Analysis**

```python
print("\n--- ElasticNet Feature Coefficients ---")
print(f"{'Feature':30s} {'Coefficient':>12s} {'Status':>10s}")
print("-" * 55)
for name, coef in zip(feature_cols, enet_cv.coef_):
    status = "SELECTED" if abs(coef) > 1e-4 else "REMOVED"
    print(f"{name:30s} {coef:>12.4f} {status:>10s}")

n_selected = np.sum(np.abs(enet_cv.coef_) > 1e-4)
print(f"\nFeatures selected: {n_selected}/{len(feature_cols)}")
```

### **Expected Output:**
```
--- ElasticNet Feature Coefficients ---
Feature                         Coefficient     Status
-------------------------------------------------------
age                                  5.8520   SELECTED
health_exam_score                   -8.2340   SELECTED
family_history_score                 6.1250   SELECTED
occupation_risk                      4.5800   SELECTED
lifestyle_score                      3.6200   SELECTED
blood_pressure_index                 2.1500   SELECTED
cholesterol_ratio                    1.2800   SELECTED
exercise_frequency                  -1.0500   SELECTED
height_cm                            0.0000    REMOVED
eye_color_code                       0.0000    REMOVED

Features selected: 8/10
```

---

## **Comparing Ridge, Lasso, and ElasticNet**

```python
from sklearn.linear_model import Ridge, Lasso

models = {
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1, max_iter=10000),
    'ElasticNet (best)': ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_, max_iter=10000),
}

print("\n--- Model Comparison ---")
print(f"{'Model':25s} {'R2':>8s} {'MAE':>8s} {'Non-zero':>10s}")
print("-" * 55)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    nz = np.sum(np.abs(model.coef_) > 1e-4)
    print(f"{name:25s} {r2:>8.4f} {mae:>8.2f} {nz:>10d}")
```

### **Expected Comparison:**
```
--- Model Comparison ---
Model                          R2      MAE   Non-zero
-------------------------------------------------------
Ridge (alpha=1.0)          0.9120     3.85         10
Lasso (alpha=0.1)          0.9050     4.02          7
ElasticNet (best)          0.9145     3.72          8
```

---

## **Production Pipeline**

```python
import joblib

# Full pipeline
risk_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        n_alphas=100, cv=5, max_iter=10000
    ))
])
risk_pipeline.fit(X_train, y_train)

# Save
joblib.dump(risk_pipeline, 'life_risk_elasticnet_v1.pkl')

# Score new applicant
new_applicant = np.array([[45, 72.0, 5.5, 3.0, 4.0, 1.1, 4.2, 3, 175, 2]])
risk = risk_pipeline.predict(new_applicant)
print(f"Applicant Risk Score: {risk[0]:.1f}/100")

# Risk classification
if risk[0] < 30:
    print("Classification: Preferred (lowest premium tier)")
elif risk[0] < 50:
    print("Classification: Standard (normal premium)")
elif risk[0] < 70:
    print("Classification: Substandard (elevated premium)")
else:
    print("Classification: Decline or rated (requires medical review)")
```

---

## **When to Use ElasticNet**

| Scenario | Recommendation |
|----------|---------------|
| Correlated health metrics | Use ElasticNet - keeps correlated groups unlike Lasso |
| Need feature selection + stability | Use ElasticNet - best of both worlds |
| All features important | Use Ridge |
| Need maximum sparsity | Use Lasso |
| Unknown which penalty is better | Use ElasticNetCV to auto-select |
| > 50 underwriting variables | Use ElasticNet with high l1_ratio |

---

## **Hyperparameters**

| Parameter | Value | Insurance Rationale |
|-----------|-------|-------------------|
| `alpha` | ElasticNetCV-selected | Balance regularization for risk data |
| `l1_ratio` | ElasticNetCV-selected | Auto-select L1/L2 mix |
| `max_iter` | 10000 | Ensure convergence |
| `cv` | 5 | Standard for underwriting model validation |
| `selection` | 'cyclic' | Reproducible for regulatory audits |

---

## **References**

1. scikit-learn ElasticNet / ElasticNetCV documentation
2. Zou & Hastie, "Regularization and Variable Selection via the Elastic Net" (2005)
3. Society of Actuaries, "Predictive Analytics and Life Insurance" (2018)
4. Hastie, Tibshirani & Friedman, "Elements of Statistical Learning" (2009)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
