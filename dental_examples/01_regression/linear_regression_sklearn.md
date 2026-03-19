# Linear Regression (Scikit-Learn) - Dental Treatment Cost Prediction

## 🦷 **Use Case: Predicting Dental Treatment Costs**

Predict treatment costs ($200-$5000) from procedure type, tooth count, patient age, insurance tier, and clinic location using scikit-learn's production-ready API.

---

## 📦 **Quick Start**

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Dental cost dataset: 500 patients, 5 features
# Features: procedure_type, tooth_count, patient_age, insurance_tier, clinic_location
X, y = load_dental_cost_data()  # y = treatment cost in dollars

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline with scaling + regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
print(f"R^2 Score: {pipeline.score(X_test, y_test):.4f}")
```

---

## 🔧 **Scikit-Learn API Details**

### **Key Parameters**
```python
LinearRegression(
    fit_intercept=True,   # Include bias term (recommended for cost prediction)
    copy_X=True,          # Don't modify original dental data
    n_jobs=-1,            # Parallelize across CPU cores
    positive=False        # Set True to force positive coefficients
)
```

### **Attributes After Fitting**
```python
model = pipeline.named_steps['regressor']
print("Coefficients:", model.coef_)
# [618.42, 84.57, 11.93, -178.65, 93.21]
# procedure_type, tooth_count, age, insurance_tier, clinic_location

print("Intercept:", model.intercept_)
# 152.30 (base cost in dollars)
```

---

## 🏥 **Dental Workflow Integration**

### **Feature Engineering for Dental Data**
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Define feature types
categorical_features = ['procedure_type', 'insurance_tier', 'clinic_location']
numerical_features = ['tooth_count', 'patient_age']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

dental_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

dental_pipeline.fit(X_train, y_train)
```

### **Predicting for a New Patient**
```python
import pandas as pd

new_patient = pd.DataFrame({
    'procedure_type': ['root_canal'],
    'tooth_count': [3],
    'patient_age': [52],
    'insurance_tier': ['gold'],
    'clinic_location': ['urban']
})

predicted_cost = dental_pipeline.predict(new_patient)
print(f"Estimated treatment cost: ${predicted_cost[0]:,.2f}")
# Estimated treatment cost: $2,415.80
```

---

## 📊 **Hyperparameter Tuning with GridSearchCV**

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso

# Compare linear regression variants for dental cost prediction
param_grid = {}  # LinearRegression has no tunable hyperparameters

# Use cross-validation instead
cv_scores = cross_val_score(dental_pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"CV R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
# CV R^2: 0.9134 (+/- 0.0182)

cv_rmse = cross_val_score(dental_pipeline, X_train, y_train, cv=5,
                          scoring='neg_root_mean_squared_error')
print(f"CV RMSE: ${-cv_rmse.mean():,.2f} (+/- ${cv_rmse.std():,.2f})")
# CV RMSE: $218.45 (+/- $32.10)
```

---

## 📈 **Model Evaluation for Dental Domain**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = dental_pipeline.predict(X_test)

print(f"R^2 Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:       ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f"MAE:        ${mean_absolute_error(y_test, y_pred):,.2f}")

# R^2 Score:  0.9087
# RMSE:       $221.15
# MAE:        $168.42

# Dental-specific: percentage error relative to cost
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE:       {mape:.1f}%")
# MAPE:       8.3%
```

### **Residual Analysis**
```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Cost ($)')
plt.ylabel('Residual ($)')
plt.title('Residual Plot: Dental Treatment Cost Prediction')
plt.show()
```

---

## 💾 **Model Persistence**

```python
import joblib

# Save the trained dental cost model
joblib.dump(dental_pipeline, 'dental_cost_predictor.pkl')

# Load for production use
loaded_model = joblib.load('dental_cost_predictor.pkl')
cost_estimate = loaded_model.predict(new_patient)
```

---

## 🎯 **Production Tips for Dental Clinics**

1. **Retrain quarterly** -- dental material costs and insurance rates change
2. **Monitor for drift** -- track prediction errors over time
3. **Handle missing insurance** -- use imputation for uninsured patients
4. **Log predictions** -- store estimates for audit and model improvement
5. **Set confidence intervals** -- provide cost ranges, not point estimates

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/linear_regression_sklearn.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
