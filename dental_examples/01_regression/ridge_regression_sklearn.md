# Ridge Regression (Scikit-Learn) - Orthodontic Treatment Duration Prediction

## 🦷 **Use Case: Predicting Orthodontic Treatment Duration**

Predict treatment duration (6-48 months) from malocclusion severity, patient age, compliance score, and bracket type using scikit-learn's Ridge with built-in cross-validation.

---

## 📦 **Quick Start**

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Orthodontic dataset: 600 patients, 4 features
X, y = load_orthodontic_data()  # y = treatment duration in months

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
print(f"R^2 Score: {pipeline.score(X_test, y_test):.4f}")
```

---

## 🔧 **Scikit-Learn API Details**

### **Key Parameters**
```python
Ridge(
    alpha=1.0,            # Regularization strength (higher = more shrinkage)
    fit_intercept=True,   # Include bias term
    solver='auto',        # 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
    max_iter=None,        # Max iterations for iterative solvers
    tol=1e-4,             # Convergence tolerance
    random_state=42       # For 'sag' and 'saga' solvers
)
```

### **RidgeCV: Built-in Cross-Validation**
```python
from sklearn.linear_model import RidgeCV

# Automatically find the best alpha
ridge_cv = RidgeCV(
    alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
    cv=5,
    scoring='neg_mean_squared_error'
)

pipeline_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge_cv', ridge_cv)
])

pipeline_cv.fit(X_train, y_train)
best_alpha = pipeline_cv.named_steps['ridge_cv'].alpha_
print(f"Best alpha: {best_alpha}")
# Best alpha: 1.0
```

---

## 🏥 **Dental Workflow Integration**

### **Feature Engineering for Orthodontic Data**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

categorical_features = ['bracket_type']
numerical_features = ['malocclusion_severity', 'patient_age', 'compliance_score']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

ortho_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5))
])

ortho_pipeline.fit(X_train, y_train)
```

### **Predicting for a New Patient**
```python
import pandas as pd

new_patient = pd.DataFrame({
    'malocclusion_severity': [7],
    'patient_age': [14],
    'compliance_score': [85],
    'bracket_type': ['ceramic']
})

predicted_duration = ortho_pipeline.predict(new_patient)
print(f"Estimated treatment duration: {predicted_duration[0]:.1f} months")
# Estimated treatment duration: 22.5 months
```

---

## 📊 **Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'ridge__alpha': np.logspace(-3, 3, 20),
}

simple_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

grid_search = GridSearchCV(
    simple_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

grid_search.fit(X_train, y_train)
print(f"Best alpha: {grid_search.best_params_['ridge__alpha']:.4f}")
print(f"Best CV RMSE: {(-grid_search.best_score_)**0.5:.2f} months")
```

### **Coefficient Path Analysis**
```python
import matplotlib.pyplot as plt

alphas = np.logspace(-3, 3, 50)
coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
for i, name in enumerate(['severity', 'age', 'compliance', 'bracket']):
    plt.plot(alphas, [c[i] for c in coefs], label=name)
plt.xscale('log')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Coefficient value')
plt.title('Ridge Coefficient Shrinkage Path')
plt.legend()
plt.show()
```

---

## 📈 **Model Evaluation**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = ortho_pipeline.predict(X_test)

print(f"R^2 Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:       {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} months")
print(f"MAE:        {mean_absolute_error(y_test, y_pred):.2f} months")

# R^2 Score:  0.8721
# RMSE:       3.02 months
# MAE:        2.34 months
```

---

## 💾 **Model Persistence**

```python
import joblib

joblib.dump(ortho_pipeline, 'orthodontic_duration_predictor.pkl')
loaded_model = joblib.load('orthodontic_duration_predictor.pkl')
```

---

## 🎯 **Production Tips for Orthodontic Practice**

1. **Use RidgeCV** over manual alpha tuning -- it's faster and more reliable
2. **Monitor compliance drift** -- patient compliance changes over treatment; retrain periodically
3. **Stratify by age group** -- pediatric vs adult orthodontics may need separate models
4. **Add interaction features** -- severity x compliance interaction may be clinically meaningful
5. **Report prediction intervals** -- patients want ranges, not point estimates

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/ridge_regression_sklearn.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
