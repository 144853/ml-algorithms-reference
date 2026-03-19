# ElasticNet (Scikit-Learn) - Patient Lifetime Dental Spend Prediction

## 🦷 **Use Case: Predicting Patient Lifetime Dental Spend**

Predict patient lifetime dental spend ($500-$50,000) from hygiene habits, diet score, genetics risk, visit frequency, and fluoride exposure using ElasticNet -- a combination of L1 (Lasso) and L2 (Ridge) regularization that handles correlated features while still performing feature selection.

---

## 📦 **Quick Start**

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Patient dataset: 700 patients, 5 features
X, y = load_lifetime_spend_data()  # y = lifetime dental spend in dollars

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
])

pipeline.fit(X_train, y_train)
print(f"R^2 Score: {pipeline.score(X_test, y_test):.4f}")
```

---

## 🔧 **Scikit-Learn API Details**

### **Key Parameters**
```python
ElasticNet(
    alpha=1.0,         # Overall regularization strength
    l1_ratio=0.5,      # Mix ratio: 0.0 = pure Ridge, 1.0 = pure Lasso, 0.5 = equal mix
    fit_intercept=True,
    max_iter=10000,    # Dental data may need more iterations
    tol=1e-4,
    warm_start=False,
    selection='cyclic',
    random_state=42
)
```

### **Understanding l1_ratio**
```
Loss = (1/2n) * ||Xw - y||^2 + alpha * (l1_ratio * ||w||_1 + (1-l1_ratio)/2 * ||w||_2^2)

l1_ratio = 0.0  -->  Pure Ridge (no feature selection)
l1_ratio = 0.5  -->  Equal L1 + L2 (balanced)
l1_ratio = 0.7  -->  More L1 (more feature selection)
l1_ratio = 1.0  -->  Pure Lasso (maximum sparsity)
```

### **ElasticNetCV: Automatic Tuning**
```python
from sklearn.linear_model import ElasticNetCV

enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
    alphas=None,        # Auto-generate alpha path for each l1_ratio
    cv=5,
    max_iter=10000,
    n_alphas=100,
    random_state=42
)

pipeline_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('enet_cv', enet_cv)
])

pipeline_cv.fit(X_train, y_train)
model = pipeline_cv.named_steps['enet_cv']
print(f"Best alpha:    {model.alpha_:.4f}")
print(f"Best l1_ratio: {model.l1_ratio_:.2f}")
# Best alpha:    0.0523
# Best l1_ratio: 0.70
```

---

## 🏥 **Dental Workflow Integration**

### **Feature Engineering**
```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

feature_names = ['hygiene_habits', 'diet_score', 'genetics_risk',
                 'visit_frequency', 'fluoride_exposure']

# Sample data
data = pd.DataFrame({
    'hygiene_habits': [8, 3, 9, 5, 7],      # 1-10 scale
    'diet_score': [6, 2, 8, 4, 7],           # 1-10 (10=healthiest)
    'genetics_risk': [0.3, 0.8, 0.1, 0.6, 0.4],  # 0-1 probability
    'visit_frequency': [2, 0.5, 4, 1, 2],     # visits per year
    'fluoride_exposure': [7, 2, 9, 4, 6],     # 1-10 index
    'lifetime_spend': [8500, 32000, 3200, 22000, 11000]  # dollars
})

dental_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('enet', ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, max_iter=10000))
])
```

### **Predicting for a New Patient**
```python
new_patient = pd.DataFrame({
    'hygiene_habits': [7],
    'diet_score': [6],
    'genetics_risk': [0.4],
    'visit_frequency': [2],
    'fluoride_exposure': [6]
})

predicted_spend = dental_pipeline.predict(new_patient[feature_names])
print(f"Estimated lifetime dental spend: ${predicted_spend[0]:,.2f}")
# Estimated lifetime dental spend: $12,450.00
```

---

## 📊 **Feature Importance Analysis**

```python
model = dental_pipeline.named_steps['enet']
coefs = model.coef_

print("Feature Importance (ElasticNet coefficients):")
for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
    status = "ACTIVE" if abs(coef) > 1e-4 else "ELIMINATED"
    print(f"  {name:20s}: ${coef:10,.2f}  [{status}]")

# genetics_risk       : $ 8,420.00  [ACTIVE]
# hygiene_habits      : $-3,150.00  [ACTIVE]
# diet_score          : $-2,840.00  [ACTIVE]
# visit_frequency     : $-1,200.00  [ACTIVE]
# fluoride_exposure   : $     0.00  [ELIMINATED]
```

### **Comparing ElasticNet vs Lasso vs Ridge**
```python
from sklearn.linear_model import Ridge, Lasso

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.7)
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    n_nonzero = np.sum(np.abs(pipe.named_steps['model'].coef_) > 1e-4)
    print(f"{name:12s}: R^2={score:.4f}, Active features={n_nonzero}/5")

# Ridge       : R^2=0.8650, Active features=5/5
# Lasso       : R^2=0.8580, Active features=3/5
# ElasticNet  : R^2=0.8720, Active features=4/5
```

---

## 📈 **Model Evaluation**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = pipeline_cv.predict(X_test)

print(f"R^2 Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:       ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f"MAE:        ${mean_absolute_error(y_test, y_pred):,.2f}")

# R^2 Score:  0.8720
# RMSE:       $2,845.30
# MAE:        $2,180.50

# Dental-specific: percentage error
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE:       {mape:.1f}%")
# MAPE:       11.2%
```

---

## 💾 **Model Persistence**

```python
import joblib

joblib.dump(pipeline_cv, 'lifetime_spend_predictor.pkl')
loaded_model = joblib.load('lifetime_spend_predictor.pkl')
```

---

## 🎯 **Production Tips**

1. **Use ElasticNetCV** -- it searches both alpha and l1_ratio simultaneously
2. **ElasticNet handles correlated features better than Lasso** -- hygiene and diet often correlate
3. **Start with l1_ratio=0.5** -- then let CV optimize the mix
4. **Increase max_iter to 10000+** -- dental spend has wide range ($500-$50K) which may slow convergence
5. **Consider log-transforming the target** -- dental spend is often right-skewed

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/elasticnet_sklearn.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
