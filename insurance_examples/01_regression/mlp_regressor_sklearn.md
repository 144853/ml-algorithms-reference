# MLP Regressor (scikit-learn) - Insurance Customer Lifetime Value

## **Use Case: Predicting Insurance Customer Lifetime Value (CLV)**

### **The Problem**
An insurance company uses scikit-learn's `MLPRegressor` to predict Customer Lifetime Value. The model captures non-linear relationships between customer behavior features and long-term value. Features: policy count, tenure years, claim frequency, premium tier, cross-sell score. Target: CLV ($500 - $50,000).

---

## **scikit-learn API Overview**

```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # Two hidden layers: 32 and 16 neurons
    activation='relu',             # 'relu', 'tanh', 'logistic', 'identity'
    solver='adam',                 # 'adam', 'sgd', 'lbfgs'
    alpha=0.0001,                  # L2 regularization strength
    batch_size='auto',             # Min(200, n_samples)
    learning_rate='adaptive',      # 'constant', 'invscaling', 'adaptive'
    learning_rate_init=0.001,      # Initial learning rate
    max_iter=500,                  # Maximum epochs
    early_stopping=True,           # Use validation set for early stopping
    validation_fraction=0.1,       # 10% held out for validation
    n_iter_no_change=20,           # Patience for early stopping
    random_state=42,
)
```

---

## **Insurance Workflow Implementation**

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --- Generate insurance CLV data ---
np.random.seed(42)
n = 5000

data = pd.DataFrame({
    'policy_count': np.random.randint(1, 9, n),
    'tenure_years': np.random.uniform(0.5, 30, n),
    'claim_frequency': np.random.exponential(0.5, n).clip(0, 5),
    'premium_tier': np.random.randint(1, 5, n),
    'cross_sell_score': np.random.uniform(0, 10, n),
})

# Non-linear CLV with interactions and diminishing returns
data['clv'] = (
    500
    + 1200 * data['policy_count']
    + 800 * np.log1p(data['tenure_years'])
    + 1500 * data['premium_tier'] ** 1.3
    - 2000 * np.sqrt(data['claim_frequency'])
    + 300 * data['cross_sell_score'] * np.log1p(data['tenure_years'])
    + np.random.normal(0, 1500, n)
).clip(500, 50000)

feature_cols = ['policy_count', 'tenure_years', 'claim_frequency',
                'premium_tier', 'cross_sell_score']
X = data[feature_cols].values
y = data['clv'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **Pipeline with Scaling**

```python
# MLP requires feature scaling for stable training
clv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    ))
])

clv_pipeline.fit(X_train, y_train)
y_pred = clv_pipeline.predict(X_test)

print(f"R-squared:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE:        ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Iterations: {clv_pipeline.named_steps['mlp'].n_iter_}")
```

---

## **Hyperparameter Tuning**

```python
# Grid search for optimal architecture and regularization
param_grid = {
    'mlp__hidden_layer_sizes': [(16,), (32,), (32, 16), (64, 32)],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.005],
    'mlp__activation': ['relu', 'tanh'],
}

grid_search = GridSearchCV(
    clv_pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV MAE: ${-grid_search.best_score_:,.2f}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Test R-squared: {r2_score(y_test, y_pred_best):.4f}")
print(f"Test MAE:       ${mean_absolute_error(y_test, y_pred_best):,.2f}")
```

---

## **Model Comparison: Linear vs. MLP**

```python
from sklearn.linear_model import LinearRegression, Ridge

models = {
    'Linear Regression': Pipeline([('scaler', StandardScaler()),
                                    ('model', LinearRegression())]),
    'Ridge (alpha=1.0)': Pipeline([('scaler', StandardScaler()),
                                    ('model', Ridge(alpha=1.0))]),
    'MLP (32,)':         Pipeline([('scaler', StandardScaler()),
                                    ('model', MLPRegressor(hidden_layer_sizes=(32,),
                                    max_iter=500, early_stopping=True, random_state=42))]),
    'MLP (64, 32)':      Pipeline([('scaler', StandardScaler()),
                                    ('model', MLPRegressor(hidden_layer_sizes=(64, 32),
                                    max_iter=500, early_stopping=True, random_state=42))]),
}

print("\n--- Model Comparison ---")
print(f"{'Model':25s} {'R2':>8s} {'MAE':>10s}")
print("-" * 45)
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"{name:25s} {r2:>8.4f} ${mae:>9,.2f}")
```

### **Expected Comparison:**
```
--- Model Comparison ---
Model                          R2        MAE
---------------------------------------------
Linear Regression          0.8250  $2,850.40
Ridge (alpha=1.0)          0.8245  $2,860.15
MLP (32,)                  0.9180  $1,720.30
MLP (64, 32)               0.9320  $1,540.80
```

---

## **Customer Segmentation by CLV**

```python
# Score all customers and segment
all_pred = clv_pipeline.predict(X)

segments = pd.DataFrame({
    'predicted_clv': all_pred,
    'policy_count': data['policy_count'],
    'tenure_years': data['tenure_years'],
})

segments['tier'] = pd.cut(all_pred,
                          bins=[0, 5000, 15000, 30000, 50000],
                          labels=['Bronze', 'Silver', 'Gold', 'Platinum'])

print("\n--- CLV Segments ---")
print(segments.groupby('tier')[['predicted_clv', 'policy_count', 'tenure_years']].mean().round(2))
```

---

## **Production Deployment**

```python
import joblib

# Save pipeline
joblib.dump(clv_pipeline, 'insurance_clv_mlp_v1.pkl')

# Score new customer
new_customer = np.array([[3, 7.5, 0.3, 3, 7.0]])
predicted_clv = clv_pipeline.predict(new_customer)
print(f"Predicted CLV: ${predicted_clv[0]:,.2f}")

# Batch scoring for marketing campaign
campaign_targets = np.array([
    [1, 1.0, 0.5, 1, 3.0],
    [2, 5.0, 0.2, 2, 6.0],
    [4, 12.0, 0.1, 3, 8.5],
    [6, 20.0, 0.0, 4, 9.5],
])
batch_clv = clv_pipeline.predict(campaign_targets)
for i, clv in enumerate(batch_clv):
    print(f"  Customer {i+1}: ${clv:,.2f}")
```

---

## **Learning Curve Analysis**

```python
# Check if more data would help
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    clv_pipeline, X, y,
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
    cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
)

print("\n--- Learning Curve ---")
for size, train_s, val_s in zip(train_sizes, -train_scores.mean(axis=1), -val_scores.mean(axis=1)):
    print(f"  n={size:>5d} | Train MAE: ${train_s:,.2f} | Val MAE: ${val_s:,.2f}")
```

---

## **When to Use sklearn MLPRegressor**

| Scenario | Recommendation |
|----------|---------------|
| Non-linear CLV patterns | Use MLP - captures interactions and curves |
| Small dataset (< 500) | Use linear models instead |
| Need feature importance | Use tree-based models (XGBoost, LightGBM) |
| Quick prototyping | Use MLP - simple API |
| GPU needed | Switch to PyTorch implementation |
| Interpretability required | Use linear models for regulators |

---

## **Hyperparameters**

| Parameter | Value | Insurance Rationale |
|-----------|-------|-------------------|
| `hidden_layer_sizes` | (64, 32) | Captures CLV non-linearities without overfitting |
| `activation` | 'relu' | Fast training, handles positive CLV values well |
| `alpha` | 0.001 | Prevents memorizing individual customer patterns |
| `early_stopping` | True | Prevents overfitting on customer data |
| `solver` | 'adam' | Best convergence for medium-sized datasets |
| `max_iter` | 500 | Sufficient with early stopping |

---

## **References**

1. scikit-learn MLPRegressor documentation
2. Fader & Hardie, "CLV in Contractual Settings" (2010)
3. Goodfellow et al., "Deep Learning" (2016), Ch. 6
4. Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization" (2012)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
