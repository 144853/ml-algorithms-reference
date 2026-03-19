# MLP Regressor (Scikit-Learn) - Dental Crown Longevity Prediction

## 🦷 **Use Case: Predicting Dental Crown Longevity**

Predict dental crown longevity (2-25 years) from material type, bite force, oral pH, grinding habit, and crown position using scikit-learn's MLPRegressor with non-linear feature interaction modeling.

---

## 📦 **Quick Start**

```python
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Crown dataset: 800 patients, 5 features
X, y = load_crown_data()  # y = longevity in years

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        max_iter=1000,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
print(f"R^2 Score: {pipeline.score(X_test, y_test):.4f}")
```

---

## 🔧 **Scikit-Learn API Details**

### **Key Parameters**
```python
MLPRegressor(
    hidden_layer_sizes=(16, 8),    # Two hidden layers: 16 and 8 neurons
    activation='relu',              # 'relu', 'tanh', 'logistic', 'identity'
    solver='adam',                  # 'adam', 'sgd', 'lbfgs'
    alpha=0.0001,                   # L2 regularization strength
    batch_size='auto',              # Min(200, n_samples) for 'adam'/'sgd'
    learning_rate='constant',       # 'constant', 'invscaling', 'adaptive'
    learning_rate_init=0.001,       # Initial learning rate
    max_iter=1000,                  # Maximum epochs
    early_stopping=True,            # Stop when validation score plateaus
    validation_fraction=0.1,        # 10% held out for early stopping
    n_iter_no_change=20,            # Patience for early stopping
    random_state=42
)
```

### **Attributes After Fitting**
```python
model = pipeline.named_steps['mlp']
print(f"Number of layers: {model.n_layers_}")
print(f"Layer sizes: {[layer.shape for layer in model.coefs_]}")
print(f"Training loss: {model.loss_:.4f}")
print(f"Iterations: {model.n_iter_}")
print(f"Best validation score: {model.best_validation_score_:.4f}")
```

---

## 🏥 **Dental Workflow Integration**

### **Feature Engineering**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_features = ['material_type', 'crown_position']
numerical_features = ['bite_force', 'oral_ph', 'grinding_habit']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

crown_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        early_stopping=True,
        max_iter=1000,
        random_state=42
    ))
])

crown_pipeline.fit(X_train, y_train)
```

### **Predicting for a New Patient**
```python
import pandas as pd

new_patient = pd.DataFrame({
    'material_type': ['zirconia'],
    'bite_force': [150],
    'oral_ph': [7.2],
    'grinding_habit': [0],
    'crown_position': ['premolar']
})

predicted_years = crown_pipeline.predict(new_patient)
print(f"Estimated crown longevity: {predicted_years[0]:.1f} years")
# Estimated crown longevity: 21.4 years
```

---

## 📊 **Hyperparameter Tuning with GridSearchCV**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'mlp__hidden_layer_sizes': [(16, 8), (32, 16), (64, 32), (32, 16, 8)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.01],
    'mlp__solver': ['adam'],
}

grid_search = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(max_iter=1000, random_state=42))]),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV RMSE: {(-grid_search.best_score_)**0.5:.2f} years")
```

---

## 📈 **Model Evaluation**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = crown_pipeline.predict(X_test)

print(f"R^2 Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:       {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} years")
print(f"MAE:        {mean_absolute_error(y_test, y_pred):.2f} years")

# R^2 Score:  0.9215
# RMSE:       1.70 years
# MAE:        1.25 years

# Compare with linear regression
from sklearn.linear_model import LinearRegression
lr_pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
lr_r2 = lr_pipeline.score(X_test, y_test)
print(f"\nLinear Regression R^2: {lr_r2:.4f}")
print(f"MLP improvement: +{(r2_score(y_test, y_pred) - lr_r2)*100:.1f}% R^2")
```

### **Learning Curve**
```python
import matplotlib.pyplot as plt

model = crown_pipeline.named_steps['mlp']
plt.plot(model.loss_curve_)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('MLP Training Loss for Crown Longevity')
plt.show()
```

---

## 💾 **Model Persistence**

```python
import joblib

joblib.dump(crown_pipeline, 'crown_longevity_mlp.pkl')
loaded_model = joblib.load('crown_longevity_mlp.pkl')
```

---

## 🎯 **Production Tips for Dental Labs**

1. **Always scale features** -- MLPRegressor is sensitive to feature scales (bite force 50-400 vs pH 5-8)
2. **Use early_stopping=True** -- prevents overfitting on small dental datasets
3. **Start with (16, 8)** -- for 5 features, deeper networks risk overfitting with 800 samples
4. **Adam solver** is recommended for dental data -- more robust than SGD for small datasets
5. **Monitor loss_curve_** -- if not converging, increase max_iter or adjust learning_rate_init

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/mlp_regressor_sklearn.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
