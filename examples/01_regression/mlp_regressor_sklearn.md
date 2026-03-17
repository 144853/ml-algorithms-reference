# MLP Regressor - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of MLP Regressor (Multi-Layer Perceptron). For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[MLP Regressor - Full Documentation](mlp_regressor_numpy.md)**

---

## 🚀 **Scikit-learn Advantages**

- **Multiple solvers**: adam, sgd, lbfgs (auto-selects based on dataset size)
- **Early stopping**: Automatic validation-based stopping
- **Learning rate schedules**: constant, invscaling, adaptive
- **Minimal setup**: No manual batch handling or training loops
- **Production-ready**: Integrated with sklearn pipelines

---

## 💻 **Quick Start**

```python
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize features (IMPORTANT for neural networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create MLP with two hidden layers
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),  # 2 layers: 64 and 32 neurons
    activation='relu',
    solver='adam',
    alpha=0.0001,  # L2 regularization
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,  # Use validation set to stop early
    random_state=42
)

# Train
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
print(f"Training iterations: {model.n_iter_}")
```

---

## 🔧 **Key Parameters**

```python
MLPRegressor(
    hidden_layer_sizes=(100,),  # Architecture: (100,) = 1 layer, (64,32) = 2 layers
    activation='relu',          # 'relu', 'tanh', 'logistic', 'identity'
    solver='adam',              # 'adam', 'sgd', 'lbfgs'
    alpha=0.0001,               # L2 regularization strength
    batch_size='auto',          # Mini-batch size (default: min(200, n_samples))
    learning_rate='constant',   # 'constant', 'invscaling', 'adaptive'
    learning_rate_init=0.001,   # Initial learning rate
    max_iter=200,               # Maximum epochs
    shuffle=True,               # Shuffle train data each epoch
    random_state=None,          
    tol=1e-4,                   # Optimization tolerance
    early_stopping=False,       # Use validation set for early stopping
    validation_fraction=0.1,    # Fraction for validation (if early_stopping=True)
    n_iter_no_change=10         # Stop if no improvement for N iterations
)
```

---

## 🎯 **Architecture Selection**

### **Common Architectures:**
```python
# Small network (simple patterns)
model = MLPRegressor(hidden_layer_sizes=(32,))

# Medium network (default - good starting point)
model = MLPRegressor(hidden_layer_sizes=(64, 32))

# Large network (complex patterns)
model = MLPRegressor(hidden_layer_sizes=(128, 64, 32))

# Deep network (very complex patterns)
model = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32))
```

### **Rule of Thumb:**
- Start with `(64, 32)`
- Neurons between input size and output size
- More layers = more capacity (but risk overfitting)

---

## 📊 **Solver Comparison**

### **'adam' (Adaptive Moment Estimation) - RECOMMENDED**
```python
model = MLPRegressor(solver='adam', learning_rate_init=0.001)
```
- **Best for**: Most cases (default choice)
- **Pros**: Fast, robust, works well out-of-the-box
- **Cons**: Can overfit on small datasets

### **'lbfgs' (L-BFGS Quasi-Newton)**
```python
model = MLPRegressor(solver='lbfgs', max_iter=500)
```
- **Best for**: Small datasets (n < 10,000)
- **Pros**: Faster convergence for small data
- **Cons**: Doesn't scale to large datasets (memory-intensive)

### **'sgd' (Stochastic Gradient Descent)**
```python
model = MLPRegressor(
    solver='sgd', 
    learning_rate='adaptive',  # Decrease lr when loss plateaus
    momentum=0.9,
    nesterovs_momentum=True
)
```
- **Best for**: Fine-grained control, large datasets
- **Pros**: Most flexible, can escape local minima
- **Cons**: Requires more tuning

---

## 🔧 **Hyperparameter Tuning**

### **Grid Search**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(32,), (64, 32), (128, 64), (128, 64, 32)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(
    MLPRegressor(max_iter=500, early_stopping=True),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_:.4f}")
```

### **Random Search (faster)**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'hidden_layer_sizes': [(32,), (64,32), (128,64), (128,64,32)],
    'activation': ['relu', 'tanh'],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate_init': uniform(0.0001, 0.01)
}

random_search = RandomizedSearchCV(
    MLPRegressor(max_iter=500, early_stopping=True),
    param_dist,
    n_iter=20,  # 20 random combinations
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

---

## 📈 **Training Curves & Diagnostics**

### **Loss Curves**
```python
model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
model.fit(X_train, y_train)

import matplotlib.pyplot as plt
plt.plot(model.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

print(f"Final loss: {model.loss_:.4f}")
print(f"Iterations: {model.n_iter_}")
```

### **Early Stopping**
```python
model = MLPRegressor(
    hidden_layer_sizes=(64,32),
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,  # Stop if no improvement for 20 epochs
    max_iter=1000
)

model.fit(X_train, y_train)

# Access validation scores
plt.plot(model.loss_curve_, label='Training')
plt.plot(model.validation_scores_, label='Validation')
plt.legend()
plt.show()
```

---

## 🎯 **Preventing Overfitting**

### **1. Early Stopping (recommended)**
```python
model = MLPRegressor(early_stopping=True, validation_fraction=0.2)
```

### **2. L2 Regularization**
```python
model = MLPRegressor(alpha=0.01)  # Increase alpha for more regularization
```

### **3. Smaller Architecture**
```python
model = MLPRegressor(hidden_layer_sizes=(32,))  # Fewer neurons
```

### **4. Dropout (Not available in sklearn - use PyTorch)**

---

## ⚖️ **MLP vs Linear Models**

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'MLP (small)': MLPRegressor(hidden_layer_sizes=(32,)),
    'MLP (medium)': MLPRegressor(hidden_layer_sizes=(64,32)),
    'MLP (large)': MLPRegressor(hidden_layer_sizes=(128,64,32))
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"{name:15s} Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
```

---

## 📝 **Code Reference**

Full implementation: [`01_regression/mlp_regressor_sklearn.py`](../../01_regression/mlp_regressor_sklearn.py)

Related:
- [MLP Regressor - NumPy (from scratch)](mlp_regressor_numpy.md)
- [MLP Regressor - PyTorch](mlp_regressor_pytorch.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
