# ElasticNet Regression - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of ElasticNet Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[ElasticNet Regression - Full Documentation](elasticnet_numpy.md)**

---

## 🚀 **Scikit-learn Advantages**

- **ElasticNetCV**: Automatic tuning of both alpha and l1_ratio
- **Optimized coordinate descent**: Fast, production-ready
- **Regularization paths**: Efficient computation across parameter grids
- **Best of both worlds**: Combines Lasso's sparsity + Ridge's stability

---

## 💻 **Quick Start**

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and standardize data (IMPORTANT!)
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Option 1: Fixed hyperparameters
model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_scaled, y_train)

# Option 2: ElasticNetCV with automatic tuning (recommended!)
model_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],  # L1/L2 mix
    alphas=None,  # Auto-generate
    cv=5,
    max_iter=10000
)
model_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {model_cv.alpha_}")
print(f"Best l1_ratio: {model_cv.l1_ratio_}")
print(f"Selected {np.sum(model_cv.coef_ != 0)}/{len(model_cv.coef_)} features")
```

---

## 🔧 **Key Parameters**

```python
ElasticNet(
    alpha=1.0,              # Overall regularization strength
    l1_ratio=0.5,           # L1/L2 mix (0=Ridge, 1=Lasso, 0.5=balanced)
    fit_intercept=True,     
    max_iter=1000,          # Increase for high-dim data
    tol=1e-4,               
    positive=False,         
    selection='cyclic'      # 'cyclic' or 'random'
)
```

### **l1_ratio Interpretation:**
- **l1_ratio = 0.0**: Pure Ridge (no sparsity)
- **l1_ratio = 0.5**: Balanced ElasticNet (RECOMMENDED starting point)
- **l1_ratio = 0.9**: Mostly Lasso (aggressive sparsity)
- **l1_ratio = 0.99**: Nearly pure Lasso (maximum sparsity)
- **l1_ratio = 1.0**: Pure Lasso

---

## 📊 **Automatic Hyperparameter Tuning**

### **ElasticNetCV - Two-Dimensional Grid Search**
```python
from sklearn.linear_model import ElasticNetCV

# Define grid
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
alphas = np.logspace(-3, 2, 50)  # [0.001, ..., 100]

model = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=alphas,
    cv=5,
    max_iter=10000,
    n_jobs=-1  # Use all CPU cores
)

model.fit(X_train, y_train)

print(f"Optimal alpha: {model.alpha_:.4f}")
print(f"Optimal l1_ratio: {model.l1_ratio_:.4f}")

# Access full CV results
print(f"MSE path shape: {model.mse_path_.shape}")  # (l1_ratios, alphas, n_folds)
```

---

## 🎯 **Use Case: Finding Correlated Groups**

ElasticNet's strength is selecting features while keeping correlated groups:

```python
# Train ElasticNet
model = ElasticNetCV(l1_ratio=0.5, cv=5, max_iter=10000)
model.fit(X_train, y_train)

# Find selected features
selected_mask = model.coef_ != 0
selected_indices = np.where(selected_mask)[0]

print(f"Selected {len(selected_indices)} features:")
print(selected_indices)

# Analyze correlations among selected features
selected_corr = np.corrcoef(X_train[:, selected_mask].T)
print(f"Mean abs correlation among selected: {np.abs(selected_corr).mean():.3f}")
```

---

## 📈 **Comparing ElasticNet vs Lasso vs Ridge**

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

models = {
    'Lasso': Lasso(alpha=1.0),
    'Ridge': Ridge(alpha=1.0),
    'ElasticNet (0.5)': ElasticNet(alpha=1.0, l1_ratio=0.5),
    'ElasticNet (0.9)': ElasticNet(alpha=1.0, l1_ratio=0.9)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    n_selected = np.sum(model.coef_ != 0)
    
    print(f"{name:20s} RMSE: {rmse:.4f}, Features: {n_selected}")
```

---

## ⚖️ **ElasticNet vs Lasso vs Ridge**

| Property | Ridge | Lasso | ElasticNet |
|----------|-------|-------|------------|
| Sparsity | No | Yes | Yes ✅ |
| Grouped selection | No | No | Yes ✅ |
| Correlated features | Keeps all | Arbitrary pick | Keeps groups ✅ |
| Tuning | 1 param (α) | 1 param (α) | 2 params (α, l1_ratio) |

---

## 📝 **Code Reference**

Full implementation: [`01_regression/elasticnet_sklearn.py`](../../01_regression/elasticnet_sklearn.py)

Related:
- [ElasticNet - NumPy (from scratch)](elasticnet_numpy.md)
- [ElasticNet - PyTorch](elasticnet_pytorch.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
