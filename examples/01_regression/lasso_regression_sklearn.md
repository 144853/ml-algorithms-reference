# Lasso Regression - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of Lasso Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Lasso Regression - Full Documentation](lasso_regression_numpy.md)**

---

## 🚀 **Scikit-learn Advantages**

- **LassoCV**: Automatic alpha selection via cross-validation
- **Optimized coordinate descent**: Fast Cython implementation
- **Regularization paths**: Efficient `lasso_path` for multiple alphas
- **Selection strategies**: Cyclic or random feature updates
- **Warm starts**: Reuse previous solution for speed

---

## 💻 **Quick Start**

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and standardize data (IMPORTANT for Lasso!)
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Option 1: Lasso with fixed alpha
model = Lasso(alpha=0.1, max_iter=10000)
model.fit(X_train_scaled, y_train)

# Option 2: LassoCV with automatic alpha selection (recommended!)
model_cv = LassoCV(alphas=None, cv=5, max_iter=10000)  # alphas=None: auto-generate
model_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {model_cv.alpha_}")

# Check sparsity
n_nonzero = np.sum(model_cv.coef_ != 0)
print(f"Selected {n_nonzero}/{len(model_cv.coef_)} features")

# Predictions
y_pred = model_cv.predict(X_test_scaled)
```

---

## 🔧 **Key Parameters**

```python
Lasso(
    alpha=1.0,              # L1 penalty strength (KEY PARAMETER)
    fit_intercept=True,     
    max_iter=1000,          # Increase if convergence warning
    tol=1e-4,               # Convergence tolerance
    positive=False,         # Force positive coefficients
    selection='cyclic',     # 'cyclic' or 'random' feature update order
    warm_start=False        # Reuse previous solution
)
```

---

## 📊 **Feature Selection with Lasso**

### **Get Selected Features**
```python
# Train LassoCV
model = LassoCV(cv=5, max_iter=10000)
model.fit(X_train, y_train)

# Get non-zero coefficients
nonzero_mask = model.coef_ != 0
selected_features = np.where(nonzero_mask)[0]
selected_coefs = model.coef_[nonzero_mask]

print(f"Selected {len(selected_features)} features:")
for idx, coef in zip(selected_features, selected_coefs):
    print(f"  Feature {idx}: {coef:.4f}")

# Retrain on selected features only (optional)
X_train_selected = X_train[:, nonzero_mask]
X_test_selected = X_test[:, nonzero_mask]
```

### **Regularization Path Visualization**
```python
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X_train, y_train, alphas=50)

# Plot coefficient paths
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs.T)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso Regularization Path')
plt.show()
```

---

## 🎯 **Automatic Alpha Selection**

### **LassoCV - Cross-Validated Selection**
```python
from sklearn.linear_model import LassoCV

# Auto-generate alphas (good default)
model = LassoCV(cv=5, max_iter=10000, n_alphas=100)
model.fit(X_train, y_train)

print(f"Optimal alpha: {model.alpha_}")
print(f"Tried {len(model.alphas_)} alpha values")

# Access CV scores for all alphas
print(f"MSE path shape: {model.mse_path_.shape}")  # (n_alphas, n_folds)
mean_mse = model.mse_path_.mean(axis=1)
```

### **Manual Alpha Grid**
```python
alphas = np.logspace(-4, 1, 50)  # [0.0001, ..., 10]
model = LassoCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)
```

---

## 📈 **High-Dimensional Feature Selection**

For datasets with p >> n (more features than samples):

```python
from sklearn.linear_model import LassoLarsCV

# LARS variant: efficient for very sparse solutions
model = LassoLarsCV(cv=5, max_iter=500)
model.fit(X_train, y_train)

print(f"Selected {np.sum(model.coef_ != 0)} features")
```

---

## ⚖️ **Lasso vs LassoCV vs LassoLarsCV**

| Model | Best For | Speed |
|-------|----------|-------|
| Lasso | Known alpha | Fast ✅ |
| LassoCV | Alpha tuning | Medium |
| LassoLarsCV | Very sparse (p >> n) | Medium |

---

## 📝 **Code Reference**

Full implementation: [`01_regression/lasso_regression_sklearn.py`](../../01_regression/lasso_regression_sklearn.py)

Related:
- [Lasso Regression - NumPy (from scratch)](lasso_regression_numpy.md)
- [Lasso Regression - PyTorch](lasso_regression_pytorch.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
