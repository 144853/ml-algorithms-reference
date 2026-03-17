# Ridge Regression - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of Ridge Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Ridge Regression - Full Documentation](ridge_regression_numpy.md)**

---

## 🚀 **Scikit-learn Advantages**

- **Multiple solvers**: auto, svd, cholesky, lsqr, sparse_cg, sag, saga
- **Built-in CV**: `RidgeCV` for automatic alpha tuning
- **Production-ready**: Highly optimized, battle-tested
- **Regularization path**: Efficient computation across multiple alpha values

---

## 💻 **Quick Start**

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Option 1: Ridge with fixed alpha
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Option 2: RidgeCV with automatic alpha selection (recommended!)
model_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
model_cv.fit(X_train, y_train)
print(f"Best alpha: {model_cv.alpha_}")

# Predictions
y_pred = model_cv.predict(X_test)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
```

---

## 🔧 **Key Parameters**

```python
Ridge(
    alpha=1.0,              # L2 penalty strength (KEY PARAMETER)
    fit_intercept=True,     
    copy_X=True,            
    max_iter=None,          # Max iterations (for iterative solvers)
    tol=1e-4,               # Convergence tolerance
    solver='auto',          # 'auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'
    positive=False          # Force positive coefficients
)
```

### **Solver Selection:**
- **'auto'**: Chooses best solver based on data type
- **'svd'**: Singular Value Decomposition (most stable, works always)
- **'cholesky'**: Fast for small to medium datasets
- **'sag'/'saga'**: Stochastic solvers for large n (n > 10,000)
- **'lsqr'**: Iterative, good for sparse matrices

---

## 📊 **Automatic Hyperparameter Tuning**

### **RidgeCV - Cross-Validated Alpha Selection**
```python
from sklearn.linear_model import RidgeCV

# Grid of alphas to try
alphas = np.logspace(-3, 3, 20)  # [0.001, ..., 1000]

model = RidgeCV(
    alphas=alphas,
    cv=5,                    # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    store_cv_results=True    # Store all CV scores
)

model.fit(X_train, y_train)

print(f"Optimal alpha: {model.alpha_}")
print(f"CV scores shape: {model.cv_results_.shape}")  # (n_alphas, n_folds)
```

### **Integration with GridSearchCV**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10, 100],
    'solver': ['auto', 'svd', 'cholesky']
}

grid_search = GridSearchCV(
    Ridge(), 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

---

## ⚖️ **Ridge vs RidgeCV**

| Feature | Ridge | RidgeCV |
|---------|-------|---------|
| Alpha selection | Manual | Automatic (CV) ✅ |
| Use case | Known alpha | Tuning needed |
| Speed | Faster (single fit) | Slower (multiple fits) |
| Recommended | Production (fixed α) | Development/tuning |

---

## 📝 **Code Reference**

Full implementation: [`01_regression/ridge_regression_sklearn.py`](../../01_regression/ridge_regression_sklearn.py)

Related:
- [Ridge Regression - NumPy (from scratch)](ridge_regression_numpy.md)
- [Ridge Regression - PyTorch](ridge_regression_pytorch.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
