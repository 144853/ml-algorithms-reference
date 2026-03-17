# Lasso Regression - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Lasso Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Lasso Regression - Full Documentation](lasso_regression_numpy.md)**

---

## 🚀 **PyTorch Approach**

- **Explicit L1 penalty**: Add `alpha * ||w||₁` to loss
- **Post-training thresholding**: Zero out small coefficients
- **GPU acceleration**: Handle large datasets
- **Note**: SGD doesn't naturally produce exact zeros like coordinate descent

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data
X_train = torch.FloatTensor(X_train_numpy)
y_train = torch.FloatTensor(y_train_numpy).reshape(-1, 1)

# Define model
model = nn.Linear(n_features, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training with L1 penalty
alpha = 0.1
for epoch in range(1000):
    y_pred = model(X_train)
    
    # MSE loss
    mse_loss = criterion(y_pred, y_train)
    
    # L1 penalty (sum of absolute weights)
    l1_penalty = alpha * sum(p.abs().sum() for p in model.parameters())
    
    # Total loss
    loss = mse_loss + l1_penalty
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Post-training: threshold small weights to zero
threshold = 1e-4
with torch.no_grad():
    for param in model.parameters():
        param[param.abs() < threshold] = 0

# Count non-zero weights
weights = model.weight.data.numpy()[0]
n_nonzero = np.sum(weights != 0)
print(f"Selected {n_nonzero}/{len(weights)} features")
```

---

## 🔧 **Soft-Thresholding (Manual Implementation)**

For better sparsity, implement soft-thresholding operator:

```python
def soft_threshold(x, threshold):
    """Soft-thresholding operator: S(x, t) = sign(x) * max(|x| - t, 0)"""
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)

# Apply after each optimizer step
for epoch in range(epochs):
    # ... forward pass, loss computation
    
    optimizer.step()
    
    # Apply soft-thresholding
    with torch.no_grad():
        for param in model.parameters():
            param.data = soft_threshold(param.data, alpha * lr)
```

---

## 📊 **Proximal Gradient Descent**

More principled approach for Lasso:

```python
# Manual optimizer for proximal gradient
lr = 0.01
alpha = 0.1

for epoch in range(1000):
    # Gradient step on MSE (smooth part)
    y_pred = model(X_train)
    mse_loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    mse_loss.backward()
    
    # Manual gradient update + soft-thresholding (non-smooth part)
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
            param.data = soft_threshold(param.data, lr * alpha)
```

---

## 🎯 **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Linear(n_features, 1).to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

# Training loop with GPU
for epoch in range(epochs):
    y_pred = model(X_train)
    mse_loss = criterion(y_pred, y_train)
    
    # L1 penalty
    l1_penalty = alpha * sum(p.abs().sum() for p in model.parameters())
    loss = mse_loss + l1_penalty
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## ⚖️ **PyTorch Lasso Limitations**

| Aspect | PyTorch | Sklearn |
|--------|---------|---------|
| Exact zeros | ⚠️ Needs thresholding | ✅ Natural |
| Speed (CPU) | Slower | Faster (Cython) |
| Speed (GPU) | ✅ Much faster | N/A |
| Algorithm | SGD-based | Coordinate descent |
| Best for | Large n | General use |

**Recommendation**: Use sklearn for most cases, PyTorch for massive datasets with GPU.

---

## 📝 **Code Reference**

Full implementation: [`01_regression/lasso_regression_pytorch.py`](../../01_regression/lasso_regression_pytorch.py)

Related:
- [Lasso Regression - NumPy (from scratch)](lasso_regression_numpy.md)
- [Lasso Regression - Scikit-learn](lasso_regression_sklearn.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
