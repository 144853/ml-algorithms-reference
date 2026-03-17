# Ridge Regression - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Ridge Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Ridge Regression - Full Documentation](ridge_regression_numpy.md)**

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Handle massive datasets on GPU
- **Weight decay**: Built-in L2 regularization in optimizers
- **Automatic differentiation**: No manual gradient computation
- **Flexible**: Easy to extend to neural networks

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data to tensors
X_train = torch.FloatTensor(X_train_numpy)
y_train = torch.FloatTensor(y_train_numpy).reshape(-1, 1)

# Define model
model = nn.Linear(n_features, 1)

# Define loss and optimizer WITH weight_decay (Ridge penalty!)
criterion = nn.MSELoss()
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.01, 
    weight_decay=2.0  # weight_decay = 2 * alpha in Ridge formulation
)

# Training loop
for epoch in range(1000):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test, y_test)
```

---

## 🔧 **Weight Decay vs Alpha**

Ridge formulation: `Loss = MSE + alpha * ||w||²`

PyTorch formulation: `Loss = MSE`, with weight_decay in optimizer

**Relationship:**
```
weight_decay = 2 * alpha
```

Example:
- Ridge alpha=0.5 → PyTorch weight_decay=1.0
- Ridge alpha=1.0 → PyTorch weight_decay=2.0

---

## 📊 **Alternative: Explicit L2 Penalty**

Instead of weight_decay, you can add L2 penalty manually:

```python
# No weight_decay in optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    y_pred = model(X_train)
    
    # MSE loss
    mse_loss = criterion(y_pred, y_train)
    
    # Explicit L2 regularization
    l2_penalty = alpha * sum(p.pow(2).sum() for p in model.parameters())
    
    # Total loss
    loss = mse_loss + l2_penalty
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🎯 **GPU Acceleration for Large Datasets**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Linear(n_features, 1).to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

# Training loop stays the same!
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    # ... rest of training
```

---

## ⚖️ **When to Use PyTorch Ridge**

| Scenario | Use PyTorch |
|----------|-------------|
| Dataset > 1M samples | ✅ Yes (GPU) |
| Need GPU acceleration | ✅ Yes |
| Plan to extend to neural net | ✅ Yes |
| Standard tabular data | No (use sklearn) |

---

## 📝 **Code Reference**

Full implementation: [`01_regression/ridge_regression_pytorch.py`](../../01_regression/ridge_regression_pytorch.py)

Related:
- [Ridge Regression - NumPy (from scratch)](ridge_regression_numpy.md)
- [Ridge Regression - Scikit-learn](ridge_regression_sklearn.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
