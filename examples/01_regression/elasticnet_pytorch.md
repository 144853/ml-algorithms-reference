# ElasticNet Regression - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of ElasticNet Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[ElasticNet Regression - Full Documentation](elasticnet_numpy.md)**

---

## 🚀 **PyTorch Approach**

- **Combined penalty**: alpha × (l1_ratio × L1 + (1-l1_ratio) × L2)
- **L1 via explicit penalty**: Add to loss function
- **L2 via weight_decay OR explicit penalty**: Two options
- **GPU acceleration**: For large datasets

---

## 💻 **Quick Start (Method 1: Explicit Penalties)**

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
optimizer = optim.Adam(model.parameters(), lr=0.01)  # No weight_decay here

# Hyperparameters
alpha = 1.0
l1_ratio = 0.5

# Training loop
for epoch in range(1000):
    y_pred = model(X_train)
    
    # MSE loss
    mse_loss = criterion(y_pred, y_train)
    
    # L1 penalty
    l1_penalty = alpha * l1_ratio * sum(p.abs().sum() for p in model.parameters())
    
    # L2 penalty
    l2_penalty = alpha * (1 - l1_ratio) * sum(p.pow(2).sum() for p in model.parameters())
    
    # Total ElasticNet loss
    loss = mse_loss + l1_penalty + l2_penalty
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Post-training thresholding for sparsity
threshold = 1e-4
with torch.no_grad():
    for param in model.parameters():
        param[param.abs() < threshold] = 0

# Report sparsity
weights = model.weight.data.numpy()[0]
print(f"Selected {np.sum(weights != 0)}/{len(weights)} features")
```

---

## 💻 **Quick Start (Method 2: weight_decay + L1)**

```python
# L2 regularization via weight_decay
alpha = 1.0
l1_ratio = 0.5

# weight_decay handles L2 part
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.01, 
    weight_decay=2 * alpha * (1 - l1_ratio)  # L2 component
)

# Training loop
for epoch in range(1000):
    y_pred = model(X_train)
    
    # MSE loss
    mse_loss = criterion(y_pred, y_train)
    
    # Only add L1 penalty explicitly
    l1_penalty = alpha * l1_ratio * sum(p.abs().sum() for p in model.parameters())
    
    loss = mse_loss + l1_penalty  # L2 is in optimizer
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🔧 **Hyperparameter Mapping**

ElasticNet loss: `MSE + α × l1_ratio × ||w||₁ + α × (1-l1_ratio) × ||w||₂²`

PyTorch implementation:
```python
# L1 part (explicit in loss)
l1_term = alpha * l1_ratio * weights.abs().sum()

# L2 part (either explicit or via weight_decay)
# Option A: Explicit
l2_term = alpha * (1 - l1_ratio) * weights.pow(2).sum()

# Option B: weight_decay (in optimizer)
weight_decay = 2 * alpha * (1 - l1_ratio)
```

---

## 📊 **Grid Search for Optimal l1_ratio**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

# Grid search
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []

for l1_ratio in l1_ratios:
    model = nn.Linear(n_features, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train model
    for epoch in range(1000):
        y_pred = model(X_train_t)
        mse = criterion(y_pred, y_train_t)
        l1 = alpha * l1_ratio * sum(p.abs().sum() for p in model.parameters())
        l2 = alpha * (1 - l1_ratio) * sum(p.pow(2).sum() for p in model.parameters())
        loss = mse + l1 + l2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
        val_mse = criterion(y_val_pred, y_val_t).item()
    
    results.append((l1_ratio, val_mse))
    print(f"l1_ratio={l1_ratio:.1f}, Val MSE={val_mse:.4f}")

# Find best
best_l1_ratio = min(results, key=lambda x: x[1])[0]
print(f"\nBest l1_ratio: {best_l1_ratio}")
```

---

## 🎯 **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Linear(n_features, 1).to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

# Training loop (same as before, data already on GPU)
for epoch in range(epochs):
    y_pred = model(X_train)
    mse = criterion(y_pred, y_train)
    l1 = alpha * l1_ratio * sum(p.abs().sum() for p in model.parameters())
    l2 = alpha * (1-l1_ratio) * sum(p.pow(2).sum() for p in model.parameters())
    loss = mse + l1 + l2
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## ⚖️ **When to Use PyTorch ElasticNet**

| Scenario | Use PyTorch |
|----------|-------------|
| Massive dataset (n > 1M) | ✅ Yes (GPU) |
| Standard tabular data | No (use sklearn) |
| Need exact coordinate descent | No (use sklearn) |
| Extending to neural networks | ✅ Yes |

**Recommendation**: Use sklearn's ElasticNetCV for most cases. Use PyTorch only for very large datasets with GPU, or as a stepping stone to neural networks.

---

## 📝 **Code Reference**

Full implementation: [`01_regression/elasticnet_pytorch.py`](../../01_regression/elasticnet_pytorch.py)

Related:
- [ElasticNet - NumPy (from scratch)](elasticnet_numpy.md)
- [ElasticNet - Scikit-learn](elasticnet_sklearn.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
