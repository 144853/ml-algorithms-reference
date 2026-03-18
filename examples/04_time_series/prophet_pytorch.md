# Prophet - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Prophet. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Prophet - Full Documentation](prophet_numpy.md)**

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data to tensors
X_train = torch.FloatTensor(X_train_numpy)
y_train = torch.LongTensor(y_train_numpy)

# Define model (placeholder - see actual implementation)
model = MODEL_CLASS_PLACEHOLDER()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🎯 **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
```

---

## 📝 **Code Reference**

Full implementation: [`04_time_series/prophet_pytorch.py`](../../04_time_series/prophet_pytorch.py)

Related:
- [Prophet - NumPy (from scratch)](prophet_numpy.md)
- [Prophet - Scikit-learn](prophet_sklearn.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
