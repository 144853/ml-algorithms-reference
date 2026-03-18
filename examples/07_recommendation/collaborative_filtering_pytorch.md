# Collaborative Filtering - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Collaborative Filtering. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Collaborative Filtering - Full Documentation](collaborative_filtering_numpy.md)**

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

Full implementation: [`07_recommendation/collaborative_filtering_pytorch.py`](../../07_recommendation/collaborative_filtering_pytorch.py)

Related:
- [Collaborative Filtering - NumPy (from scratch)](collaborative_filtering_numpy.md)
- [Collaborative Filtering - Scikit-learn](collaborative_filtering_sklearn.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
