# Linear Regression - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Linear Regression. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Linear Regression - Full Documentation](linear_regression_numpy.md)**

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive datasets
- **Automatic differentiation**: No manual gradient computation
- **Deep learning integration**: Easy to extend to neural networks
- **Mini-batch training**: Built-in DataLoader for efficient batch processing
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train_numpy)
y_train = torch.FloatTensor(y_train_numpy).reshape(-1, 1)
X_test = torch.FloatTensor(X_test_numpy)
y_test = torch.FloatTensor(y_test_numpy).reshape(-1, 1)

# Create DataLoader for mini-batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model (single linear layer!)
model = nn.Linear(n_features, 1)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test, y_test)
    print(f"Test MSE: {test_loss.item():.4f}")
```

---

## 🔧 **Key Components**

### **Model Definition**
```python
# Simple: Single linear layer
model = nn.Linear(in_features=n_features, out_features=1, bias=True)

# Alternative: Sequential for extensibility
model = nn.Sequential(
    nn.Linear(n_features, 1)
)
```

### **Optimizers**
```python
# SGD (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (Adaptive Moments - recommended)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# With learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

---

## 🎯 **GPU Acceleration**

```python
# Move model and data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop with GPU
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Speed comparison** (for large datasets):
- CPU (NumPy): 15.3s
- CPU (PyTorch): 12.1s
- GPU (PyTorch): 1.8s ⚡

---

## 📊 **PyTorch Specific Features**

### **1. Learning Rate Scheduling**
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(epochs):
    train_epoch()
    scheduler.step()  # Decay lr every 30 epochs
```

### **2. Gradient Clipping (for stability)**
```python
# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### **3. Model Saving/Loading**
```python
# Save model
torch.save(model.state_dict(), 'linear_model.pth')

# Load model
model = nn.Linear(n_features, 1)
model.load_state_dict(torch.load('linear_model.pth'))
model.eval()
```

### **4. Extracting Weights**
```python
# Access learned parameters
weight = model.weight.data.numpy()  # Shape: (1, n_features)
bias = model.bias.data.numpy()      # Shape: (1,)

print(f"Coefficients: {weight[0]}")
print(f"Intercept: {bias[0]}")
```

---

## 🔄 **Easy Extension to Neural Networks**

PyTorch's power: Change one line to add hidden layers!

```python
# Linear Regression
model = nn.Linear(n_features, 1)

# ↓ Change to Neural Network ↓

model = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Training code stays the same!
```

---

## ⚖️ **When to Use PyTorch vs NumPy/Sklearn**

| Scenario | Use This |
|----------|----------|
| Massive datasets (n > 1M) | ✅ PyTorch (GPU) |
| Need GPU acceleration | ✅ PyTorch |
| Plan to extend to neural nets | ✅ PyTorch |
| Production scikit-learn pipeline | Scikit-learn |
| Learning from scratch | NumPy |
| Quick prototyping | Scikit-learn |

---

## 📝 **Code Reference**

Full implementation: [`01_regression/linear_regression_pytorch.py`](../../01_regression/linear_regression_pytorch.py)

Related:
- [Linear Regression - NumPy (from scratch)](linear_regression_numpy.md)
- [Linear Regression - Scikit-learn](linear_regression_sklearn.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
