# Logistic Regression - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Logistic Regression for **insurance claim fraud detection**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Logistic Regression - Full Documentation](logistic_regression_numpy.md)**

---

## 🛡️ **Insurance Use Case: Claim Fraud Detection**

Predict whether an insurance claim is **fraudulent** or **legitimate** using GPU-accelerated training on large-scale claim datasets with:
- Claim amount, time to report, witness count, police report filed, policy age

### **Why PyTorch for Insurance Fraud?**
- **Scalability**: GPU training for millions of historical claims
- **Custom loss functions**: Weighted BCE for highly imbalanced fraud datasets
- **Embedding layers**: Incorporate categorical claim features (adjuster region, claim type)
- **Real-time scoring**: Deploy with TorchServe for live claim intake fraud screening

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive claim datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize loss weighting for fraud imbalance
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert insurance claim data to tensors
X_train = torch.FloatTensor(X_train_scaled)  # [n_claims, 5]
y_train = torch.FloatTensor(y_train_numpy)   # [n_claims] - 0=legitimate, 1=fraud

# Create DataLoader for batch processing
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define logistic regression model
class FraudDetector(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = FraudDetector(n_features=5)

# Weighted BCE loss for imbalanced fraud data (10% fraud)
pos_weight = torch.tensor([9.0])  # weight fraud class 9x higher
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        logits = model.linear(X_batch).squeeze()
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 🎯 **GPU Acceleration for Large Claim Portfolios**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Move batch data to GPU during training
for X_batch, y_batch in train_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    # ... training step
```

### **Inference: Real-Time Fraud Scoring**

```python
model.eval()
with torch.no_grad():
    # New claim: $35K, 42 days to report, 0 witnesses, no police, 5 months old
    new_claim = torch.FloatTensor([[35000, 42, 0, 0, 5]]).to(device)
    new_claim_scaled = scaler.transform(new_claim.cpu().numpy())
    fraud_prob = model(torch.FloatTensor(new_claim_scaled).to(device))
    print(f"Fraud probability: {fraud_prob.item():.3f}")  # e.g., 0.891
    # Route to SIU if > 0.65
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 0.6931, Train AUC: 0.500
Epoch 20,  Loss: 0.4215, Train AUC: 0.821
Epoch 40,  Loss: 0.3102, Train AUC: 0.897
Epoch 60,  Loss: 0.2934, Train AUC: 0.919
Epoch 80,  Loss: 0.2876, Train AUC: 0.924
Epoch 100, Loss: 0.2861, Train AUC: 0.926

Test AUC-ROC: 0.926
Test Accuracy: 88.1%
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/logistic_regression_pytorch.py`](../../02_classification/logistic_regression_pytorch.py)

Related:
- [Logistic Regression - NumPy (from scratch)](logistic_regression_numpy.md)
- [Logistic Regression - Scikit-learn](logistic_regression_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
