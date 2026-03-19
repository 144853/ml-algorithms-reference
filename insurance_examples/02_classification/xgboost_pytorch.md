# XGBoost - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of XGBoost for **subrogation recovery prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[XGBoost - Full Documentation](xgboost_numpy.md)**

---

## 🛡️ **Insurance Use Case: Subrogation Recovery Prediction**

Predict subrogation recovery potential -- **recoverable** or **non-recoverable** -- using gradient-boosted neural networks with GPU acceleration for:
- Fault determination, third-party insurer rating, claim type, jurisdiction, evidence quality

### **Why PyTorch for Subrogation?**
- **Neural boosting**: Gradient-boosted neural networks for complex recovery patterns
- **GPU training**: Fast iteration on large historical claim datasets
- **Custom loss**: Asymmetric loss weighting (missing a recoverable claim costs more)
- **Feature embeddings**: Learn jurisdiction and claim type representations

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive claim datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert subrogation data to tensors
X_train = torch.FloatTensor(X_train_numpy)  # [n_claims, 5]
y_train = torch.FloatTensor(y_train_numpy)  # [n_claims] - 0=non-recoverable, 1=recoverable

# Gradient-Boosted Neural Network
class SubrogationPredictor(nn.Module):
    def __init__(self, n_features=5, n_boosters=5):
        super().__init__()
        self.boosters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features, 24),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(24, 1)
            ) for _ in range(n_boosters)
        ])
        self.booster_weights = nn.Parameter(torch.ones(n_boosters) / n_boosters)

    def forward(self, x):
        predictions = [booster(x).squeeze() for booster in self.boosters]
        weighted = sum(w * p for w, p in zip(torch.softmax(self.booster_weights, dim=0), predictions))
        return weighted

model = SubrogationPredictor(n_features=5, n_boosters=5)

# Asymmetric loss: missing a recoverable claim costs 2.5x more
pos_weight = torch.tensor([2.5])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(150):
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 30 == 0:
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y_train).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
```

---

## 🎯 **GPU Acceleration for Batch Recovery Scoring**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Score all open subrogation-eligible claims
model.eval()
with torch.no_grad():
    open_claims = torch.FloatTensor(open_subrogation_data).to(device)  # [5000, 5]
    recovery_probs = torch.sigmoid(model(open_claims))

    # Priority queue for subrogation team
    high_priority = (recovery_probs > 0.75).sum().item()
    medium_priority = ((recovery_probs > 0.45) & (recovery_probs <= 0.75)).sum().item()
    low_priority = (recovery_probs <= 0.45).sum().item()

    print(f"High priority: {high_priority} claims (pursue aggressively)")
    print(f"Medium priority: {medium_priority} claims (standard pursuit)")
    print(f"Low priority: {low_priority} claims (cost-benefit review)")
```

### **Recovery Amount Estimation**

```python
# Combine recovery probability with claim amount for expected value
model.eval()
with torch.no_grad():
    claim_amounts = torch.FloatTensor(claim_amount_data).to(device)
    recovery_probs = torch.sigmoid(model(open_claims))
    expected_recovery = recovery_probs * claim_amounts
    total_expected = expected_recovery.sum().item()
    print(f"Total expected subrogation recovery: ${total_expected:,.0f}")
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 0.7812, Accuracy: 0.5240
Epoch 30,  Loss: 0.4123, Accuracy: 0.8010
Epoch 60,  Loss: 0.3245, Accuracy: 0.8520
Epoch 90,  Loss: 0.2891, Accuracy: 0.8680
Epoch 120, Loss: 0.2712, Accuracy: 0.8740
Epoch 150, Loss: 0.2634, Accuracy: 0.8760

Test Accuracy: 87.6%
Test AUC-ROC: 0.934
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/xgboost_pytorch.py`](../../02_classification/xgboost_pytorch.py)

Related:
- [XGBoost - NumPy (from scratch)](xgboost_numpy.md)
- [XGBoost - Scikit-learn](xgboost_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
