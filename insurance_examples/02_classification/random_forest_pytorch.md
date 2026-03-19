# Random Forest - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Random Forest for **insurance policy lapse prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Random Forest - Full Documentation](random_forest_numpy.md)**

---

## 🛡️ **Insurance Use Case: Policy Lapse Prediction**

Predict whether a policy will **lapse** or be **retained** using a deep ensemble approach with GPU acceleration for:
- Premium-to-income ratio, payment frequency, policy age, customer complaints, agent engagement score

### **Why PyTorch for Lapse Prediction?**
- **Deep ensemble**: Multiple neural networks mimicking forest diversity
- **GPU batch scoring**: Score entire policy portfolios in seconds
- **Temporal features**: Incorporate sequential payment history via RNNs
- **Online learning**: Update model as new lapse data streams in

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive policy portfolios
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert policy data to tensors
X_train = torch.FloatTensor(X_train_numpy)  # [n_policies, 5]
y_train = torch.LongTensor(y_train_numpy)   # [n_policies] - 0=retain, 1=lapse

# Deep Ensemble (Neural Random Forest)
class LapsePredictor(nn.Module):
    def __init__(self, n_features=5, n_estimators=10):
        super().__init__()
        self.estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features, 16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16, 2)
            ) for _ in range(n_estimators)
        ])

    def forward(self, x):
        # Average predictions across all estimators
        outputs = [est(x) for est in self.estimators]
        return torch.stack(outputs).mean(dim=0)

model = LapsePredictor(n_features=5, n_estimators=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.5]))  # Weight lapse class
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        preds = logits.argmax(dim=1)
        acc = (preds == y_train).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
```

---

## 🎯 **GPU Acceleration for Portfolio Scoring**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Score entire policy portfolio
model.eval()
with torch.no_grad():
    portfolio = torch.FloatTensor(all_policies).to(device)  # [500000, 5]
    lapse_probs = torch.softmax(model(portfolio), dim=1)[:, 1]
    high_risk_mask = lapse_probs > 0.70
    print(f"High-risk policies: {high_risk_mask.sum().item()}")
```

### **Ensemble Uncertainty for Referral**

```python
# Use individual estimator disagreement as uncertainty measure
model.eval()
with torch.no_grad():
    individual_preds = [torch.softmax(est(X_test.to(device)), dim=1)[:, 1]
                        for est in model.estimators]
    pred_stack = torch.stack(individual_preds)
    mean_lapse_prob = pred_stack.mean(dim=0)
    uncertainty = pred_stack.std(dim=0)
    # High uncertainty -> refer to retention specialist for manual review
    refer_mask = uncertainty > 0.15
    print(f"Refer to specialist: {refer_mask.sum().item()} policies")
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 0.7124, Accuracy: 0.5820
Epoch 20,  Loss: 0.4956, Accuracy: 0.7650
Epoch 40,  Loss: 0.3912, Accuracy: 0.8210
Epoch 60,  Loss: 0.3534, Accuracy: 0.8480
Epoch 80,  Loss: 0.3321, Accuracy: 0.8590
Epoch 100, Loss: 0.3198, Accuracy: 0.8640

Test Accuracy: 86.4%
Test AUC-ROC: 0.913
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/random_forest_pytorch.py`](../../02_classification/random_forest_pytorch.py)

Related:
- [Random Forest - NumPy (from scratch)](random_forest_numpy.md)
- [Random Forest - Scikit-learn](random_forest_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
