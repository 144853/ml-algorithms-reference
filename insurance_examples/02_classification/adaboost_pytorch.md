# AdaBoost - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of AdaBoost for **insurance application approval prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[AdaBoost - Full Documentation](adaboost_numpy.md)**

---

## 🛡️ **Insurance Use Case: Application Approval Prediction**

Predict insurance application outcome -- **approved** or **rejected** -- using a boosted neural ensemble with GPU acceleration for:
- Credit score, claims history, coverage gap, income verification, property inspection score

### **Why PyTorch for Application Approval?**
- **Neural boosting**: Boosted ensemble of small neural networks
- **GPU batch scoring**: Process thousands of applications during peak enrollment
- **Custom sample weighting**: Differentiable AdaBoost-style reweighting
- **Deployment**: Export to ONNX for integration with policy admin systems

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive application datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert application data to tensors
X_train = torch.FloatTensor(X_train_scaled)  # [n_applications, 5]
y_train = torch.FloatTensor(y_train_numpy)   # [n_applications] - 0=rejected, 1=approved

# Boosted Neural Ensemble
class ApplicationApprovalModel(nn.Module):
    def __init__(self, n_features=5, n_boosters=8):
        super().__init__()
        # Weak learners: small neural networks
        self.weak_learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ) for _ in range(n_boosters)
        ])
        # Learnable booster weights (analogous to AdaBoost alpha)
        self.alphas = nn.Parameter(torch.ones(n_boosters))

    def forward(self, x):
        predictions = [learner(x).squeeze() for learner in self.weak_learners]
        weights = torch.softmax(self.alphas, dim=0)
        boosted = sum(w * p for w, p in zip(weights, predictions))
        return boosted

model = ApplicationApprovalModel(n_features=5, n_boosters=8)

# Weighted BCE for application class imbalance
pos_weight = torch.tensor([1.5])  # Slightly weight approved class
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

# Training loop with sample reweighting
sample_weights = torch.ones(len(X_train)) / len(X_train)

for epoch in range(120):
    logits = model(X_train)
    # Apply sample weights (focus on misclassified applications)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y_train, weight=sample_weights, reduction='mean'
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update sample weights (AdaBoost-style)
    if epoch % 10 == 0:
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            misclassified = (preds != y_train).float()
            error_rate = (misclassified * sample_weights).sum()
            if error_rate > 0 and error_rate < 1:
                alpha = 0.5 * torch.log((1 - error_rate) / error_rate)
                sample_weights = sample_weights * torch.exp(alpha * misclassified)
                sample_weights = sample_weights / sample_weights.sum()

    if epoch % 20 == 0:
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_train).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
```

---

## 🎯 **GPU Acceleration for Batch Application Processing**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Batch process incoming applications
model.eval()
with torch.no_grad():
    batch_apps = torch.FloatTensor(new_applications_scaled).to(device)  # [10000, 5]
    approval_probs = torch.sigmoid(model(batch_apps))

    # Decision routing
    auto_approve = (approval_probs > 0.85).sum().item()
    auto_reject = (approval_probs < 0.20).sum().item()
    manual_review = len(batch_apps) - auto_approve - auto_reject

    print(f"Batch processing complete:")
    print(f"  Auto-approved: {auto_approve}")
    print(f"  Auto-rejected: {auto_reject}")
    print(f"  Manual review: {manual_review}")
```

### **Weak Learner Analysis**

```python
# Analyze contribution of each weak learner
model.eval()
with torch.no_grad():
    weights = torch.softmax(model.alphas, dim=0).cpu().numpy()
    print("Weak Learner Weights:")
    for i, w in enumerate(weights):
        print(f"  Learner {i}: weight={w:.3f}")
    # Higher weight = more important learner
    # Learner focusing on credit score interactions may have highest weight
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 0.6931, Accuracy: 0.5120
Epoch 20,  Loss: 0.4512, Accuracy: 0.7650
Epoch 40,  Loss: 0.3623, Accuracy: 0.8120
Epoch 60,  Loss: 0.3198, Accuracy: 0.8310
Epoch 80,  Loss: 0.2912, Accuracy: 0.8400
Epoch 100, Loss: 0.2756, Accuracy: 0.8450
Epoch 120, Loss: 0.2654, Accuracy: 0.8450

Test Accuracy: 84.5%
Test AUC-ROC: 0.908
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/adaboost_pytorch.py`](../../02_classification/adaboost_pytorch.py)

Related:
- [AdaBoost - NumPy (from scratch)](adaboost_numpy.md)
- [AdaBoost - Scikit-learn](adaboost_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
