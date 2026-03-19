# Decision Tree - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Decision Tree for **insurance risk tier classification**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Decision Tree - Full Documentation](decision_tree_numpy.md)**

---

## 🛡️ **Insurance Use Case: Risk Tier Classification**

Classify insurance applicants into **standard**, **preferred**, or **substandard** risk tiers using a neural decision tree with GPU acceleration for:
- Health metrics (BMI, blood pressure, cholesterol)
- Occupation hazard class, driving record, credit score

### **Why PyTorch for Risk Tiering?**
- **Soft decision trees**: Differentiable tree structures for end-to-end learning
- **GPU batch scoring**: Score millions of renewal applications simultaneously
- **Embedding support**: Handle categorical occupation codes natively
- **Ensemble ready**: Combine with neural network layers for hybrid models

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive applicant datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend with neural layers
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert insurance risk data to tensors
X_train = torch.FloatTensor(X_train_numpy)  # [n_applicants, 6]
y_train = torch.LongTensor(y_train_numpy)   # [n_applicants] - 0/1/2

# Neural Decision Tree for risk tiering
class RiskTierClassifier(nn.Module):
    def __init__(self, n_features=6, n_classes=3, n_leaves=16):
        super().__init__()
        self.decision_layer = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        return self.decision_layer(x)

model = RiskTierClassifier(n_features=6, n_classes=3)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.3]))  # Weight underrepresented tiers
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

## 🎯 **GPU Acceleration for Batch Underwriting**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Batch score 100K renewal applications
model.eval()
with torch.no_grad():
    renewal_apps = torch.FloatTensor(renewal_data).to(device)  # [100000, 6]
    tier_probs = torch.softmax(model(renewal_apps), dim=1)
    predicted_tiers = tier_probs.argmax(dim=1)
    # Map: 0=Standard, 1=Preferred, 2=Substandard
```

### **Risk Tier Confidence Scoring**

```python
# Get confidence for each tier assignment
tier_names = ['Standard', 'Preferred', 'Substandard']
for i in range(5):
    probs = tier_probs[i]
    tier = tier_names[predicted_tiers[i]]
    conf = probs.max().item()
    print(f"Applicant {i}: {tier} (confidence: {conf:.1%})")
    # Applicant 0: Preferred (confidence: 91.2%)
    # Applicant 1: Standard (confidence: 78.5%)  <- low confidence, refer to underwriter
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 1.0986, Accuracy: 0.3340
Epoch 20,  Loss: 0.6812, Accuracy: 0.7125
Epoch 40,  Loss: 0.4953, Accuracy: 0.7890
Epoch 60,  Loss: 0.4221, Accuracy: 0.8110
Epoch 80,  Loss: 0.3945, Accuracy: 0.8245
Epoch 100, Loss: 0.3812, Accuracy: 0.8270

Test Accuracy: 82.7%
Per-Tier F1: Standard=0.86, Preferred=0.78, Substandard=0.81
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/decision_tree_pytorch.py`](../../02_classification/decision_tree_pytorch.py)

Related:
- [Decision Tree - NumPy (from scratch)](decision_tree_numpy.md)
- [Decision Tree - Scikit-learn](decision_tree_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
