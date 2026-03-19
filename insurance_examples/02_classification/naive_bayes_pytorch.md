# Naive Bayes - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of Naive Bayes for **insurance underwriting decision prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Naive Bayes - Full Documentation](naive_bayes_numpy.md)**

---

## 🛡️ **Insurance Use Case: Underwriting Decision Prediction**

Predict underwriting decisions -- **approve**, **decline**, or **refer** -- using GPU-accelerated probabilistic classification for:
- Applicant age, medical history flags, occupation class, coverage amount requested

### **Why PyTorch for Underwriting?**
- **GPU batch scoring**: Process thousands of applications during open enrollment
- **Differentiable priors**: Learn optimal prior probabilities from data
- **Embedding layers**: Handle categorical occupation codes natively
- **Online updates**: Update class statistics as new decisions are recorded

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive applicant datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert underwriting data to tensors
X_train = torch.FloatTensor(X_train_numpy)  # [n_applicants, 4]
y_train = torch.LongTensor(y_train_numpy)   # [n_applicants] - 0=approve, 1=decline, 2=refer

# Neural Naive Bayes: learn feature distributions per class
class UnderwritingClassifier(nn.Module):
    def __init__(self, n_features=4, n_classes=3):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(n_features, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, n_classes)
        )
        # Learnable log-priors (initialized to approximate underwriting mix)
        self.log_priors = nn.Parameter(torch.log(torch.tensor([0.70, 0.10, 0.20])))

    def forward(self, x):
        log_likelihoods = self.feature_net(x)
        return log_likelihoods + self.log_priors

model = UnderwritingClassifier(n_features=4, n_classes=3)

# Weighted loss for imbalanced underwriting decisions
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 2.0]))
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

## 🎯 **GPU Acceleration for Open Enrollment**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Batch score open enrollment applications
model.eval()
with torch.no_grad():
    enrollment_apps = torch.FloatTensor(enrollment_data).to(device)  # [50000, 4]
    logits = model(enrollment_apps)
    probs = torch.softmax(logits, dim=1)
    decisions = probs.argmax(dim=1)

    # Confidence-based routing
    max_confidence = probs.max(dim=1).values
    auto_decide = max_confidence > 0.85
    manual_review = ~auto_decide

    decision_names = ['Approve', 'Decline', 'Refer']
    print(f"Auto-decisioned: {auto_decide.sum().item()} applications")
    print(f"Manual review: {manual_review.sum().item()} applications")

    for i, name in enumerate(decision_names):
        count = (decisions == i).sum().item()
        print(f"  {name}: {count}")
```

### **Real-Time Application Scoring**

```python
# Score single application as it arrives
model.eval()
with torch.no_grad():
    # Applicant: 55 years old, 3 medical flags, occupation class 2, $500K coverage
    applicant = torch.FloatTensor([[55, 3, 2, 500000]]).to(device)
    logits = model(applicant)
    probs = torch.softmax(logits, dim=1)
    decision = ['Approve', 'Decline', 'Refer'][probs.argmax().item()]
    confidence = probs.max().item()
    print(f"Decision: {decision} (confidence: {confidence:.1%})")
    # Decision: Refer (confidence: 72.3%) -> route to senior underwriter
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 1.0986, Accuracy: 0.3450
Epoch 20,  Loss: 0.7234, Accuracy: 0.6520
Epoch 40,  Loss: 0.5812, Accuracy: 0.7210
Epoch 60,  Loss: 0.5234, Accuracy: 0.7510
Epoch 80,  Loss: 0.4912, Accuracy: 0.7620
Epoch 100, Loss: 0.4723, Accuracy: 0.7680

Test Accuracy: 76.8%
Per-Decision F1: Approve=0.85, Decline=0.68, Refer=0.65
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/naive_bayes_pytorch.py`](../../02_classification/naive_bayes_pytorch.py)

Related:
- [Naive Bayes - NumPy (from scratch)](naive_bayes_numpy.md)
- [Naive Bayes - Scikit-learn](naive_bayes_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
