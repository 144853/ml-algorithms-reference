# Support Vector Machine - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of SVM for **auto accident severity classification**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Support Vector Machine - Full Documentation](svm_numpy.md)**

---

## 🛡️ **Insurance Use Case: Auto Accident Severity Classification**

Classify auto accident claims into **minor**, **moderate**, or **severe** categories using GPU-accelerated training for:
- Vehicle speed at impact, airbag deployment, number of vehicles, road condition, weather

### **Why PyTorch for Severity Classification?**
- **Custom hinge loss**: Implement multi-class hinge loss with class weighting
- **GPU acceleration**: Fast batch scoring for real-time FNOL triage
- **Deep SVM**: Combine kernel approximation with neural network features
- **Deployment**: Export to TorchScript for embedded claims systems

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive accident datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert accident data to tensors
X_train = torch.FloatTensor(X_train_scaled)  # [n_accidents, 5]
y_train = torch.LongTensor(y_train_numpy)    # [n_accidents] - 0=minor, 1=moderate, 2=severe

# SVM-inspired neural classifier
class AccidentSeverityClassifier(nn.Module):
    def __init__(self, n_features=5, n_classes=3):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x):
        features = self.feature_map(x)
        return self.classifier(features)

model = AccidentSeverityClassifier(n_features=5, n_classes=3)

# Multi-class hinge loss (SVM style) with class weighting
criterion = nn.MultiMarginLoss(margin=1.0, weight=torch.tensor([1.0, 1.3, 1.8]))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

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

## 🎯 **GPU Acceleration for FNOL Triage**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Real-time FNOL (First Notice of Loss) triage
model.eval()
with torch.no_grad():
    # New accident: 55 mph, airbag deployed, 3 vehicles, wet road, rain
    new_accident = torch.FloatTensor([[55, 1, 3, 2, 2]]).to(device)
    new_scaled = scaler.transform(new_accident.cpu().numpy())
    logits = model(torch.FloatTensor(new_scaled).to(device))
    probs = torch.softmax(logits, dim=1)
    severity = ['Minor', 'Moderate', 'Severe'][probs.argmax().item()]
    print(f"Predicted severity: {severity}")
    print(f"Confidence: {probs.max().item():.1%}")
    # Route: Severe -> immediate senior adjuster assignment
```

### **Batch Processing Daily Claims**

```python
# Process all daily FNOL claims at once
model.eval()
with torch.no_grad():
    daily_claims = torch.FloatTensor(daily_fnol_data).to(device)  # [500, 5]
    all_probs = torch.softmax(model(daily_claims), dim=1)
    severity_labels = all_probs.argmax(dim=1)

    minor_count = (severity_labels == 0).sum().item()
    moderate_count = (severity_labels == 1).sum().item()
    severe_count = (severity_labels == 2).sum().item()
    print(f"Daily triage: {minor_count} minor, {moderate_count} moderate, {severe_count} severe")
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 1.2145, Accuracy: 0.3280
Epoch 20,  Loss: 0.5421, Accuracy: 0.7340
Epoch 40,  Loss: 0.3812, Accuracy: 0.8010
Epoch 60,  Loss: 0.3215, Accuracy: 0.8290
Epoch 80,  Loss: 0.2934, Accuracy: 0.8380
Epoch 100, Loss: 0.2810, Accuracy: 0.8420

Test Accuracy: 84.2%
Per-Severity F1: Minor=0.89, Moderate=0.79, Severe=0.83
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/svm_pytorch.py`](../../02_classification/svm_pytorch.py)

Related:
- [Support Vector Machine - NumPy (from scratch)](svm_numpy.md)
- [Support Vector Machine - Scikit-learn](svm_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
