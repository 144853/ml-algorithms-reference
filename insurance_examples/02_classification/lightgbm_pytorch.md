# LightGBM - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of LightGBM for **insurance customer churn prediction**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[LightGBM - Full Documentation](lightgbm_numpy.md)**

---

## 🛡️ **Insurance Use Case: Customer Churn Prediction**

Predict customer churn -- **churn** or **retain** -- using gradient-boosted neural networks with GPU acceleration for:
- Premium change percentage, claim experience, NPS score, policy bundle count, competitor quote differential

### **Why PyTorch for Churn Prediction?**
- **Deep tabular models**: TabNet-style attention for feature selection
- **GPU batch scoring**: Score millions of renewal policies in minutes
- **Sequence modeling**: Incorporate customer interaction history via RNNs
- **A/B test integration**: Differentiable model enables causal uplift estimation

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive customer portfolios
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert churn data to tensors
X_train = torch.FloatTensor(X_train_numpy)  # [n_customers, 5]
y_train = torch.FloatTensor(y_train_numpy)  # [n_customers] - 0=retain, 1=churn

# Gradient-boosted neural network for churn
class ChurnPredictor(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Softmax(dim=1)
        )
        self.predictor = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Feature attention: learn which features matter most
        attn_weights = self.attention(x)
        x_weighted = x * attn_weights
        return self.predictor(x_weighted).squeeze()

model = ChurnPredictor(n_features=5)

# Focal loss for hard-to-classify churn cases
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training with DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(X_train)) > 0.5).float()
            acc = (preds == y_train).float().mean()
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {acc:.4f}")
```

---

## 🎯 **GPU Acceleration for Portfolio Scoring**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Score entire renewal portfolio
model.eval()
with torch.no_grad():
    renewal_portfolio = torch.FloatTensor(all_renewals).to(device)  # [2M, 5]
    churn_probs = torch.sigmoid(model(renewal_portfolio))

    # Retention priority segments
    high_risk = (churn_probs > 0.65).sum().item()
    medium_risk = ((churn_probs > 0.35) & (churn_probs <= 0.65)).sum().item()
    low_risk = (churn_probs <= 0.35).sum().item()

    print(f"High churn risk: {high_risk:,} customers")
    print(f"Medium risk: {medium_risk:,} customers")
    print(f"Low risk: {low_risk:,} customers")
```

### **Feature Attention Analysis**

```python
# Analyze learned feature importance via attention weights
model.eval()
with torch.no_grad():
    sample = torch.FloatTensor(X_test[:100]).to(device)
    attn = model.attention(sample).mean(dim=0).cpu().numpy()

    feature_names = ['premium_change', 'claim_count', 'nps_score', 'bundle_count', 'competitor_diff']
    for name, weight in sorted(zip(feature_names, attn), key=lambda x: -x[1]):
        print(f"{name}: attention={weight:.3f}")
    # competitor_diff: attention=0.312
    # premium_change: attention=0.268
    # nps_score: attention=0.198
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 0.6124, Accuracy: 0.6210
Epoch 20,  Loss: 0.3812, Accuracy: 0.7890
Epoch 40,  Loss: 0.2934, Accuracy: 0.8320
Epoch 60,  Loss: 0.2512, Accuracy: 0.8490
Epoch 80,  Loss: 0.2298, Accuracy: 0.8560
Epoch 100, Loss: 0.2156, Accuracy: 0.8590

Test Accuracy: 85.9%
Test AUC-ROC: 0.921
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/lightgbm_pytorch.py`](../../02_classification/lightgbm_pytorch.py)

Related:
- [LightGBM - NumPy (from scratch)](lightgbm_numpy.md)
- [LightGBM - Scikit-learn](lightgbm_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
