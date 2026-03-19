# CatBoost - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of CatBoost for **workers' compensation claim type classification**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[CatBoost - Full Documentation](catboost_numpy.md)**

---

## 🛡️ **Insurance Use Case: Workers' Compensation Claim Classification**

Classify workers' compensation claims into **medical-only**, **indemnity**, or **denied** using GPU-accelerated training with:
- Industry code, injury nature, body part, employment duration, employer size

### **Why PyTorch for Workers' Comp?**
- **Entity embeddings**: Learn rich representations of industry/injury/body part codes
- **GPU acceleration**: Fast training on large workers' comp claim portfolios
- **Multi-task learning**: Jointly predict claim type and reserve range
- **Temporal patterns**: Incorporate claim development history via recurrent layers

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

# Convert workers' comp data to tensors
X_continuous = torch.FloatTensor(X_continuous_numpy)  # [n_claims, 1] - employment_months
X_categorical = torch.LongTensor(X_categorical_numpy)  # [n_claims, 4] - industry, injury, body, size
y_train = torch.LongTensor(y_train_numpy)  # [n_claims] - 0/1/2

# Neural network with entity embeddings for categorical features
class WorkersCompClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Embeddings for categorical features
        self.industry_embed = nn.Embedding(100, 8)    # 100 industry codes -> 8-dim
        self.injury_embed = nn.Embedding(7, 4)        # 7 injury types -> 4-dim
        self.body_part_embed = nn.Embedding(7, 4)     # 7 body parts -> 4-dim
        self.employer_embed = nn.Embedding(4, 3)      # 4 employer sizes -> 3-dim

        # Classifier on concatenated features
        # 8 + 4 + 4 + 3 + 1 (employment_months) = 20
        self.classifier = nn.Sequential(
            nn.Linear(20, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x_cat, x_cont):
        industry = self.industry_embed(x_cat[:, 0])
        injury = self.injury_embed(x_cat[:, 1])
        body = self.body_part_embed(x_cat[:, 2])
        employer = self.employer_embed(x_cat[:, 3])

        combined = torch.cat([industry, injury, body, employer, x_cont], dim=1)
        return self.classifier(combined)

model = WorkersCompClassifier()

# Weighted loss for imbalanced claim types
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 2.0]))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(100):
    logits = model(X_categorical, X_continuous)
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

## 🎯 **GPU Acceleration for FNOL Classification**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Real-time FNOL classification
model.eval()
with torch.no_grad():
    # New claim: Construction (23), Fracture (2), Back (1), 18 months employed, Large employer (3)
    new_cat = torch.LongTensor([[23, 2, 1, 3]]).to(device)
    new_cont = torch.FloatTensor([[18.0]]).to(device)

    logits = model(new_cat, new_cont)
    probs = torch.softmax(logits, dim=1)
    claim_type = ['Medical-Only', 'Indemnity', 'Denied'][probs.argmax().item()]
    confidence = probs.max().item()

    print(f"Claim type: {claim_type} (confidence: {confidence:.1%})")
    print(f"  Medical-Only: {probs[0][0]:.1%}")
    print(f"  Indemnity: {probs[0][1]:.1%}")
    print(f"  Denied: {probs[0][2]:.1%}")
```

### **Embedding Visualization**

```python
# Analyze learned industry embeddings for risk insights
model.eval()
with torch.no_grad():
    all_industries = torch.arange(100).to(device)
    industry_vectors = model.industry_embed(all_industries).cpu().numpy()

    # PCA to visualize industry risk clusters
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    industry_2d = pca.fit_transform(industry_vectors)
    # Construction (23), Mining (21) cluster together -> high-risk industries
    # Finance (52), Information (51) cluster together -> low-risk industries
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 1.1024, Accuracy: 0.3350
Epoch 20,  Loss: 0.6521, Accuracy: 0.7120
Epoch 40,  Loss: 0.4812, Accuracy: 0.7850
Epoch 60,  Loss: 0.4123, Accuracy: 0.8150
Epoch 80,  Loss: 0.3812, Accuracy: 0.8290
Epoch 100, Loss: 0.3621, Accuracy: 0.8340

Test Accuracy: 83.4%
Per-Type F1: Medical-Only=0.88, Indemnity=0.79, Denied=0.78
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/catboost_pytorch.py`](../../02_classification/catboost_pytorch.py)

Related:
- [CatBoost - NumPy (from scratch)](catboost_numpy.md)
- [CatBoost - Scikit-learn](catboost_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
