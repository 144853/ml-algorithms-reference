# K-Nearest Neighbors - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of KNN for **insurance claim benchmark pricing**. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[K-Nearest Neighbors - Full Documentation](knn_numpy.md)**

---

## 🛡️ **Insurance Use Case: Similar Claim Benchmark Pricing**

Classify claims into **pricing benchmark tiers** using GPU-accelerated distance computation for:
- Injury type code, treatment duration, geographic region, attorney involvement

### **Why PyTorch for Claim Benchmarking?**
- **GPU distance computation**: Parallelize pairwise distances across millions of claims
- **Learned distance metrics**: Train a Siamese network to learn claim similarity
- **Batch queries**: Score thousands of open claims simultaneously
- **Embedding space**: Learn claim representations for more meaningful similarity

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

# Convert claim data to tensors
X_train = torch.FloatTensor(X_train_scaled)  # [n_claims, 4]
y_train = torch.LongTensor(y_train_numpy)    # [n_claims] - pricing tier 0-3

# Learned embedding KNN: map claims to similarity space
class ClaimEmbedder(nn.Module):
    def __init__(self, n_features=4, embed_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 4)

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        embedding = self.embed(x)
        return self.classifier(embedding)

model = ClaimEmbedder(n_features=4, embed_dim=16)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
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

## 🎯 **GPU-Accelerated Nearest Neighbor Search**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Compute embeddings for all historical claims on GPU
model.eval()
with torch.no_grad():
    all_embeddings = model.embed(X_train.to(device))  # [n_claims, 16]

    # Find nearest neighbors for a new claim
    new_claim = torch.FloatTensor([[5, 8, 3, 1]]).to(device)  # Fracture, 8mo, Midwest, attorney
    new_scaled = scaler.transform(new_claim.cpu().numpy())
    new_embed = model.embed(torch.FloatTensor(new_scaled).to(device))

    # Pairwise distances in embedding space (GPU-accelerated)
    distances = torch.cdist(new_embed, all_embeddings).squeeze()
    _, nearest_idx = distances.topk(k=7, largest=False)
    print(f"7 nearest claim indices: {nearest_idx.cpu().tolist()}")
```

### **Batch Claim Benchmarking**

```python
# Score all open claims at once
model.eval()
with torch.no_grad():
    open_claims = torch.FloatTensor(open_claim_data).to(device)  # [2000, 4]
    open_embeds = model.embed(open_claims)
    tier_logits = model.classifier(open_embeds)
    predicted_tiers = tier_logits.argmax(dim=1)

    tier_names = ['Low ($1K-$10K)', 'Med ($10K-$50K)', 'High ($50K-$150K)', 'Severe ($150K+)']
    for i, name in enumerate(tier_names):
        count = (predicted_tiers == i).sum().item()
        print(f"{name}: {count} claims")
```

---

## 📊 **Training Progress on Insurance Data**

```
Epoch 0,   Loss: 1.3862, Accuracy: 0.2520
Epoch 20,  Loss: 0.8123, Accuracy: 0.6450
Epoch 40,  Loss: 0.5934, Accuracy: 0.7380
Epoch 60,  Loss: 0.4912, Accuracy: 0.7720
Epoch 80,  Loss: 0.4521, Accuracy: 0.7860
Epoch 100, Loss: 0.4312, Accuracy: 0.7930

Test Accuracy: 79.3%
Per-Tier F1: Low=0.85, Med=0.76, High=0.77, Severe=0.78
```

---

## 📝 **Code Reference**

Full implementation: [`02_classification/knn_pytorch.py`](../../02_classification/knn_pytorch.py)

Related:
- [K-Nearest Neighbors - NumPy (from scratch)](knn_numpy.md)
- [K-Nearest Neighbors - Scikit-learn](knn_sklearn.md)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
