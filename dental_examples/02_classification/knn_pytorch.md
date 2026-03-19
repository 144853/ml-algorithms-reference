# KNN (PyTorch) - Dental Classification

## **Use Case: Classifying Dental Material Compatibility**

### **The Problem**
A dental materials laboratory uses a PyTorch-based learned distance metric to classify **550 patient-material pairs** as compatible or incompatible. Features include allergy profile, material composition index, and pH sensitivity.

### **Why PyTorch for KNN?**
- Learned distance metrics via Siamese networks
- GPU-accelerated nearest neighbor search
- Differentiable KNN for end-to-end learning
- Integration with material property embeddings

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MaterialCompatibilityDataset(Dataset):
    """Dataset for dental material compatibility classification."""

    def __init__(self, n_pairs=550, seed=42):
        np.random.seed(seed)
        allergy = np.random.uniform(0, 10, n_pairs)
        composition = np.random.uniform(0, 10, n_pairs)
        ph_sensitivity = np.random.uniform(0, 10, n_pairs)

        X = np.column_stack([allergy, composition, ph_sensitivity])

        score = 10 - abs(allergy - composition) - 0.3 * ph_sensitivity
        y = (score > 5.0).astype(np.float32)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))
        self.feature_names = ['allergy_profile', 'material_composition', 'ph_sensitivity']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## **Differentiable KNN Model**

```python
class LearnedDistanceKNN(nn.Module):
    """KNN with a learned feature embedding for distance computation."""

    def __init__(self, n_features=3, embed_dim=16, k=7):
        super().__init__()
        self.k = k
        # Learn a feature embedding that improves distance computation
        self.embedding = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        # Attention weights for distance dimensions
        self.distance_weights = nn.Parameter(torch.ones(embed_dim))

    def embed(self, x):
        return self.embedding(x)

    def forward(self, query, support_x, support_y):
        """
        query: (batch, features) - new patient-material pairs
        support_x: (N, features) - known cases
        support_y: (N,) - known outcomes
        """
        q_embed = self.embed(query)       # (batch, embed_dim)
        s_embed = self.embed(support_x)   # (N, embed_dim)

        # Weighted Euclidean distance
        weights = torch.softmax(self.distance_weights, dim=0)
        diff = q_embed.unsqueeze(1) - s_embed.unsqueeze(0)  # (batch, N, embed_dim)
        distances = (diff ** 2 * weights).sum(dim=2)          # (batch, N)

        # Find k nearest neighbors
        _, knn_indices = distances.topk(self.k, largest=False)  # (batch, k)

        # Weighted vote
        knn_labels = support_y[knn_indices].float()  # (batch, k)
        knn_dists = torch.gather(distances, 1, knn_indices)
        knn_weights = 1.0 / (knn_dists + 1e-8)
        knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)

        # Soft prediction
        prob_compatible = (knn_weights * knn_labels).sum(dim=1)
        return prob_compatible


class PrototypicalKNN(nn.Module):
    """Prototypical network approach - learns class prototypes."""

    def __init__(self, n_features=3, embed_dim=16):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        return self.classifier(embedded)
```

---

## **Training Loop**

```python
def train_proto_knn(model, train_loader, val_loader, epochs=100, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = model(X_batch).argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {correct/total:.4f}")

    return model
```

---

## **Full Pipeline**

```python
dataset = MaterialCompatibilityDataset(n_pairs=550)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

model = PrototypicalKNN(n_features=3, embed_dim=16)
model = train_proto_knn(model, train_loader, val_loader, epochs=100)
```

---

## **Results**

```
Epoch 20/100  | Loss: 0.5876 | Val Acc: 0.7636
Epoch 40/100  | Loss: 0.4321 | Val Acc: 0.8091
Epoch 60/100  | Loss: 0.3567 | Val Acc: 0.8364
Epoch 80/100  | Loss: 0.3102 | Val Acc: 0.8455
Epoch 100/100 | Loss: 0.2876 | Val Acc: 0.8545
```

---

## **Inference**

```python
def check_compatibility(model, dataset, patient_material_features):
    """Check material compatibility for a new patient-material pair."""
    features = np.array(patient_material_features, dtype=np.float32)
    normalized = (features - dataset.mean) / dataset.std
    tensor = torch.FloatTensor(normalized).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted = probs.argmax().item()

    labels = ['Incompatible', 'Compatible']
    return {
        'result': labels[predicted],
        'compatible_probability': f"{probs[1]:.1%}",
        'confidence': f"{probs[predicted]:.1%}",
        'recommendation': 'Safe to use' if predicted == 1 else 'Select alternative material'
    }

result = check_compatibility(model, dataset, [8.0, 2.0, 7.5])
print(result)
# {'result': 'Incompatible', 'compatible_probability': '18.2%', ...}
```

---

## **Model Export**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_means': dataset.mean,
    'feature_stds': dataset.std,
    'feature_names': dataset.feature_names
}, 'material_compatibility_knn.pt')
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Traditional KNN | Use sklearn (simpler, exact neighbors) |
| Learned distance metric | PyTorch Siamese/Prototypical networks |
| Few-shot learning (new materials) | Prototypical networks excel |
| Large material database + GPU | PyTorch scales with GPU |
| Interpretable neighbor lookup | Use sklearn for direct neighbor access |

---

## **Running the Demo**

```bash
cd examples/02_classification
python knn_pytorch.py
```

---

## **References**

1. Cover & Hart "Nearest Neighbor Pattern Classification" (1967)
2. Snell et al. "Prototypical Networks for Few-shot Learning" (2017), NeurIPS
3. Schmalz & Fair "Biocompatibility of dental materials" (2009)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
