# Random Forest (PyTorch) - Dental Classification

## **Use Case: Predicting Dental Caries Risk Level**

### **The Problem**
A preventive dentistry program classifies **800 patients** into caries risk levels (low/medium/high) using a neural ensemble approach in PyTorch. Features include saliva flow rate, bacterial count, diet sugar score, fluoride exposure, and brushing frequency.

### **Why PyTorch for Random Forest?**
- Neural ensemble with differentiable tree-like modules
- GPU-accelerated ensemble training
- Integration with dental imaging pipelines
- End-to-end learning with custom dental loss functions

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CariesRiskDataset(Dataset):
    """Dataset for dental caries risk classification (3 classes)."""

    def __init__(self, n_patients=800, seed=42):
        np.random.seed(seed)
        saliva = np.random.uniform(0.1, 2.0, n_patients)
        bacteria = np.random.uniform(1e3, 1e7, n_patients)
        sugar = np.random.uniform(0, 10, n_patients)
        fluoride = np.random.choice([0, 1, 2, 3], n_patients).astype(float)
        brushing = np.random.choice([0, 1, 2, 3], n_patients).astype(float)

        X = np.column_stack([saliva, np.log10(bacteria), sugar, fluoride, brushing])

        risk = (-0.8 * saliva + np.log10(bacteria) / 7 +
                0.3 * sugar - 0.4 * fluoride - 0.3 * brushing)
        terciles = np.percentile(risk, [33.3, 66.6])
        y = np.digitize(risk, terciles)  # 0=low, 1=medium, 2=high

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.n_classes = 3
        self.feature_names = ['saliva_flow', 'log_bacteria', 'sugar_score', 'fluoride', 'brushing']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## **Neural Forest Model**

```python
class NeuralTree(nn.Module):
    """Single neural tree with randomized feature subsets."""

    def __init__(self, n_features, n_classes, hidden_dim=16, feature_fraction=0.6):
        super().__init__()
        n_selected = max(2, int(n_features * feature_fraction))
        self.feature_indices = torch.randperm(n_features)[:n_selected]

        self.tree = nn.Sequential(
            nn.Linear(n_selected, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x):
        x_subset = x[:, self.feature_indices]
        return self.tree(x_subset)


class NeuralForest(nn.Module):
    """Ensemble of neural trees mimicking Random Forest."""

    def __init__(self, n_trees=20, n_features=5, n_classes=3, hidden_dim=16):
        super().__init__()
        self.trees = nn.ModuleList([
            NeuralTree(n_features, n_classes, hidden_dim) for _ in range(n_trees)
        ])

    def forward(self, x):
        # Average predictions across all trees
        predictions = torch.stack([tree(x) for tree in self.trees])
        return predictions.mean(dim=0)
```

---

## **Training Loop**

```python
def train_forest(model, train_loader, val_loader, epochs=100, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        scheduler.step()

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
dataset = CariesRiskDataset(n_patients=800)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

model = NeuralForest(n_trees=20, n_features=5, n_classes=3, hidden_dim=16)
model = train_forest(model, train_loader, val_loader, epochs=100)
```

---

## **Results**

```
Epoch 20/100  | Loss: 0.8543 | Val Acc: 0.7125
Epoch 40/100  | Loss: 0.6234 | Val Acc: 0.7875
Epoch 60/100  | Loss: 0.4987 | Val Acc: 0.8250
Epoch 80/100  | Loss: 0.4123 | Val Acc: 0.8438
Epoch 100/100 | Loss: 0.3765 | Val Acc: 0.8500
```

---

## **Inference**

```python
def assess_caries_risk(model, dataset, patient_features):
    """Predict caries risk level for a new patient."""
    risk_labels = ['LOW', 'MEDIUM', 'HIGH']
    features = np.array(patient_features, dtype=np.float32)
    normalized = (features - dataset.mean) / dataset.std
    tensor = torch.FloatTensor(normalized).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted = probs.argmax().item()

    return {
        'risk_level': risk_labels[predicted],
        'probabilities': {risk_labels[i]: f"{probs[i]:.1%}" for i in range(3)},
        'confidence': f"{probs[predicted]:.1%}"
    }

# High risk patient: low saliva, high bacteria, high sugar, no fluoride, rare brushing
result = assess_caries_risk(model, dataset, [0.2, np.log10(8e6), 9.0, 0, 1])
print(result)
# {'risk_level': 'HIGH', 'probabilities': {'LOW': '3.2%', 'MEDIUM': '12.1%', 'HIGH': '84.7%'}, ...}
```

---

## **Model Export**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_means': dataset.mean,
    'feature_stds': dataset.std,
    'n_trees': 20,
    'n_classes': 3
}, 'caries_risk_neural_forest.pt')
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Traditional Random Forest | Use sklearn (faster, simpler) |
| GPU-accelerated ensemble | PyTorch Neural Forest |
| Integration with imaging | Combine tabular + CNN features |
| Custom loss for dental data | Full control over training |
| Production deployment | Consider ONNX export |

---

## **Running the Demo**

```bash
cd examples/02_classification
python random_forest_pytorch.py
```

---

## **References**

1. Breiman, L. "Random Forests" (2001), Machine Learning
2. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019)
3. Fontana & Zero "Assessing patients' caries risk" (2006), JADA

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
