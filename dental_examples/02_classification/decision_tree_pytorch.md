# Decision Tree (PyTorch) - Dental Classification

## **Use Case: Classifying Tooth Extraction Necessity**

### **The Problem**
An oral surgery department uses a neural network-based decision tree approach to classify **600 teeth** as requiring extraction or preservation, based on mobility grade, bone loss percentage, infection status, and restorability score.

### **Why PyTorch for Decision Trees?**
- Soft decision trees with differentiable splits (Neural Decision Trees)
- GPU-accelerated training for large patient datasets
- End-to-end learning with gradient descent
- Can be combined with embedding layers for complex dental features

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ToothExtractionDataset(Dataset):
    """Dataset for tooth extraction classification."""

    def __init__(self, n_patients=600, seed=42):
        np.random.seed(seed)
        mobility = np.random.choice([0, 1, 2, 3], n_patients, p=[0.4, 0.3, 0.2, 0.1]).astype(float)
        bone_loss = np.random.uniform(0, 80, n_patients)
        infection = np.random.choice([0.0, 1.0], n_patients, p=[0.6, 0.4])
        restorability = np.random.uniform(0, 10, n_patients)

        X = np.column_stack([mobility, bone_loss, infection, restorability])

        score = mobility * 2 + bone_loss / 20 + infection * 1.5 - restorability * 0.5
        y = (score > 2.0).astype(np.float32)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.feature_names = ['mobility_grade', 'bone_loss_pct', 'infection_status', 'restorability_score']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## **Neural Decision Tree Model**

```python
class NeuralDecisionTree(nn.Module):
    """
    Soft decision tree implemented as a neural network.
    Each internal node learns a soft split using sigmoid.
    """

    def __init__(self, n_features=4, depth=4):
        super().__init__()
        self.depth = depth
        n_internal = 2 ** depth - 1
        n_leaves = 2 ** depth

        # Each internal node has a linear split function
        self.split_weights = nn.Linear(n_features, n_internal)
        # Each leaf has a probability output
        self.leaf_probs = nn.Parameter(torch.randn(n_leaves))

    def forward(self, x):
        batch_size = x.size(0)
        # Compute split decisions (soft routing)
        split_decisions = torch.sigmoid(self.split_weights(x))

        # Route through tree
        # Start at root (all probability mass at node 0)
        node_probs = torch.ones(batch_size, 1, device=x.device)

        for d in range(self.depth):
            n_nodes = 2 ** d
            start_idx = 2 ** d - 1

            left_probs = []
            right_probs = []

            for i in range(n_nodes):
                node_idx = start_idx + i
                split = split_decisions[:, node_idx:node_idx+1]
                parent_prob = node_probs[:, i:i+1]
                left_probs.append(parent_prob * (1 - split))
                right_probs.append(parent_prob * split)

            # Interleave left and right children
            children = []
            for l, r in zip(left_probs, right_probs):
                children.extend([l, r])
            node_probs = torch.cat(children, dim=1)

        # Weighted sum of leaf predictions
        leaf_outputs = torch.sigmoid(self.leaf_probs)
        prediction = (node_probs * leaf_outputs.unsqueeze(0)).sum(dim=1)
        return prediction
```

---

## **Training Loop**

```python
def train_neural_tree(model, train_loader, val_loader, epochs=150, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
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

        if (epoch + 1) % 30 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = (model(X_batch) >= 0.5).float()
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {correct/total:.4f}")

    return model
```

---

## **Full Pipeline**

```python
dataset = ToothExtractionDataset(n_patients=600)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

model = NeuralDecisionTree(n_features=4, depth=4)
model = train_neural_tree(model, train_loader, val_loader, epochs=150)
```

---

## **Results**

```
Epoch 30/150  | Loss: 0.4987 | Val Acc: 0.7917
Epoch 60/150  | Loss: 0.3654 | Val Acc: 0.8500
Epoch 90/150  | Loss: 0.2987 | Val Acc: 0.8750
Epoch 120/150 | Loss: 0.2654 | Val Acc: 0.8833
Epoch 150/150 | Loss: 0.2489 | Val Acc: 0.8917
```

---

## **Inference**

```python
def predict_extraction(model, dataset, patient_features):
    """Predict extraction necessity for a new tooth."""
    features = np.array(patient_features, dtype=np.float32)
    normalized = (features - dataset.mean) / dataset.std
    tensor = torch.FloatTensor(normalized).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prob = model(tensor).item()

    return {
        'decision': 'Extract' if prob >= 0.5 else 'Preserve',
        'extraction_probability': f"{prob:.1%}",
        'confidence': f"{max(prob, 1-prob):.1%}"
    }

# Severely compromised tooth: mobility=3, bone_loss=60%, infected, restorability=2.0
result = predict_extraction(model, dataset, [3, 60.0, 1, 2.0])
print(result)
# {'decision': 'Extract', 'extraction_probability': '93.5%', 'confidence': '93.5%'}
```

---

## **Model Export**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_means': dataset.mean,
    'feature_stds': dataset.std,
    'feature_names': dataset.feature_names,
    'tree_depth': 4
}, 'tooth_extraction_neural_tree.pt')
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Need traditional decision tree | Use sklearn DecisionTreeClassifier |
| Differentiable decision tree | Use this Neural Decision Tree |
| Large datasets with GPU | PyTorch scales well |
| Ensemble with other neural nets | Easy integration with PyTorch models |
| Interpretability priority | Traditional sklearn tree is more interpretable |

---

## **Running the Demo**

```bash
cd examples/02_classification
python decision_tree_pytorch.py
```

---

## **References**

1. Kontschieder et al. "Deep Neural Decision Forests" (2015), ICCV
2. Frosst & Hinton "Distilling a Neural Network Into a Soft Decision Tree" (2017)
3. Breiman et al. "Classification and Regression Trees" (1984)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
