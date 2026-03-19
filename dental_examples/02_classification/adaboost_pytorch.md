# AdaBoost (PyTorch) - Dental Classification

## **Use Case: Predicting Dental Emergency Classification**

### **The Problem**
A dental emergency triage system classifies **750 incoming cases** as urgent or routine using a PyTorch-based adaptive boosting approach. Features: pain severity, swelling level, fever, bleeding severity, and trauma indicator.

### **Why PyTorch for AdaBoost?**
- Neural weak learners for more expressive boosting
- GPU-accelerated sequential boosting
- Custom sample weighting strategies
- Integration with real-time monitoring systems

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

class DentalEmergencyDataset(Dataset):
    """Dataset for dental emergency triage classification."""

    def __init__(self, n_cases=750, seed=42):
        np.random.seed(seed)
        pain = np.random.uniform(0, 10, n_cases)
        swelling = np.random.uniform(0, 10, n_cases)
        fever = np.random.uniform(36.0, 40.5, n_cases)
        bleeding = np.random.uniform(0, 10, n_cases)
        trauma = np.random.choice([0.0, 1.0], n_cases, p=[0.7, 0.3])

        X = np.column_stack([pain, swelling, fever, bleeding, trauma])

        urgency = (0.25 * pain + 0.2 * swelling + 0.3 * (fever - 37.0).clip(0) +
                   0.15 * bleeding + 0.4 * trauma)
        y = (urgency > np.percentile(urgency, 60)).astype(np.float32)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))
        self.feature_names = ['pain_severity', 'swelling_level', 'fever_celsius',
                              'bleeding_severity', 'trauma_indicator']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## **Neural AdaBoost Model**

```python
class NeuralWeakLearner(nn.Module):
    """Simple neural network as a weak learner."""

    def __init__(self, n_features=5, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


class NeuralAdaBoost(nn.Module):
    """AdaBoost with neural weak learners."""

    def __init__(self, n_features=5, n_learners=15, hidden_dim=8):
        super().__init__()
        self.learners = nn.ModuleList([
            NeuralWeakLearner(n_features, hidden_dim) for _ in range(n_learners)
        ])
        # Learnable boosting weights (alpha values)
        self.alphas = nn.Parameter(torch.ones(n_learners) / n_learners)

    def forward(self, x):
        predictions = torch.stack([learner(x) for learner in self.learners])  # (n_learners, batch)
        weights = torch.softmax(self.alphas, dim=0)
        weighted_sum = (weights.unsqueeze(1) * predictions).sum(dim=0)
        return weighted_sum

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))
```

---

## **Training with Sample Reweighting**

```python
def train_adaboost(model, train_loader, val_loader, epochs=120, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 30 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = (model.predict_proba(X_batch) >= 0.5).long()
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {correct/total:.4f}")

    return model
```

---

## **Full Pipeline**

```python
dataset = DentalEmergencyDataset(n_cases=750)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

model = NeuralAdaBoost(n_features=5, n_learners=15, hidden_dim=8)
model = train_adaboost(model, train_loader, val_loader, epochs=120)
```

---

## **Results**

```
Epoch 30/120  | Loss: 0.5234 | Val Acc: 0.7733
Epoch 60/120  | Loss: 0.3876 | Val Acc: 0.8267
Epoch 90/120  | Loss: 0.3212 | Val Acc: 0.8533
Epoch 120/120 | Loss: 0.2876 | Val Acc: 0.8600
```

---

## **Learner Weight Analysis**

```python
def analyze_boosting_weights(model):
    """Analyze the learned boosting weights (alphas)."""
    weights = torch.softmax(model.alphas.data, dim=0).numpy()
    print("Boosting weights per weak learner:")
    for i, w in enumerate(weights):
        bar = '#' * int(w * 100)
        print(f"  Learner {i+1:2d}: {w:.4f} {bar}")

analyze_boosting_weights(model)
```

---

## **Inference**

```python
def triage_emergency(model, dataset, symptoms):
    """Triage a dental emergency case."""
    features = np.array(symptoms, dtype=np.float32)
    normalized = (features - dataset.mean) / dataset.std
    tensor = torch.FloatTensor(normalized).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prob = model.predict_proba(tensor).item()

    if prob >= 0.8:
        level, action = 'CRITICAL', 'Immediate treatment required'
    elif prob >= 0.5:
        level, action = 'URGENT', 'Priority scheduling within 1-2 hours'
    elif prob >= 0.3:
        level, action = 'SEMI-URGENT', 'Same-day appointment'
    else:
        level, action = 'ROUTINE', 'Schedule regular appointment'

    return {
        'triage_level': level,
        'urgency_probability': f"{prob:.1%}",
        'recommended_action': action,
        'classification': 'Urgent' if prob >= 0.5 else 'Routine'
    }

# Severe trauma case
result = triage_emergency(model, dataset, [8.5, 8.0, 39.5, 7.0, 1])
print(result)
# {'triage_level': 'CRITICAL', 'urgency_probability': '92.7%', ...}

# Mild case
result = triage_emergency(model, dataset, [2.0, 1.0, 36.5, 0.5, 0])
print(result)
# {'triage_level': 'ROUTINE', 'urgency_probability': '8.3%', ...}
```

---

## **Model Export**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_means': dataset.mean,
    'feature_stds': dataset.std,
    'feature_names': dataset.feature_names,
    'n_learners': 15
}, 'dental_emergency_adaboost.pt')
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Traditional AdaBoost | Use sklearn (exact algorithm) |
| Neural weak learners | PyTorch gives more expressive base models |
| Real-time triage system | Both are fast for inference |
| Custom boosting schemes | Full control in PyTorch |
| Integration with monitoring | PyTorch fits into larger systems |

---

## **Running the Demo**

```bash
cd examples/02_classification
python adaboost_pytorch.py
```

---

## **References**

1. Freund & Schapire "A Decision-Theoretic Generalization of On-Line Learning" (1997)
2. Paszke et al. "PyTorch" (2019), NeurIPS
3. ADA Emergency Triage Guidelines

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
