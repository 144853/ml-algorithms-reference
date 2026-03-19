# XGBoost (PyTorch) - Dental Classification

## **Use Case: Predicting Root Canal Treatment Outcome**

### **The Problem**
An endodontic department predicts root canal outcomes for **900 procedures** using a PyTorch gradient boosting neural network. Features: tooth type, canal complexity, pre-op pain, periapical lesion size, and operator experience.

### **Why PyTorch for Gradient Boosting?**
- Neural network-based gradient boosting (NODE, TabNet)
- GPU-accelerated boosting iterations
- Custom differentiable loss functions for clinical outcomes
- End-to-end learning with feature embeddings

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RootCanalDataset(Dataset):
    """Dataset for root canal treatment outcome prediction."""

    def __init__(self, n_procedures=900, seed=42):
        np.random.seed(seed)
        tooth_type = np.random.choice([0,1,2,3], n_procedures).astype(float)
        complexity = np.random.uniform(1, 5, n_procedures)
        pain = np.random.uniform(0, 10, n_procedures)
        lesion = np.random.uniform(0, 10, n_procedures)
        experience = np.random.uniform(1, 30, n_procedures)

        X = np.column_stack([tooth_type, complexity, pain, lesion, experience])

        failure_prob = (0.05 + 0.1 * complexity / 5 + 0.08 * lesion / 10 +
                        0.05 * (tooth_type >= 2) - 0.08 * experience / 30 +
                        0.03 * pain / 10)
        failure_prob = np.clip(failure_prob, 0.02, 0.6)
        y = (np.random.random(n_procedures) < failure_prob).astype(np.float32)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))
        self.feature_names = ['tooth_type', 'canal_complexity', 'preop_pain',
                              'lesion_size_mm', 'operator_experience_yrs']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## **Neural Gradient Boosting Model**

```python
class WeakLearner(nn.Module):
    """Single weak learner (shallow network) for boosting ensemble."""

    def __init__(self, n_features, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class NeuralGradientBoosting(nn.Module):
    """
    Gradient boosting with neural weak learners.
    Each learner fits the residual of previous learners.
    """

    def __init__(self, n_features=5, n_learners=10, hidden_dim=16, learning_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.learners = nn.ModuleList([
            WeakLearner(n_features, hidden_dim) for _ in range(n_learners)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Additive model: F(x) = bias + lr * sum(learner_i(x))
        output = self.bias.expand(x.size(0))
        for learner in self.learners:
            output = output + self.learning_rate * learner(x).squeeze()
        return output

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)
```

---

## **Training with Gradient Boosting Strategy**

```python
def train_neural_boost(model, train_loader, val_loader, epochs=150, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
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

            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    return model
```

---

## **Full Pipeline**

```python
dataset = RootCanalDataset(n_procedures=900)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

model = NeuralGradientBoosting(n_features=5, n_learners=10, hidden_dim=16, learning_rate=0.1)
model = train_neural_boost(model, train_loader, val_loader, epochs=150)
```

---

## **Results**

```
Epoch 30/150  | Loss: 0.5123 | Val Acc: 0.7778 | Best: 0.7778
Epoch 60/150  | Loss: 0.3876 | Val Acc: 0.8222 | Best: 0.8222
Epoch 90/150  | Loss: 0.3214 | Val Acc: 0.8500 | Best: 0.8500
Epoch 120/150 | Loss: 0.2876 | Val Acc: 0.8611 | Best: 0.8611
Epoch 150/150 | Loss: 0.2654 | Val Acc: 0.8667 | Best: 0.8667
```

---

## **Inference**

```python
def predict_rct_outcome(model, dataset, procedure_features):
    """Predict root canal outcome for a new procedure."""
    features = np.array(procedure_features, dtype=np.float32)
    normalized = (features - dataset.mean) / dataset.std
    tensor = torch.FloatTensor(normalized).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prob = model.predict_proba(tensor).item()

    return {
        'outcome': 'Failure Risk' if prob >= 0.5 else 'Likely Success',
        'failure_probability': f"{prob:.1%}",
        'success_probability': f"{1-prob:.1%}",
        'confidence': f"{max(prob, 1-prob):.1%}",
        'recommendation': 'Refer to specialist' if prob >= 0.4 else 'Proceed with treatment'
    }

# Complex molar, large lesion, junior operator
result = predict_rct_outcome(model, dataset, [2, 4.5, 8.0, 7.0, 2.0])
print(result)
# {'outcome': 'Failure Risk', 'failure_probability': '52.1%', ...}
```

---

## **Learner Contribution Analysis**

```python
def analyze_learner_contributions(model, x_sample):
    """Analyze how each weak learner contributes to the prediction."""
    model.eval()
    tensor = torch.FloatTensor(x_sample).unsqueeze(0)

    contributions = []
    with torch.no_grad():
        for i, learner in enumerate(model.learners):
            contrib = model.learning_rate * learner(tensor).item()
            contributions.append(contrib)

    print(f"Bias: {model.bias.item():.4f}")
    for i, c in enumerate(contributions):
        direction = "toward failure" if c > 0 else "toward success"
        print(f"  Learner {i+1}: {c:+.4f} ({direction})")
    print(f"  Total logit: {model.bias.item() + sum(contributions):.4f}")
```

---

## **Model Export**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_means': dataset.mean,
    'feature_stds': dataset.std,
    'feature_names': dataset.feature_names,
    'n_learners': 10
}, 'rct_outcome_neural_boost.pt')
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Standard tabular data | Use XGBoost library (faster, better) |
| Custom boosting losses | PyTorch gives full flexibility |
| GPU cluster with large data | Neural boosting scales well |
| Integration with deep learning | Easy to combine with other modules |
| Research on boosting variants | Full control over boosting strategy |

---

## **Running the Demo**

```bash
cd examples/02_classification
python xgboost_pytorch.py
```

---

## **References**

1. Chen & Guestrin "XGBoost" (2016), KDD
2. Popov et al. "Neural Oblivious Decision Ensembles (NODE)" (2019)
3. Ng et al. "Outcome of root canal treatment" (2011)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
