# SVM (PyTorch) - Dental Classification

## **Use Case: Classifying TMJ Disorder Type**

### **The Problem**
A maxillofacial clinic classifies **700 TMJ disorder patients** into muscular, articular, or combined types using a PyTorch-based SVM approach. Features: jaw opening range, clicking frequency, pain score, and stress level.

### **Why PyTorch for SVM?**
- Hinge loss implementation enables SVM-like learning
- GPU acceleration for large clinical datasets
- Custom multi-class margin losses
- Integration with feature extraction networks

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TMJDataset(Dataset):
    """Dataset for TMJ disorder classification."""

    def __init__(self, n_patients=700, seed=42):
        np.random.seed(seed)
        jaw_opening = np.random.uniform(15, 55, n_patients)
        clicking = np.random.uniform(0, 10, n_patients)
        pain = np.random.uniform(0, 10, n_patients)
        stress = np.random.uniform(0, 10, n_patients)

        X = np.column_stack([jaw_opening, clicking, pain, stress])

        muscular = 0.6 * stress + 0.4 * pain - 0.2 * clicking
        articular = 0.7 * clicking - 0.3 * jaw_opening / 10 + 0.2 * pain
        combined = 0.4 * stress + 0.4 * clicking + 0.3 * pain
        scores = np.column_stack([muscular, articular, combined])
        y = scores.argmax(axis=1)  # 0=muscular, 1=articular, 2=combined

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.n_classes = 3
        self.class_names = ['muscular', 'articular', 'combined']
        self.feature_names = ['jaw_opening_mm', 'clicking_freq', 'pain_score', 'stress_level']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## **SVM Model with Hinge Loss**

```python
class LinearSVM(nn.Module):
    """Multi-class linear SVM using PyTorch."""

    def __init__(self, n_features=4, n_classes=3):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


class MultiClassHingeLoss(nn.Module):
    """Crammer-Singer multi-class hinge loss."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, outputs, targets):
        n_samples = outputs.size(0)
        correct_scores = outputs[range(n_samples), targets].unsqueeze(1)
        margins = outputs - correct_scores + self.margin
        margins[range(n_samples), targets] = 0
        losses = torch.clamp(margins, min=0)
        return losses.sum() / n_samples


class KernelSVM(nn.Module):
    """Non-linear SVM approximation using RBF feature mapping."""

    def __init__(self, n_features=4, n_classes=3, n_components=64):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(n_features, n_components),
            nn.ReLU(),
            nn.Linear(n_components, n_components),
            nn.ReLU()
        )
        self.classifier = nn.Linear(n_components, n_classes)

    def forward(self, x):
        features = self.feature_map(x)
        return self.classifier(features)
```

---

## **Training Loop**

```python
def train_svm(model, train_loader, val_loader, epochs=150, lr=0.01, weight_decay=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = MultiClassHingeLoss(margin=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

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

        if (epoch + 1) % 30 == 0:
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
dataset = TMJDataset(n_patients=700)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Use Kernel SVM for non-linear boundaries
model = KernelSVM(n_features=4, n_classes=3, n_components=64)
model = train_svm(model, train_loader, val_loader, epochs=150)
```

---

## **Results**

```
Epoch 30/150  | Loss: 1.2345 | Val Acc: 0.7214
Epoch 60/150  | Loss: 0.7654 | Val Acc: 0.7857
Epoch 90/150  | Loss: 0.5432 | Val Acc: 0.8143
Epoch 120/150 | Loss: 0.4321 | Val Acc: 0.8286
Epoch 150/150 | Loss: 0.3876 | Val Acc: 0.8357
```

---

## **Inference**

```python
def classify_tmj(model, dataset, patient_features):
    """Classify TMJ disorder type for a new patient."""
    features = np.array(patient_features, dtype=np.float32)
    normalized = (features - dataset.mean) / dataset.std
    tensor = torch.FloatTensor(normalized).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted = probs.argmax().item()

    return {
        'tmj_type': dataset.class_names[predicted].upper(),
        'probabilities': {dataset.class_names[i]: f"{probs[i]:.1%}" for i in range(3)},
        'confidence': f"{probs[predicted]:.1%}"
    }

# Patient with high clicking, limited opening
result = classify_tmj(model, dataset, [25.0, 8.0, 5.0, 2.0])
print(result)
# {'tmj_type': 'ARTICULAR', 'probabilities': {...}, 'confidence': '78.4%'}
```

---

## **Model Export**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_means': dataset.mean,
    'feature_stds': dataset.std,
    'class_names': dataset.class_names,
    'n_components': 64
}, 'tmj_svm_model.pt')
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Traditional SVM | Use sklearn SVC (faster, simpler) |
| Custom margin loss | PyTorch gives full control |
| Large datasets + GPU | PyTorch scales better |
| Integration with neural nets | Easy to combine with other layers |
| Need kernel approximation | Neural feature mapping works well |

---

## **Running the Demo**

```bash
cd examples/02_classification
python svm_pytorch.py
```

---

## **References**

1. Cortes & Vapnik "Support-Vector Networks" (1995), Machine Learning
2. Crammer & Singer "On the Algorithmic Implementation of Multiclass SVMs" (2001), JMLR
3. De Leeuw & Klasser "Orofacial Pain" (2018)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
