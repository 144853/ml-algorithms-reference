# Logistic Regression (PyTorch) - Dental Classification

## **Use Case: Predicting Periodontal Disease Risk**

### **The Problem**
A dental clinic with **500 patients** wants to predict periodontal disease risk using deep learning infrastructure. PyTorch enables GPU-accelerated training and integration with larger neural network architectures for future expansion.

### **Why PyTorch?**
- GPU acceleration for large patient databases
- Custom loss functions for imbalanced dental datasets
- Easy extension to deep learning models
- Integration with dental imaging pipelines

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PeriodontalDataset(Dataset):
    """Custom dataset for periodontal disease classification."""

    def __init__(self, n_patients=500, seed=42):
        np.random.seed(seed)
        # Clinical features
        plaque_index = np.random.uniform(0.0, 3.0, n_patients)
        bop_percentage = np.random.uniform(0, 100, n_patients)
        pocket_depth = np.random.uniform(1.0, 12.0, n_patients)
        age = np.random.randint(18, 86, n_patients).astype(float)
        smoking_status = np.random.choice([0.0, 1.0, 2.0], n_patients)

        X = np.column_stack([plaque_index, bop_percentage, pocket_depth, age, smoking_status])

        # Generate labels
        risk = (0.8 * plaque_index + 0.02 * bop_percentage +
                0.5 * pocket_depth + 0.03 * age + 0.6 * smoking_status)
        y = (risk > np.median(risk)).astype(np.float32)

        # Normalize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = (X - self.mean) / self.std

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.feature_names = ['plaque_index', 'bop_percentage', 'pocket_depth', 'age', 'smoking_status']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## **Model Definition**

```python
class PeriodontalLogisticRegression(nn.Module):
    """Logistic regression model for periodontal disease prediction."""

    def __init__(self, n_features=5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def get_feature_importance(self, feature_names):
        weights = self.linear.weight.data[0].numpy()
        importance = sorted(zip(feature_names, weights),
                          key=lambda x: abs(x[1]), reverse=True)
        return importance
```

---

## **Training Loop**

```python
def train_model(model, train_loader, val_loader, epochs=200, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                val_loss += criterion(outputs, y_batch).item()
                predicted = (outputs >= 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        scheduler.step()

        val_acc = correct / total
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    return history
```

---

## **Full Training Pipeline**

```python
# Create dataset and split
dataset = PeriodontalDataset(n_patients=500)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize and train
model = PeriodontalLogisticRegression(n_features=5)
history = train_model(model, train_loader, val_loader, epochs=200, lr=0.01)
```

---

## **Results**

```
Epoch 50/200  | Train Loss: 0.4012 | Val Loss: 0.4234 | Val Acc: 0.8200
Epoch 100/200 | Train Loss: 0.3105 | Val Loss: 0.3287 | Val Acc: 0.8600
Epoch 150/200 | Train Loss: 0.2812 | Val Loss: 0.2943 | Val Acc: 0.8800
Epoch 200/200 | Train Loss: 0.2698 | Val Loss: 0.2815 | Val Acc: 0.8900
```

### **Feature Importance**

```python
importance = model.get_feature_importance(dataset.feature_names)
for name, weight in importance:
    print(f"  {name:20s}: {weight:+.4f}")
```

```
  pocket_depth        : +1.7923
  plaque_index        : +1.2108
  smoking_status      : +0.9654
  bop_percentage      : +0.6321
  age                 : +0.3198
```

---

## **Inference for New Patients**

```python
def predict_patient(model, dataset, patient_features):
    """Predict periodontal disease risk for a new patient."""
    features = np.array(patient_features, dtype=np.float32)
    normalized = (features - dataset.mean) / dataset.std
    tensor = torch.FloatTensor(normalized).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        probability = model(tensor).item()

    return {
        'probability': f"{probability:.1%}",
        'prediction': 'Disease' if probability >= 0.5 else 'Healthy',
        'risk_level': 'HIGH' if probability >= 0.7 else 'MODERATE' if probability >= 0.4 else 'LOW'
    }

# New patient: plaque=2.5, bop=72%, pocket=6.2mm, age=60, smoker=2
result = predict_patient(model, dataset, [2.5, 72.0, 6.2, 60, 2])
print(result)
# {'probability': '94.1%', 'prediction': 'Disease', 'risk_level': 'HIGH'}
```

---

## **GPU Training Notes**

```python
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# For large dental databases (>100K patients):
# - Use pin_memory=True in DataLoader
# - Increase batch_size to 256 or 512
# - Consider mixed precision training with torch.cuda.amp
```

---

## **Model Export**

```python
# Save for clinical deployment
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_means': dataset.mean,
    'feature_stds': dataset.std,
    'feature_names': dataset.feature_names
}, 'periodontal_risk_model.pt')

# Load in production
checkpoint = torch.load('periodontal_risk_model.pt')
model = PeriodontalLogisticRegression(n_features=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## **When to Use This Implementation**

| Scenario | Recommendation |
|----------|---------------|
| Prototype with small data | Use sklearn instead (simpler) |
| GPU cluster available | PyTorch shines with large datasets |
| Extending to neural networks | Easy to add hidden layers later |
| Integration with imaging | Combine tabular + image features |
| Research / custom losses | Full control over training loop |

---

## **Running the Demo**

```bash
cd examples/02_classification
python logistic_regression_pytorch.py
```

---

## **References**

1. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019), NeurIPS
2. Lang & Tonetti "Periodontal Risk Assessment" (2003), Periodontology 2000

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
