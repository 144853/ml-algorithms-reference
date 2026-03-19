# Ridge Regression (PyTorch) - Orthodontic Treatment Duration Prediction

## 🦷 **Use Case: Predicting Orthodontic Treatment Duration**

GPU-accelerated Ridge regression for predicting treatment duration (6-48 months) from malocclusion severity, patient age, compliance score, and bracket type. In PyTorch, L2 regularization is implemented via the `weight_decay` parameter in the optimizer.

---

## 📦 **Quick Start**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_tensor = torch.FloatTensor(X_train_scaled)
y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Linear(in_features=4, out_features=1)
criterion = nn.MSELoss()

# weight_decay is the L2 regularization (alpha) parameter
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1.0)

for epoch in range(1000):
    for X_batch, y_batch in loader:
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🏗️ **Custom Dataset for Orthodontic Data**

```python
from torch.utils.data import Dataset

class OrthodonticDataset(Dataset):
    """Dataset for orthodontic treatment duration prediction."""

    def __init__(self, features, durations):
        """
        Args:
            features: numpy array of shape (n_patients, 4)
                      [malocclusion_severity, patient_age, compliance_score, bracket_type]
            durations: numpy array of treatment durations in months
        """
        self.features = torch.FloatTensor(features)
        self.durations = torch.FloatTensor(durations).unsqueeze(1)

    def __len__(self):
        return len(self.durations)

    def __getitem__(self, idx):
        return self.features[idx], self.durations[idx]

train_dataset = OrthodonticDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## 🔧 **Model with Explicit L2 Regularization**

```python
class RidgeRegressionModel(nn.Module):
    """Ridge regression with explicit L2 penalty for orthodontic duration prediction."""

    def __init__(self, n_features=4):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def l2_penalty(self):
        """Compute L2 penalty on weights (not bias)."""
        return torch.sum(self.linear.weight ** 2)

model = RidgeRegressionModel(n_features=4)
```

### **Training with Explicit L2 Loss**
```python
def train_with_explicit_l2(model, train_loader, alpha=1.0, lr=0.01, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            l2_loss = alpha * model.l2_penalty()
            total_loss = mse_loss + l2_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}: MSE={mse_loss.item():.2f}, "
                  f"L2={l2_loss.item():.4f}, Total={total_loss.item():.2f}")
```

### **Alternative: Using weight_decay (Simpler)**
```python
# Equivalent to explicit L2, but built into the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0)
# Only MSE loss needed -- L2 is applied during parameter update
```

---

## 🏋️ **Training Loop with Alpha Search**

```python
def find_best_alpha(X_train, y_train, X_val, y_val, alphas):
    """Cross-validate to find best L2 regularization strength."""
    best_alpha, best_val_loss = None, float('inf')

    for alpha in alphas:
        model = RidgeRegressionModel(n_features=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=alpha)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(torch.FloatTensor(X_train),
                                  torch.FloatTensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        for epoch in range(500):
            model.train()
            for xb, yb in train_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_pred, torch.FloatTensor(y_val).unsqueeze(1)).item()

        print(f"alpha={alpha:.4f}: Val MSE={val_loss:.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_alpha = alpha

    return best_alpha

alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
best = find_best_alpha(X_train_scaled, y_train, X_val_scaled, y_val, alphas)
print(f"Best alpha: {best}")
```

---

## 🖥️ **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RidgeRegressionModel(n_features=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0)

for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    pred = model(X_batch)
    loss = nn.MSELoss()(pred, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 📈 **Evaluation**

```python
model.eval()
with torch.no_grad():
    preds = model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()

rmse = np.sqrt(np.mean((y_test - preds) ** 2))
print(f"Test RMSE: {rmse:.2f} months")
# Test RMSE: 3.02 months

# Coefficient interpretation
weights = model.linear.weight.data.squeeze().numpy()
feature_names = ['severity', 'age', 'compliance', 'bracket_type']
for name, w in zip(feature_names, weights):
    print(f"  {name}: {w:.4f}")
```

---

## 💾 **Model Saving**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': 1.0,
    'feature_names': ['severity', 'age', 'compliance', 'bracket_type'],
    'scaler_mean': scaler.mean_,
    'scaler_std': scaler.scale_,
}, 'ortho_ridge_model.pt')
```

---

## 🎯 **When to Use PyTorch Ridge**

- When **extending to deeper architectures** with L2 regularization
- For **batch-level control** over the regularization computation
- When you need **per-parameter group** weight decay (e.g., different alpha for severity vs bracket)
- For **large-scale orthodontic datasets** benefiting from GPU acceleration
- When integrating into an existing **PyTorch training pipeline**

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/ridge_regression_pytorch.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
