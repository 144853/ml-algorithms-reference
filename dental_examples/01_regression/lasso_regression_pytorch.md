# Lasso Regression (PyTorch) - Dental Implant Success Prediction

## 🦷 **Use Case: Predicting Dental Implant Success Score**

PyTorch implementation of Lasso (L1-regularized) regression for predicting implant success scores (0-100) from bone density, gum health, smoking status, diabetes indicator, and implant material. Unlike Ridge, L1 regularization requires proximal gradient descent since the L1 norm is not differentiable at zero.

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

model = nn.Linear(in_features=5, out_features=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
alpha = 0.1  # L1 regularization strength

for epoch in range(1000):
    for X_batch, y_batch in loader:
        predictions = model(X_batch)
        mse_loss = criterion(predictions, y_batch)
        l1_loss = alpha * torch.sum(torch.abs(model.weight))
        total_loss = mse_loss + l1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## 🏗️ **Custom Dataset**

```python
from torch.utils.data import Dataset

class ImplantDataset(Dataset):
    """Dataset for dental implant success prediction."""

    def __init__(self, features, scores):
        """
        Args:
            features: numpy array (n_patients, 5)
                      [bone_density, gum_health, smoking, diabetes, material]
            scores: numpy array of success scores (0-100)
        """
        self.features = torch.FloatTensor(features)
        self.scores = torch.FloatTensor(scores).unsqueeze(1)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.features[idx], self.scores[idx]
```

---

## 🔧 **Model with Proximal Gradient Descent**

```python
class LassoRegressionModel(nn.Module):
    """Lasso regression for dental implant success prediction."""

    def __init__(self, n_features=5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def l1_penalty(self):
        """Compute L1 penalty on weights (not bias)."""
        return torch.sum(torch.abs(self.linear.weight))

    def get_sparse_coefficients(self, feature_names, threshold=1e-4):
        """Return coefficients, marking near-zero as exactly zero."""
        weights = self.linear.weight.data.squeeze().numpy()
        sparse_weights = np.where(np.abs(weights) < threshold, 0.0, weights)
        return {name: w for name, w in zip(feature_names, sparse_weights)}
```

### **Proximal Gradient Step (Soft Thresholding)**
```python
def proximal_l1_step(model, alpha, lr):
    """Apply soft-thresholding to enforce sparsity after gradient step."""
    with torch.no_grad():
        weight = model.linear.weight.data
        threshold = alpha * lr
        model.linear.weight.data = torch.sign(weight) * torch.clamp(
            torch.abs(weight) - threshold, min=0.0
        )
```

---

## 🏋️ **Training Loop with Proximal Gradient**

```python
def train_lasso(model, train_loader, val_loader, alpha=0.1, lr=0.01, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Proximal step: enforce L1 sparsity
            proximal_l1_step(model, alpha, lr)

            epoch_loss += loss.item()

        if (epoch + 1) % 200 == 0:
            # Count non-zero weights
            n_nonzero = torch.sum(torch.abs(model.linear.weight.data) > 1e-4).item()
            print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.2f}, "
                  f"Active features={n_nonzero}/5")

    return model
```

### **Alternative: L1 Penalty in Loss (Differentiable Approximation)**
```python
def train_lasso_smooth(model, train_loader, alpha=0.1, lr=0.01, epochs=1000):
    """Use smooth L1 approximation instead of proximal step."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            l1_loss = alpha * model.l1_penalty()
            total_loss = mse_loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

---

## 🖥️ **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LassoRegressionModel(n_features=5).to(device)

for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    pred = model(X_batch)
    loss = nn.MSELoss()(pred, y_batch) + alpha * model.l1_penalty()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    proximal_l1_step(model, alpha, lr)
```

---

## 📈 **Evaluation and Feature Selection**

```python
model.eval()
feature_names = ['bone_density', 'gum_health', 'smoking_status',
                 'diabetes_indicator', 'implant_material']

coefs = model.get_sparse_coefficients(feature_names)
print("Selected features:")
for name, w in coefs.items():
    status = "ACTIVE" if w != 0 else "ZEROED"
    print(f"  {name:25s}: {w:8.4f}  [{status}]")

# bone_density             :  18.4800  [ACTIVE]
# gum_health               :   4.1900  [ACTIVE]
# smoking_status           : -12.6500  [ACTIVE]
# diabetes_indicator       :   0.0000  [ZEROED]
# implant_material         :   0.0000  [ZEROED]

with torch.no_grad():
    preds = model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()
rmse = np.sqrt(np.mean((y_test - preds) ** 2))
print(f"\nTest RMSE: {rmse:.2f} points")
```

---

## 💾 **Model Saving**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': alpha,
    'feature_names': feature_names,
    'active_features': [n for n, w in coefs.items() if w != 0],
}, 'implant_lasso_model.pt')
```

---

## 🎯 **When to Use PyTorch Lasso**

- For **custom sparsity-inducing penalties** beyond standard L1
- When building **sparse neural network layers** that start with Lasso
- For **group Lasso** variants requiring custom implementations
- When integrating with an existing **PyTorch research pipeline**
- For **large implant databases** that benefit from GPU batch processing

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/lasso_regression_pytorch.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
