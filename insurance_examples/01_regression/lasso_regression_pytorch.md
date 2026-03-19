# Lasso Regression (PyTorch) - Property Insurance Loss Severity

## **Use Case: Predicting Property Insurance Loss Severity**

### **The Problem**
A property insurer uses PyTorch to build a Lasso regression model for predicting claim severity with automatic feature selection. Features: building age, square footage, fire protection class, weather exposure score, occupancy type. Target: loss severity ($500 - $80,000).

---

## **PyTorch Implementation**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Custom Dataset ---
class PropertyClaimDataset(Dataset):
    """Property insurance claims dataset."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Lasso Regression Model ---
class PropertyLossLasso(nn.Module):
    """Lasso regression with L1 penalty for property loss prediction."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def l1_penalty(self):
        """L1 norm of weights (not bias)."""
        return torch.sum(torch.abs(self.linear.weight))

    def get_selected_features(self, feature_names, threshold=1e-4):
        """Identify selected vs eliminated features."""
        weights = self.linear.weight.data.squeeze().cpu().numpy()
        selected = [(n, w) for n, w in zip(feature_names, weights) if abs(w) > threshold]
        eliminated = [n for n, w in zip(feature_names, weights) if abs(w) <= threshold]
        return selected, eliminated
```

---

## **Training with L1 Regularization**

```python
def train_lasso_model(X_train, y_train, X_test, y_test,
                      alpha=1.0, n_epochs=500, lr=0.01, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    train_loader = DataLoader(PropertyClaimDataset(X_train_n, y_train),
                              batch_size=batch_size, shuffle=True)

    model = PropertyLossLasso(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            mse_loss = criterion(y_pred, y_batch)
            l1_loss = alpha * model.l1_penalty()
            total_loss = mse_loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * len(X_batch)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/len(X_train_n):.2f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test_n).to(device)
        preds = model(X_t).cpu().squeeze().numpy()

    mae = np.mean(np.abs(y_test - preds))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nTest MAE: ${mae:,.2f}")
    print(f"Test R2:  {r2:.4f}")
    return model
```

---

## **Proximal Gradient Descent (True Sparsity)**

```python
def proximal_lasso_train(X_train, y_train, X_test, y_test,
                          alpha=1.0, n_epochs=500, lr=0.01):
    """
    Proximal gradient descent achieves exact zeros (true sparsity),
    unlike standard gradient descent which only approaches zero.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    model = PropertyLossLasso(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X_train_n).to(device)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    for epoch in range(n_epochs):
        # Standard gradient step (MSE only)
        y_pred = model(X_tensor)
        mse_loss = criterion(y_pred, y_tensor)

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        # Proximal step: soft thresholding on weights
        with torch.no_grad():
            w = model.linear.weight.data
            threshold = alpha * lr
            model.linear.weight.data = torch.sign(w) * torch.clamp(torch.abs(w) - threshold, min=0)

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test_n).to(device)).cpu().squeeze().numpy()
    mae = np.mean(np.abs(y_test - preds))
    print(f"Proximal Lasso MAE: ${mae:,.2f}")
    return model
```

---

## **Demo: Feature Selection**

```python
np.random.seed(42)
n = 2000

building_age = np.random.uniform(0, 150, n)
sq_footage = np.random.uniform(500, 50000, n)
fire_class = np.random.randint(1, 11, n).astype(float)
weather = np.random.uniform(1, 10, n)
occupancy = np.random.randint(1, 4, n).astype(float)
noise1 = np.random.randn(n)  # Irrelevant
noise2 = np.random.randn(n)  # Irrelevant

loss = (-5000 + 180*building_age + 0.02*sq_footage + 2400*fire_class
        + 1800*weather + 3400*occupancy + np.random.normal(0, 3500, n)).clip(500, 80000)

X = np.column_stack([building_age, sq_footage, fire_class, weather, occupancy, noise1, noise2])
feature_names = ['building_age', 'square_footage', 'fire_protection_class',
                 'weather_exposure', 'occupancy_type', 'noise_1', 'noise_2']

split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = loss[:split], loss[split:]

# Standard L1 training
model = train_lasso_model(X_train, y_train, X_test, y_test, alpha=0.5, n_epochs=500)

selected, eliminated = model.get_selected_features(feature_names)
print("\n--- Selected Features ---")
for name, w in selected:
    print(f"  {name:30s}: {w:.4f}")
print("--- Eliminated Features ---")
for name in eliminated:
    print(f"  {name:30s}: 0.0000")
```

---

## **Model Saving & Loading**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': alpha,
    'feature_names': feature_names,
    'normalization': {'mean': X_mean, 'std': X_std}
}, 'property_loss_lasso_pytorch_v1.pt')

# Load
checkpoint = torch.load('property_loss_lasso_pytorch_v1.pt')
model = PropertyLossLasso(n_features=7)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## **When to Use PyTorch for Lasso**

| Scenario | Use PyTorch? |
|----------|-------------|
| Simple feature selection | No - sklearn LassoCV is better |
| Custom L1 + other penalties | Yes - compose penalty terms freely |
| Proximal gradient methods | Yes - implement soft thresholding |
| GPU batch scoring millions of properties | Yes - significant speedup |
| Part of neural network pipeline | Yes - add L1 to any layer |

---

## **Hyperparameters**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `alpha` | 1.0 | 0.001 - 100 | L1 penalty strength |
| `lr` | 0.01 | 0.001 - 0.1 | Learning rate |
| `batch_size` | 64 | 32 - 256 | Batch size |
| `n_epochs` | 500 | 200 - 2000 | Training iterations |
| `threshold` | 1e-4 | 1e-6 - 1e-2 | Sparsity detection threshold |

---

## **References**

1. PyTorch documentation - Custom Loss Functions
2. Tibshirani, "Regression Shrinkage and Selection via the Lasso" (1996)
3. Parikh & Boyd, "Proximal Algorithms" (2014)
4. Klugman et al., "Loss Models" (2012)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
