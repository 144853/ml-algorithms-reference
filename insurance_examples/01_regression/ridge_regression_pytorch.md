# Ridge Regression (PyTorch) - Health Insurance Claim Prediction

## **Use Case: Predicting Health Insurance Claim Amounts**

### **The Problem**
A health insurer uses PyTorch to build a GPU-accelerated Ridge regression model for predicting annual claim amounts. Features: age, BMI, chronic conditions, prescription count, hospitalization days. Target: annual claim ($100 - $100,000).

---

## **PyTorch Implementation**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Custom Dataset ---
class HealthClaimDataset(Dataset):
    """Health insurance member claims dataset."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Ridge Regression Model ---
class HealthClaimRidge(nn.Module):
    """Ridge regression for health claim prediction with L2 weight decay."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def l2_penalty(self):
        """Compute L2 penalty (weight decay) on model weights."""
        return torch.sum(self.linear.weight ** 2)
```

---

## **Training with Explicit L2 Penalty**

```python
def train_ridge_model(X_train, y_train, X_test, y_test,
                      alpha=1.0, n_epochs=300, lr=0.01, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    train_loader = DataLoader(HealthClaimDataset(X_train_n, y_train),
                              batch_size=batch_size, shuffle=True)
    test_ds = HealthClaimDataset(X_test_n, y_test)

    model = HealthClaimRidge(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()

    # Method 1: Explicit L2 in loss
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Method 2 (equivalent): Use weight_decay parameter
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=alpha)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            mse_loss = criterion(y_pred, y_batch)
            l2_loss = alpha * model.l2_penalty()
            total_loss = mse_loss + l2_loss

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
        y_t = torch.FloatTensor(y_test)
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

## **Demo: Comparing Alpha Values**

```python
np.random.seed(42)
n = 2000

age = np.random.randint(18, 80, n).astype(float)
bmi = np.random.normal(27, 5, n).clip(15, 50)
chronic = np.random.poisson(1.5, n).clip(0, 8).astype(float)
rx = (chronic * 2.0 + np.random.normal(0, 1.5, n)).clip(0, 15)
hosp = (chronic * 3.0 + np.random.exponential(2, n)).clip(0, 60)

claim = (
    -12000 + 280*age + 310*bmi + 4100*chronic + 1100*rx + 2800*hosp
    + np.random.normal(0, 3000, n)
).clip(100, 100000)

X = np.column_stack([age, bmi, chronic, rx, hosp])
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = claim[:split], claim[split:]

# Compare regularization strengths
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    print(f"\n=== Alpha = {alpha} ===")
    model = train_ridge_model(X_train, y_train, X_test, y_test,
                              alpha=alpha, n_epochs=300, lr=0.01)
```

---

## **Weight Decay vs. Explicit L2**

```python
# PyTorch weight_decay in optimizer is equivalent to L2 regularization:
#   optimizer = SGD(params, lr=0.01, weight_decay=alpha)
# This adds: loss += (weight_decay / 2) * sum(param^2)
#
# For exact Ridge equivalence, set weight_decay = 2 * alpha
# because Ridge uses: loss += alpha * sum(w^2)
# and PyTorch uses:   loss += (weight_decay / 2) * sum(w^2)

optimizer_wd = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=2.0)
```

---

## **Model Saving & Loading**

```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': alpha,
    'normalization': {'mean': X_mean, 'std': X_std},
    'feature_names': ['age', 'bmi', 'chronic_conditions', 'prescription_count', 'hospitalization_days']
}, 'health_claim_ridge_pytorch_v1.pt')

# Load
checkpoint = torch.load('health_claim_ridge_pytorch_v1.pt')
model = HealthClaimRidge(n_features=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## **When to Use PyTorch for Ridge**

| Scenario | Use PyTorch? |
|----------|-------------|
| Small dataset (< 10K members) | No - sklearn RidgeCV is faster |
| Large population (> 1M members) | Yes - GPU batch processing |
| Custom loss (e.g., asymmetric claim loss) | Yes - flexible loss definition |
| Part of deep learning pipeline | Yes - seamless integration |
| Comparing L2 with other penalties | Yes - easy to swap penalty terms |

---

## **Hyperparameters**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `alpha` | 1.0 | 0.001 - 100 | Regularization strength |
| `lr` | 0.01 | 0.001 - 0.1 | Learning rate |
| `batch_size` | 64 | 32 - 512 | Larger for GPU efficiency |
| `n_epochs` | 300 | 100 - 1000 | Monitor convergence |
| `weight_decay` | 2*alpha | - | Equivalent to explicit L2 penalty |

---

## **References**

1. PyTorch documentation - Weight Decay in Optimizers
2. Krogh & Hertz, "A Simple Weight Decay Can Improve Generalization" (1991)
3. Duncan, "Healthcare Risk Adjustment and Predictive Modeling" (2011)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
