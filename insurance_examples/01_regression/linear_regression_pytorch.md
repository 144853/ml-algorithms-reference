# Linear Regression (PyTorch) - Insurance Premium Prediction

## **Use Case: Predicting Auto Insurance Premiums**

### **The Problem**
An auto insurance company uses PyTorch to build a GPU-accelerated premium prediction model. Features: driver age, vehicle value, accidents in past 3 years, coverage level, and zip code risk score. Target: annual premium ($500 - $5,000).

---

## **PyTorch Implementation**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Custom Dataset ---
class InsuranceDataset(Dataset):
    """Auto insurance policyholder dataset."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Linear Regression Model ---
class AutoPremiumModel(nn.Module):
    """Linear regression for auto insurance premium prediction."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def get_rating_factors(self, feature_names):
        """Extract interpretable rating factors."""
        weights = self.linear.weight.data.squeeze().cpu().numpy()
        bias = self.linear.bias.data.item()
        factors = {name: w for name, w in zip(feature_names, weights)}
        factors['base_premium'] = bias
        return factors
```

---

## **Training Pipeline**

```python
def train_premium_model(X_train, y_train, X_test, y_test,
                        n_epochs=200, lr=0.01, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # DataLoaders
    train_ds = InsuranceDataset(X_train_norm, y_train)
    test_ds = InsuranceDataset(X_test_norm, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model, loss, optimizer
    model = AutoPremiumModel(n_features=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / len(train_ds)
            print(f"Epoch {epoch+1}/{n_epochs} | MSE: {avg_loss:.2f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_actuals = []
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            test_preds.extend(preds.flatten())
            test_actuals.extend(y_batch.numpy().flatten())

    test_preds = np.array(test_preds)
    test_actuals = np.array(test_actuals)
    mae = np.mean(np.abs(test_actuals - test_preds))
    ss_res = np.sum((test_actuals - test_preds) ** 2)
    ss_tot = np.sum((test_actuals - test_actuals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nTest MAE:  ${mae:.2f}")
    print(f"Test R^2:  {r2:.4f}")

    return model
```

---

## **Demo: Full Training Run**

```python
# Generate synthetic insurance data
np.random.seed(42)
n = 2000

driver_age = np.random.randint(16, 85, n).astype(float)
vehicle_value = np.random.uniform(2000, 75000, n)
accidents = np.random.poisson(0.4, n).clip(0, 5).astype(float)
coverage = np.random.randint(1, 4, n).astype(float)
zip_risk = np.random.uniform(1.0, 10.0, n)

premium = (
    600 - 6.0 * driver_age + 0.02 * vehicle_value
    + 350 * accidents + 180 * coverage + 95 * zip_risk
    + np.random.normal(0, 150, n)
)
premium = np.clip(premium, 500, 5000)

X = np.column_stack([driver_age, vehicle_value, accidents, coverage, zip_risk])
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = premium[:split], premium[split:]

feature_names = ['driver_age', 'vehicle_value', 'accidents_3yr', 'coverage_level', 'zip_risk_score']

model = train_premium_model(X_train, y_train, X_test, y_test, n_epochs=200, lr=0.01)

# Display rating factors
factors = model.get_rating_factors(feature_names)
print("\n--- Learned Rating Factors ---")
for name, value in factors.items():
    print(f"  {name:20s}: {value:>10.4f}")
```

---

## **GPU Acceleration Notes**

```python
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# For large policy books (millions of records), GPU batch scoring:
model.eval()
with torch.no_grad():
    X_gpu = torch.FloatTensor(X_large).to('cuda')
    premiums = model(X_gpu).cpu().numpy()
```

---

## **Model Saving & Loading**

```python
# Save for production deployment
torch.save({
    'model_state_dict': model.state_dict(),
    'feature_names': feature_names,
    'normalization': {'mean': X_mean, 'std': X_std}
}, 'auto_premium_pytorch_v1.pt')

# Load in production
checkpoint = torch.load('auto_premium_pytorch_v1.pt')
model = AutoPremiumModel(n_features=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## **When to Use PyTorch for Linear Regression**

| Scenario | Use PyTorch? |
|----------|-------------|
| Simple premium model | No - sklearn is simpler |
| GPU-accelerated batch scoring | Yes - significant speedup for millions of policies |
| Part of a larger deep learning pipeline | Yes - seamless integration |
| Custom loss functions (e.g., asymmetric loss for underpricing risk) | Yes - easy to implement |
| Streaming / online learning | Yes - natural mini-batch support |

---

## **Hyperparameters**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `lr` | 0.01 | 0.001 - 0.1 | Lower for stable convergence |
| `batch_size` | 64 | 32 - 512 | Larger for GPU utilization |
| `n_epochs` | 200 | 50 - 1000 | Monitor validation loss for early stopping |
| `optimizer` | SGD | SGD/Adam | Adam converges faster but SGD generalizes better |

---

## **References**

1. PyTorch documentation - nn.Linear
2. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019)
3. Werner & Modlin, "Basic Ratemaking" (CAS, 2016)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
