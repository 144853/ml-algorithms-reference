# MLP Regressor (PyTorch) - Insurance Customer Lifetime Value

## **Use Case: Predicting Insurance Customer Lifetime Value (CLV)**

### **The Problem**
An insurance company uses PyTorch to build a production-grade MLP for predicting Customer Lifetime Value. The model leverages GPU acceleration for training on large customer databases and supports custom loss functions for insurance-specific objectives. Features: policy count, tenure years, claim frequency, premium tier, cross-sell score. Target: CLV ($500 - $50,000).

---

## **PyTorch Implementation**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Custom Dataset ---
class CLVDataset(Dataset):
    """Insurance customer lifetime value dataset."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- MLP Model ---
class CLVPredictor(nn.Module):
    """Multi-layer perceptron for customer lifetime value prediction."""

    def __init__(self, n_features, hidden_sizes=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev_size = n_features

        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.ReLU(),
                nn.BatchNorm1d(h_size),
                nn.Dropout(dropout),
            ])
            prev_size = h_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

---

## **Training Pipeline with Best Practices**

```python
def train_clv_model(X_train, y_train, X_val, y_val,
                    hidden_sizes=(64, 32), dropout=0.2,
                    n_epochs=300, lr=0.001, batch_size=64,
                    weight_decay=1e-4, patience=20):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std

    # DataLoaders
    train_loader = DataLoader(CLVDataset(X_train_n, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CLVDataset(X_val_n, y_val),
                            batch_size=batch_size)

    # Model
    model = CLVPredictor(X_train.shape[1], hidden_sizes, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item() * len(X_batch)

        avg_train = train_loss / len(X_train_n)
        avg_val = val_loss / len(X_val_n)
        scheduler.step(avg_val)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{n_epochs} | Train: {avg_train:.0f} | Val: {avg_val:.0f} | LR: {curr_lr:.6f}")

    # Restore best model
    model.load_state_dict(best_state)
    return model, X_mean, X_std
```

---

## **Custom Insurance Loss Functions**

```python
class AsymmetricCLVLoss(nn.Module):
    """
    Penalize under-prediction more than over-prediction.
    Under-predicting CLV means missing retention opportunities.
    """
    def __init__(self, under_weight=2.0, over_weight=1.0):
        super().__init__()
        self.under_weight = under_weight
        self.over_weight = over_weight

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        weights = torch.where(errors > 0,
                              torch.tensor(self.under_weight),  # Under-prediction
                              torch.tensor(self.over_weight))   # Over-prediction
        return torch.mean(weights * errors ** 2)


class HuberCLVLoss(nn.Module):
    """
    Huber loss for CLV - robust to extreme value customers.
    Less sensitive to outlier CLV (e.g., $50K+ whale customers).
    """
    def __init__(self, delta=5000.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        errors = torch.abs(y_true - y_pred)
        quadratic = torch.clamp(errors, max=self.delta)
        linear = errors - quadratic
        return torch.mean(0.5 * quadratic ** 2 + self.delta * linear)
```

---

## **Demo: Full Training Run**

```python
np.random.seed(42)
n = 5000

policy_count = np.random.randint(1, 9, n).astype(float)
tenure = np.random.uniform(0.5, 30, n)
claim_freq = np.random.exponential(0.5, n).clip(0, 5)
premium_tier = np.random.randint(1, 5, n).astype(float)
cross_sell = np.random.uniform(0, 10, n)

clv = (
    500 + 1200 * policy_count + 800 * np.log1p(tenure)
    + 1500 * premium_tier ** 1.3 - 2000 * np.sqrt(claim_freq)
    + 300 * cross_sell * np.log1p(tenure) + np.random.normal(0, 1500, n)
).clip(500, 50000)

X = np.column_stack([policy_count, tenure, claim_freq, premium_tier, cross_sell])

# Train/Val/Test split
n_train, n_val = int(0.7 * n), int(0.15 * n)
X_train, y_train = X[:n_train], clv[:n_train]
X_val, y_val = X[n_train:n_train+n_val], clv[n_train:n_train+n_val]
X_test, y_test = X[n_train+n_val:], clv[n_train+n_val:]

# Train
model, X_mean, X_std = train_clv_model(
    X_train, y_train, X_val, y_val,
    hidden_sizes=(64, 32), dropout=0.2,
    n_epochs=300, lr=0.001, batch_size=64
)

# Evaluate on test set
model.eval()
X_test_n = (X_test - X_mean) / X_std
with torch.no_grad():
    preds = model(torch.FloatTensor(X_test_n)).squeeze().numpy()

mae = np.mean(np.abs(y_test - preds))
r2 = 1 - np.sum((y_test - preds)**2) / np.sum((y_test - y_test.mean())**2)
print(f"\nTest MAE: ${mae:,.2f}")
print(f"Test R2:  {r2:.4f}")
```

---

## **Batch Scoring for Production**

```python
def score_customers(model, X_new, X_mean, X_std, batch_size=1024):
    """Score customers in batches (GPU-friendly)."""
    model.eval()
    device = next(model.parameters()).device
    X_norm = (X_new - X_mean) / X_std

    predictions = []
    dataset = torch.FloatTensor(X_norm)

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size].to(device)
            pred = model(batch).cpu().numpy()
            predictions.append(pred)

    return np.concatenate(predictions).flatten()

# Score entire customer base
all_clv = score_customers(model, X, X_mean, X_std)
print(f"Scored {len(all_clv)} customers")
print(f"CLV range: ${all_clv.min():,.0f} - ${all_clv.max():,.0f}")
print(f"Mean CLV:  ${all_clv.mean():,.0f}")
```

---

## **Model Saving & Loading**

```python
# Save complete model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'hidden_sizes': (64, 32),
    'dropout': 0.2,
    'n_features': 5,
    'feature_names': ['policy_count', 'tenure_years', 'claim_frequency',
                       'premium_tier', 'cross_sell_score'],
    'normalization': {'mean': X_mean, 'std': X_std},
    'metrics': {'test_mae': mae, 'test_r2': r2}
}, 'insurance_clv_mlp_pytorch_v1.pt')

# Load
checkpoint = torch.load('insurance_clv_mlp_pytorch_v1.pt')
model = CLVPredictor(
    n_features=checkpoint['n_features'],
    hidden_sizes=checkpoint['hidden_sizes'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export to TorchScript for production
scripted = torch.jit.script(model)
scripted.save('insurance_clv_mlp_torchscript.pt')
```

---

## **When to Use PyTorch MLP for CLV**

| Scenario | Use PyTorch? |
|----------|-------------|
| Quick CLV prototype | No - use sklearn MLPRegressor |
| Custom loss functions (asymmetric, Huber) | Yes |
| GPU training on large customer bases | Yes |
| Production deployment (TorchScript/ONNX) | Yes |
| Need BatchNorm, Dropout, LR scheduling | Yes |
| Integration with deep learning pipelines | Yes |
| < 10K customers | No - sklearn is sufficient |

---

## **Hyperparameters**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `hidden_sizes` | (64, 32) | (16,) to (128, 64, 32) | Balance capacity vs. overfitting |
| `dropout` | 0.2 | 0.0 - 0.5 | Higher for more regularization |
| `lr` | 0.001 | 0.0001 - 0.01 | Reduced by scheduler on plateau |
| `batch_size` | 64 | 32 - 256 | Larger for GPU utilization |
| `weight_decay` | 1e-4 | 1e-5 - 1e-2 | L2 regularization |
| `patience` | 20 | 10 - 50 | Early stopping patience |
| `n_epochs` | 300 | 100 - 1000 | Max epochs (early stopping decides) |

---

## **References**

1. PyTorch documentation - nn.Sequential, nn.BatchNorm1d
2. Paszke et al., "PyTorch: An Imperative Style Deep Learning Library" (2019)
3. Fader & Hardie, "CLV in Contractual Settings" (2010)
4. Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
5. Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
