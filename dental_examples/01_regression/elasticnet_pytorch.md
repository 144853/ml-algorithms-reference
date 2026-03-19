# ElasticNet (PyTorch) - Patient Lifetime Dental Spend Prediction

## 🦷 **Use Case: Predicting Patient Lifetime Dental Spend**

PyTorch implementation of ElasticNet regression combining L1 and L2 penalties for predicting lifetime dental spend ($500-$50,000) from hygiene habits, diet score, genetics risk, visit frequency, and fluoride exposure.

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

alpha = 0.1     # Overall regularization strength
l1_ratio = 0.7  # 70% L1, 30% L2

for epoch in range(1000):
    for X_batch, y_batch in loader:
        predictions = model(X_batch)
        mse_loss = criterion(predictions, y_batch)

        l1_loss = torch.sum(torch.abs(model.weight))
        l2_loss = torch.sum(model.weight ** 2)
        elastic_penalty = alpha * (l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss / 2)

        total_loss = mse_loss + elastic_penalty
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## 🏗️ **Custom Dataset**

```python
from torch.utils.data import Dataset

class DentalSpendDataset(Dataset):
    """Dataset for patient lifetime dental spend prediction."""

    def __init__(self, features, spend):
        """
        Args:
            features: numpy array (n_patients, 5)
                      [hygiene_habits, diet_score, genetics_risk,
                       visit_frequency, fluoride_exposure]
            spend: numpy array of lifetime spend in dollars
        """
        self.features = torch.FloatTensor(features)
        self.spend = torch.FloatTensor(spend).unsqueeze(1)

    def __len__(self):
        return len(self.spend)

    def __getitem__(self, idx):
        return self.features[idx], self.spend[idx]
```

---

## 🔧 **ElasticNet Model**

```python
class ElasticNetModel(nn.Module):
    """ElasticNet regression with combined L1 + L2 penalties."""

    def __init__(self, n_features=5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def elastic_penalty(self, alpha, l1_ratio):
        """Compute combined L1 + L2 penalty on weights."""
        l1 = torch.sum(torch.abs(self.linear.weight))
        l2 = torch.sum(self.linear.weight ** 2)
        return alpha * (l1_ratio * l1 + (1 - l1_ratio) * l2 / 2)

    def get_coefficients(self, feature_names, threshold=1e-3):
        """Return coefficients with near-zero values snapped to zero."""
        weights = self.linear.weight.data.squeeze().numpy()
        sparse_w = np.where(np.abs(weights) < threshold, 0.0, weights)
        bias = self.linear.bias.data.item()
        coefs = {name: w for name, w in zip(feature_names, sparse_w)}
        coefs['bias'] = bias
        return coefs

model = ElasticNetModel(n_features=5)
```

---

## 🏋️ **Training Loop with Proximal Step**

```python
def proximal_l1_step(model, alpha, l1_ratio, lr):
    """Soft-threshold weights to enforce L1 sparsity component."""
    with torch.no_grad():
        weight = model.linear.weight.data
        threshold = alpha * l1_ratio * lr
        model.linear.weight.data = torch.sign(weight) * torch.clamp(
            torch.abs(weight) - threshold, min=0.0
        )

def train_elasticnet(model, train_loader, val_loader,
                     alpha=0.1, l1_ratio=0.7, lr=0.001, epochs=1000):
    criterion = nn.MSELoss()
    # Use weight_decay for the L2 component
    l2_weight_decay = alpha * (1 - l1_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=l2_weight_decay)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            # L1 component added explicitly
            l1_loss = alpha * l1_ratio * torch.sum(torch.abs(model.linear.weight))
            total_loss = mse_loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Proximal step for L1 sparsity
            proximal_l1_step(model, alpha, l1_ratio, lr)
            epoch_loss += total_loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_elasticnet_dental.pt')

        if (epoch + 1) % 200 == 0:
            n_active = torch.sum(torch.abs(model.linear.weight.data) > 1e-3).item()
            print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):,.0f}, "
                  f"Val={val_loss:,.0f}, Active={n_active}/5")
```

---

## 📊 **Grid Search for Alpha and L1 Ratio**

```python
def grid_search_elasticnet(X_train, y_train, X_val, y_val):
    """Find best alpha and l1_ratio combination."""
    best_params, best_loss = None, float('inf')

    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model = ElasticNetModel(n_features=5)
            train_ds = TensorDataset(torch.FloatTensor(X_train),
                                      torch.FloatTensor(y_train).unsqueeze(1))
            loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                          weight_decay=alpha * (1 - l1_ratio))
            criterion = nn.MSELoss()

            for epoch in range(500):
                for xb, yb in loader:
                    pred = model(xb)
                    loss = criterion(pred, yb) + alpha * l1_ratio * model.linear.weight.abs().sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(torch.FloatTensor(X_val))
                val_loss = criterion(val_pred, torch.FloatTensor(y_val).unsqueeze(1)).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}

    print(f"Best: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
    return best_params
```

---

## 📈 **Evaluation**

```python
model.eval()
feature_names = ['hygiene_habits', 'diet_score', 'genetics_risk',
                 'visit_frequency', 'fluoride_exposure']

coefs = model.get_coefficients(feature_names)
print("ElasticNet Coefficients:")
for name, w in coefs.items():
    if name != 'bias':
        status = "ACTIVE" if w != 0 else "ZEROED"
        print(f"  {name:20s}: ${w:10,.2f}  [{status}]")

with torch.no_grad():
    preds = model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()

rmse = np.sqrt(np.mean((y_test - preds) ** 2))
mae = np.mean(np.abs(y_test - preds))
print(f"\nTest RMSE: ${rmse:,.2f}")
print(f"Test MAE:  ${mae:,.2f}")
```

---

## 💾 **Model Saving**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': alpha,
    'l1_ratio': l1_ratio,
    'feature_names': feature_names,
}, 'dental_spend_elasticnet.pt')
```

---

## 🎯 **When to Use PyTorch ElasticNet**

- For **custom elastic penalties** with non-standard L1/L2 mixes
- When extending to **sparse neural networks** with ElasticNet-style regularization
- For **per-layer regularization** with different alpha/l1_ratio per layer
- When integrating with an existing **PyTorch training pipeline**
- For **large patient databases** (100K+) benefiting from GPU batch processing

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/elasticnet_pytorch.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
