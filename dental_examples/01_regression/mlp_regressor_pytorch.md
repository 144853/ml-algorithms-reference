# MLP Regressor (PyTorch) - Dental Crown Longevity Prediction

## 🦷 **Use Case: Predicting Dental Crown Longevity**

GPU-accelerated Multi-Layer Perceptron for predicting crown longevity (2-25 years) from material type, bite force, oral pH, grinding habit, and crown position. PyTorch provides full control over architecture, training, and custom loss functions.

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

model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    for X_batch, y_batch in loader:
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🏗️ **Custom Dataset**

```python
from torch.utils.data import Dataset

class CrownDataset(Dataset):
    """Dataset for dental crown longevity prediction."""

    def __init__(self, features, longevity, transform=None):
        """
        Args:
            features: numpy array (n_patients, 5)
                      [material_type, bite_force, oral_ph, grinding_habit, crown_position]
            longevity: numpy array of crown lifespan in years
        """
        self.features = torch.FloatTensor(features)
        self.longevity = torch.FloatTensor(longevity).unsqueeze(1)
        self.transform = transform

    def __len__(self):
        return len(self.longevity)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.longevity[idx]
```

---

## 🔧 **Model Definition**

```python
class CrownLongevityMLP(nn.Module):
    """MLP for predicting dental crown longevity."""

    def __init__(self, n_features=5, hidden_sizes=(32, 16), dropout=0.1):
        super().__init__()

        layers = []
        in_size = n_features
        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_size = h_size
        layers.append(nn.Linear(in_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = CrownLongevityMLP(n_features=5, hidden_sizes=(32, 16), dropout=0.1)
print(model)
```

---

## 🏋️ **Training Loop with Early Stopping**

```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def train_crown_model(model, train_loader, val_loader,
                      epochs=500, lr=0.001, patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    early_stopping = EarlyStopping(patience=patience)

    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < early_stopping.best_loss:
            best_model_state = model.state_dict().copy()

        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train RMSE={train_losses[-1]**0.5:.2f} yrs, "
                  f"Val RMSE={val_loss**0.5:.2f} yrs")

    model.load_state_dict(best_model_state)
    return train_losses, val_losses
```

---

## 🖥️ **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = CrownLongevityMLP(n_features=5, hidden_sizes=(64, 32)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    predictions = model(X_batch)
    loss = nn.MSELoss()(predictions, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 📊 **Custom Loss Functions for Dental Domain**

```python
class AsymmetricMSELoss(nn.Module):
    """Penalize over-predicting longevity more than under-predicting.
    Under-predicting is safer for patient expectations."""

    def __init__(self, over_predict_weight=2.0):
        super().__init__()
        self.over_weight = over_predict_weight

    def forward(self, predictions, targets):
        errors = predictions - targets
        weights = torch.where(errors > 0, self.over_weight, 1.0)
        return torch.mean(weights * errors ** 2)

# Use asymmetric loss: penalize over-estimating crown life
criterion = AsymmetricMSELoss(over_predict_weight=2.0)
```

---

## 📈 **Evaluation**

```python
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()
            all_preds.append(preds)
            all_targets.append(y_batch)

    preds = torch.cat(all_preds).squeeze().numpy()
    targets = torch.cat(all_targets).squeeze().numpy()

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    r2 = 1 - np.sum((targets - preds)**2) / np.sum((targets - targets.mean())**2)

    print(f"RMSE:  {rmse:.2f} years")
    print(f"MAE:   {mae:.2f} years")
    print(f"R^2:   {r2:.4f}")

    # Clinical interpretation
    within_1yr = np.mean(np.abs(preds - targets) < 1.0) * 100
    within_2yr = np.mean(np.abs(preds - targets) < 2.0) * 100
    print(f"Within 1 year: {within_1yr:.1f}%")
    print(f"Within 2 years: {within_2yr:.1f}%")
```

---

## 💾 **Model Saving and Loading**

```python
# Save complete model info
torch.save({
    'model_state_dict': model.state_dict(),
    'architecture': {'n_features': 5, 'hidden_sizes': (32, 16), 'dropout': 0.1},
    'feature_names': ['material_type', 'bite_force', 'oral_ph',
                      'grinding_habit', 'crown_position'],
    'scaler_params': {'mean': scaler.mean_, 'std': scaler.scale_},
    'training_losses': train_losses,
    'val_losses': val_losses,
}, 'crown_longevity_mlp.pt')

# Load for inference
checkpoint = torch.load('crown_longevity_mlp.pt')
model = CrownLongevityMLP(**checkpoint['architecture'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict for a new crown
new_crown = torch.FloatTensor([[4, 150, 7.2, 0, 2]])  # Zirconia, low force, neutral pH
with torch.no_grad():
    years = model(new_crown)
    print(f"Predicted longevity: {years.item():.1f} years")
```

---

## 🎯 **When to Use PyTorch MLP**

- When you need **custom architectures** (residual connections, attention, etc.)
- For **custom loss functions** (asymmetric penalties for dental safety)
- When **GPU acceleration** matters for large patient datasets
- For **transfer learning** -- pretrain on large dental databases, fine-tune on clinic data
- When building **end-to-end systems** that combine imaging and tabular dental data

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/mlp_regressor_pytorch.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
