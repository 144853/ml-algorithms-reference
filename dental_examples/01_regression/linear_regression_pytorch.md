# Linear Regression (PyTorch) - Dental Treatment Cost Prediction

## 🦷 **Use Case: Predicting Dental Treatment Costs**

GPU-accelerated linear regression for predicting treatment costs ($200-$5000) from procedure type, tooth count, patient age, insurance tier, and clinic location.

---

## 📦 **Quick Start**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Dental cost data: 500 patients, 5 features
X_tensor = torch.FloatTensor(X_train_scaled)
y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Linear regression as a single-layer neural network
model = nn.Linear(in_features=5, out_features=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    for X_batch, y_batch in loader:
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🏗️ **Custom Dataset for Dental Data**

```python
from torch.utils.data import Dataset

class DentalCostDataset(Dataset):
    """Dataset for dental treatment cost prediction."""

    def __init__(self, features, costs, transform=None):
        """
        Args:
            features: numpy array of shape (n_patients, 5)
                      [procedure_type, tooth_count, age, insurance_tier, location]
            costs: numpy array of treatment costs in dollars
            transform: optional feature transformation
        """
        self.features = torch.FloatTensor(features)
        self.costs = torch.FloatTensor(costs).unsqueeze(1)
        self.transform = transform

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.costs[idx]

# Usage
train_dataset = DentalCostDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## 🔧 **Model Definition**

```python
class DentalCostPredictor(nn.Module):
    """Linear regression model for dental treatment cost prediction."""

    def __init__(self, n_features=5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def get_coefficients(self, feature_names):
        """Return named coefficients for interpretability."""
        weights = self.linear.weight.data.squeeze().numpy()
        bias = self.linear.bias.data.item()
        coefs = {name: w for name, w in zip(feature_names, weights)}
        coefs['bias'] = bias
        return coefs

model = DentalCostPredictor(n_features=5)
```

---

## 🏋️ **Training Loop with Validation**

```python
def train_dental_model(model, train_loader, val_loader, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training
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

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
            val_losses.append(val_loss / len(val_loader))

        scheduler.step(val_losses[-1])

        # Early stopping check
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), 'best_dental_cost_model.pt')

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Train RMSE=${train_losses[-1]**0.5:,.2f}, "
                  f"Val RMSE=${val_losses[-1]**0.5:,.2f}")

    return train_losses, val_losses
```

---

## 🖥️ **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = DentalCostPredictor(n_features=5).to(device)

# Move data to GPU
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    predictions = model(X_batch)
    # ... training step
```

---

## 📈 **Evaluation**

```python
def evaluate_dental_model(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            all_preds.append(preds)
            all_targets.append(y_batch)

    preds = torch.cat(all_preds).squeeze().numpy()
    targets = torch.cat(all_targets).squeeze().numpy()

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    r2 = 1 - np.sum((targets - preds)**2) / np.sum((targets - targets.mean())**2)

    print(f"RMSE:  ${rmse:,.2f}")
    print(f"MAE:   ${mae:,.2f}")
    print(f"R^2:   {r2:.4f}")

    # Dental-specific: interpret coefficients
    feature_names = ['procedure_type', 'tooth_count', 'patient_age',
                     'insurance_tier', 'clinic_location']
    coefs = model.get_coefficients(feature_names)
    print("\nLearned cost factors:")
    for name, value in coefs.items():
        print(f"  {name}: ${value:,.2f}")
```

---

## 💾 **Model Saving and Loading**

```python
# Save full model
torch.save(model.state_dict(), 'dental_cost_linear.pt')

# Load for inference
loaded_model = DentalCostPredictor(n_features=5)
loaded_model.load_state_dict(torch.load('dental_cost_linear.pt'))
loaded_model.eval()

# Predict for a new patient
new_patient = torch.FloatTensor([[3, 4, 45, 2, 1]])  # Root canal, 4 teeth, age 45
with torch.no_grad():
    cost = loaded_model(new_patient)
    print(f"Estimated cost: ${cost.item():,.2f}")
```

---

## 🎯 **When to Use PyTorch for Linear Regression**

- As a **building block** before adding neural network layers
- When you need **GPU acceleration** for large dental datasets (100K+ records)
- For **seamless extension** to deep learning models
- When integrating with an existing **PyTorch pipeline**
- For **research** on custom loss functions (e.g., asymmetric cost penalties)

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/linear_regression_pytorch.py
```

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
