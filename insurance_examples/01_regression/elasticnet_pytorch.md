# ElasticNet (PyTorch) - Life Insurance Risk Score Prediction

## **Use Case: Predicting Life Insurance Risk Score**

### **The Problem**
A life insurer uses PyTorch to implement ElasticNet regression for predicting applicant risk scores. The model combines L1 (feature selection) and L2 (stability) penalties. Features: age, health exam results, family history score, occupation risk, lifestyle habits. Target: risk score (1-100).

---

## **PyTorch Implementation**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Custom Dataset ---
class LifeInsuranceDataset(Dataset):
    """Life insurance applicant dataset."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- ElasticNet Model ---
class LifeRiskElasticNet(nn.Module):
    """ElasticNet regression combining L1 and L2 penalties."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def elastic_penalty(self, alpha, l1_ratio):
        """Combined L1 + L2 penalty."""
        w = self.linear.weight
        l1 = torch.sum(torch.abs(w))
        l2 = torch.sum(w ** 2)
        return alpha * (l1_ratio * l1 + (1 - l1_ratio) / 2 * l2)

    def get_coefficients(self, feature_names, threshold=1e-4):
        """Extract and classify coefficients."""
        weights = self.linear.weight.data.squeeze().cpu().numpy()
        bias = self.linear.bias.data.item()
        result = {}
        for name, w in zip(feature_names, weights):
            result[name] = {'weight': w, 'selected': abs(w) > threshold}
        result['intercept'] = {'weight': bias, 'selected': True}
        return result
```

---

## **Training Pipeline**

```python
def train_elasticnet(X_train, y_train, X_test, y_test,
                     alpha=1.0, l1_ratio=0.5, n_epochs=500, lr=0.01, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    train_loader = DataLoader(LifeInsuranceDataset(X_train_n, y_train),
                              batch_size=batch_size, shuffle=True)

    model = LifeRiskElasticNet(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            mse_loss = criterion(y_pred, y_batch)
            reg_loss = model.elastic_penalty(alpha, l1_ratio)
            total_loss = mse_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * len(X_batch)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/len(X_train_n):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test_n).to(device)
        preds = model(X_t).cpu().squeeze().numpy()

    mae = np.mean(np.abs(y_test - preds))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nTest MAE:  {mae:.2f} risk points")
    print(f"Test R2:   {r2:.4f}")
    return model, X_mean, X_std
```

---

## **Demo: Full Training with L1/L2 Mix Comparison**

```python
np.random.seed(42)
n = 2000

age = np.random.randint(20, 75, n).astype(float)
health_exam = np.random.uniform(40, 100, n)
family_hist = np.random.uniform(0, 10, n)
occ_risk = np.random.uniform(1, 10, n)
lifestyle = np.random.uniform(1, 10, n)
bp_index = (100 - health_exam) / 60 + np.random.normal(0, 0.2, n)  # Correlated
noise1 = np.random.randn(n)  # Irrelevant
noise2 = np.random.randn(n)  # Irrelevant

risk_score = (
    0.5 * age - 0.6 * health_exam + 3.5 * family_hist
    + 2.8 * occ_risk + 2.2 * lifestyle + 5.0 * bp_index
    + np.random.normal(0, 5, n)
).clip(1, 100)

X = np.column_stack([age, health_exam, family_hist, occ_risk, lifestyle, bp_index, noise1, noise2])
feature_names = ['age', 'health_exam', 'family_history', 'occupation_risk',
                 'lifestyle', 'blood_pressure', 'noise_1', 'noise_2']

split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = risk_score[:split], risk_score[split:]

# Compare l1_ratio values
print("=== L1/L2 Mix Comparison ===\n")
for l1_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
    label = {0.0: 'Pure Ridge', 0.25: 'Mostly L2', 0.5: 'Equal Mix',
             0.75: 'Mostly L1', 1.0: 'Pure Lasso'}[l1_ratio]
    print(f"--- l1_ratio={l1_ratio} ({label}) ---")
    model, _, _ = train_elasticnet(X_train, y_train, X_test, y_test,
                                    alpha=0.1, l1_ratio=l1_ratio, n_epochs=300)
    coefs = model.get_coefficients(feature_names)
    n_selected = sum(1 for v in coefs.values() if v.get('selected', False) and 'intercept' not in str(v))
    print(f"Features selected: {n_selected}/{len(feature_names)}\n")
```

---

## **Proximal Gradient for True Sparsity**

```python
def proximal_elasticnet(model, alpha, l1_ratio, lr):
    """Apply proximal operator after gradient step for exact sparsity."""
    with torch.no_grad():
        w = model.linear.weight.data
        # L1 proximal: soft thresholding
        threshold = alpha * l1_ratio * lr
        shrunk = torch.sign(w) * torch.clamp(torch.abs(w) - threshold, min=0)
        # L2 shrinkage is already handled by weight decay or explicit gradient
        model.linear.weight.data = shrunk
```

---

## **Model Saving & Loading**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': 0.1,
    'l1_ratio': 0.5,
    'feature_names': feature_names,
    'normalization': {'mean': X_mean, 'std': X_std}
}, 'life_risk_elasticnet_pytorch_v1.pt')

# Load
checkpoint = torch.load('life_risk_elasticnet_pytorch_v1.pt')
model = LifeRiskElasticNet(n_features=8)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Score new applicant
new_app = np.array([[45, 72.0, 5.5, 3.0, 4.0, 1.1, 0, 0]])
new_app_n = (new_app - checkpoint['normalization']['mean']) / checkpoint['normalization']['std']
with torch.no_grad():
    risk = model(torch.FloatTensor(new_app_n)).item()
print(f"Risk Score: {risk:.1f}/100")
```

---

## **When to Use PyTorch for ElasticNet**

| Scenario | Use PyTorch? |
|----------|-------------|
| Simple ElasticNet with CV | No - sklearn ElasticNetCV is simpler |
| Custom penalty schedules | Yes - vary alpha/l1_ratio during training |
| Combined with neural network layers | Yes - add elastic penalty to any layer |
| GPU scoring for large applicant pools | Yes |
| Custom loss (e.g., asymmetric risk scoring) | Yes |
| Proximal gradient methods | Yes - implement exact sparsity |

---

## **Hyperparameters**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `alpha` | 1.0 | 0.001 - 10 | Overall penalty strength |
| `l1_ratio` | 0.5 | 0.0 - 1.0 | 0=Ridge, 1=Lasso |
| `lr` | 0.01 | 0.001 - 0.1 | Learning rate |
| `batch_size` | 64 | 32 - 256 | Mini-batch size |
| `n_epochs` | 500 | 200 - 2000 | Training epochs |

---

## **References**

1. Zou & Hastie, "Regularization and Variable Selection via the Elastic Net" (2005)
2. PyTorch documentation - Custom Regularization
3. Society of Actuaries, "Predictive Analytics and Life Insurance" (2018)
4. Parikh & Boyd, "Proximal Algorithms" (2014)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
