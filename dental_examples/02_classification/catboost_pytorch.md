# CatBoost (PyTorch) - Dental Classification

## **Use Case: Classifying Dental Insurance Claim Approval**

### **The Problem**
A dental insurance company classifies **1200 claims** as approved or denied using a PyTorch model with categorical embeddings, inspired by CatBoost's native categorical handling. Features: procedure code, policy tier, pre-authorization, claim amount, and provider network status.

### **Why PyTorch for CatBoost-style Models?**
- Learnable categorical embeddings (superior to one-hot encoding)
- GPU-accelerated training for high-volume claims processing
- Custom loss functions for asymmetric approval/denial costs
- Integration with NLP models for claim text analysis

---

## **Custom Dental Dataset**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class InsuranceClaimDataset(Dataset):
    """Dataset for dental insurance claim classification."""

    def __init__(self, n_claims=1200, seed=42):
        np.random.seed(seed)

        # Categorical features (stored as indices)
        procedure_codes = np.random.choice(range(8), n_claims)   # 8 procedure types
        policy_tiers = np.random.choice(range(4), n_claims)       # 4 tiers
        provider_network = np.random.choice(range(3), n_claims)   # 3 network types

        # Numerical features
        pre_auth = np.random.choice([0.0, 1.0], n_claims, p=[0.4, 0.6])
        claim_amount = np.random.uniform(50, 5000, n_claims)

        # Normalize numerical features
        self.amount_mean = claim_amount.mean()
        self.amount_std = claim_amount.std()
        claim_amount_norm = (claim_amount - self.amount_mean) / self.amount_std

        # Labels
        approval_score = (0.3 * pre_auth + 0.2 * policy_tiers / 3 -
                          0.15 * claim_amount / 5000 + 0.25 * provider_network / 2 +
                          0.1 * (procedure_codes < 3).astype(float))
        y = (approval_score > np.median(approval_score)).astype(np.float32)

        self.cat_features = torch.LongTensor(np.column_stack([procedure_codes, policy_tiers, provider_network]))
        self.num_features = torch.FloatTensor(np.column_stack([pre_auth, claim_amount_norm]))
        self.y = torch.LongTensor(y.astype(int))

        self.cat_dims = [8, 4, 3]  # cardinality of each categorical feature
        self.cat_names = ['procedure_code', 'policy_tier', 'provider_network']
        self.num_names = ['pre_authorization', 'claim_amount']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.cat_features[idx], self.num_features[idx], self.y[idx]
```

---

## **Model with Categorical Embeddings**

```python
class CategoricalEmbeddingClassifier(nn.Module):
    """Neural network with learned categorical embeddings (CatBoost-inspired)."""

    def __init__(self, cat_dims, n_numerical=2, embed_dim=4, hidden_dim=32, n_classes=2):
        super().__init__()

        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in cat_dims
        ])

        total_embed_dim = len(cat_dims) * embed_dim + n_numerical

        self.network = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, cat_x, num_x):
        # Embed each categorical feature
        embedded = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embedded = torch.cat(embedded, dim=1)

        # Concatenate with numerical features
        combined = torch.cat([cat_embedded, num_x], dim=1)
        return self.network(combined)

    def get_embeddings(self, feature_idx):
        """Get learned embeddings for a categorical feature."""
        return self.embeddings[feature_idx].weight.data
```

---

## **Training Loop**

```python
def train_claim_model(model, train_loader, val_loader, epochs=120, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for cat_batch, num_batch, y_batch in train_loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(cat_batch, num_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 30 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for cat_b, num_b, y_b in val_loader:
                    cat_b, num_b, y_b = cat_b.to(device), num_b.to(device), y_b.to(device)
                    predicted = model(cat_b, num_b).argmax(dim=1)
                    correct += (predicted == y_b).sum().item()
                    total += y_b.size(0)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {correct/total:.4f}")

    return model
```

---

## **Full Pipeline**

```python
dataset = InsuranceClaimDataset(n_claims=1200)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

model = CategoricalEmbeddingClassifier(
    cat_dims=dataset.cat_dims, n_numerical=2, embed_dim=4, hidden_dim=32
)
model = train_claim_model(model, train_loader, val_loader, epochs=120)
```

---

## **Results**

```
Epoch 30/120  | Loss: 0.5432 | Val Acc: 0.7750
Epoch 60/120  | Loss: 0.3876 | Val Acc: 0.8292
Epoch 90/120  | Loss: 0.3123 | Val Acc: 0.8542
Epoch 120/120 | Loss: 0.2765 | Val Acc: 0.8625
```

---

## **Embedding Analysis**

```python
def analyze_embeddings(model, dataset):
    """Analyze learned categorical embeddings."""
    procedure_codes = ['D0120', 'D0274', 'D1110', 'D2750', 'D3310', 'D4341', 'D7210', 'D8080']
    embeddings = model.get_embeddings(0).numpy()

    print("Procedure Code Embeddings (first 2 dimensions):")
    for code, emb in zip(procedure_codes, embeddings):
        print(f"  {code}: [{emb[0]:+.3f}, {emb[1]:+.3f}, ...]")

    # Similar codes cluster together in embedding space
    from scipy.spatial.distance import cosine
    print(f"\nSimilarity D0120 vs D0274 (both diagnostic): {1-cosine(embeddings[0], embeddings[1]):.3f}")
    print(f"Similarity D0120 vs D7210 (diagnostic vs surgical): {1-cosine(embeddings[0], embeddings[6]):.3f}")

analyze_embeddings(model, dataset)
```

---

## **Inference**

```python
def process_claim(model, dataset, claim_data):
    """Process a dental insurance claim."""
    proc_map = {'D0120':0, 'D0274':1, 'D1110':2, 'D2750':3, 'D3310':4, 'D4341':5, 'D7210':6, 'D8080':7}
    tier_map = {'basic':0, 'standard':1, 'premium':2, 'comprehensive':3}
    net_map = {'out_of_network':0, 'in_network':1, 'preferred':2}

    cat = torch.LongTensor([[proc_map[claim_data['procedure_code']],
                              tier_map[claim_data['policy_tier']],
                              net_map[claim_data['provider_network']]]])
    num = torch.FloatTensor([[claim_data['pre_authorization'],
                               (claim_data['claim_amount'] - dataset.amount_mean) / dataset.amount_std]])

    model.eval()
    with torch.no_grad():
        logits = model(cat, num)
        probs = torch.softmax(logits, dim=1)[0]
        predicted = probs.argmax().item()

    labels = ['DENIED', 'APPROVED']
    return {
        'decision': labels[predicted],
        'approval_probability': f"{probs[1]:.1%}",
        'confidence': f"{probs[predicted]:.1%}"
    }

result = process_claim(model, dataset, {
    'procedure_code': 'D2750',
    'policy_tier': 'premium',
    'pre_authorization': 1,
    'claim_amount': 800.00,
    'provider_network': 'preferred'
})
print(result)
# {'decision': 'APPROVED', 'approval_probability': '89.3%', ...}
```

---

## **Model Export**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'cat_dims': dataset.cat_dims,
    'amount_mean': dataset.amount_mean,
    'amount_std': dataset.amount_std
}, 'claim_approval_model.pt')
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Heavy categorical data | CatBoost library (native support) |
| Learned embeddings for categories | PyTorch categorical embeddings |
| Integration with text claims | Add NLP branch to the model |
| Custom asymmetric loss | PyTorch gives full control |
| GPU batch processing | Fast batch inference for claims |

---

## **Running the Demo**

```bash
cd examples/02_classification
python catboost_pytorch.py
```

---

## **References**

1. Prokhorenkova et al. "CatBoost" (2018), NeurIPS
2. Guo & Berkhahn "Entity Embeddings of Categorical Variables" (2016)
3. ADA Procedure Code Reference Guide

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
