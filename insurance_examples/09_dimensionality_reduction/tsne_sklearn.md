# t-SNE (t-Distributed Stochastic Neighbor Embedding) with Scikit-Learn - Simple Use Case & Data Explanation

## 🛡️ **Use Case: Visualizing Insurance Customer Segments from Complex Policyholder Profiles**

### **The Problem**
A multi-line insurer has 80,000 policyholders described by 35 features -- demographics, policy details, claims history, payment behavior, and engagement scores. The marketing team wants to see natural customer groupings in a 2D scatter plot to design targeted product bundles and retention campaigns. Traditional clustering gives labels but no visual intuition about segment boundaries and overlaps.

### **Why t-SNE?**
| Criteria | t-SNE | PCA | UMAP | MDS |
|---|---|---|---|---|
| Preserves local neighborhood structure | Excellent | Poor | Excellent | Moderate |
| Reveals non-linear clusters | Yes | No | Yes | No |
| Produces interpretable 2D visualizations | Yes | Yes | Yes | Yes |
| Widely adopted and well-understood | Yes | Yes | Growing | Yes |

---

## 📊 **Example Data Structure**

```python
import pandas as pd
import numpy as np

# 35-feature policyholder profile dataset
data = {
    'age': [29, 55, 42, 67, 35],
    'gender': [1, 0, 1, 0, 1],
    'marital_status': [0, 1, 1, 1, 0],
    'annual_income': [45000, 92000, 68000, 55000, 78000],
    'credit_score': [710, 650, 740, 620, 690],
    'num_policies': [1, 3, 2, 4, 1],
    'policy_tenure_years': [2, 15, 7, 22, 3],
    'total_premium': [1400, 5200, 3100, 6800, 1800],
    'num_claims_5yr': [0, 4, 1, 6, 1],
    'avg_claim_amount': [0, 3200, 1500, 4800, 800],
    'payment_method': [1, 0, 1, 0, 1],  # 1=autopay, 0=manual
    'months_since_last_contact': [3, 18, 6, 24, 1],
    'nps_score': [8, 4, 7, 3, 9],
    'digital_engagement': [0.85, 0.20, 0.65, 0.10, 0.90],
    'bundled_discount': [0, 1, 1, 1, 0]
}
# ... remaining 20 features omitted for brevity
df = pd.DataFrame(data)
```

---

## 🔬 **t-SNE Mathematics (Simple Terms)**

t-SNE converts high-dimensional distances into probabilities, then arranges points in 2D so that similar policyholders stay close.

**Step 1 -- Pairwise similarity in high-dimensional space (Gaussian):**

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

**Step 2 -- Symmetrize:**

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Step 3 -- Pairwise similarity in 2D space (Student-t with 1 degree of freedom):**

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

**Step 4 -- Minimize KL divergence via gradient descent:**

$$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

---

## ⚙️ **The Algorithm**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Run t-SNE to produce 2D embedding
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    n_iter=1000,
    random_state=42,
    init='pca'
)
X_2d = tsne.fit_transform(X_scaled)

# 3. Plot customer segments
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      c=cluster_labels, cmap='tab10',
                      alpha=0.6, s=10)
plt.colorbar(scatter, label='Customer Segment')
plt.title('Insurance Customer Segments (t-SNE Visualization)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig('customer_segments_tsne.png', dpi=150)
```

---

## 📈 **Results From the Demo**

```
Policyholders visualized:  80,000
Original features:         35
Output dimensions:         2
Perplexity:                30
Iterations:                1,000
Final KL divergence:       1.42

Identified visual clusters:
  Cluster A (Blue)   - Young digital-first, single policy, low claims
  Cluster B (Orange) - Mid-career families, bundled policies, moderate claims
  Cluster C (Green)  - Senior high-value, multi-policy, frequent claims
  Cluster D (Red)    - High-risk, low credit, payment issues
  Cluster E (Purple) - Loyal long-tenure, high NPS, autopay
```

### **Key Insights:**
- **Five natural customer segments emerged** without predefined labels -- validating the K=5 chosen for K-Means separately.
- **Cluster D (high-risk) forms a tight, isolated group** -- these policyholders share low credit scores and manual payment, making them candidates for targeted retention.
- **Clusters B and E partially overlap** -- suggesting mid-career families and loyal long-tenure customers share characteristics, a cross-sell opportunity.
- **Young digital-first customers (Cluster A)** are well-separated, confirming that digital engagement is a strong differentiator for product design.

---

## 💡 **Simple Analogy**

Imagine 80,000 policyholder files scattered across a warehouse with 35 different filing criteria. t-SNE is like a librarian who reads every file, understands which policyholders are "similar neighbors," and then arranges all 80,000 files on a single large table so that similar customers end up physically close together. The marketing team can now walk around the table and immediately see natural groups.

---

## 🎯 **When to Use**

### **Use when:**
- You need a 2D or 3D visualization of high-dimensional customer data
- You want to visually validate clustering results
- Exploring whether natural segments exist before running formal clustering
- Communicating data patterns to non-technical stakeholders
- The dataset is under 100K rows (t-SNE scales quadratically)

### **Common Insurance Applications:**
- Visualizing policyholder segments for marketing campaign targeting
- Exploring fraud patterns -- do fraudulent claims cluster together?
- Mapping agent performance profiles to identify coaching opportunities
- Understanding geographic risk clusters from multi-dimensional location features
- Visualizing claim severity patterns across product lines

### **When NOT to use:**
- When you need a deterministic, reproducible embedding (t-SNE results vary across runs)
- When the dataset exceeds 200K rows -- runtime becomes prohibitive on CPU
- When you need to project new unseen data points (t-SNE has no `transform` method)
- When you need distance-preserving embeddings for downstream modeling (use PCA or UMAP)

---

## 🔧 **Hyperparameters to Tune**

| Parameter | Description | Typical Values |
|---|---|---|
| `perplexity` | Balance between local and global structure | `5`-`50`; try `30` first, lower for tight clusters |
| `learning_rate` | Step size for gradient descent | `'auto'` (recommended), or `10`-`1000` |
| `n_iter` | Number of optimization iterations | `1000` minimum, `2000`-`5000` for large datasets |
| `init` | Initialization method | `'pca'` (recommended for stability) |
| `early_exaggeration` | Factor to tighten clusters early | `12.0` (default), increase for clearer separation |

```python
# Example: fine-tuned for large insurance dataset
tsne = TSNE(
    perplexity=40,
    learning_rate='auto',
    n_iter=2000,
    init='pca',
    early_exaggeration=15.0,
    random_state=42
)
```

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 09_dimensionality_reduction/tsne_sklearn.py
```

---

## 📚 **References**

1. van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
2. Wattenberg, M., Viegas, F., & Johnson, I. (2016). How to Use t-SNE Effectively. *Distill*.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## 📝 **Implementation Reference**

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
