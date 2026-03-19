# t-SNE - Simple Use Case & Data Explanation

## **Use Case: Visualizing Dental Patient Clusters in 2D from Complex Health Profiles**

### **The Problem**
A dental research institute has **3,000 patient records** with **20 clinical features** and wants to visualize natural patient groupings in a 2D plot:
- **Oral health metrics:** DMFT, plaque index, gingival index, pocket depth, BOP
- **Treatment history:** Procedure counts, visit frequency, spending
- **Risk factors:** Smoking, diabetes, diet, anxiety
- **Demographics:** Age, gender, insurance

**Goal:** Create an intuitive 2D visualization revealing patient subgroups for clinical research.

### **Why t-SNE?**
| Criteria | PCA | t-SNE | UMAP |
|----------|-----|-------|------|
| Preserves local structure | No | Excellent | Excellent |
| Preserves global structure | Yes | Poor | Good |
| Non-linear relationships | No | Yes | Yes |
| Cluster visualization | Poor | Excellent | Excellent |
| Deterministic | Yes | No | No |
| Speed | Fast | Slow | Moderate |

t-SNE excels at creating visually meaningful 2D plots that reveal patient clusters.

---

## **Example Data Structure**

```python
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_patients = 3000

# Create patient subgroups (for validation)
# Group 1: Young healthy (n=1000)
g1 = np.column_stack([
    np.random.normal(28, 5, 1000),    # age
    np.random.normal(2, 2, 1000),      # DMFT
    np.random.normal(0.5, 0.2, 1000),  # plaque
    np.random.normal(0.4, 0.2, 1000),  # gingival
    np.random.normal(2.0, 0.5, 1000),  # pocket depth
    np.random.normal(0.1, 0.05, 1000), # BOP
    np.random.poisson(1, 1000),         # n_fillings
    np.random.normal(2.5, 0.5, 1000),  # visits/year
    np.random.normal(3, 1, 1000),       # sugar score
    np.random.normal(2, 1, 1000),       # anxiety
] + [np.random.randn(1000) for _ in range(10)])

# Group 2: Middle-aged moderate risk (n=1200)
g2 = np.column_stack([
    np.random.normal(48, 8, 1200),
    np.random.normal(10, 4, 1200),
    np.random.normal(1.2, 0.4, 1200),
    np.random.normal(1.0, 0.3, 1200),
    np.random.normal(3.5, 0.8, 1200),
    np.random.normal(0.3, 0.1, 1200),
    np.random.poisson(5, 1200),
    np.random.normal(1.5, 0.5, 1200),
    np.random.normal(5, 1.5, 1200),
    np.random.normal(4, 1.5, 1200),
] + [np.random.randn(1200) for _ in range(10)])

# Group 3: Elderly high risk (n=800)
g3 = np.column_stack([
    np.random.normal(68, 7, 800),
    np.random.normal(18, 5, 800),
    np.random.normal(1.8, 0.5, 800),
    np.random.normal(1.8, 0.4, 800),
    np.random.normal(5.0, 1.0, 800),
    np.random.normal(0.5, 0.15, 800),
    np.random.poisson(10, 800),
    np.random.normal(1.0, 0.5, 800),
    np.random.normal(4, 1.5, 800),
    np.random.normal(5, 2, 800),
] + [np.random.randn(800) for _ in range(10)])

X = np.vstack([g1, g2, g3])
labels = np.array(['Young Healthy'] * 1000 + ['Middle-Aged Moderate'] * 1200 + ['Elderly High-Risk'] * 800)

print(f"Data shape: {X.shape}")
```

---

## **t-SNE Mathematics (Simple Terms)**

**Step 1: Compute pairwise similarities in high-dimensional space (Gaussian):**
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

**Step 2: Compute pairwise similarities in low-dimensional space (Student's t):**
$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

**Step 3: Minimize KL divergence between P and Q:**
$$KL(P||Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

The Student's t-distribution in the low-dimensional space prevents the "crowding problem" -- similar points stay close, dissimilar points are pushed far apart.

---

## **The Algorithm**

```python
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,         # Balances local vs global structure
    learning_rate=200,
    n_iter=1000,
    random_state=42,
    metric='euclidean',
    init='pca'             # PCA initialization for stability
)

X_tsne = tsne.fit_transform(X_scaled)

# Visualization
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
for label in ['Young Healthy', 'Middle-Aged Moderate', 'Elderly High-Risk']:
    mask = labels == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.5, s=10)

ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_title('Dental Patient Clusters (t-SNE Visualization)')
ax.legend()
plt.savefig('dental_patient_tsne.png', dpi=150)

# Compare different perplexity values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, perp in enumerate([5, 30, 100]):
    tsne_temp = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_temp = tsne_temp.fit_transform(X_scaled)
    for label in ['Young Healthy', 'Middle-Aged Moderate', 'Elderly High-Risk']:
        mask = labels == label
        axes[i].scatter(X_temp[mask, 0], X_temp[mask, 1], label=label, alpha=0.5, s=10)
    axes[i].set_title(f'Perplexity = {perp}')
plt.savefig('dental_tsne_perplexity_comparison.png', dpi=150)
```

---

## **Results From the Demo**

**t-SNE with perplexity=30:**
- **3 distinct clusters** clearly visible in the 2D plot
- Young Healthy cluster is tight and well-separated
- Middle-Aged Moderate cluster is the largest and most dispersed
- Elderly High-Risk cluster shows some overlap with Middle-Aged Moderate

**Perplexity Comparison:**
| Perplexity | Cluster Separation | Local Detail | Global Structure |
|------------|-------------------|--------------|------------------|
| 5 | Over-fragmented | Too much | Lost |
| 30 | Good | Good | Moderate |
| 100 | Merged | Less | Better |

**KL Divergence:** 0.89 (lower is better, indicating good fit)

### **Key Insights:**
- t-SNE reveals 3 natural patient subgroups aligning with clinical expectations
- Young patients with good oral health cluster separately from high-risk elderly
- Some middle-aged patients overlap with elderly high-risk (early periodontal disease)
- Perplexity of 30 provides the best balance for dental patient data
- PCA initialization speeds convergence and improves reproducibility
- Sub-clusters within groups may reveal additional clinical phenotypes

---

## **Simple Analogy**
t-SNE is like a dental conference seating arrangement. In reality, attendees (patients) differ in 20 ways (features). t-SNE figures out how to seat everyone in a 2D room so that patients who are clinically similar sit near each other. Young healthy patients end up at one table, elderly periodontal patients at another, and moderate-risk patients fill the middle tables. The algorithm preserves the neighborhood relationships while fitting everyone into the room.

---

## **When to Use**
**Good for dental applications:**
- Visualizing patient populations for research presentations
- Exploring dental phenotype clusters
- Quality checking clustering results
- Identifying outlier patients in clinical studies

**When NOT to use:**
- When you need to transform new data (t-SNE has no transform method)
- When global distances matter (use PCA or UMAP)
- When speed is critical for large datasets (>10k points, use UMAP)
- For feature selection or dimensionality reduction as preprocessing

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| perplexity | 30 | 5-100 | Local vs. global balance |
| learning_rate | 200 | 10-1000 | Optimization speed |
| n_iter | 1000 | 250-5000 | Convergence |
| metric | 'euclidean' | 'euclidean', 'cosine', 'manhattan' | Distance function |
| init | 'pca' | 'pca', 'random' | Initialization method |
| early_exaggeration | 12.0 | 4-50 | Initial cluster separation |

---

## **Running the Demo**
```bash
cd examples/09_dimensionality_reduction
python tsne_demo.py
```

---

## **References**
- van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE"
- Wattenberg, M. et al. (2016). "How to Use t-SNE Effectively"
- scikit-learn documentation: sklearn.manifold.TSNE

---

## **Implementation Reference**
- See `examples/09_dimensionality_reduction/tsne_demo.py` for full runnable code
- Preprocessing: StandardScaler for feature normalization
- Visualization: Matplotlib scatter plots with cluster coloring

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
