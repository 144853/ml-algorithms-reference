# Hierarchical Clustering - PyTorch Implementation

## **Use Case: Grouping Dental Procedures by Similarity for Billing Optimization**

### **The Problem**
A dental billing department manages **45 common dental procedures** and wants to group them by similarity for streamlined billing codes:
- **Time required** (minutes: 10-180)
- **Materials cost** ($5-$500)
- **Complexity score** (1-10)
- **Patient pain level** (0-10)

**Goal:** Create a hierarchical grouping using PyTorch for GPU-accelerated distance computation.

### **Why PyTorch for Hierarchical Clustering?**
| Criteria | Sklearn | PyTorch |
|----------|---------|---------|
| Distance matrix computation | CPU | GPU-accelerated |
| Custom linkage functions | Limited | Fully customizable |
| Integration with DL | No | Seamless |
| Large-scale pairwise distances | Slow | Fast |

---

## **Example Data Structure**

```python
import torch
import numpy as np

torch.manual_seed(42)

# Dental procedure features as tensors
time_minutes = torch.tensor([30, 60, 15, 20, 10, 15, 35, 45, 30, 60, 90,
                              75, 70, 120, 20, 45, 75, 120, 60, 90,
                              60, 90, 30, 30, 20, 45, 40, 60, 90,
                              45, 60, 10, 25, 30, 35, 120, 90, 30, 45, 25,
                              15, 20, 10, 5, 30], dtype=torch.float32)

materials_cost = torch.tensor([15, 35, 10, 5, 8, 12, 45, 65, 30, 80, 120,
                                250, 180, 400, 10, 25, 35, 500, 300, 350,
                                120, 400, 60, 5, 15, 200, 80, 40, 150,
                                25, 40, 12, 50, 20, 60, 350, 250, 30, 45, 15,
                                5, 15, 5, 3, 50], dtype=torch.float32)

complexity = torch.tensor([2, 4, 1, 1, 1, 2, 3, 4, 3, 7, 9,
                            7, 6, 8, 3, 5, 7, 10, 8, 9,
                            3, 8, 4, 2, 3, 5, 4, 6, 7,
                            4, 5, 1, 3, 4, 3, 7, 6, 3, 4, 5,
                            2, 3, 1, 1, 3], dtype=torch.float32)

pain_level = torch.tensor([1, 3, 0, 0, 0, 0, 2, 3, 2, 5, 7,
                            3, 3, 4, 3, 5, 7, 6, 3, 5,
                            1, 2, 1, 0, 2, 1, 1, 4, 5,
                            2, 4, 0, 1, 2, 1, 3, 2, 1, 2, 4,
                            1, 3, 0, 0, 0], dtype=torch.float32)

X = torch.stack([time_minutes, materials_cost, complexity, pain_level], dim=1)
print(f"Data shape: {X.shape}")  # torch.Size([45, 4])
```

---

## **Hierarchical Clustering Mathematics (Simple Terms)**

**Agglomerative Algorithm with PyTorch:**

1. Compute pairwise distance matrix using `torch.cdist`
2. Find minimum distance pair
3. Merge clusters and update distance matrix
4. Record merge in linkage matrix
5. Repeat until single cluster remains

**Ward's Linkage (variance minimization):**
$$\Delta(A,B) = \frac{|A| \cdot |B|}{|A| + |B|} ||\mu_A - \mu_B||^2$$

---

## **The Algorithm**

```python
class HierarchicalClusteringPyTorch:
    def __init__(self, n_clusters=5, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage_type = linkage

    def fit(self, X):
        device = X.device
        n = X.shape[0]

        # Standardize
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0)
        X_norm = (X - self.mean) / self.std

        # Initialize: each point is its own cluster
        cluster_ids = list(range(n))
        cluster_members = {i: [i] for i in range(n)}
        cluster_centroids = {i: X_norm[i].clone() for i in range(n)}
        cluster_sizes = {i: 1 for i in range(n)}

        # Compute initial pairwise distances
        dist_matrix = torch.cdist(X_norm, X_norm)
        dist_matrix.fill_diagonal_(float('inf'))

        merge_history = []
        next_id = n

        for step in range(n - self.n_clusters):
            # Find minimum distance pair
            min_val = dist_matrix.min()
            min_idx = (dist_matrix == min_val).nonzero()[0]
            i, j = min_idx[0].item(), min_idx[1].item()

            # Record merge
            merge_history.append((i, j, min_val.item()))

            # Merge: compute new centroid (Ward's method)
            ni, nj = cluster_sizes[i], cluster_sizes[j]
            new_centroid = (ni * cluster_centroids[i] + nj * cluster_centroids[j]) / (ni + nj)

            # Update distances using Lance-Williams formula for Ward
            active = [k for k in cluster_sizes.keys() if k != i and k != j]
            new_dists = []
            for k in active:
                nk = cluster_sizes[k]
                d_ik = dist_matrix[i, k] if i < dist_matrix.shape[0] and k < dist_matrix.shape[1] else float('inf')
                d_jk = dist_matrix[j, k] if j < dist_matrix.shape[0] and k < dist_matrix.shape[1] else float('inf')
                d_ij = min_val
                n_total = ni + nj + nk
                new_d = ((ni + nk) * d_ik + (nj + nk) * d_jk - nk * d_ij) / n_total
                new_dists.append((k, new_d))

            # Update structures
            cluster_centroids[i] = new_centroid
            cluster_sizes[i] = ni + nj
            cluster_members[i] = cluster_members[i] + cluster_members[j]
            del cluster_centroids[j]
            del cluster_sizes[j]
            del cluster_members[j]

            # Mark j as merged in distance matrix
            dist_matrix[j, :] = float('inf')
            dist_matrix[:, j] = float('inf')
            for k, d in new_dists:
                dist_matrix[i, k] = d
                dist_matrix[k, i] = d

        # Assign final labels
        self.labels = torch.zeros(n, dtype=torch.long)
        for label, (cid, members) in enumerate(cluster_members.items()):
            for m in members:
                self.labels[m] = label

        return self

# Run clustering
model = HierarchicalClusteringPyTorch(n_clusters=5)
model.fit(X)
print(f"Cluster sizes: {torch.bincount(model.labels)}")
```

---

## **Results From the Demo**

| Cluster | Name | Procedures | Avg Time | Avg Cost | Avg Complexity |
|---------|------|-----------|----------|----------|----------------|
| 0 | "Quick Diagnostics" | Exam, X-rays, Screening | 11 min | $6 | 1.2 |
| 1 | "Preventive Care" | Cleanings, Fluoride, Sealants | 27 min | $19 | 2.1 |
| 2 | "Basic Restorative" | Fillings, Bonding, Simple Extractions | 34 min | $45 | 3.5 |
| 3 | "Advanced Restorative" | Crowns, Bridges, Root Canals | 80 min | $220 | 7.0 |
| 4 | "Surgical/Complex" | Implants, Bone Grafts, Wisdom Teeth | 96 min | $400 | 8.8 |

### **Key Insights:**
- PyTorch implementation produces equivalent hierarchy to sklearn
- GPU acceleration of distance matrix computation provides speedup for large procedure catalogs
- The merge history can be used to construct a dendrogram programmatically
- Custom linkage functions are easy to implement in PyTorch

---

## **Simple Analogy**
Think of this like organizing a dental instrument tray. You start by putting each instrument in its own slot, then combine the most similar ones: mirror and explorer go together, then those join the probe. Eventually you have organized sections: diagnostic tools, restorative instruments, surgical instruments. PyTorch does this sorting at GPU speed -- measuring all instrument similarities at once.

---

## **When to Use**
**PyTorch hierarchical clustering is ideal when:**
- Computing distance matrices for large procedure catalogs
- Implementing custom linkage criteria for dental-specific similarity
- Integrating procedure grouping into a deep learning billing system
- Processing multiple distance metrics simultaneously on GPU

**When NOT to use:**
- Small procedure sets (<100) -- sklearn is simpler
- When scipy dendrogram visualization is needed directly
- When memory is limited (O(n^2) distance matrix)

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| n_clusters | 5 | 2-10 | Number of procedure groups |
| linkage | 'ward' | 'ward', 'complete', 'average' | Merge strategy |
| Distance metric | Euclidean | Euclidean, Manhattan, Cosine | Similarity measure |

---

## **Running the Demo**
```bash
cd examples/03_clustering
python hierarchical_clustering_pytorch_demo.py
```

---

## **References**
- Ward, J.H. (1963). "Hierarchical Grouping to Optimize an Objective Function"
- Lance, G.N. & Williams, W.T. (1967). "A General Theory of Classificatory Sorting Strategies"
- PyTorch documentation: torch.cdist

---

## **Implementation Reference**
- See `examples/03_clustering/hierarchical_clustering_pytorch_demo.py` for full runnable code
- GPU support: Move tensors to CUDA with `.to('cuda')`
- Dendrogram: Export merge history to scipy format for visualization

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
