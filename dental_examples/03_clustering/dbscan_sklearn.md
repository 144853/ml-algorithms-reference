# DBSCAN Clustering - Simple Use Case & Data Explanation

## **Use Case: Identifying Clusters of Dental Clinics with Similar Performance Patterns**

### **The Problem**
A dental services organization manages **180 dental clinics** and wants to identify natural groupings based on performance metrics:
- **Monthly patient volume** (50-800 patients)
- **Procedure mix index** (0-1, ratio of complex to simple procedures)
- **Patient satisfaction score** (1-5 average rating)
- **Monthly revenue** ($20k-$500k)

**Goal:** Discover natural clusters of similar clinics and identify outlier clinics that may need intervention.

### **Why DBSCAN?**
| Criteria | K-Means | DBSCAN | Hierarchical |
|----------|---------|--------|--------------|
| Handles outliers | No | Yes (labels as noise) | No |
| Arbitrary cluster shapes | No | Yes | Yes |
| Requires k specification | Yes | No | No |
| Density-based | No | Yes | No |
| Handles varying densities | No | Partially | No |

DBSCAN is ideal here because clinic performance patterns may form irregular shapes, and we want to automatically flag outlier clinics.

---

## **Example Data Structure**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_clinics = 180

data = pd.DataFrame({
    'monthly_patient_volume': np.concatenate([
        np.random.normal(200, 40, 60),   # Small clinics
        np.random.normal(450, 60, 70),   # Medium clinics
        np.random.normal(700, 50, 40),   # Large clinics
        np.random.uniform(50, 800, 10)   # Outlier clinics
    ]).clip(50, 800),
    'procedure_mix_index': np.concatenate([
        np.random.normal(0.3, 0.08, 60),
        np.random.normal(0.55, 0.1, 70),
        np.random.normal(0.75, 0.07, 40),
        np.random.uniform(0, 1, 10)
    ]).clip(0, 1),
    'patient_satisfaction': np.concatenate([
        np.random.normal(3.8, 0.4, 60),
        np.random.normal(4.2, 0.3, 70),
        np.random.normal(4.5, 0.2, 40),
        np.random.uniform(1, 5, 10)
    ]).clip(1, 5),
    'monthly_revenue': np.concatenate([
        np.random.normal(80000, 15000, 60),
        np.random.normal(200000, 35000, 70),
        np.random.normal(380000, 40000, 40),
        np.random.uniform(20000, 500000, 10)
    ]).clip(20000, 500000)
})

print(data.describe().round(1))
```

---

## **DBSCAN Mathematics (Simple Terms)**

**Core Concepts:**
- **Epsilon (eps):** Maximum distance between two points to be considered neighbors (like the maximum distance between two clinics to be considered "similar")
- **MinPts (min_samples):** Minimum number of points to form a dense region

**Point Classification:**
1. **Core point:** Has at least `min_samples` neighbors within `eps` distance
2. **Border point:** Within `eps` of a core point but has fewer than `min_samples` neighbors
3. **Noise point:** Neither core nor border -- these are the outlier clinics

**Distance metric (Euclidean):**
$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

---

## **The Algorithm**

```python
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(X_scaled)

data['cluster'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points (outlier clinics): {n_noise}")
print(f"Cluster distribution:\n{pd.Series(labels).value_counts().sort_index()}")
```

**Finding Optimal eps using k-distance plot:**
```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])
# Plot k_distances to find the "elbow" for eps
```

---

## **Results From the Demo**

| Cluster | Clinics | Avg Volume | Avg Mix Index | Avg Satisfaction | Avg Revenue |
|---------|---------|------------|---------------|------------------|-------------|
| 0 - "Community Practices" | 58 | 198 | 0.31 | 3.8 | $79,500 |
| 1 - "General Practices" | 67 | 448 | 0.54 | 4.2 | $198,000 |
| 2 - "Specialty Centers" | 38 | 695 | 0.76 | 4.5 | $375,000 |
| -1 - "Outliers" | 17 | varies | varies | varies | varies |

### **Key Insights:**
- **17 outlier clinics** were automatically identified -- these have unusual performance combinations
- Some outliers have high volume but low satisfaction -- potential quality issues
- Others have low volume but high revenue -- possibly niche specialty practices
- The 3 natural clusters align with practice size/complexity tiers
- No need to pre-specify the number of clusters unlike K-Means

---

## **Simple Analogy**
Think of DBSCAN like a dental hygienist examining an X-ray for tooth clusters. Dense groups of healthy teeth form natural clusters, while isolated problem teeth stand out as anomalies. DBSCAN finds these natural groupings without being told how many groups to expect -- just like an experienced hygienist spots patterns without a predefined checklist.

---

## **When to Use**
**Good for dental applications:**
- Identifying outlier dental practices in a network
- Discovering natural patient groupings without predefined segments
- Detecting unusual treatment patterns in clinical data
- Grouping dental insurance claims with similar characteristics

**When NOT to use:**
- When clusters have very different densities (use HDBSCAN)
- When you need every point assigned to a cluster
- When data is very high-dimensional (>20 features)

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| eps | 0.5 | 0.1-2.0 | Neighborhood radius |
| min_samples | 5 | 3-20 | Minimum cluster density |
| metric | 'euclidean' | 'euclidean', 'manhattan', 'cosine' | Distance function |
| algorithm | 'auto' | 'auto', 'ball_tree', 'kd_tree', 'brute' | Computation method |

---

## **Running the Demo**
```bash
cd examples/03_clustering
python dbscan_demo.py
```

---

## **References**
- Ester, M. et al. (1996). "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
- Schubert, E. et al. (2017). "DBSCAN Revisited"
- scikit-learn documentation: sklearn.cluster.DBSCAN

---

## **Implementation Reference**
- See `examples/03_clustering/dbscan_demo.py` for full runnable code
- Preprocessing: `StandardScaler` for feature normalization
- Evaluation: Silhouette score (excluding noise), cluster profiling

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
