# Hierarchical Clustering - Simple Use Case & Data Explanation

## **Use Case: Grouping Dental Procedures by Similarity for Billing Optimization**

### **The Problem**
A dental billing department manages **45 common dental procedures** and wants to group them by similarity for streamlined billing codes and bundled pricing:
- **Time required** (minutes: 10-180)
- **Materials cost** ($5-$500)
- **Complexity score** (1-10, based on skill required)
- **Patient pain level** (0-10, average reported pain)

**Goal:** Create a hierarchical grouping of procedures for tiered billing packages without specifying the number of groups upfront.

### **Why Hierarchical Clustering?**
| Criteria | K-Means | DBSCAN | Hierarchical |
|----------|---------|--------|--------------|
| Dendrogram visualization | No | No | Yes |
| No need to pre-specify k | No | Yes | Yes |
| Nested cluster structure | No | No | Yes |
| Deterministic | No | Yes | Yes |
| Works with any linkage | No | No | Yes |

Hierarchical clustering is ideal because dental procedures have natural hierarchies (e.g., restorative > crowns > porcelain crowns).

---

## **Example Data Structure**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

procedures = [
    'Routine Cleaning', 'Deep Cleaning', 'Fluoride Treatment',
    'Dental Exam', 'Bitewing X-ray', 'Panoramic X-ray',
    'Composite Filling (1 surface)', 'Composite Filling (2 surface)',
    'Amalgam Filling', 'Root Canal (Anterior)', 'Root Canal (Molar)',
    'Crown (Porcelain)', 'Crown (Metal)', 'Bridge (3 unit)',
    'Extraction (Simple)', 'Extraction (Surgical)', 'Wisdom Tooth Extraction',
    'Dental Implant', 'Implant Crown', 'Bone Graft',
    'Teeth Whitening', 'Veneer (Porcelain)', 'Dental Bonding',
    'Orthodontic Consultation', 'Braces Adjustment', 'Clear Aligner Fitting',
    'Night Guard', 'TMJ Treatment', 'Gum Graft',
    'Periodontal Maintenance', 'Scaling and Root Planing',
    'Sealant (per tooth)', 'Space Maintainer', 'Pulpotomy',
    'Stainless Steel Crown', 'Denture (Full)', 'Denture (Partial)',
    'Denture Repair', 'Denture Reline', 'Tooth Reattachment',
    'Emergency Exam', 'Palliative Treatment', 'Oral Cancer Screening',
    'Intraoral Photo', 'CT Scan (CBCT)'
]

data = pd.DataFrame({
    'procedure': procedures,
    'time_minutes': [30, 60, 15, 20, 10, 15, 35, 45, 30, 60, 90,
                     75, 70, 120, 20, 45, 75, 120, 60, 90,
                     60, 90, 30, 30, 20, 45, 40, 60, 90,
                     45, 60, 10, 25, 30, 35, 120, 90, 30, 45, 25,
                     15, 20, 10, 5, 30],
    'materials_cost': [15, 35, 10, 5, 8, 12, 45, 65, 30, 80, 120,
                       250, 180, 400, 10, 25, 35, 500, 300, 350,
                       120, 400, 60, 5, 15, 200, 80, 40, 150,
                       25, 40, 12, 50, 20, 60, 350, 250, 30, 45, 15,
                       5, 15, 5, 3, 50],
    'complexity_score': [2, 4, 1, 1, 1, 2, 3, 4, 3, 7, 9,
                         7, 6, 8, 3, 5, 7, 10, 8, 9,
                         3, 8, 4, 2, 3, 5, 4, 6, 7,
                         4, 5, 1, 3, 4, 3, 7, 6, 3, 4, 5,
                         2, 3, 1, 1, 3],
    'pain_level': [1, 3, 0, 0, 0, 0, 2, 3, 2, 5, 7,
                   3, 3, 4, 3, 5, 7, 6, 3, 5,
                   1, 2, 1, 0, 2, 1, 1, 4, 5,
                   2, 4, 0, 1, 2, 1, 3, 2, 1, 2, 4,
                   1, 3, 0, 0, 0]
})

print(data.head(10))
```

---

## **Hierarchical Clustering Mathematics (Simple Terms)**

**Agglomerative (Bottom-Up) Approach:**
1. Start with each procedure as its own cluster
2. Find the two closest clusters and merge them
3. Repeat until all procedures are in one cluster

**Linkage Methods:**
- **Ward's method** (minimizes total variance): $d(A,B) = \sqrt{\frac{2|A||B|}{|A|+|B|}} ||\mu_A - \mu_B||$
- **Complete linkage** (maximum distance): $d(A,B) = \max_{a \in A, b \in B} ||a - b||$
- **Average linkage**: $d(A,B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} ||a - b||$

---

## **The Algorithm**

```python
# Standardize features
feature_cols = ['time_minutes', 'materials_cost', 'complexity_score', 'pain_level']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[feature_cols])

# Create linkage matrix for dendrogram
Z = linkage(X_scaled, method='ward')

# Apply Agglomerative Clustering with 5 groups
agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels = agg.fit_predict(X_scaled)

data['cluster'] = labels

# View cluster assignments
for cluster_id in sorted(data['cluster'].unique()):
    cluster_procs = data[data['cluster'] == cluster_id]['procedure'].tolist()
    print(f"\nCluster {cluster_id}: {cluster_procs}")
```

**Dendrogram Visualization:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 8))
dendrogram(Z, labels=procedures, leaf_rotation=90, leaf_font_size=8)
plt.title('Dental Procedure Hierarchy')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('dental_procedure_dendrogram.png')
```

---

## **Results From the Demo**

| Cluster | Name | Procedures | Avg Time | Avg Cost | Avg Complexity |
|---------|------|-----------|----------|----------|----------------|
| 0 | "Quick Diagnostics" | Exam, X-rays, Screening, Photos | 12 min | $7 | 1.3 |
| 1 | "Preventive Care" | Cleaning, Fluoride, Sealants, Maintenance | 28 min | $20 | 2.2 |
| 2 | "Basic Restorative" | Fillings, Bonding, Simple Extractions | 33 min | $43 | 3.4 |
| 3 | "Advanced Restorative" | Crowns, Bridges, Dentures, Root Canals | 82 min | $215 | 6.8 |
| 4 | "Surgical/Complex" | Implants, Bone Grafts, Surgical Extractions | 95 min | $395 | 8.7 |

### **Key Insights:**
- The dendrogram reveals natural tiers of dental procedures matching clinical categorization
- Billing packages can be created for each tier with appropriate pricing
- "Quick Diagnostics" and "Preventive Care" merge first, suggesting similar resource needs
- "Surgical/Complex" procedures are most distant from all others, justifying premium billing codes
- The hierarchy helps insurance companies define coverage tiers

---

## **Simple Analogy**
Think of hierarchical clustering like organizing a dental supply catalog. You start by grouping identical items (all composite filling materials together), then merge similar groups (fillings with bonding agents), then combine related categories (all restorative materials), working up to broad sections (restorative vs. preventive vs. surgical). The dendrogram is the table of contents showing this nested organization.

---

## **When to Use**
**Good for dental applications:**
- Grouping dental procedures for billing code optimization
- Creating treatment package tiers
- Organizing dental supply catalogs
- Hierarchical patient risk stratification

**When NOT to use:**
- Large datasets (>10,000 items) -- O(n^3) complexity
- When flat partitions are sufficient (use K-Means)
- When outlier detection is the primary goal (use DBSCAN)

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| n_clusters | 2 | 2-10 | Number of final groups |
| linkage | 'ward' | 'ward', 'complete', 'average', 'single' | Merge strategy |
| metric | 'euclidean' | 'euclidean', 'manhattan', 'cosine' | Distance metric |
| distance_threshold | None | float | Cut dendrogram at distance |

---

## **Running the Demo**
```bash
cd examples/03_clustering
python hierarchical_clustering_demo.py
```

---

## **References**
- Ward, J.H. (1963). "Hierarchical Grouping to Optimize an Objective Function"
- Murtagh, F. & Contreras, P. (2012). "Algorithms for Hierarchical Clustering"
- scikit-learn documentation: sklearn.cluster.AgglomerativeClustering

---

## **Implementation Reference**
- See `examples/03_clustering/hierarchical_clustering_demo.py` for full runnable code
- Visualization: scipy.cluster.hierarchy.dendrogram
- Evaluation: Cophenetic correlation coefficient

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
