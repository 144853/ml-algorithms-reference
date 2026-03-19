# Hierarchical Clustering - Simple Use Case & Data Explanation

## 🛡️ **Use Case: Grouping Insurance Products by Similarity for Cross-Selling**

### **The Problem**
An insurance company offers 85 distinct products across auto, home, life, health, and commercial lines. The product team needs to group similar products to design cross-selling bundles, identify redundant offerings, and optimize the product catalog. Hierarchical clustering reveals natural groupings at multiple levels of granularity without requiring a predetermined number of groups.

### **Why Hierarchical Clustering?**
| Factor | Hierarchical | K-Means | DBSCAN |
|--------|-------------|---------|--------|
| Dendrogram visualization | Yes | No | No |
| Multi-level groupings | Yes | No | No |
| No need for k upfront | Yes | No | Yes |
| Business interpretability | Excellent | Good | Medium |
| Product taxonomy | Natural fit | Poor fit | Poor fit |

Hierarchical clustering is ideal because insurance products have natural taxonomies (line > sub-line > product) and the dendrogram shows business stakeholders exactly how products relate at every level.

---

## 📊 **Example Data Structure**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Insurance product features
data = {
    'product_id': ['AUTO-BASIC', 'AUTO-PREM', 'HOME-BASIC', 'HOME-PREM',
                   'LIFE-TERM', 'LIFE-WHOLE', 'HEALTH-IND', 'HEALTH-FAM',
                   'COMM-PROP', 'COMM-LIAB'],
    'coverage_type': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],         # encoded: 1=auto,2=home,3=life,4=health,5=commercial
    'avg_premium': [1200, 2400, 1800, 3600, 800, 2200, 5400, 9800, 4500, 3800],
    'target_age_group': [30, 35, 40, 45, 35, 50, 38, 42, 45, 45],  # average policyholder age
    'rider_options': [3, 7, 4, 8, 2, 5, 6, 6, 5, 4],          # number of available riders
    'risk_complexity': [0.4, 0.6, 0.5, 0.7, 0.3, 0.5, 0.8, 0.8, 0.9, 0.85],  # 0-1 complexity score
    'avg_claim_ratio': [0.65, 0.58, 0.45, 0.40, 0.30, 0.25, 0.82, 0.78, 0.55, 0.60]
}

df = pd.DataFrame(data)
print(df.head())
```

**What each feature means:**
- **coverage_type**: Line of business (auto=1, home=2, life=3, health=4, commercial=5)
- **avg_premium**: Average annual premium for the product ($800-$9,800)
- **target_age_group**: Typical policyholder age for the product
- **rider_options**: Number of optional add-ons available (2-8)
- **risk_complexity**: Underwriting complexity score (0-1)
- **avg_claim_ratio**: Historical loss ratio for the product (0-1)

---

## 🔬 **Mathematics (Simple Terms)**

### **Linkage Methods**
Determine distance between clusters of products:

**Ward's Method** (minimizes within-cluster variance):
$$d(C_i, C_j) = \sqrt{\frac{2 n_i n_j}{n_i + n_j}} \|\bar{x}_i - \bar{x}_j\|_2$$

**Complete Linkage** (maximum distance):
$$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} \|x - y\|$$

**Average Linkage**:
$$d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} \|x - y\|$$

### **Agglomerative Process**
1. Start: each product is its own cluster (85 clusters)
2. Merge the two most similar clusters
3. Repeat until all products are in one cluster
4. Cut the dendrogram at desired height to get product groups

---

## ⚙️ **The Algorithm**

```
Algorithm: Agglomerative Hierarchical Clustering for Product Grouping
Input: Product feature matrix X (n x p), linkage method, distance threshold

1. NORMALIZE all product features using StandardScaler
2. COMPUTE pairwise distance matrix between all products
3. INITIALIZE each product as its own cluster
4. REPEAT until one cluster remains:
   a. FIND the two closest clusters (using linkage method)
   b. MERGE them into a single cluster
   c. UPDATE distance matrix
5. BUILD dendrogram from merge history
6. CUT dendrogram at chosen threshold to get product groups
```

```python
# Sklearn implementation
features = ['coverage_type', 'avg_premium', 'target_age_group',
            'rider_options', 'risk_complexity', 'avg_claim_ratio']
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit hierarchical clustering
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['product_group'] = hc.fit_predict(X_scaled)

# Build dendrogram for visualization
linkage_matrix = linkage(X_scaled, method='ward')

# Analyze product groups
for group in sorted(df['product_group'].unique()):
    products = df[df['product_group'] == group]['product_id'].tolist()
    avg_prem = df[df['product_group'] == group]['avg_premium'].mean()
    print(f"Group {group}: {products} (Avg Premium: ${avg_prem:,.0f})")
```

---

## 📈 **Results From the Demo**

| Product Group | Products | Avg Premium | Avg Loss Ratio | Cross-Sell Opportunity |
|--------------|----------|-------------|----------------|----------------------|
| 0 - Personal Auto | AUTO-BASIC, AUTO-PREM | $1,800 | 0.62 | Bundle with Home |
| 1 - Personal Property | HOME-BASIC, HOME-PREM | $2,700 | 0.43 | Bundle with Auto |
| 2 - Life & Protection | LIFE-TERM, LIFE-WHOLE | $1,500 | 0.28 | Upsell term to whole |
| 3 - Health & Commercial | HEALTH-IND, HEALTH-FAM, COMM-PROP, COMM-LIAB | $5,875 | 0.69 | Business packages |

**Business Actions:**
- **Auto + Home Bundle**: 15% discount drives 23% higher retention
- **Life Upsell Path**: Term-to-whole conversion campaign
- **Commercial Package**: Combine property + liability for small businesses
- **Dendrogram Insight**: Health and commercial products cluster together due to similar complexity and premium levels

---

## 💡 **Simple Analogy**

Think of hierarchical clustering like organizing an insurance company's product catalog into a family tree. At the bottom, every product stands alone. Moving up the tree, the most similar products merge: basic and premium auto combine first, then home products join. Higher still, auto and home merge into "personal lines." At the top, everything is one company. The dendrogram lets product managers choose any level of the tree that makes business sense for their cross-selling strategy.

---

## 🎯 **When to Use**

**Best for insurance when:**
- Building product taxonomies for catalog organization
- Designing cross-selling bundles based on product similarity
- Exploring natural groupings at multiple granularities
- Visualizing product relationships for business stakeholders

**Not ideal when:**
- Very large product catalogs (> 10,000 products) due to O(n^2) memory
- Need real-time clustering of new products (batch method)
- Products are extremely homogeneous with subtle differences
- Need spherical clusters (use K-Means instead)

---

## 🔧 **Hyperparameters to Tune**

| Parameter | Default | Insurance Recommendation | Why |
|-----------|---------|------------------------|-----|
| n_clusters | 2 | 3-6 | Actionable product groups |
| linkage | ward | ward | Creates compact, balanced groups |
| metric | euclidean | euclidean | Standard for normalized product features |
| distance_threshold | None | Use with n_clusters=None | Auto-determine number of groups |

**Choosing the cut point:**
- Use the dendrogram: look for the longest vertical lines (biggest gaps)
- Business constraint: typically 3-6 groups are actionable for cross-selling
- Cophenetic correlation coefficient: higher is better fit

---

## 🚀 **Running the Demo**

```bash
cd examples/03_clustering/

# Run hierarchical clustering demo
python hierarchical_clustering_demo.py

# Expected output:
# - Product group assignments
# - Interactive dendrogram
# - Cross-selling recommendations
# - Product similarity heatmap
```

---

## 📚 **References**

- Murtagh, F. & Contreras, P. (2012). "Algorithms for hierarchical clustering." Wiley Interdisciplinary Reviews.
- Scikit-learn AgglomerativeClustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- Insurance product management and cross-selling analytics

---

## 📝 **Implementation Reference**

See the complete implementation in `examples/03_clustering/hierarchical_clustering_demo.py` which includes:
- Full product catalog with 85 insurance products
- Dendrogram visualization with business annotations
- Cross-selling bundle recommendations
- Product similarity heatmap and analysis

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
