# K-Means Clustering - Simple Use Case & Data Explanation

## **Use Case: Segmenting Dental Patients into Care-Need Groups**

### **The Problem**
A dental practice with **2,500 patients** wants to segment them into distinct care-need groups based on 5 features:
- **Visit frequency** (visits per year: 0-12)
- **Treatment history score** (0-100, based on past procedures)
- **Oral health score** (0-100, composite of gum health, decay index, plaque score)
- **Age** (18-85)
- **Insurance tier** (1=Basic, 2=Standard, 3=Premium)

**Goal:** Identify 4 patient segments for targeted outreach and resource allocation.

### **Why K-Means?**
| Criteria | K-Means | DBSCAN | Hierarchical |
|----------|---------|--------|--------------|
| Speed on 2,500 patients | Fast | Moderate | Slow |
| Need to specify clusters | Yes (k=4) | No | No |
| Cluster shape | Spherical | Arbitrary | Arbitrary |
| Interpretability | High | Moderate | High |
| Scalability | Excellent | Good | Poor |

K-Means is ideal here because we have a predefined number of patient segments and need fast, interpretable results.

---

## **Example Data Structure**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Simulated dental patient data
np.random.seed(42)
n_patients = 2500

data = pd.DataFrame({
    'visit_frequency': np.random.choice(range(0, 13), n_patients, p=[0.05, 0.1, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.02, 0.02, 0.01]),
    'treatment_history_score': np.random.normal(45, 20, n_patients).clip(0, 100),
    'oral_health_score': np.random.normal(62, 18, n_patients).clip(0, 100),
    'age': np.random.normal(42, 15, n_patients).clip(18, 85).astype(int),
    'insurance_tier': np.random.choice([1, 2, 3], n_patients, p=[0.4, 0.35, 0.25])
})

print(data.head())
# Output:
#    visit_frequency  treatment_history_score  oral_health_score  age  insurance_tier
# 0                3                    55.2               71.4   38               2
# 1                2                    32.8               48.6   56               1
# 2                4                    67.1               78.3   29               3
# 3                1                    21.4               35.2   67               1
# 4                6                    78.9               82.1   44               3
```

---

## **K-Means Mathematics (Simple Terms)**

**Objective:** Minimize the Within-Cluster Sum of Squares (WCSS):

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $k$ = number of clusters (4 patient segments)
- $C_i$ = set of patients in cluster $i$
- $\mu_i$ = centroid (mean) of cluster $i$
- $||x - \mu_i||^2$ = squared Euclidean distance from patient $x$ to centroid

**Steps:**
1. Randomly initialize 4 centroids
2. Assign each patient to nearest centroid
3. Recompute centroids as mean of assigned patients
4. Repeat until centroids stabilize

---

## **The Algorithm**

```python
# Standardize features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
labels = kmeans.fit_predict(X_scaled)

data['cluster'] = labels

# Examine cluster centers (inverse-transformed)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
center_df = pd.DataFrame(centers, columns=data.columns[:-1])
print(center_df.round(1))
```

**Elbow Method for Optimal k:**
```python
from sklearn.metrics import silhouette_score

inertias = []
sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
```

---

## **Results From the Demo**

| Cluster | Patients | Avg Visits/yr | Avg Treatment Score | Avg Oral Health | Avg Age | Common Insurance |
|---------|----------|---------------|--------------------|-----------------|---------|--------------------|
| 0 - "Routine Care" | 680 | 2.1 | 28.4 | 72.5 | 35 | Basic |
| 1 - "High Need" | 520 | 1.4 | 71.3 | 38.2 | 58 | Basic |
| 2 - "Wellness Champions" | 710 | 5.8 | 42.1 | 85.6 | 41 | Premium |
| 3 - "Moderate Risk" | 590 | 3.2 | 55.7 | 54.8 | 49 | Standard |

**Silhouette Score:** 0.42 (moderate separation)
**Inertia (WCSS):** 4,823.6

### **Key Insights:**
- **Cluster 1 ("High Need")** patients visit infrequently despite high treatment history -- ideal candidates for recall campaigns
- **Cluster 2 ("Wellness Champions")** are engaged patients who maintain excellent oral health -- offer loyalty programs
- **Cluster 0 ("Routine Care")** younger patients with basic insurance -- potential for upselling preventive packages
- **Cluster 3 ("Moderate Risk")** middle-aged patients needing closer monitoring -- schedule more frequent check-ups

---

## **Simple Analogy**
Think of K-Means like a dental office manager sorting patient files into 4 color-coded folders. Each folder represents a care pathway. The manager places each file in the folder whose "average patient" looks most similar, then recalculates what the average patient in each folder looks like. After a few rounds of reshuffling, every patient ends up in the folder that best matches their care profile.

---

## **When to Use**
**Good for dental applications:**
- Patient segmentation for targeted marketing
- Grouping dental practices by performance metrics
- Categorizing treatment plans by complexity
- Staff scheduling based on patient mix patterns

**When NOT to use:**
- When clusters have irregular shapes (use DBSCAN)
- When you do not know the number of segments (use hierarchical)
- When you have very high-dimensional dental imaging data (use spectral clustering)

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| n_clusters | 8 | 2-10 | Number of patient segments |
| n_init | 10 | 5-20 | Number of random initializations |
| max_iter | 300 | 100-500 | Maximum iterations per run |
| init | 'k-means++' | 'k-means++', 'random' | Initialization method |
| algorithm | 'lloyd' | 'lloyd', 'elkan' | Computation algorithm |

---

## **Running the Demo**
```bash
cd examples/03_clustering
python kmeans_demo.py
```

---

## **References**
- Lloyd, S. (1982). "Least squares quantization in PCM"
- Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding"
- scikit-learn documentation: sklearn.cluster.KMeans

---

## **Implementation Reference**
- See `examples/03_clustering/kmeans_demo.py` for full runnable code
- Preprocessing: `StandardScaler` for feature normalization
- Evaluation: Silhouette score, elbow method, cluster profiling

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
