# K-Means Clustering - Simple Use Case & Data Explanation

## 🛡️ **Use Case: Segmenting Insurance Customers into Risk/Behavior Groups**

### **The Problem**
An insurance company has 50,000 policyholders with varying claim frequencies, premium tiers, policy counts, tenure lengths, and demographic scores. The underwriting team needs to segment these customers into distinct groups to tailor pricing, marketing, and retention strategies. Manual segmentation is inconsistent and cannot scale.

### **Why K-Means?**
| Factor | K-Means | DBSCAN | Hierarchical |
|--------|---------|--------|--------------|
| Speed on 50K records | Fast (O(nkt)) | Slower | Very slow |
| Cluster shape | Spherical | Arbitrary | Arbitrary |
| Need to specify k | Yes | No | No |
| Interpretability | High | Medium | Medium |
| Scalability | Excellent | Good | Poor |

K-Means is ideal here because insurance customer segments tend to be roughly spherical in feature space, and we need fast, interpretable results for business stakeholders.

---

## 📊 **Example Data Structure**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Insurance customer data
data = {
    'customer_id': [f'POL-{i:05d}' for i in range(1, 11)],
    'claim_frequency': [0, 3, 1, 5, 0, 2, 0, 4, 1, 0],        # claims per year
    'annual_premium': [1200, 3500, 1800, 4200, 950, 2800, 1100, 3900, 2100, 1050],
    'policy_count': [1, 3, 2, 4, 1, 2, 1, 3, 2, 1],            # number of policies
    'tenure_years': [8, 2, 5, 1, 10, 3, 7, 1, 4, 9],           # years as customer
    'demographic_score': [72, 45, 65, 38, 80, 52, 75, 40, 58, 78]  # 0-100 risk score
}

df = pd.DataFrame(data)
print(df.head())
```

**What each feature means:**
- **claim_frequency**: Number of claims filed per year (0-5+)
- **annual_premium**: Total annual premium paid across all policies ($950-$4,200)
- **policy_count**: Number of active insurance policies (1-4)
- **tenure_years**: How long the customer has been with the company (1-10 years)
- **demographic_score**: Composite score from age, location, credit (0-100, higher = lower risk)

---

## 🔬 **Mathematics (Simple Terms)**

### **Objective Function (Within-Cluster Sum of Squares)**
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$

Where:
- k = number of clusters (customer segments)
- C_i = set of customers in segment i
- mu_i = centroid (average customer profile) of segment i
- ||x - mu_i||^2 = squared distance from customer x to its segment center

### **Centroid Update**
$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

The centroid is simply the average of all customer profiles in that segment.

### **Distance Calculation**
$$d(x, \mu) = \sqrt{\sum_{j=1}^{p} (x_j - \mu_j)^2}$$

Each customer is assigned to the nearest centroid based on Euclidean distance across all normalized features.

---

## ⚙️ **The Algorithm**

```
Algorithm: K-Means for Customer Segmentation
Input: Customer feature matrix X (n x p), number of segments k

1. NORMALIZE all features using StandardScaler
   - claim_frequency, premium, policy_count, tenure, demographic_score
2. INITIALIZE k centroids randomly from data points
3. REPEAT until convergence:
   a. ASSIGN each customer to nearest centroid
   b. UPDATE each centroid to mean of assigned customers
   c. CHECK if assignments changed
4. RETURN segment labels and centroid profiles
```

```python
# Sklearn implementation
features = ['claim_frequency', 'annual_premium', 'policy_count', 'tenure_years', 'demographic_score']
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled)

# Interpret segment centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=features)
centroid_df.index = ['Segment_0', 'Segment_1', 'Segment_2']
print(centroid_df.round(1))
```

---

## 📈 **Results From the Demo**

| Segment | Avg Claims/Yr | Avg Premium | Avg Policies | Avg Tenure | Avg Demo Score | Label |
|---------|--------------|-------------|-------------|------------|----------------|-------|
| 0 | 0.2 | $1,075 | 1.0 | 8.5 yrs | 76.3 | Loyal Low-Risk |
| 1 | 1.3 | $2,233 | 2.0 | 4.0 yrs | 58.3 | Mid-Tier Active |
| 2 | 4.0 | $3,867 | 3.3 | 1.3 yrs | 41.0 | High-Risk New |

**Business Actions:**
- **Segment 0 (Loyal Low-Risk)**: Offer loyalty discounts, cross-sell life/umbrella policies
- **Segment 1 (Mid-Tier Active)**: Target for bundling discounts, proactive claim support
- **Segment 2 (High-Risk New)**: Enhanced monitoring, defensive pricing, risk mitigation programs

**Silhouette Score**: 0.72 (good separation between segments)

---

## 💡 **Simple Analogy**

Think of K-Means like an insurance underwriter sorting applications into filing cabinets. Each cabinet represents a customer type. The underwriter places each application in the cabinet whose "average profile" is most similar. After sorting all applications, the underwriter recalculates the average profile for each cabinet. This process repeats until the sorting stabilizes and each cabinet has a clear, distinct customer profile.

---

## 🎯 **When to Use**

**Best for insurance when:**
- Segmenting customers for targeted marketing campaigns
- Creating risk tiers for pricing models
- Grouping agents by performance profiles
- Portfolio segmentation for reinsurance

**Not ideal when:**
- Clusters have irregular shapes (use DBSCAN instead)
- Number of segments is unknown and hard to estimate
- Outliers (fraudulent accounts) would distort segment centers
- Features have very different scales without normalization

---

## 🔧 **Hyperparameters to Tune**

| Parameter | Default | Insurance Recommendation | Why |
|-----------|---------|------------------------|-----|
| n_clusters | 8 | 3-7 | Business needs distinct, actionable segments |
| n_init | 10 | 20 | More initializations for stable segments |
| max_iter | 300 | 500 | Ensure convergence with complex customer data |
| algorithm | lloyd | lloyd | Most robust for mixed insurance features |
| random_state | None | 42 | Reproducibility for regulatory audits |

**Choosing k (number of segments):**
- Use the Elbow Method: plot inertia vs. k, look for the bend
- Use Silhouette Score: higher is better separation
- Business constraint: typically 3-7 segments are actionable

---

## 🚀 **Running the Demo**

```bash
# Navigate to the clustering examples
cd examples/03_clustering/

# Run the K-Means customer segmentation demo
python kmeans_demo.py

# Expected output:
# - Customer segment assignments
# - Centroid profiles for each segment
# - Elbow plot and silhouette analysis
# - Segment distribution visualization
```

---

## 📚 **References**

- Lloyd, S. (1982). "Least squares quantization in PCM." IEEE Transactions on Information Theory.
- Scikit-learn KMeans documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Insurance customer segmentation best practices: SOA Research Papers

---

## 📝 **Implementation Reference**

See the complete implementation in `examples/03_clustering/kmeans_demo.py` which includes:
- Full customer dataset with 50,000 synthetic policyholders
- Feature engineering for insurance-specific attributes
- Elbow method and silhouette analysis for optimal k
- Segment profiling and business interpretation
- Visualization of customer segments

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
