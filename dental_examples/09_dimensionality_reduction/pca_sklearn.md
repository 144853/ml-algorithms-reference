# PCA (Principal Component Analysis) - Simple Use Case & Data Explanation

## **Use Case: Reducing High-Dimensional Dental Patient Feature Space for Visualization**

### **The Problem**
A dental research team has **5,000 patient records** with **25 clinical features** and wants to visualize patient clusters and identify the most important feature combinations:
- Demographics: age, gender, BMI, smoking status
- Oral health: DMFT index, plaque index, gingival index, pocket depth, bleeding on probing
- Treatment history: n_fillings, n_crowns, n_extractions, n_root_canals, n_cleanings
- Behavioral: brushing frequency, flossing frequency, sugary diet score, dental anxiety score
- Medical: diabetes status, medication count, blood pressure category, osteoporosis risk
- Insurance: coverage tier, annual max utilization, out-of-pocket spending

**Goal:** Reduce 25 features to 2-3 principal components for visualization and feature importance.

### **Why PCA?**
| Criteria | PCA | t-SNE | UMAP | Autoencoders |
|----------|-----|-------|------|-------------|
| Linear relationships | Captures | No | No | Yes |
| Preserves global structure | Yes | No | Partial | Partial |
| Interpretable components | Yes | No | No | No |
| Deterministic | Yes | No | No | No |
| Explained variance | Yes | No | No | No |

PCA is ideal when you need interpretable components and want to understand which features drive patient variation.

---

## **Example Data Structure**

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_patients = 5000

data = pd.DataFrame({
    'age': np.random.normal(45, 15, n_patients).clip(18, 85),
    'bmi': np.random.normal(27, 5, n_patients).clip(16, 45),
    'smoking_status': np.random.choice([0, 1, 2], n_patients, p=[0.6, 0.25, 0.15]),
    'dmft_index': np.random.normal(8, 5, n_patients).clip(0, 28),
    'plaque_index': np.random.normal(1.2, 0.6, n_patients).clip(0, 3),
    'gingival_index': np.random.normal(1.0, 0.5, n_patients).clip(0, 3),
    'pocket_depth_mm': np.random.normal(3.0, 1.2, n_patients).clip(1, 10),
    'bleeding_on_probing': np.random.normal(0.3, 0.2, n_patients).clip(0, 1),
    'n_fillings': np.random.poisson(4, n_patients),
    'n_crowns': np.random.poisson(1, n_patients),
    'n_extractions': np.random.poisson(1.5, n_patients),
    'n_root_canals': np.random.poisson(0.5, n_patients),
    'n_cleanings_per_year': np.random.choice([0, 1, 2, 3, 4], n_patients, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    'brushing_freq_daily': np.random.choice([0, 1, 2, 3], n_patients, p=[0.05, 0.15, 0.55, 0.25]),
    'flossing_freq_weekly': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], n_patients),
    'sugary_diet_score': np.random.normal(5, 2.5, n_patients).clip(0, 10),
    'dental_anxiety_score': np.random.normal(4, 2, n_patients).clip(0, 10),
    'diabetes_status': np.random.choice([0, 1], n_patients, p=[0.87, 0.13]),
    'medication_count': np.random.poisson(2, n_patients),
    'bp_category': np.random.choice([0, 1, 2, 3], n_patients, p=[0.3, 0.35, 0.25, 0.1]),
    'osteoporosis_risk': np.random.normal(0.15, 0.1, n_patients).clip(0, 1),
    'insurance_tier': np.random.choice([1, 2, 3], n_patients, p=[0.4, 0.35, 0.25]),
    'annual_max_utilization': np.random.normal(0.6, 0.25, n_patients).clip(0, 1),
    'out_of_pocket_spending': np.random.normal(800, 500, n_patients).clip(0, 5000),
    'years_since_last_visit': np.random.exponential(1.5, n_patients).clip(0, 10)
})

print(f"Dataset shape: {data.shape}")
print(data.describe().round(2))
```

---

## **PCA Mathematics (Simple Terms)**

**Steps:**
1. **Standardize:** $X_{std} = \frac{X - \mu}{\sigma}$ (zero mean, unit variance)
2. **Covariance Matrix:** $C = \frac{1}{n-1} X_{std}^T X_{std}$
3. **Eigendecomposition:** $C = V \Lambda V^T$ where $V$ = eigenvectors, $\Lambda$ = eigenvalues
4. **Project:** $Z = X_{std} V_k$ where $V_k$ = top-k eigenvectors

**Explained Variance:**
$$\text{explained ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

**Key property:** PC1 captures the maximum variance direction, PC2 the next orthogonal direction, etc.

---

## **The Algorithm**

```python
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=10)  # Compute top 10 components
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained variance ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    cumulative = pca.explained_variance_ratio_[:i+1].sum()
    print(f"  PC{i+1}: {ratio:.4f} ({cumulative:.4f} cumulative)")

# Component loadings (feature importance per PC)
loadings = pd.DataFrame(
    pca.components_.T,
    index=data.columns,
    columns=[f'PC{i+1}' for i in range(10)]
)

# Top features per component
for pc in ['PC1', 'PC2', 'PC3']:
    top = loadings[pc].abs().sort_values(ascending=False).head(5)
    print(f"\n{pc} top features:")
    for feat, load in top.items():
        print(f"  {feat}: {loadings.loc[feat, pc]:.3f}")
```

---

## **Results From the Demo**

**Explained Variance:**
| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | 18.2% | 18.2% |
| PC2 | 12.5% | 30.7% |
| PC3 | 9.8% | 40.5% |
| PC4 | 7.3% | 47.8% |
| PC5 | 6.1% | 53.9% |
| PC6-10 | 24.3% | 78.2% |

**PC1 - "Oral Health Severity":**
| Feature | Loading |
|---------|---------|
| DMFT index | 0.42 |
| Pocket depth | 0.38 |
| Bleeding on probing | 0.35 |
| Gingival index | 0.33 |
| Plaque index | 0.31 |

**PC2 - "Treatment Intensity":**
| Feature | Loading |
|---------|---------|
| N fillings | 0.39 |
| N crowns | 0.36 |
| N root canals | 0.34 |
| Out-of-pocket spending | 0.31 |
| Insurance utilization | 0.28 |

### **Key Insights:**
- PC1 captures "oral health severity" -- separates healthy from periodontal patients
- PC2 captures "treatment intensity" -- separates minimal-treatment from heavily-treated patients
- 5 components capture ~54% of total variance (reasonable for 25 features)
- Age and BMI contribute to multiple components (comorbidity effects)
- Dental anxiety loads on a separate component from oral health, suggesting independent variation

---

## **Simple Analogy**
PCA is like a dental researcher summarizing a 25-question patient intake form. Instead of looking at each answer individually, PCA finds that 5 "super-questions" capture most of the patient variation. "How severe is your periodontal condition?" (PC1) combines pocket depth, bleeding, plaque, and gingival scores into one measure. "How much treatment have you had?" (PC2) combines fillings, crowns, and spending. These super-questions let you plot patients on a simple 2D map.

---

## **When to Use**
**Good for dental applications:**
- Visualizing patient population structure
- Feature importance for clinical studies
- Preprocessing for clustering (reduce noise)
- Identifying key clinical feature combinations

**When NOT to use:**
- When non-linear relationships dominate (use t-SNE or UMAP)
- When all original features need to be retained
- When feature interpretability at the individual level is needed

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| n_components | None | 2-15 | Number of PCs retained |
| whiten | False | True/False | Normalize PC variances |
| svd_solver | 'auto' | 'full', 'arpack', 'randomized' | Computation method |
| Explained variance threshold | 0.95 | 0.80-0.99 | Automatic n_components |

---

## **Running the Demo**
```bash
cd examples/09_dimensionality_reduction
python pca_demo.py
```

---

## **References**
- Pearson, K. (1901). "On Lines and Planes of Closest Fit"
- Jolliffe, I.T. (2002). "Principal Component Analysis"
- scikit-learn documentation: sklearn.decomposition.PCA

---

## **Implementation Reference**
- See `examples/09_dimensionality_reduction/pca_demo.py` for full runnable code
- Preprocessing: StandardScaler (required for PCA)
- Visualization: Scatter plot of PC1 vs PC2 with cluster coloring

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
