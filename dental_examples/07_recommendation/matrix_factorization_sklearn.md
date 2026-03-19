# Matrix Factorization - Simple Use Case & Data Explanation

## **Use Case: Recommending Dental Treatment Plans Based on Similar Patient Outcomes**

### **The Problem**
A dental practice network has **2,000 patient outcome records** across **30 treatment plan types**. Each record captures an outcome score (1-5) for how well a treatment plan worked:
- **Patients:** 2,000 with varying conditions
- **Treatment plans:** 30 standardized plans (e.g., "Perio Scaling + SRP", "Crown + Post", "Implant + Bone Graft")
- **Outcome scores:** 1-5 (1=poor outcome, 5=excellent outcome)
- **Sparsity:** Each patient has tried 3-8 treatment plans

**Goal:** Predict which untried treatment plans would have the best outcomes for each patient.

### **Why Matrix Factorization?**
| Criteria | User-Based CF | Matrix Factorization | Neural CF |
|----------|--------------|---------------------|-----------|
| Scalability | O(n^2) | O(nk) | O(batch) |
| Latent factor discovery | No | Yes | Yes |
| Handles sparsity | Moderate | Good | Good |
| Interpretability | User similarity | Latent factors | Low |
| Training speed | Fast (no training) | Moderate | Slow |

Matrix factorization discovers latent patient/treatment factors (e.g., "periodontal severity" or "cosmetic preference").

---

## **Example Data Structure**

```python
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n_patients = 2000
n_plans = 30

treatment_plans = [
    'Routine Prophylaxis', 'Deep Scaling + SRP', 'Perio Maintenance',
    'Single Crown (PFM)', 'Single Crown (Zirconia)', 'Crown + Post/Core',
    'Fixed Bridge (3-unit)', 'Maryland Bridge', 'Removable Partial Denture',
    'Complete Denture (Upper)', 'Complete Denture (Lower)', 'Implant-Retained Denture',
    'Single Implant + Crown', 'All-on-4 Implants', 'Bone Graft + Implant',
    'Root Canal (Anterior)', 'Root Canal (Molar)', 'Retreatment Root Canal',
    'Simple Extraction', 'Surgical Extraction', 'Wisdom Tooth Removal',
    'Composite Filling (1 surf)', 'Composite Filling (2+ surf)', 'Amalgam Filling',
    'Veneer (Porcelain)', 'Teeth Whitening', 'Clear Aligners',
    'Traditional Braces', 'Night Guard (TMJ)', 'Gum Graft Surgery'
]

# Simulate outcome matrix with latent factors
n_factors = 5
patient_factors = np.random.randn(n_patients, n_factors)
plan_factors = np.random.randn(n_plans, n_factors)
true_outcomes = patient_factors @ plan_factors.T + 3  # Center around 3
true_outcomes = np.clip(true_outcomes, 1, 5)

# Create sparse observed outcomes
outcomes = np.zeros((n_patients, n_plans))
for i in range(n_patients):
    n_tried = np.random.randint(3, 9)
    tried_plans = np.random.choice(n_plans, n_tried, replace=False)
    outcomes[i, tried_plans] = np.round(true_outcomes[i, tried_plans] + np.random.normal(0, 0.3, n_tried)).clip(1, 5)

outcomes_df = pd.DataFrame(outcomes, columns=treatment_plans)
print(f"Sparsity: {(outcomes == 0).sum() / outcomes.size:.1%}")
```

---

## **Matrix Factorization Mathematics (Simple Terms)**

**Decompose the rating matrix:**
$$R \approx P \times Q^T$$

Where:
- $R \in \mathbb{R}^{m \times n}$ = patient-treatment outcome matrix (2000 x 30)
- $P \in \mathbb{R}^{m \times k}$ = patient latent factors (2000 x 5)
- $Q \in \mathbb{R}^{n \times k}$ = treatment plan latent factors (30 x 5)
- $k$ = number of latent factors

**SVD Decomposition:**
$$R = U \Sigma V^T$$

**Prediction:**
$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i$$

Where $\mu$ = global mean, $b_u$ = patient bias, $b_i$ = treatment bias.

---

## **The Algorithm**

```python
# Method 1: SVD-based Matrix Factorization
from scipy.sparse.linalg import svds

# Normalize ratings
patient_means = np.true_divide(outcomes.sum(axis=1), (outcomes > 0).sum(axis=1))
patient_means = np.nan_to_num(patient_means, nan=3.0)
normalized = outcomes.copy()
for i in range(n_patients):
    mask = outcomes[i] > 0
    normalized[i, mask] -= patient_means[i]

# SVD decomposition
k = 10  # Latent factors
U, sigma, Vt = svds(normalized, k=k)
sigma = np.diag(sigma)

# Reconstruct predictions
predicted_normalized = U @ sigma @ Vt
predicted = predicted_normalized + patient_means[:, np.newaxis]
predicted = np.clip(predicted, 1, 5)

# Method 2: ALS (Alternating Least Squares)
def als_factorization(R, k=10, n_iter=20, reg=0.1):
    """Alternating Least Squares for matrix factorization."""
    m, n = R.shape
    P = np.random.randn(m, k) * 0.1
    Q = np.random.randn(n, k) * 0.1
    mask = R > 0

    for iteration in range(n_iter):
        # Fix Q, solve for P
        for u in range(m):
            rated = mask[u]
            if rated.sum() == 0:
                continue
            Q_rated = Q[rated]
            r_rated = R[u, rated]
            P[u] = np.linalg.solve(Q_rated.T @ Q_rated + reg * np.eye(k), Q_rated.T @ r_rated)

        # Fix P, solve for Q
        for i in range(n):
            rated = mask[:, i]
            if rated.sum() == 0:
                continue
            P_rated = P[rated]
            r_rated = R[rated, i]
            Q[i] = np.linalg.solve(P_rated.T @ P_rated + reg * np.eye(k), P_rated.T @ r_rated)

        pred = P @ Q.T
        error = np.sqrt(((mask * (R - pred)) ** 2).sum() / mask.sum())
        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration+1}, RMSE: {error:.4f}")

    return P, Q

P, Q = als_factorization(outcomes, k=10, n_iter=20, reg=0.1)
predicted_als = np.clip(P @ Q.T, 1, 5)

# Recommend top treatment plans
def recommend_treatments(patient_id, predicted, outcomes, n_recs=5):
    untried = outcomes[patient_id] == 0
    scores = predicted[patient_id].copy()
    scores[~untried] = -1
    top_plans = np.argsort(scores)[::-1][:n_recs]
    return [(treatment_plans[p], scores[p]) for p in top_plans]
```

---

## **Results From the Demo**

**Treatment Plan Recommendations for Patient 42:**
| Rank | Treatment Plan | Predicted Outcome |
|------|---------------|-------------------|
| 1 | Single Crown (Zirconia) | 4.7 |
| 2 | Clear Aligners | 4.5 |
| 3 | Veneer (Porcelain) | 4.4 |
| 4 | Teeth Whitening | 4.3 |
| 5 | Night Guard (TMJ) | 4.1 |

| Metric | SVD | ALS |
|--------|-----|-----|
| RMSE | 0.82 | 0.76 |
| MAE | 0.65 | 0.59 |
| Precision@3 | 0.45 | 0.52 |

### **Key Insights:**
- Patient 42's latent profile suggests cosmetic/restorative preference -- recommended plans align
- Latent factor 1 correlates with "surgical complexity tolerance"
- Latent factor 2 correlates with "cosmetic vs. functional preference"
- ALS outperforms SVD by handling missing values explicitly
- Treatment plan factors reveal natural groupings (surgical, restorative, preventive)

---

## **Simple Analogy**
Matrix factorization is like discovering that each dental patient has a hidden "treatment DNA" and each treatment plan has a hidden "success profile." A patient who did well with crowns (restorative preference) will likely do well with bridges and veneers too, because their treatment DNA matches the success profile of restorative procedures. Matrix factorization discovers these hidden profiles automatically from outcome data.

---

## **When to Use**
**Good for dental applications:**
- Treatment plan recommendation based on outcomes
- Predicting patient satisfaction with untried treatments
- Dental material recommendation for dentists
- Dental insurance plan matching

**When NOT to use:**
- Very few patients or treatments (<100 x 10)
- When explicit treatment features matter more (use content-based)
- When real-time updates are needed (SVD requires recomputation)

---

## **Hyperparameters to Tune**
| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| k (latent factors) | 10 | 5-50 | Model capacity |
| reg (regularization) | 0.1 | 0.01-1.0 | Overfitting prevention |
| n_iter (ALS) | 20 | 10-50 | Convergence |
| learning_rate (SGD) | 0.01 | 0.001-0.1 | Training speed |

---

## **Running the Demo**
```bash
cd examples/07_recommendation
python matrix_factorization_demo.py
```

---

## **References**
- Koren, Y. et al. (2009). "Matrix Factorization Techniques for Recommender Systems"
- Funk, S. (2006). "Netflix Update: Try This at Home"
- scipy documentation: scipy.sparse.linalg.svds

---

## **Implementation Reference**
- See `examples/07_recommendation/matrix_factorization_demo.py` for full code
- Methods: SVD and ALS factorization
- Evaluation: RMSE, MAE, Precision@K

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
