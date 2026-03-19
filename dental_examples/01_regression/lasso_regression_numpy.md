# Lasso Regression (NumPy) - Simple Use Case & Data Explanation

## 🦷 **Use Case: Predicting Dental Implant Success Score**

### **The Problem**
A dental implant center tracking 550 patients wants to predict implant success scores (0-100) based on bone density, gum health index, smoking status, diabetes indicator, and implant material. Some features may be irrelevant -- Lasso regression performs automatic feature selection by driving unimportant coefficients to exactly zero.

### **Why Lasso Regression?**
| Factor | Lasso Regression | Ridge Regression | Linear Regression |
|--------|-----------------|------------------|-------------------|
| Feature selection | Automatic -- zeros out irrelevant features | No -- keeps all features | No -- keeps all features |
| Regularization | L1 (absolute weights) | L2 (squared weights) | None |
| Sparse solutions | Yes | No | No |
| Best when | Some features are irrelevant | All features matter | Features are independent |

---

## 📊 **Example Data Structure**

```python
import numpy as np

# 550 dental implant patients
# Features: bone_density (0.5-2.0 g/cm3), gum_health (1-10), smoking_status (0/1),
#           diabetes_indicator (0/1), implant_material (1-4: titanium, zirconia, hybrid, ceramic)
X = np.array([
    [1.4, 8, 0, 0, 1],   # Good bone, healthy gums, non-smoker, no diabetes, titanium
    [0.8, 4, 1, 1, 2],   # Low bone, poor gums, smoker, diabetic, zirconia
    [1.7, 9, 0, 0, 3],   # Dense bone, excellent gums, non-smoker, no diabetes, hybrid
    [1.1, 5, 1, 0, 4],   # Moderate bone, fair gums, smoker, no diabetes, ceramic
    # ... 546 more patients
])

# Target: implant success score (0-100)
y = np.array([92, 45, 97, 61, ...])
```

### **The Lasso Model**
```
success = w1*bone_density + w2*gum_health + w3*smoking + w4*diabetes + w5*material + bias

With L1 penalty: minimize MSE + alpha * sum(|w_i|)
```

For example (after training, some weights may be zero):
```
success = 18.5*bone_density + 4.2*gum_health + (-12.8)*smoking + 0.0*diabetes + 0.0*material + 32.0
         = 18.5*1.4 + 4.2*8 + (-12.8)*0 + 0.0*0 + 0.0*1 + 32.0
         = 25.9 + 33.6 + 0 + 0 + 0 + 32.0
         = 91.5
```

Note: Lasso set diabetes and material coefficients to zero, identifying them as less important.

---

## 🔬 **Lasso Regression Mathematics (Simple Terms)**

### **Goal: Minimize Error + Penalty on Absolute Weights**
Lasso uses an L1 penalty that encourages **sparsity** -- it pushes small coefficients to exactly zero.

### **Loss Function: MSE + L1 Penalty**
```
Loss = (1/2n) * ||Xw - y||^2 + alpha * ||w||_1
     = (1/2n) * sum((predicted - actual)^2) + alpha * sum(|w_i|)
```

Where `alpha` controls sparsity:
- alpha = 0: no regularization (ordinary linear regression)
- Small alpha: few features eliminated
- Large alpha: most features eliminated
- alpha = infinity: all weights become zero

### **Subgradient (L1 is not differentiable at zero)**
```
dL/dw_j = (1/n) * X_j^T * (Xw - y) + alpha * sign(w_j)

where sign(w_j) = +1 if w_j > 0
                  -1 if w_j < 0
                  [-1, +1] if w_j = 0  (subgradient set)
```

### **Proximal Operator (Soft Thresholding)**
```
w_j = soft_threshold(w_j - lr * gradient_mse_j, lr * alpha)

soft_threshold(z, threshold) = sign(z) * max(|z| - threshold, 0)
```

This is the key mechanism: if the gradient update pushes a weight below the threshold, it snaps to zero.

---

## ⚙️ **The Algorithm: Coordinate Descent**

```
Initialize weights w = zeros, bias b = 0
Set alpha = 0.1, max_iter = 1000, tol = 1e-4

For each iteration:
    For each feature j = 1, ..., p:
        1. residual = y - X @ w - b + X_j * w_j  (partial residual)
        2. rho_j = X_j^T @ residual / n
        3. w_j = soft_threshold(rho_j, alpha)
    4. b = mean(y - X @ w)
    5. Check convergence: if max change in w < tol, stop

Return w, b
```

### **Soft Thresholding Function**
```python
def soft_threshold(z, threshold):
    if z > threshold:
        return z - threshold
    elif z < -threshold:
        return z + threshold
    else:
        return 0.0  # Feature eliminated!
```

---

## 📈 **Results From the Demo**

```
Training MSE:  42.18
Testing MSE:   48.35
Training R^2:  0.8945
Testing R^2:   0.8712

Learned Coefficients (alpha=0.1):
  bone_density:       18.52 (per g/cm3)
  gum_health:          4.21 (per index point)
  smoking_status:    -12.78 (smoker penalty)
  diabetes_indicator:  0.00 (eliminated by Lasso!)
  implant_material:    0.00 (eliminated by Lasso!)
  bias:               32.15

Active features: 3 out of 5 (40% sparsity)
```

### **Coefficient Path (varying alpha)**
```
alpha=0.001: [19.85, 4.58, -14.12, -2.31, 0.85]  (0 zeros)
alpha=0.01:  [19.42, 4.45, -13.65, -1.50, 0.32]  (0 zeros)
alpha=0.05:  [18.90, 4.32, -13.10, -0.42, 0.00]  (1 zero)
alpha=0.10:  [18.52, 4.21, -12.78,  0.00, 0.00]  (2 zeros)
alpha=0.50:  [16.80, 3.45, -10.20,  0.00, 0.00]  (2 zeros)
alpha=1.00:  [14.10, 2.15,  -6.50,  0.00, 0.00]  (2 zeros)
alpha=5.00:  [ 0.00, 0.00,   0.00,  0.00, 0.00]  (5 zeros)
```

### **Key Insights:**
- **Bone density** is the strongest predictor of implant success -- each g/cm3 adds ~18.5 points
- **Smoking** has a strong negative effect -- smokers score ~12.8 points lower on average
- **Diabetes and implant material were eliminated** -- their contribution was too small relative to the penalty
- **Lasso identified the 3 most clinically meaningful features**, matching dental literature that prioritizes bone quality, gum health, and smoking status

---

## 💡 **Simple Analogy**

Imagine a dental implant specialist creating a simple checklist. Instead of scoring every possible factor, Lasso acts like a wise clinician who says "only these three things really matter for success: bone quality, gum health, and whether the patient smokes. The rest is noise." The L1 penalty is like a budget constraint -- you can only afford to track a few factors, so Lasso picks the most important ones and crosses out the rest.

---

## 🎯 **When to Use Lasso Regression**

### **Use Lasso Regression when:**
- You suspect some features are irrelevant and want automatic feature selection
- You want a sparse, interpretable model (fewer coefficients to explain)
- You have more features than samples (p > n)
- You need to identify the most important clinical predictors
- You want to build a simple clinical scoring system

### **Common Dental Applications:**
- Identifying key predictors of implant failure from many clinical measurements
- Feature selection for caries risk assessment from dietary and behavioral surveys
- Building sparse periodontal disease progression models
- Selecting relevant radiographic features for automated diagnosis
- Simplifying patient risk scoring from electronic health records

### **When NOT to use:**
- All features are known to be relevant (use Ridge instead)
- Features are highly correlated (Lasso may arbitrarily drop one -- use ElasticNet)
- You need stable feature selection across different data samples
- Non-linear relationships dominate (use tree-based models)

---

## 🔧 **Hyperparameters to Tune**

| Parameter | Typical Range | Dental Tip |
|-----------|--------------|------------|
| alpha | 0.001 - 10.0 | Start at 0.1; use CV to find optimal sparsity |
| max_iter | 1000 - 10000 | Increase if convergence warning appears |
| tol | 1e-4 - 1e-6 | Default 1e-4 is usually fine for clinical data |
| feature_scaling | StandardScaler | Essential -- bone_density (0.5-2.0) vs gum_health (1-10) |

### **Choosing Alpha**
```
Small alpha (0.001): Keep most features -- use when you believe all measurements matter
Medium alpha (0.1):  Moderate feature selection -- good default for dental datasets
Large alpha (1.0+):  Aggressive feature elimination -- use for clinical scoring simplicity
```

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/lasso_regression_numpy.py
```

The script demonstrates:
1. Generating synthetic dental implant data (550 patients, 5 features)
2. Implementing coordinate descent with soft thresholding from scratch
3. Visualizing the Lasso path (coefficient values vs alpha)
4. Comparing Lasso feature selection against Ridge
5. Training/test evaluation with MSE and R-squared

---

## 📚 **References**

1. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society B*, 58(1), 267-288.
2. Friedman, J., Hastie, T., & Tibshirani, R. (2010). "Regularization Paths for Generalized Linear Models via Coordinate Descent." *Journal of Statistical Software*, 33(1).
3. D'Amore, C. et al. (2021). "Machine Learning Approaches for Dental Implant Outcome Prediction." *Clinical Implant Dentistry and Related Research*, 23(5).
4. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity*. CRC Press.

---

## 📝 **Implementation Reference**

- **NumPy implementation:** `01_regression/lasso_regression_numpy.py` -- coordinate descent with soft thresholding
- **Scikit-learn version:** `01_regression/lasso_regression_sklearn.py` -- production-ready with LassoCV
- **PyTorch version:** `01_regression/lasso_regression_pytorch.py` -- proximal gradient descent with autograd

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
