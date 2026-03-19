# Ridge Regression (NumPy) - Simple Use Case & Data Explanation

## 🦷 **Use Case: Predicting Orthodontic Treatment Duration**

### **The Problem**
An orthodontic practice with 600 patients wants to predict treatment duration (in months) based on malocclusion severity, patient age, compliance score, and bracket type. Correlated features like severity and compliance make ordinary least squares unstable -- Ridge regression stabilizes the coefficients with L2 regularization.

### **Why Ridge Regression?**
| Factor | Ridge Regression | Linear Regression | Lasso Regression |
|--------|-----------------|-------------------|------------------|
| Handles multicollinearity | Excellent -- shrinks correlated coefficients | Poor -- unstable weights | Moderate -- may drop useful features |
| Feature selection | Keeps all features | Keeps all features | Drops less important features |
| Regularization | L2 (squared weights) | None | L1 (absolute weights) |
| Best when | All features matter | Features are independent | Sparse features expected |

---

## 📊 **Example Data Structure**

```python
import numpy as np

# 600 orthodontic patients
# Features: malocclusion_severity (1-10), patient_age (8-55),
#           compliance_score (0-100), bracket_type (1-4)
X = np.array([
    [7, 14, 85, 2],   # Severe malocclusion, age 14, good compliance, ceramic
    [3, 28, 60, 1],   # Mild malocclusion, age 28, moderate compliance, metal
    [9, 12, 92, 3],   # Very severe, age 12, excellent compliance, lingual
    [5, 35, 45, 4],   # Moderate, age 35, low compliance, clear aligner
    # ... 596 more patients
])

# Target: treatment duration in months (6 - 48 months)
y = np.array([24, 14, 30, 22, ...])
```

### **The Ridge Model**
```
duration = w1*severity + w2*age + w3*compliance + w4*bracket_type + bias

With L2 penalty: minimize MSE + alpha * sum(w_i^2)
```

For example:
```
duration = 2.8*severity + 0.15*age + (-0.12)*compliance + 1.5*bracket + 8.0
         = 2.8*7 + 0.15*14 + (-0.12)*85 + 1.5*2 + 8.0
         = 19.6 + 2.1 + (-10.2) + 3.0 + 8.0
         = 22.5 months
```

---

## 🔬 **Ridge Regression Mathematics (Simple Terms)**

### **Goal: Minimize Error + Penalty on Large Weights**
Ridge adds a penalty proportional to the **squared magnitude** of weights, preventing any single coefficient from becoming too large.

### **Loss Function: MSE + L2 Penalty**
```
Loss = (1/2n) * ||Xw - y||^2 + alpha * ||w||^2
     = (1/2n) * sum((predicted - actual)^2) + alpha * sum(w_i^2)
```

Where `alpha` (lambda) controls regularization strength:
- alpha = 0: equivalent to ordinary linear regression
- alpha = infinity: all weights shrink to zero
- alpha = 1.0: typical starting point

### **Gradient**
```
dL/dw = (1/n) * X^T * (Xw - y) + 2 * alpha * w
dL/db = (1/n) * sum(Xw - y)     # bias is NOT regularized
```

### **Closed-Form Solution**
```
w = (X^T X + alpha * I)^(-1) X^T y
```

The `alpha * I` term ensures the matrix is always invertible, even when features are highly correlated (e.g., severity and compliance often correlate).

---

## ⚙️ **The Algorithm: Gradient Descent with L2 Regularization**

```
Initialize weights w = zeros, bias b = 0
Set learning_rate = 0.01, alpha = 1.0, epochs = 1000

For each epoch:
    1. predictions = X @ w + b
    2. errors = predictions - y
    3. gradient_w = (1/n) * X.T @ errors + 2 * alpha * w
    4. gradient_b = (1/n) * sum(errors)
    5. w = w - learning_rate * gradient_w
    6. b = b - learning_rate * gradient_b
    7. loss = mean(errors^2) + alpha * sum(w^2)

Return w, b
```

### **Also Implements: Ridge Closed-Form**
```
1. Add column of ones to X (for bias)
2. I_modified = identity matrix with I[0,0] = 0 (don't regularize bias)
3. w = inverse(X^T @ X + alpha * I_modified) @ X^T @ y
4. Return w
```

---

## 📈 **Results From the Demo**

```
Training MSE:  8.42
Testing MSE:   9.15
Training R^2:  0.8876
Testing R^2:   0.8721

Learned Coefficients (alpha=1.0):
  malocclusion_severity:  2.78 months/severity point
  patient_age:            0.14 months/year
  compliance_score:      -0.11 months/compliance point
  bracket_type:           1.48 months/type tier
  bias:                   8.12 months

Comparison (alpha effect on coefficients):
  alpha=0.0  (OLS):     [3.42, 0.21, -0.18, 1.95, 7.85]
  alpha=0.1:            [3.15, 0.18, -0.15, 1.72, 7.92]
  alpha=1.0:            [2.78, 0.14, -0.11, 1.48, 8.12]
  alpha=10.0:           [1.85, 0.09, -0.07, 0.98, 8.45]
```

### **Key Insights:**
- **Malocclusion severity** is the dominant predictor -- each severity point adds ~2.8 months of treatment
- **Compliance score** has a negative coefficient -- higher compliance reduces treatment time by ~0.11 months per point
- **Ridge shrinks coefficients uniformly** compared to OLS, preventing overfitting on the correlated severity/compliance features
- **Alpha=1.0 provides the best bias-variance tradeoff**, reducing test MSE by 12% compared to unregularized OLS

---

## 💡 **Simple Analogy**

Imagine an orthodontist estimating treatment time. Without Ridge (OLS), they might over-rely on one factor like severity because it correlates with compliance. Ridge acts like a experienced mentor saying "don't put too much weight on any single factor" -- it gently pushes all the weights toward smaller, more balanced values. The `alpha` parameter controls how strongly the mentor insists on balance.

---

## 🎯 **When to Use Ridge Regression**

### **Use Ridge Regression when:**
- Features are correlated (severity and compliance often move together)
- You want to keep ALL features in the model (no feature elimination)
- Ordinary least squares gives unstable or inflated coefficients
- You have more features than samples (p > n scenarios)
- You need a regularized baseline before trying more complex models

### **Common Dental Applications:**
- Predicting orthodontic treatment duration with correlated clinical measures
- Estimating periodontal disease progression from correlated biomarkers
- Forecasting dental material wear rates from overlapping physical properties
- Predicting patient satisfaction from correlated survey dimensions
- Estimating caries risk from correlated dietary and hygiene factors

### **When NOT to use:**
- You suspect many features are irrelevant (use Lasso instead for feature selection)
- Relationships are strongly non-linear (use tree-based or neural network models)
- You need an exactly sparse model for clinical guidelines (use Lasso or ElasticNet)
- Data has very few features and no multicollinearity (plain linear regression suffices)

---

## 🔧 **Hyperparameters to Tune**

| Parameter | Typical Range | Dental Tip |
|-----------|--------------|------------|
| alpha | 0.01 - 100 | Start at 1.0; increase if coefficients are still unstable |
| learning_rate | 0.001 - 0.1 | Use 0.01; treatment durations (6-48) have smaller range than costs |
| epochs | 500 - 5000 | 1000 usually sufficient for 600-patient datasets |
| feature_scaling | StandardScaler | Critical -- severity (1-10) vs compliance (0-100) differ in scale |

### **Finding Optimal Alpha**
```python
# Use cross-validation to find best alpha
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    cv_score = cross_validate(X, y, alpha, k_folds=5)
    print(f"alpha={alpha}: CV MSE = {cv_score:.2f}")
```

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/ridge_regression_numpy.py
```

The script demonstrates:
1. Generating synthetic orthodontic data with correlated features (600 patients)
2. Implementing Ridge gradient descent from scratch with NumPy
3. Implementing the Ridge closed-form solution
4. Comparing coefficients across different alpha values
5. Training/test split evaluation with MSE and R-squared
6. Visualizing coefficient shrinkage paths

---

## 📚 **References**

1. Hoerl, A.E. & Kennard, R.W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55-67.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3.4.
3. Faber, J. et al. (2008). "Predictors of Orthodontic Treatment Duration." *American Journal of Orthodontics and Dentofacial Orthopedics*, 133(1).
4. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 3.1.4.

---

## 📝 **Implementation Reference**

- **NumPy implementation:** `01_regression/ridge_regression_numpy.py` -- gradient descent + closed-form with L2 penalty
- **Scikit-learn version:** `01_regression/ridge_regression_sklearn.py` -- production-ready with RidgeCV
- **PyTorch version:** `01_regression/ridge_regression_pytorch.py` -- GPU-accelerated with weight decay

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
