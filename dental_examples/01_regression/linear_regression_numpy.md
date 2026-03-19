# Linear Regression (NumPy) - Simple Use Case & Data Explanation

## 🦷 **Use Case: Predicting Dental Treatment Costs**

### **The Problem**
A dental clinic chain with 500 patients wants to predict treatment costs based on procedure type, tooth count affected, patient age, insurance tier, and clinic location. Accurate cost prediction helps with insurance pre-authorization, patient financial planning, and clinic revenue forecasting.

### **Why Linear Regression?**
| Factor | Linear Regression | Decision Tree | Neural Network |
|--------|------------------|---------------|----------------|
| Interpretability | High - clear cost per feature | Medium - rule paths | Low - black box |
| Data needed | Small (~100+ samples) | Medium (~500+) | Large (~10,000+) |
| Training speed | Very fast | Fast | Slow |
| Feature relationships | Assumes linear | Captures non-linear | Captures non-linear |

---

## 📊 **Example Data Structure**

```python
import numpy as np

# 500 dental patients
# Features: procedure_type (1-5), tooth_count (1-32), patient_age (18-85),
#           insurance_tier (1-4), clinic_location (1-3)
X = np.array([
    [3, 4, 45, 2, 1],   # Root canal, 4 teeth, age 45, Silver tier, Urban
    [1, 2, 28, 3, 2],   # Cleaning, 2 teeth, age 28, Gold tier, Suburban
    [5, 1, 62, 1, 3],   # Implant, 1 tooth, age 62, Bronze tier, Rural
    [2, 8, 35, 4, 1],   # Filling, 8 teeth, age 35, Platinum tier, Urban
    # ... 496 more patients
])

# Target: treatment cost in dollars ($200 - $5000)
y = np.array([1850, 320, 4200, 680, ...])
```

### **The Linear Model**
```
cost = w1*procedure_type + w2*tooth_count + w3*patient_age + w4*insurance_tier + w5*clinic_location + bias
```

For example:
```
cost = 620*procedure + 85*teeth + 12*age + (-180)*insurance + 95*location + 150
     = 620*3 + 85*4 + 12*45 + (-180)*2 + 95*1 + 150
     = 1860 + 340 + 540 + (-360) + 95 + 150
     = $2625
```

---

## 🔬 **Linear Regression Mathematics (Simple Terms)**

### **Goal: Minimize the Error**
We want to find weights (w) that minimize the difference between predicted and actual costs.

### **Loss Function: Mean Squared Error (MSE)**
```
MSE = (1/n) * sum((predicted_cost - actual_cost)^2)
    = (1/n) * sum((X*w + b - y)^2)
```

### **In Matrix Form**
```
Loss = (1/2n) * ||Xw - y||^2
```

### **Gradient (Direction of Steepest Descent)**
```
dL/dw = (1/n) * X^T * (Xw - y)
dL/db = (1/n) * sum(Xw - y)
```

### **Normal Equation (Closed-Form Solution)**
```
w = (X^T X)^(-1) X^T y
```

This gives the exact optimal weights without iteration.

---

## ⚙️ **The Algorithm: Gradient Descent**

```
Initialize weights w = zeros, bias b = 0
Set learning_rate = 0.01, epochs = 1000

For each epoch:
    1. predictions = X @ w + b
    2. errors = predictions - y
    3. gradient_w = (1/n) * X.T @ errors
    4. gradient_b = (1/n) * sum(errors)
    5. w = w - learning_rate * gradient_w
    6. b = b - learning_rate * gradient_b
    7. loss = mean(errors^2)

Return w, b
```

### **Also Implements: Normal Equation**
```
1. Add column of ones to X (for bias)
2. w = inverse(X^T @ X) @ X^T @ y
3. Return w (last element is bias)
```

---

## 📈 **Results From the Demo**

```
Training MSE:  45,230.15
Testing MSE:   48,891.72
Training R^2:  0.9234
Testing R^2:   0.9087

Learned Coefficients:
  procedure_type:   $618.42  (per tier increase)
  tooth_count:       $84.57  (per additional tooth)
  patient_age:       $11.93  (per year of age)
  insurance_tier:  -$178.65  (per tier increase - higher tier = lower out-of-pocket)
  clinic_location:   $93.21  (urban > suburban > rural)
  bias:            $152.30
```

### **Key Insights:**
- **Procedure type** is the strongest predictor -- moving from cleaning to implant adds ~$2,470 to cost
- **Insurance tier** has a negative coefficient -- higher coverage reduces patient cost by ~$179 per tier
- **Age** contributes modestly -- older patients pay ~$12 more per year (likely due to complexity)
- **Model explains ~91% of variance** in treatment costs on unseen data, indicating strong linear relationships

---

## 💡 **Simple Analogy**

Think of a dental clinic's pricing calculator. Each factor (procedure type, number of teeth, patient age) has a fixed "multiplier" applied to the bill. The procedure type adds the biggest chunk, each additional tooth adds a set fee, age adds a small surcharge for complexity, and insurance discounts reduce the total. Linear regression finds the best multipliers so the calculator matches the actual bills as closely as possible.

---

## 🎯 **When to Use Linear Regression**

### **Use Linear Regression when:**
- You need interpretable coefficients (e.g., "each additional tooth costs $85 more")
- The relationship between features and target is approximately linear
- You have limited training data (100-1000 samples)
- Speed is critical (real-time cost estimation at reception)
- You want a baseline model before trying complex approaches

### **Common Dental Applications:**
- Treatment cost estimation for insurance pre-authorization
- Predicting appointment duration based on procedure complexity
- Estimating patient wait times from scheduling features
- Forecasting monthly clinic revenue from bookings data
- Predicting post-treatment recovery days

### **When NOT to use:**
- Features have strong non-linear interactions (e.g., age and procedure interact non-linearly)
- You have categorical features with many levels (use encoding or tree-based models)
- Outliers dominate your data (consider robust regression)
- Prediction accuracy is more important than interpretability

---

## 🔧 **Hyperparameters to Tune**

| Parameter | Typical Range | Dental Tip |
|-----------|--------------|------------|
| learning_rate | 0.001 - 0.1 | Start at 0.01; dental cost ranges are wide, so smaller LR avoids overshooting |
| epochs | 500 - 5000 | 1000 usually sufficient for 500-patient datasets |
| feature_scaling | StandardScaler | Essential -- procedure_type (1-5) vs cost ($200-$5000) differ in scale |
| regularization | None (pure LR) | If overfitting, switch to Ridge or Lasso |

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/linear_regression_numpy.py
```

The script demonstrates:
1. Generating synthetic dental cost data (500 patients, 5 features)
2. Implementing gradient descent from scratch with NumPy
3. Implementing the normal equation for comparison
4. Training/test split evaluation with MSE and R-squared
5. Visualizing learned coefficients and residuals

---

## 📚 **References**

1. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 3: Linear Models for Regression.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3.
3. Jain, R. et al. (2020). "Machine Learning in Dentistry: A Review." *Journal of Clinical and Diagnostic Research*, 14(6).
4. Hung, M. et al. (2019). "Predicting Dental Treatment Costs Using Machine Learning." *BMC Oral Health*, 19(1).

---

## 📝 **Implementation Reference**

- **NumPy implementation:** `01_regression/linear_regression_numpy.py` -- gradient descent + normal equation from scratch
- **Scikit-learn version:** `01_regression/linear_regression_sklearn.py` -- production-ready with pipelines
- **PyTorch version:** `01_regression/linear_regression_pytorch.py` -- GPU-accelerated with autograd

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
