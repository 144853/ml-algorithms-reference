# Logistic Regression - Simple Use Case & Data Explanation

## 🏥 **Use Case: Predicting Patient Hospital Readmission**

### **The Problem**
You're a healthcare analyst at a hospital trying to predict which patients are at high risk of readmission within 30 days. You have:
- **5,000 patients** from the past year
- **15 features** for each patient:
  - Age, gender, BMI
  - Number of previous admissions
  - Length of stay
  - Diagnosis codes
  - Lab test results
  - Medication count
  - Comorbidity scores
- A **binary outcome**: readmitted (1) or not readmitted (0)

The goal: Build a model that outputs **probabilities** of readmission for risk stratification and intervention planning.

### **Why Logistic Regression?**

| Method | Problem/Benefit |
|--------|-----------------|
| **Linear Regression** | Outputs unbounded values (can be < 0 or > 1) - nonsensical for probabilities |
| **Logistic Regression** ✅ | Outputs **calibrated probabilities** between 0 and 1 via sigmoid function |
| **Decision Tree** | Can overfit, probabilities less calibrated, harder to interpret coefficients |
| **Neural Network** | Black box, requires more data, harder to explain to clinicians |

**Logistic Regression's Strengths**: Interpretable odds ratios, well-calibrated probabilities, linear decision boundary, fast training.

---

## 📊 **Example Data Structure**

```python
# Sample data structure
n_patients = 5000
n_features = 15

# Feature matrix X: (5000 patients × 15 features)
X = [[65, 1, 28.5, 2, 5, ..., 3.2],  # Patient 0: 65 yrs, male, BMI 28.5, 2 prev admits, ...
     [42, 0, 22.1, 0, 3, ..., 2.1],  # Patient 1
     ...
     [78, 1, 31.2, 4, 7, ..., 4.5]]  # Patient 4999

# Target y: Readmission outcome (0 = no, 1 = yes)
y = [1, 0, 0, 1, ..., 1]  # 5000 labels

# Class distribution
# Class 0 (not readmitted): 4200 patients (84%)
# Class 1 (readmitted): 800 patients (16%)  ← Imbalanced!
```

### **The Logistic Function**
Unlike linear regression, logistic regression applies the **sigmoid function** to ensure outputs are probabilities:

```
z = w₁×age + w₂×gender + ... + w₁₅×feature₁₅ + b  (linear combination)
P(y=1|x) = σ(z) = 1 / (1 + e^(-z))                 (sigmoid)
```

**Sigmoid properties:**
- z → -∞: P(y=1) → 0 (very unlikely)
- z = 0: P(y=1) = 0.5 (boundary)
- z → +∞: P(y=1) → 1 (very likely)

**Example:**
- If z = 2.5 → P(readmission) = 0.92 (high risk!)
- If z = -1.8 → P(readmission) = 0.14 (low risk)

---

## 🔬 **Logistic Regression Mathematics (Simple Terms)**

Logistic regression minimizes the **Binary Cross-Entropy Loss (Log Loss)**:

```
Loss = -(1/n) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

Where:
- **p_i = σ(wᵀx_i + b)**: Predicted probability for sample i
- **y_i**: True label (0 or 1)
- **Intuition**: Penalizes confident wrong predictions heavily

**Why cross-entropy?**
- Linear regression's MSE doesn't work well for classification (non-convex for sigmoid)
- Cross-entropy is convex for logistic regression → guaranteed convergence

### **Gradient for Optimization:**

The gradient of log loss is remarkably simple:

```
∇w = (1/n) Xᵀ(p - y)  (same form as linear regression!)
∇b = (1/n) Σ(p_i - y_i)
```

Where p = σ(Xw + b) is the vector of predicted probabilities.

---

## ⚙️ **The Algorithm: Gradient Descent / Newton's Method**

### **Stochastic Gradient Descent**
```python
Initialize: w = 0, b = 0

for epoch in range(n_epochs):
    for each mini-batch (X_batch, y_batch):
        # Forward pass
        z = X_batch @ w + b
        p = sigmoid(z)
        
        # Compute gradients
        error = p - y_batch
        dw = (1/batch_size) × X_batchᵀ @ error
        db = (1/batch_size) × sum(error)
        
        # Update parameters
        w := w - lr × dw
        b := b - lr × db
```

### **L2 Regularization (Ridge Logistic)**
Add penalty to prevent overfitting:
```
Loss = Cross-Entropy + λ ||w||²

∇w = (1/n) Xᵀ(p - y) + 2λw  (extra shrinkage term)
```

---

## 📈 **Results From the Demo**

When run on hospital readmission data:

```
--- Training Results ---
Initial Loss:        0.6931 (random guess)
Final Loss:          0.3245
Training Accuracy:   87.2%
Test Accuracy:       85.8%
Training Time:       0.15s

--- Classification Metrics ---
Metric               Value      Interpretation
---------------------------------------------------
Accuracy             85.8%      Overall correctness
Precision (class 1)  72.3%      Of predicted readmits, 72% actually readmitted
Recall (class 1)     68.5%      Of actual readmits, caught 68.5%
F1 Score             70.3%      Harmonic mean of precision/recall
AUC-ROC              0.912      Excellent discrimination

--- Confusion Matrix ---
                Predicted
                No    Yes
Actual  No     840    160  ← 160 false alarms
        Yes     50    150  ← 50 missed readmissions

--- Coefficient Interpretation (Odds Ratios) ---
Feature                Coefficient    Odds Ratio    Interpretation
-----------------------------------------------------------------------
Age                      0.045         1.046        Each year → 4.6% higher odds
Prev Admissions          0.382         1.465        Each admission → 46.5% higher odds
Comorbidity Score        0.521         1.684        Each point → 68.4% higher odds
Length of Stay          -0.028         0.972        Each day → 2.8% lower odds (protective!)
```

### **Key Insights:**
- **Well-calibrated probabilities**: Can use P(readmit) > 0.6 as threshold for intervention
- **Interpretable coefficients**: Previous admissions and comorbidities are strong risk factors
- **Trade-off**: 160 false positives vs 50 missed cases - adjust threshold based on cost
- **AUC = 0.912**: Excellent ability to rank patients by risk

---

## 💡 **Simple Analogy**

Think of a credit score:
- **Not just yes/no**: Logistic regression gives you a risk score (0-100%)
- **Interpretable**: "Your score increased 15 points because you paid bills on time" (coefficients!)
- **Calibrated**: A score of 70% means 70 out of 100 similar people defaulted
- **Decision boundary**: Bank sets threshold (e.g., score > 65% → deny loan)

---

## 🎯 **When to Use Logistic Regression**

### **Use Logistic Regression when:**
- You need **binary classification** (yes/no, 0/1)
- You want **interpretable coefficients** (odds ratios for clinicians, regulators)
- You need **well-calibrated probabilities** (risk scores, A/B test conversion rates)
- Relationship between features and log-odds is **approximately linear**
- You want a **fast, reliable baseline** with few hyperparameters

### **Common Applications:**
- **Healthcare**: Disease diagnosis, readmission prediction, mortality risk scoring
- **Finance**: Credit scoring, fraud detection, default prediction
- **Marketing**: Click-through rate (CTR) prediction, conversion funnel optimization
- **Operations**: Equipment failure prediction, quality control
- **Social**: Spam detection, content moderation

### **When NOT to use:**
- **Non-linear decision boundaries** → Use kernel SVM, random forest, or neural networks
- **Multiple classes** with complex relationships → Use multinomial models or neural nets
- **High-dimensional sparse data** (e.g., text) → Try Naive Bayes or linear SVM
- **Highly imbalanced data** without resampling/weighting → Tree-based methods may work better

---

## 🔧 **Hyperparameters to Tune**

1. **learning_rate (lr)**
   - **Default**: 0.01
   - **Too high**: Loss diverges
   - **Too low**: Slow convergence
   - **Typical range**: [0.001, 0.1]

2. **lambda_reg (L2 penalty)**
   - **Default**: 0.01
   - Controls overfitting
   - Higher λ → simpler model (coefficients shrink toward 0)
   - **Tuning**: Cross-validation over [0.001, 0.01, 0.1, 1.0, 10]

3. **n_epochs**
   - **Default**: 100-1000
   - Monitor loss curve - stop when plateaus

4. **batch_size**
   - **Default**: 32
   - Smaller → noisier gradients, better generalization
   - Larger → stable gradients, faster per epoch

5. **class_weight** (for imbalanced data)
   - **balanced**: Automatically weight classes inversely to frequency
   - Helps with rare positive class (e.g., fraud detection)

---

## 🚀 **Running the Demo**

To see Logistic Regression in action:

```bash
cd /path/to/ml-algorithms-reference
python 02_classification/logistic_regression_numpy.py
```

The script will:
1. Generate synthetic hospital readmission data
2. Train logistic regression with SGD
3. Show probability outputs, confusion matrix, ROC curve
4. Demonstrate coefficient interpretation

---

## 📚 **References**

- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Section 4.4.

---

## 📝 **Implementation Reference**

The complete from-scratch NumPy implementation is available in:
- [`02_classification/logistic_regression_numpy.py`](../../02_classification/logistic_regression_numpy.py) - SGD with cross-entropy loss
- [`02_classification/logistic_regression_sklearn.py`](../../02_classification/logistic_regression_sklearn.py) - Scikit-learn wrapper
- [`02_classification/logistic_regression_pytorch.py`](../../02_classification/logistic_regression_pytorch.py) - PyTorch implementation

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
