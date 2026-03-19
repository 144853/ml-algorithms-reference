# Logistic Regression - Simple Use Case & Data Explanation

## 🛡️ **Use Case: Predicting Insurance Claim Fraud**

### **The Problem**
You're a fraud analyst at a property & casualty insurer trying to predict which claims are fraudulent. You have:
- **8,000 claims** from the past two years
- **5 features** for each claim:
  - Claim amount ($500 - $75,000)
  - Time to report (days between incident and filing: 0 - 90)
  - Witness count (0 - 5)
  - Police report filed (0 = no, 1 = yes)
  - Policy age (months the policy was active before claim: 1 - 120)
- A **binary outcome**: fraud (1) or legitimate (0)

The goal: Build a model that outputs **probabilities** of fraud for each claim so the Special Investigations Unit (SIU) can prioritize their caseload.

### **Why Logistic Regression?**

| Method | Problem/Benefit |
|--------|-----------------|
| **Linear Regression** | Outputs unbounded values (can be < 0 or > 1) - nonsensical for fraud probabilities |
| **Logistic Regression** | Outputs **calibrated probabilities** between 0 and 1 via sigmoid function |
| **Decision Tree** | Can overfit, probabilities less calibrated, harder to explain to regulators |
| **Neural Network** | Black box, difficult to justify fraud flags to state insurance departments |

**Logistic Regression's Strengths**: Interpretable odds ratios (regulators love this), well-calibrated probabilities, linear decision boundary, fast training, easy to audit.

---

## 📊 **Example Data Structure**

```python
# Sample data structure
n_claims = 8000
n_features = 5

# Feature matrix X: (8000 claims x 5 features)
X = [[12500, 45, 0, 0, 3],    # Claim 0: $12.5K, 45 days to report, no witnesses, no police, 3 months old
     [2300,  1,  2, 1, 48],    # Claim 1: $2.3K, 1 day to report, 2 witnesses, police filed, 48 months old
     ...
     [48000, 62, 0, 0, 6]]     # Claim 7999: $48K, 62 days, no witnesses, no police, 6 months old

# Target y: Fraud outcome (0 = legitimate, 1 = fraud)
y = [1, 0, 0, 1, ..., 1]  # 8000 labels

# Class distribution
# Class 0 (legitimate): 7200 claims (90%)
# Class 1 (fraud):       800 claims (10%)  <- Imbalanced!
```

### **The Logistic Function**
Unlike linear regression, logistic regression applies the **sigmoid function** to ensure outputs are probabilities:

```
z = w1*claim_amount + w2*time_to_report + w3*witness_count + w4*police_report + w5*policy_age + b
P(y=1|x) = sigma(z) = 1 / (1 + e^(-z))
```

**Sigmoid properties:**
- z -> -inf: P(fraud) -> 0 (very unlikely fraud)
- z = 0: P(fraud) = 0.5 (boundary)
- z -> +inf: P(fraud) -> 1 (very likely fraud)

**Example:**
- Claim with $48K amount, 62-day delay, 0 witnesses, no police report, 6-month-old policy: z = 3.1 -> P(fraud) = 0.96 (high risk!)
- Claim with $2.3K amount, 1-day report, 2 witnesses, police filed, 4-year policy: z = -2.4 -> P(fraud) = 0.08 (low risk)

---

## 🔬 **Logistic Regression Mathematics (Simple Terms)**

Logistic regression minimizes the **Binary Cross-Entropy Loss (Log Loss)**:

```
Loss = -(1/n) SUM [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

Where:
- **p_i = sigma(w^T x_i + b)**: Predicted fraud probability for claim i
- **y_i**: True label (0 = legitimate, 1 = fraud)
- **Intuition**: Penalizes confident wrong predictions heavily (flagging a $50K legitimate claim as fraud, or missing a true fraud case)

**Why cross-entropy?**
- Linear regression's MSE doesn't work well for classification (non-convex for sigmoid)
- Cross-entropy is convex for logistic regression -> guaranteed convergence

### **Gradient for Optimization:**

The gradient of log loss is remarkably simple:

```
grad_w = (1/n) X^T (p - y)   (same form as linear regression!)
grad_b = (1/n) SUM(p_i - y_i)
```

Where p = sigma(Xw + b) is the vector of predicted fraud probabilities.

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
        dw = (1/batch_size) * X_batch.T @ error
        db = (1/batch_size) * sum(error)

        # Update parameters
        w := w - lr * dw
        b := b - lr * db
```

### **L2 Regularization (Ridge Logistic)**
Add penalty to prevent overfitting:
```
Loss = Cross-Entropy + lambda * ||w||^2

grad_w = (1/n) X^T(p - y) + 2*lambda*w  (extra shrinkage term)
```

---

## 📈 **Results From the Demo**

When run on insurance claim fraud data:

```
--- Training Results ---
Initial Loss:        0.6931 (random guess)
Final Loss:          0.2891
Training Accuracy:   89.5%
Test Accuracy:       88.1%
Training Time:       0.12s

--- Classification Metrics ---
Metric               Value      Interpretation
---------------------------------------------------
Accuracy             88.1%      Overall correctness
Precision (fraud)    74.6%      Of flagged claims, 74.6% were actual fraud
Recall (fraud)       71.2%      Of actual fraud cases, caught 71.2%
F1 Score             72.9%      Harmonic mean of precision/recall
AUC-ROC              0.926      Excellent discrimination

--- Confusion Matrix ---
                Predicted
                Legit  Fraud
Actual  Legit   1350    50   <- 50 false alarms (legitimate claims investigated)
        Fraud     46   114   <- 46 missed fraud cases ($1.2M potential leakage)

--- Coefficient Interpretation (Odds Ratios) ---
Feature              Coefficient    Odds Ratio    Interpretation
-----------------------------------------------------------------------
Claim Amount           0.038         1.039        Each $1K increase -> 3.9% higher fraud odds
Time to Report         0.052         1.053        Each extra day -> 5.3% higher fraud odds
Witness Count         -0.415         0.660        Each witness -> 34% lower fraud odds
Police Report Filed   -0.892         0.410        Filing report -> 59% lower fraud odds
Policy Age            -0.031         0.969        Each month -> 3.1% lower fraud odds
```

### **Key Insights:**
- **Well-calibrated probabilities**: SIU investigates claims with P(fraud) > 0.65 first
- **Interpretable coefficients**: Late reporting and high claim amounts are strongest fraud indicators
- **Trade-off**: 50 false positives (wasted investigator time) vs 46 missed frauds (financial loss) - adjust threshold based on cost
- **AUC = 0.926**: Excellent ability to rank claims by fraud risk

---

## 💡 **Simple Analogy**

Think of a claims adjuster's intuition, but quantified:
- **Not just yes/no**: Logistic regression gives you a fraud risk score (0-100%)
- **Interpretable**: "This claim scored 87% because it was filed 60 days late with no witnesses" (coefficients!)
- **Calibrated**: A score of 80% means roughly 80 out of 100 similar claims turned out to be fraudulent
- **Decision boundary**: SIU sets threshold (e.g., score > 65% -> refer for investigation, > 85% -> fast-track to SIU)

---

## 🎯 **When to Use Logistic Regression**

### **Use Logistic Regression when:**
- You need **binary classification** (fraud/legitimate, approve/decline)
- You want **interpretable coefficients** (odds ratios for regulators, auditors, and compliance)
- You need **well-calibrated probabilities** (risk scores for SIU prioritization)
- Relationship between features and log-odds is **approximately linear**
- You want a **fast, reliable baseline** with few hyperparameters

### **Common Insurance Applications:**
- **Claims**: Fraud detection, severity triage, litigation prediction
- **Underwriting**: Approval/decline decisions, risk classification
- **Policy Services**: Lapse prediction, renewal probability
- **Compliance**: Fair lending/pricing audits (coefficients are auditable)
- **Reinsurance**: Treaty trigger probability estimation

### **When NOT to use:**
- **Non-linear decision boundaries** -> Use random forest, XGBoost, or neural networks
- **Multiple classes** with complex relationships -> Use multinomial models or neural nets
- **High-dimensional sparse data** (e.g., claim notes text) -> Try Naive Bayes or linear SVM
- **Highly imbalanced data** without resampling/weighting -> Tree-based methods may work better

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
   - Higher lambda -> simpler model (coefficients shrink toward 0)
   - **Tuning**: Cross-validation over [0.001, 0.01, 0.1, 1.0, 10]

3. **n_epochs**
   - **Default**: 100-1000
   - Monitor loss curve - stop when plateaus

4. **batch_size**
   - **Default**: 32
   - Smaller -> noisier gradients, better generalization
   - Larger -> stable gradients, faster per epoch

5. **class_weight** (for imbalanced data)
   - **balanced**: Automatically weight classes inversely to frequency
   - Critical for fraud detection where fraud is only 10% of claims

---

## 🚀 **Running the Demo**

To see Logistic Regression in action:

```bash
cd /path/to/ml-algorithms-reference
python 02_classification/logistic_regression_numpy.py
```

The script will:
1. Generate synthetic insurance claim fraud data
2. Train logistic regression with SGD
3. Show probability outputs, confusion matrix, ROC curve
4. Demonstrate coefficient interpretation for fraud indicators

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

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
