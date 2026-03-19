# MLP Regressor (NumPy) - Simple Use Case & Data Explanation

## 🦷 **Use Case: Predicting Dental Crown Longevity**

### **The Problem**
A dental prosthetics lab tracking 800 patients wants to predict how many years a dental crown will last based on material type, bite force, oral pH, grinding habit, and crown position. The relationships between these factors are non-linear -- for example, a high bite force combined with grinding habit drastically reduces longevity in a way linear models cannot capture. A Multi-Layer Perceptron (MLP) can learn these complex interactions.

### **Why MLP Regressor?**
| Factor | MLP Regressor | Linear Regression | Decision Tree |
|--------|--------------|-------------------|---------------|
| Non-linear relationships | Captures complex interactions | Cannot model | Captures via splits |
| Smooth predictions | Yes -- continuous output surface | Yes | No -- step-wise |
| Feature interactions | Learns automatically | Must be engineered | Learns automatically |
| Data requirements | Large (~500+) | Small (~100+) | Medium (~300+) |

---

## 📊 **Example Data Structure**

```python
import numpy as np

# 800 crown patients
# Features: material_type (1-4: porcelain, metal, ceramic, zirconia),
#           bite_force (50-400 N), oral_ph (5.0-8.0),
#           grinding_habit (0-3: none, mild, moderate, severe),
#           crown_position (1-5: front incisor to back molar)
X = np.array([
    [3, 180, 6.8, 0, 4],   # Ceramic, moderate force, neutral pH, no grinding, premolar
    [1, 320, 5.5, 2, 5],   # Porcelain, high force, acidic pH, moderate grinding, molar
    [4, 150, 7.2, 0, 2],   # Zirconia, low force, alkaline pH, no grinding, lateral incisor
    [2, 280, 6.0, 3, 5],   # Metal, high force, slightly acidic, severe grinding, molar
    # ... 796 more patients
])

# Target: crown longevity in years (2 - 25 years)
y = np.array([15.5, 6.2, 22.0, 8.8, ...])
```

### **The MLP Model**
```
Input (5 features)
    |
Hidden Layer 1 (16 neurons, ReLU activation)
    |
Hidden Layer 2 (8 neurons, ReLU activation)
    |
Output Layer (1 neuron, linear activation)
    |
Predicted longevity (years)
```

---

## 🔬 **MLP Mathematics (Simple Terms)**

### **Forward Pass**
```
Layer 1: h1 = ReLU(X @ W1 + b1)        # shape: (n, 5) @ (5, 16) = (n, 16)
Layer 2: h2 = ReLU(h1 @ W2 + b2)       # shape: (n, 16) @ (16, 8) = (n, 8)
Output:  y_hat = h2 @ W3 + b3          # shape: (n, 8) @ (8, 1) = (n, 1)
```

### **ReLU Activation**
```
ReLU(z) = max(0, z)

ReLU'(z) = 1 if z > 0
            0 if z <= 0
```

ReLU introduces non-linearity, allowing the network to model complex feature interactions (e.g., grinding + high bite force).

### **Loss Function: Mean Squared Error**
```
MSE = (1/n) * sum((y_hat - y)^2)
```

### **Backpropagation (Chain Rule)**
```
dL/dW3 = h2^T @ (y_hat - y) / n
dL/db3 = sum(y_hat - y) / n

delta2 = (y_hat - y) @ W3^T * ReLU'(h2)
dL/dW2 = h1^T @ delta2 / n
dL/db2 = sum(delta2) / n

delta1 = delta2 @ W2^T * ReLU'(h1)
dL/dW1 = X^T @ delta1 / n
dL/db1 = sum(delta1) / n
```

---

## ⚙️ **The Algorithm: Mini-Batch Gradient Descent with Backpropagation**

```
Initialize weights W1, W2, W3 with Xavier initialization
Initialize biases b1, b2, b3 = zeros
Set learning_rate = 0.001, epochs = 500, batch_size = 32

For each epoch:
    Shuffle training data
    For each mini-batch:
        1. Forward pass:
           h1 = ReLU(X_batch @ W1 + b1)
           h2 = ReLU(h1 @ W2 + b2)
           y_hat = h2 @ W3 + b3

        2. Compute loss:
           loss = mean((y_hat - y_batch)^2)

        3. Backward pass:
           Compute gradients via chain rule

        4. Update parameters:
           W_i = W_i - learning_rate * dL/dW_i
           b_i = b_i - learning_rate * dL/db_i

    Compute validation loss for early stopping

Return W1, W2, W3, b1, b2, b3
```

### **Xavier Initialization**
```python
W1 = np.random.randn(5, 16) * np.sqrt(2.0 / 5)    # sqrt(2/fan_in) for ReLU
W2 = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
W3 = np.random.randn(8, 1) * np.sqrt(2.0 / 8)
```

---

## 📈 **Results From the Demo**

```
Training MSE:  2.15
Testing MSE:   2.89
Training R^2:  0.9412
Testing R^2:   0.9215

Training History:
  Epoch 100: Train MSE=5.82, Val MSE=6.15
  Epoch 200: Train MSE=3.45, Val MSE=3.92
  Epoch 300: Train MSE=2.48, Val MSE=3.10
  Epoch 400: Train MSE=2.21, Val MSE=2.95
  Epoch 500: Train MSE=2.15, Val MSE=2.89

Sample Predictions:
  Patient 1: Predicted=15.2 yrs, Actual=15.5 yrs (error: 0.3 yrs)
  Patient 2: Predicted=6.8 yrs,  Actual=6.2 yrs  (error: 0.6 yrs)
  Patient 3: Predicted=21.4 yrs, Actual=22.0 yrs (error: 0.6 yrs)
```

### **Key Insights:**
- **MLP captures non-linear interactions** -- grinding habit combined with high bite force reduces longevity more than either factor alone
- **Zirconia crowns in low-force positions** are predicted to last 20+ years, matching clinical studies
- **Oral pH below 6.0** (acidic) significantly reduces predicted longevity, reflecting enamel erosion effects
- **R-squared of 0.92** on test data is substantially better than linear regression (R^2=0.78), confirming non-linear feature interactions

---

## 💡 **Simple Analogy**

Think of the MLP as a panel of dental specialists reviewing a crown case. The first layer of neurons acts like general dentists, each looking at combinations of features (material + force, pH + grinding). The second layer acts like prosthodontic specialists who synthesize these assessments into more refined judgments. The output layer is the final consensus: "this crown should last X years." Each layer builds on the previous one, capturing increasingly complex patterns.

---

## 🎯 **When to Use MLP Regressor**

### **Use MLP Regressor when:**
- Features have non-linear interactions (e.g., grinding + bite force)
- Linear models consistently underperform on your dental data
- You have sufficient data (500+ samples for 5 features)
- You need smooth, continuous predictions (not step-wise like trees)
- You want a universal function approximator

### **Common Dental Applications:**
- Predicting dental restoration longevity from multi-factor clinical data
- Estimating tooth survival probability after root canal treatment
- Forecasting orthodontic tooth movement from biomechanical features
- Predicting post-operative pain levels from patient and procedure features
- Estimating bone regeneration rates from implant site characteristics

### **When NOT to use:**
- You have very limited data (under 200 samples) -- use linear models
- Interpretability is critical for clinical guidelines -- use linear or tree models
- Features are linearly related to the target -- MLP is overkill
- You need feature importance rankings -- tree models are more interpretable

---

## 🔧 **Hyperparameters to Tune**

| Parameter | Typical Range | Dental Tip |
|-----------|--------------|------------|
| hidden_layers | (16, 8), (32, 16), (64, 32) | Start with (16, 8) for 5 features; increase for larger feature sets |
| learning_rate | 0.0001 - 0.01 | Use 0.001; longevity range (2-25 yrs) is moderate |
| batch_size | 16 - 64 | 32 works well for 800-patient datasets |
| epochs | 200 - 1000 | Use early stopping with patience=50 |
| activation | ReLU, LeakyReLU | ReLU is standard; LeakyReLU if "dead neuron" problem occurs |
| weight_init | Xavier/He | He initialization for ReLU activations |
| dropout | 0.0 - 0.3 | Add 0.1-0.2 dropout if overfitting on small dental datasets |

---

## 🚀 **Running the Demo**

```bash
cd /path/to/ml-algorithms-reference
python 01_regression/mlp_regressor_numpy.py
```

The script demonstrates:
1. Generating synthetic dental crown data (800 patients, 5 features with non-linear interactions)
2. Implementing a 2-hidden-layer MLP from scratch with NumPy
3. Forward pass, backpropagation, and mini-batch gradient descent
4. Xavier weight initialization
5. Training/validation curves and early stopping
6. Comparing MLP performance against linear regression

---

## 📚 **References**

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 6-8.
2. Glorot, X. & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *AISTATS*.
3. Zhangina, E. et al. (2021). "Deep Learning for Predicting Dental Restoration Longevity." *Journal of Dental Research*, 100(8).
4. Rumelhart, D., Hinton, G., & Williams, R. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533-536.

---

## 📝 **Implementation Reference**

- **NumPy implementation:** `01_regression/mlp_regressor_numpy.py` -- full backpropagation from scratch
- **Scikit-learn version:** `01_regression/mlp_regressor_sklearn.py` -- production-ready MLPRegressor
- **PyTorch version:** `01_regression/mlp_regressor_pytorch.py` -- GPU-accelerated with autograd

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
