# Autoencoder - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of Autoencoder. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[Autoencoder - Full Documentation](autoencoder_numpy.md)**

---

## 🚀 **Scikit-learn Advantages**

- **Production-ready**: Battle-tested, optimized implementation
- **Rich ecosystem**: Integrates with pipelines, cross-validation, preprocessing
- **Minimal code**: Fit in 2-3 lines
- **Multiple solvers**: Auto-selects best solver based on data characteristics

---

## 💻 **Quick Start**

```python
from sklearn MODEL_IMPORT_PLACEHOLDER
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = MODEL_CLASS_PLACEHOLDER()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

## 🔧 **Key Parameters**

See scikit-learn documentation for full parameter list.

---

## 📝 **Code Reference**

Full implementation: [`08_anomaly_detection/autoencoder_sklearn.py`](../../08_anomaly_detection/autoencoder_sklearn.py)

Related:
- [Autoencoder - NumPy (from scratch)](autoencoder_numpy.md)
- [Autoencoder - PyTorch](autoencoder_pytorch.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
