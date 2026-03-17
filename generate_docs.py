#!/usr/bin/env python3
"""
Documentation Generator for ML Algorithms Repository
Generates markdown documentation for all Python files based on established templates
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = "/Users/ashokasangapallar/Desktop/studyrepos/ml-algorithms-reference"
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

# Algorithm metadata (add entries as needed)
ALGORITHM_INFO = {
   
    # Classification
    "logistic_regression": {
        "name": "Logistic Regression",
        "category": "classification",
        "use_case": "Binary classification with probability outputs",
        "key_feature": "Interpretable coefficients as odds ratios",
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "category": "classification",
        "use_case": "Instance-based classification via similarity",
        "key_feature": "No training phase, lazy learning",
    },
    "decision_tree": {
        "name": "Decision Tree",
        "category": "classification",
        "use_case": "Rule-based classification with high interpretability",
        "key_feature": "Greedy recursive feature splitting",
    },
    "random_forest": {
        "name": "Random Forest",
        "category": "classification",
        "use_case": "Ensemble classifier reducing overfitting",
        "key_feature": "Bootstrap aggregating + random features",
    },
    "svm": {
        "name": "Support Vector Machine",
        "category": "classification",
        "use_case": "Maximum margin classification",
        "key_feature": "Kernel trick for non-linear boundaries",
    },
    "naive_bayes": {
        "name": "Naive Bayes",
        "category": "classification",
        "use_case": "Fast probabilistic classification",
        "key_feature": "Conditional independence assumption",
    },
    "adaboost": {
        "name": "AdaBoost",
        "category": "classification",
        "use_case": "Adaptive boosting focusing on hard examples",
        "key_feature": "Sequential weak learner weighting",
    },
    "xgboost": {
        "name": "XGBoost",
        "category": "classification",
        "use_case": "Optimized gradient boosting for tabular data",
        "key_feature": "Regularized boosting with 2nd-order gradients",
    },
    "lightgbm": {
        "name": "LightGBM",
        "category": "classification",
        "use_case": "Fast gradient boosting for large-scale data",
        "key_feature": "Histogram-based + leaf-wise growth",
    },
    "catboost": {
        "name": "CatBoost",
        "category": "classification",
        "use_case": "Gradient boosting with native categorical support",
        "key_feature": "Ordered boosting + target encoding",
    },
    
    # Clustering
    "kmeans": {
        "name": "K-Means",
        "category": "clustering",
        "use_case": "Partition-based clustering into K spherical clusters",
        "key_feature": "Iterative centroid assignment",
    },
    "dbscan": {
        "name": "DBSCAN",
        "category": "clustering",
        "use_case": "Density-based clustering finding arbitrary shapes",
        "key_feature": "Automatic outlier detection",
    },
    "hierarchical_clustering": {
        "name": "Hierarchical Clustering",
        "category": "clustering",
        "use_case": "Build cluster hierarchy via successive merging",
        "key_feature": "Dendrogram visualization",
    },
    
    # Time Series
    "arima": {
        "name": "ARIMA",
        "category": "time_series",
        "use_case": "Forecasting univariate time series",
        "key_feature": "AutoRegressive + Moving Average + Differencing",
    },
    "sarima": {
        "name": "SARIMA",
        "category": "time_series",
        "use_case": "Seasonal time series forecasting",
        "key_feature": "ARIMA + seasonal components",
    },
    "lstm": {
        "name": "LSTM",
        "category": "time_series",
        "use_case": "Sequence modeling with long-term dependencies",
        "key_feature": "Gated recurrent units preventing vanishing gradients",
    },
    "prophet": {
        "name": "Prophet",
        "category": "time_series",
        "use_case": "Automated forecasting with seasonality",
        "key_feature": "Additive model with trend + seasonality",
    },
    "temporal_fusion_transformer": {
        "name": "Temporal Fusion Transformer",
        "category": "time_series",
        "use_case": "Multi-horizon forecasting with attention",
        "key_feature": "Transformer architecture for time series",
    },
    
    # NLP
    "tfidf_classifier": {
        "name": "TF-IDF Classifier",
        "category": "nlp",
        "use_case": "Text classification using term frequency features",
        "key_feature": "Sparse bag-of-words representation",
    },
    "bert_classifier": {
        "name": "BERT Classifier",
        "category": "nlp",
        "use_case": "Transfer learning for text classification",
        "key_feature": "Pre-trained transformer embeddings",
    },
    "llm_rag": {
        "name": "LLM RAG",
        "category": "nlp",
        "use_case": "Retrieval-augmented generation for Q&A",
        "key_feature": "Combining retrieval with generation",
    },
    
    # Computer Vision
    "cnn": {
        "name": "CNN",
        "category": "computer_vision",
        "use_case": "Image classification with convolutional layers",
        "key_feature": "Spatial feature extraction via convolution",
    },
    "resnet": {
        "name": "ResNet",
        "category": "computer_vision",
        "use_case": "Deep image classification with residual connections",
        "key_feature": "Skip connections preventing degradation",
    },
    "efficientnet": {
        "name": "EfficientNet",
        "category": "computer_vision",
        "use_case": "Efficient image classification with compound scaling",
        "key_feature": "Balanced width/depth/resolution scaling",
    },
    
    # Recommendation
    "collaborative_filtering": {
        "name": "Collaborative Filtering",
        "category": "recommendation",
        "use_case": "User-item recommendation via similarity",
        "key_feature": "User-user or item-item similarity",
    },
    "matrix_factorization": {
        "name": "Matrix Factorization",
        "category": "recommendation",
        "use_case": "Latent factor recommendation models",
        "key_feature": "Low-rank matrix decomposition",
    },
    
    # Anomaly Detection
    "isolation_forest": {
        "name": "Isolation Forest",
        "category": "anomaly_detection",
        "use_case": "Detecting outliers via random partitioning",
        "key_feature": "Anomalies isolated quickly in trees",
    },
    "autoencoder": {
        "name": "Autoencoder",
        "category": "anomaly_detection",
        "use_case": "Reconstruction-based anomaly detection",
        "key_feature": "High reconstruction error for anomalies",
    },
    
    # Dimensionality Reduction
    "pca": {
        "name": "PCA",
        "category": "dimensionality_reduction",
        "use_case": "Linear dimensionality reduction",
        "key_feature": "Orthogonal projection maximizing variance",
    },
    "tsne": {
        "name": "t-SNE",
        "category": "dimensionality_reduction",
        "use_case": "Non-linear embedding for visualization",
        "key_feature": "Preserves local neighborhood structure",
    },
}


def generate_sklearn_doc(algo_name, algo_info, category):
    """Generate sklearn implementation reference doc"""
    return f"""# {algo_info['name']} - Scikit-learn Implementation

## 📚 **Quick Reference**

This is the **scikit-learn** implementation of {algo_info['name']}. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[{algo_info['name']} - Full Documentation]({algo_name}_numpy.md)**

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
print(f"Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))
```

---

## 🔧 **Key Parameters**

See scikit-learn documentation for full parameter list.

---

## 📝 **Code Reference**

Full implementation: [`{category}/{algo_name}_sklearn.py`](../../{category}/{algo_name}_sklearn.py)

Related:
- [{algo_info['name']} - NumPy (from scratch)]({algo_name}_numpy.md)
- [{algo_info['name']} - PyTorch]({algo_name}_pytorch.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
"""


def generate_pytorch_doc(algo_name, algo_info, category):
    """Generate PyTorch implementation reference doc"""
    return f"""# {algo_info['name']} - PyTorch Implementation

## 📚 **Quick Reference**

This is the **PyTorch** implementation of {algo_info['name']}. For the complete algorithm explanation, mathematical background, and use cases, see:

👉 **[{algo_info['name']} - Full Documentation]({algo_name}_numpy.md)**

---

## 🚀 **PyTorch Advantages**

- **GPU acceleration**: Train on GPU for massive datasets
- **Automatic differentiation**: No manual gradient computation
- **Flexible architecture**: Easy to customize and extend
- **Modern optimizers**: Adam, RMSprop, learning rate schedules

---

## 💻 **Quick Start**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data to tensors
X_train = torch.FloatTensor(X_train_numpy)
y_train = torch.LongTensor(y_train_numpy)

# Define model (placeholder - see actual implementation)
model = MODEL_CLASS_PLACEHOLDER()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🎯 **GPU Acceleration**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
```

---

## 📝 **Code Reference**

Full implementation: [`{category}/{algo_name}_pytorch.py`](../../{category}/{algo_name}_pytorch.py)

Related:
- [{algo_info['name']} - NumPy (from scratch)]({algo_name}_numpy.md)
- [{algo_info['name']} - Scikit-learn]({algo_name}_sklearn.md)

---

**Created:** March 17, 2026  
**Repository:** ml-algorithms-reference
"""


def find_all_python_files():
    """Find all Python files that need documentation"""
    files_to_document = []
    categories = [
        "02_classification", "03_clustering", "04_time_series", 
        "05_nlp", "06_computer_vision", "07_recommendation",
        "08_anomaly_detection", "09_dimensionality_reduction"
    ]
    
    for category in categories:
        category_path = os.path.join(BASE_DIR, category)
        if os.path.exists(category_path):
            for py_file in Path(category_path).glob("*.py"):
                files_to_document.append((category, py_file.stem))
    
    return files_to_document


def main():
    """Generate documentation for all files"""
    files = find_all_python_files()
    print(f"Found {len(files)} Python files to document\n")
    
    generated = []
    skipped = []
    
    for category, file_stem in files:
        # Parse file_stem to get algorithm and implementation type
        parts = file_stem.rsplit('_', 1)
        if len(parts) == 2:
            algo_name, impl_type = parts
        else:
            continue
        
        # Skip if not sklearn or pytorch (numpy docs should be manual)
        if impl_type not in ['sklearn', 'pytorch']:
            continue
        
        # Check if we have metadata
        if algo_name not in ALGORITHM_INFO:
            skipped.append((category, file_stem))
            continue
        
        # Generate documentation
        algo_info = ALGORITHM_INFO[algo_name]
        if impl_type == 'sklearn':
            content = generate_sklearn_doc(algo_name, algo_info, category)
        else:
            content = generate_pytorch_doc(algo_name, algo_info, category)
        
        # Write file
        output_dir = os.path.join(EXAMPLES_DIR, category)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{file_stem}.md")
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        generated.append(output_file)
    
    print(f"✅ Generated {len(generated)} documentation files")
    print(f"⚠️  Skipped {len(skipped)} files (need metadata or are numpy versions)")
    
    if skipped:
        print("\nSkipped files:")
        for category, file_stem in skipped[:10]:
            print(f"  - {category}/{file_stem}.py")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")


if __name__ == "__main__":
    main()
