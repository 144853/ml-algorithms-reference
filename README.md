# ML Algorithms Reference

A comprehensive machine learning reference repository with **105 algorithm implementations** across **9 categories**. Every algorithm is implemented in **3 variants**: scikit-learn, NumPy (from scratch), and PyTorch.

Each file is a self-contained, runnable tutorial with:
- Detailed theory and mathematics in the docstring
- Heavy line-by-line commenting explaining WHAT and WHY
- `train()`, `validate()`, `test()` pipeline
- Hyperparameter optimization with **Ray Tune** and **Optuna**
- `compare_parameter_sets()` — multiple configs with trade-off reasoning
- `real_world_demo()` — domain-specific use case with named features

## Repository Structure

```
ml-algorithms-reference/
├── 01_regression/          (15 files)
├── 02_classification/      (30 files)
├── 03_clustering/          (9 files)
├── 04_time_series/         (15 files)
├── 05_nlp/                 (9 files)
├── 06_computer_vision/     (9 files)
├── 07_recommendation/      (6 files)
├── 08_anomaly_detection/   (6 files)
├── 09_dimensionality_reduction/ (6 files)
├── requirements.txt
└── README.md
```

## Algorithms

### 1. Regression (15 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| Linear Regression | `linear_regression_sklearn.py` | `linear_regression_numpy.py` | `linear_regression_pytorch.py` |
| Ridge Regression | `ridge_regression_sklearn.py` | `ridge_regression_numpy.py` | `ridge_regression_pytorch.py` |
| Lasso Regression | `lasso_regression_sklearn.py` | `lasso_regression_numpy.py` | `lasso_regression_pytorch.py` |
| ElasticNet | `elasticnet_sklearn.py` | `elasticnet_numpy.py` | `elasticnet_pytorch.py` |
| MLP Regressor | `mlp_regressor_sklearn.py` | `mlp_regressor_numpy.py` | `mlp_regressor_pytorch.py` |

### 2. Classification (30 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| Logistic Regression | `logistic_regression_sklearn.py` | `logistic_regression_numpy.py` | `logistic_regression_pytorch.py` |
| Decision Tree | `decision_tree_sklearn.py` | `decision_tree_numpy.py` | `decision_tree_pytorch.py` |
| Random Forest | `random_forest_sklearn.py` | `random_forest_numpy.py` | `random_forest_pytorch.py` |
| XGBoost | `xgboost_sklearn.py` | `xgboost_numpy.py` | `xgboost_pytorch.py` |
| LightGBM | `lightgbm_sklearn.py` | `lightgbm_numpy.py` | `lightgbm_pytorch.py` |
| CatBoost | `catboost_sklearn.py` | `catboost_numpy.py` | `catboost_pytorch.py` |
| AdaBoost | `adaboost_sklearn.py` | `adaboost_numpy.py` | `adaboost_pytorch.py` |
| SVM | `svm_sklearn.py` | `svm_numpy.py` | `svm_pytorch.py` |
| Naive Bayes | `naive_bayes_sklearn.py` | `naive_bayes_numpy.py` | `naive_bayes_pytorch.py` |
| KNN | `knn_sklearn.py` | `knn_numpy.py` | `knn_pytorch.py` |

### 3. Clustering (9 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| K-Means | `kmeans_sklearn.py` | `kmeans_numpy.py` | `kmeans_pytorch.py` |
| DBSCAN | `dbscan_sklearn.py` | `dbscan_numpy.py` | `dbscan_pytorch.py` |
| Hierarchical | `hierarchical_clustering_sklearn.py` | `hierarchical_clustering_numpy.py` | `hierarchical_clustering_pytorch.py` |

### 4. Time Series (15 files)
| Algorithm | statsmodels/sklearn | NumPy | PyTorch |
|-----------|-------------------|-------|---------|
| ARIMA | `arima_statsmodels.py` | `arima_numpy.py` | `arima_pytorch.py` |
| SARIMA | `sarima_statsmodels.py` | `sarima_numpy.py` | `sarima_pytorch.py` |
| Prophet | `prophet_sklearn.py` | `prophet_numpy.py` | `prophet_pytorch.py` |
| LSTM | `lstm_sklearn.py` | `lstm_numpy.py` | `lstm_pytorch.py` |
| Temporal Fusion Transformer | `temporal_fusion_transformer_sklearn.py` | `temporal_fusion_transformer_numpy.py` | `temporal_fusion_transformer_pytorch.py` |

### 5. NLP (9 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| TF-IDF + Classifier | `tfidf_classifier_sklearn.py` | `tfidf_classifier_numpy.py` | `tfidf_classifier_pytorch.py` |
| BERT Classifier | `bert_classifier_sklearn.py` | `bert_classifier_numpy.py` | `bert_classifier_pytorch.py` |
| LLM / RAG | `llm_rag_sklearn.py` | `llm_rag_numpy.py` | `llm_rag_pytorch.py` |

### 6. Computer Vision (9 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| CNN | `cnn_sklearn.py` | `cnn_numpy.py` | `cnn_pytorch.py` |
| ResNet | `resnet_sklearn.py` | `resnet_numpy.py` | `resnet_pytorch.py` |
| EfficientNet | `efficientnet_sklearn.py` | `efficientnet_numpy.py` | `efficientnet_pytorch.py` |

### 7. Recommendation (6 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| Collaborative Filtering | `collaborative_filtering_sklearn.py` | `collaborative_filtering_numpy.py` | `collaborative_filtering_pytorch.py` |
| Matrix Factorization | `matrix_factorization_sklearn.py` | `matrix_factorization_numpy.py` | `matrix_factorization_pytorch.py` |

### 8. Anomaly Detection (6 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| Isolation Forest | `isolation_forest_sklearn.py` | `isolation_forest_numpy.py` | `isolation_forest_pytorch.py` |
| Autoencoder | `autoencoder_sklearn.py` | `autoencoder_numpy.py` | `autoencoder_pytorch.py` |

### 9. Dimensionality Reduction (6 files)
| Algorithm | sklearn | NumPy | PyTorch |
|-----------|---------|-------|---------|
| PCA | `pca_sklearn.py` | `pca_numpy.py` | `pca_pytorch.py` |
| t-SNE | `tsne_sklearn.py` | `tsne_numpy.py` | `tsne_pytorch.py` |

## How to Use

### Setup
```bash
pip install -r requirements.txt
```

### Run any algorithm
```bash
# Run a specific algorithm
python 01_regression/linear_regression_sklearn.py

# Each file runs a full pipeline:
# 1. Generate synthetic data
# 2. Train with default hyperparameters
# 3. Validate on validation set
# 4. Compare parameter sets (with reasoning)
# 5. Optuna hyperparameter optimization
# 6. Ray Tune hyperparameter search
# 7. Train with best parameters
# 8. Final test evaluation
# 9. Real-world demo with domain-specific data
```

### File Structure Pattern
Every file follows this consistent structure:
```python
"""
Algorithm Name - Framework Implementation
==========================================
Theory & Mathematics: ...
Business Use Cases: ...
Advantages / Disadvantages: ...
Hyperparameters: ...
"""

def generate_data(...)         # Create synthetic dataset
def train(...)                 # Train the model
def validate(...)              # Validate on validation set
def test(...)                  # Final test evaluation
def optuna_objective(...)      # Optuna HPO objective
def ray_tune_search(...)       # Ray Tune HPO
def compare_parameter_sets(...) # Compare configs with reasoning
def real_world_demo(...)       # Domain-specific use case
def main(...)                  # Run full pipeline
```

## Three Implementation Variants

| Variant | Purpose | Best For |
|---------|---------|----------|
| **sklearn** | Production-ready, well-optimized | Quick prototyping, baselines, production |
| **NumPy** | From-scratch implementation | Learning algorithm internals, understanding math |
| **PyTorch** | GPU-accelerated, differentiable | Deep learning integration, custom modifications |

## Key Features

- **Hyperparameter Optimization**: Every algorithm includes both Optuna (Bayesian) and Ray Tune (distributed) HPO
- **Parameter Comparison**: `compare_parameter_sets()` runs multiple configurations and explains trade-offs
- **Real-World Demos**: Domain-specific examples (medical diagnosis, fraud detection, customer churn, etc.)
- **Heavy Commenting**: Line-by-line explanations of WHAT each line does and WHY that approach was chosen
- **Metrics**: Appropriate metrics per task type (MSE/R2 for regression, accuracy/F1/AUC for classification, silhouette for clustering, etc.)
