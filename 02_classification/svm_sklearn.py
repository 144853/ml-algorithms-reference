"""
Support Vector Machine - Scikit-Learn Implementation
=====================================================

Theory & Mathematics:
    SVMs find the hyperplane that maximizes the margin between classes.

    Linear SVM (Hard Margin):
        Minimize:   (1/2) * ||w||^2
        Subject to: y_i * (w^T x_i + b) >= 1 for all i

    Soft Margin SVM (allowing misclassification):
        Minimize:   (1/2) * ||w||^2 + C * sum(xi_i)
        Subject to: y_i * (w^T x_i + b) >= 1 - xi_i, xi_i >= 0

    Kernel Trick:
        - Linear: K(x, y) = x^T y
        - RBF (Gaussian): K(x, y) = exp(-gamma * ||x - y||^2)
        - Polynomial: K(x, y) = (gamma * x^T y + r)^d

    Hinge Loss:
        L = max(0, 1 - y_i * f(x_i))
        Total: (1/2) * ||w||^2 + C * sum max(0, 1 - y_i * f(x_i))

Business Use Cases:
    - Text classification and document categorization
    - Image recognition and handwritten digit classification
    - Protein structure prediction in bioinformatics
    - Email spam detection

Hyperparameters:
    - C: Regularization parameter (penalty for misclassification)
    - kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
    - gamma: Kernel coefficient for RBF/poly/sigmoid
    - degree: Degree of polynomial kernel
"""

import logging
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from sklearn.datasets import make_classification
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # sklearn's SVM implementation using libsvm.
# WHY SVC: Wraps libsvm C++ code. Supports kernel trick, probability estimates via Platt scaling.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def generate_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    """Generate synthetic data. SVM is very sensitive to feature scaling."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features//2,
                               n_redundant=n_features//4, n_classes=n_classes, random_state=random_state)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)
    # SVM is VERY sensitive to feature scaling because it maximizes geometric margin.
    # WHY: Features on different scales would distort the distance metric,
    # making the margin computation meaningless.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on train only (prevent data leakage).
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    logger.info("Data generated: train=%d, val=%d, test=%d", X_train.shape[0], X_val.shape[0], X_test.shape[0])
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(X_train, y_train, **hyperparams):
    """Train a Support Vector Machine classifier."""
    defaults = dict(
        C=1.0,            # Regularization: higher C = less regularization = tighter fit.
        # WHY C=1.0: Balanced default. Low C (0.01) = wide margin with more errors.
        # High C (100) = narrow margin with fewer training errors but risk of overfitting.
        kernel="rbf",     # Radial Basis Function kernel.
        # WHY RBF: Most versatile kernel. Can model complex nonlinear boundaries.
        # Equivalent to infinite-dimensional feature space.
        gamma="scale",    # Gamma = 1 / (n_features * X.var()).
        # WHY scale: Adapts to the data's variance. 'auto' uses 1/n_features.
        # Small gamma = wide Gaussian = smooth boundary. Large gamma = tight = overfit.
        degree=3,         # Degree for polynomial kernel (only used when kernel='poly').
        probability=True, # Enable probability estimates via Platt scaling.
        # WHY True: Needed for AUC-ROC computation. Adds overhead (~2x training time)
        # because it fits a logistic regression on decision values via cross-validation.
        random_state=42,
    )
    defaults.update(hyperparams)
    model = SVC(**defaults)
    model.fit(X_train, y_train)
    n_sv = model.n_support_  # Number of support vectors per class.
    # WHY log support vectors: They define the decision boundary.
    # Fewer SVs = simpler model. Many SVs = complex boundary or noisy data.
    logger.info("SVM trained: kernel=%s, C=%.3f, support_vectors=%s (total=%d)",
                defaults["kernel"], defaults["C"], n_sv, sum(n_sv))
    return model


def _evaluate(model, X, y):
    y_pred, y_proba = model.predict(X), model.predict_proba(X)
    n_classes = len(np.unique(y))
    auc = roc_auc_score(y, y_proba[:, 1]) if n_classes == 2 else roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
    return {"accuracy": accuracy_score(y, y_pred), "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0), "f1": f1_score(y, y_pred, average="weighted", zero_division=0), "auc_roc": auc}

def validate(model, X_val, y_val):
    metrics = _evaluate(model, X_val, y_val)
    logger.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics

def test(model, X_test, y_test):
    metrics = _evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    logger.info("Support vectors per class: %s", model.n_support_)
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter Comparison
# ---------------------------------------------------------------------------

def compare_parameter_sets(X_train, y_train, X_val, y_val):
    """Compare SVM kernel and regularization configurations."""
    configs = {
        "Linear Kernel": {
            # Config 1: Linear SVM. No kernel trick; finds a straight hyperplane.
            # WHY: Best for linearly separable data or high-dimensional sparse data (text).
            # Fastest to train. Most interpretable (weight vector gives feature importance).
            # Best for: text classification, high-dimensional data with n_features >> n_samples.
            "C": 1.0, "kernel": "linear",
        },
        "RBF Kernel (tight)": {
            # Config 2: RBF with high C and moderate gamma.
            # WHY: High C (10) allows very few margin violations (tight fit).
            # gamma=0.1 creates a moderately narrow Gaussian, capturing local patterns.
            # Risk: May overfit by creating highly complex decision boundaries.
            # Best for: when training accuracy matters and you have enough data.
            "C": 10.0, "kernel": "rbf", "gamma": 0.1,
        },
        "RBF Kernel (wide)": {
            # Config 3: RBF with moderate C and small gamma.
            # WHY: C=1 allows more margin violations (smoother boundary).
            # gamma=0.01 creates a wide Gaussian, producing a very smooth boundary.
            # Risk: May underfit complex patterns by over-smoothing.
            # Best for: noisy data where you want a conservative, smooth boundary.
            "C": 1.0, "kernel": "rbf", "gamma": 0.01,
        },
        "Polynomial (degree=3)": {
            # Config 4: Polynomial kernel of degree 3.
            # WHY: Captures polynomial feature interactions explicitly.
            # degree=3 finds cubic relationships between features.
            # Better than RBF when you know the data has polynomial structure.
            # Best for: problems where feature interactions follow polynomial patterns.
            "C": 1.0, "kernel": "poly", "degree": 3,
        },
    }
    print("\n" + "=" * 90)
    print("SUPPORT VECTOR MACHINE - HYPERPARAMETER COMPARISON")
    print("=" * 90)
    print(f"{'Config':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'SVs':>8}")
    print("-" * 90)
    for name, params in configs.items():
        model = train(X_train, y_train, **params)
        metrics = _evaluate(model, X_val, y_val)
        n_sv = sum(model.n_support_)
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f} {n_sv:>8}")
    print("-" * 90)
    print("INTERPRETATION:")
    print("  - Linear: Fastest, most interpretable. Use for high-dim or linearly separable data.")
    print("  - RBF (tight): Complex boundary, risk of overfitting. High C = few margin violations.")
    print("  - RBF (wide): Smooth boundary, may underfit. Small gamma = wide influence radius.")
    print("  - Polynomial: Explicit polynomial interactions. Good when structure is known.")
    print("  - Fewer SVs = simpler model. Many SVs = data is hard to separate or noisy.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Real-World Demo: Email Spam Detection
# ---------------------------------------------------------------------------

def real_world_demo():
    """Demonstrate SVM on email spam detection.

    WHY this scenario: SVM was the state-of-the-art for spam detection for decades
    because of its effectiveness in high-dimensional feature spaces (bag-of-words).
    Domain context: Features represent word frequencies and email statistics.
    """
    print("\n" + "=" * 90)
    print("REAL-WORLD DEMO: Email Spam Detection (SVM)")
    print("=" * 90)
    np.random.seed(42)
    n_samples = 2000

    # Generate features inspired by the classic Spambase dataset.
    # WHY these features: They represent characteristics of spam vs legitimate emails.
    word_freq_free = np.random.exponential(0.3, n_samples)     # Frequency of "free" (spammy word).
    word_freq_money = np.random.exponential(0.2, n_samples)    # Frequency of "money".
    word_freq_offer = np.random.exponential(0.15, n_samples)   # Frequency of "offer".
    word_freq_click = np.random.exponential(0.25, n_samples)   # Frequency of "click".
    capital_run_avg = np.random.exponential(3, n_samples)      # Avg length of CAPITAL letter runs.
    # WHY capital runs: Spam emails often use excessive CAPITALIZATION.
    capital_run_max = np.random.exponential(20, n_samples)     # Max length of CAPITAL runs.
    char_freq_excl = np.random.exponential(0.3, n_samples)     # Frequency of '!' characters.
    # WHY exclamation marks: Spam tends to use excessive punctuation (BUY NOW!!!).
    char_freq_dollar = np.random.exponential(0.1, n_samples)   # Frequency of '$' characters.

    # Generate spam label based on realistic patterns.
    spam_score = (
        1.5 * word_freq_free + 2.0 * word_freq_money + 1.0 * word_freq_offer
        + 1.2 * word_freq_click + 0.1 * capital_run_avg + 0.02 * capital_run_max
        + 1.5 * char_freq_excl + 2.0 * char_freq_dollar
        - 2.5 + np.random.normal(0, 1, n_samples)
    )
    y = (np.random.random(n_samples) < 1 / (1 + np.exp(-spam_score))).astype(int)
    X = np.column_stack([word_freq_free, word_freq_money, word_freq_offer, word_freq_click,
                         capital_run_avg, capital_run_max, char_freq_excl, char_freq_dollar])
    feature_names = ["word_freq_free", "word_freq_money", "word_freq_offer", "word_freq_click",
                     "capital_run_avg", "capital_run_max", "char_freq_excl", "char_freq_dollar"]

    print(f"\nDataset: {n_samples} emails, {len(feature_names)} features")
    print(f"Spam rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM with RBF kernel (classic choice for spam detection).
    model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\n--- Performance ---")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}  F1: {f1_score(y_test, y_pred):.4f}  AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
    print(f"  Support vectors: {sum(model.n_support_)} out of {len(y_train)} training samples")
    print(f"\n--- Business Insights ---")
    print("  1. SVM excels at spam detection due to effectiveness in high-dimensional text features.")
    print("  2. Support vectors are the 'borderline' emails that define the spam/ham boundary.")
    print("  3. RBF kernel captures nonlinear word frequency patterns that linear models miss.")
    print("  4. In production, a linear SVM would be faster for real-time email filtering.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Hyperparameter Optimization & Main
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    params = {"kernel": kernel, "C": C}
    if kernel == "rbf": params["gamma"] = trial.suggest_float("gamma", 1e-5, 10.0, log=True)
    elif kernel == "poly":
        params["gamma"] = trial.suggest_float("gamma_poly", 1e-5, 10.0, log=True)
        params["degree"] = trial.suggest_int("degree", 2, 5)
    return validate(train(X_train, y_train, **params), X_val, y_val)["f1"]

def ray_tune_search(X_train, y_train, X_val, y_val, num_samples=20):
    import ray; from ray import tune as ray_tune
    if not ray.is_initialized(): ray.init(ignore_reinit_error=True, log_to_driver=False)
    def _t(config): metrics = validate(train(X_train, y_train, **config), X_val, y_val); ray_tune.report(f1=metrics["f1"], accuracy=metrics["accuracy"])
    space = {"C": ray_tune.loguniform(1e-3, 100.0), "kernel": ray_tune.choice(["linear", "rbf", "poly"]), "gamma": ray_tune.loguniform(1e-5, 10.0)}
    tuner = ray_tune.Tuner(_t, param_space=space, tune_config=ray_tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"))
    best = tuner.fit().get_best_result(metric="f1", mode="max"); logger.info("Best: %s", best.config); ray.shutdown(); return best.config

def main():
    logger.info("=" * 70); logger.info("Support Vector Machine - Scikit-Learn Implementation"); logger.info("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()
    logger.info("\n--- Baseline ---"); model = train(X_train, y_train); validate(model, X_val, y_val)
    compare_parameter_sets(X_train, y_train, X_val, y_val)
    real_world_demo()
    logger.info("\n--- Optuna ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: optuna_objective(t, X_train, y_train, X_val, y_val), n_trials=30, show_progress_bar=True)
    logger.info("Best: %s  F1: %.4f", study.best_params, study.best_value)
    best_params = {("gamma" if k == "gamma_poly" else k): v for k, v in study.best_params.items()}
    best_model = train(X_train, y_train, **best_params)
    logger.info("\n--- Final Test ---"); test(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
