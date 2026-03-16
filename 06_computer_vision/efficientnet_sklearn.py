"""
EfficientNet Feature Extraction + Sklearn Classifier Implementation
====================================================================

Architecture Overview:
    This module uses EfficientNet-like feature extraction combined with
    traditional sklearn classifiers. EfficientNet features are extracted
    using either a pretrained model or a simulated feature extractor
    that mimics EfficientNet's key innovations.

    Pipeline:
    1. Images -> 2. EfficientNet Feature Extraction
    -> 3. Feature Vector -> 4. StandardScaler -> 5. Sklearn Classifier

    EfficientNet Key Components (simulated):
    - Depthwise Separable Convolutions (factored convolutions)
    - Squeeze-and-Excitation blocks (channel attention)
    - Compound scaling (balanced width/depth/resolution)

Theory & Mathematics:
    Depthwise Separable Convolution:
        Standard conv: C_out * C_in * k * k * H * W FLOPs
        Depthwise separable:
            1. Depthwise: C_in * k * k * H * W (apply one filter per channel)
            2. Pointwise: C_out * C_in * H * W  (1x1 conv to combine channels)
        Total: C_in * (k*k + C_out) * H * W
        Reduction factor: ~k^2 (roughly 8-9x fewer FLOPs for 3x3 kernels)

    Squeeze-and-Excitation (SE):
        1. Squeeze: Global Average Pooling -> (1, C, 1, 1)
        2. Excitation: FC(C, C//r) -> ReLU -> FC(C//r, C) -> Sigmoid
        3. Scale: Multiply original features by channel weights
        This learns channel-wise attention: which channels are important.

    Compound Scaling:
        depth = alpha^phi, width = beta^phi, resolution = gamma^phi
        subject to: alpha * beta^2 * gamma^2 ~= 2
        This balances network depth, width, and input resolution.

Business Use Cases:
    - Mobile and edge deployment (efficiency matters)
    - High-throughput image classification
    - Transfer learning with efficient backbones
    - Medical imaging with computational constraints
    - Batch processing of large image datasets

Advantages:
    - EfficientNet features capture rich representations efficiently
    - Sklearn classifiers run on CPU without GPU
    - Fast to train once features are extracted
    - Good for small labeled datasets with pretrained features
    - Combines efficiency of EfficientNet with interpretability of sklearn

Disadvantages:
    - Feature extraction still needs forward pass through deep model
    - Fixed features (no fine-tuning)
    - May not capture domain-specific patterns
    - Simulated features are a rough approximation
    - Feature dimensionality may be high

Key Hyperparameters:
    Feature Extraction:
        - feature_dim: Output feature dimension (64-512)
        - se_ratio: Squeeze-excitation reduction ratio (4-16)
    SVM:
        - C: Regularization (0.01-100)
        - kernel: 'rbf', 'linear'
    Random Forest:
        - n_estimators: 50-500
        - max_depth: 5-50

References:
    - Tan, M. and Le, Q.V. (2019). "EfficientNet: Rethinking Model Scaling
      for Convolutional Neural Networks." ICML. arXiv:1905.11946.
    - Howard, A.G. et al. (2017). "MobileNets: Efficient Convolutional
      Neural Networks for Mobile Vision Applications." arXiv:1704.04861.
    - Hu, J. et al. (2018). "Squeeze-and-Excitation Networks." CVPR.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any, Optional

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    from ray import tune

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 2000,
    img_size: int = 32,
    n_channels: int = 3,
    n_classes: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic RGB images with class-specific patterns.

    Returns X: (N, C, H, W), y: (N,).
    """
    rng = np.random.RandomState(random_state)
    spc = n_samples // n_classes
    X_list, y_list = [], []

    for cls in range(n_classes):
        for _ in range(spc):
            img = rng.rand(n_channels, img_size, img_size).astype(np.float32) * 0.15
            ch = cls % n_channels

            if cls == 0:
                for r in range(0, img_size, 4):
                    img[ch, r : r + 2, :] += 0.7
            elif cls == 1:
                for c in range(0, img_size, 4):
                    img[ch, :, c : c + 2] += 0.7
            elif cls == 2:
                cx, cy = img_size // 2, img_size // 2
                yy, xx = np.ogrid[:img_size, :img_size]
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (img_size // 4) ** 2
                img[ch][mask] += 0.8
            elif cls == 3:
                for i in range(img_size):
                    img[ch, i, i % img_size] += 0.7
            elif cls == 4:
                block = 4
                for r in range(0, img_size, block):
                    for c in range(0, img_size, block):
                        if ((r // block) + (c // block)) % 2 == 0:
                            img[ch, r : r + block, c : c + block] += 0.6
            elif cls == 5:
                grad = np.linspace(0, 1, img_size).reshape(1, -1).astype(np.float32)
                img[ch] += grad * 0.7
            elif cls == 6:
                grad = np.linspace(0, 1, img_size).reshape(-1, 1).astype(np.float32)
                img[ch] += grad * 0.7
            elif cls == 7:
                mid = img_size // 2
                img[ch, mid - 2 : mid + 2, :] += 0.7
                img[ch, :, mid - 2 : mid + 2] += 0.7
            elif cls == 8:
                q = img_size // 4
                img[ch, :q, :q] += 0.8
                img[ch, -q:, -q:] += 0.8
            elif cls == 9:
                b = 3
                img[ch, :b, :] += 0.7
                img[ch, -b:, :] += 0.7
                img[ch, :, :b] += 0.7
                img[ch, :, -b:] += 0.7

            img += rng.randn(n_channels, img_size, img_size).astype(np.float32) * 0.05
            img = np.clip(img, 0, 1)
            X_list.append(img)
            y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# EfficientNet-like Feature Extractor (Simulated)
# ---------------------------------------------------------------------------


class EfficientNetFeatureExtractor:
    """
    Simulated EfficientNet feature extractor using depthwise separable
    convolutions and squeeze-excitation blocks implemented from scratch.

    This produces fixed features (like a pretrained model would) by using
    pre-generated random filters that remain consistent across calls.

    Parameters
    ----------
    feature_dim : int
        Output feature dimension.
    se_ratio : int
        Squeeze-excitation reduction ratio.
    random_state : int
        Random seed for filter generation.
    """

    def __init__(self, feature_dim: int = 128, se_ratio: int = 4, random_state: int = 42):
        self.feature_dim = feature_dim
        self.se_ratio = se_ratio
        rng = np.random.RandomState(random_state)

        # Block 1: Depthwise separable conv (3 channels -> 16 channels)
        self.dw_filters1 = rng.randn(3, 3, 3).astype(np.float32) * 0.3  # 3 channels, 3x3 each
        self.pw_filters1 = rng.randn(16, 3, 1, 1).astype(np.float32) * 0.2  # 3->16 pointwise

        # SE block 1 weights
        self.se1_fc1 = rng.randn(16, 16 // se_ratio).astype(np.float32) * 0.3
        self.se1_fc2 = rng.randn(16 // se_ratio, 16).astype(np.float32) * 0.3

        # Block 2: Depthwise separable conv (16 -> 32 channels)
        self.dw_filters2 = rng.randn(16, 3, 3).astype(np.float32) * 0.2
        self.pw_filters2 = rng.randn(32, 16, 1, 1).astype(np.float32) * 0.15

        # SE block 2 weights
        self.se2_fc1 = rng.randn(32, 32 // se_ratio).astype(np.float32) * 0.3
        self.se2_fc2 = rng.randn(32 // se_ratio, 32).astype(np.float32) * 0.3

        # Final projection
        self.proj = None  # Will be set dynamically

    def _depthwise_conv(self, X: np.ndarray, dw_filters: np.ndarray) -> np.ndarray:
        """
        Apply depthwise convolution: one filter per input channel.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
        dw_filters : np.ndarray of shape (C, kH, kW)

        Returns
        -------
        out : np.ndarray of shape (N, C, H-kH+1, W-kW+1)
        """
        N, C, H, W = X.shape
        kH, kW = dw_filters.shape[1], dw_filters.shape[2]
        Ho, Wo = H - kH + 1, W - kW + 1
        out = np.zeros((N, C, Ho, Wo), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        out[n, c, i, j] = np.sum(X[n, c, i : i + kH, j : j + kW] * dw_filters[c])
        return out

    def _pointwise_conv(self, X: np.ndarray, pw_filters: np.ndarray) -> np.ndarray:
        """
        Apply pointwise (1x1) convolution: mix channels.

        Parameters
        ----------
        X : np.ndarray of shape (N, C_in, H, W)
        pw_filters : np.ndarray of shape (C_out, C_in, 1, 1)

        Returns
        -------
        out : np.ndarray of shape (N, C_out, H, W)
        """
        N, C_in, H, W = X.shape
        C_out = pw_filters.shape[0]
        # Reshape for efficient matrix multiplication
        X_flat = X.reshape(N, C_in, H * W)  # (N, C_in, H*W)
        pw_flat = pw_filters.reshape(C_out, C_in)  # (C_out, C_in)
        out_flat = np.einsum("ij,njk->nik", pw_flat, X_flat)  # (N, C_out, H*W)
        return out_flat.reshape(N, C_out, H, W)

    def _squeeze_excitation(self, X: np.ndarray, fc1: np.ndarray, fc2: np.ndarray) -> np.ndarray:
        """
        Apply Squeeze-and-Excitation attention.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
        fc1, fc2 : weight matrices for the two FC layers

        Returns
        -------
        out : np.ndarray of shape (N, C, H, W)
        """
        N, C, H, W = X.shape

        # Squeeze: Global Average Pooling
        squeezed = X.mean(axis=(2, 3))  # (N, C)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excited = np.maximum(0, squeezed @ fc1)  # (N, C//r)
        excited = 1.0 / (1.0 + np.exp(-excited @ fc2))  # (N, C) sigmoid

        # Scale: channel-wise multiplication
        scale = excited.reshape(N, C, 1, 1)
        return X * scale

    def _relu(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    def _avg_pool(self, X: np.ndarray, pool_size: int = 2) -> np.ndarray:
        N, C, H, W = X.shape
        Ho, Wo = H // pool_size, W // pool_size
        out = np.zeros((N, C, Ho, Wo), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        out[n, c, i, j] = np.mean(
                            X[n, c, i * pool_size : (i + 1) * pool_size, j * pool_size : (j + 1) * pool_size]
                        )
        return out

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extract EfficientNet-like features.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)

        Returns
        -------
        features : np.ndarray of shape (N, feature_dim)
        """
        # Block 1: Depthwise Separable Conv + SE
        h = self._depthwise_conv(X, self.dw_filters1)
        h = self._relu(h)
        h = self._pointwise_conv(h, self.pw_filters1)
        h = self._relu(h)
        h = self._squeeze_excitation(h, self.se1_fc1, self.se1_fc2)
        h = self._avg_pool(h, 2)

        # Block 2: Depthwise Separable Conv + SE
        h = self._depthwise_conv(h, self.dw_filters2)
        h = self._relu(h)
        h = self._pointwise_conv(h, self.pw_filters2)
        h = self._relu(h)
        h = self._squeeze_excitation(h, self.se2_fc1, self.se2_fc2)
        h = self._avg_pool(h, 2)

        # Global Average Pooling
        h = h.mean(axis=(2, 3))  # (N, 32)

        # Project to desired dimension
        flat_dim = h.shape[1]
        if self.proj is None or self.proj.shape[0] != flat_dim:
            rng = np.random.RandomState(42)
            self.proj = rng.randn(flat_dim, self.feature_dim).astype(np.float32) * 0.1

        features = h @ self.proj
        return features


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_type: str = "svm",
    feature_dim: int = 128,
    se_ratio: int = 4,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    use_pca: bool = False,
    pca_components: int = 64,
) -> Dict[str, Any]:
    """
    Train EfficientNet features + sklearn classifier.

    Returns dict with pipeline and extraction settings.
    """
    print(f"Extracting EfficientNet-like features (dim={feature_dim}, se_ratio={se_ratio})...")
    extractor = EfficientNetFeatureExtractor(feature_dim=feature_dim, se_ratio=se_ratio)
    X_feat = extractor.extract(X_train)

    steps = [("scaler", StandardScaler())]
    if use_pca:
        n_comp = min(pca_components, X_feat.shape[1], X_feat.shape[0])
        steps.append(("pca", PCA(n_components=n_comp)))

    if classifier_type == "svm":
        steps.append(("clf", SVC(C=C, kernel=kernel, gamma=gamma, random_state=42, max_iter=5000)))
        print(f"Training SVM (C={C}, kernel={kernel})...")
    elif classifier_type == "rf":
        steps.append(
            ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1))
        )
        print(f"Training RF (n_estimators={n_estimators})...")
    elif classifier_type == "gbm":
        steps.append(
            ("clf", GradientBoostingClassifier(n_estimators=min(n_estimators, 100), max_depth=min(max_depth or 5, 10), random_state=42))
        )
        print(f"Training GBM (n_estimators={min(n_estimators, 100)})...")
    else:
        steps.append(("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)))
        print(f"Training Logistic Regression (C={C})...")

    pipeline = Pipeline(steps)
    pipeline.fit(X_feat, y_train)
    train_acc = pipeline.score(X_feat, y_train)
    print(f"Training accuracy: {train_acc:.4f}")

    return {
        "pipeline": pipeline,
        "extractor": extractor,
        "feature_dim": feature_dim,
        "se_ratio": se_ratio,
    }


def validate(
    model_dict: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Validate the trained pipeline."""
    extractor = model_dict["extractor"]
    pipeline = model_dict["pipeline"]

    X_feat = extractor.extract(X_val)
    y_pred = pipeline.predict(X_feat)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None, zero_division=0)
    report = classification_report(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    print(f"Validation Accuracy: {acc:.4f}")
    print(report)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }


def test(
    model_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Final test evaluation."""
    print("\n--- Test Set Evaluation ---")
    return validate(model_dict, X_test, y_test)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: "optuna.Trial",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for EfficientNet+sklearn tuning."""
    classifier_type = trial.suggest_categorical("classifier_type", ["svm", "rf", "logistic"])
    feature_dim = trial.suggest_categorical("feature_dim", [64, 128, 256])
    se_ratio = trial.suggest_categorical("se_ratio", [2, 4, 8])

    kwargs = {
        "classifier_type": classifier_type,
        "feature_dim": feature_dim,
        "se_ratio": se_ratio,
    }

    if classifier_type == "svm":
        kwargs["C"] = trial.suggest_float("svm_C", 0.01, 100.0, log=True)
        kwargs["kernel"] = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
    elif classifier_type == "rf":
        kwargs["n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        kwargs["max_depth"] = trial.suggest_int("rf_max_depth", 5, 40)
    else:
        kwargs["C"] = trial.suggest_float("lr_C", 0.01, 100.0, log=True)

    md = train(X_train, y_train, **kwargs)
    metrics = validate(md, X_val, y_val)
    return metrics["accuracy"]


def ray_tune_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 8,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search."""
    if not RAY_AVAILABLE:
        print("Ray not installed. Skipping.")
        return {}

    def trainable(config):
        md = train(
            X_train, y_train,
            classifier_type=config["classifier_type"],
            feature_dim=config["feature_dim"],
            se_ratio=config["se_ratio"],
            C=config.get("C", 1.0),
            kernel=config.get("kernel", "rbf"),
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
        )
        metrics = validate(md, X_val, y_val)
        tune.report({"accuracy": metrics["accuracy"]})

    search_space = {
        "classifier_type": tune.choice(["svm", "rf", "logistic"]),
        "feature_dim": tune.choice([64, 128, 256]),
        "se_ratio": tune.choice([2, 4, 8]),
        "C": tune.loguniform(0.01, 100.0),
        "kernel": tune.choice(["rbf", "linear"]),
        "n_estimators": tune.choice([50, 100, 200]),
        "max_depth": tune.choice([10, 20, None]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, metric="accuracy", mode="max"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config: {best.config}")
    print(f"Best accuracy: {best.metrics['accuracy']:.4f}")
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the full EfficientNet + sklearn pipeline."""
    print("=" * 70)
    print("EfficientNet Feature Extraction + Sklearn Classifier Pipeline")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/6] Generating synthetic data...")
    X, y = generate_data(n_samples=2000, img_size=32, n_classes=10)
    print(f"    Dataset: {X.shape}")

    # 2. Split
    print("\n[2/6] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Train
    print("\n[3/6] Training SVM with EfficientNet features...")
    model_dict = train(X_train, y_train, classifier_type="svm", feature_dim=128, C=10.0)

    # 4. Validate
    print("\n[4/6] Validating...")
    val_metrics = validate(model_dict, X_val, y_val)

    # 5. Optuna
    print("\n[5/6] Optuna hyperparameter search...")
    if OPTUNA_AVAILABLE:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=12,
        )
        print(f"    Best accuracy: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        bp = study.best_params
        retrain_kwargs = {
            "classifier_type": bp["classifier_type"],
            "feature_dim": bp["feature_dim"],
            "se_ratio": bp["se_ratio"],
        }
        if bp["classifier_type"] == "svm":
            retrain_kwargs["C"] = bp["svm_C"]
            retrain_kwargs["kernel"] = bp["svm_kernel"]
        elif bp["classifier_type"] == "rf":
            retrain_kwargs["n_estimators"] = bp["rf_n_estimators"]
            retrain_kwargs["max_depth"] = bp["rf_max_depth"]
        else:
            retrain_kwargs["C"] = bp["lr_C"]

        model_dict = train(X_train, y_train, **retrain_kwargs)
    else:
        print("    Optuna not installed, skipping.")

    # 6. Test
    print("\n[6/6] Test evaluation...")
    test_metrics = test(model_dict, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
