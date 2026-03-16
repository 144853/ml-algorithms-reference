"""
ResNet Feature Extraction + Sklearn Classifier Implementation
=============================================================

Architecture Overview:
    This module uses a pretrained or simulated ResNet model as a fixed
    feature extractor, then feeds those features into traditional sklearn
    classifiers (SVM, Random Forest, Logistic Regression).

    Pipeline:
    1. Images -> 2. ResNet Feature Extraction (pretrained or simulated)
    -> 3. Feature Vector (512-d or 2048-d) -> 4. Sklearn Classifier

    ResNet Feature Extraction:
    - In production: Use torchvision.models.resnet18 (pretrained=True),
      remove the final FC layer, use the 512-d average pool output.
    - In this demo: We simulate ResNet-like features using a series
      of convolutional feature extractions or use the actual pretrained
      model if available.

Theory & Mathematics:
    Residual Learning:
        The key insight of ResNet is the residual connection:
        y = F(x) + x    (identity shortcut)

        Where F(x) is the residual mapping learned by stacked conv layers.
        This allows gradients to flow directly through the identity path,
        solving the vanishing gradient problem in very deep networks.

    Feature Extraction for Transfer Learning:
        1. Load pretrained ResNet (trained on ImageNet with 1000 classes)
        2. Remove the final classification layer
        3. Pass images through the truncated network
        4. The output is a rich feature representation
        5. Train a simple classifier on these features

        This works because early layers learn universal features
        (edges, textures) while deeper layers learn task-specific features.

    Why Sklearn on Top of Deep Features:
        - Deep features from ResNet capture hierarchical visual patterns
        - SVM can find optimal linear or non-linear decision boundaries
          in the deep feature space
        - RF can capture feature interactions
        - Often works surprisingly well with small datasets

Business Use Cases:
    - Transfer learning when labeled data is scarce
    - Rapid prototyping: quickly test if visual features are discriminative
    - Edge deployment: extract features once, deploy lightweight classifier
    - Medical imaging with limited labeled examples
    - Fine-grained classification (species, product variants)

Advantages:
    - Leverages powerful pretrained representations
    - No GPU needed for classifier training
    - Works well with small labeled datasets (few hundred images)
    - Fast iteration: try different classifiers without retraining CNN
    - Interpretable classifier layer (SVM margins, RF feature importances)

Disadvantages:
    - Feature extraction still needs GPU for large datasets
    - Fixed features may not capture domain-specific patterns
    - No end-to-end fine-tuning of representations
    - Feature dimensionality can be high (needs PCA for some classifiers)
    - Pretrained models assume ImageNet-like data distribution

Key Hyperparameters:
    Feature Extraction:
        - resnet_variant: resnet18, resnet34, resnet50 (affects feature dim)
        - pooling: 'avg' or 'max' global pooling
    SVM:
        - C: Regularization (0.001 to 1000)
        - kernel: 'rbf', 'linear', 'poly'
    Random Forest:
        - n_estimators: 50-500
        - max_depth: None or 5-50
    Logistic Regression:
        - C: Inverse regularization strength

References:
    - He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
    - Razavian, A.S. et al. (2014). "CNN Features off-the-shelf:
      An Astounding Baseline for Recognition." CVPR Workshops.
"""

# numpy provides array operations for image manipulation, feature extraction computation,
# and metric calculation before data enters the sklearn pipeline
import numpy as np

# warnings module suppresses sklearn convergence warnings during hyperparameter search
import warnings

# Type hints for self-documenting function signatures and static analysis
from typing import Dict, Tuple, Any, Optional

# SVC implements Support Vector Classification with kernel trick -- SVM excels at
# classifying high-dimensional feature vectors extracted from deep networks because
# the kernel function can find complex decision boundaries in feature space
from sklearn.svm import SVC

# RandomForestClassifier provides an ensemble alternative to SVM -- useful because
# it gives feature importance scores that can reveal which ResNet features matter most
from sklearn.ensemble import RandomForestClassifier

# LogisticRegression provides a linear baseline classifier -- often sufficient for
# deep features because ResNet already learns good linear separability in its feature space
from sklearn.linear_model import LogisticRegression

# StandardScaler normalizes features to zero mean and unit variance -- critical for SVM
# because kernel functions are distance-based, so features on different scales would dominate
from sklearn.preprocessing import StandardScaler

# PCA reduces feature dimensionality -- useful when ResNet features are high-dimensional
# (512 or 2048) and the dataset is small, to prevent overfitting and speed up SVM training
from sklearn.decomposition import PCA

# Comprehensive evaluation metrics for model assessment
from sklearn.metrics import (
    accuracy_score,  # Overall correct predictions / total predictions
    classification_report,  # Per-class precision, recall, F1 in formatted table
    precision_recall_fscore_support,  # Raw per-class metric arrays for programmatic use
    confusion_matrix,  # N x N matrix showing prediction patterns between classes
)

# train_test_split handles stratified random splitting to maintain class balance
from sklearn.model_selection import train_test_split

# Pipeline chains preprocessing steps with the classifier to prevent data leakage
from sklearn.pipeline import Pipeline

# PyTorch is optional -- needed for pretrained ResNet feature extraction but the
# simulated extractor works without it for educational demonstration
try:
    import torch  # Core PyTorch library for tensor operations
    import torch.nn as nn  # Neural network modules (used to modify pretrained ResNet)

    TORCH_AVAILABLE = True  # Flag to conditionally use real pretrained models
except ImportError:
    TORCH_AVAILABLE = False  # Fall back to simulated feature extractor

# Optuna for Bayesian hyperparameter optimization
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Ray Tune for distributed hyperparameter search
try:
    import ray
    from ray import tune

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 2000,  # 2000 samples balances dataset size with feature extraction speed
    img_size: int = 32,  # 32x32 standard size; pretrained ResNet will upscale to 224x224
    n_channels: int = 3,  # RGB channels match ResNet's expected input
    n_classes: int = 10,  # 10-class problem for meaningful multi-class evaluation
    random_state: int = 42,  # Fixed seed for reproducibility
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic RGB images with class-specific patterns.

    Parameters
    ----------
    n_samples : int
        Total samples.
    img_size : int
        Image dimensions.
    n_channels : int
        Number of channels.
    n_classes : int
        Number of classes.
    random_state : int
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_channels, img_size, img_size)
    y : np.ndarray of shape (n_samples,)
    """
    # Isolated random generator for reproducibility
    rng = np.random.RandomState(random_state)

    # Equal samples per class for balanced dataset
    spc = n_samples // n_classes
    X_list, y_list = [], []

    # Generate distinct visual patterns for each class
    for cls in range(n_classes):
        for _ in range(spc):
            # Low-amplitude noise background across all 3 RGB channels
            img = rng.rand(n_channels, img_size, img_size).astype(np.float32) * 0.15

            # Rotate primary channel across classes for cross-channel pattern variation
            ch = cls % n_channels

            if cls == 0:  # Horizontal stripes
                for r in range(0, img_size, 4):
                    img[ch, r : r + 2, :] += 0.7
            elif cls == 1:  # Vertical stripes
                for c in range(0, img_size, 4):
                    img[ch, :, c : c + 2] += 0.7
            elif cls == 2:  # Filled circle
                cx, cy = img_size // 2, img_size // 2
                yy, xx = np.ogrid[:img_size, :img_size]
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (img_size // 4) ** 2
                img[ch][mask] += 0.8
            elif cls == 3:  # Diagonal line
                for i in range(img_size):
                    j = i % img_size
                    img[ch, i, j] += 0.7
                    if j + 1 < img_size:
                        img[ch, i, j + 1] += 0.5
            elif cls == 4:  # Checkerboard
                block = 4
                for r in range(0, img_size, block):
                    for c in range(0, img_size, block):
                        if ((r // block) + (c // block)) % 2 == 0:
                            img[ch, r : r + block, c : c + block] += 0.6
            elif cls == 5:  # Horizontal gradient
                grad = np.linspace(0, 1, img_size).reshape(1, -1).astype(np.float32)
                img[ch] += grad * 0.7
            elif cls == 6:  # Vertical gradient
                grad = np.linspace(0, 1, img_size).reshape(-1, 1).astype(np.float32)
                img[ch] += grad * 0.7
            elif cls == 7:  # Cross pattern
                mid = img_size // 2
                img[ch, mid - 2 : mid + 2, :] += 0.7
                img[ch, :, mid - 2 : mid + 2] += 0.7
            elif cls == 8:  # Corner patches
                q = img_size // 4
                img[ch, :q, :q] += 0.8
                img[ch, -q:, -q:] += 0.8
            elif cls == 9:  # Border frame
                b = 3
                img[ch, :b, :] += 0.7
                img[ch, -b:, :] += 0.7
                img[ch, :, :b] += 0.7
                img[ch, :, -b:] += 0.7

            # Add Gaussian noise for realism
            img += rng.randn(n_channels, img_size, img_size).astype(np.float32) * 0.05
            img = np.clip(img, 0, 1)
            X_list.append(img)
            y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


class SimpleResNetFeatureExtractor:
    """
    Simulated ResNet-like feature extractor using simple convolutions.

    For demo purposes, we apply a series of convolution + pooling operations
    to produce a fixed-size feature vector, mimicking how a pretrained ResNet
    would extract features. The filters are random but fixed (not learned),
    simulating a pretrained network whose weights are frozen during transfer learning.

    In a real ResNet, these filters would have been learned on ImageNet (1.2M images,
    1000 classes) and would encode rich visual patterns like edges, textures, and shapes.
    """

    def __init__(
        self,
        feature_dim: int = 128,  # Output feature vector dimensionality
        random_state: int = 42,  # Seed for reproducible filter initialization
    ):
        self.feature_dim = feature_dim
        rng = np.random.RandomState(random_state)

        # Simulate first conv layer filters: 8 filters, each 3x3 across 3 input channels.
        # In a real ResNet-18, the first layer has 64 filters of size 7x7 with stride 2.
        # We use fewer, smaller filters because our images are only 32x32 (not 224x224).
        self.filters1 = rng.randn(8, 3, 3, 3).astype(np.float32) * 0.3

        # Second conv layer: 16 filters processing the 8-channel output of layer 1.
        # This simulates ResNet's second residual block which increases channel depth.
        self.filters2 = rng.randn(16, 8, 3, 3).astype(np.float32) * 0.2

        # Random projection matrix: maps flattened conv features to the desired feature_dim.
        # This simulates ResNet's global average pooling + FC layer that produces
        # the final feature vector. The input dimension (16*6*6=576) is estimated
        # for a 32x32 input after two conv+pool operations.
        self.proj_matrix = rng.randn(16 * 6 * 6, feature_dim).astype(np.float32) * 0.1

    def _conv2d(self, X: np.ndarray, filters: np.ndarray) -> np.ndarray:
        """
        Apply convolution (cross-correlation) with stride 1, no padding.
        This is a simplified version of the convolution in ResNet's residual blocks.
        """
        N, C, H, W = X.shape  # Batch, Channels, Height, Width
        n_filters, _, kH, kW = filters.shape  # Number of output filters and kernel dims

        # Valid convolution: output shrinks by (kernel_size - 1)
        H_out = H - kH + 1
        W_out = W - kW + 1

        out = np.zeros((N, n_filters, H_out, W_out), dtype=np.float32)

        # Naive convolution loop -- slow but transparent for educational purposes
        for n in range(N):
            for f in range(n_filters):
                for i in range(H_out):
                    for j in range(W_out):
                        # Extract patch and compute dot product with filter
                        out[n, f, i, j] = np.sum(X[n, :, i : i + kH, j : j + kW] * filters[f])
        return out

    def _relu(self, X: np.ndarray) -> np.ndarray:
        """ReLU activation: zero out negative values to introduce non-linearity."""
        return np.maximum(0, X)

    def _avg_pool2d(self, X: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """
        Average pooling: computes mean in each non-overlapping window.
        Unlike max pooling, average pooling preserves more information about
        the feature map distribution, which is preferred in ResNet's final pooling layer.
        """
        N, C, H, W = X.shape
        H_out = H // pool_size
        W_out = W // pool_size
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # Mean of pool_size x pool_size window
                        out[n, c, i, j] = np.mean(
                            X[n, c, i * pool_size : (i + 1) * pool_size, j * pool_size : (j + 1) * pool_size]
                        )
        return out

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of images through simulated ResNet layers.

        This mimics ResNet's feature extraction pipeline:
        conv -> relu -> pool -> conv -> relu -> pool -> flatten -> project

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)

        Returns
        -------
        features : np.ndarray of shape (N, feature_dim)
        """
        # Block 1: simulates ResNet's first residual stage
        h = self._conv2d(X, self.filters1)  # Extract low-level features
        h = self._relu(h)  # Non-linearity
        h = self._avg_pool2d(h, 2)  # Spatial downsampling

        # Block 2: simulates ResNet's second residual stage
        h = self._conv2d(h, self.filters2)  # Extract higher-level features
        h = self._relu(h)
        h = self._avg_pool2d(h, 2)

        # Flatten spatial dimensions into a 1D vector per image
        h = h.reshape(h.shape[0], -1)

        # Adjust projection matrix if flattened dimensions don't match expected size
        # (happens when input image size differs from the default 32x32)
        if h.shape[1] != self.proj_matrix.shape[0]:
            rng = np.random.RandomState(42)
            self.proj_matrix = rng.randn(h.shape[1], self.feature_dim).astype(np.float32) * 0.1

        # Project to fixed-dimension feature vector (simulates ResNet's final FC layer)
        features = h @ self.proj_matrix
        return features


def extract_resnet_features_torch(
    X: np.ndarray,  # Images in NCHW format
    feature_dim: int = 512,  # ResNet-18 produces 512-dim features
) -> np.ndarray:
    """
    Extract features using a real pretrained ResNet18 if torchvision is available.
    Falls back to the simple extractor otherwise.

    In production transfer learning, this is the standard approach:
    load a pretrained model, remove the final classification head, and use
    the penultimate layer's output as a universal feature representation.
    """
    if not TORCH_AVAILABLE:
        # Fall back to simulated features if PyTorch is not installed
        extractor = SimpleResNetFeatureExtractor(feature_dim=feature_dim)
        return extractor.extract(X)

    try:
        from torchvision import models

        # Load ResNet-18 architecture without pretrained weights (weights=None)
        # In production, use weights='IMAGENET1K_V1' for pretrained features
        model = models.resnet18(weights=None)

        # Replace the final FC layer with Identity to get the 512-dim feature vector
        # instead of the 1000-class classification output
        model.fc = nn.Identity()

        # Set to eval mode: disables dropout and uses running stats for BatchNorm
        model.eval()

        features_list = []
        batch_size = 64  # Process in batches to manage memory

        # No gradient computation needed for feature extraction
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i : i + batch_size])

                # ResNet was designed for 224x224 ImageNet images -- upscale if needed.
                # Bilinear interpolation preserves visual content during resizing.
                if batch.shape[-1] < 224:
                    batch = torch.nn.functional.interpolate(
                        batch, size=(224, 224), mode="bilinear", align_corners=False
                    )

                # Forward pass through truncated ResNet produces 512-dim features
                out = model(batch)
                features_list.append(out.numpy())

        return np.concatenate(features_list, axis=0)
    except Exception:
        # Fall back to simulated features if torchvision import or model loading fails
        extractor = SimpleResNetFeatureExtractor(feature_dim=feature_dim)
        return extractor.extract(X)


def extract_features(
    X: np.ndarray,  # Input images in NCHW format
    method: str = "simple",  # 'simple' for simulated, 'pretrained' for real ResNet
    feature_dim: int = 128,  # Feature vector dimensionality
) -> np.ndarray:
    """
    Extract features using specified method.

    Parameters
    ----------
    X : np.ndarray of shape (N, C, H, W)
    method : str
        'simple' for simulated features, 'pretrained' for real ResNet.
    feature_dim : int
        Dimension of output features.

    Returns
    -------
    features : np.ndarray of shape (N, feature_dim)
    """
    if method == "pretrained":
        return extract_resnet_features_torch(X, feature_dim)
    else:
        extractor = SimpleResNetFeatureExtractor(feature_dim=feature_dim)
        return extractor.extract(X)


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,  # Training images (N, C, H, W)
    y_train: np.ndarray,  # Training labels (N,)
    classifier_type: str = "svm",  # 'svm', 'rf', or 'logistic'
    feature_method: str = "simple",  # Feature extraction method
    feature_dim: int = 128,  # Feature vector size
    C: float = 1.0,  # SVM/LogReg regularization parameter
    kernel: str = "rbf",  # SVM kernel function
    gamma: str = "scale",  # SVM RBF kernel width
    n_estimators: int = 100,  # Random Forest tree count
    max_depth: Optional[int] = None,  # Random Forest max tree depth
    use_pca: bool = False,  # Whether to apply PCA dimensionality reduction
    pca_components: int = 64,  # Number of PCA components if use_pca=True
) -> Dict[str, Any]:
    """
    Train feature extraction + sklearn classifier pipeline.

    Parameters
    ----------
    X_train : np.ndarray of shape (N, C, H, W)
    y_train : np.ndarray of shape (N,)
    Various classifier hyperparameters.

    Returns
    -------
    result : dict with 'pipeline', 'feature_method', 'feature_dim'.
    """
    print(f"Extracting {feature_method} features (dim={feature_dim})...")

    # Extract features from raw images -- this is the transfer learning step
    # where pretrained ResNet representations replace manual feature engineering
    X_feat = extract_features(X_train, method=feature_method, feature_dim=feature_dim)

    # Build sklearn pipeline: scaler -> (optional PCA) -> classifier
    steps = [("scaler", StandardScaler())]

    if use_pca:
        # PCA reduces dimensionality to prevent overfitting and speed up SVM.
        # n_components is bounded by both the desired count and available dimensions.
        n_comp = min(pca_components, X_feat.shape[1], X_feat.shape[0])
        steps.append(("pca", PCA(n_components=n_comp)))

    # Select classifier based on configuration
    if classifier_type == "svm":
        # SVM with RBF kernel works well on deep features because the Gaussian kernel
        # can model complex, non-linear decision boundaries in the feature space
        steps.append(("clf", SVC(C=C, kernel=kernel, gamma=gamma, random_state=42, max_iter=5000)))
        print(f"Training SVM (C={C}, kernel={kernel})...")
    elif classifier_type == "rf":
        # Random Forest: ensemble of trees that can capture feature interactions
        # n_jobs=-1 parallelizes tree training across all CPU cores
        steps.append(
            ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1))
        )
        print(f"Training RF (n_estimators={n_estimators}, max_depth={max_depth})...")
    else:
        # Logistic Regression: linear classifier, often sufficient for deep features
        # because ResNet features are already quite linearly separable
        steps.append(("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)))
        print(f"Training Logistic Regression (C={C})...")

    # Build and fit the pipeline
    pipeline = Pipeline(steps)
    pipeline.fit(X_feat, y_train)

    # Training accuracy as a sanity check
    train_acc = pipeline.score(X_feat, y_train)
    print(f"Training accuracy: {train_acc:.4f}")

    return {
        "pipeline": pipeline,
        "feature_method": feature_method,
        "feature_dim": feature_dim,
    }


def validate(
    model_dict: Dict[str, Any],  # Output from train()
    X_val: np.ndarray,  # Validation images
    y_val: np.ndarray,  # Validation labels
) -> Dict[str, Any]:
    """
    Validate on held-out data.

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1, confusion_matrix, report.
    """
    pipeline = model_dict["pipeline"]

    # Extract features using the SAME method as training for consistency
    X_feat = extract_features(
        X_val,
        method=model_dict["feature_method"],
        feature_dim=model_dict["feature_dim"],
    )

    # Generate predictions
    y_pred = pipeline.predict(X_feat)

    # Compute comprehensive metrics
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
    """Final test evaluation -- used once for unbiased performance assessment."""
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
    """Optuna objective for ResNet+sklearn hyperparameter tuning."""
    classifier_type = trial.suggest_categorical("classifier_type", ["svm", "rf", "logistic"])
    feature_dim = trial.suggest_categorical("feature_dim", [64, 128, 256])
    use_pca = trial.suggest_categorical("use_pca", [True, False])

    kwargs = {
        "classifier_type": classifier_type,
        "feature_method": "simple",
        "feature_dim": feature_dim,
        "use_pca": use_pca,
    }

    if classifier_type == "svm":
        kwargs["C"] = trial.suggest_float("svm_C", 0.01, 100.0, log=True)
        kwargs["kernel"] = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
    elif classifier_type == "rf":
        kwargs["n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        kwargs["max_depth"] = trial.suggest_int("rf_max_depth", 5, 40)
    else:
        kwargs["C"] = trial.suggest_float("lr_C", 0.01, 100.0, log=True)

    if use_pca:
        kwargs["pca_components"] = trial.suggest_int("pca_components", 16, min(128, feature_dim))

    model_dict = train(X_train, y_train, **kwargs)
    metrics = validate(model_dict, X_val, y_val)
    return metrics["accuracy"]


def ray_tune_search(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
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
            feature_method="simple", feature_dim=config["feature_dim"],
            C=config.get("C", 1.0), kernel=config.get("kernel", "rbf"),
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
        )
        metrics = validate(md, X_val, y_val)
        tune.report({"accuracy": metrics["accuracy"]})

    search_space = {
        "classifier_type": tune.choice(["svm", "rf", "logistic"]),
        "feature_dim": tune.choice([64, 128, 256]),
        "C": tune.loguniform(0.01, 100.0),
        "kernel": tune.choice(["rbf", "linear"]),
        "n_estimators": tune.choice([50, 100, 200]),
        "max_depth": tune.choice([10, 20, None]),
    }

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    tuner = tune.Tuner(
        trainable, param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, metric="accuracy", mode="max"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config: {best.config}")
    print(f"Best accuracy: {best.metrics['accuracy']:.4f}")
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------


def compare_parameter_sets(
    n_samples: int = 1500,
    img_size: int = 32,
) -> Dict[str, Any]:
    """
    Compare different ResNet feature extraction and classifier configurations.

    Demonstrates the impact of:
    1. Feature dimensionality: 64 vs 128 vs 256 (affects information capacity)
    2. Classifier choice: SVM vs RF vs LogisticRegression (different decision boundaries)
    3. PCA dimensionality reduction: with vs without (regularization vs information loss)
    4. SVM kernel: linear vs RBF (simple vs complex decision boundary)
    5. SVM regularization C: 0.1 vs 10 vs 100 (underfitting vs overfitting tradeoff)

    Parameters
    ----------
    n_samples : int
        Number of synthetic images.
    img_size : int
        Image spatial dimension.

    Returns
    -------
    results : dict mapping configuration names to accuracy scores.
    """
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: ResNet Feature + Classifier Configurations")
    print("=" * 70)

    # Shared dataset for fair comparison
    X, y = generate_data(n_samples=n_samples, img_size=img_size, n_classes=10, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    configs = {
        # --- Feature dimension comparison ---
        "feat_dim=64_SVM": {
            "classifier_type": "svm", "feature_dim": 64, "C": 10.0, "kernel": "rbf",
        },
        "feat_dim=128_SVM": {
            "classifier_type": "svm", "feature_dim": 128, "C": 10.0, "kernel": "rbf",
        },
        "feat_dim=256_SVM": {
            "classifier_type": "svm", "feature_dim": 256, "C": 10.0, "kernel": "rbf",
        },
        # --- Classifier comparison ---
        "SVM_rbf": {
            "classifier_type": "svm", "feature_dim": 128, "C": 10.0, "kernel": "rbf",
        },
        "SVM_linear": {
            "classifier_type": "svm", "feature_dim": 128, "C": 10.0, "kernel": "linear",
        },
        "RandomForest": {
            "classifier_type": "rf", "feature_dim": 128, "n_estimators": 200, "max_depth": 20,
        },
        "LogisticRegression": {
            "classifier_type": "logistic", "feature_dim": 128, "C": 10.0,
        },
        # --- PCA comparison ---
        "SVM_with_PCA": {
            "classifier_type": "svm", "feature_dim": 128, "C": 10.0, "kernel": "rbf",
            "use_pca": True, "pca_components": 64,
        },
        "SVM_without_PCA": {
            "classifier_type": "svm", "feature_dim": 128, "C": 10.0, "kernel": "rbf",
            "use_pca": False,
        },
        # --- SVM C comparison ---
        "SVM_C=0.1": {
            "classifier_type": "svm", "feature_dim": 128, "C": 0.1, "kernel": "rbf",
        },
        "SVM_C=100": {
            "classifier_type": "svm", "feature_dim": 128, "C": 100.0, "kernel": "rbf",
        },
    }

    results = {}
    for name, params in configs.items():
        print(f"\n--- Configuration: {name} ---")
        model_dict = train(X_train, y_train, feature_method="simple", **params)
        metrics = validate(model_dict, X_val, y_val)
        results[name] = metrics["accuracy"]

    # Print summary
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Val Accuracy':>12}")
    print("-" * 50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<35} {acc:>12.4f}")
    print("-" * 50)

    print("\nKey Takeaways:")
    print("  - Higher feature dimensions capture more information but may overfit")
    print("  - SVM with RBF kernel typically outperforms linear classifiers on deep features")
    print("  - PCA can help when feature dimension >> number of samples")
    print("  - Moderate regularization (C=1-10) usually works best for SVM")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Manufacturing Defect Detection
# ---------------------------------------------------------------------------


def real_world_demo() -> Dict[str, Any]:
    """
    Demonstrate ResNet features + sklearn classifier on a manufacturing
    defect detection scenario: classifying product images as Clean, Scratch, or Dent.

    In real factories, visual inspection systems use cameras to photograph
    products on conveyor belts. ResNet features can capture surface texture
    patterns that distinguish different defect types without training a
    full deep learning model from scratch.

    Returns
    -------
    results : dict with model performance metrics.
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Manufacturing Defect Detection (Scratch vs Dent vs Clean)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_samples = 900  # 300 per class
    img_size = 32
    n_channels = 3
    n_classes = 3
    class_names = ["Clean", "Scratch", "Dent"]
    samples_per_class = n_samples // n_classes

    X_list, y_list = [], []

    for cls_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Base: uniform surface with slight texture variation
            # Real product surfaces have subtle grain patterns
            base = np.ones((img_size, img_size), dtype=np.float32) * 0.5
            base += rng.rand(img_size, img_size).astype(np.float32) * 0.06

            # Add fine surface texture (simulates metal/plastic grain)
            for row in range(0, img_size, 2):
                base[row, :] += rng.uniform(-0.02, 0.02)

            if cls_idx == 0:
                # CLEAN: uniform surface with minimal variation
                # Just the base texture -- no defects
                base += rng.randn(img_size, img_size).astype(np.float32) * 0.01

            elif cls_idx == 1:
                # SCRATCH: linear marks on the surface
                # Scratches appear as narrow, elongated bright/dark lines
                n_scratches = rng.randint(1, 4)
                for _ in range(n_scratches):
                    # Random scratch direction and position
                    y_start = rng.randint(0, img_size)
                    x_start = rng.randint(0, img_size)
                    angle = rng.uniform(0, np.pi)
                    length = rng.randint(8, 20)
                    intensity = rng.choice([-0.15, 0.15])  # Dark or bright scratch

                    for t in range(length):
                        sx = int(x_start + t * np.cos(angle))
                        sy = int(y_start + t * np.sin(angle))
                        if 0 <= sx < img_size and 0 <= sy < img_size:
                            base[sy, sx] += intensity
                            # Add width to scratch
                            if sy + 1 < img_size:
                                base[sy + 1, sx] += intensity * 0.5

            elif cls_idx == 2:
                # DENT: circular depressions on the surface
                # Dents appear as circular regions with altered intensity
                n_dents = rng.randint(1, 3)
                for _ in range(n_dents):
                    dx = rng.randint(5, img_size - 5)
                    dy = rng.randint(5, img_size - 5)
                    radius = rng.randint(3, 7)
                    yy, xx = np.ogrid[:img_size, :img_size]
                    dist = np.sqrt((xx - dx) ** 2 + (yy - dy) ** 2)
                    # Dent creates a concave surface -- darker center, brighter rim
                    dent_profile = np.exp(-dist**2 / (2 * radius**2)).astype(np.float32)
                    base -= dent_profile * 0.12  # Darker center
                    # Bright rim around the dent (light reflection on edge)
                    rim_mask = (dist > radius * 0.7) & (dist < radius * 1.2)
                    base[rim_mask] += 0.06

            # Add sensor noise
            base += rng.randn(img_size, img_size).astype(np.float32) * 0.02
            base = np.clip(base, 0, 1)

            # Replicate to 3 channels (CHW format)
            img = np.stack([base, base, base], axis=0)
            X_list.append(img)
            y_list.append(cls_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Generated {len(X)} synthetic product images ({n_channels}x{img_size}x{img_size})")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train multiple classifiers for comparison
    print("\n--- Training SVM classifier ---")
    svm_model = train(
        X_train, y_train, classifier_type="svm",
        feature_dim=128, C=10.0, kernel="rbf",
    )
    print("\n--- Training Random Forest classifier ---")
    rf_model = train(
        X_train, y_train, classifier_type="rf",
        feature_dim=128, n_estimators=150, max_depth=20,
    )

    # Evaluate
    print("\n--- SVM Validation ---")
    svm_metrics = validate(svm_model, X_val, y_val)
    print("\n--- RF Validation ---")
    rf_metrics = validate(rf_model, X_val, y_val)

    # Select best and test
    best_name = "SVM" if svm_metrics["accuracy"] >= rf_metrics["accuracy"] else "Random Forest"
    best_model = svm_model if svm_metrics["accuracy"] >= rf_metrics["accuracy"] else rf_model
    print(f"\n--- Final Test with {best_name} ---")
    test_metrics = test(best_model, X_test, y_test)

    # Manufacturing interpretation
    print("\n--- Manufacturing Quality Metrics ---")
    print(f"Best Model: {best_name}")
    print(f"Overall Defect Detection Accuracy: {test_metrics['accuracy']:.4f}")
    for i, name in enumerate(class_names):
        if i < len(test_metrics["precision"]):
            print(f"  {name}: Precision={test_metrics['precision'][i]:.4f}, "
                  f"Recall={test_metrics['recall'][i]:.4f}")

    print("\nNote: In production, this system would be calibrated to minimize")
    print("false negatives (missed defects) even at the cost of more false positives.")

    return {
        "svm_accuracy": svm_metrics["accuracy"],
        "rf_accuracy": rf_metrics["accuracy"],
        "best_model": best_name,
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the full ResNet feature extraction + sklearn pipeline."""
    print("=" * 70)
    print("ResNet Feature Extraction + Sklearn Classifier Pipeline")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/8] Generating synthetic data...")
    X, y = generate_data(n_samples=2000, img_size=32, n_classes=10)
    print(f"    Dataset: {X.shape}, Classes: {np.unique(y)}")

    # 2. Split
    print("\n[2/8] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Train
    print("\n[3/8] Training SVM with simulated ResNet features...")
    model_dict = train(X_train, y_train, classifier_type="svm", feature_dim=128, C=10.0, kernel="rbf")

    # 4. Validate
    print("\n[4/8] Validating...")
    val_metrics = validate(model_dict, X_val, y_val)

    # 5. Optuna
    print("\n[5/8] Optuna hyperparameter search...")
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
            "use_pca": bp["use_pca"],
        }
        if bp["classifier_type"] == "svm":
            retrain_kwargs["C"] = bp["svm_C"]
            retrain_kwargs["kernel"] = bp["svm_kernel"]
        elif bp["classifier_type"] == "rf":
            retrain_kwargs["n_estimators"] = bp["rf_n_estimators"]
            retrain_kwargs["max_depth"] = bp["rf_max_depth"]
        else:
            retrain_kwargs["C"] = bp["lr_C"]

        if bp["use_pca"]:
            retrain_kwargs["pca_components"] = bp["pca_components"]

        model_dict = train(X_train, y_train, **retrain_kwargs)
    else:
        print("    Optuna not installed, skipping.")

    # 6. Test
    print("\n[6/8] Test evaluation...")
    test_metrics = test(model_dict, X_test, y_test)

    # 7. Parameter comparison
    print("\n[7/8] Running parameter comparison...")
    compare_parameter_sets(n_samples=1000, img_size=32)

    # 8. Real-world demo
    print("\n[8/8] Running manufacturing defect detection demo...")
    real_world_demo()

    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
