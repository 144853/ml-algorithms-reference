"""
EfficientNet - PyTorch Implementation
======================================

Architecture:
    Full implementation of EfficientNet using PyTorch, featuring MBConv
    blocks with inverted residuals, squeeze-and-excitation, and compound
    scaling principles.

    MBConv Block (Mobile Inverted Bottleneck Convolution):
    x --> Expand(1x1, channels*expand_ratio) -> BN -> SiLU
      --> DepthwiseConv(3x3 or 5x5) -> BN -> SiLU
      --> SqueezeExcitation(reduction_ratio)
      --> Project(1x1, out_channels) -> BN
      --> (+x if skip connection) --> output

    EfficientNet-B0 Architecture (adapted for 32x32):
    Stage  | Operator         | Resolution | Channels | Layers
    -------|-----------------|------------|----------|-------
    0      | Conv3x3, stride1 | 32x32      | 32       | 1
    1      | MBConv1, k3      | 32x32      | 16       | 1
    2      | MBConv6, k3, s2  | 16x16      | 24       | 2
    3      | MBConv6, k5, s2  | 8x8        | 40       | 2
    4      | MBConv6, k3, s2  | 4x4        | 80       | 3
    5      | Conv1x1          | 4x4        | 320      | 1
    6      | AvgPool+FC       | 1x1        | n_classes| 1

    Compound Scaling:
    EfficientNet scales depth, width, and resolution uniformly:
    depth  = alpha^phi     (number of layers)
    width  = beta^phi      (number of channels)
    resolution = gamma^phi (input image size)
    subject to: alpha * beta^2 * gamma^2 ~= 2

    For EfficientNet-B0: phi=1, alpha=1.2, beta=1.1, gamma=1.15

Theory & Mathematics:
    Inverted Residuals (MobileNetV2):
        Traditional bottleneck: compress -> process -> expand
        Inverted bottleneck:    expand -> process -> compress
        The expansion creates a high-dimensional representation where
        depthwise convolutions can capture richer patterns.

    Depthwise Separable Convolution:
        Factored into two operations:
        1. Depthwise: groups=in_channels (one filter per channel)
           Parameters: C * k * k (vs C_in * C_out * k * k for standard)
        2. Pointwise: 1x1 convolution to mix channels
           Parameters: C_in * C_out

    Squeeze-and-Excitation:
        Learns to re-weight channel features:
        z = GlobalAvgPool(x)           -- squeeze to (B, C)
        s = sigmoid(FC2(SiLU(FC1(z)))) -- learn channel weights
        y = x * s                       -- scale channels

    SiLU (Swish) Activation:
        f(x) = x * sigmoid(x)
        - Smooth, non-monotonic activation
        - Better gradient properties than ReLU
        - Used throughout EfficientNet

    Stochastic Depth (Drop Path):
        During training, randomly skip entire blocks with probability p.
        Equivalent to an implicit ensemble of exponentially many subnetworks.
        Survival probability increases for earlier layers.

Business Use Cases:
    - State-of-the-art image classification with efficiency constraints
    - Mobile and edge deployment (EfficientNet-B0 is 5.3M params)
    - Transfer learning backbone for detection and segmentation
    - Medical imaging with high accuracy requirements
    - Production ML systems where inference cost matters
    - AutoML and neural architecture search applications

Advantages:
    - Best accuracy-efficiency trade-off in image classification
    - Scales gracefully from mobile (B0) to server (B7)
    - Uses compound scaling for principled architecture design
    - MBConv blocks are hardware-friendly
    - Strong transfer learning performance
    - SiLU/Swish activation improves training dynamics

Disadvantages:
    - More complex architecture than standard CNN/ResNet
    - Training can be slower due to SE blocks and larger channels
    - Compound scaling requires careful tuning of alpha, beta, gamma
    - Depthwise convolutions may be slower on some hardware
    - Memory-intensive during training due to expanded channels

Key Hyperparameters:
    - width_mult: Channel width multiplier (compound scaling)
    - depth_mult: Depth multiplier (number of repeated blocks)
    - dropout_rate: Classifier dropout (0.2-0.5)
    - drop_path_rate: Stochastic depth rate (0.0-0.2)
    - learning_rate: Typically 0.01-0.1 with SGD, 1e-3 with Adam
    - se_ratio: SE reduction ratio (0.25 = 1/4 of channels)

References:
    - Tan, M. and Le, Q.V. (2019). "EfficientNet: Rethinking Model Scaling
      for Convolutional Neural Networks." ICML. arXiv:1905.11946.
    - Tan, M. and Le, Q.V. (2021). "EfficientNetV2: Smaller Models and
      Faster Training." ICML. arXiv:2104.00298.
    - Sandler, M. et al. (2018). "MobileNetV2: Inverted Residuals and
      Linear Bottlenecks." CVPR.
    - Hu, J. et al. (2018). "Squeeze-and-Excitation Networks." CVPR.
"""

import math
import numpy as np
import warnings
from typing import Dict, Tuple, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 3000,
    img_size: int = 32,
    n_channels: int = 3,
    n_classes: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic RGB images with class-specific visual patterns.

    Returns X: (N, C, H, W) float32, y: (N,) int64.
    """
    rng = np.random.RandomState(random_state)
    spc = n_samples // n_classes
    X_list, y_list = [], []

    for cls in range(n_classes):
        for _ in range(spc):
            img = rng.rand(n_channels, img_size, img_size).astype(np.float32) * 0.1
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
                cx, cy = img_size // 2, img_size // 2
                yy, xx = np.ogrid[:img_size, :img_size]
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                img[ch] += np.exp(-dist**2 / (2 * (img_size / 4) ** 2)).astype(np.float32) * 0.8
            elif cls == 6:
                grad = np.linspace(0, 1, img_size).reshape(1, -1).astype(np.float32)
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


def create_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# EfficientNet Building Blocks
# ---------------------------------------------------------------------------


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.

    Learns channel-wise attention weights using global context.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    se_channels : int
        Number of channels in the bottleneck.
    """

    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, in_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class DropPath(nn.Module):
    """
    Stochastic Depth / Drop Path regularization.

    During training, randomly drops entire residual branches.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device, dtype=x.dtype)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block.

    This is the core building block of EfficientNet.

    Structure:
    1. Expansion: 1x1 conv to expand channels (if expand_ratio > 1)
    2. Depthwise: kxk depthwise conv (groups=channels)
    3. Squeeze-Excitation: channel attention
    4. Projection: 1x1 conv to reduce channels
    5. Skip connection: add input if dimensions match

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    expand_ratio : int
        Expansion factor for the inverted bottleneck.
    kernel_size : int
        Kernel size for depthwise conv (3 or 5).
    stride : int
        Stride for depthwise conv.
    se_ratio : float
        SE reduction ratio (fraction of input channels).
    drop_path_rate : float
        Drop path probability for stochastic depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.use_skip = (stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio

        layers = []

        # 1. Expansion phase (only if expand_ratio > 1)
        if expand_ratio > 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            ])

        # 2. Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride,
                      padding=padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        ])

        self.main = nn.Sequential(*layers)

        # 3. Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(mid_channels, se_channels)

        # 4. Projection (pointwise conv)
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # 5. Drop path for stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.main(x)
        out = self.se(out)
        out = self.project(out)

        if self.use_skip:
            out = self.drop_path(out) + residual

        return out


# ---------------------------------------------------------------------------
# EfficientNet Model
# ---------------------------------------------------------------------------


class EfficientNetModel(nn.Module):
    """
    EfficientNet implementation with compound scaling.

    Adapted for small images (32x32) with reduced channel counts
    for CPU-friendly demo usage.

    Parameters
    ----------
    in_channels : int
        Input image channels.
    n_classes : int
        Number of output classes.
    width_mult : float
        Channel width multiplier.
    depth_mult : float
        Block depth multiplier.
    dropout_rate : float
        Dropout rate for the classifier.
    drop_path_rate : float
        Maximum stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 10,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        def _round_channels(c: int) -> int:
            """Round channels to nearest multiple of 8."""
            c = int(c * width_mult)
            return max(8, (c + 4) // 8 * 8)

        def _round_repeats(r: int) -> int:
            """Round repeats based on depth multiplier."""
            return max(1, int(math.ceil(r * depth_mult)))

        # Stage configurations: (expand_ratio, channels, repeats, stride, kernel_size)
        # Adapted from EfficientNet-B0, with reduced channels for small images
        stage_configs = [
            (1, 16, 1, 1, 3),   # Stage 1
            (6, 24, 2, 2, 3),   # Stage 2
            (6, 40, 2, 2, 5),   # Stage 3
            (6, 80, 3, 2, 3),   # Stage 4
        ]

        # Stem
        stem_channels = _round_channels(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        # Build stages
        total_blocks = sum(_round_repeats(r) for _, _, r, _, _ in stage_configs)
        block_idx = 0
        stages = []
        prev_channels = stem_channels

        for expand_ratio, channels, repeats, stride, kernel_size in stage_configs:
            out_channels = _round_channels(channels)
            n_repeats = _round_repeats(repeats)

            for i in range(n_repeats):
                s = stride if i == 0 else 1
                in_ch = prev_channels if i == 0 else out_channels
                dp_rate = drop_path_rate * block_idx / total_blocks

                stages.append(
                    MBConvBlock(
                        in_channels=in_ch,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        kernel_size=kernel_size,
                        stride=s,
                        se_ratio=0.25,
                        drop_path_rate=dp_rate,
                    )
                )
                block_idx += 1

            prev_channels = out_channels

        self.stages = nn.Sequential(*stages)

        # Head
        head_channels = _round_channels(320)
        self.head = nn.Sequential(
            nn.Conv2d(prev_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(head_channels, n_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def efficientnet_b0(in_channels: int = 3, n_classes: int = 10, **kwargs) -> EfficientNetModel:
    """EfficientNet-B0 (small variant for demos)."""
    return EfficientNetModel(in_channels, n_classes, width_mult=1.0, depth_mult=1.0, **kwargs)


def efficientnet_b1(in_channels: int = 3, n_classes: int = 10, **kwargs) -> EfficientNetModel:
    """EfficientNet-B1 (compound scaled up)."""
    return EfficientNetModel(in_channels, n_classes, width_mult=1.1, depth_mult=1.1, **kwargs)


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    in_channels: int = 3,
    n_classes: int = 10,
    width_mult: float = 1.0,
    depth_mult: float = 1.0,
    dropout_rate: float = 0.2,
    drop_path_rate: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 15,
    optimizer_name: str = "adam",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the EfficientNet model.

    Parameters
    ----------
    train_loader : DataLoader
    val_loader : DataLoader, optional
    Various architecture and training hyperparameters.

    Returns
    -------
    result : dict with 'model', 'train_losses', 'val_accs'.
    """
    model = EfficientNetModel(
        in_channels=in_channels,
        n_classes=n_classes,
        width_mult=width_mult,
        depth_mult=depth_mult,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    train_losses, val_accs = [], []
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  Model parameters: {n_params:,}")

    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        correct, total = 0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += loss.item() * Xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += Xb.size(0)

        scheduler.step()
        avg_loss = ep_loss / total
        train_acc = correct / total
        train_losses.append(avg_loss)

        val_acc = None
        if val_loader:
            vm = validate({"model": model}, val_loader=val_loader, verbose=False)
            val_acc = vm["accuracy"]
            val_accs.append(val_acc)

        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            msg = f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, TrainAcc: {train_acc:.4f}"
            if val_acc is not None:
                msg += f", ValAcc: {val_acc:.4f}"
            print(msg)

    return {"model": model, "train_losses": train_losses, "val_accs": val_accs}


def validate(
    model_dict: Dict[str, Any],
    val_loader: Optional[DataLoader] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate/evaluate the model."""
    model = model_dict["model"]
    model.eval()

    if val_loader is None:
        ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * Xb.size(0)

            preds = logits.argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
            total += Xb.size(0)

            k = min(5, logits.size(1))
            _, topk = logits.topk(k, dim=1)
            top5_correct += (topk == yb.unsqueeze(1)).any(dim=1).sum().item()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    top5_acc = top5_correct / total

    classes = np.unique(all_labels)
    per_class = {}
    for c in classes:
        mask = all_labels == c
        per_class[int(c)] = float(np.mean(all_preds[mask] == all_labels[mask])) if mask.sum() > 0 else 0.0

    if verbose:
        print(f"Accuracy: {accuracy:.4f}, Top-5: {top5_acc:.4f}, Loss: {total_loss / total:.4f}")
        for c, acc in per_class.items():
            print(f"  Class {c}: {acc:.4f}")

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
        "loss": total_loss / total,
        "per_class_accuracy": per_class,
    }


def test(
    model_dict: Dict[str, Any],
    test_loader: Optional[DataLoader] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Final test evaluation."""
    print("\n--- Test Set Evaluation ---")
    return validate(model_dict, val_loader=test_loader, X_val=X_test, y_val=y_test)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: "optuna.Trial",
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int = 10,
) -> float:
    """Optuna objective for EfficientNet tuning."""
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)
    drop_path = trial.suggest_float("drop_path_rate", 0.0, 0.2)
    width_mult = trial.suggest_categorical("width_mult", [0.5, 0.75, 1.0])
    depth_mult = trial.suggest_categorical("depth_mult", [0.5, 0.75, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

    tl, vl = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
    md = train(
        tl, n_classes=n_classes,
        width_mult=width_mult, depth_mult=depth_mult,
        dropout_rate=dropout, drop_path_rate=drop_path,
        learning_rate=lr, weight_decay=wd,
        n_epochs=8, optimizer_name=optimizer_name, verbose=False,
    )
    metrics = validate(md, val_loader=vl, verbose=False)
    return metrics["accuracy"]


def ray_tune_search(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int = 10,
    num_samples: int = 8,
) -> Dict[str, Any]:
    """Ray Tune hyperparameter search."""
    if not RAY_AVAILABLE:
        print("Ray not installed. Skipping.")
        return {}

    def trainable(config):
        tl, vl = create_dataloaders(X_train, y_train, X_val, y_val, config["batch_size"])
        md = train(
            tl, n_classes=n_classes,
            width_mult=config["width_mult"], depth_mult=config["depth_mult"],
            dropout_rate=config["dropout"], drop_path_rate=config["drop_path"],
            learning_rate=config["lr"], weight_decay=config["wd"],
            n_epochs=8, optimizer_name=config["optimizer"], verbose=False,
        )
        metrics = validate(md, val_loader=vl, verbose=False)
        tune.report({"accuracy": metrics["accuracy"]})

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "wd": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.1, 0.5),
        "drop_path": tune.uniform(0.0, 0.2),
        "width_mult": tune.choice([0.5, 0.75, 1.0]),
        "depth_mult": tune.choice([0.5, 0.75, 1.0]),
        "batch_size": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "adamw", "sgd"]),
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
    """Run the full PyTorch EfficientNet pipeline."""
    print("=" * 70)
    print("EfficientNet - PyTorch Implementation")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    N_CLASSES = 10

    # 1. Generate data
    print("\n[1/6] Generating synthetic data...")
    X, y = generate_data(n_samples=3000, img_size=32, n_classes=N_CLASSES)
    print(f"    Dataset: {X.shape}")

    # 2. Split
    print("\n[2/6] Splitting data...")
    n = len(X)
    nt = int(0.7 * n)
    nv = int(0.15 * n)
    X_train, y_train = X[:nt], y[:nt]
    X_val, y_val = X[nt : nt + nv], y[nt : nt + nv]
    X_test, y_test = X[nt + nv :], y[nt + nv :]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Train EfficientNet-B0
    print("\n[3/6] Training EfficientNet-B0...")
    tl, vl = create_dataloaders(X_train, y_train, X_val, y_val, 64)
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model_dict = train(
        tl, val_loader=vl, n_classes=N_CLASSES,
        width_mult=1.0, depth_mult=1.0,
        dropout_rate=0.2, drop_path_rate=0.1,
        learning_rate=1e-3, n_epochs=15,
    )

    # 4. Validate
    print("\n[4/6] Validation...")
    val_metrics = validate(model_dict, val_loader=vl)

    # 5. Optuna
    print("\n[5/6] Optuna hyperparameter search...")
    if OPTUNA_AVAILABLE:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, N_CLASSES),
            n_trials=10,
        )
        print(f"    Best accuracy: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        bp = study.best_params
        tl, vl = create_dataloaders(X_train, y_train, X_val, y_val, bp["batch_size"])
        model_dict = train(
            tl, val_loader=vl, n_classes=N_CLASSES,
            width_mult=bp["width_mult"], depth_mult=bp["depth_mult"],
            dropout_rate=bp["dropout_rate"], drop_path_rate=bp["drop_path_rate"],
            learning_rate=bp["learning_rate"], weight_decay=bp["weight_decay"],
            n_epochs=15, optimizer_name=bp["optimizer"],
        )
    else:
        print("    Optuna not installed, skipping.")

    # 6. Test
    print("\n[6/6] Test evaluation...")
    test_metrics = test(model_dict, test_loader=test_loader)

    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
