"""
Residual Network (ResNet) - PyTorch Implementation
===================================================

Architecture:
    Full implementation of ResNet with BasicBlock and Bottleneck residual
    blocks, following the original paper's design principles.

    BasicBlock (used in ResNet-18, ResNet-34):
        x --> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN --> (+) --> ReLU --> y
        |                                                  |
        +------------- identity or 1x1 conv ---------------+

    Bottleneck (used in ResNet-50, ResNet-101, ResNet-152):
        x --> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN --> (+) --> ReLU --> y
        |               (reduce)                                (expand)           |
        +------------------------- identity or 1x1 conv --------------------------+

    Full Architecture (ResNet-18 variant for 32x32 images):
    Input (3, 32, 32)
    -> Conv(3->64, 3x3, stride=1, pad=1) -> BN -> ReLU
    -> Layer1: 2x BasicBlock(64->64)           # 32x32
    -> Layer2: 2x BasicBlock(64->128, stride=2)  # 16x16
    -> Layer3: 2x BasicBlock(128->256, stride=2) # 8x8
    -> Layer4: 2x BasicBlock(256->512, stride=2) # 4x4
    -> AdaptiveAvgPool(1x1) -> Flatten -> FC(512->n_classes)

Theory & Mathematics:
    Residual Learning:
        y = F(x, {W_i}) + x      (identity shortcut)
        y = F(x, {W_i}) + W_s*x  (projection shortcut when dimensions change)

        For BasicBlock:
        F(x) = W_2 * ReLU(BN(W_1 * x))

        For Bottleneck:
        F(x) = W_3 * ReLU(BN(W_2 * ReLU(BN(W_1 * x))))
        where W_1: 1x1 (reduce channels), W_2: 3x3, W_3: 1x1 (expand channels)

    Gradient Flow Analysis:
        dL/dx_l = dL/dx_L * product_{i=l}^{L-1} (1 + dF(x_i)/dx_i)
        The "1+" ensures that gradient magnitude is preserved regardless of depth.

    Why ResNet Works:
        1. Identity shortcuts provide gradient highways
        2. Residual functions F(x) can easily learn zero mappings
        3. Enables training of very deep networks (100+ layers)
        4. Implicit ensemble of many shallow networks

    Bottleneck Design:
        1x1 conv (reduce channels) -> 3x3 conv -> 1x1 conv (expand channels)
        This reduces computation while maintaining network expressiveness.

Business Use Cases:
    - ImageNet-scale image classification
    - Feature backbone for object detection (Faster R-CNN, SSD)
    - Semantic segmentation (DeepLab, PSPNet)
    - Medical image analysis (pathology, radiology)
    - Autonomous driving perception systems
    - Industrial quality inspection

Advantages:
    - Solves vanishing gradient problem via skip connections
    - Scales to very deep networks (50, 101, 152 layers)
    - Strong performance across many vision tasks
    - Well-studied architecture with many pretrained variants
    - Modular design with interchangeable blocks
    - Can be used as feature backbone for downstream tasks

Disadvantages:
    - Larger memory footprint than simpler CNNs
    - Deeper variants require significant compute
    - May be overkill for simple classification tasks
    - Fixed topology (no dynamic routing)
    - Higher latency compared to lightweight architectures

Key Hyperparameters:
    - Depth: ResNet-18/34 (BasicBlock) vs 50/101/152 (Bottleneck)
    - Initial learning rate: Typically 0.1 with SGD, 1e-3 with Adam
    - Learning rate schedule: StepLR, CosineAnnealingLR, OneCycleLR
    - Weight decay: L2 regularization (typical: 1e-4)
    - Batch size: 64-256

References:
    - He, K. et al. (2016). "Deep Residual Learning for Image Recognition."
      CVPR. arXiv:1512.03385.
    - He, K. et al. (2016). "Identity Mappings in Deep Residual Networks."
      ECCV. arXiv:1603.05027.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any, Optional, List, Type

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
    Generate synthetic RGB images with class-specific patterns.

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
                    j = i % img_size
                    img[ch, i, j] += 0.7
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
    """Create PyTorch DataLoaders from NumPy arrays."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# ResNet Building Blocks
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34.

    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50, ResNet-101, ResNet-152.

    Architecture:
        x -> Conv1x1(reduce) -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
          -> Conv1x1(expand) -> BN -> (+x) -> ReLU
    """

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# ResNet Model
# ---------------------------------------------------------------------------


class ResNetModel(nn.Module):
    """
    ResNet implementation supporting both BasicBlock and Bottleneck.

    Adapted for small images (32x32) by using stride-1 initial conv
    and no initial max pooling (unlike ImageNet ResNet which uses 7x7
    conv with stride 2 and max pool).

    Parameters
    ----------
    block : Type[nn.Module]
        BasicBlock or Bottleneck.
    layers : List[int]
        Number of blocks in each of the 4 stages.
    in_channels : int
        Input image channels.
    n_classes : int
        Number of output classes.
    base_width : int
        Base channel width for the first stage.
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        in_channels: int = 3,
        n_classes: int = 10,
        base_width: int = 64,
    ):
        super().__init__()
        self.in_planes = base_width

        # Initial convolution (adapted for 32x32 images)
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, n_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, block: Type[nn.Module], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(in_channels: int = 3, n_classes: int = 10, base_width: int = 32) -> ResNetModel:
    """Construct a ResNet-18-like model (small variant for demos)."""
    return ResNetModel(BasicBlock, [2, 2, 2, 2], in_channels, n_classes, base_width)


def resnet34(in_channels: int = 3, n_classes: int = 10, base_width: int = 32) -> ResNetModel:
    """Construct a ResNet-34-like model (small variant)."""
    return ResNetModel(BasicBlock, [3, 4, 6, 3], in_channels, n_classes, base_width)


def resnet50_small(in_channels: int = 3, n_classes: int = 10, base_width: int = 16) -> ResNetModel:
    """Construct a small ResNet-50 variant (reduced width for CPU demos)."""
    return ResNetModel(Bottleneck, [3, 4, 6, 3], in_channels, n_classes, base_width)


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    in_channels: int = 3,
    n_classes: int = 10,
    model_type: str = "resnet18",
    base_width: int = 32,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    n_epochs: int = 15,
    optimizer_name: str = "sgd",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the PyTorch ResNet.

    Parameters
    ----------
    train_loader : DataLoader
    val_loader : DataLoader, optional
    model_type : str
        'resnet18', 'resnet34', or 'resnet50'.
    Various hyperparameters.

    Returns
    -------
    result : dict with 'model', 'train_losses', 'val_accs'.
    """
    if model_type == "resnet34":
        model = resnet34(in_channels, n_classes, base_width)
    elif model_type == "resnet50":
        model = resnet50_small(in_channels, n_classes, max(8, base_width // 2))
    else:
        model = resnet18(in_channels, n_classes, base_width)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    train_losses, val_accs = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            opt.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += X_batch.size(0)

        scheduler.step()
        avg_loss = epoch_loss / total
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
    """Optuna objective for ResNet tuning."""
    lr = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    model_type = trial.suggest_categorical("model_type", ["resnet18", "resnet34"])
    base_width = trial.suggest_categorical("base_width", [16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
    md = train(
        train_loader, n_classes=n_classes, model_type=model_type,
        base_width=base_width, learning_rate=lr, weight_decay=wd,
        n_epochs=8, optimizer_name=optimizer_name, verbose=False,
    )
    metrics = validate(md, val_loader=val_loader, verbose=False)
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
            tl, n_classes=n_classes, model_type=config["model_type"],
            base_width=config["base_width"], learning_rate=config["lr"],
            weight_decay=config["wd"], n_epochs=8,
            optimizer_name=config["optimizer"], verbose=False,
        )
        metrics = validate(md, val_loader=vl, verbose=False)
        tune.report({"accuracy": metrics["accuracy"]})

    search_space = {
        "lr": tune.loguniform(1e-4, 0.1),
        "wd": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([32, 64, 128]),
        "model_type": tune.choice(["resnet18", "resnet34"]),
        "base_width": tune.choice([16, 32]),
        "optimizer": tune.choice(["adam", "sgd"]),
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
    """Run the full PyTorch ResNet pipeline."""
    print("=" * 70)
    print("Residual Network (ResNet) - PyTorch Implementation")
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

    # 3. Train ResNet-18
    print("\n[3/6] Training ResNet-18...")
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, 64)
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model_dict = train(
        train_loader, val_loader=val_loader, n_classes=N_CLASSES,
        model_type="resnet18", base_width=32,
        learning_rate=0.01, n_epochs=15, optimizer_name="sgd",
    )

    # 4. Validate
    print("\n[4/6] Validation...")
    val_metrics = validate(model_dict, val_loader=val_loader)

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
            model_type=bp["model_type"], base_width=bp["base_width"],
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
