# EfficientNet - Complete Guide with Stock Market Applications

## Overview

EfficientNet, introduced by Mingxing Tan and Quoc Le from Google in 2019, is a family of convolutional neural networks that achieves state-of-the-art accuracy while being significantly smaller and faster than previous architectures. The key innovation is compound scaling: instead of arbitrarily scaling network width, depth, or resolution independently, EfficientNet uses a principled compound coefficient to uniformly scale all three dimensions simultaneously. This balanced scaling leads to much better efficiency-to-accuracy trade-offs.

In stock market applications, EfficientNet's efficiency is a game-changer for multi-timeframe chart pattern recognition. Trading firms need to analyze charts across multiple timeframes (1-minute, 5-minute, hourly, daily, weekly) for thousands of stocks simultaneously. EfficientNet achieves accuracy comparable to or better than ResNet-152 while using 8.4x fewer parameters and being 6.1x faster, making real-time multi-timeframe scanning feasible on standard hardware.

The EfficientNet family ranges from EfficientNet-B0 (5.3M parameters) to EfficientNet-B7 (66M parameters), providing a spectrum of model sizes to match different deployment constraints. For stock market chart analysis, EfficientNet-B0 to B3 offer the best balance of accuracy and speed, enabling deployment on both GPU servers for batch scanning and edge devices for individual trader workstations.

## How It Works - The Math Behind It

### Compound Scaling

The core insight is that network width (w), depth (d), and resolution (r) should be scaled together using a compound coefficient phi:

```
depth:      d = alpha ^ phi
width:      w = beta ^ phi
resolution: r = gamma ^ phi

Subject to: alpha * beta^2 * gamma^2 ≈ 2
            alpha >= 1, beta >= 1, gamma >= 1
```

The constraint ensures that total FLOPS increase by approximately 2^phi. For EfficientNet-B0, a grid search found optimal values:

```
alpha = 1.2, beta = 1.1, gamma = 1.15
```

### MBConv Block (Mobile Inverted Bottleneck)

EfficientNet uses MBConv blocks as its primary building block:

```
1. Expansion: 1x1 Conv to expand channels by ratio t
   x_expanded = Conv1x1(x) -> BN -> Swish
   channels: c -> c*t

2. Depthwise Convolution: 3x3 or 5x5 depthwise conv
   x_dw = DepthwiseConv(x_expanded) -> BN -> Swish

3. Squeeze-and-Excitation: Channel attention
   s = GlobalAvgPool(x_dw)
   s = FC(ReLU(FC(s, c*t/r)), c*t) -> Sigmoid
   x_se = x_dw * s

4. Projection: 1x1 Conv to reduce channels
   x_proj = Conv1x1(x_se) -> BN

5. Residual Connection (if input/output shapes match):
   output = x + x_proj (with stochastic depth during training)
```

### Swish Activation Function

EfficientNet uses the Swish activation instead of ReLU:

```
Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Swish is smooth and non-monotonic, providing better gradient flow than ReLU.

### Squeeze-and-Excitation (SE) Block

Channel-wise attention mechanism:

```
z = GlobalAvgPool(x)                          # (C,)
s = sigmoid(W_2 * ReLU(W_1 * z))              # (C,)
output = x * s                                 # (C, H, W) * (C, 1, 1)
```

Where W_1 has shape (C/r, C) and W_2 has shape (C, C/r), with reduction ratio r (typically 4).

### Depthwise Separable Convolution

Splits standard convolution into two operations:

```
Standard conv: C_in * C_out * K * K * H * W  FLOPs

Depthwise: C_in * K * K * H * W  (spatial filtering per channel)
Pointwise: C_in * C_out * H * W  (channel mixing)

Savings: K^2 * C_out / (K^2 + C_out) ≈ 8-9x for K=3
```

### Stochastic Depth

During training, randomly drop entire residual blocks with probability p_l that increases linearly with depth:

```
p_l = 1 - l/L * (1 - p_survive)
output = x + b_l * F(x)    where b_l ~ Bernoulli(p_l)
```

### EfficientNet Architecture (B0 baseline)

```
Stage | Operator     | Resolution | Channels | Layers
------+--------------+------------+----------+-------
1     | Conv 3x3     | 224x224    | 32       | 1
2     | MBConv1, 3x3 | 112x112   | 16       | 1
3     | MBConv6, 3x3 | 112x112   | 24       | 2
4     | MBConv6, 5x5 | 56x56     | 40       | 2
5     | MBConv6, 3x3 | 28x28     | 80       | 3
6     | MBConv6, 5x5 | 14x14     | 112      | 3
7     | MBConv6, 5x5 | 14x14     | 192      | 4
8     | MBConv6, 3x3 | 7x7       | 320      | 1
9     | Conv 1x1, Pool, FC | 7x7 | 1280     | 1
```

## Stock Market Use Case: Efficient Multi-Timeframe Chart Pattern Recognition

### The Problem

A multi-strategy hedge fund runs pattern recognition across 3,000 US equities on five different timeframes (1-min, 5-min, hourly, daily, weekly), generating 15,000 chart images that must be classified every scanning cycle. Each chart needs to be analyzed for trend direction, pattern type, and signal strength. Using ResNet-50 for this workload would require multiple expensive GPU servers. EfficientNet achieves comparable accuracy with a fraction of the compute, allowing the fund to run the entire scanning pipeline on a single GPU server, significantly reducing infrastructure costs while maintaining classification quality.

### Stock Market Features (Input Data)

| Feature | Description | Example |
|---------|-------------|---------|
| chart_image | Multi-indicator chart image | 224x224 RGB |
| timeframe | Chart period | 1min, 5min, 1hour, daily, weekly |
| ticker | Stock symbol | AAPL, TSLA |
| indicators | Overlaid technical indicators | SMA, EMA, Bollinger Bands, MACD |
| volume_included | Volume bars rendered | True/False |
| chart_style | Rendering format | candlestick, OHLC, Heikin-Ashi |
| pattern_label | Classification target | uptrend, downtrend, reversal, breakout, consolidation |
| signal_strength | Strength of detected pattern | strong, moderate, weak |

### Example Data Structure

```python
import numpy as np

# Multi-timeframe chart data structure
class MultiTimeframeChartData:
    def __init__(self):
        self.timeframes = ['1min', '5min', '1hour', 'daily', 'weekly']
        self.pattern_classes = [
            'uptrend', 'downtrend', 'bullish_reversal',
            'bearish_reversal', 'breakout_up', 'breakout_down',
            'consolidation', 'no_pattern'
        ]
        self.label_to_idx = {p: i for i, p in enumerate(self.pattern_classes)}

    def generate_chart_image(self, trend='uptrend', size=64):
        """Generate a synthetic chart image for a given trend type."""
        image = np.ones((size, size, 3), dtype=np.uint8) * 240  # Light background
        n_bars = 40
        bar_w = max(1, size // (n_bars + 2))
        base = size // 2

        for i in range(n_bars):
            x = (i + 1) * (size // (n_bars + 1))
            noise = np.random.randn() * 3

            if trend == 'uptrend':
                y_center = base - i * 0.4 + noise
            elif trend == 'downtrend':
                y_center = base + i * 0.4 + noise
            elif trend == 'consolidation':
                y_center = base + np.sin(i * 0.5) * 3 + noise
            elif trend in ['bullish_reversal', 'breakout_up']:
                y_center = base + max(0, 15 - i) * 0.5 + noise
            elif trend in ['bearish_reversal', 'breakout_down']:
                y_center = base - max(0, 15 - i) * 0.5 + noise
            else:
                y_center = base + noise

            y = int(np.clip(y_center, 5, size - 5))
            body_h = max(1, int(np.random.randint(2, 6)))

            # Draw candle
            color = [0, 150, 0] if np.random.random() > 0.4 else [200, 0, 0]
            for dy in range(-body_h, body_h + 1):
                py = max(0, min(size - 1, y + dy))
                for dx in range(bar_w):
                    px = max(0, min(size - 1, x + dx))
                    image[py, px] = color

            # Draw wick
            wick_h = body_h + np.random.randint(1, 4)
            for dy in range(-wick_h, wick_h + 1):
                py = max(0, min(size - 1, y + dy))
                cx = max(0, min(size - 1, x + bar_w // 2))
                image[py, cx] = [80, 80, 80]

        return image

    def generate_multi_timeframe_batch(self, ticker, n_per_tf=10):
        """Generate chart images across all timeframes for a stock."""
        data = []
        for tf in self.timeframes:
            for _ in range(n_per_tf):
                pattern = np.random.choice(self.pattern_classes)
                img = self.generate_chart_image(pattern, size=64)
                data.append({
                    'ticker': ticker,
                    'timeframe': tf,
                    'image': img,
                    'label': self.label_to_idx[pattern],
                    'pattern_name': pattern
                })
        return data

# Generate data
chart_data = MultiTimeframeChartData()
sample_data = chart_data.generate_multi_timeframe_batch('AAPL', n_per_tf=5)
print(f"Generated {len(sample_data)} chart images across {len(chart_data.timeframes)} timeframes")
print(f"Pattern classes: {chart_data.pattern_classes}")
print(f"Image shape: {sample_data[0]['image'].shape}")
```

### The Model in Action

```python
class MBConvBlock:
    """Mobile Inverted Bottleneck Convolution block."""

    def __init__(self, in_ch, out_ch, expand_ratio=6, stride=1, se_ratio=0.25):
        self.stride = stride
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = int(in_ch * expand_ratio)
        se_ch = max(1, int(in_ch * se_ratio))

        scale = lambda fan_in: np.sqrt(2.0 / fan_in)

        # Expansion 1x1 conv
        if expand_ratio != 1:
            self.expand_w = np.random.randn(mid_ch, in_ch, 1, 1) * scale(in_ch)
            self.expand_b = np.zeros(mid_ch)
            self.has_expand = True
        else:
            mid_ch = in_ch
            self.has_expand = False

        # Depthwise 3x3 conv (simplified as regular conv for demo)
        self.dw_w = np.random.randn(mid_ch, mid_ch, 3, 3) * scale(mid_ch * 9)
        self.dw_b = np.zeros(mid_ch)

        # SE block
        self.se_fc1_w = np.random.randn(se_ch, mid_ch) * scale(mid_ch)
        self.se_fc1_b = np.zeros(se_ch)
        self.se_fc2_w = np.random.randn(mid_ch, se_ch) * scale(se_ch)
        self.se_fc2_b = np.zeros(mid_ch)

        # Projection 1x1 conv
        self.proj_w = np.random.randn(out_ch, mid_ch, 1, 1) * scale(mid_ch)
        self.proj_b = np.zeros(out_ch)

    def _conv(self, x, w, b, s=1, p=0):
        if p > 0:
            x = np.pad(x, ((0,0),(p,p),(p,p)), mode='constant')
        nf, nc, kh, kw = w.shape
        _, ih, iw = x.shape
        oh, ow = (ih-kh)//s+1, (iw-kw)//s+1
        out = np.zeros((nf, oh, ow))
        for f in range(nf):
            for i in range(oh):
                for j in range(ow):
                    out[f,i,j] = np.sum(x[:, i*s:i*s+kh, j*s:j*s+kw] * w[f]) + b[f]
        return out

    def _swish(self, x):
        return x / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        identity = x

        # Expansion
        if self.has_expand:
            out = self._swish(self._conv(x, self.expand_w, self.expand_b))
        else:
            out = x

        # Depthwise conv
        out = self._swish(self._conv(out, self.dw_w, self.dw_b, s=self.stride, p=1))

        # Squeeze-and-Excitation
        se = out.mean(axis=(1, 2))  # Global avg pool
        se = np.maximum(0, self.se_fc1_w @ se + self.se_fc1_b)  # ReLU
        se = self._sigmoid(self.se_fc2_w @ se + self.se_fc2_b)
        out = out * se[:, np.newaxis, np.newaxis]

        # Projection
        out = self._conv(out, self.proj_w, self.proj_b)

        # Residual connection
        if self.use_residual:
            out = out + identity

        return out


class SimpleEfficientNet:
    """Simplified EfficientNet for multi-timeframe chart pattern recognition."""

    def __init__(self, n_classes=8, image_size=64):
        self.n_classes = n_classes
        np.random.seed(42)

        # Stem convolution
        self.stem_w = np.random.randn(16, 3, 3, 3) * np.sqrt(2/27)
        self.stem_b = np.zeros(16)

        # MBConv blocks (simplified)
        self.block1 = MBConvBlock(16, 16, expand_ratio=1, stride=1)
        self.block2 = MBConvBlock(16, 24, expand_ratio=6, stride=2)
        self.block3 = MBConvBlock(24, 40, expand_ratio=6, stride=2)
        self.block4 = MBConvBlock(40, 80, expand_ratio=6, stride=2)

        # Head
        self.head_w = np.random.randn(128, 80, 1, 1) * np.sqrt(2/80)
        self.head_b = np.zeros(128)

        self._fc_ready = False
        self.pattern_names = [
            'uptrend', 'downtrend', 'bullish_reversal',
            'bearish_reversal', 'breakout_up', 'breakout_down',
            'consolidation', 'no_pattern'
        ]

    def _conv(self, x, w, b, s=1, p=0):
        if p > 0:
            x = np.pad(x, ((0,0),(p,p),(p,p)), mode='constant')
        nf, nc, kh, kw = w.shape
        _, ih, iw = x.shape
        oh, ow = (ih-kh)//s+1, (iw-kw)//s+1
        out = np.zeros((nf, oh, ow))
        for f in range(nf):
            for i in range(oh):
                for j in range(ow):
                    out[f,i,j] = np.sum(x[:, i*s:i*s+kh, j*s:j*s+kw]*w[f]) + b[f]
        return out

    def _swish(self, x):
        return x / (1 + np.exp(-np.clip(x, -500, 500)))

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, image):
        x = image.astype(np.float32) / 255.0
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.transpose(2, 0, 1)

        # Stem
        x = self._swish(self._conv(x, self.stem_w, self.stem_b, s=1, p=1))

        # MBConv blocks
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.block4.forward(x)

        # Head
        x = self._swish(self._conv(x, self.head_w, self.head_b))

        # Global average pooling
        features = x.mean(axis=(1, 2))

        if not self._fc_ready:
            self.fc_w = np.random.randn(self.n_classes, len(features)) * np.sqrt(2/len(features))
            self.fc_b = np.zeros(self.n_classes)
            self._fc_ready = True

        logits = self.fc_w @ features + self.fc_b
        probs = self._softmax(logits)
        return probs, features

    def classify(self, image):
        probs, features = self.forward(image)
        pred = np.argmax(probs)
        pattern = self.pattern_names[pred]

        # Generate trading signal
        bullish_patterns = ['uptrend', 'bullish_reversal', 'breakout_up']
        bearish_patterns = ['downtrend', 'bearish_reversal', 'breakout_down']

        if pattern in bullish_patterns:
            signal = 'BUY'
        elif pattern in bearish_patterns:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'pattern': pattern,
            'confidence': float(probs[pred]),
            'signal': signal,
            'probabilities': {n: float(p) for n, p in zip(self.pattern_names, probs)}
        }

# Run multi-timeframe analysis
model = SimpleEfficientNet(n_classes=8, image_size=64)

print("=" * 70)
print("EfficientNet MULTI-TIMEFRAME PATTERN SCANNER")
print("=" * 70)

for sample in sample_data[:10]:
    result = model.classify(sample['image'])
    print(f"\n{sample['ticker']} [{sample['timeframe']:6s}] "
          f"True: {sample['pattern_name']:20s} | "
          f"Pred: {result['pattern']:20s} "
          f"({result['confidence']:.0%}) -> {result['signal']}")
```

## Advantages

1. **Superior Efficiency-Accuracy Trade-off**: EfficientNet-B0 achieves accuracy comparable to ResNet-50 with 8.4x fewer parameters and 6.1x fewer FLOPs. For stock market scanning across thousands of charts, this efficiency translates directly to lower infrastructure costs and higher throughput.

2. **Compound Scaling Produces Balanced Networks**: Unlike ad-hoc scaling of width, depth, or resolution independently, compound scaling ensures all dimensions grow proportionally. This balanced architecture captures chart patterns at appropriate scales without wasting parameters on underutilized dimensions.

3. **Squeeze-and-Excitation Attention**: The SE blocks in MBConv allow the network to learn which feature channels are most important for each input. For chart images, this means the model can dynamically emphasize trend-related features for trending charts and pattern-related features for reversal patterns.

4. **Multiple Model Sizes for Different Deployment Needs**: EfficientNet-B0 (5.3M params) works well on edge devices for individual traders, while B3 (12M params) provides higher accuracy for server-based scanning. This flexibility allows firms to deploy appropriate models at different points in their infrastructure.

5. **Depthwise Separable Convolutions Reduce Computation**: By separating spatial and channel-wise convolutions, EfficientNet dramatically reduces the number of operations per layer. This is particularly impactful for high-resolution chart images where standard convolutions would be prohibitively expensive.

6. **Excellent Transfer Learning from ImageNet**: Pre-trained EfficientNet models transfer well to chart image classification. The Swish activation and SE blocks help the model adapt quickly to financial domain features during fine-tuning.

7. **Stochastic Depth Regularization**: The built-in stochastic depth during training acts as a powerful regularizer, preventing overfitting on the typically small labeled datasets available for chart pattern classification.

## Disadvantages

1. **Complex Architecture Implementation**: MBConv blocks with SE attention, depthwise convolutions, and stochastic depth are significantly more complex to implement and debug than standard ResNet blocks. Custom optimizations for financial chart inputs require deep understanding of the architecture.

2. **Depthwise Convolutions Have Hardware Limitations**: Depthwise convolutions have low arithmetic intensity and are memory-bandwidth bound on GPUs. This means the theoretical FLOP savings do not always translate to proportional wall-clock speedups, especially on older GPU hardware.

3. **Sensitive to Image Resolution**: EfficientNet's compound scaling ties model size to input resolution. Using a resolution different from the prescribed one (e.g., 224x224 for B0) can degrade performance. Chart images may have non-standard aspect ratios that require careful resizing.

4. **Training Instability with Large Batch Sizes**: EfficientNet can be difficult to train with very large batch sizes due to the interaction between batch normalization and stochastic depth. This limits the degree of data parallelism for distributed training on large chart datasets.

5. **Swish Activation Overhead**: While Swish provides better gradients than ReLU, it is computationally more expensive due to the sigmoid computation. On CPU-only deployments, this overhead can be significant compared to simple ReLU-based architectures.

6. **NAS-Derived Architecture is Not Intuitive**: The base architecture was found through Neural Architecture Search (NAS), making it harder to intuitively understand why specific design choices were made. Modifying the architecture for financial chart-specific improvements requires careful experimentation.

## When to Use in Stock Market

- **Multi-timeframe scanning**: When analyzing charts across 5+ timeframes for thousands of stocks
- **Resource-constrained deployment**: When GPU budget is limited but accuracy requirements are high
- **Edge deployment**: For individual trader workstations without dedicated GPU servers
- **Large-scale backtesting**: When processing millions of historical chart images for strategy research
- **Production systems with SLA requirements**: When both accuracy and inference time must meet specific thresholds
- **Mobile trading applications**: When chart pattern alerts need to run on mobile devices

## When NOT to Use in Stock Market

- **When simplicity is paramount**: If a 3-layer CNN solves the problem, adding EfficientNet complexity is unjustified
- **Very low latency requirements**: When even a few extra milliseconds from SE blocks and Swish is unacceptable
- **Extremely small datasets**: When fewer than 200 labeled chart images are available
- **Custom hardware**: When deploying on FPGAs or ASICs where depthwise convolutions are not well-supported
- **Rapidly evolving architectures**: When the latest vision transformers may be more appropriate

## Hyperparameters Guide

| Hyperparameter | Typical Range | Stock Market Recommendation | Effect |
|---------------|---------------|---------------------------|--------|
| model_variant | B0 - B7 | B0 - B3 | Larger = more accurate but slower |
| image_size | 224 - 600 | 224 (B0), 300 (B3) | Must match variant's prescribed resolution |
| learning_rate | 1e-4 to 1e-2 | 1e-3 (scratch), 1e-4 (finetune) | Lower for fine-tuning pre-trained models |
| batch_size | 16 - 256 | 32 - 64 | Moderate for stable BN statistics |
| dropout | 0.2 - 0.5 | 0.3 (B0), 0.4 (B3) | Higher dropout for larger models |
| stochastic_depth | 0.0 - 0.3 | 0.2 | Survival probability for residual blocks |
| weight_decay | 1e-5 to 1e-3 | 1e-4 | Standard regularization |
| optimizer | Adam, RMSprop | RMSprop (original), AdamW (modern) | RMSprop was used in original paper |
| se_ratio | 0.0 - 0.5 | 0.25 | Squeeze-and-excitation reduction ratio |

## Stock Market Performance Tips

1. **Use compound scaling for multi-timeframe**: Apply different EfficientNet variants for different timeframes. B0 for intraday (speed priority), B2-B3 for daily/weekly (accuracy priority). This matches model capacity to the complexity of patterns at each timeframe.

2. **Multi-task learning**: Train a single EfficientNet with multiple output heads: one for trend classification, one for pattern type, and one for signal strength. Shared features reduce total compute compared to running three separate models.

3. **Knowledge distillation for deployment**: Train a large EfficientNet-B5 as a teacher and distill into an EfficientNet-B0 student. The student learns from the teacher's soft probability distributions, achieving B2-level accuracy at B0 speed.

4. **Quantize for production**: Apply INT8 quantization to reduce model size by 4x and improve inference speed by 2-3x on CPU. EfficientNet responds well to quantization with minimal accuracy loss (typically <1%).

5. **Use test-time augmentation (TTA)**: For critical trading signals, run inference on the original chart image plus 2-4 augmented versions (slight scale/crop variations) and average the predictions. This reduces false signals with modest compute overhead.

6. **Integrate with attention visualization**: Use Grad-CAM on SE block outputs to visualize which chart regions the model considers most important. This provides interpretability for risk management review of automated trading signals.

## Comparison with Other Algorithms

| Feature | EfficientNet-B0 | EfficientNet-B3 | ResNet-50 | MobileNetV3 | ViT-Small |
|---------|-----------------|-----------------|-----------|-------------|-----------|
| Parameters | 5.3M | 12M | 25.6M | 5.4M | 22M |
| FLOPs | 0.39B | 1.8B | 4.1B | 0.22B | 4.6B |
| Top-1 Accuracy (ImageNet) | 77.1% | 81.6% | 76.1% | 75.2% | 81.4% |
| Chart Pattern Accuracy | 84-88% | 88-92% | 85-90% | 80-85% | 87-91% |
| Inference (ms, GPU) | 4-8 | 8-15 | 10-25 | 3-6 | 15-30 |
| Model Size (MB) | 20 | 48 | 98 | 22 | 86 |
| Transfer Learning | Excellent | Excellent | Excellent | Good | Excellent |
| Mobile Deployable | Yes | Marginal | No | Yes | No |

## Real-World Stock Market Example

```python
import numpy as np
from datetime import datetime

class MultiTimeframeEfficientNetScanner:
    """
    Complete EfficientNet-based multi-timeframe chart pattern scanner.
    """

    def __init__(self, n_classes=8):
        self.n_classes = n_classes
        self.patterns = [
            'uptrend', 'downtrend', 'bullish_reversal',
            'bearish_reversal', 'breakout_up', 'breakout_down',
            'consolidation', 'no_pattern'
        ]
        self.timeframe_weights = {
            'weekly': 2.0,
            'daily': 1.5,
            '1hour': 1.0,
            '5min': 0.5,
            '1min': 0.3
        }

        np.random.seed(42)
        self._init_model()

    def _init_model(self):
        """Initialize simplified EfficientNet weights."""
        self.stem_w = np.random.randn(16, 3, 3, 3) * np.sqrt(2/27)
        self.stem_b = np.zeros(16)

        self.blocks = [
            {'w': np.random.randn(24, 16, 3, 3) * np.sqrt(2/(16*9)),
             'b': np.zeros(24), 'stride': 2},
            {'w': np.random.randn(48, 24, 3, 3) * np.sqrt(2/(24*9)),
             'b': np.zeros(48), 'stride': 2},
            {'w': np.random.randn(96, 48, 3, 3) * np.sqrt(2/(48*9)),
             'b': np.zeros(96), 'stride': 2},
        ]
        self._fc_ready = False

    def _conv(self, x, w, b, s=1, p=1):
        if p > 0:
            x = np.pad(x, ((0,0),(p,p),(p,p)), mode='constant')
        nf, nc, kh, kw = w.shape
        _, ih, iw = x.shape
        oh, ow = (ih-kh)//s+1, (iw-kw)//s+1
        out = np.zeros((nf, oh, ow))
        for f in range(nf):
            for i in range(oh):
                for j in range(ow):
                    out[f,i,j] = np.sum(x[:,i*s:i*s+kh,j*s:j*s+kw]*w[f]) + b[f]
        return out

    def _swish(self, x):
        return x / (1 + np.exp(-np.clip(x, -500, 500)))

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def classify_single(self, image):
        x = image.astype(np.float32) / 255.0
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.transpose(2, 0, 1)

        x = self._swish(self._conv(x, self.stem_w, self.stem_b))
        for block in self.blocks:
            x = self._swish(self._conv(x, block['w'], block['b'], s=block['stride']))

        features = x.mean(axis=(1, 2))

        if not self._fc_ready:
            self.fc_w = np.random.randn(self.n_classes, len(features)) * np.sqrt(2/len(features))
            self.fc_b = np.zeros(self.n_classes)
            self._fc_ready = True

        logits = self.fc_w @ features + self.fc_b
        return self._softmax(logits)

    def multi_timeframe_analysis(self, ticker, timeframe_charts):
        """
        Analyze charts across multiple timeframes and produce
        a weighted composite signal.
        """
        results = {}
        composite_probs = np.zeros(self.n_classes)
        total_weight = 0

        for tf, chart_img in timeframe_charts.items():
            probs = self.classify_single(chart_img)
            pred = np.argmax(probs)
            pattern = self.patterns[pred]

            weight = self.timeframe_weights.get(tf, 1.0)
            composite_probs += probs * weight
            total_weight += weight

            results[tf] = {
                'pattern': pattern,
                'confidence': float(probs[pred]),
                'probs': probs
            }

        # Normalize composite probabilities
        composite_probs /= total_weight
        final_pred = np.argmax(composite_probs)
        final_pattern = self.patterns[final_pred]

        # Determine signal
        bullish = ['uptrend', 'bullish_reversal', 'breakout_up']
        bearish = ['downtrend', 'bearish_reversal', 'breakout_down']

        bull_score = sum(composite_probs[self.patterns.index(p)] for p in bullish)
        bear_score = sum(composite_probs[self.patterns.index(p)] for p in bearish)

        if bull_score > bear_score + 0.1:
            signal = 'BUY'
            strength = bull_score
        elif bear_score > bull_score + 0.1:
            signal = 'SELL'
            strength = bear_score
        else:
            signal = 'HOLD'
            strength = 1 - bull_score - bear_score

        return {
            'ticker': ticker,
            'timeframe_results': results,
            'composite_pattern': final_pattern,
            'composite_confidence': float(composite_probs[final_pred]),
            'signal': signal,
            'signal_strength': float(strength),
            'bull_score': float(bull_score),
            'bear_score': float(bear_score),
            'timestamp': datetime.now().isoformat()
        }


# Run full portfolio scan
scanner = MultiTimeframeEfficientNetScanner(n_classes=8)

# Generate synthetic multi-timeframe charts for portfolio
portfolio_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'MA']
timeframes = ['1min', '5min', '1hour', 'daily', 'weekly']

print("=" * 75)
print("EfficientNet MULTI-TIMEFRAME PORTFOLIO SCANNER")
print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Stocks: {len(portfolio_tickers)} | Timeframes: {len(timeframes)} | "
      f"Total Charts: {len(portfolio_tickers) * len(timeframes)}")
print("=" * 75)

all_results = []
for ticker in portfolio_tickers:
    # Generate charts for each timeframe
    tf_charts = {}
    for tf in timeframes:
        tf_charts[tf] = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    result = scanner.multi_timeframe_analysis(ticker, tf_charts)
    all_results.append(result)

# Display results sorted by signal strength
all_results.sort(key=lambda r: r['bull_score'] - r['bear_score'], reverse=True)

for r in all_results:
    tf_summary = " | ".join([
        f"{tf}:{v['pattern'][:4]}" for tf, v in r['timeframe_results'].items()
    ])
    print(f"\n{r['ticker']:5s} => {r['signal']:4s} "
          f"(strength: {r['signal_strength']:.0%}) "
          f"pattern: {r['composite_pattern']}")
    print(f"  Bull: {r['bull_score']:.0%} | Bear: {r['bear_score']:.0%}")
    print(f"  Timeframes: {tf_summary}")

# Summary
buys = [r for r in all_results if r['signal'] == 'BUY']
sells = [r for r in all_results if r['signal'] == 'SELL']
holds = [r for r in all_results if r['signal'] == 'HOLD']

print(f"\n{'='*75}")
print(f"SCAN SUMMARY: BUY={len(buys)} | HOLD={len(holds)} | SELL={len(sells)}")
if buys:
    print(f"Top BUY: {buys[0]['ticker']} ({buys[0]['signal_strength']:.0%} strength)")
if sells:
    print(f"Top SELL: {sells[-1]['ticker']} ({sells[-1]['signal_strength']:.0%} strength)")
```

## Key Takeaways

1. **Compound scaling is the key principle**: Scaling width, depth, and resolution together produces better models than scaling any single dimension. Apply this principle when choosing which EfficientNet variant to deploy for different scanning requirements.

2. **EfficientNet is the best choice** for production multi-timeframe chart scanning where both accuracy and computational efficiency matter. It achieves ResNet-level accuracy at a fraction of the compute cost.

3. **SE attention blocks add significant value** for chart pattern recognition by allowing the model to dynamically weight different feature channels. This helps distinguish between similar-looking patterns that differ in subtle channel-specific features.

4. **Model variant selection matters more than hyperparameter tuning**. Choosing between B0, B2, and B3 has a larger impact on the accuracy-speed trade-off than most hyperparameter changes. Start with B0 and scale up only if accuracy is insufficient.

5. **Multi-timeframe consensus signals are more reliable** than single-timeframe predictions. Weight higher timeframes (weekly, daily) more heavily than lower timeframes (1-min, 5-min) for position decisions, as higher timeframe patterns are more reliable.

6. **Quantization and distillation are essential** for production deployment. INT8 quantization reduces EfficientNet-B0 to approximately 5 MB while maintaining 99% of original accuracy, enabling deployment on mobile and edge devices.

7. **Combine with fundamental and volume data** for complete trading signals. Chart pattern recognition is one component of a robust trading system; always validate visual signals with volume confirmation and fundamental context.
