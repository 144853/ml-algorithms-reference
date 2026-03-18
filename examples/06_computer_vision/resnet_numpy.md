# ResNet (Residual Network) - Complete Guide with Stock Market Applications

## Overview

ResNet (Residual Network), introduced by Kaiming He et al. in 2015, solved one of the most fundamental problems in deep learning: the degradation problem. As neural networks get deeper, they paradoxically become harder to train, with accuracy saturating and then degrading. ResNet addresses this through skip connections (residual connections) that allow gradients to flow directly through the network, enabling the training of extremely deep architectures (50, 101, even 152 layers) that were previously impossible to optimize.

In stock market applications, ResNet's depth advantage is crucial for pattern recognition on stock chart images. While shallow CNNs can detect simple patterns like individual candlestick formations, ResNet can learn complex hierarchical patterns: multi-candle formations, trend structures, support/resistance patterns, and chart patterns that span different scales. The deeper architecture captures both fine-grained details (individual candle shapes) and high-level structures (head-and-shoulders, cup-and-handle, double bottoms) simultaneously.

ResNet's pre-trained versions (trained on ImageNet) provide excellent transfer learning capabilities for financial chart analysis. The low-level features learned from natural images (edges, textures, shapes) transfer surprisingly well to chart images, where edges correspond to price levels, textures to congestion areas, and shapes to pattern formations. This means ResNet can achieve strong performance on chart classification even with relatively small labeled financial datasets (2,000-5,000 images).

## How It Works - The Math Behind It

### The Degradation Problem

In a plain deep network, stacking more layers leads to higher training error (not just test error), suggesting an optimization difficulty. If a deeper network could at least copy the learned features from a shallower network and add identity mappings for extra layers, it should be no worse. But plain networks fail to find this solution.

### Residual Learning

Instead of learning the desired mapping H(x) directly, ResNet learns the residual F(x) = H(x) - x:

```
H(x) = F(x) + x
```

The network only needs to learn the residual (the difference from identity), which is easier to optimize. If the optimal mapping is close to identity, it is much easier to push F(x) toward zero than to fit an identity mapping with a stack of nonlinear layers.

### Skip Connection (Shortcut Connection)

```
y = F(x, {W_i}) + x
```

Where F(x, {W_i}) represents the residual function learned by the stacked layers (typically 2-3 convolutional layers).

### Bottleneck Block (ResNet-50+)

For deeper networks, a bottleneck design reduces computation:

```
1x1 conv (reduce dimensions): 256 -> 64 channels
3x3 conv (learn features): 64 -> 64 channels
1x1 conv (restore dimensions): 64 -> 256 channels

y = F(x) + x  (where F includes all three convolutions)
```

### Dimension Matching

When input and output dimensions differ, a projection shortcut is used:

```
y = F(x, {W_i}) + W_s * x
```

Where W_s is a 1x1 convolution that matches dimensions.

### Batch Normalization

Applied after each convolution:

```
BN(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

### Global Average Pooling

Instead of flattening, ResNet uses global average pooling before the classifier:

```
GAP(feature_map) = (1/(H*W)) * SUM_i SUM_j feature_map(i, j)
```

### Full Forward Pass

```
Input -> Conv(7x7, stride=2) -> BN -> ReLU -> MaxPool(3x3, stride=2)
-> [Residual Block] x N1 (Stage 1: 64 filters)
-> [Residual Block] x N2 (Stage 2: 128 filters, downsample)
-> [Residual Block] x N3 (Stage 3: 256 filters, downsample)
-> [Residual Block] x N4 (Stage 4: 512 filters, downsample)
-> Global Average Pooling -> FC -> Softmax
```

## Stock Market Use Case: Deep Pattern Recognition on Stock Chart Images for Trend Classification

### The Problem

A quantitative fund wants to classify the current market trend of any stock by analyzing its recent chart image. They categorize trends into five classes: strong uptrend, weak uptrend, sideways/consolidation, weak downtrend, and strong downtrend. Unlike simple candlestick pattern detection, trend classification requires understanding multi-scale structures: the overall direction, the strength of moves, consolidation areas, and the relationship between recent price action and established support/resistance levels. A shallow CNN cannot capture all these hierarchical features simultaneously, making ResNet's depth essential.

### Stock Market Features (Input Data)

| Feature | Description | Example |
|---------|-------------|---------|
| chart_image | Rendered stock chart with indicators | 224x224 RGB |
| timeframe | Chart period | Daily, Weekly, Monthly |
| n_bars | Number of price bars shown | 60 - 120 |
| indicators_shown | Technical indicators overlaid | SMA_20, SMA_50, Bollinger |
| volume_panel | Whether volume bars are included | True/False |
| ticker | Stock symbol | AAPL |
| trend_label | Classification target | strong_up, weak_up, sideways, weak_down, strong_down |
| chart_type | Visual representation style | candlestick, OHLC, line |

### Example Data Structure

```python
import numpy as np

# Synthetic stock chart data generator for trend classification
class StockChartImageGenerator:
    def __init__(self, image_size=224, n_bars=60):
        self.image_size = image_size
        self.n_bars = n_bars

    def _generate_price_series(self, trend_type, n_points=60):
        """Generate synthetic OHLC data for different trend types."""
        np.random.seed()
        noise = np.random.randn(n_points) * 0.5
        base = 100.0

        if trend_type == 'strong_up':
            trend = np.linspace(0, 30, n_points) + noise
        elif trend_type == 'weak_up':
            trend = np.linspace(0, 10, n_points) + noise * 1.5
        elif trend_type == 'sideways':
            trend = np.sin(np.linspace(0, 4*np.pi, n_points)) * 5 + noise * 2
        elif trend_type == 'weak_down':
            trend = np.linspace(0, -10, n_points) + noise * 1.5
        elif trend_type == 'strong_down':
            trend = np.linspace(0, -30, n_points) + noise

        closes = base + trend
        opens = closes + np.random.randn(n_points) * 0.5
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_points)) * 1.0
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_points)) * 1.0

        return opens, highs, lows, closes

    def render_chart(self, opens, highs, lows, closes):
        """Render OHLC data as a chart image."""
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        # Normalize prices to image coordinates
        all_prices = np.concatenate([highs, lows])
        p_min, p_max = all_prices.min(), all_prices.max()
        p_range = p_max - p_min if p_max > p_min else 1.0

        margin = 20
        chart_h = self.image_size - 2 * margin
        chart_w = self.image_size - 2 * margin

        bar_width = max(1, chart_w // len(closes) - 1)

        for i in range(len(closes)):
            x = margin + i * (chart_w // len(closes))
            # Normalize to pixel coordinates (inverted y-axis)
            oy = margin + int((1 - (opens[i] - p_min) / p_range) * chart_h)
            cy = margin + int((1 - (closes[i] - p_min) / p_range) * chart_h)
            hy = margin + int((1 - (highs[i] - p_min) / p_range) * chart_h)
            ly = margin + int((1 - (lows[i] - p_min) / p_range) * chart_h)

            # Clamp to image bounds
            for val in [oy, cy, hy, ly]:
                val = max(margin, min(self.image_size - margin - 1, val))

            # Draw wick
            for y in range(min(hy, ly), max(hy, ly) + 1):
                y = max(0, min(self.image_size - 1, y))
                cx = min(x + bar_width // 2, self.image_size - 1)
                image[y, cx] = [100, 100, 100]

            # Draw body
            color = [34, 139, 34] if closes[i] >= opens[i] else [220, 20, 60]
            body_top = min(oy, cy)
            body_bot = max(oy, cy)
            for y in range(max(0, body_top), min(self.image_size, body_bot + 1)):
                for dx in range(bar_width):
                    px = min(x + dx, self.image_size - 1)
                    image[y, px] = color

        return image

    def generate_dataset(self, n_per_class=200):
        trend_types = ['strong_up', 'weak_up', 'sideways', 'weak_down', 'strong_down']
        images, labels = [], []
        label_map = {t: i for i, t in enumerate(trend_types)}

        for trend in trend_types:
            for _ in range(n_per_class):
                o, h, l, c = self._generate_price_series(trend, self.n_bars)
                img = self.render_chart(o, h, l, c)
                images.append(img)
                labels.append(label_map[trend])

        images = np.array(images)
        labels = np.array(labels)
        perm = np.random.permutation(len(images))
        return images[perm], labels[perm], label_map

# Generate dataset
gen = StockChartImageGenerator(image_size=64, n_bars=60)
X_train, y_train, label_map = gen.generate_dataset(n_per_class=20)
print(f"Training data shape: {X_train.shape}")
print(f"Number of classes: {len(label_map)}")
print(f"Classes: {label_map}")
```

### The Model in Action

```python
class ResidualBlock:
    """Single residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        scale = np.sqrt(2.0 / (out_channels * 3 * 3))
        self.conv1_w = np.random.randn(out_channels, in_channels, 3, 3) * scale
        self.conv1_b = np.zeros(out_channels)
        self.conv2_w = np.random.randn(out_channels, out_channels, 3, 3) * scale
        self.conv2_b = np.zeros(out_channels)
        self.stride = stride
        self.use_projection = (in_channels != out_channels) or (stride != 1)
        if self.use_projection:
            self.proj_w = np.random.randn(out_channels, in_channels, 1, 1) * \
                         np.sqrt(2.0 / in_channels)

    def _conv2d(self, x, w, b, stride=1, pad=1):
        if pad > 0:
            x = np.pad(x, ((0,0), (pad,pad), (pad,pad)), mode='constant')
        nf, nc, kh, kw = w.shape
        _, ih, iw = x.shape
        oh = (ih - kh) // stride + 1
        ow = (iw - kw) // stride + 1
        out = np.zeros((nf, oh, ow))
        for f in range(nf):
            for i in range(oh):
                for j in range(ow):
                    out[f, i, j] = np.sum(
                        x[:, i*stride:i*stride+kh, j*stride:j*stride+kw] * w[f]
                    ) + b[f]
        return out

    def forward(self, x):
        identity = x

        out = self._conv2d(x, self.conv1_w, self.conv1_b, stride=self.stride)
        out = np.maximum(0, out)  # ReLU

        out = self._conv2d(out, self.conv2_w, self.conv2_b, stride=1)

        if self.use_projection:
            identity = self._conv2d(x, self.proj_w, np.zeros(self.proj_w.shape[0]),
                                    stride=self.stride, pad=0)

        out = out + identity  # Skip connection
        out = np.maximum(0, out)  # ReLU
        return out


class SimpleResNet:
    """Simplified ResNet for stock chart trend classification."""

    def __init__(self, n_classes=5, image_size=64):
        self.n_classes = n_classes
        self.image_size = image_size
        np.random.seed(42)

        # Initial convolution
        self.conv1_w = np.random.randn(16, 3, 3, 3) * np.sqrt(2/(3*3*3))
        self.conv1_b = np.zeros(16)

        # Residual blocks
        self.block1 = ResidualBlock(16, 16, stride=1)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.block3 = ResidualBlock(32, 64, stride=2)

        self._fc_init = False
        self.trend_names = ['strong_up', 'weak_up', 'sideways', 'weak_down', 'strong_down']
        self.signal_map = {
            'strong_up': 'STRONG BUY - Clear uptrend momentum',
            'weak_up': 'BUY - Modest upward bias',
            'sideways': 'HOLD - Range-bound, wait for breakout',
            'weak_down': 'SELL - Modest downward bias',
            'strong_down': 'STRONG SELL - Clear downtrend momentum'
        }

    def _conv2d(self, x, w, b, stride=1, pad=1):
        if pad > 0:
            x = np.pad(x, ((0,0), (pad,pad), (pad,pad)), mode='constant')
        nf, nc, kh, kw = w.shape
        _, ih, iw = x.shape
        oh = (ih - kh) // stride + 1
        ow = (iw - kw) // stride + 1
        out = np.zeros((nf, oh, ow))
        for f in range(nf):
            for i in range(oh):
                for j in range(ow):
                    out[f, i, j] = np.sum(
                        x[:, i*stride:i*stride+kh, j*stride:j*stride+kw] * w[f]
                    ) + b[f]
        return out

    def _global_avg_pool(self, x):
        return x.mean(axis=(1, 2))

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, image):
        x = image.astype(np.float32) / 255.0
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.transpose(2, 0, 1)

        # Initial conv
        x = np.maximum(0, self._conv2d(x, self.conv1_w, self.conv1_b))

        # Residual blocks with skip connections
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)

        # Global average pooling
        features = self._global_avg_pool(x)

        # FC layer
        if not self._fc_init:
            self.fc_w = np.random.randn(self.n_classes, len(features)) * np.sqrt(2/len(features))
            self.fc_b = np.zeros(self.n_classes)
            self._fc_init = True

        logits = self.fc_w @ features + self.fc_b
        probs = self._softmax(logits)
        return probs, features

    def classify_trend(self, chart_image):
        probs, features = self.forward(chart_image)
        pred_idx = np.argmax(probs)
        trend = self.trend_names[pred_idx]
        return {
            'trend': trend,
            'confidence': float(probs[pred_idx]),
            'signal': self.signal_map[trend],
            'probabilities': {n: float(p) for n, p in zip(self.trend_names, probs)},
            'feature_dim': len(features)
        }

# Run trend classification
resnet = SimpleResNet(n_classes=5, image_size=64)

print("=" * 70)
print("ResNet STOCK CHART TREND CLASSIFIER")
print("=" * 70)

for i in range(min(8, len(X_train))):
    result = resnet.classify_trend(X_train[i])
    true_trend = list(label_map.keys())[y_train[i]]
    print(f"\nSample {i+1}:")
    print(f"  True Trend: {true_trend}")
    print(f"  Predicted:  {result['trend']} ({result['confidence']:.1%})")
    print(f"  Signal: {result['signal']}")
```

## Advantages

1. **Trains Much Deeper Networks**: ResNet's skip connections solve the vanishing gradient problem, allowing 50-152 layer networks that learn rich hierarchical features from stock charts. This depth is essential for recognizing complex multi-scale patterns like head-and-shoulders or cup-and-handle formations that span many price bars.

2. **Superior Feature Hierarchy**: The residual blocks learn features at multiple abstraction levels simultaneously. Low-level blocks detect edges and candle shapes, mid-level blocks recognize pattern components (double tops, support lines), and high-level blocks compose these into complete trend assessments. This hierarchy matches how expert chartists analyze markets.

3. **Excellent Transfer Learning Performance**: Pre-trained ResNet models (ImageNet) transfer remarkably well to stock chart classification. The low-level features (edges, gradients, textures) learned from natural images apply directly to chart features, reducing the need for large labeled financial datasets.

4. **Skip Connections Preserve Fine Details**: In plain deep networks, information about subtle chart features (thin wicks, small gaps) can be lost through many layers. Skip connections ensure these fine-grained details reach deeper layers, improving classification of patterns that depend on precise visual details.

5. **Well-Established Architecture with Proven Results**: ResNet has been extensively studied and benchmarked since 2015. Optimized implementations exist in all major frameworks (PyTorch, TensorFlow), with well-understood training recipes and hyperparameter guidelines that reduce experimentation time.

6. **Scalable Depth for Different Complexities**: ResNet comes in variants (ResNet-18, 34, 50, 101, 152) allowing you to match model complexity to task difficulty. Simple trend classification may work with ResNet-18, while complex multi-pattern recognition benefits from ResNet-50 or deeper.

## Disadvantages

1. **Computationally Expensive**: ResNet-50 has 25.6 million parameters and requires significant GPU memory and compute. Processing 224x224 chart images through a full ResNet takes 10-30ms per image, limiting throughput for scanning large stock universes in real-time.

2. **Overkill for Simple Patterns**: For straightforward trend classification or single candlestick pattern detection, ResNet's depth is unnecessary. A simpler CNN may achieve similar accuracy with 10x less computation, making ResNet a poor choice when simpler models suffice.

3. **Large Model Size for Deployment**: A ResNet-50 model is approximately 100 MB, and deploying multiple models (one per timeframe or pattern type) requires significant storage and memory. Edge deployment for mobile trading applications is challenging without model compression.

4. **Requires Substantial Training Data**: While transfer learning helps, fine-tuning ResNet effectively still requires at least 2,000-5,000 labeled chart images per class. Creating this labeled dataset with consistent expert annotations is a significant upfront investment.

5. **Black Box Nature**: Despite producing accurate classifications, ResNet provides minimal insight into which visual features drove the prediction. Gradient-based visualization methods (Grad-CAM) can highlight important regions, but explaining exactly why a chart was classified as "strong uptrend" in human-understandable terms remains difficult.

6. **Sensitivity to Image Quality**: ResNet is sensitive to chart rendering details: resolution, color scheme, indicator overlay, and background style. Models trained on one charting platform may perform poorly on charts from a different provider without additional fine-tuning or augmentation.

## When to Use in Stock Market

- **Complex pattern recognition**: When detecting multi-candle and multi-scale chart patterns that shallow CNNs miss
- **Trend classification**: For categorizing the overall market structure of stock charts
- **Transfer learning scenarios**: When you have limited labeled financial data and want to leverage ImageNet pre-training
- **Research and backtesting**: When accuracy is more important than inference speed
- **Multi-pattern detection**: When identifying multiple pattern types simultaneously in a single chart
- **High-accuracy requirements**: When trading signal quality must exceed what simpler models provide

## When NOT to Use in Stock Market

- **Real-time HFT systems**: When sub-millisecond latency is required
- **Simple pattern detection**: When basic candlestick patterns can be detected with rules or shallow CNNs
- **Resource-constrained environments**: When GPU access is limited or deploying to mobile devices
- **Small datasets**: When fewer than 500 labeled examples per class are available
- **When numerical features suffice**: If OHLC data with technical indicators provides adequate signal

## Hyperparameters Guide

| Hyperparameter | Typical Range | Stock Market Recommendation | Effect |
|---------------|---------------|---------------------------|--------|
| architecture | ResNet-18/34/50/101 | ResNet-34 or ResNet-50 | Deeper for complex patterns |
| learning_rate | 1e-5 to 1e-3 | 1e-4 (fine-tune), 1e-3 (scratch) | Lower for fine-tuning pre-trained |
| batch_size | 16 - 64 | 32 | Limited by GPU memory with 224x224 images |
| image_size | 64 - 384 | 224 | Standard ImageNet size for transfer learning |
| optimizer | SGD, Adam, AdamW | AdamW | Weight decay helps prevent overfitting |
| weight_decay | 1e-4 to 1e-2 | 1e-3 | Regularization for small financial datasets |
| epochs | 10 - 100 | 20 - 50 | Early stopping based on validation loss |
| dropout | 0.0 - 0.5 | 0.3 | Add before final FC layer |
| augmentation | various | flip, scale, color jitter | Critical for chart image robustness |

## Stock Market Performance Tips

1. **Freeze early layers during fine-tuning**: When using pre-trained ResNet, freeze the first 2-3 stages and only fine-tune the later stages. Early layers contain general edge/texture detectors that are already well-suited for chart images.

2. **Use channel-wise attention**: Add a squeeze-and-excitation (SE) block after residual blocks to allow the network to emphasize important feature channels (e.g., channels detecting trend lines vs. volume patterns).

3. **Multi-scale input processing**: Feed the same chart at multiple resolutions (zoom levels) through separate ResNet branches and fuse features. This captures both fine-grained patterns and broad trend structures.

4. **Grad-CAM visualization for validation**: Use gradient-weighted class activation mapping to visualize which regions of the chart the model focuses on. Verify that attention aligns with actual pattern locations rather than artifacts.

5. **Temporal data augmentation**: Create training augmentations specific to financial charts: shift the visible window, add or remove indicators, change candle colors/widths, and vary the number of bars shown.

6. **Ensemble multiple architectures**: Combine ResNet predictions with EfficientNet and basic CNN predictions. Different architectures attend to different chart features, and their ensemble is more robust than any individual model.

## Comparison with Other Algorithms

| Feature | ResNet | Basic CNN | EfficientNet | ViT (Vision Transformer) | Rule-Based |
|---------|--------|-----------|-------------|------------------------|------------|
| Accuracy | 85-92% | 78-85% | 87-93% | 88-94% | 60-75% |
| Parameters | 11-60M | 0.5-5M | 5-30M | 86-307M | 0 |
| Inference Speed | 10-30ms | 5-15ms | 8-25ms | 20-50ms | <1ms |
| Depth | 18-152 layers | 3-8 layers | 18-84 layers | 12-24 layers | N/A |
| Transfer Learning | Excellent | Limited | Excellent | Excellent | N/A |
| Model Size | 50-200 MB | 5-50 MB | 20-100 MB | 300-1200 MB | <1 MB |
| GPU Required | Yes | Recommended | Yes | Yes | No |
| Multi-Scale Features | Excellent | Limited | Excellent | Good | None |

## Real-World Stock Market Example

```python
import numpy as np

class StockTrendResNet:
    """
    Complete ResNet-based stock chart trend classifier.
    """

    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.trends = ['strong_up', 'weak_up', 'sideways', 'weak_down', 'strong_down']
        np.random.seed(42)

        # Simplified ResNet architecture
        self.stem_w = np.random.randn(16, 3, 3, 3) * np.sqrt(2/(27))
        self.stem_b = np.zeros(16)

        self.blocks = [
            ResidualBlock(16, 16, stride=1),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 64, stride=2),
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
                    out[f,i,j] = np.sum(x[:, i*s:i*s+kh, j*s:j*s+kw]*w[f]) + b[f]
        return out

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict(self, image):
        x = image.astype(np.float32) / 255.0
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.transpose(2, 0, 1)

        # Stem
        x = np.maximum(0, self._conv(x, self.stem_w, self.stem_b))

        # Residual blocks
        for block in self.blocks:
            x = block.forward(x)

        # Global average pooling
        features = x.mean(axis=(1, 2))

        if not self._fc_ready:
            self.fc_w = np.random.randn(self.n_classes, len(features)) * np.sqrt(2/len(features))
            self.fc_b = np.zeros(self.n_classes)
            self._fc_ready = True

        logits = self.fc_w @ features + self.fc_b
        probs = self._softmax(logits)
        return probs

    def analyze_stock(self, ticker, chart_image):
        probs = self.predict(chart_image)
        pred = np.argmax(probs)
        trend = self.trends[pred]

        strength = abs(probs[0] + probs[1] - probs[3] - probs[4])
        position_size = "FULL" if strength > 0.6 else "HALF" if strength > 0.3 else "QUARTER"

        return {
            'ticker': ticker,
            'trend': trend,
            'confidence': float(probs[pred]),
            'trend_score': float(probs[0]*2 + probs[1] - probs[3] - probs[4]*2),
            'position_suggestion': position_size,
            'probabilities': {t: float(p) for t, p in zip(self.trends, probs)}
        }

# Run analysis
model = StockTrendResNet(n_classes=5)

# Simulate portfolio analysis
stocks = {
    'AAPL': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    'MSFT': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    'GOOGL': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    'NVDA': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    'TSLA': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    'AMZN': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    'META': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    'JPM': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
}

print("=" * 70)
print("ResNet PORTFOLIO TREND ANALYSIS")
print("=" * 70)

results = []
for ticker, chart in stocks.items():
    result = model.analyze_stock(ticker, chart)
    results.append(result)

# Sort by trend score (most bullish first)
results.sort(key=lambda r: r['trend_score'], reverse=True)

for r in results:
    emoji_map = {'strong_up': '^', 'weak_up': '/', 'sideways': '-',
                 'weak_down': '\\', 'strong_down': 'v'}
    indicator = emoji_map.get(r['trend'], '?')
    print(f"\n{r['ticker']:6s} [{indicator}] {r['trend']:12s} "
          f"conf={r['confidence']:.0%} score={r['trend_score']:+.2f} "
          f"size={r['position_suggestion']}")

# Portfolio summary
bullish = [r for r in results if r['trend'] in ['strong_up', 'weak_up']]
bearish = [r for r in results if r['trend'] in ['strong_down', 'weak_down']]
neutral = [r for r in results if r['trend'] == 'sideways']

print(f"\n{'='*70}")
print(f"PORTFOLIO SUMMARY")
print(f"Bullish: {len(bullish)} | Neutral: {len(neutral)} | Bearish: {len(bearish)}")
print(f"Overall bias: {'BULLISH' if len(bullish) > len(bearish) else 'BEARISH' if len(bearish) > len(bullish) else 'NEUTRAL'}")
```

## Key Takeaways

1. **ResNet's skip connections are the key innovation** that enables training deep networks for complex chart pattern recognition. Without residual connections, networks deeper than 20 layers degrade in performance, limiting their ability to learn hierarchical visual features.

2. **Transfer learning from ImageNet** is highly effective for stock chart classification. Pre-trained ResNet models already understand edges, shapes, and textures that map directly to price levels, candle formations, and trend structures.

3. **Match model depth to task complexity**. Use ResNet-18 for simple trend classification, ResNet-34 for multi-pattern detection, and ResNet-50+ only when the full complexity is justified by the task and training data size.

4. **Global average pooling is better than flattening** for chart images because it makes the model more robust to slight variations in chart positioning and produces a more compact feature representation.

5. **Ensemble with different architectures** for production systems. Combining ResNet with EfficientNet and simpler CNNs produces more robust predictions than any single model, reducing the risk of model-specific biases in trading signals.

6. **Always use Grad-CAM or similar visualization** to validate that the model attends to meaningful chart features rather than artifacts. Models that focus on gridlines, axis labels, or background colors may achieve high training accuracy but will fail on differently styled charts.

7. **Regularly retrain on recent chart data**. Market microstructure and volatility regimes change over time, affecting chart appearance. Models trained on low-volatility period charts may perform poorly during high-volatility regimes.
