"""
CNN Feature Extraction + Sklearn Classifier Implementation
==========================================================

Architecture Overview:
    This module implements a classical computer vision pipeline that separates
    feature extraction from classification. Instead of end-to-end deep learning,
    we extract hand-crafted or learned features from images and feed them into
    traditional machine learning classifiers (SVM, Random Forest, etc.).

    Pipeline:
    1. Image Preprocessing -> 2. Feature Extraction (HOG / Pixel / LBP) -> 3. Classifier (SVM / RF)

Theory & Mathematics:
    Histogram of Oriented Gradients (HOG):
        HOG captures edge and gradient structure by computing gradient magnitudes
        and orientations in local image patches, then binning them into histograms.

        Steps:
        1. Compute image gradients: Gx = I * [-1, 0, 1], Gy = I * [-1, 0, 1]^T
        2. Magnitude: M = sqrt(Gx^2 + Gy^2)
        3. Orientation: theta = arctan(Gy / Gx)
        4. Divide image into cells (e.g., 8x8 pixels)
        5. For each cell, create histogram of gradient orientations (9 bins, 0-180 degrees)
        6. Normalize histograms across blocks of cells (e.g., 2x2 cells per block)
        7. Concatenate all block histograms into a feature vector

    Support Vector Machine (SVM):
        Finds the optimal hyperplane that maximizes the margin between classes.
        For non-linear boundaries, uses kernel trick: K(x, y) = phi(x) . phi(y)
        Common kernels: RBF (Gaussian), polynomial, linear.

    Random Forest:
        Ensemble of decision trees, each trained on a bootstrap sample with
        random feature subsets. Final prediction is majority vote.

Business Use Cases:
    - Document classification and OCR preprocessing
    - Simple object detection in constrained environments
    - Quality control in manufacturing (defect detection)
    - Medical image screening (preliminary classification)
    - Satellite image land-use classification

Advantages:
    - No GPU required; runs efficiently on CPU
    - Interpretable features (HOG visualizations)
    - Works well with small datasets (hundreds of images)
    - Fast training compared to deep learning
    - Robust feature extraction with HOG for shape-based recognition

Disadvantages:
    - Hand-crafted features may miss complex patterns
    - Does not learn hierarchical representations
    - Performance ceiling compared to deep learning on large datasets
    - HOG features are not rotation-invariant by default
    - Feature engineering requires domain knowledge

Key Hyperparameters:
    HOG:
        - pixels_per_cell: Size of cells for gradient histograms (default: (8,8))
        - cells_per_block: Number of cells per normalization block (default: (2,2))
        - orientations: Number of gradient orientation bins (default: 9)
    SVM:
        - C: Regularization parameter (trade-off between margin and misclassification)
        - kernel: 'rbf', 'linear', 'poly'
        - gamma: Kernel coefficient for 'rbf' and 'poly'
    Random Forest:
        - n_estimators: Number of trees
        - max_depth: Maximum depth of each tree
        - min_samples_split: Minimum samples to split a node

References:
    - Dalal, N. and Triggs, B. (2005). "Histograms of Oriented Gradients
      for Human Detection." CVPR.
    - Cortes, C. and Vapnik, V. (1995). "Support-Vector Networks."
      Machine Learning, 20(3), 273-297.
    - Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
"""

# numpy is the fundamental numerical computing library -- we use it for all array/matrix
# operations including image manipulation, gradient computation, and feature vector storage
import numpy as np

# warnings module lets us suppress sklearn convergence warnings that clutter output
# during hyperparameter search when SVM doesn't converge within max_iter
import warnings

# Type hints from typing module make function signatures self-documenting and enable
# static analysis tools (mypy) to catch type errors before runtime
from typing import Dict, Tuple, Any, Optional

# SVC (Support Vector Classifier) implements the SVM algorithm with kernel trick --
# we use it because SVM excels at high-dimensional classification (HOG features are
# typically 1000+ dimensions) and generalizes well with small datasets
from sklearn.svm import SVC

# RandomForestClassifier is an ensemble of decision trees -- we include it as an
# alternative to SVM because it handles non-linear boundaries without kernel selection,
# provides feature importance scores, and is less sensitive to hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier

# StandardScaler normalizes features to zero mean and unit variance -- this is critical
# for SVM because the kernel function (especially RBF) is distance-based, so features
# on different scales would dominate the distance computation unfairly
from sklearn.preprocessing import StandardScaler

# These metrics functions provide comprehensive model evaluation beyond simple accuracy:
# - accuracy_score: overall correct predictions / total predictions
# - classification_report: per-class precision, recall, F1 in a formatted table
# - precision_recall_fscore_support: returns raw arrays for programmatic analysis
# - confusion_matrix: N x N matrix showing where predictions go wrong between classes
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)

# train_test_split handles stratified random splitting -- stratify ensures each split
# has the same class distribution as the full dataset, preventing class imbalance issues
from sklearn.model_selection import train_test_split

# Pipeline chains preprocessing (scaling) and classification into a single estimator --
# this prevents data leakage by ensuring the scaler is fit only on training data and
# applied consistently to validation/test data using the same transform parameters
from sklearn.pipeline import Pipeline

# Optuna is a Bayesian hyperparameter optimization framework -- we prefer it over
# grid search because it uses Tree-structured Parzen Estimators (TPE) to intelligently
# explore the search space, focusing on promising regions rather than exhaustive search
try:
    import optuna  # Attempt import; Optuna is an optional dependency

    OPTUNA_AVAILABLE = True  # Flag to conditionally enable Optuna-based tuning
except ImportError:
    OPTUNA_AVAILABLE = False  # Gracefully degrade if Optuna is not installed

# Ray Tune provides distributed hyperparameter tuning -- we include it as an alternative
# to Optuna for scenarios where you want to parallelize trials across multiple CPUs/GPUs
# or integrate with Ray's ecosystem for production ML pipelines
try:
    import ray  # Ray is the distributed computing framework
    from ray import tune  # tune is Ray's hyperparameter tuning module

    RAY_AVAILABLE = True  # Flag to conditionally enable Ray-based tuning
except ImportError:
    RAY_AVAILABLE = False  # Gracefully degrade if Ray is not installed

# Suppress all warnings globally -- in production you'd want more targeted suppression,
# but for educational demos this prevents sklearn convergence warnings from cluttering
# the output during hyperparameter search where many configs may not converge
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int = 2000,  # Total dataset size -- 2000 is enough for HOG+SVM to learn patterns
    img_size: int = 32,  # 32x32 pixels matches CIFAR-10 resolution, a common benchmark size
    n_classes: int = 10,  # 10 classes provides enough variety to test multi-class performance
    random_state: int = 42,  # Fixed seed ensures reproducible experiments across runs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic image-like data with class-specific visual patterns.

    Each class gets a distinct pattern (horizontal lines, vertical lines,
    diagonal lines, circles, gradients, checkerboards, dots, crosses, corners,
    random noise with bias).

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    img_size : int
        Width and height of each image.
    n_classes : int
        Number of distinct classes (up to 10).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, img_size, img_size)
        Grayscale images with values in [0, 1].
    y : np.ndarray of shape (n_samples,)
        Integer class labels.
    """
    # Create a seeded random number generator for reproducibility -- using RandomState
    # instead of np.random.seed() allows multiple independent generators in parallel code
    rng = np.random.RandomState(random_state)

    # Integer division ensures equal samples per class -- any remainder samples are dropped
    # to maintain perfect class balance, which prevents classifier bias toward majority class
    samples_per_class = n_samples // n_classes

    # Accumulate images and labels in lists (faster than pre-allocating when sizes vary)
    # because list.append() is O(1) amortized while np.concatenate in a loop is O(n)
    X_list, y_list = [], []

    # Outer loop iterates over each class to generate class-specific visual patterns
    # This ensures each class has a distinct, learnable structure in the image data
    for cls in range(n_classes):
        # Inner loop generates individual samples for the current class
        for _ in range(samples_per_class):
            # Initialize with low-amplitude uniform noise (0 to 0.1) as background --
            # this simulates real-world sensor noise and prevents the classifier from
            # overfitting to perfectly clean synthetic patterns
            img = rng.rand(img_size, img_size) * 0.1  # low-level background noise

            if cls == 0:  # Horizontal lines -- tests HOG's ability to detect 0-degree edges
                for row in range(0, img_size, 4):  # Lines every 4 pixels for clear spacing
                    thickness = rng.randint(1, 3)  # Random thickness adds within-class variation
                    # Add bright horizontal stripe; += preserves background noise for realism
                    img[row : row + thickness, :] += 0.8
            elif cls == 1:  # Vertical lines -- tests HOG's ability to detect 90-degree edges
                for col in range(0, img_size, 4):  # Same spacing as horizontal for fair comparison
                    thickness = rng.randint(1, 3)  # Variable thickness prevents overfitting to exact pattern
                    # Add bright vertical stripe spanning full image height
                    img[:, col : col + thickness] += 0.8
            elif cls == 2:  # Diagonal lines (top-left to bottom-right) -- tests 45-degree HOG bins
                for i in range(img_size):  # Iterate row by row to draw diagonal
                    for d in range(-1, 2):  # Draw 3-pixel-wide diagonal for visibility
                        j = i + d  # Column index offset creates thickness around the diagonal
                        if 0 <= j < img_size:  # Bounds check prevents index-out-of-range
                            img[i, j] += 0.8  # Bright pixel along the diagonal
                    # Add a secondary parallel diagonal at random offset for more complex pattern
                    offset = rng.randint(3, 8)  # Random offset creates unique within-class variation
                    j2 = (i + offset) % img_size  # Modulo wraps around to create continuous pattern
                    img[i, j2] += 0.6  # Slightly dimmer secondary diagonal
            elif cls == 3:  # Circles -- tests HOG's response to curved edges across multiple orientations
                # Random center position adds translation invariance challenge
                cx, cy = img_size // 2 + rng.randint(-4, 5), img_size // 2 + rng.randint(-4, 5)
                radius = rng.randint(5, img_size // 3)  # Random radius varies circle size
                # ogrid creates open meshgrids for efficient distance computation without
                # materializing a full 2D coordinate array (memory efficient)
                yy, xx = np.ogrid[:img_size, :img_size]
                # Create ring mask: points where squared distance is close to radius squared
                # The threshold (radius * 3) controls ring thickness
                mask = np.abs((xx - cx) ** 2 + (yy - cy) ** 2 - radius**2) < radius * 3
                img[mask] += 0.8  # Apply bright pixels only where mask is True
            elif cls == 4:  # Gradient left-to-right -- tests response to smooth intensity changes
                # linspace creates smooth 0-to-1 transition; reshape to (1, W) for broadcasting
                gradient = np.linspace(0, 1, img_size).reshape(1, -1)
                # Broadcasting adds gradient to every row simultaneously
                img += gradient * 0.8  # Scale to 0.8 max to leave room for noise
            elif cls == 5:  # Gradient top-to-bottom -- orthogonal to class 4 gradient
                # Reshape to (H, 1) for column-wise broadcasting
                gradient = np.linspace(0, 1, img_size).reshape(-1, 1)
                img += gradient * 0.8  # Same intensity scale as horizontal gradient
            elif cls == 6:  # Checkerboard -- tests response to regular 2D spatial frequency
                block = rng.randint(3, 7)  # Random block size varies spatial frequency
                for r in range(0, img_size, block):
                    for c in range(0, img_size, block):
                        # XOR-like pattern: alternate bright/dark blocks based on position
                        if ((r // block) + (c // block)) % 2 == 0:
                            img[r : r + block, c : c + block] += 0.7
            elif cls == 7:  # Center dot / blob -- tests response to radially symmetric patterns
                cx, cy = img_size // 2, img_size // 2  # Centered blob
                yy, xx = np.ogrid[:img_size, :img_size]  # Open meshgrid for distance calc
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)  # Euclidean distance from center
                sigma = rng.uniform(3, 6)  # Random spread controls blob size
                # Gaussian blob: exp(-d^2 / 2*sigma^2) creates smooth circular gradient
                img += np.exp(-dist**2 / (2 * sigma**2)) * 0.9
            elif cls == 8:  # Cross pattern -- combines horizontal and vertical edges
                mid = img_size // 2  # Center of the cross
                thickness = rng.randint(2, 5)  # Random arm thickness
                # Horizontal arm of the cross
                img[mid - thickness : mid + thickness, :] += 0.7
                # Vertical arm of the cross -- overlapping center gets brighter
                img[:, mid - thickness : mid + thickness] += 0.7
            elif cls == 9:  # Corner patterns -- tests spatially localized feature detection
                quarter = img_size // 4  # Corner region size
                # Top-left corner bright patch
                img[:quarter, :quarter] += 0.8
                # Bottom-right corner bright patch -- diagonal arrangement
                img[-quarter:, -quarter:] += 0.8

            # Add Gaussian noise to simulate real-world image sensor noise and camera artifacts
            # Standard deviation of 0.05 adds subtle variation without overwhelming the pattern
            img += rng.randn(img_size, img_size) * 0.05

            # Clip pixel values to valid [0, 1] range -- additions above may push values > 1
            # and noise may create negative values, both of which are physically meaningless
            img = np.clip(img, 0, 1)

            # Append the generated image and its class label to the accumulation lists
            X_list.append(img)
            y_list.append(cls)

    # Convert lists to contiguous numpy arrays for efficient vectorized operations downstream
    # float32 saves memory (4 bytes vs 8 for float64) while maintaining sufficient precision
    X = np.array(X_list, dtype=np.float32)
    # int64 labels are standard for sklearn classifiers and loss functions
    y = np.array(y_list, dtype=np.int64)

    # Random permutation shuffles data to break the ordered class structure --
    # this is critical because sequential class ordering would cause issues with
    # non-shuffled batch processing and certain train/test split strategies
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def compute_hog_features(
    image: np.ndarray,  # Single grayscale image as 2D array with values in [0,1]
    pixels_per_cell: Tuple[int, int] = (8, 8),  # 8x8 cells are the standard from Dalal & Triggs (2005)
    cells_per_block: Tuple[int, int] = (2, 2),  # 2x2 blocks provide good L2 normalization context
    orientations: int = 9,  # 9 bins spanning 0-180 degrees (unsigned gradients) is the standard
) -> np.ndarray:
    """
    Compute HOG features from scratch (simplified implementation).

    Parameters
    ----------
    image : np.ndarray of shape (H, W)
        Grayscale image with values in [0, 1].
    pixels_per_cell : tuple
        Cell size in pixels.
    cells_per_block : tuple
        Number of cells per normalization block.
    orientations : int
        Number of orientation bins.

    Returns
    -------
    hog_features : np.ndarray
        1-D feature vector.
    """
    # Extract image dimensions for loop bounds and grid calculations
    H, W = image.shape

    # ---- Step 1: Compute gradients using simple [-1, 0, 1] centered difference filters ----
    # These filters approximate the first derivative of the image intensity function.
    # We use centered differences [-1, 0, 1] rather than forward differences [-1, 1] because
    # centered differences provide second-order accuracy (error ~ O(h^2) vs O(h)) and
    # produce smoother gradient estimates that are less sensitive to noise.

    # Initialize gradient arrays to zero -- edges of image will remain zero (implicit padding)
    # because the centered difference filter needs one pixel on each side
    gx = np.zeros_like(image)  # Horizontal gradient (detects vertical edges)
    gy = np.zeros_like(image)  # Vertical gradient (detects horizontal edges)

    # Horizontal gradient: difference between right neighbor and left neighbor pixels
    # Slicing image[:, 2:] - image[:, :-2] computes centered differences for all interior pixels
    # simultaneously via vectorized operations, which is 100x faster than pixel-by-pixel loops
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]

    # Vertical gradient: difference between bottom neighbor and top neighbor pixels
    # Same centered difference approach applied along the vertical axis
    gy[1:-1, :] = image[2:, :] - image[:-2, :]

    # Gradient magnitude combines horizontal and vertical components using L2 norm:
    # M = sqrt(Gx^2 + Gy^2). This gives the edge strength at each pixel regardless
    # of edge direction, which is important because we want to capture edge presence
    # separately from edge orientation
    magnitude = np.sqrt(gx**2 + gy**2)

    # Gradient orientation: arctan2 computes the angle in radians, then convert to degrees.
    # We use unsigned gradients (mod 180) because for object recognition, a dark-to-light
    # edge and a light-to-dark edge at the same angle carry the same structural information.
    # The modulo 180 maps angles from [-180, 180] to [0, 180) range.
    orientation = np.arctan2(gy, gx) * (180.0 / np.pi) % 180  # [0, 180)

    # ---- Step 2: Compute cell histograms ----
    # Divide the image into a grid of cells, and for each cell, compute a histogram
    # of gradient orientations weighted by gradient magnitudes. This captures the
    # local edge structure while providing some spatial tolerance.

    # Unpack cell dimensions from the tuple for cleaner code
    cell_h, cell_w = pixels_per_cell

    # Calculate the number of complete cells that fit in each dimension
    # Integer division drops any partial cells at the image borders
    n_cells_y = H // cell_h  # Number of cell rows
    n_cells_x = W // cell_w  # Number of cell columns

    # Pre-allocate the 3D histogram array: (cell_rows, cell_cols, orientation_bins)
    # Each cell gets one histogram with 'orientations' bins
    cell_hists = np.zeros((n_cells_y, n_cells_x, orientations))

    # Width of each orientation bin in degrees -- for 9 bins, each bin covers 20 degrees
    bin_width = 180.0 / orientations

    # Iterate over every cell in the grid to compute its orientation histogram
    for cy in range(n_cells_y):  # Cell row index
        for cx in range(n_cells_x):  # Cell column index
            # Calculate pixel coordinates of the top-left corner of this cell
            y_start = cy * cell_h
            x_start = cx * cell_w

            # Extract magnitude and orientation patches for this cell using slicing
            # This avoids redundant index calculations inside the inner pixel loop
            cell_mag = magnitude[y_start : y_start + cell_h, x_start : x_start + cell_w]
            cell_ori = orientation[y_start : y_start + cell_h, x_start : x_start + cell_w]

            # Iterate over every pixel in this cell to accumulate its contribution
            # to the orientation histogram. Each pixel votes for its orientation bin
            # with a weight equal to its gradient magnitude (stronger edges = more votes)
            for i in range(cell_h):  # Pixel row within cell
                for j in range(cell_w):  # Pixel column within cell
                    angle = cell_ori[i, j]  # Gradient angle at this pixel in degrees
                    mag = cell_mag[i, j]  # Gradient strength at this pixel

                    # Determine which orientation bin this angle falls into
                    # For 9 bins: bin 0 = [0,20), bin 1 = [20,40), ..., bin 8 = [160,180)
                    bin_idx = int(angle / bin_width)

                    # Clamp bin index to valid range -- handles edge case where angle = 180.0
                    # exactly, which would compute bin_idx = 9 (out of bounds for 9 bins)
                    if bin_idx >= orientations:
                        bin_idx = orientations - 1

                    # Accumulate magnitude-weighted vote into the histogram bin
                    # Magnitude weighting ensures strong edges contribute more than weak ones
                    cell_hists[cy, cx, bin_idx] += mag

    # ---- Step 3: Block normalization ----
    # Normalize histograms over overlapping blocks of cells. This provides invariance
    # to local illumination changes because normalization rescales the histogram values
    # relative to neighboring cells. L2-norm is used following Dalal & Triggs (2005).

    # Unpack block dimensions from the tuple
    block_h, block_w = cells_per_block

    # Calculate how many blocks fit with overlap -- each block overlaps by (block_size - 1) cells
    # This overlap means each cell contributes to multiple blocks, improving robustness
    n_blocks_y = n_cells_y - block_h + 1  # Number of block positions vertically
    n_blocks_x = n_cells_x - block_w + 1  # Number of block positions horizontally

    # Accumulate normalized block features into a flat list for concatenation
    hog_features = []

    # Small epsilon to prevent division by zero in normalization
    eps = 1e-5

    # Slide the block window over the cell grid with single-cell stride
    for by in range(max(n_blocks_y, 1)):  # max(., 1) handles degenerate case of tiny images
        for bx in range(max(n_blocks_x, 1)):
            # Extract all cell histograms within this block and flatten to 1D
            # For a 2x2 block with 9 orientations, this gives a 36-element vector
            block = cell_hists[by : by + block_h, bx : bx + block_w].ravel()

            # L2 normalization: divide by the L2 norm of the block vector
            # This makes the descriptor robust to linear illumination changes because
            # if all magnitudes scale by factor k, the normalized vector stays the same
            norm = np.sqrt(np.sum(block**2) + eps)
            hog_features.append(block / norm)

    # Concatenate all block feature vectors into a single 1D descriptor for this image
    # If no blocks were computed (very small image), return a zero vector as fallback
    return np.concatenate(hog_features) if hog_features else np.zeros(orientations)


def extract_pixel_features(image: np.ndarray) -> np.ndarray:
    """
    Flatten image pixels as features -- the simplest possible feature extraction.
    Each pixel intensity becomes one feature dimension. For a 32x32 image, this
    produces a 1024-dimensional feature vector. This baseline method preserves all
    spatial information but lacks the translation/scale invariance that HOG provides.
    """
    # ravel() returns a contiguous flattened array (row-major order by default)
    # This is preferred over flatten() because ravel() returns a view when possible,
    # avoiding unnecessary memory copies for read-only operations
    return image.ravel()


def extract_features(
    images: np.ndarray,  # Batch of grayscale images with shape (N, H, W)
    method: str = "hog",  # Feature extraction strategy: 'hog' or 'pixel'
    pixels_per_cell: Tuple[int, int] = (8, 8),  # HOG cell size parameter
    cells_per_block: Tuple[int, int] = (2, 2),  # HOG block size parameter
    orientations: int = 9,  # HOG orientation bins parameter
) -> np.ndarray:
    """
    Extract features from a batch of images.

    Parameters
    ----------
    images : np.ndarray of shape (N, H, W)
        Batch of grayscale images.
    method : str
        'hog' for HOG features, 'pixel' for raw pixel features.
    pixels_per_cell : tuple
        HOG cell size.
    cells_per_block : tuple
        HOG block size.
    orientations : int
        Number of HOG orientation bins.

    Returns
    -------
    features : np.ndarray of shape (N, D)
        Feature matrix.
    """
    # Accumulate feature vectors in a list -- we don't pre-allocate because
    # the feature dimension D depends on HOG parameters and image size,
    # which are not known until the first feature vector is computed
    feature_list = []

    # Process each image independently -- HOG features are per-image descriptors
    # In production, you might vectorize this with batch processing, but the per-pixel
    # loop in compute_hog_features makes true vectorization complex
    for img in images:
        if method == "hog":
            # Compute HOG descriptor: captures edge structure and gradient patterns
            # HOG is preferred for shape-based recognition tasks because it provides
            # some invariance to small translations and illumination changes
            feat = compute_hog_features(img, pixels_per_cell, cells_per_block, orientations)
        else:
            # Raw pixel features: simplest baseline, preserves all information but
            # is sensitive to translation, rotation, and illumination changes
            feat = extract_pixel_features(img)
        feature_list.append(feat)

    # Stack feature vectors into a 2D array (N samples x D features) for sklearn
    # float32 is sufficient precision and halves memory vs float64
    return np.array(feature_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training, Validation, Testing
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,  # Training images with shape (N, H, W)
    y_train: np.ndarray,  # Training labels with shape (N,) containing integer class indices
    classifier_type: str = "svm",  # Which classifier to use: 'svm' or 'rf' (Random Forest)
    feature_method: str = "hog",  # Feature extraction method: 'hog' or 'pixel'
    C: float = 1.0,  # SVM regularization -- higher C = less regularization = tighter fit
    kernel: str = "rbf",  # SVM kernel function: 'rbf' maps to infinite-dimensional space
    gamma: str = "scale",  # RBF kernel width: 'scale' = 1/(n_features * var) is a good default
    n_estimators: int = 100,  # Number of trees in Random Forest ensemble
    max_depth: Optional[int] = None,  # Max tree depth -- None means grow until pure leaves
    pixels_per_cell: Tuple[int, int] = (8, 8),  # HOG cell size
    cells_per_block: Tuple[int, int] = (2, 2),  # HOG block size for normalization
    orientations: int = 9,  # Number of HOG orientation bins
) -> Dict[str, Any]:
    """
    Train a feature extraction + classifier pipeline.

    Parameters
    ----------
    X_train : np.ndarray of shape (N, H, W)
        Training images.
    y_train : np.ndarray of shape (N,)
        Training labels.
    classifier_type : str
        'svm' or 'rf' (Random Forest).
    feature_method : str
        'hog' or 'pixel'.
    C : float
        SVM regularization.
    kernel : str
        SVM kernel type.
    gamma : str or float
        SVM kernel coefficient.
    n_estimators : int
        Number of trees for Random Forest.
    max_depth : int or None
        Maximum tree depth for Random Forest.
    pixels_per_cell : tuple
        HOG parameter.
    cells_per_block : tuple
        HOG parameter.
    orientations : int
        HOG parameter.

    Returns
    -------
    result : dict
        'pipeline': trained sklearn Pipeline,
        'feature_method': str,
        'hog_params': dict of HOG parameters.
    """
    # Log the feature extraction phase -- this can be slow for large datasets
    # because HOG computation involves gradient calculation and histogram binning per image
    print(f"Extracting {feature_method.upper()} features from {len(X_train)} training images...")

    # Transform raw images into feature vectors using the specified method
    # This converts (N, 32, 32) images into (N, D) feature matrix where D depends
    # on the method: HOG typically produces ~324 features for 32x32 images with default params,
    # while pixel features produce 1024 (= 32 * 32) features
    X_feat = extract_features(
        X_train,
        method=feature_method,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        orientations=orientations,
    )

    # Select and configure the classifier based on the specified type
    if classifier_type == "svm":
        # SVM with RBF kernel -- the RBF kernel K(x,y) = exp(-gamma * ||x-y||^2) maps
        # data to an infinite-dimensional space where linear separation may be possible.
        # max_iter=5000 caps training time -- SVM optimization (SMO algorithm) can be slow
        # on large datasets, and convergence beyond 5000 iterations rarely improves accuracy
        clf = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42, max_iter=5000)
        print(f"Training SVM (C={C}, kernel={kernel}, gamma={gamma})...")
    else:
        # Random Forest -- ensemble of n_estimators decision trees, each trained on a
        # bootstrap sample of the data with sqrt(n_features) random features at each split.
        # n_jobs=-1 uses all available CPU cores for parallel tree construction,
        # which provides near-linear speedup for forest training
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,  # Fixed seed for reproducible tree construction
            n_jobs=-1,  # Parallelize across all CPU cores
        )
        print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")

    # Create a Pipeline that chains StandardScaler -> Classifier
    # Pipeline ensures that scaling parameters (mean, std) are learned only from training data
    # and applied consistently during prediction, preventing data leakage from test/val sets
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    # Fit the pipeline: first StandardScaler computes mean and std of training features,
    # then transforms X_feat to zero mean and unit variance, then the classifier is trained
    # on the scaled features. This is equivalent to scaler.fit_transform(X_feat) followed
    # by clf.fit(scaled_X_feat) but bundled for safety and convenience
    pipeline.fit(X_feat, y_train)

    # Compute training accuracy as a sanity check -- very low training accuracy indicates
    # the model cannot even fit the training data (underfitting), while near-perfect training
    # accuracy combined with poor validation accuracy indicates overfitting
    train_acc = pipeline.score(X_feat, y_train)
    print(f"Training accuracy: {train_acc:.4f}")

    # Return the trained pipeline and all parameters needed to reproduce feature extraction
    # on new data -- the caller needs hog_params to extract features consistently
    return {
        "pipeline": pipeline,  # Trained scaler + classifier chain
        "feature_method": feature_method,  # Which feature extraction was used
        "hog_params": {  # HOG configuration for consistent feature extraction at inference
            "pixels_per_cell": pixels_per_cell,
            "cells_per_block": cells_per_block,
            "orientations": orientations,
        },
    }


def validate(
    model_dict: Dict[str, Any],  # Output from train() containing pipeline and feature params
    X_val: np.ndarray,  # Validation images with shape (N, H, W)
    y_val: np.ndarray,  # Validation labels with shape (N,)
) -> Dict[str, Any]:
    """
    Validate the trained pipeline on a held-out validation set.

    Parameters
    ----------
    model_dict : dict
        Output from train().
    X_val : np.ndarray of shape (N, H, W)
        Validation images.
    y_val : np.ndarray of shape (N,)
        Validation labels.

    Returns
    -------
    metrics : dict
        'accuracy': float, 'precision': array, 'recall': array,
        'f1': array, 'confusion_matrix': array, 'report': str.
    """
    # Unpack the trained pipeline and feature extraction parameters from the model dict
    pipeline = model_dict["pipeline"]  # Contains fitted scaler and trained classifier
    hog_params = model_dict["hog_params"]  # HOG configuration for consistent feature extraction
    feature_method = model_dict["feature_method"]  # 'hog' or 'pixel'

    # Extract features from validation images using the SAME method and parameters as training
    # This consistency is critical -- using different HOG params would produce incompatible features
    X_feat = extract_features(X_val, method=feature_method, **hog_params)

    # Generate predictions -- the pipeline automatically applies the same scaling transform
    # (using mean/std learned during training) before passing features to the classifier
    y_pred = pipeline.predict(X_feat)

    # Compute overall accuracy: fraction of correct predictions across all classes
    acc = accuracy_score(y_val, y_pred)

    # Compute per-class precision, recall, and F1 scores (average=None returns arrays)
    # Precision = TP / (TP + FP): how many predicted positives are actually positive
    # Recall = TP / (TP + FN): how many actual positives are correctly identified
    # F1 = 2 * P * R / (P + R): harmonic mean balancing precision and recall
    # zero_division=0 handles classes with no predictions gracefully (returns 0 instead of warning)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None, zero_division=0)

    # Generate formatted classification report string showing per-class and aggregate metrics
    report = classification_report(y_val, y_pred, zero_division=0)

    # Compute confusion matrix: cm[i][j] = number of samples with true label i predicted as label j
    # Diagonal elements are correct predictions; off-diagonal shows specific misclassification patterns
    cm = confusion_matrix(y_val, y_pred)

    # Print results for immediate feedback during training runs
    print(f"Validation Accuracy: {acc:.4f}")
    print(report)

    # Return all metrics as a dictionary for programmatic analysis and comparison
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }


def test(
    model_dict: Dict[str, Any],  # Output from train() containing pipeline and feature params
    X_test: np.ndarray,  # Test images -- only used ONCE for final evaluation
    y_test: np.ndarray,  # Test labels -- never used for model selection or tuning
) -> Dict[str, Any]:
    """
    Final evaluation on the test set.

    Parameters
    ----------
    model_dict : dict
        Output from train().
    X_test : np.ndarray of shape (N, H, W)
        Test images.
    y_test : np.ndarray of shape (N,)
        Test labels.

    Returns
    -------
    metrics : dict
        Same structure as validate().
    """
    # Print header to visually separate test results from validation output
    print("\n--- Test Set Evaluation ---")
    # Reuse validate() function -- same evaluation logic applies, the only difference
    # is semantic: test set should be used exactly once for final unbiased evaluation
    return validate(model_dict, X_test, y_test)


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: "optuna.Trial",  # Optuna trial object that manages hyperparameter suggestions
    X_train: np.ndarray,  # Training images passed through closure
    y_train: np.ndarray,  # Training labels passed through closure
    X_val: np.ndarray,  # Validation images for evaluating each trial's configuration
    y_val: np.ndarray,  # Validation labels for computing the objective (accuracy)
) -> float:
    """
    Optuna objective function for hyperparameter tuning.

    Searches over:
        - classifier_type: SVM or RF
        - feature_method: HOG or pixel
        - SVM: C, kernel, gamma
        - RF: n_estimators, max_depth
        - HOG: orientations, pixels_per_cell

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    X_train, y_train : np.ndarray
        Training data.
    X_val, y_val : np.ndarray
        Validation data.

    Returns
    -------
    accuracy : float
        Validation accuracy to maximize.
    """
    # Suggest categorical hyperparameters -- Optuna's TPE sampler learns which
    # categories perform best and focuses sampling on promising configurations
    classifier_type = trial.suggest_categorical("classifier_type", ["svm", "rf"])
    feature_method = trial.suggest_categorical("feature_method", ["hog", "pixel"])

    # HOG-specific hyperparameters -- these affect feature dimensionality and quality
    # More orientations captures finer angular distinctions but increases feature dimension
    hog_orientations = trial.suggest_int("hog_orientations", 6, 12)
    # Smaller cells capture finer spatial detail but produce higher-dimensional features
    ppc = trial.suggest_categorical("pixels_per_cell", [4, 8])

    # Build keyword arguments dict for the train() function
    kwargs = {
        "feature_method": feature_method,
        "pixels_per_cell": (ppc, ppc),  # Convert scalar to tuple for both dimensions
        "orientations": hog_orientations,
    }

    # Add classifier-specific hyperparameters based on which classifier was selected
    if classifier_type == "svm":
        kwargs["classifier_type"] = "svm"
        # C controls regularization strength: log-uniform search spans multiple orders
        # of magnitude because C's effect is multiplicative (C=0.1 vs C=100 are both common)
        kwargs["C"] = trial.suggest_float("svm_C", 0.01, 100.0, log=True)
        # Kernel choice determines the feature space transformation
        kwargs["kernel"] = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        if kwargs["kernel"] == "rbf":
            # Gamma controls RBF kernel width -- 'scale' and 'auto' are data-adaptive defaults
            kwargs["gamma"] = trial.suggest_categorical("svm_gamma", ["scale", "auto"])
    else:
        kwargs["classifier_type"] = "rf"
        # More trees generally improve performance but with diminishing returns and higher cost
        kwargs["n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        # Deeper trees can capture more complex patterns but risk overfitting
        kwargs["max_depth"] = trial.suggest_int("rf_max_depth", 5, 50)

    # Train the model with this trial's hyperparameter configuration
    model_dict = train(X_train, y_train, **kwargs)

    # Evaluate on validation set -- this accuracy becomes the objective to maximize
    metrics = validate(model_dict, X_val, y_val)

    # Return validation accuracy -- Optuna uses this to guide the TPE sampler toward
    # better configurations in subsequent trials
    return metrics["accuracy"]


def ray_tune_search(
    X_train: np.ndarray,  # Training images
    y_train: np.ndarray,  # Training labels
    X_val: np.ndarray,  # Validation images
    y_val: np.ndarray,  # Validation labels
    num_samples: int = 8,  # Number of random configurations to evaluate
) -> Dict[str, Any]:
    """
    Ray Tune hyperparameter search.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    X_val, y_val : np.ndarray
        Validation data.
    num_samples : int
        Number of random configurations to try.

    Returns
    -------
    best_config : dict
        Best hyperparameter configuration found.
    """
    # Guard clause: skip if Ray is not installed
    if not RAY_AVAILABLE:
        print("Ray is not installed. Skipping Ray Tune search.")
        return {}

    # Define the trainable function that Ray Tune will call for each configuration
    # This function receives a config dict and reports metrics back to Ray
    def trainable(config):
        # Build keyword arguments from the Ray Tune config dict
        kwargs = {
            "classifier_type": config["classifier_type"],
            "feature_method": config["feature_method"],
            "C": config.get("C", 1.0),  # Default C=1.0 if not in config
            "kernel": config.get("kernel", "rbf"),  # Default kernel if not specified
            "n_estimators": config.get("n_estimators", 100),  # Default tree count
            "max_depth": config.get("max_depth", None),  # Default unlimited depth
            "pixels_per_cell": (config["ppc"], config["ppc"]),  # Convert to tuple
            "orientations": config["orientations"],
        }
        # Train model with this configuration
        model_dict = train(X_train, y_train, **kwargs)
        # Evaluate and report accuracy to Ray Tune's scheduler
        metrics = validate(model_dict, X_val, y_val)
        tune.report({"accuracy": metrics["accuracy"]})

    # Define the hyperparameter search space using Ray Tune's sampling distributions
    search_space = {
        "classifier_type": tune.choice(["svm", "rf"]),  # Random choice between classifiers
        "feature_method": tune.choice(["hog", "pixel"]),  # Random choice of features
        "C": tune.loguniform(0.01, 100.0),  # Log-uniform for SVM regularization
        "kernel": tune.choice(["rbf", "linear"]),  # SVM kernel options
        "n_estimators": tune.choice([50, 100, 200]),  # Discrete tree count options
        "max_depth": tune.choice([10, 20, 30, None]),  # Tree depth options including unlimited
        "ppc": tune.choice([4, 8]),  # HOG pixels per cell
        "orientations": tune.choice([6, 9, 12]),  # HOG orientation bins
    }

    # Initialize Ray runtime if not already running -- num_cpus=2 limits resource usage
    # ignore_reinit_error=True prevents crashes if Ray is already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    # Create and run the Tuner -- Ray will execute trainable() num_samples times
    # with different random configurations drawn from search_space
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,  # Total number of configurations to try
            metric="accuracy",  # Metric name to optimize
            mode="max",  # Maximize accuracy (not minimize)
        ),
    )

    # Run all trials and collect results
    results = tuner.fit()

    # Extract the best-performing configuration
    best = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best config: {best.config}")
    print(f"Best accuracy: {best.metrics['accuracy']:.4f}")

    # Shut down Ray to free resources -- important in notebook/script environments
    # where Ray workers would otherwise persist in the background
    ray.shutdown()
    return best.config


# ---------------------------------------------------------------------------
# Parameter Comparison
# ---------------------------------------------------------------------------


def compare_parameter_sets(
    n_samples: int = 1500,  # Moderate dataset size for meaningful comparison without excessive runtime
    img_size: int = 32,  # Standard image size matching generate_data default
) -> Dict[str, Any]:
    """
    Compare different HOG and classifier parameter configurations to show their
    impact on classification accuracy. This function demonstrates how:

    1. Number of HOG orientation bins affects feature discriminativeness
    2. HOG cell size (pixels_per_cell) trades off spatial detail vs feature dimension
    3. SVM kernel choice and regularization affect decision boundary complexity
    4. Random Forest depth and ensemble size control model capacity

    These comparisons help practitioners understand which hyperparameters matter
    most and what ranges to search during tuning.

    Parameters
    ----------
    n_samples : int
        Number of synthetic images to generate for the comparison.
    img_size : int
        Size of each synthetic image (img_size x img_size pixels).

    Returns
    -------
    results : dict
        Dictionary mapping configuration names to their accuracy scores.
    """
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: HOG + Classifier Configurations")
    print("=" * 70)

    # Generate a shared dataset for fair comparison -- all configurations
    # are evaluated on the exact same train/val split so differences in
    # accuracy are attributable only to the parameter changes
    X, y = generate_data(n_samples=n_samples, img_size=img_size, n_classes=10, random_state=42)

    # Stratified split ensures each fold has the same class distribution
    # 70% train, 15% val, 15% test maintains enough data for reliable metrics
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Define parameter configurations to compare -- each entry specifies
    # a complete configuration that will be trained and evaluated independently
    configs = {
        # --- HOG orientation bins comparison ---
        # Fewer bins (6) merge similar angles together, losing angular precision
        # but producing more compact features. Standard (9) balances detail and dimensionality.
        # More bins (12) can capture finer angular distinctions but increases feature dimension.
        "HOG_6orientations_SVM": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 10.0, "kernel": "rbf",
            "orientations": 6, "pixels_per_cell": (8, 8),
        },
        "HOG_9orientations_SVM": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 10.0, "kernel": "rbf",
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        "HOG_12orientations_SVM": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 10.0, "kernel": "rbf",
            "orientations": 12, "pixels_per_cell": (8, 8),
        },
        # --- Pixels per cell comparison ---
        # Smaller cells (4x4) capture finer spatial detail but produce higher-dimensional
        # features (more cells per image = more histograms). This is analogous to using
        # a smaller receptive field in CNNs: more local detail but less context.
        "HOG_cell4x4_SVM": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 10.0, "kernel": "rbf",
            "orientations": 9, "pixels_per_cell": (4, 4),
        },
        "HOG_cell8x8_SVM": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 10.0, "kernel": "rbf",
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        # --- SVM kernel comparison ---
        # Linear kernel: decision boundary is a hyperplane in feature space.
        # Works well when HOG features are already discriminative enough that
        # classes are approximately linearly separable in the HOG feature space.
        "HOG_SVM_linear": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 10.0, "kernel": "linear",
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        # RBF kernel: maps to infinite-dimensional space via Gaussian similarity.
        # Better for capturing non-linear class boundaries but slower and may overfit.
        "HOG_SVM_rbf": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 10.0, "kernel": "rbf",
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        # --- SVM regularization (C) comparison ---
        # Low C (0.1): strong regularization, wider margin, tolerates more misclassification.
        # High C (100): weak regularization, narrow margin, tries harder to classify every point.
        "HOG_SVM_C=0.1": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 0.1, "kernel": "rbf",
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        "HOG_SVM_C=100": {
            "classifier_type": "svm", "feature_method": "hog",
            "C": 100.0, "kernel": "rbf",
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        # --- Random Forest comparison ---
        # Fewer trees (50): faster but higher variance in predictions.
        # More trees (200): more stable but diminishing returns beyond ~100-200.
        "HOG_RF_50trees": {
            "classifier_type": "rf", "feature_method": "hog",
            "n_estimators": 50, "max_depth": 20,
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        "HOG_RF_200trees": {
            "classifier_type": "rf", "feature_method": "hog",
            "n_estimators": 200, "max_depth": 20,
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
        # --- Feature method comparison ---
        # Raw pixel features as baseline: no feature engineering, just flattened intensities.
        # This shows how much value HOG feature extraction adds over naive pixel features.
        "Pixel_SVM": {
            "classifier_type": "svm", "feature_method": "pixel",
            "C": 10.0, "kernel": "rbf",
            "orientations": 9, "pixels_per_cell": (8, 8),
        },
    }

    # Store results for comparison table
    results = {}

    # Train and evaluate each configuration
    for name, params in configs.items():
        print(f"\n--- Configuration: {name} ---")
        # Train the model with this configuration's parameters
        model_dict = train(X_train, y_train, **params)
        # Evaluate on validation set to get accuracy
        metrics = validate(model_dict, X_val, y_val)
        # Store accuracy for the summary table
        results[name] = metrics["accuracy"]

    # Print summary comparison table sorted by accuracy (best first)
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Val Accuracy':>12}")
    print("-" * 50)

    # Sort configurations by accuracy in descending order for easy comparison
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<35} {acc:>12.4f}")

    print("-" * 50)

    # Print interpretation guidance for practitioners
    print("\nKey Takeaways:")
    print("  - HOG features typically outperform raw pixels due to edge/gradient invariance")
    print("  - 9 orientation bins is usually optimal (diminishing returns beyond)")
    print("  - Smaller cells (4x4) capture more detail but increase dimensionality")
    print("  - RBF kernel generally outperforms linear for complex visual patterns")
    print("  - SVM C value: moderate regularization (C=1-10) often works best")

    return results


# ---------------------------------------------------------------------------
# Real-World Demo: Medical X-Ray Classification
# ---------------------------------------------------------------------------


def real_world_demo() -> Dict[str, Any]:
    """
    Demonstrate the HOG + SVM/RF pipeline on a realistic medical imaging scenario:
    classifying chest X-ray images as Normal vs Pneumonia.

    In real clinical practice, this type of system would serve as a preliminary
    screening tool to flag potentially abnormal X-rays for radiologist review,
    reducing workload by filtering out clearly normal cases.

    The synthetic data mimics key visual characteristics of chest X-rays:
    - Normal: relatively uniform lung fields with subtle rib patterns
    - Pneumonia: localized opacities (white patches) representing fluid/inflammation

    Returns
    -------
    results : dict
        Dictionary containing model performance metrics and predictions.
    """
    print("\n" + "=" * 70)
    print("REAL-WORLD DEMO: Medical X-Ray Classification (Normal vs Pneumonia)")
    print("=" * 70)

    # ---- Generate synthetic chest X-ray data ----
    # In a real application, you would load actual DICOM/PNG X-ray images from
    # datasets like CheXpert, MIMIC-CXR, or the NIH Chest X-ray dataset
    rng = np.random.RandomState(42)  # Fixed seed for reproducible demo results
    n_samples = 800  # Moderate dataset simulating a small clinical study
    img_size = 48  # Larger than default 32 to capture more spatial detail in X-ray patterns

    # Define clinically meaningful class labels
    class_names = ["Normal", "Pneumonia"]
    n_classes = len(class_names)
    samples_per_class = n_samples // n_classes

    X_list, y_list = [], []  # Accumulate images and labels

    for cls_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Base X-ray appearance: mid-gray background representing soft tissue
            # Real chest X-rays have intensity around 0.4-0.6 in lung fields
            img = np.ones((img_size, img_size)) * 0.4 + rng.rand(img_size, img_size) * 0.1

            # Add rib-like horizontal patterns -- both normal and pneumonia X-rays
            # show rib shadows as subtle horizontal bands across the lung fields
            for row in range(4, img_size - 4, 6):
                rib_intensity = rng.uniform(0.05, 0.12)  # Subtle intensity variation
                rib_thickness = rng.randint(1, 3)  # Variable rib shadow width
                img[row : row + rib_thickness, 4:-4] += rib_intensity

            # Add lung field boundaries -- darker regions representing air-filled lungs
            # Lungs appear darker on X-ray because air absorbs fewer X-rays
            lung_left = img_size // 4  # Left lung boundary
            lung_right = 3 * img_size // 4  # Right lung boundary
            lung_top = img_size // 6  # Top of lung fields
            lung_bottom = 5 * img_size // 6  # Bottom of lung fields

            # Darken the lung field regions to simulate air-filled lung tissue
            img[lung_top:lung_bottom, lung_left-2:lung_left+img_size//4] -= 0.08
            img[lung_top:lung_bottom, lung_right-img_size//4:lung_right+2] -= 0.08

            if cls_idx == 0:
                # NORMAL X-ray: clear lung fields with only subtle anatomical structures
                # Normal lungs show uniform darkness in lung fields with visible vascular
                # markings (thin branching patterns radiating from the hilum)

                # Add subtle vascular markings radiating from center -- these represent
                # blood vessels that are visible in normal, well-aerated lungs
                center_x, center_y = img_size // 2, img_size // 2
                for _ in range(8):
                    # Random vascular branch direction
                    angle = rng.uniform(0, 2 * np.pi)
                    length = rng.randint(8, 18)
                    for t in range(length):
                        vx = int(center_x + t * np.cos(angle))
                        vy = int(center_y + t * np.sin(angle))
                        if 0 <= vx < img_size and 0 <= vy < img_size:
                            img[vy, vx] += 0.03  # Very subtle vascular shadow

            elif cls_idx == 1:
                # PNEUMONIA X-ray: localized opacities (white patches) in one or both
                # lung fields representing fluid accumulation, inflammation, or infection.
                # These opacities are the key diagnostic feature radiologists look for.

                # Generate 1-3 opacity patches (pneumonia can be focal or multifocal)
                n_opacities = rng.randint(1, 4)
                for _ in range(n_opacities):
                    # Random position within lung field boundaries
                    ox = rng.randint(lung_left, lung_right)
                    oy = rng.randint(lung_top, lung_bottom)
                    # Variable size simulates different stages of pneumonia
                    opacity_radius = rng.randint(4, 10)

                    # Create Gaussian-shaped opacity (soft edges like real infiltrates)
                    yy, xx = np.ogrid[:img_size, :img_size]
                    dist = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2)
                    opacity_mask = np.exp(-dist**2 / (2 * opacity_radius**2))
                    # Increase brightness in opacity region (pneumonia appears white on X-ray)
                    img += opacity_mask * rng.uniform(0.15, 0.3)

                # Add air bronchograms -- a classic sign of pneumonia where air-filled
                # bronchi become visible against the opacified (fluid-filled) lung
                if rng.random() > 0.4:  # Present in ~60% of pneumonia cases
                    bx = rng.randint(lung_left + 3, lung_right - 3)
                    by = rng.randint(lung_top + 3, lung_bottom - 3)
                    for t in range(6):
                        if 0 <= by + t < img_size and 0 <= bx < img_size:
                            img[by + t, bx] -= 0.05  # Dark air-filled bronchus

            # Add realistic noise (quantum noise in X-ray imaging follows Poisson distribution,
            # but Gaussian is a reasonable approximation for educational purposes)
            img += rng.randn(img_size, img_size) * 0.03
            img = np.clip(img, 0, 1)  # Ensure valid pixel range

            X_list.append(img)
            y_list.append(cls_idx)

    # Convert to numpy arrays for sklearn compatibility
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle to prevent ordering bias during training
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Generated {len(X)} synthetic chest X-ray images ({img_size}x{img_size})")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

    # ---- Split data ----
    # 70/15/15 split is standard for small medical datasets where you want
    # enough validation data for reliable model selection
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- Train SVM with HOG features ----
    # HOG is well-suited for X-ray classification because pneumonia opacities
    # alter the local gradient structure (edges become blurred in affected regions),
    # which HOG captures through its gradient orientation histograms
    print("\n--- Training SVM classifier on HOG features ---")
    svm_model = train(
        X_train, y_train,
        classifier_type="svm",
        feature_method="hog",
        C=10.0,  # Moderate regularization for medical data
        kernel="rbf",  # RBF captures non-linear opacity patterns
        pixels_per_cell=(8, 8),  # Standard cell size
        orientations=9,  # Standard orientation bins
    )

    # ---- Train Random Forest for comparison ----
    # RF provides an alternative that can give feature importance scores,
    # which is valuable for interpretability in medical applications
    print("\n--- Training Random Forest classifier on HOG features ---")
    rf_model = train(
        X_train, y_train,
        classifier_type="rf",
        feature_method="hog",
        n_estimators=150,  # More trees for stable predictions in medical context
        max_depth=25,  # Moderate depth to prevent overfitting
        pixels_per_cell=(8, 8),
        orientations=9,
    )

    # ---- Evaluate both models ----
    print("\n--- SVM Validation Results ---")
    svm_metrics = validate(svm_model, X_val, y_val)

    print("\n--- Random Forest Validation Results ---")
    rf_metrics = validate(rf_model, X_val, y_val)

    # ---- Select best model and run final test ----
    # In medical imaging, we might prefer the model with higher sensitivity (recall)
    # for the pneumonia class to minimize missed diagnoses
    best_model_name = "SVM" if svm_metrics["accuracy"] >= rf_metrics["accuracy"] else "Random Forest"
    best_model = svm_model if svm_metrics["accuracy"] >= rf_metrics["accuracy"] else rf_model

    print(f"\n--- Final Test with best model ({best_model_name}) ---")
    test_metrics = test(best_model, X_test, y_test)

    # ---- Clinical interpretation ----
    print("\n--- Clinical Interpretation ---")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    # In medical screening, sensitivity (recall for pneumonia class) is critical
    # because missing a pneumonia case (false negative) is more dangerous than
    # a false alarm (false positive) that would be caught by radiologist review
    if len(test_metrics["recall"]) > 1:
        pneumonia_recall = test_metrics["recall"][1]  # Class 1 = Pneumonia
        pneumonia_precision = test_metrics["precision"][1]
        print(f"Pneumonia Sensitivity (Recall): {pneumonia_recall:.4f}")
        print(f"Pneumonia Specificity (Precision): {pneumonia_precision:.4f}")

        if pneumonia_recall >= 0.9:
            print("  -> High sensitivity: suitable for preliminary screening")
        elif pneumonia_recall >= 0.7:
            print("  -> Moderate sensitivity: may miss some cases, use with caution")
        else:
            print("  -> Low sensitivity: NOT suitable for clinical screening without improvement")

    print("\nNote: This is a synthetic demo. Real medical AI requires:")
    print("  - Validated on diverse patient populations")
    print("  - Regulatory approval (FDA 510(k) or equivalent)")
    print("  - Integration with clinical workflow and radiologist oversight")
    print("  - Continuous monitoring for distribution shift and performance degradation")

    return {
        "svm_accuracy": svm_metrics["accuracy"],
        "rf_accuracy": rf_metrics["accuracy"],
        "best_model": best_model_name,
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the full CNN feature extraction + sklearn classifier pipeline."""
    print("=" * 70)
    print("CNN Feature Extraction + Sklearn Classifier Pipeline")
    print("=" * 70)

    # ---- Step 1: Generate synthetic data ----
    # We create structured synthetic images because real image datasets are large
    # and require separate downloads. The synthetic patterns test the pipeline's
    # ability to distinguish different spatial structures (lines, circles, gradients)
    print("\n[1/8] Generating synthetic image data...")
    X, y = generate_data(n_samples=2000, img_size=32, n_classes=10)
    print(f"    Dataset shape: {X.shape}, Labels: {np.unique(y)}")

    # ---- Step 2: Split data into train/validation/test ----
    # Three-way split is essential: train for fitting, val for model selection/tuning,
    # test for final unbiased evaluation. Using val for tuning prevents overfitting
    # to the test set through repeated evaluation (a common mistake in ML practice)
    print("\n[2/8] Splitting data into train/val/test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- Step 3: Train with default parameters (SVM + HOG) ----
    # Start with a known-good configuration: SVM with RBF kernel on HOG features.
    # C=10 provides moderate regularization, and HOG's default 9 orientations with
    # 8x8 cells is the standard configuration from the original Dalal & Triggs paper
    print("\n[3/8] Training SVM with HOG features...")
    model_dict = train(
        X_train, y_train, classifier_type="svm", feature_method="hog", C=10.0, kernel="rbf"
    )

    # ---- Step 4: Validate ----
    # Check performance on held-out validation data to detect overfitting
    # and establish a baseline before hyperparameter tuning
    print("\n[4/8] Validating...")
    val_metrics = validate(model_dict, X_val, y_val)

    # ---- Step 5: Optuna hyperparameter search ----
    # Automated search explores classifier type, feature method, and all
    # associated hyperparameters to find the best configuration
    print("\n[5/8] Running Optuna hyperparameter optimization...")
    if OPTUNA_AVAILABLE:
        # Set Optuna verbosity to WARNING to suppress per-trial progress logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create a study that maximizes validation accuracy
        study = optuna.create_study(direction="maximize")

        # Run 15 trials -- each trial trains a model with different hyperparameters
        # suggested by Optuna's TPE sampler based on previous trial results
        study.optimize(
            lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=15,
        )
        print(f"    Best trial accuracy: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        # Retrain with the best hyperparameters found by Optuna
        bp = study.best_params
        best_kwargs = {
            "classifier_type": bp["classifier_type"],
            "feature_method": bp["feature_method"],
            "pixels_per_cell": (bp["pixels_per_cell"], bp["pixels_per_cell"]),
            "orientations": bp["hog_orientations"],
        }
        # Add classifier-specific parameters based on the best classifier type
        if bp["classifier_type"] == "svm":
            best_kwargs["C"] = bp["svm_C"]
            best_kwargs["kernel"] = bp["svm_kernel"]
            if bp["svm_kernel"] == "rbf":
                best_kwargs["gamma"] = bp.get("svm_gamma", "scale")
        else:
            best_kwargs["n_estimators"] = bp["rf_n_estimators"]
            best_kwargs["max_depth"] = bp["rf_max_depth"]

        print("\n    Retraining with best params...")
        model_dict = train(X_train, y_train, **best_kwargs)
    else:
        print("    Optuna not installed, skipping.")

    # ---- Step 6: Final test evaluation ----
    # Use the test set exactly once for unbiased performance estimation
    print("\n[6/8] Final test evaluation...")
    test_metrics = test(model_dict, X_test, y_test)

    # ---- Step 7: Parameter comparison ----
    # Compare different configurations to understand hyperparameter sensitivity
    print("\n[7/8] Running parameter comparison...")
    compare_parameter_sets(n_samples=1000, img_size=32)

    # ---- Step 8: Real-world demo ----
    # Demonstrate the pipeline on a realistic medical imaging scenario
    print("\n[8/8] Running real-world medical X-ray classification demo...")
    real_world_demo()

    print("\n" + "=" * 70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
