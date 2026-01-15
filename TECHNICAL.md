# Technical Documentation

## Overview

This project implements a hand gesture recognition system using a Convolutional Neural Network (CNN). The system captures images from a webcam, applies custom data augmentation techniques, trains a CNN classifier, and evaluates performance using standard metrics.

---

## 1. Data Collection (`collect_data.py`)

### How It Works

The data collection script uses OpenCV to capture frames from the default webcam (device index 0). Each frame is a BGR image captured in real-time.

### Technical Flow

1. **Initialize webcam**: `cv2.VideoCapture(0)` opens the default camera
2. **Create directories**: Creates folder structure for each gesture class
3. **Capture loop**: Continuously reads frames using `cap.read()`
4. **Display overlay**: Shows live count of images per class on the frame
5. **Save on keypress**: When user presses 1-5, saves the current frame as JPEG

### Image Storage

Images are saved as JPEG files with the naming convention `{class_name}_{index}.jpg`. The JPEG format uses lossy compression which reduces file size while maintaining visual quality suitable for training.

### Frame Capture Details

- Resolution: Default camera resolution (typically 640x480 or 1280x720)
- Color space: BGR (OpenCV default)
- Format: 8-bit unsigned integer per channel (0-255)

---

## 2. Data Augmentation (`augmentations.py`)

All augmentation functions are implemented using basic OpenCV operations without high-level frameworks like torchvision.

### 2.1 Horizontal Flip

```python
def horizontal_flip(image):
    return cv2.flip(image, 1)
```

**How it works**: `cv2.flip(image, 1)` flips the image around the y-axis. The second parameter specifies the flip code:
- 0 = flip around x-axis (vertical flip)
- 1 = flip around y-axis (horizontal flip)
- -1 = flip around both axes

**Mathematical operation**: For a pixel at position (x, y) in an image of width W:
```
new_x = W - 1 - x
new_y = y
```

**Why it helps**: Hand gestures can appear from either direction. Flipping doubles the effective dataset size and makes the model invariant to left/right orientation.

### 2.2 Rotation

```python
def rotate(image, angle=None):
    if angle is None:
        angle = random.uniform(-30, 30)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))
```

**How it works**:

1. `cv2.getRotationMatrix2D(center, angle, scale)` creates a 2x3 affine transformation matrix:
```
M = | cos(θ)  -sin(θ)  (1-cos(θ))*cx + sin(θ)*cy |
    | sin(θ)   cos(θ)  -sin(θ)*cx + (1-cos(θ))*cy |
```
Where θ is the angle in degrees and (cx, cy) is the center point.

2. `cv2.warpAffine(image, matrix, (w, h))` applies the transformation to every pixel:
```
dst(x', y') = src(M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2])
```

**Angle range**: -30 to +30 degrees. This range simulates natural hand tilt without extreme rotations that would be unrealistic.

**Why it helps**: Users may hold their hand at slightly different angles. Rotation augmentation makes the model robust to these variations.

### 2.3 Brightness Adjustment

```python
def adjust_brightness(image, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

**How it works**:

1. Convert BGR to HSV color space:
   - H (Hue): Color type (0-179 in OpenCV)
   - S (Saturation): Color intensity (0-255)
   - V (Value): Brightness (0-255)

2. Multiply the V channel by the brightness factor

3. Clip values to valid range [0, 255] to prevent overflow

4. Convert back to BGR

**Factor range**: 0.5 to 1.5
- Factor < 1.0: Darker image
- Factor = 1.0: No change
- Factor > 1.0: Brighter image

**Why it helps**: Lighting conditions vary significantly between environments. This augmentation simulates different lighting scenarios.

### 2.4 Gaussian Blur

```python
def gaussian_blur(image, kernel_size=None):
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

**How it works**:

Gaussian blur convolves the image with a Gaussian kernel. The kernel is a 2D matrix where values follow a Gaussian distribution:

```
G(x, y) = (1 / 2πσ²) * e^(-(x² + y²) / 2σ²)
```

For a 3x3 kernel with σ=1, approximate values:
```
| 0.075  0.124  0.075 |
| 0.124  0.204  0.124 |
| 0.075  0.124  0.075 |
```

The kernel is applied via convolution:
```
output(x, y) = Σ Σ input(x+i, y+j) * kernel(i, j)
```

**Kernel sizes**: 3, 5, or 7 (must be odd numbers)
- Larger kernel = more blur
- σ = 0 means OpenCV calculates σ automatically from kernel size

**Why it helps**: Simulates camera focus issues and motion blur. Makes the model robust to slightly out-of-focus images.

### 2.5 Color Jitter

```python
def color_jitter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

**How it works**:

1. **Hue shift**: Adds a random value (-10 to +10) to the H channel. Modulo 180 ensures the value stays in valid range. This shifts the color around the color wheel.

2. **Saturation scale**: Multiplies the S channel by a random factor (0.8 to 1.2). This makes colors more vivid or more muted.

**Why it helps**: Different cameras have different color profiles. Skin tones vary between people. This augmentation makes the model robust to color variations.

### 2.6 Zoom

```python
def zoom(image, factor=None):
    if factor is None:
        factor = random.uniform(1.0, 1.3)
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(image, (new_w, new_h))
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    return resized[start_y:start_y + h, start_x:start_x + w]
```

**How it works**:

1. Resize image to larger dimensions using `cv2.resize()` which uses bilinear interpolation by default
2. Crop the center portion to match original dimensions
3. This effectively zooms into the center of the image

**Interpolation**: Bilinear interpolation calculates new pixel values as weighted average of 4 nearest pixels:
```
f(x, y) ≈ f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
```

**Factor range**: 1.0 to 1.3 (up to 30% zoom)

**Why it helps**: Hand distance from camera varies. Zoom augmentation simulates this variation.

---

## 3. Dataset Loading (`dataset.py`)

### 3.1 GestureDataset Class

This class extends PyTorch's `Dataset` class to create a custom dataset.

```python
class GestureDataset(Dataset):
    def __init__(self, images, labels, augment_list=None, augment_prob=0.5):
```

**Parameters**:
- `images`: NumPy array of image data
- `labels`: NumPy array of class indices (0-4)
- `augment_list`: List of augmentation names to apply
- `augment_prob`: Probability of applying augmentation (default 50%)

### 3.2 Data Preprocessing in `__getitem__`

```python
def __getitem__(self, idx):
    image = self.images[idx].copy()
    label = self.labels[idx]

    if self.augment_list and random.random() < self.augment_prob:
        image = apply_augmentations(image, self.augment_list)

    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    return torch.tensor(image), torch.tensor(label)
```

**Step-by-step**:

1. **Copy image**: Prevents modifying the original data

2. **Apply augmentations**: With probability `augment_prob`, applies the specified augmentations

3. **Resize to 64x64**: Standardizes input size for the CNN. Uses bilinear interpolation.

4. **Normalize to [0, 1]**: Divides by 255 to convert from uint8 (0-255) to float32 (0.0-1.0). This helps with gradient stability during training.

5. **Transpose dimensions**: Converts from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) format required by PyTorch.
   - Input shape: (64, 64, 3)
   - Output shape: (3, 64, 64)

### 3.3 Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)
```

**Parameters**:
- `test_size=0.2`: 20% of data goes to test set
- `random_state=42`: Fixed seed for reproducibility
- `stratify=labels`: Ensures each class has same proportion in train and test sets

**Stratification**: Without stratification, random splitting might put 90% of one class in training and only 10% in test. Stratified splitting maintains the original class distribution in both sets.

### 3.4 DataLoader

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

**batch_size=32**: Groups 32 images into a single batch for parallel processing

**shuffle=True (training)**: Randomizes order each epoch to prevent the model from learning sequence patterns

**shuffle=False (testing)**: Maintains consistent order for reproducible evaluation

---

## 4. CNN Architecture (`model.py`)

### 4.1 Network Structure

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)
```

### 4.2 Layer-by-Layer Analysis

**Input**: (batch_size, 3, 64, 64)

#### Convolutional Layer 1
```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
```
- Input channels: 3 (RGB)
- Output channels: 32 (learned filters)
- Kernel size: 3x3
- Padding: 1 (maintains spatial dimensions)

**Convolution operation**:
```
output[b, c_out, h, w] = Σ Σ Σ input[b, c_in, h+i, w+j] * weight[c_out, c_in, i, j] + bias[c_out]
```

**Parameters**: 3 × 32 × 3 × 3 + 32 = 896

**Output shape**: (batch_size, 32, 64, 64)

After ReLU + MaxPool: (batch_size, 32, 32, 32)

#### Convolutional Layer 2
```python
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
```
**Parameters**: 32 × 64 × 3 × 3 + 64 = 18,496

**Output after ReLU + MaxPool**: (batch_size, 64, 16, 16)

#### Convolutional Layer 3
```python
self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
```
**Parameters**: 64 × 128 × 3 × 3 + 128 = 73,856

**Output after ReLU + MaxPool**: (batch_size, 128, 8, 8)

#### Flatten
```python
x = x.view(x.size(0), -1)
```
Reshapes from (batch_size, 128, 8, 8) to (batch_size, 8192)

#### Fully Connected Layer 1
```python
self.fc1 = nn.Linear(128 * 8 * 8, 256)
```
**Parameters**: 8192 × 256 + 256 = 2,097,408

#### Dropout
```python
self.dropout = nn.Dropout(0.5)
```
During training, randomly sets 50% of neurons to zero. This prevents overfitting by forcing the network to learn redundant representations.

**Scaling**: During training, outputs are scaled by 1/(1-p) = 2 to maintain expected values. During evaluation, dropout is disabled.

#### Fully Connected Layer 2 (Output)
```python
self.fc2 = nn.Linear(256, num_classes)
```
**Parameters**: 256 × 5 + 5 = 1,285

**Total parameters**: ~2.19 million

### 4.3 Activation Functions

**ReLU (Rectified Linear Unit)**:
```
f(x) = max(0, x)
```
- Introduces non-linearity
- Computationally efficient
- Helps with vanishing gradient problem
- Sparse activation (many zeros)

### 4.4 MaxPooling

```python
self.pool = nn.MaxPool2d(2, 2)
```

Takes maximum value in each 2x2 window:
```
| 1  3 |
| 2  4 |  → 4
```

**Purpose**:
- Reduces spatial dimensions by half
- Provides translation invariance
- Reduces computational cost
- Helps prevent overfitting

---

## 5. Training (`train.py`)

### 5.1 Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

Cross-entropy loss for multi-class classification:

```
L = -Σ y_true[c] * log(softmax(y_pred)[c])
```

For single correct class c:
```
L = -log(softmax(y_pred)[c])
```

**Softmax** converts raw logits to probabilities:
```
softmax(x)[i] = e^(x[i]) / Σ e^(x[j])
```

### 5.2 Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Adam (Adaptive Moment Estimation)** combines:
- Momentum: Uses exponentially decaying average of past gradients
- RMSprop: Uses exponentially decaying average of squared gradients

**Update rule**:
```
m_t = β1 * m_(t-1) + (1 - β1) * g_t          # First moment
v_t = β2 * v_(t-1) + (1 - β2) * g_t²         # Second moment
m̂_t = m_t / (1 - β1^t)                       # Bias correction
v̂_t = v_t / (1 - β2^t)                       # Bias correction
θ_t = θ_(t-1) - lr * m̂_t / (√v̂_t + ε)       # Parameter update
```

**Default values**: β1=0.9, β2=0.999, ε=1e-8

### 5.3 Training Loop

```python
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**Step-by-step**:

1. `model.train()`: Enables dropout and batch normalization training mode

2. `optimizer.zero_grad()`: Clears gradients from previous iteration (PyTorch accumulates gradients by default)

3. `outputs = model(images)`: Forward pass through the network

4. `loss = criterion(outputs, labels)`: Compute loss value

5. `loss.backward()`: Backpropagation - computes gradients for all parameters using chain rule

6. `optimizer.step()`: Updates parameters using computed gradients

### 5.4 Backpropagation

The chain rule computes gradients layer by layer:

```
∂L/∂w = ∂L/∂y * ∂y/∂w
```

For a network with layers f1, f2, f3:
```
∂L/∂w1 = ∂L/∂f3 * ∂f3/∂f2 * ∂f2/∂f1 * ∂f1/∂w1
```

### 5.5 Evaluation Mode

```python
model.eval()
with torch.no_grad():
    # evaluation code
```

`model.eval()`: Disables dropout and sets batch norm to use running statistics

`torch.no_grad()`: Disables gradient computation for memory efficiency

---

## 6. Evaluation Metrics (`evaluate.py`)

### 6.1 Accuracy

```
Accuracy = (True Positives + True Negatives) / Total Samples
```

**Per-class accuracy**:
```
Accuracy[c] = Correct predictions for class c / Total samples of class c
```

### 6.2 Confusion Matrix

A 5x5 matrix where entry (i, j) represents:
- Number of samples with true label i
- Predicted as label j

```
              Predicted
            0   1   2   3   4
        0 | TP  .   .   .   . |
True    1 | .   TP  .   .   . |
        2 | .   .   TP  .   . |
        3 | .   .   .   TP  . |
        4 | .   .   .   .   TP|
```

Diagonal elements are correct predictions. Off-diagonal elements are misclassifications.

### 6.3 Precision, Recall, F1-Score

**Precision** (per class): Of all predictions for this class, how many were correct?
```
Precision[c] = TP[c] / (TP[c] + FP[c])
```

**Recall** (per class): Of all actual samples of this class, how many did we find?
```
Recall[c] = TP[c] / (TP[c] + FN[c])
```

**F1-Score** (per class): Harmonic mean of precision and recall
```
F1[c] = 2 * (Precision[c] * Recall[c]) / (Precision[c] + Recall[c])
```

### 6.4 Averaging Methods

**Macro F1**: Simple average across classes (treats all classes equally)
```
F1_macro = (1/C) * Σ F1[c]
```

**Weighted F1**: Weighted average by class frequency (accounts for class imbalance)
```
F1_weighted = Σ (n[c] / N) * F1[c]
```

Where n[c] is the number of samples in class c and N is total samples.

---

## 7. Experiment Design

### 7.1 Augmentation Experiments

Tests 7 configurations:
1. No augmentation (baseline)
2. Flip only
3. Rotate only
4. Brightness only
5. Blur only
6. Flip + rotate
7. All augmentations

**Purpose**: Identify which augmentations help most and find optimal combination.

### 7.2 Dataset Size Experiments

Tests 4 sizes: 50, 100, 200, full dataset

Each size tested with and without augmentation.

**Purpose**:
- Understand minimum data requirements
- Measure augmentation benefit at different scales
- Typically, augmentation helps more with smaller datasets

---

## 8. Technical Specifications

| Component | Specification |
|-----------|---------------|
| Input image size | 64 × 64 × 3 |
| Color format | BGR (OpenCV) → RGB normalized [0, 1] |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross-entropy |
| Epochs | 20 |
| Train/test split | 80% / 20% |
| Dropout rate | 50% |
| Number of classes | 5 |
| Total parameters | ~2.19 million |

---

## 9. File Dependencies

```
collect_data.py
    └── cv2 (OpenCV)

augmentations.py
    ├── cv2 (OpenCV)
    ├── numpy
    └── random

dataset.py
    ├── cv2 (OpenCV)
    ├── numpy
    ├── sklearn.model_selection
    ├── torch
    └── augmentations.py

model.py
    └── torch.nn

train.py
    ├── torch
    ├── torch.nn
    ├── torch.optim
    ├── model.py
    ├── dataset.py
    └── json

evaluate.py
    ├── torch
    ├── numpy
    ├── sklearn.metrics
    ├── matplotlib
    ├── seaborn
    ├── model.py
    └── dataset.py
```
