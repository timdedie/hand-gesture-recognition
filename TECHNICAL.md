# Technical Documentation

## Overview

This project implements a hand gesture recognition system using a Convolutional Neural Network (CNN). The system uses MediaPipe for hand detection, captures cropped hand images from a webcam, applies custom data augmentation techniques, trains a CNN classifier, and evaluates performance using standard metrics.

---

## 1. Hand Detection (MediaPipe)

### How It Works

MediaPipe Hands detects hands in real-time and provides 21 hand landmarks. We use these landmarks to compute a bounding box around the hand.

### Technical Flow

1. Convert frame from BGR to RGB (MediaPipe expects RGB)
2. Process frame with `hands.process(rgb_frame)`
3. Extract x,y coordinates from all 21 landmarks
4. Compute bounding box from min/max coordinates
5. Add 20px padding and clip to image boundaries
6. Crop the hand region

### Bounding Box Calculation

```python
x_coords = [lm.x for lm in landmarks.landmark]
y_coords = [lm.y for lm in landmarks.landmark]

x_min = int(min(x_coords) * w) - 20
x_max = int(max(x_coords) * w) + 20
y_min = int(min(y_coords) * h) - 20
y_max = int(max(y_coords) * h) + 20
```

Landmarks are normalized to [0, 1], so we multiply by image dimensions to get pixel coordinates.

### Why Hand Detection Matters

Without hand detection, the model learns background features instead of hand gestures. This causes high test accuracy but poor real-world performance. Cropping isolates the hand and forces the model to learn gesture features.

---

## 2. Data Collection (`collect_data.py`)

### How It Works

The data collection script uses OpenCV to capture frames from the default webcam and MediaPipe to detect and crop hands.

### Technical Flow

1. Initialize webcam: `cv2.VideoCapture(0)`
2. Initialize MediaPipe Hands detector
3. Create directories for each gesture class
4. Capture loop: Read frame, detect hand, draw bounding box
5. On keypress 1-5: Save cropped hand region as JPEG
6. For "no_hand" class: Save full frame (no hand detection needed)

### Image Storage

Images are saved as JPEG files with naming convention `{class_name}_{index}.jpg`. Only the cropped hand region is saved (except for no_hand class).

---

## 3. Data Augmentation (`augmentations.py`)

All augmentation functions are implemented using basic OpenCV operations without high-level frameworks like torchvision.

### 3.1 Horizontal Flip

```python
def horizontal_flip(image):
    return cv2.flip(image, 1)
```

Flips the image around the y-axis. Makes the model invariant to left/right hand orientation.

### 3.2 Rotation

```python
def rotate(image, angle=None):
    if angle is None:
        angle = random.uniform(-30, 30)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))
```

Rotates by -30 to +30 degrees around the center. Simulates natural hand tilt.

### 3.3 Brightness Adjustment

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

Multiplies the V (value/brightness) channel in HSV color space. Factor 0.5-1.5 simulates different lighting.

### 3.4 Gaussian Blur

```python
def gaussian_blur(image, kernel_size=None):
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

Applies Gaussian blur with kernel size 3, 5, or 7. Simulates camera focus issues.

### 3.5 Color Jitter

```python
def color_jitter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

Shifts hue by ±10 and scales saturation by 0.8-1.2. Accounts for different skin tones and camera color profiles.

### 3.6 Zoom

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

Resizes image larger then crops center. Up to 30% zoom simulates distance variation.

---

## 4. Dataset Loading (`dataset.py`)

### 4.1 GestureDataset Class

```python
class GestureDataset(Dataset):
    def __init__(self, images, labels, augment_list=None, augment_prob=0.5):
```

Parameters:
- `images`: List of image arrays
- `labels`: NumPy array of class indices (0-4)
- `augment_list`: List of augmentation names to apply
- `augment_prob`: Probability of applying augmentation (default 50%)

### 4.2 Data Preprocessing

```python
def __getitem__(self, idx):
    image = self.images[idx].copy()
    if self.augment_list and random.random() < self.augment_prob:
        image = apply_augmentations(image, self.augment_list)
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image), torch.tensor(label)
```

Steps:
1. Copy image to avoid modifying original
2. Apply augmentations with 50% probability
3. Resize to 64x64
4. Normalize to [0, 1]
5. Transpose from HWC to CHW format

### 4.3 Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)
```

80/20 split with stratification to maintain class balance.

---

## 5. CNN Architecture (`model.py`)

### 5.1 Network Structure

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
```

### 5.2 Forward Pass

Input: (batch, 3, 64, 64)

| Layer | Output Shape |
|-------|--------------|
| Conv1 + ReLU + Pool | (batch, 32, 32, 32) |
| Conv2 + ReLU + Pool | (batch, 64, 16, 16) |
| Conv3 + ReLU + Pool | (batch, 128, 8, 8) |
| Flatten | (batch, 8192) |
| FC1 + ReLU + Dropout | (batch, 256) |
| FC2 | (batch, 5) |

Total parameters: ~2.19 million

---

## 6. Training (`train.py`)

### 6.1 Loss and Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Cross-entropy loss for multi-class classification. Adam optimizer with learning rate 0.001.

### 6.2 Training Loop

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

Standard PyTorch training: zero gradients, forward pass, compute loss, backward pass, update weights.

---

## 7. Evaluation (`evaluate.py`)

### Metrics

- **Accuracy**: Correct predictions / Total samples
- **Precision**: TP / (TP + FP) per class
- **Recall**: TP / (TP + FN) per class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: 5x5 matrix showing prediction distribution

---

## 8. Experiment Design

### 8.1 Augmentation Experiments

Tests 7 configurations:
1. No augmentation (baseline)
2. Flip only
3. Rotate only
4. Brightness only
5. Blur only
6. Flip + rotate
7. All augmentations

### 8.2 Dataset Size Experiments

Tests 4 sizes: 50, 100, 200, full dataset. Each tested with and without augmentation.

---

## 9. Technical Specifications

| Component | Specification |
|-----------|---------------|
| Input image size | 64 × 64 × 3 |
| Hand detection | MediaPipe Hands 0.10.9 |
| Bounding box padding | 20 pixels |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross-entropy |
| Epochs | 20 |
| Train/test split | 80% / 20% |
| Dropout rate | 50% |
| Number of classes | 5 |

---

## 10. File Dependencies

```
collect_data.py
    ├── cv2 (OpenCV)
    └── mediapipe

crop_dataset.py
    ├── cv2 (OpenCV)
    └── mediapipe

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

live_demo.py
    ├── torch
    ├── cv2 (OpenCV)
    ├── numpy
    ├── mediapipe
    └── model.py
```
