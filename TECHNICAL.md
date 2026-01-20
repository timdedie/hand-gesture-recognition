# Technical Deep Dive: Hand Gesture Recognition

This document explains the computer vision and deep learning math behind this project in detail.

---

## Table of Contents
1. [Data Flow Overview](#data-flow-overview)
2. [Image Representation](#image-representation)
3. [Convolution Operation](#convolution-operation)
4. [Our CNN Layer by Layer](#our-cnn-layer-by-layer)
5. [Activation Functions](#activation-functions)
6. [Pooling](#pooling)
7. [Fully Connected Layers](#fully-connected-layers)
8. [Dropout](#dropout)
9. [Softmax and Cross-Entropy Loss](#softmax-and-cross-entropy-loss)
10. [Backpropagation](#backpropagation)
11. [Adam Optimizer](#adam-optimizer)
12. [MediaPipe Hand Detection](#mediapipe-hand-detection)
13. [Data Augmentation Math](#data-augmentation-math)

---

## Data Flow Overview

```
Input Image (64×64×3)
        ↓
    Conv1 + ReLU + MaxPool → (32×32×32)
        ↓
    Conv2 + ReLU + MaxPool → (16×16×64)
        ↓
    Conv3 + ReLU + MaxPool → (8×8×128)
        ↓
    Flatten → (8192,)
        ↓
    FC1 + ReLU + Dropout → (256,)
        ↓
    FC2 → (5,)
        ↓
    Softmax → Probabilities
```

---

## Image Representation

### Pixels and Channels
A digital image is a 3D tensor (array) with dimensions **Height × Width × Channels**.

For our 64×64 RGB image:
- **Height (H)**: 64 pixels
- **Width (W)**: 64 pixels
- **Channels (C)**: 3 (Red, Green, Blue)

Each pixel value is typically 0-255 (uint8). We normalize to 0-1 (float32) by dividing by 255:

```
normalized_pixel = raw_pixel / 255.0
```

### Memory Layout
PyTorch uses **CHW** format (Channels, Height, Width), so our input tensor shape is `(3, 64, 64)`.

With a batch of 32 images: `(32, 3, 64, 64)` = **Batch × Channels × Height × Width**

Total values per image: 64 × 64 × 3 = **12,288 floats**

---

## Convolution Operation

### What is Convolution?
Convolution slides a small **kernel** (filter) across the image, computing element-wise multiplication and summing at each position.

### The Math
For a 2D convolution (ignoring channels for simplicity):

```
Output[i,j] = Σ Σ Input[i+m, j+n] × Kernel[m, n]
              m n
```

Where the sums run over the kernel dimensions.

### Visual Example

```
Input (5×5):                 Kernel (3×3):
[1, 1, 1, 0, 0]              [1, 0, 1]
[0, 1, 1, 1, 0]              [0, 1, 0]
[0, 0, 1, 1, 1]              [1, 0, 1]
[0, 0, 1, 1, 0]
[0, 1, 1, 0, 0]

Position (0,0):
1×1 + 1×0 + 1×1 +
0×0 + 1×1 + 1×0 +
0×1 + 0×0 + 1×1 = 4

Output[0,0] = 4
```

### Multi-Channel Convolution
With multiple input channels (like RGB), each filter has depth equal to input channels:

```
Filter shape: (kernel_h, kernel_w, in_channels)
```

For our first conv layer:
- Input: (64, 64, 3)
- Filter: (3, 3, 3) — a 3×3 kernel with depth 3 for RGB
- We have 32 such filters

Each filter produces ONE output channel. 32 filters → 32 output channels.

### Convolution Formula (with all dimensions)

For input `X` of shape `(C_in, H, W)` and kernel `K` of shape `(C_in, k_h, k_w)`:

```
Output[h, w] = Σ   Σ   Σ  X[c, h+i, w+j] × K[c, i, j] + bias
               c=0 i=0 j=0
```

Where:
- `c` iterates over input channels
- `i, j` iterate over kernel spatial dimensions
- `bias` is a learnable scalar added to each output

### Padding
We use `padding=1` with a 3×3 kernel. This adds 1 pixel of zeros around the input:

```
Original: 64×64
After padding: 66×66
After 3×3 conv: 64×64 (same as original!)
```

Formula for output size:
```
output_size = (input_size + 2×padding - kernel_size) / stride + 1
            = (64 + 2×1 - 3) / 1 + 1
            = 64
```

### Stride
Stride = how many pixels the kernel moves each step. We use stride=1 (default).

---

## Our CNN Layer by Layer

### Layer 1: Conv1

```python
nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

**Input:** `(batch, 3, 64, 64)`

**Parameters:**
- 32 filters, each of shape (3, 3, 3)
- 32 biases (one per filter)
- Total: 32 × (3 × 3 × 3) + 32 = **896 parameters**

**Calculation:**
```
For each of 32 filters:
    For each output position (64×64 = 4096 positions):
        Sum over 3×3×3 = 27 multiplications + bias
```

**Output:** `(batch, 32, 64, 64)`

**What it learns:** Low-level features like edges, corners, color gradients.

---

### After ReLU + MaxPool (Layer 1)

**After ReLU:** Same shape `(batch, 32, 64, 64)`, negative values → 0

**After MaxPool(2,2):**
```
Output size = 64 / 2 = 32
```
**Output:** `(batch, 32, 32, 32)`

Each 2×2 region is reduced to its maximum value.

---

### Layer 2: Conv2

```python
nn.Conv2d(32, 64, kernel_size=3, padding=1)
```

**Input:** `(batch, 32, 32, 32)`

**Parameters:**
- 64 filters, each of shape (3, 3, 32)
- 64 biases
- Total: 64 × (3 × 3 × 32) + 64 = **18,496 parameters**

**Output:** `(batch, 64, 32, 32)`

**What it learns:** Mid-level features like textures, simple shapes, finger segments.

---

### After ReLU + MaxPool (Layer 2)

**Output:** `(batch, 64, 16, 16)`

---

### Layer 3: Conv3

```python
nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

**Input:** `(batch, 64, 16, 16)`

**Parameters:**
- 128 filters, each of shape (3, 3, 64)
- 128 biases
- Total: 128 × (3 × 3 × 64) + 128 = **73,856 parameters**

**Output:** `(batch, 128, 16, 16)`

**What it learns:** High-level features like finger positions, hand shapes, gesture patterns.

---

### After ReLU + MaxPool (Layer 3)

**Output:** `(batch, 128, 8, 8)`

---

### Flatten

Reshape from 4D to 2D for fully connected layers:

```
(batch, 128, 8, 8) → (batch, 128 × 8 × 8) = (batch, 8192)
```

Each image is now a 1D vector of 8192 features.

---

### Layer 4: FC1 (Fully Connected)

```python
nn.Linear(8192, 256)
```

**Input:** `(batch, 8192)`

**Operation:** Matrix multiplication + bias
```
output = input @ W.T + b
```

Where:
- `W` has shape (256, 8192)
- `b` has shape (256,)

**Parameters:** 256 × 8192 + 256 = **2,097,408 parameters**

This is where most parameters live!

**Output:** `(batch, 256)`

---

### After ReLU + Dropout (Layer 4)

**After ReLU:** Same shape, negative → 0

**After Dropout(0.5):** During training, randomly set 50% of values to 0 and scale remaining by 2. Shape unchanged.

---

### Layer 5: FC2 (Output Layer)

```python
nn.Linear(256, 5)
```

**Input:** `(batch, 256)`

**Parameters:** 5 × 256 + 5 = **1,285 parameters**

**Output:** `(batch, 5)` — raw logits (unnormalized scores)

---

### Total Parameters

| Layer | Parameters |
|-------|-----------|
| Conv1 | 896 |
| Conv2 | 18,496 |
| Conv3 | 73,856 |
| FC1 | 2,097,408 |
| FC2 | 1,285 |
| **Total** | **2,191,941** |

~2.19 million parameters, most in FC1.

---

## Activation Functions

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

Graph:
```
        |     /
        |    /
        |   /
   -----+--/----
        | 0
        |
```

**Why ReLU?**
1. **Non-linearity:** Without activation, stacking linear layers = one linear layer. Non-linearity lets the network learn complex functions.
2. **Sparse activation:** Negative values become 0, leading to sparse representations
3. **No vanishing gradient:** Gradient is 1 for positive values (unlike sigmoid/tanh which saturate)

**Derivative:**
```
dReLU/dx = 1 if x > 0
         = 0 if x ≤ 0
```

The gradient is simply passed through for positive values, blocked for negative.

---

## Pooling

### Max Pooling

```python
nn.MaxPool2d(2, 2)  # kernel_size=2, stride=2
```

Takes the maximum value in each 2×2 window:

```
Input:              Output:
[1, 3 | 2, 4]
[5, 6 | 7, 8]   →   [6, 8]
------+------       [4, 8]
[1, 2 | 3, 4]
[0, 1 | 5, 8]
```

**Why pooling?**
1. **Reduce spatial dimensions:** 64→32→16→8 (less computation)
2. **Translation invariance:** Small shifts in input don't change output much
3. **Increase receptive field:** Each neuron "sees" more of the original image

### Receptive Field

The receptive field is how much of the original image each neuron "sees".

After each layer:
- After Conv1: 3×3
- After Pool1: 6×6
- After Conv2: 10×10
- After Pool2: 20×20
- After Conv3: 28×28
- After Pool3: 56×56

By the final conv layer, each neuron integrates information from most of the 64×64 input!

---

## Fully Connected Layers

### The Math

A fully connected layer computes:

```
y = Wx + b
```

Where:
- `x` is input vector of shape (n_in,)
- `W` is weight matrix of shape (n_out, n_in)
- `b` is bias vector of shape (n_out,)
- `y` is output vector of shape (n_out,)

**Expanded:**
```
y[0] = W[0,0]×x[0] + W[0,1]×x[1] + ... + W[0,8191]×x[8191] + b[0]
y[1] = W[1,0]×x[0] + W[1,1]×x[1] + ... + W[1,8191]×x[8191] + b[1]
...
y[255] = W[255,0]×x[0] + ... + W[255,8191]×x[8191] + b[255]
```

Each output neuron is connected to ALL 8192 input values. That's why it's called "fully connected".

---

## Dropout

```python
nn.Dropout(0.5)
```

During **training**:
1. Randomly select 50% of neurons
2. Set their output to 0
3. Scale remaining outputs by 1/(1-0.5) = 2

Example:
```
Input:     [1.0, 2.0, 3.0, 4.0]
Mask:      [1,   0,   1,   0  ]  (random)
Dropped:   [1.0, 0,   3.0, 0  ]
Scaled:    [2.0, 0,   6.0, 0  ]
```

During **inference** (eval mode):
- No dropout, use all neurons
- No scaling needed (training scaling compensates)

**Why dropout?**
- Prevents **co-adaptation**: neurons can't rely on specific other neurons
- Like training an **ensemble** of smaller networks
- Reduces **overfitting** significantly

---

## Softmax and Cross-Entropy Loss

### Softmax

Converts raw logits to probabilities:

```
softmax(z)_i = exp(z_i) / Σ exp(z_j)
                          j
```

**Step by step example:**
```
Logits z = [2.0, 1.0, 0.1, -1.0, -2.0]

Step 1: Exponentiate
exp(z) = [e^2.0, e^1.0, e^0.1, e^-1.0, e^-2.0]
       = [7.39, 2.72, 1.11, 0.37, 0.14]

Step 2: Sum
sum = 7.39 + 2.72 + 1.11 + 0.37 + 0.14 = 11.73

Step 3: Normalize
softmax(z) = [7.39/11.73, 2.72/11.73, 1.11/11.73, 0.37/11.73, 0.14/11.73]
           = [0.63, 0.23, 0.09, 0.03, 0.01]
```

Properties:
- All outputs in (0, 1)
- Sum to exactly 1.0
- Preserves ordering (highest logit → highest probability)

### Cross-Entropy Loss

Measures how different predicted distribution is from true distribution:

```
L = -Σ y_i × log(p_i)
     i
```

Where:
- `y` is one-hot true label (e.g., [0, 0, 1, 0, 0] for class 2)
- `p` is predicted probability from softmax

Since only one `y_i = 1`, this simplifies to:
```
L = -log(p_correct_class)
```

**Example:**
- True class: 2 (peace sign)
- Predicted probabilities: [0.1, 0.1, 0.7, 0.05, 0.05]
- Loss = -log(0.7) = 0.357

If prediction was worse: [0.1, 0.1, 0.3, 0.25, 0.25]
- Loss = -log(0.3) = 1.204 (higher loss = worse)

If prediction was confident and wrong: [0.9, 0.02, 0.02, 0.03, 0.03]
- Loss = -log(0.02) = 3.91 (very high! penalizes confident mistakes)

**Why cross-entropy?**
- Heavily penalizes confident wrong predictions
- Gradient is clean and simple: `∂L/∂z_i = p_i - y_i`

---

## Backpropagation

### The Chain Rule

To train, we need: "How much does each weight affect the loss?"

For a chain of functions `L(f(g(x)))`:
```
∂L/∂x = (∂L/∂f) × (∂f/∂g) × (∂g/∂x)
```

### Example: Simple Network

```
x → [W1] → h → [ReLU] → a → [W2] → z → [Softmax+Loss] → L
```

**Forward pass:** compute all intermediate values

**Backward pass:** compute gradients from end to start

```
∂L/∂z = softmax(z) - y           # Gradient at output

∂L/∂W2 = ∂L/∂z × a.T             # Weight gradient

∂L/∂a = W2.T × ∂L/∂z             # Pass gradient back

∂L/∂h = ∂L/∂a × (h > 0)          # Through ReLU

∂L/∂W1 = ∂L/∂h × x.T             # First layer weights
```

### Gradient of Softmax + Cross-Entropy

Combined gradient (this is why we use them together):
```
∂L/∂z_i = p_i - y_i
```

If true class is 2 and p = [0.1, 0.2, 0.6, 0.05, 0.05]:
```
y = [0, 0, 1, 0, 0]
∂L/∂z = [0.1-0, 0.2-0, 0.6-1, 0.05-0, 0.05-0]
      = [0.1, 0.2, -0.4, 0.05, 0.05]
```

The gradient pushes z[2] up (negative gradient) and others down (positive gradient).

### Gradient Through Convolution

For `Y = Conv(X, K)`:
```
∂L/∂K[m,n] = Σ X[i+m, j+n] × ∂L/∂Y[i,j]    # Kernel gradient
             i,j

∂L/∂X[i,j] = Σ K[m,n] × ∂L/∂Y[i-m, j-n]    # Input gradient (for deeper layers)
             m,n
```

The input gradient is essentially a convolution with a flipped kernel.

---

## Adam Optimizer

### Why Not Plain SGD?

Plain SGD: `W = W - lr × gradient`

Problems:
- Same learning rate for all parameters
- Oscillates in steep directions, slow in flat directions
- Gets stuck easily

### Adam Algorithm

Adam maintains two moving averages per parameter:
- `m`: First moment (mean of gradients) — momentum
- `v`: Second moment (mean of squared gradients) — adaptive learning rate

```python
# Initialize
m = 0
v = 0
t = 0

# Each training step
t += 1
g = compute_gradient()

# Update moments
m = β1 × m + (1 - β1) × g
v = β2 × v + (1 - β2) × g²

# Bias correction (important early in training)
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)

# Update weights
W = W - lr × m_hat / (sqrt(v_hat) + ε)
```

**Our settings:**
- `lr = 0.001`
- `β1 = 0.9` (momentum decay)
- `β2 = 0.999` (RMSprop decay)
- `ε = 1e-8` (prevent division by zero)

**Intuition:**
- `m` gives **momentum**: keeps moving in consistent gradient direction
- `v` gives **adaptive rate**: parameters with large gradients get smaller updates
- Bias correction fixes initialization at 0

---

## MediaPipe Hand Detection

### Architecture Overview

MediaPipe Hands uses a two-stage pipeline:

**Stage 1: Palm Detector**
- BlazePalm: lightweight CNN that finds palm bounding boxes
- Why palm? Palms are rigid (unlike fingers), easier to detect
- Output: bounding box + confidence

**Stage 2: Hand Landmark Model**
- Takes cropped palm region
- Outputs 21 3D landmarks (x, y, z)
- z is depth relative to wrist

### The 21 Hand Landmarks

```
Index:  Finger:
0       Wrist
1-4     Thumb (CMC, MCP, IP, TIP)
5-8     Index finger (MCP, PIP, DIP, TIP)
9-12    Middle finger
13-16   Ring finger
17-20   Pinky

Visual:
            8   12  16  20    ← Fingertips
            |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18
        |   |   |   |
        5   9   13  17
         \  |   |  /
          \ |   | /
       4   \|   |/
        \   3   2   1
         \  |  /
          \ | /
           \|/
            0  ← Wrist
```

### Our Bounding Box Extraction

```python
# landmarks.landmark is a list of 21 NormalizedLandmark objects
# Each has x, y, z in [0, 1] normalized coordinates

x_coords = [lm.x for lm in landmarks.landmark]
y_coords = [lm.y for lm in landmarks.landmark]

# Convert to pixel coordinates and add padding
x_min = int(min(x_coords) * frame_width) - 20
x_max = int(max(x_coords) * frame_width) + 20
y_min = int(min(y_coords) * frame_height) - 20
y_max = int(max(y_coords) * frame_height) + 20

# Clamp to image bounds
x_min = max(0, x_min)
x_max = min(frame_width, x_max)
y_min = max(0, y_min)
y_max = min(frame_height, y_max)

# Crop
hand_crop = frame[y_min:y_max, x_min:x_max]
```

The 20px padding captures context around the hand edges.

---

## Data Augmentation Math

### Horizontal Flip

```python
flipped = cv2.flip(image, 1)  # 1 = flip around y-axis
```

**Transformation:**
```
new_x = width - 1 - old_x
new_y = old_y
```

**Matrix form (homogeneous coordinates):**
```
[x']   [-1  0  w-1] [x]
[y'] = [ 0  1   0 ] [y]
[1 ]   [ 0  0   1 ] [1]
```

### Rotation

```python
angle = random.uniform(-30, 30)  # degrees
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
rotated = cv2.warpAffine(image, M, (w, h))
```

**Rotation matrix around origin:**
```
R = [cos(θ)  -sin(θ)]
    [sin(θ)   cos(θ)]
```

**Rotation around center (cx, cy):**
1. Translate center to origin
2. Rotate
3. Translate back

```
M = [cos(θ)  -sin(θ)  (1-cos(θ))×cx + sin(θ)×cy]
    [sin(θ)   cos(θ)  (1-cos(θ))×cy - sin(θ)×cx]
```

For θ = 30°:
- cos(30°) ≈ 0.866
- sin(30°) = 0.5

### Brightness Adjustment

```python
factor = random.uniform(0.5, 1.5)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

**HSV Color Space:**
- **H** (Hue): Color wheel position, 0-180 in OpenCV (0=red, 60=green, 120=blue)
- **S** (Saturation): Color intensity, 0=gray, 255=vivid
- **V** (Value): Brightness, 0=black, 255=bright

Multiplying V directly scales brightness while preserving color.

### Gaussian Blur

```python
kernel_size = random.choice([3, 5, 7])
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
```

**2D Gaussian function:**
```
G(x, y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)
```

When sigmaX=0, OpenCV computes σ from kernel size:
```
σ = 0.3 × ((ksize-1) × 0.5 - 1) + 0.8
```

**5×5 Gaussian kernel (approximate):**
```
     1   4   7   4   1
     4  16  26  16   4
     7  26  41  26   7    × (1/273)
     4  16  26  16   4
     1   4   7   4   1
```

Center has highest weight (41), edges have lowest (1). Smooths by weighted averaging.

### Color Jitter

```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)

# Hue shift (circular)
hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180

# Saturation scale
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)

result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

Hue is circular (0° and 180° are both red), so we use modulo.

### Zoom (Center Crop)

```python
factor = random.uniform(1.0, 1.3)
h, w = image.shape[:2]

# Scale up
new_h, new_w = int(h * factor), int(w * factor)
scaled = cv2.resize(image, (new_w, new_h))

# Crop center
start_x = (new_w - w) // 2
start_y = (new_h - h) // 2
zoomed = scaled[start_y:start_y+h, start_x:start_x+w]
```

**Example with factor=1.3:**
```
64×64 → scale to 83×83 → crop center 64×64

What happens:
- Original edges (9-10 pixels each side) are removed
- Center is magnified
- Simulates camera zoom or hand being closer
```

---

## Putting It All Together

### Complete Forward Pass

```python
def forward(image):
    # image: (64, 64, 3) uint8

    # 1. Preprocess
    x = image.astype(np.float32) / 255.0  # Normalize
    x = x.transpose(2, 0, 1)               # HWC → CHW
    x = torch.tensor(x).unsqueeze(0)       # Add batch dim: (1, 3, 64, 64)

    # 2. Conv Block 1
    x = conv1(x)      # (1, 32, 64, 64)  - 3×3 conv, 32 filters
    x = relu(x)       # (1, 32, 64, 64)  - max(0, x)
    x = maxpool(x)    # (1, 32, 32, 32)  - 2×2 max

    # 3. Conv Block 2
    x = conv2(x)      # (1, 64, 32, 32)
    x = relu(x)       # (1, 64, 32, 32)
    x = maxpool(x)    # (1, 64, 16, 16)

    # 4. Conv Block 3
    x = conv3(x)      # (1, 128, 16, 16)
    x = relu(x)       # (1, 128, 16, 16)
    x = maxpool(x)    # (1, 128, 8, 8)

    # 5. Flatten
    x = x.view(1, -1) # (1, 8192)

    # 6. FC Block
    x = fc1(x)        # (1, 256)
    x = relu(x)       # (1, 256)
    x = dropout(x)    # (1, 256) - 50% zeroed during training

    # 7. Output
    x = fc2(x)        # (1, 5) - logits

    # 8. Softmax (for inference)
    probs = softmax(x)  # (1, 5) - probabilities

    return probs.argmax()  # Predicted class
```

### Training Loop Explained

```python
for epoch in range(20):
    for images, labels in dataloader:
        # images: (32, 3, 64, 64) - batch of 32
        # labels: (32,) - class indices 0-4

        # Forward pass
        logits = model(images)           # (32, 5)
        loss = cross_entropy(logits, labels)  # scalar

        # Backward pass
        optimizer.zero_grad()  # Clear gradients from last iteration
        loss.backward()        # Compute ∂L/∂W for all weights
        optimizer.step()       # W = W - lr × Adam(gradient)
```

**What happens in `loss.backward()`:**
1. PyTorch traces computation graph from loss back to inputs
2. Applies chain rule at each operation
3. Accumulates gradients in `weight.grad` for each parameter

**What happens in `optimizer.step()`:**
1. For each parameter, update momentum (`m`) and variance (`v`)
2. Compute bias-corrected estimates
3. Update: `W = W - lr × m_hat / (sqrt(v_hat) + ε)`

---

## Key Takeaways

1. **Images are 3D tensors** — height × width × channels, normalized to [0,1]

2. **Convolution = sliding dot product** — learns local patterns regardless of position

3. **Depth increases through network** — 3 → 32 → 64 → 128 channels, capturing increasingly abstract features

4. **Spatial size decreases** — 64 → 32 → 16 → 8 via pooling, reducing computation and gaining invariance

5. **FC layers are expensive** — FC1 alone has 2M of our 2.19M parameters

6. **Softmax + cross-entropy** — converts logits to probabilities and measures prediction quality

7. **Backprop = chain rule** — gradient flows backward through every operation

8. **Adam adapts per-parameter** — frequently updated params get smaller learning rates

9. **Hand cropping was essential** — without it, CNN learned backgrounds instead of gestures
