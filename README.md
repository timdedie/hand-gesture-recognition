# Hand Gesture Recognition

A CNN-based hand gesture recognition system that classifies 5 gesture types using MediaPipe for hand detection.

## Classes

1. Thumbs up
2. Thumbs down
3. Peace sign
4. Open palm
5. No hand (background)

## Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## Usage

All scripts should be run from the repository root directory.

### Step 1: Collect Dataset

Run the data collection script to capture images using your webcam:

```bash
python collect_data.py
```

Controls:
- Press `1` to save thumbs up image
- Press `2` to save thumbs down image
- Press `3` to save peace sign image
- Press `4` to save open palm image
- Press `5` to save no hand image
- Press `q` to quit

The script uses MediaPipe to detect your hand and saves only the cropped hand region. A green bounding box shows the detected hand. Aim for 100-200 images per class.

### Step 2: Train the Model

Run training with all experiments:

```bash
python train.py
```

This runs two types of experiments:

**Augmentation experiments:**
- No augmentation (baseline)
- Flip only
- Rotate only
- Brightness only
- Blur only
- Flip + rotate combined
- All augmentations combined

**Dataset size experiments:**
- 50, 100, 200, and full dataset
- Each tested with and without augmentation

Results are saved to `results.json` and models are saved as `model_*.pth` files.

### Step 3: Evaluate

Evaluate the default trained model:

```bash
python evaluate.py
```

Or specify a different model:

```bash
python evaluate.py models/model_flip_rotate.pth
```

This prints:
- Overall accuracy
- F1-score (macro and weighted)
- Per-class accuracy
- Classification report
- Confusion matrix (saved as `confusion_matrix.png`)

### Step 4: Live Demo

Test the default model in real-time using your webcam:

```bash
python live_demo.py
```

Or specify a different model:

```bash
python live_demo.py models/model_flip_rotate.pth
```

Controls:
- Press `q` to quit

The demo detects your hand with a bounding box and shows the predicted gesture and confidence score.

## File Structure

```
hand-gesture-recognition/
├── collect_data.py      # Webcam capture with hand detection
├── augmentations.py     # Custom augmentation functions
├── dataset.py           # Data loading and splitting
├── model.py             # CNN architecture
├── train.py             # Training and experiments
├── evaluate.py          # Evaluation metrics
├── live_demo.py         # Real-time webcam demo
├── dataset/             # Image dataset (created by collect_data.py)
│   ├── thumbs_up/
│   ├── thumbs_down/
│   ├── peace/
│   ├── open_palm/
│   └── no_hand/
├── models/              # Trained models (created by train.py)
│   └── model_*.pth
└── results.json         # Experiment results (created by train.py)
```

## Model Architecture

Simple CNN with 3 convolutional layers:

- Conv2D(3, 32) -> ReLU -> MaxPool
- Conv2D(32, 64) -> ReLU -> MaxPool
- Conv2D(64, 128) -> ReLU -> MaxPool
- Flatten -> Dense(256) -> Dropout(0.5) -> Dense(5)

Input size: 64x64 RGB images

## Hand Detection

MediaPipe Hands is used to:
- Detect hand presence in the frame
- Extract bounding box coordinates from hand landmarks
- Crop the hand region with 20px padding

This ensures the model learns hand features rather than background.

## Augmentations

Custom implementations using OpenCV:

| Augmentation | Description |
|--------------|-------------|
| Horizontal flip | Mirrors the image |
| Rotation | Random rotation between -30 and 30 degrees |
| Brightness | Random brightness factor between 0.5 and 1.5 |
| Gaussian blur | Random kernel size (3, 5, or 7) |
| Color jitter | Random hue and saturation adjustment |
| Zoom | Random zoom factor between 1.0 and 1.3 |

## Data Split

- 80% training data
- 20% test data
- Stratified split to maintain class balance
