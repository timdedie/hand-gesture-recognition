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
uv sync
```

This automatically creates a virtual environment with Python 3.11 (from `.python-version`) and installs all dependencies.

## Usage

All scripts should be run from the repository root directory.

### Step 1: Collect Dataset

Run the data collection script to capture images using your webcam:

```bash
uv run collect_data.py
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
uv run train.py
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
uv run evaluate.py
```

Or specify a different model:

```bash
uv run evaluate.py models/model_flip_rotate.pth
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
uv run live_demo.py
```

Or specify a different model:

```bash
uv run live_demo.py models/model_flip_rotate.pth
```

Controls:
- Press `q` to quit

The demo detects your hand with a bounding box and shows the predicted gesture and confidence score.
