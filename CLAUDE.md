# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CNN-based hand gesture recognition system that classifies 5 gestures (thumbs up, thumbs down, peace sign, open palm, no hand) from webcam input. Uses PyTorch for deep learning, OpenCV for image processing, and MediaPipe for hand detection.

## Commands

### Setup
```bash
uv sync  # Creates venv with Python 3.11 (from .python-version) and installs dependencies
```

### Run Scripts
```bash
uv run collect_data.py                    # Collect training images via webcam (keys 1-5 save, q quits)
uv run train.py                           # Run all training experiments (augmentation + dataset size)
uv run evaluate.py                        # Evaluate default model
uv run evaluate.py models/model_*.pth     # Evaluate a specific model
uv run live_demo.py                       # Run real-time demo with default model
uv run live_demo.py models/model_*.pth    # Run real-time demo with specific model
```

No test suite exists - validation is done via `evaluate.py` metrics and `live_demo.py` visual testing.

## Architecture

### Model (model.py)
SimpleCNN with 3 conv layers (3→32→64→128 channels) + 2 dense layers. Input: 64x64 RGB. Output: 5 classes. ~2.19M parameters.

### Hand Detection
MediaPipe Hands detects hand landmarks, extracts bounding box, and crops the hand region with 20px padding. This isolates hand features from background.

### Data Pipeline
- `collect_data.py`: Detects hand via MediaPipe, crops region, saves to `dataset/{class_name}/`
- `dataset.py`: `GestureDataset` class loads images, applies augmentations at 50% probability, normalizes to [0,1], transposes to CHW format. Uses stratified 80/20 train/test split.
- `augmentations.py`: Custom OpenCV implementations (flip, rotate ±30°, brightness, blur, color jitter, zoom). No torchvision - all augmentations are hand-rolled.

### Training (train.py)
Runs two experiment sets:
1. **Augmentation experiments**: Tests 7 configs (none, each augmentation alone, flip+rotate, all combined)
2. **Dataset size experiments**: Tests 50/100/200/full samples with and without augmentation

Results saved to `results.json`, models to `models/model_{config}.pth`.

### Key Design Decisions
- Hand detection via MediaPipe ensures model learns gestures not backgrounds
- Augmentations applied probabilistically at runtime (not pre-generated)
- BGR→RGB conversion handled in preprocessing
- Device-agnostic (auto-detects CUDA)
- Fixed random seed (42) for reproducible splits
- MediaPipe pinned to 0.10.9 (newer versions removed solutions API)
