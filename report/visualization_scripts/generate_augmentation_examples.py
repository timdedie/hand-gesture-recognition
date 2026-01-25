#!/usr/bin/env python3
"""Generate visual examples of each augmentation type."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_FILE = SCRIPT_DIR.parent / "images" / "augmentation_examples.png"

# Add project root to path for augmentations import
sys.path.insert(0, str(PROJECT_ROOT))
from augmentations import horizontal_flip, rotate, adjust_brightness, gaussian_blur, color_jitter, zoom

def main():
    # Find a sample image
    sample_folder = DATASET_DIR / "thumbs_up"
    images = sorted(sample_folder.glob("*.jpg"))

    if not images:
        print("No sample images found!")
        return

    # Load original image
    original = cv2.imread(str(images[0]))

    # Apply each augmentation with fixed parameters for reproducibility
    augmentations = [
        ("Original", original),
        ("Flip", horizontal_flip(original)),
        ("Rotate", rotate(original, angle=25)),
        ("Brightness", adjust_brightness(original, factor=1.4)),
        ("Blur", gaussian_blur(original, kernel_size=7)),
        ("Color Jitter", color_jitter(original)),
        ("Zoom", zoom(original, factor=1.25)),
    ]

    # Create figure with 2 rows x 4 columns (7 augmentations + 1 empty cell)
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    axes = axes.flatten()

    for i, (name, img) in enumerate(augmentations):
        ax = axes[i]
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(name, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    # Hide the empty 8th cell
    axes[7].axis("off")

    plt.suptitle("Data Augmentation Examples", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
