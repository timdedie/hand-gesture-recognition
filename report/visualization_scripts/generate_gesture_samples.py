#!/usr/bin/env python3
"""Generate grid of gesture sample images."""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_FILE = SCRIPT_DIR.parent / "images" / "gesture_samples.png"

# Gesture classes with display names
GESTURES = [
    ("thumbs_up", "Thumbs Up"),
    ("thumbs_down", "Thumbs Down"),
    ("peace", "Peace"),
    ("open_palm", "Open Palm"),
    ("no_hand", "No Hand"),
]

def main():
    # 2 rows x 3 columns layout (5 gestures + 1 empty cell)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()

    for i, (folder_name, display_name) in enumerate(GESTURES):
        ax = axes[i]
        # Find first image in the folder
        folder = DATASET_DIR / folder_name
        images = sorted(folder.glob("*.jpg"))

        if images:
            # Load and convert BGR to RGB
            img = cv2.imread(str(images[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img)
            ax.set_title(display_name, fontsize=14, fontweight="bold", pad=10)
        else:
            ax.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(display_name, fontsize=14, fontweight="bold", pad=10)

        ax.axis("off")

    # Hide the empty 6th cell
    axes[5].axis("off")

    plt.suptitle("Gesture Classes", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
