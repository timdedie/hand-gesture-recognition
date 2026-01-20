#!/usr/bin/env python3
"""Generate training curves comparing different configurations."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_FILE = PROJECT_ROOT / "results.json"
OUTPUT_FILE = SCRIPT_DIR.parent / "images" / "training_curves.png"

# Colors
COLORS = {
    "no_augmentation": "#003C78",  # UIBK Blue
    "flip_rotate": "#00A651",      # Green (best)
    "all_augmentations": "#E87722", # Orange
}

LABELS = {
    "no_augmentation": "No Augmentation",
    "flip_rotate": "Flip + Rotate",
    "all_augmentations": "All Augmentations",
}

def main():
    # Load results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    aug_results = results["augmentation_experiments"]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = np.arange(1, 21)

    configs_to_plot = ["no_augmentation", "flip_rotate", "all_augmentations"]

    # Plot each metric
    metrics = [
        ("train_loss", "Training Loss", axes[0, 0]),
        ("train_acc", "Training Accuracy", axes[0, 1]),
        ("test_loss", "Test Loss", axes[1, 0]),
        ("test_acc", "Test Accuracy", axes[1, 1]),
    ]

    for metric_key, metric_name, ax in metrics:
        for config in configs_to_plot:
            history = aug_results[config]["history"]
            values = history[metric_key]

            # Convert accuracy to percentage
            if "acc" in metric_key:
                values = [v * 100 for v in values]

            ax.plot(epochs, values, label=LABELS[config],
                   color=COLORS[config], linewidth=2, marker="o", markersize=3)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set y-axis limits for accuracy plots
        if "acc" in metric_key:
            ax.set_ylim(30, 105)
            ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Training Dynamics Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
