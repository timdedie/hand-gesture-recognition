#!/usr/bin/env python3
"""Generate bar chart comparing augmentation strategies."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_FILE = PROJECT_ROOT / "results.json"
OUTPUT_FILE = SCRIPT_DIR.parent / "images" / "augmentation_comparison.png"

# University of Innsbruck blue
UIBK_BLUE = "#003C78"
HIGHLIGHT_COLOR = "#00A651"  # Green for best result

def main():
    # Load results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    aug_results = results["augmentation_experiments"]

    # Extract data
    configs = {
        "No Augmentation": aug_results["no_augmentation"]["final_test_acc"] * 100,
        "Flip Only": aug_results["flip_only"]["final_test_acc"] * 100,
        "Rotate Only": aug_results["rotate_only"]["final_test_acc"] * 100,
        "Brightness Only": aug_results["brightness_only"]["final_test_acc"] * 100,
        "Blur Only": aug_results["blur_only"]["final_test_acc"] * 100,
        "Flip + Rotate": aug_results["flip_rotate"]["final_test_acc"] * 100,
        "All Augmentations": aug_results["all_augmentations"]["final_test_acc"] * 100,
    }

    # Sort by accuracy
    sorted_configs = dict(sorted(configs.items(), key=lambda x: x[1]))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(sorted_configs.keys())
    values = list(sorted_configs.values())

    # Color bars - highlight the best one
    colors = [HIGHLIGHT_COLOR if v == max(values) else UIBK_BLUE for v in values]

    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=11, fontweight="bold")

    # Styling
    ax.set_xlabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Augmentation Strategy Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(94, 102)
    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
