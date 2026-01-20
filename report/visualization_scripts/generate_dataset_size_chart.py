#!/usr/bin/env python3
"""Generate chart showing dataset size vs accuracy with/without augmentation."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_FILE = PROJECT_ROOT / "results.json"
OUTPUT_FILE = SCRIPT_DIR.parent / "images" / "dataset_size_effect.png"

# Colors
UIBK_BLUE = "#003C78"
ORANGE = "#E87722"

def main():
    # Load results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    size_results = results["dataset_size_experiments"]

    # Extract data
    sizes = ["50", "100", "200", "Full\n(~300)"]
    no_aug = [
        size_results["50_no_aug"]["final_test_acc"] * 100,
        size_results["100_no_aug"]["final_test_acc"] * 100,
        size_results["200_no_aug"]["final_test_acc"] * 100,
        size_results["full_no_aug"]["final_test_acc"] * 100,
    ]
    with_aug = [
        size_results["50_with_aug"]["final_test_acc"] * 100,
        size_results["100_with_aug"]["final_test_acc"] * 100,
        size_results["200_with_aug"]["final_test_acc"] * 100,
        size_results["full_with_aug"]["final_test_acc"] * 100,
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_aug, width, label="Without Augmentation",
                   color=UIBK_BLUE, edgecolor="white")
    bars2 = ax.bar(x + width/2, with_aug, width, label="With Augmentation",
                   color=ORANGE, edgecolor="white")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Highlight the improvement at 50 samples
    ax.annotate("", xy=(x[0] + width/2, with_aug[0]),
                xytext=(x[0] - width/2, no_aug[0]),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(x[0], (no_aug[0] + with_aug[0])/2, "+8%", fontsize=11,
            fontweight="bold", color="green", ha="center")

    # Styling
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Samples per Class", fontsize=12, fontweight="bold")
    ax.set_title("Impact of Dataset Size and Augmentation", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_ylim(85, 105)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(loc="lower right", fontsize=11)

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
