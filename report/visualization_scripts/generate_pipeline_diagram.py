#!/usr/bin/env python3
"""Generate end-to-end processing pipeline diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILE = SCRIPT_DIR.parent / "images" / "pipeline.png"

# Colors
UIBK_BLUE = "#003C78"
LIGHT_BLUE = "#E8F4FC"
ORANGE = "#E87722"
GREEN = "#00A651"

def draw_box(ax, x, y, width, height, text, color=UIBK_BLUE, text_color="white"):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor="black", linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=text_color, wrap=True)

def draw_arrow(ax, start, end):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle="-|>",
                            mutation_scale=20,
                            color="gray",
                            linewidth=2)
    ax.add_patch(arrow)

def main():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Box dimensions
    box_width = 2.0
    box_height = 1.2
    y_center = 2.4

    # Pipeline stages
    stages = [
        (0.5, "Webcam\nInput", UIBK_BLUE),
        (3.0, "MediaPipe\nHand Detection", ORANGE),
        (5.5, "Crop +\n20px Padding", ORANGE),
        (8.0, "Resize\n64x64", LIGHT_BLUE),
        (10.5, "Normalize\n[0,1]", LIGHT_BLUE),
        (13.0, "CNN\nClassifier", GREEN),
    ]

    # Draw boxes
    for x, text, color in stages:
        text_color = "black" if color == LIGHT_BLUE else "white"
        draw_box(ax, x, y_center - box_height/2, box_width, box_height, text, color, text_color)

    # Draw arrows between boxes
    for i in range(len(stages) - 1):
        start_x = stages[i][0] + box_width
        end_x = stages[i+1][0]
        draw_arrow(ax, (start_x + 0.1, y_center), (end_x - 0.1, y_center))

    # Output label
    ax.text(15.5, y_center, "Gesture\nClass", ha="center", va="center",
            fontsize=12, fontweight="bold", color=UIBK_BLUE)
    draw_arrow(ax, (15.0 + 0.1, y_center), (15.3, y_center))

    # Section labels
    ax.text(1.5, 4.2, "Input", ha="center", fontsize=10, color="gray", style="italic")
    ax.text(4.75, 4.2, "Hand Detection", ha="center", fontsize=10, color="gray", style="italic")
    ax.text(9.75, 4.2, "Preprocessing", ha="center", fontsize=10, color="gray", style="italic")
    ax.text(14.0, 4.2, "Classification", ha="center", fontsize=10, color="gray", style="italic")

    # Title
    ax.set_title("Hand Gesture Recognition Pipeline", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
