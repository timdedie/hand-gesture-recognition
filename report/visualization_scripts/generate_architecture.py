#!/usr/bin/env python3
"""Generate CNN architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILE = SCRIPT_DIR.parent / "images" / "architecture.png"

# Colors
CONV_COLOR = "#003C78"  # UIBK Blue
POOL_COLOR = "#4A90A4"  # Lighter blue
FC_COLOR = "#00A651"    # Green
INPUT_COLOR = "#E87722" # Orange
OUTPUT_COLOR = "#C41E3A" # Red

def draw_layer_block(ax, x, y, width, height, depth, color, label, sublabel=""):
    """Draw a 3D-ish layer block."""
    # Main rectangle
    rect = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.01,rounding_size=0.05",
                          facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)

    # 3D effect (depth lines)
    if depth > 0:
        offset = min(depth * 0.03, 0.15)
        # Top edge
        ax.plot([x, x + offset], [y + height, y + height + offset], color="black", linewidth=1)
        ax.plot([x + width, x + width + offset], [y + height, y + height + offset], color="black", linewidth=1)
        ax.plot([x + offset, x + width + offset], [y + height + offset, y + height + offset], color="black", linewidth=1)
        # Side edge
        ax.plot([x + width, x + width + offset], [y, y + offset], color="black", linewidth=1)
        ax.plot([x + width + offset, x + width + offset], [y + offset, y + height + offset], color="black", linewidth=1)
        # Fill 3D parts with lighter color
        ax.fill([x, x + offset, x + width + offset, x + width],
                [y + height, y + height + offset, y + height + offset, y + height],
                color=color, alpha=0.6)
        ax.fill([x + width, x + width + offset, x + width + offset, x + width],
                [y, y + offset, y + height + offset, y + height],
                color=color, alpha=0.4)

    # Label
    ax.text(x + width/2, y + height/2 + 0.05, label,
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    if sublabel:
        ax.text(x + width/2, y + height/2 - 0.15, sublabel,
                ha="center", va="center", fontsize=7, color="white")

def draw_arrow(ax, start, end):
    """Draw an arrow between two points."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

def main():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    # Layer specifications: (x, width, height, depth, color, label, sublabel, dimension_label)
    layers = [
        # Input
        (0, 0.5, 2.0, 3, INPUT_COLOR, "Input", "64x64", "3"),
        # Conv1 + Pool1
        (1.5, 0.5, 1.8, 32, CONV_COLOR, "Conv1", "3x3", "32"),
        (2.2, 0.5, 1.4, 32, POOL_COLOR, "Pool", "2x2", ""),
        # Conv2 + Pool2
        (3.5, 0.5, 1.4, 64, CONV_COLOR, "Conv2", "3x3", "64"),
        (4.2, 0.5, 1.0, 64, POOL_COLOR, "Pool", "2x2", ""),
        # Conv3 + Pool3
        (5.5, 0.5, 1.0, 128, CONV_COLOR, "Conv3", "3x3", "128"),
        (6.2, 0.5, 0.6, 128, POOL_COLOR, "Pool", "2x2", ""),
        # Flatten indicator
        (7.5, 0.3, 2.5, 0, "#888888", "Flat", "", "8192"),
        # FC1
        (8.8, 0.4, 1.5, 0, FC_COLOR, "FC1", "+Drop", "256"),
        # FC2 (Output)
        (10.2, 0.4, 0.8, 0, OUTPUT_COLOR, "FC2", "", "5"),
    ]

    y_base = 1.0

    # Draw layers
    for i, (x, w, h, d, color, label, sublabel, dim) in enumerate(layers):
        draw_layer_block(ax, x, y_base, w, h, d/10 if d > 0 else 0, color, label, sublabel)

        # Dimension labels below
        if dim:
            ax.text(x + w/2, y_base - 0.3, dim, ha="center", va="top", fontsize=8, color="gray")

        # Draw arrows between layers
        if i > 0:
            prev_x, prev_w, prev_h = layers[i-1][0], layers[i-1][1], layers[i-1][2]
            draw_arrow(ax, (prev_x + prev_w + 0.15, y_base + prev_h/2), (x - 0.1, y_base + h/2))

    # Output arrow and label
    last_x, last_w, last_h = layers[-1][0], layers[-1][1], layers[-1][2]
    draw_arrow(ax, (last_x + last_w + 0.1, y_base + last_h/2), (11.5, y_base + last_h/2))

    # Output classes
    classes = ["Thumbs Up", "Thumbs Down", "Peace", "Open Palm", "No Hand"]
    for i, cls in enumerate(classes):
        y_pos = y_base + last_h/2 + 0.8 - i * 0.35
        ax.text(12.0, y_pos, cls, fontsize=9, va="center")
        ax.plot([11.6, 11.9], [y_base + last_h/2, y_pos], color="gray", linewidth=0.5, alpha=0.5)

    # Annotations
    ax.text(0.25, 3.5, "64x64x3", ha="center", fontsize=8, color="gray")
    ax.text(2.45, 3.3, "32x32x32", ha="center", fontsize=8, color="gray")
    ax.text(4.45, 2.7, "16x16x64", ha="center", fontsize=8, color="gray")
    ax.text(6.45, 2.3, "8x8x128", ha="center", fontsize=8, color="gray")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=INPUT_COLOR, label="Input"),
        mpatches.Patch(facecolor=CONV_COLOR, label="Conv + ReLU"),
        mpatches.Patch(facecolor=POOL_COLOR, label="MaxPool"),
        mpatches.Patch(facecolor=FC_COLOR, label="FC + Dropout"),
        mpatches.Patch(facecolor=OUTPUT_COLOR, label="Output"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Title
    ax.set_title("SimpleCNN Architecture (~2.19M parameters)", fontsize=14, fontweight="bold", pad=15)

    # Parameter count annotation
    ax.text(7.5, -0.2, "Total: ~2.19M parameters", ha="center", fontsize=10, style="italic", color="gray")

    plt.tight_layout()

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
