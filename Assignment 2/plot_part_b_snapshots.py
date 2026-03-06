#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create heatmaps, surface plots, and optional animations from Part B snapshots.")
    parser.add_argument("--snapshot-dir", default="part_b_snapshots", help="Directory containing CSV snapshots.")
    parser.add_argument("--output-dir", default="part_b_plots", help="Directory for generated plots.")
    parser.add_argument("--animation", action="store_true", help="Generate MP4 animations when ffmpeg is available.")
    return parser.parse_args()


def load_snapshot(path):
    metadata = {}
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        if first_line.startswith("#"):
            parts = first_line[1:].split(",")
            for part in parts:
                key, value = part.strip().split("=", 1)
                metadata[key.strip()] = value.strip()
        else:
            handle.seek(0)

        reader = csv.reader(handle)
        for row in reader:
            if row:
                rows.append([float(value) for value in row])

    field = np.array(rows, dtype=np.float64)
    metadata["step"] = int(metadata.get("step", 0))
    metadata["length"] = float(metadata.get("length", 0.0))
    metadata["backend"] = metadata.get("backend", "unknown")
    return metadata, field


def save_heatmap(output_path, metadata, field):
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(field, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(f"{metadata['backend']} heatmap, L={metadata['length']:.2f}, step={metadata['step']}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(image, ax=ax, label="u")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_surface(output_path, metadata, field):
    ny, nx = field.shape
    x = np.linspace(0.0, metadata["length"], nx)
    y = np.linspace(0.0, metadata["length"], ny)
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, field, cmap="viridis", linewidth=0.0, antialiased=True)
    ax.set_title(f"{metadata['backend']} surface, L={metadata['length']:.2f}, step={metadata['step']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_animation(output_path, snapshots):
    fig, ax = plt.subplots(figsize=(7, 5))
    field0 = snapshots[0][1]
    image = ax.imshow(field0, origin="lower", cmap="viridis", aspect="auto")
    colorbar = fig.colorbar(image, ax=ax, label="u")
    colorbar.ax.set_ylabel("u")

    def update(frame_idx):
        metadata, field = snapshots[frame_idx]
        image.set_data(field)
        image.set_clim(vmin=np.min(field), vmax=np.max(field))
        ax.set_title(f"{metadata['backend']} animation, L={metadata['length']:.2f}, step={metadata['step']}")
        return [image]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=500, blit=False)
    try:
        writer = animation.FFMpegWriter(fps=2)
        ani.save(output_path, writer=writer, dpi=180)
    except Exception as exc:
        print(f"Skipping animation {output_path.name}: {exc}")
    plt.close(fig)


def main():
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for csv_path in sorted(snapshot_dir.glob("*.csv")):
        metadata, field = load_snapshot(csv_path)
        tag = f"{metadata['backend']}_L{metadata['length']:.2f}"
        grouped[tag].append((metadata, field))

        stem = csv_path.stem
        save_heatmap(output_dir / f"{stem}_heatmap.jpg", metadata, field)
        save_surface(output_dir / f"{stem}_surface.jpg", metadata, field)

    if args.animation:
        for tag, snapshots in grouped.items():
            snapshots.sort(key=lambda item: item[0]["step"])
            save_animation(output_dir / f"{tag}_animation.mp4", snapshots)


if __name__ == "__main__":
    main()
