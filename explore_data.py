"""
explore_data.py
---------------
Run this FIRST before any training to understand your dataset.
Produces plots and prints statistics that you need for your report.

Usage:
    python explore_data.py --config config.yaml

Output:
    outputs/eda/
        class_distribution.png   — change vs no-change pixel counts
        sample_grid.png          — example EO/SAR image pairs
        dataset_stats.txt        — numbers for your report
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import remap_mask
from src.utils   import load_config


def analyse_class_distribution(data_root: str, splits=("train", "val", "test")):
    """Count change vs no-change pixels across all splits."""
    print("\n[EDA] Analysing class distribution...")
    stats = {}

    for split in splits:
        mask_dir = Path(data_root) / split / "masks"
        if not mask_dir.exists():
            print(f"  [skip] {split} masks not found at {mask_dir}")
            continue

        total_pixels  = 0
        change_pixels = 0
        n_samples     = 0

        for p in tqdm(list(mask_dir.iterdir()), desc=f"  {split}"):
            if p.suffix.lower() not in {".png", ".tif", ".tiff"}:
                continue
            mask = np.array(Image.open(p))
            mask = remap_mask(mask)
            total_pixels  += mask.size
            change_pixels += (mask == 1).sum()
            n_samples     += 1

        if total_pixels == 0:
            continue
        no_change_pct = 100 * (total_pixels - change_pixels) / total_pixels
        change_pct    = 100 * change_pixels / total_pixels
        stats[split]  = {
            "n_samples": n_samples,
            "total_pixels": total_pixels,
            "change_pixels": int(change_pixels),
            "no_change_pct": round(no_change_pct, 2),
            "change_pct": round(change_pct, 2),
        }
        print(f"  {split}: {n_samples} images | "
              f"Change: {change_pct:.2f}% | No-change: {no_change_pct:.2f}%")

    return stats


def plot_class_distribution(stats: dict, out_dir: str):
    """Bar chart of class distribution per split."""
    os.makedirs(out_dir, exist_ok=True)
    splits = list(stats.keys())
    change    = [stats[s]["change_pct"]    for s in splits]
    no_change = [stats[s]["no_change_pct"] for s in splits]

    x = np.arange(len(splits))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, no_change, w, label="No-change (0)", color="#2ecc71")
    ax.bar(x + w/2, change,    w, label="Change (1)",    color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("% of pixels"); ax.set_title("Class distribution per split")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (nc, c) in enumerate(zip(no_change, change)):
        ax.text(i - w/2, nc + 0.5, f"{nc:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, c  + 0.5, f"{c:.1f}%",  ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, dpi=100); plt.close()
    print(f"[EDA] Saved → {path}")


def plot_sample_grid(data_root: str, out_dir: str, n_samples: int = 4):
    """Show n_samples rows of: Pre EO | Post EO | Pre SAR | Post SAR | Mask."""
    os.makedirs(out_dir, exist_ok=True)
    split     = "train"
    pre_eo_dir  = Path(data_root) / split / "pre_eo"
    pre_sar_dir = Path(data_root) / split / "pre_sar"
    post_eo_dir = Path(data_root) / split / "post_eo"
    post_sar_dir= Path(data_root) / split / "post_sar"
    mask_dir    = Path(data_root) / split / "masks"

    if not pre_eo_dir.exists():
        print("[EDA] Cannot plot sample grid — data not found"); return

    files = sorted(list(pre_eo_dir.iterdir()))[:n_samples]
    if not files:
        print("[EDA] No files found for sample grid"); return

    fig, axes = plt.subplots(n_samples, 5, figsize=(16, 3.5 * n_samples))
    if n_samples == 1:
        axes = [axes]
    cols = ["Pre EO", "Post EO", "Pre SAR", "Post SAR", "Change mask"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=11, fontweight="bold")

    for row, f in enumerate(files):
        name = f.stem

        def load(folder):
            for ext in [".png", ".tif", ".tiff"]:
                p = folder / (name + ext)
                if p.exists():
                    return np.array(Image.open(p))
            return None

        pre_eo  = load(pre_eo_dir)
        post_eo = load(post_eo_dir)
        pre_sar = load(pre_sar_dir)
        post_sar= load(post_sar_dir)
        mask    = load(mask_dir)
        if mask is not None:
            mask = remap_mask(mask)

        for col_idx, (img, cmap) in enumerate([
            (pre_eo, None), (post_eo, None),
            (pre_sar, "gray"), (post_sar, "gray"),
            (mask, "RdYlGn_r"),
        ]):
            axes[row][col_idx].imshow(img, cmap=cmap)
            axes[row][col_idx].axis("off")
        axes[row][0].set_ylabel(name, fontsize=7, rotation=0, labelpad=60)

    plt.tight_layout()
    path = os.path.join(out_dir, "sample_grid.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"[EDA] Saved → {path}")


def main(args):
    cfg = load_config(args.config)
    out_dir = "outputs/eda"

    # 1. Class distribution
    stats = analyse_class_distribution(cfg["data"]["root"])
    if stats:
        plot_class_distribution(stats, out_dir)

        # Save text summary
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "dataset_stats.txt"), "w") as f:
            f.write("Dataset Statistics\n" + "="*40 + "\n")
            for split, s in stats.items():
                f.write(f"\n{split.upper()}\n")
                for k, v in s.items():
                    f.write(f"  {k}: {v}\n")
        print(f"[EDA] Stats saved → {out_dir}/dataset_stats.txt")

    # 2. Sample grid
    plot_sample_grid(cfg["data"]["root"], out_dir)

    print("\n[EDA] Done! Check outputs/eda/ for plots.")
    print("      Use these statistics in your report's Data Analysis section.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args)
