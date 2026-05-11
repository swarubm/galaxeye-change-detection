"""
eval.py
-------
Evaluation script for EO-SAR binary change detection.

Usage:
    # Evaluate on test split
    python eval.py --config config.yaml --weights checkpoints/best.pth --split test

    # Evaluate on val split
    python eval.py --config config.yaml --weights checkpoints/best.pth --split val

    # With custom data path
    python eval.py --config config.yaml --weights checkpoints/best.pth \\
                   --data_path /path/to/your/data --split test

Output:
    - Metrics printed to console (IoU, Precision, Recall, F1)
    - Confusion matrix saved to outputs/
    - Prediction visualisations saved to outputs/
    - metrics.txt with all numbers for the report
"""

import argparse
import os
import json

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from src import (
    EOSARChangeDataset, build_model, ChangeMetrics,
    load_config, get_device, load_checkpoint,
    visualise_predictions, plot_confusion_matrix,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(args):
    # ── Setup ──
    cfg    = load_config(args.config)
    device = get_device()

    # Override data root if provided
    if args.data_path:
        cfg["data"]["root"] = args.data_path

    # ── Dataset ──
    dataset = EOSARChangeDataset(
        root=cfg["data"]["root"],
        split=args.split,
        img_size=cfg["data"]["img_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # ── Model ──
    model = build_model(cfg).to(device)
    model, _, epoch, _ = load_checkpoint(args.weights, model, device=device)
    model.eval()

    # ── Evaluate ──
    metrics   = ChangeMetrics(threshold=cfg["eval"]["threshold"])
    all_images = []
    all_masks  = []
    all_logits = []
    all_names  = []

    print(f"\nEvaluating on {args.split} split...")
    for batch in tqdm(loader):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        names  = batch["name"]

        logits = model(images)
        metrics.update(logits, masks)

        # Collect for visualisation (keep on CPU)
        all_images.append(images.cpu())
        all_masks.append(masks.cpu())
        all_logits.append(logits.cpu())
        all_names.extend(names)

    # ── Results ──
    results = metrics.compute()
    cm      = metrics.confusion_matrix()

    print("\n" + "="*55)
    print(f"  RESULTS — {args.split.upper()} split  (from epoch {epoch})")
    print("="*55)
    print(f"  IoU       : {results['iou']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1 Score  : {results['f1']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print("-"*55)
    print(f"  TP: {results['tp']:,}  FP: {results['fp']:,}")
    print(f"  FN: {results['fn']:,}  TN: {results['tn']:,}")
    print("="*55)

    # ── Save outputs ──
    out_dir = os.path.join("outputs", args.split)
    os.makedirs(out_dir, exist_ok=True)

    # Confusion matrix plot
    plot_confusion_matrix(cm, out_dir, split=args.split)

    # Prediction visualisations (5 examples)
    all_images_cat = torch.cat(all_images, dim=0)
    all_masks_cat  = torch.cat(all_masks,  dim=0)
    all_logits_cat = torch.cat(all_logits, dim=0)

    visualise_predictions(
        images=all_images_cat,
        masks=all_masks_cat,
        logits=all_logits_cat,
        names=all_names,
        save_dir=os.path.join(out_dir, "predictions"),
        threshold=cfg["eval"]["threshold"],
        n=8,  # save 8 examples (include both success and failure cases)
    )

    # Save metrics as JSON (easy to copy into report)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"split": args.split, "epoch": epoch, **results}, f, indent=2)
    print(f"\n[Saved] Metrics → {metrics_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--weights",   type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split",     type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override data root in config")
    args = parser.parse_args()
    evaluate(args)
