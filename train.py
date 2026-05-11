"""
train.py
--------
Training script for EO-SAR binary change detection.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/best.pth

What this script does:
    1. Loads config, sets seed, detects device
    2. Builds dataset, model, loss, optimizer, scheduler
    3. Trains with per-epoch validation
    4. Saves best checkpoint (by val F1) and latest checkpoint
    5. Plots training curves at the end
"""

import argparse
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from src import (
    get_dataloaders, build_model, build_loss, ChangeMetrics,
    set_seed, load_config, get_device,
    save_checkpoint, load_checkpoint,
    plot_training_curves,
)


# ─── Training loop (one epoch) ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─── Validation loop ────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5) -> tuple:
    model.eval()
    metrics = ChangeMetrics(threshold=threshold)
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()
        metrics.update(logits, masks)

    return total_loss / len(loader), metrics.compute()


# ─── Main ───────────────────────────────────────────────────────────────────

def main(args):
    cfg    = load_config(args.config)
    seed   = cfg["seed"]
    set_seed(seed)
    device = get_device()

    # ── Data ──
    print("\n[1/4] Building dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── Model ──
    print("\n[2/4] Building model...")
    model = build_model(cfg).to(device)

    # ── Loss + Optimiser ──
    print("\n[3/4] Setting up loss and optimiser...")
    criterion = build_loss(cfg)

    if cfg["train"]["optimizer"] == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"]["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg['train']['optimizer']}")

    epochs = cfg["train"]["epochs"]

    if cfg["train"]["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs - cfg["train"]["warmup_epochs"],
            eta_min=1e-6,
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )

    # ── Resume ──
    start_epoch = 0
    best_f1     = 0.0
    if args.resume and os.path.isfile(args.resume):
        model, optimizer, start_epoch, best_f1 = load_checkpoint(
            args.resume, model, optimizer, device
        )

    # ── Training ──
    print(f"\n[4/4] Training for {epochs} epochs...\n")
    train_losses, val_f1s = [], []
    patience_counter = 0
    patience = cfg["train"]["early_stopping_patience"]

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Warmup LR
        if epoch < cfg["train"]["warmup_epochs"]:
            warmup_factor = (epoch + 1) / cfg["train"]["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["train"]["learning_rate"] * warmup_factor

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device,
            threshold=cfg["eval"]["threshold"],
        )

        # Scheduler step
        if cfg["train"]["scheduler"] == "cosine":
            if epoch >= cfg["train"]["warmup_epochs"]:
                scheduler.step()
        else:
            scheduler.step(val_metrics["f1"])

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # ── Logging ──
        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f} | "
            f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
            f"LR: {lr_now:.2e} | {elapsed:.1f}s"
        )

        train_losses.append(train_loss)
        val_f1s.append(val_metrics["f1"])

        # ── Save best checkpoint ──
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "val_metrics": val_metrics,
                    "cfg": cfg,
                },
                cfg["train"]["save_dir"],
                "best.pth",
            )
            print(f"  ★ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        # ── Save periodic checkpoint ──
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            save_checkpoint(
                {"epoch": epoch+1, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "best_f1": best_f1, "cfg": cfg},
                cfg["train"]["save_dir"],
                f"epoch_{epoch+1:03d}.pth",
            )

        # ── Early stopping ──
        if patience_counter >= patience:
            print(f"\n[Early stopping] No improvement for {patience} epochs.")
            break

    # ── Plot training curves ──
    plot_training_curves(train_losses, val_f1s, cfg["train"]["save_dir"])
    print(f"\n[Done] Best validation F1 = {best_f1:.4f}")
    print(f"       Best checkpoint → {cfg['train']['save_dir']}/best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args)
