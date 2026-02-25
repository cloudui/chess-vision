import argparse
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ChessDataset, NUM_CLASSES, NUM_SQUARES
from model import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]):
    """Apply dot-notation overrides like 'training.epochs=10'."""
    for item in overrides:
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # Try to cast to the original type
        orig = d[keys[-1]]
        if orig is None:
            # Try int, then float, then keep as string
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
        elif isinstance(orig, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(orig, int):
            value = int(value)
        elif isinstance(orig, float):
            value = float(value)
        d[keys[-1]] = value


def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup_epochs = cfg["scheduler"]["warmup_epochs"]
    total_epochs = cfg["training"]["epochs"]
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler.last_epoch = -1  # suppress false warning about step() order
    return scheduler


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    correct_squares = 0
    correct_boards = 0
    total_squares = 0
    total_boards = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images).view(-1, NUM_SQUARES, NUM_CLASSES)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct_squares += (preds == labels).sum().item()
        correct_boards += (preds == labels).all(dim=1).sum().item()
        total_squares += labels.numel()
        total_boards += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = total_boards
    return {
        "loss": total_loss / n,
        "square_acc": correct_squares / total_squares,
        "board_acc": correct_boards / total_boards,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct_squares = 0
    correct_boards = 0
    total_squares = 0
    total_boards = 0

    pbar = tqdm(loader, desc="  val  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images).view(-1, NUM_SQUARES, NUM_CLASSES)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct_squares += (preds == labels).sum().item()
        correct_boards += (preds == labels).all(dim=1).sum().item()
        total_squares += labels.numel()
        total_boards += batch_size

    n = total_boards
    return {
        "loss": total_loss / n,
        "square_acc": correct_squares / total_squares,
        "board_acc": correct_boards / total_boards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chess ViT")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset-schedule", action="store_true",
                        help="Reset optimizer, scheduler, and epoch counter when resuming (warm restart)")
    parser.add_argument("--set", nargs="*", default=[], help="Override config values, e.g. training.epochs=10")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)

    device = get_device()
    print(f"Device: {device}")

    use_amp = cfg["training"]["mixed_precision"]
    # MPS supports autocast but not GradScaler — use autocast only on MPS
    use_scaler = use_amp and device.type == "cuda"

    # --- Data ---
    model_name = cfg["model"]["name"]
    max_samples = cfg["data"]["max_samples"]

    full_dataset = ChessDataset(
        cfg["data"]["train_dir"],
        model_name=model_name,
        max_samples=max_samples,
        is_training=True,
    )
    val_size = int(len(full_dataset) * cfg["data"]["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    num_workers = cfg["data"]["num_workers"]
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # --- Model ---
    model = build_model(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # --- Optimizer / Scheduler / Scaler ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    criterion = nn.CrossEntropyLoss()

    # --- Resume ---
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if args.reset_schedule:
            # Warm restart: keep model weights, reset everything else
            print(f"Loaded weights from {args.resume}, reset schedule (warm restart)")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            print(f"Resumed from epoch {start_epoch}")

    # --- Logging ---
    os.makedirs(cfg["logging"]["tensorboard_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["logging"]["tensorboard_dir"])

    # --- Checkpointing ---
    save_dir = cfg["checkpointing"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    patience = cfg["checkpointing"]["early_stopping_patience"]
    epochs_without_improvement = 0

    # --- Training loop ---
    epochs = cfg["training"]["epochs"]
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device, use_amp
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Train — loss: {train_metrics['loss']:.4f}, "
            f"sq_acc: {train_metrics['square_acc']:.4f}, "
            f"board_acc: {train_metrics['board_acc']:.4f}"
        )
        print(
            f"  Val   — loss: {val_metrics['loss']:.4f}, "
            f"sq_acc: {val_metrics['square_acc']:.4f}, "
            f"board_acc: {val_metrics['board_acc']:.4f}"
        )
        print(f"  LR: {lr:.2e} | Time: {elapsed:.1f}s")

        # TensorBoard
        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/square_acc", train_metrics["square_acc"], epoch)
        writer.add_scalar("train/board_acc", train_metrics["board_acc"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/square_acc", val_metrics["square_acc"], epoch)
        writer.add_scalar("val/board_acc", val_metrics["board_acc"], epoch)
        writer.add_scalar("lr", lr, epoch)

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(save_dir, "latest.pth"))

        if val_metrics["square_acc"] > best_val_acc:
            best_val_acc = val_metrics["square_acc"]
            torch.save(ckpt, os.path.join(save_dir, "best.pth"))
            print(f"  >> New best val square_acc: {best_val_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    writer.close()
    print(f"\nTraining complete. Best val square_acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to {save_dir}/")
