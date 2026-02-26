import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ChessDataset, NUM_CLASSES, NUM_SQUARES, fen_to_labels
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


def compute_class_weights(dataset, device):
    """Compute inverse-sqrt-frequency class weights from manifest FENs."""
    counts = torch.zeros(NUM_CLASSES)
    for sample in dataset.samples:
        fen = sample.get("fen")
        if fen:
            labels = fen_to_labels(fen.split()[0])
            counts += torch.bincount(labels, minlength=NUM_CLASSES).float()
    if counts.sum() == 0:
        return None
    freq = counts / counts.sum()
    weights = 1.0 / freq.clamp(min=1e-6).sqrt()
    weights /= weights.mean()  # normalize so mean weight = 1
    return weights.to(device)


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
# Train / Validate (shared)
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, use_amp, cfg, training=False,
              optimizer=None, scheduler=None, scaler=None, class_weights=None,
              writer=None, global_step=0):
    """Run one epoch of training or validation.

    When *training* is True, performs optimizer/scheduler/scaler steps.
    All training data is assumed legal, so turn and castling losses are
    computed on every sample without masking.

    Returns (metrics_dict, updated_global_step).
    """
    model.train(training)

    piece_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
    )
    turn_criterion = nn.BCEWithLogitsLoss()
    castling_criterion = nn.BCEWithLogitsLoss()

    turn_w = cfg["training"].get("turn_loss_weight", 1.0)
    castling_w = cfg["training"].get("castling_loss_weight", 1.0)

    total_loss = 0.0
    correct_squares = 0
    correct_boards = 0
    total_squares = 0
    total_boards = 0
    correct_turn = 0
    correct_castling_rights = 0
    correct_castling_all = 0
    correct_full_fen = 0

    desc = "  train" if training else "  val  "
    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.set_grad_enabled(training):
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            sq_labels = labels["squares"].to(device, non_blocking=True)
            turn_labels = labels["turn"].to(device, non_blocking=True)
            castling_labels = labels["castling"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                sq_logits = outputs["squares"].view(-1, NUM_SQUARES, NUM_CLASSES)
                piece_loss = piece_criterion(
                    sq_logits.reshape(-1, NUM_CLASSES), sq_labels.reshape(-1)
                )
                turn_loss = turn_criterion(outputs["turn"], turn_labels)
                castling_loss = castling_criterion(outputs["castling"], castling_labels)
                loss = piece_loss + turn_w * turn_loss + castling_w * castling_loss

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["grad_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Per-step TensorBoard logging
                if writer is not None:
                    writer.add_scalar("step/loss", loss.item(), global_step)
                    writer.add_scalar("step/piece_loss", piece_loss.item(), global_step)
                    writer.add_scalar("step/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            # Piece accuracy
            preds = sq_logits.argmax(dim=-1)
            sq_correct = preds == sq_labels
            correct_squares += sq_correct.sum().item()
            board_correct = sq_correct.all(dim=1)
            correct_boards += board_correct.sum().item()
            total_squares += sq_labels.numel()
            total_boards += batch_size

            # Turn accuracy
            turn_preds = (outputs["turn"].detach() > 0).float()
            turn_correct = (turn_preds == turn_labels).squeeze(1)
            correct_turn += turn_correct.sum().item()

            # Castling accuracy
            castling_preds = (outputs["castling"].detach() > 0).float()
            castling_right_correct = castling_preds == castling_labels
            correct_castling_rights += castling_right_correct.sum().item()
            castling_all_correct = castling_right_correct.all(dim=1)
            correct_castling_all += castling_all_correct.sum().item()

            # Full FEN accuracy
            full_correct = board_correct & turn_correct & castling_all_correct
            correct_full_fen += full_correct.sum().item()

            if training:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = total_boards
    metrics = {
        "loss": total_loss / n,
        "square_acc": correct_squares / total_squares,
        "board_acc": correct_boards / n,
        "turn_acc": correct_turn / n,
        "castling_right_acc": correct_castling_rights / (n * 4),
        "castling_acc": correct_castling_all / n,
        "full_fen_acc": correct_full_fen / n,
    }
    return metrics, global_step


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
    input_size = cfg["model"].get("input_size")

    # Create two datasets: one with training augmentation, one clean for validation.
    # Both share the same underlying data and use a seeded split for reproducibility.
    train_full = ChessDataset(
        cfg["data"]["train_dir"],
        model_name=model_name,
        max_samples=max_samples,
        is_training=True,
        input_size=input_size,
    )
    val_full = ChessDataset(
        cfg["data"]["train_dir"],
        model_name=model_name,
        max_samples=max_samples,
        is_training=False,
        input_size=input_size,
    )
    val_size = int(len(train_full) * cfg["data"]["val_split"])
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(
        train_full, [train_size, val_size], generator=generator
    )
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        val_full, [train_size, val_size], generator=generator
    )

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

    # --- Class weights ---
    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        class_weights = compute_class_weights(train_full, device)
        print(f"Class weights: {class_weights}")

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
    from datetime import datetime
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(cfg["logging"]["tensorboard_dir"], run_name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # --- Checkpointing ---
    save_dir = cfg["checkpointing"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    patience = cfg["checkpointing"]["early_stopping_patience"]
    epochs_without_improvement = 0

    # --- Run metadata ---
    def get_git_info():
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip())
            return git_hash, git_dirty
        except Exception:
            return None, None

    git_hash, git_dirty = get_git_info()
    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "command": sys.argv,
        "config": cfg,
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "device": str(device),
        "train_size": train_size,
        "val_size": val_size,
        "tb_dir": tb_dir,
    }
    meta_path = os.path.join(save_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run metadata: {meta_path}")

    # --- Training loop ---
    epochs = cfg["training"]["epochs"]
    global_step = 0
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        t0 = time.time()

        train_metrics, global_step = run_epoch(
            model, train_loader, device, use_amp, cfg,
            training=True, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            class_weights=class_weights, writer=writer, global_step=global_step,
        )
        val_metrics, _ = run_epoch(
            model, val_loader, device, use_amp, cfg,
            class_weights=class_weights,
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Train — loss: {train_metrics['loss']:.4f}, "
            f"sq_acc: {train_metrics['square_acc']:.4f}, "
            f"board_acc: {train_metrics['board_acc']:.4f}, "
            f"turn: {train_metrics['turn_acc']:.4f}, "
            f"castling: {train_metrics['castling_acc']:.4f}, "
            f"full_fen: {train_metrics['full_fen_acc']:.4f}"
        )
        print(
            f"  Val   — loss: {val_metrics['loss']:.4f}, "
            f"sq_acc: {val_metrics['square_acc']:.4f}, "
            f"board_acc: {val_metrics['board_acc']:.4f}, "
            f"turn: {val_metrics['turn_acc']:.4f}, "
            f"castling: {val_metrics['castling_acc']:.4f}, "
            f"full_fen: {val_metrics['full_fen_acc']:.4f}"
        )
        print(f"  LR: {lr:.2e} | Time: {elapsed:.1f}s")

        # TensorBoard — epoch-level metrics grouped for comparison
        for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
            writer.add_scalar(f"loss/{prefix}", metrics["loss"], epoch)
            writer.add_scalar(f"accuracy/board_{prefix}", metrics["board_acc"], epoch)
            writer.add_scalar(f"accuracy/square_{prefix}", metrics["square_acc"], epoch)
            writer.add_scalar(f"accuracy/turn_{prefix}", metrics["turn_acc"], epoch)
            writer.add_scalar(f"accuracy/castling_{prefix}", metrics["castling_acc"], epoch)
            writer.add_scalar(f"accuracy/full_fen_{prefix}", metrics["full_fen_acc"], epoch)

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

        if val_metrics["board_acc"] > best_val_acc:
            best_val_acc = val_metrics["board_acc"]
            torch.save(ckpt, os.path.join(save_dir, "best.pth"))
            print(f"  >> New best val board_acc: {best_val_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    writer.close()

    # Shut down DataLoader workers promptly (avoids slow exit with persistent_workers)
    del train_loader, val_loader

    # Update run metadata with final results
    run_meta["best_val_acc"] = best_val_acc
    run_meta["total_epochs"] = epoch + 1
    run_meta["final_train_metrics"] = train_metrics
    run_meta["final_val_metrics"] = val_metrics
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\nTraining complete. Best val board_acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to {save_dir}/")
