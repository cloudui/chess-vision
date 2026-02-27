#!/usr/bin/env python
"""Visualize failing predictions on a test set.

Usage:
    python visualize_failures.py --checkpoint checkpoints/best.pth --test-dir kaggle/test --max-samples 5000
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from dataset import (
    ChessDataset, NUM_CLASSES, NUM_SQUARES,
    INDEX_TO_PIECE, labels_to_fen, filename_to_fen,
)
from models import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def find_failures(model, dataset, loader, device, max_failures=50):
    model.eval()
    failures = []
    sample_idx = 0

    for images, labels in tqdm(loader, desc="Scanning"):
        images = images.to(device, non_blocking=True)
        sq_labels = labels["squares"].to(device, non_blocking=True)

        outputs = model(images)
        sq_logits = outputs["squares"].view(-1, NUM_SQUARES, NUM_CLASSES)
        preds = sq_logits.argmax(dim=-1)

        batch_size = images.size(0)
        for i in range(batch_size):
            num_wrong = (preds[i] != sq_labels[i]).sum().item()
            if num_wrong > 0:
                failures.append({
                    "idx": sample_idx + i,
                    "num_wrong": num_wrong,
                    "true_fen": labels_to_fen(sq_labels[i].cpu()),
                    "pred_fen": labels_to_fen(preds[i].cpu()),
                })
            if len(failures) >= max_failures:
                return failures
        sample_idx += batch_size

    return failures


def save_failure_grid(failures, dataset, test_dir, output_path, cols=5):
    """Save a grid of failing images with true/pred FEN annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = min(len(failures), 30)
    failures_sorted = sorted(failures, key=lambda x: -x["num_wrong"])[:rows]

    fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows))
    if rows == 1:
        axes = [axes]

    for i, fail in enumerate(failures_sorted):
        idx = fail["idx"]
        sample = dataset.samples[idx] if hasattr(dataset, 'samples') else None

        if sample:
            filename = sample["filename"]
        else:
            filenames = sorted([
                f for f in os.listdir(test_dir)
                if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')
            ])
            filename = filenames[idx]

        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path).convert("RGB")

        axes[i].imshow(img)
        axes[i].set_title(
            f"#{idx} — {fail['num_wrong']} wrong\n"
            f"True: {fail['true_fen']}\n"
            f"Pred: {fail['pred_fen']}",
            fontsize=8, fontfamily="monospace", loc="left",
        )
        axes[i].axis("off")

    plt.suptitle(f"Failing Predictions ({len(failures_sorted)} worst)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved failure grid to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-failures", type=int, default=50)
    parser.add_argument("--output", default="failures.png")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    input_size = cfg["model"].get("input_size")
    dataset = ChessDataset(
        args.test_dir,
        model_name=cfg["model"].get("name", "vit_base_patch16_224.augreg_in21k"),
        is_training=False,
        input_size=input_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    failures = find_failures(model, dataset, loader, device, max_failures=args.max_failures)
    print(f"Found {len(failures)} failing boards")

    save_failure_grid(failures, dataset, args.test_dir, args.output)
