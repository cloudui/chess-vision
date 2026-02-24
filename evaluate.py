import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    ChessDataset,
    NUM_CLASSES,
    NUM_SQUARES,
    INDEX_TO_PIECE,
    labels_to_fen,
)
from model import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_squares = 0
    correct_boards = 0
    total_squares = 0
    total_boards = 0

    # Per-piece tracking: for each class, track TP and total ground-truth count
    piece_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    piece_total = torch.zeros(NUM_CLASSES, dtype=torch.long)

    # Confusion matrix
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    # Track worst predictions
    worst = []  # list of (num_wrong, fen_true, fen_pred, filename_idx)

    sample_idx = 0
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images).view(-1, NUM_SQUARES, NUM_CLASSES)
        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)

        correct_squares += (preds == labels).sum().item()
        correct_boards += (preds == labels).all(dim=1).sum().item()
        total_squares += labels.numel()
        total_boards += batch_size

        # Per-piece and confusion
        preds_cpu = preds.cpu()
        labels_cpu = labels.cpu()
        for c in range(NUM_CLASSES):
            mask = labels_cpu == c
            piece_total[c] += mask.sum()
            piece_correct[c] += (preds_cpu[mask] == c).sum()

        for t, p in zip(labels_cpu.reshape(-1), preds_cpu.reshape(-1)):
            confusion[t, p] += 1

        # Worst predictions
        for i in range(batch_size):
            num_wrong = (preds_cpu[i] != labels_cpu[i]).sum().item()
            if num_wrong > 0:
                worst.append((
                    num_wrong,
                    labels_to_fen(labels_cpu[i]),
                    labels_to_fen(preds_cpu[i]),
                    sample_idx + i,
                ))

        sample_idx += batch_size

    # --- Print results ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall:")
    print(f"  Loss:            {total_loss / total_boards:.4f}")
    print(f"  Per-square acc:  {correct_squares / total_squares:.4f} "
          f"({correct_squares}/{total_squares})")
    print(f"  Full-board acc:  {correct_boards / total_boards:.4f} "
          f"({correct_boards}/{total_boards})")

    print(f"\nPer-piece accuracy:")
    piece_names = {0: "empty", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
                   7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k"}
    for c in range(NUM_CLASSES):
        if piece_total[c] > 0:
            acc = piece_correct[c].item() / piece_total[c].item()
            print(f"  {piece_names[c]:>5s}: {acc:.4f}  ({piece_correct[c]}/{piece_total[c]})")

    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    header = "       " + "".join(f"{piece_names[c]:>6s}" for c in range(NUM_CLASSES))
    print(header)
    for t in range(NUM_CLASSES):
        row = f"  {piece_names[t]:>4s} " + "".join(f"{confusion[t, p].item():>6d}" for p in range(NUM_CLASSES))
        print(row)

    worst.sort(key=lambda x: -x[0])
    print(f"\nTop 10 worst predictions:")
    for num_wrong, fen_true, fen_pred, idx in worst[:10]:
        print(f"  Image {idx}: {num_wrong}/64 squares wrong")
        print(f"    True: {fen_true}")
        print(f"    Pred: {fen_pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chess ViT on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-dir", default=None, help="Override test directory")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    test_dir = args.test_dir or cfg["data"]["test_dir"]
    test_dataset = ChessDataset(
        test_dir,
        model_name=cfg["model"]["name"],
        is_training=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    print(f"Test set: {len(test_dataset)} images from {test_dir}")

    evaluate(model, test_loader, device)
