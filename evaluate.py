import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    ChessDataset,
    NUM_CLASSES,
    NUM_SQUARES,
    INDEX_TO_PIECE,
    labels_to_fen,
)
from models import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def piece_count_bucket(count):
    """Bucket piece counts into game phases."""
    count = int(count)
    if count <= 10:
        return "endgame (2-10)"
    elif count <= 20:
        return "midgame (11-20)"
    else:
        return "opening (21-32)"


def castling_category(castling_str):
    """Categorize castling rights."""
    if castling_str == "-":
        return "none"
    return "has_rights"


@torch.no_grad()
def evaluate(model, dataset, loader, device):
    model.eval()
    piece_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_squares = 0
    correct_boards = 0
    total_squares = 0
    total_boards = 0
    correct_turn = 0
    total_legal = 0
    correct_castling_per_right = torch.zeros(4, dtype=torch.long)
    correct_castling_all = 0
    correct_full_fen = 0

    piece_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    piece_total = torch.zeros(NUM_CLASSES, dtype=torch.long)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    turn_confusion = torch.zeros(2, 2, dtype=torch.long)

    worst = []

    # Per-sample results for grouped metrics
    sample_results = []

    sample_idx = 0
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        sq_labels = labels["squares"].to(device, non_blocking=True)
        turn_labels = labels["turn"].to(device, non_blocking=True)
        castling_labels = labels["castling"].to(device, non_blocking=True)
        legal_mask = labels["legal"].to(device, non_blocking=True).squeeze(1)

        outputs = model(images)
        sq_logits = outputs["squares"].view(-1, NUM_SQUARES, NUM_CLASSES)

        batch_size = images.size(0)
        n_legal = int(legal_mask.sum().item())

        preds = sq_logits.argmax(dim=-1)
        sq_correct = (preds == sq_labels)
        correct_squares += sq_correct.sum().item()
        board_correct = sq_correct.all(dim=1)
        correct_boards += board_correct.sum().item()
        total_squares += sq_labels.numel()
        total_boards += batch_size

        piece_loss = piece_criterion(sq_logits.reshape(-1, NUM_CLASSES), sq_labels.reshape(-1))
        total_loss += piece_loss.item() * batch_size

        # Turn/castling
        turn_preds = (outputs["turn"] > 0).float()
        turn_correct_mask = (turn_preds == turn_labels).squeeze(1)
        castling_preds = (outputs["castling"] > 0).float()
        castling_right_correct = (castling_preds == castling_labels)
        castling_all_correct = castling_right_correct.all(dim=1)

        if n_legal > 0:
            legal_idx = legal_mask.bool()
            total_legal += n_legal
            correct_turn += (turn_correct_mask & legal_idx).sum().item()

            turn_true = turn_labels.squeeze(1).long().cpu()
            turn_pred = turn_preds.squeeze(1).long().cpu()
            legal_cpu = legal_idx.cpu()
            for j in range(batch_size):
                if legal_cpu[j]:
                    turn_confusion[turn_true[j], turn_pred[j]] += 1

            for r in range(4):
                correct_castling_per_right[r] += (castling_right_correct[:, r] & legal_idx).sum().item()
            correct_castling_all += (castling_all_correct & legal_idx).sum().item()

            full_correct = board_correct & turn_correct_mask & castling_all_correct & legal_idx
            correct_full_fen += full_correct.sum().item()

        # Per-piece and confusion
        preds_cpu = preds.cpu()
        labels_cpu = sq_labels.cpu()
        for c in range(NUM_CLASSES):
            mask = labels_cpu == c
            piece_total[c] += mask.sum()
            piece_correct[c] += (preds_cpu[mask] == c).sum()

        for t, p in zip(labels_cpu.reshape(-1), preds_cpu.reshape(-1)):
            confusion[t, p] += 1

        # Store per-sample results for grouped analysis
        for i in range(batch_size):
            num_wrong = (preds_cpu[i] != labels_cpu[i]).sum().item()
            is_legal = legal_mask[i].item() > 0
            result = {
                "idx": sample_idx + i,
                "board_correct": board_correct[i].item(),
                "squares_wrong": num_wrong,
                "turn_correct": turn_correct_mask[i].item() if is_legal else None,
                "castling_correct": castling_all_correct[i].item() if is_legal else None,
            }
            sample_results.append(result)

            if num_wrong > 0:
                worst.append((
                    num_wrong,
                    labels_to_fen(labels_cpu[i]),
                    labels_to_fen(preds_cpu[i]),
                    sample_idx + i,
                ))

        sample_idx += batch_size

    # --- Print overall results ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall ({total_boards} images, {total_legal} legal):")
    print(f"  Loss:            {total_loss / total_boards:.4f}")
    print(f"  Per-square acc:  {correct_squares / total_squares:.4f} "
          f"({correct_squares}/{total_squares})")
    print(f"  Full-board acc:  {correct_boards / total_boards:.4f} "
          f"({correct_boards}/{total_boards})")

    if total_legal > 0:
        print(f"\nTurn prediction (legal positions only):")
        print(f"  Accuracy:        {correct_turn / total_legal:.4f} "
              f"({correct_turn}/{total_legal})")
        print(f"  Confusion (rows=true, cols=pred):")
        print(f"             White  Black")
        print(f"    White  {turn_confusion[0, 0]:>6d} {turn_confusion[0, 1]:>6d}")
        print(f"    Black  {turn_confusion[1, 0]:>6d} {turn_confusion[1, 1]:>6d}")

        castling_names = ["K", "Q", "k", "q"]
        print(f"\nCastling prediction (legal positions only):")
        for r in range(4):
            acc = correct_castling_per_right[r].item() / total_legal
            print(f"  {castling_names[r]:>1s}: {acc:.4f} "
                  f"({correct_castling_per_right[r]}/{total_legal})")
        print(f"  All-4-correct:   {correct_castling_all / total_legal:.4f} "
              f"({correct_castling_all}/{total_legal})")

        print(f"\nFull FEN accuracy (position + turn + castling, legal only):")
        print(f"  {correct_full_fen / total_legal:.4f} "
              f"({correct_full_fen}/{total_legal})")
    else:
        print(f"\nNo legal positions in dataset — turn/castling metrics skipped.")

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

    # --- Grouped metrics by manifest properties ---
    print_grouped_metrics(dataset, sample_results)

    # Return summary metrics for logging
    return {
        "loss": total_loss / total_boards,
        "square_acc": correct_squares / total_squares,
        "board_acc": correct_boards / total_boards,
        "turn_acc": correct_turn / max(total_legal, 1),
        "castling_acc": correct_castling_all / max(total_legal, 1),
        "full_fen_acc": correct_full_fen / max(total_legal, 1),
        "total_boards": total_boards,
        "total_legal": total_legal,
    }


def print_grouped_metrics(dataset, sample_results):
    """Print accuracy breakdowns grouped by manifest metadata fields."""
    if not getattr(dataset, 'use_manifest', False):
        return

    # Fields to group by and how to bucket them
    grouping_fields = {
        "piece_count": piece_count_bucket,
        "castling": castling_category,
        "turn": lambda x: "white" if x == "w" else "black",
        "has_highlight": lambda x: "highlighted" if x == "1" else "no highlight",
        "style": lambda x: x,
        "flipped": lambda x: "flipped" if x == "1" else "normal",
    }

    print("\n" + "=" * 60)
    print("GROUPED METRICS")
    print("=" * 60)

    for field, bucket_fn in grouping_fields.items():
        # Check if this field exists in metadata
        sample_meta = dataset.get_metadata(0)
        if field not in sample_meta:
            continue

        groups = defaultdict(lambda: {"total": 0, "board_correct": 0,
                                       "turn_correct": 0, "turn_total": 0,
                                       "castling_correct": 0, "castling_total": 0})

        for result in sample_results:
            meta = dataset.get_metadata(result["idx"])
            raw_val = meta.get(field, "")
            bucket = bucket_fn(raw_val)
            g = groups[bucket]
            g["total"] += 1
            g["board_correct"] += result["board_correct"]
            if result["turn_correct"] is not None:
                g["turn_total"] += 1
                g["turn_correct"] += result["turn_correct"]
            if result["castling_correct"] is not None:
                g["castling_total"] += 1
                g["castling_correct"] += result["castling_correct"]

        print(f"\nBy {field}:")
        for bucket in sorted(groups.keys()):
            g = groups[bucket]
            board_acc = g["board_correct"] / g["total"] if g["total"] > 0 else 0
            line = f"  {bucket:>20s}: board_acc={board_acc:.4f} (n={g['total']})"
            if g["turn_total"] > 0:
                turn_acc = g["turn_correct"] / g["turn_total"]
                line += f"  turn={turn_acc:.4f}"
            if g["castling_total"] > 0:
                castling_acc = g["castling_correct"] / g["castling_total"]
                line += f"  castling={castling_acc:.4f}"
            print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chess ViT on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-dir", default=None, help="Override test directory")
    parser.add_argument("--manifest", default=None, help="Path to manifest CSV for test set")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    test_dir = args.test_dir or cfg["data"]["test_dir"]
    input_size = cfg["model"].get("input_size")
    test_dataset = ChessDataset(
        test_dir,
        model_name=cfg["model"]["name"],
        is_training=False,
        manifest=args.manifest,
        input_size=input_size,
        max_samples=args.max_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    print(f"Test set: {len(test_dataset)} images from {test_dir}")

    metrics = evaluate(model, test_dataset, test_loader, device)

    # Append eval results to JSONL in the checkpoint directory
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    eval_log = os.path.join(ckpt_dir, "eval_results.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "test_dir": test_dir,
        "num_samples": len(test_dataset),
        "metrics": metrics,
    }
    with open(eval_log, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nResults appended to {eval_log}")
