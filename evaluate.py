import argparse

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
    piece_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_squares = 0
    correct_boards = 0
    total_squares = 0
    total_boards = 0
    correct_turn = 0
    total_legal = 0
    # Per-castling-right tracking: [K, Q, k, q]
    correct_castling_per_right = torch.zeros(4, dtype=torch.long)
    correct_castling_all = 0
    correct_full_fen = 0

    # Per-piece tracking
    piece_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    piece_total = torch.zeros(NUM_CLASSES, dtype=torch.long)

    # Confusion matrix
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    # Turn confusion (legal positions only)
    turn_confusion = torch.zeros(2, 2, dtype=torch.long)

    # Track worst predictions
    worst = []

    sample_idx = 0
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        sq_labels = labels["squares"].to(device, non_blocking=True)
        turn_labels = labels["turn"].to(device, non_blocking=True)
        castling_labels = labels["castling"].to(device, non_blocking=True)
        legal_mask = labels["legal"].to(device, non_blocking=True).squeeze(1)  # (B,)

        outputs = model(images)
        sq_logits = outputs["squares"].view(-1, NUM_SQUARES, NUM_CLASSES)

        batch_size = images.size(0)
        n_legal = int(legal_mask.sum().item())

        # Piece predictions (all positions)
        preds = sq_logits.argmax(dim=-1)
        sq_correct = (preds == sq_labels)
        correct_squares += sq_correct.sum().item()
        board_correct = sq_correct.all(dim=1)
        correct_boards += board_correct.sum().item()
        total_squares += sq_labels.numel()
        total_boards += batch_size

        # Loss (piece only, since turn/castling may not be valid)
        piece_loss = piece_criterion(sq_logits.reshape(-1, NUM_CLASSES), sq_labels.reshape(-1))
        total_loss += piece_loss.item() * batch_size

        # Turn/castling accuracy (legal positions only)
        if n_legal > 0:
            legal_idx = legal_mask.bool()
            total_legal += n_legal

            turn_preds = (outputs["turn"] > 0).float()
            turn_correct = (turn_preds == turn_labels).squeeze(1)
            correct_turn += (turn_correct & legal_idx).sum().item()

            # Turn confusion matrix (legal only)
            turn_true = turn_labels.squeeze(1).long().cpu()
            turn_pred = turn_preds.squeeze(1).long().cpu()
            legal_cpu = legal_idx.cpu()
            for j in range(batch_size):
                if legal_cpu[j]:
                    turn_confusion[turn_true[j], turn_pred[j]] += 1

            castling_preds = (outputs["castling"] > 0).float()
            castling_right_correct = (castling_preds == castling_labels)
            for r in range(4):
                correct_castling_per_right[r] += (castling_right_correct[:, r] & legal_idx).sum().item()
            castling_all_correct = castling_right_correct.all(dim=1)
            correct_castling_all += (castling_all_correct & legal_idx).sum().item()

            # Full FEN accuracy (legal only)
            full_correct = board_correct & turn_correct & castling_all_correct & legal_idx
            correct_full_fen += full_correct.sum().item()

        # Per-piece and confusion (all positions)
        preds_cpu = preds.cpu()
        labels_cpu = sq_labels.cpu()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chess ViT on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-dir", default=None, help="Override test directory")
    parser.add_argument("--manifest", default=None, help="Path to manifest CSV for test set")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    test_dir = args.test_dir or cfg["data"]["test_dir"]
    manifest = args.manifest or cfg["data"].get("manifest")
    test_dataset = ChessDataset(
        test_dir,
        model_name=cfg["model"]["name"],
        is_training=False,
        manifest=manifest,
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
