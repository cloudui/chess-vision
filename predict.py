import argparse

import torch
from PIL import Image

from dataset import get_transform, labels_to_fen, NUM_CLASSES, NUM_SQUARES
from models import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)

        # Piece placement
        sq_logits = outputs["squares"].view(NUM_SQUARES, NUM_CLASSES)
        preds = sq_logits.argmax(dim=-1)
        placement = labels_to_fen(preds.cpu())

        # Turn
        turn = "b" if outputs["turn"].item() > 0 else "w"

        # Castling
        castling_preds = (outputs["castling"].squeeze(0) > 0).tolist()
        castling_chars = ""
        for flag, ch in zip(castling_preds, ["K", "Q", "k", "q"]):
            if flag:
                castling_chars += ch
        castling = castling_chars or "-"

    return f"{placement} {turn} {castling}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict FEN from a chess board image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", required=True, help="Path to chess board image")
    args = parser.parse_args()

    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    input_size = cfg["model"].get("input_size")
    transform = get_transform(cfg["model"]["name"], is_training=False, input_size=input_size)

    fen = predict(model, args.image, transform, device)
    print(fen)
