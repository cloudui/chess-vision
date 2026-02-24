import argparse

import torch
from PIL import Image

from dataset import get_transform, labels_to_fen, NUM_CLASSES, NUM_SQUARES
from model import build_model


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
        logits = model(tensor).view(NUM_SQUARES, NUM_CLASSES)
        preds = logits.argmax(dim=-1)

    return labels_to_fen(preds.cpu())


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

    transform = get_transform(cfg["model"]["name"], is_training=False)

    fen = predict(model, args.image, transform, device)
    print(fen)
