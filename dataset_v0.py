import os
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# 13 classes: empty + 6 white pieces + 6 black pieces
PIECE_TO_INDEX = {
    '.': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}
INDEX_TO_PIECE = {v: k for k, v in PIECE_TO_INDEX.items()}
NUM_CLASSES = 13
NUM_SQUARES = 64


def fen_to_labels(fen: str) -> torch.Tensor:
    """Convert a FEN board string to a (64,) tensor of class indices.

    FEN ranks go from rank 8 (top) to rank 1 (bottom), left to right,
    so index 0 = a8, index 7 = h8, index 8 = a7, ..., index 63 = h1.
    """
    squares = []
    for rank in fen.split('/'):
        for ch in rank:
            if ch.isdigit():
                squares.extend([0] * int(ch))
            else:
                squares.append(PIECE_TO_INDEX[ch])
    assert len(squares) == 64, f"Expected 64 squares, got {len(squares)} from FEN: {fen}"
    return torch.tensor(squares, dtype=torch.long)


def filename_to_fen(filename: str) -> str:
    """Convert a filename like '1B1B1K2-3p1N2-...-1B6.jpeg' to a FEN string."""
    name = os.path.splitext(filename)[0]  # strip .jpeg
    return name.replace('-', '/')


class ChessDataset(Dataset):
    """Dataset of chess board images with per-square piece labels.

    Each item returns:
        image: (3, 224, 224) tensor, normalized for ImageNet-pretrained models
        labels: (64,) tensor of class indices in [0, 12]
    """

    def __init__(self, root_dir: str, max_samples: int | None = None, transform=None):
        self.root_dir = root_dir
        self.filenames = sorted([
            f for f in os.listdir(root_dir) if f.endswith('.jpeg')
        ])
        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]

        if transform is not None:
            self.transform = transform
        else:
            # Use the model's own preprocessing config
            data_cfg = resolve_data_config(
                timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=False).pretrained_cfg
            )
            self.transform = create_transform(**data_cfg, is_training=False)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        fen = filename_to_fen(filename)
        labels = fen_to_labels(fen)

        return image, labels
