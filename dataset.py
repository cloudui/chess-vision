import csv
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

# Type + color decomposition for piece classification
# Type: 7 classes (empty, pawn, knight, bishop, rook, queen, king)
# Color: 3 classes (empty, white, black)
NUM_PIECE_TYPES = 7
NUM_PIECE_COLORS = 3

# Maps each of the 13 joint classes to its type index and color index
#              .  P  N  B  R  Q  K  p  n  b  r  q  k
CLASS_TO_TYPE = [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
CLASS_TO_COLOR = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


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


def labels_to_fen(labels: torch.Tensor) -> str:
    """Convert a (64,) tensor of class indices back to a FEN board string."""
    fen_ranks = []
    for rank_start in range(0, 64, 8):
        rank_str = ""
        empty_count = 0
        for sq in range(rank_start, rank_start + 8):
            piece = INDEX_TO_PIECE[labels[sq].item()]
            if piece == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece
        if empty_count > 0:
            rank_str += str(empty_count)
        fen_ranks.append(rank_str)
    return '/'.join(fen_ranks)


def filename_to_fen(filename: str) -> str:
    """Convert a filename like '1B1B1K2-3p1N2-...-1B6.jpeg' to a FEN string."""
    name = os.path.splitext(filename)[0]
    return name.replace('-', '/')


def parse_full_fen(fen_str: str) -> dict:
    """Parse a full FEN string into placement, turn, and castling components.

    Args:
        fen_str: FEN string, e.g. "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
                 Can have 2-6 space-separated fields. Only placement, turn, castling are used.

    Returns:
        dict with:
            "squares": (64,) long tensor of piece classes
            "turn": (1,) float tensor, 0.0 = white, 1.0 = black
            "castling": (4,) float tensor, [K, Q, k, q]
    """
    parts = fen_str.strip().split()
    placement = parts[0]

    # Turn
    turn_char = parts[1] if len(parts) > 1 else "w"
    turn_val = 1.0 if turn_char == "b" else 0.0

    # Castling
    castling_str = parts[2] if len(parts) > 2 else "-"
    castling = [0.0, 0.0, 0.0, 0.0]  # K, Q, k, q
    if castling_str != "-":
        if "K" in castling_str:
            castling[0] = 1.0
        if "Q" in castling_str:
            castling[1] = 1.0
        if "k" in castling_str:
            castling[2] = 1.0
        if "q" in castling_str:
            castling[3] = 1.0

    return {
        "squares": fen_to_labels(placement),
        "turn": torch.tensor([turn_val], dtype=torch.float),
        "castling": torch.tensor(castling, dtype=torch.float),
    }


# ---------------------------------------------------------------------------
# Training augmentations for OOD robustness
# ---------------------------------------------------------------------------

class RandomChannelPermutation:
    """Randomly permute RGB channels of a tensor (after ToTensor)."""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            perm = torch.randperm(3)
            return tensor[perm]
        return tensor


class RandomInvert:
    """Randomly invert pixel values: 1 - x (after ToTensor, before Normalize)."""
    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            return 1.0 - tensor
        return tensor


def get_transform(model_name: str, is_training: bool = False, input_size: int | None = None):
    """Build the image transform from a timm model's pretrained config.

    For training, uses chess-safe augmentations designed for OOD robustness:
    - No horizontal flip (would swap board columns, misaligning labels)
    - No aggressive random crop (would lose squares)
    - Mild color jitter with small hue range (different board themes)
    - Random grayscale (forces shape-based piece recognition)
    - Occasional Gaussian blur (resolution robustness)
    - Occasional Gaussian blur (resolution robustness)
    """
    pretrained_cfg = timm.create_model(model_name, pretrained=False).pretrained_cfg
    data_cfg = resolve_data_config(pretrained_cfg)
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    if input_size is None:
        input_size = data_cfg["input_size"][-1]  # 224

    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class ChessDataset(Dataset):
    """Dataset of chess board images with per-square piece labels.

    Supports two modes:
    1. Manifest CSV mode: auto-detected at ``{root_dir}/manifest.csv``,
       or an explicit path via *manifest*.  FEN includes placement + turn +
       castling (+ optional en passant).  Extra columns (piece_count, style,
       flipped, etc.) are available via ``get_metadata`` for eval grouping.
    2. Filename mode (legacy/Kaggle): parses FEN from filenames.

    Each item returns:
        image: (3, H, H) tensor (H defaults to 224, configurable via input_size)
        labels: dict with:
            "squares": (64,) long tensor of piece classes [0..12]
            "turn": (1,) float tensor, 0.0 = white, 1.0 = black
            "castling": (4,) float tensor, [K, Q, k, q]
            "legal": (1,) float tensor, 1.0 = legal position, 0.0 = random
    """

    def __init__(
        self,
        root_dir: str,
        model_name: str = "vit_base_patch16_224.augreg_in21k",
        max_samples: int | None = None,
        is_training: bool = False,
        transform=None,
        manifest: str | None = None,
        input_size: int | None = None,
    ):
        self.root_dir = root_dir
        self.transform = transform or get_transform(
            model_name, is_training=is_training, input_size=input_size
        )

        # Auto-detect manifest when not explicitly provided
        if manifest is None:
            manifest = os.path.join(root_dir, "manifest.csv")

        if os.path.exists(manifest):
            self.samples = []
            with open(manifest, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append(dict(row))
            self.use_manifest = True
        else:
            self.samples = [
                {"filename": f}
                for f in sorted(os.listdir(root_dir))
                if f.endswith('.jpeg') or f.endswith('.png')
            ]
            self.use_manifest = False

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def get_metadata(self, idx: int) -> dict:
        """Get the raw manifest row dict for a sample (for eval grouping)."""
        return self.samples[idx]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample["filename"]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        fen = sample.get("fen")
        if self.use_manifest and fen:
            labels = parse_full_fen(fen)
        else:
            placement_fen = filename_to_fen(filename)
            labels = {
                "squares": fen_to_labels(placement_fen),
                "turn": torch.tensor([0.0], dtype=torch.float),
                "castling": torch.zeros(4, dtype=torch.float),
            }

        if self.use_manifest:
            legal = sample.get("legal", "1") == "1"
        else:
            legal = False
        labels["legal"] = torch.tensor([1.0 if legal else 0.0], dtype=torch.float)

        return image, labels
