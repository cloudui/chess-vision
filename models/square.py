import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_CLASSES, NUM_PIECE_TYPES, NUM_PIECE_COLORS
from .common import combine_type_color, register_type_color_buffers


class SquareBackbone(nn.Module):
    """Lightweight CNN for classifying a single chess square crop."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 64 → 32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 32 → 16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 16 → 8

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),            # 8 → 1
        )

    def forward(self, x):
        return self.features(x).flatten(1)      # (N, 128)


class ChessSquareCNN(nn.Module):
    """Per-square chess board recognition.

    Crops each of the 64 squares (with neighbor overlap) from the full
    board image, classifies each independently with a lightweight CNN,
    then aggregates features for turn/castling prediction.

    ~825K parameters total.
    """

    def __init__(self, square_overlap: float = 1.5, square_input_size: int = 64,
                 head_dropout: float = 0.0):
        super().__init__()
        self.square_overlap = square_overlap
        self.square_input_size = square_input_size
        self.feature_dim = 128

        # Per-square piece classifier
        self.backbone = SquareBackbone()
        drop = nn.Dropout(head_dropout)
        self.type_head = nn.Sequential(drop, nn.Linear(self.feature_dim, NUM_PIECE_TYPES))
        self.color_head = nn.Sequential(drop, nn.Linear(self.feature_dim, NUM_PIECE_COLORS))
        register_type_color_buffers(self)

        # Global heads from aggregated square features
        self.global_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(64 * self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
        )
        self.turn_head = nn.Linear(64, 1)
        self.castling_head = nn.Linear(64, 4)

    def _crop_squares(self, images):
        """Crop 64 squares from each board image with overlap.

        Args:
            images: (B, 3, H, H) board images

        Returns:
            (B*64, 3, square_input_size, square_input_size) square crops
        """
        B, C, H, W = images.shape
        sq_size = H // 8  # pixels per square (e.g., 32 for 256x256)
        crop_size = int(sq_size * self.square_overlap)
        pad = (crop_size - sq_size) // 2

        # Pad the image with border replication
        padded = F.pad(images, [pad, pad, pad, pad], mode="replicate")

        crops = []
        for row in range(8):
            for col in range(8):
                y = row * sq_size
                x = col * sq_size
                crop = padded[:, :, y:y + crop_size, x:x + crop_size]
                crops.append(crop)

        # (64, B, C, crop_size, crop_size) → (B*64, C, crop_size, crop_size)
        crops = torch.stack(crops, dim=1).reshape(B * 64, C, crop_size, crop_size)

        # Resize to target input size
        if crop_size != self.square_input_size:
            crops = F.interpolate(
                crops, size=self.square_input_size, mode="bilinear", align_corners=False
            )

        return crops

    def forward(self, x):
        B = x.size(0)

        # Crop and classify each square
        crops = self._crop_squares(x)               # (B*64, 3, 64, 64)
        features = self.backbone(crops)              # (B*64, 128)

        # Type + color → 13-class logits per square
        type_logits = self.type_head(features)       # (B*64, 7)
        color_logits = self.color_head(features)     # (B*64, 3)
        squares = combine_type_color(
            type_logits, color_logits, self.class_to_type, self.class_to_color
        )                                            # (B*64, 13)
        squares = squares.reshape(B, -1)             # (B, 832)

        # Global: aggregate per-square features for turn/castling
        global_feat = features.reshape(B, -1)        # (B, 64*128)
        global_feat = self.global_head(global_feat)  # (B, 64)

        return {
            "squares": squares,
            "turn": self.turn_head(global_feat),
            "castling": self.castling_head(global_feat),
        }


def build_square(model_cfg: dict) -> ChessSquareCNN:
    return ChessSquareCNN(
        square_overlap=model_cfg.get("square_overlap", 1.5),
        square_input_size=model_cfg.get("square_input_size", 64),
        head_dropout=model_cfg.get("head_dropout", 0.0),
    )
