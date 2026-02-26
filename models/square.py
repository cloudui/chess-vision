import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from dataset import NUM_CLASSES, NUM_PIECE_TYPES, NUM_PIECE_COLORS
from .common import combine_type_color, register_type_color_buffers


class ChessSquareCNN(nn.Module):
    """Per-square chess board recognition.

    Crops each of the 64 squares (with neighbor overlap) from the full
    board image, classifies each independently with a pretrained
    MobileNetV4 backbone, then aggregates features for turn/castling.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int,
                 square_overlap: float = 1.5, square_input_size: int = 64,
                 head_dropout: float = 0.0):
        super().__init__()
        self.square_overlap = square_overlap
        self.square_input_size = square_input_size
        self.feature_dim = feature_dim

        # Per-square piece classifier
        self.backbone = backbone
        drop = nn.Dropout(head_dropout)
        self.type_head = nn.Sequential(drop, nn.Linear(feature_dim, NUM_PIECE_TYPES))
        self.color_head = nn.Sequential(drop, nn.Linear(feature_dim, NUM_PIECE_COLORS))
        register_type_color_buffers(self)

        # Global heads from aggregated square features
        self.global_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(64 * feature_dim, 64),
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
        sq_size = H // 8
        crop_size = int(sq_size * self.square_overlap)
        pad = (crop_size - sq_size) // 2

        padded = F.pad(images, [pad, pad, pad, pad], mode="replicate")

        crops = []
        for row in range(8):
            for col in range(8):
                y = row * sq_size
                x = col * sq_size
                crop = padded[:, :, y:y + crop_size, x:x + crop_size]
                crops.append(crop)

        crops = torch.stack(crops, dim=1).reshape(B * 64, C, crop_size, crop_size)

        if crop_size != self.square_input_size:
            crops = F.interpolate(
                crops, size=self.square_input_size, mode="bilinear", align_corners=False
            )

        return crops

    def _extract_features(self, crops):
        """Extract features from square crops.

        Backbone is kept in eval mode to preserve pretrained BatchNorm
        statistics, which are more robust than stats computed from
        chess square crops alone.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.set_grad_enabled(was_training):
            feat = self.backbone.forward_features(crops)
            pooled = self.backbone.global_pool(feat)
        if was_training:
            self.backbone.train()
        return pooled.flatten(1)

    def forward(self, x):
        B = x.size(0)

        crops = self._crop_squares(x)
        features = self._extract_features(crops)     # (B*64, feature_dim)

        # Type + color → 13-class logits per square
        type_logits = self.type_head(features)
        color_logits = self.color_head(features)
        squares = combine_type_color(
            type_logits, color_logits, self.class_to_type, self.class_to_color
        )
        squares = squares.reshape(B, -1)             # (B, 832)

        # Global: aggregate per-square features for turn/castling
        global_feat = features.reshape(B, -1)        # (B, 64*feature_dim)
        global_feat = self.global_head(global_feat)

        return {
            "squares": squares,
            "turn": self.turn_head(global_feat),
            "castling": self.castling_head(global_feat),
        }


def build_square(model_cfg: dict) -> ChessSquareCNN:
    model_name = model_cfg.get("name", "mobilenetv4_conv_small_050.e3000_r224_in1k")
    pretrained = model_cfg.get("pretrained", True)

    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
    )
    feature_dim = backbone.num_features

    if model_cfg.get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False

    return ChessSquareCNN(
        backbone=backbone,
        feature_dim=feature_dim,
        square_overlap=model_cfg.get("square_overlap", 1.5),
        square_input_size=model_cfg.get("square_input_size", 64),
        head_dropout=model_cfg.get("head_dropout", 0.0),
    )
