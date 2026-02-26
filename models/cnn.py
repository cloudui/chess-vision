import torch.nn as nn
import torch.nn.functional as F
import timm

from dataset import NUM_PIECE_TYPES, NUM_PIECE_COLORS
from .common import combine_type_color, register_type_color_buffers


class ChessCNN(nn.Module):
    """CNN-based chess board recognition.

    Uses a stride-32 CNN backbone (e.g. ConvNeXt, ResNet).  A 256x256
    input produces an 8x8 feature map naturally aligned with the chess
    grid.  Pieces are classified via separate type (7-class) and color
    (3-class) 1x1 conv heads whose logits are additively combined.
    Turn and castling use globally-pooled features.
    """

    def __init__(self, backbone: nn.Module, feature_channels: int, head_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.type_head = nn.Sequential(
            nn.Dropout2d(head_dropout),
            nn.Conv2d(feature_channels, NUM_PIECE_TYPES, kernel_size=1),
        )
        self.color_head = nn.Sequential(
            nn.Dropout2d(head_dropout),
            nn.Conv2d(feature_channels, NUM_PIECE_COLORS, kernel_size=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        drop = nn.Dropout(head_dropout)
        self.turn_head = nn.Sequential(drop, nn.Linear(feature_channels, 1))
        self.castling_head = nn.Sequential(drop, nn.Linear(feature_channels, 4))
        register_type_color_buffers(self)

    def forward(self, x):
        features = self.backbone(x)
        spatial = F.adaptive_avg_pool2d(features, (8, 8))

        type_logits = self.type_head(spatial).permute(0, 2, 3, 1)
        color_logits = self.color_head(spatial).permute(0, 2, 3, 1)
        squares = combine_type_color(
            type_logits, color_logits, self.class_to_type, self.class_to_color
        )
        B = squares.size(0)

        pooled = self.global_pool(features).flatten(1)

        return {
            "squares": squares.reshape(B, -1),
            "turn": self.turn_head(pooled),
            "castling": self.castling_head(pooled),
        }


def build_cnn(model_cfg: dict) -> ChessCNN:
    backbone = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=0,
        global_pool="",
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )
    feature_channels = backbone.num_features

    if model_cfg.get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False

    head_dropout = model_cfg.get("head_dropout", 0.0)
    return ChessCNN(backbone, feature_channels, head_dropout=head_dropout)
