import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from dataset import (
    NUM_CLASSES, NUM_SQUARES,
    NUM_PIECE_TYPES, NUM_PIECE_COLORS,
    CLASS_TO_TYPE, CLASS_TO_COLOR,
)


def _combine_type_color(type_logits, color_logits, class_to_type, class_to_color):
    """Combine type and color logits into joint 13-class logits.

    joint[..., c] = type_logits[..., CLASS_TO_TYPE[c]] + color_logits[..., CLASS_TO_COLOR[c]]

    Args:
        type_logits:  (..., NUM_PIECE_TYPES)  — 7 classes
        color_logits: (..., NUM_PIECE_COLORS) — 3 classes
        class_to_type:  (13,) long tensor mapping joint class → type index
        class_to_color: (13,) long tensor mapping joint class → color index

    Returns:
        (..., NUM_CLASSES) — 13-class joint logits
    """
    return type_logits[..., class_to_type] + color_logits[..., class_to_color]


# ---------------------------------------------------------------------------
# ViT architecture
# ---------------------------------------------------------------------------

class ChessViT(nn.Module):
    """ViT-based chess board recognition.

    Piece placement uses spatial patch tokens pooled to an 8x8 grid.
    Pieces are classified via separate type (7-class) and color (3-class)
    heads whose logits are additively combined into 13 joint classes.
    Turn and castling use the CLS token.
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int, head_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        drop = nn.Dropout(head_dropout)
        # Decomposed per-square classifiers
        self.type_head = nn.Sequential(drop, nn.Linear(hidden_dim, NUM_PIECE_TYPES))
        self.color_head = nn.Sequential(drop, nn.Linear(hidden_dim, NUM_PIECE_COLORS))
        # Global heads from CLS token
        self.turn_head = nn.Sequential(drop, nn.Linear(hidden_dim, 1))
        self.castling_head = nn.Sequential(drop, nn.Linear(hidden_dim, 4))
        # Mapping buffers (move to device with model)
        self.register_buffer("class_to_type", torch.tensor(CLASS_TO_TYPE, dtype=torch.long))
        self.register_buffer("class_to_color", torch.tensor(CLASS_TO_COLOR, dtype=torch.long))

    def forward(self, x):
        features = self.backbone.forward_features(x)  # (B, 1 + H*W, D)
        cls_token = features[:, 0]                     # (B, D)

        # Spatial tokens → 8x8 grid for per-square classification
        patch_tokens = features[:, 1:]                 # (B, H*W, D)
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        spatial = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        pooled = F.adaptive_avg_pool2d(spatial, (8, 8))                  # (B, D, 8, 8)
        pooled = pooled.permute(0, 2, 3, 1)                             # (B, 8, 8, D)

        # Type + color → joint 13-class logits
        type_logits = self.type_head(pooled)                             # (B, 8, 8, 7)
        color_logits = self.color_head(pooled)                           # (B, 8, 8, 3)
        squares = _combine_type_color(
            type_logits, color_logits, self.class_to_type, self.class_to_color
        )                                                                # (B, 8, 8, 13)

        return {
            "squares": squares.reshape(B, -1),         # (B, 832)
            "turn": self.turn_head(cls_token),          # (B, 1)
            "castling": self.castling_head(cls_token),  # (B, 4)
        }


# ---------------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------------

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
        # Decomposed per-square classification via 1x1 convs
        self.type_head = nn.Sequential(
            nn.Dropout2d(head_dropout),
            nn.Conv2d(feature_channels, NUM_PIECE_TYPES, kernel_size=1),
        )
        self.color_head = nn.Sequential(
            nn.Dropout2d(head_dropout),
            nn.Conv2d(feature_channels, NUM_PIECE_COLORS, kernel_size=1),
        )
        # Global heads from average-pooled features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        drop = nn.Dropout(head_dropout)
        self.turn_head = nn.Sequential(drop, nn.Linear(feature_channels, 1))
        self.castling_head = nn.Sequential(drop, nn.Linear(feature_channels, 4))
        # Mapping buffers
        self.register_buffer("class_to_type", torch.tensor(CLASS_TO_TYPE, dtype=torch.long))
        self.register_buffer("class_to_color", torch.tensor(CLASS_TO_COLOR, dtype=torch.long))

    def forward(self, x):
        features = self.backbone(x)                    # (B, C, H, W)

        # Pool to 8x8 if needed (no-op when input is 256x256 with stride-32)
        spatial = F.adaptive_avg_pool2d(features, (8, 8))  # (B, C, 8, 8)

        # Type + color → joint 13-class logits
        type_logits = self.type_head(spatial)           # (B, 7, 8, 8)
        color_logits = self.color_head(spatial)         # (B, 3, 8, 8)
        # Permute to (B, 8, 8, C) for the combine function
        type_logits = type_logits.permute(0, 2, 3, 1)  # (B, 8, 8, 7)
        color_logits = color_logits.permute(0, 2, 3, 1)  # (B, 8, 8, 3)
        squares = _combine_type_color(
            type_logits, color_logits, self.class_to_type, self.class_to_color
        )                                               # (B, 8, 8, 13)
        B = squares.size(0)
        squares = squares.reshape(B, -1)                # (B, 832)

        # Global features for turn and castling
        pooled = self.global_pool(features).flatten(1)  # (B, C)

        return {
            "squares": squares,                         # (B, 832)
            "turn": self.turn_head(pooled),             # (B, 1)
            "castling": self.castling_head(pooled),     # (B, 4)
        }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """Build a chess recognition model from config.

    Set ``cfg["model"]["arch"]`` to ``"vit"`` (default) or ``"cnn"``.
    """
    model_cfg = cfg["model"]
    arch = model_cfg.get("arch", "vit")

    if arch == "vit":
        return _build_vit(model_cfg)
    elif arch == "cnn":
        return _build_cnn(model_cfg)
    else:
        raise ValueError(f"Unknown architecture: {arch!r} (expected 'vit' or 'cnn')")


def _build_vit(model_cfg: dict) -> ChessViT:
    input_size = model_cfg.get("input_size")
    kwargs = {}
    if input_size:
        kwargs["img_size"] = input_size

    backbone = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=0,
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        **kwargs,
    )
    hidden_dim = backbone.num_features

    if model_cfg.get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False

    head_dropout = model_cfg.get("head_dropout", 0.0)
    return ChessViT(backbone, hidden_dim, head_dropout=head_dropout)


def _build_cnn(model_cfg: dict) -> ChessCNN:
    backbone = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=0,
        global_pool="",        # preserve spatial feature maps
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )
    feature_channels = backbone.num_features

    if model_cfg.get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False

    head_dropout = model_cfg.get("head_dropout", 0.0)
    return ChessCNN(backbone, feature_channels, head_dropout=head_dropout)
