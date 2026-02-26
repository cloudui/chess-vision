import torch.nn as nn
import torch.nn.functional as F
import timm

from dataset import NUM_PIECE_TYPES, NUM_PIECE_COLORS
from .common import combine_type_color, register_type_color_buffers


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
        self.type_head = nn.Sequential(drop, nn.Linear(hidden_dim, NUM_PIECE_TYPES))
        self.color_head = nn.Sequential(drop, nn.Linear(hidden_dim, NUM_PIECE_COLORS))
        self.turn_head = nn.Sequential(drop, nn.Linear(hidden_dim, 1))
        self.castling_head = nn.Sequential(drop, nn.Linear(hidden_dim, 4))
        register_type_color_buffers(self)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]

        patch_tokens = features[:, 1:]
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        spatial = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(spatial, (8, 8))
        pooled = pooled.permute(0, 2, 3, 1)

        type_logits = self.type_head(pooled)
        color_logits = self.color_head(pooled)
        squares = combine_type_color(
            type_logits, color_logits, self.class_to_type, self.class_to_color
        )

        return {
            "squares": squares.reshape(B, -1),
            "turn": self.turn_head(cls_token),
            "castling": self.castling_head(cls_token),
        }


def build_vit(model_cfg: dict) -> ChessViT:
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
