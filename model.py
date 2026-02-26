import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from dataset import NUM_CLASSES, NUM_SQUARES


class ChessViT(nn.Module):
    """Multi-head ViT for chess board recognition.

    Piece placement uses spatial patch tokens pooled to an 8x8 grid,
    giving each square its own local features.  Turn and castling use
    the CLS token since they are global board properties.
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int, head_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        drop = nn.Dropout(head_dropout)
        # Per-square classifier applied to pooled spatial features
        self.square_head = nn.Sequential(drop, nn.Linear(hidden_dim, NUM_CLASSES))
        # Global heads from CLS token
        self.turn_head = nn.Sequential(drop, nn.Linear(hidden_dim, 1))
        self.castling_head = nn.Sequential(drop, nn.Linear(hidden_dim, 4))

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
        squares = self.square_head(pooled)                               # (B, 8, 8, 13)

        return {
            "squares": squares.reshape(B, -1),         # (B, 832)
            "turn": self.turn_head(cls_token),          # (B, 1)
            "castling": self.castling_head(cls_token),  # (B, 4)
        }


def build_model(cfg: dict) -> nn.Module:
    """Build a multi-head ViT model from config."""
    model_cfg = cfg["model"]
    input_size = model_cfg.get("input_size")

    kwargs = {}
    if input_size:
        kwargs["img_size"] = input_size

    backbone = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=0,  # remove default classification head
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        **kwargs,
    )
    hidden_dim = backbone.num_features

    if model_cfg.get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False

    head_dropout = model_cfg.get("head_dropout", 0.0)
    return ChessViT(backbone, hidden_dim, head_dropout=head_dropout)
