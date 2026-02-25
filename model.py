import timm
import torch.nn as nn

from dataset import NUM_CLASSES, NUM_SQUARES


class ChessViT(nn.Module):
    """Multi-head ViT for chess board recognition.

    Predicts piece placement, turn-to-move, and castling rights
    from the CLS token of a ViT backbone.
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.piece_head = nn.Linear(hidden_dim, NUM_SQUARES * NUM_CLASSES)
        self.turn_head = nn.Linear(hidden_dim, 1)
        self.castling_head = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        features = self.backbone.forward_features(x)  # (B, num_tokens, D)
        cls_token = features[:, 0]                     # (B, D)
        return {
            "squares": self.piece_head(cls_token),     # (B, 832)
            "turn": self.turn_head(cls_token),         # (B, 1)
            "castling": self.castling_head(cls_token), # (B, 4)
        }


def build_model(cfg: dict) -> nn.Module:
    """Build a multi-head ViT model from config."""
    backbone = timm.create_model(
        cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=0,  # remove default classification head
    )
    hidden_dim = backbone.num_features

    if cfg["model"].get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False

    return ChessViT(backbone, hidden_dim)
