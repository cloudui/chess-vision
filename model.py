import timm
import torch.nn as nn

from dataset import NUM_CLASSES, NUM_SQUARES


def build_model(cfg: dict) -> nn.Module:
    """Build a ViT model from config with the chess classification head."""
    model = timm.create_model(
        cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=NUM_SQUARES * NUM_CLASSES,
    )

    if cfg["model"].get("freeze_backbone", False):
        # Freeze everything except the classification head
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    return model
