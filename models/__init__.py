import torch.nn as nn

from .vit import ChessViT, build_vit
from .cnn import ChessCNN, build_cnn
from .square import ChessSquareCNN, build_square


def build_model(cfg: dict) -> nn.Module:
    """Build a chess recognition model from config.

    Set ``cfg["model"]["arch"]`` to:
      - ``"vit"``    — ViT with spatial token heads (default)
      - ``"cnn"``    — stride-32 CNN with 1x1 conv heads
      - ``"square"`` — per-square crop CNN classifier
    """
    model_cfg = cfg["model"]
    arch = model_cfg.get("arch", "vit")

    builders = {
        "vit": build_vit,
        "cnn": build_cnn,
        "square": build_square,
    }

    if arch not in builders:
        raise ValueError(
            f"Unknown architecture: {arch!r} (expected one of {list(builders.keys())})"
        )

    return builders[arch](model_cfg)
