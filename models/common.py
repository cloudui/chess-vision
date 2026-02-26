import torch

from dataset import (
    NUM_CLASSES, NUM_SQUARES,
    NUM_PIECE_TYPES, NUM_PIECE_COLORS,
    CLASS_TO_TYPE, CLASS_TO_COLOR,
)


def combine_type_color(type_logits, color_logits, class_to_type, class_to_color):
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


def register_type_color_buffers(module):
    """Register CLASS_TO_TYPE and CLASS_TO_COLOR as model buffers."""
    module.register_buffer("class_to_type", torch.tensor(CLASS_TO_TYPE, dtype=torch.long))
    module.register_buffer("class_to_color", torch.tensor(CLASS_TO_COLOR, dtype=torch.long))
