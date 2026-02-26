# Experiment Log

## Project Goal

Predict the full FEN (piece placement + turn + castling) from a 2D chess board screenshot. The model should generalize across different piece styles, board color themes, highlights, and orientations — including styles not seen during training (OOD robustness).

## Architecture Evolution

### v1: CLS token + Linear (baseline)
- ViT-B/16 backbone, CLS token → `Linear(768, 832)` for all 64 squares
- Single vector bottleneck forces entire board through one 768-dim representation

### v2: Spatial tokens + per-square classifier
- ViT: patch tokens pooled to 8x8, shared `Linear(768, 13)` per square
- CNN (ConvNeXtV2-Tiny): native 8x8 feature map, `Conv2d(768, 13, 1x1)`
- 256x256 input for patch-to-square alignment (2x2 patches per square)
- Piece placement solved in 1 epoch (100% accuracy)

### v3: Type + color decomposition (current)
- Split piece classification into type head (7 classes) + color head (3 classes)
- Logits combined additively: `joint[c] = type[type_of(c)] + color[color_of(c)]`
- Forces shape features to be learned separately from color features
- Goal: improve OOD generalization by doubling examples per piece shape

## Training Data

- **Generated dataset**: 80k train / 20k test from Lichess PGN games
  - 9 piece styles, 10 fixed + random board colors, 50/50 flipped, 60% highlights
  - Manifest CSV with full FEN + metadata for eval grouping
- **Kaggle dataset** (OOD eval only): 20k test, different piece styles and textured boards
- **Random positions**: 2.5k test, random piece placement for robustness

## Results Summary

### v1: CLS token, no regularization (80k data, 20 epochs)

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 99.79% | — |
| Board acc | 88.79% | — |
| Turn | 84.94% | — |
| Castling | 99.36% | — |
| Full FEN | 75.56% | — |

Heavy overfitting: train loss 0.001, val loss 0.7. Turn prediction entirely relies on highlights (99.98% with, 64% without).

### v2: Spatial tokens + regularization (80k data, 10 epochs)

**ViT (no augmentation):**

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 98.67% |
| Board acc | 99.99% | 68.51% |
| Turn | 80.55% | — |
| Castling | 98.43% | — |
| Full FEN | 79.27% | — |

**ViT (with mild augmentation):**

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 97.50% |
| Board acc | 100.00% | 60.88% |
| Turn | 85.37% | — |
| Castling | 99.64% | — |
| Full FEN | 85.08% | — |

Augmentation improved in-distribution (turn 80→85%, full FEN 79→85%) but hurt OOD — aggressive augmentations (hue=0.5, channel permutation, inversion) caused hallucinated pieces on unfamiliar board textures.

**CNN (ConvNeXtV2-Tiny, latest checkpoint, 3 epochs):**

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 98.39% |
| Board acc | 100.00% | 64.54% |
| Turn | 83.75% | — |
| Castling | 96.30% | — |
| Full FEN | 80.77% | — |

CNN matches ViT on pieces, slightly worse on turn/castling (global avg pool vs CLS attention). 3x fewer parameters (28M vs 86M).

## Key Findings

1. **Piece placement is solved** with spatial token/conv approach — 100% on in-distribution after 1 epoch
2. **Turn prediction is limited by highlights** — ~100% with highlights, ~52-64% without (turn is invisible metadata, can't be determined from position alone)
3. **OOD gap is driven by piece style variety** — the model has seen 9 styles, Kaggle has different ones. Color inversion and knight↔rook confusion are the main failure modes
4. **Aggressive augmentation hurts** — hue=0.5, channel permutation, and inversion created unrealistic training images that degraded OOD performance. Dialed back to mild augmentation (hue=0.1, blur p=0.2, perspective p=0.2)
5. **Best checkpoint metric matters** — `square_acc` saturates at epoch 1, switched to `full_fen_acc` to capture turn/castling improvements
6. **CNN is viable** but ViT is better for global properties (turn/castling) due to self-attention

## Current Configuration

- **Regularization**: head dropout 0.1, drop_path_rate 0.1, label smoothing 0.1, weight decay 0.05 (devserver)
- **Class weights**: inverse-sqrt-frequency from manifest FENs
- **Augmentation**: ColorJitter (hue=0.1), RandomGrayscale (10%), GaussianBlur (20%), RandomPerspective (20%)
- **Schedule**: 10 epochs, cosine LR, 1 epoch warmup, early stopping patience 3
- **Data generation**: 50% random board colors, 50/50 flipped, 60% highlights

## What's Next

- Training with type+color decomposition to see if OOD improves
- Adding more piece styles to the data generator
- Board texture variation (wood grain, marble) in renderer
- Potentially mixing Kaggle training data for OOD exposure
