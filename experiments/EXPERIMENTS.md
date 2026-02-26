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
- Doubles training examples per piece shape (white knight + black knight both train "knight")

## Training Data

- **Generated dataset**: 80k train / 20k test from Lichess PGN games
  - Originally 9 piece styles, now 26 styles available after downloading lichess sets
  - 10 fixed + 50% random cohesive board colors, 50/50 flipped, 60% highlights
  - Manifest CSV with full FEN + metadata for eval grouping
- **Kaggle dataset** (OOD eval only): 20k test, different piece styles and textured boards
- **Random positions**: 2.5k test, random piece placement for robustness

## Results — Sequential Experiments

### Experiment 1: CLS token baseline (v1, 80k data, 20 epochs, no regularization)

| Metric | Test |
|---|---|
| Per-square | 99.79% |
| Board acc | 88.79% |
| Turn | 84.94% |
| Castling | 99.36% |
| Full FEN | 75.56% |

Heavy overfitting: train loss 0.001, val loss 0.7. Turn prediction entirely relies on highlights (99.98% with, 64% without). No Kaggle OOD eval was done at this point.

### Experiment 2: Spatial tokens + regularization (v2, 80k data)

Added dropout, drop_path, label smoothing, class weights, 256x256 input. Switched from CLS token to spatial per-square classification. Piece placement jumped to 100% after 1 epoch.

**ViT (no augmentation, 10 epochs):**

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 98.67% |
| Board acc | 99.99% | 68.51% |
| Turn | 80.55% | — |
| Castling | 98.43% | — |
| Full FEN | 79.27% | — |

First OOD eval on Kaggle revealed major gap (68.5% board). Main failure modes: color inversion (white↔black) and knight↔rook confusion on unseen piece styles.

### Experiment 3: Aggressive augmentation (v2, 80k data, 10 epochs)

Added hue=0.5, channel permutation (20%), random inversion (5%), always-on Gaussian blur.

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 97.50% |
| Board acc | 100.00% | 60.88% |
| Turn | 85.37% | — |
| Castling | 99.64% | — |
| Full FEN | 85.08% | — |

**Lesson learned**: Augmentation improved in-distribution (turn 80→85%, full FEN 79→85%) but *hurt* OOD (68.5→60.9%). The model hallucinated pieces on empty squares because extreme color transforms made board textures look like pieces. Dialed back to mild augmentation: hue=0.1, blur p=0.2, removed channel permutation and inversion.

### Experiment 4: CNN comparison (v2, ConvNeXtV2-Tiny, 80k data)

Added CNN architecture alongside ViT for comparison. CNN uses native 8x8 feature map from stride-32 backbone with 1x1 conv classification. 28M params vs ViT's 86M.

**Important lesson**: initial CNN eval used `best.pth` which was the epoch 1 checkpoint (selected by `square_acc` which saturated at epoch 1). This made CNN look much worse than it was. Switched best checkpoint metric from `square_acc` to `full_fen_acc`.

**CNN (latest checkpoint, 3 epochs):**

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 98.39% |
| Board acc | 100.00% | 64.54% |
| Turn | 83.75% | — |
| Castling | 96.30% | — |
| Full FEN | 80.77% | — |

CNN matches ViT on piece placement. Competitive on turn/castling when given enough epochs. Worse on OOD (64.5% vs 68.5%) — CNNs have a known texture bias vs ViTs' shape bias.

### Experiment 5: Type + color decomposition (v3, 80k data, 10 epochs + warm restart)

Split 13-class piece head into 7-class type head + 3-class color head. Logits combined additively. External interface unchanged (still outputs 832 joint logits).

**ViT with type+color decomposition:**

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 98.54% |
| Board acc | 100.00% | **70.75%** |
| Turn | 85.37% | — |
| Castling | 99.64% | — |
| Full FEN | 85.08% | — |

OOD improved from 68.5% → 70.8% board accuracy. Pawn accuracy jumped (95.7→99.0%), queen accuracy improved (77.1→84.0%). The decomposition helps by forcing shape and color features to be learned independently.

Converged after ~10 epochs + warm restart. Turn plateaued at ~84% (highlight ceiling: 100% with highlights, ~53-64% without).

**CNN with type+color decomposition:**

| Metric | Test | Kaggle OOD |
|---|---|---|
| Per-square | 100.00% | 95.97% |
| Board acc | 100.00% | 51.45% |
| Turn | 85.15% | — |
| Castling | 99.69% | — |
| Full FEN | 84.91% | — |

CNN beat ViT on in-distribution metrics (turn 85.2% vs 83.9%, full FEN 84.9% vs 83.5%) but performed much worse on OOD (51.5% vs 70.8%). The CNN catastrophically hallucinated pieces on empty squares for unfamiliar board styles — a structured pattern of filling boards with repeating piece textures. This aligns with the well-documented CNN texture bias: the CNN learned texture patterns for pieces that break on OOD board styles.

## Key Findings

1. **Piece placement is solved** with spatial token/conv approach — 100% on in-distribution after 1 epoch for both architectures
2. **Turn prediction is limited by highlights** — ~100% with highlights, ~53-64% without (turn is invisible metadata, can't be determined from the position alone)
3. **OOD gap is driven by piece style variety** — the model has seen 9 styles, Kaggle has different ones. Color inversion and knight↔rook confusion are the main failure modes
4. **Aggressive augmentation hurts OOD** — unrealistic transforms (extreme hue, channel permutation, inversion) cause hallucinated pieces on unfamiliar board textures. Mild augmentation is better
5. **Best checkpoint metric matters** — `square_acc` saturates at epoch 1, switched to `full_fen_acc`
6. **Type+color decomposition helps OOD** — forces shape-independent features, improved Kaggle from 68.5→70.8% for ViT
7. **CNN has texture bias** — strong in-distribution (beats ViT on turn/castling) but catastrophically worse OOD due to texture-dependent features. May improve with more training styles
8. **Both models converge fast** — 10 epochs is sufficient, reduced from 20. Cosine schedule with 1 epoch warmup

## Current Configuration

- **Architecture**: v3 type+color decomposition (both ViT and CNN)
- **Regularization**: head dropout 0.1, drop_path_rate 0.1, label smoothing 0.1, weight decay 0.05 (devserver)
- **Class weights**: inverse-sqrt-frequency from manifest FENs
- **Augmentation**: ColorJitter (brightness/contrast/sat=0.3, hue=0.1), RandomGrayscale (10%), GaussianBlur (20%, sigma 0.1-1.5)
- **Schedule**: 10 epochs, cosine LR, 1 epoch warmup, early stopping patience 3
- **Piece styles**: 26 available (12 bundled + 14 downloaded from lichess)
- **Data generation**: 50% random cohesive board colors, 50/50 flipped, 60% highlights

## What's Next

- Regenerate dataset with 26 piece styles (up from 9) and 150k+ training images
- Retrain both ViT and CNN to see if more styles close the OOD gap (especially for CNN)
- Add board textures (wood grain, marble) to renderer
- Potentially mix Kaggle training data for direct OOD exposure
