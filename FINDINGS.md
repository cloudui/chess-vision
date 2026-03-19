# Findings

Comprehensive record of what we learned building a chess board image → FEN predictor. Organized by topic rather than chronologically.

## Goal

Predict the full FEN (piece placement + turn to move + castling rights) from a 2D chess board screenshot image. The model should work across different piece styles, board color themes, highlights, and orientations — including styles not seen during training (OOD robustness). Current scope is clean digital screenshots; real-world photos (camera, book) are a future goal.

---

## Architecture Findings

### Three architectures tested

| Architecture | Params | In-Dist Board Acc | Kaggle OOD Board Acc | Key Strength | Key Weakness |
|---|---|---|---|---|---|
| **ViT** (ViT-B/16) | 86M | 100% | **82.6%** | Best OOD, attention captures global context | Slow, large, overfits OOD with more epochs |
| **CNN** (ConvNeXtV2-Tiny) | 28M | 100% | ~42-65% | Strong in-dist, fast | Catastrophic texture bias OOD — hallucinates pieces on empty squares |
| **Square** (MobileNetV4 per-square) | 2.9M | 100% | 67.5% | Tiny, 100% empty accuracy OOD, no hallucination | Limited cross-square context, king/queen confusion |

### CLS token vs spatial tokens (v1 → v2)

The original ViT used the CLS token to predict all 64 squares via a single `Linear(768, 832)`. This bottleneck forced the model to compress all spatial information into one vector.

Switching to spatial patch tokens pooled to an 8×8 grid (one feature vector per square) was transformative: piece accuracy jumped from ~89% to 100% in a single epoch. The per-square classification became trivially easy because each square gets its own dedicated 768-dim feature vector instead of sharing one.

### Type + color decomposition (v3)

Split the 13-class piece prediction into:
- **Type head**: 7 classes (empty, pawn, knight, bishop, rook, queen, king)
- **Color head**: 3 classes (empty, white, black)
- Combined additively: `joint[c] = type[type_of(c)] + color[color_of(c)]`

This forces the model to learn shape features independently from color features, doubling training examples per piece shape (white knight + black knight both train "knight"). Improved ViT OOD from 68.5% → 70.8%.

### Per-square architecture

Crops each of the 64 squares with 1.5× overlap (48px crop from 32px square) using border replication padding, resizes to 64×64, and classifies independently with a MobileNetV4 backbone. A linear bottleneck aggregator (64×feature_dim → 64) feeds turn/castling heads.

Key finding: **100% empty square accuracy on OOD**. The per-square approach never hallucinates pieces on empty squares, unlike the ViT and CNN which both suffer from this on unfamiliar board textures. The main weakness is king↔queen and knight↔rook confusion, which may be resolution-limited (32px squares upscaled to 64×64).

### CNN texture bias

The CNN (ConvNeXtV2-Tiny) matches or beats the ViT on in-distribution data but is dramatically worse OOD. On Kaggle, the CNN fills empty boards with repeating piece texture patterns (e.g., `QQQqQqQq/qBrQrQqQ/...`). This aligns with the well-documented CNN texture bias vs ViTs' shape bias in the literature.

Adding more piece styles (9→26) actually made the CNN's OOD worse (51.5% → 42.3%), suggesting more visual variety overwhelms texture-matching rather than helping generalization.

---

## Data & Augmentation Findings

### Training data composition matters more than quantity

| Dataset | Piece Styles | Textures | ViT OOD |
|---|---|---|---|
| 80k images, 9 styles, flat colors | 9 | 0 | 68.5% |
| 150k images, 26 styles, flat colors | 26 | 0 | 75.4% (epoch 3) |
| 50k images, 26 styles, 23 textures | 26 | 23 | **81-83%** |

Adding board textures (wood, marble, leather, etc.) was the single biggest OOD improvement. Going from 0 textures to 23 textures on only 50k images outperformed 150k images without textures by 6 percentage points.

### Aggressive augmentation hurts OOD

Tried extreme augmentations hoping to improve generalization:
- `hue=0.5` (full ±180° hue rotation)
- `RandomChannelPermutation` (swap RGB channels, p=0.2)
- `RandomInvert` (negate pixels, p=0.05)
- Always-on `GaussianBlur`

**Result**: In-distribution improved (turn 80→85%) but OOD dropped from 68.5% to 60.9%. The model hallucinated pieces on empty squares because extreme color transforms made board textures look like pieces. The empty square accuracy dropped from 99.99% to 98.94% — 11,458 false positives.

**Working augmentation set** (mild, realistic):
- `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)` — subtle theme variation
- `RandomGrayscale(p=0.1)` — forces shape recognition
- `GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))` with `p=0.2` — occasional softness

Removed: channel permutation, inversion, perspective distortion (not needed for clean screenshots).

### Class weights and label smoothing

- **Class weights** (inverse-sqrt-frequency): Tested but ultimately removed. Interacted poorly with label smoothing and didn't clearly help OOD.
- **Label smoothing** (0.1): Minimum achievable loss is ~0.5 per square even with perfect predictions. Makes loss numbers less interpretable but prevents overconfident predictions.

### Random board colors

50% of boards use one of 4 fixed color themes, 50% use randomly generated cohesive palettes. The random generator picks a single base hue and creates light/dark variants by blending toward white/dark — producing plausible pairs like "light tan / dark brown" rather than garish random RGB.

### Dataset size reduction

Rendering at 256px (model input size) instead of 480px + JPEG format instead of PNG reduced dataset size from ~6GB to ~500MB-1GB for 50k images. Textured boards were the main culprit at 480px PNG (~400KB each).

---

## Training Findings

### Piece accuracy converges in 1-3 epochs

With spatial tokens or per-square heads, piece placement reaches ~100% validation accuracy by epoch 1-3. The remaining training time is spent on turn/castling heads.

### OOD performance peaks early and degrades

Per-epoch OOD tracking on the 150k dataset showed dramatic degradation:
- Epoch 1: **80.2%** OOD
- Epoch 2: 63.7% OOD
- Epoch 3: 61.0% OOD

On the 50k dataset with textures, the pattern is less severe but still present:
- Epoch 3: 75.4% OOD → Epoch 9: 61.9% OOD (without textures)
- Epoch 4: 80.1% → Epoch 10: 81.1% (with textures — more stable)

The backbone overfits to training-distribution board textures, making features less general. The train/val metrics continue improving (turn, castling) while OOD silently degrades. Textures in the training set significantly reduce this effect.

**Implication**: OOD monitoring during training is essential. Currently tracking `board_acc` for best checkpoint, with `ood_val_dir` logging Kaggle OOD each epoch.

### Frozen backbone doesn't work

Tested freezing the ViT backbone and training only the heads (11,535 trainable params). Pretrained ImageNet features alone can't classify chess pieces well enough — val board_acc plateaued at 57%, OOD at 40.6%. The backbone needs fine-tuning, but fine-tuning causes style overfitting. This tension is unresolved.

### Validation transform bug

The original code created one dataset with `is_training=True` and split it into train/val. Both subsets received training augmentation. Fixed by creating two datasets with the same seeded split — train gets augmentation, val gets clean transforms.

### Turn prediction ceiling

Turn accuracy is capped at ~84-85% because turn is invisible metadata that can't be determined from the board position alone:
- **With last-move highlights**: ~100% accuracy (model reads highlight color)
- **Without highlights**: ~53-64% (near random)

This is correct behavior — a human also can't determine whose turn it is from a static board image without highlights.

### Best checkpoint metric evolution

| Metric | Problem |
|---|---|
| `square_acc` | Saturates at epoch 1 with spatial heads — best.pth is always epoch 1 |
| `full_fen_acc` | Captures turn/castling improvement but doesn't reflect OOD |
| `board_acc` | Current choice — reflects piece placement quality |

### LR and stability

- ViT with type+color decomposition sometimes shows training instability (loss spikes at epoch 5-6). Lower LR (5e-6 vs 4e-4) or the decomposition's constrained optimization landscape may contribute.
- The cosine schedule with 1 epoch warmup works well. 10 epochs is sufficient.
- For warm restarts: use `--reset-schedule --set training.lr=1e-4 scheduler.warmup_epochs=0`

---

## OOD Analysis (Kaggle)

### Main failure modes

1. **Color inversion** (~30% of errors): Complete white↔black flip — model gets piece type correct but inverts the color for every piece. Happens when the Kaggle board has unusual piece-to-background color relationships (e.g., dark red pieces on brown squares instead of the expected contrast).

2. **Knight↔Rook confusion** (~25% of errors): N→R: 854 errors, n→r: 917 errors in 10k Kaggle eval. Some Kaggle piece styles have knight and rook silhouettes that overlap more than the 26 training styles.

3. **Queen↔King confusion** (~20% of errors): K→Q: 288 errors, q→k: 225 errors. At 32px, the visual difference between a king's cross and queen's pointed crown is subtle on unfamiliar styles.

4. **Hallucinated pieces on empty squares** (CNN/ViT-specific): The CNN fills empty boards with repeating piece textures. The ViT does this less but still hallucinates on some OOD boards. The per-square model has 100% empty accuracy.

### What the Kaggle set reveals

The Kaggle boards are standard-looking chess boards with mostly normal pieces. The main OOD challenge is that **piece colors don't match board square colors** in the way our training data assumes. In our data, "black pieces on dark squares" always have a clear contrast relationship. Kaggle boards can have dark red pieces on brown squares, or white pieces on cream squares.

The remaining errors aren't really about resolution or architecture — they're about the model never having seen those specific piece renderings.

---

## Infrastructure Findings

### TensorBoard organization

All runs were initially written to a flat `runs/` directory, appearing as one run "." in TensorBoard. Fixed by creating timestamped subdirectories (`runs/20260226_143052/`). Per-step logging (loss, piece_loss, LR every batch) provides much more useful curves than epoch-only logging.

### OOD monitoring during training

Added `ood_val_dir` config option that loads a Kaggle subset and evaluates after each epoch. Logs `accuracy/board_ood` to TensorBoard. This was essential for discovering that OOD peaks early and degrades — without it, we'd have kept training to convergence and gotten worse OOD results.

### Run metadata

`run_meta.json` saved in checkpoint dir captures: full command, resolved config, git hash, git dirty state, dataset sizes, TensorBoard dir path, and final metrics. `eval_results.jsonl` accumulates eval results with timestamps. This makes it possible to trace any checkpoint back to the exact command and config that produced it.

### Dataset format

JPEG at 256px is ~15-20× smaller than PNG at 480px for textured boards. The `image_format` and `image_quality` fields in `dataset.yaml` make this configurable without code changes.

---

## Open Questions

- **Why does the model confuse pieces that look visually distinct to humans?** Knight↔rook and queen↔king confusion persists at ~81% OOD even though these shapes are unambiguous to a human at any resolution. The remaining errors may be concentrated on a small number of genuinely unusual Kaggle piece styles.
- **Is the Kaggle set the right benchmark?** Kaggle boards have unusual piece-to-board color relationships (dark red pieces on brown squares) that most real chess apps don't have. Real deployment targets (lichess, chess.com) use styles we already train on.
- **Can the per-square model close the gap with higher resolution?** At 32px per square, king/queen distinction is a few pixels. Testing input_size=400+ would give 50px squares (slight downscale to 64×64 instead of upscale).

## What's Next

1. **More piece styles** from other sources (chess.com, Wikipedia, etc.) — each genuinely novel style directly reduces failures on similar OOD styles
2. **Higher resolution for the square model** — test input_size=400 to see if finer piece details close the king/queen and knight/rook gap
3. **Two-phase training** — unfreeze backbone for 1 epoch, then freeze and train heads only. May preserve general features while learning chess-specific heads
4. **Real-world photos** — would require a board detection/perspective correction frontend and camera-specific augmentations (JPEG artifacts, noise, motion blur, moiré)
