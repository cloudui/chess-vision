# Chess Image to FEN

Predicts the full [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) from an image of a chess board. Supports three model architectures for comparison.

## Models

| Architecture | Config | Params | Description |
|---|---|---|---|
| **ViT** (`arch: vit`) | `config_devserver.yaml` | 86M | ViT-B/16 with spatial token heads pooled to 8x8 |
| **CNN** (`arch: cnn`) | `config_cnn_devserver.yaml` | 28M | ConvNeXtV2-Tiny with 1x1 conv heads on 8x8 feature map |
| **Square** (`arch: square`) | `config_square_devserver.yaml` | 2.9M | Per-square MobileNetV4 crops with neighbor overlap |

All models predict:
- **Piece placement** -- type (7 classes) + color (3 classes) decomposed heads, combined into 13-class output
- **Turn to move** -- binary (white / black)
- **Castling rights** -- four binary predictions (K, Q, k, q)

## Setup

```bash
conda create -n chess_vision python=3.10
conda activate chess_vision
conda install pytorch torchvision -c pytorch
pip install timm pyyaml tqdm tensorboard matplotlib
```

## Dataset

### Generated dataset (primary)

The `datagen/` pipeline renders board images from real game positions extracted from PGN files. 26 piece styles, 23 board textures, random board colors, highlights, and flipped orientations. Output is 256x256 JPEG with a manifest CSV for labels and eval grouping.

```bash
cd datagen

# Download piece styles from lichess (14 additional styles)
node download_pieces.js

# Download board textures (wood, marble, leather, etc.)
node download_boards.js

# Generate train/test images
node generate.js --config dataset.yaml
```

Output structure:

```
data/
├── train/
│   ├── 000000.jpg
│   ├── ...
│   └── manifest.csv
├── test/
│   ├── ...
│   └── manifest.csv
└── test_random/
    ├── ...
    └── manifest.csv
```

Rendering config (`datagen/dataset.yaml`):
- `image_size`: render resolution (default 256)
- `image_format`: `jpeg` or `png` (default jpeg)
- `image_quality`: JPEG quality 1-100 (default 90)
- `highlight_pct`: fraction of boards with last-move highlights (default 0.6)
- `texture_pct`: fraction using texture backgrounds vs flat colors (default 0.5)

`ChessDataset` auto-detects `manifest.csv` inside the data directory. Falls back to parsing FEN from filenames (Kaggle format) if no manifest found.

### Kaggle dataset (OOD eval)

The [Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions) dataset (80k train / 20k test). Used as an out-of-distribution test set -- different piece styles and textured boards not seen during training.

## Usage

### Train

```bash
# ViT (default)
python train.py --config config_devserver.yaml

# CNN
python train.py --config config_cnn_devserver.yaml

# Per-square
python train.py --config config_square_devserver.yaml

# Quick smoke test
python train.py --config config_square.yaml --set training.epochs=2 data.max_samples=50

# Resume from checkpoint
python train.py --config config_devserver.yaml --resume checkpoints/latest.pth

# Warm restart with lower LR
python train.py --config config_devserver.yaml --resume checkpoints/latest.pth --reset-schedule --set training.lr=1e-4
```

### Evaluate

```bash
# In-distribution test set
python evaluate.py --checkpoint checkpoints/best.pth

# OOD evaluation on Kaggle
python evaluate.py --checkpoint checkpoints/best.pth --test-dir kaggle/test

# Limit sample count for quick checks
python evaluate.py --checkpoint checkpoints/best.pth --max-samples 3000
```

### Predict

```bash
python predict.py --checkpoint checkpoints/best.pth --image path/to/board.jpg
# Output: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq
```

### Monitor Training

```bash
tensorboard --logdir runs/
```

Each training run creates a timestamped subdirectory. OOD validation (Kaggle) is logged as `accuracy/board_ood` when `ood_val_dir` is set in config.

## Configuration

All hyperparameters are in config YAML files. Any value can be overridden from the command line:

```bash
python train.py --set training.lr=3e-4 training.batch_size=128 training.turn_loss_weight=0
```

Config files:
- `config.yaml` / `config_devserver.yaml` -- ViT
- `config_cnn.yaml` / `config_cnn_devserver.yaml` -- CNN
- `config_square.yaml` / `config_square_devserver.yaml` -- Per-square

## Project Structure

```
├── config*.yaml               # Training configs (local + devserver for each arch)
├── dataset.py                 # Dataset class, FEN parsing, transforms
├── models/                    # Model architectures
│   ├── __init__.py            # build_model() dispatcher
│   ├── common.py              # Shared type+color decomposition logic
│   ├── vit.py                 # ChessViT (ViT-B/16 + spatial token heads)
│   ├── cnn.py                 # ChessCNN (ConvNeXtV2 + 1x1 conv heads)
│   └── square.py              # ChessSquareCNN (per-square MobileNetV4)
├── train.py                   # Training loop with OOD monitoring
├── evaluate.py                # Test set evaluation with grouped metrics
├── predict.py                 # Single-image inference
├── visualize.ipynb            # Data and augmentation visualization
├── datagen/                   # Node.js data generation pipeline
│   ├── dataset.yaml           # Generation config
│   ├── generate.js            # Main generator (multi-threaded)
│   ├── render.js              # Board rendering (flat colors + textures)
│   ├── render-worker.js       # Worker thread for parallel rendering
│   ├── positions.js           # PGN position extraction
│   ├── rand.js                # Seeded PRNG utilities
│   ├── download_pieces.js     # Fetch lichess piece SVGs → PNG
│   ├── download_boards.js     # Fetch lichess board textures
│   ├── preview_styles.js      # Generate piece style preview grid
│   ├── preview_textures.js    # Generate board texture preview grid
│   └── preview_colors.js      # Generate board color preview grid
├── EXPERIMENTS.md             # Full experiment log with results
├── CONTINUATION_PROMPT.md     # Session handoff prompt
├── FRESH_REVIEW_PROMPT.md     # Independent review prompt
├── checkpoints*/              # Saved model weights (per architecture)
└── runs*/                     # TensorBoard logs (per architecture)
```

## Logging

- **TensorBoard**: per-step loss/LR curves, per-epoch accuracy (train/val/OOD)
- **`run_meta.json`**: saved in checkpoint dir with command, config, git hash, final metrics
- **`eval_results.jsonl`**: eval results appended per run with timestamp, checkpoint, test dir, metrics
