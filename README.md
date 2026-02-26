# Chess Image to FEN

Predicts the full [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) from an image of a chess board using a fine-tuned Vision Transformer (ViT).

The model has three heads sharing a single ViT backbone:

- **Piece placement** -- classifies each of the 64 squares into one of 13 classes (empty + 6 white + 6 black pieces)
- **Turn to move** -- binary prediction (white / black)
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

The `datagen/` pipeline renders board images from real game positions extracted from PGN files. It produces a manifest CSV alongside the images with full FEN labels and metadata columns for eval grouping (piece count, style, highlight, flipped, etc.).

```bash
# Download a Lichess PGN (script in datagen/)
cd datagen && bash download_pgn.sh

# Generate train/test images
node generate.js --config dataset.yaml
```

Output structure:

```
data/
├── train/
│   ├── 000000.png
│   ├── ...
│   └── manifest.csv
├── test/
│   ├── ...
│   └── manifest.csv
└── test_random/       # random positions for robustness eval
    ├── ...
    └── manifest.csv
```

`ChessDataset` auto-detects `manifest.csv` inside the data directory. If no manifest is found, it falls back to parsing the FEN from filenames (legacy Kaggle format).

### Kaggle dataset (legacy)

The [Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions) dataset (80k train / 20k test, 400x400 JPEG). FEN is encoded in each filename with `-` as the rank separator. This dataset has piece placement only -- no turn or castling labels.

## Usage

### Train

```bash
# Full training run with defaults from config.yaml
python train.py

# Quick smoke test
python train.py --set training.epochs=2 data.max_samples=50

# Resume from checkpoint
python train.py --resume checkpoints/latest.pth

# Warm restart (keep weights, reset optimizer/scheduler)
python train.py --resume checkpoints/best.pth --reset-schedule

# Use devserver config (larger batch size, more workers)
python train.py --config config_devserver.yaml
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pth

# Override test directory
python evaluate.py --checkpoint checkpoints/best.pth --test-dir data/test_random
```

Reports per-square accuracy, full-board accuracy, turn/castling accuracy, full FEN accuracy, per-piece breakdown, confusion matrix, worst predictions, and grouped metrics by manifest fields (piece count, style, etc.).

### Predict

```bash
python predict.py --checkpoint checkpoints/best.pth --image path/to/board.png
# Output: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq
```

### Monitor Training

```bash
tensorboard --logdir runs/
```

## Configuration

All hyperparameters are in `config.yaml`. Any value can be overridden from the command line:

```bash
python train.py --set training.lr=3e-4 training.batch_size=128 scheduler.warmup_epochs=1
```

`config_devserver.yaml` has settings tuned for a GPU devserver (larger batch size, more workers).

## Project Structure

```
├── config.yaml            # Hyperparameters and paths (local)
├── config_devserver.yaml   # Config for GPU devserver
├── dataset.py             # Dataset class, FEN parsing, transforms
├── model.py               # Multi-head ViT model
├── train.py               # Training loop
├── evaluate.py            # Test set evaluation with grouped metrics
├── predict.py             # Single-image inference
├── visualize.ipynb        # Data visualization notebook
├── datagen/               # Node.js data generation pipeline
│   ├── dataset.yaml       # Generation config (splits, counts, rendering)
│   ├── generate.js        # Main generator entry point
│   ├── positions.js       # PGN position extraction
│   ├── rand.js            # Random position generation
│   └── render.js          # Board rendering
├── checkpoints/           # Saved model weights
└── runs/                  # TensorBoard logs
```

## Model

`vit_base_patch16_224.augreg_in21k` from [timm](https://github.com/huggingface/pytorch-image-models) -- a ViT-B/16 pretrained on ImageNet-21k with augmentation and regularization. The default classification head is replaced with three task-specific heads (piece placement, turn, castling) fed from the CLS token.
