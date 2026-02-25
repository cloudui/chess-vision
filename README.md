# Chess Image to FEN

Predicts the [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) board position from an image of a chess board using a fine-tuned Vision Transformer (ViT).

The model classifies each of the 64 squares into one of 13 classes (empty + 6 white pieces + 6 black pieces) and reconstructs the FEN string from the predictions.

## Setup

```bash
conda create -n chess_vision python=3.10
conda activate chess_vision
conda install pytorch torchvision -c pytorch
pip install timm pyyaml tqdm tensorboard matplotlib
```

## Dataset

Uses the [Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions) dataset from Kaggle (80k train / 20k test, 400x400 JPEG images). The FEN is encoded in each filename with `-` as the rank separator.

Place the downloaded data so the directory structure looks like:

```
kaggle/
├── train/    # 80,000 images
└── test/     # 20,000 images
```

## Usage

### Train

```bash
# Full training run with defaults from config.yaml
python train.py

# Quick run with overrides
python train.py --set data.max_samples=1000 training.epochs=5

# Resume from checkpoint
python train.py --resume checkpoints/latest.pth
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pth
```

Reports per-square accuracy, full-board accuracy, per-piece accuracy, confusion matrix, and worst predictions.

### Predict

```bash
python predict.py --checkpoint checkpoints/best.pth --image path/to/board.jpeg
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

## Project Structure

```
├── config.yaml        # Hyperparameters and paths
├── dataset.py         # Dataset class and FEN parsing
├── model.py           # Model construction
├── train.py           # Training loop
├── evaluate.py        # Test set evaluation
├── predict.py         # Single-image inference
├── visualize.ipynb    # Data visualization notebook
├── checkpoints/       # Saved model weights
└── runs/              # TensorBoard logs
```

## Model

`vit_base_patch16_224.augreg_in21k` from [timm](https://github.com/huggingface/pytorch-image-models) — a ViT-B/16 pretrained on ImageNet-21k with augmentation and regularization. The classification head is replaced with a linear layer outputting 64 × 13 = 832 values (one 13-class prediction per square).
