# Experiment Log — 26-Style Dataset (150k train)

All experiments use the 26-piece-style, 150k-sample dataset with varied board colors.
OOD evaluation is on Kaggle test set (20k images, unseen piece styles and textured boards).

## Summary Table

| # | Experiment | Epochs | Config Overrides | Val Board Acc | **Kaggle OOD Board Acc** | Notes |
|---|---|---|---|---|---|---|
| 01 | ViT baseline | 3 (of 10) | — | 98.46% | 75.4% | Interrupted at epoch 4 |
| 02 | ViT baseline | 9 (of 10) | — | ~99.7% | 61.9% | OOD regressed from overfitting |
| 03 | CNN baseline | 3 (of 10) | `config_cnn_devserver.yaml` | 99.43% | 42.3% | CNN texture bias, mass hallucination |
| 04 | ViT frozen backbone | 10 | `freeze_backbone=true, label_smoothing=0, class_weights=false, turn/castling=0` | 57.19% | 40.6% | Pretrained features insufficient alone |
| 05a | ViT per-epoch OOD: epoch 1 | 1 | — | ~96.5% | **80.2%** | OOD peaks at epoch 1 |
| 05b | ViT per-epoch OOD: epoch 2 | 2 | — | ~98.8% | 63.7% | Massive OOD drop, hallucination begins |
| 05c | ViT per-epoch OOD: epoch 3 | 3 | — | ~99.3% | 61.0% | Continued degradation |
| 06 | ViT resumed to 10 epochs | 10 | resumed from exp 05 epoch 1 | 99.75% | 61.4% | Same overfitting pattern |
| 07 | ViT 3 epochs (best run) | 3 | — | ~98.3% | **82.6%** | Current best OOD result |

## Key Findings

### OOD peaks early, then degrades
- Epoch 1: 80.2% OOD → Epoch 2: 63.7% → Epoch 3: 61.0%
- The backbone overfits to training styles within 1-2 epochs
- Longer training improves in-distribution (turn, castling) but destroys OOD

### CNN has severe texture bias
- CNN (ConvNeXtV2-Tiny) achieves 42.3% OOD vs ViT's 75-82%
- Mass hallucination: filling empty squares with phantom pieces
- CNN learns board texture patterns, breaks on unseen textures

### Frozen backbone is insufficient
- Only 11,535 trainable parameters (heads only)
- Pretrained ImageNet features can't classify chess pieces well enough
- Val board_acc plateaued at 57%, OOD at 40.6%

### Best result: ViT, 3 epochs, default config
- 82.6% Kaggle OOD board accuracy
- Remaining errors are mostly knight confusion (N↔R) and queen confusion (Q↔K)
- Worst prediction: 11/64 squares wrong (clean errors, no hallucination)
- Empty square accuracy: 99.93% (no hallucination)

## Experiment Details

### Default ViT config (`config_devserver.yaml`)
```yaml
model:
  name: vit_base_patch16_224.augreg_in21k
  pretrained: true
  freeze_backbone: false
  input_size: 256
  head_dropout: 0.1
  drop_path_rate: 0.1

training:
  epochs: 10
  batch_size: 256
  lr: 4.0e-4
  weight_decay: 0.05
  label_smoothing: 0.1
  use_class_weights: true
  turn_loss_weight: 1.0
  castling_loss_weight: 1.0
```

### CNN config (`config_cnn_devserver.yaml`)
```yaml
model:
  arch: cnn
  name: convnextv2_tiny.fcmae_ft_in22k_in1k
  input_size: 256
  head_dropout: 0.1
  drop_path_rate: 0.1

training:
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 0.01
```

## Commands

```bash
# Experiment 01: ViT baseline 3 epochs
CUDA_VISIBLE_DEVICES=0 python train.py --config config_devserver.yaml

# Experiment 03: CNN baseline
CUDA_VISIBLE_DEVICES=1 python train.py --config config_cnn_devserver.yaml

# Experiment 04: ViT frozen backbone
python train.py --config config_devserver.yaml \
  --set training.epochs=10 \
  model.freeze_backbone=true \
  training.label_smoothing=0 \
  training.use_class_weights=false \
  training.turn_loss_weight=0 \
  training.castling_loss_weight=0

# Experiment 06: Resume from epoch 1
CUDA_VISIBLE_DEVICES=1 python train.py --config config_devserver.yaml \
  --resume checkpoints2/latest.pth

# Kaggle OOD eval
python evaluate.py --checkpoint checkpoints2/best.pth --test-dir kaggle/test
```

## Log Files

Full training and evaluation logs are in `experiment_logs/`:
- `01_vit_baseline_3ep_{train,eval_kaggle}.log`
- `02_vit_baseline_9ep_{train,eval_kaggle}.log`
- `03_cnn_baseline_3ep_{train,eval_kaggle}.log`
- `04_vit_frozen_10ep_{train,eval_kaggle}.log`
- `05_vit_per_epoch_ood_{train,eval_kaggle}.log`
- `06_vit_resumed_10ep_{train,eval_kaggle}.log`
- `07_vit_3ep_best_eval_kaggle.log`

## Next Steps

- Try lower learning rate (1e-5) to slow backbone overfitting, allow more epochs
- Try removing label smoothing and class weights (untested with this dataset)
- Two-phase training: 1 epoch unfrozen, then freeze backbone + train heads
- Add board textures to renderer for OOD data diversity
