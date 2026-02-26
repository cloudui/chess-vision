# Session Continuation Prompt

Use this prompt to continue development if starting a new session.

---

I'm building a chess board image → FEN prediction model. The codebase is at the current working directory. Read `EXPERIMENTS.md` for the full experiment log with sequential results.

## Current state

The repo has two model architectures (ViT-B/16 and ConvNeXtV2-Tiny CNN) that predict piece placement, turn, and castling from 2D chess board screenshots. Both use type+color decomposition: a 7-class type head + 3-class color head whose logits combine additively into 13-class piece predictions. This forces shape features to be learned independently from color features.

Both achieve 100% piece accuracy and ~85% turn/full FEN on in-distribution test data (80k generated training images, 9 piece styles). The main challenge is **OOD generalization** — on the Kaggle dataset (different piece styles and textured boards):
- ViT: 70.8% full-board accuracy (main errors: color inversion, knight↔rook confusion)
- CNN: 51.5% full-board accuracy (catastrophic hallucination of pieces on empty squares due to texture bias)

## What needs to happen next

1. **Regenerate dataset with 26 piece styles** — downloaded 14 lichess piece sets (via `datagen/download_pieces.js`), render.js auto-discovers them. Bump `dataset.yaml` train count to 150k+ since more styles need more examples.

2. **Retrain both models** on the new dataset and eval on Kaggle OOD to measure improvement.

3. **Consider board textures** — the Kaggle boards have wood grain/marble textures that our flat-color renderer doesn't produce. Could add texture support to render.js.

4. **Consider mixing Kaggle training data** — 80k Kaggle images with piece-placement-only labels could be mixed in for direct OOD exposure (would need to handle missing turn/castling labels in training loop).

## Key files

- `model.py` — ChessViT + ChessCNN with type/color decomposition, build_model dispatcher (`arch: vit` or `arch: cnn`)
- `dataset.py` — ChessDataset (manifest auto-detect), transforms, FEN parsing, type/color constants
- `train.py` — run_epoch, class weights, training loop (best checkpoint by `full_fen_acc`)
- `evaluate.py` — full eval with grouped metrics by style/flipped/highlight/etc.
- `predict.py` — single image inference
- `datagen/` — Node.js pipeline: generate.js, positions.js, render.js, dataset.yaml
- `datagen/download_pieces.js` — downloads lichess piece SVGs, converts to 80x80 PNGs
- `config.yaml` / `config_devserver.yaml` — ViT configs (local / GPU)
- `config_cnn.yaml` / `config_cnn_devserver.yaml` — CNN configs
- `EXPERIMENTS.md` — full experiment log with all results

## Important lessons learned (read EXPERIMENTS.md for details)

- Aggressive augmentation (extreme hue, channel permutation, inversion) hurts OOD — stick to mild augmentation
- `square_acc` saturates at epoch 1 with spatial heads — use `full_fen_acc` for best checkpoint
- CNN beats ViT in-distribution but is much worse OOD (texture bias vs shape bias)
- Turn accuracy ceiling is ~84-85% due to highlight dependency (100% with highlights, ~53-64% without)
- Type+color decomposition improved ViT OOD from 68.5% → 70.8%
- Both models converge in ~10 epochs with cosine schedule
