# Session Continuation Prompt

Use this prompt to continue development if starting a new session.

---

I'm building a chess board image → FEN prediction model. The codebase is at the current working directory. Read `EXPERIMENTS.md` for full context on what's been tried and results.

## Current state

The repo has two model architectures (ViT and CNN) that predict piece placement, turn, and castling from 2D chess board screenshots. Both achieve 100% piece accuracy on in-distribution test data (80k generated training images, 9 piece styles, various board colors). The main challenge is **OOD generalization** — on the Kaggle dataset (different piece styles and textured boards), accuracy drops to ~60-68% full-board.

## What was just implemented (not yet trained/tested)

1. **Type + color decomposition** (`model.py`): The 13-class piece prediction is decomposed into a 7-class type head (empty/pawn/knight/bishop/rook/queen/king) and a 3-class color head (empty/white/black). Logits are additively combined to produce 13-class output. The external interface (B, 832) is unchanged so train/eval/predict don't need modifications.

2. **Revised augmentation** (`dataset.py`): Dialed back from aggressive augmentations that hurt OOD. Current: ColorJitter(hue=0.1), RandomGrayscale(10%), GaussianBlur(20%), RandomPerspective(20%). Removed: channel permutation, inversion, extreme hue shifts.

3. **Random board colors** (`datagen/render.js`): 50% of generated boards use random RGB color pairs instead of 10 fixed themes. Requires dataset regeneration.

## Key files

- `model.py` — ChessViT + ChessCNN with type/color decomposition, build_model dispatcher
- `dataset.py` — ChessDataset (manifest auto-detect), transforms, FEN parsing
- `train.py` — run_epoch, class weights, training loop
- `evaluate.py` — full eval with grouped metrics by style/flipped/highlight/etc.
- `predict.py` — single image inference
- `datagen/` — Node.js pipeline: generate.js, positions.js, render.js, dataset.yaml
- `config.yaml` / `config_devserver.yaml` — ViT configs (local / GPU)
- `config_cnn.yaml` / `config_cnn_devserver.yaml` — CNN configs
- `EXPERIMENTS.md` — experiment log with all results

## Next steps to try

1. Train both ViT and CNN with type+color decomposition, eval on test + Kaggle OOD
2. Regenerate dataset with random board colors, retrain
3. Add more piece styles to `datagen/` (find additional SVG/PNG piece sets)
4. Add board textures (wood grain, marble) to renderer
5. Consider mixing Kaggle training data for direct OOD exposure
