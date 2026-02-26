# Fresh Review Prompt

Use this prompt with a new Claude session for an independent architectural and training review.

---

I have a chess board image → FEN prediction project. I want a fresh, objective review of the codebase — architecture, training pipeline, data generation, and augmentation — to identify improvements I might be missing.

## Task

Given a clean 2D screenshot of a chess board (from any chess app or website), predict the full FEN: piece placement (which piece on which square), turn to move (white/black), and castling rights (KQkq).

The input images are always:
- Perfect digital renders (no camera photos, no perspective distortion)
- Square boards filling the entire image (no borders, UI elements, or cropping needed)
- Various piece styles, board color themes, and orientations (normal or flipped)
- Some have last-move highlights, some don't

## What to review

Please read through the codebase and evaluate:

1. **Architecture** (`model.py`): Are the model architectures well-designed for this task? Are there better approaches for the spatial piece classification, the type+color decomposition, or the global heads (turn/castling)? Consider both the ViT and CNN variants.

2. **Training pipeline** (`train.py`): Is the training loop, loss computation, class weighting, and scheduler setup sound? Any issues with the optimization strategy?

3. **Data generation** (`datagen/`): Is the rendering pipeline producing good training data? Are there gaps in the visual variety? Look at `render.js` for how boards are rendered and `dataset.yaml` for the generation config.

4. **Augmentation** (`dataset.py`): Are the training transforms appropriate for this task? Too aggressive? Too mild? Missing something important?

5. **Evaluation** (`evaluate.py`): Is the evaluation comprehensive? Are we measuring the right things?

6. **Dataset design** (`dataset.py`): Is the dataset class well-structured? Any issues with how labels are parsed or data is loaded?

## Key constraint

The current OOD challenge: the model trains on generated images with known piece styles but needs to generalize to unseen piece styles and board themes (e.g., the Kaggle chess positions dataset which has different piece art and textured backgrounds). In-distribution accuracy is essentially 100% for pieces. The gap is entirely on out-of-distribution visual styles.

## Files to read

Start with these in order:
- `EXPERIMENTS.md` — experiment log (read for context but form your own conclusions)
- `model.py` — both architectures
- `train.py` — training loop
- `dataset.py` — data loading and augmentation
- `evaluate.py` — evaluation
- `datagen/render.js` — board rendering
- `datagen/dataset.yaml` — generation config
- `config_devserver.yaml` — training hyperparameters

Don't read `CONTINUATION_PROMPT.md` — that has my prior conclusions which I want you to independently validate or challenge.
