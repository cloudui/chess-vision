#!/usr/bin/env node
/**
 * Render board texture previews with pieces.
 * Outputs to datagen/texture_preview.png
 */

const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const { renderBoard, preloadPieceImages, BOARD_TEXTURES } = require('./render');

const FEN = 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R';
const BOARD_SIZE = 200;
const COLS = 6;
const LABEL_HEIGHT = 20;

async function main() {
  if (BOARD_TEXTURES.length === 0) {
    console.log('No board textures found in boards/ directory');
    process.exit(1);
  }

  console.log(`Found ${BOARD_TEXTURES.length} textures: ${BOARD_TEXTURES.join(', ')}`);

  const rows = Math.ceil(BOARD_TEXTURES.length / COLS);
  const width = COLS * BOARD_SIZE;
  const height = rows * (BOARD_SIZE + LABEL_HEIGHT);

  await preloadPieceImages(['cburnett']);

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < BOARD_TEXTURES.length; i++) {
    const col = i % COLS;
    const row = Math.floor(i / COLS);
    const x = col * BOARD_SIZE;
    const y = row * (BOARD_SIZE + LABEL_HEIGHT);

    const boardBuf = await renderBoard(FEN, {
      size: BOARD_SIZE,
      light: null,
      dark: null,
      style: 'cburnett',
      flipped: false,
      lastMove: null,
      highlightColor: null,
      showHighlights: false,
      texture: BOARD_TEXTURES[i],
    });

    const img = await loadImage(boardBuf);
    ctx.drawImage(img, x, y);

    ctx.fillStyle = '#000000';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    const label = BOARD_TEXTURES[i].replace(/\.(jpg|png|svg)$/i, '');
    ctx.fillText(label, x + BOARD_SIZE / 2, y + BOARD_SIZE + 14);
  }

  const outPath = path.join(__dirname, 'texture_preview.png');
  fs.writeFileSync(outPath, canvas.toBuffer('image/png'));
  console.log(`Saved ${BOARD_TEXTURES.length} texture previews to ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
