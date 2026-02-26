#!/usr/bin/env node
/**
 * Render one sample board per piece style for visual verification.
 * Outputs a grid image to datagen/style_preview.png
 */

const fs = require('fs');
const path = require('path');
const { createCanvas } = require('@napi-rs/canvas');
const { renderBoard, preloadPieceImages } = require('./render');

const PIECE_RES_DIR = path.join(
  __dirname, 'node_modules', 'chess-fen2img', 'src', 'resources'
);

const STYLES = fs.readdirSync(PIECE_RES_DIR)
  .filter(d => {
    const p = path.join(PIECE_RES_DIR, d);
    return fs.statSync(p).isDirectory() &&
      fs.readdirSync(p).filter(f => f.endsWith('.png')).length === 12;
  })
  .sort();

// A recognizable position (Italian Game)
const FEN = 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R';

const BOARD_SIZE = 200;
const COLS = 6;
const LABEL_HEIGHT = 20;

async function main() {
  const rows = Math.ceil(STYLES.length / COLS);
  const width = COLS * BOARD_SIZE;
  const height = rows * (BOARD_SIZE + LABEL_HEIGHT);

  await preloadPieceImages(STYLES);

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < STYLES.length; i++) {
    const col = i % COLS;
    const row = Math.floor(i / COLS);
    const x = col * BOARD_SIZE;
    const y = row * (BOARD_SIZE + LABEL_HEIGHT);

    const boardBuf = await renderBoard(FEN, {
      size: BOARD_SIZE,
      light: '#f0d9b5',
      dark: '#b58863',
      style: STYLES[i],
      flipped: false,
      lastMove: null,
      highlightColor: null,
      showHighlights: false,
    });

    const { loadImage } = require('@napi-rs/canvas');
    const img = await loadImage(boardBuf);
    ctx.drawImage(img, x, y);

    // Label
    ctx.fillStyle = '#000000';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(STYLES[i], x + BOARD_SIZE / 2, y + BOARD_SIZE + 14);
  }

  const outPath = path.join(__dirname, 'style_preview.png');
  fs.writeFileSync(outPath, canvas.toBuffer('image/png'));
  console.log(`Saved ${STYLES.length} style previews to ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
