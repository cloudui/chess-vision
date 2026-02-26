#!/usr/bin/env node
/**
 * Render board color previews: 10 fixed themes + 10 random cohesive palettes.
 * Outputs to datagen/color_preview.png
 */

const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const { setSeed, randInt } = require('./rand');
const { renderBoard, preloadPieceImages } = require('./render');

const BOARD_COLORS = [
  { light: '#f0d9b5', dark: '#b58863', name: 'classic brown' },
  { light: '#eeeed2', dark: '#769656', name: 'chess.com green' },
  { light: '#dee3e6', dark: '#8ca2ad', name: 'lichess blue' },
  { light: '#e8dac2', dark: '#a57551', name: 'wood' },
  { light: '#f0f0e0', dark: '#6a9b5e', name: 'green variant' },
  { light: '#e0e0d0', dark: '#7b8ea0', name: 'slate' },
  { light: '#fce4b8', dark: '#d4924a', name: 'amber' },
  { light: '#d7eaf3', dark: '#5a8bb0', name: 'sky blue' },
  { light: '#D9E2E8', dark: '#7093A9', name: 'glass' },
  { light: '#91B5A4', dark: '#6E9281', name: 'game room' },
];

function randomBoardColors(idx) {
  const hr = randInt(0, 255), hg = randInt(0, 255), hb = randInt(0, 255);
  const lMix = 0.3 + Math.random() * 0.2;
  const lr = Math.round(255 * (1 - lMix) + hr * lMix);
  const lg = Math.round(255 * (1 - lMix) + hg * lMix);
  const lb = Math.round(255 * (1 - lMix) + hb * lMix);
  const dMix = 0.5 + Math.random() * 0.3;
  const dBase = randInt(80, 140);
  const dr = Math.round(dBase * (1 - dMix) + hr * dMix * 0.6);
  const dg = Math.round(dBase * (1 - dMix) + hg * dMix * 0.6);
  const db = Math.round(dBase * (1 - dMix) + hb * dMix * 0.6);
  return {
    light: `rgb(${lr}, ${lg}, ${lb})`,
    dark: `rgb(${dr}, ${dg}, ${db})`,
    name: `random #${idx + 1}`,
  };
}

const FEN = 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R';
const BOARD_SIZE = 160;
const COLS = 5;
const LABEL_HEIGHT = 20;

async function main() {
  setSeed(42);

  // 10 fixed + 10 random
  const colors = [
    ...BOARD_COLORS,
    ...Array.from({ length: 10 }, (_, i) => randomBoardColors(i)),
  ];

  const rows = Math.ceil(colors.length / COLS);
  const width = COLS * BOARD_SIZE;
  const height = rows * (BOARD_SIZE + LABEL_HEIGHT);

  await preloadPieceImages(['cburnett']);

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < colors.length; i++) {
    const col = i % COLS;
    const row = Math.floor(i / COLS);
    const x = col * BOARD_SIZE;
    const y = row * (BOARD_SIZE + LABEL_HEIGHT);

    const boardBuf = await renderBoard(FEN, {
      size: BOARD_SIZE,
      light: colors[i].light,
      dark: colors[i].dark,
      style: 'cburnett',
      flipped: false,
      lastMove: null,
      highlightColor: null,
      showHighlights: false,
    });

    const img = await loadImage(boardBuf);
    ctx.drawImage(img, x, y);

    ctx.fillStyle = '#000000';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(colors[i].name, x + BOARD_SIZE / 2, y + BOARD_SIZE + 14);
  }

  const outPath = path.join(__dirname, 'color_preview.png');
  fs.writeFileSync(outPath, canvas.toBuffer('image/png'));
  console.log(`Saved ${colors.length} color previews to ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
