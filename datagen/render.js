/**
 * Board rendering with @napi-rs/canvas + piece PNGs from chess-fen2img.
 */

const path = require('path');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const { choice, sample } = require('./rand');

const PIECE_RES_DIR = path.join(
  __dirname, 'node_modules', 'chess-fen2img', 'src', 'resources'
);

const PIECE_STYLES = [
  'neo', 'game_room', 'glass', 'wood', 'alpha',
  'cburnett', 'cheq', 'leipzig', 'merida',
];

const BOARD_COLORS = [
  { light: '#f0d9b5', dark: '#b58863' },  // classic brown
  { light: '#eeeed2', dark: '#769656' },  // chess.com green
  { light: '#dee3e6', dark: '#8ca2ad' },  // lichess blue
  { light: '#e8dac2', dark: '#a57551' },  // wood
  { light: '#f0f0e0', dark: '#6a9b5e' },  // green variant
  { light: '#e0e0d0', dark: '#7b8ea0' },  // slate
  { light: '#fce4b8', dark: '#d4924a' },  // amber
  { light: '#d7eaf3', dark: '#5a8bb0' },  // sky blue
  { light: '#D9E2E8', dark: '#7093A9' },  // glass
  { light: '#91B5A4', dark: '#6E9281' },  // game room
];

const HIGHLIGHT_COLORS = [
  'rgba(235, 97, 80, 0.5)',
  'rgba(255, 255, 0, 0.4)',
  'rgba(0, 255, 0, 0.3)',
  'rgba(0, 150, 255, 0.3)',
  'rgba(255, 170, 0, 0.4)',
];

const PIECE_FILE_NAMES = {
  wp: 'WhitePawn', bp: 'BlackPawn',
  wb: 'WhiteBishop', bb: 'BlackBishop',
  wn: 'WhiteKnight', bn: 'BlackKnight',
  wr: 'WhiteRook', br: 'BlackRook',
  wq: 'WhiteQueen', bq: 'BlackQueen',
  wk: 'WhiteKing', bk: 'BlackKing',
};

// Cache loaded piece images
const pieceImageCache = {};

async function getPieceImage(style, colorType) {
  const key = `${style}/${colorType}`;
  if (!pieceImageCache[key]) {
    const imgPath = path.join(PIECE_RES_DIR, style, PIECE_FILE_NAMES[colorType] + '.png');
    pieceImageCache[key] = await loadImage(imgPath);
  }
  return pieceImageCache[key];
}

/** Parse FEN placement into array of {color, type} or null per square. */
function parsePlacement(placement) {
  const squares = [];
  for (const ch of placement) {
    if (ch === '/') continue;
    if (ch >= '1' && ch <= '8') {
      for (let n = 0; n < parseInt(ch); n++) squares.push(null);
    } else {
      squares.push({
        color: ch === ch.toUpperCase() ? 'w' : 'b',
        type: ch.toLowerCase(),
      });
    }
  }
  return squares;
}

/** Pick random visual style options for a board. */
function randomStyle() {
  const highlights = [];
  if (choice([true, false])) {
    const num = choice([1, 2, 3, 4]);
    highlights.push(...sample(Array.from({ length: 64 }, (_, i) => i), num));
  }

  return {
    style: choice(PIECE_STYLES),
    colors: choice(BOARD_COLORS),
    flipped: choice([false, false, false, true]),  // ~25% chance
    highlightColor: choice(HIGHLIGHT_COLORS),
    highlights,
  };
}

/** Render a board position to a PNG buffer. */
async function renderBoard(placement, opts) {
  const { size, light, dark, style, flipped, highlights, highlightColor } = opts;

  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  const sq = size / 8;
  const squares = parsePlacement(placement);

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const vr = flipped ? 7 - r : r;
      const vf = flipped ? 7 - f : f;

      ctx.fillStyle = (vr + vf) % 2 === 0 ? light : dark;
      ctx.fillRect(f * sq, r * sq, sq, sq);

      const sqIdx = vr * 8 + vf;
      if (highlights.includes(sqIdx)) {
        ctx.fillStyle = highlightColor;
        ctx.fillRect(f * sq, r * sq, sq, sq);
      }

      const piece = squares[sqIdx];
      if (piece) {
        const img = await getPieceImage(style, piece.color + piece.type);
        ctx.drawImage(img, f * sq, r * sq, sq, sq);
      }
    }
  }

  return canvas.toBuffer('image/png');
}

module.exports = { renderBoard, randomStyle };
