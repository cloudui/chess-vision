/**
 * Board rendering with @napi-rs/canvas + piece PNGs from chess-fen2img.
 */

const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const { choice, randInt } = require('./rand');

const PIECE_RES_DIR = path.join(
  __dirname, 'node_modules', 'chess-fen2img', 'src', 'resources'
);

const BOARDS_DIR = path.join(__dirname, 'boards');

// Auto-discover piece styles: any subdirectory with 12 PNG files
const PIECE_STYLES = fs.readdirSync(PIECE_RES_DIR)
  .filter(d => {
    const dir = path.join(PIECE_RES_DIR, d);
    return fs.statSync(dir).isDirectory() &&
      fs.readdirSync(dir).filter(f => f.endsWith('.png')).length === 12;
  })
  .sort();

// Auto-discover board textures: any image file in boards/
const BOARD_TEXTURES = fs.existsSync(BOARDS_DIR)
  ? fs.readdirSync(BOARDS_DIR)
      .filter(f => /\.(jpg|jpeg|png|svg)$/i.test(f))
      .sort()
  : [];

const BOARD_COLORS = [
  { light: '#f0d9b5', dark: '#b58863' },  // classic brown
  { light: '#eeeed2', dark: '#769656' },  // chess.com green
  { light: '#dee3e6', dark: '#8ca2ad' },  // lichess blue
  { light: '#fce4b8', dark: '#d4924a' },  // amber
  // Covered by textures — kept for reference:
  // { light: '#e8dac2', dark: '#a57551' },  // wood
  // { light: '#f0f0e0', dark: '#6a9b5e' },  // green variant
  // { light: '#e0e0d0', dark: '#7b8ea0' },  // slate
  // { light: '#d7eaf3', dark: '#5a8bb0' },  // sky blue
  // { light: '#D9E2E8', dark: '#7093A9' },  // glass
  // { light: '#91B5A4', dark: '#6E9281' },  // game room
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

// Cache loaded images
const pieceImageCache = {};
const boardTextureCache = {};

async function getPieceImage(style, colorType) {
  const key = `${style}/${colorType}`;
  if (!pieceImageCache[key]) {
    const imgPath = path.join(PIECE_RES_DIR, style, PIECE_FILE_NAMES[colorType] + '.png');
    pieceImageCache[key] = await loadImage(imgPath);
  }
  return pieceImageCache[key];
}

async function getBoardTexture(name) {
  if (!boardTextureCache[name]) {
    const imgPath = path.join(BOARDS_DIR, name);
    boardTextureCache[name] = await loadImage(imgPath);
  }
  return boardTextureCache[name];
}

/** Convert algebraic square (e.g. "e4") to board index (0=a8, 63=h1). */
function algebraicToIndex(sq) {
  const file = sq.charCodeAt(0) - 97; // 'a' = 0
  const rank = parseInt(sq[1]) - 1;   // '1' = 0
  return (7 - rank) * 8 + file;
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

/** Generate a random light/dark board color pair with a cohesive palette. */
function randomBoardColors() {
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
  };
}

/** Pick random visual style options for a board. */
function randomStyle(renderConfig = {}) {
  const highlightPct = renderConfig.highlight_pct != null ? renderConfig.highlight_pct : 0.6;
  const texturePct = renderConfig.texture_pct != null ? renderConfig.texture_pct : 0.5;

  // Board background: texture, known color, or random color
  let colors = null;
  let texture = null;
  if (BOARD_TEXTURES.length > 0 && Math.random() < texturePct) {
    texture = choice(BOARD_TEXTURES);
  } else if (Math.random() < 0.5) {
    colors = choice(BOARD_COLORS);
  } else {
    colors = randomBoardColors();
  }

  return {
    style: choice(PIECE_STYLES),
    colors,
    texture,
    flipped: choice([false, true]),
    highlightColor: choice(HIGHLIGHT_COLORS),
    showHighlights: Math.random() < highlightPct,
  };
}

/**
 * Pre-load all piece images for the given styles into the cache.
 */
async function preloadPieceImages(styles) {
  const pieceTypes = Object.keys(PIECE_FILE_NAMES);
  const uniqueStyles = [...new Set(styles)];
  const promises = [];
  for (const style of uniqueStyles) {
    for (const pieceType of pieceTypes) {
      promises.push(getPieceImage(style, pieceType));
    }
  }
  await Promise.all(promises);
}

/**
 * Render a board position to a PNG buffer.
 * lastMove: { from: "e2", to: "e4" } or null
 */
async function renderBoard(placement, opts) {
  const { size, light, dark, style, flipped, lastMove, highlightColor, showHighlights, texture } = opts;

  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  const sq = size / 8;
  const squares = parsePlacement(placement);

  // Compute highlighted square indices from last move
  const highlights = [];
  if (showHighlights && lastMove) {
    highlights.push(algebraicToIndex(lastMove.from));
    highlights.push(algebraicToIndex(lastMove.to));
  }

  // Draw board background
  if (texture) {
    const texImg = await getBoardTexture(texture);
    ctx.drawImage(texImg, 0, 0, size, size);
  }

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const vr = flipped ? 7 - r : r;
      const vf = flipped ? 7 - f : f;

      // Only draw flat color squares if no texture
      if (!texture) {
        ctx.fillStyle = (vr + vf) % 2 === 0 ? light : dark;
        ctx.fillRect(f * sq, r * sq, sq, sq);
      }

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

module.exports = { renderBoard, randomStyle, preloadPieceImages, BOARD_TEXTURES };
