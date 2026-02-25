#!/usr/bin/env node
/**
 * Chess board image dataset generator using chess-fen2img.
 *
 * Generates diverse chess board images with varied piece styles, board colors,
 * orientations, highlights, and coordinate toggles. Outputs PNGs + manifest.csv
 * compatible with the Python training pipeline.
 *
 * Usage:
 *   node generate.js --num-images 10000 --output-dir ../data/generated \
 *     --pgn ../data/pgn/lichess_2013-01.pgn --legal-ratio 0.5 --seed 42
 */

const fs = require('fs');
const path = require('path');
const { ChessImageGenerator } = require('chess-fen2img');
const Chess = require('chess.js').Chess || require('chess.js');

// ---------------------------------------------------------------------------
// Seeded PRNG (mulberry32)
// ---------------------------------------------------------------------------

function mulberry32(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rand = Math.random;

function randInt(min, max) {
  return Math.floor(rand() * (max - min + 1)) + min;
}

function choice(arr) {
  return arr[Math.floor(rand() * arr.length)];
}

function sample(arr, n) {
  const copy = arr.slice();
  const result = [];
  for (let i = 0; i < n && copy.length > 0; i++) {
    const idx = Math.floor(rand() * copy.length);
    result.push(copy.splice(idx, 1)[0]);
  }
  return result;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

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

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const RANKS = ['1', '2', '3', '4', '5', '6', '7', '8'];

// ---------------------------------------------------------------------------
// Position generation — Random (legal=0)
// ---------------------------------------------------------------------------

function randomPosition() {
  // Place pieces on an 8x8 board represented as an array of 64 slots
  const board = new Array(64).fill(null);

  // Always place both kings
  const wkSq = randInt(0, 63);
  let bkSq = randInt(0, 63);
  while (bkSq === wkSq) bkSq = randInt(0, 63);
  board[wkSq] = 'K';
  board[bkSq] = 'k';

  const otherPieces = ['P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q'];
  const numExtra = randInt(0, 28);
  const available = [];
  for (let i = 0; i < 64; i++) {
    if (i !== wkSq && i !== bkSq) available.push(i);
  }
  shuffle(available);

  for (let i = 0; i < Math.min(numExtra, available.length); i++) {
    let piece = choice(otherPieces);
    const sq = available[i];
    const rank = Math.floor(sq / 8); // 0=rank 8, 7=rank 1
    // Don't place pawns on rank 1 or 8 (rank index 0 or 7)
    if ((piece === 'P' || piece === 'p') && (rank === 0 || rank === 7)) {
      piece = choice(['N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q']);
      // Preserve color: if original was lowercase (black), pick lowercase
      // Actually, just pick randomly from the non-pawn set, since original
      // code doesn't preserve color for the substitution
    }
    board[sq] = piece;
  }

  // Build FEN placement string
  const ranks = [];
  for (let r = 0; r < 8; r++) {
    let rankStr = '';
    let empty = 0;
    for (let f = 0; f < 8; f++) {
      const piece = board[r * 8 + f];
      if (piece) {
        if (empty > 0) { rankStr += empty; empty = 0; }
        rankStr += piece;
      } else {
        empty++;
      }
    }
    if (empty > 0) rankStr += empty;
    ranks.push(rankStr);
  }

  // Random positions: white to move, no castling, no en passant
  return {
    placement: ranks.join('/'),
    turn: 'w',
    castling: '-',
  };
}

// ---------------------------------------------------------------------------
// Position generation — PGN (legal=1)
// ---------------------------------------------------------------------------

function loadPgnGames(pgnPath) {
  const content = fs.readFileSync(pgnPath, 'utf-8');
  // Split into individual games by looking for [Event lines
  const gameTexts = [];
  let current = '';

  for (const line of content.split('\n')) {
    if (line.startsWith('[Event ') && current.trim()) {
      gameTexts.push(current);
      current = '';
    }
    current += line + '\n';
  }
  if (current.trim()) gameTexts.push(current);

  return gameTexts;
}

function samplePgnPosition(gameText) {
  const chess = new Chess();
  const loaded = chess.load_pgn
    ? chess.load_pgn(gameText)
    : chess.loadPgn(gameText);

  if (!loaded) return null;

  // Get the move history
  const history = chess.history();
  if (history.length === 0) return null;

  // Reset and replay to a random point
  chess.reset();
  const targetMove = randInt(0, history.length);

  // We need to re-parse and replay moves one by one
  const chess2 = new Chess();
  chess2.load_pgn
    ? chess2.load_pgn(gameText)
    : chess2.loadPgn(gameText);
  const moves = chess2.history();

  const chess3 = new Chess();
  for (let i = 0; i < targetMove && i < moves.length; i++) {
    chess3.move(moves[i]);
  }

  const fen = chess3.fen();
  const parts = fen.split(' ');
  return {
    placement: parts[0],
    turn: parts[1],
    castling: parts[2],
  };
}

function positionsFromPgn(pgnPath, numPositions) {
  console.log(`Loading PGN from ${pgnPath}...`);
  const gameTexts = loadPgnGames(pgnPath);
  console.log(`  Found ${gameTexts.length} games`);

  if (gameTexts.length === 0) {
    throw new Error(`No games found in ${pgnPath}`);
  }

  const positions = [];
  let attempts = 0;
  const maxAttempts = numPositions * 10;

  while (positions.length < numPositions && attempts < maxAttempts) {
    attempts++;
    const gameText = choice(gameTexts);
    const pos = samplePgnPosition(gameText);
    if (pos) positions.push(pos);
  }

  if (positions.length < numPositions) {
    console.warn(`  Warning: only extracted ${positions.length}/${numPositions} positions`);
  }

  return positions.slice(0, numPositions);
}

// ---------------------------------------------------------------------------
// FEN adjustment for flipped boards
// ---------------------------------------------------------------------------

function boardFenForOrientation(pos, flipped) {
  let { placement, turn, castling } = pos;

  if (flipped) {
    // Reverse rank order and reverse each rank string
    const ranks = placement.split('/');
    ranks.reverse();
    const flippedRanks = ranks.map(r => r.split('').reverse().join(''));
    placement = flippedRanks.join('/');
  }

  return `${placement} ${turn} ${castling}`;
}

// ---------------------------------------------------------------------------
// Square name helpers
// ---------------------------------------------------------------------------

function squareIndexToAlgebraic(idx) {
  // idx 0 = a8, idx 7 = h8, idx 8 = a7, ..., idx 63 = h1
  const file = idx % 8;
  const rank = 7 - Math.floor(idx / 8);
  return FILES[file] + RANKS[rank];
}

function randomHighlightSquares() {
  if (rand() >= 0.5) return [];
  const num = randInt(1, 4);
  const indices = sample(
    Array.from({ length: 64 }, (_, i) => i),
    num,
  );
  return indices.map(squareIndexToAlgebraic);
}

// ---------------------------------------------------------------------------
// Main generation
// ---------------------------------------------------------------------------

async function generateDataset(opts) {
  const {
    numImages,
    outputDir,
    pgnPath,
    legalRatio,
    imageSize,
  } = opts;

  fs.mkdirSync(outputDir, { recursive: true });

  // Determine how many legal vs random positions
  let numLegal = 0;
  let numRandom = numImages;
  let legalPositions = [];

  if (pgnPath) {
    numLegal = Math.round(numImages * legalRatio);
    numRandom = numImages - numLegal;
    legalPositions = positionsFromPgn(pgnPath, numLegal);
    // Adjust if we got fewer positions than expected
    numLegal = legalPositions.length;
    numRandom = numImages - numLegal;
  }

  console.log(`Generating ${numImages} images: ${numLegal} legal + ${numRandom} random`);

  // Build position list: (position, legal)
  const positions = [];
  for (const pos of legalPositions) {
    positions.push({ pos, legal: true });
  }
  for (let i = 0; i < numRandom; i++) {
    positions.push({ pos: randomPosition(), legal: false });
  }

  // Shuffle so legal and random are interleaved
  shuffle(positions);

  // Open manifest
  const manifestPath = path.join(outputDir, 'manifest.csv');
  const manifestLines = ['filename,fen,legal'];

  for (let i = 0; i < positions.length; i++) {
    const { pos, legal } = positions[i];

    // Random visual style
    const style = choice(PIECE_STYLES);
    const colors = choice(BOARD_COLORS);
    const flipped = rand() < 0.3;
    const notations = rand() < 0.7;
    const highlights = randomHighlightSquares();

    // Create generator
    const generator = new ChessImageGenerator({
      size: imageSize,
      light: colors.light,
      dark: colors.dark,
      style: style,
      flipped: flipped,
      notations: notations,
    });

    // Load FEN (only the placement part)
    generator.loadFEN(pos.placement);

    // Apply highlights
    if (highlights.length > 0) {
      generator.highlightSquares(highlights);
    }

    // Generate image
    const filename = `${String(i).padStart(6, '0')}.png`;
    const outputPath = path.join(outputDir, filename);
    await generator.generatePNG(outputPath);

    // Write manifest entry (FEN adjusted for orientation)
    const fen = boardFenForOrientation(pos, flipped);
    manifestLines.push(`${filename},${fen},${legal ? 1 : 0}`);

    if ((i + 1) % 100 === 0 || i + 1 === positions.length) {
      console.log(`  ${i + 1}/${positions.length}`);
    }
  }

  fs.writeFileSync(manifestPath, manifestLines.join('\n') + '\n');
  console.log(`Done. Images and manifest saved to ${outputDir}/`);
  console.log(`  Manifest: ${manifestPath}`);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    numImages: 1000,
    outputDir: '../data/generated',
    pgnPath: null,
    legalRatio: 0.5,
    imageSize: 480,
    seed: null,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--num-images':
        opts.numImages = parseInt(args[++i], 10);
        break;
      case '--output-dir':
        opts.outputDir = args[++i];
        break;
      case '--pgn':
        opts.pgnPath = args[++i];
        break;
      case '--legal-ratio':
        opts.legalRatio = parseFloat(args[++i]);
        break;
      case '--image-size':
        opts.imageSize = parseInt(args[++i], 10);
        break;
      case '--seed':
        opts.seed = parseInt(args[++i], 10);
        break;
      default:
        console.error(`Unknown argument: ${args[i]}`);
        process.exit(1);
    }
  }

  return opts;
}

async function main() {
  const opts = parseArgs();

  if (opts.seed !== null) {
    rand = mulberry32(opts.seed);
  }

  await generateDataset(opts);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
