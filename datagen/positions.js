/**
 * Chess position generation — random placement and PGN sampling.
 */

const fs = require('fs');
const Chess = require('chess.js').Chess || require('chess.js');
const { randInt, choice, shuffle } = require('./rand');

/**
 * Generate a random position (not necessarily legal).
 * Turn defaults to 'w', castling to '-' since move history is unknown.
 */
function randomPosition() {
  const board = new Array(64).fill(null);

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
    const rank = Math.floor(sq / 8);
    if ((piece === 'P' || piece === 'p') && (rank === 0 || rank === 7)) {
      piece = choice(['N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q']);
    }
    board[sq] = piece;
  }

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

  return { placement: ranks.join('/'), turn: 'w', castling: '-', lastMove: null };
}

/**
 * Load games from a PGN file, up to maxGames.
 * Only reads enough bytes to find maxGames games, avoiding loading
 * the entire file for huge PGN databases.
 */
function loadPgnGames(pgnPath, maxGames = 1000) {
  // Read in chunks to avoid loading multi-GB files
  const CHUNK_SIZE = 4 * 1024 * 1024; // 4MB
  const fd = fs.openSync(pgnPath, 'r');
  const stat = fs.fstatSync(fd);
  const gameTexts = [];
  let current = '';
  let leftover = '';
  let bytesRead = 0;

  const buf = Buffer.alloc(CHUNK_SIZE);

  while (bytesRead < stat.size && gameTexts.length < maxGames) {
    const n = fs.readSync(fd, buf, 0, CHUNK_SIZE, bytesRead);
    if (n === 0) break;
    bytesRead += n;

    const text = leftover + buf.toString('utf-8', 0, n);
    const lines = text.split('\n');
    // Last element may be incomplete — save for next chunk
    leftover = lines.pop();

    for (const line of lines) {
      if (line.startsWith('[Event ') && current.trim()) {
        gameTexts.push(current);
        if (gameTexts.length >= maxGames) {
          fs.closeSync(fd);
          return gameTexts;
        }
        current = '';
      }
      current += line + '\n';
    }
  }

  // Handle remaining
  if (leftover) current += leftover + '\n';
  if (current.trim() && gameTexts.length < maxGames) gameTexts.push(current);

  fs.closeSync(fd);
  return gameTexts;
}

/**
 * Sample a random position from a single PGN game string.
 * Returns the position plus the last move (from/to squares) if any.
 */
function samplePgnPosition(gameText) {
  const chess = new Chess();
  const loaded = chess.load_pgn
    ? chess.load_pgn(gameText)
    : chess.loadPgn(gameText);

  if (!loaded) return null;

  const history = chess.history({ verbose: true });
  if (history.length === 0) return null;

  const targetMove = randInt(0, history.length);

  const chess2 = new Chess();
  for (let i = 0; i < targetMove && i < history.length; i++) {
    chess2.move(history[i].san);
  }

  const fen = chess2.fen();
  const parts = fen.split(' ');

  // Last move's from/to squares (null if at starting position)
  const lastMove = targetMove > 0
    ? { from: history[targetMove - 1].from, to: history[targetMove - 1].to }
    : null;

  return {
    placement: parts[0],
    turn: parts[1],
    castling: parts[2],
    lastMove,
  };
}

/**
 * Sample multiple positions from a PGN file.
 */
function positionsFromPgn(pgnPath, numPositions) {
  console.log(`Loading PGN from ${pgnPath}...`);
  // Load at most 10x the requested positions worth of games to avoid
  // reading a massive file when we only need a few positions
  const maxGames = Math.max(numPositions * 2, 1000);
  const gameTexts = loadPgnGames(pgnPath, maxGames);
  console.log(`  Loaded ${gameTexts.length} games`);

  if (gameTexts.length === 0) {
    throw new Error(`No games found in ${pgnPath}`);
  }

  const positions = [];
  let attempts = 0;
  const maxAttempts = numPositions * 10;

  while (positions.length < numPositions && attempts < maxAttempts) {
    attempts++;
    const pos = samplePgnPosition(choice(gameTexts));
    if (pos) positions.push(pos);
  }

  if (positions.length < numPositions) {
    console.warn(`  Warning: only extracted ${positions.length}/${numPositions} positions`);
  }

  return positions.slice(0, numPositions);
}

module.exports = { randomPosition, positionsFromPgn };
