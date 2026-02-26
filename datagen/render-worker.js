/**
 * Worker thread for parallel board rendering.
 *
 * Receives a chunk of items via workerData, renders each board to PNG,
 * writes the file with async I/O, and posts manifest lines back to the
 * main thread.
 */

const { parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');
const { renderBoard, preloadPieceImages } = require('./render');

function boardFenForOrientation(pos, flipped) {
  let { placement, turn, castling, enPassant } = pos;

  if (flipped) {
    const ranks = placement.split('/');
    ranks.reverse();
    placement = ranks.map(r => r.split('').reverse().join('')).join('/');
  }

  return `${placement} ${turn} ${castling} ${enPassant}`;
}

function countPieces(placement) {
  let count = 0;
  for (const ch of placement) {
    if (ch !== '/' && (ch < '1' || ch > '8')) count++;
  }
  return count;
}

async function run() {
  const { items, outputDir, imageSize } = workerData;

  // Pre-load all piece images needed by this chunk
  const styles = items.map(item => item.style.style);
  await preloadPieceImages(styles);

  for (let i = 0; i < items.length; i++) {
    const { index, pos, legal, style: vis } = items[i];

    const hasHighlight = vis.showHighlights && pos.lastMove != null;

    const buffer = await renderBoard(pos.placement, {
      size: imageSize,
      light: vis.colors.light,
      dark: vis.colors.dark,
      style: vis.style,
      flipped: vis.flipped,
      lastMove: pos.lastMove || null,
      highlightColor: vis.highlightColor,
      showHighlights: vis.showHighlights,
    });

    const filename = `${String(index).padStart(6, '0')}.png`;
    await fs.promises.writeFile(path.join(outputDir, filename), buffer);

    const fen = boardFenForOrientation(pos, vis.flipped);
    const pieceCount = countPieces(pos.placement);

    parentPort.postMessage({
      type: 'manifest',
      index,
      line: [
        filename,
        fen,
        legal ? 1 : 0,
        pos.turn,
        pos.castling,
        pos.enPassant,
        pieceCount,
        hasHighlight ? 1 : 0,
        vis.style,
        vis.flipped ? 1 : 0,
      ].join(','),
    });
  }

  parentPort.postMessage({ type: 'done' });
}

run().catch(err => {
  parentPort.postMessage({ type: 'error', message: err.message, stack: err.stack });
  process.exit(1);
});
