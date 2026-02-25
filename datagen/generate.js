#!/usr/bin/env node
/**
 * Chess board image dataset generator.
 *
 * Outputs PNGs + manifest.csv compatible with the Python training pipeline.
 *
 * Usage:
 *   node generate.js --num-images 10000 --output-dir ../data/generated --seed 42
 *   node generate.js --num-images 10000 --output-dir ../data/generated \
 *     --pgn ../data/pgn/lichess_2013-01.pgn --legal-ratio 0.5
 */

const fs = require('fs');
const path = require('path');
const { setSeed, shuffle } = require('./rand');
const { randomPosition, positionsFromPgn } = require('./positions');
const { renderBoard, randomStyle } = require('./render');

// ---------------------------------------------------------------------------
// FEN orientation adjustment
// ---------------------------------------------------------------------------

function boardFenForOrientation(pos, flipped) {
  let { placement, turn, castling } = pos;

  if (flipped) {
    const ranks = placement.split('/');
    ranks.reverse();
    placement = ranks.map(r => r.split('').reverse().join('')).join('/');
  }

  return `${placement} ${turn} ${castling}`;
}

// ---------------------------------------------------------------------------
// Main generation
// ---------------------------------------------------------------------------

async function generateDataset(opts) {
  const { numImages, outputDir, pgnPath, legalRatio, imageSize } = opts;

  fs.mkdirSync(outputDir, { recursive: true });

  // Build positions
  let numLegal = 0;
  let numRandom = numImages;
  let legalPositions = [];

  if (pgnPath) {
    numLegal = Math.round(numImages * legalRatio);
    numRandom = numImages - numLegal;
    legalPositions = positionsFromPgn(pgnPath, numLegal);
    numLegal = legalPositions.length;
    numRandom = numImages - numLegal;
  }

  console.log(`Generating ${numImages} images: ${numLegal} legal + ${numRandom} random`);

  const positions = [
    ...legalPositions.map(pos => ({ pos, legal: true })),
    ...Array.from({ length: numRandom }, () => ({ pos: randomPosition(), legal: false })),
  ];
  shuffle(positions);

  // Generate images + manifest
  const manifestPath = path.join(outputDir, 'manifest.csv');
  const manifestLines = ['filename,fen,legal'];

  for (let i = 0; i < positions.length; i++) {
    const { pos, legal } = positions[i];
    const vis = randomStyle();

    const buffer = await renderBoard(pos.placement, {
      size: imageSize,
      light: vis.colors.light,
      dark: vis.colors.dark,
      style: vis.style,
      flipped: vis.flipped,
      highlights: vis.highlights,
      highlightColor: vis.highlightColor,
    });

    const filename = `${String(i).padStart(6, '0')}.png`;
    fs.writeFileSync(path.join(outputDir, filename), buffer);

    const fen = boardFenForOrientation(pos, vis.flipped);
    manifestLines.push(`${filename},${fen},${legal ? 1 : 0}`);

    if ((i + 1) % 100 === 0 || i + 1 === positions.length) {
      console.log(`  ${i + 1}/${positions.length}`);
    }
  }

  fs.writeFileSync(manifestPath, manifestLines.join('\n') + '\n');
  console.log(`Done. Manifest: ${manifestPath}`);
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
      case '--num-images':  opts.numImages = parseInt(args[++i], 10); break;
      case '--output-dir':  opts.outputDir = args[++i]; break;
      case '--pgn':         opts.pgnPath = args[++i]; break;
      case '--legal-ratio': opts.legalRatio = parseFloat(args[++i]); break;
      case '--image-size':  opts.imageSize = parseInt(args[++i], 10); break;
      case '--seed':        opts.seed = parseInt(args[++i], 10); break;
      default:
        console.error(`Unknown argument: ${args[i]}`);
        process.exit(1);
    }
  }

  return opts;
}

async function main() {
  const opts = parseArgs();
  if (opts.seed !== null) setSeed(opts.seed);
  await generateDataset(opts);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
