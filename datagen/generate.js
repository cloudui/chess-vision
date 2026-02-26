#!/usr/bin/env node
/**
 * Chess board image dataset generator.
 *
 * Two modes:
 *   CLI:    node generate.js --num-images 1000 --output-dir ../data/generated --seed 42
 *   Config: node generate.js --config dataset.yaml
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
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
// Generate a single split
// ---------------------------------------------------------------------------

async function generateSplit(name, splitConfig, rendering) {
  const { output_dir: outputDir, sources } = splitConfig;
  const imageSize = rendering.image_size || 480;

  fs.mkdirSync(outputDir, { recursive: true });

  // Collect positions from all sources
  const positions = [];
  for (const source of sources) {
    if (source.type === 'pgn') {
      const pgnPositions = positionsFromPgn(source.pgn, source.count);
      for (const pos of pgnPositions) {
        positions.push({ pos, legal: true });
      }
    } else if (source.type === 'random') {
      for (let i = 0; i < source.count; i++) {
        positions.push({ pos: randomPosition(), legal: false });
      }
    }
  }

  shuffle(positions);

  const totalImages = positions.length;
  console.log(`\n[${name}] Generating ${totalImages} images → ${outputDir}`);

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
      lastMove: pos.lastMove || null,
      highlightColor: vis.highlightColor,
      showHighlights: vis.showHighlights,
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
  console.log(`  Manifest: ${manifestPath}`);
}

// ---------------------------------------------------------------------------
// Config mode
// ---------------------------------------------------------------------------

async function runFromConfig(configPath) {
  const config = yaml.load(fs.readFileSync(configPath, 'utf-8'));
  const rendering = config.rendering || {};

  if (rendering.seed != null) setSeed(rendering.seed);

  const splits = config.splits || {};
  for (const [name, splitConfig] of Object.entries(splits)) {
    await generateSplit(name, splitConfig, rendering);
  }

  console.log('\nAll splits complete.');
}

// ---------------------------------------------------------------------------
// CLI mode (legacy)
// ---------------------------------------------------------------------------

async function runFromCli(opts) {
  if (opts.seed !== null) setSeed(opts.seed);

  // Build a synthetic split config from CLI args
  const sources = [];
  if (opts.pgnPath) {
    const numLegal = Math.round(opts.numImages * opts.legalRatio);
    const numRandom = opts.numImages - numLegal;
    if (numLegal > 0) sources.push({ type: 'pgn', pgn: opts.pgnPath, count: numLegal });
    if (numRandom > 0) sources.push({ type: 'random', count: numRandom });
  } else {
    sources.push({ type: 'random', count: opts.numImages });
  }

  await generateSplit('default', {
    output_dir: opts.outputDir,
    sources,
  }, { image_size: opts.imageSize });
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    configPath: null,
    numImages: 1000,
    outputDir: '../data/generated',
    pgnPath: null,
    legalRatio: 0.5,
    imageSize: 480,
    seed: null,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--config':      opts.configPath = args[++i]; break;
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

  if (opts.configPath) {
    await runFromConfig(opts.configPath);
  } else {
    await runFromCli(opts);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
