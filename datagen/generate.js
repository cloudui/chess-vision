#!/usr/bin/env node
/**
 * Chess board image dataset generator.
 *
 * Two modes:
 *   CLI:    node generate.js --num-images 1000 --output-dir ../data/generated --seed 42
 *   Config: node generate.js --config dataset.yaml
 */

const fs = require('fs');
const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');
const yaml = require('js-yaml');
const { setSeed, shuffle } = require('./rand');
const { randomPosition, positionsFromPgn } = require('./positions');
const { randomStyle } = require('./render');

// ---------------------------------------------------------------------------
// Generate a single split using worker threads
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

  // Pre-generate styles in main thread (preserves seeded PRNG determinism)
  const items = positions.map((p, i) => ({
    index: i,
    pos: p.pos,
    legal: p.legal,
    style: randomStyle(rendering),
  }));

  // Split into chunks for workers
  const numWorkers = Math.min(os.cpus().length, totalImages);
  const chunkSize = Math.ceil(totalImages / numWorkers);
  const chunks = [];
  for (let i = 0; i < totalImages; i += chunkSize) {
    chunks.push(items.slice(i, i + chunkSize));
  }

  console.log(`  Using ${chunks.length} worker threads`);

  // Spawn workers and collect manifest lines
  const manifestLines = [];
  let completed = 0;

  await Promise.all(chunks.map(chunk => new Promise((resolve, reject) => {
    const worker = new Worker(path.join(__dirname, 'render-worker.js'), {
      workerData: { items: chunk, outputDir, imageSize },
    });

    worker.on('message', msg => {
      if (msg.type === 'manifest') {
        manifestLines.push({ index: msg.index, line: msg.line });
        completed++;
        if (completed % 500 === 0 || completed === totalImages) {
          console.log(`  ${completed}/${totalImages}`);
        }
      } else if (msg.type === 'error') {
        reject(new Error(msg.message));
      }
    });

    worker.on('error', reject);
    worker.on('exit', code => {
      if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
      else resolve();
    });
  })));

  // Sort by index and write manifest
  manifestLines.sort((a, b) => a.index - b.index);
  const header = 'filename,fen,legal,turn,castling,en_passant,piece_count,has_highlight,style,flipped';
  const manifestContent = [header, ...manifestLines.map(m => m.line)].join('\n') + '\n';
  const manifestPath = path.join(outputDir, 'manifest.csv');
  fs.writeFileSync(manifestPath, manifestContent);
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
