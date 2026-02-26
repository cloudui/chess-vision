#!/usr/bin/env node
/**
 * Download and convert lichess piece sets from SVG to PNG.
 *
 * Usage: node download_pieces.js [--styles style1,style2,...]
 *
 * Downloads SVGs from the lichess GitHub repo, renders them to 80x80 PNGs
 * using @napi-rs/canvas, and saves them to the chess-fen2img resources
 * directory with the naming convention expected by render.js.
 *
 * Default styles are a curated subset that adds visual variety beyond
 * the 9 styles already bundled with chess-fen2img.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { createCanvas, loadImage } = require('@napi-rs/canvas');

const PIECE_SIZE = 80; // match existing piece PNGs

const DEST_DIR = path.join(
  __dirname, 'node_modules', 'chess-fen2img', 'src', 'resources'
);

// Lichess piece file names → our naming convention
const PIECE_MAP = {
  'wP': 'WhitePawn',   'bP': 'BlackPawn',
  'wN': 'WhiteKnight', 'bN': 'BlackKnight',
  'wB': 'WhiteBishop', 'bB': 'BlackBishop',
  'wR': 'WhiteRook',   'bR': 'BlackRook',
  'wQ': 'WhiteQueen',  'bQ': 'BlackQueen',
  'wK': 'WhiteKing',   'bK': 'BlackKing',
};

// Curated styles that add variety beyond the bundled 9
const DEFAULT_STYLES = [
  'california',  // clean modern
  'cardinal',    // distinctive serif
  'celtic',      // ornate knotwork
  'chessnut',    // classic rounded
  'companion',   // clean outlined
  'fantasy',     // decorative
  'gioco',       // bold geometric
  'governor',    // traditional
  'horsey',      // fun/distinctive
  'kosal',       // clean simple
  'maestro',     // lichess default
  'pixel',       // pixel art
  'staunty',     // classic Staunton
  'tatiana',     // ornate traditional
];

function download(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'chess-vision-datagen' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return download(res.headers.location).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    }).on('error', reject);
  });
}

async function svgToPng(svgBuffer, size) {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  const img = await loadImage(svgBuffer);
  ctx.drawImage(img, 0, 0, size, size);
  return canvas.toBuffer('image/png');
}

async function downloadStyle(style) {
  const styleDir = path.join(DEST_DIR, style);

  if (fs.existsSync(styleDir) && fs.readdirSync(styleDir).length === 12) {
    console.log(`  ${style}: already exists, skipping`);
    return { style, status: 'skipped' };
  }

  fs.mkdirSync(styleDir, { recursive: true });

  const baseUrl = `https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/${style}`;
  let downloaded = 0;

  for (const [lichessName, ourName] of Object.entries(PIECE_MAP)) {
    const url = `${baseUrl}/${lichessName}.svg`;
    const outPath = path.join(styleDir, `${ourName}.png`);

    try {
      const svgData = await download(url);
      const pngData = await svgToPng(svgData, PIECE_SIZE);
      fs.writeFileSync(outPath, pngData);
      downloaded++;
    } catch (err) {
      console.error(`    Failed: ${lichessName} — ${err.message}`);
    }
  }

  console.log(`  ${style}: ${downloaded}/12 pieces`);
  return { style, status: downloaded === 12 ? 'ok' : 'partial', downloaded };
}

async function main() {
  // Parse --styles flag
  let styles = DEFAULT_STYLES;
  const stylesIdx = process.argv.indexOf('--styles');
  if (stylesIdx !== -1 && process.argv[stylesIdx + 1]) {
    styles = process.argv[stylesIdx + 1].split(',');
  }

  console.log(`Downloading ${styles.length} piece sets to ${DEST_DIR}\n`);

  const results = [];
  for (const style of styles) {
    results.push(await downloadStyle(style));
  }

  const ok = results.filter(r => r.status === 'ok').length;
  const skipped = results.filter(r => r.status === 'skipped').length;
  const failed = results.filter(r => r.status === 'partial').length;
  console.log(`\nDone: ${ok} downloaded, ${skipped} skipped, ${failed} partial`);

  // Print the PIECE_STYLES array to paste into render.js
  const allStyles = fs.readdirSync(DEST_DIR)
    .filter(d => fs.statSync(path.join(DEST_DIR, d)).isDirectory())
    .filter(d => fs.readdirSync(path.join(DEST_DIR, d)).length === 12)
    .sort();
  console.log(`\nAll available styles (${allStyles.length}):`);
  console.log(`const PIECE_STYLES = ${JSON.stringify(allStyles, null, 2).replace(/\n/g, '\n')};`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
