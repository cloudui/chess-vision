#!/usr/bin/env node
/**
 * Download lichess board texture images.
 *
 * Usage: node download_boards.js
 *
 * Downloads board background images from the lichess GitHub repo
 * and saves them to datagen/boards/ for use by render.js.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const DEST_DIR = path.join(__dirname, 'boards');

const BASE_URL = 'https://raw.githubusercontent.com/lichess-org/lila/master/public/images/board';

// Board textures to download (skip thumbnails, originals, and overlay PNGs)
const BOARDS = [
  'wood.jpg',
  'wood2.jpg',
  'wood3.jpg',
  'wood4.jpg',
  'blue-marble.jpg',
  'blue2.jpg',
  'blue3.jpg',
  'blue.png',
  'brown.png',
  'canvas2.jpg',
  'green-plastic.png',
  'green.png',
  'grey.jpg',
  'leather.jpg',
  'maple.jpg',
  'maple2.jpg',
  'marble.jpg',
  'metal.jpg',
  'ncf-board.png',
  'newspaper.svg',
  'olive.jpg',
  'pink-pyramid.png',
  'purple-diag.png',
  'purple.png',
];

function download(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'chess-vision-datagen' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return download(res.headers.location).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode}`));
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    }).on('error', reject);
  });
}

async function main() {
  fs.mkdirSync(DEST_DIR, { recursive: true });
  console.log(`Downloading board textures to ${DEST_DIR}\n`);

  let downloaded = 0, skipped = 0, failed = 0;

  for (const filename of BOARDS) {
    const outPath = path.join(DEST_DIR, filename);

    if (fs.existsSync(outPath)) {
      console.log(`  ${filename}: already exists, skipping`);
      skipped++;
      continue;
    }

    try {
      const data = await download(`${BASE_URL}/${filename}`);
      fs.writeFileSync(outPath, data);
      console.log(`  ${filename}: OK (${(data.length / 1024).toFixed(0)} KB)`);
      downloaded++;
    } catch (err) {
      console.error(`  ${filename}: FAILED — ${err.message}`);
      failed++;
    }
  }

  console.log(`\nDone: ${downloaded} downloaded, ${skipped} skipped, ${failed} failed`);

  const available = fs.readdirSync(DEST_DIR)
    .filter(f => /\.(jpg|jpeg|png)$/i.test(f) && !f.includes('thumbnail'));
  console.log(`Available textures (${available.length}): ${available.join(', ')}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
