#!/bin/bash
# Download a Lichess monthly database PGN file.
# Early months (2013-01 through 2013-06) are small — good for testing.
#
# Usage: ./download_pgn.sh [YYYY-MM] [output_dir]

MONTH=${1:-"2013-01"}
URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"
OUTPUT_DIR="${2:-../data/pgn}"

mkdir -p "$OUTPUT_DIR"
echo "Downloading Lichess ${MONTH}..."
curl -L -o "${OUTPUT_DIR}/lichess_${MONTH}.pgn.zst" "$URL"
echo "Decompressing..."
zstd -d "${OUTPUT_DIR}/lichess_${MONTH}.pgn.zst" -o "${OUTPUT_DIR}/lichess_${MONTH}.pgn"
echo "Done: ${OUTPUT_DIR}/lichess_${MONTH}.pgn"
