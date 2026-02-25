"""Synthetic chess board image generator.

Generates chess board images with varied visual styles and full FEN labels.
Uses python-chess for positions/SVG rendering and cairosvg for SVG→PNG.

Output: PNG images + manifest.csv (filename,fen).
"""

import argparse
import csv
import io
import os
import random
import re

import cairosvg
import chess
import chess.pgn
import chess.svg
from PIL import Image


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def random_position() -> chess.Board:
    """Generate a random position by placing random pieces.

    These are NOT legal game positions — turn, castling, and en passant are
    unknowable from random placement. They're set to neutral defaults
    (white to move, no castling, no en passant) and marked as non-legal
    in the manifest so training can skip turn/castling loss on them.

    Still useful for training piece recognition with visual diversity.
    """
    board = chess.Board.empty()

    # Always place both kings
    wk_sq = random.randint(0, 63)
    bk_sq = random.randint(0, 63)
    while bk_sq == wk_sq:
        bk_sq = random.randint(0, 63)
    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

    # Place a random number of other pieces
    other_pieces = [
        (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK),
        (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK),
        (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK),
        (chess.ROOK, chess.WHITE), (chess.ROOK, chess.BLACK),
        (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.BLACK),
    ]

    num_extra = random.randint(0, 28)
    available = [sq for sq in range(64) if sq != wk_sq and sq != bk_sq]
    random.shuffle(available)

    for i in range(min(num_extra, len(available))):
        piece_type, color = random.choice(other_pieces)
        sq = available[i]
        # Don't place pawns on rank 1 or 8
        rank = chess.square_rank(sq)
        if piece_type == chess.PAWN and rank in (0, 7):
            piece_type = random.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        board.set_piece_at(sq, chess.Piece(piece_type, color))

    # Unknowable metadata — use neutral defaults
    board.turn = chess.WHITE
    board.castling_rights = 0  # no castling (can't know move history)
    board.ep_square = None

    return board


def positions_from_pgn(pgn_path: str, num_positions: int) -> list[chess.Board]:
    """Sample random positions from a PGN file."""
    positions = []
    games = []

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)

    if not games:
        raise ValueError(f"No games found in {pgn_path}")

    while len(positions) < num_positions:
        game = random.choice(games)
        board = game.board()
        moves = list(game.mainline_moves())
        if not moves:
            continue
        # Pick a random point in the game
        num_moves = random.randint(0, len(moves))
        for move in moves[:num_moves]:
            board.push(move)
        positions.append(board.copy())

    return positions[:num_positions]


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def random_color_pair() -> tuple[str, str]:
    """Generate a random light/dark square color pair."""
    presets = [
        ("#f0d9b5", "#b58863"),  # classic brown
        ("#eeeed2", "#769656"),  # chess.com green
        ("#dee3e6", "#8ca2ad"),  # lichess blue
        ("#e8dac2", "#a57551"),  # wood
        ("#f0f0e0", "#6a9b5e"),  # green variant
        ("#e0e0d0", "#7b8ea0"),  # slate
        ("#fce4b8", "#d4924a"),  # amber
        ("#e8e0d0", "#8b7355"),  # walnut
        ("#d7eaf3", "#5a8bb0"),  # sky blue
        ("#f5e6cc", "#c48b4a"),  # warm wood
    ]
    return random.choice(presets)


def random_highlight_squares() -> dict[int, str]:
    """Generate random square highlights (simulating last-move/selection)."""
    fills = {}
    if random.random() < 0.5:
        # Highlight 1-4 random squares
        num = random.randint(1, 4)
        squares = random.sample(range(64), num)
        for sq in squares:
            # Semi-transparent yellow/green/blue highlights
            colors = [
                "rgba(255, 255, 0, 0.4)",
                "rgba(0, 255, 0, 0.3)",
                "rgba(0, 150, 255, 0.3)",
                "rgba(255, 170, 0, 0.4)",
            ]
            fills[sq] = random.choice(colors)
    return fills


def render_board(
    board: chess.Board,
    image_size: int = 400,
    flipped: bool = False,
    colors: tuple[str, str] | None = None,
    highlights: dict[int, str] | None = None,
    coordinates: bool = True,
) -> Image.Image:
    """Render a chess board to a PIL Image."""
    light, dark = colors or ("#f0d9b5", "#b58863")

    svg_str = chess.svg.board(
        board,
        flipped=flipped,
        coordinates=coordinates,
        fill=highlights or {},
        colors={
            "square light": light,
            "square dark": dark,
        },
        size=image_size,
    )

    png_bytes = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=image_size,
        output_height=image_size,
    )
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def board_fen_for_orientation(board: chess.Board, flipped: bool) -> str:
    """Get the FEN placement string adjusted for board orientation.

    When flipped (black on bottom), the visual board is rotated 180°,
    so we reverse the FEN placement to match what's visually shown.
    The turn and castling remain the same (they're metadata, not visual).
    """
    fen = board.fen()
    parts = fen.split(" ")
    placement = parts[0]

    if flipped:
        # Reverse the visual order: flip ranks and reverse each rank
        ranks = placement.split("/")
        ranks.reverse()
        flipped_ranks = []
        for rank in ranks:
            flipped_ranks.append(rank[::-1])
        placement = "/".join(flipped_ranks)

    # Reconstruct: placement + turn + castling (skip en passant, halfmove, fullmove)
    turn = parts[1]
    castling = parts[2]
    return f"{placement} {turn} {castling}"


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_images: int,
    output_dir: str,
    image_size: int = 400,
    pgn_path: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Get positions: list of (board, legal) tuples
    if pgn_path:
        print(f"Sampling {num_images} positions from {pgn_path}...")
        boards = [(b, True) for b in positions_from_pgn(pgn_path, num_images)]
    else:
        print(f"Generating {num_images} random positions...")
        boards = [(random_position(), False) for _ in range(num_images)]

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "fen", "legal"])

        for i, (board, legal) in enumerate(boards):
            # Randomize visual style
            colors = random_color_pair()
            highlights = random_highlight_squares()
            flipped = random.random() < 0.3  # 30% chance black-on-bottom
            coordinates = random.random() < 0.7  # 70% chance show coordinates

            img = render_board(
                board,
                image_size=image_size,
                flipped=flipped,
                colors=colors,
                highlights=highlights,
                coordinates=coordinates,
            )

            filename = f"{i:06d}.png"
            img.save(os.path.join(output_dir, filename))

            fen = board_fen_for_orientation(board, flipped)
            writer.writerow([filename, fen, "1" if legal else "0"])

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_images}")

    print(f"Done. Images and manifest saved to {output_dir}/")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic chess board images")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory for images + manifest")
    parser.add_argument("--pgn", default=None, help="Optional PGN file to sample positions from")
    parser.add_argument("--image-size", type=int, default=400, help="Output image size in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generate_dataset(
        num_images=args.num_images,
        output_dir=args.output_dir,
        image_size=args.image_size,
        pgn_path=args.pgn,
    )
