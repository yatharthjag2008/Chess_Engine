"""Build the website's opening-name table from the Lichess opening data set.

Usage:
    python scripts/build_openings.py          # third_party/chess-openings/*.tsv -> site/data/openings.json

Each opening's moves are played out with python-chess and its name is filed under the position reached (piece
placement, side to move and castling rights), so the site finds a name however the moves were ordered
(transpositions). When several openings reach the same position, the one with the fewest moves names it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "chess-openings"
TARGET = ROOT / "site" / "data" / "openings.json"


def position_key(board: chess.Board) -> str:
    """Placement, side to move and castling: the first three FEN fields (en passant is left out on purpose)."""
    return " ".join(board.fen().split(" ")[:3])


def read_openings(files: Iterable[Path]) -> list[tuple[str, str, str]]:
    """(eco, name, pgn) rows from the TSV files."""
    rows = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows.append((row["eco"], row["name"], row["pgn"]))
    return rows


def play(pgn: str) -> chess.Board:
    """The position after a move list such as "1. e4 e5 2. Nf3"."""
    board = chess.Board()
    for token in pgn.split():
        if not token[0].isdigit():
            board.push_san(token)
    return board


def build(rows: Iterable[tuple[str, str, str]]) -> dict[str, list[str]]:
    """{position key: [eco, name]}, the shortest line winning when several reach one position."""
    best: dict[str, tuple[int, str, str]] = {}
    for eco, name, pgn in rows:
        board = play(pgn)
        key = position_key(board)
        plies = len(board.move_stack)
        if key not in best or plies < best[key][0]:
            best[key] = (plies, eco, name)
    return {key: [eco, name] for key, (_, eco, name) in sorted(best.items())}


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(SOURCE), help="folder with a.tsv ... e.tsv")
    parser.add_argument("--out", default=str(TARGET))
    args = parser.parse_args(argv)
    files = sorted(Path(args.source).glob("*.tsv"))
    if not files:
        print(f"error: no .tsv files in {args.source}", file=sys.stderr)
        return 1
    table = build(read_openings(files))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write(f"{len(table):,} positions named, written to {out} ({out.stat().st_size / 1e3:.0f} kB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
