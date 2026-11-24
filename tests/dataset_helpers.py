"""Helpers for dataset tests: small games files and configs that write into a temp directory."""

from __future__ import annotations

import copy
import functools
import random
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from chess_engine.config import Config, load_config
from chess_engine.data.game_store import GAME_SCHEMA


def random_game_rows(count: int, seed: int = 0, prefix: str = "rand") -> list[dict[str, Any]]:
    """``count`` seeded random games as rows of the step-2 games table (legal SAN, plausible clocks)."""
    return copy.deepcopy(_random_game_rows(count, seed, prefix))


@functools.lru_cache(maxsize=None)
def _random_game_rows(count: int, seed: int, prefix: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        board = chess.Board()
        moves: list[str] = []
        for _ in range(rng.randrange(20, 60)):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(board.san(move))
            board.push(move)
        clocks, left = [], [180.0, 180.0]
        for ply in range(len(moves)):
            if ply >= 2:
                left[ply % 2] = max(0.0, left[ply % 2] - rng.randrange(0, 8) + 2)
            clocks.append(left[ply % 2])
        rows.append({
            "game_id": f"{prefix}{i:05d}", "event": "Rated blitz game", "utc_date": "2026.10.01",
            "utc_time": "00:00:00", "white": "w", "black": "b",
            "white_elo": rng.randrange(800, 2600), "black_elo": rng.randrange(800, 2600),
            "white_title": None, "black_title": None, "result": rng.choice(["1-0", "0-1", "1/2-1/2"]),
            "termination": "Normal", "base_seconds": 180, "increment_seconds": 2, "speed": "blitz",
            "eco": "A00", "opening": "?", "num_plies": len(moves), "moves": moves, "clocks": clocks,
        })
    return rows


def write_game_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write rows to a games Parquet file with the step-2 schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=GAME_SCHEMA), path)
    return path


def tmp_config(tmp_path: Path, **dataset: Any) -> Config:
    """Default config with all paths under ``tmp_path`` and ``data.dataset`` overrides."""
    override = {
        "paths": {
            "data_dir": str(tmp_path),
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "data": {"dataset": {"workers": 1, **dataset}},
    }
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(override), encoding="utf-8")
    return load_config(path)
