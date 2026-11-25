"""Review a PGN game in one call, with Stockfish from the config and a disk cache.

    review = run_review(cfg, pgn_text)                    # GameReview (Stockfish found as in step 1)
    review = run_review(cfg, pgn_text, game_index=2)      # the third game of the PGN

A finished review is stored under ``data/reviews/`` (``review.cache``), keyed by the
moves, the start position, the depth, the number of engine lines and the class
thresholds, so changing any of them reviews the game again.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chess_engine.config import Config
from chess_engine.review.analyzer import GameReview, review_from_dict, review_game, review_to_dict
from chess_engine.review.engine import Analyzer, StockfishAnalyzer
from chess_engine.stockfish import find_stockfish


def read_game(text: str, game_index: int = 0) -> chess.pgn.Game:
    """Game number ``game_index`` (0-based) of a PGN text, if it is readable and has moves."""
    stream = io.StringIO(text)
    for _ in range(game_index):
        if chess.pgn.read_game(stream) is None:
            break
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or next(iter(game.mainline_moves()), None) is None:
        raise ValueError(f"no reviewable game number {game_index + 1} in the PGN")
    return game


class ReviewCache:
    """Finished reviews as JSON files, one per (game, settings)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(game: chess.pgn.Game, cfg: Config) -> str:
        settings: dict[str, Any] = {"depth": cfg.stockfish.analysis_depth, **cfg.review.model_dump()}
        settings.pop("key_moments", None)
        settings.pop("cache", None)
        payload = json.dumps([game.board().fen(), [m.uci() for m in game.mainline_moves()], settings], sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> GameReview | None:
        path = self.directory / f"{key}.json"
        try:
            return review_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, review: GameReview) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(review_to_dict(review)), encoding="utf-8")
        tmp.replace(path)


def run_review(cfg: Config, pgn_text: str, game_index: int = 0, analyzer: Analyzer | None = None,
               cache: ReviewCache | None = None) -> GameReview:
    """Review one game of ``pgn_text`` (see the module docstring).

    Args:
        analyzer: Engine to use; default: Stockfish from the config, started and stopped here.
        cache: Default: ``<data_dir>/reviews`` when ``review.cache`` is on.

    Raises:
        ValueError: no such game. ``chess_engine.stockfish.StockfishError``: Stockfish not found.
    """
    game = read_game(pgn_text, game_index)
    if cache is None and cfg.review.cache:
        cache = ReviewCache(cfg.paths.data_dir / "reviews")
    key = ReviewCache.key(game, cfg)
    if cache is not None and (cached := cache.get(key)) is not None:
        return cached
    depth = cfg.stockfish.analysis_depth
    if analyzer is not None:
        review = review_game(game, analyzer, cfg.review, depth)
    else:
        path = find_stockfish(cfg.stockfish.path)
        with StockfishAnalyzer(str(path), depth, cfg.review.multipv, cfg.stockfish.threads,
                               cfg.stockfish.hash_mb) as engine:
            review = review_game(game, engine, cfg.review, depth)
    if cache is not None:
        cache.put(key, review)
    return review
