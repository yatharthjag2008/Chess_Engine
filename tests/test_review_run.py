"""Tests for one-call reviews with a disk cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine.config import Config
from chess_engine.review.analyzer import review_to_dict
from chess_engine.review.run import ReviewCache, read_game, run_review
from test_review_analyzer import PGN, _scripted


def test_read_game() -> None:
    assert len(list(read_game(PGN).mainline_moves())) == 7
    with pytest.raises(ValueError, match="number 2"):
        read_game(PGN, game_index=1)


def test_run_review_caches_by_game_and_settings(default_config: Config, tmp_path: Path) -> None:
    cache = ReviewCache(tmp_path / "reviews")
    analyzer = _scripted()
    first = run_review(default_config, PGN, analyzer=analyzer, cache=cache)
    calls = analyzer.calls
    again = run_review(default_config, PGN, analyzer=analyzer, cache=cache)
    assert analyzer.calls == calls                                   # answered from the cache
    assert review_to_dict(again) == review_to_dict(first)
    stricter = default_config.model_copy(update={"review": default_config.review.model_copy(
        update={"inaccuracy_drop": 3})})
    run_review(stricter, PGN, analyzer=analyzer, cache=cache)
    assert analyzer.calls > calls                                    # different thresholds: reviewed again
    assert len(list((tmp_path / "reviews").glob("*.json"))) == 2


def test_cache_is_off_when_configured(default_config: Config, tmp_path: Path) -> None:
    cfg = default_config.model_copy(update={
        "review": default_config.review.model_copy(update={"cache": False}),
        "paths": default_config.paths.model_copy(update={"data_dir": tmp_path}),
    })
    run_review(cfg, PGN, analyzer=_scripted())
    assert not (tmp_path / "reviews").exists()
    assert ReviewCache(tmp_path / "x").get("missing") is None


@pytest.mark.stockfish
def test_run_review_with_the_configured_stockfish(default_config: Config, tmp_path: Path, stockfish) -> None:
    cfg = default_config.model_copy(update={
        "stockfish": default_config.stockfish.model_copy(update={"analysis_depth": 8}),
        "paths": default_config.paths.model_copy(update={"data_dir": tmp_path}),
    })
    review = run_review(cfg, PGN)
    assert review.depth == 8 and review.moves[5].classification.value == "blunder"
    assert any((tmp_path / "reviews").glob("*.json"))
