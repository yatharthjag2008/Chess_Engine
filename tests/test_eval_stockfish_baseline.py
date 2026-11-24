"""Tests for the Stockfish baseline (with a fake UCI engine; a real-Stockfish test skips if it is missing)."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_engine.config import Config, ModelConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.encoding.moves import index_to_move
from chess_engine.eval.positions import board_of
from chess_engine.eval.predictions import predict_split
from chess_engine.eval.stockfish_baseline import MoveCache, engine_moves, sample_rows, stockfish_baseline
from chess_engine.model.network import RatingConditionedNet

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uci_engine.py"), "--first-legal"]


def test_sample_rows() -> None:
    rows = sample_rows(1000, 50, seed=1)
    assert len(rows) == 50 == len(set(rows.tolist())) and (np.diff(rows) > 0).all()
    assert np.array_equal(rows, sample_rows(1000, 50, seed=1))
    assert sample_rows(5, 50).tolist() == [0, 1, 2, 3, 4]


def test_engine_moves_and_cache(tmp_path: Path) -> None:
    boards = [chess.Board(), chess.Board("8/8/8/8/8/2k5/8/K7 w - - 0 1")]
    cache = tmp_path / "cache.json"
    moves = engine_moves(boards, FAKE, depth=3, cache_path=cache)
    assert [m.uci() for m in moves] == [min(m.uci() for m in b.legal_moves) for b in boards]
    assert len(MoveCache(cache).moves) == 2
    seen = []
    engine_moves(boards, ["no-such-engine"], depth=3, cache_path=cache, progress=lambda d, t: seen.append(d))
    assert seen == []  # everything came from the cache; the (broken) engine was never started


def test_baseline_on_real_positions(tiny_dataset: Path, default_config: Config, tmp_path: Path) -> None:
    dataset = PositionDataset(tiny_dataset, "test")
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(architecture="resnet", channels=8, residual_blocks=1,
                                             rating_embedding_dim=4), 8)
    table = predict_split(model, dataset, default_config.ratings)
    rows = sample_rows(len(table), 30, seed=0)
    result = stockfish_baseline(table, dataset.samples, rows, FAKE, depth=2,
                                labels=default_config.ratings.bucket_labels(), cache_path=tmp_path / "c.json")
    expected = []
    for i in rows:
        board = board_of(dataset.samples, int(i))
        played = index_to_move(int(dataset.samples["move"][i]), board).uci()
        expected.append(float(played == min(m.uci() for m in board.legal_moves)))
    assert result["hits"] == expected
    assert result["overall"]["positions"] == 30 and result["overall"]["top1"] == pytest.approx(np.mean(expected))
    assert result["overall"]["model_top1_covered"] == pytest.approx(table.top1_hit[rows].mean())
    assert result["name"] == "Stockfish depth 2" and result["rows"] == rows.tolist()


@pytest.mark.stockfish
def test_real_stockfish_finds_mate_in_one(stockfish) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    assert engine_moves([board], stockfish.path, depth=6)[0].uci() == "d1d8"
