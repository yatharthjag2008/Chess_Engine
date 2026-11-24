"""Tests for predicting moves on real boards and for saving / loading models."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from chess_engine.config import Config, ModelConfig, RatingsConfig
from chess_engine.model.inference import (
    MovePredictor,
    ModelFileError,
    load_model,
    save_model,
)
from chess_engine.model.network import RatingConditionedNet

SMALL = ModelConfig(architecture="resnet", channels=16, residual_blocks=2, rating_embedding_dim=8)
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1"


@pytest.fixture()
def model() -> RatingConditionedNet:
    torch.manual_seed(0)
    net = RatingConditionedNet(SMALL, num_buckets=8)
    for block in net.blocks:  # make the rating matter, as after training
        torch.nn.init.normal_(block.modulation.weight, std=0.5)
    return net


@pytest.fixture()
def predictor(model: RatingConditionedNet, default_config: Config) -> MovePredictor:
    return MovePredictor(model, default_config.ratings)


def test_predict_covers_every_legal_move(predictor: MovePredictor) -> None:
    board = chess.Board()
    prediction = predictor.predict(board, rating=1500)
    assert len(prediction) == 20
    assert {p.move for p in prediction.moves} == set(board.legal_moves)
    assert sum(p.prob for p in prediction.moves) == pytest.approx(1.0, abs=1e-5)
    probs = [p.prob for p in prediction.moves]
    assert probs == sorted(probs, reverse=True)
    assert prediction[0].san == board.san(prediction[0].move)
    assert set(prediction.as_dict()) == {m.uci() for m in board.legal_moves}


def test_black_to_move_and_special_moves(predictor: MovePredictor) -> None:
    board = chess.Board(KIWIPETE)
    moves = {p.san for p in predictor.predict(board, rating=2000).moves}
    assert moves == {board.san(m) for m in board.legal_moves}
    assert "O-O" in moves and "O-O-O" in moves


def test_promotions_are_named_correctly(predictor: MovePredictor) -> None:
    board = chess.Board("1r5k/2P5/8/8/8/8/5p2/K5R1 w - - 0 1")
    sans = {p.san for p in predictor.predict(board, rating=1200).moves}
    promotions = {board.san(m) for m in board.legal_moves if m.promotion}
    assert len(promotions) == 8 and promotions <= sans  # c8 and cxb8, each to Q, R, B, N
    assert {"c8=Q+", "c8=N", "cxb8=N"} <= sans


def test_wdl_sums_to_one(predictor: MovePredictor) -> None:
    wdl = predictor.predict(chess.Board(), rating=1500).wdl
    assert wdl is not None and sum(wdl) == pytest.approx(1.0, abs=1e-5)


def test_rating_changes_the_prediction(predictor: MovePredictor) -> None:
    low = predictor.predict(chess.Board(), rating=850).as_dict()
    high = predictor.predict(chess.Board(), rating=2300).as_dict()
    assert low != pytest.approx(high, abs=1e-6)


def test_batch_equals_one_at_a_time(predictor: MovePredictor) -> None:
    boards = [chess.Board(), chess.Board(KIWIPETE)]
    batch = predictor.predict_many(boards, ratings=[1100, 1900], clocks=[100.0, 20.0], base_seconds=180)
    for board, rating, clock, pred in zip(boards, (1100, 1900), (100.0, 20.0), batch):
        single = predictor.predict(board, rating=rating, clocks=clock, base_seconds=180)
        assert pred.as_dict() == pytest.approx(single.as_dict(), abs=1e-5)


def test_temperature_sharpens(predictor: MovePredictor) -> None:
    normal = predictor.predict(chess.Board(), rating=1500)
    sharp = predictor.predict(chess.Board(), rating=1500, temperature=0.1)
    assert sharp[0].move == normal[0].move and sharp[0].prob > normal[0].prob


def test_bad_inputs(predictor: MovePredictor) -> None:
    with pytest.raises(ValueError, match="no legal moves"):
        predictor.predict(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), rating=1500)
    with pytest.raises(ValueError, match="one per board"):
        predictor.predict_many([chess.Board(), chess.Board()], ratings=[1500, 1600, 1700])
    assert predictor.predict_many([], ratings=1500) == []


def test_bucket_count_must_match(model: RatingConditionedNet) -> None:
    with pytest.raises(ValueError, match="buckets"):
        MovePredictor(model, RatingsConfig(bucket_edges=[800, 1500]))


def test_save_and_load_roundtrip(model: RatingConditionedNet, default_config: Config, tmp_path: Path) -> None:
    path = save_model(model, tmp_path / "models" / "m.pt", default_config.ratings, extra={"step": 123})
    assert not list(path.parent.glob("*.tmp"))
    loaded = load_model(path)
    assert loaded.extra == {"step": 123}
    assert loaded.ratings.bucket_edges == default_config.ratings.bucket_edges
    assert loaded.model.config == model.config and not loaded.model.training
    board = chess.Board(KIWIPETE)
    before = MovePredictor(model, default_config.ratings).predict(board, rating=1700).as_dict()
    after = loaded.predictor().predict(board, rating=1700).as_dict()
    assert before == pytest.approx(after, abs=1e-6)


def test_load_rejects_other_encodings(model: RatingConditionedNet, default_config: Config, tmp_path: Path) -> None:
    path = save_model(model, tmp_path / "m.pt", default_config.ratings)
    payload = torch.load(path, weights_only=True)
    payload["encoding"]["num_moves"] = 4672
    torch.save(payload, path)
    with pytest.raises(ModelFileError, match="encoding"):
        load_model(path)


def test_load_rejects_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelFileError, match="readable"):
        load_model(bad)
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pt")
