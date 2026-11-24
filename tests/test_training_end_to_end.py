"""End to end: games -> built dataset -> Trainer -> saved model that plays like each rating level.

In the synthetic games below, low-rated White players always open 1.e4 and
high-rated ones always open 1.d4; everything after that is random. The start
position is identical in every game, so only the rating can tell the model which
move to expect. The trained model (loaded back from ``best_model.pt``) must have
learned that, through the whole pipeline: dataset building, bucket balancing,
the training loop and model saving.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import pytest

from chess_engine.data.dataset_builder import build_dataset
from chess_engine.model.inference import load_model
from chess_engine.training.train import Trainer
from dataset_helpers import random_game_rows, write_game_rows
from training_helpers import train_config


def _games(count: int, seed: int) -> list[dict]:
    rows = random_game_rows(count, seed=seed, prefix="e2e")
    rng = random.Random(seed)
    for row in rows:
        rating = rng.choice([rng.randrange(800, 1100), rng.randrange(1900, 2500)])
        row["white_elo"], row["black_elo"] = rating, rating
        board = chess.Board()
        moves = ["e4" if rating < 1500 else "d4"]
        board.push_san(moves[0])
        for _ in range(11):
            move = rng.choice(list(board.legal_moves))
            moves.append(board.san(move))
            board.push(move)
            if board.is_game_over():
                break
        row["moves"], row["num_plies"] = moves, len(moves)
        row["clocks"] = [180.0 - i for i in range(len(moves))]
    return rows


@pytest.mark.parametrize("balancing", ["weight", "sample"])
def test_pipeline_learns_rating_dependent_openings(tmp_path: Path, balancing: str) -> None:
    games = write_game_rows(_games(150, seed=3), tmp_path / "games.parquet")
    cfg = train_config(tmp_path, model={"channels": 16, "residual_blocks": 2, "rating_embedding_dim": 8},
                       epochs=8, batch_size=64, learning_rate=0.005, bucket_balancing=balancing,
                       eval_every_steps=10_000, log_every_steps=50)
    dataset = build_dataset(cfg, [games], tmp_path / "ds", show_progress=False).output_dir
    result = Trainer(cfg, tmp_path / "run", dataset, log=lambda _: None).train()
    assert result.best_model is not None

    predictor = load_model(result.final_model).predictor()
    low = predictor.predict(chess.Board(), rating=900).as_dict()
    high = predictor.predict(chess.Board(), rating=2300).as_dict()
    assert low["e2e4"] > 0.5 and low["e2e4"] > 3 * low["d2d4"]
    assert high["d2d4"] > 0.5 and high["d2d4"] > 3 * high["e2e4"]
