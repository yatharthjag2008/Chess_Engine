"""Tests for replaying games into training samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_engine.config import Config, DatasetConfig
from chess_engine.data.pgn_stream import GameFilter, filter_games, iter_raw_games
from chess_engine.data.samples import (
    SAMPLE_COLUMNS,
    GameRejected,
    Samples,
    SampleStats,
    clocks_before_moves,
    game_to_samples,
    games_to_samples,
)
from chess_engine.encoding import encode_board, legal_move_indices, legal_move_mask, move_to_index
from chess_engine.encoding.compact import planes_from_compact

FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"
DEFAULT_DATASET = DatasetConfig()


def _game(**overrides: Any) -> dict[str, Any]:
    game = {
        "game_id": "fools0001",
        "white_elo": 1500,
        "black_elo": 1700,
        "base_seconds": 180,
        "increment_seconds": 0,
        "result": "0-1",
        "moves": ["f3", "e5", "g4", "Qh4#"],
        "clocks": [180.0, 179.0, 178.0, 175.0],
    }
    game.update(overrides)
    return game


@pytest.fixture(scope="module")
def fixture_games() -> list[dict[str, Any]]:
    from chess_engine.config import load_config

    cfg = load_config()
    with FIXTURE.open(encoding="utf-8") as f:
        return [r.to_dict() for r in filter_games(iter_raw_games(f), GameFilter.from_config(cfg.data))]


def _boards(moves: list[str]) -> list[chess.Board]:
    board = chess.Board()
    boards = []
    for san in moves:
        boards.append(board.copy())
        board.push_san(san)
    return boards


# ----------------------------------------------------------------------------- clocks


def test_clocks_before_moves() -> None:
    mine, theirs = clocks_before_moves([180, 179, 178, 175], base_seconds=180)
    np.testing.assert_array_equal(mine, [180, 180, 180, 179])
    np.testing.assert_array_equal(theirs, [180, 180, 179, 178])


def test_clocks_before_moves_empty() -> None:
    mine, theirs = clocks_before_moves([], base_seconds=60)
    assert len(mine) == len(theirs) == 0


# ----------------------------------------------------------------------------- one game


def test_short_game_columns(default_config: Config) -> None:
    samples = game_to_samples(_game(), 7, default_config.ratings, DEFAULT_DATASET)
    assert len(samples) == 4
    np.testing.assert_array_equal(samples["ply"], [0, 1, 2, 3])
    np.testing.assert_array_equal(samples["rating"], [1500, 1700, 1500, 1700])
    np.testing.assert_array_equal(samples["opp_rating"], [1700, 1500, 1700, 1500])
    np.testing.assert_array_equal(samples["bucket"], [3, 4, 3, 4])
    np.testing.assert_array_equal(samples["result"], [-1, 1, -1, 1])
    np.testing.assert_array_equal(samples["clock"], [180, 180, 180, 179])
    np.testing.assert_array_equal(samples["opp_clock"], [180, 180, 179, 178])
    np.testing.assert_array_equal(samples["time_spent"], [0, 1, 2, 4])
    np.testing.assert_array_equal(samples["game"], [7] * 4)
    np.testing.assert_array_equal(samples["base_seconds"], [180] * 4)
    for name, (dtype, shape) in SAMPLE_COLUMNS.items():
        assert samples[name].dtype == dtype and samples[name].shape == (4, *shape)


def test_samples_match_the_encoders(default_config: Config) -> None:
    game = _game()
    samples = game_to_samples(game, 0, default_config.ratings, DEFAULT_DATASET)
    boards = _boards(game["moves"])
    planes = planes_from_compact(samples["pieces"], samples["flags"], samples["ep"], samples["halfmove"])
    for i, board in enumerate(boards):
        np.testing.assert_array_equal(planes[i], encode_board(board))
        assert samples["move"][i] == move_to_index(board.parse_san(game["moves"][i]), board.turn)
        np.testing.assert_array_equal(samples.legal_moves_of(i), legal_move_indices(board))
    np.testing.assert_array_equal(samples.legal_mask(), np.stack([legal_move_mask(b) for b in boards]))


def test_increment_is_added_to_time_spent(default_config: Config) -> None:
    # Real Lichess 3+2 clocks: the first move of each side is untimed and gets no increment.
    game = _game(base_seconds=180, increment_seconds=2, clocks=[180.0, 180.0, 181.0, 175.0])
    samples = game_to_samples(game, 0, default_config.ratings, DEFAULT_DATASET)
    # ply 2: White had 180, gets +2, ends with 181 -> 1 second spent; ply 3: 180 + 2 - 175 = 7.
    np.testing.assert_array_equal(samples["time_spent"], [0, 0, 1, 7])


def test_draw_result_is_zero(default_config: Config) -> None:
    samples = game_to_samples(_game(result="1/2-1/2"), 0, default_config.ratings, DEFAULT_DATASET)
    assert (samples["result"] == 0).all()


def test_skip_first_plies(default_config: Config) -> None:
    stats = SampleStats()
    samples = game_to_samples(_game(), 0, default_config.ratings, DatasetConfig(skip_first_plies=3), stats)
    np.testing.assert_array_equal(samples["ply"], [3])
    assert stats.skipped_opening == 3


def test_min_clock_seconds(default_config: Config) -> None:
    stats = SampleStats()
    samples = game_to_samples(_game(), 0, default_config.ratings, DatasetConfig(min_clock_seconds=179.5), stats)
    np.testing.assert_array_equal(samples["ply"], [0, 1, 2])
    assert stats.skipped_low_clock == 1


def test_everything_skipped_gives_empty_samples(default_config: Config) -> None:
    samples = game_to_samples(_game(), 0, default_config.ratings, DatasetConfig(skip_first_plies=10))
    assert len(samples) == 0


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"moves": ["e4", "e4"], "clocks": [180.0, 180.0]}, "illegal_move"),
        ({"moves": ["e4", "Zz9"], "clocks": [180.0, 180.0]}, "illegal_move"),
        ({"clocks": [180.0, 179.0]}, "bad_clocks"),
        ({"white_elo": 600}, "rating_below_buckets"),
        ({"result": "*"}, "unknown_result"),
    ],
)
def test_rejected_games(default_config: Config, overrides: dict, reason: str) -> None:
    with pytest.raises(GameRejected) as info:
        game_to_samples(_game(**overrides), 0, default_config.ratings, DEFAULT_DATASET)
    assert info.value.reason == reason


# ----------------------------------------------------------------------------- many games


def test_real_games_every_played_move_is_legal(default_config: Config, fixture_games: list) -> None:
    samples, stats, per_game = games_to_samples(fixture_games, 100, default_config.ratings, DEFAULT_DATASET)
    assert stats.games_in == stats.games_used == len(fixture_games) == 8
    assert stats.positions == len(samples) == sum(per_game) == sum(len(g["moves"]) for g in fixture_games)
    mask = samples.legal_mask()
    assert mask[np.arange(len(samples)), samples["move"]].all()
    assert (samples["legal_count"] == mask.sum(axis=1)).all()
    assert sorted(set(samples["game"])) == list(range(100, 108))
    assert (samples["clock"] >= 0).all() and (samples["time_spent"] >= 0).all()


def test_games_to_samples_counts_rejections(default_config: Config) -> None:
    games = [_game(), _game(result="*"), _game(moves=["e4", "e4"], clocks=[1.0, 1.0]), _game()]
    samples, stats, per_game = games_to_samples(games, 0, default_config.ratings, DEFAULT_DATASET)
    assert per_game == [4, 0, 0, 4]
    assert stats.games_used == 2 and stats.games_in == 4
    assert stats.games_rejected == {"unknown_result": 1, "illegal_move": 1}
    np.testing.assert_array_equal(np.unique(samples["game"]), [0, 3])


def test_stats_merge() -> None:
    a = SampleStats(games_in=2, positions=10, games_rejected={"x": 1})
    b = SampleStats(games_in=3, positions=5, skipped_opening=4, games_rejected={"x": 2, "y": 1})
    a.merge(b)
    assert a.as_dict() == {
        "games_in": 5, "games_used": 0, "positions": 15, "skipped_opening": 4,
        "skipped_low_clock": 0, "games_rejected": {"x": 3, "y": 1},
    }


# ----------------------------------------------------------------------------- the Samples container


def test_take_reorders_legal_moves(default_config: Config, fixture_games: list) -> None:
    samples, _, _ = games_to_samples(fixture_games[:2], 0, default_config.ratings, DEFAULT_DATASET)
    order = np.random.default_rng(0).permutation(len(samples))
    taken = samples.take(order)
    for new, old in enumerate(order[:50]):
        np.testing.assert_array_equal(taken.legal_moves_of(new), samples.legal_moves_of(int(old)))
        assert taken["move"][new] == samples["move"][old]
    np.testing.assert_array_equal(taken.legal_mask(), samples.legal_mask(order))


def test_take_with_repeats_and_empty(default_config: Config) -> None:
    samples = game_to_samples(_game(), 0, default_config.ratings, DEFAULT_DATASET)
    repeated = samples.take([1, 1, 0])
    np.testing.assert_array_equal(repeated["ply"], [1, 1, 0])
    assert len(samples.take([])) == 0


def test_concat_and_empty(default_config: Config) -> None:
    one = game_to_samples(_game(), 0, default_config.ratings, DEFAULT_DATASET)
    both = Samples.concat([one, Samples.empty(), one])
    assert len(both) == 8
    np.testing.assert_array_equal(both.legal_mask()[4:], one.legal_mask())
    assert len(Samples.concat([])) == 0


def test_samples_validation(default_config: Config) -> None:
    one = game_to_samples(_game(), 0, default_config.ratings, DEFAULT_DATASET)
    with pytest.raises(ValueError, match="missing"):
        Samples({k: v for k, v in one.columns.items() if k != "clock"}, one.legal_moves)
    with pytest.raises(ValueError, match="shape"):
        Samples({**one.columns, "clock": one["clock"][:2]}, one.legal_moves)
    with pytest.raises(ValueError, match="legal_count"):
        Samples(one.columns, one.legal_moves[:-1])
