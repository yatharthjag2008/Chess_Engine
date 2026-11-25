"""Tests for the Elo-dial validation: schedule, players, resumable match runs."""

from __future__ import annotations

import sys
from pathlib import Path

import chess

from chess_engine.config import EloValidationConfig
from chess_engine.eval.elo_validation import GameLog, Pairing, make_players, run_matches, schedule
from chess_engine.eval.players import BotAgent, StockfishPlayer
from bot_helpers import small_predictor

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uci_engine.py"), "--first-legal"]


class Quick:
    """A player that plays its first legal move (fast and deterministic)."""

    def __init__(self, name: str, rating: int) -> None:
        self.name, self.rating = name, rating

    def choose(self, board: chess.Board, opponent_rating: int | None = None) -> chess.Move:
        return min(board.legal_moves, key=lambda m: m.uci())

    def close(self) -> None:
        pass


def test_schedule() -> None:
    cfg = EloValidationConfig(bot_ratings=[1400, 800, 2000], stockfish_elos=[1600, 1320], self_play="adjacent")
    keys = [p.key for p in schedule(cfg)]
    assert keys[:2] == ["bot:800 vs sf:1320", "bot:800 vs sf:1600"]
    assert len(keys) == 3 * 2 + 2 and keys[-2:] == ["bot:800 vs bot:1400", "bot:1400 vs bot:2000"]
    assert len(schedule(cfg.model_copy(update={"self_play": "all"}))) == 6 + 3
    assert len(schedule(cfg.model_copy(update={"self_play": "none"}))) == 6


def test_make_players() -> None:
    cfg = EloValidationConfig(bot_ratings=[900, 1500], stockfish_elos=[1320])
    players = make_players(cfg, small_predictor(), FAKE)
    assert set(players) == {"bot:900", "bot:1500", "sf:1320"}
    assert isinstance(players["bot:900"], BotAgent) and isinstance(players["sf:1320"], StockfishPlayer)
    assert players["bot:900"].bot.rng is not players["bot:1500"].bot.rng
    assert set(make_players(cfg, small_predictor(), None)) == {"bot:900", "bot:1500"}


def test_run_alternates_colours_logs_and_resumes(tmp_path: Path) -> None:
    players = {"bot:900": Quick("bot 900", 900), "bot:1500": Quick("bot 1500", 1500)}
    pairings = [Pairing("bot:900", "bot:1500"), Pairing("bot:900", "sf:1320")]  # the second is skipped
    seen = []
    log = run_matches(pairings, players, None, tmp_path / "games.jsonl", games=3, max_plies=10,
                      progress=lambda p, i, g: seen.append(i))
    records = log.records()
    assert seen == [0, 1, 2] and len(records) == 3
    assert [r["white_key"] for r in records] == ["bot:900", "bot:1500", "bot:900"]
    assert records[0]["pairing"] == "bot:900 vs bot:1500" and records[0]["plies"] == 10
    assert log.done() == {("bot:900 vs bot:1500", i) for i in range(3)}

    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"pairing": "cut off')                 # an interrupted write is ignored
    more = run_matches(pairings, players, None, log.path, games=5, max_plies=10,
                       progress=lambda p, i, g: seen.append(i))
    assert seen == [0, 1, 2, 3, 4] and len(more.records()) == 5   # only the 2 missing games were played
    assert GameLog(tmp_path / "missing.jsonl").records() == []


def test_run_against_a_fake_stockfish(tmp_path: Path) -> None:
    cfg = EloValidationConfig(bot_ratings=[900, 1500], stockfish_elos=[1320], games_per_pairing=2)
    players = make_players(cfg, small_predictor(), FAKE)
    try:
        log = run_matches([Pairing("bot:900", "sf:1320")], players, None, tmp_path / "g.jsonl", games=2, max_plies=12)
    finally:
        for p in players.values():
            p.close()
    records = log.records()
    assert [r["white"] for r in records] == ["bot 900", "Stockfish 1320"] and all(r["plies"] <= 12 for r in records)


def _record(pairing: str, index: int, white_key: str, black_key: str, result: str, adjudicated: bool = False,
            plies: int = 80, termination: str = "checkmate") -> dict:
    return {"pairing": pairing, "index": index, "white_key": white_key, "black_key": black_key, "result": result,
            "adjudicated": adjudicated, "plies": plies, "termination": termination, "white": white_key,
            "black": black_key, "moves": []}


def _synthetic_log() -> list[dict]:
    """bot:1600 is clearly stronger than bot:1000; Stockfish 1320 sits in between."""
    import numpy as np

    from chess_engine.eval.elo_fit import expected_score

    true = {"bot:1000": 1150, "bot:1600": 1600, "sf:1320": 1320}
    rng = np.random.default_rng(1)
    records = []
    for a, b in [("bot:1000", "sf:1320"), ("bot:1600", "sf:1320"), ("bot:1000", "bot:1600")]:
        for i in range(60):
            white, black = (a, b) if i % 2 == 0 else (b, a)
            e = expected_score(true[white], true[black])
            result = "1-0" if rng.random() < e else "0-1"
            records.append(_record(f"{a} vs {b}", i, white, black, result, adjudicated=i % 3 == 0))
    return records


def test_pairing_table_scores_from_the_first_players_view() -> None:
    from chess_engine.eval.elo_validation import pairing_table

    records = [_record("bot:900 vs sf:1320", 0, "bot:900", "sf:1320", "1-0", plies=60),
               _record("bot:900 vs sf:1320", 1, "sf:1320", "bot:900", "1-0", plies=40, adjudicated=True),
               _record("bot:900 vs sf:1320", 2, "bot:900", "sf:1320", "1/2-1/2", plies=50)]
    (row,) = pairing_table(records)
    assert (row["wins"], row["draws"], row["losses"]) == (1, 1, 1) and row["score"] == 0.5
    assert row["adjudicated"] == 1 and row["average_plies"] == 50 and row["a"] == "bot:900"


def test_analyze_puts_bots_on_the_stockfish_scale() -> None:
    from chess_engine.eval.elo_validation import analyze

    result = analyze(_synthetic_log(), bootstrap=60)
    assert result["games"] == 180 and result["anchored"] and result["stockfish"] == [1320.0]
    low, high = (b for b in result["bots"])
    assert low["key"] == "bot:1000" and high["dial"] == 1600
    assert low["measured"] < 1320 < high["measured"]
    assert low["interval"][0] < low["measured"] < low["interval"][1]
    assert 0 < low["scores"]["sf:1320"] < 0.5 < high["scores"]["sf:1320"]
    assert high["scores"]["bot:1000"] > 0.5
    assert abs(result["dial"]["spearman"] - 1.0) < 1e-9 and result["dial"]["steps"][0]["prob_stronger"] > 0.9
    assert result["terminations"] == {"checkmate": 180} and abs(result["adjudicated_share"] - 1 / 3) < 1e-9


def test_analyze_without_stockfish_and_errors() -> None:
    import pytest

    from chess_engine.eval.elo_validation import analyze

    only_bots = [r for r in _synthetic_log() if "sf:" not in r["pairing"]]
    result = analyze(only_bots, bootstrap=20)
    assert not result["anchored"] and result["stockfish"] == []
    measured = [b["measured"] for b in result["bots"]]
    assert sum(measured) / 2 == pytest.approx(1300)  # mean fixed at the mean dial rating
    with pytest.raises(ValueError):
        analyze([])
