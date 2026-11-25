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
