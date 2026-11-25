"""Tests for match games and adjudication (scripted players and judges, no engines needed)."""

from __future__ import annotations

import chess
import pytest

from chess_engine.eval.arena import MatchGame, play_match_game


class Scripted:
    """Plays the given moves in order (SAN), then the first legal move."""

    def __init__(self, name: str, moves: list[str] | None = None, rating: int | None = None) -> None:
        self.name, self.rating = name, rating
        self.moves = list(moves or [])
        self.seen_opponent: list[int | None] = []

    def choose(self, board: chess.Board, opponent_rating: int | None = None) -> chess.Move:
        self.seen_opponent.append(opponent_rating)
        if self.moves:
            return board.parse_san(self.moves.pop(0))
        return min(board.legal_moves, key=lambda m: m.uci())

    def close(self) -> None:
        pass


class FixedJudge:
    """Returns the evaluations in order (White's view), repeating the last one."""

    def __init__(self, *scores: int) -> None:
        self.scores = list(scores)
        self.calls = 0

    def evaluate(self, board: chess.Board) -> int:
        self.calls += 1
        return self.scores[min(self.calls, len(self.scores)) - 1]


def test_natural_end_and_ratings_passed() -> None:
    white = Scripted("A", ["f3", "g4"], rating=1500)
    black = Scripted("B", ["e5", "Qh4#"], rating=1700)
    game = play_match_game(white, black)
    assert game.result == "0-1" and game.termination == "checkmate" and not game.adjudicated
    assert game.moves == ["f2f3", "e7e5", "g2g4", "d8h4"] and game.plies == 4
    assert white.seen_opponent == [1700, 1700] and black.seen_opponent == [1500, 1500]
    assert game.white_score == 0.0 and game.as_dict()["black"] == "B"


def test_decisive_advantage_is_adjudicated_after_a_streak() -> None:
    judge = FixedJudge(0, 900, 900, 50, 900, 900, 900)  # the streak is broken once, then held 3 plies
    game = play_match_game(Scripted("A"), Scripted("B"), judge, adjudicate_cp=800, adjudicate_plies=3)
    assert game.result == "1-0" and game.adjudicated and game.termination.startswith("adjudicated")
    assert game.plies == 7 and judge.calls == 7

    black_wins = play_match_game(Scripted("A"), Scripted("B"), FixedJudge(-1000), adjudicate_plies=2)
    assert black_wins.result == "0-1" and black_wins.plies == 2


def test_move_limit_uses_the_last_evaluation() -> None:
    level = play_match_game(Scripted("A"), Scripted("B"), FixedJudge(100), max_plies=6)
    assert level.result == "1/2-1/2" and level.termination == "move limit (6 plies)" and level.plies == 6
    ahead = play_match_game(Scripted("A"), Scripted("B"), FixedJudge(0, 0, 0, 0, 0, -900), max_plies=6,
                            adjudicate_plies=50)
    assert ahead.result == "0-1"
    no_judge = play_match_game(Scripted("A"), Scripted("B"), None, max_plies=4)
    assert no_judge.result == "1/2-1/2" and no_judge.plies == 4


def test_starting_position_and_score() -> None:
    game = play_match_game(Scripted("A", ["Ra7"]), Scripted("B"), fen="7k/8/6K1/8/8/8/R7/1R6 w - - 0 1")
    # Black's only move is Kg8; White's scripted list is empty after Ra7 and plays its first legal move.
    assert game.moves[:2] == ["a2a7", "h8g8"]
    assert MatchGame("A", "B", "1/2-1/2", "stalemate", 10).white_score == pytest.approx(0.5)
