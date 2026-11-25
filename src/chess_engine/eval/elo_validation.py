"""Check the rating dial: do bots set to higher ratings actually play stronger chess?

Each bot rating plays every Stockfish level (fixed, known strengths), and bots
play each other (neighbouring ratings, or all pairs). Colours alternate. Every
finished game is appended to a JSON-lines file, so a long run can be stopped and
resumed without replaying anything.

    players = make_players(cfg, predictor, stockfish_path)      # {"bot:1400": ..., "sf:1600": ...}
    run_matches(schedule(cfg.elo_validation), players, judge, "reports/elo/games.jsonl", games=20)

Player keys are ``bot:<rating>`` and ``sf:<UCI_Elo>``.

:func:`analyze` turns the log into results: a score table per pairing, every
bot's measured Elo on Stockfish's scale (anchored to the levels' UCI_Elo, with
bootstrap intervals) and how measured strength follows the dial.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_engine.bot.player import BotPlayer
from chess_engine.bot.sampling import SamplingSettings
from chess_engine.config import EloValidationConfig
from chess_engine.eval.arena import MatchGame, play_match_game
from chess_engine.eval.elo_fit import Result, bootstrap_ratings, dial_summary
from chess_engine.eval.players import BotAgent, Judge, Player, StockfishPlayer
from chess_engine.model.inference import MovePredictor


@dataclass(frozen=True)
class Pairing:
    """Two players (keys) who play a mini-match."""

    a: str
    b: str

    @property
    def key(self) -> str:
        return f"{self.a} vs {self.b}"


def bot_key(rating: int) -> str:
    return f"bot:{rating}"


def stockfish_key(elo: int) -> str:
    return f"sf:{elo}"


def schedule(cfg: EloValidationConfig) -> list[Pairing]:
    """Every bot against every Stockfish level, plus bot-vs-bot games per ``cfg.self_play``."""
    bots = [bot_key(r) for r in sorted(cfg.bot_ratings)]
    pairings = [Pairing(b, stockfish_key(e)) for b in bots for e in sorted(cfg.stockfish_elos)]
    if cfg.self_play == "adjacent":
        pairings += [Pairing(a, b) for a, b in zip(bots, bots[1:])]
    elif cfg.self_play == "all":
        pairings += [Pairing(a, b) for a, b in itertools.combinations(bots, 2)]
    return pairings


def make_players(
    cfg: EloValidationConfig,
    predictor: MovePredictor,
    stockfish_command: str | Sequence[str] | None,
    sampling: SamplingSettings | None = None,
) -> dict[str, Player]:
    """One bot per rating (sharing the model, each with its own seed) and one Stockfish per level.

    With ``stockfish_command`` None, only the bots are created (Stockfish pairings are then skipped).
    """
    sampling = sampling or SamplingSettings()
    players: dict[str, Player] = {}
    for i, rating in enumerate(sorted(cfg.bot_ratings)):
        players[bot_key(rating)] = BotAgent(BotPlayer(predictor, rating, sampling, seed=cfg.seed * 1000 + i))
    if stockfish_command is not None:
        for elo in sorted(cfg.stockfish_elos):
            players[stockfish_key(elo)] = StockfishPlayer(stockfish_command, elo, cfg.stockfish_move_seconds)
    return players


class GameLog:
    """Finished games, one JSON object per line: the :class:`MatchGame` plus its pairing and game number."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut off by an interruption
        return out

    def done(self) -> set[tuple[str, int]]:
        return {(r["pairing"], r["index"]) for r in self.records()}

    def append(self, pairing: Pairing, index: int, white_key: str, black_key: str, game: MatchGame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"pairing": pairing.key, "index": index, "white_key": white_key, "black_key": black_key,
                  **game.as_dict()}
        # After an interrupted write the file may end mid-line; start a fresh line so this record survives.
        needs_newline = False
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                needs_newline = f.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(("\n" if needs_newline else "") + json.dumps(record) + "\n")


def run_matches(
    pairings: Sequence[Pairing],
    players: Mapping[str, Player],
    judge: Judge | None,
    log_path: str | Path,
    games: int,
    max_plies: int = 300,
    adjudicate_cp: int = 800,
    adjudicate_plies: int = 8,
    progress: Callable[[Pairing, int, MatchGame], None] | None = None,
) -> GameLog:
    """Play ``games`` games per pairing (``a`` is White in even-numbered games), skipping games already logged.

    Pairings with a player missing from ``players`` are skipped.
    """
    log = GameLog(log_path)
    done = log.done()
    for pairing in pairings:
        if pairing.a not in players or pairing.b not in players:
            continue
        for index in range(games):
            if (pairing.key, index) in done:
                continue
            white_key, black_key = (pairing.a, pairing.b) if index % 2 == 0 else (pairing.b, pairing.a)
            game = play_match_game(players[white_key], players[black_key], judge, max_plies, adjudicate_cp,
                                   adjudicate_plies)
            log.append(pairing, index, white_key, black_key, game)
            if progress is not None:
                progress(pairing, index, game)
    return log


# --------------------------------------------------------------------------- analysis


def _rating_of(key: str) -> int:
    return int(key.split(":", 1)[1])


def _score_for(record: Mapping[str, Any], key: str) -> float:
    white = {"1-0": 1.0, "0-1": 0.0}.get(record["result"], 0.5)
    return white if record["white_key"] == key else 1 - white


def pairing_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per pairing: games, A's wins / draws / losses and score, adjudicated games, average length."""
    rows: dict[str, dict[str, Any]] = {}
    for r in records:
        a, b = r["pairing"].split(" vs ")
        row = rows.setdefault(r["pairing"], {"pairing": r["pairing"], "a": a, "b": b, "games": 0, "wins": 0,
                                             "draws": 0, "losses": 0, "adjudicated": 0, "plies": 0})
        score = _score_for(r, a)
        row["games"] += 1
        row["wins" if score == 1 else "losses" if score == 0 else "draws"] += 1
        row["adjudicated"] += int(r.get("adjudicated", False))
        row["plies"] += r["plies"]
    for row in rows.values():
        row["score"] = (row["wins"] + 0.5 * row["draws"]) / row["games"]
        row["average_plies"] = row.pop("plies") / row["games"]
    return list(rows.values())


def analyze(records: Sequence[Mapping[str, Any]], bootstrap: int = 200, seed: int = 0) -> dict[str, Any]:
    """Everything the report shows, from logged games (see the module docstring).

    Returns:
        ``pairings`` (see :func:`pairing_table`), ``bots`` (per dial rating: measured Elo, interval,
        score against every opponent), ``stockfish`` (the anchor levels), ``dial`` (from
        :func:`chess_engine.eval.elo_fit.dial_summary`), ``terminations`` and ``games``.
    """
    if not records:
        raise ValueError("no games to analyze")
    results = [Result(r["white_key"], r["black_key"], _score_for(r, r["white_key"])) for r in records]
    keys = {r["white_key"] for r in records} | {r["black_key"] for r in records}
    anchors = {k: float(_rating_of(k)) for k in keys if k.startswith("sf:")}
    bots = sorted((k for k in keys if k.startswith("bot:")), key=_rating_of)
    center = sum(_rating_of(k) for k in bots) / len(bots) if bots else 1500.0
    estimate, intervals, fits = bootstrap_ratings(results, [r["pairing"] for r in records], anchors or None,
                                                  samples=bootstrap, seed=seed, center=center)
    table = pairing_table(records)
    bot_rows = []
    for key in bots:
        scores = {}
        for row in table:
            if key in (row["a"], row["b"]):
                other = row["b"] if row["a"] == key else row["a"]
                scores[other] = row["score"] if row["a"] == key else 1 - row["score"]
        bot_rows.append({"key": key, "dial": _rating_of(key), "measured": estimate[key],
                         "interval": list(intervals[key]), "scores": scores})
    return {
        "games": len(records),
        "pairings": table,
        "bots": bot_rows,
        "stockfish": sorted(anchors.values()),
        "anchored": bool(anchors),
        "dial": dial_summary({k: _rating_of(k) for k in bots}, estimate, fits),
        "terminations": dict(Counter(r["termination"] for r in records).most_common()),
        "adjudicated_share": sum(bool(r.get("adjudicated")) for r in records) / len(records),
    }
