"""From PGN text to a saved style: pick the player's games, train the adapter, record how well it imitates them.

    info = build_style(cfg, base_model, pgn_text, name="ann", player="Ann")                 # a player's style
    info = build_style(cfg, base_model, pgn_text, name="me-2024", player="me", until=date)  # ghost: you, to a date
"""

from __future__ import annotations

import datetime
import statistics
from collections.abc import Callable

from chess_engine.config import Config
from chess_engine.eval.player_games import player_games_from_pgn
from chess_engine.model.network import RatingConditionedNet
from chess_engine.style.data import examples_from_games, select_games, split_games
from chess_engine.style.store import StyleInfo, StyleLibrary, check_name
from chess_engine.style.train import train_style


def library_for(cfg: Config) -> StyleLibrary:
    return StyleLibrary(cfg.paths.data_dir / cfg.style.directory)


def build_style(cfg: Config, base: RatingConditionedNet, pgn_text: str, name: str, player: str,
                until: datetime.date | None = None, since: datetime.date | None = None,
                rating: int | None = None, base_model: str = "", log: Callable[[str], None] | None = None,
                library: StyleLibrary | None = None) -> StyleInfo:
    """Train and save the style ``name`` from ``player``'s games in ``pgn_text``.

    A ghost is a style with ``until``: only games up to that date are used, so you can play your past self.
    ``rating``: the player's rating when the games do not say (default: their median rating in the games, or 1500).

    Raises:
        ValueError: too few games (``style.min_games``), or a bad name.
    """
    sc = cfg.style
    check_name(name)
    games = select_games(player_games_from_pgn(pgn_text, player=player), until=until, since=since)
    if len(games) < sc.min_games:
        raise ValueError(f"{len(games)} games of {player!r} found; a style needs at least {sc.min_games}")
    known = [g.rating for g in games if g.rating]
    typical = rating or (int(statistics.median(known)) if known else 1500)
    train_games, val_games = split_games(games, sc.val_fraction, sc.seed)
    train = examples_from_games(train_games, cfg.ratings, typical, sc.skip_opening_plies, sc.max_positions)
    val = examples_from_games(val_games, cfg.ratings, typical, sc.skip_opening_plies) if val_games else None
    if log:
        log(f"{len(games)} games of {player} ({len(train_games)} to learn from, {len(val_games)} held out); "
            f"{len(train)} positions")
    result = train_style(base, train, val, sc, log)
    metrics = {}
    if result.base is not None and result.styled is not None:
        metrics = {"base_loss": round(result.base.loss, 4), "styled_loss": round(result.styled.loss, 4),
                   "base_top1": round(result.base.top1, 4), "styled_top1": round(result.styled.top1, 4)}
    info = StyleInfo(name=name, kind="ghost" if until else "style", player=player, games=len(games),
                     positions=result.train_positions + result.val_positions, rating=typical,
                     until=until.isoformat() if until else None, created=datetime.date.today().isoformat(),
                     base_model=base_model, metrics=metrics)
    (library or library_for(cfg)).save(result.adapter, info, base)
    return info
