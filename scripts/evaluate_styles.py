"""Does a style imitate its player better than the rating model? Held-out games, several players (step 19).

For each player, a style is trained on their games (in a throwaway folder) and both the rating model and the
style are scored on the player's held-out games: average loss (lower = the player's real moves are more likely)
and top-1 move match.

Usage:
    python scripts/evaluate_styles.py --pgn games.pgn --players alice bob carol
    python scripts/evaluate_styles.py --pgn games.pgn --top 10          # the 10 players with the most games
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.eval.player_games import read_games
from chess_engine.model.inference import ModelFileError, load_model
from chess_engine.style.build import build_style
from chess_engine.style.store import StyleLibrary


def busiest_players(text: str, top: int) -> list[str]:
    """The ``top`` names that appear in the most games."""
    counts: Counter[str] = Counter()
    for game in read_games(text):
        counts.update({game.headers.get("White", "?"), game.headers.get("Black", "?")} - {"?"})
    return [name for name, _ in counts.most_common(top)]


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pgn", required=True)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--players", nargs="+")
    who.add_argument("--top", type=int)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        text = Path(args.pgn).read_text(encoding="utf-8")
        players = args.players or busiest_players(text, args.top)
        base = load_model(args.model or cfg.bot.model).model
    except (ConfigError, ModelFileError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        library = StyleLibrary(tmp)
        for index, player in enumerate(players):
            try:
                info = build_style(cfg, base, text, f"p{index}", player, library=library)
            except ValueError as exc:
                write(f"{player}: skipped ({exc})")
                continue
            m = info.metrics
            if not m:
                continue
            rows.append((player, info.games, m))
            write(f"{player:24} {info.games:4} games  loss {m['base_loss']:.3f} -> {m['styled_loss']:.3f}  "
                  f"top-1 {m['base_top1']:.1%} -> {m['styled_top1']:.1%}")
    if rows:
        n = len(rows)
        loss = sum(m["base_loss"] - m["styled_loss"] for _, _, m in rows) / n
        top1 = sum(m["styled_top1"] - m["base_top1"] for _, _, m in rows) / n
        better = sum(m["styled_loss"] < m["base_loss"] for _, _, m in rows)
        write(f"\n{n} players: held-out loss {loss:+.3f} lower on average (better for {better} of {n}); "
              f"top-1 {top1:+.1%} on average.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
