"""Check the rating dial (step 11): bots at different ratings play Stockfish levels and each other.

Games are appended to reports/elo/<run>_<model>/games.jsonl as they finish, so the
run can be stopped (Ctrl+C) and continued with the same command. The report
(report.md, results.json, charts) is written at the end, or with --analyze-only.

Usage:
    python scripts/elo_validation.py                                   # the config's elo_validation settings
    python scripts/elo_validation.py --games 10 --ratings 1000 1500 2000
    python scripts/elo_validation.py --temperature 0.5 --out reports/elo/t05
    python scripts/elo_validation.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from chess_engine.bot.player import BotError, sampling_from_config
from chess_engine.bot.sampling import SamplingSettings
from chess_engine.config import ConfigError, EloValidationConfig, load_config
from chess_engine.eval.elo_report import CHARTS, plot_dial, plot_scores, render_report
from chess_engine.eval.elo_validation import GameLog, analyze, make_players, run_matches, schedule
from chess_engine.eval.evaluate import model_name, shown_path
from chess_engine.eval.players import Judge
from chess_engine.model.inference import ModelFileError, load_model
from chess_engine.stockfish import StockfishError, find_stockfish
from chess_engine.training.run_log import json_safe


def main(argv: list[str] | None = None, stockfish_command=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="model file (default: bot.model)")
    parser.add_argument("--ratings", type=int, nargs="+", default=None, help="bot dial settings")
    parser.add_argument("--stockfish-elos", type=int, nargs="*", default=None, help="Stockfish UCI_Elo levels")
    parser.add_argument("--games", type=int, default=None, help="games per pairing")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--self-play", choices=["none", "adjacent", "all"], default=None)
    parser.add_argument("--no-stockfish", action="store_true", help="bots only (relative strengths, no judge)")
    parser.add_argument("--out", default=None, help="output directory (default: reports/elo/<run>_<model>)")
    parser.add_argument("--analyze-only", action="store_true", help="only write the report from the logged games")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        overrides = {k: v for k, v in (("bot_ratings", args.ratings), ("stockfish_elos", args.stockfish_elos),
                                       ("games_per_pairing", args.games), ("temperature", args.temperature),
                                       ("self_play", args.self_play)) if v is not None}
        e = EloValidationConfig.model_validate({**cfg.elo_validation.model_dump(), **overrides})
        model_path = Path(args.model).resolve() if args.model else cfg.bot.model
        out = Path(args.out) if args.out else cfg.paths.reports_dir / "elo" / model_name(model_path)
        log_path = out / "games.jsonl"
        base = sampling_from_config(cfg.bot)
        temperature = e.temperature if e.temperature is not None else base.temperature
        start = time.perf_counter()
        command = None
        if not args.analyze_only:
            loaded = load_model(model_path)
            if not args.no_stockfish:
                command = stockfish_command or str(find_stockfish(cfg.stockfish.path))
            sampling = SamplingSettings(temperature, base.top_k, base.top_p, base.min_prob)
            players = make_players(e, loaded.predictor(), command, sampling)
            pairings = schedule(e)
            total = len([p for p in pairings if p.a in players and p.b in players]) * e.games_per_pairing
            already = len(GameLog(log_path).done())
            print(f"{total} games planned, {already} already played; logging to {log_path}", flush=True)

            def progress(pairing, index, game) -> None:
                print(f"[{time.perf_counter() - start:6.0f}s] {pairing.key:<24} #{index:<3} {game.result:<7} "
                      f"{game.plies:>3} plies  {game.termination}", flush=True)

            judge = Judge(command, e.judge_nodes) if command else None
            try:
                run_matches(pairings, players, judge, log_path, e.games_per_pairing, e.max_plies,
                            e.adjudicate_cp, e.adjudicate_plies, progress)
            except KeyboardInterrupt:
                print("\nstopped; run the same command again to continue", file=sys.stderr)
                return 130
            finally:
                for player in players.values():
                    player.close()
                if judge is not None:
                    judge.close()
        records = GameLog(log_path).records()
        if not records:
            raise ValueError(f"no games in {log_path}")
        analysis = analyze(records, bootstrap=args.bootstrap, seed=e.seed)
    except (ConfigError, BotError, ModelFileError, StockfishError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    setup = {"model_name": model_name(model_path), "model_path": shown_path(model_path), "temperature": temperature,
             "stockfish_move_seconds": e.stockfish_move_seconds, "judge_nodes": e.judge_nodes,
             "adjudicate_cp": e.adjudicate_cp, "adjudicate_plies": e.adjudicate_plies, "max_plies": e.max_plies,
             "bootstrap": args.bootstrap, "seconds": time.perf_counter() - start, "self_play": e.self_play,
             "stockfish_version": "19"}
    plot_dial(analysis, out / CHARTS["dial"])
    plot_scores(analysis, out / CHARTS["scores"])
    (out / "results.json").write_text(json.dumps(json_safe({"setup": setup, **analysis}), indent=1) + "\n",
                                      encoding="utf-8")
    (out / "report.md").write_text(render_report(analysis, setup), encoding="utf-8")
    for b in analysis["bots"]:
        print(f"dial {b['dial']:>5}: measured {b['measured']:6.0f}  ({b['interval'][0]:.0f} to {b['interval'][1]:.0f})")
    print(f"dial vs. measured: rank correlation {analysis['dial']['spearman']:.2f}, "
          f"{analysis['dial']['slope'] * 100:.0f} Elo per 100 dial points")
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
