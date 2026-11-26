"""Write the website's honest rating dial (site/data/dial.json) from the Elo-dial validation (step 11).

Usage:
    python scripts/build_dial.py        # reports/elo/laptop_64x6_best_model/results.json -> site/data/dial.json

The dial's labels become measured strength: steps on it are measured Elo steps, the input 1500 keeps the label
1500 (an assumption: the measurement is on Stockfish's scale), and only the strengths the bot can play are offered.
See chess_engine.eval.dial. Re-run after a new model and a new scripts/elo_validation.py run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_engine.eval.dial import load_calibration  # noqa: E402

RESULTS = ROOT / "reports" / "elo" / "laptop_64x6_best_model" / "results.json"   # the model the site runs
TARGET = ROOT / "site" / "data" / "dial.json"


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(RESULTS), help="an Elo-dial validation's results.json")
    parser.add_argument("--out", default=str(TARGET), help="where to write (default: site/data/dial.json)")
    parser.add_argument("--anchor", type=int, default=1500, help="the input that keeps its label (default 1500)")
    args = parser.parse_args(argv)
    calibration = load_calibration(args.results, args.anchor)
    data = {"source": Path(args.results).parent.name, **calibration.to_dict()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    write(f"dial {data['min']}-{data['max']} (anchor {data['anchor']}): "
          + ", ".join(f"{p['input']}->{p['label']:.0f}" for p in data["points"]) + f" -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
