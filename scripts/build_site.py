"""Build the public website (GitHub Pages): play the human-like bot in the browser, no server needed.

Usage:
    python scripts/build_site.py --export      # export the trained model to site/model/ (needs PyTorch and the model)
    python scripts/build_site.py --styles      # export the trained styles and ghosts to site/styles/ (the same)
    python scripts/build_site.py               # assemble the site into _site/ (standard library only)
    python -m http.server -d _site 8001        # look at it: http://localhost:8001

The site is site/ (its page, scripts, the exported model and the opening names from scripts/build_openings.py)
plus the board and styles shared with the local web app (web/js/board.js, web/css/style.css, web/pieces/). The exported model is committed, so the GitHub Actions workflow
(.github/workflows/pages.yml) only runs the second command and publishes _site/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SHARED = {"js/board.js": ROOT / "web" / "js" / "board.js", "css/style.css": ROOT / "web" / "css" / "style.css",
          "css/review.css": ROOT / "web" / "css" / "review.css"}


def assemble(out: Path, tests: bool = False) -> list[str]:
    """Copy the site into ``out`` (replacing it); with ``tests`` also the browser tests. Returns the files."""
    if not (SITE / "model" / "model.json").is_file():
        raise FileNotFoundError("site/model/model.json is missing: run with --export first")
    if not (SITE / "data" / "openings.json").is_file():
        raise FileNotFoundError("site/data/openings.json is missing: run python scripts/build_openings.py")
    if out.exists():
        shutil.rmtree(out)
    ignore = None if tests else shutil.ignore_patterns("tests")
    shutil.copytree(SITE, out, ignore=ignore)
    for target, source in SHARED.items():
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, out / target)
    shutil.copytree(ROOT / "web" / "pieces", out / "pieces")     # the piece images style.css refers to
    if tests:
        shutil.copyfile(ROOT / "web" / "tests" / "harness.js", out / "tests" / "harness.js")
    (out / ".nojekyll").write_text("", encoding="utf-8")      # serve every file as it is
    return sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())


def export(config: str | None, model_path: str | None, write: Callable[[str], None]) -> None:
    from chess_engine.config import load_config
    from chess_engine.model.inference import load_model
    from chess_engine.web_export import export_model

    cfg = load_config(config)
    path = Path(model_path or cfg.bot.model)
    loaded = load_model(path)
    manifest = export_model(loaded.model, loaded.ratings, SITE / "model", name=path.parent.name or path.stem)
    size = (SITE / "model" / "model.bin").stat().st_size
    write(f"Exported {path.name} ({manifest['floats']:,} numbers, {size / 1e6:.1f} MB) to site/model/")


def export_styles(config: str | None, model_path: str | None, write: Callable[[str], None],
                  names: list[str] | None = None, out_dir: Path | None = None) -> list[str]:
    """Export trained styles (all, or ``names``) from the style folder to site/styles/ with an index."""
    from chess_engine.config import load_config
    from chess_engine.model.inference import load_model
    from chess_engine.style.store import StyleLibrary, blend_weight
    from chess_engine.web_export import export_style, write_style_index

    cfg = load_config(config)
    base = load_model(Path(model_path or cfg.bot.model)).predictor()
    library = StyleLibrary(cfg.paths.data_dir / cfg.style.directory)
    out = out_dir or SITE / "styles"
    chosen = names or [info.name for info in library.list()]
    exported = []
    for name in chosen:
        adapter, info = library.load(name, base.model)
        exported.append(export_style(base, adapter, info, blend_weight(info.games, cfg.style), out))
        write(f"Exported style {name} ({info.player}, {info.kind}, {info.games} games)")
    write_style_index(out, exported)
    return [s["name"] for s in exported]


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", action="store_true", help="export the trained model to site/model/ first")
    parser.add_argument("--out", default=str(ROOT / "_site"), help="where to assemble the site (default: _site/)")
    parser.add_argument("--tests", action="store_true", help="include the browser tests (site/tests/)")
    parser.add_argument("--styles", nargs="*", default=None, metavar="NAME",
                        help="export trained styles to site/styles/ first (all, or the ones named)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        if args.export:
            export(args.config, args.model, write)
        if args.styles is not None:
            export_styles(args.config, args.model, write, args.styles or None)
        files = assemble(Path(args.out), tests=args.tests)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write(f"Site assembled in {args.out}: {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
