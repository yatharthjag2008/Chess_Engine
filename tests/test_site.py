"""The public website (GitHub Pages): the model running in the browser gives PyTorch's numbers (step 20+)."""

from __future__ import annotations

import functools
import http.server
import importlib.util
import json
import threading
from pathlib import Path

import pytest

from test_web_js import _Quiet, browser_or_skip, check_results, dump_dom

ROOT = Path(__file__).resolve().parents[1]
UNIT_TESTS = 15     # tests in site/tests/unit_tests.js
PLAY_TESTS = 28    # tests in site/tests/play_tests.js


def _build_site():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serve(directory: Path):
    handler = functools.partial(_Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_assemble_copies_the_site_and_the_shared_files(tmp_path) -> None:
    build = _build_site()
    files = build.assemble(tmp_path / "out")
    assert {"index.html", "js/engine.js", "js/board.js", "css/style.css", "vendor/chess.js",
            "model/model.json", "model/model.bin", ".nojekyll", "pieces/wK.svg", "pieces/bP.svg",
            "pieces/LICENSE.txt"} <= set(files)
    assert not any(f.startswith("tests/") for f in files), "tests are not published"
    with_tests = build.assemble(tmp_path / "out", tests=True)
    assert "tests/engine.html" in with_tests and "tests/harness.js" in with_tests


def test_the_committed_model_matches_its_test_vectors(tmp_path) -> None:
    """The exported model in site/model/ (the trained one) is computed correctly by the browser."""
    out = tmp_path / "site"
    _build_site().assemble(out, tests=True)
    server, url = _serve(out)
    try:
        dom = dump_dom(browser_or_skip(), f"{url}/tests/engine.html", tmp_path / "profile", budget_ms=20000)
    finally:
        server.shutdown()
    check_results(dom, 7)


def test_a_fresh_export_matches_pytorch(tmp_path) -> None:
    """Export a small model with non-trivial batch-norm statistics and check the browser against PyTorch."""
    from bot_helpers import small_predictor
    from chess_engine.web_export import export_model
    from test_web_export import _trained_bn

    out = tmp_path / "site"
    _build_site().assemble(out, tests=True)
    predictor = _trained_bn(small_predictor())
    export_model(predictor.model, predictor.ratings, out / "model", name="small")
    assert json.loads((out / "model" / "model.json").read_text(encoding="utf-8"))["name"] == "small"
    server, url = _serve(out)
    try:
        dom = dump_dom(browser_or_skip(), f"{url}/tests/engine.html", tmp_path / "profile", budget_ms=20000)
    finally:
        server.shutdown()
    check_results(dom, 7)


def _run_page(tmp_path, page: str, minimum: int, budget_ms: int = 30000) -> None:
    out = tmp_path / "site"
    _build_site().assemble(out, tests=True)
    server, url = _serve(out)
    try:
        dom = dump_dom(browser_or_skip(), f"{url}/tests/{page}", tmp_path / "profile", budget_ms=budget_ms)
    finally:
        server.shutdown()
    check_results(dom, minimum)


def test_the_site_modules(tmp_path) -> None:
    """Sounds, material, clocks and the other modules of the play page, one by one."""
    _run_page(tmp_path, "unit.html", UNIT_TESTS)


def test_the_play_page_end_to_end(tmp_path) -> None:
    _run_page(tmp_path, "play.html", PLAY_TESTS)


def test_a_game_survives_a_reload(tmp_path) -> None:
    """A real reload in the browser: the saved game is restored."""
    _run_page(tmp_path, "reload.html", 1, budget_ms=120000)   # two page loads, each loading the model
