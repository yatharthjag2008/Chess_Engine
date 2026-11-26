"""The public website (GitHub Pages): the model running in the browser gives PyTorch's numbers (step 20+)."""

from __future__ import annotations

import functools
import http.server
import importlib.util
import json
import subprocess
import threading
from pathlib import Path

import pytest

from test_web_js import _Quiet, browser_or_skip, check_results, dump_dom

ROOT = Path(__file__).resolve().parents[1]
UNIT_TESTS = 43     # tests in site/tests/unit_tests.js
PLAY_TESTS = 58    # tests in site/tests/play_tests.js
ANALYSE_TESTS = 11  # tests in site/tests/analyse_tests.js
STOCKFISH_TESTS = 7  # tests in site/tests/stockfish_tests.js
REVIEW_TESTS = 7  # tests in site/tests/review_tests.js


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
            "pieces/LICENSE.txt", "data/openings.json", "js/openings.js",
            "analyse.html", "js/analyse.js", "js/rating.js", "js/games.js", "data/rating.json",
            "js/stockfish.js", "js/pool.js", "js/dial.js", "data/dial.json", "js/rating_worker.js", "vendor/stockfish/stockfish-18-lite-single.js",
            "vendor/stockfish/stockfish-18-lite-single.wasm", "vendor/stockfish/COPYING.txt",
            "review.html", "js/review.js", "js/review_page.js", "css/review.css"} <= set(files)
    assert not any(f.startswith("tests/") for f in files), "tests are not published"
    with_tests = build.assemble(tmp_path / "out", tests=True)
    assert "tests/engine.html" in with_tests and "tests/harness.js" in with_tests


def test_assemble_needs_the_opening_names(tmp_path, monkeypatch) -> None:
    build = _build_site()
    fake = tmp_path / "site"
    (fake / "model").mkdir(parents=True)
    (fake / "model" / "model.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(build, "SITE", fake)
    with pytest.raises(FileNotFoundError, match="build_openings.py"):
        build.assemble(tmp_path / "out")


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


def test_an_exported_style_matches_pytorch(tmp_path) -> None:
    """A style adapter exported for the site gives the PyTorch styled predictor's numbers in the browser."""
    from bot_helpers import small_predictor
    from chess_engine.web_export import export_model, export_style, write_style_index
    from test_style import _biased_adapter, _info

    out = tmp_path / "site"
    _build_site().assemble(out, tests=True)
    predictor = small_predictor()
    export_model(predictor.model, predictor.ratings, out / "model", name="small")
    data = export_style(predictor, _biased_adapter(predictor.model, "a2a3"), _info("ann", games=200), 1.0,
                        out / "styles")
    write_style_index(out / "styles", [data])
    assert len(data["policy_bias"]) == 1858 and len(data["tests"]) == 6
    server, url = _serve(out)
    try:
        dom = dump_dom(browser_or_skip(), f"{url}/tests/style.html", tmp_path / "profile", budget_ms=20000)
    finally:
        server.shutdown()
    check_results(dom, 4)


def _run_page(tmp_path, page: str, minimum: int, budget_ms: int = 30000, attempts: int = 2) -> None:
    """Run a test page in headless Chrome; ``attempts`` > 1 retries a page that did not finish (never a failure)."""
    out = tmp_path / "site"
    _build_site().assemble(out, tests=True)
    server, url = _serve(out)
    try:
        for attempt in range(attempts):
            dom = dump_dom(browser_or_skip(), f"{url}/tests/{page}", tmp_path / f"profile{attempt}", budget_ms=budget_ms)
            if "RESULT passed=" in dom:
                break
    finally:
        server.shutdown()
    check_results(dom, minimum)


def test_the_site_modules(tmp_path) -> None:
    """Sounds, material, clocks and the other modules of the play page, one by one."""
    _run_page_live(tmp_path, "unit.html", UNIT_TESTS)


def test_the_play_page_end_to_end(tmp_path) -> None:
    _run_page(tmp_path, "play.html", PLAY_TESTS)


def test_the_analyse_page_end_to_end(tmp_path) -> None:
    """Rating estimates in the browser, checked against Python's for a real game."""
    _run_page_live(tmp_path, "analyse.html", ANALYSE_TESTS, seconds=240)


class _Results(_Quiet):
    """Serves the site, and takes the results a test page posts to /__results."""

    posted: list[str]
    done: threading.Event

    def do_POST(self) -> None:  # noqa: N802 (http.server's naming)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8", "replace")
        if self.path == "/__results":
            type(self).posted.append(body)
            type(self).done.set()
        self.send_response(204)
        self.end_headers()


def _run_page_live(tmp_path, page: str, minimum: int, seconds: float = 120) -> None:
    """Run a test page in headless Chrome in real time, until it posts its results (for pages with workers)."""
    out = tmp_path / "site"
    _build_site().assemble(out, tests=True)
    handler_class = type("Handler", (_Results,), {"posted": [], "done": threading.Event()})
    handler = functools.partial(handler_class, directory=str(out))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/tests/{page}"
    chrome = subprocess.Popen([browser_or_skip(), "--headless=new", "--disable-gpu", "--no-first-run",
                               "--no-default-browser-check", f"--user-data-dir={tmp_path / 'profile'}", url],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        handler_class.done.wait(seconds)
    finally:
        chrome.kill()
        chrome.wait(timeout=30)
        server.shutdown()
    check_results(handler_class.posted[0] if handler_class.posted else "(nothing posted)", minimum)


def test_stockfish_runs_in_the_browser(tmp_path) -> None:
    """The vendored Stockfish WebAssembly build in a worker: mates, lines, evaluations for the side to move."""
    _run_page_live(tmp_path, "stockfish.html", STOCKFISH_TESTS)


def test_the_review_page_end_to_end(tmp_path) -> None:
    """Game review in the browser: a scripted analyzer for the details, and the real Stockfish for a blunder."""
    _run_page_live(tmp_path, "review.html", REVIEW_TESTS)


def test_a_game_survives_a_reload(tmp_path) -> None:
    """A real reload in the browser: the saved game is restored."""
    # Two page loads, each loading the model: under a busy machine the first try sometimes runs out of time.
    _run_page(tmp_path, "reload.html", 1, budget_ms=120000, attempts=3)
