"""Run the web UI's JavaScript tests (web/tests/) in headless Chrome or Edge.

* ``tests/index.html``: unit tests of the board, served by a plain local HTTP server
  (browsers refuse ES modules from ``file://``).
* ``tests/app.html``: the play page end to end, against the real FastAPI app (small
  in-memory model) running on a local port.

The browser runs the tests and its DOM is dumped back. Skipped when no Chrome /
Edge is installed.
"""

from __future__ import annotations

import functools
import http.server
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from chess_engine.api.app import create_app
from chess_engine.config import PROJECT_ROOT, Config
from bot_helpers import small_predictor

WEB = PROJECT_ROOT / "web"
CANDIDATES = (
    os.environ.get("CHROME_PATH", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_browser() -> str | None:
    for path in CANDIDATES:
        if path and Path(path).is_file():
            return path
    for name in ("google-chrome", "chromium", "chrome", "msedge"):
        if shutil.which(name):
            return shutil.which(name)
    return None


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


@pytest.fixture(scope="module")
def web_server():
    handler = functools.partial(_Quiet, directory=str(WEB))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def dump_dom(browser: str, url: str, profile: Path, budget_ms: int = 5000) -> str:
    result = subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
         f"--user-data-dir={profile}", f"--virtual-time-budget={budget_ms}", "--dump-dom", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
    )
    return result.stdout


@pytest.fixture(scope="module")
def api_server(default_config: Config, tmp_path_factory: pytest.TempPathFactory):
    """The real app (API + web UI) with a small model and a scripted analyser (no Stockfish needed),
    writing into a temporary data directory, served by uvicorn on a free port."""
    from review_helpers import ScriptedAnalyzer

    data = tmp_path_factory.mktemp("web_data")
    cfg = default_config.model_copy(update={"paths": default_config.paths.model_copy(
        update={"data_dir": data, "processed_dir": data / "processed", "reports_dir": data / "reports"})})
    # One scripted blunder for the review page tests: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6?? (3...g6 holds).
    from review_helpers import position_key

    script = {position_key("e4", "e5", "Qh5", "Nc6", "Bc4"): ("g7g6", 0, {"g8f6": -900}),
              position_key("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"): ("h5f7", 900, {})}
    app = create_app(cfg, small_predictor(), review_analyzer=ScriptedAnalyzer(script))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def browser_or_skip() -> str:
    browser = find_browser()
    if browser is None:
        pytest.skip("no Chrome / Edge found for the JavaScript tests (set CHROME_PATH)")
    return browser


def check_results(dom: str, minimum: int) -> None:
    match = re.search(r"RESULT passed=(\d+) failed=(\d+)", dom)
    assert match, f"the test page did not finish:\n{dom[-2000:]}"
    failures = [line for line in dom.splitlines() if line.startswith("FAIL")]
    assert int(match.group(2)) == 0, "\n".join(failures)
    assert int(match.group(1)) >= minimum


def test_web_ui_javascript_unit_tests(web_server: str, tmp_path: Path) -> None:
    check_results(dump_dom(browser_or_skip(), f"{web_server}/tests/index.html", tmp_path / "profile"), 12)


def test_play_page_end_to_end(api_server: str, tmp_path: Path) -> None:
    dom = dump_dom(browser_or_skip(), f"{api_server}/tests/app.html", tmp_path / "profile", budget_ms=20000)
    check_results(dom, 18)


def test_review_page_end_to_end(api_server: str, tmp_path: Path) -> None:
    dom = dump_dom(browser_or_skip(), f"{api_server}/tests/review.html", tmp_path / "profile", budget_ms=20000)
    check_results(dom, 11)
