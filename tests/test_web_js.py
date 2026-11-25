"""Run the web UI's JavaScript unit tests (web/tests/) in headless Chrome or Edge.

The page is served over a local HTTP server (browsers refuse ES modules from
``file://``); the browser runs the tests and the DOM is dumped back. Skipped
when no Chrome / Edge is installed.
"""

from __future__ import annotations

import functools
import http.server
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from chess_engine.config import PROJECT_ROOT

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


def test_web_ui_javascript_unit_tests(web_server: str, tmp_path: Path) -> None:
    browser = find_browser()
    if browser is None:
        pytest.skip("no Chrome / Edge found for the JavaScript tests (set CHROME_PATH)")
    dom = dump_dom(browser, f"{web_server}/tests/index.html", tmp_path / "profile")
    match = re.search(r"RESULT passed=(\d+) failed=(\d+)", dom)
    assert match, f"the test page did not finish:\n{dom[-2000:]}"
    failures = [line for line in dom.splitlines() if line.startswith("FAIL")]
    assert int(match.group(2)) == 0, "\n".join(failures)
    assert int(match.group(1)) >= 5
