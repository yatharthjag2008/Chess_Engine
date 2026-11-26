"""Time the website's network and rating estimate in headless Chrome (real time) and print the numbers.

Usage:
    python scripts/bench_site.py            # builds the site with its test pages into a temporary folder

Prints the milliseconds per network pass and the seconds to score one side of a real 60-move game at every
rating (site/tests/bench.html). Needs Chrome or Chromium.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    import build_site
    from test_web_js import _Quiet, find_browser

    browser = find_browser()
    if browser is None:
        print("error: Chrome or Chromium is needed", file=sys.stderr)
        return 1
    posted: list[str] = []
    done = threading.Event()

    class Handler(_Quiet):
        def do_POST(self) -> None:  # noqa: N802
            posted.append(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8", "replace"))
            done.set()
            self.send_response(204)
            self.end_headers()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "site"
        build_site.assemble(out, tests=True)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(out)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/tests/bench.html"
        chrome = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--no-first-run",
                                   f"--user-data-dir={Path(tmp) / 'profile'}", url],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            done.wait(300)
        finally:
            chrome.kill()
            chrome.wait(timeout=30)
            server.shutdown()
    if not posted:
        print("error: the page did not finish", file=sys.stderr)
        return 1
    print("\n".join(line for line in posted[0].splitlines() if not line.startswith("RESULT")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
