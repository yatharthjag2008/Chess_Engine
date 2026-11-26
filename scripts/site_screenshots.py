"""Screenshots of the website in headless Chrome, at desktop and phone width, with a check for sideways overflow.

Usage:
    python scripts/site_screenshots.py                  # into reports/web/site/ (desktop and phone, every page)
    python scripts/site_screenshots.py --readme         # also update reports/web/website.png (the play page's top)

Each page is set up as a visitor would leave it (a game with a hint and an arrow; an estimated game; a reviewed
game at a key moment with the coach), in real time, so Stockfish and the background workers finish. Chrome is
driven through its DevTools protocol (a minimal client on the standard library). A page that is wider than the
screen is reported, and the script exits with 1.
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

OUT = ROOT / "reports" / "web" / "site"
SIZES = {"desktop": (1280, 900, 1, False), "phone": (390, 844, 3, True)}
README_HEIGHT = 1310          # the README's picture: the board, the new-game form, the moves and the hint

GAME = (ROOT / "tests" / "fixtures" / "lichess_real.pgn").read_text(encoding="utf-8").split("\n\n[Event")[0].strip()

# What to do on each page before the screenshot (JavaScript, awaited in the page).
SCENES = {
    "play": ("index.html", """
        const site = await import("./js/site.js");
        await until(() => !document.getElementById("play").disabled);
        site.ui.delay = 0; site.ui.thinkScale = 0.001; site.ui.random = () => 0.37;
        document.querySelector('input[name="time"][value="5"]').checked = true;
        await site.newGame({ rating: 1500, color: "white", time: "5", hints: 3 });
        for (const m of ["e2e4", "g1f3", "f1c4"]) await site.play(m);
        await site.askHint(); await site.askHint();
        site.ui.board.setShapes([{ orig: "f3", dest: "g5" }, { orig: "f7", brush: "red" }]);
        site.ui.clock.remaining.w = 251000; site.ui.clock.remaining.b = 263000; site.updateClocks();
    """),
    "analyse": ("analyse.html", """
        const page = await import("./js/analyse.js");
        await until(() => !document.getElementById("estimate").disabled);
        document.getElementById("pgn-text").value = GAME;
        await page.estimate();
    """),
    "review": ("review.html", """
        const page = await import("./js/review_page.js");
        await page.startReview(GAME, 10);
        await page.startCoach();
        document.querySelector("#summary-panel .moments button")?.click();
    """),
}


# --------------------------------------------------------------------------- a minimal DevTools client


class DevTools:
    """One page's DevTools WebSocket: send a command, wait for its answer (events are skipped)."""

    def __init__(self, ws_url: str) -> None:
        rest = ws_url.split("://", 1)[1]
        hostport, path = rest.split("/", 1)
        host, port = hostport.split(":")
        self.sock = socket.create_connection((host, int(port)), timeout=300)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"no WebSocket: {response[:200]!r}")
        self.buffer = response.split(b"\r\n\r\n", 1)[1]
        self.next_id = 0

    def _read(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError("DevTools closed the connection")
            self.buffer += chunk
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        return data

    def _message(self) -> str:
        parts = []
        while True:
            head = self._read(2)
            fin, opcode, length = head[0] & 0x80, head[0] & 0x0F, head[1] & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(length)
            if opcode in (1, 0):
                parts.append(payload)
                if fin:
                    return b"".join(parts).decode("utf-8")

    def send(self, method: str, **params: object) -> dict:
        self.next_id += 1
        data = json.dumps({"id": self.next_id, "method": method, "params": params}).encode()
        mask = os.urandom(4)
        header = bytes([0x81])
        n = len(data)
        header += bytes([0x80 | n]) if n < 126 else bytes([0xFE]) + struct.pack(">H", n) if n < 65536 \
            else bytes([0xFF]) + struct.pack(">Q", n)
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        while True:
            message = json.loads(self._message())
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression: str) -> object:
        result = self.send("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in result:
            raise RuntimeError(f"page error: {result['exceptionDetails'].get('exception', {}).get('description')}")
        return result.get("result", {}).get("value")


def _chrome(browser: str, profile: Path) -> subprocess.Popen:
    return subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                             "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _page_socket(profile: Path) -> str:
    port_file = profile / "DevToolsActivePort"
    for _ in range(300):
        if port_file.is_file() and port_file.read_text().strip():
            break
        time.sleep(0.1)
    port = port_file.read_text().split()[0]
    for _ in range(100):
        pages = [t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
                 if t.get("type") == "page"]
        if pages:
            return pages[0]["webSocketDebuggerUrl"]
        time.sleep(0.1)
    raise RuntimeError("Chrome opened no page")


# --------------------------------------------------------------------------- the screenshots


def shoot(url_base: str, browser: str, tmp: Path, out: Path) -> list[str]:
    """Every scene at every size; returns the problems found (pages wider than the screen)."""
    problems = []
    for size_name, (width, height, scale, mobile) in SIZES.items():
        profile = tmp / f"profile-{size_name}"
        chrome = _chrome(browser, profile)
        try:
            tools = DevTools(_page_socket(profile))
            tools.send("Page.enable")
            tools.send("Runtime.enable")
            tools.send("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale,
                       mobile=mobile)
            for scene, (page, script) in SCENES.items():
                target = f"{url_base}/{page}"
                tools.send("Page.navigate", url=target)
                for _ in range(600):                 # navigate returns before the new page exists: wait for it
                    try:
                        if tools.evaluate(f"location.href === {json.dumps(target)} && document.readyState === 'complete'"):
                            break
                    except RuntimeError:             # the old page went away mid-question
                        pass
                    time.sleep(0.05)
                helpers = ("const until = (f, ms = 120000) => new Promise((ok, no) => { const t0 = Date.now(); "
                           "const step = () => f() ? ok() : Date.now() - t0 > ms ? no(new Error('timed out; status: ' + "
                           "(document.querySelector('.status') || {}).textContent)) : "
                           f"setTimeout(step, 50); step(); }}); const GAME = {json.dumps(GAME)};")
                tools.evaluate(f"(async () => {{ localStorage.clear(); {helpers} {script} }})()")
                tools.evaluate("new Promise((r) => setTimeout(r, 400))")      # let the board's animation settle
                overflow = tools.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                if overflow and overflow > 0:
                    problems.append(f"{scene} at {size_name} width: {overflow}px wider than the screen")
                full = tools.evaluate("Math.ceil(document.documentElement.scrollHeight)")
                shot = tools.send("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                                  clip={"x": 0, "y": 0, "width": width, "height": full, "scale": 1})
                path = out / f"{scene}-{size_name}.png"
                path.write_bytes(base64.b64decode(shot["data"]))
                if scene == "play" and size_name == "desktop":          # the README's: the board and the panels
                    top = tools.send("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                                     clip={"x": 0, "y": 0, "width": width, "height": min(full, README_HEIGHT), "scale": 1})
                    (out / "play-desktop-top.png").write_bytes(base64.b64decode(top["data"]))
                print(f"{path}  ({width}x{full})")
        finally:
            chrome.kill()
            chrome.wait(timeout=30)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--readme", action="store_true", help="also update reports/web/website.png (desktop play)")
    args = parser.parse_args(argv)
    import build_site
    from test_web_js import _Quiet, find_browser

    browser = find_browser()
    if browser is None:
        print("error: Chrome or Chromium is needed", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp) / "site"
        build_site.assemble(site)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(site)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            problems = shoot(f"http://127.0.0.1:{server.server_address[1]}", browser, Path(tmp), out)
        finally:
            server.shutdown()
    if args.readme:
        shutil.copyfile(out / "play-desktop-top.png", ROOT / "reports" / "web" / "website.png")
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
