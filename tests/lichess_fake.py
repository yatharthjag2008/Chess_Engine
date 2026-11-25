"""A fake Lichess for the bot tests: canned NDJSON streams, recorded posts."""

from __future__ import annotations

import json


class FakeResponse:
    def __init__(self, status: int = 200, body: object = None, lines: list[dict] | None = None) -> None:
        self.status_code = status
        self._body = body if body is not None else {}
        self._lines = lines or []

    def json(self) -> object:
        return self._body

    def iter_lines(self):
        for item in self._lines:
            yield b""                                   # keep-alive lines are blank
            yield json.dumps(item).encode("utf-8")


class FakeLichess:
    """``get`` answers from ``routes`` (path -> FakeResponse); ``post`` records (path, data) and answers 200
    unless ``fail_posts`` has the path."""

    def __init__(self, routes: dict[str, FakeResponse] | None = None) -> None:
        self.routes = routes or {}
        self.posts: list[tuple[str, dict]] = []
        self.fail_posts: dict[str, FakeResponse] = {}
        self.headers: list[dict] = []

    def _path(self, url: str) -> str:
        return url.split("lichess.org", 1)[1] if "lichess.org" in url else url

    def get(self, url: str, **kwargs):
        self.headers.append(kwargs.get("headers", {}))
        return self.routes.get(self._path(url), FakeResponse(404, {"error": "Not found"}))

    def post(self, url: str, **kwargs):
        self.headers.append(kwargs.get("headers", {}))
        path = self._path(url)
        self.posts.append((path, kwargs.get("data", {})))
        return self.fail_posts.get(path, FakeResponse(200, {"ok": True}))
