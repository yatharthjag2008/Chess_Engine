"""A local HTTP server with scripted routes, standing in for the Lichess and Chess.com APIs in tests."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


@dataclass
class Reply:
    status: int = 200
    body: bytes | str = b""
    headers: dict[str, str] = field(default_factory=dict)


Route = Callable[[dict[str, list[str]], dict[str, str]], Reply] | Reply | list[Reply]


class FakeApi:
    """Routes by path (without the query string). A route is a :class:`Reply`, a list of replies served
    in order (the last one repeats), or a function ``(query, headers) -> Reply``.
    Requests are recorded as ``(path, query, headers)``."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.requests: list[tuple[str, dict[str, list[str]], dict[str, str]]] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                parts = urlsplit(self.path)
                query = parse_qs(parts.query)
                headers = {k.lower(): v for k, v in self.headers.items()}
                api.requests.append((parts.path, query, headers))
                route = api.routes.get(parts.path, Reply(404, b'{"error": "Not found"}'))
                if isinstance(route, list):
                    reply = route.pop(0) if len(route) > 1 else route[0]
                elif callable(route):
                    reply = route(query, headers)
                else:
                    reply = route
                body = reply.body.encode("utf-8") if isinstance(reply.body, str) else reply.body
                self.send_response(reply.status)
                for key, value in reply.headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeApi:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
