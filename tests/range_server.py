"""A tiny local HTTP server for tests: serves in-memory files with Range support and injected faults."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class ServedFile:
    """One file the server can serve, plus faults to inject.

    Attributes:
        data: The file's bytes.
        drop_after: For the first ``len(drop_after)`` requests, close the connection after
            sending this many bytes of the body (simulates a dropped connection).
        fail_first: Respond with HTTP 503 to this many requests before serving normally.
        support_ranges: If False, ignore ``Range`` headers (always send the whole file).
    """

    data: bytes
    drop_after: list[int] = field(default_factory=list)
    fail_first: int = 0
    support_ranges: bool = True


class RangeServer:
    """Serve :class:`ServedFile` objects at ``http://127.0.0.1:<port>/<name>``."""

    def __init__(self) -> None:
        self.files: dict[str, ServedFile] = {}
        self.requests: list[tuple[str, str | None]] = []  # (path, Range header)
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:  # keep test output quiet
                pass

            def _serve(self, send_body: bool) -> None:
                name = self.path.lstrip("/")
                range_header = self.headers.get("Range")
                server.requests.append((name, range_header))
                served = server.files.get(name)
                if served is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if served.fail_first > 0:
                    served.fail_first -= 1
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                data, start = served.data, 0
                if range_header and served.support_ranges:
                    start = int(range_header.removeprefix("bytes=").split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if not send_body:
                    return
                if served.drop_after:
                    limit = served.drop_after.pop(0)
                    self.wfile.write(body[:limit])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 (http.server naming)
                self._serve(send_body=True)

            def do_HEAD(self) -> None:  # noqa: N802
                self._serve(send_body=False)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def url(self, name: str) -> str:
        """URL of a served file."""
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/{name}"

    def start(self) -> RangeServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
