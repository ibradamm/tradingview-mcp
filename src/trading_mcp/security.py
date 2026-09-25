"""HTTP hardening middlewares: redacted access log, rate limiting and security headers."""

from __future__ import annotations

import logging
import threading
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

access_logger = logging.getLogger("trading_mcp.access")

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
    (b"strict-transport-security", b"max-age=31536000"),
]


def client_ip(scope: Scope) -> str:
    """Client IP, honouring the first X-Forwarded-For hop set by the hosting proxy."""
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            return value.decode().split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


class AccessLogMiddleware:
    """Access log WITHOUT query strings, so ``?key=<token>`` never reaches the logs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started, status = time.perf_counter(), 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            access_logger.info('%s "%s %s" %s %.0fms', client_ip(scope), scope["method"], scope["path"], status,
                               (time.perf_counter() - started) * 1000)


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *SECURITY_HEADERS]}
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RateLimitMiddleware:
    """Per-IP token bucket on a path prefix. In-memory: per process, reset on restart."""

    def __init__(self, app: ASGIApp, per_minute: int, prefix: str = "/mcp") -> None:
        self.app = app
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.prefix = prefix
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def _allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1
            self._buckets[key] = (tokens - 1 if allowed else tokens, now)
            if len(self._buckets) > 10_000:  # bound memory under IP churn
                self._buckets.clear()
            return allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(self.prefix) or self._allow(client_ip(scope)):
            await self.app(scope, receive, send)
            return
        await send({"type": "http.response.start", "status": 429,
                    "headers": [(b"content-type", b"application/json"), (b"retry-after", b"10")]})
        await send({"type": "http.response.body", "body": b'{"error":"rate_limited"}'})
