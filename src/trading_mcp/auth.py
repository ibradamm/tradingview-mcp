"""Shared-secret authentication for the /mcp endpoint.

Claude.ai custom connectors support either OAuth or no auth at all; they cannot send a custom
header. So the token is accepted in two places:

* ``Authorization: Bearer <token>`` - for Claude Code, scripts and other clients;
* ``?key=<token>`` query parameter - for Claude.ai (the key is part of the connector URL).

Treat the full connector URL as a password. OAuth 2.1 is the upgrade path (see ARCHITECTURE.md).
"""

from __future__ import annotations

import hmac
import json
import logging
from urllib.parse import parse_qs

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class TokenAuthMiddleware:
    """Rejects requests to protected path prefixes without the correct token."""

    def __init__(self, app: ASGIApp, token: str, protected_prefix: str = "/mcp") -> None:
        self.app = app
        self.token = token.encode()
        self.prefix = protected_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(self.prefix) or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        logger.warning("rejected unauthenticated request to %s", scope["path"])
        body = json.dumps({"error": "unauthorized", "detail": "Missing or invalid token."}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b'Bearer realm="mcp"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _authorized(self, scope: Scope) -> bool:
        candidates: list[bytes] = []
        for name, value in scope.get("headers", []):
            if name == b"authorization" and value[:7].lower() == b"bearer ":
                candidates.append(value[7:].strip())
        query = parse_qs(scope.get("query_string", b"").decode(), keep_blank_values=False)
        candidates.extend(v.encode() for v in query.get("key", []))
        return any(hmac.compare_digest(c, self.token) for c in candidates)
