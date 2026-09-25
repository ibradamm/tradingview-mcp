"""Shared-secret authentication for the /mcp endpoint.

Claude.ai custom connectors support either OAuth or no auth at all; they cannot send a custom
header. So the token is accepted in two places:

* ``Authorization: Bearer <token>`` - for Claude Code, scripts and other clients;
* ``?key=<token>`` query parameter - for Claude.ai (the key is part of the connector URL).

Treat the full connector URL as a password. OAuth 2.1 is the upgrade path (see ARCHITECTURE.md).

The middleware only keeps SHA-256 digests of accepted tokens, so a deployment can be configured
with ``MCP_AUTH_TOKEN_SHA256`` and never hold the plain token at all.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from urllib.parse import parse_qs

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class TokenAuthMiddleware:
    """Rejects requests to protected path prefixes without the correct token."""

    def __init__(self, app: ASGIApp, token_hashes: list[str], protected_prefix: str = "/mcp") -> None:
        if not token_hashes:
            raise ValueError("at least one token hash is required")
        self.app = app
        self.hashes = [bytes.fromhex(h) for h in token_hashes]
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
        digests = [hashlib.sha256(c).digest() for c in candidates]
        return any(hmac.compare_digest(d, h) for d in digests for h in self.hashes)


def sha256_hex(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
