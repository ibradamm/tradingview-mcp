"""Entry point: ``python -m trading_mcp.main`` or the ``trading-mcp`` console script."""

from __future__ import annotations

import logging
import os

import uvicorn

from trading_mcp.config import get_settings


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Most PaaS inject PORT; it wins over the .env value.
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run(
        "trading_mcp.server:create_app",
        factory=True,
        host=settings.host,
        port=port,
        log_level=settings.log_level.lower(),
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=False,  # uvicorn's access log would record ?key=<token>; AccessLogMiddleware redacts it
        server_header=False,
    )


if __name__ == "__main__":
    main()
