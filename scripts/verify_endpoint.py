"""End-to-end check of a deployed MCP endpoint, exactly as Claude.ai will use it.

Usage:
    MCP_VERIFY_TOKEN=... python scripts/verify_endpoint.py https://host [--json out.json]

The token is read from the environment and is never printed.
Checks: HTTPS /health, 401 without token, MCP initialize + tools/list, and real tool calls.
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
from mcp import Client

EXPECTED_TOOLS = {
    "get_quote", "get_historical_data", "search_symbol", "calculate_rsi", "calculate_macd", "calculate_vwap",
    "multi_timeframe_analysis", "market_screener", "backtest_strategy", "walk_forward_backtest", "train_model",
    "evaluate_model", "predict", "calculate_position_size", "risk_analysis", "portfolio_analysis",
}


async def run(base: str, token: str) -> dict[str, Any]:
    results: dict[str, Any] = {"base_url": base, "checks": {}}
    checks = results["checks"]
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(f"{base}/health")
        checks["health"] = {"status": r.status_code, "body": r.json() if r.status_code == 200 else None}
        r = await http.post(f"{base}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                            headers={"Accept": "application/json, text/event-stream"})
        checks["unauthenticated_rejected"] = {"status": r.status_code, "ok": r.status_code == 401}

    started = time.perf_counter()
    async with Client(f"{base}/mcp?key={token}", read_timeout_seconds=120) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        checks["tools_list"] = {"count": len(tools), "missing": sorted(EXPECTED_TOOLS - tools)}
        calls = {
            "get_quote": {"ticker": "NVDA"},
            "get_historical_data": {"ticker": "BTCUSD", "timeframe": "1h", "limit": 3},
            "calculate_rsi": {"ticker": "SPY", "tail": 2},
            "multi_timeframe_analysis": {"ticker": "NVDA", "timeframes": ["15m", "1h", "1d"]},
            "backtest_strategy": {"ticker": "SPY", "strategy": "sma_crossover", "start": "2020-01-01"},
        }
        for name, args in calls.items():
            t0 = time.perf_counter()
            res = await client.call_tool(name, args)
            summary: dict[str, Any] = {"ok": not res.is_error, "seconds": round(time.perf_counter() - t0, 2)}
            data = res.structured_content or {}
            if res.is_error:
                summary["error"] = res.content[0].text[:300] if res.content else "unknown"
            elif name == "get_quote":
                summary.update({k: data.get(k) for k in ("ticker", "price", "change_percent", "timestamp", "source")})
            elif name == "get_historical_data":
                summary["last_bar"] = (data.get("bars") or [{}])[-1]
            elif name == "calculate_rsi":
                summary["latest"] = data.get("latest")
            elif name == "multi_timeframe_analysis":
                summary["trends"] = {tf: v.get("trend", v.get("error")) for tf, v in data["timeframes"].items()}
            elif name == "backtest_strategy":
                st = data.get("statistics", {})
                summary.update({k: st.get(k) for k in ("total_return", "sharpe_ratio", "max_drawdown",
                                                       "number_of_trades")})
            checks[f"call:{name}"] = summary
    results["mcp_seconds"] = round(time.perf_counter() - started, 2)
    results["passed"] = (
        checks["health"]["status"] == 200
        and checks["unauthenticated_rejected"]["ok"]
        and not checks["tools_list"]["missing"]
        and all(v["ok"] for k, v in checks.items() if k.startswith("call:"))
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", help="https://host (without /mcp)")
    parser.add_argument("--json", help="write results to this file")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    token = os.environ.get("MCP_VERIFY_TOKEN")
    if not token:
        sys.exit("MCP_VERIFY_TOKEN is not set")
    base = args.base_url.rstrip("/")
    last_error: Exception | None = None
    for attempt in range(args.retries):
        try:
            results = asyncio.run(run(base, token))
            break
        except Exception as exc:  # noqa: BLE001 - DNS/TLS propagation of fresh endpoints
            last_error = exc
            print(f"attempt {attempt + 1} failed: {type(exc).__name__}: {str(exc).replace(token, '***')[:200]}")
            time.sleep(10)
    else:
        sys.exit(f"endpoint unreachable: {last_error!r}".replace(token, "***"))
    text = json.dumps(results, indent=2, default=str).replace(token, "***")
    print(text)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
