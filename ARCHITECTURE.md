# Architecture

## Overview

```
┌──────────────┐   HTTPS (TLS terminated by host: HF Spaces / Render / Cloud Run / tunnel)
│  Claude.ai   │───────────────────────────────────────────────────────────────┐
│  connector   │   POST /mcp?key=<token>   (Streamable HTTP, JSON-RPC)          │
└──────────────┘                                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ ASGI app (Starlette, served by uvicorn)                                   server.py │
│                                                                                      │
│  request → AccessLog (no query strings) → SecurityHeaders → CORS → RateLimit(/mcp)   │
│          → TokenAuth(/mcp, SHA-256 compare) → route                                  │
│                                                                                      │
│   GET /health  (public)            /mcp  → MCPServer (mcp SDK 2.x, stateless, JSON) │
└───────────────────────────────────────────────┬──────────────────────────────────────┘
                                                │ tools/call
┌───────────────────────────────────────────────▼──────────────────────────────────────┐
│ tools/*  thin MCP wrappers: validation (Pydantic), @safe_tool error mapping + audit  │
│   market · technical · analysis(MTF, screener) · backtest · ml · risk/portfolio      │
└──────┬───────────────┬──────────────┬────────────────┬──────────────┬────────────────┘
       │               │              │                │              │
┌──────▼─────┐ ┌───────▼──────┐ ┌─────▼──────┐ ┌───────▼──────┐ ┌─────▼──────┐
│ indicators │ │ analysis/    │ │ backtest/  │ │ ml/          │ │ risk/      │
│ technical  │ │ snapshot     │ │ strategies │ │ features     │ │ sizing     │
│ (pandas)   │ │ screener/    │ │ engine     │ │ models (reg.)│ │ VaR/CVaR   │
│            │ │ filters      │ │ metrics    │ │ training/WF  │ │ portfolio  │
│            │ │              │ │ walk_fwd   │ │ store        │ │            │
└──────┬─────┘ └───────┬──────┘ └─────┬──────┘ └───────┬──────┘ └─────┬──────┘
       └───────────────┴──────┬───────┴────────────────┴──────────────┘
                              │ get_history / get_quote / search_symbol
┌─────────────────────────────▼────────────────────────────────────────────────────────┐
│ data/factory.CachedProvider   in-memory TTL cache → provider → persist bars to DB    │
│                               (DB copy served as flagged fallback if provider fails) │
│   MarketDataProvider (ABC)  ← YahooProvider (yfinance)  |  SyntheticProvider (tests) │
└─────────────────────────────┬───────────────────────────────────────────┬────────────┘
                              │                                           │
                 ┌────────────▼────────────┐                  ┌───────────▼──────────┐
                 │ db/ SQLAlchemy 2        │                  │ ml/store  joblib +   │
                 │ SQLite (WAL) | Postgres │                  │ JSON in MODEL_DIR    │
                 │ price_bars, backtest_   │                  └──────────────────────┘
                 │ runs, ml_experiments,   │
                 │ tool_calls              │
                 └─────────────────────────┘
```

## Design decisions

| Decision | Reason |
|---|---|
| MCP SDK 2.x `MCPServer`, **stateless** Streamable HTTP with JSON responses | No server-side sessions: survives restarts and free-tier sleep, scales horizontally, simplest for remote clients. |
| Token via `?key=` **or** `Authorization: Bearer` | Claude.ai custom connectors only support OAuth or no auth; they cannot send custom headers. The query key is the simplest secure-enough option. Only SHA-256 digests are kept in memory; comparison is constant-time. |
| Access log without query string, uvicorn access log disabled | Otherwise the token in `?key=` would be written to the host's logs. Tested. |
| Server refuses to start without a token | Fail closed: a misconfigured deployment cannot expose the tools publicly. |
| `MarketDataProvider` ABC + decorator cache | Swap Yahoo for a licensed feed by writing one class; caching and persistence are provider-agnostic. |
| Indicators implemented in pandas | Small dependency surface, exact control over definitions (Wilder smoothing = TradingView `ta.rma`), tested against hand-computed values. |
| Signals at close, fills at next open | Structural protection against look-ahead bias. A signal on the last bar can never trade (tested). |
| Walk-forward TRAIN → VALIDATION → TEST, stitched OOS equity | Only test segments are out-of-sample; parameter search is included in the estimate. Train-vs-test degradation drives the overfitting flags. |
| ML: causal features, forward labels, purged chronological split, `TimeSeriesSplit(gap=h)`, scaler inside the pipeline | Prevents the classic leakage paths (shuffled splits, overlapping labels, scaler fitted on test data). |
| Model registry (`register_model`) | XGBoost/LightGBM can be added with one line, without touching tools or training code. |
| Joblib models loaded only from `MODEL_DIR` with validated ids | Pickle is code execution; never load a caller-supplied path. |
| SQLAlchemy generic types, `DATABASE_URL` | Same code on SQLite (default) and PostgreSQL (tested in CI with a Postgres service). |

## Walk-forward (rule-based strategies)

```
bars: ├──────── TRAIN ────────┤─ VALIDATION ─┤── TEST ──┤
      grid search (Sharpe)     pick among      measure    ← fold 1
                               top-3 of train  (OOS)
            ├──────── TRAIN ────────┤─ VALIDATION ─┤── TEST ──┤    ← fold 2 (rolled by test length)
                  ...
OOS equity = concatenation of all TEST segments
```

## ML walk-forward (`evaluate_model(method="walk_forward")`)

```
├──────── TRAIN (retrain) ────────┤gap h├── TEST (predict, trade next open) ──┤  → roll by test length
```

## Security model

- Threats covered: unauthenticated use of the public endpoint, token leakage through logs, brute force
  (rate limiting), DNS rebinding (optional `ALLOWED_HOSTS`), pickle loading of arbitrary files, secrets in git.
- Not covered: token leakage on the client side (the URL is a password), per-user authorisation (single
  shared token), distributed rate limiting.
- **Upgrade path: OAuth 2.1.** The MCP SDK provides `TokenVerifier` / `AuthSettings`; Claude.ai supports OAuth
  for custom connectors. Put an OAuth provider (e.g. Auth0, WorkOS, Cloudflare Access, Keycloak) in front, verify
  JWTs in `TokenVerifier`, and drop the query key.

## Extending

- **New data source**: subclass `MarketDataProvider`, add it to `data/factory.build_provider`, set `MARKET_DATA_PROVIDER`.
- **New indicator**: pure function in `indicators/technical.py` + wrapper in `tools/technical.py` + test.
- **New strategy**: signal function returning -1/0/1 in `backtest/strategies.py` and a `StrategySpec` entry (with a grid for walk-forward).
- **New model**: `register_model("xgboost", "classification", lambda: XGBClassifier(...))`.
- **New tool module**: `tools/<name>.py` with `register(mcp)`, then add it to `tools/__init__.register_all`.
