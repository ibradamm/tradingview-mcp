# Trading MCP

A remote **MCP server** (Model Context Protocol) in Python that gives Claude market data, technical
analysis, a multi-criteria screener, backtesting with walk-forward validation, machine learning, risk
and portfolio analytics.

> **Research, education and paper trading only.** Nothing here is investment advice. Backtests and
> model outputs are simulations and probabilistic estimates, not predictions of future results.

```
Claude.ai ──HTTPS──> /mcp (Streamable HTTP, token auth) ──> tools ──> data provider (Yahoo) + SQLite/Postgres + model store
```

- 33 MCP tools (full list below)
- Python 3.12, official MCP Python SDK 2.x (`MCPServer`, stateless Streamable HTTP, JSON responses)
- pandas / NumPy / SciPy / scikit-learn, SQLAlchemy 2 (SQLite default, PostgreSQL supported)
- 84 automated tests (`pytest`), CI on GitHub Actions (unit, PostgreSQL, Docker end-to-end, live data)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [TEST_PROMPTS.md](TEST_PROMPTS.md) for prompts
to try in Claude.

---

## 1. Installation (local)

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone https://github.com/ibradamm/tradingview-mcp.git
cd tradingview-mcp
uv venv --python 3.12            # creates .venv
uv pip install -e ".[dev]"       # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env             # then edit .env
```

Generate a connector token and put it in `.env` as `MCP_AUTH_TOKEN`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 2. Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MCP_AUTH_TOKEN` | *(empty)* | Connector token (>= 24 chars). The server **refuses to start** without a token unless `ALLOW_UNAUTHENTICATED=true`. |
| `MCP_AUTH_TOKEN_SHA256` | *(empty)* | Alternative: comma-separated SHA-256 digests of accepted tokens (server never holds the plain token). |
| `ALLOW_UNAUTHENTICATED` | `false` | Local development only. |
| `MARKET_DATA_PROVIDER` | `yahoo` | `yahoo` (real data) or `synthetic` (fake, deterministic, offline). |
| `DATABASE_URL` | `sqlite:///./data/trading_mcp.db` | Any SQLAlchemy URL. `postgres://…` / `postgresql://…` are accepted (psycopg 3). |
| `MODEL_DIR` | `./data/models` | Where trained models are saved. |
| `CACHE_TTL_SECONDS` | `300` | In-memory cache of historical bars (0 disables). |
| `PERSIST_BARS` | `true` | Store fetched bars in the DB; served as fallback if the provider fails. |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-IP limit on `/mcp` (0 disables). |
| `CORS_ORIGINS` | `https://claude.ai,https://claude.com` | Allowed browser origins. |
| `ALLOWED_HOSTS` | *(empty)* | Optional Host-header allow-list (DNS-rebinding protection). |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address. `PORT` injected by the host wins. |
| `LOG_LEVEL` | `INFO` | |

Secrets live only in `.env` (git-ignored) or in the hosting platform's secret store. Never commit them.

## 3. Run locally

```bash
.venv/bin/trading-mcp            # or: python -m trading_mcp.main
curl http://localhost:8000/health
```

Endpoints:

| Path | Auth | Description |
|---|---|---|
| `POST/GET/DELETE /mcp` | token | MCP Streamable HTTP endpoint |
| `GET /health` | none | Liveness probe (no sensitive data) |

Nothing else is exposed (no docs UI, no admin routes).

Token can be sent as `Authorization: Bearer <token>` (Claude Code, scripts) or as `?key=<token>` in the
URL (Claude.ai custom connectors, which cannot send custom headers).

Use it from **Claude Code**:

```bash
claude mcp add --transport http trading "http://localhost:8000/mcp" --header "Authorization: Bearer $MCP_AUTH_TOKEN"
```

## 4. Tests

```bash
.venv/bin/pytest                 # full offline suite (synthetic data), ~20 s
.venv/bin/pytest -m live -v      # real Yahoo Finance data (needs internet)
.venv/bin/ruff check src tests   # lint
MCP_VERIFY_TOKEN=<token> .venv/bin/python scripts/verify_endpoint.py http://localhost:8000   # end-to-end
```

The offline suite covers market data normalisation, every indicator (against hand-computed reference
values), the screener, the backtest engine (exact fee/slippage arithmetic, no-look-ahead checks), walk-forward
folds, feature causality, leakage-free splits, models learning a real signal and *not* finding one in a
random walk, risk/VaR maths, the database layer, authentication, rate limiting, log redaction, and every MCP
tool through a real MCP client.

## 5. Deployment (public HTTPS URL for Claude.ai)

### Option A (recommended, free): Hugging Face Spaces

Why: free CPU tier with 2 vCPU / 16 GB RAM (enough for the ML tools), HTTPS, stable URL
`https://<user>-<space>.hf.space`, sleeps only after ~48 h without traffic (it wakes on the next request,
which then takes ~1 min). Free-tier specs as documented by Hugging Face at the time of writing; check
[huggingface.co/pricing](https://huggingface.co/pricing).

One-time setup (about 5 minutes, no command line):

1. Create a free account on [huggingface.co](https://huggingface.co/join).
2. Create an access token: *Settings → Access Tokens → Create new token*, type **Write**. Copy it.
3. In this GitHub repository: *Settings → Secrets and variables → Actions*:
   - *Secrets* tab → **New repository secret** `HF_TOKEN` = the Hugging Face token.
   - *Secrets* tab → **New repository secret** `MCP_AUTH_TOKEN` = your connector token (>= 24 random characters).
   - *Variables* tab → **New repository variable** `HF_SPACE` = `<your-hf-username>/trading-mcp`.
4. *Actions* tab → **Deploy to Hugging Face Spaces** → **Run workflow**.

The workflow creates the Space (public, so Claude.ai can reach it; `/mcp` stays token-protected), stores the
token as a Space **secret**, builds the Docker image, waits until it runs and then verifies the public URL
end-to-end with real market data. Every later push redeploys automatically.

Your URLs: `https://<user>-trading-mcp.hf.space/health` and `https://<user>-trading-mcp.hf.space/mcp?key=<token>`.

### Option B: Render (free, weaker)

*Render dashboard → New → Blueprint →* select this repository (uses `render.yaml`). Render generates
`MCP_AUTH_TOKEN`; copy it from the service's *Environment* tab. Free plan: 512 MB RAM, shared CPU, sleeps
after 15 min idle (~1 min cold start, which may make the first Claude call time out). ML walk-forward tools
are slow there.

### Option C: Google Cloud Run (generous free tier, requires a billing account)

```bash
gcloud run deploy trading-mcp --source . --region europe-west1 --allow-unauthenticated \
  --set-secrets MCP_AUTH_TOKEN=mcp-token:latest --memory 1Gi --cpu 1
```
(`--allow-unauthenticated` refers to Google IAM; the app still requires its own token.)

### Temporary test endpoint (GitHub Actions + Cloudflare quick tunnel)

`.github/workflows/public-demo.yml` starts the server on a GitHub runner behind a Cloudflare quick tunnel
for up to 120 minutes and verifies it end-to-end. It exists to **test** the public HTTPS path. It is not
hosting: the URL changes every run and it stops automatically. Only the SHA-256 of the token is passed to the
workflow, because this repository is public.

### Cloud comparison (September 2026, verify current terms before relying on them)

| Platform | Free tier | Fit for this project |
|---|---|---|
| Hugging Face Spaces (Docker) | Yes: 2 vCPU, 16 GB RAM, sleeps after ~48 h idle | **Best free fit** (ML needs RAM/CPU) |
| Render | Yes: 512 MB, shared CPU, sleeps after 15 min | Works, slow cold starts, weak for ML |
| Google Cloud Run | Yes (monthly free quota) but billing account required | Excellent, not "no card" |
| Fly.io | No free tier for new accounts (pay-as-you-go) | Paid |
| Railway | Trial credit only, then paid | Paid |
| Cloudflare Workers (Python) | Yes | **Not compatible**: Pyodide runtime, CPU/size limits, no scikit-learn training at this scale |

## 6. Connect to Claude.ai

1. Claude.ai → **Settings → Connectors → Add custom connector**.
2. Name: `Trading MCP`. URL: `https://<your-host>/mcp?key=<your token>`.
3. Leave OAuth fields empty. Save, then enable the connector in a chat ("Search and tools" menu).
4. Try the prompts in [TEST_PROMPTS.md](TEST_PROMPTS.md).

Treat the full connector URL as a password. To rotate the token, change it on the host and update the connector.

## 7. MCP tools

| Group | Tool | What it does |
|---|---|---|
| Market data | `get_quote` | Price, change vs previous close, volume, timestamp, source |
| | `get_historical_data` | OHLCV bars, timeframes 1m 5m 15m 30m 1h 4h 1d 1w |
| | `search_symbol` | Find tickers by name |
| Technical | `calculate_rsi`, `calculate_macd`, `calculate_sma`, `calculate_ema`, `calculate_bollinger_bands`, `calculate_atr`, `calculate_adx`, `calculate_stochastic`, `calculate_vwap`, `calculate_volatility`, `calculate_returns`, `calculate_drawdown`, `calculate_correlation` | Configurable indicators; latest value + recent series |
| Analysis | `multi_timeframe_analysis` | Trend, RSI, MACD, MAs, VWAP, volatility, volume and descriptive signals per timeframe, plus alignment |
| Screener | `market_screener` | Combine filters: price, volume, % change, RSI, volatility, market cap, trend, relative volume, breakout, distance to VWAP / SMA50 / SMA200 / EMA20 |
| Backtesting | `list_strategies`, `backtest_strategy`, `walk_forward_backtest`, `list_saved_runs`, `get_backtest_run` | Strategies with fees, slippage and position sizing; TRAIN→VALIDATION→TEST walk-forward optimisation |
| Machine learning | `create_features`, `train_model`, `evaluate_model` (hold-out or walk-forward), `predict`, `save_model`, `load_model`, `list_models` | Logistic regression, random forest, gradient boosting; classification or regression; configurable horizon |
| Risk | `calculate_position_size`, `risk_analysis` | Fixed-fractional sizing; volatility, drawdown, VaR/CVaR (historical, normal, Cornish-Fisher), concentration, correlation, strategy risk |
| Portfolio | `portfolio_analysis` | Return, volatility, Sharpe, Sortino, drawdown, VaR, correlations, diversification ratio, risk contributions |

Symbols: `NVDA`, `NASDAQ:NVDA`, `SPY`, `SPX`/`^GSPC`, `EURUSD`, `EUR/USD`, `BTCUSD`, `BTC-USD`,
`BINANCE:BTCUSDT`, `ES1!`/`ES=F`, `MC` + `exchange=PA` (Paris) are all understood.

## 8. Usage examples

- "Get the latest market data for NVDA."
- "Screen US mega caps for relative volume > 1.5, change > 3 % and RSI between 50 and 70."
- "Backtest an SMA 20/50 crossover on SPY since 2015 with 5 bps fees and 5 bps slippage, then run a walk-forward."
- "Train a random forest on NVDA daily bars to classify the 5-day direction, then evaluate it walk-forward."

## 9. Data sources

| Source | Used for | Status |
|---|---|---|
| Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) | Stocks, ETFs, indices, forex, crypto, futures, symbol search, market cap | **Unofficial** client of Yahoo's public endpoints. Not affiliated with Yahoo. Data can be delayed (typically ~15 min for equities), adjusted, incomplete, or break without notice. Yahoo's terms allow personal use only. |
| Synthetic provider | Tests / offline development | Fake data, clearly labelled |

TradingView is **not** used: it offers no public market-data API, and scraping it would breach its terms.
To add a licensed source (Polygon/Massive, Twelve Data, Alpaca, Tiingo, Kraken/Coinbase for crypto...),
implement `MarketDataProvider` (`src/trading_mcp/data/base.py`: `get_quote`, `get_history`,
`search_symbol`) and register it in `data/factory.py`.

Yahoo limits: 1m bars ≈ last 7 days, 5m–30m ≈ last 60 days, 1h ≈ last 730 days (4h is resampled from 1h).
The server clamps requests and reports this in `meta.notes`.

## 10. Limitations (read before trusting any number)

- **Data**: free, delayed, unofficial; corporate actions handled by Yahoo's adjustment only; no point-in-time
  fundamentals. **Survivorship bias**: screener universes are today's constituents.
- **Backtests**: next-bar-open execution, fixed fees/slippage in bps, fractional shares, no borrow costs,
  dividends only through adjusted prices, no market impact, no partial fills, one position per strategy.
- **Walk-forward**: out-of-sample stitched returns are the honest estimate. A "no overfitting signal" verdict
  does not prove an edge exists.
- **ML**: probabilities are model outputs, only as reliable as the calibration table shows. On liquid markets
  expect test ROC-AUC close to 0.5. **Look-ahead bias** is prevented by causal features, next-bar execution and
  purged chronological splits; **regime change** is not something any split can remove.
- **Hosting**: free tiers sleep (first call slow) and have ephemeral disks: SQLite data and saved models are lost
  on redeploy/restart. For persistence use a free PostgreSQL (e.g. Neon/Supabase) via `DATABASE_URL`;
  models then need an object store (not implemented).
- **Auth**: a shared-secret token in the URL is simple but weaker than OAuth (it can leak through browser
  history or screenshots). OAuth 2.1 is the upgrade path (see ARCHITECTURE.md).
- Rate limiting is in-memory per process.

## 11. Costs

- Code, Yahoo data, GitHub Actions on a public repository: **free**.
- Hugging Face Spaces CPU basic: **free**. Render free plan: **free**. Cloud Run: free quota, card required.
- No paid service is used by default. Licensed data feeds would be the first real cost if needed.

## 12. Project layout

```
src/trading_mcp/
  server.py, main.py, config.py, auth.py, security.py
  data/        MarketDataProvider, Yahoo + synthetic providers, symbols, caches
  indicators/  technical indicators (pandas)
  analysis/    technical snapshot (multi-timeframe, screener)
  screener/    filters and universes
  backtest/    strategies, engine, metrics, walk-forward
  ml/          features/targets, model registry, training/evaluation, model store
  risk/        position sizing, VaR, portfolio statistics
  db/          SQLAlchemy models, session, repository
  tools/       MCP tool wrappers (one module per domain)
tests/         pytest suite
scripts/       verify_endpoint.py, deploy_hf.py
```
