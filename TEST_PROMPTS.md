# Test prompts for Claude

Enable the **Trading MCP** connector in a Claude.ai chat, then try these prompts in order. Each line says
which tools should be called and what to check. All outputs are research data, not advice.

## 1. Connectivity and market data

| Prompt | Expected tools | Check |
|---|---|---|
| Get the latest market data for NVDA. | `get_quote` | price, change %, volume, timestamp, source = yahoo_finance |
| Show me the last 10 hourly candles for BTC. | `get_historical_data` (BTCUSD, 1h) | 10 OHLCV bars in UTC |
| Find the ticker for LVMH and give me its quote in Paris. | `search_symbol`, `get_quote` (MC, exchange PA) | MC.PA in EUR |
| Quote EURUSD, the S&P 500 index and the E-mini S&P future. | `get_quote` ×3 | EURUSD=X, ^GSPC, ES=F |

## 2. Technical analysis

| Prompt | Expected tools |
|---|---|
| Calculate RSI, MACD, VWAP and ATR for NVDA. | `calculate_rsi`, `calculate_macd`, `calculate_vwap`, `calculate_atr` |
| Compute Bollinger Bands (20, 2.5) and ADX on AAPL daily. | `calculate_bollinger_bands`, `calculate_adx` |
| What is the maximum drawdown of TSLA since 2021 and when did it happen? | `calculate_drawdown` |
| Correlation between NVDA, AMD, SPY and BTC over the last 2 years. | `calculate_correlation` |
| 30-day rolling volatility and log returns of ETH. | `calculate_volatility`, `calculate_returns` |

## 3. Multi-timeframe analysis

- "Analyze BTC on 5m, 15m and 1h timeframes." → `multi_timeframe_analysis` (descriptive conditions, no buy/sell call)
- "Multi-timeframe analysis of NVDA on 5m, 15m, 1h, 4h and 1d. Do the timeframes agree?"

## 4. Screener

- "Find US mega caps with high relative volume (> 1.5), up more than 3 % and RSI between 50 and 70." → `market_screener`
- "Which crypto majors are in an uptrend and within 2 % of their VWAP?"
- "Screen the CAC 40 for stocks above their 200-day SMA with a breakout of the 20-day high."
- "Scan AAPL, MSFT, NVDA, AMD, INTC for market cap above 500 billion and RSI below 40."

## 5. Backtesting

- "List the available strategies." → `list_strategies`
- "Run a backtest of a moving average strategy on SPY since 2015: SMA 20/50, 10 000 USD, 5 bps fees, 5 bps slippage." → `backtest_strategy`
- "Same strategy with volatility-target sizing at 12 % and allow shorts. Compare with buy and hold."
- "Backtest an RSI mean-reversion strategy on QQQ hourly bars over the last year."
- "Run a walk-forward optimisation of the SMA crossover on SPY and tell me if it looks overfit." → `walk_forward_backtest`
- "Show my last saved backtests." → `list_saved_runs`

## 6. Machine learning

- "Show me the ML features for NVDA daily." → `create_features`
- "Train a machine learning model using historical data: random forest on NVDA daily bars, predict the 5-day direction." → `train_model`
- "Evaluate the model using walk-forward validation." → `evaluate_model(method="walk_forward")`
- "Train a gradient boosting regression on SPY 1h bars for the next 4 hours (horizon 4), then predict." → `train_model`, `predict`
- "Compare logistic regression, random forest and gradient boosting on AAPL. Which one overfits most?"
- "List my models and reload the best one." → `list_models`, `load_model`

What to look for: test ROC-AUC vs 0.5, train vs test gap, calibration table, trading simulation vs buy and
hold, and the warnings. A near-0.5 AUC is the normal, honest result on liquid markets.

## 7. Risk and portfolio

- "I have 25 000 USD, I risk 1 % per trade, entry 120, stop 114. Position size?" → `calculate_position_size`
- "Analyze the risk of this strategy." (after a backtest) → `risk_analysis(strategy=...)`
- "Risk analysis of 50 % NVDA, 30 % SPY, 20 % BTC with 20 000 USD: VaR 95 % and 99 %, concentration." → `risk_analysis`
- "Portfolio analysis of 40 % SPY, 30 % TLT, 20 % GLD, 10 % BTC since 2020: Sharpe, drawdown, diversification." → `portfolio_analysis`

## 8. Error handling (should fail gracefully)

- "Get 3h candles for NVDA." → clear "Unsupported timeframe" message
- "Quote the ticker ZZZZZZZZ." → clear "No data returned" message
- "Backtest a strategy called magic_money." → list of valid strategies
