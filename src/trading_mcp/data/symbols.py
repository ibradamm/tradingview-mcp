"""Translate user-friendly / TradingView-style symbols into Yahoo Finance symbols.

Examples: ``NASDAQ:NVDA -> NVDA``, ``BTCUSD -> BTC-USD``, ``EUR/USD -> EURUSD=X``,
``ES1! -> ES=F``, ``SPX -> ^GSPC``, ``MC`` + exchange ``PA`` -> ``MC.PA``.
"""

from __future__ import annotations

import re

CRYPTO = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX", "DOT", "LINK", "LTC", "MATIC",
    "TRX", "SHIB", "BCH", "XLM", "ATOM", "UNI", "ETC", "NEAR", "APT", "ARB", "OP", "SUI", "PEPE",
}
FIAT = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "CNY", "HKD", "SGD", "MXN"}
CRYPTO_QUOTES = {"USD", "USDT", "USDC", "EUR", "GBP", "BTC", "ETH"}

INDEX_ALIASES = {
    "SPX": "^GSPC", "SP500": "^GSPC", "NDX": "^NDX", "NASDAQ100": "^NDX", "IXIC": "^IXIC",
    "DJI": "^DJI", "DJIA": "^DJI", "RUT": "^RUT", "VIX": "^VIX", "DAX": "^GDAXI",
    "CAC40": "^FCHI", "PX1": "^FCHI", "FTSE": "^FTSE", "UKX": "^FTSE", "N225": "^N225",
    "NI225": "^N225", "STOXX50": "^STOXX50E", "SX5E": "^STOXX50E", "HSI": "^HSI",
}

EXCHANGE_SUFFIX = {
    "NASDAQ": "", "NYSE": "", "AMEX": "", "ARCA": "", "NYSEARCA": "", "BATS": "", "US": "",
    "PA": ".PA", "EURONEXT": ".PA", "EPA": ".PA", "XPAR": ".PA",
    "AS": ".AS", "AMS": ".AS", "BR": ".BR", "MI": ".MI", "MIL": ".MI",
    "L": ".L", "LSE": ".L", "LON": ".L",
    "DE": ".DE", "XETRA": ".DE", "XETR": ".DE", "F": ".F", "FWB": ".F",
    "SW": ".SW", "SIX": ".SW", "TO": ".TO", "TSX": ".TO", "V": ".V", "TSXV": ".V",
    "T": ".T", "TSE": ".T", "HK": ".HK", "HKEX": ".HK", "AX": ".AX", "ASX": ".AX",
    "MC": ".MC", "BME": ".MC", "ST": ".ST", "OL": ".OL", "CO": ".CO",
}

# TradingView venue prefixes that are not stock exchanges (no Yahoo suffix applies).
_NON_EQUITY_VENUES = {"BINANCE", "COINBASE", "KRAKEN", "BITSTAMP", "CRYPTO", "FX", "OANDA", "FX_IDC", "FOREXCOM"}
_TV_FUTURE = re.compile(r"^([A-Z0-9]{1,4})1!$")


def to_yahoo_symbol(ticker: str, exchange: str | None = None) -> str:
    """Return the Yahoo Finance symbol for a user supplied ticker."""
    raw = ticker.strip().upper()
    if not raw:
        raise ValueError("ticker must not be empty")

    if ":" in raw:  # TradingView style "EXCHANGE:SYMBOL"
        prefix, raw = raw.split(":", 1)
        exchange = exchange or prefix
        if prefix in _NON_EQUITY_VENUES:
            exchange = None

    if raw.startswith("^") or raw.endswith(("=X", "=F")) or "-" in raw:
        return raw
    if raw in INDEX_ALIASES:
        return INDEX_ALIASES[raw]

    future = _TV_FUTURE.match(raw)
    if future:
        return f"{future.group(1)}=F"

    if "/" in raw:
        base, quote = raw.split("/", 1)
        if base in CRYPTO:
            return f"{base}-{'USD' if quote in {'USDT', 'USDC'} else quote}"
        return f"{base}{quote}=X"

    if raw in CRYPTO:
        return f"{raw}-USD"
    for quote in sorted(CRYPTO_QUOTES, key=len, reverse=True):
        base = raw[: -len(quote)]
        if raw.endswith(quote) and base in CRYPTO:
            return f"{base}-{'USD' if quote in {'USDT', 'USDC'} else quote}"

    if len(raw) == 6 and raw[:3] in FIAT and raw[3:] in FIAT:
        return f"{raw}=X"

    if exchange:
        suffix = EXCHANGE_SUFFIX.get(exchange.strip().upper())
        if suffix is None:
            suffix = "." + exchange.strip().upper()
        if suffix and not raw.endswith(suffix):
            return f"{raw}{suffix}"
    return raw
