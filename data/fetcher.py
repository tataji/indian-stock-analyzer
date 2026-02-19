"""
data/fetcher.py
Handles all data fetching from Yahoo Finance (NSE/BSE stocks).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import DEFAULT_PERIOD, DEFAULT_INTERVAL

logger = logging.getLogger(__name__)


def _nse(symbol: str) -> str:
    """Append .NS suffix for NSE listing."""
    symbol = symbol.upper().strip()
    if not symbol.endswith((".NS", ".BO")):
        return f"{symbol}.NS"
    return symbol


def fetch_ohlcv(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a given NSE symbol.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    or None if the fetch fails.
    """
    ticker_sym = _nse(symbol)
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(ticker_sym)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                logger.warning("No OHLCV data returned for %s", ticker_sym)
                return None
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            logger.info("Fetched %d rows for %s", len(df), ticker_sym)
            return df
        except Exception as exc:
            logger.error("Attempt %d failed for %s: %s", attempt + 1, ticker_sym, exc)
            time.sleep(2)
    return None


def fetch_fundamentals(symbol: str) -> dict:
    """
    Fetch key fundamental metrics for a stock.
    Returns a dict; missing fields default to None.
    """
    ticker_sym = _nse(symbol)
    try:
        info = yf.Ticker(ticker_sym).info
    except Exception as exc:
        logger.error("Failed to fetch fundamentals for %s: %s", ticker_sym, exc)
        return {}

    keys = [
        "trailingPE", "forwardPE", "priceToBook", "debtToEquity",
        "returnOnEquity", "returnOnAssets", "revenueGrowth", "earningsGrowth",
        "trailingEps", "forwardEps", "marketCap", "dividendYield",
        "currentRatio", "quickRatio", "freeCashflow", "operatingMargins",
        "profitMargins", "grossMargins", "totalRevenue", "totalDebt",
        "totalCash", "bookValue", "beta", "52WeekChange",
        "shortName", "longName", "sector", "industry", "website",
        "country", "fullTimeEmployees", "longBusinessSummary",
    ]
    return {k: info.get(k) for k in keys}


def fetch_batch_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Fetch fundamentals for multiple symbols with a small delay."""
    results = {}
    for sym in symbols:
        results[sym] = fetch_fundamentals(sym)
        time.sleep(0.3)   # be polite to Yahoo Finance
    return results


def get_company_info(symbol: str) -> dict:
    """Return company name, sector, industry for display purposes."""
    f = fetch_fundamentals(symbol)
    return {
        "name":     f.get("longName") or f.get("shortName") or symbol,
        "sector":   f.get("sector", "N/A"),
        "industry": f.get("industry", "N/A"),
        "website":  f.get("website", "N/A"),
        "employees": f.get("fullTimeEmployees", "N/A"),
        "description": f.get("longBusinessSummary", "N/A"),
    }
