"""
utils/screener.py
Stock screener — runs technical + fundamental analysis across multiple symbols
and ranks them by composite score.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from config.settings import NIFTY50_SYMBOLS
from data.fetcher import fetch_ohlcv, fetch_fundamentals
from data.preprocessor import build_features
from models.technical import TechnicalModel
from models.fundamental import score_fundamentals

logger = logging.getLogger(__name__)


def _analyze_symbol(symbol: str) -> Optional[dict]:
    try:
        df_raw = fetch_ohlcv(symbol, period="1y")
        if df_raw is None or len(df_raw) < 60:
            return None

        df = build_features(df_raw, include_labels=False)
        if df.empty:
            return None

        fundamentals = fetch_fundamentals(symbol)
        fund_result  = score_fundamentals(fundamentals)

        # Technical model
        model = TechnicalModel(symbol)
        trained = model.load()
        if not trained:
            try:
                model.train(df_raw)
                model.save()
            except Exception:
                return None

        tech_signal = model.predict(df_raw)

        # Composite score (weighted)
        tech_score = tech_signal["proba_bull"] * 10     # 0-10
        fund_score = fund_result.score                   # 0-18
        composite  = round(0.45 * tech_score + 0.55 * (fund_score / 18 * 10), 2)

        row = df.iloc[-1]
        return {
            "symbol":       symbol,
            "close":        round(float(row["Close"]), 2),
            "rsi":          round(float(row.get("RSI", 0)), 1),
            "macd_bullish": int(row.get("MACD_bullish", 0)),
            "adx":          round(float(row.get("ADX", 0)), 1),
            "tech_signal":  tech_signal["signal"],
            "tech_conf":    round(tech_signal["confidence"], 3),
            "proba_bull":   round(tech_signal["proba_bull"], 3),
            "fund_score":   fund_result.score,
            "fund_max":     fund_result.max_score,
            "fund_grade":   fund_result.grade,
            "composite":    composite,
            "pe":           fundamentals.get("trailingPE"),
            "roe":          fundamentals.get("returnOnEquity"),
            "rev_growth":   fundamentals.get("revenueGrowth"),
        }
    except Exception as exc:
        logger.error("Error screening %s: %s", symbol, exc)
        return None


def run_screener(
    symbols:      list[str] = None,
    max_workers:  int = 5,
    top_n:        int = 20,
) -> pd.DataFrame:
    """
    Run screener on a list of symbols (defaults to Nifty 50).
    Returns a DataFrame sorted by composite score.
    """
    symbols = symbols or NIFTY50_SYMBOLS
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("composite", ascending=False).head(top_n).reset_index(drop=True)
    df.index += 1   # rank starts at 1
    return df


def filter_screener(
    df:             pd.DataFrame,
    signal:         str  = "ALL",       # BULLISH / BEARISH / ALL
    min_rsi:        float = 0,
    max_rsi:        float = 100,
    min_fund_score: int  = 0,
    min_adx:        float = 0,
) -> pd.DataFrame:
    """Apply filters to a screener result DataFrame."""
    out = df.copy()
    if signal != "ALL":
        out = out[out["tech_signal"] == signal]
    out = out[out["rsi"].between(min_rsi, max_rsi)]
    out = out[out["fund_score"] >= min_fund_score]
    out = out[out["adx"] >= min_adx]
    return out
