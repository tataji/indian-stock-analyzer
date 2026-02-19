"""
models/trainer.py
Batch training pipeline — train models for multiple symbols.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.fetcher import fetch_ohlcv
from models.technical import TechnicalModel

logger = logging.getLogger(__name__)


def train_single(symbol: str, period: str = "3y", force: bool = False) -> Optional[dict]:
    """
    Train and save a TechnicalModel for one symbol.
    Returns training report or None on failure.
    """
    model = TechnicalModel(symbol)

    # Load cached model unless forced
    if not force and model.load():
        logger.info("Using cached model for %s", symbol)
        return {"symbol": symbol, "status": "cached"}

    df = fetch_ohlcv(symbol, period=period)
    if df is None or len(df) < 300:
        logger.warning("Insufficient data for %s", symbol)
        return None

    try:
        report = model.train(df)
        model.save()
        report["symbol"] = symbol
        report["status"] = "trained"
        return report
    except Exception as exc:
        logger.error("Training failed for %s: %s", symbol, exc)
        return None


def train_batch(symbols: list[str], period: str = "3y", force: bool = False) -> pd.DataFrame:
    """
    Train models for a list of symbols.
    Returns a summary DataFrame.
    """
    records = []
    for sym in symbols:
        logger.info("Processing %s …", sym)
        report = train_single(sym, period=period, force=force)
        if report:
            records.append(report)
    return pd.DataFrame(records)
