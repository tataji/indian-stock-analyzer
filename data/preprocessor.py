"""
data/preprocessor.py
Feature engineering: technical indicators + derived features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, SMA_SHORT, SMA_LONG, EMA_PERIOD,
    ATR_PERIOD, ADX_PERIOD, STOCH_K, STOCH_D, CCI_PERIOD,
    WILLIAMS_PERIOD, VOLUME_MA_PERIOD,
    PREDICTION_HORIZON_DAYS, UPSIDE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) used in RSI."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _stochastic(high, low, close, k=14, d=3):
    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low   = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def _adx(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = _rma(tr, period)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di  = 100 * _rma(pd.Series(plus_dm,  index=close.index), period) / atr
    minus_di = 100 * _rma(pd.Series(minus_dm, index=close.index), period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx, period)
    return adx, plus_di, minus_di


# ── Main Feature Builder ──────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to an OHLCV DataFrame.
    Returns a copy with new columns appended.
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Trend ──────────────────────────────────────────────────────────────
    df["SMA_20"]  = close.rolling(SMA_SHORT).mean()
    df["SMA_50"]  = close.rolling(SMA_LONG).mean()
    df["EMA_20"]  = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"]  = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=200, adjust=False).mean()

    # Golden / Death cross signal
    df["golden_cross"] = (df["EMA_20"] > df["EMA_50"]).astype(int)

    # Price relative to moving averages
    df["price_vs_sma20"]  = (close - df["SMA_20"]) / df["SMA_20"]
    df["price_vs_sma50"]  = (close - df["SMA_50"]) / df["SMA_50"]
    df["price_vs_ema200"] = (close - df["EMA_200"]) / df["EMA_200"]

    # ── Momentum ───────────────────────────────────────────────────────────
    df["RSI"] = _rsi(close, RSI_PERIOD)
    df["RSI_overbought"] = (df["RSI"] > 70).astype(int)
    df["RSI_oversold"]   = (df["RSI"] < 30).astype(int)

    # MACD
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    df["MACD_bullish"] = (df["MACD"] > df["MACD_signal"]).astype(int)

    # Stochastic
    df["Stoch_K"], df["Stoch_D"] = _stochastic(high, low, close, STOCH_K, STOCH_D)

    # CCI
    df["CCI"] = _cci(high, low, close, CCI_PERIOD)

    # Williams %R
    df["Williams_R"] = _williams_r(high, low, close, WILLIAMS_PERIOD)

    # ROC – Rate of Change
    df["ROC_5"]  = close.pct_change(5) * 100
    df["ROC_10"] = close.pct_change(10) * 100
    df["ROC_20"] = close.pct_change(20) * 100

    # ── Volatility ─────────────────────────────────────────────────────────
    df["ATR"] = _atr(high, low, close, ATR_PERIOD)
    df["ATR_pct"] = df["ATR"] / close

    # Bollinger Bands
    bb_ma  = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std()
    df["BB_upper"] = bb_ma + BB_STD * bb_std
    df["BB_lower"] = bb_ma - BB_STD * bb_std
    df["BB_mid"]   = bb_ma
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_ma
    df["BB_pct"]   = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(50).mean()).astype(int)

    # Historical volatility (annualised)
    df["HV_20"]  = close.pct_change().rolling(20).std()  * np.sqrt(252)
    df["HV_60"]  = close.pct_change().rolling(60).std()  * np.sqrt(252)

    # ── Volume ─────────────────────────────────────────────────────────────
    df["Vol_MA20"]   = volume.rolling(VOLUME_MA_PERIOD).mean()
    df["Vol_ratio"]  = volume / df["Vol_MA20"].replace(0, np.nan)
    df["Vol_surge"]  = (df["Vol_ratio"] > 2).astype(int)

    # On-Balance Volume
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_MA"] = obv.rolling(20).mean()
    df["OBV_trend"] = (df["OBV"] > df["OBV_MA"]).astype(int)

    # ── ADX / Trend Strength ───────────────────────────────────────────────
    df["ADX"], df["DI_plus"], df["DI_minus"] = _adx(high, low, close, ADX_PERIOD)
    df["ADX_trending"] = (df["ADX"] > 25).astype(int)

    # ── Support / Resistance via 52-week ──────────────────────────────────
    df["52w_high"] = close.rolling(252).max()
    df["52w_low"]  = close.rolling(252).min()
    df["pct_from_52w_high"] = (close - df["52w_high"]) / df["52w_high"]
    df["pct_from_52w_low"]  = (close - df["52w_low"])  / df["52w_low"]

    # ── Candle features ───────────────────────────────────────────────────
    df["candle_body"]    = (close - df["Open"]).abs() / df["Open"]
    df["candle_upper_shadow"] = (high - pd.concat([close, df["Open"]], axis=1).max(axis=1)) / df["Open"]
    df["candle_lower_shadow"] = (pd.concat([close, df["Open"]], axis=1).min(axis=1) - low) / df["Open"]
    df["bullish_candle"] = (close > df["Open"]).astype(int)

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward-return labels for ML training."""
    df = df.copy()
    fwd_return = df["Close"].shift(-PREDICTION_HORIZON_DAYS) / df["Close"] - 1
    df["target"]      = (fwd_return > UPSIDE_THRESHOLD).astype(int)
    df["fwd_return"]  = fwd_return
    return df.dropna(subset=["target"])


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only numeric feature columns (exclude raw OHLCV + labels)."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "target", "fwd_return",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def build_features(df: pd.DataFrame, include_labels: bool = True) -> pd.DataFrame:
    """Full pipeline: OHLCV → indicators → labels → dropna."""
    df = add_technical_indicators(df)
    if include_labels:
        df = add_labels(df)
    return df.replace([np.inf, -np.inf], np.nan).dropna()
