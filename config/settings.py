"""
config/settings.py
Central configuration for the Indian Stock Analyzer.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ── Model Settings ────────────────────────────────────────────────────────────
LLM_MODEL: str = "claude-sonnet-4-6"
LLM_MAX_TOKENS: int = 1500

# ── Data Settings ─────────────────────────────────────────────────────────────
DEFAULT_PERIOD: str = "2y"           # yfinance period string
DEFAULT_INTERVAL: str = "1d"
PREDICTION_HORIZON_DAYS: int = 5     # predict n-day forward return
UPSIDE_THRESHOLD: float = 0.02       # 2% to label as "bullish"

# ── Technical Indicator Parameters ───────────────────────────────────────────
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BB_PERIOD: int = 20
BB_STD: int = 2
SMA_SHORT: int = 20
SMA_LONG: int = 50
EMA_PERIOD: int = 50
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
STOCH_K: int = 14
STOCH_D: int = 3
CCI_PERIOD: int = 20
WILLIAMS_PERIOD: int = 14
VOLUME_MA_PERIOD: int = 20

# ── Popular NSE Symbols for Screening ────────────────────────────────────────
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
    "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "MARUTI", "NESTLEIND", "ULTRACEMCO", "WIPRO", "TITAN",
    "ADANIENT", "ADANIPORTS", "TATAMOTORS", "TATASTEEL", "SUNPHARMA",
    "TECHM", "NTPC", "POWERGRID", "ONGC", "COALINDIA",
    "BAJAJFINSV", "BAJAJ-AUTO", "GRASIM", "INDUSINDBK", "DIVISLAB",
    "CIPLA", "EICHERMOT", "BPCL", "JSWSTEEL", "HINDALCO",
    "BRITANNIA", "DRREDDY", "UPL", "SBILIFE", "APOLLOHOSP",
    "HDFCLIFE", "HEROMOTOCO", "M&M", "SHREECEM", "TATACONSUM"
]

SECTOR_MAP = {
    "IT":          ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "Banking":     ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "FMCG":        ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Auto":        ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "M&M", "HEROMOTOCO"],
    "Pharma":      ["SUNPHARMA", "DIVISLAB", "CIPLA", "DRREDDY", "APOLLOHOSP"],
    "Energy":      ["RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID", "COALINDIA"],
    "Metal":       ["TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIENT"],
    "Finance":     ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE"],
    "Infra":       ["LT", "ADANIPORTS", "GRASIM", "ULTRACEMCO", "SHREECEM"],
    "Consumer":    ["ASIANPAINT", "TITAN"],
}

# ── Fundamental Scoring Thresholds ────────────────────────────────────────────
FUNDAMENTAL_THRESHOLDS = {
    "pe_attractive":     20,
    "pe_expensive":      40,
    "pb_attractive":     2,
    "roe_strong":        0.15,
    "de_low":            0.5,
    "de_high":           2.0,
    "revenue_growth_good": 0.15,
    "current_ratio_good":  1.5,
    "profit_margin_good":  0.10,
}

# ── Model Paths ───────────────────────────────────────────────────────────────
MODEL_DIR: str = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)
