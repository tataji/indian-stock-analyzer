"""
data/news.py
News fetching and sentiment analysis for Indian stocks.
Uses Google News RSS (no API key required) + keyword-based sentiment scoring.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Sentiment Lexicon (Finance-tuned) ────────────────────────────────────────
POSITIVE_WORDS = {
    "surge", "rally", "gain", "rise", "jump", "soar", "beat", "strong",
    "profit", "growth", "bullish", "upgrade", "outperform", "record",
    "buy", "positive", "up", "high", "expand", "win", "award", "deal",
    "partnership", "launch", "acquisition", "dividend", "bonus", "split",
    "breakout", "momentum", "recovery", "rebound", "upside", "boost",
    "earnings beat", "revenue growth", "market share", "order win",
}

NEGATIVE_WORDS = {
    "fall", "drop", "decline", "loss", "plunge", "crash", "weak", "sell",
    "downgrade", "underperform", "bearish", "cut", "miss", "debt", "fine",
    "penalty", "fraud", "investigation", "lawsuit", "down", "low", "exit",
    "delay", "recall", "warning", "risk", "concern", "slowdown", "miss",
    "disappointing", "restructure", "layoff", "write-off", "impairment",
}

INTENSIFIERS = {"very", "highly", "significantly", "sharply", "massively", "deeply"}


def _score_text(text: str) -> float:
    """
    Simple lexicon-based sentiment score.
    Returns float in [-1.0, 1.0]:  -1 = very negative, +1 = very positive
    """
    text_lower = text.lower()
    words      = re.findall(r'\b\w+\b', text_lower)

    pos_count  = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count  = sum(1 for w in words if w in NEGATIVE_WORDS)
    intensify  = sum(0.3 for w in words if w in INTENSIFIERS)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    raw = (pos_count - neg_count) / total
    return max(-1.0, min(1.0, raw + (intensify if raw > 0 else -intensify)))


def _sentiment_label(score: float) -> str:
    if score >= 0.3:
        return "🟢 Positive"
    elif score <= -0.3:
        return "🔴 Negative"
    return "🟡 Neutral"


def fetch_news(symbol: str, company_name: str = "", max_articles: int = 10) -> list[dict]:
    """
    Fetch recent news from Google News RSS for a stock.

    Parameters
    ----------
    symbol       : NSE symbol (e.g. RELIANCE)
    company_name : Full company name for better search results
    max_articles : Max number of articles to return

    Returns list of dicts with: title, link, published, sentiment_score, sentiment_label, source
    """
    query   = company_name if company_name else symbol
    # Add "NSE" or "stock" to narrow to financial news
    query   = f"{query} NSE stock India"
    encoded = quote(query)
    url     = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    articles = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")[:max_articles]

        for item in items:
            title     = item.find("title").get_text(strip=True) if item.find("title") else ""
            link      = item.find("link").get_text(strip=True)  if item.find("link")  else ""
            pub_date  = item.find("pubDate")
            source_el = item.find("source")

            published = ""
            if pub_date:
                try:
                    from email.utils import parsedate_to_datetime
                    published = parsedate_to_datetime(pub_date.get_text()).strftime("%d %b %Y %H:%M")
                except Exception:
                    published = pub_date.get_text(strip=True)

            source = source_el.get_text(strip=True) if source_el else "Google News"

            score = _score_text(title)
            articles.append({
                "title":           title,
                "link":            link,
                "published":       published,
                "source":          source,
                "sentiment_score": round(score, 3),
                "sentiment_label": _sentiment_label(score),
            })

    except Exception as exc:
        logger.error("News fetch failed for %s: %s", symbol, exc)

    return articles


def aggregate_sentiment(articles: list[dict]) -> dict:
    """
    Aggregate sentiment scores across multiple articles.
    Returns overall sentiment summary.
    """
    if not articles:
        return {
            "overall_score":  0.0,
            "overall_label":  "🟡 Neutral",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count":  0,
            "total":          0,
        }

    scores = [a["sentiment_score"] for a in articles]
    avg    = sum(scores) / len(scores)

    return {
        "overall_score":  round(avg, 3),
        "overall_label":  _sentiment_label(avg),
        "positive_count": sum(1 for s in scores if s >= 0.3),
        "negative_count": sum(1 for s in scores if s <= -0.3),
        "neutral_count":  sum(1 for s in scores if -0.3 < s < 0.3),
        "total":          len(articles),
    }
