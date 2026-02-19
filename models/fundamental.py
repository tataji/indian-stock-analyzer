"""
models/fundamental.py
Rule-based fundamental scoring engine tuned for Indian equities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config.settings import FUNDAMENTAL_THRESHOLDS as T


@dataclass
class FundamentalSignal:
    name:        str
    value:       Optional[float]
    score_delta: int
    message:     str
    positive:    bool


@dataclass
class FundamentalResult:
    score:      int = 0
    max_score:  int = 18
    grade:      str = "N/A"
    signals:    list[FundamentalSignal] = field(default_factory=list)
    summary:    str = ""

    @property
    def score_pct(self) -> float:
        return round(self.score / self.max_score * 100, 1)

    @property
    def positive_signals(self) -> list[FundamentalSignal]:
        return [s for s in self.signals if s.positive]

    @property
    def negative_signals(self) -> list[FundamentalSignal]:
        return [s for s in self.signals if not s.positive]


def _fmt(val, pct: bool = False, prefix: str = "") -> str:
    if val is None:
        return "N/A"
    if pct:
        return f"{prefix}{val * 100:.1f}%"
    return f"{prefix}{val:.2f}"


def score_fundamentals(fundamentals: dict) -> FundamentalResult:
    """
    Score a stock on fundamental parameters relevant to Indian markets.

    Scoring rubric (max 18 points):
    - Valuation  : P/E, P/B        (0-4)
    - Quality    : ROE, Margins     (0-4)
    - Growth     : Revenue, EPS     (0-4)
    - Safety     : D/E, Current     (0-4)
    - Dividends  : Yield            (0-2)
    """
    result = FundamentalResult()
    f = fundamentals

    def add(name, value, delta, msg, positive):
        result.signals.append(FundamentalSignal(name, value, delta, msg, positive))
        result.score += delta

    # ── Valuation ─────────────────────────────────────────────────────────
    pe = f.get("trailingPE") or f.get("forwardPE")
    if pe is not None:
        if pe < T["pe_attractive"]:
            add("P/E Ratio", pe, 2, f"Attractive P/E of {pe:.1f} (< {T['pe_attractive']})", True)
        elif pe < T["pe_expensive"]:
            add("P/E Ratio", pe, 1, f"Moderate P/E of {pe:.1f}", True)
        else:
            add("P/E Ratio", pe, -1, f"Expensive P/E of {pe:.1f} (> {T['pe_expensive']})", False)

    pb = f.get("priceToBook")
    if pb is not None:
        if pb < T["pb_attractive"]:
            add("P/B Ratio", pb, 2, f"Trading below 2x book value (P/B={pb:.1f})", True)
        elif pb < 4:
            add("P/B Ratio", pb, 1, f"Fair P/B of {pb:.1f}", True)
        else:
            add("P/B Ratio", pb, 0, f"High P/B of {pb:.1f} — premium valuation", False)

    # ── Quality ───────────────────────────────────────────────────────────
    roe = f.get("returnOnEquity")
    if roe is not None:
        if roe > T["roe_strong"]:
            add("ROE", roe, 2, f"Strong ROE of {roe*100:.1f}% (> 15%)", True)
        elif roe > 0.08:
            add("ROE", roe, 1, f"Adequate ROE of {roe*100:.1f}%", True)
        else:
            add("ROE", roe, -1, f"Weak ROE of {roe*100:.1f}%", False)

    pm = f.get("profitMargins") or f.get("operatingMargins")
    if pm is not None:
        if pm > T["profit_margin_good"]:
            add("Profit Margin", pm, 2, f"Healthy margin of {pm*100:.1f}%", True)
        elif pm > 0.05:
            add("Profit Margin", pm, 1, f"Thin but positive margin of {pm*100:.1f}%", True)
        else:
            add("Profit Margin", pm, -1, f"Very thin margin of {pm*100:.1f}%", False)

    # ── Growth ────────────────────────────────────────────────────────────
    rev_growth = f.get("revenueGrowth")
    if rev_growth is not None:
        if rev_growth > T["revenue_growth_good"]:
            add("Revenue Growth", rev_growth, 2, f"Strong revenue growth: {rev_growth*100:.1f}%", True)
        elif rev_growth > 0.05:
            add("Revenue Growth", rev_growth, 1, f"Moderate growth: {rev_growth*100:.1f}%", True)
        else:
            add("Revenue Growth", rev_growth, -1, f"Weak/negative revenue growth: {rev_growth*100:.1f}%", False)

    eps_growth = f.get("earningsGrowth")
    if eps_growth is not None:
        if eps_growth > 0.20:
            add("EPS Growth", eps_growth, 2, f"Excellent earnings growth: {eps_growth*100:.1f}%", True)
        elif eps_growth > 0.05:
            add("EPS Growth", eps_growth, 1, f"Positive earnings growth: {eps_growth*100:.1f}%", True)
        else:
            add("EPS Growth", eps_growth, -1, f"Earnings declining: {eps_growth*100:.1f}%", False)

    # ── Safety ────────────────────────────────────────────────────────────
    de = f.get("debtToEquity")
    if de is not None:
        de_ratio = de / 100 if de > 10 else de     # yfinance returns % sometimes
        if de_ratio < T["de_low"]:
            add("Debt/Equity", de_ratio, 2, f"Very low leverage (D/E={de_ratio:.2f})", True)
        elif de_ratio < T["de_high"]:
            add("Debt/Equity", de_ratio, 1, f"Manageable debt (D/E={de_ratio:.2f})", True)
        else:
            add("Debt/Equity", de_ratio, -2, f"High leverage risk (D/E={de_ratio:.2f})", False)

    cr = f.get("currentRatio")
    if cr is not None:
        if cr > T["current_ratio_good"]:
            add("Current Ratio", cr, 2, f"Strong liquidity (current ratio={cr:.1f})", True)
        elif cr > 1.0:
            add("Current Ratio", cr, 1, f"Adequate liquidity (current ratio={cr:.1f})", True)
        else:
            add("Current Ratio", cr, -1, f"Liquidity concern (current ratio={cr:.1f} < 1)", False)

    # ── Dividends ─────────────────────────────────────────────────────────
    div_yield = f.get("dividendYield")
    if div_yield is not None and div_yield > 0:
        if div_yield > 0.03:
            add("Dividend Yield", div_yield, 2, f"Attractive dividend yield: {div_yield*100:.1f}%", True)
        else:
            add("Dividend Yield", div_yield, 1, f"Modest dividend yield: {div_yield*100:.1f}%", True)

    # ── Grade ─────────────────────────────────────────────────────────────
    pct = result.score_pct
    if pct >= 75:
        result.grade = "A  (Strong Buy)"
    elif pct >= 55:
        result.grade = "B  (Buy)"
    elif pct >= 35:
        result.grade = "C  (Neutral)"
    elif pct >= 15:
        result.grade = "D  (Caution)"
    else:
        result.grade = "F  (Avoid)"

    # ── Narrative summary ─────────────────────────────────────────────────
    pos = len(result.positive_signals)
    neg = len(result.negative_signals)
    result.summary = (
        f"Fundamental score: {result.score}/{result.max_score} ({pct}%) — Grade {result.grade}. "
        f"{pos} positive factors, {neg} risk factors identified."
    )

    return result
