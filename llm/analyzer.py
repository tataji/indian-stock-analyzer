"""
llm/analyzer.py
LLM-powered analysis layer using Claude.
Converts ML signals + technical data into human-readable research reports.
"""

from __future__ import annotations

import logging
from typing import Optional

import anthropic
import pandas as pd

from config.settings import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from models.fundamental import FundamentalResult

logger = logging.getLogger(__name__)


def _client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Please add it to your .env file."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _latest_bar_summary(df: pd.DataFrame) -> str:
    """Format the most recent bar's key indicators as a text block."""
    row = df.iloc[-1]

    def _g(col, fmt=".2f"):
        val = row.get(col)
        if val is None or pd.isna(val):
            return "N/A"
        try:
            return f"{val:{fmt}}"
        except Exception:
            return str(val)

    lines = [
        f"Price (Close)      : ₹{_g('Close')}",
        f"Open               : ₹{_g('Open')}",
        f"High               : ₹{_g('High')}",
        f"Low                : ₹{_g('Low')}",
        f"Volume vs 20d avg  : {_g('Vol_ratio')}x",
        f"RSI (14)           : {_g('RSI')}",
        f"MACD Histogram     : {_g('MACD_hist')}",
        f"Stochastic K/D     : {_g('Stoch_K')} / {_g('Stoch_D')}",
        f"CCI (20)           : {_g('CCI')}",
        f"Williams %R        : {_g('Williams_R')}",
        f"ADX                : {_g('ADX')}",
        f"ATR %              : {_g('ATR_pct', '.4f')}",
        f"BB %B              : {_g('BB_pct')}",
        f"BB Squeeze         : {'Yes' if row.get('BB_squeeze') == 1 else 'No'}",
        f"Price vs SMA20     : {_g('price_vs_sma20', '.2%')}",
        f"Price vs SMA50     : {_g('price_vs_sma50', '.2%')}",
        f"Price vs EMA200    : {_g('price_vs_ema200', '.2%')}",
        f"52w High           : ₹{_g('52w_high')}",
        f"52w Low            : ₹{_g('52w_low')}",
        f"% from 52w High    : {_g('pct_from_52w_high', '.2%')}",
        f"Golden Cross       : {'Yes' if row.get('golden_cross') == 1 else 'No'}",
        f"OBV Trend          : {'Bullish' if row.get('OBV_trend') == 1 else 'Bearish'}",
        f"HV 20d             : {_g('HV_20', '.2%')}",
    ]
    return "\n".join(lines)


def _fundamental_summary(fund_result: FundamentalResult) -> str:
    pos = "\n".join(f"  ✅ {s.message}" for s in fund_result.positive_signals)
    neg = "\n".join(f"  ⚠️  {s.message}" for s in fund_result.negative_signals)
    return (
        f"Score: {fund_result.score}/{fund_result.max_score} ({fund_result.score_pct}%) | "
        f"Grade: {fund_result.grade}\n"
        f"Positives:\n{pos or '  None identified'}\n"
        f"Risks:\n{neg or '  None identified'}"
    )


def generate_analysis(
    symbol:        str,
    df:            pd.DataFrame,
    tech_signal:   dict,
    fund_result:   FundamentalResult,
    fundamentals:  dict,
    company_info:  dict,
) -> str:
    """
    Generate a full stock research report using Claude.

    Parameters
    ----------
    symbol       : NSE ticker symbol
    df           : OHLCV + indicator DataFrame (post-preprocessing)
    tech_signal  : output from TechnicalModel.predict()
    fund_result  : output from score_fundamentals()
    fundamentals : raw fundamentals dict from fetcher
    company_info : company meta dict from get_company_info()
    """
    client = _client()

    bar_text  = _latest_bar_summary(df)
    fund_text = _fundamental_summary(fund_result)

    market_cap = fundamentals.get("marketCap")
    cap_str = f"₹{market_cap/1e9:.1f}B" if market_cap else "N/A"

    prompt = f"""You are a senior equity research analyst specializing in Indian stock markets (NSE/BSE).
Provide a structured, professional stock analysis report. Be specific, data-driven, and actionable.

════════════════════════════════════════
COMPANY OVERVIEW
════════════════════════════════════════
Symbol      : {symbol} (NSE)
Company     : {company_info.get('name', 'N/A')}
Sector      : {company_info.get('sector', 'N/A')}
Industry    : {company_info.get('industry', 'N/A')}
Market Cap  : {cap_str}
Beta        : {fundamentals.get('beta', 'N/A')}

════════════════════════════════════════
TECHNICAL INDICATORS (Latest Bar)
════════════════════════════════════════
{bar_text}

ML MODEL SIGNAL
---------------
Direction    : {tech_signal['signal']}
Confidence   : {tech_signal['confidence']:.1%}
Prob Bullish : {tech_signal['proba_bull']:.1%}
Prob Bearish : {tech_signal['proba_bear']:.1%}

════════════════════════════════════════
FUNDAMENTAL ANALYSIS
════════════════════════════════════════
{fund_text}

════════════════════════════════════════
YOUR ANALYSIS REPORT
════════════════════════════════════════
Please write a professional research note with the following sections:

1. **Executive Summary** (2-3 sentences covering the overall thesis)

2. **Technical Setup**
   - Current trend (using EMA, ADX, golden cross)
   - Momentum assessment (RSI, MACD, Stochastic)
   - Key support and resistance levels (based on BB, 52w data)
   - Volume confirmation

3. **Fundamental Assessment**
   - Valuation commentary (P/E, P/B vs sector norms)
   - Profitability & quality
   - Growth trajectory
   - Balance sheet strength

4. **Key Catalysts & Risks**
   - 2-3 potential upside catalysts
   - 2-3 key risks to monitor

5. **Overall View**
   - Verdict: Bullish / Neutral / Bearish (with conviction level: High / Medium / Low)
   - Short-term outlook (1-2 weeks)
   - Medium-term outlook (1-3 months)
   - Suggested entry zone / stop-loss levels (price ranges based on technical data)

⚠️ Disclaimer: This analysis is for educational purposes only and does not constitute SEBI-registered investment advice.
"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        logger.error("LLM analysis failed: %s", exc)
        return f"⚠️ Analysis generation failed: {exc}"


def generate_comparison(
    symbols:      list[str],
    results:      list[dict],
) -> str:
    """Generate a comparative analysis across multiple stocks."""
    client = _client()

    rows = []
    for r in results:
        rows.append(
            f"- {r['symbol']}: Tech={r['tech_signal']}, "
            f"Confidence={r['confidence']:.0%}, "
            f"Fund Score={r['fund_score']}/{r['fund_max']}, "
            f"Grade={r['fund_grade']}"
        )

    prompt = f"""You are a senior Indian equity research analyst.

Compare the following stocks based on their technical and fundamental scores:

{chr(10).join(rows)}

Provide:
1. A ranked recommendation (best to worst) with brief reasoning
2. Sector/thematic commentary if applicable
3. A suggested portfolio allocation idea if an investor wanted to pick 1-2 of these

Be concise and actionable. Add a standard disclaimer at the end.
"""
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        return f"⚠️ Comparison generation failed: {exc}"


def generate_market_summary(sector_scores: dict) -> str:
    """Generate a brief market breadth / sector rotation summary."""
    client = _client()

    lines = [f"- {sec}: avg score {scores['avg']:.1f}/18, avg conf {scores['avg_conf']:.0%}"
             for sec, scores in sector_scores.items()]

    prompt = f"""You are an Indian equity market strategist.

Based on the following sector-level fundamental and technical scores, provide:
1. Top 2 sectors to overweight and why
2. Sectors to underweight / avoid
3. A 2-sentence overall market sentiment comment

Sector Data:
{chr(10).join(lines)}

Be concise. Add a standard disclaimer.
"""
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        return f"⚠️ Market summary failed: {exc}"
