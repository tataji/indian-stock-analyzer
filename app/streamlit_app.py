"""
app/streamlit_app.py
Main Streamlit UI for Indian Stock Analyzer.
Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time

import pandas as pd
import streamlit as st

from config.settings import NIFTY50_SYMBOLS, SECTOR_MAP, ANTHROPIC_API_KEY
from data.fetcher import fetch_ohlcv, fetch_fundamentals, get_company_info
from data.preprocessor import build_features, add_technical_indicators
from models.technical import TechnicalModel
from models.fundamental import score_fundamentals
from llm.analyzer import generate_analysis, generate_comparison, generate_market_summary
from utils.charts import (
    candlestick_chart, fundamental_radar,
    score_gauge, confidence_bar, returns_histogram,
)
from utils.screener import run_screener, filter_screener

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252b3b 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #2d3348;
        margin-bottom: 8px;
    }
    .signal-bull { color: #00C851; font-weight: 700; font-size: 1.2em; }
    .signal-bear { color: #FF4444; font-weight: 700; font-size: 1.2em; }
    .signal-neutral { color: #FFBB33; font-weight: 700; font-size: 1.2em; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; padding: 10px 20px; }
    div[data-testid="stMetric"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #2d3348;
    }
    .section-header {
        background: linear-gradient(90deg, #1565C0, #0D47A1);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 1.05em;
        font-weight: 600;
    }
    .analysis-box {
        background: #1a1f2e;
        border-left: 4px solid #2196F3;
        border-radius: 6px;
        padding: 16px 20px;
        font-size: 0.95em;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/47/NSE_logo.svg",
             width=180, use_column_width=False)
    st.markdown("## Indian Stock Analyzer")
    st.caption("Technical + Fundamental + LLM Analysis")
    st.divider()

    mode = st.selectbox(
        "📋 Mode",
        ["Single Stock Analysis", "Stock Screener", "Sector Comparison"],
        index=0,
    )
    st.divider()

    if mode == "Single Stock Analysis":
        symbol = st.text_input(
            "NSE Symbol",
            value="RELIANCE",
            placeholder="e.g. TCS, INFY, HDFCBANK",
        ).upper().strip()
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "3y"], index=2)
        n_bars = st.slider("Chart Bars", 60, 252, 120)
        run_llm = st.toggle("🤖 Generate AI Analysis", value=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    elif mode == "Stock Screener":
        sector_filter = st.multiselect(
            "Filter by Sector", list(SECTOR_MAP.keys()), default=[]
        )
        signal_filter = st.selectbox("Signal Filter", ["ALL", "BULLISH", "BEARISH"])
        min_rsi  = st.slider("Min RSI", 0, 100, 30)
        max_rsi  = st.slider("Max RSI", 0, 100, 70)
        min_fund = st.slider("Min Fundamental Score", 0, 18, 6)
        min_adx  = st.slider("Min ADX (Trend Strength)", 0, 60, 15)
        top_n    = st.slider("Top N Results", 5, 50, 20)
        screen_btn = st.button("🔎 Run Screener", type="primary", use_container_width=True)

    elif mode == "Sector Comparison":
        selected_sectors = st.multiselect(
            "Select Sectors", list(SECTOR_MAP.keys()),
            default=list(SECTOR_MAP.keys())[:4],
        )
        compare_btn = st.button("📊 Compare Sectors", type="primary", use_container_width=True)

    st.divider()
    if not ANTHROPIC_API_KEY:
        st.warning("⚠️ No ANTHROPIC_API_KEY found.\nSet it in `.env` to enable AI analysis.")
    else:
        st.success("✅ Anthropic API connected")

    st.caption("Data: Yahoo Finance | Models: Sklearn Ensemble | LLM: Claude")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_ohlcv(symbol: str, period: str) -> pd.DataFrame | None:
    return fetch_ohlcv(symbol, period=period)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(symbol: str) -> dict:
    return fetch_fundamentals(symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_company_info(symbol: str) -> dict:
    return get_company_info(symbol)


def _signal_badge(signal: str) -> str:
    css = "signal-bull" if signal == "BULLISH" else (
          "signal-bear" if signal == "BEARISH" else "signal-neutral")
    icon = "📈" if signal == "BULLISH" else ("📉" if signal == "BEARISH" else "➡️")
    return f'<span class="{css}">{icon} {signal}</span>'


def _delta_color(val):
    if val is None:
        return None
    return "normal" if val >= 0 else "inverse"


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: SINGLE STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_single_analysis(symbol: str, period: str, n_bars: int, run_llm: bool):
    st.markdown(f"## 📊 Analysis: **{symbol}**")

    with st.spinner(f"Fetching data for {symbol}…"):
        df_raw = cached_ohlcv(symbol, period)

    if df_raw is None or df_raw.empty:
        st.error(f"❌ Could not fetch data for **{symbol}**. Check the symbol and try again.")
        return

    with st.spinner("Engineering features…"):
        df_feat = build_features(df_raw, include_labels=False)
        df_ind  = add_technical_indicators(df_raw)

    if df_feat.empty:
        st.error("Feature engineering returned empty DataFrame.")
        return

    with st.spinner("Running ML model…"):
        model = TechnicalModel(symbol)
        if not model.load():
            try:
                model.train(df_raw)
                model.save()
            except Exception as e:
                st.warning(f"Model training failed: {e}. Showing technical data only.")
                tech_signal = {"signal": "N/A", "confidence": 0.0, "proba_bull": 0.5, "proba_bear": 0.5}
            else:
                tech_signal = model.predict(df_raw)
        else:
            tech_signal = model.predict(df_raw)

    with st.spinner("Scoring fundamentals…"):
        fundamentals = cached_fundamentals(symbol)
        fund_result  = score_fundamentals(fundamentals)
        company_info = cached_company_info(symbol)

    # ── Company Header ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0">🏢 {company_info.get('name', symbol)}</h3>
        <span style="color:#9e9e9e">{company_info.get('sector','N/A')} · {company_info.get('industry','N/A')}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────
    row = df_ind.iloc[-1]
    prev_row = df_ind.iloc[-2] if len(df_ind) > 1 else row

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    close  = float(row["Close"])
    prev   = float(prev_row["Close"])
    change = (close - prev) / prev * 100

    c1.metric("📈 Close",    f"₹{close:,.2f}",  f"{change:+.2f}%")
    c2.metric("RSI (14)",   f"{row.get('RSI', 0):.1f}")
    c3.metric("ADX",        f"{row.get('ADX', 0):.1f}")
    c4.metric("BB %B",      f"{row.get('BB_pct', 0):.2f}")
    c5.metric("Vol Ratio",  f"{row.get('Vol_ratio', 1):.2f}x")
    c6.metric("HV 20d",     f"{row.get('HV_20', 0)*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score Row ─────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.plotly_chart(score_gauge(fund_result.score, fund_result.max_score, "Fundamental Score"),
                        use_container_width=True)
    with sc2:
        st.plotly_chart(score_gauge(int(tech_signal["proba_bull"] * 10), 10, "Technical Score"),
                        use_container_width=True)
    with sc3:
        composite = round(0.45 * tech_signal["proba_bull"] * 10 +
                          0.55 * (fund_result.score / fund_result.max_score * 10), 2)
        st.plotly_chart(score_gauge(int(composite * 10), 100, "Composite Score"),
                        use_container_width=True)

    # ── ML Signal Banner ──────────────────────────────────────────────────
    if tech_signal["signal"] != "N/A":
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
            ML Signal: {_signal_badge(tech_signal['signal'])}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Confidence: <strong>{tech_signal['confidence']:.1%}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            P(Bull): <strong style="color:#00C851">{tech_signal['proba_bull']:.1%}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            P(Bear): <strong style="color:#FF4444">{tech_signal['proba_bear']:.1%}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(confidence_bar(tech_signal), use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Chart", "🔢 Technicals", "📊 Fundamentals", "🤖 AI Analysis", "📈 Stats"
    ])

    with tab1:
        st.plotly_chart(candlestick_chart(df_ind, symbol, n_bars=n_bars),
                        use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">📐 Technical Indicator Values</div>',
                    unsafe_allow_html=True)
        indicator_data = {
            "Indicator": [
                "RSI (14)", "MACD", "MACD Signal", "MACD Histogram",
                "Stochastic K", "Stochastic D", "CCI (20)", "Williams %R",
                "ADX", "DI+", "DI-", "ATR %",
                "BB Upper", "BB Lower", "BB %B", "BB Width",
                "EMA 20", "EMA 50", "EMA 200",
                "SMA 20", "SMA 50",
                "52w High", "52w Low", "% from 52w High",
                "OBV Trend", "Volume Ratio",
            ],
            "Value": [
                f"{row.get('RSI', 0):.2f}",
                f"{row.get('MACD', 0):.4f}",
                f"{row.get('MACD_signal', 0):.4f}",
                f"{row.get('MACD_hist', 0):.4f}",
                f"{row.get('Stoch_K', 0):.2f}",
                f"{row.get('Stoch_D', 0):.2f}",
                f"{row.get('CCI', 0):.2f}",
                f"{row.get('Williams_R', 0):.2f}",
                f"{row.get('ADX', 0):.2f}",
                f"{row.get('DI_plus', 0):.2f}",
                f"{row.get('DI_minus', 0):.2f}",
                f"{row.get('ATR_pct', 0)*100:.3f}%",
                f"₹{row.get('BB_upper', 0):.2f}",
                f"₹{row.get('BB_lower', 0):.2f}",
                f"{row.get('BB_pct', 0):.3f}",
                f"{row.get('BB_width', 0):.4f}",
                f"₹{row.get('EMA_20', 0):.2f}",
                f"₹{row.get('EMA_50', 0):.2f}",
                f"₹{row.get('EMA_200', 0):.2f}",
                f"₹{row.get('SMA_20', 0):.2f}",
                f"₹{row.get('SMA_50', 0):.2f}",
                f"₹{row.get('52w_high', 0):.2f}",
                f"₹{row.get('52w_low', 0):.2f}",
                f"{row.get('pct_from_52w_high', 0)*100:.2f}%",
                "Bullish 🟢" if row.get("OBV_trend") == 1 else "Bearish 🔴",
                f"{row.get('Vol_ratio', 1):.2f}x",
            ],
            "Signal": [
                "🟢 Oversold" if row.get("RSI", 50) < 35 else (
                    "🔴 Overbought" if row.get("RSI", 50) > 65 else "🟡 Neutral"),
                "🟢 Bullish" if row.get("MACD_bullish", 0) == 1 else "🔴 Bearish",
                "-", "-",
                "🟢 Oversold" if row.get("Stoch_K", 50) < 20 else (
                    "🔴 Overbought" if row.get("Stoch_K", 50) > 80 else "🟡 Neutral"),
                "-",
                "🟢 Oversold" if row.get("CCI", 0) < -100 else (
                    "🔴 Overbought" if row.get("CCI", 0) > 100 else "🟡 Neutral"),
                "🟢 Oversold" if row.get("Williams_R", -50) < -80 else (
                    "🔴 Overbought" if row.get("Williams_R", -50) > -20 else "🟡 Neutral"),
                "🟢 Strong Trend" if row.get("ADX", 0) > 25 else "🟡 Weak Trend",
                "-", "-", "-", "-", "-",
                "🟢 Oversold" if row.get("BB_pct", 0.5) < 0.2 else (
                    "🔴 Overbought" if row.get("BB_pct", 0.5) > 0.8 else "🟡 Neutral"),
                "🔴 Squeeze" if row.get("BB_squeeze", 0) == 1 else "🟢 Normal",
                "🟢 Above EMA20" if close > row.get("EMA_20", 0) else "🔴 Below EMA20",
                "🟢 Above EMA50" if close > row.get("EMA_50", 0) else "🔴 Below EMA50",
                "🟢 Bullish" if close > row.get("EMA_200", 0) else "🔴 Bearish",
                "-", "-", "-", "-",
                "🟢 Near 52w High" if row.get("pct_from_52w_high", -1) > -0.05 else "🟡 Below 52w High",
                "-",
                "🟢 High Volume" if row.get("Vol_ratio", 1) > 1.5 else "🟡 Normal Volume",
            ],
        }
        st.dataframe(
            pd.DataFrame(indicator_data),
            use_container_width=True,
            hide_index=True,
            height=700,
        )

        # Feature importance
        try:
            fi = model.feature_importance(top_n=15)
            if not fi.empty:
                st.markdown('<div class="section-header">🏆 Top Feature Importances (ML Model)</div>',
                            unsafe_allow_html=True)
                st.bar_chart(fi.set_index("feature")["importance"])
        except Exception:
            pass

    with tab3:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown(f"""
            **Fundamental Score:** {fund_result.score}/{fund_result.max_score}  
            **Grade:** `{fund_result.grade}`  
            **Score %:** {fund_result.score_pct}%
            """)

            st.markdown("**✅ Positive Signals**")
            for sig in fund_result.positive_signals:
                st.success(sig.message)

            st.markdown("**⚠️ Risk Signals**")
            for sig in fund_result.negative_signals:
                st.error(sig.message)

        with col_r:
            st.plotly_chart(fundamental_radar(fund_result), use_container_width=True)

        # Raw fundamentals table
        st.markdown('<div class="section-header">📋 Raw Fundamental Data</div>',
                    unsafe_allow_html=True)
        raw_data = {
            "Metric": [
                "Trailing P/E", "Forward P/E", "Price/Book",
                "Return on Equity", "Return on Assets", "Profit Margin",
                "Operating Margin", "Gross Margin",
                "Revenue Growth", "Earnings Growth",
                "Debt/Equity", "Current Ratio", "Quick Ratio",
                "Market Cap", "Free Cashflow", "Dividend Yield", "Beta",
                "EPS (TTM)", "EPS Forward",
            ],
            "Value": [
                f"{fundamentals.get('trailingPE', 'N/A'):.2f}" if isinstance(fundamentals.get('trailingPE'), float) else "N/A",
                f"{fundamentals.get('forwardPE', 'N/A'):.2f}" if isinstance(fundamentals.get('forwardPE'), float) else "N/A",
                f"{fundamentals.get('priceToBook', 'N/A'):.2f}" if isinstance(fundamentals.get('priceToBook'), float) else "N/A",
                f"{fundamentals.get('returnOnEquity', 0)*100:.1f}%" if isinstance(fundamentals.get('returnOnEquity'), float) else "N/A",
                f"{fundamentals.get('returnOnAssets', 0)*100:.1f}%" if isinstance(fundamentals.get('returnOnAssets'), float) else "N/A",
                f"{fundamentals.get('profitMargins', 0)*100:.1f}%" if isinstance(fundamentals.get('profitMargins'), float) else "N/A",
                f"{fundamentals.get('operatingMargins', 0)*100:.1f}%" if isinstance(fundamentals.get('operatingMargins'), float) else "N/A",
                f"{fundamentals.get('grossMargins', 0)*100:.1f}%" if isinstance(fundamentals.get('grossMargins'), float) else "N/A",
                f"{fundamentals.get('revenueGrowth', 0)*100:.1f}%" if isinstance(fundamentals.get('revenueGrowth'), float) else "N/A",
                f"{fundamentals.get('earningsGrowth', 0)*100:.1f}%" if isinstance(fundamentals.get('earningsGrowth'), float) else "N/A",
                f"{fundamentals.get('debtToEquity', 'N/A'):.2f}" if isinstance(fundamentals.get('debtToEquity'), float) else "N/A",
                f"{fundamentals.get('currentRatio', 'N/A'):.2f}" if isinstance(fundamentals.get('currentRatio'), float) else "N/A",
                f"{fundamentals.get('quickRatio', 'N/A'):.2f}" if isinstance(fundamentals.get('quickRatio'), float) else "N/A",
                f"₹{fundamentals.get('marketCap', 0)/1e9:.1f}B" if isinstance(fundamentals.get('marketCap'), (int, float)) else "N/A",
                f"₹{fundamentals.get('freeCashflow', 0)/1e9:.1f}B" if isinstance(fundamentals.get('freeCashflow'), (int, float)) else "N/A",
                f"{fundamentals.get('dividendYield', 0)*100:.2f}%" if isinstance(fundamentals.get('dividendYield'), float) else "N/A",
                f"{fundamentals.get('beta', 'N/A'):.2f}" if isinstance(fundamentals.get('beta'), float) else "N/A",
                f"₹{fundamentals.get('trailingEps', 'N/A'):.2f}" if isinstance(fundamentals.get('trailingEps'), float) else "N/A",
                f"₹{fundamentals.get('forwardEps', 'N/A'):.2f}" if isinstance(fundamentals.get('forwardEps'), float) else "N/A",
            ],
        }
        st.dataframe(pd.DataFrame(raw_data), use_container_width=True, hide_index=True)

    with tab4:
        if not ANTHROPIC_API_KEY:
            st.warning("⚠️ Please set ANTHROPIC_API_KEY in your .env file to use AI analysis.")
        elif not run_llm:
            st.info("AI Analysis is toggled off. Enable it in the sidebar.")
        else:
            with st.spinner("🤖 Claude is generating your analysis…"):
                try:
                    report = generate_analysis(
                        symbol=symbol,
                        df=df_ind,
                        tech_signal=tech_signal,
                        fund_result=fund_result,
                        fundamentals=fundamentals,
                        company_info=company_info,
                    )
                    st.markdown(f'<div class="analysis-box">{report}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    with tab5:
        st.plotly_chart(returns_histogram(df_raw), use_container_width=True)

        # Stats table
        returns = df_raw["Close"].pct_change().dropna()
        stats = {
            "Total Return (period)":        f"{(df_raw['Close'].iloc[-1]/df_raw['Close'].iloc[0]-1)*100:.2f}%",
            "Annualised Volatility":         f"{returns.std()*252**0.5*100:.2f}%",
            "Max Drawdown":                  f"{((df_raw['Close']/df_raw['Close'].cummax())-1).min()*100:.2f}%",
            "Sharpe Ratio (approx)":         f"{(returns.mean()/returns.std()*252**0.5):.3f}",
            "Positive Days":                 f"{(returns>0).sum()} ({(returns>0).mean()*100:.1f}%)",
            "Average Daily Return":          f"{returns.mean()*100:.4f}%",
            "Best Day":                      f"{returns.max()*100:.2f}%",
            "Worst Day":                     f"{returns.min()*100:.2f}%",
        }
        for k, v in stats.items():
            st.metric(k, v)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: SCREENER
# ══════════════════════════════════════════════════════════════════════════════

def run_stock_screener(sector_filter, signal_filter, min_rsi, max_rsi, min_fund, min_adx, top_n):
    st.markdown("## 🔎 Stock Screener — Nifty Universe")

    symbols = []
    if sector_filter:
        for sec in sector_filter:
            symbols.extend(SECTOR_MAP.get(sec, []))
    else:
        symbols = NIFTY50_SYMBOLS

    symbols = list(set(symbols))
    st.info(f"Screening {len(symbols)} stocks… This may take 2-3 minutes.")
    prog = st.progress(0, "Starting…")

    try:
        df_result = run_screener(symbols=symbols, max_workers=4, top_n=top_n * 2)
        prog.progress(100, "Done!")
    except Exception as e:
        st.error(f"Screener error: {e}")
        return

    if df_result.empty:
        st.warning("No results found. Try adjusting filters.")
        return

    # Apply filters
    df_filtered = filter_screener(
        df_result,
        signal=signal_filter,
        min_rsi=min_rsi,
        max_rsi=max_rsi,
        min_fund_score=min_fund,
        min_adx=min_adx,
    ).head(top_n)

    st.success(f"✅ Found {len(df_filtered)} stocks matching your criteria")

    # Display
    display_cols = {
        "symbol":      "Symbol",
        "close":       "Price (₹)",
        "tech_signal": "ML Signal",
        "tech_conf":   "Confidence",
        "rsi":         "RSI",
        "adx":         "ADX",
        "fund_score":  "Fund Score",
        "fund_grade":  "Grade",
        "composite":   "Composite",
        "pe":          "P/E",
        "roe":         "ROE",
    }
    df_disp = df_filtered[list(display_cols.keys())].rename(columns=display_cols)

    def color_signal(val):
        if val == "BULLISH":
            return "background-color: rgba(0,200,81,0.2); color: #00C851"
        elif val == "BEARISH":
            return "background-color: rgba(255,68,68,0.2); color: #FF4444"
        return ""

    styled = df_disp.style.map(color_signal, subset=["ML Signal"])
    st.dataframe(styled, use_container_width=True, height=500)

    # Compare with LLM
    if ANTHROPIC_API_KEY and len(df_filtered) >= 2:
        if st.button("🤖 Get AI Comparison for Top Results"):
            top5 = df_filtered.head(5)
            comp_data = [
                {
                    "symbol":    r["symbol"],
                    "tech_signal": r["tech_signal"],
                    "confidence": r["tech_conf"],
                    "fund_score": r["fund_score"],
                    "fund_max":   r["fund_max"],
                    "fund_grade": r["fund_grade"],
                }
                for _, r in top5.iterrows()
            ]
            with st.spinner("Generating comparison…"):
                comparison = generate_comparison(top5["symbol"].tolist(), comp_data)
            st.markdown(f'<div class="analysis-box">{comparison}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: SECTOR COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def run_sector_comparison(selected_sectors):
    st.markdown("## 📊 Sector Comparison")

    if not selected_sectors:
        st.warning("Please select at least one sector.")
        return

    sector_scores = {}
    progress = st.progress(0)

    for i, sector in enumerate(selected_sectors):
        symbols = SECTOR_MAP.get(sector, [])
        bull_confs, fund_scores = [], []

        for sym in symbols:
            try:
                df_raw = fetch_ohlcv(sym, period="6mo")
                if df_raw is None or len(df_raw) < 60:
                    continue

                model = TechnicalModel(sym)
                if model.load():
                    sig = model.predict(df_raw)
                    bull_confs.append(sig["proba_bull"])

                fund = score_fundamentals(fetch_fundamentals(sym))
                fund_scores.append(fund.score)

            except Exception:
                pass

        sector_scores[sector] = {
            "avg":      round(sum(fund_scores) / len(fund_scores), 2) if fund_scores else 0,
            "avg_conf": round(sum(bull_confs)  / len(bull_confs),  3) if bull_confs  else 0.5,
            "n":        len(symbols),
        }
        progress.progress((i + 1) / len(selected_sectors))

    # Display sector table
    rows = [
        {
            "Sector":            sec,
            "Avg Fund Score":    v["avg"],
            "Avg Bull Prob":     f"{v['avg_conf']:.1%}",
            "Stocks Covered":    v["n"],
            "Outlook":           "🟢 Bullish" if v["avg_conf"] > 0.55 else (
                                 "🔴 Bearish" if v["avg_conf"] < 0.45 else "🟡 Neutral"),
        }
        for sec, v in sorted(sector_scores.items(), key=lambda x: x[1]["avg"], reverse=True)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # LLM Market Summary
    if ANTHROPIC_API_KEY:
        if st.button("🤖 Get AI Market Summary"):
            with st.spinner("Generating market summary…"):
                summary = generate_market_summary(sector_scores)
            st.markdown(f'<div class="analysis-box">{summary}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTING
# ══════════════════════════════════════════════════════════════════════════════

st.title("Indian Stock Analyzer")
st.caption("Powered by ML (Gradient Boosting Ensemble) + Claude LLM | Data: Yahoo Finance")
st.divider()

if mode == "Single Stock Analysis":
    if "analyze_btn" in dir() and analyze_btn:  # triggered by button
        run_single_analysis(symbol, period, n_bars, run_llm)
    else:
        # Auto-run on first load
        if st.session_state.get("last_symbol") != symbol:
            st.session_state["last_symbol"] = symbol
        run_single_analysis(symbol, period, n_bars, run_llm)

elif mode == "Stock Screener":
    if screen_btn:
        run_stock_screener(sector_filter, signal_filter, min_rsi, max_rsi, min_fund, min_adx, top_n)
    else:
        st.info("👈 Configure filters in the sidebar and click **Run Screener**.")

elif mode == "Sector Comparison":
    if compare_btn:
        run_sector_comparison(selected_sectors)
    else:
        st.info("👈 Select sectors in the sidebar and click **Compare Sectors**.")
