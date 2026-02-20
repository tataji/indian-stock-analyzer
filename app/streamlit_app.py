"""
app/streamlit_app.py  — v2.0 (Improved)
Indian Stock Analyzer — Full Dashboard
Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config.settings import NIFTY50_SYMBOLS, SECTOR_MAP, ANTHROPIC_API_KEY
from data.fetcher import fetch_ohlcv, fetch_fundamentals, get_company_info
from data.preprocessor import build_features, add_technical_indicators
from data.news import fetch_news, aggregate_sentiment
from models.technical import TechnicalModel
from models.fundamental import score_fundamentals
from models.backtester import run_backtest
from llm.analyzer import generate_analysis, generate_comparison, generate_market_summary
from utils.charts import (
    candlestick_chart, fundamental_radar, score_gauge,
    confidence_bar, returns_histogram,
    backtest_equity_curve, portfolio_pie, portfolio_pnl_bar,
)
from utils.screener import run_screener, filter_screener
from utils.portfolio import Portfolio

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background: #0a0e1a; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #21262d;
}
.kpi-card {
    background: linear-gradient(135deg, #161b27 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.badge-bull {
    background: rgba(0,200,81,0.15); color: #00C851;
    border: 1px solid #00C851; border-radius: 20px;
    padding: 4px 14px; font-weight: 700; font-size: 0.9em; display: inline-block;
}
.badge-bear {
    background: rgba(255,68,68,0.15); color: #FF4444;
    border: 1px solid #FF4444; border-radius: 20px;
    padding: 4px 14px; font-weight: 700; font-size: 0.9em; display: inline-block;
}
.badge-neutral {
    background: rgba(255,187,51,0.15); color: #FFBB33;
    border: 1px solid #FFBB33; border-radius: 20px;
    padding: 4px 14px; font-weight: 700; font-size: 0.9em; display: inline-block;
}
.section-title {
    font-size: 1.05em; font-weight: 600; color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px; margin: 20px 0 12px 0;
}
.analysis-box {
    background: #161b27; border-left: 3px solid #2196F3;
    border-radius: 8px; padding: 20px 24px;
    line-height: 1.8; font-size: 0.95em;
}
.news-card {
    background: #161b27; border: 1px solid #21262d;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
}
div[data-testid="stMetric"] {
    background: #161b27; border: 1px solid #21262d;
    border-radius: 10px; padding: 14px 16px;
}
.hero-header {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #0d47a1 100%);
    border-radius: 14px; padding: 24px 28px; margin-bottom: 20px;
    border: 1px solid #1976d2;
}
</style>
""", unsafe_allow_html=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_ohlcv(symbol, period):  return fetch_ohlcv(symbol, period=period)
@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(symbol):   return fetch_fundamentals(symbol)
@st.cache_data(ttl=3600, show_spinner=False)
def cached_company_info(symbol):   return get_company_info(symbol)
@st.cache_data(ttl=1800, show_spinner=False)
def cached_news(symbol, name):     return fetch_news(symbol, company_name=name, max_articles=12)


def _badge(signal: str) -> str:
    css  = {"BULLISH": "badge-bull", "BEARISH": "badge-bear"}.get(signal, "badge-neutral")
    icon = {"BULLISH": "▲", "BEARISH": "▼"}.get(signal, "—")
    return f'<span class="{css}">{icon} {signal}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0;">
        <h2 style="margin:0; color:#58a6ff;">📈 StockAnalyzer</h2>
        <p style="color:#8b949e; font-size:0.8em; margin:4px 0 0 0;">Indian Markets · ML + LLM</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    mode = st.radio("Navigation", [
        "🔍 Single Stock", "🔎 Screener", "📊 Sector View",
        "⏪ Backtest", "💼 Portfolio",
    ], label_visibility="collapsed")
    st.divider()

    if mode == "🔍 Single Stock":
        symbol   = st.text_input("NSE Symbol", "RELIANCE").upper().strip()
        period   = st.selectbox("Period", ["6mo","1y","2y","3y"], index=2)
        n_bars   = st.slider("Chart Bars", 60, 252, 120)
        run_llm  = st.toggle("🤖 AI Report", value=bool(ANTHROPIC_API_KEY))
        run_news = st.toggle("📰 News Sentiment", value=True)

    elif mode == "🔎 Screener":
        sec_filter = st.multiselect("Sectors", list(SECTOR_MAP.keys()))
        sig_filter = st.selectbox("Signal", ["ALL","BULLISH","BEARISH"])
        min_rsi    = st.slider("RSI ≥", 0, 100, 30)
        max_rsi    = st.slider("RSI ≤", 0, 100, 70)
        min_fund   = st.slider("Min Fund Score", 0, 18, 6)
        min_adx    = st.slider("Min ADX", 0, 60, 15)
        top_n      = st.slider("Top N", 5, 50, 20)
        screen_btn = st.button("Run Screener →", type="primary", use_container_width=True)

    elif mode == "📊 Sector View":
        sel_sectors = st.multiselect("Sectors", list(SECTOR_MAP.keys()),
                                     default=list(SECTOR_MAP.keys())[:5])
        sector_btn  = st.button("Compare →", type="primary", use_container_width=True)

    elif mode == "⏪ Backtest":
        bt_symbol   = st.text_input("Symbol", "RELIANCE").upper().strip()
        bt_period   = st.selectbox("Period", ["2y","3y","5y"], index=1)
        bt_capital  = st.number_input("Capital (₹)", value=100000, step=10000)
        bt_stop     = st.slider("Stop Loss %", 1, 20, 5)
        bt_tp       = st.slider("Take Profit %", 2, 30, 10)
        bt_conf     = st.slider("Min Confidence %", 50, 90, 60)
        bt_longonly = st.toggle("Long Only", value=True)
        bt_btn      = st.button("Run Backtest →", type="primary", use_container_width=True)

    elif mode == "💼 Portfolio":
        pf_tab = st.radio("Action", ["View","Add Holding","Remove Holding"],
                          label_visibility="collapsed")

    st.divider()
    st.caption("✅ API Connected" if ANTHROPIC_API_KEY else "⚠️ No ANTHROPIC_API_KEY")
    st.caption("Data: Yahoo Finance | ML: Sklearn | LLM: Claude")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE STOCK
# ══════════════════════════════════════════════════════════════════════════════
def page_single_stock(symbol, period, n_bars, run_llm, run_news):
    info = cached_company_info(symbol)
    st.markdown(f"""
    <div class="hero-header">
        <h2 style="margin:0;color:white;">{info.get('name', symbol)}</h2>
        <p style="color:#90caf9;margin:4px 0 0 0;">
            {symbol} · NSE &nbsp;|&nbsp; {info.get('sector','N/A')} &nbsp;|&nbsp; {info.get('industry','N/A')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching market data…"):
        df_raw = cached_ohlcv(symbol, period)
    if df_raw is None or df_raw.empty:
        st.error(f"No data for **{symbol}**. Check the symbol.")
        return

    with st.spinner("Computing indicators…"):
        df_ind  = add_technical_indicators(df_raw)

    with st.spinner("Running ML model…"):
        model = TechnicalModel(symbol)
        if not model.load():
            try:
                model.train(df_raw); model.save()
            except Exception as e:
                st.warning(f"ML failed: {e}")
                tech_signal = {"signal":"N/A","confidence":0,"proba_bull":0.5,"proba_bear":0.5}
            else:
                tech_signal = model.predict(df_raw)
        else:
            tech_signal = model.predict(df_raw)

    fundamentals = cached_fundamentals(symbol)
    fund_result  = score_fundamentals(fundamentals)

    row   = df_ind.iloc[-1]
    prev  = df_ind.iloc[-2] if len(df_ind) > 1 else row
    close = float(row["Close"])
    chg   = (close - float(prev["Close"])) / float(prev["Close"]) * 100

    # KPIs
    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("Close",       f"₹{close:,.2f}", f"{chg:+.2f}%")
    k2.metric("RSI (14)",    f"{row.get('RSI',0):.1f}")
    k3.metric("ADX",         f"{row.get('ADX',0):.1f}")
    k4.metric("MACD",        "Bullish 🟢" if row.get("MACD_bullish",0)==1 else "Bearish 🔴")
    k5.metric("BB %B",       f"{row.get('BB_pct',0):.2f}")
    k6.metric("Vol Ratio",   f"{row.get('Vol_ratio',1):.2f}x")
    k7.metric("Fund Score",  f"{fund_result.score}/{fund_result.max_score}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Score gauges
    sc1,sc2,sc3 = st.columns(3)
    with sc1: st.plotly_chart(score_gauge(fund_result.score, fund_result.max_score, "Fundamental"), use_container_width=True)
    with sc2: st.plotly_chart(score_gauge(int(tech_signal["proba_bull"]*10), 10, "Technical"), use_container_width=True)
    with sc3:
        comp = int((0.45*tech_signal["proba_bull"]*10 + 0.55*fund_result.score/18*10)*10)
        st.plotly_chart(score_gauge(comp, 100, "Composite"), use_container_width=True)

    if tech_signal["signal"] != "N/A":
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;padding:14px">
            Signal: {_badge(tech_signal['signal'])}
            &nbsp;·&nbsp; Confidence: <strong>{tech_signal['confidence']:.1%}</strong>
            &nbsp;·&nbsp; P(Bull):<strong style="color:#00C851"> {tech_signal['proba_bull']:.1%}</strong>
            &nbsp;·&nbsp; P(Bear):<strong style="color:#FF4444"> {tech_signal['proba_bear']:.1%}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(confidence_bar(tech_signal), use_container_width=True)

    tabs = st.tabs(["📉 Chart","🔢 Indicators","📊 Fundamentals","📰 News","🤖 AI Report","📈 Stats"])

    # Chart
    with tabs[0]:
        st.plotly_chart(candlestick_chart(df_ind, symbol, n_bars), use_container_width=True)

    # Indicators
    with tabs[1]:
        st.markdown('<p class="section-title">Technical Indicator Values</p>', unsafe_allow_html=True)
        def _v(col, fmt=".2f"):
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
            return f"{val:{fmt}}"
        rows_table = [
            ("RSI (14)",        _v("RSI"),        "🟢 Oversold" if row.get("RSI",50)<35 else ("🔴 Overbought" if row.get("RSI",50)>65 else "🟡 Neutral")),
            ("MACD",            _v("MACD",".4f"), "🟢 Bullish" if row.get("MACD_bullish",0)==1 else "🔴 Bearish"),
            ("MACD Signal",     _v("MACD_signal",".4f"), "—"),
            ("MACD Histogram",  _v("MACD_hist",".4f"), "—"),
            ("Stochastic K",    _v("Stoch_K"),    "🟢 Oversold" if row.get("Stoch_K",50)<20 else ("🔴 Overbought" if row.get("Stoch_K",50)>80 else "🟡 Neutral")),
            ("Stochastic D",    _v("Stoch_D"),    "—"),
            ("CCI (20)",        _v("CCI"),        "🟢 Oversold" if row.get("CCI",0)<-100 else ("🔴 Overbought" if row.get("CCI",0)>100 else "🟡 Neutral")),
            ("Williams %R",     _v("Williams_R"), "🟢 Oversold" if row.get("Williams_R",-50)<-80 else ("🔴 Overbought" if row.get("Williams_R",-50)>-20 else "🟡 Neutral")),
            ("ADX",             _v("ADX"),        "🟢 Trending" if row.get("ADX",0)>25 else "🟡 Weak"),
            ("DI+ / DI-",       f"{_v('DI_plus')} / {_v('DI_minus')}", "—"),
            ("ATR %",           _v("ATR_pct",".3%"), "—"),
            ("BB Upper",        f"₹{_v('BB_upper')}", "—"),
            ("BB Lower",        f"₹{_v('BB_lower')}", "—"),
            ("BB %B",           _v("BB_pct"),     "🟢 Oversold" if row.get("BB_pct",0.5)<0.2 else ("🔴 Overbought" if row.get("BB_pct",0.5)>0.8 else "🟡 Neutral")),
            ("BB Squeeze",      "Yes 🔴" if row.get("BB_squeeze",0)==1 else "No 🟢", "Breakout likely" if row.get("BB_squeeze",0)==1 else "—"),
            ("EMA 20",          f"₹{_v('EMA_20')}", "🟢 Above" if close>row.get("EMA_20",0) else "🔴 Below"),
            ("EMA 50",          f"₹{_v('EMA_50')}", "🟢 Above" if close>row.get("EMA_50",0) else "🔴 Below"),
            ("EMA 200",         f"₹{_v('EMA_200')}", "🟢 Bull Trend" if close>row.get("EMA_200",0) else "🔴 Bear Trend"),
            ("Golden Cross",    "Yes 🟢" if row.get("golden_cross",0)==1 else "No", "—"),
            ("52w High",        f"₹{_v('52w_high')}", "—"),
            ("52w Low",         f"₹{_v('52w_low')}", "—"),
            ("% from 52w High", _v("pct_from_52w_high",".2%"), "—"),
            ("OBV Trend",       "Bullish 🟢" if row.get("OBV_trend",0)==1 else "Bearish 🔴", "—"),
            ("Volume Ratio",    _v("Vol_ratio"), "🟢 High" if row.get("Vol_ratio",1)>1.5 else "🟡 Normal"),
            ("HV 20d",          _v("HV_20",".2%"), "—"),
        ]
        st.dataframe(pd.DataFrame(rows_table, columns=["Indicator","Value","Signal"]),
                     use_container_width=True, hide_index=True, height=650)
        try:
            fi = model.feature_importance(15)
            if not fi.empty:
                st.markdown('<p class="section-title">Top ML Feature Importances</p>', unsafe_allow_html=True)
                st.bar_chart(fi.set_index("feature")["importance"])
        except Exception:
            pass

    # Fundamentals
    with tabs[2]:
        fl, fr = st.columns(2)
        with fl:
            st.markdown(f"""
            <div class="kpi-card">
                <h4 style="margin:0">Fundamental Score</h4>
                <h2 style="margin:4px 0;color:#58a6ff">{fund_result.score} / {fund_result.max_score}</h2>
                <p style="margin:0;color:#8b949e">Grade: <strong style="color:white">{fund_result.grade}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**✅ Positive Factors**")
            for s in fund_result.positive_signals: st.success(s.message)
            st.markdown("**⚠️ Risk Factors**")
            for s in fund_result.negative_signals: st.error(s.message)
        with fr:
            st.plotly_chart(fundamental_radar(fund_result), use_container_width=True)

        st.markdown('<p class="section-title">Raw Fundamental Data</p>', unsafe_allow_html=True)
        def _f(key, pct=False, crore=False, billion=False):
            v = fundamentals.get(key)
            if v is None: return "N/A"
            if pct:     return f"{v*100:.1f}%"
            if crore:   return f"₹{v/1e7:.1f}Cr"
            if billion: return f"₹{v/1e9:.1f}B"
            return f"{v:.2f}"
        raw_rows = [
            ("Trailing P/E",     _f("trailingPE")),
            ("Forward P/E",      _f("forwardPE")),
            ("Price/Book",       _f("priceToBook")),
            ("ROE",              _f("returnOnEquity", pct=True)),
            ("ROA",              _f("returnOnAssets", pct=True)),
            ("Profit Margin",    _f("profitMargins",  pct=True)),
            ("Oper. Margin",     _f("operatingMargins", pct=True)),
            ("Revenue Growth",   _f("revenueGrowth",  pct=True)),
            ("Earnings Growth",  _f("earningsGrowth", pct=True)),
            ("Debt/Equity",      _f("debtToEquity")),
            ("Current Ratio",    _f("currentRatio")),
            ("Free Cashflow",    _f("freeCashflow",   crore=True)),
            ("Market Cap",       _f("marketCap",      billion=True)),
            ("Dividend Yield",   _f("dividendYield",  pct=True)),
            ("Beta",             _f("beta")),
            ("EPS (TTM)",        _f("trailingEps")),
        ]
        st.dataframe(pd.DataFrame(raw_rows, columns=["Metric","Value"]),
                     use_container_width=True, hide_index=True)

    # News
    with tabs[3]:
        if not run_news:
            st.info("Enable 'News Sentiment' in the sidebar.")
        else:
            with st.spinner("Fetching news…"):
                articles = cached_news(symbol, info.get("name",""))
                agg      = aggregate_sentiment(articles)

            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Sentiment Score", f"{agg['overall_score']:+.2f}")
            s2.metric("Overall Mood",    agg["overall_label"])
            s3.metric("Positive",        f"🟢 {agg['positive_count']}")
            s4.metric("Negative",        f"🔴 {agg['negative_count']}")

            if agg["total"] > 0:
                pos_pct = agg["positive_count"] / agg["total"] * 100
                neg_pct = agg["negative_count"] / agg["total"] * 100
                neu_pct = agg["neutral_count"]  / agg["total"] * 100
                fig_s   = go.Figure()
                for lbl, val, col in [("Positive",pos_pct,"#00C851"),("Neutral",neu_pct,"#FFBB33"),("Negative",neg_pct,"#FF4444")]:
                    fig_s.add_trace(go.Bar(y=[""], x=[val], name=lbl, orientation="h",
                                           marker_color=col, text=[f"{lbl} {val:.0f}%"], textposition="inside"))
                fig_s.update_layout(barmode="stack", template="plotly_dark", height=90,
                                    margin=dict(l=10,r=10,t=10,b=10),
                                    xaxis=dict(range=[0,100],showticklabels=False), showlegend=False)
                st.plotly_chart(fig_s, use_container_width=True)

            st.markdown('<p class="section-title">Latest News</p>', unsafe_allow_html=True)
            if not articles:
                st.warning("No news found. Check internet connection.")
            for art in articles:
                sc = art["sentiment_score"]
                border = "#00C851" if sc>=0.3 else ("#FF4444" if sc<=-0.3 else "#FFBB33")
                st.markdown(f"""
                <div class="news-card" style="border-left:4px solid {border}">
                    <a href="{art['link']}" target="_blank" style="color:white;text-decoration:none;font-weight:600;">
                        {art['title']}
                    </a>
                    <p style="color:#8b949e;font-size:0.82em;margin:6px 0 0 0;">
                        {art['source']} · {art['published']} · Sentiment: <strong>{art['sentiment_label']}</strong> ({sc:+.2f})
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # AI Report
    with tabs[4]:
        if not ANTHROPIC_API_KEY:
            st.warning("Set ANTHROPIC_API_KEY in .env to enable AI reports.")
        elif not run_llm:
            st.info("Enable 'AI Report' in the sidebar.")
        else:
            with st.spinner("🤖 Claude is generating your research report…"):
                try:
                    report = generate_analysis(
                        symbol=symbol, df=df_ind,
                        tech_signal=tech_signal, fund_result=fund_result,
                        fundamentals=fundamentals, company_info=info,
                    )
                    st.markdown(f'<div class="analysis-box">{report}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed: {e}")

    # Stats
    with tabs[5]:
        st.plotly_chart(returns_histogram(df_raw), use_container_width=True)
        returns  = df_raw["Close"].pct_change().dropna()
        eq_curve = (1+returns).cumprod() * 100
        fig_eq   = go.Figure(go.Scatter(x=eq_curve.index, y=eq_curve.values, name="Growth of ₹100",
                                        line=dict(color="#2196F3", width=2),
                                        fill="tozeroy", fillcolor="rgba(33,150,243,0.1)"))
        fig_eq.update_layout(template="plotly_dark", height=300, title="Growth of ₹100",
                              margin=dict(l=40,r=20,t=50,b=30))
        st.plotly_chart(fig_eq, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            for k, v in {
                "Total Return":          f"{(df_raw['Close'].iloc[-1]/df_raw['Close'].iloc[0]-1)*100:.2f}%",
                "Annualised Volatility": f"{returns.std()*252**0.5*100:.2f}%",
                "Max Drawdown":          f"{((df_raw['Close']/df_raw['Close'].cummax())-1).min()*100:.2f}%",
                "Sharpe Ratio":          f"{(returns.mean()/returns.std()*252**0.5):.3f}",
                "Best Day":              f"{returns.max()*100:.2f}%",
                "Worst Day":             f"{returns.min()*100:.2f}%",
            }.items(): st.metric(k, v)
        with c2:
            for k, v in {
                "Positive Days": f"{(returns>0).mean()*100:.1f}%",
                "Total Bars":    str(len(df_raw)),
                "Avg Daily Ret": f"{returns.mean()*100:.4f}%",
                "Skewness":      f"{returns.skew():.3f}",
                "Kurtosis":      f"{returns.kurt():.3f}",
            }.items(): st.metric(k, v)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
def page_screener(sec_filter, sig_filter, min_rsi, max_rsi, min_fund, min_adx, top_n):
    st.markdown("## 🔎 Stock Screener")
    symbols = list(set(sum([SECTOR_MAP.get(s,[]) for s in (sec_filter or [])], []))) or NIFTY50_SYMBOLS
    st.info(f"Screening **{len(symbols)} stocks**… ~2-4 minutes")

    with st.spinner("Running…"):
        df_result = run_screener(symbols=symbols, max_workers=4, top_n=top_n*2)

    if df_result.empty:
        st.warning("No results. Adjust filters.")
        return

    df_f = filter_screener(df_result, signal=sig_filter, min_rsi=min_rsi,
                           max_rsi=max_rsi, min_fund_score=min_fund, min_adx=min_adx).head(top_n)
    st.success(f"✅ {len(df_f)} stocks found")

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Bullish",  f"🟢 {(df_f['tech_signal']=='BULLISH').sum()}")
    k2.metric("Bearish",  f"🔴 {(df_f['tech_signal']=='BEARISH').sum()}")
    k3.metric("Avg RSI",  f"{df_f['rsi'].mean():.1f}")
    k4.metric("Avg Fund", f"{df_f['fund_score'].mean():.1f}")

    disp = df_f[["symbol","close","tech_signal","tech_conf","rsi","adx","fund_score","fund_grade","composite","pe"]].copy()
    disp.columns = ["Symbol","Price","Signal","Conf","RSI","ADX","Fund Score","Grade","Composite","P/E"]

    def _cs(v):
        if v=="BULLISH": return "background-color:rgba(0,200,81,0.2);color:#00C851"
        if v=="BEARISH": return "background-color:rgba(255,68,68,0.2);color:#FF4444"
        return ""
    st.dataframe(disp.style.map(_cs, subset=["Signal"]), use_container_width=True, height=500)

    if ANTHROPIC_API_KEY and len(df_f) >= 2 and st.button("🤖 AI Compare Top 5"):
        top5 = df_f.head(5)
        comp_data = [{"symbol":r.symbol,"tech_signal":r.tech_signal,"confidence":r.tech_conf,
                      "fund_score":r.fund_score,"fund_max":r.fund_max,"fund_grade":r.fund_grade}
                     for r in top5.itertuples()]
        with st.spinner("Generating…"):
            cmp = generate_comparison(top5["symbol"].tolist(), comp_data)
        st.markdown(f'<div class="analysis-box">{cmp}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SECTOR VIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_sector_view(sel_sectors):
    st.markdown("## 📊 Sector Comparison")
    if not sel_sectors:
        st.warning("Select at least one sector.")
        return

    sector_scores = {}
    prog = st.progress(0)
    for i, sector in enumerate(sel_sectors):
        syms = SECTOR_MAP.get(sector, [])
        bull_confs, fund_scores = [], []
        for sym in syms:
            try:
                df = fetch_ohlcv(sym, period="6mo")
                if df is None or len(df) < 60: continue
                m = TechnicalModel(sym)
                if m.load(): bull_confs.append(m.predict(df)["proba_bull"])
                fund_scores.append(score_fundamentals(fetch_fundamentals(sym)).score)
            except Exception: pass
        sector_scores[sector] = {
            "avg":      round(np.mean(fund_scores), 2) if fund_scores else 0,
            "avg_conf": round(np.mean(bull_confs),  3) if bull_confs  else 0.5,
            "n":        len(syms),
        }
        prog.progress((i+1)/len(sel_sectors))

    rows = [{"Sector": s, "Avg Fund Score": v["avg"], "Avg Bull Prob": f"{v['avg_conf']:.1%}",
             "Stocks": v["n"], "Outlook": "🟢 Bullish" if v["avg_conf"]>0.55 else ("🔴 Bearish" if v["avg_conf"]<0.45 else "🟡 Neutral")}
            for s, v in sorted(sector_scores.items(), key=lambda x: x[1]["avg"], reverse=True)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fig = go.Figure()
    snames = [r["Sector"] for r in rows]
    fig.add_trace(go.Bar(x=snames, y=[sector_scores[s]["avg"] for s in snames],
                         name="Avg Fund Score", marker_color="#2196F3"))
    fig.add_trace(go.Bar(x=snames, y=[sector_scores[s]["avg_conf"]*18 for s in snames],
                         name="Tech Score (scaled)", marker_color="#00C851"))
    fig.update_layout(barmode="group", template="plotly_dark", height=380,
                      title="Sector Scores", margin=dict(l=40,r=20,t=50,b=40))
    st.plotly_chart(fig, use_container_width=True)

    if ANTHROPIC_API_KEY and st.button("🤖 AI Market Summary"):
        with st.spinner("Generating…"):
            summ = generate_market_summary(sector_scores)
        st.markdown(f'<div class="analysis-box">{summ}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def page_backtest(symbol, period, capital, stop_pct, tp_pct, min_conf, long_only):
    st.markdown(f"## ⏪ Backtest — {symbol}")

    with st.spinner("Fetching data…"):
        df_raw = fetch_ohlcv(symbol, period=period)
    if df_raw is None or len(df_raw) < 500:
        st.error("Need at least 2 years of data.")
        return

    with st.spinner("Running walk-forward backtest… (~30-60s)"):
        result = run_backtest(
            symbol=symbol, df_raw=df_raw, initial_capital=capital,
            stop_loss_pct=stop_pct/100, take_profit_pct=tp_pct/100,
            min_confidence=min_conf/100, long_only=long_only,
        )

    summary = result.summary()
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Return",  summary["Total Return"])
    k2.metric("Win Rate",      summary["Win Rate"])
    k3.metric("Total Trades",  summary["Total Trades"])
    k4.metric("Sharpe Ratio",  summary["Sharpe Ratio"])
    k5.metric("Max Drawdown",  summary["Max Drawdown"])
    k6.metric("Profit Factor", summary["Profit Factor"])

    st.plotly_chart(backtest_equity_curve(result), use_container_width=True)

    st.markdown('<p class="section-title">Trade Log</p>', unsafe_allow_html=True)
    df_trades = result.trades_df()
    if df_trades.empty:
        st.warning("No trades executed. Lower the confidence threshold.")
    else:
        def _pc(v):
            return "color:#00C851" if "+" in str(v) else "color:#FF4444"
        st.dataframe(df_trades.style.map(_pc, subset=["P&L %"]),
                     use_container_width=True, hide_index=True)

    st.markdown('<p class="section-title">Performance Summary</p>', unsafe_allow_html=True)
    for k, v in summary.items():
        st.metric(k, v)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
def page_portfolio(pf_tab):
    st.markdown("## 💼 Portfolio Tracker")
    pf = Portfolio.load()

    if pf_tab == "Add Holding":
        with st.form("add_form"):
            c1, c2 = st.columns(2)
            sym    = c1.text_input("Symbol (NSE)", "RELIANCE").upper()
            qty    = c2.number_input("Quantity", min_value=0.01, value=10.0)
            cost   = c1.number_input("Avg Cost (₹)", min_value=0.01, value=1000.0)
            date   = c2.date_input("Buy Date")
            sector = st.selectbox("Sector", ["Unknown"] + list(SECTOR_MAP.keys()))
            notes  = st.text_input("Notes (optional)")
            cash   = st.number_input("Cash Balance (₹)", value=pf.cash, min_value=0.0)
            if st.form_submit_button("Add Holding", type="primary"):
                pf.cash = cash
                pf.add_holding(sym, qty, cost, str(date), sector, notes)
                pf.save()
                st.success(f"✅ Added {qty} × {sym} @ ₹{cost:.2f}")

    elif pf_tab == "Remove Holding":
        if not pf.holdings:
            st.info("No holdings to remove.")
        else:
            rem = st.selectbox("Select to Remove", [h.symbol for h in pf.holdings])
            if st.button("Remove", type="primary"):
                pf.remove_holding(rem); pf.save()
                st.success(f"Removed {rem}")

    if not pf.holdings:
        st.info("No holdings yet. Add your first stock above.")
        return

    with st.spinner("Fetching live prices…"):
        metrics = pf.get_metrics()
        df_sum  = pf.get_summary()
        df_sec  = pf.sector_allocation()

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Invested",   f"₹{metrics['total_invested']:,.0f}")
    k2.metric("Current",    f"₹{metrics['total_current']:,.0f}", f"{metrics['total_pnl_pct']:+.2f}%")
    k3.metric("P&L",        f"₹{metrics['total_pnl']:+,.0f}")
    k4.metric("Cash",       f"₹{metrics['cash']:,.0f}")
    k5.metric("Holdings",   metrics["n_holdings"])

    st.metric("Best",  metrics["best_performer"])
    st.metric("Worst", metrics["worst_performer"])

    if not df_sec.empty:
        ch1, ch2 = st.columns(2)
        with ch1: st.plotly_chart(portfolio_pie(df_sec),       use_container_width=True)
        with ch2: st.plotly_chart(portfolio_pnl_bar(df_sum),   use_container_width=True)

    st.markdown('<p class="section-title">Holdings Detail</p>', unsafe_allow_html=True)
    def _cp(v):
        if isinstance(v, float): return "color:#00C851" if v>=0 else "color:#FF4444"
        return ""
    st.dataframe(df_sum.style.map(_cp, subset=["P&L %","P&L (₹)"]),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
    <h1 style="margin:0;">🇮🇳 Indian Stock Analyzer Pro</h1>
    <span style="background:#1565c0;color:white;border-radius:6px;padding:3px 10px;font-size:0.75em;font-weight:600;">v2.0</span>
</div>
<p style="color:#8b949e;margin-bottom:20px;">ML Ensemble + Fundamental Scoring + LLM · NSE/BSE</p>
""", unsafe_allow_html=True)

if mode == "🔍 Single Stock":
    page_single_stock(symbol, period, n_bars, run_llm, run_news)
elif mode == "🔎 Screener":
    if screen_btn: page_screener(sec_filter, sig_filter, min_rsi, max_rsi, min_fund, min_adx, top_n)
    else: st.info("👈 Configure filters and click **Run Screener →**")
elif mode == "📊 Sector View":
    if sector_btn: page_sector_view(sel_sectors)
    else: st.info("👈 Select sectors and click **Compare →**")
elif mode == "⏪ Backtest":
    if bt_btn: page_backtest(bt_symbol, bt_period, bt_capital, bt_stop, bt_tp, bt_conf, bt_longonly)
    else: st.info("👈 Configure parameters and click **Run Backtest →**")
elif mode == "💼 Portfolio":
    page_portfolio(pf_tab)
