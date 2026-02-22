"""
utils/charts.py
Plotly chart builders for the Streamlit UI.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BULL_COLOR = "#00C851"
BEAR_COLOR = "#FF4444"
NEUTRAL_COLOR = "#FFBB33"
ACCENT_COLOR = "#2196F3"


def candlestick_chart(df: pd.DataFrame, symbol: str, n_bars: int = 120) -> go.Figure:
    """
    Multi-panel candlestick chart with:
    - Panel 1: Candlesticks + EMA20/50/200 + Bollinger Bands
    - Panel 2: Volume bars
    - Panel 3: RSI
    - Panel 4: MACD
    """
    df = df.tail(n_bars).copy()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.175, 0.175],
        vertical_spacing=0.025,
        subplot_titles=["", "Volume", "RSI (14)", "MACD"],
    )

    # ── Panel 1: Candlesticks ─────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing_line_color=BULL_COLOR,
        decreasing_line_color=BEAR_COLOR,
        showlegend=False,
    ), row=1, col=1)

    for col_name, color, label in [
        ("EMA_20", "#FFA500", "EMA 20"),
        ("EMA_50", "#9C27B0", "EMA 50"),
        ("EMA_200", "#2196F3", "EMA 200"),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                name=label, line=dict(color=color, width=1.5),
                opacity=0.85,
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"],
            name="BB Upper", line=dict(color="rgba(33,150,243,0.3)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            name="BB Lower", line=dict(color="rgba(33,150,243,0.3)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(33,150,243,0.05)",
            showlegend=False,
        ), row=1, col=1)

    # ── Panel 2: Volume ───────────────────────────────────────────────────
    colors = [BULL_COLOR if c >= o else BEAR_COLOR
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors,
        showlegend=False, opacity=0.8,
    ), row=2, col=1)

    if "Vol_MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Vol_MA20"],
            name="Vol MA20", line=dict(color="#FFA500", width=1.2),
        ), row=2, col=1)

    # ── Panel 3: RSI ──────────────────────────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI", line=dict(color=ACCENT_COLOR, width=1.5),
        ), row=3, col=1)
        # Overbought / oversold bands
        for level, color in [(70, BEAR_COLOR), (30, BULL_COLOR), (50, "grey")]:
            fig.add_hline(y=level, row=3, col=1,
                         line_dash="dash", line_color=color, opacity=0.5)

    # ── Panel 4: MACD ─────────────────────────────────────────────────────
    if "MACD" in df.columns:
        hist_colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"],
            name="MACD Histogram", marker_color=hist_colors,
            showlegend=False, opacity=0.7,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            name="MACD", line=dict(color=ACCENT_COLOR, width=1.5),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"],
            name="Signal", line=dict(color="#FF9800", width=1.5),
        ), row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"{symbol} — Technical Chart", font=dict(size=18)),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
        margin=dict(l=50, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",    row=2, col=1)
    fig.update_yaxes(title_text="RSI",       row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",      row=4, col=1)

    return fig


def fundamental_radar(fund_result) -> go.Figure:
    """Radar/spider chart for fundamental score breakdown."""
    categories, values = [], []

    score_map = {
        "P/E Ratio":      0, "P/B Ratio":      0,
        "ROE":            0, "Profit Margin":   0,
        "Revenue Growth": 0, "EPS Growth":      0,
        "Debt/Equity":    0, "Current Ratio":   0,
        "Dividend Yield": 0,
    }
    for sig in fund_result.signals:
        if sig.name in score_map:
            score_map[sig.name] += max(0, sig.score_delta)   # only positives for radar

    for k, v in score_map.items():
        categories.append(k)
        values.append(v)

    # Normalize to 0-100
    max_val = max(values) if max(values) > 0 else 1
    values_norm = [v / max_val * 100 for v in values]

    fig = go.Figure(go.Scatterpolar(
        r=values_norm + [values_norm[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(33,150,243,0.2)",
        line=dict(color=ACCENT_COLOR, width=2),
        name="Fundamentals",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Fundamental Strength Radar",
        template="plotly_dark",
        height=400,
        showlegend=False,
    )
    return fig


def score_gauge(score: int, max_score: int, title: str = "Score") -> go.Figure:
    pct = score / max_score * 100
    color = BULL_COLOR if pct >= 60 else (NEUTRAL_COLOR if pct >= 35 else BEAR_COLOR)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  35],  "color": "rgba(255,68,68,0.15)"},
                {"range": [35, 60],  "color": "rgba(255,187,51,0.15)"},
                {"range": [60, 100], "color": "rgba(0,200,81,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": pct,
            },
        },
    ))
    fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def confidence_bar(tech_signal: dict) -> go.Figure:
    bull_pct = tech_signal["proba_bull"] * 100
    bear_pct = tech_signal["proba_bear"] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Signal"],
        x=[bull_pct],
        name="Bullish",
        orientation="h",
        marker_color=BULL_COLOR,
        text=[f"Bullish {bull_pct:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=["Signal"],
        x=[bear_pct],
        name="Bearish",
        orientation="h",
        marker_color=BEAR_COLOR,
        text=[f"Bearish {bear_pct:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        title="ML Signal Confidence",
        template="plotly_dark",
        height=130,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(range=[0, 100], showticklabels=False),
        showlegend=False,
    )
    return fig


def returns_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of daily returns."""
    returns = df["Close"].pct_change().dropna() * 100
    fig = go.Figure(go.Histogram(
        x=returns,
        nbinsx=50,
        marker_color=ACCENT_COLOR,
        opacity=0.75,
        name="Daily Returns",
    ))
    fig.add_vline(x=0, line_color="white", line_dash="dash")
    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=300,
        margin=dict(l=30, r=20, t=40, b=30),
    )
    return fig


def backtest_equity_curve(result) -> go.Figure:
    """Plot equity curve + drawdown from a BacktestResult."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=["Equity Curve", "Drawdown"],
    )

    if result.equity_curve.empty:
        fig.add_annotation(text="No trades executed", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18))
        return fig

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        name="Portfolio Value",
        line=dict(color=BULL_COLOR, width=2),
        fill="tozeroy", fillcolor="rgba(0,200,81,0.1)",
    ), row=1, col=1)

    # Buy & hold reference
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=[result.equity_curve.iloc[0]] * len(result.equity_curve),
        name="Initial Capital",
        line=dict(color="grey", width=1, dash="dash"),
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=result.drawdown_series.index,
        y=result.drawdown_series.values * 100,
        name="Drawdown %",
        line=dict(color=BEAR_COLOR, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,68,68,0.15)",
    ), row=2, col=1)

    fig.update_layout(
        title=f"Backtest Results — {result.symbol}",
        template="plotly_dark",
        height=550,
        margin=dict(l=50, r=30, t=60, b=30),
    )
    fig.update_yaxes(title_text="Value (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig


def portfolio_pie(df_allocation: "pd.DataFrame") -> go.Figure:
    """Pie chart for sector allocation."""
    import pandas as pd
    fig = go.Figure(go.Pie(
        labels=df_allocation["Sector"],
        values=df_allocation["Current (₹)"],
        hole=0.45,
        marker=dict(colors=[
            "#2196F3", "#00C851", "#FF9800", "#E91E63",
            "#9C27B0", "#00BCD4", "#FF5722", "#8BC34A",
        ]),
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Sector Allocation",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def portfolio_pnl_bar(df_summary: "pd.DataFrame") -> go.Figure:
    """Horizontal bar chart of individual P&L %."""
    df = df_summary.sort_values("P&L %")
    colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in df["P&L %"]]

    fig = go.Figure(go.Bar(
        x=df["P&L %"],
        y=df["Symbol"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["P&L %"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(
        title="Individual Stock P&L",
        template="plotly_dark",
        height=max(300, len(df) * 40 + 80),
        xaxis_title="P&L %",
        margin=dict(l=80, r=60, t=50, b=30),
    )
    return fig
