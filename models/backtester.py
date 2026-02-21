"""
models/backtester.py
Walk-forward backtesting engine for the TechnicalModel.
Produces trade log, equity curve, and performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data.preprocessor import build_features
from models.technical import TechnicalModel

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    direction:   str        # LONG / SHORT
    pnl_pct:     float
    pnl_abs:     float
    bars_held:   int
    signal_conf: float


@dataclass
class BacktestResult:
    symbol:          str
    trades:          list[Trade]    = field(default_factory=list)
    equity_curve:    pd.Series      = field(default_factory=pd.Series)
    drawdown_series: pd.Series      = field(default_factory=pd.Series)

    # ── Computed metrics ──────────────────────────────────────────────────
    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_pct > 0) / len(self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_pct for t in self.trades if t.pnl_pct <= 0]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss   = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct <= 0))
        return round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return round((self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100, 2)

    @property
    def max_drawdown(self) -> float:
        if self.drawdown_series.empty:
            return 0.0
        return round(self.drawdown_series.min() * 100, 2)

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        daily_ret = self.equity_curve.pct_change().dropna()
        if daily_ret.std() == 0:
            return 0.0
        return round(daily_ret.mean() / daily_ret.std() * np.sqrt(252), 3)

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown == 0:
            return 0.0
        return round(self.total_return / abs(self.max_drawdown), 3)

    @property
    def avg_bars_held(self) -> float:
        if not self.trades:
            return 0.0
        return round(np.mean([t.bars_held for t in self.trades]), 1)

    def summary(self) -> dict:
        return {
            "Symbol":            self.symbol,
            "Total Trades":      self.n_trades,
            "Win Rate":          f"{self.win_rate:.1%}",
            "Avg Win":           f"{self.avg_win:.2%}",
            "Avg Loss":          f"{self.avg_loss:.2%}",
            "Profit Factor":     self.profit_factor,
            "Total Return":      f"{self.total_return:.2f}%",
            "Max Drawdown":      f"{self.max_drawdown:.2f}%",
            "Sharpe Ratio":      self.sharpe_ratio,
            "Calmar Ratio":      self.calmar_ratio,
            "Avg Bars Held":     self.avg_bars_held,
        }

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Entry Date":  t.entry_date.date(),
            "Exit Date":   t.exit_date.date(),
            "Entry (₹)":   round(t.entry_price, 2),
            "Exit (₹)":    round(t.exit_price,  2),
            "Direction":   t.direction,
            "P&L %":       f"{t.pnl_pct*100:.2f}%",
            "Bars Held":   t.bars_held,
            "Confidence":  f"{t.signal_conf:.1%}",
        } for t in self.trades])


def run_backtest(
    symbol:           str,
    df_raw:           pd.DataFrame,
    initial_capital:  float = 100_000,
    position_size:    float = 0.95,      # fraction of capital per trade
    stop_loss_pct:    float = 0.05,      # 5% stop loss
    take_profit_pct:  float = 0.10,      # 10% take profit
    min_confidence:   float = 0.60,      # only trade if model confidence >= this
    train_window:     int   = 400,       # bars used for training
    test_window:      int   = 60,        # bars in each out-of-sample window
    long_only:        bool  = True,
) -> BacktestResult:
    """
    Walk-forward backtest using TechnicalModel.

    Parameters
    ----------
    symbol          : NSE ticker
    df_raw          : Raw OHLCV DataFrame (must have >= train_window + test_window rows)
    initial_capital : Starting capital in ₹
    position_size   : Fraction of capital to deploy per trade
    stop_loss_pct   : Hard stop loss below entry
    take_profit_pct : Hard take profit above entry
    min_confidence  : Minimum model confidence to enter a trade
    train_window    : Number of bars for each training window
    test_window     : Number of bars for each OOS test window
    long_only       : Only take long trades if True
    """
    result = BacktestResult(symbol=symbol)

    if len(df_raw) < train_window + test_window:
        logger.warning("Not enough data for backtest (%d rows)", len(df_raw))
        return result

    capital     = initial_capital
    equity      = []
    dates       = []
    in_trade    = False
    entry_price = 0.0
    entry_date  = None
    entry_conf  = 0.0
    direction   = "LONG"

    total_bars  = len(df_raw)
    start_idx   = train_window

    i = start_idx
    while i < total_bars:
        train_df = df_raw.iloc[max(0, i - train_window): i].copy()
        test_end = min(i + test_window, total_bars)

        # Train model on window
        try:
            model = TechnicalModel(symbol)
            model.train(train_df)
        except Exception as exc:
            logger.debug("Train failed at bar %d: %s", i, exc)
            i = test_end
            continue

        # Walk through OOS bars
        for j in range(i, test_end):
            bar      = df_raw.iloc[j]
            close    = float(bar["Close"])
            bar_date = df_raw.index[j]

            # ── Exit logic ────────────────────────────────────────────
            if in_trade:
                pnl_pct = (close - entry_price) / entry_price
                if direction == "SHORT":
                    pnl_pct = -pnl_pct

                exit_triggered = (
                    pnl_pct <= -stop_loss_pct or
                    pnl_pct >= take_profit_pct or
                    j == test_end - 1          # end of OOS window
                )

                if exit_triggered:
                    pnl_abs  = capital * position_size * pnl_pct
                    capital += pnl_abs
                    result.trades.append(Trade(
                        entry_date  = entry_date,
                        exit_date   = bar_date,
                        entry_price = entry_price,
                        exit_price  = close,
                        direction   = direction,
                        pnl_pct     = pnl_pct,
                        pnl_abs     = pnl_abs,
                        bars_held   = j - df_raw.index.get_loc(entry_date),
                        signal_conf = entry_conf,
                    ))
                    in_trade = False

            # ── Entry logic ───────────────────────────────────────────
            if not in_trade:
                try:
                    sig = model.predict(df_raw.iloc[: j + 1])
                except Exception:
                    continue

                if sig["confidence"] >= min_confidence:
                    if sig["signal"] == "BULLISH":
                        in_trade    = True
                        entry_price = close
                        entry_date  = bar_date
                        entry_conf  = sig["confidence"]
                        direction   = "LONG"
                    elif sig["signal"] == "BEARISH" and not long_only:
                        in_trade    = True
                        entry_price = close
                        entry_date  = bar_date
                        entry_conf  = sig["confidence"]
                        direction   = "SHORT"

            equity.append(capital)
            dates.append(bar_date)

        i = test_end

    # ── Equity curve & drawdown ───────────────────────────────────────────
    if equity:
        eq_series = pd.Series(equity, index=dates)
        result.equity_curve = eq_series
        roll_max = eq_series.cummax()
        result.drawdown_series = (eq_series - roll_max) / roll_max

    return result
