"""
utils/portfolio.py
Portfolio tracker — tracks holdings, computes P&L, risk metrics,
and generates allocation analysis.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from data.fetcher import fetch_ohlcv, fetch_fundamentals

PORTFOLIO_FILE = "portfolio.json"


@dataclass
class Holding:
    symbol:      str
    quantity:    float
    avg_cost:    float
    buy_date:    str
    sector:      str = "Unknown"
    notes:       str = ""


@dataclass
class Portfolio:
    name:        str              = "My Portfolio"
    holdings:    list[Holding]    = field(default_factory=list)
    cash:        float            = 0.0
    created_at:  str              = field(default_factory=lambda: datetime.now().isoformat())

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str = PORTFOLIO_FILE):
        data = {
            "name":       self.name,
            "cash":       self.cash,
            "created_at": self.created_at,
            "holdings":   [asdict(h) for h in self.holdings],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = PORTFOLIO_FILE) -> "Portfolio":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        holdings = [Holding(**h) for h in data.get("holdings", [])]
        return cls(
            name       = data.get("name", "My Portfolio"),
            holdings   = holdings,
            cash       = data.get("cash", 0.0),
            created_at = data.get("created_at", datetime.now().isoformat()),
        )

    # ── Holding management ────────────────────────────────────────────────
    def add_holding(self, symbol: str, quantity: float, avg_cost: float,
                    buy_date: str = "", sector: str = "", notes: str = ""):
        # Update existing or add new
        for h in self.holdings:
            if h.symbol == symbol.upper():
                total_qty  = h.quantity + quantity
                h.avg_cost = (h.avg_cost * h.quantity + avg_cost * quantity) / total_qty
                h.quantity = total_qty
                return
        self.holdings.append(Holding(
            symbol   = symbol.upper(),
            quantity = quantity,
            avg_cost = avg_cost,
            buy_date = buy_date or datetime.now().strftime("%Y-%m-%d"),
            sector   = sector,
            notes    = notes,
        ))

    def remove_holding(self, symbol: str):
        self.holdings = [h for h in self.holdings if h.symbol != symbol.upper()]

    # ── Live valuation ────────────────────────────────────────────────────
    def get_live_prices(self) -> dict[str, float]:
        prices = {}
        for h in self.holdings:
            try:
                df = fetch_ohlcv(h.symbol, period="5d")
                if df is not None and not df.empty:
                    prices[h.symbol] = float(df["Close"].iloc[-1])
            except Exception:
                prices[h.symbol] = h.avg_cost
        return prices

    def get_summary(self) -> pd.DataFrame:
        if not self.holdings:
            return pd.DataFrame()

        prices = self.get_live_prices()
        rows   = []

        for h in self.holdings:
            ltp        = prices.get(h.symbol, h.avg_cost)
            invested   = h.quantity * h.avg_cost
            current    = h.quantity * ltp
            pnl_abs    = current - invested
            pnl_pct    = (ltp - h.avg_cost) / h.avg_cost * 100

            rows.append({
                "Symbol":        h.symbol,
                "Qty":           h.quantity,
                "Avg Cost (₹)":  round(h.avg_cost, 2),
                "LTP (₹)":       round(ltp, 2),
                "Invested (₹)":  round(invested, 2),
                "Current (₹)":   round(current, 2),
                "P&L (₹)":       round(pnl_abs, 2),
                "P&L %":         round(pnl_pct, 2),
                "Sector":        h.sector,
            })

        return pd.DataFrame(rows)

    def get_metrics(self) -> dict:
        if not self.holdings:
            return {}

        df = self.get_summary()
        total_invested = df["Invested (₹)"].sum()
        total_current  = df["Current (₹)"].sum()
        total_pnl      = total_current - total_invested
        total_pnl_pct  = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        best  = df.loc[df["P&L %"].idxmax()]
        worst = df.loc[df["P&L %"].idxmin()]

        return {
            "total_invested":  round(total_invested, 2),
            "total_current":   round(total_current, 2),
            "total_pnl":       round(total_pnl, 2),
            "total_pnl_pct":   round(total_pnl_pct, 2),
            "total_with_cash": round(total_current + self.cash, 2),
            "cash":            round(self.cash, 2),
            "n_holdings":      len(self.holdings),
            "best_performer":  f"{best['Symbol']} (+{best['P&L %']:.1f}%)",
            "worst_performer": f"{worst['Symbol']} ({worst['P&L %']:.1f}%)",
            "winners":         int((df["P&L %"] > 0).sum()),
            "losers":          int((df["P&L %"] <= 0).sum()),
        }

    def sector_allocation(self) -> pd.DataFrame:
        if not self.holdings:
            return pd.DataFrame()
        df = self.get_summary()
        grouped = df.groupby("Sector")["Current (₹)"].sum().reset_index()
        grouped["Weight %"] = (grouped["Current (₹)"] / grouped["Current (₹)"].sum() * 100).round(1)
        return grouped.sort_values("Weight %", ascending=False)
