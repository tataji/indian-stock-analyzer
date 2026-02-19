# 🇮🇳 Indian Stock Analyzer

A production-ready stock analysis system for NSE/BSE stocks combining:
- **ML Ensemble Models** (Gradient Boosting + Random Forest + Logistic Regression)
- **30+ Technical Indicators** (RSI, MACD, BB, ADX, Stochastic, Williams %R, CCI, OBV…)
- **Fundamental Scoring Engine** (P/E, ROE, D/E, margins, growth)
- **LLM-powered Analysis** via Claude (Anthropic)
- **Streamlit Dashboard** with interactive charts

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone <your-repo>
cd indian-stock-analyzer
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run the app
```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
indian-stock-analyzer/
├── config/
│   └── settings.py          # All configuration (API keys, thresholds, symbols)
├── data/
│   ├── fetcher.py            # NSE/BSE data via Yahoo Finance
│   └── preprocessor.py      # Feature engineering (30+ indicators)
├── models/
│   ├── technical.py          # ML ensemble classifier
│   ├── fundamental.py        # Rule-based fundamental scorer
│   ├── trainer.py            # Batch training pipeline
│   └── saved/               # Persisted model files (.joblib)
├── llm/
│   └── analyzer.py           # Claude LLM integration
├── utils/
│   ├── charts.py             # Plotly chart builders
│   └── screener.py           # Multi-stock screener
├── app/
│   └── streamlit_app.py      # Main UI
├── requirements.txt
└── .env.example
```

---

## 📊 Features

### Single Stock Analysis
- Live OHLCV data from NSE (Yahoo Finance `.NS` suffix)
- 30+ technical indicators with signal interpretation
- ML model trained on historical data using TimeSeriesSplit (no leakage)
- Fundamental scoring across 9 dimensions
- Claude-powered research report

### Stock Screener
- Screen across Nifty 50 universe (or custom sector)
- Filter by: signal, RSI range, fundamental score, ADX
- AI comparison of top picks

### Sector Comparison
- Compare sector-level technical and fundamental strength
- AI market summary with overweight/underweight recommendations

---

## 🔧 Configuration

Edit `config/settings.py` to adjust:
- `PREDICTION_HORIZON_DAYS` — ML target horizon (default: 5 days)
- `UPSIDE_THRESHOLD` — Bullish threshold (default: 2%)
- `FUNDAMENTAL_THRESHOLDS` — Scoring thresholds
- `NIFTY50_SYMBOLS` — Stock universe
- `SECTOR_MAP` — Sector groupings

---

## 📈 Technical Indicators Used

| Category    | Indicators |
|-------------|-----------|
| Trend       | EMA 20/50/200, SMA 20/50, Golden Cross |
| Momentum    | RSI, MACD, Stochastic K/D, CCI, Williams %R, ROC |
| Volatility  | ATR, Bollinger Bands (width, %B, squeeze), HV 20/60 |
| Volume      | OBV, Volume Ratio, Volume MA, Surge detection |
| Strength    | ADX, DI+, DI- |
| Price       | 52-week high/low, support/resistance |
| Candles     | Body size, upper/lower shadows, bullish candle |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute SEBI-registered investment advice. Always do your own due diligence before investing.

---

## 📦 Dependencies

- `yfinance` — Market data
- `scikit-learn` — ML models
- `pandas`, `numpy` — Data processing
- `streamlit` — UI
- `plotly` — Charts
- `anthropic` — LLM analysis
- `joblib` — Model persistence
