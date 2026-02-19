"""
models/technical.py
Gradient Boosting classifier for technical signal prediction.
Includes training, evaluation, persistence, and prediction.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import MODEL_DIR
from data.preprocessor import build_features, get_feature_columns

logger = logging.getLogger(__name__)

MODEL_PATH  = os.path.join(MODEL_DIR, "{symbol}_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "{symbol}_scaler.joblib")
FEATS_PATH  = os.path.join(MODEL_DIR, "{symbol}_features.joblib")


class TechnicalModel:
    """
    Ensemble classifier trained on technical indicators.

    Usage
    -----
    m = TechnicalModel("RELIANCE")
    report = m.train(df_ohlcv)
    result = m.predict(df_ohlcv)
    m.save()
    """

    def __init__(self, symbol: str):
        self.symbol   = symbol.upper()
        self.scaler   = StandardScaler()
        self.model    = None
        self.features = []
        self._trained = False

    # ── Build ensemble ────────────────────────────────────────────────────
    def _build_model(self):
        gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)

        voting = VotingClassifier(
            estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
            voting="soft",
            weights=[3, 2, 1],
        )
        return CalibratedClassifierCV(voting, cv=3, method="isotonic")

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, df_raw: pd.DataFrame) -> dict:
        """
        Train on raw OHLCV DataFrame.
        Returns a dict with accuracy, roc_auc, and classification report.
        """
        df = build_features(df_raw, include_labels=True)
        if len(df) < 200:
            raise ValueError(f"Not enough data to train ({len(df)} rows after preprocessing).")

        self.features = get_feature_columns(df)
        X = df[self.features].values
        y = df["target"].values

        # Time-series cross-validation (no data leakage)
        tscv = TimeSeriesSplit(n_splits=5)
        oof_preds = np.zeros(len(y))

        self.model = self._build_model()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr         = y[train_idx]

            X_tr_sc  = self.scaler.fit_transform(X_tr)
            X_val_sc = self.scaler.transform(X_val)

            self.model.fit(X_tr_sc, y_tr)
            oof_preds[val_idx] = self.model.predict_proba(X_val_sc)[:, 1]

        # Final fit on all data
        X_all = self.scaler.fit_transform(X)
        self.model.fit(X_all, y)
        self._trained = True

        # Metrics
        preds_binary = (oof_preds > 0.5).astype(int)
        try:
            auc = roc_auc_score(y, oof_preds)
        except Exception:
            auc = float("nan")

        report = {
            "accuracy":      round(accuracy_score(y, preds_binary), 4),
            "roc_auc":       round(auc, 4),
            "class_report":  classification_report(y, preds_binary),
            "n_samples":     len(y),
            "n_features":    len(self.features),
            "class_balance": round(y.mean(), 3),
        }
        logger.info("Trained %s | acc=%.4f auc=%.4f", self.symbol, report["accuracy"], report["roc_auc"])
        return report

    # ── Prediction ────────────────────────────────────────────────────────
    def predict(self, df_raw: pd.DataFrame) -> dict:
        """
        Predict on the most recent bar of an OHLCV DataFrame.

        Returns
        -------
        dict with keys: signal, confidence, proba_bull, proba_bear
        """
        if not self._trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first or load().")

        df = build_features(df_raw, include_labels=False)
        if df.empty:
            raise ValueError("Empty DataFrame after feature engineering.")

        # Use only known features, fill any missing with 0
        available = [f for f in self.features if f in df.columns]
        row = df[available].iloc[[-1]]

        # Align columns
        full = pd.DataFrame(0.0, index=row.index, columns=self.features)
        full[available] = row.values
        X = self.scaler.transform(full.values)

        proba = self.model.predict_proba(X)[0]
        pred  = int(np.argmax(proba))

        return {
            "signal":     "BULLISH" if pred == 1 else "BEARISH",
            "confidence": round(float(proba[pred]), 4),
            "proba_bull": round(float(proba[1]), 4),
            "proba_bear": round(float(proba[0]), 4),
        }

    # ── Feature importance ────────────────────────────────────────────────
    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return top-N feature importances from the GradientBoosting sub-model."""
        if not self._trained:
            raise RuntimeError("Model not trained.")
        try:
            gb_estimator = self.model.calibrated_classifiers_[0].estimator.named_estimators_["gb"]
            importances  = gb_estimator.feature_importances_
        except Exception:
            return pd.DataFrame()

        df_imp = pd.DataFrame({"feature": self.features, "importance": importances})
        return df_imp.sort_values("importance", ascending=False).head(top_n)

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self):
        """Persist model, scaler, and feature list to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model,    MODEL_PATH.format(symbol=self.symbol))
        joblib.dump(self.scaler,   SCALER_PATH.format(symbol=self.symbol))
        joblib.dump(self.features, FEATS_PATH.format(symbol=self.symbol))
        logger.info("Saved model for %s", self.symbol)

    def load(self) -> bool:
        """Load persisted model. Returns True if successful."""
        mp = MODEL_PATH.format(symbol=self.symbol)
        sp = SCALER_PATH.format(symbol=self.symbol)
        fp = FEATS_PATH.format(symbol=self.symbol)
        if not all(os.path.exists(p) for p in [mp, sp, fp]):
            return False
        try:
            self.model    = joblib.load(mp)
            self.scaler   = joblib.load(sp)
            self.features = joblib.load(fp)
            self._trained = True
            logger.info("Loaded model for %s", self.symbol)
            return True
        except Exception as exc:
            logger.error("Failed to load model for %s: %s", self.symbol, exc)
            return False
