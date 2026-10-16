from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from gold_trader.domain import FeatureVector


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative))


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def build_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """All inputs are trailing; no backward fill or centered window is allowed."""
    frame = bars.copy().sort_values("timestamp").reset_index(drop=True)
    close = frame["close"]
    frame["return_1"] = close.pct_change()
    frame["return_3"] = close.pct_change(3)
    frame["return_12"] = close.pct_change(12)
    frame["volatility_12"] = frame["return_1"].rolling(12).std()
    frame["volatility_48"] = frame["return_1"].rolling(48).std()
    frame["ema_fast_gap"] = close / close.ewm(span=12, adjust=False).mean() - 1
    frame["ema_slow_gap"] = close / close.ewm(span=48, adjust=False).mean() - 1
    frame["rsi_14"] = rsi(close)
    frame["atr_14"] = atr(frame)
    frame["range_fraction"] = (frame["high"] - frame["low"]) / close
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    frame["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    frame["weekday"] = timestamp.dt.weekday
    return frame


class PointInTimeFeaturePipeline:
    def transform(self, bars: pd.DataFrame, as_of: datetime) -> FeatureVector:
        cutoff = as_of.astimezone(UTC)
        available = bars[pd.to_datetime(bars["timestamp"], utc=True) <= cutoff]
        if len(available) < 50:
            raise ValueError("at least 50 historical bars are required")
        features = build_feature_frame(available).iloc[-1]
        values = {
            key: float(features[key])
            for key in (
                "return_1",
                "return_3",
                "return_12",
                "volatility_12",
                "volatility_48",
                "ema_fast_gap",
                "ema_slow_gap",
                "rsi_14",
                "atr_14",
                "range_fraction",
                "hour_sin",
                "hour_cos",
                "weekday",
            )
            if not pd.isna(features[key])
        }
        return FeatureVector(as_of=cutoff, available_at=cutoff, features=values)
