from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import pandas as pd

from gold_trader.domain import Side, Signal
from gold_trader.features import atr, rsi

GENERATION_TWO_VARIANTS = (
    "session-breakout-l12",
    "session-breakout-l36",
    "volatility-mean-reversion-l12",
    "volatility-mean-reversion-l36",
    "trend-pullback-ema12-36",
    "trend-pullback-ema30-80",
    "time-series-momentum-l20",
    "time-series-momentum-l35",
)


@dataclass(frozen=True)
class StrategyParameters:
    lookback: int
    atr_period: int = 14
    stop_atr: float = 1.5
    reward_risk: float = 2.0
    fast_window: int = 20
    slow_window: int = 50
    band_std: float = 2.0
    trend_threshold: float = 0.012


class BaseStrategy:
    strategy_id = "base"
    timeframe = "5min"
    warmup = 60

    def __init__(self, parameters: StrategyParameters, variant_id: str | None = None) -> None:
        self.parameters = parameters
        self.variant_id = variant_id or f"{self.strategy_id}-baseline"

    def generate(self, bars: pd.DataFrame) -> Signal | None:
        if len(bars) < self.warmup:
            return None
        candidate = self.precompute(bars).iloc[-1]
        if pd.isna(candidate["side"]):
            return None
        return self._signal(
            bars,
            Side(str(candidate["side"])),
            str(candidate["reason"]),
            float(candidate["atr"]),
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def candidate_signal(
        self, bars: pd.DataFrame, candidates: pd.DataFrame, index: int
    ) -> Signal | None:
        if index < self.warmup:
            return None
        candidate = candidates.iloc[index]
        if pd.isna(candidate["side"]):
            return None
        return self._signal(
            bars.iloc[[index]],
            Side(str(candidate["side"])),
            str(candidate["reason"]),
            float(candidate["atr"]),
        )

    @property
    def strategy_hash(self) -> str:
        payload = {
            "id": self.strategy_id,
            "variant_id": self.variant_id,
            "timeframe": self.timeframe,
            **asdict(self.parameters),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _signal(self, bars: pd.DataFrame, side: Side, reason: str, atr_value: float) -> Signal:
        latest = bars.iloc[-1]
        price = float(latest["close"])
        distance = max(atr_value * self.parameters.stop_atr, price * 0.0005)
        if side is Side.BUY:
            stop = price - distance
            target = price + distance * self.parameters.reward_risk
        else:
            stop = price + distance
            target = price - distance * self.parameters.reward_risk
        return Signal(
            side=side,
            generated_at=pd.Timestamp(latest["timestamp"]).to_pydatetime(),
            reference_price=price,
            stop_loss=stop,
            take_profit=target,
            strategy_id=self.strategy_id,
            strategy_hash=self.strategy_hash,
            reason=reason,
        )


class SessionBreakoutStrategy(BaseStrategy):
    strategy_id = "session-breakout"
    timeframe = "15min"

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index, data={"side": None, "reason": None})
        previous_high = bars["high"].shift(1).rolling(self.parameters.lookback).max()
        previous_low = bars["low"].shift(1).rolling(self.parameters.lookback).min()
        hours = pd.to_datetime(bars["timestamp"], utc=True).dt.hour
        session = hours.between(7, 16)
        buy = session & (bars["close"] > previous_high)
        sell = session & (bars["close"] < previous_low)
        result.loc[buy, ["side", "reason"]] = [
            Side.BUY.value,
            "London/New York range broke upward",
        ]
        result.loc[sell, ["side", "reason"]] = [
            Side.SELL.value,
            "London/New York range broke downward",
        ]
        result["atr"] = atr(bars, self.parameters.atr_period)
        return result


class VolatilityMeanReversionStrategy(BaseStrategy):
    strategy_id = "volatility-mean-reversion"
    timeframe = "15min"

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index, data={"side": None, "reason": None})
        close = bars["close"]
        mean = close.rolling(self.parameters.lookback).mean()
        deviation = close.rolling(self.parameters.lookback).std()
        rsi_value = rsi(close)
        slow_trend = (close / close.ewm(span=80, adjust=False).mean() - 1).abs()
        regime = slow_trend <= 0.012
        buy = regime & (close < mean - self.parameters.band_std * deviation) & (rsi_value < 30)
        sell = regime & (close > mean + self.parameters.band_std * deviation) & (rsi_value > 70)
        result.loc[buy, ["side", "reason"]] = [
            Side.BUY.value,
            "Low-volatility lower-band exhaustion",
        ]
        result.loc[sell, ["side", "reason"]] = [
            Side.SELL.value,
            "Low-volatility upper-band exhaustion",
        ]
        result["atr"] = atr(bars, self.parameters.atr_period)
        return result


class TrendPullbackStrategy(BaseStrategy):
    strategy_id = "trend-pullback"
    timeframe = "1h"

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index, data={"side": None, "reason": None})
        close = bars["close"]
        fast = close.ewm(span=self.parameters.fast_window, adjust=False).mean()
        slow = close.ewm(span=self.parameters.slow_window, adjust=False).mean()
        crossed_up = (close.shift(1) <= fast.shift(1)) & (close > fast)
        crossed_down = (close.shift(1) >= fast.shift(1)) & (close < fast)
        buy = (fast > slow) & crossed_up
        sell = (fast < slow) & crossed_down
        result.loc[buy, ["side", "reason"]] = [
            Side.BUY.value,
            "Pullback reclaimed fast EMA in uptrend",
        ]
        result.loc[sell, ["side", "reason"]] = [
            Side.SELL.value,
            "Pullback rejected fast EMA in downtrend",
        ]
        result["atr"] = atr(bars, self.parameters.atr_period)
        return result


class TimeSeriesMomentumStrategy(BaseStrategy):
    strategy_id = "time-series-momentum"
    timeframe = "4h"

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index, data={"side": None, "reason": None})
        previous_high = bars["high"].shift(1).rolling(self.parameters.lookback).max()
        previous_low = bars["low"].shift(1).rolling(self.parameters.lookback).min()
        buy = bars["close"] > previous_high
        sell = bars["close"] < previous_low
        result.loc[buy, ["side", "reason"]] = [
            Side.BUY.value,
            "Long-horizon price-channel momentum",
        ]
        result.loc[sell, ["side", "reason"]] = [
            Side.SELL.value,
            "Long-horizon price-channel momentum",
        ]
        result["atr"] = atr(bars, self.parameters.atr_period)
        return result


def strategy_registry() -> dict[str, BaseStrategy]:
    baseline: dict[str, BaseStrategy] = {
        "session-breakout": SessionBreakoutStrategy(StrategyParameters(lookback=20)),
        "volatility-mean-reversion": VolatilityMeanReversionStrategy(
            StrategyParameters(lookback=20, stop_atr=1.25, reward_risk=1.5)
        ),
        "trend-pullback": TrendPullbackStrategy(StrategyParameters(lookback=20)),
        "time-series-momentum": TimeSeriesMomentumStrategy(
            StrategyParameters(lookback=55, stop_atr=2.0, reward_risk=2.5)
        ),
    }
    generation_two: dict[str, BaseStrategy] = {
        "session-breakout-l12": SessionBreakoutStrategy(
            StrategyParameters(lookback=12), "session-breakout-l12"
        ),
        "session-breakout-l36": SessionBreakoutStrategy(
            StrategyParameters(lookback=36), "session-breakout-l36"
        ),
        "volatility-mean-reversion-l12": VolatilityMeanReversionStrategy(
            StrategyParameters(lookback=12, stop_atr=1.25, reward_risk=1.5),
            "volatility-mean-reversion-l12",
        ),
        "volatility-mean-reversion-l36": VolatilityMeanReversionStrategy(
            StrategyParameters(lookback=36, stop_atr=1.25, reward_risk=1.5),
            "volatility-mean-reversion-l36",
        ),
        "trend-pullback-ema12-36": TrendPullbackStrategy(
            StrategyParameters(lookback=36, fast_window=12, slow_window=36),
            "trend-pullback-ema12-36",
        ),
        "trend-pullback-ema30-80": TrendPullbackStrategy(
            StrategyParameters(lookback=80, fast_window=30, slow_window=80),
            "trend-pullback-ema30-80",
        ),
        "time-series-momentum-l20": TimeSeriesMomentumStrategy(
            StrategyParameters(lookback=20, stop_atr=2.0, reward_risk=2.5),
            "time-series-momentum-l20",
        ),
        "time-series-momentum-l35": TimeSeriesMomentumStrategy(
            StrategyParameters(lookback=35, stop_atr=2.0, reward_risk=2.5),
            "time-series-momentum-l35",
        ),
    }
    if tuple(generation_two) != GENERATION_TWO_VARIANTS:
        raise RuntimeError("Generation-2 registry no longer matches its preregistration")
    return {**baseline, **generation_two}
