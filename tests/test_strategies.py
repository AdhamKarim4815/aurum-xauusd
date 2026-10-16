from __future__ import annotations

import pandas as pd

from gold_trader.data import generate_demo_data, normalize_bars, resample_bars
from gold_trader.strategies import strategy_registry


def test_generation_two_registry_is_bounded_and_hashes_are_unique() -> None:
    strategies = strategy_registry()
    generation_two = {
        key: strategy for key, strategy in strategies.items() if key != strategy.strategy_id
    }
    assert set(generation_two) == {
        "session-breakout-l12",
        "session-breakout-l36",
        "volatility-mean-reversion-l12",
        "volatility-mean-reversion-l36",
        "trend-pullback-ema12-36",
        "trend-pullback-ema30-80",
        "time-series-momentum-l20",
        "time-series-momentum-l35",
    }
    assert len({strategy.strategy_hash for strategy in strategies.values()}) == len(strategies)


def test_every_preregistered_strategy_runs_without_future_access() -> None:
    normalized, _ = normalize_bars(generate_demo_data(rows=12_000))
    for strategy in strategy_registry().values():
        bars = resample_bars(normalized, strategy.timeframe)
        result = strategy.generate(bars)
        assert result is None or result.generated_at == pd.Timestamp(bars.iloc[-1]["timestamp"])


def test_breakout_and_momentum_emit_auditable_signal() -> None:
    bars = generate_demo_data(rows=500)
    bars.loc[bars.index[-1], "close"] = bars.iloc[:-1]["high"].max() + 20
    bars.loc[bars.index[-1], "high"] = bars.loc[bars.index[-1], "close"] + 1
    bars.loc[bars.index[-1], "low"] = (
        min(bars.loc[bars.index[-1], "open"], bars.loc[bars.index[-1], "close"]) - 1
    )
    bars.loc[bars.index[-1], "timestamp"] = pd.Timestamp("2025-01-03 10:00:00", tz="UTC")
    breakout = strategy_registry()["session-breakout"].generate(bars)
    momentum = strategy_registry()["time-series-momentum"].generate(bars)
    assert breakout is not None and breakout.reason
    assert momentum is not None and momentum.reason
