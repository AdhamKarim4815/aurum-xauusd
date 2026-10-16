from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from gold_trader.domain import Bar, FeatureVector, Quote


def test_quote_rejects_crossed_market() -> None:
    with pytest.raises(ValidationError, match="crossed quote"):
        Quote(bid=2401, ask=2400, provider="fixture", available_at=datetime.now(UTC))


def test_bar_rejects_inconsistent_ohlc() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="bar high"):
        Bar(
            timeframe="5min",
            open=2400,
            high=2399,
            low=2390,
            close=2401,
            available_at=now,
            occurred_at=now,
        )


def test_feature_rejects_future_availability() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="not available"):
        FeatureVector(as_of=now, available_at=now + timedelta(seconds=1), features={"x": 1.0})


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Quote(bid=2400, ask=2401, provider="fixture", available_at=datetime.now())
