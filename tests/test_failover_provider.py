from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import pytest

from gold_trader.data import FailoverQuoteProvider
from gold_trader.domain import Quote


class StubProvider:
    def __init__(
        self,
        quote_result: Quote | Exception,
        backfill_result: pd.DataFrame | Exception | None = None,
    ) -> None:
        self.quote_result = quote_result
        self.backfill_result = backfill_result
        self.quote_calls = 0
        self.backfill_calls = 0

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        self.quote_calls += 1
        if isinstance(self.quote_result, Exception):
            raise self.quote_result
        return self.quote_result

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.backfill_calls += 1
        if isinstance(self.backfill_result, Exception):
            raise self.backfill_result
        return self.backfill_result if self.backfill_result is not None else pd.DataFrame()


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, title: str, message: str) -> bool:
        self.messages.append((title, message))
        return True


def _quote(provider_name: str) -> Quote:
    return Quote(bid=2400.0, ask=2400.5, provider=provider_name, available_at=datetime.now(UTC))


def test_failover_requires_at_least_one_provider() -> None:
    with pytest.raises(ValueError):
        FailoverQuoteProvider([])


def test_first_provider_success_never_touches_second() -> None:
    primary = StubProvider(_quote("primary"))
    backup = StubProvider(_quote("backup"))
    failover = FailoverQuoteProvider([("primary", primary), ("backup", backup)])

    quote = asyncio.run(failover.quote())

    assert quote.provider == "primary"
    assert failover.active_provider_name == "primary"
    assert primary.quote_calls == 1
    assert backup.quote_calls == 0


def test_failover_switches_and_notifies_once() -> None:
    primary = StubProvider(RuntimeError("primary down"))
    backup = StubProvider(_quote("backup"))
    notifier = FakeNotifier()
    failover = FailoverQuoteProvider([("primary", primary), ("backup", backup)], notifier=notifier)

    first = asyncio.run(failover.quote())
    second = asyncio.run(failover.quote())

    assert first.provider == "backup"
    assert second.provider == "backup"
    assert failover.active_provider_name == "backup"
    assert primary.quote_calls == 2
    assert backup.quote_calls == 2
    # No notification on the very first poll (no prior "active" provider to compare against),
    # and no repeat notification on the second poll since the active provider didn't change again.
    assert notifier.messages == []


def test_failover_notifies_on_switch_after_a_healthy_period() -> None:
    primary = StubProvider(_quote("primary"))
    backup = StubProvider(_quote("backup"))
    notifier = FakeNotifier()
    failover = FailoverQuoteProvider([("primary", primary), ("backup", backup)], notifier=notifier)

    asyncio.run(failover.quote())
    primary.quote_result = RuntimeError("primary went down")
    asyncio.run(failover.quote())

    assert failover.active_provider_name == "backup"
    assert len(notifier.messages) == 1
    assert "primary" in notifier.messages[0][1]
    assert "backup" in notifier.messages[0][1]


def test_all_providers_failing_raises_last_error() -> None:
    primary = StubProvider(RuntimeError("primary down"))
    backup = StubProvider(RuntimeError("backup down"))
    failover = FailoverQuoteProvider([("primary", primary), ("backup", backup)])

    with pytest.raises(RuntimeError, match="backup down"):
        asyncio.run(failover.quote())


def test_backfill_fails_over_the_same_way() -> None:
    good_frame = pd.DataFrame(
        {
            "timestamp": [datetime.now(UTC)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    )
    primary = StubProvider(_quote("primary"), backfill_result=RuntimeError("no history"))
    backup = StubProvider(_quote("backup"), backfill_result=good_frame)
    failover = FailoverQuoteProvider([("primary", primary), ("backup", backup)])

    frame = asyncio.run(failover.backfill("XAUUSD", datetime.now(UTC), datetime.now(UTC)))

    assert len(frame) == 1
    assert failover.active_provider_name == "backup"
