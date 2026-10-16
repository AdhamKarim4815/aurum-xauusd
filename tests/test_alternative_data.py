from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from gold_trader.alternative_data import (
    CftcPublicReportingProvider,
    GovernedWebCollector,
    PointInTimeRecord,
    SourcePolicy,
    SourceRegistry,
    parse_rss_headlines,
)


def policy(**overrides: object) -> SourcePolicy:
    values: dict[str, object] = {
        "name": "official-feed",
        "base_url": "https://example.com",
        "terms_url": "https://example.com/terms",
        "purpose": "timestamped research headlines",
        "allow_scraping": True,
    }
    values.update(overrides)
    return SourcePolicy(**values)  # type: ignore[arg-type]


def test_point_in_time_record_rejects_prepublication_availability() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="cannot precede"):
        PointInTimeRecord(
            source="test",
            kind="macro",
            event_time=now - timedelta(days=1),
            published_at=now,
            available_at=now - timedelta(seconds=1),
            values={"value": 1.0},
        )


def test_source_registry_controls_scraping() -> None:
    registry = SourceRegistry()
    registry.register(policy())
    assert registry.require("official-feed", scraping=True).enabled
    assert registry.manifest()[0]["terms_url"].endswith("/terms")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(policy())
    registry.disable("official-feed")
    with pytest.raises(ValueError, match="not enabled"):
        registry.require("official-feed")


def test_collector_and_cftc_require_governed_sources() -> None:
    with pytest.raises(ValueError, match="does not permit"):
        GovernedWebCollector(policy(allow_scraping=False))
    with pytest.raises(ValueError, match="official"):
        CftcPublicReportingProvider("https://example.com/data.json")


def test_rss_headlines_preserve_publication_and_retrieval_time() -> None:
    xml = """<rss><channel><item><title>Gold and inflation</title>
      <link>https://example.com/gold</link>
      <pubDate>Wed, 22 Jul 2026 12:00:00 GMT</pubDate>
    </item></channel></rss>"""
    retrieved = datetime(2026, 7, 22, 12, 5, tzinfo=UTC)
    records = parse_rss_headlines("official-feed", xml, retrieved)
    assert records[0].values["headline"] == "Gold and inflation"
    assert records[0].available_at == retrieved
