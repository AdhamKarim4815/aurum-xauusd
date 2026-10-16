from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import pandas as pd
from defusedxml import ElementTree
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("point-in-time data requires timezone-aware timestamps")
    return value.astimezone(UTC)


class PointInTimeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    kind: str
    event_time: datetime
    published_at: datetime
    available_at: datetime
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    values: dict[str, str | float | int | bool | None]

    @field_validator("event_time", "published_at", "available_at", "retrieved_at")
    @classmethod
    def validate_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_availability_order(self) -> PointInTimeRecord:
        if self.available_at < self.published_at:
            raise ValueError("available_at cannot precede publication")
        return self


class SourcePolicy(BaseModel):
    name: str
    base_url: str
    terms_url: str
    purpose: str
    requests_per_minute: int = Field(default=6, ge=1, le=60)
    allow_scraping: bool = False
    enabled: bool = True
    parser_version: str = "1"


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, SourcePolicy] = {}

    def register(self, policy: SourcePolicy) -> None:
        if policy.name in self._sources:
            raise ValueError(f"source {policy.name!r} is already registered")
        self._sources[policy.name] = policy

    def require(self, name: str, scraping: bool = False) -> SourcePolicy:
        policy = self._sources.get(name)
        if policy is None or not policy.enabled:
            raise ValueError(f"source {name!r} is not enabled")
        if scraping and not policy.allow_scraping:
            raise ValueError(f"source {name!r} is not approved for scraping")
        return policy

    def disable(self, name: str) -> None:
        policy = self.require(name)
        self._sources[name] = policy.model_copy(update={"enabled": False})

    def manifest(self) -> list[dict[str, Any]]:
        return [policy.model_dump() for policy in self._sources.values()]


class FredVintageProvider:
    name = "fred-alfred"
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("FRED API key is required")
        self.api_key = api_key

    async def records(
        self, series_id: str, start: datetime, end: datetime
    ) -> list[PointInTimeRecord]:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.date().isoformat(),
            "observation_end": end.date().isoformat(),
            "realtime_start": start.date().isoformat(),
            "realtime_end": end.date().isoformat(),
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            observations = response.json().get("observations", [])
        records: list[PointInTimeRecord] = []
        for observation in observations:
            if observation.get("value") in {None, "."}:
                continue
            event_time = datetime.fromisoformat(observation["date"]).replace(tzinfo=UTC)
            available = datetime.fromisoformat(observation["realtime_start"]).replace(tzinfo=UTC)
            records.append(
                PointInTimeRecord(
                    source=self.name,
                    kind="macro-vintage",
                    event_time=event_time,
                    published_at=available,
                    available_at=available,
                    values={"series_id": series_id, "value": float(observation["value"])},
                )
            )
        return records


class CftcPublicReportingProvider:
    name = "cftc-cot"

    def __init__(self, dataset_url: str) -> None:
        if not dataset_url.startswith("https://publicreporting.cftc.gov/resource/"):
            raise ValueError("CFTC provider requires an official public-reporting dataset URL")
        self.dataset_url = dataset_url

    async def records(self, market_name: str, start: datetime, end: datetime) -> pd.DataFrame:
        query = (
            f"contract_market_name like '%{market_name.upper()}%' "
            f"and report_date_as_yyyy_mm_dd between '{start.date()}' and '{end.date()}'"
        )
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(self.dataset_url, params={"$where": query, "$limit": 5000})
            response.raise_for_status()
            frame = pd.DataFrame(response.json())
        if not frame.empty:
            frame["report_date_as_yyyy_mm_dd"] = pd.to_datetime(
                frame["report_date_as_yyyy_mm_dd"], utc=True
            )
            # COT positions are Tuesday observations normally published Friday.
            frame["available_at"] = frame["report_date_as_yyyy_mm_dd"] + pd.Timedelta(
                days=3, hours=20, minutes=30
            )
        return frame


class GovernedWebCollector:
    """Fetches allowlisted public pages only after an explicit robots check."""

    def __init__(self, policy: SourcePolicy, user_agent: str = "AurumResearchBot/0.1") -> None:
        if not policy.allow_scraping or not policy.enabled:
            raise ValueError("source policy does not permit scraping")
        self.policy = policy
        self.user_agent = user_agent

    async def fetch(self, url: str) -> dict[str, Any]:
        if urlparse(url).netloc != urlparse(self.policy.base_url).netloc:
            raise ValueError("URL is outside the allowlisted source host")
        robots_url = urljoin(self.policy.base_url, "/robots.txt")
        async with httpx.AsyncClient(
            timeout=20, headers={"User-Agent": self.user_agent}, follow_redirects=True
        ) as client:
            robots_response = await client.get(robots_url)
            robots_response.raise_for_status()
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(robots_response.text.splitlines())
            if not parser.can_fetch(self.user_agent, url):
                raise PermissionError("robots.txt does not permit this fetch")
            response = await client.get(url)
            response.raise_for_status()
        return {
            "source": self.policy.name,
            "url": str(response.url),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "parser_version": self.policy.parser_version,
            "terms_url": self.policy.terms_url,
            "content": response.text,
        }


def parse_rss_headlines(
    source: str, xml_text: str, retrieved_at: datetime
) -> list[PointInTimeRecord]:
    root = ElementTree.fromstring(xml_text)
    records: list[PointInTimeRecord] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        published = item.findtext("pubDate")
        if not title or not published:
            continue
        from email.utils import parsedate_to_datetime

        published_at = parsedate_to_datetime(published).astimezone(UTC)
        records.append(
            PointInTimeRecord(
                source=source,
                kind="headline",
                event_time=published_at,
                published_at=published_at,
                available_at=max(published_at, retrieved_at.astimezone(UTC)),
                retrieved_at=retrieved_at,
                values={"headline": title, "link": item.findtext("link")},
            )
        )
    return records
