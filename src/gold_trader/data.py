from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast
from uuid import uuid4

import httpx
import pandas as pd

from gold_trader.domain import DatasetManifest, Quote

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")


class LiveQuoteProvider(Protocol):
    async def quote(self, symbol: str = "XAUUSD") -> Quote: ...

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame: ...


_TIMEZONE_ALIASES = {
    "utc": "UTC",
    "gmt": "UTC",
    # JForex labels its DST-aware Eastern European export as EET even when
    # summer rows are emitted in EEST. Helsinki supplies the matching rules.
    "eet": "Europe/Helsinki",
    "eest": "Europe/Helsinki",
}


@dataclass(frozen=True)
class DataQualityReport:
    valid: bool
    rows: int
    duplicate_timestamps: int = 0
    missing_values: int = 0
    inconsistent_ohlc: int = 0
    non_monotonic_timestamps: int = 0
    crossed_quotes: int = 0
    large_gaps: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "gmt time": "timestamp",
        "bidopen": "open",
        "bidhigh": "high",
        "bidlow": "low",
        "bidclose": "close",
        "tick_volume": "volume",
        "vol": "volume",
    }
    lower = {str(column).strip().lower(): column for column in frame.columns}
    rename: dict[Any, str] = {}
    timestamp_header = next(
        (name for name in lower if re.fullmatch(r"(?:time|timestamp|date/time)\s*\([^)]+\)", name)),
        None,
    )
    if timestamp_header is not None and "timestamp" not in lower:
        rename[lower[timestamp_header]] = "timestamp"
    for source, destination in aliases.items():
        if source in lower and destination not in lower and destination not in rename.values():
            rename[lower[source]] = destination
    frame = frame.rename(columns=rename)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _timestamp_timezone(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        match = re.fullmatch(
            r"(?:time|timestamp|date/time)\s*\(([^)]+)\)",
            str(column).strip().lower(),
        )
        if match:
            return _TIMEZONE_ALIASES.get(match.group(1).strip().lower())
    return None


def normalize_bars(
    frame: pd.DataFrame, expected_interval: str = "5min"
) -> tuple[pd.DataFrame, DataQualityReport]:
    source_timezone = _timestamp_timezone(frame)
    data = _column_aliases(frame.copy())
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

    if source_timezone and source_timezone != "UTC":
        parsed = pd.to_datetime(data["timestamp"], errors="coerce")
        data["timestamp"] = parsed.dt.tz_localize(
            source_timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")
    else:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "bid_close", "ask_close"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_timestamps = int(data["timestamp"].isna().sum())
    duplicate_timestamps = int(data["timestamp"].duplicated(keep="last").sum())
    data = data.dropna(subset=list(REQUIRED_COLUMNS)).drop_duplicates("timestamp", keep="last")
    data = data.sort_values("timestamp").reset_index(drop=True)

    inconsistent = int(
        (
            (data["high"] < data[["open", "close", "low"]].max(axis=1))
            | (data["low"] > data[["open", "close", "high"]].min(axis=1))
            | (data[["open", "high", "low", "close"]].min(axis=1) <= 0)
        ).sum()
    )
    crossed = 0
    if {"bid_close", "ask_close"}.issubset(data.columns):
        crossed = int((data["ask_close"] < data["bid_close"]).sum())

    diffs = data["timestamp"].diff().dropna()
    expected = pd.Timedelta(expected_interval)
    large_gaps = int((diffs > expected * 6).sum())
    non_monotonic = int((diffs <= pd.Timedelta(0)).sum())
    missing_values = int(data[list(REQUIRED_COLUMNS)].isna().sum().sum()) + invalid_timestamps
    warnings: list[str] = []
    if large_gaps:
        warnings.append(f"{large_gaps} gaps exceed six expected intervals")
    if duplicate_timestamps:
        warnings.append(f"deduplicated {duplicate_timestamps} timestamps")

    report = DataQualityReport(
        valid=not any((missing_values, inconsistent, non_monotonic, crossed)),
        rows=len(data),
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        inconsistent_ohlc=inconsistent,
        non_monotonic_timestamps=non_monotonic,
        crossed_quotes=crossed,
        large_gaps=large_gaps,
        warnings=warnings,
    )
    return data, report


def resample_bars(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    data = frame.set_index("timestamp")
    aggregation: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in data.columns:
        aggregation["volume"] = "sum"
    if "bid_close" in data.columns:
        aggregation["bid_close"] = "last"
    if "ask_close" in data.columns:
        aggregation["ask_close"] = "last"
    return (
        data.resample(timeframe, label="right", closed="right")
        .agg(aggregation)  # type: ignore[arg-type]
        .dropna()
        .reset_index()
    )


def import_market_file(
    source: Path,
    destination: Path,
    source_name: str = "user-import",
    source_url: str | None = None,
    terms_url: str | None = None,
) -> DatasetManifest:
    raw = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    normalized, quality = normalize_bars(raw)
    if not quality.valid:
        raise ValueError(f"data quality failed: {quality.as_dict()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(destination, index=False)
    retrieved_at = datetime.now(UTC)
    return DatasetManifest(
        dataset_id=str(uuid4()),
        source=source_name,
        source_url=source_url,
        retrieved_at=retrieved_at,
        content_sha256=file_sha256(destination),
        parser_version="1",
        row_count=len(normalized),
        start_at=normalized["timestamp"].iloc[0].to_pydatetime(),
        end_at=normalized["timestamp"].iloc[-1].to_pydatetime(),
        terms_url=terms_url,
        quality=quality.as_dict(),
    )


class DukascopyExportProvider:
    """Imports Dukascopy's documented CSV historical exports without scraping."""

    name = "dukascopy-export"

    def import_export(self, source: Path, destination: Path) -> DatasetManifest:
        return import_market_file(
            source,
            destination,
            source_name=self.name,
            source_url="https://www.dukascopy.com/swiss/english/marketwatch/historical/",
            terms_url="https://www.dukascopy.com/swiss/english/legal/",
        )


class TwelveDataProvider:
    name = "twelve-data"
    base_url = "https://api.twelvedata.com"

    def __init__(self, api_key: str, poll_seconds: int = 60) -> None:
        if not api_key:
            raise ValueError("Twelve Data API key is required")
        self.api_key = api_key
        self.poll_seconds = poll_seconds

    @staticmethod
    def _payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = cast(dict[str, Any], response.json())
        except ValueError:
            payload = {}
        if response.is_error:
            message = payload.get("message", "provider returned an HTTP error")
            raise RuntimeError(
                f"Twelve Data request failed (HTTP {response.status_code}): {message}"
            )
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("message", "Twelve Data request failed"))
        return payload

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        params: dict[str, str | int] = {
            "symbol": "XAU/USD" if symbol.upper() == "XAUUSD" else symbol,
            "interval": "5min",
            "start_date": start.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": 5000,
            "timezone": "UTC",
            "apikey": self.api_key,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/time_series", params=params)
        payload = self._payload(response)
        frame = pd.DataFrame(payload.get("values", []))
        if frame.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        frame = frame.rename(columns={"datetime": "timestamp"})
        normalized, quality = normalize_bars(frame)
        if not quality.valid:
            raise RuntimeError(f"Twelve Data quality failed: {quality.as_dict()}")
        return normalized

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        params: dict[str, str | int] = {
            "symbol": "XAU/USD" if symbol.upper() == "XAUUSD" else symbol,
            "interval": "1min",
            "outputsize": 1,
            "timezone": "UTC",
            "apikey": self.api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{self.base_url}/time_series", params=params)
        payload = self._payload(response)
        values = payload.get("values") or []
        if not values:
            raise RuntimeError("Twelve Data request failed: no current one-minute bar returned")
        latest = values[0]
        close = float(latest["close"])
        parsed = pd.Timestamp(latest["datetime"])
        available_at = (
            parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
        ).to_pydatetime()
        return Quote(
            bid=close,
            ask=close,
            provider=self.name,
            entitlement="latest 1-minute bar; provider plan determines latency",
            available_at=available_at,
        )

    async def stream(self, symbol: str):  # type: ignore[no-untyped-def]
        while True:
            yield await self.quote(symbol)
            await asyncio.sleep(self.poll_seconds)

    async def symbol_specification(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "contract_size_oz": 100.0, "source": "configuration"}

    async def health(self) -> dict[str, Any]:
        try:
            quote = await self.quote()
            return {"healthy": True, "last_quote_at": quote.available_at.isoformat()}
        except Exception as exc:
            return {"healthy": False, "error": type(exc).__name__}


def _parse_provider_timestamp(value: Any) -> datetime:
    now = datetime.now(UTC)
    if value is None:
        return now
    try:
        if str(value).replace(".", "", 1).isdigit():
            return datetime.fromtimestamp(float(value), UTC)
        parsed = pd.Timestamp(value)
        return (
            parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
        ).to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        return now


class AlphaVantageProvider:
    name = "alpha-vantage"
    base_url = "https://www.alphavantage.co"

    def __init__(self, api_key: str, poll_seconds: int = 60) -> None:
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.api_key = api_key
        self.poll_seconds = poll_seconds

    @staticmethod
    def _payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = cast(dict[str, Any], response.json())
        except ValueError:
            payload = {}
        if response.is_error:
            raise RuntimeError(f"Alpha Vantage request failed (HTTP {response.status_code})")
        if payload.get("Error Message") or payload.get("Note") or payload.get("Information"):
            raise RuntimeError("Alpha Vantage request failed: provider returned no data")
        return payload

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        from_currency = "XAU" if symbol.upper() == "XAUUSD" else symbol[:3]
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": "USD",
            "apikey": self.api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{self.base_url}/query", params=params)
        payload = self._payload(response)
        rate = payload.get("Realtime Currency Exchange Rate")
        if not rate:
            raise RuntimeError("Alpha Vantage request failed: missing exchange rate payload")
        mid = float(rate["5. Exchange Rate"])
        bid = float(rate.get("8. Bid Price", mid))
        ask = float(rate.get("9. Ask Price", mid))
        available_at = _parse_provider_timestamp(rate.get("6. Last Refreshed"))
        return Quote(
            bid=bid,
            ask=max(ask, bid),
            provider=self.name,
            entitlement="provider-dependent; verify in dashboard",
            available_at=available_at,
        )

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        from_symbol = "XAU" if symbol.upper() == "XAUUSD" else symbol[:3]
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol,
            "to_symbol": "USD",
            "interval": "5min",
            "outputsize": "full",
            "apikey": self.api_key,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/query", params=params)
        payload = self._payload(response)
        series = payload.get("Time Series FX (5min)")
        if not series:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        frame = pd.DataFrame.from_dict(series, orient="index").reset_index()
        frame = frame.rename(
            columns={
                "index": "timestamp",
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
            }
        )
        window = frame[
            (pd.to_datetime(frame["timestamp"], utc=True) >= pd.Timestamp(start, tz="UTC"))
            & (pd.to_datetime(frame["timestamp"], utc=True) <= pd.Timestamp(end, tz="UTC"))
        ]
        normalized, quality = normalize_bars(window)
        if not quality.valid:
            raise RuntimeError(f"Alpha Vantage quality failed: {quality.as_dict()}")
        return normalized


class MetalsDevProvider:
    """Dedicated precious-metals spot price API; free tier has minute-level delay."""

    name = "metals-dev"
    base_url = "https://api.metals.dev/v1"

    def __init__(self, api_key: str, poll_seconds: int = 60) -> None:
        if not api_key:
            raise ValueError("metals.dev API key is required")
        self.api_key = api_key
        self.poll_seconds = poll_seconds

    @staticmethod
    def _payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = cast(dict[str, Any], response.json())
        except ValueError:
            payload = {}
        if response.is_error or payload.get("status") == "error" or payload.get("success") is False:
            raise RuntimeError(f"metals.dev request failed (HTTP {response.status_code})")
        return payload

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        params = {"api_key": self.api_key, "metal": "gold", "currency": "USD"}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{self.base_url}/metal/spot", params=params)
        payload = self._payload(response)
        rate = payload.get("rate", payload)
        mid = float(rate.get("price", rate.get("ask", rate.get("bid", 0))) or 0)
        if mid <= 0:
            raise RuntimeError("metals.dev request failed: missing spot price")
        bid = float(rate.get("bid", mid))
        ask = float(rate.get("ask", mid))
        available_at = _parse_provider_timestamp(payload.get("timestamp") or payload.get("date"))
        return Quote(
            bid=bid,
            ask=max(ask, bid),
            provider=self.name,
            entitlement="provider-dependent; verify in dashboard",
            available_at=available_at,
        )

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            "api_key": self.api_key,
            "metal": "gold",
            "currency": "USD",
            "start_date": start.astimezone(UTC).date().isoformat(),
            "end_date": end.astimezone(UTC).date().isoformat(),
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/timeseries", params=params)
        payload = self._payload(response)
        rates = payload.get("rates", {})
        if not rates:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        rows = []
        for day, values in rates.items():
            price = float(values.get("metals", values).get("gold", 0) or 0)
            if price <= 0:
                continue
            rows.append(
                {"timestamp": day, "open": price, "high": price, "low": price, "close": price}
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        normalized, quality = normalize_bars(frame, expected_interval="1D")
        if not quality.valid:
            raise RuntimeError(f"metals.dev quality failed: {quality.as_dict()}")
        return normalized


class FinnhubProvider:
    """Finnhub forex quote. Free tier returns last price only, no live bid/ask."""

    name = "finnhub"
    base_url = "https://finnhub.io/api/v1"
    symbol_map: ClassVar[dict[str, str]] = {"XAUUSD": "OANDA:XAU_USD"}

    def __init__(self, api_key: str, poll_seconds: int = 60) -> None:
        if not api_key:
            raise ValueError("Finnhub API key is required")
        self.api_key = api_key
        self.poll_seconds = poll_seconds

    @staticmethod
    def _payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = cast(dict[str, Any], response.json())
        except ValueError:
            payload = {}
        if response.is_error:
            raise RuntimeError(f"Finnhub request failed (HTTP {response.status_code})")
        return payload

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        params = {"symbol": self.symbol_map.get(symbol.upper(), symbol), "token": self.api_key}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{self.base_url}/quote", params=params)
        payload = self._payload(response)
        price = float(payload.get("c") or 0)
        if price <= 0:
            raise RuntimeError("Finnhub request failed: no current price returned")
        available_at = _parse_provider_timestamp(payload.get("t"))
        return Quote(
            bid=price,
            ask=price,
            provider=self.name,
            entitlement="provider-dependent; verify in dashboard",
            available_at=available_at,
        )

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        params: dict[str, str | int] = {
            "symbol": self.symbol_map.get(symbol.upper(), symbol),
            "resolution": "5",
            "from": int(start.astimezone(UTC).timestamp()),
            "to": int(end.astimezone(UTC).timestamp()),
            "token": self.api_key,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/forex/candle", params=params)
        payload = self._payload(response)
        if payload.get("s") != "ok" or not payload.get("t"):
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(payload["t"], unit="s", utc=True),
                "open": payload["o"],
                "high": payload["h"],
                "low": payload["l"],
                "close": payload["c"],
            }
        )
        normalized, quality = normalize_bars(frame)
        if not quality.valid:
            raise RuntimeError(f"Finnhub quality failed: {quality.as_dict()}")
        return normalized


class FailoverQuoteProvider:
    """Tries quote providers in priority order, falling back on any failure.

    Never swallows a total outage: if every provider fails, the last error is
    re-raised so the live-paper engine's existing network-error handling
    (feed_status, retry backoff, notifications) keeps working unchanged.
    """

    def __init__(
        self,
        providers: list[tuple[str, LiveQuoteProvider]],
        notifier: Any | None = None,
    ) -> None:
        if not providers:
            raise ValueError("at least one live quote provider is required")
        self.providers = providers
        self.notifier = notifier
        self._active_name: str | None = None

    @property
    def active_provider_name(self) -> str | None:
        return self._active_name

    def _mark_active(self, name: str) -> None:
        changed = self._active_name is not None and self._active_name != name
        if changed and self.notifier is not None:
            self.notifier.notify(
                "Aurum feed failover",
                f"Switched live quote source from {self._active_name} to {name}",
            )
        self._active_name = name

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        last_error: Exception | None = None
        for name, provider in self.providers:
            try:
                quote = await provider.quote(symbol)
            except Exception as exc:
                last_error = exc
                continue
            self._mark_active(name)
            return quote
        assert last_error is not None
        raise last_error

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        last_error: Exception | None = None
        for name, provider in self.providers:
            try:
                frame = await provider.backfill(symbol, start, end)
            except Exception as exc:
                last_error = exc
                continue
            self._mark_active(name)
            return frame
        assert last_error is not None
        raise last_error


def generate_demo_data(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic bars for workflow testing, never research claims."""
    import numpy as np

    random = np.random.default_rng(seed)
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    returns = random.normal(0.000002, 0.00055, rows)
    close = 2400 * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([close[0]], close[:-1]))
    range_ = np.maximum(np.abs(random.normal(0.8, 0.35, rows)), 0.1)
    high = np.maximum(open_, close) + range_
    low = np.minimum(open_, close) - range_
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": random.integers(50, 500, rows),
        }
    )
