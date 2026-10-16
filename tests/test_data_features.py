from __future__ import annotations

import asyncio
from datetime import UTC

import pandas as pd

from gold_trader.data import (
    AlphaVantageProvider,
    FinnhubProvider,
    MetalsDevProvider,
    TwelveDataProvider,
    generate_demo_data,
    normalize_bars,
    resample_bars,
)
from gold_trader.features import PointInTimeFeaturePipeline


def _fake_http_client(
    monkeypatch, payload: dict, *, status_code: int = 200, is_error: bool = False
):
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = status_code
            self.is_error = is_error

        @staticmethod
        def json() -> dict:
            return payload

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("gold_trader.data.httpx.AsyncClient", FakeClient)


def test_normalization_deduplicates_and_reports() -> None:
    frame = generate_demo_data(rows=500)
    frame = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    normalized, report = normalize_bars(frame)
    assert report.valid
    assert report.duplicate_timestamps == 1
    assert len(normalized) == 500
    assert str(normalized["timestamp"].dt.tz) == "UTC"


def test_normalization_converts_jforex_eet_export_to_utc() -> None:
    frame = pd.DataFrame(
        {
            "Time (EET)": ["2016.01.18 01:00:00", "2016.07.18 01:00:00"],
            "Open": [1100.0, 1332.0],
            "High": [1101.0, 1333.0],
            "Low": [1099.0, 1331.0],
            "Close": [1100.5, 1332.5],
            "Volume": [1.0, 1.0],
        }
    )

    normalized, report = normalize_bars(frame)

    assert report.valid
    assert normalized["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z").tolist() == [
        "2016-01-17 23:00:00+0000",
        "2016-07-17 22:00:00+0000",
    ]


def test_twelve_data_error_does_not_expose_request_url_or_key(monkeypatch) -> None:
    secret = "sensitive-api-key"

    class FakeResponse:
        is_error = True
        status_code = 401

        @staticmethod
        def json() -> dict[str, str]:
            return {"status": "error", "message": "invalid API key"}

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("gold_trader.data.httpx.AsyncClient", FakeClient)

    try:
        asyncio.run(TwelveDataProvider(secret).quote())
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected a sanitized provider error")

    assert secret not in message
    assert "https://" not in message
    assert "HTTP 401" in message


def test_twelve_data_quote_uses_timestamped_latest_minute(monkeypatch) -> None:
    _fake_http_client(
        monkeypatch,
        {
            "status": "ok",
            "values": [
                {
                    "datetime": "2026-07-23 01:27:00",
                    "open": "4137.63",
                    "high": "4140.13",
                    "low": "4137.48",
                    "close": "4139.62",
                }
            ],
        },
    )

    quote = asyncio.run(TwelveDataProvider("key").quote())

    assert quote.bid == 4139.62
    assert quote.ask == 4139.62
    assert quote.available_at.isoformat() == "2026-07-23T01:27:00+00:00"
    assert "1-minute" in quote.entitlement


def test_alpha_vantage_quote_parses_bid_ask(monkeypatch) -> None:
    payload = {
        "Realtime Currency Exchange Rate": {
            "5. Exchange Rate": "2400.12",
            "6. Last Refreshed": "2026-07-23 12:00:00",
            "8. Bid Price": "2399.90",
            "9. Ask Price": "2400.30",
        }
    }
    _fake_http_client(monkeypatch, payload)
    quote = asyncio.run(AlphaVantageProvider("key").quote())
    assert quote.bid == 2399.90
    assert quote.ask == 2400.30
    assert quote.provider == "alpha-vantage"


def test_alpha_vantage_error_does_not_expose_request_url_or_key(monkeypatch) -> None:
    secret = "sensitive-av-key"
    _fake_http_client(monkeypatch, {"Error Message": "invalid API call"})

    try:
        asyncio.run(AlphaVantageProvider(secret).quote())
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected a sanitized provider error")

    assert secret not in message
    assert "https://" not in message


def test_metals_dev_quote_parses_bid_ask(monkeypatch) -> None:
    payload = {
        "status": "success",
        "rate": {"price": 2400.5, "bid": 2399.8, "ask": 2401.0},
        "timestamp": "2026-07-23T12:00:00Z",
    }
    _fake_http_client(monkeypatch, payload)
    quote = asyncio.run(MetalsDevProvider("key").quote())
    assert quote.bid == 2399.8
    assert quote.ask == 2401.0
    assert quote.provider == "metals-dev"


def test_metals_dev_error_does_not_expose_request_url_or_key(monkeypatch) -> None:
    secret = "sensitive-metals-key"
    _fake_http_client(monkeypatch, {"message": "invalid key"}, status_code=401, is_error=True)

    try:
        asyncio.run(MetalsDevProvider(secret).quote())
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected a sanitized provider error")

    assert secret not in message
    assert "https://" not in message
    assert "HTTP 401" in message


def test_finnhub_quote_synthesizes_bid_ask_from_last_price(monkeypatch) -> None:
    payload = {"c": 2400.55, "t": 1753272000}
    _fake_http_client(monkeypatch, payload)
    quote = asyncio.run(FinnhubProvider("key").quote())
    assert quote.bid == 2400.55
    assert quote.ask == 2400.55
    assert quote.provider == "finnhub"


def test_finnhub_error_does_not_expose_request_url_or_key(monkeypatch) -> None:
    secret = "sensitive-finnhub-key"
    _fake_http_client(monkeypatch, {}, status_code=401, is_error=True)

    try:
        asyncio.run(FinnhubProvider(secret).quote())
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected a sanitized provider error")

    assert secret not in message
    assert "https://" not in message
    assert "HTTP 401" in message


def test_resample_preserves_ohlc_contract() -> None:
    normalized, _ = normalize_bars(generate_demo_data(rows=500))
    hourly = resample_bars(normalized, "1h")
    assert len(hourly) < len(normalized)
    assert (hourly["high"] >= hourly[["open", "close"]].max(axis=1)).all()
    assert (hourly["low"] <= hourly[["open", "close"]].min(axis=1)).all()


def test_future_bars_cannot_change_as_of_features() -> None:
    bars = generate_demo_data(rows=500)
    as_of = pd.Timestamp(bars.iloc[300]["timestamp"]).to_pydatetime().astimezone(UTC)
    pipeline = PointInTimeFeaturePipeline()
    before = pipeline.transform(bars.iloc[:350], as_of)
    changed = bars.copy()
    changed.loc[400:, "close"] *= 100
    after = pipeline.transform(changed, as_of)
    assert before.features == after.features
