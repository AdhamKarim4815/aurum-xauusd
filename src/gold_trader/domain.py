from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class CommandStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DomainRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = Field(default_factory=utc_now)
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    causation_id: str | None = None
    data_hash: str | None = None
    config_hash: str | None = None
    strategy_hash: str | None = None
    model_hash: str | None = None

    @field_validator("occurred_at")
    @classmethod
    def validate_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class Quote(DomainRecord):
    symbol: str = "XAUUSD"
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    provider: str
    entitlement: str = "unknown"
    available_at: datetime

    @field_validator("available_at")
    @classmethod
    def validate_available_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_spread(self) -> Quote:
        if self.ask < self.bid:
            raise ValueError("crossed quote: ask cannot be below bid")
        return self


class Bar(DomainRecord):
    symbol: str = "XAUUSD"
    timeframe: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)
    bid_close: float | None = Field(default=None, gt=0)
    ask_close: float | None = Field(default=None, gt=0)
    available_at: datetime

    @field_validator("available_at")
    @classmethod
    def validate_available_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_ohlc(self) -> Bar:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("bar high is inconsistent")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("bar low is inconsistent")
        if (
            self.bid_close is not None
            and self.ask_close is not None
            and self.ask_close < self.bid_close
        ):
            raise ValueError("crossed close quote")
        return self


class DatasetManifest(DomainRecord):
    dataset_id: str
    symbol: str = "XAUUSD"
    source: str
    source_url: str | None = None
    retrieved_at: datetime
    content_sha256: str
    parser_version: str
    timezone: str = "UTC"
    row_count: int = Field(ge=0)
    start_at: datetime | None = None
    end_at: datetime | None = None
    terms_url: str | None = None
    quality: dict[str, Any] = Field(default_factory=dict)

    @field_validator("retrieved_at", "start_at", "end_at")
    @classmethod
    def validate_manifest_times(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


class FeatureVector(DomainRecord):
    symbol: str = "XAUUSD"
    as_of: datetime
    available_at: datetime
    features: dict[str, float | int | bool | str | None]

    @field_validator("as_of", "available_at")
    @classmethod
    def validate_feature_times(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def prevent_future_feature(self) -> FeatureVector:
        if self.available_at > self.as_of:
            raise ValueError("feature was not available as of the decision time")
        return self


class Signal(DomainRecord):
    symbol: str = "XAUUSD"
    side: Side
    generated_at: datetime
    reference_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    strategy_id: str
    reason: str
    event_context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _utc(value)


class ModelScore(DomainRecord):
    signal_id: str
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    eligible: bool
    explanation: dict[str, float] = Field(default_factory=dict)


class OrderIntent(DomainRecord):
    signal_id: str
    symbol: str = "XAUUSD"
    side: Side
    reference_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    requested_risk_fraction: float = Field(gt=0)


class RiskDecision(DomainRecord):
    intent_id: str
    approved: bool
    reason: str
    risk_amount: float = Field(default=0, ge=0)
    lots: float = Field(default=0, ge=0)
    effective_risk_fraction: float = Field(default=0, ge=0)


class SimulatedOrder(DomainRecord):
    intent_id: str
    symbol: str = "XAUUSD"
    side: Side
    lots: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    status: OrderStatus = OrderStatus.PENDING


class Fill(DomainRecord):
    order_id: str
    symbol: str = "XAUUSD"
    side: Side
    lots: float = Field(gt=0)
    price: float = Field(gt=0)
    commission: float = Field(default=0, ge=0)
    slippage: float = Field(default=0, ge=0)
    filled_at: datetime

    @field_validator("filled_at")
    @classmethod
    def validate_filled_at(cls, value: datetime) -> datetime:
        return _utc(value)


class Position(DomainRecord):
    symbol: str = "XAUUSD"
    side: Side
    lots: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    opened_at: datetime
    unrealized_pnl: float = 0

    @field_validator("opened_at")
    @classmethod
    def validate_opened_at(cls, value: datetime) -> datetime:
        return _utc(value)


class AccountSnapshot(DomainRecord):
    balance: float = Field(gt=0)
    equity: float = Field(gt=0)
    day_start_equity: float = Field(gt=0)
    rolling_five_day_high: float = Field(gt=0)
    peak_equity: float = Field(gt=0)
    open_position_count: int = Field(default=0, ge=0)
    kill_switch_active: bool = False


class Approval(DomainRecord):
    candidate_hash: str
    dataset_hash: str
    configuration_hash: str
    strategy_hash: str
    model_hash: str
    approved_at: datetime = Field(default_factory=utc_now)
    valid: bool = True

    @field_validator("approved_at")
    @classmethod
    def validate_approved_at(cls, value: datetime) -> datetime:
        return _utc(value)


class KillSwitchEvent(DomainRecord):
    active: bool
    reason: str
    operator: str = "local"


class PaperTradeClosed(DomainRecord):
    trade_id: str
    side: Side
    lots: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    opened_at: datetime
    closed_at: datetime
    pnl: float
    commission: float = Field(ge=0)
    swap: float
    reason: str

    @field_validator("opened_at", "closed_at")
    @classmethod
    def validate_trade_times(cls, value: datetime) -> datetime:
        return _utc(value)
