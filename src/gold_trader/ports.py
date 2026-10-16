from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Any, Protocol

import pandas as pd

from gold_trader.domain import (
    AccountSnapshot,
    FeatureVector,
    ModelScore,
    OrderIntent,
    Quote,
    RiskDecision,
    Signal,
    SimulatedOrder,
)


class MarketDataProvider(Protocol):
    name: str

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame: ...

    async def stream(self, symbol: str) -> AsyncIterator[Quote]: ...

    async def symbol_specification(self, symbol: str) -> dict[str, Any]: ...

    async def health(self) -> dict[str, Any]: ...


class AlternativeDataProvider(Protocol):
    name: str

    async def records(self, start: datetime, end: datetime) -> pd.DataFrame: ...


class FeaturePipeline(Protocol):
    def transform(self, bars: pd.DataFrame, as_of: datetime) -> FeatureVector: ...


class Strategy(Protocol):
    strategy_id: str
    timeframe: str

    def generate(self, bars: pd.DataFrame) -> Signal | None: ...


class ModelScorer(Protocol):
    model_id: str

    def score(self, signal: Signal, features: FeatureVector) -> ModelScore: ...


class RiskEnginePort(Protocol):
    def evaluate(self, intent: OrderIntent, account: AccountSnapshot) -> RiskDecision: ...


class ExecutionVenue(Protocol):
    venue_id: str

    def submit(
        self, intent: OrderIntent, decision: RiskDecision, market_price: float
    ) -> SimulatedOrder: ...

    def cancel_all(self) -> int: ...

    def reconcile(self) -> Sequence[SimulatedOrder]: ...


class ExperimentRunner(Protocol):
    def run(self, dataset: pd.DataFrame, strategy: Strategy) -> dict[str, Any]: ...
