from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskSettings(BaseModel):
    max_risk_fraction: float = Field(default=0.005, gt=0, le=0.005)
    daily_loss_fraction: float = Field(default=0.02, gt=0, le=0.02)
    rolling_five_day_loss_fraction: float = Field(default=0.05, gt=0, le=0.05)
    peak_drawdown_fraction: float = Field(default=0.10, gt=0, le=0.10)
    max_positions: int = Field(default=1, ge=1, le=1)
    contract_size_oz: float = Field(default=100.0, gt=0)
    min_lot: float = Field(default=0.01, gt=0)
    lot_step: float = Field(default=0.01, gt=0)


class SimulationSettings(BaseModel):
    spread_points: float = Field(default=0.30, ge=0)
    slippage_points: float = Field(default=0.10, ge=0)
    commission_per_lot_round_trip: float = Field(default=7.0, ge=0)
    swap_per_lot_day: float = -5.0


class ResearchSettings(BaseModel):
    holdout_months: int = Field(default=12, ge=6)
    intraday_min_trades: int = Field(default=100, ge=1)
    swing_min_trades: int = Field(default=40, ge=1)
    min_profit_factor: float = Field(default=1.15, ge=1)
    min_sharpe: float = Field(default=0.75)
    max_drawdown: float = Field(default=0.10, gt=0, le=1)
    min_profitable_fold_fraction: float = Field(default=0.60, ge=0.5, le=1)


class GenerationSettings(BaseModel):
    workers: int = Field(default=6, ge=1, le=12)
    trials_per_family: int = Field(default=384, ge=1, le=100_000)
    batch_size: int = Field(default=24, ge=1, le=1_000)
    minimum_batches_before_futility: int = Field(default=6, ge=1)
    no_improvement_batches: int = Field(default=8, ge=1)
    minimum_free_disk_gb: float = Field(default=2.0, ge=0.25)
    maximum_load_per_worker: float = Field(default=2.0, ge=0.5)


class LivePaperSettings(BaseModel):
    poll_seconds: float = Field(default=120.0, ge=1.0, le=900.0)
    stale_after_seconds: float = Field(default=180.0, ge=5.0, le=3600.0)
    max_retry_seconds: float = Field(default=300.0, ge=5.0, le=3600.0)
    reconnect_initial_seconds: float = Field(default=10.0, ge=1.0, le=300.0)
    reconnect_max_seconds: float = Field(default=120.0, ge=5.0, le=900.0)
    warmup_bars_5m: int = Field(default=3000, ge=500, le=100_000)
    provider_priority: list[str] = Field(default_factory=lambda: ["twelve-data"])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GOLD_TRADER_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    env: str = "development"
    database_url: str = "sqlite:///var/gold_trader.sqlite3"
    host: str = "127.0.0.1"
    port: int = 8775
    starting_equity: float = Field(default=10_000.0, gt=0)
    session_secret: str = ""
    fred_keyring_name: str = "fred"
    risk: RiskSettings = RiskSettings()
    simulation: SimulationSettings = SimulationSettings()
    research: ResearchSettings = ResearchSettings()
    generation: GenerationSettings = GenerationSettings()
    live_paper: LivePaperSettings = LivePaperSettings()

    @property
    def effective_session_secret(self) -> str:
        return self.session_secret or secrets.token_urlsafe(32)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def ensure_runtime_dirs(self) -> None:
        for relative in (
            "var",
            "var/data",
            "var/reports",
            "var/generations",
            "artifacts",
        ):
            (self.project_root / relative).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
