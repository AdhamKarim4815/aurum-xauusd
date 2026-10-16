from __future__ import annotations

from pathlib import Path

import pytest

from gold_trader.config import Settings
from gold_trader.database import Store


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
        session_secret="test-session-secret",
    )


@pytest.fixture
def store(settings: Settings) -> Store:
    selected = Store(settings.database_url)
    selected.initialize()
    return selected
