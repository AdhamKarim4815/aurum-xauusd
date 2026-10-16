.PHONY: setup setup-ml check test serve demo

setup:
	uv sync

setup-ml:
	uv sync --extra ml

check:
	uv run ruff check .
	uv run mypy src/gold_trader
	uv run pytest --cov=gold_trader --cov-report=term-missing

test:
	uv run pytest

serve:
	uv run gold-trader serve

demo:
	uv run gold-trader demo-data
	uv run gold-trader research run --dataset var/data/demo_xauusd_5m.csv --strategy trend-pullback

