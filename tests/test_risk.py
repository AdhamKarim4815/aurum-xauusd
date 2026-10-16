from __future__ import annotations

from gold_trader.config import RiskSettings
from gold_trader.domain import AccountSnapshot, OrderIntent, Side
from gold_trader.risk import RiskEngine


def account(**overrides: float | int | bool) -> AccountSnapshot:
    values: dict[str, float | int | bool] = {
        "balance": 10_000.0,
        "equity": 10_000.0,
        "day_start_equity": 10_000.0,
        "rolling_five_day_high": 10_000.0,
        "peak_equity": 10_000.0,
        "open_position_count": 0,
        "kill_switch_active": False,
    }
    values.update(overrides)
    return AccountSnapshot(**values)  # type: ignore[arg-type]


def intent(side: Side = Side.BUY, stop: float = 2390) -> OrderIntent:
    return OrderIntent(
        signal_id="signal",
        side=side,
        reference_price=2400,
        stop_loss=stop,
        take_profit=2420 if side is Side.BUY else 2380,
        requested_risk_fraction=0.005,
    )


def test_size_rounds_down_and_never_exceeds_half_percent() -> None:
    decision = RiskEngine(RiskSettings()).evaluate(intent(), account())
    assert decision.approved
    assert decision.lots == 0.05
    assert decision.risk_amount == 50
    assert decision.effective_risk_fraction <= 0.005


def test_model_request_cannot_raise_hard_cap() -> None:
    requested = intent().model_copy(update={"requested_risk_fraction": 0.50})
    decision = RiskEngine(RiskSettings()).evaluate(requested, account())
    assert decision.approved
    assert decision.effective_risk_fraction <= 0.005


def test_all_halts_are_fail_closed() -> None:
    engine = RiskEngine(RiskSettings())
    cases = [
        account(kill_switch_active=True),
        account(open_position_count=1),
        account(equity=9799),
        account(equity=9499),
        account(equity=8999),
    ]
    assert all(not engine.evaluate(intent(), snapshot).approved for snapshot in cases)


def test_stop_must_be_on_loss_side() -> None:
    engine = RiskEngine(RiskSettings())
    assert not engine.evaluate(intent(stop=2401), account()).approved
    assert not engine.evaluate(intent(Side.SELL, stop=2399), account()).approved
