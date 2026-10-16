from __future__ import annotations

import math

from gold_trader.config import RiskSettings
from gold_trader.domain import AccountSnapshot, OrderIntent, RiskDecision, Side


class RiskEngine:
    """The final authority for all order intents; models cannot override it."""

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    def evaluate(self, intent: OrderIntent, account: AccountSnapshot) -> RiskDecision:
        rejection = self._rejection_reason(intent, account)
        if rejection:
            return RiskDecision(intent_id=intent.event_id, approved=False, reason=rejection)

        risk_fraction = min(intent.requested_risk_fraction, self.settings.max_risk_fraction)
        risk_amount = account.equity * risk_fraction
        stop_distance = abs(intent.reference_price - intent.stop_loss)
        raw_lots = risk_amount / (stop_distance * self.settings.contract_size_oz)
        lots = math.floor(raw_lots / self.settings.lot_step) * self.settings.lot_step
        lots = round(lots, 8)
        if lots < self.settings.min_lot:
            return RiskDecision(
                intent_id=intent.event_id,
                approved=False,
                reason="calculated size is below the minimum lot",
            )

        actual_risk = lots * stop_distance * self.settings.contract_size_oz
        if actual_risk > risk_amount + 1e-8:
            return RiskDecision(
                intent_id=intent.event_id,
                approved=False,
                reason="rounded size exceeds risk budget",
            )
        return RiskDecision(
            intent_id=intent.event_id,
            approved=True,
            reason="all deterministic risk controls passed",
            risk_amount=actual_risk,
            lots=lots,
            effective_risk_fraction=actual_risk / account.equity,
        )

    def _rejection_reason(self, intent: OrderIntent, account: AccountSnapshot) -> str | None:
        if account.kill_switch_active:
            return "kill switch is active"
        if account.open_position_count >= self.settings.max_positions:
            return "single-position limit reached"
        if intent.stop_loss <= 0 or math.isclose(intent.stop_loss, intent.reference_price):
            return "a distinct positive stop loss is mandatory"
        if intent.side is Side.BUY and intent.stop_loss >= intent.reference_price:
            return "buy stop must be below entry"
        if intent.side is Side.SELL and intent.stop_loss <= intent.reference_price:
            return "sell stop must be above entry"
        if account.equity <= account.day_start_equity * (1 - self.settings.daily_loss_fraction):
            return "daily loss halt is active"
        if account.equity <= account.rolling_five_day_high * (
            1 - self.settings.rolling_five_day_loss_fraction
        ):
            return "rolling five-day loss halt is active"
        if account.equity <= account.peak_equity * (1 - self.settings.peak_drawdown_fraction):
            return "peak-to-trough account lock is active"
        return None
