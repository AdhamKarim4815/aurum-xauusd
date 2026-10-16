# Paper trading and risk PRD

## Account and authority

The default paper account starts at USD 10,000. The approved candidate may propose signals; the risk engine alone decides whether an `OrderIntent` becomes a `SimulatedOrder`. Only one simulated venue is registered.

## Hard ceilings

- Risk at stop: at most 0.5% of current equity.
- Exposure: one net XAUUSD position per paper account. The normal approved workflow has one account. The explicitly acknowledged four-strategy experimental lab has four isolated accounts and can therefore hold up to four simultaneous positions in aggregate, one per selected strategy.
- Daily halt: 2% below UTC day-start equity.
- Rolling halt: 5% below the five-trading-day high.
- Account lock: 10% below peak equity.
- Stop: distinct, positive, on the loss side of entry, and present before size calculation.
- Lot size: round down to the configured broker-like step; reject below the minimum rather than round up.

Risk ceilings may be lowered through configuration but cannot be raised above these values without a deliberate code, documentation, test, and approval-lineage change.

## Prohibited behavior

Martingale sizing, grids, averaging down, parallel long/short positions, stop widening after entry, uncapped leverage, manual orders, and model overrides of a risk rejection.

The four-strategy lab does not relax these rules within an account: no strategy can pyramid, average
down, or hold parallel long/short positions. Accounts are intentionally not netted or combined, so
their equity curves and returns must be interpreted separately.

## Paper fill rules

Signals enter on the next quote/bar. Every fill records spread, adverse slippage, commission, and swap. Unordered intrabar stop/target collisions resolve to the stop. Gaps fill at the available price. Event and weekend context is a model feature rather than a hard veto, so gap and swap effects must remain visible.

## Staleness and recovery

- Each provider declares its expected interval and stale threshold.
- Missing, crossed, future-dated, unentitled, or stale quotes block entries.
- A reconnect backfills the gap, verifies monotonic event time, reconciles orders/positions, and resumes only from the next new quote.
- The system never invents retroactive fills for a disconnected period.
- Starting an active paper session acquires a `caffeinate` lease; stop, kill, or worker exit releases it.

## Approval

An approval binds candidate, data, configuration, strategy, and model hashes. The dashboard exposes an approval button only for a specific eligible candidate and submits its exact full hash. Any bound hash change invalidates the approval. Start is rejected when approval is missing, invalid, or mismatched.

The experimental lab is a separate, auditable exception to normal approval. It may accept only a
finalized lineage whose failed-gate set is a subset of `trade_count` and `monthly_activity`, and it
requires an explicit paper-only acknowledgement on every start. All other final gates remain hard.

## Kill switch

Kill is synchronous at the application state boundary: mark killed, block entries, record the event, enqueue simulated cancellation/reconciliation, notify the operator, and release the sleep lease. Reset requires the worker to be stopped and creates another audit event.

## Future-live gate

There is no automatic transition. The readiness report stays false until at least 30 trading days and 50 closed paper trades exist, operational interruptions are reviewed, observed spread/slippage is compared with assumptions, and the operator explicitly commissions a separate live PRD.
