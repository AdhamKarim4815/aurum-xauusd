# Four-strategy live paper lab

## Purpose and boundary

This lab observes four exact strategy lineages simultaneously against live or delayed XAUUSD quote
data. It creates simulated orders and fills only. It has no broker account, real execution venue,
manual-order endpoint, MT5 dependency, or path to real money. Paper performance can differ materially
from broker execution and does not demonstrate future profitability.

The lab exists because a low-frequency strategy can fail the final trade-count/activity floor while
passing the other final safety and robustness gates. It does not convert that failure into a pass.
The dashboard labels the override, preserves the failed gates, and requires acknowledgement before
each start.

## Admission policy

A candidate is selectable only when:

- it is a durable finalized Generation 3, 4, 5, or 6 lineage with an exact trial specification;
- every failed final gate is either `trade_count` or `monthly_activity`;
- its dataset, configuration, strategy, and model hashes remain available; and
- its warm-up dataset is present and byte-identical to the final report.

Return, doubled-cost, profit-factor, Sharpe, drawdown, stability, concentration, calibration,
data-quality, and independent-source failures cannot be overridden. Exactly four unique candidate
hashes are required.

## Runtime model

`gold-trader serve` supervises a dedicated `PaperLabWorker`. Dashboard requests enqueue idempotent
`paperlab.start`, `paperlab.stop`, and `paperlab.kill` commands; trading logic never runs in the HTTP
process.

The coordinator fetches one provider quote per cycle. Each strategy receives an event-ID-distinct
copy with the same market values and evaluates it using its own bars, pending signal, position,
balance, equity, and risk state. Account state is stored under
`paper_lab:account:<candidate-hash>`. Audit events use a candidate-specific correlation ID.

The default provider cycle is 120 seconds. That is 720 shared requests per 24 hours and leaves a
small buffer under Twelve Data Basic's documented 800-request daily cap. Extra diagnostics or data
requests use the same allowance; exhaustion is treated as a feed failure and blocks entries.

If the quote connection fails, the entire lab enters `paused-network`. New entries are blocked,
open simulated positions are preserved at their last known mark, and the worker retries after
10, 20, 40, 80, then at most 120 seconds. A successful quote automatically triggers the gap-backfill
and reconciliation path, returns healthy accounts to `running`, and never creates retroactive trades
for signals that occurred while offline.

Each account starts at the configured default of $10,000 and may hold one position. The four accounts
may therefore hold four simultaneous XAUUSD positions in aggregate. A strategy may not pyramid,
average down, widen a stop, or hold simultaneous long and short positions. Accounts are not combined,
so four displayed returns must never be added and called a portfolio return.

## Controls and recovery

- **Start all four** validates the exact candidate set, loads independent warm-up state, acquires one
  macOS sleep-prevention lease, and begins polling.
- **Stop all safely** cancels pending simulated entries, closes open simulated positions at each
  account's last known mark, and releases the lease.
- **Kill all four** applies the same flatten-and-release behavior with a killed state and operator
  reason. The global command-center kill also kills a running lab.
- Network failure changes the lab to `paused-network`, stale/future quotes block affected entries,
  and a missing hash-bound dataset, risk halt, or worker error fails closed. Restart restores a
  previously running or network-paused lab from namespaced state and does not invent retroactive
  fills.

## Dashboard interpretation

The lab page refreshes every three seconds and displays, per strategy:

- status and feed health;
- equity and return against that strategy's own starting balance;
- closed simulated trades and current position;
- whether entries are allowed, quote age, current risk/feed block, and last decision bar.

Low-frequency 1h and 4h strategies may legitimately show no trade for days or weeks. “Running” means
the worker and feed are functioning, not that a trade is guaranteed. Keep the lab active long enough
to observe varied market regimes, and review the audit trail before drawing conclusions.
