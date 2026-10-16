# Master product PRD

## Product statement

Aurum helps one operator research, audit, and paper-test XAUUSD strategies locally on a MacBook without exposing a path to real orders. The value is disciplined evidence and operational correctness, not a promised return.

## User and jobs

The sole v1 user is the owner/operator. They need to:

1. Import or sync point-in-time market and alternative data.
2. See whether it is complete, licensed, timestamp-safe, and reproducible.
3. Run preregistered rules and optional ML signal scorers.
4. Compare base and stressed-cost results on an untouched holdout.
5. Approve one exact candidate lineage after reading its evidence.
6. Operate a paper worker, inspect state, and stop or kill it safely.
7. Collect at least 30 trading days and 50 closed paper trades before even considering a live-integration project.

## Functional requirements

- **PRD-001 Local boundary:** bind the dashboard to loopback, store state locally, and keep credentials in Keychain.
- **PRD-002 No live orders:** the runtime must not register or import a real execution venue.
- **PRD-003 Reproducible data:** every normalized dataset has a manifest and content hash.
- **PRD-004 Research evidence:** every run records metrics, stressed metrics, gates, trades, hashes, timestamps, and limitations.
- **PRD-005 Manual promotion:** only an eligible candidate explicitly selected with its dashboard approval button can be approved; the backend binds the exact full hash.
- **PRD-006 Approval invalidation:** data, configuration, strategy, or model changes invalidate the prior lineage.
- **PRD-007 Deterministic risk:** risk evaluation is authoritative and fully auditable.
- **PRD-008 Safe worker:** commands are durable and idempotent; stale/unconfigured live data blocks entries.
- **PRD-009 Emergency control:** kill immediately blocks entries, cancels pending simulated orders, releases the sleep lease, and records why.
- **PRD-010 Evidence dashboard:** show system, feed, approval, risk, research, and audit state without exposing a manual trade form.
- **PRD-025 Experimental four-strategy lab:** an explicit paper-only workflow may run exactly four finalized strategies whose only failed final gates are trade count or monthly activity. It must isolate account state and risk per strategy, permit at most one position per strategy (four aggregate), share one live quote fetch per cycle, expose progress per strategy, and never register a real execution venue.

## Non-functional requirements

- Repeating a replay with the same inputs produces the same trades within documented floating-point tolerances.
- UTC is used for event time, available time, research splits, daily risk resets, and audit records.
- The default setup works on Apple silicon using Python 3.12 managed by `uv`.
- Failures are closed: the consequence is no new entry, never an inferred quote or silent approval.
- Secrets are redacted and scraped content is treated as untrusted data.

## Success measures

- Fresh setup, doctor, synthetic data, research, and dashboard smoke flow complete without network access.
- All risk invariants and future-leak tests pass in CI.
- No executable real-order surface exists.
- Every dashboard action that changes paper or approval state produces a durable event or command.
- A readiness report remains false until both paper-observation thresholds are evidenced.

## Explicit non-goals

Commercial distribution, multi-user identity, cloud hosting, social/copy trading, manual orders, portfolio trading, options/futures execution, guaranteed uptime, and real-money MT5 integration are not v1 work.
