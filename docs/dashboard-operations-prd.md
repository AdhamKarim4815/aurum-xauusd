# Dashboard and operations PRD

## Information architecture

- **Command:** paper status, feed health, sleep lease, current approval, risk ceilings, start/stop, and kill.
- **Research:** candidate lineage, metrics, stressed metrics, individual gates, model evidence, combined generation choices, and eligible-only one-click approval.
- **Audit trail:** timestamped event type, ID, cause/correlation fields, and payload.
- **Settings:** effective non-secret configuration and provider/key presence, never secret values.
- **Live lab:** exact four-candidate selection, explicit experimental acknowledgement, shared-feed and sleep-lease status, stop/kill-all controls, and independent equity, return, feed, position, trade-count, risk, quote-age, and decision progress for each strategy.

Future pages may add data quality, source health, trade/equity detail, model-card visualization, SHAP plots, paper-observation progress, and recovery tools without adding manual trading.

## Interaction requirements

- Use normal HTML forms with CSRF protection for state changes; pages remain usable without JavaScript.
- JavaScript may poll read-only status and draw local charts.
- Buttons reflect preconditions, while the service rechecks every precondition server-side.
- Errors state why an action was rejected without disclosing secrets or internals.
- Risk and kill controls remain keyboard reachable, visibly focused, and responsive on a laptop or phone-sized local browser.

## Runtime responsibilities

`gold-trader serve` starts the localhost web process and isolated normal-paper, four-strategy-lab,
validation, and research workers. Research runs execute through the CLI/process boundary so
CPU-heavy work cannot block risk controls. SQLite commands are durable and idempotent.

## Alerts

Native macOS notifications cover stale feeds, worker errors, loss halts, approval invalidation, and kill activation. Notifications are secondary; SQLite state and audit events are authoritative.

## Operator workflow

1. Run doctor and inspect missing prerequisites.
2. Import data and review the manifest.
3. Run a candidate and inspect every failed/passed gate.
4. Type an eligible hash to approve it.
5. Start replay paper; confirm worker, feed, and sleep states.
6. Watch alerts, equity, trades, drawdowns, and data freshness.
7. Stop normally or kill immediately when state is uncertain.
8. Review audit and reconcile before reset/restart.

For experimental simultaneous observation, use **Live lab** instead of the normal approved account.
Select exactly four qualified finalized candidates, read and accept the paper-only warning, start
them together, and judge each account separately. Stop or kill applies to all four so the lab cannot
silently continue with a partial operator-selected set.
